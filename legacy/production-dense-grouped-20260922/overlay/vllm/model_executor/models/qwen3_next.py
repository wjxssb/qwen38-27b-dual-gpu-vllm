# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

import os as _diag_os
import json as _diag_json
import functools as _diag_functools
import tempfile as _diag_tempfile
from pathlib import Path as _DiagPath
from collections import deque as _DiagDeque
_DIAG_RING = _DiagDeque(maxlen=128)
# Keep synchronization at decoder boundaries, including both sides of the
# historically suspect post-attention norm. Inner GDN/attention checkpoints
# retain metadata without serializing every projection and recurrence kernel.
_DIAG_SYNC_STAGES = frozenset({
    "after_input_norm", "after_linear_attention", "after_full_attention",
    "after_post_norm", "after_mlp",
})

def _diag_failure(exc):
    # No CUDA call on the failure path. Keep only bounded CPU metadata.
    payload = {"exception": str(exc)[:2000], "stages": list(_DIAG_RING)}
    path = _DiagPath("/results") / ("first-cuda-failure-pid" + str(_diag_os.getpid()) + ".json")
    temporary = None
    try:
        # A directly created final file can survive a host crash as zero bytes
        # and then suppress every later first-failure write. Publish only the
        # fully written, synced record; never overwrite an earlier incident.
        with _diag_tempfile.NamedTemporaryFile(
            mode="w", prefix=".decoder-failure-", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            _diag_json.dump(payload, stream)
            stream.flush()
            _diag_os.fsync(stream.fileno())
        _diag_os.link(temporary, path)
        directory_fd = _diag_os.open(path.parent, _diag_os.O_RDONLY | _diag_os.O_DIRECTORY)
        try:
            _diag_os.fsync(directory_fd)
        finally:
            _diag_os.close(directory_fd)
    except OSError:
        pass  # Evidence storage failure must not replace the CUDA exception.
    finally:
        if temporary is not None:
            try:
                _diag_os.unlink(temporary)
            except OSError:
                pass
    try:
        print("DIAG_FIRST_FAILURE " + _diag_json.dumps(payload), flush=True)
    except OSError:
        pass  # A disconnected log consumer cannot replace the first failure.

def _diag_trace(function):
    @_diag_functools.wraps(function)
    def traced(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            try:
                _diag_failure(exc)
            except Exception:
                pass  # Preserve the original exception even if formatting fails.
            raise
    return traced

def _diag_tensor_metadata(tensor):
    if tensor is None:
        return None
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device),
            "storage_offset": tensor.storage_offset(),
            "ptr_mod_16": tensor.data_ptr() % 16}


def _diag_begin_norm(layer, operation, norm, hidden_states, residual):
    if hidden_states.shape[0] <= 4 or torch.cuda.is_current_stream_capturing():
        return None
    record = {
        "layer": getattr(layer, "_diag_prefix", getattr(layer, "prefix", "layer." + str(getattr(layer, "layer_idx", "unknown")))),
        "stage": operation, "status": "pending",
        "norm_type": type(norm).__module__ + "." + type(norm).__qualname__,
        "inputs": {"hidden_states": _diag_tensor_metadata(hidden_states),
                   "residual": _diag_tensor_metadata(residual),
                   "weight": _diag_tensor_metadata(getattr(norm, "weight", None))},
    }
    _DIAG_RING.append(record)
    return record


def _diag_emit_stage(record):
    # CPU-only, bounded telemetry before/after an EXISTING synchronization.
    # The in-memory diagnostic ring alone disappears on engine-side timeout.
    try:
        from vllm.stability_telemetry import emit
        emit("decoder_sync_boundary", decoder=dict(record),
             diagnostic_scope="CPU_METADATA_EXISTING_SYNC_BOUNDARY")
    except Exception:
        # Observer failure cannot replace a model/driver exception.
        pass


def _diag_numeric_fingerprint(layer, tensor, residual):
    # Separate eager numerical diagnostic. D2H reads deliberately perturb
    # timing; these records cannot qualify throughput or unobserved decode.
    if _diag_os.environ.get("NVIDIA_NUMERIC_LAYER_TRACE") != "1":
        return
    from vllm import stability_telemetry as telemetry
    generation = telemetry._generation.get()
    if generation is None or generation <= 0:
        return  # Startup dummy passes and audit RPCs are not application input.
    import hashlib

    values = {}
    for name, value in (("hidden", tensor), ("residual", residual)):
        if value is None:
            values[name] = None
            continue
        if value.dtype != torch.bfloat16 or not 4 < value.shape[0] <= 2048:
            raise ValueError("NUMERIC_TRACE_UNDECLARED_TENSOR")
        host = value.detach().to(device="cpu").contiguous()
        raw = host.view(torch.uint8).numpy().reshape(-1)
        words = raw.view("uint16")
        values[name] = {
            "shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "source_ptr": value.data_ptr(),
            "bytes": raw.nbytes,
            "sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
            "finite": not bool(((words & 0x7F80) == 0x7F80).any()),
            "first_bf16_words": words[:8].tolist(),
            "last_bf16_words": words[-8:].tolist(),
        }
    telemetry.emit("decoder_numeric_fingerprint",
                   layer=getattr(layer, "_diag_prefix", "layer." + str(getattr(layer, "layer_idx", "unknown"))),
                   stage="after_mlp", tensors=values,
                   diagnostic_scope="FULL_CPU_BYTE_HASH_AFTER_EXISTING_SYNC")


def _diag_checkpoint(layer, stage, tensor, pending_operation=None):
    if tensor.shape[0] <= 4 or torch.cuda.is_current_stream_capturing():
        return
    record = {"layer": getattr(layer, "_diag_prefix", getattr(layer, "prefix", "layer." + str(getattr(layer, "layer_idx", "unknown")))), "stage": stage,
              "shape": list(tensor.shape), "stride": list(tensor.stride()),
              "dtype": str(tensor.dtype), "ptr_mod_16": tensor.data_ptr() % 16,
              "status": "pending"}
    _DIAG_RING.append(record)
    # One bounded fault-surfacing barrier at each decoder layer output.
    if stage != "after_mlp":
        record["status"] = "submitted"
        return
    _diag_emit_stage(record)
    torch.cuda.synchronize()
    record["status"] = "pass"
    _diag_emit_stage(record)
    if pending_operation is not None:
        pending_operation["status"] = "pass"
    from vllm.nvidia_numeric_focus import checkpoint
    checkpoint(layer, stage, hidden=tensor)


from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.v1.attention.backend import AttentionType

from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]


def _is_shared_expert_fse_compatible(quant_config) -> bool:
    """Check if shared expert can be fused with routed experts.

    FSE requires that shared and routed expert weights use the same
    quantization format. Returns False when the shared expert is
    excluded from quantization (e.g. float32 shared in an MXFP4 model)
    or has a different quant spec than routed experts.
    """
    if quant_config is None:
        return True
    # Quark stores its full config dict in quant_config.quant_config
    raw_config = getattr(quant_config, "quant_config", None)
    if not isinstance(raw_config, dict):
        return True
    exclude = raw_config.get("exclude", [])
    if not exclude:
        return True
    return not any("shared_expert." in str(e) for e in exclude)


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        _fse_requested = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        _fse_enabled = _fse_requested and _is_shared_expert_fse_compatible(quant_config)
        if _fse_requested and not _fse_enabled:
            logger.warning(
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but "
                "shared expert has a different quantization spec than routed "
                "experts. Falling back to non-fused shared expert path."
            )
        if _fse_enabled or config.shared_expert_intermediate_size <= 0:
            self.shared_expert = None
        else:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_expert",
            )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=1 if self.shared_expert is None else None,
            shared_expert_gate=self.shared_expert_gate
            if self.shared_expert is None
            else None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the MoERunner class
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(orig_shape)


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._diag_prefix = prefix
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        # Late-interaction retrieval models (e.g. ColQwen3.5) run BIDIRECTIONAL
        # attention on the full_attention layers; they set config.is_causal=False
        # via a VerifyAndUpdateConfig handler. Generation models leave is_causal
        # unset (-> causal/DECODER), so this is a no-op for them. Mirrors qwen3.py.
        attn_type = (
            AttentionType.DECODER
            if getattr(config, "is_causal", True)
            else AttentionType.ENCODER_ONLY
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Fuse the gated split + QK-RMSNorm + (partial) NeoX RoPE + gate copy.
        # TODO: support MRoPE
        mm_config = model_config.multimodal_config if model_config else None
        text_only = mm_config is None or mm_config.language_model_only
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and text_only
        )

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return post-norm, post-RoPE (q, k, v) and the pre-sigmoid gate.

        Dispatches between the fused Triton kernel and the eager
        split + QK-RMSNorm + RoPE path. ``gate`` is ``None`` when output
        gating is disabled.
        """
        if self.use_fused_qk_norm_rope_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            # mRoPE passes positions as (3, n_tokens) for T/H/W. Fusion is only
            # enabled text-only, where the three rows are identical, so taking
            # the T row is exact. (1D positions pass through.)
            pos = positions[0] if positions.ndim == 2 else positions
            q, k, gate = fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                self.q_norm.weight.float() + 1.0,
                self.k_norm.weight.float() + 1.0,
                self.rotary_emb.cos_sin_cache,
                pos,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
            )
            return q, k, v, gate

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    @_diag_trace
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _diag_checkpoint(self, "before_qkv", hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        _diag_checkpoint(self, "after_qkv", qkv)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        _diag_checkpoint(self, "after_qk_norm_rope", q)
        attn_output = self.attn(q, k, v)
        _diag_checkpoint(self, "after_attention", attn_output)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        _diag_checkpoint(self, "after_gate_before_o_proj", attn_output)
        output, _ = self.o_proj(attn_output)
        _diag_checkpoint(self, "after_o_proj", output)
        return output


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._diag_prefix = prefix

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        is_moe_layer = (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        )
        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if is_moe_layer:
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

    @_diag_trace
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if hidden_states.shape[0] > 4 and not torch.cuda.is_current_stream_capturing():
            from vllm.nvidia_numeric_focus import checkpoint
            checkpoint(self, "decoder_input", hidden=hidden_states, residual=residual)
        _diag_checkpoint(self, "decoder_begin", hidden_states)
        full_num_tokens = positions.shape[-1]
        input_is_sequence_parallel = (
            self.use_attn_reduce_scatter_for_moe
            and residual is not None
            and hidden_states.shape[0] != full_num_tokens
        )

        pending_input_norm = _diag_begin_norm(
            self, "input_layernorm", self.input_layernorm, hidden_states, residual
        )
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if input_is_sequence_parallel:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        _diag_checkpoint(self, "after_input_norm", hidden_states, pending_input_norm)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")

        _diag_checkpoint(self, "after_" + self.layer_type, hidden_states)
        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        if self.use_attn_reduce_scatter_for_moe:
            tp_world_size = get_tensor_model_parallel_world_size()
            # small trick using minus, eg. -17 % 8 = 7
            sp_pad = (-hidden_states.shape[0]) % tp_world_size
            # pad if not divisible by world size
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            if not input_is_sequence_parallel:
                residual = sequence_parallel_chunk(residual)

        # Fully Connected
        pending_post_norm = _diag_begin_norm(
            self, "post_attention_layernorm", self.post_attention_layernorm,
            hidden_states, residual,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        _diag_checkpoint(self, "after_post_norm", hidden_states, pending_post_norm)
        if self.use_attn_reduce_scatter_for_moe:
            hidden_states = self.mlp(
                hidden_states,
                already_sequence_parallel=True,
            )
        else:
            hidden_states = self.mlp(hidden_states)
        _diag_checkpoint(self, "after_mlp", hidden_states)
        if hidden_states.shape[0] > 4 and not torch.cuda.is_current_stream_capturing():
            _diag_numeric_fingerprint(self, hidden_states, residual)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


def _all_gather_hidden_and_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    full_num_tokens: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if residual is None:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[:full_num_tokens]
        return hidden_states, None

    combined_states = torch.cat([hidden_states, residual], dim=-1)
    combined_states = tensor_model_parallel_all_gather(combined_states, 0)
    combined_states = combined_states[:full_num_tokens]
    hidden_states, residual = combined_states.split([hidden_size, hidden_size], dim=-1)
    return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module, EagleModelMixin):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        full_num_tokens = positions.shape[-1]
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if (
                hidden_states.shape[0] != full_num_tokens
                and not layer.use_attn_reduce_scatter_for_moe
            ):
                hidden_states, residual = _all_gather_hidden_and_residual(
                    hidden_states,
                    residual,
                    full_num_tokens,
                    self.config.hidden_size,
                )
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if (layer_idx + 1) in self.aux_hidden_state_layers and hidden_states.shape[
                0
            ] != full_num_tokens:
                hidden_states, residual = _all_gather_hidden_and_residual(
                    hidden_states,
                    residual,
                    full_num_tokens,
                    self.config.hidden_size,
                )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if hidden_states.shape[0] != full_num_tokens:
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                full_num_tokens,
                self.config.hidden_size,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0),
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class QwenNextMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
    SupportsEagle3,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkvz"],
        "in_proj_ba": ["in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights)
