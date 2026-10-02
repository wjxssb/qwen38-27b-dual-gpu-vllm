# SPDX-License-Identifier: Apache-2.0
"""Opt-in b619 experiment: graph only MTP K3's two one-token followups."""

import json
import os
from pathlib import Path

ENV = "VLLM_SM120_MTP_DECODE_GRAPH"


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def validate_proposer(proposer) -> None:
    config = proposer.vllm_config
    parallel = config.parallel_config
    compilation = config.compilation_config
    required = {
        "target_gate": os.getenv("VLLM_SM120_NVFP4_K3_GRAPH") == "1",
        "native_gate": os.getenv("VLLM_SM120_NVFP4_K3_NATIVE") == "1",
        "method": proposer.method == "mtp",
        "k3": proposer.num_speculative_tokens == 3,
        "tp2": parallel.tensor_parallel_size == 2,
        "pp1": parallel.pipeline_parallel_size == 1,
        "dp1": parallel.data_parallel_size == 1,
        "max_seqs1": config.scheduler_config.max_num_seqs == 1,
        "sealed_prefill_chunk": config.scheduler_config.max_num_batched_tokens == 4096,
        "context262144": config.model_config.max_model_len == 262144,
        "nvfp4": config.cache_config.cache_dtype == "nvfp4",
        "kv3039750144": config.cache_config.kv_cache_memory_bytes == 3039750144,
        "prefix_on": config.cache_config.enable_prefix_caching is True,
        "mamba_align": config.cache_config.mamba_cache_mode == "align",
        "target_graph4": compilation.cudagraph_capture_sizes == [4],
        "target_mode": compilation.cudagraph_mode.name == "FULL_DECODE_ONLY",
        "compilation_none": compilation.mode.name == "NONE",
        "draft_not_eager_forced": not proposer.speculative_config.enforce_eager,
        "serial_drafting": not proposer.parallel_drafting,
        "multimodal_embedding_contract": (
            not proposer.supports_mm_inputs or (
                getattr(proposer.model, "supports_multimodal_embeddings", False)
                and getattr(proposer, "_mtp_decode_boundary_version", None) == 1
            )
        ),
        "ordinary_positions": not proposer.constant_draft_positions,
        "no_shared_indexer": not proposer._share_mtp_indices,
        "no_lora": config.lora_config is None,
        "qwen_dense_mtp": type(proposer.model).__name__ == "Qwen3_5MTP",
        "one_mtp_layer": proposer.model.model.num_mtp_layers == 1,
        "one_attention_group": len(proposer.draft_attn_groups) == 1,
        "one_attention_layer": len(proposer._draft_attn_layer_names) == 1,
    }
    failures = [name for name, valid in required.items() if not valid]
    if failures:
        raise RuntimeError("MTP decode graph envelope: " + ", ".join(failures))


def tensor_signature(values) -> tuple:
    return tuple(
        (name, value.data_ptr(), tuple(value.shape), tuple(value.stride()),
         str(value.dtype), str(value.device))
        for name, value in sorted(values.items())
        if value is not None
    )


def validate_decode_boundary(proposer, common, kwargs, draft_index, phase):
    # This phase is supplied only by the generated-token followup loop.
    # The first pass (including merged Vision embeddings) never calls this.
    if phase != "generated_token_followup" or draft_index not in (1, 2):
        raise RuntimeError("MTP graph requires a generated-token followup")
    if (common.num_reqs != 1 or common.num_actual_tokens != 1
            or common.max_query_len != 1):
        raise RuntimeError("MTP graph boundary is not one decode token")
    if set(kwargs) != {"input_ids", "positions", "inputs_embeds", "hidden_states"}:
        raise RuntimeError("MTP graph boundary fields changed")
    if tuple(kwargs["hidden_states"].shape) != (1, proposer.hidden_size):
        raise RuntimeError("MTP graph hidden shape changed")
    positions = kwargs["positions"]
    if tuple(positions.shape) not in ((1,), (3, 1)):
        raise RuntimeError("MTP graph position shape changed")
    if proposer.supports_mm_inputs:
        if kwargs["input_ids"] is not None or kwargs["inputs_embeds"] is None:
            raise RuntimeError("MTP multimodal decode embedding contract changed")
        if tuple(kwargs["inputs_embeds"].shape) != (1, proposer.inputs_embeds_size):
            raise RuntimeError("MTP graph embedding shape changed")
        expected = proposer.inputs_embeds[:1]
        if tensor_signature({"embedding": expected}) != tensor_signature({"embedding": kwargs["inputs_embeds"]}):
            raise RuntimeError("MTP graph embedding is not the owned decode buffer")
    elif kwargs["inputs_embeds"] is not None or kwargs["input_ids"] is None or tuple(kwargs["input_ids"].shape) != (1,):
        raise RuntimeError("MTP text decode input contract changed")


class SM120MTPDecodeGraph:
    """Capture at startup; refuse missing capture or changed input identities."""

    def __init__(self, proposer):
        import torch

        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor

        validate_proposer(proposer)
        self.proposer = proposer
        self.descriptor = BatchDescriptor(num_tokens=1, num_reqs=1, uniform=True)
        self.wrapper = CUDAGraphWrapper(
            proposer.model, proposer.vllm_config, CUDAGraphMode.FULL
        )
        # Target hidden states may still be live during drafting.
        self.wrapper.graph_pool = torch.cuda.graph_pool_handle()
        self.signature = None
        self.fi_wrapper = None
        self.fi_signature = None
        self.captured = False
        self.replays = 0
        self.verified_steps = set()

    def _kwargs(self):
        p = self.proposer
        return {
            "input_ids": None if p.supports_mm_inputs else p.input_ids[:1],
            "positions": p._get_positions(1),
            "inputs_embeds": p.inputs_embeds[:1] if p.supports_mm_inputs else None,
            "hidden_states": p.hidden_states[:1],
        }

    def _check_metadata(self, metadata):
        from vllm.v1.attention.backends.flashinfer import FlashInferMetadata

        if set(metadata) != set(self.proposer._draft_attn_layer_names):
            raise RuntimeError("MTP graph attention layer mismatch")
        for value in metadata.values():
            if not isinstance(value, FlashInferMetadata):
                raise RuntimeError("MTP graph requires native FlashInfer metadata")
            if (value.num_actual_tokens != 1 or value.num_decodes != 1
                    or value.num_prefills != 0 or value.decode is None
                    or value.decode.q_len_per_req != 1):
                raise RuntimeError("MTP graph accepts only one decode query")
            wrapper = value.decode.wrapper
            if not wrapper.is_cuda_graph_enabled:
                raise RuntimeError("MTP graph requires planned graph-safe decode")
            if self.fi_wrapper is None:
                self.fi_wrapper = wrapper
            elif wrapper is not self.fi_wrapper:
                raise RuntimeError("MTP graph FlashInfer wrapper identity changed")
            buffers = {
                name: getattr(wrapper, name)
                for name in ("_paged_kv_indptr_buf", "_paged_kv_indices_buf",
                             "_paged_kv_last_page_len_buf", "_float_workspace_buffer",
                             "_int_workspace_buffer")
            }
            buffers["slot_mapping"] = value.slot_mapping
            signature = tensor_signature(buffers)
            if self.fi_signature is None:
                self.fi_signature = signature
            elif signature != self.fi_signature:
                raise RuntimeError("MTP graph metadata/workspace buffers changed")

    def _call(self, metadata, kwargs, mode):
        from vllm.forward_context import set_forward_context

        self._check_metadata(metadata)
        signature = tensor_signature(kwargs)
        if self.signature is None:
            self.signature = signature
        elif self.signature != signature:
            raise RuntimeError("MTP graph input buffers changed")
        p = self.proposer
        with set_forward_context(
            metadata, p.vllm_config, num_tokens=1,
            cudagraph_runtime_mode=mode, batch_descriptor=self.descriptor,
            slot_mapping=p._get_slot_mapping(1),
        ):
            return self.wrapper(**kwargs)

    def capture(self):
        import torch

        from vllm.config import CUDAGraphMode
        from vllm.v1.attention.backend import CommonAttentionMetadata

        if self.captured:
            raise RuntimeError("MTP graph capture may run only once")
        p = self.proposer
        p.input_ids[:1].zero_()
        if p.supports_mm_inputs:
            # Identical to the existing generated-token followup loop:
            # ordinary token embeddings, no image embeddings or mask.
            p.inputs_embeds[:1] = p.model.embed_input_ids(p.input_ids[:1])
        p.hidden_states[:1].zero_()
        p._get_positions(1).zero_()
        p._slot_mapping_buffer[:1].fill_(-1)
        query_cpu = torch.tensor([0, 1], dtype=torch.int32)
        seq_cpu = torch.tensor([1], dtype=torch.int32)
        common = CommonAttentionMetadata(
            query_start_loc=query_cpu.to(p.device),
            query_start_loc_cpu=query_cpu,
            seq_lens=seq_cpu.to(p.device),
            num_reqs=1, num_actual_tokens=1, max_query_len=1, max_seq_len=1,
            block_table_tensor=torch.zeros(
                (1, (p.max_model_len + p.block_size - 1) // p.block_size),
                dtype=torch.int32, device=p.device,
            ),
            slot_mapping=p._slot_mapping_buffer[:1],
            _seq_lens_cpu=seq_cpu,
        )
        _, metadata = p.build_per_group_and_layer_attn_metadata(common, draft_index=1)
        kwargs = self._kwargs()
        torch.cuda.synchronize(p.device)
        before_allocated = torch.cuda.memory_allocated(p.device)
        before_free = torch.cuda.mem_get_info(p.device)[0]
        for _ in range(3):
            self._call(metadata, kwargs, CUDAGraphMode.NONE)
        torch.cuda.synchronize(p.device)
        self._call(metadata, kwargs, CUDAGraphMode.FULL)
        torch.cuda.synchronize(p.device)
        entry = self.wrapper.concrete_cudagraph_entries.get(self.descriptor)
        if entry is None or entry.cudagraph is None:
            raise RuntimeError("MTP decode graph was not captured")
        self.captured = True
        self._emit("captured", {
            "allocated_delta": torch.cuda.memory_allocated(p.device) - before_allocated,
            "physical_free_delta": before_free - torch.cuda.mem_get_info(p.device)[0],
            "physical_free_after": torch.cuda.mem_get_info(p.device)[0],
            "first_pass": "eager", "draft_steps": [1, 2],
        })

    def run(self, common, metadata, kwargs, draft_index, *, phase=None):
        from vllm.config import CUDAGraphMode

        validate_decode_boundary(self.proposer, common, kwargs, draft_index, phase)
        if (not self.captured or draft_index not in (1, 2)
                or self.proposer.num_speculative_tokens != 3
                or common.num_reqs != 1 or common.num_actual_tokens != 1
                or common.max_query_len != 1):
            raise RuntimeError("MTP decode replay contract changed")
        entry = self.wrapper.concrete_cudagraph_entries.get(self.descriptor)
        if entry is None or entry.cudagraph is None:
            raise RuntimeError("MTP graph missing after startup; recapture refused")
        result = self._call(metadata, kwargs, CUDAGraphMode.FULL)
        if draft_index not in self.verified_steps:
            result = self._verify_graph_vs_eager(metadata, kwargs, result, draft_index)
            self.verified_steps.add(draft_index)
        self.replays += 1
        # Bounded sparse telemetry; sampling/logits remain outside this graph.
        if self.replays <= 2 or self.replays % 256 == 0:
            self._emit("replay", {"draft_index": draft_index,
                                  "replays": self.replays,
                                  "phase": phase,
                                  "embedding_input": self.proposer.supports_mm_inputs,
                                  "first_pass_multimodal": getattr(self.proposer, "_mtp_last_first_pass_multimodal", False)})
        return result

    def _verify_graph_vs_eager(self, metadata, kwargs, graph_result, draft_index):
        import torch

        from vllm.config import CUDAGraphMode
        from vllm.distributed import get_tensor_model_parallel_rank

        # Qwen3_5MTP has one full-attention layer, no recurrent GDN state.
        # The eager call rewrites the same KV slot from identical inputs.
        graph_hidden = graph_result.detach().clone()
        eager_hidden = self._call(metadata, kwargs, CUDAGraphMode.NONE).detach().clone()
        graph_logits = self.proposer.model.compute_logits(graph_hidden).detach().clone()
        eager_logits = self.proposer.model.compute_logits(eager_hidden).detach().clone()
        raw = {name: value.detach().cpu() for name, value in kwargs.items()
               if value is not None}
        raw.update(slot_mapping=next(iter(metadata.values())).slot_mapping.detach().cpu(),
                   graph_hidden=graph_hidden.cpu(), eager_hidden=eager_hidden.cpu(),
                   graph_logits=graph_logits.cpu(), eager_logits=eager_logits.cpu())
        directory = Path(os.environ["VLLM_SM120_MTP_GRAPH_EVIDENCE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        rank = get_tensor_model_parallel_rank()
        path = directory / f'mtp-graph-rank{rank}-step{draft_index}.pt'
        with path.open('xb') as stream:
            torch.save(raw, stream)
        records = {}
        for name, left, right in (("hidden", graph_hidden, eager_hidden),
                                  ("logits", graph_logits, eager_logits)):
            a, b = left.double().flatten(), right.double().flatten()
            if a.shape != b.shape or not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
                raise RuntimeError("MTP graph/eager nonfinite or shape mismatch")
            delta = a - b
            relative = float(delta.norm() / b.norm().clamp_min(1e-30))
            cosine = float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))
            records[name] = {"relative_l2": relative, "cosine": cosine,
                             "max_abs": float(delta.abs().max()),
                             "mean_abs": float(delta.abs().mean())}
            if relative > .001 or cosine < .99999:
                raise RuntimeError("MTP graph/eager numerical mismatch: " + json.dumps(records))
        if not torch.equal(graph_logits.argmax(-1), eager_logits.argmax(-1)):
            raise RuntimeError("MTP graph/eager greedy token differs")
        self._emit("graph_vs_eager", {"draft_index": draft_index, "status": "PASS",
                   "metrics": records, "greedy_token_equal": True,
                   "artifact": str(path), "relative_l2_max": .001,
                   "cosine_min": .99999})
        return graph_hidden

    def _emit(self, event, fields):
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "VLLM_MTP_DECODE_GRAPH=%s",
            json.dumps({"event": event, "rank": get_tensor_model_parallel_rank(),
                        "batch_size": 1, "tokens": 1, "mode": "FULL", **fields},
                       sort_keys=True, separators=(",", ":")),
        )
