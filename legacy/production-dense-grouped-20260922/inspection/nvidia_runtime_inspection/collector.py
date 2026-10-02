"""Pinned vLLM worker extension. Import does not import torch or initialize CUDA.

This collector never executes a forward, changes a tensor, resets a cache,
performs a distributed data collective, or issues a qualification receipt.
Scalar device-to-host copies synchronize CUDA and can expose existing failures.
Invoke only in an owned, quiescent candidate window, outside timed measurements.
"""
import dataclasses
import enum
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

IDENTITY_KEYS = ("instance_id", "boot_id", "model_revision", "runtime_manifest_sha256")
MODEL_REVISION = "dbb8f445b3145f8a4c18ddc769f032d57d32867c"
METHOD = "nvidia_collect_runtime_mapping_v1"
MODULE_ATTRS = (
    "input_size", "output_size", "input_size_per_partition", "output_size_per_partition",
    "output_partition_sizes", "output_sizes", "logical_widths", "tp_size", "tp_rank",
    "disable_tp", "disable_tp_for_ba_proj", "gather_output", "input_is_parallel",
    "num_heads", "num_kv_heads", "num_k_heads", "num_v_heads", "head_size",
    "head_k_dim", "head_v_dim", "conv_kernel_size", "num_spec", "hidden_size",
    "layer_norm_epsilon", "variance_epsilon", "eps", "gqa_interleaved_layout",
    "gdn_prefill_backend", "enable_packed_recurrent_decode", "kv_cache_dtype",
    "weights_padding_cols", "orig_vocab_size", "org_vocab_size", "org_vocab_size_padded",
    "num_embeddings", "num_embeddings_padded", "num_embeddings_per_partition",
    "shard_indices", "_k_scale_float", "_v_scale_float", "calculate_kv_scales",
)
PARAM_ATTRS = ("input_dim", "output_dim", "packed_dim", "packed_factor", "marlin_tile_size", "tp_rank", "tp_size")
SCALE_ATTRS = ("_k_scale", "_v_scale", "k_scale", "v_scale", "input_scale", "weight_scale",
               "weight_scale_2", "input_global_scale", "weight_global_scale", "alpha",
               "input_global_scale_inv")


def class_name(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def json_value(value, depth=0):
    """Bounded, explicit conversion; never repr a model/config containing secrets."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, enum.Enum):
        return {"enum": class_name(value), "name": value.name, "value": json_value(value.value)}
    if depth >= 8:
        return {"unrecorded_type": class_name(value), "reason": "depth_limit"}
    if isinstance(value, (tuple, list)):
        if len(value) > 8192:
            raise ValueError("metadata list exceeds inspection bound")
        return [json_value(x, depth + 1) for x in value]
    if isinstance(value, dict):
        if len(value) > 8192:
            raise ValueError("metadata dict exceeds inspection bound")
        return {str(k): json_value(v, depth + 1) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        # Do not use asdict: it deep-copies tensors and can allocate GPU memory.
        return {"class": class_name(value), **{
            f.name: json_value(getattr(value, f.name), depth + 1)
            for f in dataclasses.fields(value)}}
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device", "Size"):
        return str(value)
    return {"unrecorded_type": class_name(value)}


def selected_attrs(obj, names):
    return {name: json_value(getattr(obj, name)) for name in names if hasattr(obj, name)}


def validate_request(raw):
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 16384:
        raise ValueError("inspection request must be a JSON string of at most 16384 bytes")
    row = json.loads(raw)
    expected = {"identity", "request_id", "read_scalars", "quiescent_asserted"}
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError("unexpected request fields")
    identity = row["identity"]
    if not isinstance(identity, dict) or set(identity) != set(IDENTITY_KEYS):
        raise ValueError("identity fields mismatch")
    if not all(isinstance(identity[k], str) and identity[k] for k in IDENTITY_KEYS):
        raise ValueError("identity requires nonempty strings")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", identity["instance_id"]):
        raise ValueError("invalid instance_id")
    if not re.fullmatch(r"[0-9a-f-]{36}", identity["boot_id"]):
        raise ValueError("invalid boot_id")
    if identity["model_revision"] != MODEL_REVISION:
        raise ValueError("wrong checkpoint revision")
    if not re.fullmatch(r"[0-9a-f]{64}", identity["runtime_manifest_sha256"]):
        raise ValueError("invalid runtime manifest")
    if not isinstance(row["request_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", row["request_id"]):
        raise ValueError("invalid request_id")
    if type(row["read_scalars"]) is not bool or row["quiescent_asserted"] is not True:
        raise ValueError("explicit quiescent assertion and boolean read_scalars required")
    return row


def verified_binding(request):
    # Binding is supplied by the isolated launcher's sealed per-run environment.
    # It is identity evidence, not independent host Docker/label verification.
    binding = json.loads(os.environ["NVIDIA_AUDIT_BINDING_JSON"])
    if not isinstance(binding, dict) or set(binding) != set(IDENTITY_KEYS):
        raise ValueError("invalid configured audit binding")
    if binding != request["identity"]:
        raise ValueError("request does not name this candidate run")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != binding["boot_id"]:
        raise ValueError("audit binding belongs to a different boot")
    return binding


def tensor_record(tensor, torch_module, read_scalars=False):
    record = {
        "class": class_name(tensor), "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "device": str(tensor.device), "numel": int(tensor.numel()),
        "logical_bytes": int(tensor.numel() * tensor.element_size()),
        "stride": list(tensor.stride()), "storage_offset": int(tensor.storage_offset()),
        "requires_grad": bool(tensor.requires_grad),
        "loader_attributes": selected_attrs(tensor, PARAM_ATTRS),
        "content_status": "NOT_COLLECTED",
    }
    loader = getattr(tensor, "weight_loader", None)
    record["weight_loader"] = (getattr(loader, "__module__", "") + "." + getattr(loader, "__qualname__", class_name(loader))) if loader is not None else None
    if getattr(tensor, "is_meta", False):
        record["storage"] = None
        return record
    storage = tensor.untyped_storage()
    record["storage"] = {"data_ptr": int(storage.data_ptr()), "bytes": int(storage.nbytes())}
    if read_scalars and tensor.numel() <= 16:
        # Copy first, cast on CPU: do not schedule a GPU conversion/reduction kernel.
        values = tensor.detach().cpu().to(dtype=torch_module.float64).reshape(-1).tolist()
        record["values"] = [json_value(float(x)) for x in values]
        record["all_finite"] = all(math.isfinite(float(x)) for x in values)
        record["content_status"] = "SMALL_TENSOR_VALUES_COLLECTED"
    return record


def cache_tree(value, torch_module):
    if isinstance(value, torch_module.Tensor):
        return tensor_record(value, torch_module)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [cache_tree(x, torch_module) for x in value]
    if isinstance(value, dict):
        return {str(k): cache_tree(v, torch_module) for k, v in value.items()}
    return {"unrecorded_type": class_name(value)}


def unique_storage_bytes(records):
    """Account aliases once. This is live storage, not allocator reserved VRAM."""
    stores = {}
    for row in records:
        storage = row.get("storage")
        if storage:
            key = (row["device"], storage["data_ptr"])
            stores[key] = max(stores.get(key, 0), storage["bytes"])
    return sum(stores.values())


def inspect_worker(worker, request, torch_module, tp_group):
    started = time.time_ns()
    runner = worker.model_runner
    model = runner.get_model()
    config = worker.vllm_config
    parameters = {}
    buffers = {}
    for name, tensor in model.named_parameters(recurse=True, remove_duplicate=False):
        parameters[name] = tensor_record(tensor, torch_module,
                                         request["read_scalars"] and "scale" in name)
    for name, tensor in model.named_buffers(recurse=True, remove_duplicate=False):
        buffers[name] = tensor_record(tensor, torch_module,
                                      request["read_scalars"] and "scale" in name)
    modules = {}
    for name, module in model.named_modules(remove_duplicate=False):
        row = {"class": class_name(module), "attributes": selected_attrs(module, MODULE_ATTRS)}
        quant = getattr(module, "quant_method", None)
        row["quant_method_class"] = class_name(quant) if quant is not None else None
        if quant is not None:
            row["quant_method_attributes"] = selected_attrs(quant, ("backend", "input_dtype", "out_dtype", "weight_block_size"))
            row["quant_config"] = selected_attrs(getattr(quant, "quant_config", None),
                                                 ("quant_method", "group_size", "kv_cache_quant_method", "is_checkpoint_nvfp4_serialized", "is_checkpoint_fp8_serialized"))
            row["kernels"] = {attr: class_name(getattr(quant, attr))
                              for attr in ("kernel", "fp8_linear") if hasattr(quant, attr)}
        # Attention scales can be unregistered attributes. Include actual tensors.
        row["scales"] = {attr: tensor_record(getattr(module, attr), torch_module, request["read_scalars"])
                         for attr in SCALE_ATTRS
                         if isinstance(getattr(module, attr, None), torch_module.Tensor)}
        modules[name] = row
    static = getattr(config.compilation_config, "static_forward_context", {})
    static_context = {}
    for name, module in static.items():
        static_context[name] = {
            "class": class_name(module), "attributes": selected_attrs(module, MODULE_ATTRS),
            "scales": {attr: tensor_record(getattr(module, attr), torch_module, request["read_scalars"])
                       for attr in SCALE_ATTRS
                       if isinstance(getattr(module, attr, None), torch_module.Tensor)},
            "kv_cache": cache_tree(getattr(module, "kv_cache", None), torch_module),
        }
    device = worker.device
    props = torch_module.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    communicator = getattr(tp_group, "device_communicator", None)
    graph = getattr(runner, "cudagraph_dispatcher", None)
    result = {
        "schema": 1, "scope": "ACTUAL_RUNTIME_OBSERVATION_ONLY", "status": "COLLECTED_UNQUALIFIED",
        "numerical_correctness": "NOT_PROVEN", "p2p_data_integrity": "NOT_PROVEN",
        "request_id": request["request_id"], "identity": request["identity"],
        "started_unix_ns": started, "pid": os.getpid(), "hostname": os.uname().nodename,
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "worker_class": class_name(worker), "runner_class": class_name(runner), "model_class": class_name(model),
        "rank": worker.rank, "local_rank": worker.local_rank,
        "tensor_parallel": {**selected_attrs(tp_group, ("rank", "rank_in_group", "world_size", "ranks", "local_rank")),
                            "communicator_class": class_name(communicator) if communicator else None,
                            "communicator_components": {attr: class_name(getattr(communicator, attr))
                                for attr in ("pynccl_comm", "ca_comm", "all_reduce")
                                if getattr(communicator, attr, None) is not None}},
        "device": {"logical_device": str(device), "name": props.name,
                   "uuid": str(getattr(props, "uuid", "UNAVAILABLE")),
                   "capability": [props.major, props.minor], "total_bytes": total_bytes,
                   "free_bytes": free_bytes, "allocated_bytes": torch_module.cuda.memory_allocated(device),
                   "reserved_bytes": torch_module.cuda.memory_reserved(device)},
        "effective_config": {
            "model": selected_attrs(config.model_config, ("model", "dtype", "max_model_len", "served_model_name", "revision", "quantization", "enforce_eager", "is_multimodal_model", "is_encoder_decoder")),
            "parallel": selected_attrs(config.parallel_config, ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "worker_extension_cls", "disable_custom_all_reduce")),
            "cache": selected_attrs(config.cache_config, ("cache_dtype", "block_size", "kv_cache_memory_bytes", "enable_prefix_caching", "mamba_cache_dtype", "mamba_ssm_cache_dtype", "mamba_cache_mode", "num_gpu_blocks", "calculate_kv_scales")),
            "scheduler": selected_attrs(config.scheduler_config, ("max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill")),
            "compilation": selected_attrs(config.compilation_config, ("cudagraph_mode", "cudagraph_capture_sizes", "mode", "backend")),
            "multimodal": selected_attrs(getattr(config.model_config, "multimodal_config", None), ("mm_encoder_tp_mode", "limit_per_prompt", "language_model_only", "mm_processor_cache_gb")),
            "speculative_config_is_none": config.speculative_config is None,
            "speculative_config_class": class_name(config.speculative_config) if config.speculative_config is not None else None,
            "runner_num_spec_tokens": getattr(runner, "num_spec_tokens", None),
            "runner_drafter_class": class_name(runner.drafter) if getattr(runner, "drafter", None) is not None else None,
        },
        "parameters": parameters, "buffers": buffers, "modules": modules,
        "parameter_unique_storage_bytes": unique_storage_bytes(parameters.values()),
        "buffer_unique_storage_bytes": unique_storage_bytes(buffers.values()),
        "kv_cache_config": json_value(getattr(runner, "kv_cache_config", None)),
        "kv_cache_tensors": cache_tree(getattr(runner, "kv_caches", None), torch_module),
        "static_forward_context": static_context,
        "graph_dispatcher_class": class_name(graph) if graph is not None else None,
        "runner_observed": selected_attrs(runner, ("model_memory_usage", "cudagraph_batch_sizes", "use_mrope", "use_xdrope")),
        "limitations": ["Identity environment is not an independent Docker ownership proof; verify host inspect before and after collection.",
                        "No full weight/block-scale content scan, repack inversion, or TP shard reconstruction.",
                        "No forward, numerical comparison, P2P integrity test, recurrent-state content inspection, or graph pointer-lifetime test.",
                        "Vision tensor residency is not Vision semantic acceptance.",
                        "Logical and unique storage bytes are not total process VRAM or peak allocation.",
                        "Quiescence is caller asserted; this collector does not inspect the scheduler request queue."],
    }
    drafter = getattr(runner, "drafter", None)
    draft = getattr(drafter, "model", None)
    graph = getattr(drafter, "_sm120_mtp_decode_graph", None)
    spec = config.speculative_config
    result["mtp"] = {
        "method": getattr(spec, "method", None), "k": getattr(spec, "num_speculative_tokens", None),
        "use_v2_model_runner": config.use_v2_model_runner,
        "target_model": config.model_config.model, "draft_model": getattr(spec, "model", None),
        "draft_model_class": class_name(draft) if draft is not None else None,
        "draft_tp": getattr(getattr(spec, "draft_parallel_config", None), "tensor_parallel_size", None),
        "rejection_method": getattr(spec, "rejection_sample_method", None),
        "graph_captured": getattr(graph, "captured", False), "graph_replays": getattr(graph, "replays", 0),
        "graph_verified_steps": sorted(getattr(graph, "verified_steps", [])),
        "boundary_version": getattr(drafter, "_mtp_decode_boundary_version", None),
        "supports_mm_inputs": getattr(drafter, "supports_mm_inputs", None),
        "first_pass_counts": getattr(drafter, "_mtp_first_pass_counts", {}),
        "draft_parameters": {n: tensor_record(t, torch_module, False) for n,t in draft.named_parameters()} if draft is not None else {},
    }
    torch_module.cuda.synchronize(device)
    result["finished_unix_ns"] = time.time_ns()
    return result


class InspectionWorkerExtension:
    def nvidia_collect_runtime_mapping_v1(self, request_json: str):
        request = validate_request(request_json)
        verified_binding(request)
        import torch
        from vllm.distributed.parallel_state import get_tp_group
        return inspect_worker(self, request, torch, get_tp_group())

    def nvidia_collect_transport_checksums_v1(self, request_json: str):
        from .transport import run
        return run(self, request_json)

    def nvidia_collect_tensor_samples_v1(self, request_json: str):
        from .sampler import run
        return run(self, request_json)
