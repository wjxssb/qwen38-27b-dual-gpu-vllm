"""Bounded exact-coordinate scalar views; no GPU flatten or full-tensor copy."""
import hashlib
import json
import math
import re
from pathlib import Path
from .collector import validate_request, verified_binding

METHOD = "nvidia_collect_tensor_samples_v1"
MAX_BYTES = 65536


def validate_sample_request(raw):
    if not isinstance(raw, str) or len(raw.encode()) > MAX_BYTES:
        raise ValueError("sample request exceeds 64 KiB")
    row = json.loads(raw)
    if not isinstance(row, dict) or set(row) != {"identity", "request_id", "quiescent_asserted", "samples"}:
        raise ValueError("unexpected sample request fields")
    validate_request(json.dumps({k: v for k, v in row.items() if k != "samples"} | {"read_scalars": False}))
    samples = row["samples"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= 128:
        raise ValueError("sample plan requires 1..128 entries")
    ids = set()
    total = 0
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {"sample_id", "rank", "tensor_name", "expected_shape", "expected_dtype", "coordinates"}:
            raise ValueError("unexpected sample plan fields")
        if not isinstance(sample["sample_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", sample["sample_id"]) or sample["sample_id"] in ids:
            raise ValueError("invalid or duplicate sample ID")
        ids.add(sample["sample_id"])
        if type(sample["rank"]) is not int or sample["rank"] not in (0, 1):
            raise ValueError("sample rank must be 0 or 1")
        if not isinstance(sample["tensor_name"], str) or not 1 <= len(sample["tensor_name"]) <= 512:
            raise ValueError("invalid tensor name")
        if not isinstance(sample["expected_dtype"], str) or sample["expected_dtype"] not in {"torch.uint8", "torch.int8", "torch.bfloat16", "torch.float16", "torch.float32", "torch.float64", "torch.float8_e4m3fn"}:
            raise ValueError("unreviewed sample dtype")
        shape = sample["expected_shape"]
        if not isinstance(shape, list) or len(shape) > 8 or not all(type(d) is int and 0 < d <= 2**32 for d in shape):
            raise ValueError("invalid expected shape")
        coordinates = sample["coordinates"]
        if not isinstance(coordinates, list) or not 1 <= len(coordinates) <= 32:
            raise ValueError("1..32 coordinates per entry required")
        for coord in coordinates:
            if not isinstance(coord, list) or len(coord) != len(shape) or not all(type(i) is int and 0 <= i < size for i, size in zip(coord, shape)):
                raise ValueError("coordinate does not fit expected shape")
        total += len(coordinates)
    if total > 2048:
        raise ValueError("at most 2048 scalar coordinates per request")
    return row


def resolve_samples(model, request, rank):
    params = dict(model.named_parameters(recurse=True, remove_duplicate=False))
    buffers = dict(model.named_buffers(recurse=True, remove_duplicate=False))
    if set(params) & set(buffers): raise ValueError("parameter/buffer name collision")
    tensors = params | buffers
    resolved = []
    for sample in request["samples"]:
        if sample["rank"] != rank: continue
        tensor = tensors.get(sample["tensor_name"])
        if tensor is None: raise ValueError("sample tensor missing: " + sample["tensor_name"])
        if list(tensor.shape) != sample["expected_shape"] or str(tensor.dtype) != sample["expected_dtype"]:
            raise ValueError("sample tensor actual shape/dtype differs: " + sample["tensor_name"])
        if not str(tensor.device).startswith("cuda:") or getattr(tensor, "is_meta", False):
            raise ValueError("sample tensor must be actually CUDA resident")
        resolved.append((sample, tensor))
    return resolved


def inspect_samples(worker, request, torch_module, group):
    if group.world_size != 2 or group.rank_in_group != worker.rank:
        raise ValueError("actual TP rank mismatch")
    # Resolve every tensor/shape/dtype first. The parser checked every coordinate.
    resolved = resolve_samples(worker.model_runner.get_model(), request, worker.rank)
    results = []
    for sample, tensor in resolved:
        points = []
        for coordinate in sample["coordinates"]:
            scalar = tensor.detach()[tuple(coordinate)].cpu()
            # The only reshape/view happens on the scalar CPU tensor.
            raw = bytes(scalar.reshape(1).view(torch_module.uint8).tolist())
            number = float(scalar.to(dtype=torch_module.float64).item())
            points.append({"coordinate": coordinate, "raw_hex": raw.hex(),
                           "decoded_value": number if math.isfinite(number) else {"nonfinite": str(number)}})
        results.append({"sample_id": sample["sample_id"], "rank": worker.rank, "tensor_name": sample["tensor_name"],
                        "shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
                        "stride": list(tensor.stride()), "points": points})
    props = torch_module.cuda.get_device_properties(worker.device)
    torch_module.cuda.synchronize(worker.device)
    return {"schema": 1, "scope": "EXACT_COORDINATE_SCALAR_SAMPLES_ONLY", "status": "COLLECTED_UNQUALIFIED",
            "identity": request["identity"], "request_id": request["request_id"], "rank": worker.rank,
            "device": {"uuid": str(getattr(props, "uuid", "UNAVAILABLE")), "logical_device": str(worker.device)},
            "sampler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "samples": results,
            "full_tensor_content_mapping": "NOT_PROVEN", "model_numerical_correctness": "NOT_PROVEN"}


def run(worker, raw):
    request = validate_sample_request(raw)
    verified_binding(request)
    import torch
    from vllm.distributed.parallel_state import get_tp_group
    return inspect_samples(worker, request, torch, get_tp_group())
