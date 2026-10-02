"""Fixed-size bidirectional checksum test of the already active TP PyNCCL link."""
import hashlib
import inspect
import json
import random
import time
from pathlib import Path

from .collector import class_name, validate_request, verified_binding

METHOD = "nvidia_collect_transport_checksums_v1"
PROTOCOL = "TP2_PYNCCL_BROADCAST_32X2X1M_V1"
ROUNDS = 32
PAYLOAD_BYTES = 1024 * 1024
BASE_SEED = 20260913
EXPECTED_SOURCES = {
    "group": {"class": "vllm.distributed.parallel_state.GroupCoordinator", "sha256": "c6d65b96a260ea0d779da1073d1799392813eee74c9ed44753dfa56f4e7a9e4c"},
    "communicator": {"class": "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator", "sha256": "a9a0242175c3d3e0d46d586a38cb76139e6e4e92d15032d5b1e33237af2f6757"},
    "pynccl": {"class": "vllm.distributed.device_communicators.pynccl.PyNcclCommunicator", "sha256": "28596817ee9fb842026624a8a6c32b0637630b6b908ddb2a79ecccbeac531e35"},
}


def validate_transport_request(raw):
    if not isinstance(raw, str) or len(raw.encode()) > 16384:
        raise ValueError("transport request exceeds bound")
    row = json.loads(raw)
    if not isinstance(row, dict) or set(row) != {"identity", "request_id", "quiescent_asserted", "protocol"}:
        raise ValueError("unexpected transport request fields")
    if row["protocol"] != PROTOCOL:
        raise ValueError("only the fixed transport protocol is supported")
    validate_request(json.dumps({k: v for k, v in row.items() if k != "protocol"} | {"read_scalars": False}))
    return row


def payload(round_index, source_rank):
    seed = BASE_SEED + 2 * round_index + source_rank
    pattern = ("zeros", "ones", "walking_byte", "seeded_random")[round_index % 4]
    if pattern == "zeros": data = bytes(PAYLOAD_BYTES)
    elif pattern == "ones": data = b"\xff" * PAYLOAD_BYTES
    elif pattern == "walking_byte":
        data = bytes((x + seed) % 256 for x in range(256)) * (PAYLOAD_BYTES // 256)
    else: data = random.Random(seed).randbytes(PAYLOAD_BYTES)
    return data, seed, pattern


def source_hash(obj):
    path = Path(inspect.getsourcefile(type(obj)))
    return {"class": class_name(obj), "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inspect_transport(worker, request, torch_module, group):
    if group.world_size != 2 or group.ranks != [0, 1] or group.rank_in_group not in (0, 1):
        raise ValueError("transport audit requires actual TP ranks [0,1]")
    comm = group.device_communicator
    pynccl = getattr(comm, "pynccl_comm", None)
    if pynccl is None or pynccl.disabled or not callable(getattr(comm, "broadcast", None)):
        raise ValueError("enabled existing TP PyNCCL broadcast is required; no fallback")
    if pynccl.rank != group.rank_in_group or pynccl.world_size != 2:
        raise ValueError("PyNCCL rank/world identity mismatch")
    sources = {"group": source_hash(group), "communicator": source_hash(comm), "pynccl": source_hash(pynccl)}
    for name, expected in EXPECTED_SOURCES.items():
        if any(sources[name].get(key) != value for key, value in expected.items()):
            raise ValueError("transport source differs from reviewed pinned implementation: " + name)
    # Exactly one explicit 1 MiB uint8 allocation per rank, reused for all 64
    # broadcasts. NCCL/PyTorch internal allocator reservations are not bounded here.
    tensor = torch_module.empty((PAYLOAD_BYTES,), dtype=torch_module.uint8, device=worker.device)
    props = torch_module.cuda.get_device_properties(worker.device)
    reports = []
    started = time.time_ns()
    for round_index in range(ROUNDS):
        for source_rank in (0, 1):
            reference, seed, pattern = payload(round_index, source_rank)
            reference_hash = hashlib.sha256(reference).hexdigest()
            is_source = group.rank_in_group == source_rank
            # Every destination is poisoned with the bitwise opposite pattern;
            # stale allocation contents cannot accidentally satisfy this round.
            initial = reference if is_source else bytes(x ^ 255 for x in reference)
            cpu = torch_module.frombuffer(bytearray(initial), dtype=torch_module.uint8)
            tensor.copy_(cpu)
            before = tensor.detach().cpu().numpy().tobytes()
            before_hash = hashlib.sha256(before).hexdigest()
            if pynccl.disabled:
                raise ValueError("PyNCCL unexpectedly disabled during fixed protocol")
            comm.broadcast(tensor, src=source_rank)
            torch_module.cuda.synchronize(worker.device)
            observed = tensor.detach().cpu().numpy().tobytes()
            reports.append({"round": round_index, "source_rank": source_rank, "destination_rank": 1 - source_rank,
                            "observed_rank": group.rank_in_group, "payload_bytes": PAYLOAD_BYTES,
                            "seed": seed, "pattern": pattern, "source_cpu_sha256": reference_hash,
                            "before_device_sha256": before_hash, "before_device_equals_initial": before == initial,
                            "source_device_before_equals_reference": (before == reference) if is_source else None,
                            "destination_before_is_poison": (before != reference and before == initial) if not is_source else None,
                            "after_device_sha256": hashlib.sha256(observed).hexdigest(),
                            "after_device_equals_source_cpu": observed == reference})
            # Complete the fixed sequence even on byte mismatch: asymmetric early
            # return would leave the peer waiting in a collective. CUDA/RPC errors
            # propagate and are handled as unknown outcomes by the supervisor.
    del tensor
    torch_module.cuda.synchronize(worker.device)
    matches = all(r["before_device_equals_initial"] and r["after_device_equals_source_cpu"] for r in reports)
    return {"schema": 1, "scope": "TP_PYNCCL_TRANSPORT_CHECKSUMS_ONLY",
            "status": "LOCAL_CHECKSUMS_MATCH" if matches else "CHECKSUM_MISMATCH",
            "identity": request["identity"], "request_id": request["request_id"],
            "rank": worker.rank, "local_rank": worker.local_rank,
            "tensor_parallel": {"rank_in_group": group.rank_in_group, "world_size": group.world_size, "ranks": group.ranks},
            "device": {"uuid": str(getattr(props, "uuid", "UNAVAILABLE")), "logical_device": str(worker.device)},
            "protocol": PROTOCOL, "rounds": ROUNDS, "directions": 2, "payload_bytes": PAYLOAD_BYTES,
            "explicit_tensor_bytes_per_rank": PAYLOAD_BYTES, "base_seed": BASE_SEED,
            "transport_method": "existing_tp_group.device_communicator.broadcast -> existing_pynccl_comm.broadcast -> ncclBroadcast",
            "transport_source": sources, "transport_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "started_unix_ns": started, "finished_unix_ns": time.time_ns(), "observations": reports,
            "model_numerical_correctness": "NOT_PROVEN", "physical_p2p_routing": "NOT_PROVEN",
            "limitations": ["Checksum equality proves only this fixed broadcast byte-transport workload.",
                            "It does not establish actual NCCL channel P2P/CUMEM routing; correlate current NCCL logs.",
                            "It does not qualify custom all-reduce, model kernels, inference, state lifetime, or long-duration stability.",
                            "Only explicit test-tensor allocation is bounded; existing NCCL allocator behavior is separate."]}


def run(worker, raw):
    request = validate_transport_request(raw)
    verified_binding(request)
    import torch
    from vllm.distributed.parallel_state import get_tp_group
    return inspect_transport(worker, request, torch, get_tp_group())


def compare(results):
    """Require both source/destination reports and independently rebuild payloads."""
    findings = []
    if not isinstance(results, list) or len(results) != 2 or {x.get("rank") for x in results} != {0, 1}:
        return {"status": "FAIL", "scope": "TP_PYNCCL_TRANSPORT_CHECKSUMS_ONLY", "findings": ["need two distinct rank reports"]}
    if results[0].get("identity") != results[1].get("identity") or results[0].get("request_id") != results[1].get("request_id"):
        findings.append("rank request identity mismatch")
    uuids = [r.get("device", {}).get("uuid") for r in results]
    if len(set(uuids)) != 2 or any(x in (None, "UNAVAILABLE") for x in uuids): findings.append("actual distinct UUIDs missing")
    for rank in results:
        if rank.get("scope") != "TP_PYNCCL_TRANSPORT_CHECKSUMS_ONLY" or rank.get("protocol") != PROTOCOL or rank.get("status") != "LOCAL_CHECKSUMS_MATCH":
            findings.append("scope/protocol/local checksum failure")
        if rank.get("tensor_parallel") != {"rank_in_group": rank["rank"], "world_size": 2, "ranks": [0, 1]}:
            findings.append("actual TP group identity mismatch")
        for name, expected in EXPECTED_SOURCES.items():
            if any(rank.get("transport_source", {}).get(name, {}).get(key) != value for key, value in expected.items()):
                findings.append("transport source hash/class mismatch: " + name)
        if rank.get("rounds") != ROUNDS or rank.get("directions") != 2 or rank.get("payload_bytes") != PAYLOAD_BYTES:
            findings.append("protocol allocation/iteration bounds mismatch")
        observed = rank.get("observations", [])
        if len(observed) != ROUNDS * 2:
            findings.append("incomplete fixed protocol")
            continue
        for index, row in enumerate(observed):
            round_index, source = divmod(index, 2)
            data, seed, pattern = payload(round_index, source)
            expected = hashlib.sha256(data).hexdigest()
            is_source = rank["rank"] == source
            initial = data if is_source else bytes(x ^ 255 for x in data)
            wanted = {"round": round_index, "source_rank": source, "destination_rank": 1-source,
                      "observed_rank": rank["rank"], "payload_bytes": PAYLOAD_BYTES,
                      "seed": seed, "pattern": pattern, "source_cpu_sha256": expected,
                      "before_device_sha256": hashlib.sha256(initial).hexdigest(), "before_device_equals_initial": True,
                      "source_device_before_equals_reference": True if is_source else None,
                      "destination_before_is_poison": None if is_source else True,
                      "after_device_sha256": expected, "after_device_equals_source_cpu": True}
            if row != wanted: findings.append(f"rank {rank['rank']} round {round_index} src {source}: checksum/identity mismatch")
    return {"schema": 1, "status": "FAIL" if findings else "PASS", "scope": "TP_PYNCCL_TRANSPORT_CHECKSUMS_ONLY",
            "findings": findings, "model_numerical_correctness": "NOT_PROVEN", "physical_p2p_routing": "NOT_PROVEN"}
