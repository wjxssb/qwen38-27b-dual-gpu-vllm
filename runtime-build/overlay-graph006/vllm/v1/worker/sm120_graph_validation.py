# SPDX-License-Identifier: Apache-2.0
"""Validation-only state snapshots for the SM120 K3 graph campaign."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any

import torch

from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

_VALIDATION_DIR = os.getenv("VLLM_SM120_GRAPH_VALIDATION_DIR")
_VALIDATION_CASE = os.getenv("VLLM_SM120_GRAPH_VALIDATION_CASE", "unspecified")


def enabled() -> bool:
    return bool(_VALIDATION_DIR)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _index_values(tensor: torch.Tensor | None) -> list[int]:
    if tensor is None:
        return []
    values = tensor.detach().reshape(-1).cpu().tolist()
    return sorted({int(value) for value in values if int(value) >= 0})


def _gdn_indices(metadata: GDNAttentionMetadata) -> list[int]:
    values: set[int] = set()
    for tensor in (
        metadata.spec_state_indices_tensor,
        metadata.non_spec_state_indices_tensor,
        metadata.prefill_state_indices,
    ):
        values.update(_index_values(tensor))
    return sorted(values)


def _phase(attn_metadata: Any) -> str | None:
    if not isinstance(attn_metadata, dict):
        return None
    gdn = next(
        (
            value
            for value in attn_metadata.values()
            if isinstance(value, GDNAttentionMetadata)
        ),
        None,
    )
    if gdn is None:
        return None
    if gdn.num_prefills > 0:
        return "prefill"
    if (
        gdn.num_decodes == 0
        and gdn.num_spec_decodes == 1
        and gdn.num_spec_decode_tokens == 4
    ):
        return "verify"
    return None


def _slot_record(
    slot_mappings: Any,
    target_layer_names: set[str],
    num_actual_tokens: int,
) -> dict[str, Any]:
    tensors: list[tuple[str, torch.Tensor]] = []
    if isinstance(slot_mappings, dict):
        tensors.extend(
            (name, value)
            for name, value in slot_mappings.items()
            if name in target_layer_names and torch.is_tensor(value)
        )
    elif isinstance(slot_mappings, list):
        for index, entry in enumerate(slot_mappings):
            if isinstance(entry, dict):
                tensors.extend(
                    (f"{name}@ubatch{index}", value)
                    for name, value in entry.items()
                    if name in target_layer_names and torch.is_tensor(value)
                )
    records = []
    for name, tensor in tensors:
        values = [
            int(value)
            for value in tensor[:num_actual_tokens].detach().cpu().reshape(-1).tolist()
            if int(value) >= 0
        ]
        records.append(
            {
                "layer": name,
                "count": len(values),
                "unique_count": len(set(values)),
                "sha256": hashlib.sha256(
                    json.dumps(values, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    return {"groups": records}


def _snapshot(
    runner: Any,
    *,
    checkpoint: str,
    attn_metadata: Any,
    slot_mappings: Any,
    num_actual_tokens: int,
    runtime_mode: str,
    case_index: int,
) -> dict[str, Any]:
    layers = runner.compilation_config.static_forward_context
    draft_layer_names = set(
        getattr(getattr(runner, "drafter", None), "_draft_attn_layer_names", ())
    )
    gdn_states: dict[str, Any] = {}
    attention_outputs: dict[str, Any] = {}
    for name, layer in sorted(layers.items()):
        if name in draft_layer_names:
            continue
        metadata = attn_metadata.get(name) if isinstance(attn_metadata, dict) else None
        states = getattr(layer, "kv_cache", None)
        if isinstance(metadata, GDNAttentionMetadata) and isinstance(states, tuple):
            indices = _gdn_indices(metadata)
            state_records = []
            for state in states:
                active = state[indices] if indices else state[:0]
                state_records.append(
                    {
                        "shape": list(active.shape),
                        "dtype": str(active.dtype),
                        "sha256": _tensor_sha256(active),
                    }
                )
            gdn_states[name] = {"state_block_indices": indices, "states": state_records}
            continue
        impl = getattr(layer, "impl", None)
        validation_route = "graph" if runtime_mode == "FULL" else "eager"
        validation_records = getattr(impl, "_sm120_graph_validation_records", {})
        validation_record = (
            validation_records.get(validation_route, {})
            if isinstance(validation_records, dict)
            else {}
        )
        output = validation_record.get("output")
        if torch.is_tensor(output):
            impl_slots = validation_record.get("slot_mapping")
            slot_values = (
                [
                    int(value)
                    for value in impl_slots.detach().cpu().reshape(-1).tolist()
                    if int(value) >= 0
                ]
                if torch.is_tensor(impl_slots)
                else []
            )
            attention_outputs[name] = {
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "sha256": _tensor_sha256(output),
                "num_decodes": validation_record.get("num_decodes"),
                "q_len_per_req": validation_record.get("q_len_per_req"),
                "kv_write_tokens": validation_record.get("kv_write_tokens"),
                "kv_slot_count": len(slot_values),
                "kv_slot_unique_count": len(set(slot_values)),
                "kv_slot_sha256": hashlib.sha256(
                    json.dumps(slot_values, separators=(",", ":")).encode()
                ).hexdigest(),
            }

    return {
        "schema_version": 1,
        "case": _VALIDATION_CASE,
        "case_index": case_index,
        "rank": get_tensor_model_parallel_rank(),
        "checkpoint": checkpoint,
        "runtime_mode": runtime_mode,
        "num_actual_tokens": num_actual_tokens,
        "gdn_layer_count": len(gdn_states),
        "full_attention_layer_count": len(attention_outputs),
        "gdn_states": gdn_states,
        "attention_outputs": attention_outputs,
        "slot_mapping": _slot_record(
            slot_mappings, set(attention_outputs), num_actual_tokens
        ),
    }


def record_after_forward(
    runner: Any,
    *,
    attn_metadata: Any,
    slot_mappings: Any,
    num_actual_tokens: int,
    runtime_mode: str,
) -> None:
    if not enabled():
        return
    phase = _phase(attn_metadata)
    if phase is None:
        return
    req_ids = list(getattr(runner.input_batch, "req_ids", ()))
    if not req_ids:
        return
    request_id = str(req_ids[0])
    cases = getattr(runner, "_sm120_validation_cases", {})
    if request_id not in cases:
        if len(cases) >= 2:
            return
        cases[request_id] = len(cases)
        runner._sm120_validation_cases = cases
    case_index = cases[request_id]
    if phase == "prefill":
        row = runner.input_batch.req_id_to_index.get(request_id)
        if row is None:
            return
        computed = int(runner.input_batch.num_computed_tokens_cpu[row])
        prompt = int(runner.input_batch.num_prompt_tokens[row])
        if computed + num_actual_tokens < prompt:
            return
    recorded = getattr(runner, "_sm120_validation_recorded", set())
    checkpoint = f"after_{phase}"
    key = (case_index, checkpoint)
    if key in recorded:
        return
    torch.accelerator.synchronize()
    value = _snapshot(
        runner,
        checkpoint=checkpoint,
        attn_metadata=attn_metadata,
        slot_mappings=slot_mappings,
        num_actual_tokens=num_actual_tokens,
        runtime_mode=runtime_mode,
        case_index=case_index,
    )
    _append(value)
    recorded.add(key)
    runner._sm120_validation_recorded = recorded
    if phase == "verify":
        pending = getattr(runner, "_sm120_validation_pending", {})
        pending[case_index] = (
            attn_metadata,
            slot_mappings,
            num_actual_tokens,
            runtime_mode,
        )
        runner._sm120_validation_pending = pending


def record_after_accepted_commit(runner: Any) -> None:
    if not enabled():
        return
    pending_by_case = getattr(runner, "_sm120_validation_pending", {})
    if not pending_by_case:
        return
    case_index = max(pending_by_case)
    pending = pending_by_case[case_index]
    recorded = getattr(runner, "_sm120_validation_recorded", set())
    key = (case_index, "after_accepted_commit")
    if key in recorded:
        return
    attn_metadata, slot_mappings, num_actual_tokens, runtime_mode = pending
    torch.accelerator.synchronize()
    _append(
        _snapshot(
            runner,
            checkpoint="after_accepted_commit",
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings,
            num_actual_tokens=num_actual_tokens,
            runtime_mode=runtime_mode,
            case_index=case_index,
        )
    )
    recorded.add(key)
    runner._sm120_validation_recorded = recorded


def _append(value: dict[str, Any]) -> None:
    assert _VALIDATION_DIR is not None
    directory = pathlib.Path(_VALIDATION_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank-{value['rank']}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
