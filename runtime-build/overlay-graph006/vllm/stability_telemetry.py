# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in structured telemetry for isolated runtime-stability investigations."""

from __future__ import annotations

import contextvars
import json
import os
import pathlib
import queue
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

STABILITY_TELEMETRY_DIR_ENV = "VLLM_STABILITY_TELEMETRY_DIR"

_process_role = "unknown"
_rpc_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "vllm_stability_rpc_id", default=None
)
_generation: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "vllm_stability_generation", default=None
)
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "vllm_stability_request_id", default=None
)
_WRITER_QUEUE_SIZE = 8192
# A crash/failure record needs enough recent correlation to explain a stalled
# generation without turning every hot-path event into synchronous I/O.  Keep
# this deliberately small and process-local; the JSONL writer remains the
# durable, asynchronous stream.
_RECENT_EVENT_WINDOW_SIZE = 256
_writer_lock = threading.Lock()
_recent_events_lock = threading.Lock()
_writer_pid: int | None = None
_writer_directory: str | None = None
_writer_queue: queue.Queue[dict[str, Any] | _FlushRequest | object] | None = None
_writer_thread: threading.Thread | None = None
_dropped_records = 0
_recent_events: deque[dict[str, Any]] = deque(maxlen=_RECENT_EVENT_WINDOW_SIZE)
_STOP = object()


class _FlushRequest:
    def __init__(self) -> None:
        self.done = threading.Event()


def _writer_loop(
    directory: str,
    records: queue.Queue[dict[str, Any] | _FlushRequest | object],
) -> None:
    root = pathlib.Path(directory)
    handles: dict[str, Any] = {}
    try:
        root.mkdir(parents=True, exist_ok=True)
        while True:
            record = records.get()
            if record is _STOP:
                return
            if isinstance(record, _FlushRequest):
                for handle in handles.values():
                    handle.flush()
                record.done.set()
                continue
            try:
                role = record["process_role"]
                handle = handles.get(role)
                if handle is None:
                    handle = (root / f"{role}.jsonl").open(
                        "a", encoding="utf-8", buffering=1
                    )
                    handles[role] = handle
                with _writer_lock:
                    global _dropped_records
                    dropped_records = _dropped_records
                    _dropped_records = 0
                if dropped_records:
                    record["dropped_since_previous"] = dropped_records
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            except Exception:
                continue
    finally:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                continue


def _ensure_writer() -> queue.Queue[dict[str, Any] | _FlushRequest | object] | None:
    directory = os.environ.get(STABILITY_TELEMETRY_DIR_ENV)
    if not directory:
        return None

    global _writer_directory, _writer_pid, _writer_queue, _writer_thread
    pid = os.getpid()
    with _writer_lock:
        if (
            _writer_pid == pid
            and _writer_directory == directory
            and _writer_queue is not None
            and _writer_thread is not None
            and _writer_thread.is_alive()
        ):
            return _writer_queue

        old_queue = _writer_queue
        records: queue.Queue[dict[str, Any] | _FlushRequest | object] = queue.Queue(
            maxsize=_WRITER_QUEUE_SIZE
        )
        writer = threading.Thread(
            target=_writer_loop,
            args=(directory, records),
            name="vllm-stability-telemetry",
            daemon=True,
        )
        _writer_pid = pid
        _writer_directory = directory
        _writer_queue = records
        _writer_thread = writer
        writer.start()

    if old_queue is not None:
        with suppress(queue.Full):
            old_queue.put_nowait(_STOP)
    return records


def is_enabled() -> bool:
    """Return whether isolated stability telemetry is enabled."""
    return bool(os.environ.get(STABILITY_TELEMETRY_DIR_ENV))


def configure_process(role: str) -> None:
    """Set the stable process role used for subsequent telemetry records."""
    global _process_role
    _process_role = role
    if _ensure_writer() is not None:
        emit("process_configured")


@contextmanager
def rpc_scope(rpc_id: int | None) -> Iterator[None]:
    """Associate telemetry emitted in this scope with one internal RPC."""
    token = _rpc_id.set(rpc_id)
    try:
        yield
    finally:
        _rpc_id.reset(token)


@contextmanager
def lifecycle_scope(generation: int | None, request_id: str | None) -> Iterator[None]:
    """Associate telemetry with a bounded TP request generation.

    This is intentionally only metadata.  It does not retain request payloads
    or tensors and is safe to use from executor and worker control paths.
    """
    generation_token = _generation.set(generation)
    request_token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(request_token)
        _generation.reset(generation_token)


def recent_event_window(limit: int = 64) -> list[dict[str, Any]]:
    """Return a bounded, read-only snapshot for a failure diagnostic.

    The caller receives copies so a diagnostic consumer cannot mutate the
    process-local ring.  Invalid or excessive limits are clamped rather than
    raising from a failure path.
    """
    limit = max(0, min(int(limit), _RECENT_EVENT_WINDOW_SIZE))
    if limit == 0:
        return []
    with _recent_events_lock:
        return [dict(record) for record in list(_recent_events)[-limit:]]


def emit_failure_window(event: str, /, **fields: Any) -> None:
    """Emit one asynchronous diagnostic containing the recent bounded ring."""
    emit(event, recent_event_window=recent_event_window(), **fields)


def emit(event: str, /, **fields: Any) -> None:
    """Queue one best-effort record without blocking serving."""
    records = _ensure_writer()
    if records is None:
        return

    record: dict[str, Any] = {
        "schema_version": 1,
        "event": event,
        "pid": os.getpid(),
        "process_role": _process_role,
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_ns": time.time_ns(),
        **fields,
    }
    if (rpc_id := _rpc_id.get()) is not None:
        record["rpc_id"] = rpc_id
    if (generation := _generation.get()) is not None:
        record["generation"] = generation
    if (request_id := _request_id.get()) is not None:
        record["request_id"] = request_id

    # Keep the in-memory failure window bounded and independent from writer
    # health.  The append is deliberately best-effort just like the queue.
    try:
        with _recent_events_lock:
            _recent_events.append(dict(record))
    except Exception:
        pass

    try:
        records.put_nowait(record)
    except queue.Full:
        global _dropped_records
        with _writer_lock:
            _dropped_records += 1


def flush_for_test(timeout_s: float = 5.0) -> bool:
    """Flush queued telemetry records for deterministic no-GPU tests."""
    records = _ensure_writer()
    if records is None:
        return True
    request = _FlushRequest()
    try:
        records.put(request, timeout=timeout_s)
    except queue.Full:
        return False
    return request.done.wait(timeout_s)
