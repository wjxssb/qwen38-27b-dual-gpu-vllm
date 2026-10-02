# SPDX-License-Identifier: Apache-2.0
"""Best-effort telemetry with bounded records, memory and disk use."""
from __future__ import annotations

import contextvars
import copy
import json
import os
import pathlib
import queue
import re
import threading
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from itertools import islice
from typing import Any

STABILITY_TELEMETRY_DIR_ENV = "VLLM_STABILITY_TELEMETRY_DIR"
_MAX_RECORD_BYTES = 16 * 1024
_MAX_FAILURE_BYTES = 256 * 1024
_MAX_QUEUED_BYTES = 4 * 1024 * 1024
_MAX_FILE_BYTES = 32 * 1024 * 1024
_WRITER_QUEUE_SIZE = 512
_RECENT_EVENT_WINDOW_SIZE = 256
_process_role = "unknown"
_rpc_id = contextvars.ContextVar("vllm_stability_rpc_id", default=None)
_generation = contextvars.ContextVar("vllm_stability_generation", default=None)
_request_id = contextvars.ContextVar("vllm_stability_request_id", default=None)
_writer_lock = threading.Lock()
_recent_events_lock = threading.Lock()
_writer_pid = None
_writer_directory = None
_writer_queue = None
_writer_thread = None
_queued_bytes = 0
_dropped_records = 0
_recent_events = deque(maxlen=_RECENT_EVENT_WINDOW_SIZE)
_STOP = object()
_CUDA_HARD_FAILURE = re.compile(
    r'illegal[ _]memory[ _]access|device-side assert|misaligned[ _]address|'
    r'unspecified launch failure|launch timed out|launch timeout|illegal[ _]instruction|'
    r'cudaErrorLaunchTimeout|cudaErrorIllegalInstruction|CUDNN_STATUS_EXECUTION_FAILED|'
    r'cudaErrorLaunchFailure|cudaErrorMisalignedAddress|'
    r'fallen off the bus|Failed to initialize the TMA descriptor\s+716\b|'
    r'CUDA (?:error:.*out of memory|out of memory)', re.I)


def record_first_failure(reason, error, **fields):
    """Publish bounded first-fault evidence without the lossy telemetry queue.

    Atomic hard-link publication exposes either a complete JSON record or no
    record. Per-role markers retain severity if another role first published a
    secondary soft error. This performs no CUDA operation, collective or RPC.
    """
    directory = os.environ.get(STABILITY_TELEMETRY_DIR_ENV)
    if not directory:
        return False
    root = pathlib.Path(directory).parent
    chain, seen = [], set()
    current = error
    while current is not None and len(chain) < 4 and id(current) not in seen:
        seen.add(id(current))
        frames = deque(maxlen=12)
        trace = getattr(current, '__traceback__', None)
        depth = 0
        while trace is not None and depth < 64:
            frames.append({'file': trace.tb_frame.f_code.co_filename[-256:],
                           'function': trace.tb_frame.f_code.co_name[:128], 'line': trace.tb_lineno})
            trace = trace.tb_next
            depth += 1
        chain.append({'type': type(current).__name__, 'message': str(current)[:2048],
                      'frames': list(frames)})
        current = getattr(current, '__cause__', None) or getattr(current, '__context__', None)
    message = chain[0]['message'] if chain else str(error)[:2048]
    row = _sanitize(fields)
    row.update(schema_version=1, reason=str(reason)[:128], message=message,
               exception_chain=chain, hard_stop=any(_CUDA_HARD_FAILURE.search(item['message']) for item in chain),
               process_role=_process_role, pid=os.getpid(),
               monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns())
    for key, var in (('rpc_id', _rpc_id), ('generation', _generation), ('request_id', _request_id)):
        if row.get(key) is None and var.get() is not None:
            row[key] = var.get()
    raw = _encode(row)
    if len(raw) > 16384:
        keys = ('schema_version', 'reason', 'message', 'hard_stop', 'process_role', 'pid',
                'monotonic_ns', 'wall_time_ns', 'rpc_id', 'generation', 'request_id',
                'executor_identity', 'rank', 'method')
        row = {key: (row[key][:64] if type(row[key]) is str else row[key])
               for key in keys if key in row}
        row['fields_truncated'] = True
        raw = _encode(row)
    role = _process_role if re.fullmatch(r'[A-Za-z0-9_-]{1,64}', _process_role) else 'unknown'
    temporary = None
    published = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix='.first-failure-', dir=root, delete=False) as stream:
            temporary = stream.name
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        for target in (root / ('first-failure-' + role + '.json'), root / 'engine-failure.json'):
            try:
                os.link(temporary, target)
                published = True
            except FileExistsError:
                pass  # The first published cause is immutable.
        if published:
            # Sync the directory entry too: file fsync alone does not make the
            # newly linked first-fault marker durable across a host crash.
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError:
        pass  # Evidence I/O cannot allow the failed worker to resume work.
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return published


class _FlushRequest:
    def __init__(self):
        self.done = threading.Event()


def _sanitize(value, depth=0, budget=None):
    """Copy bounded builtin values; never stringify tensors/arbitrary objects.

    Failure windows are excluded at every depth. The node budget also handles
    cycles and shared subtrees before serialization allocates any large string.
    """
    if budget is None:
        budget = [512]
    budget[0] -= 1
    if budget[0] < 0 or depth > 7:
        return "<truncated>"
    if value is None or type(value) in (bool, float):
        return value
    if type(value) is int:
        return value if value.bit_length() <= 128 else "<large-int>"
    if type(value) is str:
        return value[:2048]
    if type(value) is dict:
        result = {}
        for key, item in islice(value.items(), 64):
            if budget[0] <= 0:
                break
            if type(key) is str and key != "recent_event_window":
                result[key[:128]] = _sanitize(item, depth + 1, budget)
        return result
    if type(value) in (list, tuple):
        result = []
        for item in islice(value, 64):
            if budget[0] <= 0:
                break
            result.append(_sanitize(item, depth + 1, budget))
        return result
    return "<unsupported-value>"


def _encode(record):
    return (json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _bounded_record_and_payload(record):
    """Return the detached record and its already checked JSON representation."""
    clean = _sanitize(record)
    payload = _encode(clean)
    if len(payload) <= _MAX_RECORD_BYTES:
        return clean, payload
    # Keep correlation even when a caller supplies a huge snapshot.
    keys = ("schema_version", "event", "pid", "process_role", "monotonic_ns",
            "wall_time_ns", "rpc_id", "generation", "request_id")
    result = {key: clean[key] for key in keys if key in clean}
    result["fields_truncated"] = True
    # Correlation strings themselves may contain escaped unicode.
    for key, value in result.items():
        if type(value) is str:
            result[key] = value[:128]
    return result, _encode(result)


def _bounded_record(record):
    return _bounded_record_and_payload(record)[0]


def _writer_loop(directory, records):
    global _queued_bytes
    root = pathlib.Path(directory)
    while True:
        record = records.get()
        if record is _STOP:
            return
        if isinstance(record, _FlushRequest):
            record.done.set()
            continue
        role, payload = record
        with _writer_lock:
            _queued_bytes -= len(payload)
        try:
            root.mkdir(parents=True, exist_ok=True)
            path = root / (role + ".jsonl")
            if path.exists() and path.stat().st_size + len(payload) > _MAX_FILE_BYTES:
                # Only this process's current log rotates; prior run evidence
                # is in different run directories and is never touched.
                os.replace(path, root / (role + ".jsonl.1"))
            with path.open("ab") as handle:
                handle.write(payload)
        except OSError:
            pass  # Keep draining if disk/permissions fail; never block serving.


def _ensure_writer():
    global _writer_pid, _writer_directory, _writer_queue, _writer_thread
    directory = os.environ.get(STABILITY_TELEMETRY_DIR_ENV)
    if not directory:
        return None
    with _writer_lock:
        if _writer_queue is not None:
            # A runtime role/directory is fixed for one process. Do not spawn
            # competing writers if an environment variable changes mid-run.
            return _writer_queue
        _writer_pid = os.getpid()
        _writer_directory = directory
        _writer_queue = queue.Queue(maxsize=_WRITER_QUEUE_SIZE)
        _writer_thread = threading.Thread(
            target=_writer_loop, args=(directory, _writer_queue), daemon=True,
            name="vllm-stability-telemetry")
        _writer_thread.start()
        return _writer_queue


def _after_fork():
    global _writer_lock, _recent_events_lock, _writer_pid, _writer_directory
    global _writer_queue, _writer_thread, _queued_bytes, _dropped_records, _recent_events
    _writer_lock = threading.Lock()
    _recent_events_lock = threading.Lock()
    _writer_pid = _writer_directory = _writer_queue = _writer_thread = None
    _queued_bytes = _dropped_records = 0
    _recent_events = deque(maxlen=_RECENT_EVENT_WINDOW_SIZE)


os.register_at_fork(after_in_child=_after_fork)


def is_enabled():
    return bool(os.environ.get(STABILITY_TELEMETRY_DIR_ENV))


def configure_process(role):
    global _process_role
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", role):
        raise ValueError("Invalid telemetry process role")
    _process_role = role
    emit("process_configured")


@contextmanager
def rpc_scope(rpc_id):
    token = _rpc_id.set(rpc_id)
    try:
        yield
    finally:
        _rpc_id.reset(token)


@contextmanager
def lifecycle_scope(generation, request_id):
    g = _generation.set(generation)
    r = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(r)
        _generation.reset(g)


def recent_event_window(limit=64):
    try:
        limit = max(0, min(int(limit), _RECENT_EVENT_WINDOW_SIZE))
    except (TypeError, ValueError, OverflowError):
        return []
    with _recent_events_lock:
        return copy.deepcopy(list(_recent_events)[-limit:]) if limit else []


def emit_failure_window(event, /, **fields):
    if is_enabled():
        _emit(event, fields, failure=True)


def emit(event, /, **fields):
    if is_enabled():
        _emit(event, fields, failure=False)


def _emit(event, fields, failure):
    global _queued_bytes, _dropped_records
    records = _ensure_writer()
    if records is None:
        return
    base = dict(fields)
    base.update(schema_version=1, event=event, pid=os.getpid(),
                process_role=_process_role, monotonic_ns=time.monotonic_ns(),
                wall_time_ns=time.time_ns())
    for key, var in (("rpc_id", _rpc_id), ("generation", _generation),
                     ("request_id", _request_id)):
        if var.get() is not None:
            base[key] = var.get()
    base, payload = _bounded_record_and_payload(base)
    record = dict(base)
    if failure:
        record["recent_event_window"] = recent_event_window()
        while len(_encode(record)) > _MAX_FAILURE_BYTES:
            record["recent_event_window"].pop(0)
            record["window_truncated"] = True
        payload = _encode(record)
    # Store only detached base records, NEVER a failure's embedded window.
    with _recent_events_lock:
        _recent_events.append(base)
    with _writer_lock:
        counted_drops = False
        if _dropped_records:
            record['dropped_since_previous'] = min(_dropped_records, 2**63 - 1)
            candidate = _encode(record)
            if len(candidate) <= (_MAX_FAILURE_BYTES if failure else _MAX_RECORD_BYTES):
                payload = candidate
                counted_drops = True
        if _queued_bytes + len(payload) > _MAX_QUEUED_BYTES:
            _dropped_records = min(_dropped_records + 1, 2**63 - 1)
            return
        try:
            records.put_nowait((_process_role, payload))
            _queued_bytes += len(payload)
            if counted_drops:
                _dropped_records = 0
        except queue.Full:
            _dropped_records = min(_dropped_records + 1, 2**63 - 1)


def flush_for_test(timeout_s=5.0):
    records = _ensure_writer()
    if records is None:
        return True
    request = _FlushRequest()
    try:
        records.put(request, timeout=timeout_s)
    except queue.Full:
        return False
    return request.done.wait(timeout_s)
