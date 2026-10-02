"""CPU-only HTTP admission for the single-process owned audit endpoint.

The lease covers the complete ASGI response, including SSE. It does not claim
that an aborted frontend request has drained the engine or its CUDA work.
"""
import json
import re

AUDIT_PATHS = frozenset(("/nvidia-runtime-inspection", "/nvidia-transport-checksums",
                         "/nvidia-runtime-samples"))
READ_METHODS = frozenset(("GET", "HEAD", "OPTIONS"))
SSE_ERROR_PREFIX = re.compile(
    rb'^(?:data:\s*\{\s*"error"\s*:|'
    rb'data:\s*\{\s*"type"\s*:\s*"(?:error|response\.failed)")')
SSE_ERROR_EVENT = re.compile(rb'event:\s*(?:error|response\.failed)\s*\r?')


class NativeSSEErrorObserver:
    """Observe pinned native error prefixes; retain at most 256 bytes per line.

    This is not a general SSE validator or a completion/drain proof. Normal
    model content is nested in JSON and does not become a top-level SSE error.
    """
    def __init__(self):
        self.prefix = b""
        self.failed = False

    def feed(self, body):
        parts = body.split(b"\n")
        for index, part in enumerate(parts):
            if index:
                self.prefix = b""
            self.prefix += part[:max(0, 256 - len(self.prefix))]
            # Event names are exact only after a newline, not at a partial
            # chunk end that might later extend "error" to "error.other".
            if SSE_ERROR_PREFIX.match(self.prefix) or (
                    index < len(parts) - 1 and SSE_ERROR_EVENT.fullmatch(self.prefix)):
                self.failed = True


def audit_failed(state):
    # A returned both-rank RPC followed by lost HTTP delivery does not itself
    # make worker execution unknown. Request IDs remain non-replayable.
    return bool(state.nvidia_inspection_outcome_unknown)


def frontend_unfinished_requests(state):
    """Pinned AsyncLLM -> OutputProcessor count; no engine RPC or CUDA call."""
    engine = state.nvidia_inspection_engine
    processor = getattr(engine, "output_processor", None)
    getter = getattr(processor, "get_num_unfinished_requests", None)
    if not callable(getter):
        raise ValueError("native frontend unfinished-request count unavailable")
    count = getter()
    if type(count) is not int or count < 0:
        raise ValueError("invalid native frontend unfinished-request count")
    return count


class InspectionAdmissionMiddleware:
    def __init__(self, app, *, state):
        self.app = app
        self.state = state

    async def _reject(self, send, status, detail):
        body = json.dumps({"detail": detail}, separators=(",", ":")).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method", "GET") in READ_METHODS:
            return await self.app(scope, receive, send)
        state = self.state
        if not getattr(state, "nvidia_inspection_admission_initialized", False):
            return await self._reject(send, 503, "inspection admission not initialized")
        if audit_failed(state):
            return await self._reject(send, 409, "previous audit failed; owned candidate restart required")
        is_audit = scope.get("path") in AUDIT_PATHS
        # No await occurs between admission checks and lease acquisition. All
        # HTTP handlers for the sealed single API process share this event loop.
        if is_audit:
            if state.nvidia_inspection_audit_active or state.nvidia_inspection_application_active:
                return await self._reject(send, 409, "candidate HTTP work already in progress")
            if state.nvidia_inspection_application_drain_unproven:
                return await self._reject(send, 409, "previous application response incomplete; engine drain unproven")
            state.nvidia_inspection_audit_active = True
            try:
                count = frontend_unfinished_requests(state)
            except (AttributeError, TypeError, ValueError) as exc:
                state.nvidia_inspection_audit_active = False
                return await self._reject(send, 503, str(exc))
            except BaseException:
                state.nvidia_inspection_audit_active = False
                raise
            if count:
                state.nvidia_inspection_audit_active = False
                return await self._reject(send, 409, "native frontend still owns unfinished requests")
        else:
            if state.nvidia_inspection_audit_active:
                return await self._reject(send, 409, "inspection in progress")
            state.nvidia_inspection_application_active += 1
        completed = False
        event_stream = False
        stream_errors = NativeSSEErrorObserver()
        dispatched_before = len(state.nvidia_inspection_seen)

        async def tracked_send(message):
            nonlocal completed, event_stream
            if not is_audit:
                if message["type"] == "http.response.start":
                    # Native generation failures can be returned as ordinary
                    # error responses; middleware sees no escaped exception.
                    if message["status"] >= 400:
                        state.nvidia_inspection_application_drain_unproven = True
                    event_stream = any(name.lower() == b"content-type" and
                        value.lower().startswith(b"text/event-stream")
                        for name, value in message.get("headers", ()))
                elif message["type"] == "http.response.body" and event_stream:
                    stream_errors.feed(message.get("body", b""))
                    if stream_errors.failed:
                        state.nvidia_inspection_application_drain_unproven = True
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                completed = True

        try:
            return await self.app(scope, receive, tracked_send)
        except BaseException:
            # Even if a final body was sent, an exception in the ASGI call can
            # represent incomplete background cleanup. Do not infer engine drain.
            if is_audit and len(state.nvidia_inspection_seen) > dispatched_before:
                state.nvidia_inspection_response_failed = True
            elif not is_audit:
                state.nvidia_inspection_application_drain_unproven = True
            raise
        finally:
            if is_audit:
                if not completed and len(state.nvidia_inspection_seen) > dispatched_before:
                    state.nvidia_inspection_response_failed = True
                state.nvidia_inspection_audit_active = False
            else:
                if not completed:
                    state.nvidia_inspection_application_drain_unproven = True
                state.nvidia_inspection_application_active -= 1
