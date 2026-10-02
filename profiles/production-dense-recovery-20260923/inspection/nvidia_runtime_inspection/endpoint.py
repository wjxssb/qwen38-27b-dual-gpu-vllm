"""Three fixed audit routes; no generic RPC or development endpoints."""
import asyncio
import json

from .collector import METHOD, validate_request, verified_binding
from . import sampler, transport
from .admission import InspectionAdmissionMiddleware, audit_failed


class InspectionEndpoint:
    name = "nvidia_runtime_inspection"
    required_tasks = ("generate",)

    def attach_router(self, app):
        from fastapi import HTTPException, Request
        from fastapi.responses import JSONResponse

        def make_handler(method, validator, byte_limit):
            async def collect(request: Request):
                chunks = []
                size = 0
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > byte_limit:
                        raise HTTPException(status_code=413, detail="inspection request too large")
                    chunks.append(chunk)
                try:
                    raw = b"".join(chunks).decode("utf-8")
                    parsed = validator(raw)
                    verified_binding(parsed)
                except (ValueError, KeyError, UnicodeError) as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                engine = getattr(app.state, "nvidia_inspection_engine", None)
                if engine is None:
                    raise HTTPException(status_code=503, detail="inspection engine unavailable")
                lock = app.state.nvidia_inspection_lock
                if lock.locked():
                    raise HTTPException(status_code=409, detail="inspection already in progress")
                if audit_failed(app.state):
                    raise HTTPException(status_code=409, detail="previous audit failed; no automatic retry")
                if parsed["request_id"] in app.state.nvidia_inspection_seen:
                    raise HTTPException(status_code=409, detail="request_id already dispatched")
                if len(app.state.nvidia_inspection_seen) >= 128:
                    raise HTTPException(status_code=409, detail="inspection count bound reached")
                async with lock:
                    app.state.nvidia_inspection_seen.add(parsed["request_id"])
                    try:
                        results = await engine.collective_rpc(method=method, timeout=120,
                                                              args=(json.dumps(parsed, separators=(",", ":")),))
                    except BaseException:
                        app.state.nvidia_inspection_outcome_unknown = True
                        raise
                try:
                    return JSONResponse(content={"schema": 1, "status": "COLLECTED_UNQUALIFIED", "results": results})
                except BaseException:
                    # The worker RPC returned, but its evidence cannot be
                    # encoded. This is distinct from unknown worker outcome.
                    app.state.nvidia_inspection_response_failed = True
                    raise
            return collect

        routes = (
            ("/nvidia-runtime-inspection", METHOD, validate_request, 16384),
            ("/nvidia-transport-checksums", transport.METHOD, transport.validate_transport_request, 16384),
            ("/nvidia-runtime-samples", sampler.METHOD, sampler.validate_sample_request, sampler.MAX_BYTES),
        )
        if any(getattr(route, "path", None) in {r[0] for r in routes} for route in app.routes):
            raise RuntimeError("inspection route collision")
        app.add_middleware(InspectionAdmissionMiddleware, state=app.state)
        for path, method, validator, limit in routes:
            app.add_api_route(path, make_handler(method, validator, limit), methods=["POST"], include_in_schema=False)

    async def init_state(self, engine_client, state, args):
        state.nvidia_inspection_engine = engine_client
        state.nvidia_inspection_lock = asyncio.Lock()
        state.nvidia_inspection_seen = set()
        state.nvidia_inspection_outcome_unknown = False
        state.nvidia_inspection_response_failed = False
        state.nvidia_inspection_audit_active = False
        state.nvidia_inspection_application_active = 0
        state.nvidia_inspection_application_drain_unproven = False
        state.nvidia_inspection_admission_initialized = True
