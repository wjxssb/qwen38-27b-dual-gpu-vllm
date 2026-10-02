# Staged request compatibility

`request_adapter.py` provides a pure, candidate-scoped `high`→`xhigh` alias translation. It changes only the effective reasoning-effort JSON value span and retains all other bytes. The explicit `candidate_model_id` must match the request's model field. The module does not listen on a port, call the model, touch services, or rewrite response fields.

Top-level effort wins over request `chat_template_kwargs` when non-null, matching the pinned vLLM implementation. A shadowed nested `high` is unchanged. Message roles and unsupported effort values are unchanged. Ambiguous duplicate compatibility fields are rejected. Candidate launcher/server imports are absent; this adapter is **staged, not active**.

Tests use the exact pinned vLLM merge/build methods extracted by AST plus the unmodified NVIDIA Jinja template. See `test-receipt.json` and the campaign's `TEMPLATE_COMPATIBILITY.md`. Integration later must bind only candidate Chat Completions, update changed body length, preserve cancellation/backpressure/SSE transport, and run actual E2E after GPU qualification.
