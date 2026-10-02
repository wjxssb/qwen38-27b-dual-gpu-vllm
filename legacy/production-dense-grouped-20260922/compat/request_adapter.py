"""Staged NVIDIA candidate-only reasoning alias compatibility; no network I/O.

Only the winning reasoning_effort JSON string is replaced. Every other byte,
including stream settings, image data, schemas, and message history is retained.
No production or candidate server imports this module at preparation time.
"""
from dataclasses import dataclass
import json


class InvalidRequestBody(ValueError):
    pass


@dataclass(frozen=True)
class AdaptedBody:
    body: bytes
    changed_field: str | None


def _object_fields(text: str, start: int = 0):
    """Return decoded field values with exact value spans in one JSON object."""
    decoder = json.JSONDecoder()
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != '{':
        raise InvalidRequestBody('Expected a JSON object.')
    index += 1
    fields = {}
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == '}':
            return fields
        key, index = decoder.raw_decode(text, index)
        if not isinstance(key, str):
            raise InvalidRequestBody('Expected a JSON object key.')
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != ':':
            raise InvalidRequestBody('Expected a JSON key/value separator.')
        index += 1
        while index < len(text) and text[index].isspace():
            index += 1
        value_start = index
        value, index = decoder.raw_decode(text, index)
        fields.setdefault(key, []).append((value, value_start, index))
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == ',':
            index += 1
            continue
        if index < len(text) and text[index] == '}':
            return fields
        raise InvalidRequestBody('Expected a JSON field separator.')


def _unique(fields, key):
    matches = fields.get(key, [])
    if len(matches) > 1:
        raise InvalidRequestBody('Ambiguous duplicate compatibility field: ' + key)
    return matches[0] if matches else None


def adapt_request_body(body: bytes, *, candidate_model_id: str) -> AdaptedBody:
    """Normalize effective high→xhigh for exactly one configured candidate ID.

    Caller binds this to candidate /v1/chat/completions only. vLLM's pinned
    build_chat_params gives valid non-None top-level reasoning_effort priority
    over chat_template_kwargs.reasoning_effort. No server defaults are injected.
    Invalid or unknown effort values are left to the backend's own validation.
    """
    if not candidate_model_id:
        raise ValueError('An explicit candidate model ID is required.')
    try:
        text = body.decode('utf-8')
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise InvalidRequestBody('Expected a JSON object.')
        fields = _object_fields(text)
        model = _unique(fields, 'model')
        if model is None or model[0] != candidate_model_id:
            return AdaptedBody(body, None)
        top = _unique(fields, 'reasoning_effort')
        kwargs = _unique(fields, 'chat_template_kwargs')
        selected = None
        field = None
        # merge_kwargs internally ignores "auto", but ChatCompletionRequest's
        # HTTP schema does not accept that value. Do not rescue or modify an
        # invalid request by interpreting it as an absent top-level effort.
        if top is not None and top[0] not in (None, 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
            return AdaptedBody(body, None)
        if top is not None and top[0] not in (None, 'auto'):
            selected, field = top, 'reasoning_effort'
        elif kwargs is not None and isinstance(kwargs[0], dict):
            selected = _unique(_object_fields(text, kwargs[1]), 'reasoning_effort')
            field = 'chat_template_kwargs.reasoning_effort'
        if selected is None or selected[0] != 'high':
            return AdaptedBody(body, None)
        _, start, end = selected
        return AdaptedBody((text[:start] + '"xhigh"' + text[end:]).encode('utf-8'), field)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidRequestBody('Invalid JSON request body.') from exc
