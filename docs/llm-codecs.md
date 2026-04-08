<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LLM Codecs

Codecs translate opaque `LLMRequest` payloads into structured
`AnnotatedLLMRequest` objects, giving request intercepts a typed contract
instead of raw JSON. They sit in the LLM call pipeline, decoding before
request intercepts and encoding back after.

## The Problem

`LLMRequest` carries two fields: `headers` (metadata) and `content` (the
request payload as untyped JSON). Different SDKs put different things in
`content` -- OpenAI uses `messages` + `model` + `temperature`, NVIDIA NIM
adds `guided_json` and `nvext`, and so on. Intercept authors who want to
inspect or modify messages, model parameters, or tool definitions must
reverse-engineer each SDK's format.

## How Codecs Solve It

A Codec provides two methods:

- **`decode(request)`** -- Parse `LLMRequest.content` into a structured
  `AnnotatedLLMRequest` with typed fields (messages, model, params, tools,
  etc.)
- **`encode(annotated, original)`** -- Merge structured changes back into
  `LLMRequest`, preserving unmodeled fields via merge-not-replace semantics

The pipeline calls decode before request intercepts and encode after:

```
LLMRequest
    |
    v
Codec.decode() --> AnnotatedLLMRequest
    |                   |
    v                   v
Request Intercepts (receive both)
    |                   |
    v                   v
Codec.encode() <-- modified AnnotatedLLMRequest
    |
    v
LLMRequest (updated)
    |
    v
Execution Intercepts --> func() --> Response
```

When no Codec is passed, the pipeline behaves exactly as before --
intercepts receive `None` for the annotated request.

## AnnotatedLLMRequest

The structured request type with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `list[dict]` | Role-tagged messages (`system`, `user`, `assistant`, `tool`) |
| `model` | `str \| None` | Model identifier |
| `params` | `dict \| None` | Generation parameters (`temperature`, `max_tokens`, `top_p`, `stop`) |
| `tools` | `list[dict] \| None` | Tool/function definitions |
| `tool_choice` | `dict \| str \| None` | Tool selection control (`"auto"`, `"none"`, `"required"`, or `{"type": "function", "function": {"name": "..."}}`) |
| `extra` | `dict` | Catch-all for unmodeled fields -- anything not explicitly mapped lands here and round-trips through encode |

### Message Format

Messages follow the OpenAI chat completion structure with role tagging:

```python
# System message
{"role": "system", "content": "You are a helpful assistant."}

# User message
{"role": "user", "content": "What is the weather?"}

# Assistant message with tool calls
{
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\"}"}
        }
    ]
}

# Tool result message
{"role": "tool", "tool_call_id": "call_123", "content": "{\"temp\": 72}"}
```

### Helper Methods

```python
annotated.system_prompt()       # Returns system message content (str or None)
annotated.last_user_message()   # Returns last user message content (str or None)
annotated.has_tool_calls()      # True if any assistant message has tool_calls
```

## Creating a Codec

Subclass `LlmCodec` and implement `decode` and `encode`:

```python
from nat_nexus import LLMRequest, AnnotatedLLMRequest
from nat_nexus.codecs import LlmCodec


class OpenAICodec(LlmCodec):
    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        content = request.content
        params = {}
        for key in ("temperature", "max_tokens", "top_p", "stop"):
            if key in content:
                params[key] = content[key]

        return AnnotatedLLMRequest(
            messages=content.get("messages", []),
            model=content.get("model"),
            params=params if params else None,
            tools=content.get("tools"),
            tool_choice=content.get("tool_choice"),
        )

    def encode(
        self, annotated: AnnotatedLLMRequest, original: LLMRequest
    ) -> LLMRequest:
        # Start with the original content (preserves unmodeled fields)
        content = dict(original.content)

        # Overlay structured fields
        content["messages"] = annotated.messages
        if annotated.model is not None:
            content["model"] = annotated.model
        if annotated.params:
            content.update(annotated.params)
        if annotated.tools is not None:
            content["tools"] = annotated.tools
        if annotated.tool_choice is not None:
            content["tool_choice"] = annotated.tool_choice

        # Merge extra fields (intercept-added body params)
        if annotated.extra:
            content.update(annotated.extra)

        # Use original headers (pipeline merges headers separately)
        return LLMRequest(original.headers, content)


# Pass the codec instance directly to execute calls:
openai_codec = OpenAICodec()

result = await nat_nexus.llm.execute(
    "gpt-4", request, call_openai, codec=openai_codec
)
```

### Encode Must Use Merge-Not-Replace

The `encode` method receives both the modified `AnnotatedLLMRequest` and
the pre-intercept `original` LLMRequest. Always start from
`original.content` and overlay changes. This preserves SDK-specific fields
that `AnnotatedLLMRequest` does not model (e.g., `stream`,
`response_format`, `reasoning_effort`, `seed`, `logprobs`).

If you construct a new content dict from scratch, those fields are lost.

## Using Codecs

### Per-Call Selection

Pass a Codec instance directly to the execute call:

```python
import nat_nexus

openai_codec = OpenAICodec()

result = await nat_nexus.llm.execute(
    "gpt-4",
    request,
    call_openai,
    codec=openai_codec,  # Pass the Codec instance directly
)
```

### Resolution Order

When an LLM call executes:

1. **Explicit `codec` parameter** on the execute call -- decode/encode is performed
2. **None** -- no Codec, intercepts receive `None` for annotated request

## Writing Request Intercepts

All LLM request intercepts receive the raw `LLMRequest` and the optional
`AnnotatedLLMRequest`. There is a single intercept type and a single
registration function -- no legacy/annotated split.

```python
from nat_nexus import LLMRequest, AnnotatedLLMRequest
from nat_nexus.intercepts import register_llm_request


def inject_system_prompt(name, request, annotated):
    """Add a system prompt if one doesn't exist."""
    if annotated is None:
        # No Codec passed -- fall back to raw request modification
        return request, None

    if not annotated.system_prompt():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            *annotated.messages,
        ]
        annotated = AnnotatedLLMRequest(
            messages=messages,
            model=annotated.model,
            params=annotated.params,
            tools=annotated.tools,
            tool_choice=annotated.tool_choice,
            extra=annotated.extra,
        )

    return request, annotated


register_llm_request(
    "system-prompt-injector",
    priority=10,
    break_chain=False,
    fn=inject_system_prompt,
)
```

### Intercept Signature

```python
def my_intercept(
    name: str,
    request: LLMRequest,
    annotated: AnnotatedLLMRequest | None,
) -> tuple[LLMRequest, AnnotatedLLMRequest | None]:
    ...
```

- `name` -- the LLM name passed to `llm.execute()`
- `request` -- the raw `LLMRequest` (always present)
- `annotated` -- the structured request (present when a Codec is active,
  `None` otherwise)
- Returns a tuple of `(LLMRequest, AnnotatedLLMRequest | None)`

When you modify `annotated`, the pipeline automatically calls
`Codec.encode()` to merge your changes back into `LLMRequest` before
passing to execution intercepts. You do not need to modify both -- change
whichever is appropriate:

- Modify `annotated` for structured changes (add messages, change model,
  adjust params)
- Modify `request` for raw changes (custom headers, opaque content
  manipulation)

## Practical Examples

### PII Redaction Intercept

```python
import re
from nat_nexus import AnnotatedLLMRequest
from nat_nexus.intercepts import register_llm_request

EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")


def redact_pii(name, request, annotated):
    if annotated is None:
        return request, None

    cleaned_messages = []
    for msg in annotated.messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            msg = {**msg, "content": EMAIL_PATTERN.sub("[REDACTED]", content)}
        cleaned_messages.append(msg)

    annotated = AnnotatedLLMRequest(
        messages=cleaned_messages,
        model=annotated.model,
        params=annotated.params,
        tools=annotated.tools,
        tool_choice=annotated.tool_choice,
        extra=annotated.extra,
    )
    return request, annotated


register_llm_request("pii-redactor", 5, False, redact_pii)
```

### Token Budget Enforcer

```python
from nat_nexus.intercepts import register_llm_request


def enforce_token_budget(name, request, annotated):
    if annotated is None:
        return request, None

    # Cap max_tokens to 1000 regardless of what the caller requested
    params = dict(annotated.params or {})
    if params.get("max_tokens", 0) > 1000:
        params["max_tokens"] = 1000

    annotated = AnnotatedLLMRequest(
        messages=annotated.messages,
        model=annotated.model,
        params=params,
        tools=annotated.tools,
        tool_choice=annotated.tool_choice,
        extra=annotated.extra,
    )
    return request, annotated


register_llm_request("token-budget", 20, False, enforce_token_budget)
```

### Tool Call Audit Logger

```python
import json
from nat_nexus.intercepts import register_llm_request


def log_tool_calls(name, request, annotated):
    if annotated and annotated.has_tool_calls():
        for msg in annotated.messages:
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                print(f"[AUDIT] Tool call: {fn.get('name')} args={fn.get('arguments')}")
    return request, annotated


register_llm_request("tool-audit", 100, False, log_tool_calls)
```

### Adding Extra Body Parameters

Intercepts can add arbitrary parameters to the request body via the
`extra` field. These get merged into the LLM request during encode:

```python
def add_request_metadata(name, request, annotated):
    if annotated is None:
        return request, None

    extra = dict(annotated.extra or {})
    extra["user"] = "tenant-123"
    extra["metadata"] = {"source": "api", "version": "2.0"}

    annotated = AnnotatedLLMRequest(
        messages=annotated.messages,
        model=annotated.model,
        params=annotated.params,
        tools=annotated.tools,
        tool_choice=annotated.tool_choice,
        extra=extra,
    )
    return request, annotated


register_llm_request("metadata-injector", 15, False, add_request_metadata)
```

## Cross-Language Support

The Codec system is available in all Nexus binding languages. Codecs are
passed directly at the call site:

| Language | Pass to Execute |
|----------|-----------------|
| **Python** | `llm.execute(..., codec=my_codec)` |
| **Go** | `LlmCallExecute(..., WithLLMCodec(codec))` |
| **Node.js** | `llmCallExecute(..., decodeFn, encodeFn)` |
| **WASM** | `llm_call_execute(..., decodeFn, encodeFn)` |

The `codec` parameter is available on all LLM execute functions across
all bindings.

## Related Docs

- [Middleware Pipeline](middleware-pipeline.md) -- Full pipeline stage ordering
- [Getting Started: Python](getting-started-python.md) -- Python quickstart
- [Typed API Reference](typed-api-reference.md) -- Typed wrappers (distinct from LLM Codecs)
- [Recipes](recipes.md) -- Common integration patterns
