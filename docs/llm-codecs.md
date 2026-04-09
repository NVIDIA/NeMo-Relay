<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LLM Codecs

Codecs provide structured, typed access to LLM request and response
payloads. Request codecs translate opaque `LLMRequest` payloads into
`AnnotatedLLMRequest` objects for intercepts. Response codecs parse raw
API responses into `AnnotatedLLMResponse` objects for event subscribers.

Nexus ships three built-in codecs that implement both request and
response decoding: **OpenAI Chat Completions**, **OpenAI Responses**, and
**Anthropic Messages**.

## The Problem

`LLMRequest` carries two fields: `headers` (metadata) and `content` (the
request payload as untyped JSON). Different SDKs put different things in
`content` -- OpenAI uses `messages` + `model` + `temperature`, NVIDIA NIM
adds `guided_json` and `nvext`, and so on. Intercept authors who want to
inspect or modify messages, model parameters, or tool definitions must
reverse-engineer each SDK's format.

The same problem exists on the response side. Different APIs return
responses in different shapes -- `choices[0].message` for OpenAI Chat,
heterogeneous `output` arrays for OpenAI Responses, `content` blocks for
Anthropic. Event subscribers that want to extract usage data, tool calls,
or finish reasons must parse each API's format manually.

## How Codecs Solve It

### Request Codecs (LlmCodec)

A request codec provides two methods:

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

When no request codec is passed, intercepts receive `None` for the
annotated request.

### Response Codecs (LlmResponseCodec)

A response codec provides a single method:

- **`decode_response(response)`** -- Parse raw JSON response into a
  normalized `AnnotatedLLMResponse`

Response codecs are **decode-only** (introspection, not modification).
They run after execution and before event dispatch:

```
func() --> raw JSON response
    |
    v
ResponseCodec.decode_response() --> AnnotatedLLMResponse (or None on error)
    |
    v
LLMEnd event carries annotated_response field
```

Response decode is **non-fatal**: if `decode_response()` returns an error,
the pipeline continues with `annotated_response: None` on the event. The
call itself is not affected.

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

## AnnotatedLLMResponse

The structured response type produced by response codecs:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str \| None` | Response ID from the API (e.g., `"chatcmpl-abc123"`, `"resp_abc123"`, `"msg_abc123"`) |
| `model` | `str \| None` | The model that actually served the request (may differ from requested model) |
| `message` | `str \| dict \| None` | The assistant's response content |
| `tool_calls` | `list[dict] \| None` | Tool calls requested by the model, normalized across APIs |
| `finish_reason` | `str \| None` | Why generation stopped, normalized: `"complete"`, `"length"`, `"tool_use"`, `"content_filter"`, or custom |
| `usage` | `dict \| None` | Token usage statistics |
| `api_specific` | `dict \| None` | API-specific response data that cannot be normalized |
| `extra` | `dict` | Catch-all for unmodeled top-level fields |

### Usage Fields

The `usage` dictionary contains:

| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` | `int \| None` | Tokens consumed by the prompt/input |
| `completion_tokens` | `int \| None` | Tokens generated in the completion/output |
| `total_tokens` | `int \| None` | Total tokens (prompt + completion) |
| `cache_read_tokens` | `int \| None` | Tokens served from prompt cache |
| `cache_write_tokens` | `int \| None` | Tokens written to prompt cache |

### Finish Reason Normalization

The `finish_reason` field maps provider-specific stop reasons to a
normalized set:

| Normalized | OpenAI Chat | OpenAI Responses | Anthropic |
|-----------|-------------|------------------|-----------|
| `complete` | `"stop"` | status `"completed"` | `"end_turn"` |
| `length` | `"length"` | incomplete + `"max_output_tokens"` | `"max_tokens"` |
| `tool_use` | `"tool_calls"` | (tool calls in output) | `"tool_use"` |
| `content_filter` | `"content_filter"` | incomplete + `"content_filter"` | -- |

Unrecognized reasons are passed through as-is.

### Tool Call Format

Response tool calls use parsed JSON arguments (not strings):

```python
{
    "id": "call_abc123",
    "name": "get_weather",
    "arguments": {"city": "NYC"}  # Parsed JSON, not a string
}
```

This differs from request-side tool calls where OpenAI sends arguments as
a JSON string. Response codecs parse the string during decode.

### API-Specific Fields

Each built-in codec populates the `api_specific` field with data unique
to that provider:

**OpenAI Chat** (`api_specific.api == "openai_chat"`):
- `logprobs` -- Token-level log probabilities
- `system_fingerprint` -- Reproducibility fingerprint
- `service_tier` -- Processing tier (e.g., `"default"`)

**OpenAI Responses** (`api_specific.api == "openai_responses"`):
- `output_items` -- Full heterogeneous output array
- `status` -- Response status (e.g., `"completed"`, `"incomplete"`)
- `incomplete_details` -- Details about incomplete responses

**Anthropic Messages** (`api_specific.api == "anthropic_messages"`):
- `stop_sequence` -- Which stop sequence was matched
- `content_blocks` -- Full content blocks array

### Helper Methods

```python
annotated_response.response_text()   # Returns text content (str or None)
annotated_response.has_tool_calls()  # True if tool_calls is non-empty
```

## Built-In Codecs

Nexus ships three built-in codecs. Each implements **both** `LlmCodec`
(request decode/encode) and `LlmResponseCodec` (response decode), so a
single codec instance can be passed as both `codec=` and
`response_codec=`.

### OpenAI Chat Completions (OpenAIChatCodec)

Handles the OpenAI Chat Completions API format (`/v1/chat/completions`).

**Request handling:**
- `messages` array with role-tagged messages
- `model`, `temperature`, `max_tokens` / `max_completion_tokens`, `top_p`, `stop`
- `tools` and `tool_choice`
- Unmodeled fields (`stream`, `seed`, `response_format`, etc.) preserved in `extra`

**Response handling:**
- Extracts first choice from `choices[0].message`
- Parses tool call arguments from JSON strings
- Maps `finish_reason`: `"stop"` to `complete`, `"length"` to `length`,
  `"tool_calls"` to `tool_use`, `"content_filter"` to `content_filter`
- Extracts usage including `prompt_tokens_details.cached_tokens`

### OpenAI Responses (OpenAIResponsesCodec)

Handles the OpenAI Responses API format (`/v1/responses`).

**Request handling:**
- `input` (string or array) mapped to messages; `instructions` mapped to
  system message
- `model`, `temperature`, `max_output_tokens`, `top_p`
- `tools` and `tool_choice`

**Response handling:**
- Processes heterogeneous `output` array: `message` items for text,
  `function_call` items for tool calls, `reasoning` items captured in
  `api_specific`
- Uses `call_id` (not `id`) for tool call identification
- Maps status: `"completed"` to `complete`, `"incomplete"` +
  `"max_output_tokens"` to `length`, `"incomplete"` + `"content_filter"`
  to `content_filter`
- Maps usage: `input_tokens` to `prompt_tokens`, `output_tokens` to
  `completion_tokens`

### Anthropic Messages (AnthropicMessagesCodec)

Handles the Anthropic Messages API format (`/v1/messages`).

**Request handling:**
- Top-level `system` field mapped to system message (supports both string
  and array-of-content-blocks formats)
- `messages` array with Anthropic's content block format
- `model`, `max_tokens` (required by Anthropic), `temperature`, `top_p`,
  `stop_sequences`
- `tool_choice`: `{"type": "auto"}` to `auto`, `{"type": "any"}` to
  `required`, `{"type": "tool", "name": "X"}` to specific
- `tools` with `input_schema` instead of `parameters`

**Response handling:**
- Processes `content` blocks: `text` for message content, `tool_use` for
  tool calls (arguments already parsed JSON)
- Maps `stop_reason`: `"end_turn"` to `complete`, `"max_tokens"` to
  `length`, `"tool_use"` to `tool_use`
- Extracts usage including `cache_read_input_tokens` and
  `cache_creation_input_tokens`

## Event Enrichment

When codecs are active, LLM lifecycle events carry structured data:

- **`LLMStartEvent.annotated_request`** -- Present when a request codec
  is passed. Contains the `AnnotatedLLMRequest` produced by
  `codec.decode()`.
- **`LLMEndEvent.annotated_response`** -- Present when a response codec
  is passed. Contains the `AnnotatedLLMResponse` produced by
  `response_codec.decode_response()`.

Both fields are `None` when no codec is active or when decoding fails.
Internally they are `Arc`-wrapped for zero-copy sharing across subscribers.

### Accessing Annotated Data in Subscribers

```python
import nat_nexus


def my_subscriber(event):
    if event.kind == "LLMStart" and event.annotated_request is not None:
        print(f"Model: {event.annotated_request.model}")
        print(f"System prompt: {event.annotated_request.system_prompt()}")

    if event.kind == "LLMEnd" and event.annotated_response is not None:
        resp = event.annotated_response
        print(f"Model used: {resp.model}")
        print(f"Finish reason: {resp.finish_reason}")
        if resp.usage:
            print(f"Tokens: {resp.usage}")
        if resp.has_tool_calls():
            for tc in resp.tool_calls:
                print(f"Tool call: {tc['name']}({tc['arguments']})")


nat_nexus.subscribers.register("annotated_logger", my_subscriber)
```

## Using Codecs

### Per-Call Selection

Pass codec instances directly to the execute call. Both `codec` (request)
and `response_codec` (response) are optional and independent:

```python
import nat_nexus
from nat_nexus.codecs import OpenAIChatCodec

codec = OpenAIChatCodec()

# Use the same codec for both request and response
result = await nat_nexus.llm.execute(
    "gpt-4",
    request,
    call_openai,
    codec=codec,
    response_codec=codec,
)

# Or use only response codec (no request annotation)
result = await nat_nexus.llm.execute(
    "gpt-4",
    request,
    call_openai,
    response_codec=codec,
)
```

### Codec Parameter Summary

| Parameter | Type | Purpose |
|-----------|------|---------|
| `codec` | Request codec instance | Enables `AnnotatedLLMRequest` for intercepts and `LLMStartEvent.annotated_request` |
| `response_codec` | Response codec instance | Enables `AnnotatedLLMResponse` on `LLMEndEvent.annotated_response` |

Both parameters accept codec objects (not strings). There is no
string-based codec resolution -- you always pass an instance directly.

## Cross-Language Usage

### Python

```python
import nat_nexus
from nat_nexus.codecs import AnthropicMessagesCodec, OpenAIChatCodec, OpenAIResponsesCodec

# Built-in codecs implement both request and response decoding
codec = OpenAIChatCodec()

result = await nat_nexus.llm.execute(
    "gpt-4", request, call_openai,
    codec=codec,
    response_codec=codec,
)

# Streaming also supports both codecs
stream = await nat_nexus.llm.stream_execute(
    "gpt-4", request, stream_openai, collector, finalizer,
    codec=codec,
    response_codec=codec,
)

# Direct decode (useful for testing)
annotated_req = codec.decode(request)
annotated_resp = codec.decode_response(raw_response)
```

### Go

```go
import (
    "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
)

// Create a codec handle (carries both request and response codec)
codec := nat_nexus.NewOpenAIChatCodec()
// Also: nat_nexus.NewOpenAIResponsesCodec()
// Also: nat_nexus.NewAnthropicMessagesCodec()

// Pass request codec via WithLLMCodec, response codec via WithLLMResponseCodec
response, err := nat_nexus.LlmCallExecute(
    "gpt-4", request, llmFunc,
    nat_nexus.WithLLMCodec(myCodecFunc),
    nat_nexus.WithLLMResponseCodec(codec),
    nat_nexus.WithLLMModelName("gpt-4"),
)
```

The `CodecHandle` type wraps an opaque FFI handle. Create via
`NewOpenAIChatCodec()`, `NewOpenAIResponsesCodec()`, or
`NewAnthropicMessagesCodec()`. The handle is automatically freed by the
Go garbage collector.

### Node.js

```javascript
import {
    OpenAIChatCodec,
    OpenAIResponsesCodec,
    AnthropicMessagesCodec,
    llmCallExecute,
} from './index.js';

const codec = new OpenAIChatCodec();

// Built-in codecs expose decode, encode, and decodeResponse
const annotatedReq = codec.decode(request);
const annotatedResp = codec.decodeResponse(rawResponse);
const encoded = codec.encode(annotatedReq, request);

// Pass to execute: codec's decode/encode for requests,
// codec's decodeResponse for response annotation
const response = await llmCallExecute(
    "gpt-4", request, llmFunc,
    null, null, null, null, "gpt-4",
    // request codec decode, request codec encode
    (content) => codec.decode(content),
    (annotated, original) => codec.encode(annotated, original),
    // response codec decode
    (response) => codec.decodeResponse(response),
);
```

### WebAssembly

```javascript
import init, {
    WasmOpenAIChatCodec,
    WasmOpenAIResponsesCodec,
    WasmAnthropicMessagesCodec,
    llmCallExecute,
} from './pkg/nvidia_nat_nexus_wasm.js';

await init();

const codec = new WasmOpenAIChatCodec();

// Direct decode/encode
const annotatedReq = codec.decode(request);
const annotatedResp = codec.decode_response(rawResponse);
const encoded = codec.encode(annotatedReq, request);

// Pass to execute via function callbacks
const response = await llmCallExecute(
    "gpt-4", request, llmFunc,
    null, null, null, null, "gpt-4",
    (content) => codec.decode(content),
    (annotated, original) => codec.encode(annotated, original),
    (response) => codec.decode_response(response),
);
```

### Cross-Language Type Names

| Concept | Python | Go | Node.js | WASM |
|---------|--------|----|---------|------|
| OpenAI Chat codec | `OpenAIChatCodec` | `NewOpenAIChatCodec()` | `OpenAIChatCodec` | `WasmOpenAIChatCodec` |
| OpenAI Responses codec | `OpenAIResponsesCodec` | `NewOpenAIResponsesCodec()` | `OpenAIResponsesCodec` | `WasmOpenAIResponsesCodec` |
| Anthropic Messages codec | `AnthropicMessagesCodec` | `NewAnthropicMessagesCodec()` | `AnthropicMessagesCodec` | `WasmAnthropicMessagesCodec` |
| Codec handle (Go) | -- | `CodecHandle` | -- | -- |
| Response codec option (Go) | -- | `WithLLMResponseCodec(codec)` | -- | -- |
| Request codec protocol | `LlmCodec` | `CodecFunc` | function pair | function pair |
| Response codec protocol | `LlmResponseCodec` | `CodecHandle` | function | function |

## Creating a Custom Codec

### Custom Request Codec

Subclass `LlmCodec` and implement `decode` and `encode`:

```python
from nat_nexus import LLMRequest, AnnotatedLLMRequest
from nat_nexus.codecs import LlmCodec


class MyCodec(LlmCodec):
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

        return LLMRequest(original.headers, content)
```

#### Encode Must Use Merge-Not-Replace

The `encode` method receives both the modified `AnnotatedLLMRequest` and
the pre-intercept `original` LLMRequest. Always start from
`original.content` and overlay changes. This preserves SDK-specific fields
that `AnnotatedLLMRequest` does not model (e.g., `stream`,
`response_format`, `reasoning_effort`, `seed`, `logprobs`).

### Custom Response Codec

Implement the `LlmResponseCodec` protocol with a single
`decode_response` method:

```python
from nat_nexus.codecs import LlmResponseCodec


class MyResponseCodec(LlmResponseCodec):
    def decode_response(self, response: object) -> "AnnotatedLLMResponse":
        from nat_nexus import AnnotatedLLMResponse

        # Parse your custom API format into the normalized structure.
        # Return an AnnotatedLLMResponse with whichever fields are available.
        # Fields you cannot extract should be left as None.
        return AnnotatedLLMResponse(
            id=response.get("id"),
            model=response.get("model"),
            message=response.get("text"),
            finish_reason="complete" if response.get("done") else None,
            usage={
                "prompt_tokens": response.get("input_tokens"),
                "completion_tokens": response.get("output_tokens"),
            } if response.get("input_tokens") else None,
        )
```

Response codecs should return `Err` / raise only for genuinely
unparseable input. The pipeline treats decode errors as "no annotation
available" and continues normally.

### Combined Codec

Since built-in codecs implement both traits, you can do the same:

```python
class MyFullCodec(LlmCodec, LlmResponseCodec):
    def decode(self, request):
        ...
    def encode(self, annotated, original):
        ...
    def decode_response(self, response):
        ...

codec = MyFullCodec()
result = await nat_nexus.llm.execute(
    "my-llm", request, func,
    codec=codec,
    response_codec=codec,
)
```

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

### Usage Tracking Subscriber

```python
import nat_nexus


def track_usage(event):
    if event.kind != "LLMEnd":
        return
    resp = event.annotated_response
    if resp is None:
        return
    usage = resp.usage
    if usage:
        print(
            f"[{event.name}] "
            f"prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')} "
            f"total={usage.get('total_tokens')} "
            f"cached={usage.get('cache_read_tokens')}"
        )


nat_nexus.subscribers.register("usage-tracker", track_usage)
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

## Related Docs

- [Middleware Pipeline](middleware-pipeline.md) -- Full pipeline stage ordering
- [API Reference](api-reference.md) -- Core runtime function signatures
- [Getting Started: Python](getting-started-python.md) -- Python quickstart
- [Typed Wrappers](typed-wrappers.md) -- Codec-based typed APIs (distinct from LLM Codecs)
- [Typed API Reference](typed-api-reference.md) -- Typed wrapper function signatures
- [Recipes](recipes.md) -- Common integration patterns
