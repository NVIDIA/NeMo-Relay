<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Typed API Reference

This document covers the typed helper APIs exposed through `nemo_flow.typed`.
Use [Typed Wrappers](typed-wrappers.md) for the conceptual model and data-flow
diagrams; use this page for function signatures, built-in codecs, and behavior
at the Python API boundary.

## Overview

Typed helpers keep the core runtime on JSON while letting application code work
with structured Python objects. The public Python entry points are:

- `tool_execute(...)`
- `llm_execute(...)`
- `llm_stream_execute(...)`

All three APIs are thin wrappers over the core runtime:

- tool helpers call `nemo_flow.tools.execute(...)`
- LLM helpers call `nemo_flow.llm.execute(...)`
- streaming helpers call `nemo_flow.llm.stream_execute(...)`

## Codec Protocol

Every typed API takes an explicit codec object:

```python
class Codec(Generic[T]):
    def to_json(self, value: T) -> Json: ...
    def from_json(self, data: Json) -> T: ...
```

Contract:

- `to_json(...)` must return JSON-serializable data.
- `from_json(...)` must accept the JSON shape produced by the runtime after
  middleware has had a chance to rewrite it.
- Intercepts and guardrails always operate on JSON, not typed objects.

## Built-In Python Codecs

### `JsonPassthrough`

Identity codec for already-JSON-safe data.

### `DataclassCodec[T]`

Encodes and decodes `dataclasses.dataclass` instances via `asdict()` and
constructor rehydration.

### `PydanticCodec[T]`

Encodes and decodes Pydantic `BaseModel` subclasses via `model_dump()` and
`model_validate()`.

### `BestEffortAnyCodec`

Heuristic codec for arbitrary Python values. Encoding priority:

1. Pydantic model
2. Dataclass
3. JSON-serializable object
4. Pickle fallback
5. String fallback

It also keeps an in-process runtime type registry so function-local dataclasses
and Pydantic models can round-trip within one Python process.

Security warning:

- The pickle fallback can execute arbitrary code during deserialization.
- Using `BestEffortAnyCodec` with untrusted input is unsafe.
- The runtime type registry does not make untrusted deserialization safe.

Recommendations:

- Prefer JSON-serializable objects when possible.
- Prefer explicit schemas such as dataclasses or Pydantic models.
- Disable or avoid the pickle fallback if you do not fully trust the source
  data.

Simple mitigations:

- Only accept trusted serialized data.
- Validate signatures and/or encryption before deserializing.
- Avoid pickle entirely when crossing trust boundaries.

## `tool_execute`

```python
async def tool_execute(
    name: str,
    args: TArgs,
    func: Callable[[TArgs], TResult] | Callable[[TArgs], Awaitable[TResult]],
    args_codec: Codec[TArgs],
    result_codec: Codec[TResult],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
) -> TResult
```

Behavior:

- Encodes `args` with `args_codec.to_json(...)`
- Runs the standard tool middleware pipeline on JSON
- Decodes intercepted JSON back into `TArgs` before calling `func(...)`
- Encodes the typed result with `result_codec.to_json(...)`
- Decodes the final JSON result back into `TResult`

## `llm_execute`

```python
async def llm_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], TResponse] | Callable[[LLMRequest], Awaitable[TResponse]],
    response_json_codec: Codec[TResponse],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
    codec: LlmCodec | None = None,
    response_codec: LlmResponseCodec | None = None,
) -> TResponse
```

Behavior:

- `request` stays as an `LLMRequest`; it is not codec-encoded
- `func(...)` returns a typed response
- `response_json_codec` serializes that response into JSON for the middleware path
- optional `codec` enables annotated request intercepts
- optional `response_codec` adds an `AnnotatedLLMResponse` to emitted end events
- The final JSON response is decoded back into `TResponse`

## `llm_stream_execute`

```python
async def llm_stream_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], AsyncIterator[TResponseChunk]],
    collector: Callable[[TResponseChunk], None],
    finalizer: Callable[[], TResponse],
    chunk_json_codec: Codec[TResponseChunk],
    response_json_codec: Codec[TResponse],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
    codec: LlmCodec | None = None,
    response_codec: LlmResponseCodec | None = None,
) -> LlmStream
```

Behavior:

- Each yielded typed chunk is encoded with `chunk_json_codec`
- Streaming middleware sees JSON chunks
- The user-supplied `collector(...)` receives typed chunks after JSON is
  decoded back through `chunk_json_codec`
- `finalizer()` returns a typed aggregated response, encoded with
  `response_json_codec` before sanitize-response guardrails and end-event emission
- optional `codec` enables annotated request intercepts
- optional `response_codec` adds an `AnnotatedLLMResponse` to emitted end events

## Notes and Limits

- Typed helpers do not bypass middleware; they only add encode/decode steps at
  the API boundary.
- Managed sanitize guardrails still affect emitted lifecycle payloads rather
  than the typed values seen by user code.
- Any JSON rewrites performed by intercepts must still decode successfully via
  the selected codec.
- `BestEffortAnyCodec` can round-trip more values than strict codecs, but it is
  less explicit and should not replace domain-specific codecs by default.

## Node.js Surface

Node.js exposes analogous helpers through `typed.js` and `typed.d.ts`:

- `typedToolExecute(...)`
- `typedLlmExecute(...)`
- `typedLlmStreamExecute(...)`

The codec contract is the same: middleware sees JSON and application code sees
typed values defined by the codec.

## Related Docs

- [Typed Wrappers](typed-wrappers.md)
- [API Reference](api-reference.md)
- [Language Bindings](language-bindings.md)
