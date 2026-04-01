<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Typed Wrappers

Typed wrappers provide a generic serialization layer that converts typed domain objects to/from JSON at API boundaries. The Rust core middleware pipeline operates on plain JSON throughout; typed wrappers encode before entry and decode after exit, giving user code full type safety while keeping middleware uniform.

Available in **Python** (`nat_nexus.typed`) and **Node.js** (`typed.js` / `typed.d.ts`).

## The Codec Pattern

All typed wrappers use an explicit `Codec[T]` that defines bidirectional conversion:

```python
# Python
class Codec(Generic[T]):
    def to_json(self, value: T) -> Json:
        raise NotImplementedError
    def from_json(self, data: Json) -> T:
        raise NotImplementedError
```

```typescript
// TypeScript
interface Codec<T> {
    toJson(value: T): any;
    fromJson(data: any): T;
}
```

The codec is passed explicitly to every typed execute function — no implicit serialization magic.

## Built-in Codecs

### JsonPassthrough

Identity codec — returns value unchanged. Zero overhead.

```python
passthrough = JsonPassthrough()
# to_json({"key": "val"}) → {"key": "val"}
# from_json({"key": "val"}) → {"key": "val"}
```

### DataclassCodec (Python)

Encodes/decodes `dataclasses.dataclass` types via `asdict()` / `cls(**data)`.

```python
@dataclasses.dataclass
class SearchArgs:
    query: str
    limit: int = 10

codec = DataclassCodec(SearchArgs)
# to_json(SearchArgs(query="hello")) → {"query": "hello", "limit": 10}
# from_json({"query": "hello"})      → SearchArgs(query="hello", limit=10)
```

### PydanticCodec (Python)

Encodes/decodes Pydantic `BaseModel` subclasses via `model_dump()` / `model_validate()`.

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

codec = PydanticCodec(User)
```

### BestEffortAnyCodec (Python)

Heuristic codec for arbitrary Python values. Encoding priority:

1. Pydantic `BaseModel` → `{"__nv_pydantic__": "module.Class", "data": json_str}`
2. Dataclass → `{"__nv_dataclass__": "module.Class", "data": dict}`
3. JSON-serializable → pass through
4. Pickle fallback → `{"__nv_pickle__": "module.Class", "data": b64}`
5. String fallback → `{"__nv_fallback_str__": "module.Class", "data": str}`

### Custom Codecs

Subclass `Codec[T]` for domain-specific encoding:

```python
class PointCodec(Codec[Point]):
    def to_json(self, p: Point) -> dict:
        return {"x": p.x, "y": p.y}
    def from_json(self, data: dict) -> Point:
        return Point(data["x"], data["y"])
```

```javascript
const pointCodec = {
    toJson(p) { return { x: p.x, y: p.y }; },
    fromJson(d) { return new Point(d.x, d.y); },
};
```

## Tool Execute Flow

```python
result = await typed.tool_execute(
    name, args, func, args_codec, result_codec,
    *, handle=None, attributes=None, data=None, metadata=None,
)
```

Data flow:

```
User code (typed)                Middleware (JSON)               User func (typed)
─────────────────               ─────────────────               ─────────────────
args: TArgs
  │
  ├─ args_codec.to_json() ──→  json_args
  │                               │
  │                         request intercepts
  │                         sanitize request guards
  │                         execution intercepts
  │                               │
  │                         json_args_inner ──→ args_codec.from_json()
  │                                                    │
  │                                              func(typed_args) → typed_result
  │                                                    │
  │                         result_codec.to_json() ←───┘
  │                               │
  │                         response intercepts
  │                         sanitize response guards
  │                               │
  ├─ result_codec.from_json() ←──┘
  │
result: TResult
```

**Key**: Middleware always sees JSON. Intercepts can modify values freely; changes persist through the decode step.

## LLM Execute Flow

```python
result = await typed.llm_execute(
    name, request, func, response_codec,
    *, handle=None, attributes=None, data=None, metadata=None, model_name=None,
)
```

The `LLMRequest` passes through unchanged (not encoded via a codec). Only the **response** is typed:

```
request (LLMRequest) ──→ middleware pipeline ──→ func(request) → typed_response
                                                      │
                                              response_codec.to_json()
                                                      │
                                              middleware (sanitize response)
                                                      │
                                              response_codec.from_json() → TResponse
```

## LLM Stream Execute Flow

```python
stream = await typed.llm_stream_execute(
    name, request, func, collector, finalizer,
    chunk_codec, response_codec,
    *, handle=None, attributes=None, data=None, metadata=None, model_name=None,
)
```

Two codecs are needed — one for per-chunk encoding, one for the final aggregated response:

```
func(request) yields TResponseChunk instances
  │
  ├─ chunk_codec.to_json(chunk) ──→ JSON chunks flow through middleware
  │                                      │
  │                                 chunk_codec.from_json(json_chunk)
  │                                      │
  │                                 collector(typed_chunk)  ← user accumulates
  │
  └─ stream exhausted
       │
       finalizer() → TResponse
       │
       response_codec.to_json() ──→ sanitize response guardrails
                                          │
                                     emit End event
```

## Intercept Integration

Intercepts see JSON, not typed objects. This means intercepts work uniformly regardless of codec choice:

```python
def intercept_fn(tool_name, args):
    # tool_name is the tool being called (e.g. "search")
    # args is a plain dict, not a SearchArgs dataclass
    args["limit"] = 99
    return args

intercepts.register_tool_request("cap_limit", 1, False, intercept_fn)

async def search(args: SearchArgs) -> SearchResult:
    assert args.limit == 99  # Sees the intercept's modification
    return SearchResult(items=[], total=0)

result = await typed.tool_execute(
    "search", SearchArgs(query="test", limit=5), search,
    search_args_codec, search_result_codec,
)
```

## Codec Selection Guide

| Use Case | Codec | Overhead |
|----------|-------|----------|
| JSON dicts/maps | `JsonPassthrough` | Zero |
| Python `@dataclass` | `DataclassCodec(Cls)` | Low |
| Pydantic `BaseModel` | `PydanticCodec(Cls)` | Low |
| Custom domain objects | Custom `Codec[T]` | Varies |
| Unknown/mixed types | `BestEffortAnyCodec` | Higher |

## Node.js Stream Bridge

Node.js uses a push-based stream bridge due to NAPI limitations with resolving JS Promises from native code:

1. The wrapper receives `{ __nat_nexus_native: req, __nat_nexus_stream_id: streamId }`
2. JavaScript drives async generator iteration
3. Each chunk is pushed via `lib.pushStreamChunk(streamId, json_chunk)`
4. Stream end is signaled via `lib.endStream(streamId)`

This is the only binding-specific divergence; Python and WASM use standard async iterators.
