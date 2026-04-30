<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Advanced Guide: Using Codecs

Use this guide when a framework integration needs typed application values at its public boundary while NeMo Flow still records JSON-compatible payloads.

## What You Build

You will choose typed value codecs for framework-facing wrappers so that:

- Application code can pass native objects to framework callbacks
- NeMo Flow can emit JSON-compatible lifecycle payloads
- Middleware and subscribers receive predictable serialized values
- The framework callback still receives the application type it expects

For provider-native LLM payload normalization, use [Advanced Guide: Provider Codecs](provider-codecs.md).

## Before You Start

You need:

- A stable framework callback boundary.
- Application request and response types that can be projected into JSON.
- A managed wrapper that accepts codecs, such as the typed tool or typed LLM helpers.
- A validation path that confirms the returned value still matches framework expectations.

## What Codecs Are

A typed value codec is a pure data translator at the NeMo Flow boundary. It converts application-facing values to JSON before NeMo Flow emits events or runs JSON-based middleware, then converts JSON back into the type expected by the framework callback or caller.

Typed value codecs are different from provider codecs:

| Codec Type | Purpose | Common Use |
|---|---|---|
| Typed value codec | Converts application values to and from JSON. | Dataclasses, Pydantic models, TypeScript object shapes, custom framework types. |
| Provider codec | Converts provider-specific LLM requests and responses to annotated NeMo Flow request or response data. | OpenAI Chat, OpenAI Responses, Anthropic Messages, custom provider payloads. |

Use this page for typed value codecs. Use [Provider Codecs](provider-codecs.md) when middleware needs normalized LLM messages, tools, model names, generation parameters, or provider response annotations.

## How Typed Value Codecs Work

When a managed typed wrapper receives a codec:

1. The wrapper converts the application input into JSON before entering the NeMo Flow runtime.
2. NeMo Flow emits lifecycle events and runs middleware against JSON-compatible payloads.
3. The wrapper converts JSON back into the callback type before invoking framework-owned code when needed.
4. The wrapper converts the callback result back through the result codec before returning to the caller.

Keep codecs deterministic and side-effect free. A codec should not open network connections, mutate framework state, or hide provider I/O.

## Typed Value Codecs

Python exposes:

- `JsonPassthrough`
- `DataclassCodec`
- `PydanticCodec`
- `BestEffortAnyCodec`

Node.js exposes `JsonPassthrough` and custom `Codec<T>` implementations.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from dataclasses import dataclass

from pydantic import BaseModel

from nemo_flow.typed import DataclassCodec, JsonPassthrough, PydanticCodec


@dataclass
class SearchArgs:
    query: str


class SearchResult(BaseModel):
    hits: int


args_codec = DataclassCodec(SearchArgs)
result_codec = PydanticCodec(SearchResult)
passthrough = JsonPassthrough()
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { JsonPassthrough, type Codec, type JsonValue } from 'nemo-flow-node/typed';

type SearchArgs = { query: string };

const argsCodec: Codec<SearchArgs, JsonValue> = {
  toJson: (value) => value,
  fromJson: (json) => json as SearchArgs,
};

const passthrough = new JsonPassthrough();
```
:::

::::

Use `BestEffortAnyCodec` only at boundary code where strict schemas are unavailable. Prefer dataclass, Pydantic, or explicit Node.js codecs when the framework owns a stable schema.

## Example: Typed Tool Boundary

Use typed value codecs when the framework wants native objects but NeMo Flow should emit JSON payloads.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from dataclasses import dataclass

import nemo_flow
from nemo_flow.typed import DataclassCodec, JsonPassthrough, tool_execute


@dataclass
class SearchArgs:
    query: str


async def invoke(args: SearchArgs) -> dict[str, str]:
    return {"echo": args.query}


result = await tool_execute(
    "search",
    SearchArgs("weather"),
    invoke,
    args_codec=DataclassCodec(SearchArgs),
    result_codec=JsonPassthrough(),
)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { JsonPassthrough, typedToolExecute, type Codec, type JsonValue } from 'nemo-flow-node/typed';

type SearchArgs = { query: string };
type SearchResult = { echo: string };

const argsCodec: Codec<SearchArgs, JsonValue> = {
  toJson: (value) => value,
  fromJson: (json) => json as SearchArgs,
};

const result = await typedToolExecute<SearchArgs, SearchResult>(
  'search',
  { query: 'weather' },
  async (args) => ({ echo: args.query }),
  argsCodec,
  new JsonPassthrough(),
);
```
:::

::::

## Validation Checklist

Use this checklist to confirm the implementation preserves the expected runtime
contract.

- Codec output is JSON-compatible.
- Required fields are preserved by `toJson` and `fromJson`.
- Middleware sees the expected serialized shape.
- The framework callback receives the expected application type.
- Error paths report conversion failures close to the framework boundary.

## Next Steps

Use these links to continue from this workflow into the next related task.

- Use [Advanced Guide: Provider Codecs](provider-codecs.md) when provider payloads need normalized request or response annotations.
- Use [Advanced Guide: Handle Non-Serializable Data](non-serializable-data.md) when framework objects cannot pass through JSON payloads.
- Use [Integrate into Frameworks Code Examples](code-examples.md) for explicit fallback APIs.
