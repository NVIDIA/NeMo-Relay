<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python Quick Start

This quick start shows the smallest Python workflow that emits scope, tool, and LLM events.

[LangChain](https://www.langchain.com/langchain) and [LangGraph](https://www.langchain.com/langgraph) users should start with the [LangChain integration](langchain.md) or [LangGraph integration](langgraph.md) guides for the best experience in those frameworks.

## Choose an Install Path

Pick the installation path that matches whether you are using a local checkout or a
published package.

### Install from the Repository

Use this path when you are working from a local checkout and need editable source
behavior.

```bash
uv sync
```

This is the right path when you are working from a local checkout and want the editable package plus the native extension build.

If you are consuming the local checkout from another `uv` project, add the source
path from that application's directory instead:

```bash
uv add --editable ../NeMo-Flow
```

This records the local source in the application's `pyproject.toml` through
`[tool.uv.sources]`.

### Install from a Package Manager

Use this path when you want the published package for application development.

```bash
uv add nemo-flow
```

Run `uv add` from an application project that has a `pyproject.toml`; it records
`nemo-flow` as a dependency. If you are only installing into an active virtual
environment, use `uv pip install nemo-flow`. If you are not using `uv`, install
the published package with `pip install nemo-flow`.

## Run One Scope, One Tool Call, and One LLM Call

The example below runs one minimal instrumented workflow through the binding.

```python
import asyncio

import nemo_flow


def on_event(event) -> None:
    print(f"event={event.kind} name={event.name}")


async def search(args):
    return {"echo": args["query"]}


async def model(request):
    return {
        "messages": request.content["messages"],
        "ok": True,
    }


async def main():
    nemo_flow.subscribers.register("quickstart-printer", on_event)

    with nemo_flow.scope.scope("demo-agent", nemo_flow.ScopeType.Agent) as handle:
        nemo_flow.scope.event("initialized", handle=handle, data={"binding": "python"})

        tool_result = await nemo_flow.tools.execute("search", {"query": "hello"}, search, handle=handle)
        llm_result = await nemo_flow.llm.execute(
            "demo-provider",
            nemo_flow.LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}]}),
            model,
            handle=handle,
        )

        print(tool_result)
        print(llm_result)

    nemo_flow.subscribers.deregister("quickstart-printer")


asyncio.run(main())
```

## What Success Looks Like

You should see:

- Event lines for the scope, tool, LLM, and mark lifecycle
- `{'echo': 'hello'}` from the tool call
- A final object containing `ok: True` and the echoed message payload from the LLM callback

If you only see the returned values and no event lines, the callbacks ran but
you did not verify instrumentation. The subscriber output is the fast check that
NeMo Flow actually emitted lifecycle events.

## Where the Python Surface Lives

These modules are the main Python APIs to use from applications and integrations.

- `nemo_flow.scope`
- `nemo_flow.tools`
- `nemo_flow.llm`
- `nemo_flow.guardrails`
- `nemo_flow.intercepts`
- `nemo_flow.subscribers`
- `nemo_flow.plugin`
- `nemo_flow.adaptive`
- `nemo_flow.typed`
- `nemo_flow.codecs`

## What to Learn Next

Use these links to continue from the quick start into the core runtime concepts.

- [LangChain integration](langchain.md)
- [LangGraph integration](langgraph.md)
- [Scopes](../../about/concepts/scopes.md)
- [Middleware](../../about/concepts/middleware.md)
- [Plugins](../../about/concepts/plugins.md)

## Framework Integrations

Use these guides when your Python application already uses LangChain or
LangGraph and you want NeMo Flow observability through their public APIs.

```{toctree}
:maxdepth: 1

LangChain Integration <langchain>
LangGraph Integration <langgraph>
```
