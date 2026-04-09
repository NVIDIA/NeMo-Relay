<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Best Practices: Adding a Python Agent Framework Integration

This guide walks through the recommended patterns for integrating a new Python agent
framework (e.g., CrewAI, AutoGen, Semantic Kernel) with NeMo Flow.
The [LangChain integration](../patches/langchain/) serves as the reference
implementation throughout.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Stubbing the Library](#2-stubbing-the-library)
3. [Transparent Fallback to Default Behavior](#3-transparent-fallback-to-default-behavior)
4. [Wrapping LLM Calls](#4-wrapping-llm-calls)
5. [Wrapping Tool Calls](#5-wrapping-tool-calls)
6. [Creating Scopes](#6-creating-scopes)
7. [Thread and Async Safety](#7-thread-and-async-safety)
8. [Patch Structure and File Layout](#8-patch-structure-and-file-layout)
9. [Testing](#9-testing)
10. [Checklist](#10-checklist)

---

## 1. Design Principles

Every integration must follow these rules:

- **Optional dependency** — NeMo Flow is never a hard requirement. The framework must
  work identically when `nemo_flow` is not installed.
- **Graceful degradation** — All NeMo Flow errors are caught and logged at `DEBUG`. They
  must never propagate to the framework's users.
- **Transparent wrapping** — Minimal changes to existing framework code. Existing
  tests must continue to pass without `nemo_flow` installed.
- **Explicit activation** — NeMo Flow only activates when a scope stack has been
  initialized by the caller. It never auto-creates one.

---

## 2. Stubbing the Library

Create a **lazy-import helper** module inside the framework's source tree. This is
the single point of contact for checking NeMo Flow availability.

```python
# <framework>/utils/_nemo_flow.py   (or <framework>/_nemo_flow.py)

"""Lazy import helper for optional NeMo Flow integration."""

from __future__ import annotations

import logging
from types import ModuleType

_logger = logging.getLogger(__name__)
_nemo_flow: ModuleType | None | bool = False  # False = not yet attempted


def get_nemo_flow() -> ModuleType | None:
    """Return the ``nemo_flow`` module, or ``None`` if not installed.

    The import is performed lazily on first call and cached thereafter.
    """
    global _nemo_flow  # noqa: PLW0603
    if _nemo_flow is False:
        try:
            import nemo_flow
            _nemo_flow = nemo_flow
        except ImportError:
            _nemo_flow = None
    return _nemo_flow  # type: ignore[return-value]


def is_available() -> bool:
    """Return ``True`` if NeMo Flow is installed and importable."""
    return get_nemo_flow() is not None
```

### Key decisions

| Choice | Rationale |
|---|---|
| `False` as sentinel (not `None`) | Distinguishes "not yet attempted" from "attempted and missing" |
| Module-level cache | Avoids repeated `importlib` overhead on every call |
| No top-level `import nemo_flow` | Prevents `ImportError` at framework import time |

For **provider-level** modules (LLM chat model classes), create a second bridge
module that additionally checks for an active scope stack. See
[§3 Transparent Fallback](#3-transparent-fallback-to-default-behavior) below.

---

## 3. Transparent Fallback to Default Behavior

The integration must have two code paths — one with NeMo Flow, one without — and the
framework user should never notice the difference (except for the middleware
features NeMo Flow adds).

### Provider-level availability check

For LLM providers, NeMo Flow should only activate when the caller has explicitly
initialized a scope stack. This prevents unexpected behavior when `nemo_flow` is
installed but unused:

```python
# <framework>/chat_models/_nemo_flow.py

try:
    import nemo_flow
    from nemo_flow import LLMRequest
    _HAS_NEMO_FLOW = True
except ImportError:
    _HAS_NEMO_FLOW = False


def available() -> bool:
    """Return True when nemo_flow is importable *and* a scope stack is active."""
    if not _HAS_NEMO_FLOW:
        return False
    try:
        return nemo_flow.scope_stack_active()
    except Exception:
        return False
```

### Branching pattern

At every integration point, use a simple if/else branch. The `else` branch must
be the **original, unmodified code**:

```python
# In a tool execution method:
if (nnex := get_nemo_flow()) is not None:
    # NeMo Flow-wrapped path
    codec = nnex.typed.BestEffortAnyCodec()
    response = await nnex.typed.tool_execute(
        self.name, tool_input, _func, codec, codec,
    )
else:
    # Original behavior — completely unchanged
    response = await _func(tool_input)
```

### Error silencing

Wrap **every** NeMo Flow call in try/except at the callback and scope-management
layers. Use `DEBUG`-level logging only — no warnings, no user-visible messages:

```python
try:
    handle = scope.push(name, nnex.ScopeType.Agent, handle=parent)
    self._scope_handles[run_id] = handle
except Exception:
    _logger.debug("NeMo Flow: scope push failed", exc_info=True)
```

The tool and LLM execution wrappers (`typed.tool_execute`, `typed.llm_execute`)
already handle errors internally, so you do not need extra try/except around those.

---

## 4. Wrapping LLM Calls

LLM calls are wrapped at the **provider level** (the class that actually makes HTTP
requests), not at an abstract base class level. This gives access to the raw
request payload and SDK response.

### Non-streaming calls

Use `nemo_flow.typed.llm_execute` with `JsonPassthrough` (since provider code
already converts SDK responses to dicts via `model_dump()` or equivalent):

```python
from <framework>.chat_models import _nemo_flow

# Inside the _generate or _call method:
if _nemo_flow.available():
    request = _nemo_flow.make_request(payload, extra_headers)

    async def _call(req):
        # Use req.content (dict) and req.headers (dict) to make the real call
        raw = await self._async_client.chat.completions.create(**req.content)
        return raw.model_dump()

    resp_dict = _nemo_flow.run_sync(
        _nemo_flow.llm_execute(self.model_name, request, _call)
    )
    return self._process_response(resp_dict)

# ... original code path below ...
```

### Streaming calls

Use `nemo_flow.typed.llm_stream_execute` with a collector/finalizer pattern:

```python
if _nemo_flow.available():
    request = _nemo_flow.make_request(payload)
    collected: list[dict] = []

    async def _call(req):
        """Async generator yielding chunk dicts."""
        stream = await self._async_client.chat.completions.create(
            **req.content, stream=True
        )
        async for chunk in stream:
            yield chunk.model_dump()

    def _collector(chunk):
        collected.append(chunk)

    def _finalizer():
        return collected[-1] if collected else {}

    # Async context — use directly
    stream = await _nemo_flow.llm_stream_execute(
        self.model_name, request, _call, _collector, _finalizer
    )
    async for chunk_dict in stream:
        yield self._convert_chunk(chunk_dict)
    return

# ... original streaming code path below ...
```

### The bridge module pattern

Each provider package should have its own `_nemo_flow.py` bridge that wraps the
typed API with the correct codec. This keeps the main chat model code clean:

```python
# <framework>/chat_models/_nemo_flow.py

async def llm_execute(model_name, request, func):
    codec = nemo_flow.typed.JsonPassthrough()
    return await nemo_flow.typed.llm_execute(
        model_name, request, func, codec, model_name=model_name,
    )

async def llm_stream_execute(model_name, request, func, collector, finalizer):
    codec = nemo_flow.typed.JsonPassthrough()
    return await nemo_flow.typed.llm_stream_execute(
        model_name, request, func, collector, finalizer,
        codec, codec, model_name=model_name,
    )

def make_request(payload, extra_headers=None):
    return nemo_flow.LLMRequest(extra_headers or {}, payload)

def run_sync(coro):
    """Run a coroutine from sync context, handling running event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
```

---

## 5. Wrapping Tool Calls

Tool calls are wrapped at the framework's **tool execution entry point** — wherever
the framework calls the user's tool function.

### Async tool execution

```python
from <framework>.utils._nemo_flow import get_nemo_flow

# Inside the tool's async invoke/execute method:
async def _func(_args):
    """Wraps the user's tool function."""
    return await self._arun(*tool_args, **tool_kwargs)

if (nnex := get_nemo_flow()) is not None:
    codec = nnex.typed.BestEffortAnyCodec()
    response = await nnex.typed.tool_execute(
        self.name,       # tool name
        tool_input,      # raw args (dict, dataclass, Pydantic model, etc.)
        _func,           # wrapped callable
        codec,           # args codec
        codec,           # result codec
    )
else:
    response = await _func(tool_input)
```

### Sync tool execution (with event loop handling)

When wrapping a synchronous tool call, you need to bridge into NeMo Flow's async
pipeline. This requires handling the case where an event loop may already be
running (e.g., Jupyter, nested async frameworks):

```python
def _func(_args):
    return context.run(self._run, *tool_args, **tool_kwargs)

if (nnex := get_nemo_flow()) is not None:
    import asyncio
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    codec = nnex.typed.BestEffortAnyCodec()
    ctx = contextvars.copy_context()
    scope_stack = nnex.get_scope_stack()

    async def _nemo_flow_run():
        return await nnex.typed.tool_execute(
            self.name, tool_input, _func, codec, codec,
        )

    def _run_with_scope_stack():
        # Propagate scope stack to the worker thread's Rust thread-local
        nnex.set_thread_scope_stack(scope_stack)
        return asyncio.run(_nemo_flow_run())

    try:
        asyncio.get_running_loop()
        # Already in an async context — offload to a thread
        with ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(ctx.run, _run_with_scope_stack).result()
    except RuntimeError:
        # No event loop — safe to use asyncio.run directly
        response = asyncio.run(_nemo_flow_run())
else:
    response = _func(tool_input)
```

### Choosing a codec

| Situation | Codec | Why |
|---|---|---|
| Tool args/results are arbitrary Python objects | `BestEffortAnyCodec()` | Handles Pydantic, dataclasses, plain dicts, and fallback |
| Tool args/results are always plain dicts | `JsonPassthrough()` | Zero overhead, no conversion |
| You know the exact Pydantic model | `PydanticCodec(MyModel)` | Type-safe roundtrip |
| You know the exact dataclass | `DataclassCodec(MyDC)` | Type-safe roundtrip |

For framework integrations where tool input/output types are unknown at integration
time, **always use `BestEffortAnyCodec`**.

---

## 6. Creating Scopes

Scopes track the execution hierarchy. The framework integration should map the
framework's own hierarchy concept to NeMo Flow scopes.

### Scope types

```python
import nemo_flow

nemo_flow.ScopeType.Function  # Individual function/step
nemo_flow.ScopeType.Agent     # An agent or chain invocation
```

### Agent scopes

Create an `Agent` scope when a chain, agent, or high-level runnable starts
execution. Pop it when execution ends (including on error):

```python
# LangChain maps this via a callback handler:
class NemoFlowCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self._scope_handles: dict[UUID, Any] = {}
        self._nnex = _try_import()

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
        if self._nnex is None:
            return
        try:
            name = serialized.get("name", "Unknown")
            parent = self._scope_handles.get(parent_run_id) if parent_run_id else None
            handle = self._nnex.scope.push(
                name,
                self._nnex.ScopeType.Agent,
                handle=parent,  # preserves hierarchy
            )
            self._scope_handles[run_id] = handle
        except Exception:
            _logger.debug("NeMo Flow: scope push failed", exc_info=True)

    def on_chain_end(self, outputs, *, run_id, **kw):
        self._pop_scope(run_id)

    def on_chain_error(self, error, *, run_id, **kw):
        self._pop_scope(run_id)  # always pop, even on error

    def _pop_scope(self, run_id):
        if self._nnex is None:
            return
        handle = self._scope_handles.pop(run_id, None)
        if handle is None:
            return
        try:
            self._nnex.scope.pop(handle)
        except Exception:
            _logger.debug("NeMo Flow: scope.pop failed", exc_info=True)
```

If the framework doesn't have a callback/hook system, you can use the context
manager form instead:

```python
with nemo_flow.scope.scope("my-agent", nemo_flow.ScopeType.Agent) as handle:
    result = await agent.run(task)
```

### Function scopes

Create `Function` scopes for discrete steps within an agent (e.g., a planner step,
a retrieval step). These are typically shorter-lived than agent scopes:

```python
with nemo_flow.scope.scope("plan", nemo_flow.ScopeType.Function) as handle:
    plan = await planner.generate(task)
```

### Parallel batch calls

When the framework runs multiple operations concurrently (e.g., parallel tool
calls, fan-out steps), each parallel branch should get its own scope. All branches
share the same parent:

```python
import asyncio
import nemo_flow

parent_handle = nemo_flow.scope.get_handle()

async def run_branch(name, task):
    # Each branch gets its own child scope
    with nemo_flow.scope.scope(
        name, nemo_flow.ScopeType.Function, handle=parent_handle
    ):
        return await execute(task)

# Fan-out — scopes are siblings under the same parent
results = await asyncio.gather(
    run_branch("search-web", web_query),
    run_branch("search-db", db_query),
    run_branch("search-cache", cache_query),
)
```

For frameworks that have a built-in parallel execution primitive (like LangChain's
`RunnableParallel`), the callback-based approach handles this automatically — each
parallel branch gets its own `run_id` with the same `parent_run_id`.

### Scope lifecycle rules

1. **Always pop what you push** — use try/finally or the context manager.
2. **Pop on error** — a scope must be popped even if the operation fails.
3. **Map the framework's hierarchy** — if the framework has parent/child
   relationships (run IDs, task IDs), use the `handle=` parameter to preserve them.
4. **Don't over-scope** — only create scopes at meaningful execution boundaries, not
   for every internal helper function.

---

## 7. Thread and Async Safety

NeMo Flow uses both Python `contextvars` (for async task isolation) and Rust
thread-locals (for native FFI calls). The integration must keep these in sync.

### Scope stack propagation

When spawning a new thread or offloading work to a thread pool, propagate the scope
stack using `propagate_scope_to_thread()`:

```python
import contextvars
from concurrent.futures import ThreadPoolExecutor

# Capture current scope stack (raises RuntimeError if none is active)
scope_stack = nnex.propagate_scope_to_thread()
ctx = contextvars.copy_context()

def worker():
    # Bind the Rust thread-local in the worker thread
    nnex.set_thread_scope_stack(scope_stack)
    return do_work()

with ThreadPoolExecutor(max_workers=1) as pool:
    result = pool.submit(ctx.run, worker).result()
```

You can also check whether a scope stack is active before attempting propagation:

```python
if nnex.scope_stack_active():
    scope_stack = nnex.propagate_scope_to_thread()
    # ... propagate to worker ...
```

### Sync-to-async bridge

NeMo Flow's execute pipeline is async. When integrating into a sync code path:

```python
def run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Event loop already running (Jupyter, nested frameworks)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
```

---

## 8. Patch Structure and File Layout

Integrations are maintained as git patches applied to local upstream checkouts
under `third_party/`. The repository tracks only the patch files and a pinned
manifest; the upstream clones themselves are bootstrapped locally and are not
tracked as submodules on `main`. Follow this directory structure:

```text
third_party/
  sources.lock               # tracked manifest (git-config syntax)
  <framework>/               # local git checkout bootstrapped from sources.lock

patches/
  <framework>/
    0001-add-nemo-flow-integration.patch
```

Bootstrap the local upstream checkouts before applying or refreshing patches:

```bash
./scripts/bootstrap-third-party.sh
```

### Files to add/modify

A typical integration touches these files:

| File | Purpose |
|---|---|
| `<framework>/utils/_nemo_flow.py` | Lazy import helper (new file) |
| `<framework>/chat_models/_nemo_flow.py` | Provider bridge with `available()`, `llm_execute`, `run_sync` (new file) |
| `<framework>/tools/base.py` | Tool execution wrapping (modify existing) |
| `<framework>/callbacks/nemo_flow_handler.py` | Scope lifecycle callback handler (new file, if the framework supports callbacks) |
| `tests/.../test_nemo_flow_handler.py` | Tests for scope lifecycle (new file) |

### Generating the patch

```bash
cd third_party/<framework>
# Make your changes...
git diff HEAD -- . > ../../patches/<framework>/0001-add-nemo-flow-integration.patch
```

### Applying the patch

```bash
./scripts/apply-patches.sh
```

---

## 9. Testing

### Unit tests (within the framework's test suite)

Test the callback handler / scope management with mocked NeMo Flow:

```python
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

def _make_mock_nnex():
    nnex = ModuleType("nemo_flow")
    nnex.ScopeType = SimpleNamespace(Agent="Agent")
    scope = MagicMock()
    scope.push = MagicMock(
        side_effect=lambda name, st, **kw: SimpleNamespace(uuid=str(uuid4()))
    )
    scope.pop = MagicMock()
    nnex.scope = scope
    return nnex


class TestScopeLifecycle:
    def test_chain_start_pushes_scope(self):
        handler = NemoFlowCallbackHandler()
        handler._nnex = _make_mock_nnex()
        run_id = uuid4()
        handler.on_chain_start({"name": "MyAgent"}, {}, run_id=run_id)
        handler._nnex.scope.push.assert_called_once()

    def test_chain_end_pops_scope(self):
        handler = NemoFlowCallbackHandler()
        handler._nnex = _make_mock_nnex()
        run_id = uuid4()
        handler.on_chain_start({"name": "MyAgent"}, {}, run_id=run_id)
        handler.on_chain_end({}, run_id=run_id)
        handler._nnex.scope.pop.assert_called_once()

    def test_no_nemo_flow_is_silent_noop(self):
        handler = NemoFlowCallbackHandler()
        handler._nnex = None
        handler.on_chain_start({"name": "x"}, {}, run_id=uuid4())  # no error

    def test_nemo_flow_error_is_swallowed(self):
        mock = _make_mock_nnex()
        mock.scope.push.side_effect = RuntimeError("boom")
        handler = NemoFlowCallbackHandler()
        handler._nnex = mock
        handler.on_chain_start({"name": "x"}, {}, run_id=uuid4())  # no error
```

### What to verify

- Framework's existing test suite passes **without** `nemo_flow` installed.
- Scope push/pop called correctly for agent/chain lifecycle events.
- Parent scope handles are propagated for nested invocations.
- All NeMo Flow errors are swallowed — never surface to the framework.
- End-without-start is a no-op (no crash).
- Tool and LLM execute calls go through NeMo Flow when available, fall back when not.

---

## 10. Checklist

Before submitting your integration:

- [ ] **Lazy import helper** — single module, cached, `False` sentinel
- [ ] **Provider bridge** — `available()` checks scope stack, `llm_execute`,
      `llm_stream_execute`, `make_request`, `run_sync`
- [ ] **Tool wrapping** — `BestEffortAnyCodec`, sync + async paths, scope stack
      propagation for thread offloading
- [ ] **LLM wrapping** — non-streaming via `llm_execute`, streaming via
      `llm_stream_execute` with collector/finalizer
- [ ] **Scope management** — agent scopes for chains/agents, function scopes for
      steps, proper push/pop on error
- [ ] **Parallel scopes** — each concurrent branch gets its own child scope under a
      shared parent
- [ ] **Error silencing** — all NeMo Flow errors caught and logged at `DEBUG`
- [ ] **No hard dependency** — framework works identically without `nemo_flow`
- [ ] **Tests** — scope lifecycle, graceful no-op, error swallowing
- [ ] **Patch generated** — `patches/<framework>/0001-add-nemo-flow-integration.patch`
- [ ] **SPDX headers** — Apache-2.0 on all new files

---

## Other Languages

This guide focuses on Python integrations. For Go, Node.js, and WASM
integration patterns, see [Language Bindings](language-bindings.md) which
covers setup, usage examples, and language-specific considerations for each
binding layer. The same architectural principles apply: lazy import,
transparent fallback, scope lifecycle management, and thread/async safety.
