<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Best Practices: Adding a Python Agent Framework Integration

This guide walks through the recommended patterns for integrating a new Python agent
framework (e.g., CrewAI, AutoGen, Semantic Kernel) with NeMo Agent Toolkit Nexus.
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

- **Optional dependency** — Nexus is never a hard requirement. The framework must
  work identically when `nat_nexus` is not installed.
- **Graceful degradation** — All Nexus errors are caught and logged at `DEBUG`. They
  must never propagate to the framework's users.
- **Transparent wrapping** — Minimal changes to existing framework code. Existing
  tests must continue to pass without `nat_nexus` installed.
- **Explicit activation** — Nexus only activates when a scope stack has been
  initialized by the caller. It never auto-creates one.

---

## 2. Stubbing the Library

Create a **lazy-import helper** module inside the framework's source tree. This is
the single point of contact for checking Nexus availability.

```python
# <framework>/utils/_nat_nexus.py   (or <framework>/_nat_nexus.py)

"""Lazy import helper for optional Nexus integration."""

from __future__ import annotations

import logging
from types import ModuleType

_logger = logging.getLogger(__name__)
_nat_nexus: ModuleType | None | bool = False  # False = not yet attempted


def get_nat_nexus() -> ModuleType | None:
    """Return the ``nat_nexus`` module, or ``None`` if not installed.

    The import is performed lazily on first call and cached thereafter.
    """
    global _nat_nexus  # noqa: PLW0603
    if _nat_nexus is False:
        try:
            import nat_nexus
            _nat_nexus = nat_nexus
        except ImportError:
            _nat_nexus = None
    return _nat_nexus  # type: ignore[return-value]


def is_available() -> bool:
    """Return ``True`` if Nexus is installed and importable."""
    return get_nat_nexus() is not None
```

### Key decisions

| Choice | Rationale |
|---|---|
| `False` as sentinel (not `None`) | Distinguishes "not yet attempted" from "attempted and missing" |
| Module-level cache | Avoids repeated `importlib` overhead on every call |
| No top-level `import nat_nexus` | Prevents `ImportError` at framework import time |

For **provider-level** modules (LLM chat model classes), create a second bridge
module that additionally checks for an active scope stack. See
[§3 Transparent Fallback](#3-transparent-fallback-to-default-behavior) below.

---

## 3. Transparent Fallback to Default Behavior

The integration must have two code paths — one with Nexus, one without — and the
framework user should never notice the difference (except for the middleware
features Nexus adds).

### Provider-level availability check

For LLM providers, Nexus should only activate when the caller has explicitly
initialized a scope stack. This prevents unexpected behavior when `nat_nexus` is
installed but unused:

```python
# <framework>/chat_models/_nat_nexus.py

try:
    import nat_nexus
    from nat_nexus import LLMRequest
    _HAS_NAT_NEXUS = True
except ImportError:
    _HAS_NAT_NEXUS = False


def available() -> bool:
    """Return True when nat_nexus is importable *and* a scope stack is active."""
    if not _HAS_NAT_NEXUS:
        return False
    try:
        return nat_nexus.scope_stack_active()
    except Exception:
        return False
```

### Branching pattern

At every integration point, use a simple if/else branch. The `else` branch must
be the **original, unmodified code**:

```python
# In a tool execution method:
if (nnex := get_nat_nexus()) is not None:
    # Nexus-wrapped path
    codec = nnex.typed.BestEffortAnyCodec()
    response = await nnex.typed.tool_execute(
        self.name, tool_input, _func, codec, codec,
    )
else:
    # Original behavior — completely unchanged
    response = await _func(tool_input)
```

### Error silencing

Wrap **every** Nexus call in try/except at the callback and scope-management
layers. Use `DEBUG`-level logging only — no warnings, no user-visible messages:

```python
try:
    handle = scope.push(name, nnex.ScopeType.Agent, handle=parent)
    self._scope_handles[run_id] = handle
except Exception:
    _logger.debug("Nexus: scope push failed", exc_info=True)
```

The tool and LLM execution wrappers (`typed.tool_execute`, `typed.llm_execute`)
already handle errors internally, so you do not need extra try/except around those.

---

## 4. Wrapping LLM Calls

LLM calls are wrapped at the **provider level** (the class that actually makes HTTP
requests), not at an abstract base class level. This gives access to the raw
request payload and SDK response.

### Non-streaming calls

Use `nat_nexus.typed.llm_execute` with `JsonPassthrough` (since provider code
already converts SDK responses to dicts via `model_dump()` or equivalent):

```python
from <framework>.chat_models import _nat_nexus

# Inside the _generate or _call method:
if _nat_nexus.available():
    request = _nat_nexus.make_request(payload, extra_headers)

    async def _call(req):
        # Use req.content (dict) and req.headers (dict) to make the real call
        raw = await self._async_client.chat.completions.create(**req.content)
        return raw.model_dump()

    resp_dict = _nat_nexus.run_sync(
        _nat_nexus.llm_execute(self.model_name, request, _call)
    )
    return self._process_response(resp_dict)

# ... original code path below ...
```

### Streaming calls

Use `nat_nexus.typed.llm_stream_execute` with a collector/finalizer pattern:

```python
if _nat_nexus.available():
    request = _nat_nexus.make_request(payload)
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
    stream = await _nat_nexus.llm_stream_execute(
        self.model_name, request, _call, _collector, _finalizer
    )
    async for chunk_dict in stream:
        yield self._convert_chunk(chunk_dict)
    return

# ... original streaming code path below ...
```

### The bridge module pattern

Each provider package should have its own `_nat_nexus.py` bridge that wraps the
typed API with the correct codec. This keeps the main chat model code clean:

```python
# <framework>/chat_models/_nat_nexus.py

async def llm_execute(model_name, request, func):
    codec = nat_nexus.typed.JsonPassthrough()
    return await nat_nexus.typed.llm_execute(
        model_name, request, func, codec, model_name=model_name,
    )

async def llm_stream_execute(model_name, request, func, collector, finalizer):
    codec = nat_nexus.typed.JsonPassthrough()
    return await nat_nexus.typed.llm_stream_execute(
        model_name, request, func, collector, finalizer,
        codec, codec, model_name=model_name,
    )

def make_request(payload, extra_headers=None):
    return nat_nexus.LLMRequest(extra_headers or {}, payload)

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
from <framework>.utils._nat_nexus import get_nat_nexus

# Inside the tool's async invoke/execute method:
async def _func(_args):
    """Wraps the user's tool function."""
    return await self._arun(*tool_args, **tool_kwargs)

if (nnex := get_nat_nexus()) is not None:
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

When wrapping a synchronous tool call, you need to bridge into Nexus's async
pipeline. This requires handling the case where an event loop may already be
running (e.g., Jupyter, nested async frameworks):

```python
def _func(_args):
    return context.run(self._run, *tool_args, **tool_kwargs)

if (nnex := get_nat_nexus()) is not None:
    import asyncio
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    codec = nnex.typed.BestEffortAnyCodec()
    ctx = contextvars.copy_context()
    scope_stack = nnex.get_scope_stack()

    async def _nat_nexus_run():
        return await nnex.typed.tool_execute(
            self.name, tool_input, _func, codec, codec,
        )

    def _run_with_scope_stack():
        # Propagate scope stack to the worker thread's Rust thread-local
        nnex.set_thread_scope_stack(scope_stack)
        return asyncio.run(_nat_nexus_run())

    try:
        asyncio.get_running_loop()
        # Already in an async context — offload to a thread
        with ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(ctx.run, _run_with_scope_stack).result()
    except RuntimeError:
        # No event loop — safe to use asyncio.run directly
        response = asyncio.run(_nat_nexus_run())
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
framework's own hierarchy concept to Nexus scopes.

### Scope types

```python
import nat_nexus

nat_nexus.ScopeType.Function  # Individual function/step
nat_nexus.ScopeType.Agent     # An agent or chain invocation
```

### Agent scopes

Create an `Agent` scope when a chain, agent, or high-level runnable starts
execution. Pop it when execution ends (including on error):

```python
# LangChain maps this via a callback handler:
class NatNexusCallbackHandler(BaseCallbackHandler):
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
            _logger.debug("Nexus: scope push failed", exc_info=True)

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
            _logger.debug("Nexus: scope.pop failed", exc_info=True)
```

If the framework doesn't have a callback/hook system, you can use the context
manager form instead:

```python
with nat_nexus.scope.scope("my-agent", nat_nexus.ScopeType.Agent) as handle:
    result = await agent.run(task)
```

### Function scopes

Create `Function` scopes for discrete steps within an agent (e.g., a planner step,
a retrieval step). These are typically shorter-lived than agent scopes:

```python
with nat_nexus.scope.scope("plan", nat_nexus.ScopeType.Function) as handle:
    plan = await planner.generate(task)
```

### Parallel batch calls

When the framework runs multiple operations concurrently (e.g., parallel tool
calls, fan-out steps), each parallel branch should get its own scope. All branches
share the same parent:

```python
import asyncio
import nat_nexus

parent_handle = nat_nexus.scope.get_handle()

async def run_branch(name, task):
    # Each branch gets its own child scope
    with nat_nexus.scope.scope(
        name, nat_nexus.ScopeType.Function, handle=parent_handle
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

Nexus uses both Python `contextvars` (for async task isolation) and Rust
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

Nexus's execute pipeline is async. When integrating into a sync code path:

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
    0001-add-nat-nexus-integration.patch
```

Bootstrap the local upstream checkouts before applying or refreshing patches:

```bash
./scripts/bootstrap-third-party.sh
```

### Files to add/modify

A typical integration touches these files:

| File | Purpose |
|---|---|
| `<framework>/utils/_nat_nexus.py` | Lazy import helper (new file) |
| `<framework>/chat_models/_nat_nexus.py` | Provider bridge with `available()`, `llm_execute`, `run_sync` (new file) |
| `<framework>/tools/base.py` | Tool execution wrapping (modify existing) |
| `<framework>/callbacks/nat_nexus_handler.py` | Scope lifecycle callback handler (new file, if the framework supports callbacks) |
| `tests/.../test_nat_nexus_handler.py` | Tests for scope lifecycle (new file) |

### Generating the patch

```bash
cd third_party/<framework>
# Make your changes...
git diff HEAD -- . > ../../patches/<framework>/0001-add-nat-nexus-integration.patch
```

### Applying the patch

```bash
./scripts/apply-patches.sh
```

---

## 9. Testing

### Unit tests (within the framework's test suite)

Test the callback handler / scope management with mocked Nexus:

```python
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

def _make_mock_nnex():
    nnex = ModuleType("nat_nexus")
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
        handler = NatNexusCallbackHandler()
        handler._nnex = _make_mock_nnex()
        run_id = uuid4()
        handler.on_chain_start({"name": "MyAgent"}, {}, run_id=run_id)
        handler._nnex.scope.push.assert_called_once()

    def test_chain_end_pops_scope(self):
        handler = NatNexusCallbackHandler()
        handler._nnex = _make_mock_nnex()
        run_id = uuid4()
        handler.on_chain_start({"name": "MyAgent"}, {}, run_id=run_id)
        handler.on_chain_end({}, run_id=run_id)
        handler._nnex.scope.pop.assert_called_once()

    def test_no_nexus_is_silent_noop(self):
        handler = NatNexusCallbackHandler()
        handler._nnex = None
        handler.on_chain_start({"name": "x"}, {}, run_id=uuid4())  # no error

    def test_nexus_error_is_swallowed(self):
        mock = _make_mock_nnex()
        mock.scope.push.side_effect = RuntimeError("boom")
        handler = NatNexusCallbackHandler()
        handler._nnex = mock
        handler.on_chain_start({"name": "x"}, {}, run_id=uuid4())  # no error
```

### What to verify

- Framework's existing test suite passes **without** `nat_nexus` installed.
- Scope push/pop called correctly for agent/chain lifecycle events.
- Parent scope handles are propagated for nested invocations.
- All Nexus errors are swallowed — never surface to the framework.
- End-without-start is a no-op (no crash).
- Tool and LLM execute calls go through Nexus when available, fall back when not.

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
- [ ] **Error silencing** — all Nexus errors caught and logged at `DEBUG`
- [ ] **No hard dependency** — framework works identically without `nat_nexus`
- [ ] **Tests** — scope lifecycle, graceful no-op, error swallowing
- [ ] **Patch generated** — `patches/<framework>/0001-add-nat-nexus-integration.patch`
- [ ] **SPDX headers** — Apache-2.0 on all new files

---

## Other Languages

This guide focuses on Python integrations. For Go, Node.js, and WASM
integration patterns, see [Language Bindings](language-bindings.md) which
covers setup, usage examples, and language-specific considerations for each
binding layer. The same architectural principles apply: lazy import,
transparent fallback, scope lifecycle management, and thread/async safety.
