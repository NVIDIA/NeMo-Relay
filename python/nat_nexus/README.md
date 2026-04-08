<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nat_nexus

Python wrapper package for the Nexus core runtime.

## Overview

This package provides a thin Python layer over the native Rust extension (`_native`), offering a Pythonic API with module-level organization, type stubs, and convenience wrappers. The native extension is compiled from `crates/python/` via Maturin and exposed here as `nat_nexus._native`.

## What It Provides

- **Module-level organization** -- Each domain area has its own module for clean imports (e.g., `from nat_nexus.tools import register_tool`).
- **Type stubs** -- `__init__.pyi` provides full type annotations for IDE autocompletion and static analysis.
- **Scope context managers** -- Python context managers for automatic scope push/pop with cleanup.
- **Async support** -- All async operations are exposed as native Python coroutines compatible with `asyncio`.
- **Scope-local middleware** -- `scope_local.py` exposes scope-scoped guardrail, intercept, and subscriber registration.
- **Typed and optimizer helpers** -- `typed.py` and `optimizer.py` expose higher-level typed wrappers plus config-driven optimizer runtime helpers.
- **Hosted optimizer plugins** -- `register_optimizer_plugin(...)`, `deregister_optimizer_plugin(...)`, and `ExternalComponent` support callback-backed optimizer extensions invoked from Rust.

## Key Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package root, re-exports from `_native` and submodules |
| `__init__.pyi` | Type stubs for the full public API |
| `scope.py` | Scope management (push, pop, context managers) |
| `tools.py` | Tool registration and invocation |
| `llm.py` | LLM call registration and invocation |
| `guardrails.py` | Guardrail registration (input/output) |
| `intercepts.py` | Request and execution intercept registration |
| `subscribers.py` | Event subscriber registration |
| `scope_local.py` | Scope-local middleware registration |
| `typed.py` | Typed helper utilities |
| `optimizer.py` | Optimizer config helpers, diagnostics, runtime lifecycle wrapper, and hosted plugin helpers |

## Install

```bash
uv sync
```

This builds the native Rust extension and installs the package in a virtual environment. Tests can be run with:

```bash
uv run pytest
```

If the native extension changed, rebuild it before rerunning optimizer tests:

```bash
uv run maturin develop
uv run pytest python/tests/test_optimizer.py python/tests/test_optimizer_config.py
```

## Optimizer Runtime

Python exposes typed optimizer helpers through `nat_nexus.optimizer`.

```python
from nat_nexus.optimizer import (
    BackendSpec,
    OptimizerConfig,
    OptimizerRuntime,
    StateConfig,
    TelemetryComponent,
)

runtime = OptimizerRuntime(
    OptimizerConfig(
        state=StateConfig(backend=BackendSpec.in_memory()),
        components=[TelemetryComponent(learners=["latency_sensitivity"])],
    )
)
```

## Hosted Optimizer Plugins

Python hosted optimizer plugins register a handler object first, then enable
the plugin through `ExternalComponent(...)` in `OptimizerConfig`.

```python
from nat_nexus import LLMRequest
from nat_nexus.optimizer import (
    ExternalComponent,
    OptimizerConfig,
    OptimizerRuntime,
    register_optimizer_plugin,
)

class HeaderPlugin:
    def validate(self, instance_id, plugin_config):
        return []

    def register(self, instance_id, plugin_config, context):
        def intercept(_name, request, annotated):
            headers = dict(request.headers)
            headers["x-plugin"] = instance_id
            return LLMRequest(headers, request.content), annotated

        context.register_llm_request_intercept(
            f"{instance_id}.header",
            25,
            False,
            intercept,
        )

register_optimizer_plugin("example.header_plugin", HeaderPlugin())

runtime = OptimizerRuntime(
    OptimizerConfig(
        components=[
            ExternalComponent(
                plugin_kind="example.header_plugin",
                instance_id="plugin-1",
            )
        ]
    )
)
```

`context` exposes:

- `register_subscriber(...)`
- `register_llm_request_intercept(...)`
- `register_llm_execution_intercept(...)`
- `register_llm_stream_execution_intercept(...)`
- `register_tool_request_intercept(...)`
- `register_tool_execution_intercept(...)`

## Documentation

See [docs/api-reference.md](../../docs/api-reference.md) for the core runtime
API, [docs/typed-api-reference.md](../../docs/typed-api-reference.md) for typed
helpers, and [docs/optimizer-api-reference.md](../../docs/optimizer-api-reference.md)
for optimizer APIs.
