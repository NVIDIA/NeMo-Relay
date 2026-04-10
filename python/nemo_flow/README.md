<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo_flow

Python wrapper package for the NeMo Flow core runtime.

## Overview

This package provides a thin Python layer over the native Rust extension (`_native`), offering a Pythonic API with module-level organization, type stubs, and convenience wrappers. The native extension is compiled from `crates/python/` via Maturin and exposed here as `nemo_flow._native`.

## What It Provides

- **Module-level organization** -- Each domain area has its own module for clean imports (e.g., `from nemo_flow.tools import register_tool`).
- **Type stubs** -- `__init__.pyi` provides full type annotations for IDE autocompletion and static analysis.
- **Scope context managers** -- Python context managers for automatic scope push/pop with cleanup.
- **Async support** -- All async operations are exposed as native Python coroutines compatible with `asyncio`.
- **Scope-local middleware** -- `scope_local.py` exposes scope-scoped guardrail, intercept, and subscriber registration.
- **Typed and adaptive helpers** -- `typed.py` and `adaptive.py` expose higher-level typed wrappers plus flat adaptive config helpers for the core plugin host.
- **Generic plugin host** -- `plugin.py` exposes global plugin registration, configuration, and reporting helpers.

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
| `adaptive.py` | Adaptive config helpers for the top-level `adaptive` plugin component |
| `plugin.py` | Generic plugin-host registration and configuration helpers |

## Install

```bash
uv sync
```

This builds the native Rust extension and installs the package in a virtual environment. Tests can be run with:

```bash
uv run pytest
```

If the native extension changed, rebuild it before rerunning adaptive tests:

```bash
uv run maturin develop
uv run pytest python/tests/test_adaptive.py python/tests/test_adaptive_config.py
```

## Adaptive Config

Python exposes typed adaptive helpers through `nemo_flow.adaptive`, then
activates them through `nemo_flow.plugin`.

```python
from nemo_flow.adaptive import (
    AdaptiveHintsConfig,
    BackendSpec,
    AdaptiveConfig,
    ComponentSpec,
    StateConfig,
    TelemetryConfig,
)
from nemo_flow import plugin

report = await plugin.initialize(
    plugin.PluginConfig(
        components=[
            ComponentSpec(
                AdaptiveConfig(
                    state=StateConfig(backend=BackendSpec.in_memory()),
                    telemetry=TelemetryConfig(learners=["latency_sensitivity"]),
                    adaptive_hints=AdaptiveHintsConfig(),
                )
            )
        ]
    )
)
assert report["diagnostics"] == []
```

## Hosted Plugins

Python hosted plugins register a handler object first, then enable themselves
as top-level plugin components in `nemo_flow.plugin.initialize(...)`.

```python
from nemo_flow import LLMRequest
from nemo_flow import plugin

class HeaderPlugin:
    def validate(self, plugin_config):
        return []

    def register(self, plugin_config, context):
        def intercept(_name, request, annotated):
            headers = dict(request.headers)
            headers["x-plugin"] = "enabled"
            return LLMRequest(headers, request.content), annotated

        context.register_llm_request_intercept(
            "header",
            25,
            False,
            intercept,
        )

plugin.register("example.header_plugin", HeaderPlugin())

await plugin.initialize(
    plugin.PluginConfig(
        components=[
            plugin.ComponentSpec(kind="example.header_plugin"),
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
helpers, and [docs/adaptive-api-reference.md](../../docs/adaptive-api-reference.md)
for adaptive APIs.
