<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# go/nemo_flow

Go CGo bindings for the NeMo Flow runtime, including the adaptive surface.

## Overview

This package wraps the NeMo Flow C FFI layer via CGo, providing a Go-idiomatic API with functional options patterns, native error types, generic plugin-host helpers, adaptive config helpers, and subpackages organized by domain. It requires the FFI library to be built before use.

## What It Provides

- **Go-idiomatic API** -- PascalCase naming, `error` returns, functional option patterns for configuration.
- **Subpackage organization** -- Separate packages for `scope`, `tools`, `llm`, `guardrails`, `intercepts`, and `subscribers`.
- **Callback bridging** -- Go functions are bridged to C function pointers for use as tool handlers, guardrails, intercepts, and event subscribers.
- **Stream support** -- Go-native stream handling for LLM streaming responses.
- **Scope-local middleware** -- `ScopeRegister*` functions for registering middleware scoped to a specific execution scope.
- **Context isolation** -- `CreateScopeStack`/`CurrentScopeStack`/`SetThreadScopeStack` for per-request isolation in concurrent servers.
- **Generic plugin host** -- `PluginConfig`, `RegisterPlugin`, `InitializePlugins`, and `PluginContext` expose the core plugin system directly.
- **Adaptive config helpers** -- the `adaptive` subpackage exposes typed config builders plus adaptive-owned component specs for the top-level adaptive plugin component.

## Key Files

| File | Purpose |
|------|---------|
| `nemo_flow.go` | Core CGo bindings and top-level API |
| `types.go` | Go type definitions mirroring core types |
| `callbacks.go` | Go-to-C callback bridging |
| `stream.go` | LLM stream wrapper for Go |
| `plugin.go` | Generic plugin-host config, registration, and callback APIs |
| `adaptive.go` | Adaptive config helpers and top-level adaptive plugin wrapper |
| `scope/scope.go` | Scope management subpackage |
| `tools/tools.go` | Tool registration and invocation |
| `llm/llm.go` | LLM call registration and invocation |
| `guardrails/guardrails.go` | Guardrail registration |
| `intercepts/intercepts.go` | Intercept registration |
| `subscribers/subscribers.go` | Event subscriber registration |

## Prerequisites

The FFI library must be built before running or testing the Go bindings:

```bash
cargo build --release -p nemo-flow-ffi
```

The package searches `target/release` and `target/debug` automatically, so no
extra `CGO_LDFLAGS` setup is needed when running from this repo.

## Build and Test

```bash
cd go/nemo_flow
go test -race -v ./...
```

## Adaptive Config

Go exposes typed adaptive config builders through the `adaptive` subpackage,
then activates them through the generic plugin host.

```go
import adaptive "github.com/NVIDIA/NeMo-Flow/go/nemo_flow/adaptive"

config := adaptive.NewConfig()
config.State = &adaptive.StateConfig{
    Backend: adaptive.NewInMemoryBackend(),
}
telemetry := adaptive.NewTelemetryConfig()
telemetry.Learners = []string{"latency_sensitivity"}
config.Telemetry = &telemetry

report, err := nemo_flow.InitializePlugins(nemo_flow.PluginConfig{
    Version: 1,
    Components: []nemo_flow.PluginComponentSpec{
        adaptive.NewComponentSpec(config).PluginComponent(),
    },
})
if err != nil {
    panic(err)
}
_ = report
defer func() { _ = nemo_flow.ClearPluginConfiguration() }()
```

## Hosted Plugins

Go plugins register callback handlers up front, then enable themselves as
top-level plugin components in the core plugin config.

```go
import (
    "encoding/json"

    adaptive "github.com/NVIDIA/NeMo-Flow/go/nemo_flow/adaptive"
    nemo_flow "github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
)

pluginKind := "example.header_plugin"
err := nemo_flow.RegisterPlugin(pluginKind, nemo_flow.PluginFuncs{
    ValidateFunc: func(pluginConfig map[string]any) ([]nemo_flow.ConfigDiagnostic, error) {
        return nil, nil
    },
    RegisterFunc: func(pluginConfig map[string]any, ctx *nemo_flow.PluginContext) error {
        return ctx.RegisterToolRequestIntercept(
            "tool",
            25,
            false,
            func(name string, args json.RawMessage) json.RawMessage {
                var payload map[string]any
                _ = json.Unmarshal(args, &payload)
                payload["goPlugin"] = "enabled"
                payload["tool"] = name
                out, _ := json.Marshal(payload)
                return out
            },
        )
    },
})
if err != nil {
    panic(err)
}
defer func() { _ = nemo_flow.DeregisterPlugin(pluginKind) }()

_, err = nemo_flow.InitializePlugins(nemo_flow.PluginConfig{
    Version: 1,
    Components: []nemo_flow.PluginComponentSpec{
        adaptive.NewComponentSpec(adaptive.NewConfig()).PluginComponent(),
        {
            Kind:    pluginKind,
            Enabled: true,
        },
    },
})
```

`PluginContext` exposes:

- `RegisterSubscriber(...)`
- `RegisterLlmRequestIntercept(...)`
- `RegisterLlmExecutionIntercept(...)`
- `RegisterLlmStreamExecutionIntercept(...)`
- `RegisterToolRequestIntercept(...)`
- `RegisterToolExecutionIntercept(...)`

## Documentation

See [docs/language-bindings.md](../../docs/language-bindings.md) for Go binding details.
