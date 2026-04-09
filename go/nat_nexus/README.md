<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# go/nat_nexus

Go CGo bindings for the Nexus runtime, including the optimizer surface.

## Overview

This package wraps the Nexus C FFI layer via CGo, providing a Go-idiomatic API with functional options patterns, native error types, optimizer config/runtime helpers, and subpackages organized by domain. It requires the FFI library to be built before use.

## What It Provides

- **Go-idiomatic API** -- PascalCase naming, `error` returns, functional option patterns for configuration.
- **Subpackage organization** -- Separate packages for `scope`, `tools`, `llm`, `guardrails`, `intercepts`, and `subscribers`.
- **Callback bridging** -- Go functions are bridged to C function pointers for use as tool handlers, guardrails, intercepts, and event subscribers.
- **Stream support** -- Go-native stream handling for LLM streaming responses.
- **Scope-local middleware** -- `ScopeRegister*` functions for registering middleware scoped to a specific execution scope.
- **Context isolation** -- `CreateScopeStack`/`CurrentScopeStack`/`SetThreadScopeStack` for per-request isolation in concurrent servers.
- **Optimizer runtime** -- the `optimizer` subpackage exposes `Config`, `Runtime`, validation helpers, and typed component builders.
- **Hosted optimizer plugins** -- the `optimizer` subpackage exposes `RegisterPlugin`, `DeregisterPlugin`, `ExternalComponent`, and `PluginContext` for callback-backed optimizer extensions.

## Key Files

| File | Purpose |
|------|---------|
| `nat_nexus.go` | Core CGo bindings and top-level API |
| `types.go` | Go type definitions mirroring core types |
| `callbacks.go` | Go-to-C callback bridging |
| `stream.go` | LLM stream wrapper for Go |
| `optimizer.go` | Optimizer config, diagnostics, runtime wrapper, and hosted plugin APIs |
| `scope/scope.go` | Scope management subpackage |
| `tools/tools.go` | Tool registration and invocation |
| `llm/llm.go` | LLM call registration and invocation |
| `guardrails/guardrails.go` | Guardrail registration |
| `intercepts/intercepts.go` | Intercept registration |
| `subscribers/subscribers.go` | Event subscriber registration |

## Prerequisites

The FFI library must be built before running or testing the Go bindings:

```bash
cargo build --release -p nvidia-nat-nexus-ffi
```

The package searches `target/release` and `target/debug` automatically, so no
extra `CGO_LDFLAGS` setup is needed when running from this repo.

## Build and Test

```bash
cd go/nat_nexus
go test -race -v ./...
```

## Optimizer Runtime

Go exposes typed optimizer config builders and a synchronous runtime wrapper
through the `optimizer` subpackage.

```go
import optimizer "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/optimizer"

config := optimizer.NewConfig()
config.State = &optimizer.StateConfig{
    Backend: optimizer.NewInMemoryBackend(),
}
config.Components = []optimizer.ComponentSpec{
    optimizer.TelemetryComponent(optimizer.TelemetryComponentConfig{
        Learners: []string{"latency_sensitivity"},
    }),
}

runtime, err := optimizer.NewRuntime(config)
if err != nil {
    panic(err)
}
defer runtime.Close()
```

## Hosted Optimizer Plugins

Go plugins register callback handlers up front, then enable themselves through
the optimizer config via `ExternalComponent(...)`.

```go
import (
    "encoding/json"

    optimizer "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/optimizer"
)

pluginKind := "example.header_plugin"
err := optimizer.RegisterPlugin(pluginKind, optimizer.PluginHandlerFuncs{
    ValidateFunc: func(instanceID string, pluginConfig map[string]any) ([]optimizer.ConfigDiagnostic, error) {
        return nil, nil
    },
    RegisterFunc: func(instanceID string, pluginConfig map[string]any, ctx *optimizer.PluginContext) error {
        return ctx.RegisterToolRequestIntercept(
            instanceID+".tool",
            25,
            false,
            func(name string, args json.RawMessage) json.RawMessage {
                var payload map[string]any
                _ = json.Unmarshal(args, &payload)
                payload["goPlugin"] = instanceID
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
defer func() { _ = optimizer.DeregisterPlugin(pluginKind) }()
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
