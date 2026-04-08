<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# go/nat_nexus

Go CGo bindings for the Nexus core runtime.

## Overview

This package wraps the Nexus C FFI layer via CGo, providing a Go-idiomatic API with functional options patterns, native error types, and subpackages organized by domain. It requires the FFI library to be built before use.

## What It Provides

- **Go-idiomatic API** -- PascalCase naming, `error` returns, functional option patterns for configuration.
- **Subpackage organization** -- Separate packages for `scope`, `tools`, `llm`, `guardrails`, `intercepts`, and `subscribers`.
- **Callback bridging** -- Go functions are bridged to C function pointers for use as tool handlers, guardrails, intercepts, and event subscribers.
- **Stream support** -- Go-native stream handling for LLM streaming responses.
- **Scope-local middleware** -- `ScopeRegister*` functions for registering middleware scoped to a specific execution scope.
- **Context isolation** -- `CreateScopeStack`/`CurrentScopeStack`/`SetThreadScopeStack` for per-request isolation in concurrent servers.

## Key Files

| File | Purpose |
|------|---------|
| `nat_nexus.go` | Core CGo bindings and top-level API |
| `types.go` | Go type definitions mirroring core types |
| `callbacks.go` | Go-to-C callback bridging |
| `stream.go` | LLM stream wrapper for Go |
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

## Documentation

See [docs/language-bindings.md](../../docs/language-bindings.md) for Go binding details.
