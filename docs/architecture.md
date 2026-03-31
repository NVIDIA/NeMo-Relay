<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Architecture Overview

## System Design

Nexus is a Rust-core runtime with bindings for Python, Node.js, Go, and WebAssembly. The core handles all middleware logic, event emission, and scope management. Each binding is a thin translation layer that marshals types between the host language and Rust.

```
┌────────────────────────────────────────────────────────────┐
│                     Application Code                       │
│            (Python / Node.js / Go / WASM)                  │
└──────┬──────────┬──────────┬─────────┬─────────────────────┘
       │          │          │         │
┌──────▼───┐  ┌───▼────┐ ┌───▼───┐ ┌───▼──────┐
│  PyO3    │  │  NAPI  │ │  FFI  │ │  wasm-   │
│  Python  │  │ Node.js│ │  C/Go │ │  bindgen │
└──────┬───┘  └───┬────┘ └───┬───┘ └───┬──────┘
       │          │          │         │
┌──────▼──────────▼──────────▼─────────▼──────┐
│              nvidia-nat-nexus-core (Rust)            │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Scopes  │  │  Tools   │  │    LLM    │  │
│  └──────────┘  └──────────┘  └───────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │Guardrails│  │Intercepts│  │Subscribers│  │
│  └──────────┘  └──────────┘  └───────────┘  │
│  ┌──────────────────────────────────────┐   │
│  │         Global Context State         │   │
│  │  (registries, scope stacks, events)  │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Binding Layer Architecture

```mermaid
graph TD
    subgraph "Application"
        PY[Python App]
        JS[Node.js App]
        GO[Go App]
        WA[WASM App]
    end

    subgraph "Bindings"
        PYO3[crates/python<br/>PyO3 + abi3]
        NAPI[crates/node<br/>NAPI-RS]
        FFI[crates/ffi<br/>cbindgen → C header]
        WASM[crates/wasm<br/>wasm-bindgen]
    end

    subgraph "Core"
        CORE[crates/core<br/>nvidia-nat-nexus-core]
    end

    PY --> PYO3
    JS --> NAPI
    GO --> FFI
    WA --> WASM

    PYO3 --> CORE
    NAPI --> CORE
    FFI --> CORE
    WASM --> CORE
```

## Repository Structure

```
crates/
  core/           # Core runtime (Rust)
    src/
      lib.rs          # Public re-exports
      api.rs          # All API functions
      types.rs        # LLMRequest, handles, attributes, events
      context.rs      # Global state, scope stacks, callable type aliases
      registry.rs     # SortedRegistry<T> for priority-ordered middleware
      stream.rs       # LlmStreamWrapper for streaming LLM responses
      error.rs        # NexusError enum
      json.rs         # Json type alias, merge_json helper
      atif.rs         # ATIF trajectory exporter
    tests/
      context_isolation_tests.rs
      stream_tests.rs

  python/          # PyO3 bindings
  ffi/             # C FFI (used by Go via CGo)
  node/            # NAPI Node.js bindings
  wasm/            # wasm-bindgen WebAssembly bindings

python/            # Python wrapper package (nat_nexus/)
  nat_nexus/
    __init__.py        # Re-exports all submodules and types
    scope.py           # Scope operations
    tools.py           # Tool lifecycle
    llm.py             # LLM lifecycle
    guardrails.py      # Guardrail registration
    intercepts.py      # Intercept registration
    subscribers.py     # Event subscriber registration
    scope_local.py     # Scope-local middleware registration
    typed.py           # Codec-based typed wrappers

go/nat_nexus/        # Go CGo bindings
```

## Global Context

Middleware registrations exist at two levels: **global** (shared by all scope stacks) and **scope-local** (bound to a specific scope within a stack).

### Global Registries

```mermaid
graph LR
    subgraph "NatNexusContextState (Global)"
        TSR[Tool Sanitize<br/>Request Guards]
        TSP[Tool Sanitize<br/>Response Guards]
        TCE[Tool Conditional<br/>Execution Guards]
        TRI[Tool Request<br/>Intercepts]
        TRSI[Tool Response<br/>Intercepts]
        TEI[Tool Execution<br/>Intercepts]

        LSR[LLM Sanitize<br/>Request Guards]
        LSP[LLM Sanitize<br/>Response Guards]
        LCE[LLM Conditional<br/>Execution Guards]
        LRI[LLM Request<br/>Intercepts]
        LEI[LLM Execution<br/>Intercepts]
        LSEI[LLM Stream<br/>Execution Intercepts]

        SUBS[Event<br/>Subscribers]
    end
```

Each registry is a `SortedRegistry<T>` that maintains entries by name with lazy priority-based sorting.

### Scope-Local Registries

Each `ScopeStack` can also hold per-scope middleware, stored in a `HashMap<Uuid, ScopeLocalRegistries>`:

```mermaid
graph TD
    subgraph "ScopeStack"
        ROOT["Root Scope (uuid-A)"]
        AGENT["Agent Scope (uuid-B)"]
        ROOT --> AGENT

        subgraph "scope_registries"
            SL_B["uuid-B → ScopeLocalRegistries<br/>(same 13 registry types as global)"]
        end
    end
```

- Lazily created on first `scope_register_*` call for a given scope
- Automatically removed when the scope is popped
- During pipeline execution, entries from global + all ancestor scope-local registries are merged by priority

See [Core Concepts: Scope-Local Middleware](concepts.md#scope-local-middleware) for usage details.

## Data Flow: LLM Execute

```mermaid
sequenceDiagram
    participant App
    participant Core as nvidia-nat-nexus-core
    participant Guards as Guardrails
    participant Ints as Intercepts
    participant Func as User Function
    participant Subs as Subscribers

    App->>Core: llm.execute(name, request, func)
    Core->>Guards: conditional_execution(request)
    alt Rejected
        Guards-->>Core: Some("reason")
        Core->>Subs: emit Mark event
        Core-->>App: GuardrailRejected
    end
    Guards-->>Core: None (allowed)
    Core->>Ints: request_intercepts(request)
    Ints-->>Core: transformed request
    Core->>Guards: sanitize_request(request)
    Guards-->>Core: sanitized request
    Core->>Subs: emit Start event (input=sanitized)
    Core->>Ints: execution_intercept_chain(request)
    Ints->>Func: func(request)
    Func-->>Ints: response
    Ints-->>Core: response
    Core->>Guards: sanitize_response(response)
    Guards-->>Core: sanitized response
    Core->>Subs: emit End event (output=sanitized)
    Core-->>App: response
```
