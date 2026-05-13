<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust Quick Start

This quick start shows the smallest Rust workflow that emits scope and mark events.

## Choose an Install Path

Pick the installation path that matches whether you are using a local checkout or a
published package.

### Install from the Repository

Use a path dependency when your application is consuming a local checkout:

```toml
[dependencies]
nemo-flow = { path = "../NeMo-Flow/crates/core" }
nemo-flow-adaptive = { path = "../NeMo-Flow/crates/adaptive" }
serde_json = "1"
```

- `nemo-flow` is the core Rust runtime surface.
- `nemo-flow-adaptive` is the companion crate for adaptive runtime primitives and Redis-backed learning components.
- `nemo-flow-cli` is a binary crate. Use `cargo install nemo-flow-cli` when
  you need the coding-agent gateway.

### Install from a Package Manager

Use the published crate when you are consuming a release:

```toml
[dependencies]
nemo-flow = "0.1.*"
nemo-flow-adaptive = "0.1.*"
serde_json = "1"
```

Install the published gateway binary separately when you need coding-agent hook
and LLM gateway observability:

```bash
cargo install nemo-flow-cli
```

## Push a Scope and Emit a Mark

The example below creates a scope and records a mark event from Rust.

```rust
use nemo_flow::api::scope::{
    self, EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeType,
};
use nemo_flow::api::subscriber::{deregister_subscriber, register_subscriber};
use serde_json::json;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    register_subscriber(
        "quickstart-printer",
        Arc::new(|event| {
            println!("{} {}", event.kind(), event.name());
        }),
    )?;

    let handle = scope::push_scope(
        PushScopeParams::builder()
            .name("demo-agent")
            .scope_type(ScopeType::Agent)
            .attributes(ScopeAttributes::empty())
            .data(json!({"binding": "rust"}))
            .build(),
    )?;

    scope::event(
        EmitMarkEventParams::builder()
            .name("initialized")
            .parent(&handle)
            .data(json!({"ok": true}))
            .build(),
    )?;
    scope::pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build())?;
    let _ = deregister_subscriber("quickstart-printer")?;
    Ok(())
}
```

## What Success Looks Like

The script should exit cleanly and print lifecycle lines from the subscriber.
You should see one line for the scope start event, one for the `initialized`
mark, and one for the scope end event.

That tells you two things:

- The scope API ran successfully.
- Emitted events were observable through the subscriber system.

## What to Learn Next

Use these links to continue from the quick start into the core runtime concepts.

- Use [Scopes](../about/concepts/scopes.md), [Middleware](../about/concepts/middleware.md), [Subscribers](../about/concepts/subscribers.md), and [Plugins](../about/concepts/plugins.md) for the runtime model.
