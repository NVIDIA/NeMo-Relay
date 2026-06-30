---
name: test-rust-core
description: Build and test NeMo Relay Rust core, adaptive, and dynamic plugin crates; use for crates/core, crates/adaptive, crates/plugin, crates/worker, crates/worker-proto, crates/types, or shared runtime changes
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Build And Test Rust Core

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when a change is primarily in `crates/core`, `crates/adaptive`,
dynamic plugin Rust crates, or shared Rust runtime semantics.

## Default Path

1. Run `cargo fmt --all`.
2. Run `just test-rust`.
3. Run `cargo clippy --workspace --all-targets -- -D warnings`.
4. Because this skill covers `crates/core`, `crates/adaptive`, or shared runtime
   semantics, expand to the full binding matrix with `validate-change`.

Use narrower crate tests as a local debug loop, not as the final validation
story for a Rust change.

## Common Commands

```bash
# Shared-runtime build/test wrapper
just test-rust

# Required Rust format pass
cargo fmt --all

# Required Rust lint pass
cargo clippy --workspace --all-targets -- -D warnings

# Core runtime only
cargo test -p nemo-relay

# Adaptive crate when touched
cargo test -p nemo-relay-adaptive

# Dynamic plugin crates when touched
cargo test -p nemo-relay-types
cargo test -p nemo-relay-plugin
cargo test -p nemo-relay-worker-proto
cargo test -p nemo-relay-worker
cargo test -p nemo-relay --features worker-grpc --test native_plugin_integration --test worker_plugin_integration

# Compile sweep
just build-rust

# Shared-semantics or broad runtime changes
just ci=true test-rust
```

## When To Escalate

- If a public API, event shape, middleware behavior, plugin semantics, or any
  `crates/core`/`crates/adaptive` behavior changed, also use `validate-change`.
- If native dynamic plugins, gRPC workers, `nemo-relay-plugin`,
  `nemo-relay-worker`, `nemo-relay-worker-proto`, or `nemo-relay-types` changed,
  also use `maintain-dynamic-plugins`.
- If the change is isolated to one binding wrapper on top of unchanged Rust
  semantics, prefer that binding's build/test skill instead.

## References

- `Cargo.toml`
- `crates/core/Cargo.toml`
- `crates/adaptive/Cargo.toml`
- `crates/core/README.md`
- `crates/adaptive/README.md`
- `docs/contribute/testing-and-docs.mdx`
- `validate-change`
- `maintain-dynamic-plugins`
