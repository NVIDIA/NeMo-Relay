---
name: test-ffi-surface
description: Build and test the NeMo Relay FFI surface; use this for crates/ffi changes, header generation, or ABI-facing validation
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Build And Test FFI Surface

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when the change is primarily in `crates/ffi`, the generated
header, or ABI-facing behavior consumed by Go or external native callers.

## Default Path

1. Rebuild the FFI crate in release mode so the shared library and header stay
   in sync.
2. Run `cargo fmt --all` because FFI work is Rust work.
3. Run `just test-rust`.
4. Run `cargo clippy --workspace --all-targets -- -D warnings`.
5. Check the generated header diff when any exported symbol or type changed.
6. If downstream consumers changed too, run their binding-specific skills next.

## Common Commands

```bash
# Rebuild the shared library with the repo wrapper used by FFI consumers
just build-go

# Required Rust validation
cargo fmt --all
just test-rust
cargo test -p nemo-relay-ffi
cargo clippy --workspace --all-targets -- -D warnings

# Review header drift if the FFI surface changed
git diff -- crates/ffi/nemo_relay.h
```

## When To Escalate

- If Go behavior changed, also use `test-go-binding`.
- If the FFI change reflects shared runtime semantics rather than a pure bridge
  change, also use `validate-change`.

## References

- `crates/ffi/Cargo.toml`
- `crates/ffi/build.rs`
- `crates/ffi/cbindgen.toml`
- `crates/ffi/nemo_relay.h`
- `just build-go`
- `.pre-commit-config.yaml`
- `README.md`
- `validate-change`
