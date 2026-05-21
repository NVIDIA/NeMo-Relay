---
name: test-go-binding
description: Build and test the NeMo Relay Go binding; use this for go/nemo_relay changes or Go-facing integration checks
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Build And Test Go Binding

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when the change is primarily in `go/nemo_relay` or the Go
binding behavior it depends on.

## Important Constraint

The Go binding links against the shared FFI library. `just test-go`
builds that library for you, while `just build-go` remains useful when
you want an explicit build-only pass or need the artifact for other work.

## Default Path

1. Format changed Go packages with `cd go/nemo_relay && go fmt ./...`.
2. Run Go tests with `just test-go`.
3. If any Rust files changed as part of the Go work, also run
   `cargo fmt --all`, `just test-rust`, and
   `cargo clippy --workspace --all-targets -- -D warnings`.
4. Use `just build-go` when you want an explicit build-only pass.
5. Use `just ci=true test-go` when you need the CI-style coverage and JUnit path.
6. Expand to broader validation only if the change touched shared semantics.

## Common Commands

```bash
# Full Go suite
just test-go

# Format Go files
cd go/nemo_relay && go fmt ./...

# Required when the Go change also touched Rust code
cargo fmt --all
just test-rust
cargo clippy --workspace --all-targets -- -D warnings

# CI-style Go suite with coverage and JUnit artifacts
just ci=true test-go

# Explicit shared-library build when needed separately
just build-go
```

In the `test-go` task, the `ci` variable is what `is_true "{{ ci }}"` checks.
Setting `ci=true` enables `coverage_out` and `junit_out` handling and adds
`-coverprofile=coverage.out` to `go_test_cmd`.

On macOS, also set `DYLD_LIBRARY_PATH` to the same `../../target/release`
directory before running the raw `go test` command directly.

## When To Escalate

- If the change touched `crates/ffi`, also use `test-ffi-surface`.
- If the change touched `crates/core` or shared runtime semantics, also use
  `validate-change`.

## References

- `go/nemo_relay/go.mod`
- `go/nemo_relay/nemo_relay.go`
- `README.md`
- `docs/getting-started/installation.md`
- `validate-change`
