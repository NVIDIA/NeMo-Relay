---
name: maintain-dynamic-plugins
description: Maintain NeMo Relay dynamic plugin loaders, manifests, Rust native SDKs, gRPC worker protocol, Python worker SDK, docs, tests, and release workflow coverage
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Maintain Dynamic Plugins

## Companion Guidance

Use `karpathy-guidelines`, `validate-change`, `maintain-packaging`, and
`contribute-docs` alongside this skill when implementation, packaging, CI, or
documentation changes are involved.

Use this skill for `plugin.kind = "rust_dynamic"`, `plugin.kind = "worker"`,
`nemo-relay-plugin`, `nemo-relay-worker`, `nemo-relay-worker-proto`,
`nemo-relay-types`, and the Python `nemo-relay-plugin` package.

## Rules

- Keep the stable boundary explicit: native plugins cross a C ABI; worker
  plugins cross `grpc-v1`.
- Do not pass Rust runtime types, trait objects, futures, or allocator-owned
  strings across the native dynamic-library boundary.
- Keep worker protocol DTOs in `JsonEnvelope`; protobuf owns control flow, not
  duplicated Relay data models.
- Keep `relay-plugin.toml` dynamic records separate from generic runtime
  components. Enabled dynamic records may synthesize internal component specs;
  disabled records stay inspectable but unloaded.
- Treat plugin Relay compatibility as normal SemVer. Use `>=0.5,<1.0` in
  examples unless a plugin intentionally declares a narrower range.
- Do not add tests under `src`; Rust tests belong in crate `tests/` trees and
  Python SDK tests belong under `python/tests`.
- Native and worker plugins are trusted extensions. Document that native plugins
  are in-process and unsandboxed; worker plugins provide process isolation but
  not a security sandbox.

## Checklist

- [ ] Manifest validation covers kind, compatibility, load contract, integrity,
      capability mismatch, and disabled-plugin behavior.
- [ ] Native loader keeps libraries alive until registered callbacks are cleared
      and deregisters plugin kinds before unload.
- [ ] Worker activation covers process launch, token auth, handshake, validation,
      declarative registration, proxy rollback, cancellation, and shutdown.
- [ ] Rust and Python SDKs expose every supported registration surface.
- [ ] Runtime helpers cover marks, scopes, continuations, and isolated scope
      stacks.
- [ ] CLI `plugins list/inspect/validate/doctor` diagnostics include native ABI
      and worker protocol status without leaking secret config.
- [ ] Docs include separate Rust native, Python worker, and `grpc-v1` protocol
      pages.
- [ ] `justfile`, Codecov, and CI package/test workflows include new plugin
      crates and packages.

## Validation

```bash
cargo test -p nemo-relay-types
cargo test -p nemo-relay-plugin
cargo test -p nemo-relay-worker-proto
cargo test -p nemo-relay-worker
cargo test -p nemo-relay --test native_plugin_tests --test worker_plugin_tests
uv run pytest python/tests/plugin
just test-rust
just test-python
just docs
```

For broad runtime or public API changes, run the full `validate-change` matrix.

## References

- `crates/core/src/plugin/dynamic/`
- `crates/plugin`
- `crates/worker`
- `crates/worker-proto`
- `crates/types`
- `python/plugin`
- `examples/rust-native-plugin`
- `docs/build-plugins`
- `examples/python-grpc-worker-plugin`
