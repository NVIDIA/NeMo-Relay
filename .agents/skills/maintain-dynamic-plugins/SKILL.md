---
name: maintain-dynamic-plugins
description: Change NeMo Relay dynamic plugin loaders, manifests, native ABI, gRPC worker protocol, or Python worker SDK. Do not use for ordinary built-in plugin configuration.
license: Apache-2.0
---

# Maintain Dynamic Plugins

Use this skill for `plugin.kind = "rust_dynamic"`, `plugin.kind = "worker"`,
`nemo-relay-plugin`, `nemo-relay-worker`, `nemo-relay-worker-proto`,
`nemo-relay-types`, and the Python `nemo-relay-plugin` package.

## Rules

- Keep the stable boundary explicit: native plugins cross a C ABI; worker
  plugins cross `nvidia.nemo.relay.worker.v1` (contract `1.0.0`).
- Use `abi_stable` checked records and generated trait objects, ABI-safe owned
  values, and `async-ffi` futures across the native dynamic-library boundary.
  Keep ordinary Rust runtime types, futures, and Tokio objects within their
  originating library.
- Poll typed native middleware directly on the caller, restoring Relay scope
  and optional SDK Tokio context for every poll and drop. Inline is the SDK
  default and allocates no plugin runtime. Tokio-enabled components share a
  library-local runtime and must agree on its worker count. Use `ctx.spawn` or
  `ctx.task_spawner()` for component-owned background work; keep subscribers
  synchronous and preserve raw synchronous ABI registrations.
- Define closed worker transport structures in protobuf when generated clients
  must enforce their fields. Keep open application payloads lossless by using
  `JsonValue` or `JsonObject` rather than `google.protobuf.Value`.
- Keep `relay-plugin.toml` dynamic records separate from generic runtime
  components. Enabled records can create internal component specs. Disabled records remain
  available for inspection but do not load.
- Require worker manifests to declare the protocol identity and a `compat.relay`
  range. The worker SDK uses `>=1.0.0,<2.0`.
- Treat `compat.relay` as the plugin author's supported release range. Verify
  binary layouts or worker handshake independently.
- The worker protocol is `nvidia.nemo.relay.worker.v1`, contract version `1.0.0`.
  Native plugins use checked contract `v1.0`. Freeze published module types
  and bootstrap layouts. Add minor contracts through versioned modules; plugins
  require at least the native minor of their build SDK. Breaking layouts or
  dependency ABI generations require a native major change, independently of
  Relay's release version. A breaking worker contract needs a new protocol identity.
- Keep the native contract source in `crates/plugin-abi/src/contract.rs`.
  The interface package follows the workspace release version. Its build script
  derives SDK constants and checked layout metadata from the explicit contract,
  not the package release. Keep published bootstrap and module version strings
  fixed. Validate cross-package-version fixtures when changing this mechanism.
- Do not add tests under `src`; Rust tests belong in crate `tests/` trees and
  Python SDK tests belong under `python/tests`.
- Native and worker plugins are trusted extensions. Document that native plugins
  are in-process and unsandboxed; worker plugins provide process isolation but
  not a security sandbox.

## Checklist

- [ ] Manifest validation covers kind, compatibility, load contract, integrity,
      capability mismatch, and disabled-plugin behavior.
- [ ] Native libraries and staged dependencies remain resident until process exit.
      Component close removes registrations, revokes capabilities, cancels and
      drains tasks, and releases its runtime lease; the last lease stops the
      shared runtime. Incomplete cleanup retains
      ownership for retry.
- [ ] Worker activation covers process launch, token auth, handshake, validation,
      declarative registration, proxy rollback, cancellation, and shutdown.
- [ ] Rust and Python SDKs expose every supported registration surface.
- [ ] Runtime helpers cover marks, scopes, continuations, and isolated scope
      stacks.
- [ ] `plugins list`, `plugins inspect`, and `plugins validate` report lifecycle
      and compatibility status without leaking secret config.
- [ ] Top-level `doctor` reports resolved dynamic plugin and host configuration
      status.
- [ ] When detailed dynamic plugin guides exist, they keep Rust native, Python
      worker, and `nvidia.nemo.relay.worker.v1` protocol details on separate pages.
- [ ] `justfile`, Codecov, and CI package/test workflows include new plugin
      crates and packages.

## Validation

Choose checks by the changed layer:

- Manifest or shared types: test the owning Rust crate.
- Native loader or ABI: prepare plugin fixtures and run the focused native
  plugin integration test. For ABI changes, also run
  `just build-native-abi-test-fixtures` and
  `cargo nextest run --locked -p nemo-relay-plugin-abi --test binary_compatibility`.
  Native integration fixtures do not prepare the independent ABI workspaces.
- Worker protocol or host: test the worker crates and regenerate or test the
  Python worker SDK only when its protocol surface changes.
- Shared runtime behavior: run the Rust suite and only binding suites whose
  observable plugin behavior changes.
- Documentation or packaging: run their targeted checks only when changed.

Canonical surface suites prepare plugin fixtures. Before a raw focused native
or worker integration test, run `just build-test-plugin-fixtures`; never compile
fixtures inside an individual test case.

## References

- `crates/plugin-host/src/dynamic/`
- `crates/plugin`
- `crates/worker`
- `crates/worker-proto`
- `crates/types`
- `python/plugin`
- `examples/rust-native-plugin`
- `docs/build-plugins`
- `examples/python-grpc-worker-plugin`
