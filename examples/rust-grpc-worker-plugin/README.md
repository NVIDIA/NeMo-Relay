<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust gRPC Worker Plugin

This is the Rust worker used by the NeMo Relay plugin guide. It supports all 17
`nvidia.nemo.relay.worker.v1` callback types. Configuration selects which callbacks
to register. The example shows annotations, pending marks, codec access, lazy
streams, and cleanup of scopes and cancelled work.

The worker targets Relay 1.x using the `nvidia.nemo.relay.worker.v1` protocol
(contract `1.0.0`). Its manifest requires `compat.relay = ">=1.0.0,<2.0"`.
Non-streaming and streaming LLM execution callbacks receive request codec
context before their continuation. Non-streaming callbacks also receive
completed-response codec context; streaming callbacks do not because chunks
are not complete provider responses.

Building a Rust worker requires `protoc` 31.1 or later and Buf 1.75.0 or later.
Install both on `PATH`, or select them with `PROTOC` and `BUF`. The first build
needs access to the Buf Schema Registry to download the locked validation schema.

Build and test the worker, then create its local manifest:

1. Run `cargo test --locked` from this directory. The configuration and schema
   tests can run in any order. The lifecycle test builds a worker, checks its
   manifest digest, activates it, runs middleware, checks a mark, and verifies shutdown.
2. Run `cargo build --locked --target-dir target`.
3. Copy `relay-plugin.toml` to `relay-plugin.local.toml`. Replace the platform
   worker placeholder with the built executable's name.
4. Replace only `<artifact-sha256>` with the executable's lowercase SHA-256
   digest. Keep the `sha256:` prefix. Use the digest printed by `shasum -a 256`,
   `sha256sum`, or `Get-FileHash`; do not include the filename.

The optional `registration_control` group installs one callback-based gate for the
worker activation. It starts disabled. Its default kinds are `["subscriber"]`,
its target is `documentation-controlled-subscriber`, and its reason is
`disabled by documentation plugin`. The callback returns that reason for targets
whose names start with `documentation-controlled-` and returns `None` to leave
other matching targets enabled. All three values must be nonempty. Refer to
[Conditional Middleware Guardrails](../../docs/about-nemo-relay/concepts/conditional-middleware-guardrails.mdx)
for effective-name discovery and automatic teardown behavior.
