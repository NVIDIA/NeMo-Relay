<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust gRPC Worker Plugin

This project is the checked Rust worker used by the NeMo Relay plugin authoring
guide. It validates the shared documentation configuration, registers every
safe `grpc-v1` surface, exercises continuations and lazy streams, uses
invocation-scoped codecs, and demonstrates marks and scope-stack cleanup.

Run `cargo test` and `cargo build` from this directory. The configuration and
schema tests are order-independent. The lifecycle test builds a fresh worker,
materializes a digest-checked manifest, activates it through `grpc-v1`, runs
managed middleware, observes a host-runtime mark, and verifies shutdown. Copy
`relay-plugin.toml` to `relay-plugin.local.toml`, replace the platform worker
placeholder with the built executable name, and replace only `<artifact-sha256>`
with the lowercase hexadecimal digest of the built executable. The manifest value
keeps its `sha256:` prefix; omit the filename column printed by `shasum -a 256`,
`sha256sum`, or `Get-FileHash`.

The optional `registration_control` group installs one callback-based gate for the
worker activation. It defaults to disabled, with `kinds: ["subscriber"]`, target
`documentation-controlled-subscriber`, and reason
`disabled by documentation plugin`. The callback returns that reason for targets
whose names start with `documentation-controlled-` and returns `None` to leave
other matching targets enabled. All three values must be nonempty. Refer to
[Conditional Middleware Guardrails](../../docs/about-nemo-relay/concepts/conditional-middleware-guardrails.mdx)
for effective-name discovery and automatic teardown behavior.
