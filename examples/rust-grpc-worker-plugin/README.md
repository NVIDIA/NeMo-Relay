<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust gRPC Worker Plugin

This project is the checked Rust worker used by the NeMo Relay plugin authoring
guide. It validates the shared documentation configuration, registers every
safe `grpc-v1` surface, exercises continuations and lazy streams, uses
invocation-scoped codecs, and demonstrates marks and scope-stack cleanup.

Run `cargo test` and `cargo build` from this directory. The tests are
order-independent and each asserts one configuration, schema, or manifest
contract. Copy
`relay-plugin.toml` to `relay-plugin.local.toml`, replace the platform worker
placeholder with the built executable name, and replace the digest placeholder
with `shasum -a 256`, `sha256sum`, or `Get-FileHash` output for that executable.

The complete registration, execution, verification, and shutdown procedure is
in the [Rust Worker guide](https://docs.nvidia.com/nemo/relay/build-plugins/workers/rust).
