<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-flow-ffi

`nemo-flow-ffi` provides the C-compatible ABI layer for NeMo Flow.

It exposes the Rust runtime through exported `nemo_flow_*` symbols and a
generated C header so downstream native bindings can call into the shared
runtime contract.

Use this crate when building language bindings or native integrations that need
the raw C ABI. The repository-maintained Go binding consumes this surface.

For project-level documentation, start with:

- the repo root `README.md`
- `docs/troubleshooting/troubleshooting-guide.md`
