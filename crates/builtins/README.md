<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay built-in composition

Shared default composition for the Rust, Python, Node.js, and Go SDKs and the CLI.
Registers observability, pricing, resource metrics, adaptive, and PII redaction.
Registration makes components available; plugin configuration controls activation.
Custom embeddings can use `nemo-relay-plugin-host` directly to choose their own composition.
