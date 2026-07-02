<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scripts

The canonical build and test surface now lives in the repository `justfile`.
Use `just --list` to discover supported developer workflows.

Keep `scripts/` focused on helpers that are still script-native:

## Top-Level Commands

- `build-docs.sh`: compatibility wrapper around the Fern documentation validation recipe; it regenerates ignored Fern API reference pages before checking the site
- `generate_attributions.sh`: regenerate attribution documents
- `test-install.sh`: run live GitHub release and local interface checks for the curl-based CLI installer
- `test-install.ps1`: run live GitHub release and local interface checks for the PowerShell CLI installer
- `test-install-mocks.sh`: run installer scenarios that require simulated platforms or failures

## Internal Layout

- `docs/`: Fern reference-generation, migration cleanup, and `docs-website` branch sync helpers. Generated API reference output under `docs/reference/api/*-library-reference/` is ignored and recreated by `just docs`.
- `licensing/`: attribution generation helpers, including license inventory diff scripts
- `lint/`: pre-commit and local lint helpers
- `test-support/`: shared test utilities
