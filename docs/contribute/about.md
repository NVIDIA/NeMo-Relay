<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# About

Use this section when you want to contribute to NeMo Flow source code, bindings,
documentation, examples, tests, or third-party integration patches.

Contributing to NeMo Flow often means working across the Rust core, generated and
hand-written language bindings, plugin surfaces, adaptive components,
observability exporters, and documentation. The contribution workflow keeps
those surfaces aligned so public behavior does not drift between supported
languages. Use these pages to keep cross-language changes reviewable and
validated.

## Start Here When

Use these signals to decide whether this documentation path matches your current task.

- Setting up the repository for source development
- Choosing which language tests apply to a change
- Updating docs or examples alongside behavior
- Preparing a pull request for review
- Looking for contribution workflow details beyond user-facing product docs

If you are only consuming NeMo Flow packages, start with [Getting Started](../getting-started/quick-start.md) instead.

## Guides

Use these guide links to move from the overview into task-specific instructions.

- [Development Setup](development-setup.md) covers package installation, source setup, branch naming, and code style.
- [Workflow and Reviews](workflow-and-reviews.md) covers pre-commit hooks, pull request expectations, release tag conventions, DCO sign-off, commit messages, and review rules.
- [Testing and Documentation](testing-and-docs.md) covers affected-language test selection, common build and test commands, documentation checks, and licensing expectations.

Read [Development Setup](development-setup.md) before building locally. Use
[Testing and Documentation](testing-and-docs.md) to choose the smallest
validation set that covers your change. Before opening a pull request, review
[Workflow and Reviews](workflow-and-reviews.md) and the repository root
`CONTRIBUTING.md` guide.
