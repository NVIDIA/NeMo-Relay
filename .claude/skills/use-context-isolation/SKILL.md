<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Set up and reason about NeMo Flow scope-stack isolation for concurrent application work
---

# Use Context Isolation

Use this skill when an application runs concurrent requests, worker pools, async
tasks, goroutines, or multiple agents in the same process.

## Core Rule

Each independent request, agent, or workflow needs its own scope stack. Do not
share one mutable stack across unrelated concurrent work unless you want shared
ancestry and shared scope-local middleware.

## Per-Language Defaults

- **Python**: rely on task-local behavior via `get_scope_stack()` and
  `contextvars`, or explicitly propagate when work leaves the current execution
  context
- **Go**: use `ScopeStack.Run()` for goroutine-safe usage
- **Node.js**: create and set a scope stack explicitly for the current execution
  path
- **WASM**: set the scope stack manually; single-threaded does not remove the
  need for isolation between logical runs

## Common Failures

- events from different requests appear under one root UUID
- scope-local middleware leaks across requests
- worker-thread work runs without the expected active scope
- integrations activate NeMo Flow without an explicitly initialized stack

## References

- `docs/context-isolation.md`
- `docs/concepts.md`
