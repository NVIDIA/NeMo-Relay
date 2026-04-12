<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Debug application-side NeMo Flow integration issues such as load failures, inactive scopes, missing events, or adaptive/plugin wiring problems
---

# Debug Runtime Integration

Use this skill when NeMo Flow is present in the application but something is not
working.

## First Checks

- Can the binding or native artifact load?
- Is there an active scope when the failing call runs?
- Is the work happening on the expected scope stack?
- Is the subscriber/exporter/plugin configuration actually active?
- Did the app choose the right public API layer: managed execute vs manual
  lifecycle vs typed wrappers vs adaptive/plugin host?

## Common Failure Classes

- Python native extension missing
- Go dynamic library not on the loader path
- Node native addon not built or not loading
- WASM package not initialized
- execute call outside a scope
- missing events because registration never happened
- concurrency causing the wrong scope stack to be active
- adaptive component never initialized or config validation ignored

## References

- `docs/getting-started-python.md`
- `docs/getting-started-go.md`
- `docs/getting-started-node.md`
- `docs/getting-started-wasm.md`
- `docs/context-isolation.md`
- `docs/testing.md`
