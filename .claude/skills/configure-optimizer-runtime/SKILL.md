<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Configure the NeMo Flow adaptive layer and hosted plugins through the shared plugin host; use this when users still say optimizer
---

# Configure Adaptive Layer

Use this skill when an application already intends to use adaptive features
(sometimes still called the optimizer) and now needs a correct configuration.

There is no separate public adaptive runtime object. Adaptive is configured as a
top-level plugin component inside the shared plugin host.

## Default Path

1. Build the shared plugin config document or binding-native helper config.
2. Add one top-level `adaptive.ComponentSpec(...)`.
3. Choose the state backend.
4. Enable only the adaptive sections you need: `telemetry`,
   `adaptive_hints`, and `tool_parallelism`.
5. Validate the config.
6. Initialize through the shared plugin host.
7. Clear or replace the plugin configuration cleanly when the app lifecycle
   changes.

## Checklist

- [ ] Adaptive is modeled as a top-level plugin component, not a nested runtime
- [ ] Backend chosen (`in_memory` first unless persistence is required)
- [ ] Adaptive sections chosen explicitly
- [ ] Config validated before initialization
- [ ] Custom plugins added as sibling top-level components when used
- [ ] Plugin lifecycle matched to the app lifecycle

## References

- `docs/adaptive-layer.md`
- `docs/adaptive-api-reference.md`
- `docs/hosted-plugins.md`
- `docs/recipes.md`
