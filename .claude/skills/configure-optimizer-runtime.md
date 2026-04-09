<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Configure and register the NeMo Flow optimizer runtime, components, and hosted plugins
---

# Configure Optimizer Runtime

Use this skill when an application already intends to use the optimizer and now
needs a correct runtime configuration.

## Default Path

1. Build the config document or typed helper config.
2. Choose the state backend.
3. Add the built-in components or `external_component` entries.
4. Validate the config.
5. Construct the runtime.
6. Register, use, deregister, and shut down it cleanly.

## Checklist

- [ ] backend chosen (`in_memory` first unless persistence is required)
- [ ] components chosen explicitly
- [ ] config validated before registration
- [ ] hosted plugins registered before runtime registration when used
- [ ] runtime lifecycle matched to the app lifecycle

## References

- `docs/optimizer-layer.md`
- `docs/optimizer-api-reference.md`
