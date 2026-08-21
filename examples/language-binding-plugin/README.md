<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Language Binding Plugin

These Rust, Python, and Node.js hosts implement the same application-owned
`documentation-plugin`. Every test owns one behavior and can run by itself;
setup and teardown do not depend on another test having run first.

Run each project from its own directory:

```bash
(cd rust && cargo test)
(cd python && uv run --locked --group test pytest)
(cd node && npm test)
```

The test names separate validation, activation, tool and model policies, request
rewrites, streaming, subscription, and teardown. The `main` program in each
directory is the end-to-end demonstration, while its atomic tests identify the
exact contract that failed.

Each implementation accepts the same optional registration control:

| Field | Default | Meaning |
|---|---|---|
| `registration_control.enabled` | `false` | Install one activation-owned conditional middleware guardrail. |
| `registration_control.kinds` | `["subscriber"]` | Nonempty runtime registration kinds eligible for suppression. |
| `registration_control.registration_name` | `"documentation-controlled-subscriber"` | Effective global target name discovered for the current activation. |
| `registration_control.reason` | `"disabled by documentation plugin"` | Nonempty reason returned while the gate is active. |

The disabled default prevents the runnable example from suppressing unrelated
application middleware. Refer to
[Conditional Middleware Guardrails](../../docs/about-nemo-relay/concepts/conditional-middleware-guardrails.mdx)
for discovery, ownership, snapshot timing, and cleanup.
