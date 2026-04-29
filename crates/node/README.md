<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-flow-node

`nemo-flow-node` is the Rust native extension crate behind the NeMo Flow Node.js
package.

It uses napi-rs to expose NeMo Flow runtime scopes, tool and LLM lifecycle
helpers, middleware, subscribers, typed helpers, plugin helpers, and adaptive
runtime helpers to JavaScript and TypeScript.

Most Node.js users should install the npm package rather than depend on this
crate directly:

```bash
npm install nemo-flow-node
```

For project-level documentation, start with:

- the repo root `README.md`
- `docs/getting-started/nodejs.md`
- `docs/reference/api/nodejs/index.md`
