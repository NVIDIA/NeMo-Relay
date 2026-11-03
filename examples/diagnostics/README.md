<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Embedded Diagnostics Readers

These examples enable the process diagnostics service through ordinary plugin-host
configuration, validate it before startup, inspect native and canonical pages and
reader metadata, follow bounded pagination, check a stable query error, and keep
reading after plugin teardown. They use the public APIs without a listener or an
exporter. Each program requires a fresh process because service settings are chosen
once for that process.

From the repository root, after installing/building the corresponding binding:

```bash
uv run python examples/diagnostics/python.py
node examples/diagnostics/node.cjs
cargo run -p nemo-relay --example embedded_diagnostics
```

The Rust source lives at
[`crates/sdk/examples/embedded_diagnostics.rs`](../../crates/sdk/examples/embedded_diagnostics.rs)
so Cargo can build it with the core crate's existing dependencies. Python and Node
example smoke tests run through their normal binding test suites.

The examples use the existing plugin configuration resolver. Programmatic
`diagnostics` settings replace the whole file-derived diagnostics profile, while
plugin components retain their normal file layering. Review ambient user/system
plugin configuration before running the examples in an existing application
environment. Tests suppress implicit discovery through the existing test hook.

Reader access permits inspection of potentially unsanitized evidence. The examples
do not print or export pages, attach them to exceptions, or register subscribers.
Adapt their page-consumption comments within your application's access controls.
See the [runtime diagnostics reference](../../docs/reference/diagnostics.mdx) for
ownership, memory budgets, numeric representations, and failure reasons.
