// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
Minimal Node.js example: push a scope and execute a tool.

Run from the repository root:

  npm --prefix crates/node install
  npm --prefix crates/node run build
  node examples/node/basic.cjs
*/

const { ScopeType, pushScope, popScope, toolCallExecute } = require("../../crates/node/index.js");

async function main() {
  const handle = pushScope("example-agent", ScopeType.Agent, null, null, null, null);

  const result = await toolCallExecute(
    "search",
    { query: "hello" },
    async (args) => ({ results: [`echo:${args.query}`] }),
    null,
    null,
    null,
    null,
  );

  console.log(result);
  popScope(handle);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
