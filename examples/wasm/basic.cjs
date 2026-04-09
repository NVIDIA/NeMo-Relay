// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
Minimal WASM example: initialize the generated package, push a scope, and
execute a tool.

Run from the repository root:

  wasm-pack build crates/wasm --scope nvidia --target nodejs
  node examples/wasm/basic.cjs
*/

const wasm = require("../../crates/wasm/pkg/nemo_flow_wasm.js");
const init = wasm.default;
const { pushScope, popScope, toolCallExecute, SCOPE_TYPE_AGENT } = wasm;

async function main() {
  await init();

  const handle = pushScope("example-agent", SCOPE_TYPE_AGENT, null, null, null, null);

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
