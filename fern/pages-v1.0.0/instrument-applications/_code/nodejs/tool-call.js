// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const {
  ScopeType,
  deregisterSubscriber,
  flushSubscribers,
  registerSubscriber,
  toolCallExecute,
  withScope,
} = require('nemo-relay-node');

function search(toolArguments) {
  return {
    result: {
      query: toolArguments.query,
      hits: [{ title: 'NeMo Relay' }],
    },
  };
}

async function main() {
  registerSubscriber('instrumentation-check', (event) => {
    console.log(`${event.kind} ${event.name}`);
  });

  try {
    return await withScope('agent-run', ScopeType.Agent, async (handle) => {
      const result = await toolCallExecute(
        'search',
        { query: 'runtime instrumentation' },
        search,
        handle,
        null,
        null,
        null,
      );

      console.log(result.result);
      return result.result;
    });
  } finally {
    await flushSubscribers();
    deregisterSubscriber('instrumentation-check');
  }
}

module.exports = { main, search };

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
