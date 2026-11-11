// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const {
  LlmRequest,
  ScopeType,
  deregisterSubscriber,
  flushSubscribers,
  llmCallExecute,
  registerSubscriber,
  withScope,
} = require('nemo-relay-node');

function provider(request) {
  return { text: 'hello', messages: request.content.messages };
}

async function main() {
  registerSubscriber('llm-check', (event) => {
    console.log(`${event.kind} ${event.name}`);
  });

  try {
    return await withScope('agent-run', ScopeType.Agent, async (handle) => {
      const request = new LlmRequest({}, { messages: [{ role: 'user', content: 'hello' }] });
      const result = await llmCallExecute('demo-provider', request, provider, handle, null, null, null, 'demo-model');

      console.log(result);
      return result;
    });
  } finally {
    await flushSubscribers();
    deregisterSubscriber('llm-check');
  }
}

module.exports = { main, provider };

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
