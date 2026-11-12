// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const {
  ScopeType,
  deregisterSubscriber,
  event,
  registerSubscriber,
  flushSubscribers,
  withScope,
} = require('nemo-relay-node');

async function main() {
  registerSubscriber('scope-check', (runtimeEvent) => {
    console.log(`${runtimeEvent.kind} ${runtimeEvent.name}`);
  });

  try {
    await withScope(
      'agent-run',
      ScopeType.Agent,
      async (handle) => {
        event('planning-started', handle, { step: 1 }, null);
        event('planning-finished', handle, { step: 2 }, null);
      },
      null,
      null,
      null,
      null,
      { request_id: 'req-123' },
    );
  } finally {
    await flushSubscribers();
    deregisterSubscriber('scope-check');
  }
}

module.exports = { main };

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
