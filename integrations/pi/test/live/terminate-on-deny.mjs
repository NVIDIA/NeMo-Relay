// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { chatProvider, until } from './chat-provider.mjs';
import { deny, extension, withRpc } from './rpc.mjs';

const [pi, version] = process.argv.slice(2);
assert.ok(['0.84.0', '1.6.0'].includes(version));

async function exercise({ enabled, verdict, expectedRequests }) {
  const paths = ['first.txt', 'second.txt'];
  const model = await chatProvider(n => n === 1
    ? paths.map(path => ['write', { path, content: 'executed' }]) : []);
  try {
    await withRpc(pi, async (_, agent) => {
      await model.configure(agent);
      return {
        extensions: [extension],
        env: enabled ? { NEMO_RELAY_PI_TERMINATE_ON_DENY: '1' } : {},
        reply(event) {
          if (event.hook_event_name !== 'tool_call') return {};
          if (verdict === 'fault') return { status: 503 };
          if (verdict === 'mixed' && event.input.path === paths[1]) return {};
          return deny('fixture policy');
        },
      };
    }, async ({ root, command, posts }) => {
      assert.ok((await command('prompt', { message: 'exercise writes' })).success);
      await until(() => posts.some(p => p.hook_event_name === 'agent_settled'), 'settled tool batch');
      assert.equal(model.requests.length, expectedRequests, 'automatic model requests');
      assert.equal(posts.filter(p => p.hook_event_name === 'tool_call').length, 2);
      for (const path of paths) {
        const read = readFile(join(root, path), 'utf8');
        if (verdict === 'mixed' && path === paths[1]) assert.equal(await read, 'executed');
        else await assert.rejects(read, { code: 'ENOENT' });
      }
    });
  } finally {
    await model.close();
  }
}

await exercise({ enabled: true, verdict: 'deny', expectedRequests: version === '1.6.0' ? 1 : 2 });
if (version === '1.6.0') {
  await exercise({ enabled: false, verdict: 'deny', expectedRequests: 2 });
  await exercise({ enabled: true, verdict: 'mixed', expectedRequests: 2 });
  await exercise({ enabled: true, verdict: 'fault', expectedRequests: 2 });
}
