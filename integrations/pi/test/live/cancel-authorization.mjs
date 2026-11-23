// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { chatProvider, until } from './chat-provider.mjs';
import { extension, withRpc } from './rpc.mjs';

const [pi] = process.argv.slice(2);
assert.ok(pi, 'supply the Pi 1.6.0 executable');

const model = await chatProvider(n => [
  [['write', { path: 'cancelled.txt', content: 'must not execute' }]],
  [['write', { path: 'recovered.txt', content: 'next prompt executed' }]],
][n - 1] ?? []);
const gate = Promise.withResolvers();
let held;
try {
  await withRpc(pi, async (_, agent) => {
    await model.configure(agent);
    return {
      extensions: [extension],
      env: { NEMO_RELAY_PI_FAIL: 'open', NEMO_RELAY_PI_HOOK_STREAM: '1' },
      async reply(event) {
        if (event.hook_event_name === 'tool_call' && event.input.path === 'cancelled.txt') {
          held = event;
          await gate.promise;
          return { body: { tool_call: {
            tool_call_id: event.tool_call_id,
            input: { ...event.input, path: 'late-rewrite.txt' },
          } } };
        }
        return {};
      },
    };
  }, async ({ root, command, posts }) => {
    try {
      assert.ok((await command('prompt', { message: 'exercise cancelling a write' })).success);
      await until(() => held, 'authorization received');
      const start = performance.now();
      const abort = await Promise.race([
        command('abort'),
        delay(2_000, undefined, { ref: false }),
      ]);
      assert.ok(abort?.success, 'abort must complete while Relay is still deciding');
      assert.equal(model.requests.length, 1, 'cancelling must end automatic model requests');
      console.log(`Pi abort completed in ${(performance.now() - start).toFixed(1)} ms with authorization pending`);
    } finally {
      gate.resolve();
    }
    await until(() => posts.some(post => post.hook_event_name === 'agent_settled'), 'aborted turn drained');
    for (const path of ['cancelled.txt', 'late-rewrite.txt']) {
      await assert.rejects(readFile(join(root, path)), { code: 'ENOENT' });
    }
    assert.equal(posts.filter(post => post.hook_event_name === 'tool_arguments_transformed').length, 0);
    assert.ok((await command('prompt', { message: 'exercise recovery' })).success);
    await until(() => posts.filter(post => post.hook_event_name === 'agent_settled').length === 2, 'second turn completed');
    assert.equal(await readFile(join(root, 'recovered.txt'), 'utf8'), 'next prompt executed');
  });
} finally {
  gate.resolve();
  await model.close();
}
