// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { until } from './chat-provider.mjs';
import { extension, withRpc } from './rpc.mjs';

const [pi] = process.argv.slice(2);
const dialogs = [];
await withRpc(pi, async root => {
  const probe = join(root, 'dialogs.mjs');
  await writeFile(probe, `export default pi => pi.registerCommand('fixture-dialog', {
    description: 'Open overlapping dialogs',
    handler: async (_, ctx) => Promise.all([
      ctx.ui.confirm('private title', 'private confirmation'),
      ctx.ui.input('private input', 'private placeholder'),
    ]),
  });`);
  return {
    extensions: [extension, probe],
    onEvent(event, send) {
      if (event.type !== 'extension_ui_request') return;
      dialogs.push(event.method);
      setTimeout(() => send({ type: 'extension_ui_response', id: event.id,
        ...(event.method === 'confirm' ? { confirmed: true } : { value: 'private response' }),
      }), event.method === 'confirm' ? 40 : 100);
    },
  };
}, async ({ command, posts }) => {
  assert.ok((await command('prompt', { message: '/fixture-dialog' })).success);
  await until(() => posts.some(p => p.hook_event_name === 'ui_prompt_end'), 'outer dialog end');
  const waits = posts.filter(p => p.hook_event_name.startsWith('ui_prompt_'));
  assert.deepEqual(dialogs, ['confirm', 'input']);
  assert.deepEqual(waits.map(p => [p.hook_event_name, p.kind]), [
    ['ui_prompt_start', 'confirm'], ['ui_prompt_end', 'confirm'],
  ]);
  assert.ok(waits[0].session_id);
  assert.equal(waits[0].session_id, waits[1].session_id);
  assert.ok(waits[1].wait_duration_ms >= 50);
  assert.doesNotMatch(JSON.stringify(waits), /private/);
});
