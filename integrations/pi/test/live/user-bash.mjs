// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { deny, extension, withRpc } from './rpc.mjs';

const [pi, version] = process.argv.slice(2);
assert.ok(pi && ['0.84.0', '1.6.0'].includes(version), 'supply the Pi executable and its exact version');
await withRpc(pi, async (root) => {
  const probe = join(root, 'shell-probe.mjs');
  await writeFile(probe, `export default pi => pi.on('user_bash', event => {
    if (event.command.includes('thrown')) throw new Error('fixture shell exception');
    if (event.command.includes('invalid')) return {};
  });`);
  return { extensions: [probe, extension], reply: e =>
    e.hook_event_name === 'user_bash' && e.input.command.includes('denied') ? deny('fixture shell denial') : {} };
}, async ({ root, posts, command }) => {
  for (const name of ['allowed', 'denied', 'thrown', 'invalid']) {
    const response = await command('bash', { command: 'echo ran > ' + name });
    const executes = name === 'allowed' || (version === '0.84.0' && ['thrown','invalid'].includes(name));
    assert.equal(existsSync(join(root, name)), executes, JSON.stringify(response));
    if (name === 'denied') {
      assert.equal(response.data.exitCode, 126);
      assert.match(response.data.output, /fixture shell denial/);
    }
  }
  assert.ok(posts.some(e => e.hook_event_name === 'user_bash' && e.input.command.includes('allowed')));
});
