// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { chatProvider, until } from './chat-provider.mjs';
import { extension, withRpc } from './rpc.mjs';

const [pi] = process.argv.slice(2);
assert.ok(pi, 'supply the Pi 1.6.0 executable');
const model = await chatProvider(n => n === 1 ? {error:503} : n === 2 ? [['bash',{command:'echo ok'}]] :
  n === 4 ? [['bash',{command:'sleep 15'}]] : []);
try {
  await withRpc(pi, async (root, agent) => {
    await model.configure(agent);
    await writeFile(join(agent,'settings.json'), JSON.stringify({defaultProvider:'fixture',defaultModel:'fixture',
      retry:{enabled:true,baseDelayMs:10,maxRetries:2,provider:{maxRetries:0}}}));
    const helper = join(root,'reload.mjs');
    await writeFile(helper,"export default pi => pi.registerCommand('fixture-reload',{description:'reload',handler:(_,ctx)=>ctx.reload()});");
    return {extensions:[extension,helper]};
  }, async ({command, posts}) => {
    const events = name => posts.filter(e => e.hook_event_name === name);
    assert.equal((await command('compact')).success, false);
    await until(() => events('session_compact_failed').length, 'compaction failure');
    const failure = events('session_compact_failed')[0];
    assert.deepEqual([failure.reason, failure.aborted, failure.will_retry], ['manual',false,false]);
    assert.equal(events('session_compact').length, 0);
    assert.ok((await command('prompt', {message:'exercise retry'})).success);
    await until(() => events('agent_settled').length === 1, 'settled retry');
    assert.equal(events('agent_settled')[0].aborted,false);
    assert.equal(events('agent_settled')[0].attempts,2);
    assert.ok(events('tool_execution_end')[0].pi_duration_ms >= 0);
    assert.ok((await command('prompt',{message:'exercise abort'})).success);
    await until(() => events('tool_call').length === 2, 'running second tool');
    assert.ok((await command('abort')).success);
    await until(() => events('agent_settled').length === 2, 'aborted settlement');
    assert.equal(events('agent_settled')[1].aborted,true);
    assert.ok((await command('prompt',{message:'/fixture-reload'})).success);
    await until(() => events('session_start').some(e => e.reason === 'reload'), 'reload');
    assert.equal(events('session_shutdown').length,0);
    const original = events('session_start')[0].session_id;
    assert.equal(events('session_start').at(-1).session_id, original);
    const messages = await command('get_fork_messages');
    assert.ok((await command('fork',{entryId:messages.data.messages[0].entryId})).success);
    assert.ok((await command('new_session')).success);
    await until(() => events('session_start').some(e => e.reason === 'new'), 'session replacement');
    const starts = events('session_start').filter(e => e.reason !== 'reload');
    assert.deepEqual(starts.map(e => e.reason), ['startup','fork','new']);
    assert.equal(new Set(starts.map(e=>e.session_id)).size,3);
    assert.deepEqual(events('session_shutdown').map(e=>e.session_id),starts.slice(0,2).map(e=>e.session_id));
  });
} finally { await model.close(); }
