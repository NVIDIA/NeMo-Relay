// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';
import { chatProvider, until } from './chat-provider.mjs';
import { extension, deny, withRpc } from './rpc.mjs';

const name = 'mcp__fixture__measure';
const [pi, ledger] = process.argv.slice(2);
if (pi === '--mcp') {
  for await (const line of createInterface({ input: process.stdin })) {
    const message = JSON.parse(line);
    if (message.id === undefined) continue;
    let result = {};
    if (message.method === 'initialize') result = { protocolVersion: message.params.protocolVersion,
      serverInfo: { name: 'fixture', version: '1' }, capabilities: { tools: {} } };
    if (message.method === 'tools/list') result = { tools: [{ name: 'measure', description: 'record a value',
      inputSchema: { type: 'object', properties: { value: { type: 'integer' } }, required: ['value'] } }] };
    if (message.method === 'tools/call') {
      await appendFile(ledger, JSON.stringify(message.params.arguments) + '\n');
      result = { content: [{ type: 'text', text: 'recorded' }] };
    }
    process.stdout.write(JSON.stringify({ jsonrpc:'2.0', id:message.id, result }) + '\n');
  }
} else {
  assert.ok(pi, 'supply the Pi 1.6.0 executable');
  const model = await chatProvider(n => n === 1 ? ['42','13','invalid'].map(value => [name, {value}]) : []);
  try {
    await withRpc(pi, async (root, agent) => {
      await model.configure(agent);
      await writeFile(join(agent, 'mcp.json'), JSON.stringify({ mcpServers: { fixture: { command: process.execPath,
        args: [fileURLToPath(import.meta.url), '--mcp', join(root, 'executed')], exposure:'direct' } } }));
      return { extensions:[extension,'builtin:mcp'], reply(e) {
        if (e.hook_event_name !== 'tool_call') return {};
        return e.input.value === 42 ? { body:{ tool_call:{ tool_call_id:e.tool_call_id, input:{ value:7 } } } } : deny('rejected');
      } };
    }, async ({ root, command, posts }) => {
      assert.ok((await command('prompt', { message:'run fixture' })).success);
      await until(() => posts.some(e => e.hook_event_name === 'agent_settled'), 'Pi settlement');
      const gates = posts.filter(e => e.hook_event_name === 'tool_call' && e.tool_name === name);
      assert.deepEqual(gates.map(e => e.input.value).sort((a,b) => a-b), [13,42]);
      assert.deepEqual((await readFile(join(root,'executed'),'utf8')).trim().split('\n').map(s => JSON.parse(s)), [{value:7}]);
      const ends = posts.filter(e => e.hook_event_name === 'tool_execution_end' && e.tool_name === name);
      assert.deepEqual(ends.map(e=>e.status).sort(), ['error','error','ok']);
    });
  } finally { await model.close(); }
}
