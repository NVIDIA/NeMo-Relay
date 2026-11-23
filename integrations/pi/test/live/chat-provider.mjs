// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { once } from 'node:events';
import { createServer } from 'node:http';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

/** Script a local OpenAI-compatible model through Pi's actual streaming client. */
export async function chatProvider(respond) {
  const requests = [];
  const server = createServer(async (req, res) => {
    let body = '';
    for await (const chunk of req) body += chunk;
    const request = JSON.parse(body);
    requests.push(request);
    const calls = respond(requests.length, request);
    if (calls?.error) {
      res.writeHead(calls.error, { 'content-type': 'application/json' });
      res.end(JSON.stringify({error:{message:'Temporary server error',type:'server_error'}}));
      return;
    }
    const delta = calls?.length ? { role: 'assistant', tool_calls: calls.map(([name, args], i) => ({
      index: i, id: 'call_' + i, type: 'function', function: { name, arguments: JSON.stringify(args) },
    })) } : { role: 'assistant', content: 'fixture complete' };
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    for (const [change, end] of [[delta, null], [{}, calls?.length ? 'tool_calls' : 'stop']]) {
      res.write('data: ' + JSON.stringify({ id: 'fixture', object: 'chat.completion.chunk', created: 1,
        model: request.model, choices: [{ index: 0, delta: change, finish_reason: end }] }) + '\n\n');
    }
    res.end('data: [DONE]\n\n');
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  return {
    requests,
    async configure(agent) {
      await writeFile(join(agent, 'models.json'), JSON.stringify({providers:{fixture:{
        api:'openai-completions', apiKey:'fixture', baseUrl:'http://127.0.0.1:' + server.address().port + '/v1',
        models:[{id:'fixture',contextWindow:64000,maxTokens:4096}],
      }}}));
      await writeFile(join(agent, 'settings.json'), JSON.stringify({defaultProvider:'fixture',defaultModel:'fixture'}));
    },
    async close() { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); },
  };
}

export async function until(predicate, label) {
  const end = Date.now() + 20000;
  while (!predicate()) {
    if (Date.now() >= end) throw new Error('Timed out waiting for ' + label);
    await delay(20);
  }
}
