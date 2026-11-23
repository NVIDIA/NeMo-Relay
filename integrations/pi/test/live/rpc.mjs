// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createServer } from 'node:http';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

export const extension = fileURLToPath(new URL('../../index.ts', import.meta.url));

/** Launch Pi's actual RPC mode. Environment paths, shell effects and the gateway are local to this case. */
export async function withRpc(pi, setup, run) {
  const root = await mkdtemp(join(tmpdir(), 'relay-pi-rpc-'));
  const agent = join(root, 'agent');
  await mkdir(agent);
  const forward = join(root, 'forward.mjs');
  const fixture = new URL('../fixtures/hook-forward.mjs', import.meta.url);
  await writeFile(forward, '#!/usr/bin/env node\nawait import(' + JSON.stringify(fixture.href) + ');\n', { mode: 0o755 });
  const posts = [];
  const requests = new Map();
  const server = createServer();
  let child;
  try {
    const { extensions = [], reply, env = {}, flags = [], onEvent } = await setup(root, agent, posts);
    server.on('request', async (request, response) => {
      let data = '';
      for await (const chunk of request) data += chunk;
      const body = JSON.parse(data);
      posts.push(body);
      const result = reply?.(body) ?? {};
      response.writeHead(result.status ?? 200, { 'content-type': 'application/json' });
      response.end(JSON.stringify(result.body ?? {}));
    });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    child = spawn(pi, ['--mode', 'rpc', '--no-session', '--no-extensions',
      ...extensions.flatMap((entry) => ['-e', entry]), ...flags], {
      cwd: root, stdio: ['pipe', 'pipe', 'pipe'],
      env: { PATH: process.env.PATH, HOME: root, PI_CODING_AGENT_DIR: agent, PI_OFFLINE: '1',
        NEMO_RELAY_PI_COMMAND: forward, NEMO_RELAY_PI_GATEWAY_URL: 'http://127.0.0.1:' + server.address().port,
        NEMO_RELAY_PI_REDIRECT: 'off', NEMO_RELAY_PI_FAIL: 'closed', ...env },
    });
    const closed = once(child, 'close');
    let stderr = '';
    child.stderr.on('data', (s) => { stderr = (stderr + s).slice(-2000); });
    const send = (message) => child.stdin.write(JSON.stringify(message) + '\n');
    let id = 0;
    const command = (type, fields = {}) => new Promise((resolve, reject) => {
      const key = String(++id);
      const timeout = setTimeout(() => reject(new Error('Pi RPC timeout: ' + type + '\n' + stderr)), 20000);
      requests.set(key, { resolve: (value) => { clearTimeout(timeout); resolve(value); },
        reject: (error) => { clearTimeout(timeout); reject(error); } });
      send({ type, id: key, ...fields });
    });
    child.on('close', () => {
      for (const pending of requests.values()) pending.reject(new Error('Pi RPC exited: ' + stderr));
      requests.clear();
    });
    const reader = createInterface({ input: child.stdout });
    reader.on('line', (line) => {
      const event = JSON.parse(line);
      if (event.type === 'response') {
        requests.get(event.id)?.resolve(event);
        requests.delete(event.id);
      }
      onEvent?.(event, send);
    });
    await run({ root, agent, posts, command, send });
    child.stdin.end();
    // Some Pi versions continue reading the terminal after the input pipe closes.
    child.kill();
    await closed;
  } finally {
    child?.kill();
    server.closeAllConnections();
    if (server.listening) await new Promise(resolve => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  }
}

export const deny = (reason) => ({ status: 403, body: { error: { type: 'nemo_relay_guardrail_rejected', reason } } });
