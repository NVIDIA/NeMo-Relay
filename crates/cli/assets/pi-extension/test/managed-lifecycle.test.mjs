// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it, mock } from 'node:test';

import { managedWiring } from '../managed.ts';
import { createExtension } from '../src/extension.ts';
import { createSharedLease } from '../src/managed/mcp-lease.ts';
import { postDaemonHook } from '../src/managed/transport.ts';

const CREDENTIAL = Buffer.alloc(32, 7).toString('base64url');
const DAEMON = 'https://relay.example.com';

const spawnWithNode = (command, args, options) => spawn(process.execPath, [command, ...args], options);

function fakePi() {
  const handlers = new Map();
  const registrations = [];
  return {
    pi: {
      on(name, handler) {
        handlers.set(name, handler);
      },
      registerProvider(name, config) {
        registrations.push({ name, config });
      },
    },
    fire: (name, event, context) => handlers.get(name)({ type: name, ...event }, context),
    registrations,
  };
}

function countingLease() {
  const counts = { ensureReady: 0, releases: 0 };
  return {
    counts,
    lease: {
      ensureReady: async () => {
        counts.ensureReady += 1;
      },
      release: async () => {
        counts.releases += 1;
      },
    },
  };
}

async function withRecordedFetch(run) {
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, request) => {
    requests.push({ url: String(url), headers: request.headers, body: JSON.parse(request.body) });
    return new Response('{}', { status: 200 });
  };
  try {
    return await run(requests);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

describe('managed pi MCP lease', () => {
  it('becomes ready, restarts, and releases without another launch', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'nemo-relay-pi-lease-'));
    const launches = join(directory, 'launches');
    const dispatcher = join(directory, 'dispatcher.mjs');
    await writeFile(
      dispatcher,
      String.raw`import { appendFileSync, readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
appendFileSync(${JSON.stringify(launches)}, 'launch\n');
const launchCount = readFileSync(${JSON.stringify(launches)}, 'utf8').trim().split('\n').length;
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
lines.on('line', (line) => {
  const request = JSON.parse(line);
  if (request.method === 'initialize') {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: request.id, result: {
      protocolVersion: '2025-11-25', capabilities: {}, serverInfo: { name: 'nemo-relay', version: 'test' }
    } }) + '\n');
  } else if (request.method === 'notifications/initialized' && launchCount === 1) {
    setTimeout(() => process.exit(0), 10);
  }
});
`,
    );

    let removed = 0;
    let child;
    const lease = createSharedLease(
      {
        schema: 'nemo-relay-managed-pi-v1',
        daemonAddress: 'http://127.0.0.1:47632',
        dispatcherCommand: dispatcher,
      },
      CREDENTIAL,
      () => {
        removed += 1;
      },
      (command, args, options) => (child = spawnWithNode(command, args, options)),
    );
    const launchCount = async () => (await readFile(launches, 'utf8')).trim().split('\n').length;

    try {
      await lease.ensureReady();
      const deadline = Date.now() + 10_000;
      while ((await launchCount()) < 2) {
        assert.ok(Date.now() < deadline, 'MCP child did not restart');
        await new Promise((resolve) => setTimeout(resolve, 10));
        await lease.ensureReady();
      }
      await lease.release();
      assert.equal(await launchCount(), 2);
      assert.equal(removed, 1);
      await assert.rejects(lease.ensureReady(), /released/);
      assert.equal(await launchCount(), 2);
    } finally {
      child?.kill('SIGKILL');
      await rm(directory, { recursive: true, force: true });
    }
  });

  it('kills a dispatcher that ignores stdin close and SIGTERM before releasing', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'nemo-relay-pi-lease-'));
    const dispatcher = join(directory, 'dispatcher.mjs');
    await writeFile(
      dispatcher,
      String.raw`import { createInterface } from 'node:readline';
process.on('SIGTERM', () => {});
setInterval(() => {}, 1000);
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
lines.on('line', (line) => {
  const request = JSON.parse(line);
  if (request.method === 'initialize') {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: request.id, result: {
      protocolVersion: '2025-11-25', capabilities: {}, serverInfo: { name: 'nemo-relay', version: 'test' }
    } }) + '\n');
  }
});
`,
    );
    let child;
    const lease = createSharedLease(
      { schema: 'nemo-relay-managed-pi-v1', daemonAddress: 'http://127.0.0.1:47632', dispatcherCommand: dispatcher },
      CREDENTIAL,
      () => {},
      (command, args, options) => (child = spawnWithNode(command, args, options)),
      { readyMs: 5_000, releaseMs: 100, killMs: 100 },
    );
    try {
      await lease.ensureReady();
      await lease.release();
      assert.ok(child.exitCode !== null || child.signalCode !== null, 'release must not return with the child alive');
      if (process.platform !== 'win32') assert.equal(child.signalCode, 'SIGKILL');
    } finally {
      child?.kill('SIGKILL');
      await rm(directory, { recursive: true, force: true });
    }
  });
});

describe('managed pi transport', () => {
  it('retries a route that is not ready yet within the readiness budget', async () => {
    const originalFetch = globalThis.fetch;
    let calls = 0;
    globalThis.fetch = async () => {
      calls += 1;
      if (calls === 1) {
        return new Response(JSON.stringify({ error: { message: 'route is not ready' } }), {
          status: 503,
          headers: { 'retry-after': '0' },
        });
      }
      return new Response('{}', { status: 200 });
    };
    try {
      const outcome = await postDaemonHook(
        { daemonAddress: DAEMON, credential: CREDENTIAL },
        { hook_event_name: 'turn_start' },
      );
      assert.equal(outcome.kind, 'allow');
      assert.equal(calls, 2);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('gives up on a route that stays not ready once the budget is spent', async () => {
    const originalFetch = globalThis.fetch;
    let calls = 0;
    globalThis.fetch = async () => {
      calls += 1;
      return new Response(JSON.stringify({ error: { message: 'route is not ready' } }), {
        status: 503,
        headers: { 'retry-after': '0' },
      });
    };
    try {
      const outcome = await postDaemonHook(
        { daemonAddress: DAEMON, credential: CREDENTIAL },
        { hook_event_name: 'turn_start' },
        0,
      );
      assert.equal(outcome.kind, 'fault');
      assert.match(outcome.detail, /HTTP 503/);
      assert.equal(calls, 1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('reports a payload that cannot be serialized as a fault instead of throwing', async () => {
    const originalFetch = globalThis.fetch;
    let fetched = false;
    globalThis.fetch = async () => {
      fetched = true;
      return new Response('{}', { status: 200 });
    };
    try {
      const outcome = await postDaemonHook(
        { daemonAddress: DAEMON, credential: CREDENTIAL },
        {
          hook_event_name: 'tool_call',
          input: {
            toJSON() {
              throw null;
            },
          },
        },
      );
      assert.equal(outcome.kind, 'fault');
      assert.equal(outcome.origin, 'handler');
      assert.match(outcome.detail, /JSON-safe/);
      assert.equal(fetched, false);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

describe('managed pi wiring', () => {
  const context = (model, catalog) => ({
    cwd: '/workspace',
    model,
    modelRegistry: { getAll: () => catalog, getRegisteredNativeProvider: () => undefined },
    sessionManager: { getSessionId: () => 'session' },
  });

  it('keeps hook order across session replacement and releases the lease only on quit', async () => {
    const { pi, fire } = fakePi();
    const { lease, counts } = countingLease();
    const runtime = {
      config: {
        schema: 'nemo-relay-managed-pi-v1',
        daemonAddress: DAEMON,
        dispatcherCommand: '/opt/nemo-relay',
      },
      credential: CREDENTIAL,
      lease,
    };
    await withRecordedFetch(async (requests) => {
      createExtension(pi, { onFault: 'closed', bind: async () => managedWiring(runtime) });
      const ctx = context(undefined, []);
      await fire('session_start', { reason: 'startup' }, ctx);
      await Promise.all([
        fire('turn_start', { turnIndex: 0, timestamp: 1 }, ctx),
        fire('turn_start', { turnIndex: 1, timestamp: 2 }, ctx),
      ]);
      await fire('session_shutdown', { reason: 'reload' }, ctx);
      await fire('session_shutdown', { reason: 'new' }, ctx);
      const beforeQuit = counts.ensureReady;
      await fire('session_shutdown', { reason: 'quit' }, ctx);

      assert.deepEqual(
        requests.map((request) => request.body.hook_event_name),
        ['session_start', 'turn_start', 'turn_start', 'session_shutdown', 'session_shutdown'],
      );
      assert.deepEqual(
        requests
          .filter((request) => request.body.hook_event_name === 'turn_start')
          .map((request) => request.body.turn_index),
        [0, 1],
      );
      for (const request of requests) {
        assert.equal(request.url, `${DAEMON}/hooks/pi`);
        assert.equal(request.headers['x-nemo-relay-client-token'], CREDENTIAL);
        assert.equal(request.body.session_id, 'session');
      }
      assert.equal(counts.ensureReady - beforeQuit, 0, 'quit posts its final hook without restarting MCP');
      assert.equal(counts.releases, 1);
    });
  });

  it('posts the quit hook once when the route is gone, and still releases the lease', async () => {
    const { pi, fire } = fakePi();
    const { lease, counts } = countingLease();
    const runtime = {
      config: { schema: 'nemo-relay-managed-pi-v1', daemonAddress: DAEMON, dispatcherCommand: '/opt/nemo-relay' },
      credential: CREDENTIAL,
      lease,
    };
    const originalFetch = globalThis.fetch;
    let quitPosts = 0;
    globalThis.fetch = async (_url, request) => {
      if (JSON.parse(request.body).hook_event_name === 'session_shutdown') {
        quitPosts += 1;
        return new Response(JSON.stringify({ error: { message: 'route is not ready' } }), {
          status: 503,
          headers: { 'retry-after': '0' },
        });
      }
      return new Response('{}', { status: 200 });
    };
    const spy = mock.method(console, 'error', () => {});
    try {
      createExtension(pi, { onFault: 'closed', bind: async () => managedWiring(runtime) });
      const ctx = context(undefined, []);
      await fire('session_start', { reason: 'startup' }, ctx);
      await fire('session_shutdown', { reason: 'quit' }, ctx);
      assert.equal(quitPosts, 1, 'shutdown must not wait out the readiness budget');
      assert.equal(counts.releases, 1);
    } finally {
      spy.mock.restore();
      globalThis.fetch = originalFetch;
    }
  });

  it('redirects the selected provider through the daemon, naming its endpoint', async () => {
    const { pi, fire, registrations } = fakePi();
    const { lease } = countingLease();
    const runtime = {
      config: {
        schema: 'nemo-relay-managed-pi-v1',
        daemonAddress: DAEMON,
        dispatcherCommand: '/opt/nemo-relay',
      },
      credential: CREDENTIAL,
      lease,
    };
    const selected = {
      id: 'custom-response-model',
      api: 'openai-responses',
      provider: 'custom-enterprise',
      baseUrl: 'https://llm.example/v1',
    };
    const sibling = {
      id: 'custom-messages-model',
      api: 'anthropic-messages',
      provider: 'custom-enterprise',
      baseUrl: 'https://llm.example/v1/',
    };
    await withRecordedFetch(async (requests) => {
      createExtension(pi, { onFault: 'closed', bind: async () => managedWiring(runtime) });
      const ctx = context(selected, [selected, sibling]);
      await fire('session_start', { reason: 'startup' }, ctx);
      await fire('session_shutdown', { reason: 'quit' }, ctx);

      assert.deepEqual(registrations, [
        {
          name: 'custom-enterprise',
          config: {
            baseUrl: DAEMON,
            headers: {
              'x-nemo-relay-client-token': CREDENTIAL,
              'x-nemo-relay-upstream-base-url': 'https://llm.example/v1',
            },
          },
        },
      ]);
      const [redirect] = requests.filter((request) => request.body.hook_event_name === 'model_redirect');
      assert.equal(redirect.body.outcome, 'redirect');
      assert.equal(redirect.body.provider, 'custom-enterprise');
      assert.equal(redirect.body.upstream, 'https://llm.example/v1');
    });
  });

  it('registers a provider only once the MCP route is ready, and retries after a failed start', async () => {
    const { pi, fire, registrations } = fakePi();
    let ready = false;
    let attempts = 0;
    const lease = {
      ensureReady: async () => {
        attempts += 1;
        if (!ready) throw new Error('dispatcher exited before initialization');
      },
      release: async () => {},
    };
    const runtime = {
      config: { schema: 'nemo-relay-managed-pi-v1', daemonAddress: DAEMON, dispatcherCommand: '/opt/nemo-relay' },
      credential: CREDENTIAL,
      lease,
    };
    const selected = { id: 'm', api: 'openai-responses', provider: 'custom', baseUrl: 'https://llm.example/v1' };
    const errors = [];
    const spy = mock.method(console, 'error', (line) => errors.push(String(line)));
    try {
      await withRecordedFetch(async (requests) => {
        createExtension(pi, { onFault: 'closed', bind: async () => managedWiring(runtime) });
        const ctx = context(selected, [selected]);
        await fire('session_start', { reason: 'startup' }, ctx);
        assert.equal(registrations.length, 0, 'no registration without a ready route');
        assert.equal(attempts, 1, 'startup spends one readiness attempt, not two');
        assert.ok(errors.some((line) => /session_start hook was not accepted/.test(line)));

        ready = true;
        await fire('model_select', { model: selected }, ctx);
        assert.equal(registrations.length, 1, 'a later selection retries the registration');
        await fire('session_shutdown', { reason: 'quit' }, ctx);
        assert.deepEqual(
          requests.map((request) => request.body.hook_event_name),
          ['session_start', 'model_redirect', 'session_shutdown'],
          'the skipped session_start is delivered before the provider is registered',
        );
      });
    } finally {
      spy.mock.restore();
    }
  });

  it('recovers a rejected startup at the next turn without a model change', async () => {
    const { pi, fire, registrations } = fakePi();
    const { lease } = countingLease();
    const runtime = {
      config: { schema: 'nemo-relay-managed-pi-v1', daemonAddress: DAEMON, dispatcherCommand: '/opt/nemo-relay' },
      credential: CREDENTIAL,
      lease,
    };
    const selected = { id: 'm', api: 'openai-responses', provider: 'custom', baseUrl: 'https://llm.example/v1' };
    const seen = [];
    const originalFetch = globalThis.fetch;
    let rejectStart = true;
    globalThis.fetch = async (_url, request) => {
      const body = JSON.parse(request.body);
      seen.push(body.hook_event_name);
      if (body.hook_event_name === 'session_start' && rejectStart) {
        rejectStart = false;
        return new Response(JSON.stringify({ error: { message: 'daemon draining' } }), { status: 503 });
      }
      return new Response('{}', { status: 200 });
    };
    const spy = mock.method(console, 'error', () => {});
    try {
      createExtension(pi, { onFault: 'closed', bind: async () => managedWiring(runtime) });
      const ctx = context(selected, [selected]);
      await fire('session_start', { reason: 'startup' }, ctx);
      assert.equal(registrations.length, 0);
      await fire('turn_start', { turnIndex: 0, timestamp: 1 }, ctx);
      await fire('session_shutdown', { reason: 'quit' }, ctx);
      assert.equal(registrations.length, 1, 'the provider is registered before the first model request');
      assert.deepEqual(seen, ['session_start', 'session_start', 'model_redirect', 'turn_start', 'session_shutdown']);
    } finally {
      spy.mock.restore();
      globalThis.fetch = originalFetch;
    }
  });

  it('fails closed with the configuration error when the runtime cannot bind', async () => {
    const { pi, fire } = fakePi();
    createExtension(pi, {
      onFault: 'closed',
      bind: async () => {
        throw new Error('managed pi configuration is unreadable: ENOENT');
      },
    });
    const ctx = context(undefined, []);
    const result = await fire('tool_call', { toolCallId: 'c1', toolName: 'read', input: { path: 'README.md' } }, ctx);
    assert.equal(result.block, true);
    assert.match(result.reason, /could not load its configuration/);
    assert.match(result.reason, /ENOENT/);
  });
});
