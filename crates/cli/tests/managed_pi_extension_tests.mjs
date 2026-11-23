// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { chmod, copyFile, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import test from 'node:test';

import {
  decideManagedProviderRedirect,
  decideManagedToolTransform,
  summarizeManagedToolResult,
} from '../src/daemon/managed/pi_extension/index.ts';

const CREDENTIAL = Buffer.alloc(32, 7).toString('base64url');

function requireReplace(source, pattern, replacement) {
  assert.equal(source.split(pattern).length, 2, `expected exactly one lifecycle rewrite: ${pattern}`);
  return source.replace(pattern, replacement);
}

async function loadLifecycleTestApi() {
  const sourceUrl = new URL('../src/daemon/managed/pi_extension/index.ts', import.meta.url);
  let source = await readFile(sourceUrl, 'utf8');
  const rewrites = [
    [
      'export default function managedNemoRelayPi(pi: ExtensionAPI): void {',
      'export default function managedNemoRelayPi(pi: ExtensionAPI, initialize = initializeRuntime): void {',
    ],
    ['runtimePromise ??= initializeRuntime();', 'runtimePromise ??= initialize();'],
    ['function createSharedLease(', 'export function createSharedLease('],
    [
      'const launched = spawn(config.dispatcherCommand, [',
      'const launched = spawn(process.execPath, [config.dispatcherCommand, ',
    ],
  ];
  for (const [pattern, replacement] of rewrites) source = requireReplace(source, pattern, replacement);
  source += '\nexport { managedNemoRelayPi as installManagedNemoRelayPi };\n';
  const directory = await mkdtemp(join(tmpdir(), 'nemo-relay-pi-module-'));
  const modulePath = join(directory, 'index.ts');
  await copyFile(new URL('./provider-transport.ts', sourceUrl), join(directory, 'provider-transport.ts'));
  await writeFile(modulePath, source);
  return import(pathToFileURL(modulePath).href);
}

test('managed Pi MCP lease becomes ready, restarts, and releases without another launch', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'nemo-relay-pi-lease-'));
  const launches = join(directory, 'launches');
  const dispatcher = join(directory, 'dispatcher.mjs');
  await writeFile(
    dispatcher,
    String.raw`#!/usr/bin/env node
import { appendFileSync, readFileSync } from 'node:fs';
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
  await chmod(dispatcher, 0o755);

  let removed = 0;
  const { createSharedLease } = await loadLifecycleTestApi();
  const lease = createSharedLease(
    { schema: 'nemo-relay-managed-pi-v1', daemonAddress: 'http://127.0.0.1:47632', dispatcherCommand: dispatcher },
    CREDENTIAL,
    () => { removed += 1; },
  );
  await lease.ensureReady();
  const deadline = Date.now() + 10_000;
  while ((await readFile(launches, 'utf8')).trim().split('\n').length < 2) {
    assert.ok(Date.now() < deadline, 'MCP child did not restart');
    await new Promise((resolve) => setTimeout(resolve, 10));
    await lease.ensureReady();
  }
  await lease.release();
  assert.equal((await readFile(launches, 'utf8')).trim().split('\n').length, 2);
  assert.equal(removed, 1);
  await assert.rejects(lease.ensureReady(), /released/);
  assert.equal((await readFile(launches, 'utf8')).trim().split('\n').length, 2);
});

test('managed Pi shutdown branches preserve ordering and quit releases without reinitializing', async () => {
  const handlers = new Map();
  const pi = {
    on(name, handler) { handlers.set(name, handler); },
    registerProvider() {},
  };
  let ensureReady = 0;
  let releases = 0;
  const runtime = {
    config: { schema: 'nemo-relay-managed-pi-v1', daemonAddress: 'https://relay.example.com', dispatcherCommand: '/opt/nemo-relay' },
    credential: CREDENTIAL,
    lease: {
      ensureReady: async () => { ensureReady += 1; },
      release: async () => { releases += 1; },
    },
  };
  const bodies = [];
  let startAttempts = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_url, request) => {
    const body = JSON.parse(request.body);
    if (body.hook_event_name === 'session_start' && ++startAttempts === 1) {
      return Response.json({ error: { message: 'route is not ready' } }, {
        status: 503, headers: { 'Retry-After': '1' },
      });
    }
    bodies.push(body);
    return new Response('{}', { status: 200 });
  };
  try {
    const { installManagedNemoRelayPi } = await loadLifecycleTestApi();
    installManagedNemoRelayPi(pi, async () => runtime);
    const context = { cwd: '/workspace', sessionManager: { getSessionId: () => 'session' } };
    await handlers.get('session_start')({ type: 'session_start', reason: 'startup' }, context);
    assert.equal(startAttempts, 2, 'session startup waits for the broker to accept the hook');
    await Promise.all([
      handlers.get('turn_start')({ type: 'turn_start', turnIndex: 0 }, context),
      handlers.get('turn_start')({ type: 'turn_start', turnIndex: 1 }, context),
    ]);
    await handlers.get('session_shutdown')({ type: 'session_shutdown', reason: 'reload' }, context);
    await handlers.get('session_shutdown')({ type: 'session_shutdown', reason: 'new' }, context);
    const beforeQuit = ensureReady;
    await handlers.get('session_shutdown')({ type: 'session_shutdown', reason: 'quit' }, context);

    assert.deepEqual(bodies.map((body) => body.hook_event_name), [
      'session_start', 'turn_start', 'turn_start', 'session_shutdown', 'session_shutdown',
    ]);
    assert.deepEqual(bodies.filter((body) => body.hook_event_name === 'turn_start').map((body) => body.turn_index), [0, 1]);
    assert.equal(ensureReady - beforeQuit, 0, 'quit forwards its final hook without restarting MCP');
    assert.equal(releases, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('managed Pi does not retry startup after the readiness deadline', async (t) => {
  const { installManagedNemoRelayPi } = await loadLifecycleTestApi();
  const handlers = new Map();
  installManagedNemoRelayPi({
    on(name, handler) { handlers.set(name, handler); },
    registerProvider() {},
  }, async () => ({
    config: { daemonAddress: 'https://relay.example.com' },
    credential: CREDENTIAL,
    lease: { ensureReady: async () => {} },
  }));
  let now = 0;
  t.mock.method(performance, 'now', () => now);
  const send = t.mock.method(globalThis, 'fetch', async () => Response.json(
    { error: { message: 'route is not ready' } },
    { status: 503, headers: { 'Retry-After': '1' } },
  ));
  t.mock.method(console, 'error', () => {});

  const starting = handlers.get('session_start')({ type: 'session_start', reason: 'startup' }, {
    cwd: '/workspace',
    sessionManager: { getSessionId: () => 'session' },
  });
  await new Promise((resolve) => globalThis.setImmediate(resolve));
  assert.equal(send.mock.callCount(), 1);

  // Simulate a retry timer that fires after the monotonic deadline.
  now = 180_001;
  await starting;
  assert.equal(send.mock.callCount(), 1);
});

test('custom Pi providers redirect only when every sibling API is supported', () => {
  const selected = {
    id: 'custom-response-model',
    api: 'openai-responses',
    provider: 'custom-enterprise',
    baseUrl: 'https://ignored-by-managed-policy.example/v1',
  };
  const serviceableCatalog = [
    selected,
    {
      id: 'custom-messages-model',
      api: 'anthropic-messages',
      provider: selected.provider,
      baseUrl: 'https://ignored-by-managed-policy.example/v1/',
    },
  ];

  assert.deepEqual(decideManagedProviderRedirect(selected, serviceableCatalog), {
    kind: 'redirect',
    upstream: selected.baseUrl,
    reason: 'provider uses only daemon-supported APIs and every model shares its endpoint',
  });
  assert.equal(decideManagedProviderRedirect(selected, undefined).kind, 'skip');
  assert.deepEqual(
    decideManagedProviderRedirect(selected, [
      ...serviceableCatalog,
      {
        id: 'custom-google-model',
        api: 'google-generative-ai',
        provider: selected.provider,
        baseUrl: 'https://google.example',
      },
    ]),
    {
      kind: 'skip',
      code: 'provider-mixed-apis',
      reason:
        'redirecting custom-enterprise would also move its unsupported ' +
        'google-generative-ai model custom-google-model',
    },
  );
  assert.deepEqual(
    decideManagedProviderRedirect(selected, [
      selected,
      {
        id: 'different-endpoint-model',
        api: 'openai-completions',
        provider: selected.provider,
        baseUrl: 'https://different.example/v1',
      },
    ]),
    {
      kind: 'skip',
      code: 'provider-mixed-endpoints',
      reason:
        'redirecting custom-enterprise would also move different-endpoint-model, which targets ' +
        'https://different.example/v1 rather than https://ignored-by-managed-policy.example/v1',
    },
  );
});

test('managed Pi preserves the caller on nested MCP rewrites and completions', async (t) => {
  const bodies = [];
  t.mock.method(globalThis, 'fetch', async (_url, request) => {
    const body = JSON.parse(request.body);
    bodies.push(body);
    const reply = body.hook_event_name === 'tool_call'
      ? { tool_call: { tool_call_id: body.tool_call_id, input: { query: 'public' } } }
      : {};
    return Response.json(reply);
  });
  const handlers = new Map();
  const { installManagedNemoRelayPi } = await loadLifecycleTestApi();
  installManagedNemoRelayPi({ on: (name, handler) => handlers.set(name, handler) }, async () => ({
    config: { daemonAddress: 'https://relay.example.com' },
    credential: CREDENTIAL,
    lease: { ensureReady: async () => {} },
  }));
  const context = { sessionManager: { getSessionId: () => 'session' } };
  const event = {
    toolCallId: 'outer/1', parentToolCallId: 'outer', toolName: 'mcp__docs__search',
    input: { query: 'private' },
  };
  assert.equal(await handlers.get('tool_call')(event, context), undefined);
  assert.deepEqual(event.input, { query: 'public' });
  await handlers.get('tool_execution_end')({ ...event, isError: false, result: 'ok' }, context);
  await handlers.get('session_shutdown')({ reason: 'reload' }, context);
  assert.deepEqual(bodies.map((body) => body.hook_event_name), [
    'tool_call', 'tool_arguments_transformed', 'tool_execution_end',
  ]);
  for (const body of bodies) {
    assert.equal(body.parent_tool_call_id, 'outer');
    assert.equal(body.tool_call_id, 'outer/1');
    assert.equal(body.tool_name, 'mcp__docs__search');
  }
});

test('managed Pi termination applies only to an explicit policy denial', async (t) => {
  const key = 'NEMO_RELAY_PI_TERMINATE_ON_DENY';
  const previous = process.env[key];
  t.after(() => {
    if (previous === undefined) delete process.env[key];
    else process.env[key] = previous;
  });
  let status = 403;
  t.mock.method(globalThis, 'fetch', async () => Response.json({
    error: { type: 'nemo_relay_guardrail_rejected', reason: 'fixture policy' },
  }, { status }));
  const handlers = new Map();
  const { installManagedNemoRelayPi } = await loadLifecycleTestApi();
  installManagedNemoRelayPi({ on: (name, handler) => handlers.set(name, handler) }, async () => ({
    config: { daemonAddress: 'https://relay.example.com' },
    credential: CREDENTIAL,
    lease: { ensureReady: async () => {} },
  }));
  const gate = () => handlers.get('tool_call')(
    { toolCallId: 'call', toolName: 'read', input: { path: 'fixture' } },
    { sessionManager: { getSessionId: () => 'session' } },
  );
  delete process.env[key];
  assert.deepEqual(await gate(), { block: true, reason: 'fixture policy' });
  process.env[key] = '1';
  assert.deepEqual(await gate(), { block: true, reason: 'fixture policy', terminate: true });
  status = 503;
  const fault = await gate();
  assert.equal(fault.block, true);
  assert.equal(fault.terminate, undefined);
});

test('managed Pi tool rewrites require the exact call ID and recursively preserve shape', () => {
  const current = { path: '/before', flags: [true, { retries: 2 }] };
  assert.deepEqual(
    decideManagedToolTransform(
      {
        tool_call: {
          tool_call_id: 'call-1',
          input: { path: '/after', flags: [false, { retries: 3 }] },
        },
      },
      'call-1',
      current,
    ),
    {
      kind: 'replace',
      input: { path: '/after', flags: [false, { retries: 3 }] },
    },
  );
  assert.equal(
    decideManagedToolTransform({ tool_call: { input: { path: '/after', flags: current.flags } } }, 'call-1', current)
      .kind,
    'invalid',
  );
  assert.equal(
    decideManagedToolTransform(
      {
        tool_call: {
          tool_call_id: 'call-1',
          input: { path: '/after', flags: [false, { retries: 3 }], extra: true },
        },
      },
      'call-1',
      current,
    ).kind,
    'invalid',
  );
});

test('managed Pi summaries preserve ordered text blocks and Unicode boundaries', () => {
  assert.deepEqual(
    summarizeManagedToolResult(
      {
        content: [
          { type: 'text', text: 'first' },
          { type: 'image', data: 'not-forwarded' },
          { type: 'text', text: 'second' },
        ],
      },
      false,
    ),
    { content: 'first\nsecond', result_keys: ['content'] },
  );

  const summary = summarizeManagedToolResult('x'.repeat(1_999) + '😀tail', false).content;
  assert.equal(typeof summary, 'string');
  assert.equal(summary.includes('�'), false);
  assert.equal(summary.includes('... [truncated 6 chars]'), true);
});
