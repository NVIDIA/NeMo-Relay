// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as adaptive from '../adaptive.js';
import * as plugin from '../plugin.js';
import {
  JsonPassthrough,
  typedLlmExecute,
  typedLlmStreamExecute,
  typedToolExecute,
} from '../typed.js';

test('WASM adaptive and plugin wrappers expose helper defaults', async () => {
  assert.deepEqual(adaptive.defaultConfig(), { version: 1 });
  assert.deepEqual(plugin.defaultConfig(), { version: 1, components: [] });
  assert.deepEqual(adaptive.inMemoryBackend(), { kind: 'in_memory', config: {} });
  assert.deepEqual(adaptive.redisBackend('redis://127.0.0.1:6379'), {
    kind: 'redis',
    config: { url: 'redis://127.0.0.1:6379', key_prefix: 'nemo_flow:' },
  });
  assert.deepEqual(adaptive.telemetryConfig({ learners: ['latency_sensitivity'] }), {
    learners: ['latency_sensitivity'],
  });
  assert.equal(adaptive.adaptiveHintsConfig().priority, 100);
  assert.equal(adaptive.toolParallelismConfig().mode, 'observe_only');
  assert.deepEqual(
    adaptive.ComponentSpec({ version: 1, state: { backend: adaptive.inMemoryBackend() } }),
    {
      kind: 'adaptive',
      enabled: true,
      config: { version: 1, state: { backend: { kind: 'in_memory', config: {} } } },
    },
  );

  const pluginKind = `wasm.wrapper.plugin.${Date.now()}`;
  const noValidateKind = `${pluginKind}.no_validate`;
  const validatedConfigs = [];
  plugin.register(pluginKind, {
    validate(config) {
      validatedConfigs.push(config);
      return [];
    },
    register() {},
  });
  plugin.register(noValidateKind, {
    register() {},
  });

  try {
    assert.equal(plugin.listKinds().includes(pluginKind), true);
    assert.equal(plugin.listKinds().includes(noValidateKind), true);
    assert.equal(plugin.report(), undefined);

    const report = plugin.validate({
      version: 1,
      components: [
        adaptive.ComponentSpec({
          version: 1,
          state: { backend: adaptive.inMemoryBackend() },
          telemetry: adaptive.telemetryConfig({ learners: ['latency_sensitivity'] }),
        }),
        plugin.ComponentSpec(pluginKind, {}),
      ],
    });
    assert.deepEqual(report.diagnostics, []);
    assert.deepEqual(validatedConfigs, [{}]);

    const initialized = await plugin.initialize({
      version: 1,
      components: [
        adaptive.ComponentSpec({
          version: 1,
          state: { backend: adaptive.inMemoryBackend() },
        }),
        plugin.ComponentSpec(pluginKind, {}),
      ],
    });
    assert.deepEqual(plugin.report(), initialized);
  } finally {
    plugin.clear();
    plugin.deregister(noValidateKind);
    plugin.deregister(pluginKind);
  }
});

test('WASM typed wrappers execute tool, llm, and stream flows', async () => {
  const passthrough = new JsonPassthrough();

  const toolResult = await typedToolExecute(
    'wrapper_tool',
    { value: 3 },
    (args) => ({ doubled: args.value * 2 }),
    passthrough,
    passthrough,
  );
  assert.deepEqual(toolResult, { doubled: 6 });

  const llmResult = await typedLlmExecute(
    'wrapper_llm',
    { headers: {}, content: { messages: [], model: 'test-model' } },
    () => ({ response: 'ok' }),
    passthrough,
  );
  assert.deepEqual(llmResult, { response: 'ok' });

  const seen = [];
  const stream = await typedLlmStreamExecute(
    'wrapper_stream',
    { headers: {}, content: { messages: [], model: 'test-model' } },
    async function* () {
      yield { token: 'hello' };
      yield { token: 'world' };
    },
    (chunk) => {
      seen.push(chunk);
    },
    () => ({ count: seen.length }),
    passthrough,
    passthrough,
  );

  const chunks = [];
  for (;;) {
    const next = await stream.next();
    if (next.done) {
      break;
    }
    chunks.push(next.value);
  }

  assert.deepEqual(chunks, [[{ token: 'hello' }, { token: 'world' }]]);
  assert.deepEqual(seen, chunks);

  const syncToolResult = await typedToolExecute(
    'wrapper_tool_sync',
    { value: 4 },
    (args) => ({ tripled: args.value * 3 }),
    passthrough,
    passthrough,
    { attributes: 1, data: { source: 'sync' }, metadata: { kind: 'tool' } },
  );
  assert.deepEqual(syncToolResult, { tripled: 12 });

  const asyncToolResult = await typedToolExecute(
    'wrapper_tool_async',
    { value: 5 },
    async (args) => ({ quadrupled: args.value * 4 }),
    passthrough,
    passthrough,
  );
  assert.deepEqual(asyncToolResult, { quadrupled: 20 });

  const syncLlmResult = await typedLlmExecute(
    'wrapper_llm_sync',
    { headers: {}, content: { messages: [], model: 'sync-model' } },
    () => ({ response: 'sync' }),
    passthrough,
    { attributes: 1, data: { source: 'sync' }, metadata: { kind: 'llm' }, modelName: 'sync-model' },
  );
  assert.deepEqual(syncLlmResult, { response: 'sync' });

  const asyncLlmResult = await typedLlmExecute(
    'wrapper_llm_async',
    { headers: {}, content: { messages: [], model: 'async-model' } },
    async () => ({ response: 'async' }),
    passthrough,
  );
  assert.deepEqual(asyncLlmResult, { response: 'async' });

  const streamWithoutHooks = await typedLlmStreamExecute(
    'wrapper_stream_no_hooks',
    { headers: {}, content: { messages: [], model: 'test-model' } },
    async function* () {
      yield { token: 'solo' };
    },
    null,
    null,
    passthrough,
    passthrough,
    { modelName: 'test-model' },
  );

  const noHookChunks = [];
  for (;;) {
    const next = await streamWithoutHooks.next();
    if (next.done) {
      break;
    }
    noHookChunks.push(next.value);
  }
  assert.deepEqual(noHookChunks, [[{ token: 'solo' }]]);
});
