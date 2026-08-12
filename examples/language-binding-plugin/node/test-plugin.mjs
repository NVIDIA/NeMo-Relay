// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { DEFAULT_CONFIG, config, documentationPlugin, isolateExampleEnvironment, plugin, relay } from './main.mjs';

function registeredCallbacks() {
  const callbacks = new Map();
  const context = new Proxy(
    {},
    {
      get(_target, method) {
        return (...args) => callbacks.set(method, args.at(-1));
      },
    },
  );
  documentationPlugin.register(structuredClone(DEFAULT_CONFIG), context);
  return callbacks;
}

async function withActivePlugin(run) {
  const restoreEnvironment = isolateExampleEnvironment();
  documentationPlugin.events.length = 0;
  try {
    plugin.register('documentation-plugin', documentationPlugin);
    const preflight = plugin.validate(config('enforce'));
    assert.deepEqual(preflight.diagnostics, []);
    const report = await plugin.initialize(config('enforce'));
    return await run(report);
  } finally {
    plugin.clear();
    plugin.deregister('documentation-plugin');
    restoreEnvironment();
  }
}

test('validation accepts a supported mode', () => {
  assert.deepEqual(documentationPlugin.validate({ requests: { mode: 'enforce' } }), []);
});

test('validation rejects an unsupported mode', () => {
  const diagnostics = documentationPlugin.validate({ requests: { mode: 'invalid' } });

  assert.equal(diagnostics[0].code, 'documentation-plugin.unsupported_mode');
});

test('validation rejects a wrong type', () => {
  const diagnostics = documentationPlugin.validate({ requests: { priority: 'high' } });

  assert.equal(diagnostics[0].code, 'documentation-plugin.invalid_config');
});

for (const [config, field, code] of [
  [{ tag: '' }, 'tag', 'documentation-plugin.invalid_tag'],
  [{ requests: { header_name: '' } }, 'requests.header_name', 'documentation-plugin.invalid_header'],
  [{ requests: { header_value: '' } }, 'requests.header_value', 'documentation-plugin.invalid_header'],
]) {
  test(`validation rejects an empty ${field}`, () => {
    const diagnostics = documentationPlugin.validate(config);

    assert.ok(diagnostics.some((diagnostic) => diagnostic.code === code && diagnostic.field === field));
  });
}

test('validation warns about an unknown field', () => {
  const diagnostics = documentationPlugin.validate({ unexpected: true });

  assert.equal(diagnostics[0].level, 'warning');
  assert.equal(diagnostics[0].field, 'unexpected');
});

test('activation reports no diagnostics', async () => {
  await withActivePlugin((report) => {
    assert.deepEqual(report.diagnostics, []);
  });
});

test('tool requests are rewritten', async () => {
  await withActivePlugin(async () => {
    const result = await relay.toolCallExecute('safe_tool', { value: 1 }, (args) => args);

    assert.deepEqual(result, { value: 1, plugin_tag: 'documentation' });
  });
});

test('tool policy blocks the configured tool', () => {
  const policy = registeredCallbacks().get('registerToolConditionalExecutionGuardrail');

  assert.equal(policy('dangerous_tool', { value: 1 }), "tool 'dangerous_tool' is blocked");
});

test('LLM requests are rewritten', async () => {
  await withActivePlugin(async () => {
    const result = await relay.llmCallExecute(
      'allowed-model',
      { headers: {}, content: { model: 'allowed-model' } },
      (request) => ({ headers: request.headers }),
    );

    assert.equal(result.headers['x-nemo-relay-plugin'], 'documentation');
  });
});

test('LLM policy blocks the configured model', () => {
  const policy = registeredCallbacks().get('registerLlmConditionalExecutionGuardrail');

  assert.equal(policy({ headers: {}, content: { model: 'restricted-model' } }), "model 'restricted-model' is blocked");
});

test('LLM stream chunks are transformed', async () => {
  const intercept = registeredCallbacks().get('registerLlmStreamExecutionIntercept');

  const chunks = await intercept({ headers: {}, content: { model: 'allowed-model' } }, async () => [
    { chunk: 1 },
    { chunk: 2 },
  ]);

  assert.deepEqual(chunks, [
    { chunk: 1, plugin_stream: true },
    { chunk: 2, plugin_stream: true },
  ]);
});

test('subscriber observes an emitted event', async () => {
  await withActivePlugin(async () => {
    relay.event('documentation-event', null, { emitted: true });
    await relay.flushSubscribers();

    assert.ok(documentationPlugin.events.includes('documentation-event'));
  });
});

test('teardown removes the plugin kind', async () => {
  const restoreEnvironment = isolateExampleEnvironment();
  plugin.register('documentation-plugin', documentationPlugin);
  try {
    await plugin.initialize(config('enforce'));
    plugin.clear();
    assert.equal(plugin.deregister('documentation-plugin'), true);
    assert.equal(plugin.listKinds().includes('documentation-plugin'), false);
  } finally {
    plugin.clear();
    plugin.deregister('documentation-plugin');
    restoreEnvironment();
  }
});
