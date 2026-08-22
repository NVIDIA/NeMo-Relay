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

async function withActivePlugin(run, pluginConfig = config('enforce')) {
  const restoreEnvironment = isolateExampleEnvironment();
  documentationPlugin.events.length = 0;
  try {
    plugin.register('documentation-plugin', documentationPlugin);
    const preflight = plugin.validate(pluginConfig);
    assert.deepEqual(preflight.diagnostics, []);
    const report = await plugin.initialize(pluginConfig);
    return await run(report);
  } finally {
    // Scope-end sanitizers run in Relay's queued publication path. Drain that
    // work before removing the callbacks that the active component owns.
    await relay.flushSubscribers();
    plugin.clear();
    plugin.deregister('documentation-plugin');
    restoreEnvironment();
  }
}

test('validation accepts a supported mode', () => {
  assert.deepEqual(documentationPlugin.validate({ requests: { mode: 'enforce' } }), []);
});

test('default registration control is disabled and valid', () => {
  assert.equal(DEFAULT_CONFIG.registration_control.enabled, false);
  assert.deepEqual(documentationPlugin.validate(structuredClone(DEFAULT_CONFIG)), []);
});

test('validation rejects an unsupported mode', () => {
  const diagnostics = documentationPlugin.validate({ requests: { mode: 'invalid' } });

  assert.equal(diagnostics[0].code, 'documentation-plugin.unsupported_mode');
});

test('validation rejects a wrong type', () => {
  const diagnostics = documentationPlugin.validate({ requests: { priority: 'high' } });

  assert.equal(diagnostics[0].code, 'documentation-plugin.invalid_config');
});

test('validation reports a non-object configuration', () => {
  const diagnostics = documentationPlugin.validate(null);

  assert.equal(diagnostics[0].code, 'documentation-plugin.invalid_config');
  assert.equal(diagnostics[0].field, undefined);
});

for (const [config, field, code] of [
  [{ tag: '' }, 'tag', 'documentation-plugin.invalid_tag'],
  [{ requests: { header_name: '' } }, 'requests.header_name', 'documentation-plugin.invalid_header'],
  [{ requests: { header_value: '' } }, 'requests.header_value', 'documentation-plugin.invalid_header'],
  [{ registration_control: { kinds: [] } }, 'registration_control.kinds', 'documentation-plugin.invalid_config'],
  [
    { registration_control: { registration_name: '' } },
    'registration_control.registration_name',
    'documentation-plugin.invalid_config',
  ],
  [{ registration_control: { reason: '' } }, 'registration_control.reason', 'documentation-plugin.invalid_config'],
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

test('disabled invalid configuration is still validated', () => {
  const restoreEnvironment = isolateExampleEnvironment();
  plugin.register('documentation-plugin', documentationPlugin);
  try {
    const report = plugin.validate(config('invalid', false));

    assert.equal(report.diagnostics[0].code, 'documentation-plugin.unsupported_mode');
  } finally {
    plugin.deregister('documentation-plugin');
    restoreEnvironment();
  }
});

test('registers every safe plugin surface', () => {
  const registrations = registeredCallbacks();

  assert.deepEqual([...registrations.keys()].sort(), [
    'registerLlmConditionalExecutionGuardrail',
    'registerLlmExecutionIntercept',
    'registerLlmRequestIntercept',
    'registerLlmSanitizeRequestGuardrail',
    'registerLlmSanitizeResponseGuardrail',
    'registerLlmStreamExecutionIntercept',
    'registerMarkSanitizeGuardrail',
    'registerScopeSanitizeEndGuardrail',
    'registerScopeSanitizeStartGuardrail',
    'registerSubscriber',
    'registerToolConditionalExecutionGuardrail',
    'registerToolExecutionIntercept',
    'registerToolRequestIntercept',
    'registerToolSanitizeRequestGuardrail',
    'registerToolSanitizeResponseGuardrail',
  ]);

  const enabled = structuredClone(DEFAULT_CONFIG);
  enabled.registration_control.enabled = true;
  const methods = new Set();
  documentationPlugin.register(enabled, new Proxy({}, { get: (_target, method) => () => methods.add(method) }));
  assert.ok(methods.has('registerConditionalMiddlewareGuardrail'));
});

test('registration control is owned by activation', async () => {
  const target = 'documentation-controlled-subscriber';
  const observed = [];
  const controlled = config('enforce');
  controlled.components[0].config.registration_control.enabled = true;
  relay.registerSubscriber(target, (event) => observed.push(event.name));
  try {
    const before = relay.listRuntimeRegistrations(['subscriber']);
    assert.ok(before.some((registration) => registration.effectiveName === target));
    await withActivePlugin(async () => {
      const baseline = observed.length;
      relay.event('registration-control-active', null, null);
      await relay.flushSubscribers();
      assert.equal(observed.length, baseline);
    }, controlled);
    relay.event('registration-control-cleared', null, null);
    await relay.flushSubscribers();
    assert.equal(observed.at(-1), 'registration-control-cleared');
  } finally {
    relay.deregisterSubscriber(target);
  }
});

test('activation reports no diagnostics', async () => {
  await withActivePlugin((report) => {
    assert.deepEqual(report.diagnostics, []);
  });
});

test('tool requests are rewritten', async () => {
  await withActivePlugin(async () => {
    const result = await relay.toolCallExecute('safe_tool', { value: 1 }, (args) => ({
      result: args,
      annotation: { source: 'application' },
    }));

    assert.deepEqual(result, {
      result: { value: 1, plugin_tag: 'documentation' },
      annotation: { source: 'application' },
    });
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

  const transformed = intercept({ headers: {}, content: { model: 'allowed-model' } }, async () =>
    (async function* () {
      yield { chunk: 1 };
      yield { chunk: 2 };
    })(),
  );
  const chunks = [];
  for await (const chunk of transformed) chunks.push(chunk);

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

test('runtime controls emit a mark and an isolated scope only when enabled', async () => {
  await withActivePlugin(async () => {
    await relay.toolCallExecute('safe_tool', { value: 1 }, (args) => ({ result: args }));
    await relay.flushSubscribers();

    assert.ok(documentationPlugin.events.includes('documentation-plugin.request'));
    assert.ok(documentationPlugin.events.includes('documentation-plugin.isolated'));
  });

  const runtimeDisabled = config('enforce');
  runtimeDisabled.components[0].config.runtime = { emit_marks: false, emit_isolated_scope: false };
  await withActivePlugin(async () => {
    await relay.toolCallExecute('safe_tool', { value: 1 }, (args) => ({ result: args }));
    await relay.flushSubscribers();

    assert.ok(!documentationPlugin.events.includes('documentation-plugin.request'));
    assert.ok(!documentationPlugin.events.includes('documentation-plugin.isolated'));
  }, runtimeDisabled);
});

test('runtime controls do not depend on request rewriting', async () => {
  const requestsDisabled = config('enforce');
  requestsDisabled.components[0].config.requests.enabled = false;
  await withActivePlugin(async () => {
    await relay.toolCallExecute('safe_tool', { value: 1 }, (args) => ({ result: args }));
    await relay.flushSubscribers();

    assert.ok(documentationPlugin.events.includes('documentation-plugin.request'));
  }, requestsDisabled);
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

test('registration rejects a duplicate kind and missing deregistration is false', () => {
  const restoreEnvironment = isolateExampleEnvironment();
  plugin.register('documentation-plugin', documentationPlugin);
  try {
    assert.throws(() => plugin.register('documentation-plugin', documentationPlugin));
    assert.equal(plugin.deregister('missing-documentation-plugin'), false);
  } finally {
    plugin.clear();
    plugin.deregister('documentation-plugin');
    restoreEnvironment();
  }
});
