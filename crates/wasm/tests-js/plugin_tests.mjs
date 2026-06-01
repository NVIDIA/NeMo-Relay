// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as adaptive from '../pkg/adaptive.js';
import * as plugin from '../pkg/plugin.js';
import { unique } from './test_support.mjs';

test('WebAssembly plugin wrappers expose default config', () => {
  assert.deepEqual(plugin.defaultConfig(), {
    version: 1,
    components: [],
  });
});

test('WebAssembly plugin wrappers layer config documents', () => {
  // Smoke test only: merge semantics are covered by the core crate. This
  // verifies the wrapper forwards both documents and returns merged JSON.
  assert.deepEqual(plugin.layer({ a: 1 }, { b: 2 }), { a: 1, b: 2 });
});

test('WebAssembly plugin wrappers register and validate components', () => {
  const pluginKind = unique('wasm.wrapper.plugin');
  const validatedConfigs = [];

  plugin.register(pluginKind, {
    validate(config) {
      validatedConfigs.push(config);
      return [];
    },
    register() {},
  });

  try {
    assert.equal(plugin.listKinds().includes(pluginKind), true);
    assert.equal(plugin.report(), undefined);

    const report = plugin.validate({
      version: 1,
      components: [
        adaptive.ComponentSpec({
          version: 1,
          state: {
            backend: adaptive.inMemoryBackend(),
          },
          telemetry: adaptive.telemetryConfig({
            learners: ['latency_sensitivity'],
          }),
        }),
        plugin.ComponentSpec(pluginKind, {}),
      ],
    });
    assert.deepEqual(report.diagnostics, []);
    assert.deepEqual(validatedConfigs, [{}]);
  } finally {
    plugin.clear();
    plugin.deregister(pluginKind);
  }
});

test('WebAssembly plugin wrappers initialize components and report state', async () => {
  const pluginKind = unique('wasm.wrapper.plugin.init');
  plugin.register(pluginKind, {
    register() {},
  });

  try {
    const initialized = await plugin.initialize({
      version: 1,
      components: [
        adaptive.ComponentSpec({
          version: 1,
          state: {
            backend: adaptive.inMemoryBackend(),
          },
        }),
        plugin.ComponentSpec(pluginKind, {}),
      ],
    });
    assert.deepEqual(plugin.report(), initialized);
  } finally {
    plugin.clear();
    plugin.deregister(pluginKind);
  }
});

test('WebAssembly plugin wrappers treat implicit undefined validation as no diagnostics', () => {
  const pluginKind = unique('wasm.wrapper.validate_undefined');
  plugin.register(pluginKind, {
    validate() {},
    register() {},
  });

  try {
    const report = plugin.validate({
      version: 1,
      components: [plugin.ComponentSpec(pluginKind, {})],
    });
    assert.deepEqual(report.diagnostics, []);
  } finally {
    plugin.clear();
    plugin.deregister(pluginKind);
  }
});
