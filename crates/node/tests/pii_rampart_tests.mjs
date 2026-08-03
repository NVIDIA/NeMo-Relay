// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const rampart = require('../pii_rampart.js');

describe('pii_rampart plugin helpers', () => {
  it('builds the independent component shape', () => {
    const config = rampart.defaultConfig('/models/rampart', {
      codec: 'openai_chat',
      target_path_patterns: ['/messages/*/content'],
    });
    assert.equal(config.model_path, '/models/rampart');
    assert.equal(config.max_windows_per_payload, 4);
    assert.equal(config.inference_batch_size, 16);
    assert.equal(config.custom_mark_payload_policy, 'preserve');
    assert.equal(rampart.RAMPART_MODEL_ID, 'nationaldesignstudio/rampart');
    assert.equal(rampart.RAMPART_MODEL_REVISION, 'b1993e4e68b082835b80ffc65acc03325ea2e501');
    const component = rampart.ComponentSpec(config);
    assert.equal(component.kind, rampart.RAMPART_PII_PLUGIN_KIND);
    assert.equal(component.enabled, true);
    assert.equal(rampart.ComponentSpec(config, { enabled: false }).enabled, false);
    assert.deepEqual(rampart.validateConfig(config).diagnostics, []);
  });

  it('requires one content selection mode in the config helper', () => {
    const exact = rampart.defaultConfig('/models/rampart', {
      target_paths: ['/message'],
    });
    assert.deepEqual(rampart.validateConfig(exact).diagnostics, []);

    assert.throws(
      () => rampart.defaultConfig('/models/rampart'),
      /requires preset, target_paths, or target_path_patterns/,
    );
    assert.throws(
      () => rampart.defaultConfig('/models/rampart', {}),
      /requires preset, target_paths, or target_path_patterns/,
    );
    assert.throws(
      () =>
        rampart.defaultConfig('/models/rampart', {
          target_paths: [],
          target_path_patterns: [],
        }),
      /requires preset, target_paths, or target_path_patterns/,
    );

    const preset = rampart.defaultConfig('/models/rampart', {
      preset: 'trajectory_context',
    });
    assert.deepEqual(rampart.validateConfig(preset).diagnostics, []);
    assert.throws(
      () =>
        rampart.defaultConfig('/models/rampart', {
          preset: 'trajectory_context',
          target_paths: ['/message'],
        }),
      /cannot combine preset with explicit target selectors/,
    );
    assert.equal(
      rampart.defaultConfig('/models/rampart', {
        model_path: '/unapproved/model',
        target_paths: ['/message'],
      }).model_path,
      '/models/rampart',
    );
  });

  it('is registered and validates malformed paths', () => {
    const plugin = require('../plugin.js');
    assert.equal(plugin.listKinds().includes(rampart.RAMPART_PII_PLUGIN_KIND), true);
    const report = rampart.validateConfig(
      rampart.defaultConfig('relative/model', {
        target_path_patterns: ['/messages/pre*fix/content'],
      }),
    );
    assert.deepEqual(
      new Set(report.diagnostics.map((diagnostic) => diagnostic.field)),
      new Set(['model_path', 'target_path_patterns']),
    );
  });
});
