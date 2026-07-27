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
    assert.equal(config.inference_batch_size, 16);
    assert.equal(rampart.RAMPART_MODEL_ID, 'nationaldesignstudio/rampart');
    assert.equal(rampart.RAMPART_MODEL_REVISION, 'b1993e4e68b082835b80ffc65acc03325ea2e501');
    const component = rampart.ComponentSpec(config);
    assert.equal(component.kind, rampart.RAMPART_PII_PLUGIN_KIND);
    assert.equal(component.enabled, true);
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
