// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const plugin = require('../plugin.js');
const piiRedaction = require('../pii_redaction.js');

describe('pii_redaction plugin helpers', () => {
  it('builds defaults and plugin component shape', () => {
    assert.deepEqual(piiRedaction.defaultConfig(), {
      version: 1,
      mode: 'builtin',
      input: true,
      output: true,
      tool_input: true,
      tool_output: true,
      mark: true,
      priority: 100,
    });
    assert.deepEqual(piiRedaction.builtinConfig(), { action: 'remove' });
    assert.deepEqual(piiRedaction.localModelConfig(), {});
    assert.deepEqual(piiRedaction.profileConfig(), {
      enabled: true,
      mode: 'builtin',
      priority: 100,
    });

    const component = piiRedaction.ComponentSpec({
      ...piiRedaction.defaultConfig(),
      builtin: piiRedaction.builtinConfig({ detector: 'email' }),
    });
    assert.equal(component.kind, piiRedaction.PII_REDACTION_PLUGIN_KIND);
    assert.equal(component.enabled, true);
  });

  it('builds profile composition without legacy top-level fields', () => {
    const config = {
      version: 1,
      codec: 'openai_chat',
      profiles: [
        piiRedaction.profileConfig({
          builtin: piiRedaction.builtinConfig({ detector: 'email' }),
        }),
        piiRedaction.profileConfig({
          mode: 'local_model',
          priority: 110,
          local: piiRedaction.localModelConfig({
            backend: 'acme.pii/detector',
            target_paths: ['/message'],
          }),
        }),
      ],
    };

    assert.equal(config.mode, undefined);
    assert.deepEqual(plugin.validate({ version: 1, components: [piiRedaction.ComponentSpec(config)] }).diagnostics, []);
  });

  it('lists builtin pii_redaction kind and validates bad values', () => {
    assert.equal(plugin.listKinds().includes(piiRedaction.PII_REDACTION_PLUGIN_KIND), true);
    const report = plugin.validate({
      version: 1,
      components: [
        piiRedaction.ComponentSpec({
          ...piiRedaction.defaultConfig(),
          input: false,
          output: false,
          builtin: piiRedaction.builtinConfig({ action: 'mask', detector: 'not_a_detector' }),
        }),
      ],
    });
    assert.deepEqual(
      report.diagnostics.map((diagnostic) => diagnostic.field),
      ['builtin.detector'],
    );
  });
});
