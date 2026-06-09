// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as piiRedaction from '../pkg/pii_redaction.js';
import * as plugin from '../pkg/plugin.js';

test('WebAssembly pii_redaction wrappers expose helper defaults', () => {
  assert.deepEqual(piiRedaction.defaultConfig(), {
    version: 1,
    mode: 'builtin',
    input: true,
    output: true,
    tool_input: true,
    tool_output: true,
    priority: 100,
  });
  assert.deepEqual(piiRedaction.builtinConfig(), {
    action: 'redact',
  });
  assert.deepEqual(piiRedaction.localModelConfig(), {});
});

test('WebAssembly pii_redaction wrappers build component specs and validate bad values', () => {
  assert.equal(plugin.listKinds().includes(piiRedaction.PII_REDACTION_PLUGIN_KIND), true);

  const component = piiRedaction.ComponentSpec({
    ...piiRedaction.defaultConfig(),
    builtin: piiRedaction.builtinConfig({ detector: 'email' }),
  });

  assert.deepEqual(component, {
    kind: 'pii_redaction',
    enabled: true,
    config: {
      version: 1,
      mode: 'builtin',
      input: true,
      output: true,
      tool_input: true,
      tool_output: true,
      priority: 100,
      builtin: {
        action: 'redact',
        detector: 'email',
      },
    },
  });

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
  assert.deepEqual(report.diagnostics.map((diagnostic) => diagnostic.field), ['builtin.detector']);
});
