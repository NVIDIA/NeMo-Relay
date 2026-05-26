// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as observability from '../pkg/observability.js';
import * as plugin from '../pkg/plugin.js';

test('WebAssembly observability wrappers expose helper defaults', () => {
  assert.deepEqual(observability.defaultConfig(), {
    version: 1,
  });
  assert.deepEqual(observability.atofConfig(), {
    enabled: false,
    mode: 'append',
  });
  assert.deepEqual(observability.atifConfig(), {
    enabled: false,
    agent_name: 'NeMo Relay',
    model_name: 'unknown',
    filename_template: 'nemo-relay-atif-{session_id}.json',
    subagent_export_mode: 'embedded',
  });
  assert.deepEqual(observability.otlpConfig(), {
    enabled: false,
    transport: 'http_binary',
    headers: {},
    resource_attributes: {},
    service_name: 'nemo-relay',
    timeout_millis: 3000,
  });
});

test('WebAssembly observability wrappers build component specs and validate file sinks', () => {
  assert.equal(plugin.listKinds().includes(observability.OBSERVABILITY_PLUGIN_KIND), true);

  const component = observability.ComponentSpec({
    version: 1,
    atof: observability.atofConfig({ enabled: true }),
    atif: observability.atifConfig({ enabled: true }),
  });

  assert.deepEqual(component, {
    kind: 'observability',
    enabled: true,
    config: {
      version: 1,
      atof: {
        enabled: true,
        mode: 'append',
      },
      atif: {
        enabled: true,
        agent_name: 'NeMo Relay',
        model_name: 'unknown',
        filename_template: 'nemo-relay-atif-{session_id}.json',
        subagent_export_mode: 'embedded',
      },
    },
  });

  const report = plugin.validate({
    version: 1,
    components: [component],
  });
  assert.deepEqual(report.diagnostics.map((diagnostic) => [diagnostic.component, diagnostic.field]).sort(), [
    ['atif', 'enabled'],
    ['atof', 'enabled'],
  ]);
});
