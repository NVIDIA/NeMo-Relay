// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const plugin = require('../plugin.js');

describe('plugin export activation hooks', () => {
  for (const [decision, expected] of [
    ['allow', 1],
    ['deny', 0],
  ]) {
    it(`lets one plugin ${decision} its own exporter`, async () => {
      const kind = `tests.node_export_activation_${decision}`;
      let activations = 0;
      plugin.register(kind, {
        register(_config, context) {
          context.registerExportActivationPolicy(async (request) => {
            assert.deepEqual(request, {
              target_kind: 'tests.telemetry.otlp',
              config: { country: 'US' },
            });
            return decision;
          });
          context.registerExportTarget(
            {
              id: 'self-otel',
              targetKind: 'tests.telemetry.otlp',
              activationPolicy: {
                provider: kind,
                timeout_millis: 30000,
                config: { country: 'US' },
              },
            },
            async () => {
              activations += 1;
            },
          );
        },
      });
      try {
        await plugin.initialize({
          version: 1,
          components: [{ kind, enabled: true, config: {} }],
        });
        assert.equal(activations, expected);
      } finally {
        plugin.clear();
        plugin.deregister(kind);
      }
    });
  }
});
