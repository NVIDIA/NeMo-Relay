// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as adaptive from '../pkg/adaptive.js';

test('WebAssembly adaptive wrappers expose default config and helper defaults', () => {
  assert.deepEqual(adaptive.defaultConfig(), {
    version: 1,
  });
  assert.equal(adaptive.adaptiveHintsConfig().priority, 100);
  assert.equal(adaptive.toolParallelismConfig().mode, 'observe_only');
  assert.deepEqual(adaptive.acgConfig(), {
    provider: 'passthrough',
    observation_window: 100,
    priority: 50,
    stability_thresholds: {
      stable_threshold: 0.95,
      semi_stable_threshold: 0.5,
      min_observations_for_full_confidence: 20,
    },
  });
});

test('WebAssembly adaptive wrappers expose backend and telemetry helpers', () => {
  assert.deepEqual(adaptive.inMemoryBackend(), {
    kind: 'in_memory',
    config: {},
  });
  assert.deepEqual(adaptive.redisBackend('redis://127.0.0.1:6379'), {
    kind: 'redis',
    config: {
      url: 'redis://127.0.0.1:6379',
      key_prefix: 'nemo_flow:',
    },
  });
  assert.deepEqual(
    adaptive.telemetryConfig({
      learners: ['latency_sensitivity'],
    }),
    {
      learners: ['latency_sensitivity'],
    },
  );
});

test('WebAssembly adaptive wrappers build adaptive component specs', () => {
  assert.deepEqual(
    adaptive.ComponentSpec({
      version: 1,
      state: {
        backend: adaptive.inMemoryBackend(),
      },
      acg: adaptive.acgConfig({
        provider: 'anthropic',
      }),
    }),
    {
      kind: 'adaptive',
      enabled: true,
      config: {
        version: 1,
        state: {
          backend: {
            kind: 'in_memory',
            config: {},
          },
        },
        acg: {
          provider: 'anthropic',
          observation_window: 100,
          priority: 50,
          stability_thresholds: {
            stable_threshold: 0.95,
            semi_stable_threshold: 0.5,
            min_observations_for_full_confidence: 20,
          },
        },
      },
    },
  );
});
