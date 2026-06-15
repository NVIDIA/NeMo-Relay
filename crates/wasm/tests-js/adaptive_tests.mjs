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
      key_prefix: 'nemo_relay:',
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

test('WebAssembly adaptive wrappers validate config through the native runtime', () => {
  assert.deepEqual(adaptive.validateConfig(adaptive.defaultConfig()).diagnostics, []);
});

test('WebAssembly adaptive wrappers build cache telemetry events from options', () => {
  const event = adaptive.buildCacheTelemetryEvent({
    provider: 'openai',
    requestId: '00000000-0000-0000-0000-000000000301',
    usage: {
      prompt_tokens: 100,
      completion_tokens: 10,
      cache_read_tokens: 25,
    },
    agentId: 'wasm-agent',
    templateVersion: 'v1',
    toolsetHash: 'tools',
    modelFamily: 'gpt',
    tenantScope: 'tenant',
    timestamp: '2026-06-15T00:00:00Z',
  });

  assert.equal(event.provider, 'openai');
  assert.equal(event.cache_read_tokens, 25);
  assert.equal(event.total_prompt_tokens, 100);
  assert.equal(event.hit_rate, 0.25);
  assert.equal(event.agent_identity.agent_id, 'wasm-agent');
});

test('WebAssembly adaptive runtime registers and builds cache request facts', async () => {
  const runtime = new adaptive.AdaptiveRuntime({
    version: 1,
    agent_id: 'wasm-adaptive-openai',
    state: {
      backend: adaptive.inMemoryBackend(),
    },
    acg: adaptive.acgConfig({
      provider: 'openai',
    }),
  });

  await runtime.register();
  try {
    assert.deepEqual(runtime.report().diagnostics, []);
    const facts = runtime.buildCacheRequestFacts({
      provider: 'openai',
      requestId: '00000000-0000-0000-0000-000000000302',
      annotatedRequest: {
        messages: [
          {
            role: 'user',
            content: 'Find sources about caching',
          },
        ],
        model: 'gpt-4.1-mini',
      },
      agentId: 'wasm-adaptive-openai',
    });

    assert.deepEqual(facts, {
      missing_facts: ['acg_stability_unavailable'],
      provider: 'openai',
      stable_prefix_length: 0,
    });
    runtime.waitForIdle();
    runtime.deregister();
  } finally {
    await runtime.shutdown().catch(() => {});
  }
});

test('WebAssembly adaptive wrappers reject invalid latency sensitivity values', () => {
  assert.throws(() => adaptive.setLatencySensitivity(0), /positive/);
});
