// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const adaptive = require('../adaptive.js');

function acgRuntimeConfig(provider = 'anthropic') {
  return {
    version: 1,
    agent_id: `node-adaptive-${provider}`,
    state: {
      backend: adaptive.inMemoryBackend(),
    },
    acg: adaptive.acgConfig({
      provider,
    }),
  };
}

describe('adaptive runtime bridge', () => {
  it('validates config through the native adaptive runtime', () => {
    assert.deepEqual(adaptive.validateConfig(adaptive.defaultConfig()).diagnostics, []);
  });

  it('builds cache telemetry events from one options object', () => {
    const event = adaptive.buildCacheTelemetryEvent({
      provider: 'openai',
      requestId: '00000000-0000-0000-0000-000000000201',
      usage: {
        prompt_tokens: 100,
        completion_tokens: 10,
        cache_read_tokens: 25,
      },
      agentId: 'node-agent',
      templateVersion: 'v1',
      toolsetHash: 'tools',
      modelFamily: 'gpt',
      tenantScope: 'tenant',
      timestamp: '2026-06-15T00:00:00Z',
    });

    assert.equal(event.provider, 'openai');
    assert.equal(event.request_id, '00000000-0000-0000-0000-000000000201');
    assert.equal(event.cache_read_tokens, 25);
    assert.equal(event.total_prompt_tokens, 100);
    assert.equal(event.hit_rate, 0.25);
    assert.equal(event.agent_identity.agent_id, 'node-agent');
  });

  it('returns null when cache telemetry lacks prompt tokens', () => {
    assert.equal(
      adaptive.buildCacheTelemetryEvent({
        provider: 'openai',
        requestId: '00000000-0000-0000-0000-000000000202',
        usage: {
          completion_tokens: 10,
        },
        agentId: 'node-agent',
        templateVersion: 'v1',
        toolsetHash: 'tools',
        modelFamily: 'gpt',
        tenantScope: 'tenant',
      }),
      null,
    );
  });

  it('registers an owned runtime and builds cache request facts', async () => {
    const runtime = new adaptive.AdaptiveRuntime(acgRuntimeConfig('openai'));
    await runtime.register();
    try {
      assert.deepEqual(runtime.report().diagnostics, []);
      const facts = runtime.buildCacheRequestFacts({
        provider: 'openai',
        requestId: '00000000-0000-0000-0000-000000000203',
        annotatedRequest: {
          messages: [
            {
              role: 'user',
              content: 'Find sources about caching',
            },
          ],
          model: 'gpt-4.1-mini',
        },
        agentId: 'node-adaptive-openai',
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

  it('rejects invalid latency sensitivity values', () => {
    assert.throws(() => adaptive.setLatencySensitivity(0), /positive/);
  });
});
