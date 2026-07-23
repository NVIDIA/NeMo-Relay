// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');
const plugin = require('../plugin.js');
const adaptive = require('../adaptive.js');

describe('core plugins', () => {
  it('reports active config and lists registered plugin kinds', async () => {
    const pluginKind = `node.test.report.${Date.now()}`;

    plugin.register(pluginKind, {
      register() {},
    });

    try {
      assert.equal(plugin.report(), null);
      assert.equal(plugin.listKinds().includes(pluginKind), true);

      const report = await plugin.initialize({
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

      assert.deepEqual(plugin.report(), report);
    } finally {
      plugin.clear();
      plugin.deregister(pluginKind);
    }
  });

  it('routes validation diagnostics through a registered JS plugin', () => {
    const pluginKind = `node.test.validate.${Date.now()}`;

    plugin.register(pluginKind, {
      validate(pluginConfig) {
        return [
          {
            level: 'warning',
            code: 'plugin.node_validate',
            component: pluginKind,
            field: 'threshold',
            message: `threshold:${pluginConfig.threshold}`,
          },
        ];
      },
      register() {},
    });

    try {
      const report = plugin.validate(plugin.defaultConfig());
      const wrappedReport = plugin.validate({
        version: 1,
        components: [
          plugin.ComponentSpec(pluginKind, {
            threshold: 7,
          }),
        ],
      });

      assert.equal(report.diagnostics.length, 0);
      assert.equal(wrappedReport.diagnostics.length, 1);
      assert.equal(wrappedReport.diagnostics[0].code, 'plugin.node_validate');
      assert.equal(wrappedReport.diagnostics[0].field, 'threshold');
    } finally {
      assert.equal(plugin.deregister(pluginKind), true);
    }
  });

  it('treats implicit undefined plugin validation as no diagnostics', () => {
    const pluginKind = `node.test.validate_undefined.${Date.now()}`;

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
      assert.equal(plugin.deregister(pluginKind), true);
    }
  });

  it('invokes top-level plugin registration during plugin configuration', async () => {
    const pluginKind = `node.test.register.${Date.now()}`;
    let registerCalls = 0;
    let registerContext = null;

    plugin.register(pluginKind, {
      register(pluginConfig, context) {
        registerCalls += 1;
        assert.equal(pluginConfig.priority, 17);
        registerContext = {
          priority: pluginConfig.priority,
          hasSubscriber: typeof context.registerSubscriber === 'function',
          hasToolRequest: typeof context.registerToolRequestIntercept === 'function',
          hasLlmExecution: typeof context.registerLlmExecutionIntercept === 'function',
          hasLlmStreamExecution: typeof context.registerLlmStreamExecutionIntercept === 'function',
          hasMarkSanitize: typeof context.registerMarkSanitizeGuardrail === 'function',
          hasScopeStartSanitize: typeof context.registerScopeSanitizeStartGuardrail === 'function',
          hasScopeEndSanitize: typeof context.registerScopeSanitizeEndGuardrail === 'function',
        };
        context.registerSubscriber('subscriber', () => {});
        context.registerToolRequestIntercept('toolRequest', 17, false, (_name, args) => ({
          ...args,
          nodeToolPlugin: `priority:${pluginConfig.priority}`,
        }));
        context.registerLlmExecutionIntercept('llmExec', 17, async (request, next) => {
          const result = await next(request);
          return {
            ...result,
            nodeLlmPlugin: `priority:${pluginConfig.priority}`,
          };
        });
        context.registerLlmStreamExecutionIntercept('llmStreamExec', 17, async (request, next) => next(request));
      },
    });

    try {
      const report = await plugin.initialize({
        version: 1,
        components: [
          adaptive.ComponentSpec({
            version: 1,
            state: {
              backend: adaptive.inMemoryBackend(),
            },
            adaptive_hints: adaptive.adaptiveHintsConfig(),
          }),
          plugin.ComponentSpec(pluginKind, {
            priority: 17,
          }),
        ],
      });
      assert.deepEqual(report.diagnostics, []);
      assert.equal(registerCalls, 1);
      assert.deepEqual(registerContext, {
        priority: 17,
        hasSubscriber: true,
        hasToolRequest: true,
        hasLlmExecution: true,
        hasLlmStreamExecution: true,
        hasMarkSanitize: true,
        hasScopeStartSanitize: true,
        hasScopeEndSanitize: true,
      });
    } finally {
      plugin.clear();
      plugin.deregister(pluginKind);
    }
  });

  it('turns plugin request-intercept throws into catchable errors', async () => {
    const pluginKind = `node.test.request-throw.${Date.now()}`;
    plugin.register(pluginKind, {
      register(_config, context) {
        context.registerToolRequestIntercept('throwingRequest', 10, false, () => {
          throw new Error('plugin request intercept boom');
        });
      },
    });

    try {
      const report = await plugin.initialize({
        version: 1,
        components: [
          adaptive.ComponentSpec({
            version: 1,
            state: { backend: adaptive.inMemoryBackend() },
            adaptive_hints: adaptive.adaptiveHintsConfig(),
          }),
          plugin.ComponentSpec(pluginKind, {}),
        ],
      });
      assert.deepEqual(report.diagnostics, []);
      await assert.rejects(
        () => lib.toolCallExecute('plugin_request_throw', {}, () => ({ should_not: 'run' })),
        /plugin request intercept boom/i,
      );
    } finally {
      plugin.clear();
      plugin.deregister(pluginKind);
    }
  });
});

describe('adaptive helpers', () => {
  it('builds a redis backend with the default key prefix', () => {
    assert.deepEqual(adaptive.redisBackend('redis://127.0.0.1:6379'), {
      kind: 'redis',
      config: {
        url: 'redis://127.0.0.1:6379',
        key_prefix: 'nemo_relay:',
      },
    });
  });

  it('builds an acg config with nested stability-threshold defaults', () => {
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
    assert.deepEqual(
      adaptive.acgConfig({
        provider: 'openai',
        stability_thresholds: {
          stable_threshold: 0.99,
        },
      }),
      {
        provider: 'openai',
        observation_window: 100,
        priority: 50,
        stability_thresholds: {
          stable_threshold: 0.99,
          semi_stable_threshold: 0.5,
          min_observations_for_full_confidence: 20,
        },
      },
    );
  });

  it('keeps response-cache helpers camelCase and serializes plugin config', () => {
    const responseCache = adaptive.responseCacheConfig();
    assert.deepEqual(responseCache, {
      ttlSeconds: 3600,
      namespace: '',
      priority: 50,
      bypassRate: 0,
      cacheNondeterministic: true,
      keyStrategy: 'exact_request',
      headerAllowlist: [],
      skipKeys: [],
      backend: adaptive.inMemoryBackend(),
    });
    assert.deepEqual(adaptive.ComponentSpec({ version: 1, responseCache }).config, {
      version: 1,
      response_cache: {
        ttl_seconds: 3600,
        namespace: '',
        priority: 50,
        bypass_rate: 0,
        cache_nondeterministic: true,
        key_strategy: 'exact_request',
        header_allowlist: [],
        skip_keys: [],
        backend: adaptive.inMemoryBackend(),
      },
    });
  });

  it('serializes response-cache config at both native boundaries', () => {
    const config = { version: 1, responseCache: { ttlSeconds: 0 } };
    assert.equal(adaptive.validateConfig(config).diagnostics[0].code, 'response_cache.invalid_ttl');
    assert.throws(() => new adaptive.AdaptiveRuntime(config), /ttl_seconds must be greater than 0/);
  });
});
