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

  it('validates a registered JS plugin during asynchronous initialization', async () => {
    const pluginKind = `node.test.async_validate.${Date.now()}`;
    const config = {
      version: 1,
      components: [plugin.ComponentSpec(pluginKind, { marker: 'validated' })],
    };
    let validateCalls = 0;
    let registerCalls = 0;

    plugin.register(pluginKind, {
      validate(pluginConfig) {
        validateCalls += 1;
        assert.equal(pluginConfig.marker, 'validated');
        return [];
      },
      register(pluginConfig, context) {
        registerCalls += 1;
        context.registerToolRequestIntercept('marker', 1, false, (_name, args) => ({
          ...args,
          marker: pluginConfig.marker,
        }));
      },
    });

    try {
      assert.deepEqual(plugin.validate(config).diagnostics, []);
      assert.deepEqual((await plugin.initialize(config)).diagnostics, []);
      assert.equal(validateCalls, 2);
      assert.equal(registerCalls, 1);
      assert.deepEqual(await lib.toolCallExecute('validated_plugin_tool', {}, (args) => ({ result: args })), {
        result: { marker: 'validated' },
      });

      plugin.clear();
      assert.equal(plugin.deregister(pluginKind), true);
      assert.deepEqual(await lib.toolCallExecute('cleared_plugin_tool', {}, (args) => ({ result: args })), {
        result: {},
      });
    } finally {
      plugin.clear();
      plugin.deregister(pluginKind);
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

  it('reports throwing async plugin validation without terminating Node', async () => {
    const pluginKind = `node.test.async_validate_throw.${Date.now()}`;
    plugin.register(pluginKind, {
      validate() {
        throw new Error('plugin validation boom');
      },
      register() {
        assert.fail('registration must not run after validation fails');
      },
    });

    try {
      const config = {
        version: 1,
        components: [plugin.ComponentSpec(pluginKind, {})],
      };
      await assert.rejects(() => plugin.initialize(config), /plugin validation boom/);
      const report = plugin.validate(config);
      assert.equal(report.diagnostics.length, 1);
      assert.equal(report.diagnostics[0].code, 'plugin.validate_failed');
      assert.match(report.diagnostics[0].message, /plugin validation boom/);
      assert.deepEqual(await lib.toolCallExecute('validation_error_survives', {}, (args) => ({ result: args })), {
        result: {},
      });
    } finally {
      plugin.clear();
      plugin.deregister(pluginKind);
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
        () => lib.toolCallExecute('plugin_request_throw', {}, () => ({ result: { should_not: 'run' } })),
        /plugin request intercept boom/i,
      );
    } finally {
      plugin.clear();
      plugin.deregister(pluginKind);
    }
  });

  it('snapshotted plugin execution intercepts survive configuration teardown', async () => {
    const pluginKind = `node.test.execution-snapshot.${Date.now()}`;
    let blockerEntered;
    const entered = new Promise((resolve) => {
      blockerEntered = resolve;
    });
    let releaseBlocker;
    const release = new Promise((resolve) => {
      releaseBlocker = resolve;
    });

    plugin.register(pluginKind, {
      register(_config, context) {
        context.registerToolExecutionIntercept('target', 100, async (args, next) => {
          const downstream = await next(args);
          return {
            result: {
              ...downstream.result,
              snapshotted: true,
            },
            ...(downstream.annotation == null ? {} : { annotation: downstream.annotation }),
          };
        });
        context.registerToolExecutionIntercept('blocker', -100, async (args, next) => {
          blockerEntered();
          await release;
          return await next(args);
        });
      },
    });

    try {
      await plugin.initialize({
        version: 1,
        components: [
          plugin.ComponentSpec('observability', {
            version: 3,
            atof: { enabled: false },
          }),
          adaptive.ComponentSpec({
            version: 1,
            state: { backend: adaptive.inMemoryBackend() },
            adaptive_hints: adaptive.adaptiveHintsConfig(),
          }),
          plugin.ComponentSpec(pluginKind, {}),
        ],
      });
      const execution = lib.toolCallExecute('plugin_snapshot_tool', {}, () => ({
        result: { downstream: true },
      }));
      await entered;
      plugin.clear();
      releaseBlocker();
      assert.deepEqual(await execution, {
        result: {
          downstream: true,
          snapshotted: true,
        },
      });
    } finally {
      releaseBlocker();
      plugin.clear();
      plugin.deregister(pluginKind);
    }
  });
});

describe('plugin context conditional middleware guardrails', () => {
  it('fails open and is removed during clear', async () => {
    const suffix = `${process.pid}-${Date.now()}`;
    const pluginKind = `node.context_gate.${suffix}`;
    const target = `node-context-gate-target-${suffix}`;
    const observed = [];
    let guardrailCalls = 0;
    lib.registerSubscriber(target, (event) => observed.push(event.name));
    plugin.register(pluginKind, {
      register(_config, context) {
        context.registerConditionalMiddlewareGuardrail('failing-gate', ['subscriber'], target, () => {
          guardrailCalls += 1;
          throw new Error('expected gate failure');
        });
      },
    });
    try {
      await plugin.initialize({ version: 1, components: [plugin.ComponentSpec(pluginKind)] });
      lib.event('node-context-gate-fail-open', null, null);
      await lib.flushSubscribers();
      assert.equal(observed.at(-1), 'node-context-gate-fail-open');
      assert.ok(guardrailCalls > 0);
      plugin.clear();
      const callsBeforeClear = guardrailCalls;
      lib.event('node-context-gate-cleared', null, null);
      await lib.flushSubscribers();
      assert.equal(observed.at(-1), 'node-context-gate-cleared');
      assert.equal(guardrailCalls, callsBeforeClear);
    } finally {
      plugin.clear();
      plugin.deregister(pluginKind);
      lib.deregisterSubscriber(target);
    }
  });

  it('rolls back a gate when plugin registration throws', async () => {
    const suffix = `${process.pid}-${Date.now()}`;
    const pluginKind = `node.context_gate_rollback.${suffix}`;
    const target = `node-context-gate-rollback-target-${suffix}`;
    const observed = [];
    let guardrailCalls = 0;
    lib.registerSubscriber(target, (event) => observed.push(event.name));
    plugin.register(pluginKind, {
      register(_config, context) {
        context.registerConditionalMiddlewareGuardrail('rollback-gate', ['subscriber'], target, () => {
          guardrailCalls += 1;
          return 'activation in progress';
        });
        throw new Error('expected activation failure');
      },
    });
    try {
      await assert.rejects(
        () => plugin.initialize({ version: 1, components: [plugin.ComponentSpec(pluginKind)] }),
        /expected activation failure/,
      );
      const callsAfterFailure = guardrailCalls;
      lib.event('node-context-gate-rolled-back', null, null);
      await lib.flushSubscribers();
      assert.equal(observed.at(-1), 'node-context-gate-rolled-back');
      assert.equal(guardrailCalls, callsAfterFailure);
    } finally {
      plugin.clear();
      plugin.deregister(pluginKind);
      lib.deregisterSubscriber(target);
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
      cacheNondeterministic: false,
      keyStrategy: adaptive.ResponseCacheKeyStrategy.ExactRequest,
      headerAllowlist: [],
      backend: adaptive.inMemoryBackend(),
    });
    assert.deepEqual(adaptive.ComponentSpec({ version: 1, responseCache }).config, {
      version: 1,
      response_cache: {
        ttl_seconds: 3600,
        namespace: '',
        priority: 50,
        bypass_rate: 0,
        cache_nondeterministic: false,
        key_strategy: 'exact_request',
        header_allowlist: [],
        backend: adaptive.inMemoryBackend(),
      },
    });
  });

  it('serializes nested tool-cache config', () => {
    const spec = adaptive.ComponentSpec({
      version: 1,
      responseCache: {
        tools: {
          enabled: true,
          cacheErrors: true,
          default: { ttlSeconds: 30, bypassRate: 0.1, argSkip: ['trace'] },
          classes: { readOnly: { cacheable: true, toolVersion: 'class-v1', members: ['search'] } },
          overrides: { search: { toolVersion: 'v1', argSkip: ['requestId'] } },
        },
      },
    });
    assert.deepEqual(spec.config.response_cache.tools, {
      enabled: true,
      cache_errors: true,
      default: { ttl_seconds: 30, bypass_rate: 0.1, arg_skip: ['trace'] },
      classes: { readOnly: { cacheable: true, tool_version: 'class-v1', members: ['search'] } },
      overrides: { search: { tool_version: 'v1', arg_skip: ['requestId'] } },
    });
  });

  it('exports and serializes response-cache key strategy values', () => {
    assert.deepEqual(adaptive.ResponseCacheKeyStrategy, {
      ExactRequest: 'exact_request',
      Logical: 'logical',
    });
    const spec = adaptive.ComponentSpec({
      version: 1,
      responseCache: {
        keyStrategy: adaptive.ResponseCacheKeyStrategy.Logical,
      },
    });
    assert.equal(spec.config.response_cache.key_strategy, 'logical');
  });

  it('serializes response-cache config at both native boundaries', () => {
    const unscoped = adaptive.validateConfig({ version: 1, responseCache: {} });
    assert.ok(unscoped.diagnostics.some(({ code }) => code === 'response_cache.missing_namespace'));

    const config = {
      version: 1,
      responseCache: { ttlSeconds: 0, namespace: 'node-test' },
    };
    assert.equal(adaptive.validateConfig(config).diagnostics[0].code, 'response_cache.invalid_ttl');
    assert.throws(() => new adaptive.AdaptiveRuntime(config), /ttl_seconds must be greater than 0/);
  });
});
