// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');
const {
  defaultOptimizerConfig,
  optimizerInMemoryBackend,
  externalComponent,
  registerOptimizerPlugin,
  deregisterOptimizerPlugin,
} = require('../typed.js');

describe('optimizer hosted plugins', () => {
  it('routes validation diagnostics through a registered JS plugin', () => {
    const pluginKind = `node.test.validate.${Date.now()}`;

    registerOptimizerPlugin(pluginKind, {
      validate(instanceId, pluginConfig) {
        return [{
          level: 'warning',
          code: 'optimizer.node_plugin_validate',
          component: 'external_component',
          field: 'plugin_config.threshold',
          message: `${instanceId}:${pluginConfig.threshold}`,
        }];
      },
      register() {},
    });

    try {
      const report = lib.validateOptimizerConfig({
        version: 1,
        components: [
          externalComponent(pluginKind, 'node-plugin-validate', { threshold: 7 }),
        ],
      });

      assert.equal(report.diagnostics.length, 1);
      assert.equal(report.diagnostics[0].code, 'optimizer.node_plugin_validate');
      assert.equal(report.diagnostics[0].field, 'plugin_config.threshold');
    } finally {
      assert.equal(deregisterOptimizerPlugin(pluginKind), true);
    }
  });

  it('invokes hosted plugin registration during optimizer runtime registration', async () => {
    const pluginKind = `node.test.register.${Date.now()}`;
    let registerCalls = 0;
    let registerContext = null;

    registerOptimizerPlugin(pluginKind, {
      register(instanceId, pluginConfig, context) {
        registerCalls += 1;
        assert.equal(instanceId, 'node-plugin-register');
        assert.equal(pluginConfig.priority, 17);
        registerContext = {
          instanceId,
          priority: pluginConfig.priority,
          hasSubscriber: typeof context.registerSubscriber === 'function',
          hasToolRequest: typeof context.registerToolRequestIntercept === 'function',
          hasLlmExecution: typeof context.registerLlmExecutionIntercept === 'function',
          hasLlmStreamExecution: typeof context.registerLlmStreamExecutionIntercept === 'function',
        };
        context.registerSubscriber(`${instanceId}.subscriber`, () => {});
        context.registerToolRequestIntercept(
          `${instanceId}.toolRequest`,
          17,
          false,
          (_name, args) => ({ ...args, nodeToolPlugin: `${instanceId}:${pluginConfig.priority}` }),
        );
        context.registerLlmExecutionIntercept(
          `${instanceId}.llmExec`,
          17,
          async (request, next) => {
            const result = await next(request);
            return { ...result, nodeLlmPlugin: `${instanceId}:${pluginConfig.priority}` };
          },
        );
        context.registerLlmStreamExecutionIntercept(
          `${instanceId}.llmStreamExec`,
          17,
          async (request, next) => next(request),
        );
      },
    });

    const config = defaultOptimizerConfig();
    config.state = { backend: optimizerInMemoryBackend() };
    config.components = [
      externalComponent(pluginKind, 'node-plugin-register', { priority: 17 }),
    ];

    const runtime = new lib.OptimizerRuntime(config);
    try {
      assert.deepEqual((await runtime.report()).diagnostics, []);
      await runtime.register();
      assert.equal(registerCalls, 1);
      assert.deepEqual(registerContext, {
        instanceId: 'node-plugin-register',
        priority: 17,
        hasSubscriber: true,
        hasToolRequest: true,
        hasLlmExecution: true,
        hasLlmStreamExecution: true,
      });

      await runtime.deregister();
      await runtime.shutdown();
    } finally {
      deregisterOptimizerPlugin(pluginKind);
    }
  });
});
