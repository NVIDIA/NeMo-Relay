// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const plugin = require('../plugin.js');
const adaptive = require('../adaptive.js');

describe('core hosted plugins', () => {
  it('routes validation diagnostics through a registered JS plugin', () => {
    const pluginKind = `node.test.validate.${Date.now()}`;

    plugin.register(pluginKind, {
      validate(pluginConfig) {
        return [{
          level: 'warning',
          code: 'plugin.node_validate',
          component: pluginKind,
          field: 'threshold',
          message: `threshold:${pluginConfig.threshold}`,
        }];
      },
      register() {},
    });

    try {
      const report = plugin.validate(plugin.defaultConfig());
      const wrappedReport = plugin.validate({
        version: 1,
        components: [
          plugin.ComponentSpec(pluginKind, { threshold: 7 }),
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
        };
        context.registerSubscriber('subscriber', () => {});
        context.registerToolRequestIntercept(
          'toolRequest',
          17,
          false,
          (_name, args) => ({ ...args, nodeToolPlugin: `priority:${pluginConfig.priority}` }),
        );
        context.registerLlmExecutionIntercept(
          'llmExec',
          17,
          async (request, next) => {
            const result = await next(request);
            return { ...result, nodeLlmPlugin: `priority:${pluginConfig.priority}` };
          },
        );
        context.registerLlmStreamExecutionIntercept(
          'llmStreamExec',
          17,
          async (request, next) => next(request),
        );
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
          plugin.ComponentSpec(pluginKind, { priority: 17 }),
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
      });
    } finally {
      plugin.clear();
      plugin.deregister(pluginKind);
    }
  });
});
