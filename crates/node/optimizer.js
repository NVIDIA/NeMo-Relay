// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const { createRequire } = require('module');
const path = require('path');

const nativeRequire = createRequire(path.join(__dirname, 'index.js'));
const lib = nativeRequire('./index.js');

function defaultConfig() {
  return { version: 1, components: [] };
}

function inMemoryBackend() {
  return { kind: 'in_memory', config: {} };
}

function redisBackend(url, keyPrefix = 'nemo_flow:') {
  return {
    kind: 'redis',
    config: { url, key_prefix: keyPrefix },
  };
}

function telemetryComponent(config = {}) {
  return { kind: 'telemetry', enabled: true, config };
}

function dynamoHintsComponent(config = {}) {
  return {
    kind: 'dynamo_hints',
    enabled: true,
    config: {
      priority: 100,
      break_chain: false,
      inject_header: true,
      inject_body_path: 'nvext.agent_hints',
      ...config,
    },
  };
}

function toolParallelismComponent(config = {}) {
  return {
    kind: 'tool_parallelism',
    enabled: true,
    config: {
      priority: 100,
      mode: 'observe_only',
      ...config,
    },
  };
}

function externalComponent(pluginKind, instanceId, pluginConfig = {}) {
  return {
    kind: 'external_component',
    enabled: true,
    config: {
      plugin_kind: pluginKind,
      instance_id: instanceId,
      plugin_config: pluginConfig,
    },
  };
}

function registerPlugin(pluginKind, handler) {
  return lib.registerOptimizerPlugin(
    pluginKind,
    handler.validate
      ? (instanceId, pluginConfig) => handler.validate(instanceId, pluginConfig)
      : null,
    (instanceId, pluginConfig, context) => handler.register(instanceId, pluginConfig, context),
  );
}

function deregisterPlugin(pluginKind) {
  return lib.deregisterOptimizerPlugin(pluginKind);
}

module.exports = {
  Runtime: lib.OptimizerRuntime,
  validateConfig: lib.validateOptimizerConfig,
  registerPlugin,
  deregisterPlugin,
  defaultConfig,
  inMemoryBackend,
  redisBackend,
  telemetryComponent,
  dynamoHintsComponent,
  toolParallelismComponent,
  externalComponent,
};
