// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  OptimizerRuntime as Runtime,
  validateOptimizerConfig as validateConfig,
  registerOptimizerPlugin,
  deregisterOptimizerPlugin,
} from './pkg/nvidia_nat_nexus_wasm.js';

export { Runtime, validateConfig };

export function defaultConfig() {
  return { version: 1, components: [] };
}

export function inMemoryBackend() {
  return { kind: 'in_memory', config: {} };
}

export function redisBackend(url, keyPrefix = 'nexus:') {
  return {
    kind: 'redis',
    config: { url, key_prefix: keyPrefix },
  };
}

export function telemetryComponent(config = {}) {
  return { kind: 'telemetry', enabled: true, config };
}

export function dynamoHintsComponent(config = {}) {
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

export function toolParallelismComponent(config = {}) {
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

export function externalComponent(pluginKind, instanceId, pluginConfig = {}) {
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

export function registerPlugin(pluginKind, handler) {
  return registerOptimizerPlugin(
    pluginKind,
    handler.validate ?? null,
    (instanceId, pluginConfig, context) => handler.register(instanceId, pluginConfig, context),
  );
}

export function deregisterPlugin(pluginKind) {
  return deregisterOptimizerPlugin(pluginKind);
}
