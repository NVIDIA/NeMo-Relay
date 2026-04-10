// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const plugin = require('./plugin.js');

const ADAPTIVE_PLUGIN_KIND = 'adaptive';

function defaultConfig() {
  return { version: 1 };
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

function telemetryConfig(config = {}) {
  return {
    learners: [],
    ...config,
  };
}

function adaptiveHintsConfig(config = {}) {
  return {
    priority: 100,
    break_chain: false,
    inject_header: true,
    inject_body_path: 'nvext.agent_hints',
    ...config,
  };
}

function toolParallelismConfig(config = {}) {
  return {
    priority: 100,
    mode: 'observe_only',
    ...config,
  };
}

function ComponentSpec(config, { enabled = true } = {}) {
  return plugin.ComponentSpec(ADAPTIVE_PLUGIN_KIND, config, { enabled });
}

module.exports = {
  ADAPTIVE_PLUGIN_KIND,
  defaultConfig,
  inMemoryBackend,
  redisBackend,
  telemetryConfig,
  adaptiveHintsConfig,
  toolParallelismConfig,
  ComponentSpec,
};
