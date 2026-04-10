// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as plugin from './plugin.js';

export const ADAPTIVE_PLUGIN_KIND = 'adaptive';

export function defaultConfig() {
  return { version: 1 };
}

export function inMemoryBackend() {
  return { kind: 'in_memory', config: {} };
}

export function redisBackend(url, keyPrefix = 'nemo_flow:') {
  return {
    kind: 'redis',
    config: { url, key_prefix: keyPrefix },
  };
}

export function telemetryConfig(config = {}) {
  return {
    learners: [],
    ...config,
  };
}

export function adaptiveHintsConfig(config = {}) {
  return {
    priority: 100,
    break_chain: false,
    inject_header: true,
    inject_body_path: 'nvext.agent_hints',
    ...config,
  };
}

export function toolParallelismConfig(config = {}) {
  return {
    priority: 100,
    mode: 'observe_only',
    ...config,
  };
}

export function ComponentSpec(config, { enabled = true } = {}) {
  return plugin.ComponentSpec(ADAPTIVE_PLUGIN_KIND, config, { enabled });
}
