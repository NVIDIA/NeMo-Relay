// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  validatePluginConfig,
  registerPlugin,
  deregisterPlugin,
  initializePlugins,
  clearPluginConfiguration,
  activePluginReport,
  listPluginKinds,
} from './pkg/nemo_flow_wasm.js';

export function defaultConfig() {
  return { version: 1, components: [] };
}

export function ComponentSpec(kind, config = {}, { enabled = true } = {}) {
  return {
    kind,
    enabled,
    config,
  };
}

export function validate(config) {
  return validatePluginConfig(config);
}

export function initialize(config) {
  return initializePlugins(config);
}

export function clear() {
  return clearPluginConfiguration();
}

export function report() {
  return activePluginReport();
}

export function listKinds() {
  return listPluginKinds();
}

export function register(pluginKind, handler) {
  return registerPlugin(
    pluginKind,
    handler.validate ? (pluginConfig) => handler.validate(pluginConfig) : null,
    (pluginConfig, context) => handler.register(pluginConfig, context),
  );
}

export { deregisterPlugin as deregister };
