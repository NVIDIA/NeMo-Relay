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

function ComponentSpec(kind, config = {}, { enabled = true } = {}) {
  return {
    kind,
    enabled,
    config,
  };
}

function validate(config) {
  return lib.validatePluginConfig(config);
}

function initialize(config) {
  return lib.initializePlugins(config);
}

function clear() {
  return lib.clearPluginConfiguration();
}

function report() {
  return lib.activePluginReport();
}

function listKinds() {
  return lib.listPluginKinds();
}

function register(pluginKind, handler) {
  return lib.registerPlugin(
    pluginKind,
    handler.validate ? (pluginConfig) => handler.validate(pluginConfig) : null,
    (pluginConfig, context) => handler.register(pluginConfig, context),
  );
}

function deregister(pluginKind) {
  return lib.deregisterPlugin(pluginKind);
}

module.exports = {
  defaultConfig,
  ComponentSpec,
  validate,
  initialize,
  clear,
  report,
  listKinds,
  register,
  deregister,
};
