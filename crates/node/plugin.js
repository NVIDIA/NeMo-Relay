// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const { createRequire } = require('node:module');
const path = require('node:path');

const nativeRequire = createRequire(path.join(__dirname, 'index.js'));
const lib = nativeRequire('./index.js');

/**
 * Create an empty plugin configuration.
 *
 * Returns the canonical top-level config shape with `version = 1` and no
 * configured components so callers can build a document incrementally before
 * validating or activating it.
 *
 * @returns {object} A new plugin config object.
 * @remarks Mutating the returned object does not affect runtime state until it
 * is passed to `initialize`.
 */
function defaultConfig() {
  return {
    version: 1,
    components: [],
  };
}

/**
 * Create a plugin component entry for a plugin config document.
 *
 * Packages a plugin kind, component-local config, and enablement flag into the
 * object shape expected by `PluginConfig.components`.
 *
 * @param {string} kind - Registered plugin kind to reference.
 * @param {object} [config={}] - Component-local config passed to plugin hooks.
 * @param {{ enabled?: boolean }} [options={}] - Optional component-level flags.
 * @returns {object} A component spec ready to insert into a plugin config.
 * @remarks Setting `enabled` to `false` preserves the component for validation
 * while skipping runtime registration during `initialize`.
 */
function ComponentSpec(kind, config = {}, { enabled = true } = {}) {
  return {
    kind,
    enabled,
    config,
  };
}

/**
 * Initialize the core-owned static and dynamic plugin host.
 *
 * Resolves programmatic config with either an explicit or discovered user
 * file, then the system configuration, and activates one owned lifetime.
 *
 * @param {object} config - Lowest-precedence programmatic configuration.
 * @param {string} [additionalPluginsToml] - Optional explicit `plugins.toml` layer.
 * @returns {Promise<object>} An owned activation with the unified host report.
 * @remarks Keep the returned activation alive while callbacks may run and call
 * `close()` or use `await using` for deterministic teardown.
 */
function initialize(config, additionalPluginsToml) {
  return lib.initialize(config, additionalPluginsToml);
}

/**
 * Validate the plugin host without loading plugin code.
 *
 * Resolves the same layered configuration and trust policy used by activation
 * while leaving the process-wide host lease untouched.
 *
 * @param {object} config - Lowest-precedence programmatic configuration.
 * @param {string} [additionalPluginsToml] - Optional explicit `plugins.toml` layer.
 * @returns {PluginHostReport} Structured static and dynamic validation report.
 * @remarks Validation performs no activation and does not acquire the host lease.
 */
function validate(config, additionalPluginsToml) {
  return lib.validate(config, additionalPluginsToml);
}

/**
 * Validate only the supplied static plugin configuration.
 *
 * Unlike `validate`, this does not discover or merge `plugins.toml` files.
 * Use it for component-specific validation when `config` is the complete
 * document to check.
 *
 * @param {object} config - Complete static plugin configuration.
 * @returns {PluginHostReport} Static validation results with no dynamic plugins.
 * @remarks This validates only the supplied document and intentionally skips
 * plugin discovery from the filesystem.
 */
function validateExact(config) {
  return lib.validateExact(config);
}

/**
 * List registered plugin kinds.
 *
 * Returns the plugin kind identifiers currently known to the global registry
 * so callers can inspect what can be referenced from plugin configs.
 *
 * @returns {string[]} The registered plugin kind names.
 * @remarks The list reflects registry state only; it does not indicate whether
 * a plugin kind is currently active in the runtime configuration.
 */
function listKinds() {
  return lib.listPluginKinds();
}

/**
 * Register a plugin kind with JavaScript validation and registration hooks.
 *
 * Adapts the higher-level `Plugin` object contract to the native callback
 * shape expected by the Node binding.
 *
 * @param {string} pluginKind - Unique plugin kind identifier to register.
 * @param {object} plugin - Plugin implementation with `validate` and `register` hooks.
 * @returns {void} Nothing.
 * @remarks Omitting `plugin.validate` makes the plugin permissive during
 * validation; `plugin.register` is still required and runs later during
 * `initialize`.
 */
function register(pluginKind, plugin) {
  return lib.registerPlugin(
    pluginKind,
    plugin.validate ? (pluginConfig) => plugin.validate(pluginConfig) : null,
    (pluginConfig, context) => plugin.register(pluginConfig, context),
  );
}

/**
 * Remove a previously registered plugin kind.
 *
 * Deletes the plugin kind from the registry so future config validation and
 * initialization calls can no longer reference it.
 *
 * @param {string} pluginKind - Registered plugin kind identifier to remove.
 * @returns {boolean} `true` when a plugin kind was removed, otherwise `false`.
 * @remarks Active runtime registrations remain in place until the owning
 * plugin-host activation closes.
 */
function deregister(pluginKind) {
  return lib.deregisterPlugin(pluginKind);
}

module.exports = {
  defaultConfig,
  ComponentSpec,
  initialize,
  validate,
  validateExact,
  listKinds,
  register,
  deregister,
};
