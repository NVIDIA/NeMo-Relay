// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const {
  validatePluginConfig,
  layerPluginConfig,
  registerPlugin,
  deregisterPlugin,
  initializePlugins,
  clearPluginConfiguration,
  activePluginReport,
  listPluginKinds,
} = require('./nemo_relay_wasm.js');

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
 * Layer one plugin configuration over another.
 *
 * Objects merge recursively, arrays and scalar values are replaced by the
 * overlay, and top-level components merge by `kind`.
 *
 * @param {object} base - Lower-precedence plugin config, usually loaded from files.
 * @param {object} overlay - Higher-precedence plugin config, usually built in code.
 * @returns {object} The effective raw plugin config document.
 * @remarks Passing raw objects preserves omitted fields so they can inherit
 * from the base config.
 */
function layer(base, overlay) {
  return layerPluginConfig(base, overlay);
}

/**
 * Validate a plugin configuration without activating it.
 *
 * Runs the same config validation pipeline used by initialization while
 * leaving the active plugin registry and runtime configuration unchanged.
 *
 * @param {object} config - Candidate plugin configuration document.
 * @returns {object} A structured validation report with diagnostics.
 * @remarks Use this to surface warnings or incompatibilities before replacing
 * the active plugin configuration.
 */
function validate(config) {
  return validatePluginConfig(config);
}

/**
 * Validate and activate a plugin configuration.
 *
 * Replaces the current active config, invokes each enabled component's
 * registration hooks, and resolves with the final activation report.
 *
 * @param {object} config - Plugin configuration document to activate.
 * @returns {Promise<object>} A promise resolving to the activation report.
 * @remarks Partial plugin registration is rolled back if activation fails, and
 * the returned promise rejects with the underlying validation or setup error.
 */
function initialize(config) {
  return initializePlugins(config);
}

/**
 * Clear the active plugin configuration.
 *
 * Removes the currently active component registrations while leaving plugin
 * kinds in the registry so they can be reused by a later initialization call.
 *
 * @returns {void} Nothing.
 * @remarks Registered plugin kinds remain available after the active config is
 * cleared.
 */
function clear() {
  return clearPluginConfiguration();
}

/**
 * Return the last successfully activated plugin report.
 *
 * Exposes the most recent activation report emitted by the plugin system without
 * triggering validation or registration work.
 *
 * @returns {object|null|undefined} The last activation report, if one exists.
 * @remarks This returns an empty value until `initialize` succeeds at least
 * once in the current process.
 */
function report() {
  return activePluginReport();
}

/**
 * List registered plugin kinds.
 *
 * Returns the plugin kind identifiers currently known to the registry so
 * callers can inspect what can be referenced from plugin configs.
 *
 * @returns {string[]} The registered plugin kind names.
 * @remarks The list reflects registry state only; it does not indicate whether
 * a plugin kind is currently active in the runtime configuration.
 */
function listKinds() {
  return listPluginKinds();
}

/**
 * Register a plugin kind with JavaScript validation and registration hooks.
 *
 * Adapts the higher-level `Plugin` object contract to the lower-level callback
 * shape expected by the WebAssembly wrapper surface.
 *
 * @param {string} pluginKind - Unique plugin kind identifier to register.
 * @param {object} plugin - Plugin implementation with `validate` and `register` hooks.
 * @returns {void} Nothing.
 * @remarks Omitting `plugin.validate` makes the plugin permissive during
 * validation; `plugin.register` still runs later during `initialize`.
 */
function register(pluginKind, plugin) {
  return registerPlugin(
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
 * @remarks Active runtime registrations remain in place until `clear` or a
 * later successful `initialize` replaces them.
 */
function deregister(pluginKind) {
  return deregisterPlugin(pluginKind);
}

exports.defaultConfig = defaultConfig;
exports.ComponentSpec = ComponentSpec;
exports.layer = layer;
exports.validate = validate;
exports.initialize = initialize;
exports.clear = clear;
exports.report = report;
exports.listKinds = listKinds;
exports.register = register;
exports.deregister = deregister;
