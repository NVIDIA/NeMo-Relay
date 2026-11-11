// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// <reference lib="esnext.disposable" />

import type {
  ConfigDiagnostic,
  JsonObject,
  JsonObjectCompatible,
  PluginContext,
  PluginHostActivation,
  PluginHostReport,
  ResourceMetricsConfig,
} from '../binding.js';
import type { AdaptivePluginConfig } from './adaptive.js';
import * as lib from './index.js';
import type { PricingConfig } from './model_pricing.js';
import type { Config as ObservabilityConfig } from './observability.js';
import type { Config as PiiRedactionConfig } from './pii_redaction.js';

export type {
  ConfigDiagnostic,
  ConfigReport,
  DynamicPluginCheckState,
  DynamicPluginFailure,
  DynamicPluginKind,
  DynamicPluginValidationReport,
  DynamicPluginValidationStatus,
  EventMetadata,
  EventMetadataScalar,
  EventMetadataValue,
  LlmCodecIdentity,
  LlmExecutionContext,
  LlmOptimizationContribution,
  LlmOptimizationDataSchema,
  LlmOptimizationModel,
  LlmOptimizationModelTransition,
  LlmOptimizationTokenImpact,
  LlmOptimizationTokens,
  LlmRequestContext,
  LlmRequestInterceptOutcome,
  LlmResponseContext,
  LlmSanitizeRequestContext,
  LlmSanitizeResponseContext,
  PluginContext,
  PluginHostActivation,
  PluginHostReport,
  RuntimeDiagnostic,
  ToolExecutionInterceptOutcome,
} from '../binding.js';

/** How unsupported configuration is handled. */
export type UnsupportedBehavior = 'ignore' | 'warn' | 'error';

/** Handling for unknown or unsupported plugin configuration. */
export interface ConfigPolicy {
  unknown_component?: UnsupportedBehavior;
  unknown_field?: UnsupportedBehavior;
  unsupported_value?: UnsupportedBehavior;
}

export interface ComponentConfigs {
  adaptive: AdaptivePluginConfig;
  observability: ObservabilityConfig;
  pii_redaction: PiiRedactionConfig;
  pricing: PricingConfig;
  resource_metrics: ResourceMetricsConfig;
}

export type ComponentKind = keyof ComponentConfigs;

export type KnownComponentSpec = {
  [K in ComponentKind]: { kind: K; enabled?: boolean; config?: ComponentConfigs[K] };
}[ComponentKind];

declare const openComponent: unique symbol;

/** Config returned by `ComponentSpec` for a kind registered with `registerPlugin`. */
export type OpenComponentConfig = {
  readonly [openComponent]: true;
};

/** Component created by `ComponentSpec` for a kind registered with `registerPlugin`. */
export interface OpenComponentSpec<C extends object = JsonObject> {
  kind: string;
  enabled?: boolean;
  config?: C & OpenComponentConfig;
}

export type ComponentSpecFor<K extends string> = K extends ComponentKind
  ? Extract<KnownComponentSpec, { kind: K }>
  : OpenComponentSpec;

/** One plugin component. */
export type ComponentSpec = KnownComponentSpec | OpenComponentSpec<object>;

/** Plugin configuration document. */
export interface PluginConfig {
  version?: number;
  components?: ComponentSpec[];
  policy?: ConfigPolicy;
  /** Host service profile read at startup; omitted settings come from config files. */
  diagnostics?: { enabled?: boolean; limits?: Record<string, number> };
}

/** Callbacks a plugin kind provides. */
export interface Plugin {
  /** Validates one component config. */
  validate?: (pluginConfig: JsonObject) => ConfigDiagnostic[] | null | undefined;
  /** Installs middleware and subscribers for one component; throwing aborts initialization. */
  register: (pluginConfig: JsonObject, context: PluginContext) => void;
}

/**
 * Creates an empty plugin configuration.
 *
 * The document has `version` 1 and no components. Add components to it, then
 * pass it to `validate` or `initialize`.
 *
 * @returns A plugin configuration with no components.
 * @remarks Editing the returned object changes nothing until it is passed to
 * `initialize`.
 */
export function defaultConfig(): PluginConfig {
  return {
    version: 1,
    components: [],
  };
}

/**
 * Creates one component entry for `PluginConfig.components`.
 *
 * For a built-in kind, `config` is checked against that kind's config type.
 * For a kind registered with `register`, `config` is any JSON object.
 *
 * @param kind - Plugin kind name.
 * @param config - Config handed to the plugin's hooks.
 * @param options - `enabled` controls whether the component activates.
 * @returns A component entry for a plugin configuration.
 * @remarks With `enabled` false, `initialize` validates the component but does
 * not activate it.
 */
export function ComponentSpec<K extends ComponentKind>(
  kind: K,
  config?: ComponentConfigs[K],
  options?: {
    enabled?: boolean;
  },
): ComponentSpecFor<K>;
export function ComponentSpec<K extends string, C extends JsonObjectCompatible<C> = JsonObject>(
  kind: K extends ComponentKind ? never : K,
  config?: C,
  options?: {
    enabled?: boolean;
  },
): OpenComponentSpec<C>;
export function ComponentSpec(
  kind: string,
  config: object = {},
  { enabled = true }: { enabled?: boolean } = {},
): { kind: string; enabled?: boolean; config?: object } {
  return {
    kind,
    enabled,
    config,
  };
}

/**
 * Initializes the plugin host and activates the configured plugins.
 *
 * The system and user `plugins.toml` files are merged first and `config` is
 * applied on top of them.
 *
 * @param config - Programmatic configuration. Its values override the files.
 * @param additionalPluginsToml - Path of one more `plugins.toml` to merge. If
 * the file is missing, the report carries a `plugin.configuration_file_missing` warning.
 * @returns The activation, which holds the host report.
 * @remarks Plugin callbacks run only while the activation is alive. End it
 * with `close()` or `await using`.
 */
export function initialize(config: PluginConfig, additionalPluginsToml?: string): Promise<PluginHostActivation> {
  return lib.initialize(config, additionalPluginsToml);
}

/**
 * Validates the plugin host configuration without loading plugin code.
 *
 * The files and `config` are merged exactly as `initialize` merges them, and
 * the merged result is checked.
 *
 * @param config - Programmatic configuration. Its values override the files.
 * @param additionalPluginsToml - Path of one more `plugins.toml` to merge. If
 * the file is missing, the report carries a `plugin.configuration_file_missing` warning.
 * @returns The report for static components and dynamic plugins.
 * @remarks Nothing is activated and no host lease is taken.
 */
export function validate(config: PluginConfig, additionalPluginsToml?: string): PluginHostReport {
  return lib.validate(config, additionalPluginsToml);
}

/**
 * Validates one plugin configuration document by itself.
 *
 * No `plugins.toml` file is read or merged, so `config` must be the whole
 * document.
 *
 * @param config - Complete plugin configuration.
 * @returns The report for the static components in `config`.
 * @remarks Host-only `diagnostics` settings are rejected here. Use `validate`
 * to check them.
 */
export function validateExact(config: PluginConfig): PluginHostReport {
  return lib.validateExact(config);
}

/**
 * Lists the plugin kinds in the registry.
 *
 * A plugin configuration can reference any of these kinds.
 *
 * @returns The registered plugin kind names.
 * @remarks A registered kind is not necessarily active in the current
 * configuration.
 */
export function listKinds(): string[] {
  return lib.listPluginKinds();
}

/**
 * Registers a plugin kind implemented in JavaScript.
 *
 * `plugin.validate` runs when a configuration is validated. `plugin.register`
 * runs when `initialize` activates a component of this kind.
 *
 * @param pluginKind - Unique plugin kind name.
 * @param plugin - The `validate` and `register` hooks.
 * @returns Nothing.
 * @remarks Without `plugin.validate`, any config for this kind passes
 * validation.
 */
export function register(pluginKind: string, plugin: Plugin): void {
  lib.registerPlugin(
    pluginKind,
    plugin.validate === undefined ? null : (pluginConfig) => plugin.validate?.(pluginConfig),
    (pluginConfig, context) => plugin.register(pluginConfig, context),
  );
}

/**
 * Removes a plugin kind from the registry.
 *
 * Configurations validated or initialized after this call cannot reference
 * the kind.
 *
 * @param pluginKind - Name the kind was registered under.
 * @returns `true` if the kind was registered, `false` if it was not.
 * @remarks Components of this kind that are already active keep running until
 * their activation closes.
 */
export function deregister(pluginKind: string): boolean {
  return lib.deregisterPlugin(pluginKind);
}
