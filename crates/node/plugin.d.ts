// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// <reference lib="esnext.disposable" />

import type { AdaptivePluginConfig } from './adaptive';
import type { ConfigDiagnostic, Json, JsonObject, PluginContext, ResourceMetricsConfig } from './index';
import type { PricingConfig } from './model_pricing';
import type { Config as ObservabilityConfig } from './observability';
import type { Config as PiiRedactionConfig } from './pii_redaction';

export type {
  ConfigDiagnostic,
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
  LlmRequestInterceptOutcome,
  LlmRequestContext,
  LlmResponseContext,
  LlmSanitizeRequestContext,
  LlmSanitizeResponseContext,
  PluginContext,
  ToolExecutionInterceptOutcome,
} from './index';

/** Policy behavior for unsupported configuration. */
export type UnsupportedBehavior = 'ignore' | 'warn' | 'error';

/** Plugin-level policy for unknown or unsupported plugin configuration. */
export interface ConfigPolicy {
  unknown_component?: UnsupportedBehavior;
  unknown_field?: UnsupportedBehavior;
  unsupported_value?: UnsupportedBehavior;
}

/** Validation or activation report for a plugin configuration. */
export interface ConfigReport {
  diagnostics: ConfigDiagnostic[];
  runtime_diagnostics?: RuntimeDiagnostic[];
}

/** One bounded aggregate of a runtime plugin failure. */
export interface RuntimeDiagnostic {
  code: string;
  component: string;
  field?: string;
  message: string;
  session_id?: string;
  count: number;
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

/** One top-level plugin component. */
export interface ComponentSpec {
  kind: string;
  enabled?: boolean;
  config?: JsonObject;
}

export type ComponentConfigFor<K extends string> = K extends ComponentKind ? ComponentConfigs[K] : JsonObject;

export type ComponentSpecFor<K extends string> = K extends ComponentKind
  ? Extract<KnownComponentSpec, { kind: K }>
  : ComponentSpec;

/** Canonical plugin configuration document. */
export interface PluginConfig {
  version?: number;
  components?: Array<KnownComponentSpec | ComponentSpec>;
  policy?: ConfigPolicy;
  /** Startup-only host service profile; omitted settings use file layers. */
  diagnostics?: { enabled?: boolean; limits?: Record<string, number> };
}

/** Execution lane for a dynamically loaded Relay plugin. */
export type DynamicPluginKind = 'rust_dynamic' | 'worker';

export interface PluginHostActivation extends AsyncDisposable {
  /** Validation report produced by the successful activation. */
  readonly report: PluginHostReport;
  /** Whether this activation remains open. Failed teardown can be retried. */
  readonly isActive: boolean;
  /** Clear callbacks before unloading libraries and workers. Idempotent. */
  close(): Promise<void>;
  /** Delegate structured `await using` cleanup to `close()`. */
  [Symbol.asyncDispose](): Promise<void>;
}

export interface PluginHostReport {
  config: ConfigReport;
  dynamic_plugins: DynamicPluginValidationReport[];
  /** Existing plugins.toml files that contributed to the resolved configuration. */
  config_paths: string[];
  /** Fully merged plugin configuration with sensitive values redacted. */
  resolved_config: Json;
}

export type DynamicPluginCheckState = 'unknown' | 'valid' | 'invalid';

export interface DynamicPluginValidationStatus {
  manifest: DynamicPluginCheckState;
  compatibility: DynamicPluginCheckState;
  integrity: DynamicPluginCheckState;
  environment: DynamicPluginCheckState;
  authenticity: DynamicPluginCheckState;
  policy_satisfied: DynamicPluginCheckState;
  checked_at?: string | null;
  message?: string | null;
}

export interface DynamicPluginFailure {
  phase: string;
  code: string;
  message: string;
}

export interface DynamicPluginValidationReport {
  plugin_id: string;
  manifest_ref: string;
  kind: DynamicPluginKind;
  status: DynamicPluginValidationStatus;
  failure?: DynamicPluginFailure | null;
  selected: boolean;
}

/** Plugin callback contract. */
export interface Plugin {
  /** Validate one component-local config object. */
  validate?(pluginConfig: JsonObject): ConfigDiagnostic[] | null | undefined;
  /**
   * Install middleware and subscribers for one component instance.
   *
   * Throwing aborts the current initialization and triggers rollback.
   */
  register(pluginConfig: JsonObject, context: PluginContext): void;
}

/**
 * Create an empty plugin configuration.
 *
 * Returns the canonical top-level config shape with `version = 1` and no
 * configured components so callers can build a document incrementally before
 * validating or activating it.
 *
 * @returns A new `PluginConfig` object ready for mutation or validation.
 * @remarks Mutating the returned object does not affect runtime state until it
 * is passed to `initialize`.
 */
export declare function defaultConfig(): PluginConfig;
/**
 * Create a plugin component entry for a plugin config document.
 *
 * Packages a plugin kind, component-local config, and enablement flag into the
 * object shape expected by `PluginConfig.components`.
 *
 * @param kind - Registered plugin kind to reference.
 * @param config - Component-local config passed to plugin hooks.
 * @param options - Optional component-level flags.
 * @returns A `ComponentSpec` ready to insert into a plugin config.
 * @remarks Setting `options.enabled = false` preserves the component for
 * validation while skipping runtime registration during `initialize`.
 */
export declare function ComponentSpec<K extends string>(
  kind: K,
  config?: ComponentConfigFor<K>,
  options?: {
    enabled?: boolean;
  },
): ComponentSpecFor<K>;
/**
 * Initialize the core-owned static and dynamic plugin host.
 *
 * Resolves an explicit or discovered user file with the system configuration,
 * then applies programmatic config and activates one owned lifetime.
 *
 * @param config - Programmatic configuration. It overrides file values.
 * @param additionalPluginsToml - Optional explicit `plugins.toml` layer. A
 * missing explicit file is reported as a `plugin.configuration_file_missing`
 * warning in the host report.
 * @returns An owned activation with the unified host report.
 * @remarks Keep the returned activation alive while callbacks may run and call
 * `close()` or use `await using` for deterministic teardown.
 */
export declare function initialize(config: PluginConfig, additionalPluginsToml?: string): Promise<PluginHostActivation>;
/**
 * Validate the plugin host without loading plugin code.
 *
 * Resolves the same layered configuration and trust policy used by activation
 * while leaving the process-wide host lease untouched.
 *
 * @param config - Programmatic configuration. It overrides file values.
 * @param additionalPluginsToml - Optional explicit `plugins.toml` layer. A
 * missing explicit file is reported as a `plugin.configuration_file_missing`
 * warning in the host report.
 * @returns Structured static and dynamic validation report.
 * @remarks Validation performs no activation and does not acquire the host lease.
 */
export declare function validate(config: PluginConfig, additionalPluginsToml?: string): PluginHostReport;
/**
 * Validate only the supplied static plugin configuration.
 *
 * Unlike `validate`, this does not discover or merge `plugins.toml` files.
 * Use it for component-specific validation when `config` is the complete
 * document to check. Host-only `diagnostics` settings are rejected; use
 * `validate` to validate those settings.
 *
 * @param config - Complete static plugin configuration.
 * @returns Static validation results with no dynamic plugins.
 */
export declare function validateExact(config: PluginConfig): PluginHostReport;
/**
 * List registered plugin kinds.
 *
 * Returns the plugin kind identifiers currently known to the global registry
 * so callers can inspect what can be referenced from plugin configs.
 *
 * @returns The registered plugin kind names.
 * @remarks The list reflects registry state only; it does not indicate whether
 * a plugin kind is currently active in the runtime configuration.
 */
export declare function listKinds(): string[];
/**
 * Register a plugin kind with JavaScript validation and registration hooks.
 *
 * Adapts the higher-level `Plugin` object contract to the native callback
 * shape expected by the Node binding.
 *
 * @param pluginKind - Unique plugin kind identifier to register.
 * @param plugin - Plugin implementation with `validate` and `register` hooks.
 * @returns Nothing.
 * @remarks Omitting `plugin.validate` makes the plugin permissive during
 * validation; `plugin.register` still runs later during `initialize`.
 */
export declare function register(pluginKind: string, plugin: Plugin): void;
/**
 * Remove a previously registered plugin kind.
 *
 * Deletes the plugin kind from the registry so future config validation and
 * initialization calls can no longer reference it.
 *
 * @param pluginKind - Registered plugin kind identifier to remove.
 * @returns `true` when a plugin kind was removed, otherwise `false`.
 * @remarks Active runtime registrations remain until the owning plugin-host
 * activation closes.
 */
export declare function deregister(pluginKind: string): boolean;
