// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// <reference lib="esnext.disposable" />

import type { EventSanitizeFields, Json } from './index';

/** Policy behavior for unsupported configuration. */
export type UnsupportedBehavior = 'ignore' | 'warn' | 'error';

/** Plugin-level policy for unknown or unsupported plugin configuration. */
export interface ConfigPolicy {
  unknown_component?: UnsupportedBehavior;
  unknown_field?: UnsupportedBehavior;
  unsupported_value?: UnsupportedBehavior;
}

/** One validation or compatibility diagnostic produced by the plugin system. */
export interface ConfigDiagnostic {
  level: 'warning' | 'error';
  code: string;
  component?: string;
  field?: string;
  message: string;
}

/** Validation or activation report for a plugin configuration. */
export interface ConfigReport {
  diagnostics: ConfigDiagnostic[];
}

/** One top-level plugin component. */
export interface ComponentSpec {
  kind: string;
  enabled?: boolean;
  config?: Record<string, Json>;
}

/** Canonical plugin configuration document. */
export interface PluginConfig {
  version?: number;
  components?: Array<{
    kind: string;
    enabled?: boolean;
    config?: Record<string, Json>;
  }>;
  policy?: ConfigPolicy;
}

/** Execution lane for a dynamically loaded Relay plugin. */
export type DynamicPluginKind = 'rust_dynamic' | 'worker';

/** Explicitly resolved dynamic plugin load and component configuration. */
export interface DynamicPluginActivationSpec {
  pluginId: string;
  kind: DynamicPluginKind;
  manifestRef: string;
  environmentRef?: string | null;
  config?: Record<string, Json>;
}

/** Owns one process-wide dynamic plugin host activation. */
export interface DynamicPluginActivation extends AsyncDisposable {
  /** Validation report produced by the successful activation. */
  readonly report: ConfigReport;
  /**
   * Whether this activation handle has not begun teardown. `false` does not
   * guarantee another process-wide activation can start after failed teardown.
   */
  readonly active: boolean;
  /** Clear callbacks before unloading libraries and workers. Idempotent. */
  close(): Promise<void>;
  /** Delegate structured `await using` cleanup to `close()`. */
  [Symbol.asyncDispose](): Promise<void>;
}

/** A mark Relay materializes under a managed lifecycle. */
export interface PendingMarkSpec {
  name: string;
  category?: string | null;
  categoryProfile?: Json;
  data?: Json;
  metadata?: Json;
}

/** Schema tag attached to an opaque optimization contribution payload. */
export interface LlmOptimizationDataSchema {
  name: string;
  version: string;
}

/** Model identity retained for counterfactual pricing and downstream repricing. */
export interface LlmOptimizationModel {
  model: string;
  provider?: string;
}

/** Baseline and effective model identities for a routing optimization. */
export interface LlmOptimizationModelTransition {
  baseline?: LlmOptimizationModel;
  effective?: LlmOptimizationModel;
}

/** Explicit token evidence, independent from a pricing catalog. */
export interface LlmOptimizationTokens {
  /** Token counts must be non-negative JavaScript safe integers. */
  prompt_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  completion_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  cache_read_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  cache_write_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  total_tokens?: number;
}

/** Baseline, effective, and saved token evidence for one optimization. */
export interface LlmOptimizationTokenImpact {
  baseline?: LlmOptimizationTokens;
  effective?: LlmOptimizationTokens;
  saved?: LlmOptimizationTokens;
  quality?: 'observed' | 'estimated';
  estimation_method?: string;
}

/**
 * One plugin's optimization evidence.
 *
 * `kind` is deliberately an open string so new optimizer categories round-trip
 * without a Relay release. Unknown top-level fields are retained by the wire
 * contract and represented by this interface's JSON extension surface.
 */
export interface LlmOptimizationContribution {
  id?: string;
  /** Relay ordering must remain within JavaScript's safe-integer range. */
  sequence?: number;
  producer: string;
  kind: 'input_compression' | 'model_routing' | (string & {});
  applied: boolean;
  model_transition?: LlmOptimizationModelTransition;
  token_impact?: LlmOptimizationTokenImpact;
  payload_schema?: LlmOptimizationDataSchema;
  payload?: Json;
  [key: string]: Json | undefined;
}

/** Canonical result returned by an LLM request intercept. */
export interface LlmRequestInterceptOutcome {
  request: Json;
  annotated?: Json | null;
  pendingMarks?: PendingMarkSpec[];
  optimizationContributions?: LlmOptimizationContribution[];
}

/**
 * Canonical result returned by a tool execution intercept.
 *
 * `result` is passed to the remaining middleware and application. `pendingMarks`
 * are Relay-owned lifecycle metadata emitted after the tool-end event and are
 * not included in the application-visible result.
 */
export interface ToolExecutionInterceptOutcome {
  result: Json;
  pendingMarks?: PendingMarkSpec[];
}

/** Component-scoped registration context passed to plugin handlers. */
export interface PluginContext {
  /** Register an infallible event subscriber for this component. */
  registerSubscriber(name: string, callback: (event: Json) => void): void;
  /** Register a mark event sanitizer for this component. */
  registerMarkSanitizeGuardrail(
    name: string,
    priority: number,
    callback: (event: Json, fields: EventSanitizeFields) => EventSanitizeFields,
  ): void;
  /** Register a scope-start event sanitizer for this component. */
  registerScopeSanitizeStartGuardrail(
    name: string,
    priority: number,
    callback: (event: Json, fields: EventSanitizeFields) => EventSanitizeFields,
  ): void;
  /** Register a scope-end event sanitizer for this component. */
  registerScopeSanitizeEndGuardrail(
    name: string,
    priority: number,
    callback: (event: Json, fields: EventSanitizeFields) => EventSanitizeFields,
  ): void;
  /** Register a tool sanitize-request guardrail for this component. */
  registerToolSanitizeRequestGuardrail(
    name: string,
    priority: number,
    callback: (name: string, args: Json) => Json,
  ): void;
  /** Register a tool sanitize-response guardrail for this component. */
  registerToolSanitizeResponseGuardrail(
    name: string,
    priority: number,
    callback: (name: string, result: Json) => Json,
  ): void;
  /** Register a tool conditional-execution guardrail for this component. */
  registerToolConditionalExecutionGuardrail(
    name: string,
    priority: number,
    callback: (name: string, args: Json) => string | null,
  ): void;
  /** Register an LLM sanitize-request guardrail for this component. */
  registerLlmSanitizeRequestGuardrail(name: string, priority: number, callback: (request: Json) => Json): void;
  /** Register an LLM sanitize-response guardrail for this component. */
  registerLlmSanitizeResponseGuardrail(name: string, priority: number, callback: (response: Json) => Json): void;
  /** Register an LLM conditional-execution guardrail for this component. */
  registerLlmConditionalExecutionGuardrail(
    name: string,
    priority: number,
    callback: (request: Json) => string | null,
  ): void;
  /** Register an LLM request intercept for this component. */
  registerLlmRequestIntercept(
    name: string,
    priority: number,
    breakChain: boolean,
    callback: (args: { name: string; request: Json; annotated: Json | null }) => LlmRequestInterceptOutcome,
  ): void;
  /** Register an LLM execution intercept for this component. */
  registerLlmExecutionIntercept(
    name: string,
    priority: number,
    callback: (request: Json, next: (request: Json) => Json | Promise<Json>) => Json | Promise<Json>,
  ): void;
  /** Register an LLM streaming execution intercept for this component. */
  registerLlmStreamExecutionIntercept(
    name: string,
    priority: number,
    callback: (
      request: Json,
      next: (request: Json) => AsyncIterable<Json> | Promise<AsyncIterable<Json>>,
    ) => AsyncIterable<Json> | Promise<AsyncIterable<Json>>,
  ): void;
  /** Register a tool request intercept for this component. */
  registerToolRequestIntercept(
    name: string,
    priority: number,
    breakChain: boolean,
    callback: (name: string, args: Json) => Json,
  ): void;
  /**
   * Register tool execution middleware that returns a canonical outcome.
   * The `next` callback resolves to the raw downstream result.
   */
  registerToolExecutionIntercept(
    name: string,
    priority: number,
    callback: (
      args: Json,
      next: (args: Json) => Json | Promise<Json>,
    ) => ToolExecutionInterceptOutcome | Promise<ToolExecutionInterceptOutcome>,
  ): void;
}

/** Plugin callback contract. */
export interface Plugin {
  /** Validate one component-local config object. */
  validate?(pluginConfig: Record<string, Json>): ConfigDiagnostic[] | null | undefined;
  /**
   * Install middleware and subscribers for one component instance.
   *
   * Throwing aborts the current initialization and triggers rollback.
   */
  register(pluginConfig: Record<string, Json>, context: PluginContext): void;
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
export declare function ComponentSpec(
  kind: string,
  config?: Record<string, Json>,
  options?: {
    enabled?: boolean;
  },
): ComponentSpec;
/**
 * Validate a plugin configuration without activating it.
 *
 * Runs the same config validation pipeline used by initialization while
 * leaving the active plugin registry and runtime configuration unchanged.
 *
 * @param config - Candidate plugin configuration document.
 * @returns A structured validation report with diagnostics.
 * @remarks Use this to surface warnings or incompatibilities before replacing
 * the active plugin configuration.
 */
export declare function validate(config: PluginConfig): ConfigReport;
/**
 * Validate and activate a plugin configuration.
 *
 * Replaces the current active config, invokes each enabled component's
 * registration hooks, and resolves with the final activation report.
 *
 * @param config - Plugin configuration document to activate.
 * @returns A promise resolving to the activation report.
 * @remarks Partial plugin registration is rolled back if activation fails, and
 * the promise rejects with the underlying validation or setup error.
 */
export declare function initialize(config: PluginConfig): Promise<ConfigReport>;
/**
 * Load and activate explicitly resolved dynamic plugins.
 *
 * The returned object owns loaded libraries and worker processes. Keep it
 * alive while plugin callbacks may run and call `close()` for deterministic
 * teardown. Garbage collection is a defensive fallback only.
 *
 * @param config - Base configuration layered over discovered `plugins.toml` files.
 * @param specs - Non-empty explicit manifest and component configuration for each plugin.
 * @returns The owned activation and its validation report.
 * @remarks File-configured static components initialize before dynamic
 * components. Use `initialize()` for a static-only configuration.
 */
export declare function activateDynamicPlugins(
  config: PluginConfig,
  specs: DynamicPluginActivationSpec[],
): Promise<DynamicPluginActivation>;
/**
 * Clear the active plugin configuration.
 *
 * Removes the currently active component registrations while leaving plugin
 * kinds in the registry so they can be reused by a later initialization call.
 *
 * @returns Nothing.
 * @remarks Registered plugin kinds remain available after the active config is
 * cleared.
 */
export declare function clear(): void;
/**
 * Return the last successfully activated plugin report.
 *
 * Exposes the most recent activation report emitted by the native plugin system
 * without triggering validation or registration work.
 *
 * @returns The last activation report, if one exists.
 * @remarks This returns `null` until `initialize` succeeds at least once in
 * the current process.
 */
export declare function report(): ConfigReport | null;
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
 * @remarks Active runtime registrations remain until `clear()` or the next
 * successful `initialize(...)`.
 */
export declare function deregister(pluginKind: string): boolean;
