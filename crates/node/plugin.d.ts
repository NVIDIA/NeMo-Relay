// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// <reference lib="esnext.disposable" />

import type { EventSanitizeFields, Json, RuntimeRegistrationKind, ToolExecutionResult } from './index';
import type { LlmCodec, LlmResponseCodec } from './typed';

/** Codec identity available while a managed LLM event is sanitized. */
export type LlmCodecIdentity =
  | { kind: 'none' }
  | {
      kind: 'builtin';
      id: 'openai_chat' | 'openai_responses' | 'anthropic_messages' | 'oci_genai' | 'gemini_generate_content';
    }
  | { kind: 'runtime'; id: string }
  | { kind: 'opaque' };

/** Codec context available while an LLM request is sanitized. */
export interface LlmSanitizeRequestContext {
  codec: LlmCodecIdentity;
  /** Resolve the active codec for this callback. Do not retain the result after the callback returns. */
  resolveCodec(): LlmCodec | null;
}

/** Codec context available while an LLM response is sanitized. */
export interface LlmSanitizeResponseContext {
  codec: LlmCodecIdentity;
  /** Resolve the active codec for this callback. Do not retain the result after the callback returns. */
  resolveCodec(): LlmResponseCodec | null;
}

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

export interface PluginHostActivation extends AsyncDisposable {
  /** Validation report produced by the successful activation. */
  readonly report: PluginHostReport;
  /**
   * Whether this activation handle has not begun teardown. `false` does not
   * guarantee another process-wide activation can start after failed teardown.
   */
  readonly isActive: boolean;
  /** Clear callbacks before unloading libraries and workers. Idempotent. */
  close(): Promise<void>;
  /** Delegate structured `await using` cleanup to `close()`. */
  [Symbol.asyncDispose](): Promise<void>;
}

export interface PluginHostReport {
  config: ConfigReport;
  dynamic_plugins: DynamicPluginValidationReport[];
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

/** A mark Relay materializes under a managed lifecycle. */
export interface PendingMarkSpec {
  name: string;
  category?: string | null;
  categoryProfile?: Json;
  data?: Json;
  dataSchema?: { name: string; version: string } | null;
  metadata?: Json;
  severity?: 'trace' | 'debug' | 'info' | 'warn' | 'warning' | 'error' | null;
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
  annotation?: Json;
  pendingMarks?: PendingMarkSpec[];
}

/** Scalar value accepted in event metadata additions. */
export type EventMetadataScalar = string | number | boolean;

/**
 * Flat value accepted in event metadata additions. After JSON conversion,
 * numeric arrays must contain only integer values or only floating-point values.
 */
export type EventMetadataValue = EventMetadataScalar | string[] | number[] | boolean[];

/** Metadata additions returned by an event metadata injector. */
export type EventMetadata = Record<string, EventMetadataValue>;

/** Component-scoped registration context passed to plugin handlers. */
export interface PluginContext {
  /** Register an activation-owned eligibility gate for a global runtime registration. */
  registerConditionalMiddlewareGuardrail(
    name: string,
    kinds: RuntimeRegistrationKind[],
    registrationName: string,
    guardrail: (kinds: RuntimeRegistrationKind[], registrationName: string) => string | null,
  ): void;
  /**
   * Register an event subscriber for this component. Callback failures are isolated and reported
   * through the Node binding's callback-error channel; flushSubscribers waits for returned promises.
   */
  registerSubscriber(name: string, callback: (event: Json) => void | Promise<void>): void;
  /** Register an event metadata injector for this component. */
  registerEventMetadataInjector(
    name: string,
    priority: number,
    callback: (event: Json) => EventMetadata | Promise<EventMetadata>,
  ): void;
  /** Register a mark event sanitizer for this component. */
  registerMarkSanitizeGuardrail(
    name: string,
    priority: number,
    callback: (event: Json, fields: EventSanitizeFields) => EventSanitizeFields | Promise<EventSanitizeFields>,
  ): void;
  /** Register a scope-start event sanitizer for this component. */
  registerScopeSanitizeStartGuardrail(
    name: string,
    priority: number,
    callback: (event: Json, fields: EventSanitizeFields) => EventSanitizeFields | Promise<EventSanitizeFields>,
  ): void;
  /** Register a scope-end event sanitizer for this component. */
  registerScopeSanitizeEndGuardrail(
    name: string,
    priority: number,
    callback: (event: Json, fields: EventSanitizeFields) => EventSanitizeFields | Promise<EventSanitizeFields>,
  ): void;
  /** Register a tool sanitize-request guardrail for this component. */
  registerToolSanitizeRequestGuardrail(
    name: string,
    priority: number,
    callback: (name: string, args: Json) => Json | Promise<Json>,
  ): void;
  /** Register a tool sanitize-response guardrail for this component. */
  registerToolSanitizeResponseGuardrail(
    name: string,
    priority: number,
    callback: (name: string, result: Json) => Json | Promise<Json>,
  ): void;
  /** Register a tool conditional-execution guardrail for this component. */
  registerToolConditionalExecutionGuardrail(
    name: string,
    priority: number,
    callback: (name: string, args: Json) => string | null | Promise<string | null>,
  ): void;
  /** Register an LLM sanitize-request guardrail. The callback receives `(request, context)`. */
  registerLlmSanitizeRequestGuardrail(
    name: string,
    priority: number,
    callback: (request: Json, context: LlmSanitizeRequestContext) => Json | null | Promise<Json | null>,
  ): void;
  /** Register an LLM sanitize-response guardrail. The callback receives `(response, context)`. */
  registerLlmSanitizeResponseGuardrail(
    name: string,
    priority: number,
    callback: (response: Json, context: LlmSanitizeResponseContext) => Json | null | Promise<Json | null>,
  ): void;
  /** Register an LLM conditional-execution guardrail for this component. */
  registerLlmConditionalExecutionGuardrail(
    name: string,
    priority: number,
    callback: (request: Json) => string | null | Promise<string | null>,
  ): void;
  /** Register an LLM request intercept for this component. */
  registerLlmRequestIntercept(
    name: string,
    priority: number,
    breakChain: boolean,
    callback: (args: {
      name: string;
      request: Json;
      annotated: Json | null;
    }) => LlmRequestInterceptOutcome | Promise<LlmRequestInterceptOutcome>,
  ): void;
  /** Register an LLM execution intercept for this component. */
  registerLlmExecutionIntercept(
    name: string,
    priority: number,
    callback: (request: Json, next: (request: Json) => Json | Promise<Json>) => Json | Promise<Json>,
  ): void;
  /**
   * Register an LLM streaming execution intercept for this component.
   *
   * The `next` callback resolves to a lazy stream. Return that stream to
   * preserve incremental downstream delivery.
   */
  registerLlmStreamExecutionIntercept(
    name: string,
    priority: number,
    callback: (
      request: Json,
      next: (request: Json) => Promise<AsyncIterable<Json>>,
    ) => AsyncIterable<Json> | Promise<AsyncIterable<Json>>,
  ): void;
  /** Register a tool request intercept for this component. */
  registerToolRequestIntercept(
    name: string,
    priority: number,
    breakChain: boolean,
    callback: (name: string, args: Json) => Json | Promise<Json>,
  ): void;
  /**
   * Register tool execution middleware that returns a canonical outcome.
   * The `next` callback resolves to the canonical downstream result.
   */
  registerToolExecutionIntercept(
    name: string,
    priority: number,
    callback: (
      args: Json,
      next: (args: Json) => ToolExecutionResult | Promise<ToolExecutionResult>,
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
 * Initialize the core-owned static and dynamic plugin host.
 *
 * Resolves programmatic, explicit, user, and system configuration layers and
 * activates the resulting host under one owned lifetime.
 *
 * @param config - Lowest-precedence programmatic configuration.
 * @param additionalPluginsToml - Optional explicit `plugins.toml` layer.
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
 * @param config - Lowest-precedence programmatic configuration.
 * @param additionalPluginsToml - Optional explicit `plugins.toml` layer.
 * @returns Structured static and dynamic validation report.
 * @remarks Validation performs no activation and does not acquire the host lease.
 */
export declare function validate(config: PluginConfig, additionalPluginsToml?: string): PluginHostReport;
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
