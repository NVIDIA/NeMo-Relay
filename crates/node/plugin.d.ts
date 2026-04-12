// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Json } from './index';

/** Policy behavior for unsupported configuration. */
export type UnsupportedBehavior = 'ignore' | 'warn' | 'error';

/** Host-level policy for unknown or unsupported plugin configuration. */
export interface ConfigPolicy {
  unknown_component?: UnsupportedBehavior;
  unknown_field?: UnsupportedBehavior;
  unsupported_value?: UnsupportedBehavior;
}

/** One validation or compatibility diagnostic produced by the plugin host. */
export interface ConfigDiagnostic {
  level: 'warning' | 'error';
  code: string;
  component?: string;
  field?: string;
  message: string;
}

/** Validation or activation report for a plugin-host configuration. */
export interface ConfigReport {
  diagnostics: ConfigDiagnostic[];
}

/** One top-level hosted plugin component. */
export interface ComponentSpec {
  kind: string;
  enabled?: boolean;
  config?: Record<string, Json>;
}

/** Canonical plugin-host configuration document. */
export interface PluginConfig {
  version?: number;
  components?: Array<{ kind: string; enabled?: boolean; config?: Record<string, Json> }>;
  policy?: ConfigPolicy;
}

/** Component-scoped registration context passed to hosted plugin handlers. */
export interface PluginContext {
  /** Register an infallible event subscriber for this component. */
  registerSubscriber(name: string, callback: (event: Json) => void): void;
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
  registerLlmSanitizeRequestGuardrail(
    name: string,
    priority: number,
    callback: (request: Json) => Json,
  ): void;
  /** Register an LLM sanitize-response guardrail for this component. */
  registerLlmSanitizeResponseGuardrail(
    name: string,
    priority: number,
    callback: (response: Json) => Json,
  ): void;
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
    callback: (args: {
      name: string;
      request: Json;
      annotated: Json | null;
    }) => {
      request: Json;
      annotated: Json | null;
    },
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
  /** Register a tool execution intercept for this component. */
  registerToolExecutionIntercept(
    name: string,
    priority: number,
    callback: (args: Json, next: (args: Json) => Json | Promise<Json>) => Json | Promise<Json>,
  ): void;
}

/** Hosted plugin callback contract. */
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

/** Create a default plugin-host config with `version = 1` and no components. */
export declare function defaultConfig(): PluginConfig;
/**
 * Create one top-level hosted plugin component.
 *
 * `enabled = false` keeps the component in the config for validation but skips
 * runtime registration.
 */
export declare function ComponentSpec(
  kind: string,
  config?: Record<string, Json>,
  options?: { enabled?: boolean },
): ComponentSpec;
/** Validate a plugin-host config without changing runtime state. */
export declare function validate(config: PluginConfig): ConfigReport;
/**
 * Validate and activate a plugin-host config.
 *
 * Replaces the current configuration and rolls back partial registration on
 * failure.
 */
export declare function initialize(config: PluginConfig): Promise<ConfigReport>;
/** Clear the active plugin configuration while leaving plugin kinds registered. */
export declare function clear(): void;
/** Return the last successfully activated plugin report, if any. */
export declare function report(): ConfigReport | null;
/** List registered plugin kinds known to the registry. */
export declare function listKinds(): string[];
/** Register a plugin kind for later validation and initialization. */
export declare function register(pluginKind: string, plugin: Plugin): void;
/**
 * Remove a previously registered plugin kind.
 *
 * Active runtime registrations remain until `clear()` or the next successful
 * `initialize(...)`.
 */
export declare function deregister(pluginKind: string): boolean;
