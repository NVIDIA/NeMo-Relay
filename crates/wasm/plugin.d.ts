// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { JsonObject, JsonValue, LlmRequestShape } from './typed';

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
  config?: JsonObject;
}

/** Canonical plugin-host configuration document. */
export interface Config {
  version?: number;
  components?: ComponentSpec[];
  policy?: ConfigPolicy;
}

/** Component-scoped registration context passed to hosted plugin handlers. */
export interface PluginContext {
  /** Register an infallible event subscriber for this component. */
  registerSubscriber(name: string, callback: (event: JsonValue) => void): void;
  /** Register an LLM request intercept for this component. */
  registerLlmRequestIntercept(
    name: string,
    priority: number,
    breakChain: boolean,
    callback: (
      name: string,
      request: LlmRequestShape,
      annotated: JsonValue | null,
    ) => {
      request: LlmRequestShape;
      annotated: JsonValue | null;
    },
  ): void;
  /** Register an LLM execution intercept for this component. */
  registerLlmExecutionIntercept(
    name: string,
    priority: number,
    callback: (
      request: LlmRequestShape,
      next: (request: LlmRequestShape) => JsonValue | Promise<JsonValue>,
    ) => JsonValue | Promise<JsonValue>,
  ): void;
  /** Register an LLM streaming execution intercept for this component. */
  registerLlmStreamExecutionIntercept(
    name: string,
    priority: number,
    callback: (
      request: LlmRequestShape,
      next: (request: LlmRequestShape) => AsyncIterable<JsonValue> | Promise<AsyncIterable<JsonValue>>,
    ) => AsyncIterable<JsonValue> | Promise<AsyncIterable<JsonValue>>,
  ): void;
  /** Register a tool request intercept for this component. */
  registerToolRequestIntercept(
    name: string,
    priority: number,
    breakChain: boolean,
    callback: (name: string, args: JsonValue) => JsonValue,
  ): void;
  /** Register a tool execution intercept for this component. */
  registerToolExecutionIntercept(
    name: string,
    priority: number,
    callback: (
      args: JsonValue,
      next: (args: JsonValue) => JsonValue | Promise<JsonValue>,
    ) => JsonValue | Promise<JsonValue>,
  ): void;
}

/** Hosted plugin callback contract. */
export interface PluginHandler {
  /** Validate one component-local config object. */
  validate?(pluginConfig: JsonObject): ConfigDiagnostic[] | null | undefined;
  /**
   * Install middleware and subscribers for one component instance.
   *
   * Throwing aborts the current initialization and triggers rollback.
   */
  register(pluginConfig: JsonObject, context: PluginContext): void;
}

/** Create a default plugin-host config with `version = 1` and no components. */
export declare function defaultConfig(): Config;
/**
 * Create one top-level hosted plugin component.
 *
 * `enabled = false` keeps the component in the config for validation but skips
 * runtime registration.
 */
export declare function ComponentSpec(
  kind: string,
  config?: JsonObject,
  options?: { enabled?: boolean },
): ComponentSpec;
/** Validate a plugin-host config without changing runtime state. */
export declare function validate(config: Config): ConfigReport;
/**
 * Validate and activate a plugin-host config.
 *
 * Replaces the current configuration and rolls back partial registration on
 * failure.
 */
export declare function initialize(config: Config): Promise<ConfigReport>;
/** Clear the active plugin configuration while leaving handler kinds registered. */
export declare function clear(): void;
/** Return the last successfully activated plugin report, if any. */
export declare function report(): ConfigReport | null;
/** List registered hosted plugin kinds known to the handler registry. */
export declare function listKinds(): string[];
/** Register a hosted plugin kind for later validation and initialization. */
export declare function register(pluginKind: string, handler: PluginHandler): void;
/**
 * Remove a previously registered hosted plugin kind.
 *
 * Active runtime registrations remain until `clear()` or the next successful
 * `initialize(...)`.
 */
export declare function deregister(pluginKind: string): boolean;
