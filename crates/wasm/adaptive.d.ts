// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ConfigPolicy, ConfigDiagnostic, ConfigReport } from './plugin';
import type { JsonObject } from './typed';

export { ConfigPolicy, ConfigDiagnostic, ConfigReport };

/** Adaptive state backend selection. */
export interface BackendSpec {
  kind: string;
  config?: JsonObject;
}

/** Adaptive state configuration. */
export interface StateConfig {
  backend: BackendSpec;
}

/** Built-in adaptive telemetry settings. */
export interface TelemetryConfig {
  subscriber_name?: string;
  learners?: string[];
}

/** Built-in adaptive hints injection settings. */
export interface AdaptiveHintsConfig {
  priority?: number;
  break_chain?: boolean;
  inject_header?: boolean;
  inject_body_path?: string;
}

/** Built-in adaptive tool scheduling settings. */
export interface ToolParallelismConfig {
  priority?: number;
  mode?: 'observe_only' | 'inject_hints' | 'schedule' | string;
}

/** Canonical config object for the top-level adaptive component. */
export interface Config {
  version?: number;
  agent_id?: string;
  state?: StateConfig;
  telemetry?: TelemetryConfig;
  adaptive_hints?: AdaptiveHintsConfig;
  tool_parallelism?: ToolParallelismConfig;
  policy?: ConfigPolicy;
}

/** Top-level adaptive component wrapper with fixed kind `adaptive`. */
export interface ComponentSpec {
  kind: 'adaptive';
  enabled?: boolean;
  config: Config;
}

export declare const ADAPTIVE_PLUGIN_KIND: 'adaptive';
/** Create a default adaptive config with `version = 1`. */
export declare function defaultConfig(): Config;
/** Create an in-memory adaptive state backend spec. */
export declare function inMemoryBackend(): BackendSpec;
/** Create a Redis adaptive state backend spec. */
export declare function redisBackend(url: string, keyPrefix?: string): BackendSpec;
/** Create built-in adaptive telemetry settings. */
export declare function telemetryConfig(config?: TelemetryConfig): TelemetryConfig;
/** Create built-in adaptive hints injection settings. */
export declare function adaptiveHintsConfig(config?: AdaptiveHintsConfig): AdaptiveHintsConfig;
/** Create built-in adaptive tool scheduling settings. */
export declare function toolParallelismConfig(config?: ToolParallelismConfig): ToolParallelismConfig;
/**
 * Wrap adaptive config as a top-level component.
 *
 * The returned object is placed directly into `plugin.Config.components`.
 */
export declare function ComponentSpec(
  config: Config,
  options?: { enabled?: boolean },
): ComponentSpec;
