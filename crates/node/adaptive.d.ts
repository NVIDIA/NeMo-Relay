// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Json } from './index';
import type { ConfigPolicy, ConfigDiagnostic, ConfigReport } from './plugin';

export { ConfigPolicy, ConfigDiagnostic, ConfigReport };

/** Adaptive state backend selection. */
export interface BackendSpec {
  kind: string;
  config?: Record<string, Json>;
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

/** ACG prompt-stability classification thresholds. */
export interface AcgStabilityThresholds {
  stable_threshold?: number;
  semi_stable_threshold?: number;
  min_observations_for_full_confidence?: number;
}

/** Adaptive cache-governor settings. */
export interface AcgConfig {
  provider?: 'anthropic' | 'openai' | 'passthrough' | string;
  observation_window?: number;
  priority?: number;
  stability_thresholds?: AcgStabilityThresholds;
}

/** Canonical config object for the top-level adaptive component. */
export interface Config {
  version?: number;
  agent_id?: string;
  state?: StateConfig;
  telemetry?: TelemetryConfig;
  adaptive_hints?: AdaptiveHintsConfig;
  tool_parallelism?: ToolParallelismConfig;
  acg?: AcgConfig;
  policy?: ConfigPolicy;
}

/** Top-level adaptive component wrapper with fixed kind `adaptive`. */
export interface ComponentSpec {
  kind: 'adaptive';
  enabled?: boolean;
  config: Config;
}

export declare const ADAPTIVE_PLUGIN_KIND: 'adaptive';
/**
 * Create a default adaptive component config.
 *
 * Returns the minimal top-level adaptive config shape with `version = 1` so
 * callers can add state, telemetry, and scheduler settings incrementally.
 *
 * @returns A new adaptive config object.
 * @remarks The returned object is detached from runtime state until it is
 * wrapped with `ComponentSpec` and activated through the plugin system.
 */
export declare function defaultConfig(): Config;
/**
 * Create an in-memory adaptive state backend spec.
 *
 * Produces the backend descriptor for ephemeral adaptive state stored inside
 * the current process rather than an external datastore.
 *
 * @returns An adaptive backend spec using in-memory storage.
 * @remarks This backend does not persist state across process restarts.
 */
export declare function inMemoryBackend(): BackendSpec;
/**
 * Create a Redis-backed adaptive state backend spec.
 *
 * Produces the backend descriptor expected by the adaptive plugin when state
 * should be shared or persisted through Redis.
 *
 * @param url - Redis connection URL for the backend.
 * @param keyPrefix - Prefix applied to Redis keys.
 * @returns An adaptive backend spec using Redis storage.
 * @remarks The default key prefix namespaces runtime records under
 * `nemo_flow:` unless a different prefix is supplied.
 */
export declare function redisBackend(url: string, keyPrefix?: string): BackendSpec;
/**
 * Create adaptive telemetry settings with runtime defaults applied.
 *
 * Merges caller-supplied overrides onto the built-in telemetry config shape
 * used by the adaptive plugin.
 *
 * @param config - Partial telemetry settings to override.
 * @returns A normalized adaptive telemetry config object.
 * @remarks An empty `learners` array is supplied by default so callers can
 * append learner names without checking for initialization first.
 */
export declare function telemetryConfig(config?: TelemetryConfig): TelemetryConfig;
/**
 * Create adaptive hint-injection settings with defaults applied.
 *
 * Merges caller-supplied overrides onto the default config used by the
 * adaptive hints injector.
 *
 * @param config - Partial adaptive hints settings to override.
 * @returns A normalized adaptive hints config object.
 * @remarks By default the injector runs at priority `100`, preserves the rest
 * of the chain, and writes hints to `nvext.agent_hints`.
 */
export declare function adaptiveHintsConfig(config?: AdaptiveHintsConfig): AdaptiveHintsConfig;
/**
 * Create adaptive tool-parallelism settings with defaults applied.
 *
 * Merges caller-supplied overrides onto the scheduler config shape used by the
 * adaptive plugin's tool parallelism component.
 *
 * @param config - Partial tool scheduling settings to override.
 * @returns A normalized tool-parallelism config object.
 * @remarks The default mode is `observe_only`, so recommendations are produced
 * without changing execution behavior unless the caller opts in.
 */
export declare function toolParallelismConfig(config?: ToolParallelismConfig): ToolParallelismConfig;
/**
 * Create adaptive cache-governor settings with defaults applied.
 *
 * Merges caller-supplied overrides onto the Adaptive Cache Governor (ACG)
 * config shape used by the adaptive plugin's LLM execution intercept.
 *
 * @param config - Partial Adaptive Cache Governor (ACG) settings to override.
 * @returns A normalized adaptive cache-governor config object.
 * @remarks Nested `stability_thresholds` values are defaulted individually so
 * callers can override only the thresholds they need.
 */
export declare function acgConfig(config?: AcgConfig): AcgConfig;
/**
 * Wrap adaptive config as a top-level component.
 *
 * Produces the plugin component entry that can be inserted directly
 * into `plugin.defaultConfig().components`.
 *
 * @param config - Adaptive component configuration document.
 * @param options - Optional component-level flags.
 * @returns A plugin component spec for the adaptive plugin.
 * @remarks Setting `options.enabled = false` keeps the config in the plugin
 * document for validation while skipping runtime activation.
 */
export declare function ComponentSpec(
  config: Config,
  options?: {
    enabled?: boolean;
  },
): ComponentSpec;
