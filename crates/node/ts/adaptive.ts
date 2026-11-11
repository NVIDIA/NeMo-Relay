// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  CacheRequestFacts,
  CacheRequestFactsOptions,
  CacheTelemetryEvent,
  CacheTelemetryEventOptions,
  ConfigReport,
  Json,
  ScopeHandle,
} from '../binding.js';
import * as lib from './index.js';
import type { ConfigPolicy } from './plugin.js';

export type {
  AgentIdentity,
  CacheRequestFacts,
  CacheRequestFactsOptions,
  CacheTelemetryEvent,
  CacheTelemetryEventOptions,
  CacheUsage,
  CacheWriteTokenCounts,
  ConfigDiagnostic,
  ConfigReport,
} from '../binding.js';
export type { ConfigPolicy } from './plugin.js';

/** Response-cache key strategies. */
export const ResponseCacheKeyStrategy: { readonly ExactRequest: 'exact_request'; readonly Logical: 'logical' } =
  Object.freeze({
    ExactRequest: 'exact_request',
    Logical: 'logical',
  } as const);
export type ResponseCacheKeyStrategy = (typeof ResponseCacheKeyStrategy)[keyof typeof ResponseCacheKeyStrategy];

/** Adaptive state backend. */
export interface BackendSpec {
  kind: string;
  config?: Record<string, Json>;
}

/** Adaptive state configuration. */
export interface StateConfig {
  backend: BackendSpec;
}

/** Adaptive telemetry settings. */
export interface TelemetryConfig {
  subscriber_name?: string;
  learners?: string[];
}

/** Adaptive hints injection settings. */
export interface AdaptiveHintsConfig {
  priority?: number;
  break_chain?: boolean;
  inject_header?: boolean;
  inject_body_path?: string;
}

/** Adaptive tool scheduling settings. */
export interface ToolParallelismConfig {
  priority?: number;
  mode?: 'observe_only' | 'inject_hints' | 'schedule' | (string & {});
}

/** Prompt-stability thresholds for the cache governor. */
export interface AcgStabilityThresholds {
  stable_threshold?: number;
  semi_stable_threshold?: number;
  min_observations_for_full_confidence?: number;
}

/** Cache governor settings. */
export interface AcgConfig {
  provider?: 'anthropic' | 'openai' | 'passthrough' | (string & {});
  observation_window?: number;
  priority?: number;
  stability_thresholds?: AcgStabilityThresholds;
}

/** LLM response and tool-result cache settings. */
export interface ResponseCacheConfig {
  ttlSeconds?: number;
  /** Required cache trust domain; one cache must not span untrusted tenants or upstreams. */
  namespace?: string;
  priority?: number;
  bypassRate?: number;
  cacheNondeterministic?: boolean;
  keyStrategy?: ResponseCacheKeyStrategy;
  headerAllowlist?: string[];
  backend?: BackendSpec;
  /** Tool-result cache. */
  tools?: ToolCacheConfig;
}

export interface ResponseCachePluginConfig {
  ttl_seconds?: number;
  /** Required cache trust domain. */
  namespace?: string;
  priority?: number;
  bypass_rate?: number;
  cache_nondeterministic?: boolean;
  key_strategy?: ResponseCacheKeyStrategy;
  header_allowlist?: string[];
  backend?: BackendSpec;
  tools?: ToolCachePluginConfig;
}

/** Shared tool policy; omitted TTL and bypass rate use response-cache values. */
export interface ToolClass {
  cacheable?: boolean;
  ttlSeconds?: number;
  bypassRate?: number;
  toolVersion?: string;
  argSkip?: string[];
  /** Tool names, or `prefix*`, `*suffix`, or `*contains*` patterns. */
  members?: string[];
}

/** Per-tool override; `argSkip: []` clears the class list. */
export interface ToolOverride {
  cacheable?: boolean;
  ttlSeconds?: number;
  bypassRate?: number;
  toolVersion?: string;
  argSkip?: string[];
}

/** Caching for tools that are read-only and stable for their TTL. */
export interface ToolCacheConfig {
  enabled?: boolean;
  /** Tool execution-intercept priority, default 150. */
  priority?: number;
  /** Whether error results may be cached, default false. */
  cacheErrors?: boolean;
  default?: ToolClass;
  classes?: Record<string, ToolClass>;
  /** Tool names, or `prefix*`, `*suffix`, or `*contains*` patterns. */
  overrides?: Record<string, ToolOverride>;
}

export type ToolClassPluginConfig = Omit<ToolClass, 'ttlSeconds' | 'bypassRate' | 'toolVersion' | 'argSkip'> & {
  ttl_seconds?: number;
  bypass_rate?: number;
  tool_version?: string;
  arg_skip?: string[];
};

export type ToolOverridePluginConfig = Omit<ToolOverride, 'ttlSeconds' | 'bypassRate' | 'toolVersion' | 'argSkip'> & {
  ttl_seconds?: number;
  bypass_rate?: number;
  tool_version?: string;
  arg_skip?: string[];
};

export type ToolCachePluginConfig = Omit<ToolCacheConfig, 'cacheErrors' | 'default' | 'classes' | 'overrides'> & {
  cache_errors?: boolean;
  default?: ToolClassPluginConfig;
  classes?: Record<string, ToolClassPluginConfig>;
  overrides?: Record<string, ToolOverridePluginConfig>;
};

/** Adaptive component config. */
export interface Config {
  version?: number;
  agent_id?: string;
  state?: StateConfig;
  telemetry?: TelemetryConfig;
  adaptive_hints?: AdaptiveHintsConfig;
  tool_parallelism?: ToolParallelismConfig;
  acg?: AcgConfig;
  responseCache?: ResponseCacheConfig;
  policy?: ConfigPolicy;
}

export type AdaptivePluginConfig = Omit<Config, 'responseCache'> & {
  response_cache?: ResponseCachePluginConfig;
};

/** Adaptive component with kind `adaptive`. */
export interface ComponentSpec {
  kind: 'adaptive';
  enabled?: boolean;
  config: AdaptivePluginConfig;
}

export const ADAPTIVE_PLUGIN_KIND = 'adaptive';

/**
 * Creates a default adaptive config.
 *
 * The config has `version` 1 and no sections. Add state, telemetry, and
 * scheduler sections to it.
 *
 * @returns An adaptive config with no sections.
 * @remarks The config does nothing until it is wrapped with `ComponentSpec`
 * and activated.
 */
export function defaultConfig(): Config {
  return {
    version: 1,
  };
}

/**
 * Creates a backend spec that keeps adaptive state in process memory.
 *
 * The state is private to the process and is lost when it exits.
 *
 * @returns A backend spec with kind `in_memory`.
 * @remarks Use `redisBackend` when several processes must share state.
 */
export function inMemoryBackend(): BackendSpec {
  return {
    kind: 'in_memory',
    config: {},
  };
}

/**
 * Creates a backend spec that keeps adaptive state in Redis.
 *
 * Processes that use the same Redis instance and key prefix share the state.
 *
 * @param url - Redis connection URL.
 * @param keyPrefix - Prefix for every key the backend writes.
 * @returns A backend spec with kind `redis`.
 * @remarks `keyPrefix` defaults to `nemo_relay:`.
 */
export function redisBackend(url: string, keyPrefix: string = 'nemo_relay:'): BackendSpec {
  return {
    kind: 'redis',
    config: {
      url,
      key_prefix: keyPrefix,
    },
  };
}

/**
 * Creates adaptive telemetry settings.
 *
 * Fields in `config` replace the defaults; fields left out keep them.
 *
 * @param config - Telemetry settings to set.
 * @returns Complete telemetry settings.
 * @remarks `learners` defaults to an empty list.
 */
export function telemetryConfig(config: TelemetryConfig = {}): TelemetryConfig {
  return {
    learners: [],
    ...config,
  };
}

/**
 * Creates adaptive hint-injection settings.
 *
 * Fields in `config` replace the defaults; fields left out keep them.
 *
 * @param config - Hint-injection settings to set.
 * @returns Complete hint-injection settings.
 * @remarks By default the injector runs at priority 100, lets the chain
 * continue, and writes hints to `nvext.agent_hints`.
 */
export function adaptiveHintsConfig(config: AdaptiveHintsConfig = {}): AdaptiveHintsConfig {
  return {
    priority: 100,
    break_chain: false,
    inject_header: true,
    inject_body_path: 'nvext.agent_hints',
    ...config,
  };
}

/**
 * Creates adaptive tool-scheduling settings.
 *
 * Fields in `config` replace the defaults; fields left out keep them.
 *
 * @param config - Tool-scheduling settings to set.
 * @returns Complete tool-scheduling settings.
 * @remarks The default mode `observe_only` records recommendations and leaves
 * execution unchanged.
 */
export function toolParallelismConfig(config: ToolParallelismConfig = {}): ToolParallelismConfig {
  return {
    priority: 100,
    mode: 'observe_only',
    ...config,
  };
}

/**
 * Creates adaptive cache governor settings.
 *
 * Fields in `config` replace the defaults; fields left out keep them.
 *
 * @param config - Cache governor settings to set.
 * @returns Complete cache governor settings.
 * @remarks Thresholds in `stability_thresholds` default one by one, so one
 * threshold can be set without the others.
 */
export function acgConfig(config: AcgConfig = {}): AcgConfig {
  const { stability_thresholds, ...rest } = config;
  return {
    provider: 'passthrough',
    observation_window: 100,
    priority: 50,
    stability_thresholds: {
      stable_threshold: 0.95,
      semi_stable_threshold: 0.5,
      min_observations_for_full_confidence: 20,
      ...stability_thresholds,
    },
    ...rest,
  };
}

/**
 * Creates response-cache settings.
 *
 * Fields in `config` replace the defaults; fields left out keep them. The
 * result is one section of an adaptive config, not a plugin kind.
 *
 * @param config - Response-cache settings to set.
 * @returns Complete response-cache settings.
 * @remarks The defaults are an in-memory backend, a `bypassRate` of 0, no
 * caching of nondeterministic requests, and an empty `namespace`. Validation
 * rejects the empty `namespace`, so set it before validating.
 */
export function responseCacheConfig(config: ResponseCacheConfig = {}): ResponseCacheConfig {
  const { backend, ...rest } = config;
  return {
    ttlSeconds: 3600,
    namespace: '',
    priority: 150,
    bypassRate: 0,
    cacheNondeterministic: false,
    keyStrategy: ResponseCacheKeyStrategy.ExactRequest,
    headerAllowlist: [],
    backend: backend ?? inMemoryBackend(),
    ...rest,
  };
}

function mapValues<V, R>(record: Record<string, V>, map: (value: V) => R): Record<string, R> {
  return Object.fromEntries(Object.entries(record).map(([key, value]) => [key, map(value)]));
}

function toToolClassPluginConfig({
  ttlSeconds,
  bypassRate,
  toolVersion,
  argSkip,
  ...rest
}: ToolClass): ToolClassPluginConfig {
  return {
    ...rest,
    ...(ttlSeconds === undefined ? {} : { ttl_seconds: ttlSeconds }),
    ...(bypassRate === undefined ? {} : { bypass_rate: bypassRate }),
    ...(toolVersion === undefined ? {} : { tool_version: toolVersion }),
    ...(argSkip === undefined ? {} : { arg_skip: argSkip }),
  };
}

function toToolCachePluginConfig({
  cacheErrors,
  default: defaults,
  classes,
  overrides,
  ...rest
}: ToolCacheConfig): ToolCachePluginConfig {
  return {
    ...rest,
    ...(cacheErrors === undefined ? {} : { cache_errors: cacheErrors }),
    ...(defaults === undefined ? {} : { default: toToolClassPluginConfig(defaults) }),
    ...(classes === undefined ? {} : { classes: mapValues(classes, toToolClassPluginConfig) }),
    ...(overrides === undefined ? {} : { overrides: mapValues(overrides, toToolClassPluginConfig) }),
  };
}

function toResponseCachePluginConfig({
  ttlSeconds,
  bypassRate,
  cacheNondeterministic,
  keyStrategy,
  headerAllowlist,
  tools,
  ...rest
}: ResponseCacheConfig): ResponseCachePluginConfig {
  return {
    ...rest,
    ...(ttlSeconds === undefined ? {} : { ttl_seconds: ttlSeconds }),
    ...(bypassRate === undefined ? {} : { bypass_rate: bypassRate }),
    ...(cacheNondeterministic === undefined ? {} : { cache_nondeterministic: cacheNondeterministic }),
    ...(keyStrategy === undefined ? {} : { key_strategy: keyStrategy }),
    ...(headerAllowlist === undefined ? {} : { header_allowlist: headerAllowlist }),
    ...(tools === undefined ? {} : { tools: toToolCachePluginConfig(tools) }),
  };
}

function toPluginConfig({ responseCache, ...rest }: Config): AdaptivePluginConfig {
  return {
    ...rest,
    ...(responseCache === undefined ? {} : { response_cache: toResponseCachePluginConfig(responseCache) }),
  };
}

/** Adaptive runtime owned by the caller. */
export class AdaptiveRuntime {
  readonly #native: lib.AdaptiveRuntime;

  constructor(config: Config) {
    this.#native = lib.AdaptiveRuntime.create(toPluginConfig(config));
  }

  /** Registers configured runtime features. */
  register(): Promise<void> {
    return this.#native.register();
  }

  /** Removes registered runtime features. */
  deregister(): void {
    this.#native.deregister();
  }

  /** Shuts the runtime down for good. */
  shutdown(): Promise<void> {
    return this.#native.shutdown();
  }

  /** Blocks until telemetry has processed pending events. */
  waitForIdle(): void {
    this.#native.waitForIdle();
  }

  /** Validation report from construction. */
  report(): ConfigReport {
    return this.#native.report();
  }

  /** Binds cache governor request rewrites to a scope. */
  bindScope(scopeHandle: ScopeHandle): void {
    this.#native.bindScope(scopeHandle);
  }

  /** Builds cache request facts for an annotated LLM request. */
  buildCacheRequestFacts(options: CacheRequestFactsOptions): CacheRequestFacts | null {
    return this.#native.buildCacheRequestFacts(options);
  }
}

/**
 * Wraps an adaptive config as a plugin component.
 *
 * The result is an entry for `PluginConfig.components`.
 *
 * @param config - Adaptive config.
 * @param options - `enabled` controls whether the component activates.
 * @returns A component entry with kind `adaptive`.
 * @remarks With `enabled` false, `initialize` validates the component but does
 * not activate it.
 */
export function ComponentSpec(config: Config, { enabled = true }: { enabled?: boolean } = {}): ComponentSpec {
  return {
    kind: ADAPTIVE_PLUGIN_KIND,
    enabled,
    config: toPluginConfig(config),
  };
}

/** Validates adaptive config without creating a runtime. */
export function validateConfig(config: Config): ConfigReport {
  return lib.validateAdaptiveConfig(toPluginConfig(config));
}

/** Builds one cache telemetry event from token usage. */
export function buildCacheTelemetryEvent(options: CacheTelemetryEventOptions): CacheTelemetryEvent | null {
  return lib.buildCacheTelemetryEvent(options);
}

/** Sets latency sensitivity on the current scope. */
export function setLatencySensitivity(value: number): void {
  lib.setLatencySensitivity(value);
}
