// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * User-facing configuration parsing for the OpenClaw plugin.
 *
 * Keep defaults and validation here so runtime code can consume one normalized
 * config shape and avoid repeating defensive checks around optional plugin JSON.
 */
import type { OpenClawPluginConfigSchema } from 'openclaw/plugin-sdk/plugin-entry';

import manifest from '../openclaw.plugin.json' with { type: 'json' };

export type BackendKind = 'hooks';

export type CaptureConfig = {
  includePrompts: boolean;
  includeResponses: boolean;
  stripToolArgs: boolean;
  stripToolResults: boolean;
};

export type CorrelationConfig = {
  llmOutputGraceMs: number;
  recordTtlMs: number;
  maxRecordsPerKey: number;
};

export type NemoRelayPluginHostConfig = {
  version: number;
  components: unknown[];
  [key: string]: unknown;
};

export type NemoRelayHookBackendConfig = {
  enabled: boolean;
  backend: BackendKind;
  plugins: NemoRelayPluginHostConfig;
  capture: CaptureConfig;
  correlation: CorrelationConfig;
};

const DEFAULT_PLUGIN_HOST_CONFIG: NemoRelayPluginHostConfig = {
  version: 1,
  components: [],
};

export const NEMO_RELAY_OPENCLAW_JSON_SCHEMA = manifest.configSchema;

export const DEFAULT_CONFIG: NemoRelayHookBackendConfig = {
  enabled: true,
  backend: 'hooks',
  plugins: DEFAULT_PLUGIN_HOST_CONFIG,
  capture: {
    includePrompts: true,
    includeResponses: true,
    stripToolArgs: true,
    stripToolResults: true,
  },
  correlation: {
    llmOutputGraceMs: 250,
    recordTtlMs: 600_000,
    maxRecordsPerKey: 32,
  },
};

export const nemoRelayConfigSchema = {
  safeParse(value: unknown) {
    try {
      return { success: true, data: parseConfig(value) };
    } catch (error) {
      return {
        success: false,
        error: {
          issues: [
            {
              path: [],
              message: error instanceof Error ? error.message : String(error),
            },
          ],
        },
      };
    }
  },
  jsonSchema: NEMO_RELAY_OPENCLAW_JSON_SCHEMA,
} satisfies OpenClawPluginConfigSchema;

/** Parse OpenClaw plugin JSON into the normalized hook backend config. */
export function parseConfig(value: unknown): NemoRelayHookBackendConfig {
  const raw = asRecord(value, 'config', true);
  rejectRemovedFields(raw);
  rejectUnknownFields(raw, 'config', ['enabled', 'backend', 'plugins', 'capture', 'correlation']);
  const backend = optionalString(raw.backend, 'backend') ?? DEFAULT_CONFIG.backend;

  if (backend !== 'hooks') {
    throw new Error(`unsupported nemo-relay backend: ${backend}`);
  }

  const capture = asRecord(raw.capture, 'capture', true);
  const correlation = asRecord(raw.correlation, 'correlation', true);

  return {
    enabled: optionalBoolean(raw.enabled, 'enabled') ?? DEFAULT_CONFIG.enabled,
    backend,
    plugins: parsePluginHostConfig(raw.plugins),
    capture: {
      includePrompts:
        optionalBoolean(capture.includePrompts, 'capture.includePrompts') ?? DEFAULT_CONFIG.capture.includePrompts,
      includeResponses:
        optionalBoolean(capture.includeResponses, 'capture.includeResponses') ??
        DEFAULT_CONFIG.capture.includeResponses,
      stripToolArgs:
        optionalBoolean(capture.stripToolArgs, 'capture.stripToolArgs') ?? DEFAULT_CONFIG.capture.stripToolArgs,
      stripToolResults:
        optionalBoolean(capture.stripToolResults, 'capture.stripToolResults') ??
        DEFAULT_CONFIG.capture.stripToolResults,
    },
    correlation: {
      llmOutputGraceMs:
        optionalNonNegativeInteger(correlation.llmOutputGraceMs, 'correlation.llmOutputGraceMs') ??
        DEFAULT_CONFIG.correlation.llmOutputGraceMs,
      recordTtlMs:
        optionalNonNegativeInteger(correlation.recordTtlMs, 'correlation.recordTtlMs') ??
        DEFAULT_CONFIG.correlation.recordTtlMs,
      maxRecordsPerKey:
        optionalPositiveInteger(correlation.maxRecordsPerKey, 'correlation.maxRecordsPerKey') ??
        DEFAULT_CONFIG.correlation.maxRecordsPerKey,
    },
  };
}

/** Normalize the optional generic NeMo Relay plugin-host config embedded in OpenClaw config. */
function parsePluginHostConfig(value: unknown): NemoRelayPluginHostConfig {
  if (value === undefined) {
    return clonePluginHostConfig(DEFAULT_PLUGIN_HOST_CONFIG);
  }
  const record = asRecord(value, 'plugins', false);
  const version = optionalNumber(record.version, 'plugins.version') ?? 1;
  const components = record.components === undefined ? [] : record.components;

  if (!Array.isArray(components)) {
    throw new Error('plugins.components must be an array');
  }

  return {
    ...record,
    version,
    components: [...components],
  };
}

/** Clone the mutable plugin-host component list before putting it in runtime state. */
function clonePluginHostConfig(config: NemoRelayPluginHostConfig): NemoRelayPluginHostConfig {
  return {
    ...config,
    components: [...config.components],
  };
}

/** Require an object config section, optionally treating undefined as an empty object. */
function asRecord(value: unknown, path: string, optional: boolean): Record<string, unknown> {
  if (value === undefined && optional) {
    return {};
  }
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  throw new Error(`${path} must be an object`);
}

/** Reject config fields removed by the generic plugin-host pivot with direct migration hints. */
function rejectRemovedFields(raw: Record<string, unknown>): void {
  if (raw.nemoRelay !== undefined) {
    throw new Error('nemoRelay.pluginConfig was removed; use top-level plugins instead');
  }
  if (raw.atif !== undefined) {
    throw new Error('atif was removed; configure plugins.components[].config.atif on the observability component');
  }
  if (raw.telemetry !== undefined) {
    throw new Error(
      'telemetry was removed; configure plugins.components[].config.opentelemetry or openinference on the observability component',
    );
  }
}

/** Keep parser behavior aligned with the manifest's additionalProperties=false contract. */
function rejectUnknownFields(raw: Record<string, unknown>, path: string, allowed: string[]): void {
  const allowedSet = new Set(allowed);
  for (const key of Object.keys(raw)) {
    if (!allowedSet.has(key)) {
      throw new Error(`${path}.${key} is not supported`);
    }
  }
}

/** Parse an optional boolean while producing config-path-specific error messages. */
function optionalBoolean(value: unknown, path: string): boolean | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'boolean') {
    throw new Error(`${path} must be a boolean`);
  }
  return value;
}

/** Parse an optional finite number while preserving undefined for default fallback. */
function optionalNumber(value: unknown, path: string): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${path} must be a finite number`);
  }
  return value;
}

/** Parse an optional integer where zero is valid, such as timeouts. */
function optionalNonNegativeInteger(value: unknown, path: string): number | undefined {
  const parsed = optionalNumber(value, path);
  if (parsed === undefined) {
    return undefined;
  }
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${path} must be a non-negative integer`);
  }
  return parsed;
}

/** Parse an optional integer where zero would disable required bounded storage. */
function optionalPositiveInteger(value: unknown, path: string): number | undefined {
  const parsed = optionalNumber(value, path);
  if (parsed === undefined) {
    return undefined;
  }
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${path} must be a positive integer`);
  }
  return parsed;
}

/** Parse an optional string while rejecting accidental non-string config values. */
function optionalString(value: unknown, path: string): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'string') {
    throw new Error(`${path} must be a string`);
  }
  return value;
}
