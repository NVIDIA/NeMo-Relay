// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Runtime health: the coarse state, plugin-host outputs, and counters.
 *
 * Runtime state owns status transitions; this file turns that state into a
 * stable, JSON-friendly payload that `status.ts` extends into the status file.
 */
import type { NemoRelayDeepSeekHarnessConfig } from './config.js';

export type RuntimeStatus =
  | { state: 'not_initialized'; reason?: string }
  | { state: 'disabled'; reason?: string }
  | { state: 'ready' }
  | { state: 'degraded'; reason: string }
  | { state: 'stopping' }
  | { state: 'stopped'; reason?: string };

export type OutputHealthState = 'enabled' | 'disabled' | 'degraded';

export type RuntimeCounters = {
  turnsOpened: number;
  turnsClosed: number;
  llmCalls: number;
  toolCalls: number;
  guardrailRejections: number;
  marksEmitted: number;
  middlewareErrors: number;
};

export type NemoRelayHealthSnapshot = {
  id: 'nemo-relay';
  host: 'deepseek-harness';
  status: RuntimeStatus;
  state: RuntimeStatus['state'];
  initializedPluginHost: boolean;
  configPaths: string[];
  outputs: {
    atof: OutputHealthState;
    atif: OutputHealthState;
    otel: OutputHealthState;
    openInference: OutputHealthState;
  };
  counters: RuntimeCounters;
  activeSessions: number;
  lastError?: string;
};

/** Provide zero counters before the runtime starts. */
export function emptyCounters(): RuntimeCounters {
  return {
    turnsOpened: 0,
    turnsClosed: 0,
    llmCalls: 0,
    toolCalls: 0,
    guardrailRejections: 0,
    marksEmitted: 0,
    middlewareErrors: 0,
  };
}

/** Build a complete health payload from runtime status, configured outputs, and counters. */
export function createHealthSnapshot(params: {
  status: RuntimeStatus;
  initializedPluginHost: boolean;
  pluginHostOutputsHealthy: boolean;
  config: NemoRelayDeepSeekHarnessConfig;
  resolvedConfig?: unknown;
  configPaths?: string[];
  counters?: RuntimeCounters;
  activeSessions?: number;
}): NemoRelayHealthSnapshot {
  const lastError = 'reason' in params.status ? params.status.reason : undefined;
  // Prefer the merged document the plugin host resolved (system and user plugins.toml layers).
  const outputs = configuredObservabilityOutputs(params.resolvedConfig ?? params.config.plugins);
  const pluginHostFailed = params.status.state === 'degraded' && !params.pluginHostOutputsHealthy;
  return {
    id: 'nemo-relay',
    host: 'deepseek-harness',
    status: params.status,
    state: params.status.state,
    initializedPluginHost: params.initializedPluginHost,
    configPaths: params.configPaths ?? [],
    outputs: {
      atof: outputHealth(outputs.atof, pluginHostFailed),
      atif: outputHealth(outputs.atif, pluginHostFailed),
      otel: outputHealth(outputs.otel, pluginHostFailed),
      openInference: outputHealth(outputs.openInference, pluginHostFailed),
    },
    counters: params.counters ?? emptyCounters(),
    activeSessions: params.activeSessions ?? 0,
    ...(lastError === undefined ? {} : { lastError }),
  };
}

function outputHealth(enabled: boolean, pluginHostFailed: boolean): OutputHealthState {
  if (!enabled) {
    return 'disabled';
  }
  return pluginHostFailed ? 'degraded' : 'enabled';
}

/** Inspect a generic plugin document for configured observability outputs. */
function configuredObservabilityOutputs(document: unknown): {
  atof: boolean;
  atif: boolean;
  otel: boolean;
  openInference: boolean;
} {
  const outputs = { atof: false, atif: false, otel: false, openInference: false };
  const components = asRecord(document)?.components;
  if (!Array.isArray(components)) {
    return outputs;
  }

  for (const component of components) {
    const record = asRecord(component);
    if (record?.kind !== 'observability' || record.enabled === false) {
      continue;
    }
    const componentConfig = asRecord(record.config);
    outputs.atof ||= sectionEnabled(componentConfig?.atof);
    outputs.atif ||= sectionEnabled(componentConfig?.atif);
    const opentelemetry = asRecord(componentConfig?.opentelemetry);
    const opentelemetryEnabled = sectionEnabled(opentelemetry);
    outputs.otel ||= opentelemetryEnabled;
    outputs.openInference ||=
      opentelemetryEnabled &&
      Array.isArray(opentelemetry?.endpoints) &&
      opentelemetry.endpoints.some((endpoint) => asRecord(endpoint)?.type === 'openinference');
  }

  return outputs;
}

function sectionEnabled(value: unknown): boolean {
  return asRecord(value)?.enabled === true;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return undefined;
}
