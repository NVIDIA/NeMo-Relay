// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Health snapshot construction for the plugin gateway status method.
 *
 * Runtime state owns status transitions; this file turns that state into a
 * stable, JSON-friendly status payload for operators and tests.
 */
import type { NemoRelayHookBackendConfig } from './config.js';
import type { HookReplayBackendState } from './hook-replay/session.js';

export type HookReplayBackendStatus =
  | { state: 'not_initialized'; reason?: string }
  | { state: 'disabled'; reason?: string }
  | { state: 'ready' }
  | { state: 'degraded'; reason: string }
  | { state: 'stopping' }
  | { state: 'stopped'; reason?: string };

export type OutputHealthState = 'enabled' | 'disabled' | 'degraded';

export type NemoRelayHealthSnapshot = {
  id: 'nemo-relay';
  backend: 'hooks';
  status: HookReplayBackendStatus;
  initializedPluginHost: boolean;
  state: HookReplayBackendStatus['state'];
  outputs: {
    atif: OutputHealthState;
    otel: OutputHealthState;
    openInference: OutputHealthState;
  };
  counters: HookReplayBackendState['counters'];
  lastError?: string;
};

/** Build a complete health payload from runtime status, configured outputs, and counters. */
export function createHealthSnapshot(params: {
  status: HookReplayBackendStatus;
  initializedPluginHost: boolean;
  pluginHostOutputsHealthy: boolean;
  config: NemoRelayHookBackendConfig;
  counters?: HookReplayBackendState['counters'];
}): NemoRelayHealthSnapshot {
  const lastError = 'reason' in params.status ? params.status.reason : undefined;
  const outputs = configuredObservabilityOutputs(params.config);
  const pluginHostFailed = params.status.state === 'degraded' && !params.pluginHostOutputsHealthy;
  return {
    id: 'nemo-relay',
    backend: 'hooks',
    status: params.status,
    initializedPluginHost: params.initializedPluginHost,
    state: params.status.state,
    outputs: {
      atif: outputHealth(outputs.atif, pluginHostFailed),
      otel: outputHealth(outputs.otel, pluginHostFailed),
      openInference: outputHealth(outputs.openInference, pluginHostFailed),
    },
    counters: params.counters ?? emptyCounters(),
    ...(lastError === undefined ? {} : { lastError }),
  };
}

function outputHealth(enabled: boolean, pluginHostFailed: boolean): OutputHealthState {
  if (!enabled) {
    return 'disabled';
  }
  return pluginHostFailed ? 'degraded' : 'enabled';
}

/** Inspect generic PluginConfig components for configured observability outputs. */
function configuredObservabilityOutputs(config: NemoRelayHookBackendConfig): {
  atif: boolean;
  otel: boolean;
  openInference: boolean;
} {
  const outputs = { atif: false, otel: false, openInference: false };

  for (const component of config.plugins.components) {
    const record = asRecord(component);
    if (record?.kind !== 'observability' || record.enabled === false) {
      continue;
    }

    const componentConfig = asRecord(record.config);
    outputs.atif ||= sectionEnabled(componentConfig?.atif);
    outputs.otel ||= sectionEnabled(componentConfig?.opentelemetry);
    outputs.openInference ||= sectionEnabled(componentConfig?.openinference);
  }

  return outputs;
}

function sectionEnabled(value: unknown): boolean {
  const section = asRecord(value);
  return section?.enabled === true;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return undefined;
}

/** Provide zero counters before hook replay has initialized. */
function emptyCounters(): HookReplayBackendState['counters'] {
  return {
    llmSpansReplayed: 0,
    toolSpansReplayed: 0,
    marksEmitted: 0,
    replayErrors: 0,
    skippedEvents: 0,
  };
}
