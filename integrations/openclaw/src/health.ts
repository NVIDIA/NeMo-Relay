// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Health snapshot construction for the plugin gateway status method.
 *
 * Runtime state owns status transitions; this file turns that state into a
 * stable, JSON-friendly status payload for operators and tests.
 */
import type { NemoFlowHookBackendConfig } from "./config.js";
import type { HookReplayBackendState, SessionState } from "./hook-replay/session.js";

export type HookReplayBackendStatus =
  | { state: "not_initialized"; reason?: string }
  | { state: "disabled"; reason?: string }
  | { state: "ready" }
  | { state: "degraded"; reason: string }
  | { state: "stopping" }
  | { state: "stopped"; reason?: string };

export type OutputHealthState = "enabled" | "disabled" | "degraded";

export type NemoFlowHealthSnapshot = {
  id: "nemo-flow";
  backend: "hooks";
  status: HookReplayBackendStatus;
  initializedPluginHost: boolean;
  state: HookReplayBackendStatus["state"];
  outputs: {
    atif: OutputHealthState;
    otel: OutputHealthState;
    openInference: OutputHealthState;
  };
  counters: HookReplayBackendState["counters"];
  lastError?: string;
};

/** Build a complete health payload from runtime status, outputs, and counters. */
export function createHealthSnapshot(params: {
  status: HookReplayBackendStatus;
  initializedPluginHost: boolean;
  config: NemoFlowHookBackendConfig;
  degradedOutputs: ReadonlySet<"atif" | "otel" | "openInference">;
  counters?: HookReplayBackendState["counters"];
  sessions?: Iterable<SessionState>;
}): NemoFlowHealthSnapshot {
  const lastError = "reason" in params.status ? params.status.reason : undefined;
  return {
    id: "nemo-flow",
    backend: "hooks",
    status: params.status,
    initializedPluginHost: params.initializedPluginHost,
    state: params.status.state,
    outputs: {
      atif: atifOutputHealth(params.config, params.degradedOutputs, params.sessions),
      otel: telemetryOutputHealth(params.config.telemetry.otel.enabled, params.degradedOutputs.has("otel")),
      openInference: telemetryOutputHealth(
        params.config.telemetry.openInference.enabled,
        params.degradedOutputs.has("openInference"),
      ),
    },
    counters: params.counters ?? emptyCounters(),
    ...(lastError === undefined ? {} : { lastError }),
  };
}

/** Report ATIF degradation from config, global output state, or active sessions. */
function atifOutputHealth(
  config: NemoFlowHookBackendConfig,
  degradedOutputs: ReadonlySet<"atif" | "otel" | "openInference">,
  sessions: Iterable<SessionState> | undefined,
): OutputHealthState {
  if (!config.atif.enabled) {
    return "disabled";
  }
  if (degradedOutputs.has("atif")) {
    return "degraded";
  }
  for (const session of sessions ?? []) {
    if (session.atif?.disabled || session.atif?.leakedRegistration) {
      return "degraded";
    }
  }
  return "enabled";
}

/** Report telemetry sink health using the sink's enabled and degraded states. */
function telemetryOutputHealth(enabled: boolean, degraded: boolean): OutputHealthState {
  if (!enabled) {
    return "disabled";
  }
  return degraded ? "degraded" : "enabled";
}

/** Provide zero counters before hook replay has initialized. */
function emptyCounters(): HookReplayBackendState["counters"] {
  return {
    llmSpansReplayed: 0,
    toolSpansReplayed: 0,
    marksEmitted: 0,
    atifFilesWritten: 0,
    replayErrors: 0,
    skippedEvents: 0,
  };
}
