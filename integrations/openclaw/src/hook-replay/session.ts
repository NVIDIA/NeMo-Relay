// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NemoFlowHookBackendConfig } from "../config.js";
import { createAtifExporter } from "../atif-capture.js";
import { evictExpiredRecords, tupleKey as tupleKeyFromCorrelation } from "./correlation.js";
import type { AtifExporterLike } from "../modules.js";
import type {
  PluginHookAgentContext,
  PluginHookLlmOutputEvent,
  PluginHookModelCallEndedEvent,
} from "../openclaw-hook-types.js";
import type { PluginLogger } from "openclaw/plugin-sdk/plugin-entry";
import type { JsonObject as JsonRecord } from "nemo-flow-node/typed";
import type { NemoFlowRuntimeModule } from "../modules.js";

export type SessionLookupInput = {
  sessionId?: string | undefined;
  sessionKey?: string | undefined;
  runId?: string | undefined;
  childSessionKey?: string | undefined;
  requesterSessionKey?: string | undefined;
};

export type EnsureSessionInput = SessionLookupInput & {
  agentId?: string | undefined;
  source: "session_start" | "lazy_session";
  resumedFrom?: string | undefined;
  timestamp?: number | undefined;
};

export type SessionState = {
  sessionId: string;
  sessionKey?: string;
  agentId?: string;
  source: "session_start" | "lazy_session";
  resumedFrom?: string;
  finalOutput?: JsonRecord;
  trajectoryReplayedRuns?: Set<string>;
  hookLlmOutputReplayCounts?: Map<string, number>;
  agentRunInputSnapshots?: Map<
    string,
    { historyMessageCount: number; historyMessages: unknown[]; observedAtMs: number; prompt: string }
  >;
  messageWrites?: unknown[];
  assistantMessageWrites?: AssistantMessageRecord[];
  stack: ReturnType<NemoFlowRuntimeModule["createScopeStack"]>;
  rootHandle?: ReturnType<NemoFlowRuntimeModule["pushScope"]>;
  atif?: {
    exporter: AtifExporterLike;
    registrationName: string;
    capturing: boolean;
    registeredOnce?: boolean;
    disabled?: boolean;
    leakedRegistration?: boolean;
  };
};

export type PendingLlmOutputRecord = {
  sessionKey: string;
  sessionId: string;
  runId: string;
  provider: string;
  model: string;
  event: PluginHookLlmOutputEvent;
  ctx: PluginHookAgentContext;
  observedAtMs: number;
  timer?: ReturnType<typeof setTimeout> | undefined;
};

export type LlmInputRecord = {
  sessionKey: string;
  sessionId: string;
  runId: string;
  provider: string;
  model: string;
  prompt: string;
  historyMessages: unknown[];
  imagesCount: number;
  observedAtMs: number;
  systemPrompt?: string | undefined;
  placeholderRequest?: boolean | undefined;
};

export type AssistantMessageRecord = {
  sessionKey: string;
  provider: string;
  model: string;
  assistantTexts: string[];
  assistantToolCalls: unknown[];
  historyMessages: unknown[];
  prompt: string;
  observedAtMs: number;
  replayed: boolean;
  usage?: unknown;
};

export type ModelCallRecord = {
  sessionKey: string;
  runId: string;
  callId: string;
  provider: string;
  model: string;
  consumed: boolean;
  observedAtMs: number;
  sessionId?: string | undefined;
  api?: string | undefined;
  transport?: string | undefined;
  startedAtMs?: number | undefined;
  endedAtMs?: number | undefined;
  durationMs?: number | undefined;
  outcome?: PluginHookModelCallEndedEvent["outcome"] | undefined;
  errorCategory?: string | undefined;
  failureKind?: PluginHookModelCallEndedEvent["failureKind"] | undefined;
  requestPayloadBytes?: number | undefined;
  responseStreamBytes?: number | undefined;
  timeToFirstByteMs?: number | undefined;
  upstreamRequestIdHash?: string | undefined;
  ambiguous?: boolean | undefined;
};

export type HookReplayCounters = {
  llmSpansReplayed: number;
  toolSpansReplayed: number;
  marksEmitted: number;
  atifFilesWritten: number;
  replayErrors: number;
  skippedEvents: number;
};

export type HookReplayBackendState = {
  sessions: Map<string, SessionState>;
  sessionAliases: Map<string, string>;
  llmInputs: Map<string, LlmInputRecord[]>;
  llmOutputsPendingInput: Map<string, PendingLlmOutputRecord[]>;
  modelCallsByCallId: Map<string, ModelCallRecord[]>;
  modelTimingsByLlmKey: Map<string, ModelCallRecord[]>;
  counters: HookReplayCounters;
};

export type SessionManager = {
  nf: NemoFlowRuntimeModule;
  config: NemoFlowHookBackendConfig;
  logger: PluginLogger;
  state: HookReplayBackendState;
  agentVersion: string;
  resolvedAtifOutputDir: string;
  emitCapturedUnderSession: (label: string, session: SessionState, emit: () => void) => void;
  replayPendingLlmOutputsForSession: (
    session: SessionState,
    options: { allowPlaceholderRequest: boolean },
  ) => void;
  emitUnpairedModelCallTimingMarks: (session: SessionState) => void;
  markOutputDegraded: (output: "atif" | "otel" | "openInference") => void;
  logBoundedWarn: (key: string, message: string) => void;
};

export function lookupSessionKeys(input: SessionLookupInput): string[] {
  return [input.sessionId, input.sessionKey, input.requesterSessionKey, input.childSessionKey, input.runId].filter(
    (value): value is string => typeof value === "string" && value.length > 0,
  );
}

export function aliasSessionKeys(input: SessionLookupInput): string[] {
  return [input.sessionId, input.sessionKey, input.requesterSessionKey, input.runId].filter(
    (value): value is string => typeof value === "string" && value.length > 0,
  );
}

export function resolveSessionKey(
  state: HookReplayBackendState,
  input: SessionLookupInput,
): string | undefined {
  for (const key of lookupSessionKeys(input)) {
    const canonical = state.sessionAliases.get(key);
    if (canonical) {
      return canonical;
    }
  }

  return input.sessionId ?? input.sessionKey ?? input.childSessionKey ?? input.runId;
}

export function rememberSessionAliases(
  state: HookReplayBackendState,
  session: SessionState,
  input: SessionLookupInput,
): void {
  for (const alias of aliasSessionKeys(input)) {
    state.sessionAliases.set(alias, session.sessionId);
  }
}

export function createHookReplayState(): HookReplayBackendState {
  return {
    sessions: new Map(),
    sessionAliases: new Map(),
    llmInputs: new Map(),
    llmOutputsPendingInput: new Map(),
    modelCallsByCallId: new Map(),
    modelTimingsByLlmKey: new Map(),
    counters: {
      llmSpansReplayed: 0,
      toolSpansReplayed: 0,
      marksEmitted: 0,
      atifFilesWritten: 0,
      replayErrors: 0,
      skippedEvents: 0,
    },
  };
}

export function ensureSession(manager: SessionManager, input: EnsureSessionInput): SessionState | undefined {
  const key = resolveSessionKey(manager.state, input);
  if (!key) {
    manager.state.counters.skippedEvents += 1;
    manager.logBoundedWarn("missing-session-key", "nemo-flow skipped replay because no session/run key was available");
    return undefined;
  }

  const existing = manager.state.sessions.get(key);
  if (existing) {
    rememberSessionAliases(manager.state, existing, input);
    return existing;
  }

  const canonicalSessionId = input.sessionId ?? key;
  const aliased = manager.state.sessions.get(canonicalSessionId);
  if (aliased) {
    rememberSessionAliases(manager.state, aliased, input);
    return aliased;
  }

  const stack = manager.nf.createScopeStack();
  const session: SessionState = {
    sessionId: canonicalSessionId,
    source: input.source,
    stack,
  };

  if (input.sessionKey !== undefined) {
    session.sessionKey = input.sessionKey;
  }
  if (input.agentId !== undefined) {
    session.agentId = input.agentId;
  }
  if (input.resumedFrom !== undefined) {
    session.resumedFrom = input.resumedFrom;
  }

  createAtifExporter(manager, session);
  openSessionRoot(manager, session, input);
  manager.state.sessions.set(session.sessionId, session);
  rememberSessionAliases(manager.state, session, input);
  return session;
}

export function drainSession(manager: SessionManager, session: SessionState): void {
  cancelPendingLlmOutputTimers(manager.state, session);
  manager.replayPendingLlmOutputsForSession(session, { allowPlaceholderRequest: true });
  manager.emitUnpairedModelCallTimingMarks(session);
  evictSessionCorrelationRecords(manager.state, session);
}

export function closeSessionRoot(
  manager: SessionManager,
  session: SessionState,
  summary: JsonRecord,
  rootOutput: JsonRecord = summary,
  timestamp?: number,
): void {
  manager.emitCapturedUnderSession("session_end", session, () => {
    if (!session.rootHandle) {
      return;
    }

    manager.nf.event("openclaw.session_end", session.rootHandle, summary, null, timestamp ?? null);
    manager.state.counters.marksEmitted += 1;
    manager.nf.popScope(session.rootHandle, rootOutput, timestamp ?? null);
    delete session.rootHandle;
  });
}

export function deleteSession(state: HookReplayBackendState, session: SessionState): void {
  state.sessions.delete(session.sessionId);
}

export function insertBoundedRecord<T>(
  map: Map<string, T[]>,
  key: string,
  record: T,
  maxRecordsPerKey: number,
): void {
  const records = map.get(key) ?? [];
  records.push(record);
  while (records.length > maxRecordsPerKey) {
    records.shift();
  }
  map.set(key, records);
}

export function tupleKey(parts: Array<string | undefined>): string {
  return tupleKeyFromCorrelation(parts);
}

export function evictExpiredCorrelationRecords(state: HookReplayBackendState, nowMs: number, ttlMs: number): void {
  evictExpiredRecords(state.llmInputs, nowMs, ttlMs);
  evictExpiredPendingLlmOutputs(state.llmOutputsPendingInput, nowMs, ttlMs);
  evictExpiredRecords(state.modelCallsByCallId, nowMs, ttlMs);
  evictExpiredRecords(state.modelTimingsByLlmKey, nowMs, ttlMs);
}

function openSessionRoot(manager: SessionManager, session: SessionState, input: EnsureSessionInput): void {
  const data: JsonRecord = {
    sessionId: session.sessionId,
    source: session.source,
    ...(session.sessionKey === undefined ? {} : { sessionKey: session.sessionKey }),
    ...(session.agentId === undefined ? {} : { agentId: session.agentId }),
    ...(input.runId === undefined ? {} : { runId: input.runId }),
    ...(session.resumedFrom === undefined ? {} : { resumedFrom: session.resumedFrom }),
  };

  manager.emitCapturedUnderSession("session_start", session, () => {
    session.rootHandle = manager.nf.pushScope(
      "openclaw.session",
      agentScopeType(manager.nf),
      null,
      null,
      data,
      null,
      null,
      input.timestamp ?? null,
    );
    manager.nf.event("openclaw.session_start", session.rootHandle, data, null, input.timestamp ?? null);
    manager.state.counters.marksEmitted += 1;
  });
}

function cancelPendingLlmOutputTimers(state: HookReplayBackendState, session: SessionState): void {
  for (const records of state.llmOutputsPendingInput.values()) {
    for (const record of records) {
      if (record.sessionKey === session.sessionId && record.timer) {
        clearTimeout(record.timer);
        record.timer = undefined;
      }
    }
  }
}

function evictSessionCorrelationRecords(state: HookReplayBackendState, session: SessionState): void {
  evictFromRecordMap(state.llmInputs, session.sessionId);
  evictFromRecordMap(state.llmOutputsPendingInput, session.sessionId);
  evictFromRecordMap(state.modelCallsByCallId, session.sessionId);
  evictFromRecordMap(state.modelTimingsByLlmKey, session.sessionId);

  for (const [alias, canonical] of state.sessionAliases) {
    if (canonical === session.sessionId || alias === session.sessionId) {
      state.sessionAliases.delete(alias);
    }
  }
}

function evictFromRecordMap<T extends { sessionKey: string }>(map: Map<string, T[]>, sessionKey: string): void {
  for (const [key, records] of map) {
    const retained = records.filter((record) => record.sessionKey !== sessionKey);
    if (retained.length === 0) {
      map.delete(key);
    } else {
      map.set(key, retained);
    }
  }
}

function evictExpiredPendingLlmOutputs(
  map: Map<string, PendingLlmOutputRecord[]>,
  nowMs: number,
  ttlMs: number,
): void {
  for (const [key, records] of map) {
    const retained: PendingLlmOutputRecord[] = [];
    for (const record of records) {
      if (nowMs - record.observedAtMs <= ttlMs) {
        retained.push(record);
        continue;
      }
      if (record.timer) {
        clearTimeout(record.timer);
        record.timer = undefined;
      }
    }
    if (retained.length === 0) {
      map.delete(key);
    } else {
      map.set(key, retained);
    }
  }
}

function agentScopeType(nf: NemoFlowRuntimeModule): Parameters<NemoFlowRuntimeModule["pushScope"]>[1] {
  return (nf.ScopeType?.Agent ?? 0) as Parameters<NemoFlowRuntimeModule["pushScope"]>[1];
}
