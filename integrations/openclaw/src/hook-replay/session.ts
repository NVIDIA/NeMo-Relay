// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Session identity and root-span management for hook replay.
 *
 * OpenClaw can reference the same conversation by session id, session key, run id,
 * requester key, or child key depending on the hook. This module canonicalizes
 * those identifiers and owns the root `openclaw.session` scope lifecycle.
 */
import type { NemoRelayHookBackendConfig } from '../config.js';
import {
  evictExpiredRecords,
  tupleKey as tupleKeyFromCorrelation,
} from './correlation.js';
import type {
  PluginHookAgentContext,
  PluginHookLlmOutputEvent,
  PluginHookModelCallEndedEvent,
} from '../openclaw-hook-types.js';
import type { PluginLogger } from 'openclaw/plugin-sdk/plugin-entry';
import type { JsonObject as JsonRecord } from 'nemo-relay-node/typed';
import type { NemoRelayRuntimeModule } from '../modules.js';
import { toJsonRecord } from './marks.js';

export type SessionLookupInput = {
  sessionId?: string | undefined;
  sessionKey?: string | undefined;
  runId?: string | undefined;
  childSessionKey?: string | undefined;
  requesterSessionKey?: string | undefined;
};

export type EnsureSessionInput = SessionLookupInput & {
  agentId?: string | undefined;
  parentHandle?: ReturnType<NemoRelayRuntimeModule['pushScope']> | undefined;
  scopeRole?: 'subagent' | undefined;
  source: 'session_start' | 'lazy_session';
  resumedFrom?: string | undefined;
  timestamp?: number | undefined;
  deferRootOpen?: boolean | undefined;
};

export type SessionState = {
  /** Immutable internal owner key used for replay buffers and alias lookups. */
  ownerKey: string;
  sessionId: string;
  sessionKey?: string;
  agentId?: string;
  source: 'session_start' | 'lazy_session';
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
  stack: ReturnType<NemoRelayRuntimeModule['createScopeStack']>;
  rootHandle?: ReturnType<NemoRelayRuntimeModule['pushScope']>;
  scopeRole?: 'subagent';
  pendingRootOpen?: boolean;
  pendingRootTimestampMicros?: number;
  pendingRootRunId?: string;
  pendingCapturedEmits?: Array<{ label: string; emit: () => void }>;
};

export type PendingLlmOutputRecord = {
  sessionOwnerKey: string;
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
  sessionOwnerKey: string;
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
  sessionOwnerKey: string;
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
  sessionOwnerKey: string;
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
  outcome?: PluginHookModelCallEndedEvent['outcome'] | undefined;
  errorCategory?: string | undefined;
  failureKind?: PluginHookModelCallEndedEvent['failureKind'] | undefined;
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
  nf: NemoRelayRuntimeModule;
  config: NemoRelayHookBackendConfig;
  logger: PluginLogger;
  state: HookReplayBackendState;
  agentVersion: string;
  emitCapturedUnderSession: (label: string, session: SessionState, emit: () => void) => void;
  replayPendingLlmOutputsForSession: (session: SessionState, options: { allowPlaceholderRequest: boolean }) => void;
  emitUnpairedModelCallTimingMarks: (session: SessionState) => void;
  logBoundedWarn: (key: string, message: string) => void;
  resolveSessionRootContext?: (input: EnsureSessionInput) => Partial<EnsureSessionInput> | undefined;
};

/** Merge lineage/root-context defaults without letting undefined hook fields erase them. */
function mergeEnsureSessionInput(
  input: EnsureSessionInput,
  context: Partial<EnsureSessionInput> | undefined,
): EnsureSessionInput {
  if (!context) {
    return input;
  }

  const merged: Record<string, unknown> = { ...context, ...input };
  for (const [key, value] of Object.entries(context)) {
    if (merged[key] === undefined && value !== undefined) {
      merged[key] = value;
    }
  }
  return merged as EnsureSessionInput;
}

/** Return the session key that belongs to the session itself, not a requester alias. */
function ownSessionKey(input: SessionLookupInput): string | undefined {
  return input.sessionKey ?? input.childSessionKey;
}

/** Return all keys that may identify an existing OpenClaw session. */
export function lookupSessionKeys(input: SessionLookupInput): string[] {
  return [input.sessionId, input.sessionKey, input.requesterSessionKey, input.childSessionKey, input.runId].filter(
    (value): value is string => typeof value === 'string' && value.length > 0,
  );
}

/** Return keys that should alias to one stable internal session owner. */
export function aliasSessionKeys(input: SessionLookupInput): string[] {
  return [input.sessionId, input.sessionKey, input.childSessionKey, input.requesterSessionKey, input.runId].filter(
    (value): value is string => typeof value === 'string' && value.length > 0,
  );
}

/** Resolve a hook's session identity to the stable owner key used in replay state. */
export function resolveSessionOwnerKey(state: HookReplayBackendState, input: SessionLookupInput): string | undefined {
  for (const key of lookupSessionKeys(input)) {
    const canonical = state.sessionAliases.get(key);
    if (canonical) {
      return canonical;
    }
  }

  return input.sessionId ?? input.sessionKey ?? input.childSessionKey ?? input.runId;
}

/** Remember equivalent hook identifiers so later events attach to the same owner and root span. */
export function rememberSessionAliases(
  state: HookReplayBackendState,
  session: SessionState,
  input: SessionLookupInput,
): void {
  for (const alias of aliasSessionKeys(input)) {
    state.sessionAliases.set(alias, session.ownerKey);
  }
}

/** Create the mutable in-memory state used by the hook replay backend. */
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
      replayErrors: 0,
      skippedEvents: 0,
    },
  };
}

/** Return an existing session or lazily create a root session scope for replay. */
export function ensureSession(manager: SessionManager, input: EnsureSessionInput): SessionState | undefined {
  const resolvedInput = mergeEnsureSessionInput(input, manager.resolveSessionRootContext?.(input));
  const ownerKey = resolveSessionOwnerKey(manager.state, resolvedInput);
  if (!ownerKey) {
    manager.state.counters.skippedEvents += 1;
    manager.logBoundedWarn('missing-session-key', 'nemo-relay skipped replay because no session/run key was available');
    return undefined;
  }

  const existing = manager.state.sessions.get(ownerKey);
  if (existing) {
    enrichSession(existing, resolvedInput);
    rememberSessionAliases(manager.state, existing, resolvedInput);
    if (!existing.rootHandle && resolvedInput.deferRootOpen !== true) {
      materializeSessionRoot(manager, existing, resolvedInput);
    }
    return existing;
  }

  const stack = manager.nf.createScopeStack();
  const session: SessionState = {
    ownerKey,
    sessionId: resolvedInput.sessionId ?? ownerKey,
    source: resolvedInput.source,
    stack,
  };

  const sessionKey = ownSessionKey(resolvedInput);
  if (sessionKey !== undefined) {
    session.sessionKey = sessionKey;
  }
  if (resolvedInput.agentId !== undefined) {
    session.agentId = resolvedInput.agentId;
  }
  if (resolvedInput.resumedFrom !== undefined) {
    session.resumedFrom = resolvedInput.resumedFrom;
  }
  if (resolvedInput.scopeRole !== undefined) {
    session.scopeRole = resolvedInput.scopeRole;
  }

  if (resolvedInput.deferRootOpen === true) {
    session.pendingRootOpen = true;
    if (resolvedInput.timestamp !== undefined) {
      session.pendingRootTimestampMicros = resolvedInput.timestamp;
    }
    if (resolvedInput.runId !== undefined) {
      session.pendingRootRunId = resolvedInput.runId;
    }
    session.pendingCapturedEmits = [];
  } else {
    materializeSessionRoot(manager, session, resolvedInput);
  }
  manager.state.sessions.set(session.ownerKey, session);
  rememberSessionAliases(manager.state, session, resolvedInput);
  return session;
}

/** Fill stable session identifiers from later hooks without clobbering established values. */
function enrichSession(session: SessionState, input: EnsureSessionInput): void {
  const sessionKey = ownSessionKey(input);
  if (session.sessionKey === undefined && sessionKey !== undefined) {
    session.sessionKey = sessionKey;
  }
  if (session.agentId === undefined && input.agentId !== undefined) {
    session.agentId = input.agentId;
  }
  if (session.resumedFrom === undefined && input.resumedFrom !== undefined) {
    session.resumedFrom = input.resumedFrom;
  }
  if (session.scopeRole === undefined && input.scopeRole !== undefined) {
    session.scopeRole = input.scopeRole;
  }
  if (!session.rootHandle && input.sessionId !== undefined) {
    session.sessionId = input.sessionId;
  }
  if (!session.rootHandle && session.source === 'lazy_session' && input.source === 'session_start') {
    session.source = 'session_start';
  }
  if (!session.rootHandle && input.timestamp !== undefined) {
    if (input.source === 'session_start' || session.pendingRootTimestampMicros === undefined) {
      session.pendingRootTimestampMicros = input.timestamp;
    }
  }
  if (!session.rootHandle && session.pendingRootRunId === undefined && input.runId !== undefined) {
    session.pendingRootRunId = input.runId;
  }
}

/** Queue an emit callback until a deferred session root can open with honest lineage. */
export function queueCapturedEmit(session: SessionState, label: string, emit: () => void): boolean {
  if (session.rootHandle || session.pendingRootOpen !== true) {
    return false;
  }

  (session.pendingCapturedEmits ??= []).push({ label, emit });
  return true;
}

/** Flush pending LLM output/timing state before the root session closes. */
export function drainSession(manager: SessionManager, session: SessionState): void {
  cancelPendingLlmOutputTimers(manager.state, session);
  manager.replayPendingLlmOutputsForSession(session, { allowPlaceholderRequest: true });
  manager.emitUnpairedModelCallTimingMarks(session);
  evictSessionCorrelationRecords(manager.state, session);
}

/** Close the root session scope with separate lifecycle summary and user-visible output. */
export function closeSessionRoot(
  manager: SessionManager,
  session: SessionState,
  summary: JsonRecord,
  rootOutput: JsonRecord = summary,
  metadata?: JsonRecord | null,
  timestamp?: number,
): void {
  manager.emitCapturedUnderSession('session_end', session, () => {
    if (!session.rootHandle) {
      return;
    }

    manager.nf.event('openclaw.session_end', session.rootHandle, summary, metadata ?? null, timestamp ?? null);
    manager.state.counters.marksEmitted += 1;
    manager.nf.popScope(session.rootHandle, rootOutput, timestamp ?? null);
    delete session.rootHandle;
  });
}

/** Remove a closed session from active replay state. */
export function deleteSession(state: HookReplayBackendState, session: SessionState): void {
  state.sessions.delete(session.ownerKey);
}

/** Insert a correlation record while bounding retained entries per key. */
export function insertBoundedRecord<T>(map: Map<string, T[]>, key: string, record: T, maxRecordsPerKey: number): void {
  const records = map.get(key) ?? [];
  records.push(record);
  while (records.length > maxRecordsPerKey) {
    records.shift();
  }
  map.set(key, records);
}

/** Build a stable tuple key for session alias maps. */
export function tupleKey(parts: Array<string | undefined>): string {
  return tupleKeyFromCorrelation(parts);
}

/** Evict stale cross-hook correlation records across all replay maps. */
export function evictExpiredCorrelationRecords(state: HookReplayBackendState, nowMs: number, ttlMs: number): void {
  evictExpiredRecords(state.llmInputs, nowMs, ttlMs);
  evictExpiredPendingLlmOutputs(state.llmOutputsPendingInput, nowMs, ttlMs);
  evictExpiredRecords(state.modelCallsByCallId, nowMs, ttlMs);
  evictExpiredRecords(state.modelTimingsByLlmKey, nowMs, ttlMs);
}

/** Open a deferred or new root session scope and flush queued child emissions. */
export function materializeSessionRoot(manager: SessionManager, session: SessionState, input: EnsureSessionInput): void {
  if (session.rootHandle) {
    return;
  }

  enrichSession(session, input);

  const timestampMicros = input.timestamp ?? session.pendingRootTimestampMicros ?? null;
  const rootRunId = input.runId ?? session.pendingRootRunId;

  const data: JsonRecord = {
    sessionId: session.sessionId,
    source: session.source,
    ...(session.sessionKey === undefined ? {} : { sessionKey: session.sessionKey }),
    ...(session.agentId === undefined ? {} : { agentId: session.agentId }),
    ...(rootRunId === undefined ? {} : { runId: rootRunId }),
    ...(session.resumedFrom === undefined ? {} : { resumedFrom: session.resumedFrom }),
  };
  const metadata = toJsonRecord({
    source: session.source === 'session_start' ? 'openclaw.session_start' : 'openclaw.lazy_session',
    hook_event_name: session.source === 'session_start' ? 'session_start' : undefined,
    sessionId: session.sessionId,
    sessionKey: session.sessionKey,
    agentId: session.agentId,
    runId: rootRunId,
    nemo_relay_scope_role: session.scopeRole,
  });

  const previousStack = manager.nf.currentScopeStack();
  try {
    manager.nf.setThreadScopeStack(session.stack);
    session.rootHandle = manager.nf.pushScope(
      'openclaw.session',
      agentScopeType(manager.nf),
      input.parentHandle ?? null,
      null,
      data,
      metadata,
      data,
      timestampMicros,
    );
    manager.nf.event('openclaw.session_start', session.rootHandle, data, metadata, timestampMicros);
    manager.state.counters.marksEmitted += 1;
  } finally {
    manager.nf.setThreadScopeStack(previousStack);
  }

  delete session.pendingRootOpen;
  delete session.pendingRootTimestampMicros;
  delete session.pendingRootRunId;
  const pendingEmits = session.pendingCapturedEmits ?? [];
  delete session.pendingCapturedEmits;
  for (const pending of pendingEmits) {
    manager.emitCapturedUnderSession(pending.label, session, pending.emit);
  }
}

/** Cancel timers that would otherwise replay late LLM outputs after session close. */
function cancelPendingLlmOutputTimers(state: HookReplayBackendState, session: SessionState): void {
  for (const records of state.llmOutputsPendingInput.values()) {
    for (const record of records) {
      if (record.sessionOwnerKey === session.ownerKey && record.timer) {
        clearTimeout(record.timer);
        record.timer = undefined;
      }
    }
  }
}

/** Remove all correlation records and aliases owned by a closed session. */
function evictSessionCorrelationRecords(state: HookReplayBackendState, session: SessionState): void {
  evictFromRecordMap(state.llmInputs, session.ownerKey);
  evictFromRecordMap(state.llmOutputsPendingInput, session.ownerKey);
  evictFromRecordMap(state.modelCallsByCallId, session.ownerKey);
  evictFromRecordMap(state.modelTimingsByLlmKey, session.ownerKey);

  for (const [alias, ownerKey] of state.sessionAliases) {
    if (ownerKey === session.ownerKey) {
      state.sessionAliases.delete(alias);
    }
  }
}

/** Drop records for one session from a single keyed correlation map. */
function evictFromRecordMap<T extends { sessionOwnerKey: string }>(map: Map<string, T[]>, ownerKey: string): void {
  for (const [key, records] of map) {
    const retained = records.filter((record) => record.sessionOwnerKey !== ownerKey);
    if (retained.length === 0) {
      map.delete(key);
    } else {
      map.set(key, retained);
    }
  }
}

/** Evict pending LLM outputs and clear their grace timers when their TTL expires. */
function evictExpiredPendingLlmOutputs(map: Map<string, PendingLlmOutputRecord[]>, nowMs: number, ttlMs: number): void {
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

/** Resolve the runtime's Agent scope enum while tolerating older Node bindings. */
function agentScopeType(nf: NemoRelayRuntimeModule): Parameters<NemoRelayRuntimeModule['pushScope']>[1] {
  return (nf.ScopeType?.Agent ?? 0) as Parameters<NemoRelayRuntimeModule['pushScope']>[1];
}
