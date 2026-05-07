// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NemoFlowHookBackendConfig } from "../config.js";
import type {
  PluginHookAgentContext,
  PluginHookLlmInputEvent,
  PluginHookLlmOutputEvent,
  PluginHookModelCallEndedEvent,
  PluginHookModelCallStartedEvent,
} from "../openclaw-hook-types.js";
import type { JsonRecord, JsonValue } from "../types.js";
import { emitMark, toJsonRecord, toJsonValue } from "./marks.js";
import {
  evictExpiredCorrelationRecords,
  ensureSession,
  insertBoundedRecord,
  type LlmInputRecord,
  type ModelCallRecord,
  type PendingLlmOutputRecord,
  type SessionManager,
  type SessionState,
} from "./session.js";
import {
  llmKey,
  modelTimingKey,
  modelTimingLlmKey,
  nowMicros,
  startMicrosFromDuration,
} from "./correlation.js";

export function recordLlmInput(
  manager: SessionManager,
  event: PluginHookLlmInputEvent,
  ctx: PluginHookAgentContext,
): void {
  evictExpiredReplayRecords(manager);
  const session = ensureSession(manager, {
    sessionId: event.sessionId,
    sessionKey: ctx.sessionKey,
    runId: event.runId,
    agentId: ctx.agentId,
    source: "lazy_session",
  });
  if (!session) {
    return;
  }

  const key = llmKey(event);
  const input = createInputRecord(session, event);
  insertBoundedRecord(manager.state.llmInputs, key, input, manager.config.correlation.maxRecordsPerKey);

  const pending = shiftOldest(manager.state.llmOutputsPendingInput, key, (record) => record.sessionKey === session.sessionId);
  if (!pending) {
    return;
  }

  removeRecord(manager.state.llmInputs, key, input);
  clearPendingTimer(pending);
  replayLlmOutput({
    manager,
    event: pending.event,
    ctx: pending.ctx,
    input,
    timing: consumeTimingCandidate(manager, session, pending.event),
  });
}

export function recordLlmOutput(
  manager: SessionManager,
  event: PluginHookLlmOutputEvent,
  ctx: PluginHookAgentContext,
): void {
  evictExpiredReplayRecords(manager);
  const session = ensureSession(manager, {
    sessionId: event.sessionId,
    sessionKey: ctx.sessionKey,
    runId: event.runId,
    agentId: ctx.agentId,
    source: "lazy_session",
  });
  if (!session) {
    return;
  }

  const key = llmKey(event);
  const input = shiftOldest(manager.state.llmInputs, key, (record) => record.sessionKey === session.sessionId);
  if (input) {
    replayLlmOutput({
      manager,
      event,
      ctx,
      input,
      timing: consumeTimingCandidate(manager, session, event),
    });
    return;
  }

  const pending: PendingLlmOutputRecord = {
    sessionKey: session.sessionId,
    sessionId: event.sessionId,
    runId: event.runId,
    provider: event.provider,
    model: event.model,
    event,
    ctx,
    observedAtMs: Date.now(),
  };
  pending.timer = setTimeout(
    () => replayExpiredPendingOutput(manager, key, pending),
    manager.config.correlation.llmOutputGraceMs,
  );
  pending.timer.unref?.();
  insertPendingOutput(manager, key, pending);
}

export function recordModelCallStarted(
  manager: SessionManager,
  event: PluginHookModelCallStartedEvent,
  ctx: PluginHookAgentContext,
): void {
  evictExpiredReplayRecords(manager);
  const session = ensureSession(manager, {
    sessionId: event.sessionId ?? ctx.sessionId,
    sessionKey: event.sessionKey ?? ctx.sessionKey,
    runId: event.runId,
    agentId: ctx.agentId,
    source: "lazy_session",
  });
  if (!session) {
    return;
  }

  insertBoundedRecord(
    manager.state.modelCallsByCallId,
    modelTimingKey(event),
    {
      sessionKey: session.sessionId,
      sessionId: session.sessionId,
      runId: event.runId,
      callId: event.callId,
      provider: event.provider,
      model: event.model,
      consumed: false,
      observedAtMs: Date.now(),
      startedAtMs: Date.now(),
      ...(event.api === undefined ? {} : { api: event.api }),
      ...(event.transport === undefined ? {} : { transport: event.transport }),
    },
    manager.config.correlation.maxRecordsPerKey,
  );
}

export function recordModelCallEnded(
  manager: SessionManager,
  event: PluginHookModelCallEndedEvent,
  ctx: PluginHookAgentContext,
): void {
  evictExpiredReplayRecords(manager);
  const session = ensureSession(manager, {
    sessionId: event.sessionId ?? ctx.sessionId,
    sessionKey: event.sessionKey ?? ctx.sessionKey,
    runId: event.runId,
    agentId: ctx.agentId,
    source: "lazy_session",
  });
  if (!session) {
    return;
  }

  const nowMs = Date.now();
  const byCallKey = modelTimingKey(event);
  const existing = latestUnendedRecord(manager.state.modelCallsByCallId.get(byCallKey), session);
  const record =
    existing ??
    ({
      sessionKey: session.sessionId,
      sessionId: session.sessionId,
      runId: event.runId,
      callId: event.callId,
      provider: event.provider,
      model: event.model,
      consumed: false,
      observedAtMs: nowMs,
    } satisfies ModelCallRecord);

  applyModelCallEnd(record, event, nowMs);
  if (!existing) {
    insertBoundedRecord(
      manager.state.modelCallsByCallId,
      byCallKey,
      record,
      manager.config.correlation.maxRecordsPerKey,
    );
  }
  insertBoundedRecord(
    manager.state.modelTimingsByLlmKey,
    modelTimingLlmKey({ sessionId: session.sessionId, runId: event.runId, provider: event.provider, model: event.model }),
    record,
    manager.config.correlation.maxRecordsPerKey,
  );
}

export function replayPendingLlmOutputsForSession(
  manager: SessionManager,
  session: SessionState,
  options: { allowPlaceholderRequest: boolean },
): void {
  if (!options.allowPlaceholderRequest) {
    return;
  }
  for (const [key, records] of [...manager.state.llmOutputsPendingInput]) {
    const remaining: PendingLlmOutputRecord[] = [];
    for (const record of records) {
      if (record.sessionKey !== session.sessionId) {
        remaining.push(record);
        continue;
      }
      clearPendingTimer(record);
      replayLlmOutput({
        manager,
        event: record.event,
        ctx: record.ctx,
        input: placeholderInputRecord(record),
        timing: consumeTimingCandidate(manager, session, record.event),
      });
    }
    if (remaining.length === 0) {
      manager.state.llmOutputsPendingInput.delete(key);
    } else {
      manager.state.llmOutputsPendingInput.set(key, remaining);
    }
  }
}

export function emitUnpairedModelCallTimingMarks(manager: SessionManager, session: SessionState): void {
  for (const records of manager.state.modelCallsByCallId.values()) {
    for (const record of records) {
      if (record.sessionKey !== session.sessionId || record.consumed || record.endedAtMs !== undefined) {
        continue;
      }
      emitModelTimingMark(manager, session, "openclaw.model_call_timing_unpaired", record);
      record.consumed = true;
    }
  }

  for (const records of manager.state.modelTimingsByLlmKey.values()) {
    for (const record of records) {
      if (record.sessionKey !== session.sessionId || record.consumed) {
        continue;
      }
      emitModelTimingMark(manager, session, "openclaw.model_call_timing_unpaired", record);
      record.consumed = true;
    }
  }
}

export function buildReplayLlmRequest(
  input: LlmInputRecord,
  output: PluginHookLlmOutputEvent,
  config: NemoFlowHookBackendConfig,
): JsonValue {
  const messages = config.capture.includePrompts && Array.isArray(input.historyMessages) ? [...input.historyMessages] : [];
  const replayMessages = config.capture.includePrompts ? appendPromptIfMissing(messages, input.prompt) : [];
  return toJsonValue({
    headers: {},
    content: {
      provider: output.provider,
      model: output.model,
      prompt: config.capture.includePrompts ? input.prompt : undefined,
      systemPrompt: config.capture.includePrompts ? input.systemPrompt : undefined,
      messages: replayMessages,
      imagesCount: input.imagesCount,
      placeholderRequest: input.placeholderRequest === true,
      source: "openclaw.hooks",
    },
  });
}

export function buildReplayLlmResponse(
  event: PluginHookLlmOutputEvent,
  timing: ModelCallRecord | undefined,
  config: NemoFlowHookBackendConfig,
): JsonValue {
  return toJsonValue({
    role: "assistant",
    content: config.capture.includeResponses ? event.assistantTexts.join("\n") : undefined,
    assistant_texts_count: event.assistantTexts.length,
    resolved_ref: event.resolvedRef,
    harness_id: event.harnessId,
    token_usage: mapUsage(event.usage),
    openclaw: {
      duration_ms: timing?.durationMs,
      outcome: timing?.outcome,
      error_category: timing?.errorCategory,
      failure_kind: timing?.failureKind,
      time_to_first_byte_ms: timing?.timeToFirstByteMs,
      request_payload_bytes: timing?.requestPayloadBytes,
      response_stream_bytes: timing?.responseStreamBytes,
      upstream_request_id_hash: timing?.upstreamRequestIdHash,
    },
  });
}

function replayExpiredPendingOutput(
  manager: SessionManager,
  key: string,
  record: PendingLlmOutputRecord,
): void {
  try {
    if (!removeRecord(manager.state.llmOutputsPendingInput, key, record)) {
      return;
    }
    const session = manager.state.sessions.get(record.sessionKey);
    if (!session) {
      manager.state.counters.skippedEvents += 1;
      return;
    }
    replayLlmOutput({
      manager,
      event: record.event,
      ctx: record.ctx,
      input: placeholderInputRecord(record),
      timing: consumeTimingCandidate(manager, session, record.event),
    });
  } catch (error) {
    manager.state.counters.replayErrors += 1;
    manager.logBoundedWarn(
      `llm_grace_timer_failed:${key}`,
      `nemo-flow failed to replay pending llm_output after grace timer: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function replayLlmOutput(params: {
  manager: SessionManager;
  event: PluginHookLlmOutputEvent;
  ctx: PluginHookAgentContext;
  input: LlmInputRecord;
  timing?: ModelCallRecord | undefined;
}): void {
  const { manager, event, ctx, input, timing } = params;
  const session = ensureSession(manager, {
    sessionId: event.sessionId,
    runId: event.runId,
    agentId: ctx.agentId,
    source: "lazy_session",
  });
  if (!session) {
    return;
  }

  const endMicros = nowMicros();
  const request = buildReplayLlmRequest(input, event, manager.config);
  const response = buildReplayLlmResponse(event, timing, manager.config);
  const metadata = toJsonRecord({
    source: "openclaw.llm_output",
    runId: event.runId,
    sessionId: event.sessionId,
    provider: event.provider,
    model: event.model,
  });

  manager.emitCapturedUnderSession("llm_output", session, () => {
    const handle = manager.nf.llmCall(
      event.provider,
      request,
      session.rootHandle,
      null,
      metadata,
      metadata,
      event.model,
      startMicrosFromDuration(endMicros, timing?.durationMs),
    );
    manager.nf.llmCallEnd(handle, response, response, metadata, endMicros);
    manager.state.counters.llmSpansReplayed += 1;
  });
}

function consumeTimingCandidate(
  manager: SessionManager,
  session: SessionState,
  event: PluginHookLlmOutputEvent,
): ModelCallRecord | undefined {
  const key = modelTimingLlmKey({
    sessionId: session.sessionId,
    runId: event.runId,
    provider: event.provider,
    model: event.model,
  });
  const candidates = (manager.state.modelTimingsByLlmKey.get(key) ?? []).filter(
    (record) => record.sessionKey === session.sessionId && !record.consumed,
  );
  if (candidates.length === 1) {
    const candidate = candidates[0];
    if (!candidate) {
      return undefined;
    }
    candidate.consumed = true;
    return candidate;
  }
  if (candidates.length > 1) {
    const shouldEmit = candidates.some((candidate) => candidate.ambiguous !== true);
    for (const candidate of candidates) {
      candidate.ambiguous = true;
    }
    if (shouldEmit) {
      emitModelTimingAmbiguousMark(manager, session, event, candidates.length);
    }
  }
  return undefined;
}

function emitModelTimingAmbiguousMark(
  manager: SessionManager,
  session: SessionState,
  event: PluginHookLlmOutputEvent,
  candidateCount: number,
): void {
  manager.emitCapturedUnderSession("model_call_timing_ambiguous", session, () => {
    emitMark({
      nf: manager.nf,
      state: manager.state,
      session,
      name: "openclaw.model_call_timing_ambiguous",
      data: toJsonRecord({
        runId: event.runId,
        sessionId: event.sessionId,
        provider: event.provider,
        model: event.model,
        candidateCount,
      }),
    });
  });
}

function emitModelTimingMark(
  manager: SessionManager,
  session: SessionState,
  name: string,
  record: ModelCallRecord,
): void {
  manager.emitCapturedUnderSession(name, session, () => {
    emitMark({
      nf: manager.nf,
      state: manager.state,
      session,
      name,
      data: toJsonRecord({
        runId: record.runId,
        callId: record.callId,
        provider: record.provider,
        model: record.model,
        api: record.api,
        transport: record.transport,
        durationMs: record.durationMs,
        outcome: record.outcome,
        errorCategory: record.errorCategory,
        failureKind: record.failureKind,
        requestPayloadBytes: record.requestPayloadBytes,
        responseStreamBytes: record.responseStreamBytes,
        timeToFirstByteMs: record.timeToFirstByteMs,
        upstreamRequestIdHash: record.upstreamRequestIdHash,
        ambiguous: record.ambiguous,
      }),
    });
  });
}

function createInputRecord(session: SessionState, event: PluginHookLlmInputEvent): LlmInputRecord {
  return {
    sessionKey: session.sessionId,
    sessionId: event.sessionId,
    runId: event.runId,
    provider: event.provider,
    model: event.model,
    prompt: event.prompt,
    historyMessages: event.historyMessages,
    imagesCount: event.imagesCount,
    observedAtMs: Date.now(),
    ...(event.systemPrompt === undefined ? {} : { systemPrompt: event.systemPrompt }),
  };
}

function placeholderInputRecord(record: PendingLlmOutputRecord): LlmInputRecord {
  return {
    sessionKey: record.sessionKey,
    sessionId: record.sessionId,
    runId: record.runId,
    provider: record.provider,
    model: record.model,
    prompt: "",
    historyMessages: [],
    imagesCount: 0,
    observedAtMs: Date.now(),
    placeholderRequest: true,
  };
}

function appendPromptIfMissing(historyMessages: unknown[], prompt: string): unknown[] {
  if (!prompt) {
    return historyMessages;
  }
  const last = historyMessages.at(-1);
  if (isRecord(last) && last.role === "user" && last.content === prompt) {
    return historyMessages;
  }
  return [...historyMessages, { role: "user", content: prompt }];
}

function mapUsage(usage: PluginHookLlmOutputEvent["usage"]): Record<string, number> | undefined {
  if (!usage) {
    return undefined;
  }
  const mapped: Record<string, number> = {};
  if (usage.input !== undefined) {
    mapped.prompt_tokens = usage.input;
  }
  if (usage.output !== undefined) {
    mapped.completion_tokens = usage.output;
  }
  if (usage.cacheRead !== undefined) {
    mapped.cached_tokens = usage.cacheRead;
  }
  if (usage.cacheWrite !== undefined) {
    mapped.cache_write_tokens = usage.cacheWrite;
  }
  if (usage.total !== undefined) {
    mapped.total_tokens = usage.total;
  } else if (usage.input !== undefined || usage.output !== undefined) {
    mapped.total_tokens = (usage.input ?? 0) + (usage.output ?? 0);
  }
  return Object.keys(mapped).length > 0 ? mapped : undefined;
}

function applyModelCallEnd(record: ModelCallRecord, event: PluginHookModelCallEndedEvent, nowMs: number): void {
  record.observedAtMs = nowMs;
  record.endedAtMs = nowMs;
  record.durationMs = event.durationMs;
  record.outcome = event.outcome;
  record.api = event.api;
  record.transport = event.transport;
  record.errorCategory = event.errorCategory;
  record.failureKind = event.failureKind;
  record.requestPayloadBytes = event.requestPayloadBytes;
  record.responseStreamBytes = event.responseStreamBytes;
  record.timeToFirstByteMs = event.timeToFirstByteMs;
  record.upstreamRequestIdHash = event.upstreamRequestIdHash;
}

function latestUnendedRecord(records: ModelCallRecord[] | undefined, session: SessionState): ModelCallRecord | undefined {
  if (!records) {
    return undefined;
  }
  for (let index = records.length - 1; index >= 0; index -= 1) {
    const record = records[index];
    if (record?.sessionKey === session.sessionId && record.endedAtMs === undefined) {
      return record;
    }
  }
  return undefined;
}

function insertPendingOutput(manager: SessionManager, key: string, record: PendingLlmOutputRecord): void {
  const records = manager.state.llmOutputsPendingInput.get(key) ?? [];
  records.push(record);
  while (records.length > manager.config.correlation.maxRecordsPerKey) {
    const evicted = records.shift();
    if (evicted) {
      clearPendingTimer(evicted);
    }
  }
  manager.state.llmOutputsPendingInput.set(key, records);
}

function shiftOldest<T>(map: Map<string, T[]>, key: string, predicate: (record: T) => boolean): T | undefined {
  const records = map.get(key);
  if (!records) {
    return undefined;
  }
  const index = records.findIndex(predicate);
  if (index === -1) {
    return undefined;
  }
  const [record] = records.splice(index, 1);
  if (records.length === 0) {
    map.delete(key);
  }
  return record;
}

function removeRecord<T>(map: Map<string, T[]>, key: string, record: T): boolean {
  const records = map.get(key);
  if (!records) {
    return false;
  }
  const index = records.indexOf(record);
  if (index === -1) {
    return false;
  }
  records.splice(index, 1);
  if (records.length === 0) {
    map.delete(key);
  }
  return true;
}

function clearPendingTimer(record: PendingLlmOutputRecord): void {
  if (record.timer) {
    clearTimeout(record.timer);
    record.timer = undefined;
  }
}

function evictExpiredReplayRecords(manager: SessionManager): void {
  evictExpiredCorrelationRecords(manager.state, Date.now(), manager.config.correlation.recordTtlMs);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
