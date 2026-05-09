// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NemoFlowHookBackendConfig } from "../config.js";
import type {
  PluginHookAgentContext,
  PluginHookAgentEndEvent,
  PluginHookBeforeMessageWriteContext,
  PluginHookBeforeMessageWriteEvent,
  PluginHookLlmInputEvent,
  PluginHookLlmOutputEvent,
  PluginHookModelCallEndedEvent,
  PluginHookModelCallStartedEvent,
} from "../openclaw-hook-types.js";
import type { JsonObject as JsonRecord, JsonValue } from "nemo-flow-node/typed";
import { emitMark, toJsonRecord, toJsonValue } from "./marks.js";
import {
  evictExpiredCorrelationRecords,
  ensureSession,
  insertBoundedRecord,
  resolveSessionKey,
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

  if (hasTrajectoryReplay(session, event.runId)) {
    return;
  }

  rememberAgentRunInputSnapshot(
    session,
    event.runId,
    event.historyMessages,
    event.prompt,
    manager.config.correlation.maxRecordsPerKey,
  );
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
  if (hasTrajectoryReplay(session, event.runId)) {
    shiftOldest(manager.state.llmInputs, key, (record) => record.sessionKey === session.sessionId);
    return;
  }

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

export function recordBeforeMessageWrite(
  manager: SessionManager,
  event: PluginHookBeforeMessageWriteEvent,
  ctx: PluginHookBeforeMessageWriteContext,
): void {
  evictExpiredReplayRecords(manager);
  const session = existingSessionForMessageWrite(manager, event, ctx);
  if (!session) {
    return;
  }

  const message = isRecord(event.message) ? event.message : undefined;
  if (!message || typeof message.role !== "string") {
    return;
  }
  const recordedMessage = toJsonValue(message);
  const historyMessages =
    session.messageWrites === undefined || session.messageWrites.length === 0
      ? initialHistoryFromLlmInputSnapshot(session)
      : [...session.messageWrites];
  if ((session.messageWrites === undefined || session.messageWrites.length === 0) && historyMessages.length > 0) {
    session.messageWrites = [...historyMessages];
  }

  if (message.role === "assistant") {
    const provider = stringField(message, "provider");
    const model = stringField(message, "model");
    const assistantTexts = extractTextBlocks(message);
    const assistantToolCalls = snapshotMessages(extractToolCalls(message));
    const usage = "usage" in message ? toJsonValue(message.usage) : undefined;
    if (provider && model && (assistantTexts.length > 0 || assistantToolCalls.length > 0 || usage !== undefined)) {
      session.assistantMessageWrites ??= [];
      session.assistantMessageWrites.push({
        sessionKey: session.sessionId,
        provider,
        model,
        assistantTexts,
        assistantToolCalls,
        historyMessages,
        prompt: "",
        observedAtMs: Date.now(),
        replayed: false,
        ...(usage === undefined ? {} : { usage }),
      });
      while (session.assistantMessageWrites.length > manager.config.correlation.maxRecordsPerKey) {
        session.assistantMessageWrites.shift();
      }
    }
  }

  session.messageWrites ??= [];
  session.messageWrites.push(recordedMessage);
  while (session.messageWrites.length > manager.config.correlation.maxRecordsPerKey) {
    session.messageWrites.shift();
  }
}

export function recordModelCallStarted(
  manager: SessionManager,
  event: PluginHookModelCallStartedEvent,
  ctx: PluginHookAgentContext,
): void {
  evictExpiredReplayRecords(manager);
  const nowMs = Date.now();
  const session = ensureSession(manager, {
    sessionId: event.sessionId ?? ctx.sessionId,
    sessionKey: event.sessionKey ?? ctx.sessionKey,
    runId: event.runId,
    agentId: ctx.agentId,
    source: "lazy_session",
    timestamp: nowMs * 1000,
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
      observedAtMs: nowMs,
      startedAtMs: nowMs,
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
  const nowMs = Date.now();
  const startMicros = startMicrosFromDuration(nowMs * 1000, event.durationMs) ?? nowMs * 1000;
  const session = ensureSession(manager, {
    sessionId: event.sessionId ?? ctx.sessionId,
    sessionKey: event.sessionKey ?? ctx.sessionKey,
    runId: event.runId,
    agentId: ctx.agentId,
    source: "lazy_session",
    timestamp: startMicros,
  });
  if (!session) {
    return;
  }

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
      if (hasTrajectoryReplay(session, record.runId)) {
        continue;
      }
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

export function replayAgentEndMessages(
  manager: SessionManager,
  event: PluginHookAgentEndEvent,
  ctx: PluginHookAgentContext,
  session: SessionState,
): JsonRecord | undefined {
  const runId = event.runId ?? ctx.runId;
  const runKey = trajectoryRunKey(session, runId);
  const currentRunMessages = currentRunAgentMessages(session, runId, event.messages);
  const finalOutput = finalOutputFromAgentEnd(currentRunMessages, event, runId);
  if (hasTrajectoryReplay(session, runId)) {
    return finalOutput;
  }

  const replayedFromHooks = hookLlmOutputReplayCount(session, runId);
  if (replayedFromHooks > 0) {
    markTrajectoryReplay(session, runKey, manager.config.correlation.maxRecordsPerKey);
    cleanupAgentRunReplayBookkeeping(session, runKey);
    return finalOutput;
  }

  const replayedFromMessageWrites = replayAssistantMessageWrites(manager, session, event, ctx);
  if (replayedFromMessageWrites > 0) {
    markTrajectoryReplay(session, runKey, manager.config.correlation.maxRecordsPerKey);
    cleanupAgentRunReplayBookkeeping(session, runKey);
    return finalOutput;
  }

  cleanupAgentRunReplayBookkeeping(session, runKey);
  return finalOutput;
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

  const unpairedEnded: ModelCallRecord[] = [];
  for (const records of manager.state.modelTimingsByLlmKey.values()) {
    for (const record of records) {
      if (record.sessionKey !== session.sessionId || record.consumed) {
        continue;
      }
      unpairedEnded.push(record);
      record.consumed = true;
    }
  }
  if (unpairedEnded.length === 1) {
    const [record] = unpairedEnded;
    if (record) {
      emitModelTimingMark(manager, session, "openclaw.model_call_timing_unpaired", record);
    }
  } else if (unpairedEnded.length > 1) {
    emitModelTimingSummaryMark(manager, session, unpairedEnded);
  }
}

export function buildReplayLlmRequest(
  input: LlmInputRecord,
  output: PluginHookLlmOutputEvent,
  config: NemoFlowHookBackendConfig,
  source = "openclaw.hooks",
): JsonValue {
  const messages =
    config.capture.includePrompts && Array.isArray(input.historyMessages)
      ? input.historyMessages.map((message) => sanitizePromptMessage(message, config))
      : [];
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
      source,
    },
  });
}

export function buildReplayLlmResponse(
  event: PluginHookLlmOutputEvent,
  timing: ModelCallRecord | undefined,
  config: NemoFlowHookBackendConfig,
): JsonValue {
  const usage = mapUsage(event.usage);
  const assistantToolCallNames = toolCallNames(event.assistantToolCalls);
  return toJsonValue({
    role: "assistant",
    content: config.capture.includeResponses
      ? responseContent(event.assistantTexts, assistantToolCallNames)
      : undefined,
    assistant_texts_count: event.assistantTexts.length,
    resolved_ref: event.resolvedRef,
    harness_id: event.harnessId,
    usage,
    openclaw: {
      duration_ms: timing?.durationMs,
      outcome: timing?.outcome,
      error_category: timing?.errorCategory,
      failure_kind: timing?.failureKind,
      time_to_first_byte_ms: timing?.timeToFirstByteMs,
      request_payload_bytes: timing?.requestPayloadBytes,
      response_stream_bytes: timing?.responseStreamBytes,
      upstream_request_id_hash: timing?.upstreamRequestIdHash,
      assistant_tool_call_names: assistantToolCallNames,
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
    if (hasTrajectoryReplay(session, record.runId)) {
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
  source?: "openclaw.llm_output" | "openclaw.before_message_write" | undefined;
}): void {
  const { manager, event, ctx, input, timing, source = "openclaw.llm_output" } = params;
  const observedEndMicros = nowMicros();
  const endMicros = timing?.endedAtMs === undefined ? observedEndMicros : timing.endedAtMs * 1000;
  const observedStartMicros = Math.min(input.observedAtMs * 1000, endMicros);
  const startMicros =
    timing?.startedAtMs === undefined
      ? (startMicrosFromDuration(endMicros, timing?.durationMs) ?? observedStartMicros)
      : Math.min(timing.startedAtMs * 1000, endMicros);
  const session = ensureSession(manager, {
    sessionId: event.sessionId,
    sessionKey: ctx.sessionKey,
    runId: event.runId,
    agentId: ctx.agentId,
    source: "lazy_session",
    timestamp: startMicros,
  });
  if (!session) {
    return;
  }

  const request = buildReplayLlmRequest(input, event, manager.config, source);
  const response = buildReplayLlmResponse(event, timing, manager.config);
  const metadata = toJsonRecord({
    source,
    runId: event.runId,
    sessionId: event.sessionId,
    provider: event.provider,
    model: event.model,
    callId: timing?.callId,
    correlation: source === "openclaw.before_message_write" ? "fifo_model_call_timing" : undefined,
  });

  manager.emitCapturedUnderSession("llm_output", session, () => {
    const handle = manager.nf.llmCall(
      event.provider,
      request,
      session.rootHandle,
      null,
      null,
      metadata,
      event.model,
      startMicros,
    );
    manager.nf.llmCallEnd(handle, response, null, metadata, endMicros);
    manager.state.counters.llmSpansReplayed += 1;
  });
  if (source === "openclaw.llm_output") {
    incrementHookLlmOutputReplayCount(session, event.runId, manager.config.correlation.maxRecordsPerKey);
  }
}

function replayAssistantMessageWrites(
  manager: SessionManager,
  session: SessionState,
  event: PluginHookAgentEndEvent,
  ctx: PluginHookAgentContext,
): number {
  const records = session.assistantMessageWrites ?? [];
  const pending = records.filter((record) => !record.replayed);
  let replayed = 0;

  for (const record of pending) {
    const timing = consumeNextTimingCandidate(manager, session, {
      runId: event.runId ?? ctx.runId,
      provider: record.provider,
      model: record.model,
    });
    if (!timing) {
      continue;
    }

    const runId = timing.runId || event.runId || ctx.runId || session.sessionId;
    const usage = mapHookUsage(record.usage);
    replayLlmOutput({
      manager,
      event: {
        runId,
        sessionId: session.sessionId,
        provider: record.provider,
        model: record.model,
        assistantTexts: record.assistantTexts,
        assistantToolCalls: record.assistantToolCalls,
        ...(usage === undefined ? {} : { usage }),
      },
      ctx,
      input: {
        sessionKey: session.sessionId,
        sessionId: session.sessionId,
        runId,
        provider: record.provider,
        model: record.model,
        prompt: record.prompt,
        historyMessages: record.historyMessages,
        imagesCount: 0,
        observedAtMs: record.observedAtMs,
        placeholderRequest: true,
      },
      timing,
      source: "openclaw.before_message_write",
    });
    record.replayed = true;
    replayed += 1;
  }

  session.assistantMessageWrites = [];
  return replayed;
}

function consumeNextTimingCandidate(
  manager: SessionManager,
  session: SessionState,
  input: { runId?: string | undefined; provider: string; model: string },
): ModelCallRecord | undefined {
  const key = modelTimingLlmKey({
    sessionId: session.sessionId,
    runId: input.runId,
    provider: input.provider,
    model: input.model,
  });
  const records = manager.state.modelTimingsByLlmKey.get(key) ?? [];
  const candidate = records.find((record) => record.sessionKey === session.sessionId && !record.consumed);
  if (!candidate) {
    return undefined;
  }
  candidate.consumed = true;
  return candidate;
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

function emitModelTimingSummaryMark(
  manager: SessionManager,
  session: SessionState,
  records: ModelCallRecord[],
): void {
  manager.emitCapturedUnderSession("model_call_timing_unmatched", session, () => {
    emitMark({
      nf: manager.nf,
      state: manager.state,
      session,
      name: "openclaw.model_call_timing_unmatched",
      data: toJsonRecord({
        count: records.length,
        sampleCallIds: records.slice(0, 5).map((record) => record.callId),
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
    historyMessages: snapshotMessages(event.historyMessages),
    imagesCount: event.imagesCount,
    observedAtMs: Date.now(),
    ...(event.systemPrompt === undefined ? {} : { systemPrompt: event.systemPrompt }),
  };
}

function existingSessionForMessageWrite(
  manager: SessionManager,
  event: PluginHookBeforeMessageWriteEvent,
  ctx: PluginHookBeforeMessageWriteContext,
): SessionState | undefined {
  const key = resolveSessionKey(manager.state, {
    sessionKey: event.sessionKey ?? ctx.sessionKey,
  });
  return key === undefined ? undefined : manager.state.sessions.get(key);
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
  if (isRecord(last) && last.role === "user" && extractTextBlocks(last).join("\n") === prompt) {
    return historyMessages;
  }
  return [...historyMessages, { role: "user", content: prompt }];
}

function sanitizePromptMessage(message: unknown, config: NemoFlowHookBackendConfig): unknown {
  if (!isRecord(message)) {
    return message;
  }

  let sanitized: Record<string, unknown> = { ...message };
  if ((sanitized.role === "tool" || sanitized.role === "toolResult") && config.capture.stripToolResults) {
    sanitized = { ...sanitized, content: { stripped: true } };
  }
  if (sanitized.role === "assistant" && config.capture.stripToolArgs) {
    sanitized = stripAssistantToolArgs(sanitized);
  } else if (Array.isArray(sanitized.content)) {
    sanitized = { ...sanitized, content: sanitized.content.map(stripLargeAssistantContentFields) };
  }
  return sanitized;
}

function stripAssistantToolArgs(message: Record<string, unknown>): Record<string, unknown> {
  const stripped: Record<string, unknown> = { ...message };
  if (Array.isArray(stripped.toolCalls)) {
    stripped.toolCalls = stripped.toolCalls.map(stripToolCallArgs);
  }
  if (Array.isArray(stripped.tool_calls)) {
    stripped.tool_calls = stripped.tool_calls.map(stripToolCallArgs);
  }
  if (Array.isArray(stripped.content)) {
    stripped.content = stripped.content.map((item) =>
      isToolCallLike(item) ? stripToolCallArgs(item) : stripLargeAssistantContentFields(item),
    );
  }
  return stripped;
}

function stripToolCallArgs(value: unknown): unknown {
  if (!isRecord(value)) {
    return value;
  }
  const stripped: Record<string, unknown> = { ...value };
  for (const key of ["args", "arguments", "input", "params"]) {
    if (stripped[key] !== undefined) {
      stripped[key] = { stripped: true };
    }
  }
  return stripped;
}

function stripLargeAssistantContentFields(value: unknown): unknown {
  if (!isRecord(value)) {
    return value;
  }
  if (value.type === "thinking") {
    return { type: "thinking", stripped: true };
  }
  const stripped: Record<string, unknown> = { ...value };
  if (stripped.thinking !== undefined) {
    stripped.thinking = { stripped: true };
  }
  if (stripped.thinkingSignature !== undefined) {
    stripped.thinkingSignature = { stripped: true };
  }
  return stripped;
}

function responseContent(assistantTexts: string[], assistantToolCallNames: string[]): string | undefined {
  const text = assistantTexts.join("\n").trim();
  if (text.length > 0) {
    return text;
  }
  if (assistantToolCallNames.length > 0) {
    return `tool calls: ${assistantToolCallNames.join(", ")}`;
  }
  return undefined;
}

function lastAssistantText(messages: unknown[]): string | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (!isRecord(message) || message.role !== "assistant") {
      continue;
    }
    const text = extractTextBlocks(message).join("\n").trim();
    if (text.length > 0) {
      return text;
    }
  }
  return undefined;
}

function finalOutputFromAgentEnd(
  messages: unknown[],
  event: PluginHookAgentEndEvent,
  runId?: string,
): JsonRecord | undefined {
  const lastText = lastAssistantText(messages);
  if (lastText) {
    return toJsonRecord({
      content: lastText,
      source: "openclaw.agent_end",
      runId,
      success: event.success,
    });
  }
  if (event.error) {
    return toJsonRecord({
      source: "openclaw.agent_end",
      runId,
      success: event.success,
      error: event.error,
    });
  }
  return undefined;
}

function extractTextBlocks(message: Record<string, unknown>): string[] {
  const content = message.content;
  if (typeof content === "string" && content.length > 0) {
    return [content];
  }
  if (!Array.isArray(content)) {
    return [];
  }
  const texts: string[] = [];
  for (const item of content) {
    if (typeof item === "string") {
      texts.push(item);
    } else if (isRecord(item) && typeof item.text === "string") {
      texts.push(item.text);
    }
  }
  return texts;
}

function extractToolCalls(message: Record<string, unknown>): unknown[] {
  if (Array.isArray(message.toolCalls)) {
    return message.toolCalls;
  }
  if (Array.isArray(message.tool_calls)) {
    return message.tool_calls;
  }
  const content = message.content;
  if (!Array.isArray(content)) {
    return [];
  }
  return content.filter(
    (item) => isToolCallLike(item),
  );
}

function isToolCallLike(value: unknown): boolean {
  return (
    isRecord(value) &&
    (value.type === "toolCall" ||
      value.type === "tool_use" ||
      value.type === "tool-call" ||
      value.toolName !== undefined ||
      value.name !== undefined)
  );
}

function toolCallNames(toolCalls: unknown[] | undefined): string[] {
  if (!Array.isArray(toolCalls)) {
    return [];
  }
  const names: string[] = [];
  for (const toolCall of toolCalls) {
    if (!isRecord(toolCall)) {
      continue;
    }
    const name =
      stringField(toolCall, "name") ??
      stringField(toolCall, "toolName") ??
      stringField(toolCall, "functionName");
    if (name) {
      names.push(name);
    }
  }
  return names;
}

function mapHookUsage(usage: unknown): PluginHookLlmOutputEvent["usage"] | undefined {
  const mapped = mapUsage(usage);
  if (!mapped) {
    return undefined;
  }
  const hookUsage: NonNullable<PluginHookLlmOutputEvent["usage"]> = {};
  if (mapped.prompt_tokens !== undefined) {
    hookUsage.input = mapped.prompt_tokens;
  }
  if (mapped.completion_tokens !== undefined) {
    hookUsage.output = mapped.completion_tokens;
  }
  if (mapped.cache_read_tokens !== undefined) {
    hookUsage.cacheRead = mapped.cache_read_tokens;
  }
  if (mapped.cache_write_tokens !== undefined) {
    hookUsage.cacheWrite = mapped.cache_write_tokens;
  }
  if (mapped.total_tokens !== undefined) {
    hookUsage.total = mapped.total_tokens;
  }
  if (mapped.cost_usd !== undefined) {
    hookUsage.cost = { total: mapped.cost_usd };
  }
  return Object.keys(hookUsage).length > 0 ? hookUsage : undefined;
}

function hasTrajectoryReplay(session: SessionState, runId?: string): boolean {
  return session.trajectoryReplayedRuns?.has(trajectoryRunKey(session, runId)) === true;
}

function rememberAgentRunInputSnapshot(
  session: SessionState,
  runId: string | undefined,
  historyMessages: unknown[],
  prompt: string,
  maxSnapshots: number,
): void {
  const runKey = trajectoryRunKey(session, runId);
  session.agentRunInputSnapshots ??= new Map();
  if (!session.agentRunInputSnapshots.has(runKey)) {
    session.agentRunInputSnapshots.set(runKey, {
      historyMessageCount: historyMessages.length,
      historyMessages: snapshotMessages(historyMessages),
      observedAtMs: Date.now(),
      prompt,
    });
  }
  while (session.agentRunInputSnapshots.size > maxSnapshots) {
    const oldest = session.agentRunInputSnapshots.keys().next().value;
    if (oldest === undefined) {
      break;
    }
    session.agentRunInputSnapshots.delete(oldest);
  }
}

function snapshotMessages(messages: unknown[]): unknown[] {
  const snapshot = toJsonValue(messages);
  return Array.isArray(snapshot) ? snapshot : [];
}

function initialHistoryFromLlmInputSnapshot(session: SessionState): unknown[] {
  let snapshot: { historyMessages: unknown[]; observedAtMs: number; prompt: string } | undefined;
  for (const current of session.agentRunInputSnapshots?.values() ?? []) {
    if (!snapshot || current.observedAtMs > snapshot.observedAtMs) {
      snapshot = current;
    }
  }
  if (!snapshot) {
    return [];
  }
  return appendPromptIfMissing([...snapshot.historyMessages], snapshot.prompt);
}

function currentRunAgentMessages(session: SessionState, runId: string | undefined, messages: unknown[]): unknown[] {
  const inputSnapshot = session.agentRunInputSnapshots?.get(trajectoryRunKey(session, runId));
  if (!inputSnapshot || inputSnapshot.historyMessageCount <= 0) {
    return messages;
  }
  if (inputSnapshot.historyMessageCount <= messages.length) {
    return messages.slice(inputSnapshot.historyMessageCount);
  }
  const promptIndex = findCurrentPromptIndex(messages, inputSnapshot.prompt);
  return promptIndex === undefined ? [] : messages.slice(promptIndex);
}

function findCurrentPromptIndex(messages: unknown[], prompt: string): number | undefined {
  if (!prompt) {
    return undefined;
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (!isRecord(message) || message.role !== "user") {
      continue;
    }
    if (extractTextBlocks(message).join("\n") === prompt) {
      return index;
    }
  }
  return undefined;
}

function cleanupAgentRunReplayBookkeeping(session: SessionState, runKey: string): void {
  session.agentRunInputSnapshots?.delete(runKey);
  session.hookLlmOutputReplayCounts?.delete(runKey);
}

function markTrajectoryReplay(session: SessionState, runKey: string, maxRuns: number): void {
  session.trajectoryReplayedRuns ??= new Set();
  session.trajectoryReplayedRuns.delete(runKey);
  session.trajectoryReplayedRuns.add(runKey);
  while (session.trajectoryReplayedRuns.size > maxRuns) {
    const oldest = session.trajectoryReplayedRuns.values().next().value;
    if (oldest === undefined) {
      break;
    }
    session.trajectoryReplayedRuns.delete(oldest);
  }
}

function hookLlmOutputReplayCount(session: SessionState, runId?: string): number {
  return session.hookLlmOutputReplayCounts?.get(trajectoryRunKey(session, runId)) ?? 0;
}

function incrementHookLlmOutputReplayCount(session: SessionState, runId: string | undefined, maxRuns: number): void {
  const runKey = trajectoryRunKey(session, runId);
  session.hookLlmOutputReplayCounts ??= new Map();
  const nextCount = hookLlmOutputReplayCount(session, runId) + 1;
  session.hookLlmOutputReplayCounts.delete(runKey);
  session.hookLlmOutputReplayCounts.set(runKey, nextCount);
  while (session.hookLlmOutputReplayCounts.size > maxRuns) {
    const oldest = session.hookLlmOutputReplayCounts.keys().next().value;
    if (oldest === undefined) {
      break;
    }
    session.hookLlmOutputReplayCounts.delete(oldest);
  }
}

function trajectoryRunKey(session: SessionState, runId?: string): string {
  return runId ?? session.sessionId;
}

function mapUsage(usage: unknown): Record<string, number> | undefined {
  if (!isRecord(usage)) {
    return undefined;
  }
  const mapped: Record<string, number> = {};
  const input = numberField(usage, "input") ?? numberField(usage, "prompt_tokens");
  const output = numberField(usage, "output") ?? numberField(usage, "completion_tokens");
  const cacheRead = numberField(usage, "cacheRead") ?? numberField(usage, "cache_read_tokens");
  const cacheWrite = numberField(usage, "cacheWrite") ?? numberField(usage, "cache_write_tokens");
  const total = numberField(usage, "total") ?? numberField(usage, "totalTokens") ?? numberField(usage, "total_tokens");
  const totalCanIncludeCompletion = total === undefined || output === undefined || total >= output;
  const prompt = total !== undefined && output !== undefined && totalCanIncludeCompletion ? total - output : input;
  const totalCanIncludePrompt = total === undefined || prompt === undefined || total >= prompt;
  const normalizedTotal = totalCanIncludeCompletion && totalCanIncludePrompt ? total : undefined;
  const costTotal = isRecord(usage.cost) ? numberField(usage.cost, "total") : numberField(usage, "cost_usd");
  if (prompt !== undefined) {
    mapped.prompt_tokens = prompt;
  }
  if (output !== undefined) {
    mapped.completion_tokens = output;
  }
  if (cacheRead !== undefined) {
    mapped.cached_tokens = cacheRead;
    mapped.cache_read_tokens = cacheRead;
  }
  if (cacheWrite !== undefined) {
    mapped.cache_write_tokens = cacheWrite;
  }
  if (normalizedTotal !== undefined) {
    mapped.total_tokens = normalizedTotal;
  } else if (total === undefined && (prompt !== undefined || output !== undefined)) {
    mapped.total_tokens = (prompt ?? 0) + (output ?? 0);
  }
  if (costTotal !== undefined) {
    mapped.cost_usd = costTotal;
  }
  return Object.keys(mapped).length > 0 ? mapped : undefined;
}

function stringField(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function numberField(record: Record<string, unknown>, key: string): number | undefined {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
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
