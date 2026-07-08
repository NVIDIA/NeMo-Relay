// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tool-call replay from OpenClaw hooks into NeMo Relay spans.
 *
 * Tool payloads can be large or sensitive, so this module applies capture policy
 * before exporting arguments/results while keeping enough metadata for debugging.
 */
import type {
  PluginHookAfterToolCallEvent,
  PluginHookBeforeToolCallEvent,
  PluginHookToolContext,
} from '../openclaw-hook-types.js';
import { blockedToolDetails, emitMark, errorToJson, toJsonRecord, toJsonValue } from './marks.js';
import { ensureSession, type SessionManager } from './session.js';
import { nowMicros, startMicrosFromDuration } from './correlation.js';
import {
  detectSkillLoads,
  SKILL_LOADS_METADATA_KEY,
  type SkillLoadDetection,
} from './skill-load.js';
import type { SessionState } from './session.js';
import type { NemoRelayRuntimeModule } from '../modules.js';
import type { JsonObject as JsonRecord } from 'nemo-relay-node/typed';

export type EagerSkillToolCall = {
  handle: ReturnType<NemoRelayRuntimeModule['toolCall']>;
  observedAtMs: number;
};

/** Run NeMo Relay tool conditional-execution guardrails before OpenClaw executes a tool. */
export async function guardBeforeToolCall(
  manager: SessionManager,
  event: PluginHookBeforeToolCallEvent,
  ctx: PluginHookToolContext,
): Promise<ReturnType<typeof ensureSession>> {
  const observedAtMicros = nowMicros();
  const session = ensureSession(manager, {
    sessionId: ctx.sessionId,
    sessionKey: ctx.sessionKey,
    runId: event.runId ?? ctx.runId,
    agentId: ctx.agentId,
    source: 'lazy_session',
    timestamp: observedAtMicros,
  });
  const args = toJsonValue(event.params ?? {});

  if (session) {
    const previousStack = manager.nf.currentScopeStack();
    try {
      manager.nf.setThreadScopeStack(session.stack);
      await manager.nf.toolConditionalExecution(event.toolName, args);
    } finally {
      manager.nf.setThreadScopeStack(previousStack);
    }
  } else {
    await manager.nf.toolConditionalExecution(event.toolName, args);
  }
  return session;
}

/** Start a stable OpenClaw skill-read tool span before argument capture policy strips its params. */
export function startEagerSkillToolCall(
  manager: SessionManager,
  event: PluginHookBeforeToolCallEvent,
  ctx: PluginHookToolContext,
  session: SessionState,
  detections: SkillLoadDetection[],
  timestamp: number,
): EagerSkillToolCall | undefined {
  const metadata = toolMetadata(event, ctx, session, 'openclaw.before_tool_call');
  metadata[SKILL_LOADS_METADATA_KEY] = detections.map((detection) => ({
    skill_name: detection.skillName,
    source: detection.source,
  }));
  let handle: ReturnType<NemoRelayRuntimeModule['toolCall']> | undefined;
  manager.emitCapturedUnderSession('before_tool_call', session, () => {
    handle = manager.nf.toolCall(
      event.toolName,
      toolArgsPayload(manager, event.params),
      session.rootHandle,
      null,
      null,
      metadata,
      event.toolCallId ?? ctx.toolCallId ?? null,
      timestamp,
    );
  });
  if (!handle) {
    return undefined;
  }
  manager.state.counters.marksEmitted += detections.length;
  return {
    handle,
    observedAtMs: Math.trunc(timestamp / 1000),
  };
}

/** Convert one OpenClaw after_tool_call event into a NeMo Relay tool span or blocked-tool mark. */
export function replayAfterToolCall(
  manager: SessionManager,
  event: PluginHookAfterToolCallEvent,
  ctx: PluginHookToolContext,
  eagerToolCall?: EagerSkillToolCall,
): void {
  const endMicros = nowMicros();
  const sessionTimestamp = startMicrosFromDuration(endMicros, event.durationMs) ?? endMicros;
  const session = ensureSession(manager, {
    sessionId: ctx.sessionId,
    sessionKey: ctx.sessionKey,
    runId: event.runId ?? ctx.runId,
    agentId: ctx.agentId,
    source: 'lazy_session',
    timestamp: sessionTimestamp,
  });

  if (!session) {
    return;
  }

  const metadata = toolMetadata(event, ctx, session, 'openclaw.after_tool_call');
  const endPayload = toJsonValue(
    manager.config.capture.stripToolResults
      ? toolDisplayPayload(event, true)
      : event.error
        ? { ...toolDisplayPayload(event, false), error: errorToJson(event.error), result: event.result ?? null }
        : { ...toolDisplayPayload(event, false), result: event.result ?? null },
  );
  if (eagerToolCall) {
    manager.emitCapturedUnderSession('after_tool_call', session, () => {
      manager.nf.toolCallEnd(eagerToolCall.handle, endPayload, null, metadata, endMicros);
      manager.state.counters.toolSpansReplayed += 1;
    });
    return;
  }

  const blockedDetails = blockedToolDetails(event, { runId: event.runId ?? ctx.runId });
  if (blockedDetails) {
    manager.emitCapturedUnderSession('openclaw.tool_blocked', session, () => {
      emitMark({
        nf: manager.nf,
        state: manager.state,
        session,
        name: 'openclaw.tool_blocked',
        data: blockedDetails,
        metadata: toJsonRecord({
          source: 'openclaw.after_tool_call',
          hook_event_name: 'after_tool_call',
          sessionId: session.sessionId,
          sessionKey: session.sessionKey,
          agentId: session.agentId,
          runId: event.runId ?? ctx.runId,
          toolCallId: event.toolCallId ?? ctx.toolCallId,
        }),
        timestamp: endMicros,
      });
    });
    return;
  }

  const skillLoads = detectSkillLoads(event.toolName, event.params);
  if (skillLoads.length > 0) {
    metadata[SKILL_LOADS_METADATA_KEY] = skillLoads.map((detection) => ({
      skill_name: detection.skillName,
      source: detection.source,
    }));
  }

  manager.emitCapturedUnderSession('after_tool_call', session, () => {
    const handle = manager.nf.toolCall(
      event.toolName,
      toolArgsPayload(manager, event.params),
      session.rootHandle,
      null,
      null,
      metadata,
      event.toolCallId ?? ctx.toolCallId ?? null,
      startMicrosFromDuration(endMicros, event.durationMs),
    );
    manager.state.counters.marksEmitted += skillLoads.length;
    manager.nf.toolCallEnd(handle, endPayload, null, metadata, endMicros);
    manager.state.counters.toolSpansReplayed += 1;
  });
}

function toolMetadata(
  event: PluginHookBeforeToolCallEvent | PluginHookAfterToolCallEvent,
  ctx: PluginHookToolContext,
  session: SessionState,
  source: string,
): JsonRecord {
  return toJsonRecord({
    source,
    runId: event.runId ?? ctx.runId,
    sessionId: session.sessionId,
    sessionKey: session.sessionKey,
    agentId: session.agentId,
    toolCallId: event.toolCallId ?? ctx.toolCallId,
    durationMs: 'durationMs' in event ? event.durationMs : undefined,
  });
}

function toolArgsPayload(manager: SessionManager, params: unknown) {
  return toJsonValue(
    manager.config.capture.stripToolArgs
      ? {
          stripped: true,
          argKeys: params && typeof params === 'object' && !Array.isArray(params) ? Object.keys(params) : undefined,
        }
      : (params ?? {}),
  );
}

/** Build the compact default tool output shown in trace UIs. */
function toolDisplayPayload(event: PluginHookAfterToolCallEvent, stripped: boolean): Record<string, unknown> {
  const hasError = Boolean(event.error);
  return {
    content: `Tool ${event.toolName} ${hasError ? 'failed' : 'completed'}.`,
    openclaw: {
      toolName: event.toolName,
      toolCallId: event.toolCallId,
      durationMs: event.durationMs,
      hasError,
      stripped,
      resultKeys: resultKeys(event.result),
    },
  };
}

/** Include result keys as a low-noise hint when full tool results are stripped. */
function resultKeys(result: unknown): string[] | undefined {
  return result && typeof result === 'object' && !Array.isArray(result) ? Object.keys(result) : undefined;
}
