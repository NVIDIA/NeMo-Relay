// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PluginHookAfterToolCallEvent, PluginHookToolContext } from "../openclaw-hook-types.js";
import { blockedToolDetails, emitMark, errorToJson, toJsonRecord, toJsonValue } from "./marks.js";
import { ensureSession, type SessionManager } from "./session.js";
import { nowMicros, startMicrosFromDuration } from "./correlation.js";

export function replayAfterToolCall(
  manager: SessionManager,
  event: PluginHookAfterToolCallEvent,
  ctx: PluginHookToolContext,
): void {
  const session = ensureSession(manager, {
    sessionId: ctx.sessionId,
    sessionKey: ctx.sessionKey,
    runId: event.runId ?? ctx.runId,
    agentId: ctx.agentId,
    source: "lazy_session",
  });

  const blockedDetails = blockedToolDetails(event, { runId: event.runId ?? ctx.runId });
  if (session && blockedDetails) {
    manager.emitCapturedUnderSession("openclaw.tool_blocked", session, () => {
      emitMark({
        nf: manager.nf,
        state: manager.state,
        session,
        name: "openclaw.tool_blocked",
        data: blockedDetails,
      });
    });
    return;
  }

  if (!session) {
    return;
  }

  const endMicros = nowMicros();
  const metadata = toJsonRecord({
    source: "openclaw.after_tool_call",
    runId: event.runId ?? ctx.runId,
    sessionId: ctx.sessionId,
    sessionKey: ctx.sessionKey,
    toolCallId: event.toolCallId ?? ctx.toolCallId,
    durationMs: event.durationMs,
  });
  const argsPayload = toJsonValue(
    manager.config.capture.stripToolArgs
      ? {
          stripped: true,
          argKeys:
            event.params && typeof event.params === "object" && !Array.isArray(event.params)
              ? Object.keys(event.params)
              : undefined,
        }
      : event.params ?? {},
  );
  const endPayload = toJsonValue(
    manager.config.capture.stripToolResults
      ? toolDisplayPayload(event, true)
      : event.error
        ? { ...toolDisplayPayload(event, false), error: errorToJson(event.error), result: event.result ?? null }
        : { ...toolDisplayPayload(event, false), result: event.result ?? null },
  );

  manager.emitCapturedUnderSession("after_tool_call", session, () => {
    const handle = manager.nf.toolCall(
      event.toolName,
      argsPayload,
      session.rootHandle,
      null,
      null,
      metadata,
      event.toolCallId ?? ctx.toolCallId ?? null,
      startMicrosFromDuration(endMicros, event.durationMs),
    );
    manager.nf.toolCallEnd(handle, endPayload, null, metadata, endMicros);
    manager.state.counters.toolSpansReplayed += 1;
  });
}

function toolDisplayPayload(event: PluginHookAfterToolCallEvent, stripped: boolean): Record<string, unknown> {
  const hasError = Boolean(event.error);
  return {
    content: `Tool ${event.toolName} ${hasError ? "failed" : "completed"}.`,
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

function resultKeys(result: unknown): string[] | undefined {
  return result && typeof result === "object" && !Array.isArray(result) ? Object.keys(result) : undefined;
}
