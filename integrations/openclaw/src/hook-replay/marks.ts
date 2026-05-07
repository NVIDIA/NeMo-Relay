// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PluginHookAfterToolCallEvent } from "../openclaw-hook-types.js";
import type { JsonRecord, JsonValue } from "../types.js";
import type { HookReplayBackendState, SessionState } from "./session.js";
import type { NemoFlowRuntimeModule } from "../modules.js";

export function emitMark(params: {
  nf: NemoFlowRuntimeModule;
  state: HookReplayBackendState;
  session: SessionState;
  name: string;
  data: JsonRecord;
  timestamp?: number;
}): void {
  if (!params.session.rootHandle) {
    params.state.counters.skippedEvents += 1;
    return;
  }

  params.nf.event(params.name, params.session.rootHandle, params.data, null, params.timestamp ?? null);
  params.state.counters.marksEmitted += 1;
}

export function blockedToolDetails(
  event: PluginHookAfterToolCallEvent,
  context?: { runId?: string | undefined },
): JsonRecord | undefined {
  const details = resultDetails(event.result);
  if (details?.status !== "blocked") {
    return undefined;
  }

  return toJsonRecord({
    toolName: event.toolName,
    toolCallId: event.toolCallId,
    runId: event.runId ?? context?.runId,
    blocked: true,
    deniedReason: typeof details.deniedReason === "string" ? details.deniedReason : undefined,
    durationMs: event.durationMs,
  });
}

export function toJsonRecord(input: Record<string, unknown>): JsonRecord {
  return stripUndefined(input, new WeakSet<object>());
}

export function toJsonValue(input: unknown): JsonValue {
  return normalizeJsonValue(input, new WeakSet<object>());
}

export function errorToJson(error: unknown): JsonRecord {
  if (error instanceof Error) {
    return toJsonRecord({
      name: error.name,
      message: error.message,
      stack: error.stack,
    });
  }
  if (isRecord(error)) {
    return toJsonRecord(error);
  }
  return { message: String(error) };
}

function resultDetails(result: unknown): Record<string, unknown> | undefined {
  if (!isRecord(result)) {
    return undefined;
  }
  const details = result.details;
  return isRecord(details) ? details : undefined;
}

function stripUndefined(input: Record<string, unknown>, seen: WeakSet<object>): JsonRecord {
  const output: JsonRecord = {};
  for (const [key, value] of Object.entries(input)) {
    if (value !== undefined) {
      output[key] = normalizeJsonValue(value, seen);
    }
  }
  return output;
}

function normalizeJsonValue(value: unknown, seen: WeakSet<object>): JsonValue {
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (Array.isArray(value)) {
    if (seen.has(value)) {
      return "[Circular]";
    }
    seen.add(value);
    const out = value.map((item) => normalizeJsonValue(item, seen));
    seen.delete(value);
    return out;
  }
  if (isRecord(value)) {
    if (seen.has(value)) {
      return "[Circular]";
    }
    seen.add(value);
    const out = stripUndefined(value, seen);
    seen.delete(value);
    return out;
  }
  return String(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
