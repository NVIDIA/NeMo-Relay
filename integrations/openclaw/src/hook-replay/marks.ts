// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared event/mark emission and JSON normalization helpers.
 *
 * Hook payloads can contain undefined values, circular objects, errors, and
 * prototype-sensitive keys. This module normalizes them before they cross the
 * NeMo Relay NAPI boundary.
 */
import type { PluginHookAfterToolCallEvent } from '../openclaw-hook-types.js';
import type { JsonObject as JsonRecord, JsonValue } from 'nemo-relay-node/typed';
import type { HookReplayBackendState, SessionState } from './session.js';
import type { NemoRelayRuntimeModule } from '../modules.js';

/** Emit a NeMo Relay event under an existing OpenClaw session root span. */
export function emitMark(params: {
  nf: NemoRelayRuntimeModule;
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

/** Return a blocked-tool event payload when OpenClaw denied the tool call. */
export function blockedToolDetails(
  event: PluginHookAfterToolCallEvent,
  context?: { runId?: string | undefined },
): JsonRecord | undefined {
  const details = resultDetails(event.result);
  if (details?.status !== 'blocked') {
    return undefined;
  }

  return toJsonRecord({
    toolName: event.toolName,
    toolCallId: event.toolCallId,
    runId: event.runId ?? context?.runId,
    blocked: true,
    deniedReason: typeof details.deniedReason === 'string' ? details.deniedReason : undefined,
    durationMs: event.durationMs,
  });
}

/** Convert an object to a JSON record, dropping undefined fields recursively. */
export function toJsonRecord(input: Record<string, unknown>): JsonRecord {
  return stripUndefined(input, new WeakSet<object>());
}

/** Convert arbitrary hook payload data into NAPI-safe JSON. */
export function toJsonValue(input: unknown): JsonValue {
  return normalizeJsonValue(input, new WeakSet<object>());
}

/** Preserve useful error fields in telemetry without requiring Error instances. */
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

/** Extract OpenClaw tool result details when the result uses that envelope. */
function resultDetails(result: unknown): Record<string, unknown> | undefined {
  if (!isRecord(result)) {
    return undefined;
  }
  const details = result.details;
  return isRecord(details) ? details : undefined;
}

/** Strip undefined properties and protect prototype-like keys during JSON conversion. */
function stripUndefined(input: Record<string, unknown>, seen: WeakSet<object>): JsonRecord {
  const output: JsonRecord = {};
  for (const [key, value] of Object.entries(input)) {
    if (value !== undefined) {
      const normalized = normalizeJsonValue(value, seen);
      if (key === '__proto__') {
        Object.defineProperty(output, key, {
          configurable: true,
          enumerable: true,
          value: normalized,
          writable: true,
        });
      } else {
        output[key] = normalized;
      }
    }
  }
  return output;
}

/** Normalize any hook value into JSON, replacing cycles and unsupported primitives. */
function normalizeJsonValue(value: unknown, seen: WeakSet<object>): JsonValue {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (Array.isArray(value)) {
    if (seen.has(value)) {
      return '[Circular]';
    }
    seen.add(value);
    const out = value.map((item) => normalizeJsonValue(item, seen));
    seen.delete(value);
    return out;
  }
  if (isRecord(value)) {
    if (seen.has(value)) {
      return '[Circular]';
    }
    seen.add(value);
    const out = stripUndefined(value, seen);
    seen.delete(value);
    return out;
  }
  return String(value);
}

/** Narrow unknown values to plain records for payload traversal. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
