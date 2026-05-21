// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Correlation-key and timestamp utilities for hook replay.
 *
 * OpenClaw emits request, response, timing, and tool events separately. These
 * helpers keep key construction consistent across the replay modules.
 */
export type LlmKeyInput = {
  sessionId?: string | undefined;
  runId?: string | undefined;
  provider?: string | undefined;
  model?: string | undefined;
};

export type ModelTimingKeyInput = {
  runId: string;
  callId: string;
};

export type TimestampedRecord = {
  observedAtMs?: number | undefined;
  startedAtMs?: number | undefined;
  endedAtMs?: number | undefined;
};

/** Serialize correlation tuple parts while preserving empty or missing fields as null. */
export function tupleKey(parts: unknown[]): string {
  return JSON.stringify(parts.map((part) => (typeof part === 'string' && part.length > 0 ? part : null)));
}

/** Build the best available key for pairing public llm_input and llm_output hooks. */
export function llmKey(input: LlmKeyInput): string {
  return tupleKey([input.sessionId, input.runId, input.provider, input.model]);
}

/** Build the stronger key for model timing events that include a provider call id. */
export function modelTimingKey(input: ModelTimingKeyInput): string {
  return tupleKey([input.runId, input.callId]);
}

// Model timing fallback uses the same provider/model tuple as LLM replay.
export const modelTimingLlmKey = llmKey;

/** Remove correlation records that are too old to safely pair with later events. */
export function evictExpiredRecords<T extends TimestampedRecord>(
  map: Map<string, T[]>,
  nowMs: number,
  ttlMs: number,
): void {
  for (const [key, records] of map) {
    const retained = records.filter((record) => nowMs - recordTimestamp(record) <= ttlMs);
    if (retained.length === 0) {
      map.delete(key);
    } else {
      map.set(key, retained);
    }
  }
}

/** Return wall-clock microseconds for NeMo Relay span APIs. */
export function nowMicros(): number {
  return Date.now() * 1000;
}

/** Infer a start timestamp when OpenClaw gives end time plus duration. */
export function startMicrosFromDuration(endMicros: number, durationMs: number | undefined): number | null {
  return durationMs === undefined ? null : endMicros - Math.round(durationMs * 1000);
}

/** Select the most recent timestamp field available on a correlation record. */
function recordTimestamp(record: TimestampedRecord): number {
  return record.observedAtMs ?? record.endedAtMs ?? record.startedAtMs ?? 0;
}
