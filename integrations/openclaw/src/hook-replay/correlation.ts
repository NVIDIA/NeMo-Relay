// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

export function tupleKey(parts: unknown[]): string {
  return JSON.stringify(parts.map((part) => (typeof part === "string" && part.length > 0 ? part : null)));
}

export function llmKey(input: LlmKeyInput): string {
  return tupleKey([input.sessionId, input.runId, input.provider, input.model]);
}

export function modelTimingKey(input: ModelTimingKeyInput): string {
  return tupleKey([input.runId, input.callId]);
}

export function modelTimingLlmKey(input: LlmKeyInput): string {
  return tupleKey([input.sessionId, input.runId, input.provider, input.model]);
}

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

export function nowMicros(): number {
  return Date.now() * 1000;
}

export function startMicrosFromDuration(endMicros: number, durationMs: number | undefined): number | null {
  return durationMs === undefined ? null : endMicros - Math.round(durationMs * 1000);
}

function recordTimestamp(record: TimestampedRecord): number {
  return record.observedAtMs ?? record.endedAtMs ?? record.startedAtMs ?? 0;
}
