// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  DiagnosticMemoryPage,
  DiagnosticQuery,
  DiagnosticSnapshot,
  DiagnosticsInfo,
  DiagnosticsReader as NativeDiagnosticsReader,
} from '../binding.js';
import * as lib from './index.js';
import {
  MAX_QUERY_BYTES,
  MAX_QUERY_DEPTH,
  MAX_QUERY_FIELD_NAME_BYTES,
  MAX_QUERY_NODES,
  MAX_QUERY_OBJECT_FIELDS,
  NUMERIC_QUERY_FIELDS,
} from './diagnostics_query_contract.js';

export type {
  Classification,
  ComponentIdentity,
  ComponentMemoryFootprint,
  ComponentMemoryUsage,
  ComponentOwnership,
  ComponentSnapshot,
  DiagnosticCapability,
  DiagnosticCorrelation,
  DiagnosticCount,
  DiagnosticCounter,
  DiagnosticCoverage,
  DiagnosticCursor,
  DiagnosticDetail,
  DiagnosticDiscovery,
  DiagnosticEvent,
  DiagnosticLifetimeSummary,
  DiagnosticLoss,
  DiagnosticMemoryPage,
  DiagnosticMemoryUsage,
  DiagnosticNumber,
  DiagnosticOrigin,
  DiagnosticQuery,
  DiagnosticRefreshSummary,
  DiagnosticRetention,
  DiagnosticSnapshot,
  DiagnosticTotals,
  DiagnosticWindow,
  DiagnosticWindowBucket,
  DiagnosticWindowCoverage,
  DiagnosticsInfo,
  EntryMemoryFootprint,
  EntryMemoryUsage,
  MemoryLimitEvent,
  MemorySelectionTotals,
  PluginObservationContext,
  RemoteDiagnosticOrigin,
  ResourceMemoryMeasurement,
  ResourceMemoryMetrics,
  ResourceMemoryNumber,
  ResourceMemorySample,
  ResourceMemoryUnit,
  ResourceMemoryView,
  ResourceProcessIdentity,
  SourceIdentity,
} from '../binding.js';

/** Reader failure with a stable `reason` and no evidence in the message. */
export class DiagnosticsError extends Error {
  readonly reason: DiagnosticsReason;

  constructor(reason: DiagnosticsReason) {
    super(reason);
    this.name = 'DiagnosticsError';
    this.reason = reason;
  }
}

const reasonNames = [
  'not_initialized',
  'fork_requires_exec',
  'configuration_conflict',
  'disabled',
  'busy',
  'query_disabled',
  'unsupported_schema',
  'internal',
  'invalid_cursor',
  'snapshot_capacity',
  'allocation_failed',
  'invalid_limits',
  'invalid_query',
  'invalid_snapshot',
  'invalid_metadata',
  'numeric_out_of_range',
] as const;

export type DiagnosticsReason = (typeof reasonNames)[number] | 'unavailable';

const reasons: ReadonlySet<string> = new Set(reasonNames);

function isReason(value: string): value is DiagnosticsReason {
  return reasons.has(value);
}

function call<T>(operation: () => T): T {
  try {
    return operation();
  } catch (error) {
    if (error instanceof DiagnosticsError) throw error;
    throw new DiagnosticsError(error instanceof Error && isReason(error.message) ? error.message : 'unavailable');
  }
}

function queryJson(query: DiagnosticQuery = {}): string {
  let size = 0;
  let nodes = 0;
  function normalize(value: unknown, key: string, depth: number): unknown {
    if (++nodes > MAX_QUERY_NODES || depth > MAX_QUERY_DEPTH) throw new DiagnosticsError('invalid_query');
    if (value !== null && typeof value === 'object') {
      if (Array.isArray(value)) throw new DiagnosticsError('invalid_query');
      const out: Record<string, unknown> = Object.create(null);
      let fields = 0;
      for (const [field, fieldValue] of Object.entries(value)) {
        if (++fields > MAX_QUERY_OBJECT_FIELDS || Buffer.byteLength(field) > MAX_QUERY_FIELD_NAME_BYTES) {
          throw new DiagnosticsError('invalid_query');
        }
        size += Buffer.byteLength(field);
        if (size > MAX_QUERY_BYTES) throw new DiagnosticsError('invalid_query');
        const wire = field.replace(/[A-Z]/g, (letter) => '_' + letter.toLowerCase());
        if (Object.hasOwn(out, wire)) throw new DiagnosticsError('invalid_query');
        out[wire] = normalize(fieldValue, wire, depth + 1);
      }
      return out;
    }
    return normalizeScalar(value, key);
  }
  function normalizeScalar(value: unknown, key: string): unknown {
    if (NUMERIC_QUERY_FIELDS.has(key) && typeof value === 'bigint') {
      if (value < 0n || value > 18446744073709551615n) throw new DiagnosticsError('invalid_query');
      return value.toString();
    }
    if (NUMERIC_QUERY_FIELDS.has(key) && typeof value === 'number') throw new DiagnosticsError('invalid_query');
    if (typeof value === 'string') {
      size += Buffer.byteLength(value);
      if (size > MAX_QUERY_BYTES) throw new DiagnosticsError('invalid_query');
    } else if (
      value !== null &&
      typeof value !== 'boolean' &&
      !(typeof value === 'number' && Number.isSafeInteger(value))
    ) {
      throw new DiagnosticsError('invalid_query');
    }
    return value;
  }
  const encoded = JSON.stringify(normalize(query, '', 0));
  if (Buffer.byteLength(encoded) > MAX_QUERY_BYTES) throw new DiagnosticsError('invalid_query');
  return encoded;
}

/** Reads diagnostics for this process; keep it from untrusted clients. */
export class DiagnosticsReader {
  readonly #native: NativeDiagnosticsReader;

  constructor(native: NativeDiagnosticsReader) {
    this.#native = native;
  }

  /** Returns one page as objects with `bigint` diagnostic integers. */
  snapshot(query?: DiagnosticQuery): DiagnosticSnapshot {
    return call(() => this.#native.snapshot(queryJson(query)));
  }

  /** Returns one page as canonical JSON with integers as decimal strings. */
  snapshotJson(query?: DiagnosticQuery): string {
    return call(() => this.#native.snapshotJson(queryJson(query)));
  }

  /** Returns one memory page as objects with `bigint` diagnostic integers. */
  inspectMemory(query?: DiagnosticQuery): DiagnosticMemoryPage {
    return call(() => this.#native.inspectMemory(queryJson(query)));
  }

  /** Returns one memory page as canonical JSON with integers as decimal strings. */
  inspectMemoryJson(query?: DiagnosticQuery): string {
    return call(() => this.#native.inspectMemoryJson(queryJson(query)));
  }

  /** Returns budgets, occupancy, and logging coverage. */
  info(): DiagnosticsInfo {
    return call(() => this.#native.info());
  }
}

/** Returns a reader, or throws `not_initialized`. */
export function reader(): DiagnosticsReader {
  return call(() => new DiagnosticsReader(lib.diagnosticsReader()));
}
