// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';
const lib = require('./index.js');

/** Reader failure with a stable machine-readable reason. */
class DiagnosticsError extends Error {
  constructor(reason) {
    super(reason);
    this.name = 'DiagnosticsError';
    this.reason = reason;
  }
}
const reasons = new Set([
  'not_initialized',
  'fork_requires_exec',
  'configuration_conflict',
  'disabled',
  'busy',
  'queries_disabled',
  'invalid_cursor',
  'page_capacity',
  'allocation_failed',
  'invalid_limits',
  'invalid_query',
  'numeric_out_of_range',
]);
function call(operation) {
  try {
    return operation();
  } catch (error) {
    if (error instanceof DiagnosticsError) throw error;
    throw new DiagnosticsError(reasons.has(error.message) ? error.message : 'unavailable');
  }
}
function queryJson(query = {}) {
  const numeric = new Set([
    'origin_generation',
    'after_sequence',
    'sequence',
    'component_offset',
    'event_offset',
    'scope',
  ]);
  let size = 0;
  let nodes = 0;
  function normalize(value, key = '', depth = 0) {
    if (++nodes > 64 || depth > 4) throw new DiagnosticsError('invalid_query');
    if (value !== null && typeof value === 'object') {
      if (Array.isArray(value)) throw new DiagnosticsError('invalid_query');
      const out = Object.create(null);
      let fields = 0;
      for (const field in value) {
        if (!Object.hasOwn(value, field)) continue;
        if (++fields > 20 || field.length > 128) throw new DiagnosticsError('invalid_query');
        size += Buffer.byteLength(field);
        if (size > 4096) throw new DiagnosticsError('invalid_query');
        const wire = field.replace(/[A-Z]/g, (letter) => '_' + letter.toLowerCase());
        if (Object.hasOwn(out, wire)) throw new DiagnosticsError('invalid_query');
        out[wire] = normalize(value[field], wire, depth + 1);
      }
      return out;
    }
    if (numeric.has(key) && typeof value === 'bigint') {
      if (value < 0n || value > 18446744073709551615n) throw new DiagnosticsError('invalid_query');
      return value.toString();
    }
    if (numeric.has(key) && typeof value === 'number') throw new DiagnosticsError('invalid_query');
    if (typeof value === 'string') {
      size += Buffer.byteLength(value);
      if (size > 4096) throw new DiagnosticsError('invalid_query');
    } else if (
      value !== null &&
      typeof value !== 'boolean' &&
      !(typeof value === 'number' && Number.isSafeInteger(value))
    )
      throw new DiagnosticsError('invalid_query');
    return value;
  }
  const encoded = JSON.stringify(normalize(query));
  if (Buffer.byteLength(encoded) > 4096) throw new DiagnosticsError('invalid_query');
  return encoded;
}

/** Inspection authority. It does not initialize or enable diagnostics. */
class DiagnosticsReader {
  #native;
  constructor(native) {
    this.#native = native;
  }
  /** Return one camelCase page with every diagnostic u64 represented as bigint. */
  snapshot(query) {
    return call(() => this.#native.snapshot(queryJson(query)));
  }
  /** Return portable canonical JSON (snake_case, decimal-string integers). */
  snapshotJson(query) {
    return call(() => this.#native.snapshotJson(queryJson(query)));
  }
  /** Return separate core/copy budgets, occupancy, and logging coverage. */
  info() {
    return call(() => this.#native.info());
  }
}
/** Obtain a reader without starting the service; may throw not_initialized. */
function reader() {
  return call(() => new DiagnosticsReader(lib.diagnosticsReader()));
}
module.exports = { DiagnosticsError, DiagnosticsReader, reader };
