// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { makeLlmRequest, resetScopeStack, unique, waitFor, wasm } from './test_support.mjs';

const timestampMicros = (value, micros = 0) => Date.parse(value) * 1000 + micros;
const parseTimestampMicros = (value) => {
  const match = /^(.*?)(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$/.exec(value);
  assert.ok(match, `invalid timestamp ${value}`);
  const [, base, fraction = '', zone] = match;
  return Date.parse(`${base}${zone}`) * 1000 + Number(fraction.padEnd(6, '0').slice(0, 6));
};

test('WebAssembly manual lifecycle APIs accept optional timestamp arguments', async () => {
  const stack = resetScopeStack();
  const events = [];
  const timestamps = [
    timestampMicros('2026-01-01T00:00:00.000Z', 123456),
    timestampMicros('2026-01-01T00:00:01.000Z', 223456),
    timestampMicros('2026-01-01T00:00:02.000Z', 323456),
    timestampMicros('2026-01-01T00:00:03.000Z', 423456),
    timestampMicros('2026-01-01T00:00:04.000Z', 523456),
    timestampMicros('2026-01-01T00:00:05.000Z', 623456),
    timestampMicros('2026-01-01T00:00:06.000Z', 723456),
  ];
  const subscriberName = unique('wasm_timestamp');
  let scope;
  let tool;
  let llm;

  wasm.registerSubscriber(subscriberName, (event) => events.push(event));
  try {
    scope = wasm.pushScope('wasm_ts_scope', wasm.ScopeType.Agent, null, null, null, null, null, timestamps[0]);
    wasm.event('wasm_ts_mark', scope, null, null, timestamps[1]);
    tool = wasm.toolCall('wasm_ts_tool', { x: 1 }, null, null, null, null, null, timestamps[2]);
    wasm.toolCallEnd(tool, { ok: true }, null, null, timestamps[3]);
    llm = wasm.llmCall('wasm_ts_llm', makeLlmRequest(), null, null, null, null, null, timestamps[4]);
    wasm.llmCallEnd(llm, { ok: true }, null, null, timestamps[5]);
    wasm.popScope(scope, null, timestamps[6]);
    await waitFor(() => events.filter((event) => event.name.startsWith('wasm_ts_')).length >= 7);
  } finally {
    wasm.deregisterSubscriber(subscriberName);
    if (llm) {
      llm.free();
    }
    if (tool) {
      tool.free();
    }
    if (scope) {
      scope.free();
    }
    stack.free();
  }

  assert.deepEqual(
    events
      .filter((event) => event.name.startsWith('wasm_ts_'))
      .map((event) => [event.name, parseTimestampMicros(event.timestamp)]),
    [
      ['wasm_ts_scope', timestamps[0]],
      ['wasm_ts_mark', timestamps[1]],
      ['wasm_ts_tool', timestamps[2]],
      ['wasm_ts_tool', timestamps[3]],
      ['wasm_ts_llm', timestamps[4]],
      ['wasm_ts_llm', timestamps[5]],
      ['wasm_ts_scope', timestamps[6]],
    ],
  );
});

test('WebAssembly manual lifecycle APIs reject invalid timestamp microseconds', () => {
  const stack = resetScopeStack();
  const invalidTimestamps = [
    timestampMicros('2026-01-01T00:00:00.000Z') + 0.5,
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.MAX_SAFE_INTEGER + 1,
    Number.MIN_SAFE_INTEGER - 1,
  ];

  try {
    for (const badTimestamp of invalidTimestamps) {
      assert.throws(
        () =>
          wasm.pushScope('wasm_bad_ts_scope_start', wasm.ScopeType.Agent, null, null, null, null, null, badTimestamp),
        /safe integer number of Unix microseconds/,
      );

      const scope = wasm.pushScope('wasm_bad_ts_scope', wasm.ScopeType.Agent);
      try {
        assert.throws(
          () => wasm.event('wasm_bad_ts_mark', scope, null, null, badTimestamp),
          /safe integer number of Unix microseconds/,
        );

        assert.throws(
          () => wasm.toolCall('wasm_bad_ts_tool_start', { x: 1 }, null, null, null, null, null, badTimestamp),
          /safe integer number of Unix microseconds/,
        );

        const tool = wasm.toolCall('wasm_bad_ts_tool', { x: 1 });
        try {
          assert.throws(
            () => wasm.toolCallEnd(tool, { ok: true }, null, null, badTimestamp),
            /safe integer number of Unix microseconds/,
          );
        } finally {
          wasm.toolCallEnd(tool, { ok: true });
          tool.free();
        }

        assert.throws(
          () => wasm.llmCall('wasm_bad_ts_llm_start', makeLlmRequest(), null, null, null, null, null, badTimestamp),
          /safe integer number of Unix microseconds/,
        );

        const llm = wasm.llmCall('wasm_bad_ts_llm', makeLlmRequest());
        try {
          assert.throws(
            () => wasm.llmCallEnd(llm, { ok: true }, null, null, badTimestamp),
            /safe integer number of Unix microseconds/,
          );
        } finally {
          wasm.llmCallEnd(llm, { ok: true });
          llm.free();
        }

        assert.throws(() => wasm.popScope(scope, null, badTimestamp), /safe integer number of Unix microseconds/);
      } finally {
        wasm.popScope(scope);
        scope.free();
      }
    }
  } finally {
    stack.free();
  }
});
