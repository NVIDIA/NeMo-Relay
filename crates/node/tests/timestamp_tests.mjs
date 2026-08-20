// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  ScopeType,
  deregisterSubscriber,
  event,
  flushSubscribers,
  llmCall,
  llmCallEnd,
  popScope,
  pushScope,
  registerSubscriber,
  toolCall,
  toolCallEnd,
} = require('../index.js');

const timestampMicros = (value, micros = 0) => Date.parse(value) * 1000 + micros;
const parseTimestampMicros = (value) => {
  const match = /^(.*?)(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$/.exec(value);
  assert.ok(match, `invalid timestamp ${value}`);
  const [, base, fraction = '', zone] = match;
  return Date.parse(`${base}${zone}`) * 1000 + Number(fraction.padEnd(6, '0').slice(0, 6));
};

test('manual lifecycle APIs accept optional timestamp arguments', async () => {
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
  const subscriberName = `node_timestamp_${Date.now()}`;
  registerSubscriber(subscriberName, (event) => events.push(event));
  const scope = pushScope('node_ts_scope', ScopeType.Agent, null, null, null, null, null, timestamps[0]);
  event('node_ts_mark', scope, null, null, timestamps[1]);
  const tool = toolCall('node_ts_tool', { x: 1 }, null, null, null, null, null, timestamps[2]);
  toolCallEnd(tool, { result: { ok: true } }, null, null, timestamps[3]);
  const llm = llmCall(
    'node_ts_llm',
    { headers: {}, content: { messages: [], model: 'test-model' } },
    null,
    null,
    null,
    null,
    null,
    timestamps[4],
  );
  llmCallEnd(llm, { ok: true }, null, null, timestamps[5]);
  popScope(scope, null, timestamps[6]);
  await flushSubscribers();
  deregisterSubscriber(subscriberName);

  assert.deepEqual(
    events
      .filter((event) => event.name.startsWith('node_ts_'))
      .map((event) => [event.name, parseTimestampMicros(event.timestamp)]),
    [
      ['node_ts_scope', timestamps[0]],
      ['node_ts_mark', timestamps[1]],
      ['node_ts_tool', timestamps[2]],
      ['node_ts_tool', timestamps[3]],
      ['node_ts_llm', timestamps[4]],
      ['node_ts_llm', timestamps[5]],
      ['node_ts_scope', timestamps[6]],
    ],
  );
});

test('manual lifecycle APIs reject invalid timestamp microseconds', () => {
  const invalidTimestamps = [
    timestampMicros('2026-01-01T00:00:00.000Z') + 0.5,
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.MAX_SAFE_INTEGER + 1,
    Number.MIN_SAFE_INTEGER - 1,
  ];

  for (const badTimestamp of invalidTimestamps) {
    assert.throws(
      () => pushScope('node_bad_ts_scope_start', ScopeType.Agent, null, null, null, null, null, badTimestamp),
      /safe integer number of Unix microseconds/,
    );

    const scope = pushScope('node_bad_ts_scope', ScopeType.Agent);
    try {
      assert.throws(
        () => event('node_bad_ts_mark', scope, null, null, badTimestamp),
        /safe integer number of Unix microseconds/,
      );

      assert.throws(
        () => toolCall('node_bad_ts_tool_start', { x: 1 }, null, null, null, null, null, badTimestamp),
        /safe integer number of Unix microseconds/,
      );

      const tool = toolCall('node_bad_ts_tool', { x: 1 });
      try {
        assert.throws(
          () => toolCallEnd(tool, { result: { ok: true } }, null, null, badTimestamp),
          /safe integer number of Unix microseconds/,
        );
      } finally {
        toolCallEnd(tool, { result: { ok: true } });
      }

      assert.throws(
        () =>
          llmCall(
            'node_bad_ts_llm_start',
            { headers: {}, content: { messages: [], model: 'test-model' } },
            null,
            null,
            null,
            null,
            null,
            badTimestamp,
          ),
        /safe integer number of Unix microseconds/,
      );

      const llm = llmCall('node_bad_ts_llm', { headers: {}, content: { messages: [], model: 'test-model' } });
      try {
        assert.throws(
          () => llmCallEnd(llm, { ok: true }, null, null, badTimestamp),
          /safe integer number of Unix microseconds/,
        );
      } finally {
        llmCallEnd(llm, { ok: true });
      }

      assert.throws(() => popScope(scope, null, badTimestamp), /safe integer number of Unix microseconds/);
    } finally {
      popScope(scope);
    }
  }
});
