// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  pushScope,
  popScope,
  toolCall,
  toolCallEnd,
  toolCallExecute,
  toolCallExecuteAsync,
  toolRequestIntercepts,
  toolConditionalExecution,
  registerToolSanitizeRequestGuardrail,
  deregisterToolSanitizeRequestGuardrail,
  registerToolSanitizeResponseGuardrail,
  deregisterToolSanitizeResponseGuardrail,
  registerToolConditionalExecutionGuardrail,
  deregisterToolConditionalExecutionGuardrail,
  registerToolRequestIntercept,
  deregisterToolRequestIntercept,
  registerToolExecutionIntercept,
  deregisterToolExecutionIntercept,
  clearLastCallbackError,
  getLastCallbackError,
  registerSubscriber,
  deregisterSubscriber,
  flushSubscribers,
  ScopeType,
} = lib;

const TOOL_ATTR_LOCAL = 0b01;

function rejectWithPrimitive(value) {
  return Promise.reject(value);
}

async function assertCompletesWithin(promise, message) {
  let timeout;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timeout = setTimeout(() => reject(new assert.AssertionError({ message })), 2000);
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

function sparseArray() {
  const values = new Array(2);
  values[1] = 1;
  return values;
}

function toolResult(result, annotation) {
  return annotation == null ? { result } : { result, annotation };
}

// ===========================================================================
// Tool lifecycle
// ===========================================================================

describe('Tool lifecycle', () => {
  it('tool call and end', () => {
    const handle = toolCall(
      'test_tool',
      {
        x: 1,
      },
      null,
      TOOL_ATTR_LOCAL,
      null,
      null,
      'tool-call-1',
    );
    assert.equal(handle.name, 'test_tool');
    assert.equal(handle.attributes, TOOL_ATTR_LOCAL);
    assert.ok(handle.uuid.length > 0);
    toolCallEnd(
      handle,
      {
        result: 42,
      },
      null,
      null,
    );
  });

  it('tool call with attributes', () => {
    const handle = toolCall('attr_tool', {}, null, TOOL_ATTR_LOCAL, null, null);
    assert.equal(handle.attributes, TOOL_ATTR_LOCAL);
    toolCallEnd(handle, toolResult({}), null, null);
  });

  it('tool call with data/metadata', () => {
    const handle = toolCall(
      'data_tool',
      {},
      null,
      null,
      {
        info: 'test',
      },
      {
        version: '1.0',
      },
    );
    toolCallEnd(
      handle,
      toolResult({}),
      {
        done: true,
      },
      null,
    );
  });

  it('tool call with parent', () => {
    const scope = pushScope('tool_parent', ScopeType.Agent, null, null);
    const handle = toolCall('parented_tool', {}, scope, null, null, null);
    assert.equal(handle.parentUuid, scope.uuid);
    toolCallEnd(handle, toolResult({}), null, null);
    popScope(scope);
  });

  it('tool call generates events', async () => {
    const events = [];
    registerSubscriber('node_tool_evt_sub', (e) => events.push(e));
    try {
      const handle = toolCall('evt_tool', {}, null, null, null, null);
      toolCallEnd(handle, toolResult({}), null, null);
      await flushSubscribers();
      assert.ok(events.length >= 2, 'Expected at least 2 events');
    } finally {
      deregisterSubscriber('node_tool_evt_sub');
    }
  });

  it('tool call event exposes toolCallId and payload fields', async () => {
    const events = [];
    const scope = pushScope('tool_event_parent', ScopeType.Agent, null, null);
    registerSubscriber('node_tool_field_sub', (e) => events.push(e));
    try {
      const handle = toolCall(
        'field_tool',
        {
          x: 1,
        },
        scope,
        TOOL_ATTR_LOCAL,
        {
          start: true,
        },
        {
          meta: true,
        },
        'tool-call-123',
      );
      assert.equal(handle.parentUuid, scope.uuid);
      assert.equal(handle.attributes, TOOL_ATTR_LOCAL);
      toolCallEnd(
        handle,
        toolResult(42, { provider: 'manual' }),
        {
          end: true,
        },
        {
          final: true,
        },
      );

      await flushSubscribers();

      const start = events.find(
        (e) => e.name === 'field_tool' && e.kind === 'scope' && e.category === 'tool' && e.scope_category === 'start',
      );
      const end = events.find(
        (e) => e.name === 'field_tool' && e.kind === 'scope' && e.category === 'tool' && e.scope_category === 'end',
      );
      assert.equal(start.category_profile.tool_call_id, 'tool-call-123');
      assert.deepEqual(start.data, {
        x: 1,
      });
      assert.equal(end.data, 42);
      assert.equal(end.category_profile.tool_call_id, 'tool-call-123');
      assert.deepEqual(end.category_profile.tool_result_annotation, { provider: 'manual' });
    } finally {
      deregisterSubscriber('node_tool_field_sub');
      popScope(scope);
    }
  });
});

// ===========================================================================
// Tool execute
// ===========================================================================

describe('Tool execute', () => {
  it('basic execute', async () => {
    const result = await toolCallExecute(
      'exec_tool',
      {
        x: 10,
      },
      (args) => ({
        result: args.x + 1,
      }),
      null,
      null,
      null,
      null,
    );
    assert.deepEqual(result, {
      result: 11,
    });
  });

  it('rejects legacy raw tool results from sync and async producers', async () => {
    const cases = [
      () => toolCallExecute('exec_tool_undefined', { x: 10 }, () => undefined),
      () => toolCallExecute('exec_tool_legacy_raw', {}, () => ({ legacy: true })),
      () => toolCallExecuteAsync('exec_async_tool_legacy_raw', {}, async () => ({ legacy: true })),
    ];
    for (const execute of cases) {
      await assert.rejects(execute, (error) => {
        assert.match(error.message, /must return ToolExecutionResult/i);
        assert.match(error.message, /return \{ result: payload \}, not a raw object/i);
        assert.doesNotMatch(error.message, /internal error/i);
        return true;
      });
    }
  });

  it('sync execute rejects thrown callbacks without terminating Node', async () => {
    await assert.rejects(
      () =>
        toolCallExecute('exec_tool_throw', {}, () => {
          throw new Error('sync tool callback failed');
        }),
      /sync tool callback failed/,
    );

    const result = await toolCallExecute('exec_tool_after_throw', {}, () => toolResult({ ok: true }));
    assert.deepEqual(result, toolResult({ ok: true }));
  });

  it('sync and async execute reject invalid JSON results without terminating Node', async () => {
    const cases = [
      ['sync_bigint', () => toolCallExecute('exec_tool_bigint', {}, () => toolResult(1n))],
      ['async_bigint', () => toolCallExecuteAsync('exec_async_tool_bigint', {}, async () => toolResult({ value: 1n }))],
      ['sync_date', () => toolCallExecute('exec_tool_date', {}, () => toolResult(new Date()))],
      [
        'async_map',
        () => toolCallExecuteAsync('exec_async_tool_map', {}, async () => toolResult(new Map([['value', 1]]))),
      ],
      ['sync_sparse_array', () => toolCallExecute('exec_tool_sparse_array', {}, () => toolResult(sparseArray()))],
      [
        'async_sparse_array',
        () => toolCallExecuteAsync('exec_async_tool_sparse_array', {}, async () => toolResult(sparseArray())),
      ],
    ];

    for (const [kind, execute] of cases) {
      await assert.rejects(execute, /bigint|undefined|json/i);
      const result = await toolCallExecute(`exec_tool_after_${kind}`, {}, () => toolResult({ ok: true }));
      assert.deepEqual(result, toolResult({ ok: true }));
    }
  });

  it('sync and async execute reject circular results without terminating Node', async () => {
    const circularResult = () => {
      const result = {};
      result.self = result;
      return result;
    };

    const cases = [
      ['sync', () => toolCallExecute('exec_tool_circular', {}, () => toolResult(circularResult()))],
      ['async', () => toolCallExecuteAsync('exec_async_tool_circular', {}, async () => toolResult(circularResult()))],
    ];

    for (const [kind, execute] of cases) {
      await assert.rejects(execute, /circular|json/i);

      const result = await toolCallExecute(`exec_tool_after_circular_${kind}`, {}, () => toolResult({ ok: true }));
      assert.deepEqual(result, toolResult({ ok: true }));
    }
  });

  it('sync and async execute read stateful getter results once', async () => {
    const statefulResult = () => {
      let reads = 0;
      return {
        get value() {
          reads += 1;
          return reads;
        },
      };
    };
    const cases = [
      () => toolCallExecute('exec_tool_stateful_getter', {}, () => toolResult(statefulResult())),
      () => toolCallExecuteAsync('exec_async_tool_stateful_getter', {}, async () => toolResult(statefulResult())),
    ];

    for (const execute of cases) {
      const result = await execute();
      assert.deepEqual(result, toolResult({ value: 1 }));
    }
  });

  it('sync and async execute serialize arrays by index instead of their iterator', async () => {
    const statefulArray = () => {
      const result = [1];
      result[Symbol.iterator] = function* iterator() {
        yield 99;
      };
      return result;
    };
    const cases = [
      () => toolCallExecute('exec_tool_iterator', {}, () => toolResult(statefulArray())),
      () => toolCallExecuteAsync('exec_async_tool_iterator', {}, async () => toolResult(statefulArray())),
    ];

    for (const execute of cases) {
      assert.deepEqual(await execute(), toolResult([1]));
    }
  });

  it('execute with attributes', async () => {
    const result = await toolCallExecute(
      'exec_attr_tool',
      {},
      () => toolResult({ ok: true }),
      null,
      TOOL_ATTR_LOCAL,
      null,
      null,
    );
    assert.deepEqual(result, toolResult({ ok: true }));
  });

  it('execute records OTEL status metadata on end events', async () => {
    const events = [];
    registerSubscriber('node_tool_status_metadata_sub', (e) => events.push(e));
    try {
      const result = await toolCallExecute(
        'exec_status_ok_tool',
        {
          x: 1,
        },
        (args) => ({
          result: args.x + 1,
        }),
        null,
        null,
        null,
        {
          caller: 'node-tool',
        },
        'node-managed-success-123',
      );
      assert.deepEqual(result, {
        result: 2,
      });

      await assert.rejects(
        () =>
          toolCallExecute(
            'exec_status_error_tool',
            {},
            () => {
              throw new TypeError('tool status failure');
            },
            null,
            null,
            null,
            {
              caller: 'node-tool-error',
            },
            'node-managed-error-123',
          ),
        /tool status failure/,
      );

      await flushSubscribers();
      const okStart = events.find(
        (e) =>
          e.name === 'exec_status_ok_tool' &&
          e.kind === 'scope' &&
          e.category === 'tool' &&
          e.scope_category === 'start',
      );
      const okEnd = events.find(
        (e) =>
          e.name === 'exec_status_ok_tool' && e.kind === 'scope' && e.category === 'tool' && e.scope_category === 'end',
      );
      const errorStart = events.find(
        (e) =>
          e.name === 'exec_status_error_tool' &&
          e.kind === 'scope' &&
          e.category === 'tool' &&
          e.scope_category === 'start',
      );
      const errorEnd = events.find(
        (e) =>
          e.name === 'exec_status_error_tool' &&
          e.kind === 'scope' &&
          e.category === 'tool' &&
          e.scope_category === 'end',
      );
      assert.ok(okStart, 'expected successful tool start event');
      assert.ok(okEnd, 'expected successful tool end event');
      assert.equal(okStart.category_profile.tool_call_id, 'node-managed-success-123');
      assert.equal(okEnd.category_profile.tool_call_id, 'node-managed-success-123');
      assert.equal(okEnd.metadata.caller, 'node-tool');
      assert.equal(okEnd.metadata['otel.status_code'], 'OK');
      assert.ok(errorStart, 'expected failed tool start event');
      assert.ok(errorEnd, 'expected failed tool end event');
      assert.equal(errorStart.category_profile.tool_call_id, 'node-managed-error-123');
      assert.equal(errorEnd.category_profile.tool_call_id, 'node-managed-error-123');
      assert.equal(errorEnd.metadata.caller, 'node-tool-error');
      assert.equal(errorEnd.metadata['otel.status_code'], 'ERROR');
      assert.match(errorEnd.metadata['otel.status_description'], /tool status failure/);
      assert.equal(errorEnd.metadata['error.type'], 'internal_error');
      assert.equal(errorEnd.metadata['exception.type'], 'TypeError');
    } finally {
      deregisterSubscriber('node_tool_status_metadata_sub');
    }
  });

  it('async execute awaits Promise-returning callbacks and records tool call IDs', async () => {
    const events = [];
    registerSubscriber('node_async_tool_call_id_sub', (event) => events.push(event));
    try {
      const result = await toolCallExecuteAsync(
        'exec_async_tool',
        {
          x: 10,
        },
        async (args) => ({
          result: args.x + 2,
        }),
        null,
        TOOL_ATTR_LOCAL,
        {
          data: true,
        },
        {
          meta: true,
        },
        'node-managed-async-123',
      );
      assert.deepEqual(result, {
        result: 12,
      });

      await flushSubscribers();
      const lifecycle = events.filter(
        (event) => event.kind === 'scope' && event.category === 'tool' && event.name === 'exec_async_tool',
      );
      assert.deepEqual(
        lifecycle.map((event) => event.scope_category),
        ['start', 'end'],
      );
      assert.ok(lifecycle.every((event) => event.category_profile.tool_call_id === 'node-managed-async-123'));
    } finally {
      deregisterSubscriber('node_async_tool_call_id_sub');
    }
  });

  it('generated execute declarations expose an additive trailing toolCallId', () => {
    const declarations = readFileSync(new URL('../index.d.ts', import.meta.url), 'utf8');

    assert.match(declarations, /toolCallExecute\([\s\S]*?metadata\?: Json[^,\n]*,[\s\S]*?toolCallId\?: string/);
    assert.match(declarations, /toolCallExecuteAsync\([\s\S]*?metadata\?: Json[^,\n]*,[\s\S]*?toolCallId\?: string/);
  });

  it('async execute surfaces plain string rejections', async () => {
    await assert.rejects(
      () =>
        toolCallExecuteAsync(
          'exec_async_tool_reject',
          {
            x: 10,
          },
          async () => rejectWithPrimitive(new Error('string tool error')),
          null,
          null,
          null,
          null,
        ),
      /string tool error/,
    );
  });
});

// ===========================================================================
// Tool guardrails
// ===========================================================================

describe('Tool guardrails', () => {
  it('sanitize request guardrail', () => {
    registerToolSanitizeRequestGuardrail('node_tool_san_req', 10, (name, args) => {
      args.sanitized = true;
      return args;
    });
    deregisterToolSanitizeRequestGuardrail('node_tool_san_req');
  });

  it('sanitize request guardrail rewrites start event payload', async () => {
    const events = [];
    registerSubscriber('node_tool_san_req_evt', (e) => events.push(e));
    registerToolSanitizeRequestGuardrail('node_tool_san_req_evt_guard', 10, (name, args) => ({
      ...args,
      sanitized: true,
    }));
    try {
      const result = await toolCallExecute(
        'san_req_evt_tool',
        {
          x: 1,
        },
        (args) => toolResult(args),
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, toolResult({ x: 1 }));
      await flushSubscribers();
      const start = events.find(
        (e) =>
          e.name === 'san_req_evt_tool' && e.kind === 'scope' && e.category === 'tool' && e.scope_category === 'start',
      );
      assert.deepEqual(start.data, {
        x: 1,
        sanitized: true,
      });
    } finally {
      deregisterToolSanitizeRequestGuardrail('node_tool_san_req_evt_guard');
      deregisterSubscriber('node_tool_san_req_evt');
    }
  });

  it('sanitize response guardrail', () => {
    registerToolSanitizeResponseGuardrail('node_tool_san_resp', 10, (name, result) => {
      result.checked = true;
      return result;
    });
    deregisterToolSanitizeResponseGuardrail('node_tool_san_resp');
  });

  it('sanitize response guardrail rewrites end event payload', async () => {
    const events = [];
    registerSubscriber('node_tool_san_resp_evt', (e) => events.push(e));
    registerToolSanitizeResponseGuardrail('node_tool_san_resp_evt_guard', 10, (name, result) => ({
      ...result,
      checked: true,
    }));
    try {
      const result = await toolCallExecute(
        'san_resp_evt_tool',
        {
          x: 1,
        },
        () => toolResult({ ok: true }),
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, toolResult({ ok: true }));
      await flushSubscribers();
      const end = events.find(
        (e) =>
          e.name === 'san_resp_evt_tool' && e.kind === 'scope' && e.category === 'tool' && e.scope_category === 'end',
      );
      assert.deepEqual(end.data, {
        ok: true,
        checked: true,
      });
    } finally {
      deregisterToolSanitizeResponseGuardrail('node_tool_san_resp_evt_guard');
      deregisterSubscriber('node_tool_san_resp_evt');
    }
  });

  it('conditional guardrail (allow)', () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond', 10, (name, args) => null);
    deregisterToolConditionalExecutionGuardrail('node_tool_cond');
  });

  it('conditional guardrail awaits a Promise result', async () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond_promise', 10, async () => {
      await new Promise((resolve) => setImmediate(resolve));
      return null;
    });
    try {
      const result = await toolCallExecute(
        'tool_cond_promise',
        { ok: true },
        (args) => toolResult(args),
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, toolResult({ ok: true }));
    } finally {
      deregisterToolConditionalExecutionGuardrail('node_tool_cond_promise');
    }
  });

  it('Promise middleware preserves its invocation scope before and after await', async () => {
    const originalStack = lib.currentScopeStack();
    const invocationStack = lib.createScopeStack();
    const unrelatedStack = lib.createScopeStack();
    const observed = [];
    let invocationScope;
    lib.registerToolConditionalExecutionGuardrail('node_tool_cond_scope_context', 10, async () => {
      observed.push(lib.getHandle().uuid);
      await new Promise((resolve) => setImmediate(resolve));
      observed.push(lib.getHandle().uuid);
      return null;
    });
    try {
      const execution = lib.withScopeStack(invocationStack, () => {
        invocationScope = lib.pushScope('middleware-invocation', lib.ScopeType.Agent);
        return lib.toolCallExecute('tool_cond_scope_context', {}, (args) => toolResult(args));
      });
      lib.setThreadScopeStack(unrelatedStack);
      await execution;
      assert.deepEqual(observed, [invocationScope.uuid, invocationScope.uuid]);
    } finally {
      lib.withScopeStack(invocationStack, () => lib.popScope(invocationScope));
      lib.setThreadScopeStack(originalStack);
      lib.deregisterToolConditionalExecutionGuardrail('node_tool_cond_scope_context');
    }
  });

  it('conditional guardrail propagates a rejected Promise', async () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond_reject', 10, async () => {
      throw new Error('guardrail rejected promise');
    });
    try {
      await assert.rejects(
        () => toolCallExecute('tool_cond_reject', {}, () => ({ should_not: 'run' }), null, null, null, null),
        /guardrail rejected promise/i,
      );
    } finally {
      deregisterToolConditionalExecutionGuardrail('node_tool_cond_reject');
    }
  });

  it('conditional guardrail treats implicit undefined as allow', async () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond_undefined', 10, () => undefined);
    try {
      const result = await toolCallExecute(
        'tool_cond_undefined',
        {
          ok: true,
        },
        (args) => toolResult(args),
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, toolResult({ ok: true }));
    } finally {
      deregisterToolConditionalExecutionGuardrail('node_tool_cond_undefined');
    }
  });

  it('conditional guardrail throws a catchable error without terminating Node', async () => {
    let executed = false;
    registerToolConditionalExecutionGuardrail('node_tool_cond_throw', 10, () => {
      throw new Error('guardrail boom');
    });
    try {
      await assert.rejects(
        () =>
          toolCallExecute(
            'tool_cond_throw',
            {},
            () => {
              executed = true;
              return {};
            },
            null,
            null,
            null,
            null,
          ),
        /guardrail boom/i,
      );
      assert.equal(executed, false);
      deregisterToolConditionalExecutionGuardrail('node_tool_cond_throw');

      const result = await toolCallExecute(
        'tool_after_guardrail_throw',
        {},
        () => toolResult({ ok: true }),
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, toolResult({ ok: true }));
    } finally {
      deregisterToolConditionalExecutionGuardrail('node_tool_cond_throw');
    }
  });

  it('sanitize guardrail failures omit the observable payload', async () => {
    clearLastCallbackError();
    const events = [];
    registerSubscriber('node_tool_san_throw_sub', (event) => events.push(event));
    registerToolSanitizeRequestGuardrail('node_tool_san_throw', 10, () => {
      throw new Error('sanitize boom');
    });
    try {
      await toolCallExecute(
        'tool_sanitize_throw',
        { original: true },
        (args) => toolResult(args),
        null,
        null,
        null,
        null,
      );
      await flushSubscribers();
      const start = events.find(
        (event) =>
          event.name === 'tool_sanitize_throw' &&
          event.kind === 'scope' &&
          event.category === 'tool' &&
          event.scope_category === 'start',
      );
      assert.equal(start.data, null);
      assert.match(getLastCallbackError() ?? '', /sanitize boom/i);
    } finally {
      deregisterToolSanitizeRequestGuardrail('node_tool_san_throw');
      deregisterSubscriber('node_tool_san_throw_sub');
      clearLastCallbackError();
    }
  });

  it('manual async sanitizers can flush subscribers without deadlocking', async () => {
    const events = [];
    let requestFlushed = false;
    let responseFlushed = false;
    registerSubscriber('node_manual_tool_flush_subscriber', (event) => events.push(event));
    registerToolSanitizeRequestGuardrail('node_manual_tool_flush_request', 10, async (_name, args) => {
      await flushSubscribers();
      requestFlushed = true;
      return { ...args, requestSanitized: true };
    });
    registerToolSanitizeResponseGuardrail('node_manual_tool_flush_response', 10, async (_name, response) => {
      await flushSubscribers();
      responseFlushed = true;
      return { ...response, responseSanitized: true };
    });
    try {
      const handle = toolCall('node_manual_tool_flush', { original: true });
      toolCallEnd(handle, toolResult({ ok: true }));
      await flushSubscribers();
    } finally {
      deregisterToolSanitizeRequestGuardrail('node_manual_tool_flush_request');
      deregisterToolSanitizeResponseGuardrail('node_manual_tool_flush_response');
      deregisterSubscriber('node_manual_tool_flush_subscriber');
    }
    assert.equal(requestFlushed, true);
    assert.equal(responseFlushed, true);
    const start = events.find((event) => event.name === 'node_manual_tool_flush' && event.scope_category === 'start');
    const end = events.find((event) => event.name === 'node_manual_tool_flush' && event.scope_category === 'end');
    assert.deepEqual(start.data, { original: true, requestSanitized: true });
    assert.deepEqual(end.data, { ok: true, responseSanitized: true });
  });

  it('conditional guardrail (block)', () => {
    registerToolConditionalExecutionGuardrail('node_tool_block', 10, (name, args) => 'blocked');
    deregisterToolConditionalExecutionGuardrail('node_tool_block');
  });

  it('conditional guardrail rejects non-string return values', async () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond_non_string', 10, () => ({
      blocked: true,
    }));
    try {
      await assert.rejects(
        () =>
          toolCallExecute(
            'tool_cond_non_string',
            {
              ok: true,
            },
            (args) => args,
            null,
            null,
            null,
            null,
          ),
        /expected string or null/i,
      );
    } finally {
      deregisterToolConditionalExecutionGuardrail('node_tool_cond_non_string');
    }
  });

  it('duplicate guardrail fails', () => {
    registerToolSanitizeRequestGuardrail('node_dup_guard', 10, (n, a) => a);
    assert.throws(() => registerToolSanitizeRequestGuardrail('node_dup_guard', 20, (n, a) => a));
    deregisterToolSanitizeRequestGuardrail('node_dup_guard');
  });
});

// ===========================================================================
// Tool intercepts
// ===========================================================================

describe('Tool intercepts', () => {
  it('request intercept register/deregister', () => {
    registerToolRequestIntercept('node_tool_req_int', 10, false, (name, args) => {
      args.intercepted = true;
      return args;
    });
    deregisterToolRequestIntercept('node_tool_req_int');
  });

  it('execution intercept register/deregister', () => {
    registerToolExecutionIntercept('node_tool_exec_int', 10, async (args, next) => {
      const downstream = await next(args);
      return {
        result: downstream.result,
        ...(downstream.annotation == null ? {} : { annotation: downstream.annotation }),
      };
    });
    deregisterToolExecutionIntercept('node_tool_exec_int');
  });

  it('request intercept with break_chain', () => {
    registerToolRequestIntercept('node_tool_break', 10, true, (name, args) => args);
    deregisterToolRequestIntercept('node_tool_break');
  });

  it('duplicate intercept fails', () => {
    registerToolRequestIntercept('node_dup_int', 10, false, (n, a) => a);
    assert.throws(() => registerToolRequestIntercept('node_dup_int', 20, false, (n, a) => a));
    deregisterToolRequestIntercept('node_dup_int');
  });

  it('request intercept modifies args', async () => {
    registerToolRequestIntercept('node_tool_req_mod', 10, false, (name, args) => {
      args.added = 'yes';
      return args;
    });
    const result = await toolCallExecute(
      'mod_tool',
      {
        original: true,
      },
      (args) => toolResult(args),
      null,
      null,
      null,
      null,
    );
    assert.equal(result.result.added, 'yes');
    deregisterToolRequestIntercept('node_tool_req_mod');
  });

  it('request intercept awaits a Promise result', async () => {
    registerToolRequestIntercept('node_tool_req_promise', 10, false, async (_name, args) => {
      await new Promise((resolve) => setImmediate(resolve));
      return { ...args, promised: true };
    });
    try {
      const result = await toolCallExecute(
        'tool_req_promise',
        { original: true },
        (args) => toolResult(args),
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, toolResult({ original: true, promised: true }));
    } finally {
      deregisterToolRequestIntercept('node_tool_req_promise');
    }
  });

  it('request intercept propagates a rejected Promise', async () => {
    registerToolRequestIntercept('node_tool_req_reject', 10, false, async () => {
      throw new Error('request intercept rejected promise');
    });
    try {
      await assert.rejects(
        () => toolCallExecute('tool_req_reject', {}, () => ({ should_not: 'run' }), null, null, null, null),
        /request intercept rejected promise/i,
      );
    } finally {
      deregisterToolRequestIntercept('node_tool_req_reject');
    }
  });

  it('request intercept throws a catchable error without terminating Node', async () => {
    registerToolRequestIntercept('node_tool_req_throw', 10, false, () => {
      throw new Error('tool request intercept boom');
    });
    try {
      await assert.rejects(
        () => toolCallExecute('tool_req_throw', {}, () => ({ should_not: 'run' }), null, null, null, null),
        /tool request intercept boom/i,
      );
      deregisterToolRequestIntercept('node_tool_req_throw');
      const result = await toolCallExecute(
        'tool_after_request_intercept_throw',
        {},
        () => toolResult({ ok: true }),
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, toolResult({ ok: true }));
    } finally {
      deregisterToolRequestIntercept('node_tool_req_throw');
    }
  });

  it('request intercept can return null JSON', async () => {
    registerToolRequestIntercept('node_tool_req_bad', 10, false, () => null);
    try {
      const result = await toolCallExecute(
        'bad_tool',
        {
          original: true,
        },
        (args) => toolResult(args),
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, toolResult(null));
    } finally {
      deregisterToolRequestIntercept('node_tool_req_bad');
    }
  });

  it('execution intercept composes with next', async () => {
    const events = [];
    registerSubscriber('node_tool_exec_mark_sub', (event) => events.push(event));
    registerToolExecutionIntercept('node_tool_exec_repl', 10, async (args, next) => {
      const downstream = await next({
        ...args,
        intercepted: true,
      });
      return {
        result: {
          ...downstream.result,
          wrapped: true,
        },
        annotation: downstream.annotation,
        pendingMarks: [
          {
            name: 'node.tool.execution',
            dataSchema: { name: 'example.tool.mark', version: '1' },
            severity: 'warning',
          },
        ],
      };
    });
    try {
      const result = await toolCallExecute(
        'replaced_tool',
        {},
        (args) =>
          toolResult(
            {
              original: !args.intercepted,
            },
            { source: 'provider' },
          ),
        null,
        null,
        null,
        null,
      );
      assert.equal(result.result.original, false);
      assert.equal(result.result.wrapped, true);
      assert.deepEqual(result.annotation, { source: 'provider' });
      await flushSubscribers();
      const start = events.find(
        (event) => event.name === 'replaced_tool' && event.kind === 'scope' && event.scope_category === 'start',
      );
      const end = events.find(
        (event) => event.name === 'replaced_tool' && event.kind === 'scope' && event.scope_category === 'end',
      );
      const mark = events.find((event) => event.name === 'node.tool.execution');
      assert.ok(start, 'expected tool start event');
      assert.ok(end, 'expected tool end event');
      assert.ok(mark, 'expected tool execution pending mark');
      assert.deepEqual(end.category_profile.tool_result_annotation, { source: 'provider' });
      assert.deepEqual(mark.data_schema, { name: 'example.tool.mark', version: '1' });
      assert.equal(mark.metadata['nemo_relay.log.severity'], 'warn');
      assert.equal(mark.parent_uuid, start.uuid);
      assert.ok(events.indexOf(end) < events.indexOf(mark), 'expected tool end before pending mark');
      assert.ok(mark.timestamp > end.timestamp, 'expected pending mark timestamp after tool end');
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_repl');
      deregisterSubscriber('node_tool_exec_mark_sub');
    }
  });

  it('execution intercept can remove a downstream annotation', async () => {
    const events = [];
    registerSubscriber('node_tool_exec_annotation_removal_sub', (event) => events.push(event));
    registerToolExecutionIntercept('node_tool_exec_annotation_removal', 10, async (args, next) => {
      const downstream = await next(args);
      return { result: downstream.result };
    });
    try {
      const result = await toolCallExecute('annotation_removal_tool', {}, () =>
        toolResult({ ok: true }, { source: 'provider' }),
      );
      assert.deepEqual(result, toolResult({ ok: true }));
      assert.equal(Object.hasOwn(result, 'annotation'), false);

      await flushSubscribers();
      const end = events.find(
        (event) => event.name === 'annotation_removal_tool' && event.kind === 'scope' && event.scope_category === 'end',
      );
      assert.ok(end, 'expected tool end event');
      assert.equal(Object.hasOwn(end.category_profile, 'tool_result_annotation'), false);
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_annotation_removal');
      deregisterSubscriber('node_tool_exec_annotation_removal_sub');
    }
  });

  it('normalizes a JSON-null annotation to absence', async () => {
    const events = [];
    registerSubscriber('node_tool_exec_null_annotation_sub', (event) => events.push(event));
    try {
      const result = await toolCallExecute('null_annotation_tool', {}, () => ({
        result: { ok: true },
        annotation: null,
      }));
      assert.deepEqual(result, toolResult({ ok: true }));
      assert.equal(Object.hasOwn(result, 'annotation'), false);

      await flushSubscribers();
      const end = events.find(
        (event) => event.name === 'null_annotation_tool' && event.kind === 'scope' && event.scope_category === 'end',
      );
      assert.ok(end, 'expected tool end event');
      assert.equal(Object.hasOwn(end.category_profile, 'tool_result_annotation'), false);
    } finally {
      deregisterSubscriber('node_tool_exec_null_annotation_sub');
    }
  });

  it('execution callbacks preserve the managed propagation parent across await', async () => {
    const events = [];
    const observed = [];
    registerSubscriber('node_tool_exec_propagation_parent', (event) => events.push(event));
    registerToolExecutionIntercept('node_tool_exec_propagation_parent', 10, async (args, next) => {
      observed.push(['intercept-before', lib.capturePropagationContext().parentUuid]);
      await new Promise((resolve) => setImmediate(resolve));
      observed.push(['intercept-after', lib.capturePropagationContext().parentUuid]);
      const downstream = await next(args);
      return {
        result: downstream.result,
        ...(downstream.annotation == null ? {} : { annotation: downstream.annotation }),
      };
    });
    try {
      const result = await toolCallExecuteAsync('propagation_parent_tool', {}, async () => {
        observed.push(['provider-before', lib.capturePropagationContext().parentUuid]);
        await new Promise((resolve) => setImmediate(resolve));
        observed.push(['provider-after', lib.capturePropagationContext().parentUuid]);
        return toolResult({ ok: true });
      });
      assert.deepEqual(result, toolResult({ ok: true }));
      await flushSubscribers();
      const start = events.find(
        (event) =>
          event.name === 'propagation_parent_tool' && event.kind === 'scope' && event.scope_category === 'start',
      );
      assert.ok(start, 'expected managed tool start event');
      assert.deepEqual(observed, [
        ['intercept-before', start.uuid],
        ['intercept-after', start.uuid],
        ['provider-before', start.uuid],
        ['provider-after', start.uuid],
      ]);
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_propagation_parent');
      deregisterSubscriber('node_tool_exec_propagation_parent');
    }
  });

  it('execution settlement expires scope replacements inherited by detached work', async () => {
    const baseline = {
      active: lib.scopeStackActive(),
      parentUuid: lib.capturePropagationContext().parentUuid,
    };
    const replacementStack = lib.createScopeStack();
    let releaseLateContext;
    const lateGate = new Promise((resolve) => {
      releaseLateContext = resolve;
    });
    let lateContext = Promise.resolve(baseline);

    await toolCallExecuteAsync('scope_replacement_expiry_tool', {}, async () => {
      lib.setThreadScopeStack(replacementStack);
      lateContext = lateGate.then(() => ({
        active: lib.scopeStackActive(),
        parentUuid: lib.capturePropagationContext().parentUuid,
      }));
      await new Promise((resolve) => setImmediate(resolve));
      return toolResult({ ok: true });
    });

    releaseLateContext();
    assert.deepEqual(await lateContext, baseline);
  });

  it('execution intercept rejects a detached next call after settlement', async () => {
    let releaseLateNext;
    const lateGate = new Promise((resolve) => {
      releaseLateNext = resolve;
    });
    let lateNext;
    let providerCalls = 0;
    registerToolExecutionIntercept('node_tool_exec_late_next', 10, async (args, next) => {
      lateNext = lateGate.then(() => next(args));
      return { result: { source: 'intercept' } };
    });
    try {
      const result = await toolCallExecute('late_next_tool', { value: 1 }, (args) => {
        providerCalls += 1;
        return toolResult(args);
      });
      assert.deepEqual(result, toolResult({ source: 'intercept' }));
      releaseLateNext();
      await assert.rejects(lateNext, /execution continuation is no longer active/i);
      assert.equal(providerCalls, 0);
    } finally {
      releaseLateNext?.();
      await lateNext?.catch(() => {});
      deregisterToolExecutionIntercept('node_tool_exec_late_next');
    }
  });

  it('execution intercept aborts an already-started async provider after settlement', async () => {
    let releaseProvider;
    const providerGate = new Promise((resolve) => {
      releaseProvider = resolve;
    });
    let providerStarted;
    const started = new Promise((resolve) => {
      providerStarted = resolve;
    });
    let providerAborted;
    const aborted = new Promise((resolve) => {
      providerAborted = resolve;
    });
    let downstream;
    let providerSideEffects = 0;
    registerToolExecutionIntercept('node_tool_exec_abort_started_provider', 10, async (args, next) => {
      downstream = next(args);
      downstream.catch(() => undefined);
      await started;
      return { result: { source: 'intercept' } };
    });
    try {
      const result = await toolCallExecuteAsync('abort_started_tool', { value: 1 }, async (_args, signal) => {
        providerStarted();
        await Promise.race([
          providerGate,
          new Promise((_, reject) => {
            signal.addEventListener(
              'abort',
              () => {
                providerAborted();
                reject(new Error('provider aborted'));
              },
              { once: true },
            );
          }),
        ]);
        providerSideEffects += 1;
        return toolResult({ source: 'provider' });
      });
      assert.deepEqual(result, toolResult({ source: 'intercept' }));
      await assert.rejects(downstream, /execution continuation is no longer active/i);
      await assertCompletesWithin(aborted, 'provider did not receive cancellation after continuation revocation');
      releaseProvider();
      assert.equal(providerSideEffects, 0);
    } finally {
      releaseProvider?.();
      await downstream?.catch(() => {});
      deregisterToolExecutionIntercept('node_tool_exec_abort_started_provider');
    }
  });

  it('execution intercept isolates concurrent next scope branches', async () => {
    let releaseBoth;
    const bothPushed = new Promise((resolve) => {
      releaseBoth = resolve;
    });
    let pushed = 0;
    registerToolExecutionIntercept('node_tool_exec_concurrent_next_scopes', 10, async (_args, next) => {
      const [first, second] = await Promise.all([next({ branch: 'first' }), next({ branch: 'second' })]);
      return { result: [first.result, second.result] };
    });
    try {
      const result = await toolCallExecuteAsync('concurrent_next_tool', {}, async (args) => {
        const handle = pushScope(`node-next-${args.branch}`, ScopeType.Custom);
        try {
          pushed += 1;
          if (pushed === 2) {
            releaseBoth();
          }
          await bothPushed;
          if (args.branch === 'first') {
            await new Promise((resolve) => setImmediate(resolve));
          }
          assert.equal(lib.getHandle().uuid, handle.uuid);
          return toolResult(args);
        } finally {
          popScope(handle);
        }
      });
      assert.deepEqual(result.result, [{ branch: 'first' }, { branch: 'second' }]);
    } finally {
      releaseBoth?.();
      deregisterToolExecutionIntercept('node_tool_exec_concurrent_next_scopes');
    }
  });

  it('execution intercept isolates concurrent branch scope-stack replacements', async () => {
    const firstStack = lib.createScopeStack();
    const secondStack = lib.createScopeStack();
    const firstScope = lib.withScopeStack(firstStack, () => lib.getHandle().uuid);
    const secondScope = lib.withScopeStack(secondStack, () => lib.getHandle().uuid);
    let firstStackInstalled;
    const firstInstalled = new Promise((resolve) => {
      firstStackInstalled = resolve;
    });
    let secondStackInstalled;
    const secondInstalled = new Promise((resolve) => {
      secondStackInstalled = resolve;
    });
    let parentScope;

    registerToolExecutionIntercept('node_tool_exec_concurrent_scope_replacements', 10, async (args, next) => {
      parentScope = lib.getHandle().uuid;
      const first = lib.withScopeStack(firstStack, async () => {
        firstStackInstalled();
        await secondInstalled;
        return next({ ...args, branch: 'first' });
      });
      const second = lib.withScopeStack(secondStack, async () => {
        await firstInstalled;
        secondStackInstalled();
        return next({ ...args, branch: 'second' });
      });
      const branches = await Promise.all([first, second]);
      assert.equal(lib.getHandle().uuid, parentScope);
      const parent = await next({ ...args, branch: 'parent' });
      return { result: [...branches.map((branch) => branch.result), parent.result] };
    });
    try {
      const result = await toolCallExecuteAsync('concurrent_scope_replacements_tool', {}, async (args) =>
        toolResult({
          branch: args.branch,
          scope: lib.getHandle().uuid,
        }),
      );
      assert.deepEqual(result.result, [
        { branch: 'first', scope: firstScope },
        { branch: 'second', scope: secondScope },
        { branch: 'parent', scope: parentScope },
      ]);
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_concurrent_scope_replacements');
    }
  });

  it('execution intercept rejects non-JSON next arguments without aborting Node', async () => {
    registerToolExecutionIntercept('node_tool_exec_bigint_next', 10, async (_args, next) => ({
      result: await next(1n),
    }));
    try {
      await assert.rejects(
        () => toolCallExecute('bigint_next_tool', {}, (args) => args),
        /unsupported bigint value.*JSON/i,
      );
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_bigint_next');
    }
  });

  it('execution intercept next preserves the invocation scope across the chain', async () => {
    const originalStack = lib.currentScopeStack();
    const invocationStack = lib.createScopeStack();
    const unrelatedStack = lib.createScopeStack();
    const observed = [];
    let invocationScope;

    const intercept = (label) => async (args, next) => {
      observed.push([label, 'before', lib.getHandle().uuid]);
      await new Promise((resolve) => setImmediate(resolve));
      const result = await next(args);
      observed.push([label, 'after', lib.getHandle().uuid]);
      return {
        result: result.result,
        ...(result.annotation == null ? {} : { annotation: result.annotation }),
      };
    };

    registerToolExecutionIntercept('node_tool_exec_scope_outer', 10, intercept('outer'));
    registerToolExecutionIntercept('node_tool_exec_scope_inner', 20, intercept('inner'));
    try {
      const execution = lib.withScopeStack(invocationStack, () => {
        invocationScope = lib.pushScope('execution-intercept-invocation', lib.ScopeType.Agent);
        return toolCallExecute('execution_intercept_scope', {}, (args) => toolResult(args));
      });
      lib.setThreadScopeStack(unrelatedStack);
      await execution;
      assert.deepEqual(observed, [
        ['outer', 'before', invocationScope.uuid],
        ['inner', 'before', invocationScope.uuid],
        ['inner', 'after', invocationScope.uuid],
        ['outer', 'after', invocationScope.uuid],
      ]);
    } finally {
      lib.withScopeStack(invocationStack, () => lib.popScope(invocationScope));
      lib.setThreadScopeStack(originalStack);
      deregisterToolExecutionIntercept('node_tool_exec_scope_outer');
      deregisterToolExecutionIntercept('node_tool_exec_scope_inner');
    }
  });

  it('snapshotted execution intercept survives deregistration', async () => {
    let blockerEntered;
    const entered = new Promise((resolve) => {
      blockerEntered = resolve;
    });
    let releaseBlocker;
    const release = new Promise((resolve) => {
      releaseBlocker = resolve;
    });

    registerToolExecutionIntercept('node_tool_exec_snapshot_target', 100, async (args, next) => ({
      result: {
        ...(await next(args)).result,
        snapshotted: true,
      },
    }));
    registerToolExecutionIntercept('node_tool_exec_snapshot_blocker', -100, async (args, next) => {
      blockerEntered();
      await release;
      const downstream = await next(args);
      return {
        result: downstream.result,
        ...(downstream.annotation == null ? {} : { annotation: downstream.annotation }),
      };
    });

    try {
      const execution = toolCallExecute(
        'snapshotted_tool',
        {},
        () => toolResult({ downstream: true }),
        null,
        null,
        null,
        null,
      );
      await entered;
      assert.equal(deregisterToolExecutionIntercept('node_tool_exec_snapshot_target'), true);
      releaseBlocker();
      assert.deepEqual((await execution).result, {
        downstream: true,
        snapshotted: true,
      });
    } finally {
      releaseBlocker();
      deregisterToolExecutionIntercept('node_tool_exec_snapshot_blocker');
      deregisterToolExecutionIntercept('node_tool_exec_snapshot_target');
    }
  });

  it('execution intercept propagates Error messages', async () => {
    registerToolExecutionIntercept('node_tool_exec_throw', 10, async () => {
      throw new Error('tool middleware exploded');
    });
    try {
      await assert.rejects(
        () =>
          toolCallExecute(
            'throwing_tool',
            {
              value: 1,
            },
            (args) => args,
            null,
            null,
            null,
            null,
          ),
        /tool middleware exploded/,
      );
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_throw');
    }
  });

  it('execution intercept rejects legacy raw results with forwarding guidance', async () => {
    registerToolExecutionIntercept('node_tool_exec_legacy', 10, async () => ({ legacyResult: true }));
    try {
      await assert.rejects(
        () => toolCallExecute('legacy_tool', {}, (args) => args, null, null, null, null),
        (error) => {
          assert.match(error.message, /tool execution intercept must return/i);
          assert.match(error.message, /result: downstream\.result, annotation: downstream\.annotation/i);
          assert.doesNotMatch(error.message, /internal error/i);
          return true;
        },
      );
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_legacy');
    }
  });

  it('execution intercept may directly forward the canonical Node result', async () => {
    registerToolExecutionIntercept('node_tool_exec_forward_result', 10, async (args, next) => next(args));
    try {
      const result = await toolCallExecute('forward_result_tool', { ok: true }, (args) => toolResult(args));
      assert.deepEqual(result, toolResult({ ok: true }));
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_forward_result');
    }
  });

  it('execution intercept rejects unknown pending-mark fields', async () => {
    registerToolExecutionIntercept('node_tool_exec_bad_mark', 10, async () => ({
      result: { ok: true },
      pendingMarks: [{ name: 'node.bad.mark', category_profile: { subtype: 'invalid.snake.case' } }],
    }));
    try {
      await assert.rejects(
        () => toolCallExecute('bad_mark_tool', {}, (args) => args, null, null, null, null),
        /unknown field.*category_profile/i,
      );
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_bad_mark');
    }
  });

  it('async execute preserves primitive rejection values', async () => {
    await assert.rejects(
      () =>
        toolCallExecuteAsync(
          'primitive_reject_tool',
          {
            value: 1,
          },
          async () => rejectWithPrimitive(42),
          null,
          null,
          null,
          null,
        ),
      /internal error: 42/i,
    );
  });

  it('standalone request intercepts helper applies intercept chain', async () => {
    registerToolRequestIntercept('node_tool_req_helper', 10, false, (name, args) => ({
      ...args,
      helper: true,
    }));
    try {
      const result = await toolRequestIntercepts('helper_tool', {
        original: true,
      });
      assert.deepEqual(result, {
        original: true,
        helper: true,
      });
    } finally {
      deregisterToolRequestIntercept('node_tool_req_helper');
    }
  });

  it('standalone conditional execution helper throws on rejection', async () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond_helper', 10, () => 'blocked by helper');
    try {
      await assert.rejects(
        () =>
          toolConditionalExecution('helper_tool', {
            test: true,
          }),
        /guardrail rejected/i,
      );
    } finally {
      deregisterToolConditionalExecutionGuardrail('node_tool_cond_helper');
    }
  });

  it('standalone conditional execution helper resolves when allowed', async () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond_allow', 10, () => null);
    try {
      await assert.doesNotReject(() =>
        toolConditionalExecution('helper_tool', {
          test: true,
        }),
      );
    } finally {
      deregisterToolConditionalExecutionGuardrail('node_tool_cond_allow');
    }
  });
});
