// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
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

async function waitForSubscriberCallbacks(predicate, timeoutMs = 15000) {
  flushSubscribers();
  // flushSubscribers() waits for Relay's Rust subscriber dispatcher, but JS
  // subscriber callbacks are queued onto Node's event loop through N-API
  // ThreadsafeFunction. Yield event-loop turns until the observed JS-side
  // callback state is ready, with a timeout to avoid hanging the test forever.
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) {
      throw new Error('timed out waiting for subscriber callbacks');
    }
    await new Promise((resolve) => setImmediate(resolve));
  }
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
    toolCallEnd(handle, {}, null, null);
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
      {},
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
    toolCallEnd(handle, {}, null, null);
    popScope(scope);
  });

  it('tool call generates events', async () => {
    const events = [];
    registerSubscriber('node_tool_evt_sub', (e) => events.push(e));
    try {
      const handle = toolCall('evt_tool', {}, null, null, null, null);
      toolCallEnd(handle, {}, null, null);
      const deadline = Date.now() + 2000;
      while (events.length < 2 && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 10));
      }
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
        {
          result: 42,
        },
        {
          end: true,
        },
        {
          final: true,
        },
      );

      const deadline = Date.now() + 2000;
      while (events.filter((e) => e.name === 'field_tool').length < 2 && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 10));
      }

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
      assert.deepEqual(end.data, {
        result: 42,
      });
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

  it('treats implicit undefined tool results as null', async () => {
    const result = await toolCallExecute(
      'exec_tool_undefined',
      {
        x: 10,
      },
      () => undefined,
      null,
      null,
      null,
      null,
    );
    assert.equal(result, null);
  });

  it('sync execute rejects thrown callbacks without terminating Node', async () => {
    await assert.rejects(
      () =>
        toolCallExecute('exec_tool_throw', {}, () => {
          throw new Error('sync tool callback failed');
        }),
      /sync tool callback failed/,
    );

    const result = await toolCallExecute('exec_tool_after_throw', {}, () => ({
      ok: true,
    }));
    assert.deepEqual(result, {
      ok: true,
    });
  });

  it('sync and async execute reject invalid JSON results without terminating Node', async () => {
    const cases = [
      ['sync_bigint', () => toolCallExecute('exec_tool_bigint', {}, () => 1n)],
      ['async_bigint', () => toolCallExecuteAsync('exec_async_tool_bigint', {}, async () => ({ value: 1n }))],
      ['sync_date', () => toolCallExecute('exec_tool_date', {}, () => new Date())],
      ['async_map', () => toolCallExecuteAsync('exec_async_tool_map', {}, async () => new Map([['value', 1]]))],
      ['sync_sparse_array', () => toolCallExecute('exec_tool_sparse_array', {}, () => [, 1])],
      ['async_sparse_array', () => toolCallExecuteAsync('exec_async_tool_sparse_array', {}, async () => [, 1])],
    ];

    for (const [kind, execute] of cases) {
      await assert.rejects(execute, /bigint|undefined|json/i);
      const result = await toolCallExecute(`exec_tool_after_${kind}`, {}, () => ({
        ok: true,
      }));
      assert.deepEqual(result, {
        ok: true,
      });
    }
  });

  it('sync and async execute reject circular results without terminating Node', async () => {
    const circularResult = () => {
      const result = {};
      result.self = result;
      return result;
    };

    const cases = [
      ['sync', () => toolCallExecute('exec_tool_circular', {}, circularResult)],
      ['async', () => toolCallExecuteAsync('exec_async_tool_circular', {}, async () => circularResult())],
    ];

    for (const [kind, execute] of cases) {
      await assert.rejects(execute, /circular|json/i);

      const result = await toolCallExecute(`exec_tool_after_circular_${kind}`, {}, () => ({
        ok: true,
      }));
      assert.deepEqual(result, { ok: true });
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
      () => toolCallExecute('exec_tool_stateful_getter', {}, statefulResult),
      () => toolCallExecuteAsync('exec_async_tool_stateful_getter', {}, async () => statefulResult()),
    ];

    for (const execute of cases) {
      const result = await execute();
      assert.deepEqual(result, {
        value: 1,
      });
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
      () => toolCallExecute('exec_tool_iterator', {}, statefulArray),
      () => toolCallExecuteAsync('exec_async_tool_iterator', {}, async () => statefulArray()),
    ];

    for (const execute of cases) {
      assert.deepEqual(await execute(), [1]);
    }
  });

  it('execute with attributes', async () => {
    const result = await toolCallExecute(
      'exec_attr_tool',
      {},
      () => ({
        ok: true,
      }),
      null,
      TOOL_ATTR_LOCAL,
      null,
      null,
    );
    assert.deepEqual(result, {
      ok: true,
    });
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
      );
      assert.deepEqual(result, {
        result: 2,
      });

      await assert.rejects(
        () =>
          toolCallExecuteAsync(
            'exec_status_error_tool',
            {},
            async () => {
              throw new Error('tool status failure');
            },
            null,
            null,
            null,
            {
              caller: 'node-tool-error',
            },
          ),
        /tool status failure/,
      );

      await waitForSubscriberCallbacks(
        () =>
          events.some(
            (e) =>
              e.name === 'exec_status_ok_tool' &&
              e.kind === 'scope' &&
              e.category === 'tool' &&
              e.scope_category === 'end',
          ) &&
          events.some(
            (e) =>
              e.name === 'exec_status_error_tool' &&
              e.kind === 'scope' &&
              e.category === 'tool' &&
              e.scope_category === 'end',
          ),
      );
      const okEnd = events.find(
        (e) =>
          e.name === 'exec_status_ok_tool' && e.kind === 'scope' && e.category === 'tool' && e.scope_category === 'end',
      );
      const errorEnd = events.find(
        (e) =>
          e.name === 'exec_status_error_tool' &&
          e.kind === 'scope' &&
          e.category === 'tool' &&
          e.scope_category === 'end',
      );
      assert.ok(okEnd, 'expected successful tool end event');
      assert.equal(okEnd.metadata.caller, 'node-tool');
      assert.equal(okEnd.metadata['otel.status_code'], 'OK');
      assert.ok(errorEnd, 'expected failed tool end event');
      assert.equal(errorEnd.metadata.caller, 'node-tool-error');
      assert.equal(errorEnd.metadata['otel.status_code'], 'ERROR');
      assert.match(errorEnd.metadata['otel.status_description'], /tool status failure/);
    } finally {
      deregisterSubscriber('node_tool_status_metadata_sub');
    }
  });

  it('async execute awaits Promise-returning callbacks', async () => {
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
    );
    assert.deepEqual(result, {
      result: 12,
    });
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
        (args) => args,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, {
        x: 1,
      });
      const deadline = Date.now() + 2000;
      while (
        !events.some(
          (e) =>
            e.name === 'san_req_evt_tool' &&
            e.kind === 'scope' &&
            e.category === 'tool' &&
            e.scope_category === 'start',
        ) &&
        Date.now() < deadline
      ) {
        await new Promise((r) => setTimeout(r, 10));
      }
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
        () => ({
          ok: true,
        }),
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, {
        ok: true,
      });
      const deadline = Date.now() + 2000;
      while (
        !events.some(
          (e) =>
            e.name === 'san_resp_evt_tool' && e.kind === 'scope' && e.category === 'tool' && e.scope_category === 'end',
        ) &&
        Date.now() < deadline
      ) {
        await new Promise((r) => setTimeout(r, 10));
      }
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

  it('conditional guardrail treats implicit undefined as allow', async () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond_undefined', 10, () => undefined);
    try {
      const result = await toolCallExecute(
        'tool_cond_undefined',
        {
          ok: true,
        },
        (args) => args,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, {
        ok: true,
      });
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
        () => ({ ok: true }),
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { ok: true });
    } finally {
      deregisterToolConditionalExecutionGuardrail('node_tool_cond_throw');
    }
  });

  it('sanitize guardrail failures preserve the observable payload', async () => {
    clearLastCallbackError();
    const events = [];
    registerSubscriber('node_tool_san_throw_sub', (event) => events.push(event));
    registerToolSanitizeRequestGuardrail('node_tool_san_throw', 10, () => {
      throw new Error('sanitize boom');
    });
    try {
      await toolCallExecute('tool_sanitize_throw', { original: true }, (args) => args, null, null, null, null);
      await waitForSubscriberCallbacks(() => events.some((event) => event.name === 'tool_sanitize_throw'));
      const start = events.find(
        (event) =>
          event.name === 'tool_sanitize_throw' &&
          event.kind === 'scope' &&
          event.category === 'tool' &&
          event.scope_category === 'start',
      );
      assert.deepEqual(start.data, { original: true });
      assert.match(getLastCallbackError() ?? '', /sanitize boom/i);
    } finally {
      deregisterToolSanitizeRequestGuardrail('node_tool_san_throw');
      deregisterSubscriber('node_tool_san_throw_sub');
      clearLastCallbackError();
    }
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
    registerToolExecutionIntercept('node_tool_exec_int', 10, async (args, next) => ({ result: await next(args) }));
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
      (args) => args,
      null,
      null,
      null,
      null,
    );
    assert.equal(result.added, 'yes');
    deregisterToolRequestIntercept('node_tool_req_mod');
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
        () => ({ ok: true }),
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { ok: true });
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
        (args) => args,
        null,
        null,
        null,
        null,
      );
      assert.equal(result, null);
    } finally {
      deregisterToolRequestIntercept('node_tool_req_bad');
    }
  });

  it('execution intercept composes with next', async () => {
    const events = [];
    registerSubscriber('node_tool_exec_mark_sub', (event) => events.push(event));
    registerToolExecutionIntercept('node_tool_exec_repl', 10, async (args, next) => {
      const result = await next({
        ...args,
        intercepted: true,
      });
      return {
        result: {
          ...result,
          wrapped: true,
        },
        pendingMarks: [{ name: 'node.tool.execution' }],
      };
    });
    try {
      const result = await toolCallExecute(
        'replaced_tool',
        {},
        (args) => ({
          original: !args.intercepted,
        }),
        null,
        null,
        null,
        null,
      );
      assert.equal(result.original, false);
      assert.equal(result.wrapped, true);
      await waitForSubscriberCallbacks(
        () =>
          events.some(
            (event) => event.name === 'replaced_tool' && event.kind === 'scope' && event.scope_category === 'end',
          ) && events.some((event) => event.name === 'node.tool.execution'),
      );
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
      assert.equal(mark.parent_uuid, start.uuid);
      assert.ok(events.indexOf(end) < events.indexOf(mark), 'expected tool end before pending mark');
      assert.ok(mark.timestamp > end.timestamp, 'expected pending mark timestamp after tool end');
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_repl');
      deregisterSubscriber('node_tool_exec_mark_sub');
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

  it('execution intercept rejects legacy raw results', async () => {
    registerToolExecutionIntercept('node_tool_exec_legacy', 10, async () => ({ legacyResult: true }));
    try {
      await assert.rejects(
        () => toolCallExecute('legacy_tool', {}, (args) => args, null, null, null, null),
        /invalid JS tool execution intercept outcome/i,
      );
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_legacy');
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

  it('async execute falls back to unknown error for primitive rejections', async () => {
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
      /unknown error/i,
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
