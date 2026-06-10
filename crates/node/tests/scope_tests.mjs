// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  getHandle,
  pushScope,
  popScope,
  event,
  withScope,
  toolCallExecute,
  llmCallExecute,
  registerSubscriber,
  deregisterSubscriber,
  flushSubscribers,
  ScopeType,
} = lib;

const SCOPE_ATTR_PARALLEL = 0b01;
const SCOPE_ATTR_RELOCATABLE = 0b10;

function rejectWithPrimitive(value) {
  return Promise.reject(value);
}

async function flushSubscriberCallbacks() {
  flushSubscribers();
  for (let i = 0; i < 10; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

// ===========================================================================
// Scope operations
// ===========================================================================

describe('Scope operations', () => {
  it('getHandle returns root', () => {
    const handle = getHandle();
    assert.ok(handle.uuid);
    assert.ok(handle.uuid.length > 0);
  });

  it('push and pop scope', () => {
    const scope = pushScope('node_test_scope', ScopeType.Agent, null, null);
    assert.equal(scope.name, 'node_test_scope');
    assert.equal(scope.scopeType, ScopeType.Agent);
    popScope(scope);
  });

  it('scope with attributes', () => {
    const scope = pushScope('attr_scope', ScopeType.Function, null, SCOPE_ATTR_PARALLEL | SCOPE_ATTR_RELOCATABLE);
    assert.equal(scope.attributes, SCOPE_ATTR_PARALLEL | SCOPE_ATTR_RELOCATABLE);
    popScope(scope);
  });

  it('scope with parent', () => {
    const parent = pushScope('parent_scope', ScopeType.Agent, null, null);
    const child = pushScope('child_scope', ScopeType.Function, parent, null);
    assert.equal(child.parentUuid, parent.uuid);
    popScope(child);
    popScope(parent);
  });

  it('scope nesting', () => {
    const s1 = pushScope('nest_1', ScopeType.Agent, null, null);
    const s2 = pushScope('nest_2', ScopeType.Function, null, null);
    const s3 = pushScope('nest_3', ScopeType.Tool, null, null);
    popScope(s3);
    popScope(s2);
    popScope(s1);
  });

  it('all scope types', () => {
    const types = [
      [ScopeType.Agent, 'agent_s'],
      [ScopeType.Function, 'function_s'],
      [ScopeType.Tool, 'tool_s'],
      [ScopeType.Llm, 'llm_s'],
      [ScopeType.Retriever, 'retriever_s'],
      [ScopeType.Embedder, 'embedder_s'],
      [ScopeType.Reranker, 'reranker_s'],
      [ScopeType.Guardrail, 'guardrail_s'],
      [ScopeType.Evaluator, 'evaluator_s'],
      [ScopeType.Custom, 'custom_s'],
      [ScopeType.Unknown, 'unknown_s'],
    ];
    for (const [st, name] of types) {
      const scope = pushScope(name, st, null, null);
      assert.equal(scope.scopeType, st);
      popScope(scope);
    }
  });

  it('popScope merges end metadata over scope metadata', async () => {
    const events = [];
    registerSubscriber('node_scope_pop_metadata_sub', (e) => events.push(e));
    try {
      const scope = pushScope('pop_metadata_scope', ScopeType.Agent, null, null, null, { a: 1, b: 2, c: 3 });
      popScope(scope, null, null, { c: 3.5, d: 4 });
      await flushSubscriberCallbacks();

      const end = events.find(
        (e) => e.name === 'pop_metadata_scope' && e.kind === 'scope' && e.scope_category === 'end',
      );
      assert.ok(end, 'expected scope end event');
      assert.deepEqual(end.metadata, { a: 1, b: 2, c: 3.5, d: 4 });
    } finally {
      deregisterSubscriber('node_scope_pop_metadata_sub');
    }
  });
});

// ===========================================================================
// withScope (context manager)
// ===========================================================================

describe('withScope', () => {
  it('passes handle info to callback and auto-pops scope', async () => {
    const before = getHandle();
    let receivedHandle = null;
    await withScope('with_scope_test', ScopeType.Agent, (handle) => {
      receivedHandle = handle;
    });
    assert.ok(receivedHandle, 'callback should receive handle');
    assert.ok(receivedHandle.uuid, 'handle should have uuid');
    assert.equal(receivedHandle.name, 'with_scope_test');
    assert.equal(receivedHandle.scopeType, ScopeType.Agent);

    // Scope should be popped
    const after = getHandle();
    assert.equal(after.uuid, before.uuid, 'scope should be popped after withScope');
  });

  it('callback receives a reusable ScopeHandle', async () => {
    let toolResult;
    let llmResult;
    let childParentUuid;
    await withScope('reusable_handle', ScopeType.Agent, async (handle) => {
      // The handle is a real ScopeHandle: usable as an event target,
      const handleUuid = handle.uuid;
      event('inside', handle, { ok: true }, null);

      // as an explicit parent for child scopes,
      const child = pushScope('child', ScopeType.Function, handle, null);
      childParentUuid = child.parentUuid;
      popScope(child);

      // and as the scope target for managed tool/LLM execution.
      toolResult = await toolCallExecute(
        'search',
        { query: 'hello' },
        (args) => ({ echo: args.query }),
        handle,
        null,
        null,
        null,
      );
      llmResult = await llmCallExecute(
        'demo-provider',
        { headers: {}, content: { messages: [{ role: 'user', content: 'hi' }] } },
        (request) => ({ ok: true, messages: request.content.messages }),
        handle,
        null,
        null,
        null,
        null,
      );
      assert.equal(childParentUuid, handleUuid, 'child scope should record the handle as its parent');
    });
    assert.deepEqual(toolResult, { echo: 'hello' });
    assert.deepEqual(llmResult, { ok: true, messages: [{ role: 'user', content: 'hi' }] });
  });

  it('returns callback result', async () => {
    const result = await withScope('result_test', ScopeType.Function, () => {
      return {
        value: 42,
      };
    });
    assert.deepEqual(result, {
      value: 42,
    });
  });

  it('returns async callback result', async () => {
    const result = await withScope('async_test', ScopeType.Function, async () => {
      await new Promise((r) => setTimeout(r, 10));
      return {
        async: true,
      };
    });
    assert.deepEqual(result, {
      async: true,
    });
  });

  it('records OK status metadata on successful auto-pop', async () => {
    const events = [];
    registerSubscriber('node_with_scope_ok_status_sub', (e) => events.push(e));
    try {
      await withScope('with_scope_ok_status', ScopeType.Function, () => ({ ok: true }), null, null, null, {
        caller: 'node',
      });
      await flushSubscriberCallbacks();

      const end = events.find(
        (e) => e.name === 'with_scope_ok_status' && e.kind === 'scope' && e.scope_category === 'end',
      );
      assert.ok(end, 'expected scope end event');
      assert.equal(end.metadata.caller, 'node');
      assert.equal(end.metadata['otel.status_code'], 'OK');
      assert.equal(Object.hasOwn(end.metadata, 'otel.status_description'), false);
    } finally {
      deregisterSubscriber('node_with_scope_ok_status_sub');
    }
  });

  it('pops scope on synchronous throw', async () => {
    const before = getHandle();
    await assert.rejects(
      () =>
        withScope('throw_test', ScopeType.Tool, () => {
          throw new Error('test error');
        }),
      /test error/,
    );
    const after = getHandle();
    assert.equal(after.uuid, before.uuid, 'scope should be popped after throw');
  });

  it('pops scope on async rejection', async () => {
    const before = getHandle();
    await assert.rejects(
      () =>
        withScope('reject_test', ScopeType.Tool, async () => {
          await new Promise((r) => setTimeout(r, 10));
          throw new Error('async error');
        }),
      /async error/,
    );
    const after = getHandle();
    assert.equal(after.uuid, before.uuid, 'scope should be popped after rejection');
  });

  it('records ERROR status metadata on failed auto-pop', async () => {
    const events = [];
    registerSubscriber('node_with_scope_error_status_sub', (e) => events.push(e));
    try {
      await assert.rejects(
        () =>
          withScope('with_scope_error_status', ScopeType.Tool, async () => {
            throw new Error('node status failure');
          }),
        /node status failure/,
      );
      await flushSubscriberCallbacks();

      const end = events.find(
        (e) => e.name === 'with_scope_error_status' && e.kind === 'scope' && e.scope_category === 'end',
      );
      assert.ok(end, 'expected scope end event');
      assert.equal(end.metadata['otel.status_code'], 'ERROR');
      assert.match(end.metadata['otel.status_description'], /node status failure/);
    } finally {
      deregisterSubscriber('node_with_scope_error_status_sub');
    }
  });

  it('surfaces primitive rejection values as unknown error and still pops the scope', async () => {
    const before = getHandle();
    await assert.rejects(
      () =>
        withScope('primitive_reject_test', ScopeType.Tool, async () => {
          return rejectWithPrimitive(123);
        }),
      /unknown error/i,
    );
    const after = getHandle();
    assert.equal(after.uuid, before.uuid, 'scope should be popped after primitive rejection');
  });

  it('nested withScope calls', async () => {
    const before = getHandle();
    await withScope('outer', ScopeType.Agent, async (outerHandle) => {
      assert.equal(outerHandle.name, 'outer');
      await withScope('inner', ScopeType.Function, async (innerHandle) => {
        assert.equal(innerHandle.name, 'inner');
        assert.equal(innerHandle.parentUuid, outerHandle.uuid);
      });
    });
    const after = getHandle();
    assert.equal(after.uuid, before.uuid, 'all scopes should be popped');
  });
});

// ===========================================================================
// Events
// ===========================================================================

describe('Events', () => {
  it('basic event', () => {
    event('test_event', null, null, null);
  });

  it('event with data', () => {
    event(
      'data_event',
      null,
      {
        key: 'value',
      },
      null,
    );
  });

  it('event with parent', () => {
    const scope = pushScope('event_parent', ScopeType.Agent, null, null);
    event('child_event', scope, null, null);
    popScope(scope);
  });
});

// ===========================================================================
// Subscribers
// ===========================================================================

describe('Subscribers', () => {
  it('register and deregister', () => {
    registerSubscriber('node_sub_1', () => { });
    const removed = deregisterSubscriber('node_sub_1');
    assert.equal(removed, true);
  });

  it('duplicate subscriber fails', () => {
    registerSubscriber('node_dup_sub', () => { });
    assert.throws(() => registerSubscriber('node_dup_sub', () => { }));
    deregisterSubscriber('node_dup_sub');
  });

  it('deregister nonexistent', () => {
    const removed = deregisterSubscriber('nonexistent_sub');
    assert.equal(removed, false);
  });

  it('subscriber receives events', async () => {
    const events = [];
    registerSubscriber('node_event_collector', (e) => events.push(e));
    try {
      const scope = pushScope('sub_test', ScopeType.Agent, null, null);
      popScope(scope);
      await flushSubscriberCallbacks();
      assert.ok(events.length > 0, 'Expected at least one event');
    } finally {
      deregisterSubscriber('node_event_collector');
    }
  });

  it('flushSubscribers is a native barrier before JS event-loop delivery', async () => {
    const events = [];
    registerSubscriber('node_flush_collector', (e) => events.push(e));
    try {
      event('node_flush_mark', null, null, null);
      flushSubscribers();
      assert.equal(events.length, 0);
      await new Promise((resolve) => setImmediate(resolve));
      assert.ok(events.some((e) => e.kind === 'mark' && e.name === 'node_flush_mark'));
    } finally {
      deregisterSubscriber('node_flush_collector');
    }
  });

  it('subscriber event properties', async () => {
    let captured = null;
    registerSubscriber('node_prop_collector', (e) => {
      if (!captured) captured = e;
    });
    try {
      const scope = pushScope('prop_test', ScopeType.Function, null, null);
      popScope(scope);
      await flushSubscriberCallbacks();
      assert.ok(captured, 'Expected an event');
      assert.ok(typeof captured.uuid === 'string');
      assert.ok(typeof captured.timestamp === 'string');
      assert.ok(typeof captured.kind === 'string');
      assert.equal(structuredClone(captured).kind, captured.kind);
    } finally {
      deregisterSubscriber('node_prop_collector');
    }
  });

  it('mark events', async () => {
    const events = [];
    registerSubscriber('node_mark_collector', (e) => events.push(e));
    try {
      event(
        'mark_event',
        null,
        {
          marker: 'test',
        },
        null,
      );
      await flushSubscriberCallbacks();
      const found = events.some((e) => e.kind === 'mark');
      assert.ok(found, 'Expected a Mark event');
    } finally {
      deregisterSubscriber('node_mark_collector');
    }
  });
});
