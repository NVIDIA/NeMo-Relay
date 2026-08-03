// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const lib = require('../index.js');
const nodeDir = fileURLToPath(new URL('..', import.meta.url));

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

function runSubscriberFailureChild({ callback, registration = 'global' }) {
  const register = {
    global: "registerSubscriber('bad', () => { " + callback + ' });',
    scope: [
      "globalThis.scope = pushScope('subscriber_failure_scope', ScopeType.Agent, null, null);",
      "scopeRegisterSubscriber(scope.uuid, 'bad', () => { " + callback + ' });',
    ].join('\n'),
    plugin: [
      "process.chdir(require('node:os').tmpdir());",
      'globalThis.plugin = require(' + JSON.stringify(path.join(nodeDir, 'plugin.js')) + ');',
      "globalThis.pluginKind = 'node.test.subscriber-failure';",
      'plugin.register(pluginKind, {',
      '  register(_config, context) {',
      "    context.registerSubscriber('bad', () => { " + callback + ' });',
      '  },',
      '});',
      "await plugin.initialize({ version: 1, components: [plugin.ComponentSpec('observability', { version: 3 }), plugin.ComponentSpec(pluginKind)] });",
    ].join('\n'),
  }[registration];
  const scopeEvent = registration === 'scope' ? 'scope' : 'null';
  const cleanup = {
    global: "deregisterSubscriber('bad');",
    scope: "scopeDeregisterSubscriber(scope.uuid, 'bad'); popScope(scope);",
    plugin: 'plugin.clear(); plugin.deregister(pluginKind);',
  }[registration];
  const script = `
    const lib = require(${JSON.stringify(path.join(nodeDir, 'index.js'))});
    const {
      ScopeType, clearLastCallbackError, deregisterSubscriber, event, flushSubscribers,
      getLastCallbackError, pushScope, popScope, registerSubscriber,
      scopeDeregisterSubscriber, scopeRegisterSubscriber,
    } = lib;
    (async () => {
      clearLastCallbackError();
      let healthyCalls = 0;
      registerSubscriber('healthy', (event) => {
        if (event.name === 'subscriber_failure_mark') healthyCalls += 1;
      });
      try {
        ${register}
        event('subscriber_failure_mark', ${scopeEvent}, null, null);
        await flushSubscribers();
        if (healthyCalls !== 1) throw new Error('healthy subscriber did not receive the event');
        const error = getLastCallbackError() ?? '';
        if (!/bad/.test(error) || !/subscriber boom/.test(error)) {
          throw new Error('missing subscriber error: ' + error);
        }
        console.log('subscriber failure isolated');
      } finally {
        ${cleanup}
        deregisterSubscriber('healthy');
        clearLastCallbackError();
      }
    })().catch((error) => {
      console.error(error);
      process.exitCode = 1;
    });
  `;
  let output;
  try {
    output = execFileSync(process.execPath, ['--eval', script], {
      cwd: nodeDir,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  } catch (error) {
    throw new Error(`subscriber failure child failed (${registration}):\n${error.stdout ?? ''}${error.stderr ?? ''}`);
  }
  assert.match(output, /subscriber failure isolated/);
}

function rejectWithPrimitive(value) {
  return Promise.reject(value);
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
      await flushSubscribers();

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
      await flushSubscribers();

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
      await flushSubscribers();

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

  it('surfaces primitive rejection values and still pops the scope', async () => {
    const before = getHandle();
    await assert.rejects(
      () =>
        withScope('primitive_reject_test', ScopeType.Tool, async () => {
          return rejectWithPrimitive(123);
        }),
      /internal error: 123/i,
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
    registerSubscriber('node_sub_1', () => {});
    const removed = deregisterSubscriber('node_sub_1');
    assert.equal(removed, true);
  });

  it('duplicate subscriber fails', () => {
    registerSubscriber('node_dup_sub', () => {});
    assert.throws(() => registerSubscriber('node_dup_sub', () => {}));
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
      await flushSubscribers();
    } finally {
      deregisterSubscriber('node_event_collector');
    }
  });

  it('flushSubscribers asynchronously drains the native dispatcher', async () => {
    const events = [];
    registerSubscriber('node_flush_collector', (e) => events.push(e));
    try {
      event('node_flush_mark', null, null, null);
      await flushSubscribers();
      assert.ok(events.some((e) => e.kind === 'mark' && e.name === 'node_flush_mark'));
    } finally {
      deregisterSubscriber('node_flush_collector');
    }
  });

  it('flushSubscribers waits for JavaScript callbacks without inspecting their return values', async () => {
    let called = false;
    registerSubscriber('node_flush_js_callback', () => {
      called = true;
      return 1n;
    });
    try {
      event('node_flush_js_callback_mark', null, null, null);
      await flushSubscribers();
      assert.equal(called, true);
    } finally {
      deregisterSubscriber('node_flush_js_callback');
    }
  });

  it('flushSubscribers waits for a fulfilled JavaScript subscriber promise', async () => {
    let release;
    let settled = false;
    let flushed = false;
    const deferred = new Promise((resolve) => {
      release = resolve;
    });
    registerSubscriber('node_flush_js_promise_callback', async () => {
      await deferred;
      settled = true;
    });
    try {
      event('node_flush_js_promise_callback_mark', null, null, null);
      const flushing = flushSubscribers().then(() => {
        flushed = true;
      });
      await new Promise((resolve) => setImmediate(resolve));
      assert.equal(flushed, false);
      release();
      await flushing;
      assert.equal(settled, true);
    } finally {
      deregisterSubscriber('node_flush_js_promise_callback');
    }
  });

  it('isolates a synchronous global subscriber throw', () => {
    runSubscriberFailureChild({
      callback: "throw new Error('sync subscriber boom');",
    });
  });

  it('isolates a rejected global subscriber promise', () => {
    runSubscriberFailureChild({
      callback: "return Promise.reject(new Error('async subscriber boom'));",
    });
  });

  it('uses the safe adapter for scope-local subscribers', () => {
    runSubscriberFailureChild({
      registration: 'scope',
      callback: "throw new Error('scope subscriber boom');",
    });
  });

  it('uses the safe adapter for plugin subscribers', () => {
    runSubscriberFailureChild({
      registration: 'plugin',
      callback: "return Promise.reject(new Error('plugin subscriber boom'));",
    });
  });

  it('subscriber event properties', async () => {
    let captured = null;
    registerSubscriber('node_prop_collector', (e) => {
      if (!captured) captured = e;
    });
    try {
      const scope = pushScope('prop_test', ScopeType.Function, null, null);
      popScope(scope);
      await flushSubscribers();
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
      await flushSubscribers();
      const found = events.some((e) => e.kind === 'mark');
      assert.ok(found, 'Expected a Mark event');
    } finally {
      deregisterSubscriber('node_mark_collector');
    }
  });
});
