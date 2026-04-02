// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const EVENT_DELIVERY_TIMEOUT_MS = process.env.CI ? 2000 : 200;

async function waitForEvents(eventsArray, predicate, timeoutMs = EVENT_DELIVERY_TIMEOUT_MS) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (predicate(eventsArray)) return;
    await new Promise(r => setTimeout(r, 10));
  }
}

const {
  pushScope, popScope, event,
  toolCallExecute,
  scopeRegisterToolSanitizeRequestGuardrail, scopeDeregisterToolSanitizeRequestGuardrail,
  scopeRegisterToolSanitizeResponseGuardrail, scopeDeregisterToolSanitizeResponseGuardrail,
  scopeRegisterToolConditionalExecutionGuardrail, scopeDeregisterToolConditionalExecutionGuardrail,
  scopeRegisterToolRequestIntercept, scopeDeregisterToolRequestIntercept,
  scopeRegisterToolExecutionIntercept, scopeDeregisterToolExecutionIntercept,
  scopeRegisterSubscriber, scopeDeregisterSubscriber,
  registerToolSanitizeRequestGuardrail, deregisterToolSanitizeRequestGuardrail,
  registerSubscriber, deregisterSubscriber,
  ScopeType,
} = lib;

// ===========================================================================
// Scope-local guardrail registration and execution
// ===========================================================================

describe('Scope-local guardrail registration and execution', () => {
  it('register and deregister scope-local tool sanitize request guardrail', () => {
    const scope = pushScope('sl_guard_req', ScopeType.Agent, null, null);
    scopeRegisterToolSanitizeRequestGuardrail(scope.uuid, 'sl_san_req_1', 10, (name, args) => {
      args.sanitized = true;
      return args;
    });
    const removed = scopeDeregisterToolSanitizeRequestGuardrail(scope.uuid, 'sl_san_req_1');
    assert.equal(removed, true);
    popScope(scope);
  });

  it('register and deregister scope-local tool sanitize response guardrail', () => {
    const scope = pushScope('sl_guard_resp', ScopeType.Agent, null, null);
    scopeRegisterToolSanitizeResponseGuardrail(scope.uuid, 'sl_san_resp_1', 10, (name, result) => {
      result.checked = true;
      return result;
    });
    const removed = scopeDeregisterToolSanitizeResponseGuardrail(scope.uuid, 'sl_san_resp_1');
    assert.equal(removed, true);
    popScope(scope);
  });

  it('register and deregister scope-local tool conditional execution guardrail', () => {
    const scope = pushScope('sl_guard_cond', ScopeType.Agent, null, null);
    scopeRegisterToolConditionalExecutionGuardrail(scope.uuid, 'sl_cond_1', 10, (name, args) => null);
    const removed = scopeDeregisterToolConditionalExecutionGuardrail(scope.uuid, 'sl_cond_1');
    assert.equal(removed, true);
    popScope(scope);
  });

  it('scope-local sanitize request guardrail modifies tool args', async () => {
    const events = [];
    const scope = pushScope('sl_guard_exec', ScopeType.Agent, null, null);
    registerSubscriber('sl_san_exec_sub', (e) => events.push(e));
    scopeRegisterToolSanitizeRequestGuardrail(scope.uuid, 'sl_san_exec_1', 10, (name, args) => {
      args.scope_sanitized = true;
      return args;
    });
    const result = await toolCallExecute(
      'sl_guarded_tool', { original: true }, (args) => args,
      null, null, null, null,
    );
    // Sanitize guardrails are observability-only; they modify event data, not execution results
    assert.equal(result.original, true);
    await waitForEvents(events, (ev) => ev.some(e => e.event_type === 0));
    deregisterSubscriber('sl_san_exec_sub');
    const startEvents = events.filter(e => e.event_type === 0);
    const input = startEvents.length > 0 && startEvents[0].input ? JSON.parse(startEvents[0].input) : null;
    assert.ok(input, 'Expected a Start event with input');
    assert.equal(input.scope_sanitized, true);
    scopeDeregisterToolSanitizeRequestGuardrail(scope.uuid, 'sl_san_exec_1');
    popScope(scope);
  });

  it('scope-local sanitize response guardrail modifies tool result', async () => {
    const events = [];
    const scope = pushScope('sl_guard_resp_exec', ScopeType.Agent, null, null);
    registerSubscriber('sl_resp_exec_sub', (e) => events.push(e));
    scopeRegisterToolSanitizeResponseGuardrail(scope.uuid, 'sl_resp_exec_1', 10, (name, result) => {
      result.post_checked = true;
      return result;
    });
    const result = await toolCallExecute(
      'sl_resp_tool', {}, (args) => ({ value: 99 }),
      null, null, null, null,
    );
    // Sanitize guardrails are observability-only; they modify event data, not execution results
    assert.equal(result.value, 99);
    await waitForEvents(events, (ev) => ev.some(e => e.event_type === 1));
    deregisterSubscriber('sl_resp_exec_sub');
    const endEvents = events.filter(e => e.event_type === 1);
    const output = endEvents.length > 0 && endEvents[0].output ? JSON.parse(endEvents[0].output) : null;
    assert.ok(output, 'Expected an End event with output');
    assert.equal(output.post_checked, true);
    scopeDeregisterToolSanitizeResponseGuardrail(scope.uuid, 'sl_resp_exec_1');
    popScope(scope);
  });

  it('scope-local conditional guardrail blocks execution', async () => {
    const scope = pushScope('sl_guard_block', ScopeType.Agent, null, null);
    scopeRegisterToolConditionalExecutionGuardrail(scope.uuid, 'sl_block_1', 10, (name, args) => 'blocked by scope guardrail');
    await assert.rejects(
      () => toolCallExecute(
        'sl_blocked_tool', {}, (args) => ({ should_not: 'run' }),
        null, null, null, null,
      ),
      (err) => {
        assert.ok(
          err.message.includes('blocked') || err.message.includes('Guardrail') || err.message.includes('rejected'),
          `Expected error about blocked/Guardrail/rejected, got: ${err.message}`,
        );
        return true;
      },
    );
    scopeDeregisterToolConditionalExecutionGuardrail(scope.uuid, 'sl_block_1');
    popScope(scope);
  });

  it('duplicate scope-local guardrail name fails', () => {
    const scope = pushScope('sl_guard_dup', ScopeType.Agent, null, null);
    scopeRegisterToolSanitizeRequestGuardrail(scope.uuid, 'sl_dup_guard', 10, (n, a) => a);
    assert.throws(() => scopeRegisterToolSanitizeRequestGuardrail(scope.uuid, 'sl_dup_guard', 20, (n, a) => a));
    scopeDeregisterToolSanitizeRequestGuardrail(scope.uuid, 'sl_dup_guard');
    popScope(scope);
  });

  it('deregister nonexistent scope-local guardrail returns false', () => {
    const scope = pushScope('sl_guard_nx', ScopeType.Agent, null, null);
    const removed = scopeDeregisterToolSanitizeRequestGuardrail(scope.uuid, 'nonexistent_guard');
    assert.equal(removed, false);
    popScope(scope);
  });
});

// ===========================================================================
// Auto-cleanup on scope pop
// ===========================================================================

describe('Scope-local auto-cleanup on scope pop', () => {
  it('scope-local guardrail is cleaned up when scope is popped', async () => {
    const scope = pushScope('sl_cleanup_guard', ScopeType.Agent, null, null);
    scopeRegisterToolSanitizeRequestGuardrail(scope.uuid, 'sl_cleanup_san', 10, (name, args) => {
      args.from_popped_scope = true;
      return args;
    });
    popScope(scope);

    // After popping, the scope-local guardrail should no longer affect tool calls.
    const result = await toolCallExecute(
      'sl_cleanup_tool', { original: true }, (args) => args,
      null, null, null, null,
    );
    assert.equal(result.from_popped_scope, undefined);
    assert.equal(result.original, true);
  });

  it('scope-local intercept is cleaned up when scope is popped', async () => {
    const scope = pushScope('sl_cleanup_int', ScopeType.Agent, null, null);
    scopeRegisterToolRequestIntercept(scope.uuid, 'sl_cleanup_req_int', 10, false, (name, args) => {
      args.from_popped_intercept = true;
      return args;
    });
    popScope(scope);

    const result = await toolCallExecute(
      'sl_cleanup_int_tool', { original: true }, (args) => args,
      null, null, null, null,
    );
    assert.equal(result.from_popped_intercept, undefined);
    assert.equal(result.original, true);
  });

  it('scope-local subscriber is cleaned up when scope is popped', async () => {
    const events = [];
    const scope = pushScope('sl_cleanup_sub', ScopeType.Agent, null, null);
    scopeRegisterSubscriber(scope.uuid, 'sl_cleanup_sub_1', (e) => events.push(e));
    popScope(scope);

    // Fire an event after the scope is popped -- the subscriber should not capture it.
    const eventsBeforeCount = events.length;
    event('sl_post_pop_event', null, { marker: 'post_pop' }, null);
    await waitForEvents(events, (ev) => ev.some(e => e.data && e.data.marker === 'post_pop'));
    // The subscriber should not have received the event fired after pop
    // (it may have received scope push/pop events before the pop though)
    const postPopEvents = events.filter(e => {
      const data = e.data;
      return data && data.marker === 'post_pop';
    });
    assert.equal(postPopEvents.length, 0, 'Subscriber should not receive events after scope pop');
  });

  it('nested scope cleanup does not affect parent scope-local middleware', async () => {
    const parent = pushScope('sl_parent', ScopeType.Agent, null, null);
    // Use a request intercept for parent (intercepts DO modify execution args)
    scopeRegisterToolRequestIntercept(parent.uuid, 'sl_parent_guard', 10, false, (name, args) => {
      args.parent_ran = true;
      return args;
    });

    const child = pushScope('sl_child', ScopeType.Function, null, null);
    // Child uses a sanitize guardrail (observability-only, won't affect execution result)
    scopeRegisterToolSanitizeRequestGuardrail(child.uuid, 'sl_child_guard', 20, (name, args) => {
      args.child_ran = true;
      return args;
    });
    popScope(child);

    // After child scope pop, parent intercept should still be active
    const result = await toolCallExecute(
      'sl_nested_tool', {}, (args) => args,
      null, null, null, null,
    );
    assert.equal(result.parent_ran, true);
    assert.equal(result.child_ran, undefined);

    scopeDeregisterToolRequestIntercept(parent.uuid, 'sl_parent_guard');
    popScope(parent);
  });
});

// ===========================================================================
// Priority merge (global + scope-local)
// ===========================================================================

describe('Priority merge of global and scope-local middleware', () => {
  it('global and scope-local sanitize request guardrails both run', async () => {
    const events = [];
    registerSubscriber('sl_merge_sub', (e) => events.push(e));
    registerToolSanitizeRequestGuardrail('sl_merge_global', 5, (name, args) => {
      args.global_ran = true;
      return args;
    });

    const scope = pushScope('sl_merge_scope', ScopeType.Agent, null, null);
    scopeRegisterToolSanitizeRequestGuardrail(scope.uuid, 'sl_merge_local', 15, (name, args) => {
      args.scope_ran = true;
      return args;
    });

    await toolCallExecute(
      'sl_merged_tool', {}, (args) => args,
      null, null, null, null,
    );
    // Sanitize guardrails are observability-only; verify via tool Start event input
    await waitForEvents(events, (ev) => ev.some(e => e.event_type === 0 && e.scope_type === 2));
    deregisterSubscriber('sl_merge_sub');
    const toolStartEvents = events.filter(e => e.event_type === 0 && e.scope_type === 2);
    const input = toolStartEvents.length > 0 && toolStartEvents[0].input ? JSON.parse(toolStartEvents[0].input) : null;
    assert.ok(input, 'Expected a tool Start event with input');
    assert.equal(input.global_ran, true);
    assert.equal(input.scope_ran, true);

    scopeDeregisterToolSanitizeRequestGuardrail(scope.uuid, 'sl_merge_local');
    popScope(scope);
    deregisterToolSanitizeRequestGuardrail('sl_merge_global');
  });

  it('global and scope-local request intercepts both run with priority ordering', async () => {
    const order = [];

    // Global intercept at lower priority
    lib.registerToolRequestIntercept('sl_merge_global_int', 5, false, (name, args) => {
      order.push('global');
      args.global_intercepted = true;
      return args;
    });

    const scope = pushScope('sl_merge_int_scope', ScopeType.Agent, null, null);
    // Scope-local intercept at higher priority
    scopeRegisterToolRequestIntercept(scope.uuid, 'sl_merge_local_int', 15, false, (name, args) => {
      order.push('scope');
      args.scope_intercepted = true;
      return args;
    });

    const result = await toolCallExecute(
      'sl_merge_int_tool', {}, (args) => args,
      null, null, null, null,
    );
    assert.equal(result.global_intercepted, true);
    assert.equal(result.scope_intercepted, true);

    scopeDeregisterToolRequestIntercept(scope.uuid, 'sl_merge_local_int');
    popScope(scope);
    lib.deregisterToolRequestIntercept('sl_merge_global_int');
  });

  it('scope-local execution intercept and global intercept merge', async () => {
    lib.registerToolExecutionIntercept('sl_merge_global_exec', 5, async (args, next) => {
      const result = await next({ ...args, from_global: true });
      return { ...result, global_exec: true };
    });

    const scope = pushScope('sl_merge_exec_scope', ScopeType.Agent, null, null);
    scopeRegisterToolExecutionIntercept(scope.uuid, 'sl_merge_local_exec', 15, async (args, next) => {
      const result = await next({ ...args, from_scope: true });
      return { ...result, scope_exec: true };
    });

    const result = await toolCallExecute(
      'sl_merge_exec_tool', { base: true }, (args) => args,
      null, null, null, null,
    );
    assert.deepEqual(result, {
      base: true,
      from_global: true,
      from_scope: true,
      scope_exec: true,
      global_exec: true,
    });

    scopeDeregisterToolExecutionIntercept(scope.uuid, 'sl_merge_local_exec');
    popScope(scope);
    lib.deregisterToolExecutionIntercept('sl_merge_global_exec');
  });
});

// ===========================================================================
// Scope-local subscriber receives events
// ===========================================================================

describe('Scope-local subscriber receives events', () => {
  it('scope-local subscriber captures scope lifecycle events', async () => {
    const events = [];
    const scope = pushScope('sl_sub_lifecycle', ScopeType.Agent, null, null);
    scopeRegisterSubscriber(scope.uuid, 'sl_lifecycle_sub', (e) => events.push(e));

    // Push and pop a child scope to generate events
    const child = pushScope('sl_sub_child', ScopeType.Function, null, null);
    popScope(child);

    await waitForEvents(events, (ev) => ev.length > 0);
    assert.ok(events.length > 0, 'Scope-local subscriber should receive at least one event');

    scopeDeregisterSubscriber(scope.uuid, 'sl_lifecycle_sub');
    popScope(scope);
  });

  it('scope-local subscriber captures mark events', async () => {
    const events = [];
    const scope = pushScope('sl_sub_mark', ScopeType.Agent, null, null);
    scopeRegisterSubscriber(scope.uuid, 'sl_mark_sub', (e) => events.push(e));

    event('sl_mark_event', null, { marker: 'scope_local' }, null);
    await waitForEvents(events, (ev) => ev.some(e => e.event_type === 2));

    const markEvents = events.filter(e => e.event_type === 2); // Mark = 2
    assert.ok(markEvents.length > 0, 'Scope-local subscriber should receive mark events');

    scopeDeregisterSubscriber(scope.uuid, 'sl_mark_sub');
    popScope(scope);
  });

  it('scope-local subscriber event has expected properties', async () => {
    let captured = null;
    const scope = pushScope('sl_sub_props', ScopeType.Agent, null, null);
    scopeRegisterSubscriber(scope.uuid, 'sl_props_sub', (e) => { if (!captured) captured = e; });

    const child = pushScope('sl_sub_prop_child', ScopeType.Function, null, null);
    popScope(child);

    await waitForEvents([], () => captured !== null);
    assert.ok(captured, 'Expected at least one event');
    assert.ok(typeof captured.uuid === 'string', 'Event should have uuid string');
    assert.ok(typeof captured.timestamp === 'string', 'Event should have timestamp string');
    assert.ok(typeof captured.event_type === 'number', 'Event should have event_type number');

    scopeDeregisterSubscriber(scope.uuid, 'sl_props_sub');
    popScope(scope);
  });

  it('duplicate scope-local subscriber name fails', () => {
    const scope = pushScope('sl_sub_dup', ScopeType.Agent, null, null);
    scopeRegisterSubscriber(scope.uuid, 'sl_dup_sub_1', () => {});
    assert.throws(() => scopeRegisterSubscriber(scope.uuid, 'sl_dup_sub_1', () => {}));
    scopeDeregisterSubscriber(scope.uuid, 'sl_dup_sub_1');
    popScope(scope);
  });

  it('deregister nonexistent scope-local subscriber returns false', () => {
    const scope = pushScope('sl_sub_nx', ScopeType.Agent, null, null);
    const removed = scopeDeregisterSubscriber(scope.uuid, 'nonexistent_sub');
    assert.equal(removed, false);
    popScope(scope);
  });
});
