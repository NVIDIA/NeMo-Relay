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
  llmCallExecute, llmStreamCallExecute,
  scopeRegisterToolSanitizeRequestGuardrail, scopeDeregisterToolSanitizeRequestGuardrail,
  scopeRegisterToolSanitizeResponseGuardrail, scopeDeregisterToolSanitizeResponseGuardrail,
  scopeRegisterToolConditionalExecutionGuardrail, scopeDeregisterToolConditionalExecutionGuardrail,
  scopeRegisterToolRequestIntercept, scopeDeregisterToolRequestIntercept,
  scopeRegisterToolExecutionIntercept, scopeDeregisterToolExecutionIntercept,
  scopeRegisterLlmSanitizeRequestGuardrail, scopeDeregisterLlmSanitizeRequestGuardrail,
  scopeRegisterLlmSanitizeResponseGuardrail, scopeDeregisterLlmSanitizeResponseGuardrail,
  scopeRegisterLlmConditionalExecutionGuardrail, scopeDeregisterLlmConditionalExecutionGuardrail,
  scopeRegisterLlmRequestIntercept, scopeDeregisterLlmRequestIntercept,
  scopeRegisterLlmExecutionIntercept, scopeDeregisterLlmExecutionIntercept,
  scopeRegisterLlmStreamExecutionIntercept, scopeDeregisterLlmStreamExecutionIntercept,
  scopeRegisterSubscriber, scopeDeregisterSubscriber,
  registerToolSanitizeRequestGuardrail, deregisterToolSanitizeRequestGuardrail,
  registerSubscriber, deregisterSubscriber,
  ScopeType,
} = lib;

function makeNative() {
  return { headers: {}, content: { messages: [], model: 'scope-local-model' } };
}

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
    await waitForEvents(events, (ev) => ev.some(e => e.kind === 'ToolStart'));
    deregisterSubscriber('sl_san_exec_sub');
    const startEvents = events.filter(e => e.kind === 'ToolStart');
    const input = startEvents.length > 0 ? startEvents[0].input : null;
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
    await waitForEvents(events, (ev) => ev.some(e => e.kind === 'ToolEnd'));
    deregisterSubscriber('sl_resp_exec_sub');
    const endEvents = events.filter(e => e.kind === 'ToolEnd');
    const output = endEvents.length > 0 ? endEvents[0].output : null;
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

  it('register and deregister scope-local llm sanitize request guardrail', () => {
    const scope = pushScope('sl_llm_guard_req', ScopeType.Agent, null, null);
    scopeRegisterLlmSanitizeRequestGuardrail(scope.uuid, 'sl_llm_san_req_1', 10, (request) => {
      request.headers = { ...request.headers, scoped: 'yes' };
      return request;
    });
    const removed = scopeDeregisterLlmSanitizeRequestGuardrail(scope.uuid, 'sl_llm_san_req_1');
    assert.equal(removed, true);
    popScope(scope);
  });

  it('scope-local llm sanitize request guardrail rewrites start event payload', async () => {
    const events = [];
    const scope = pushScope('sl_llm_guard_req_exec', ScopeType.Agent, null, null);
    registerSubscriber('sl_llm_san_req_evt_sub', (e) => events.push(e));
    scopeRegisterLlmSanitizeRequestGuardrail(scope.uuid, 'sl_llm_san_req_evt', 10, (request) => {
      request.headers = { ...request.headers, 'X-Scope-Local': 'yes' };
      return request;
    });

    try {
      const result = await llmCallExecute(
        'sl_llm_req_guarded',
        makeNative(),
        (request) => ({ model: request.content.model }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { model: 'scope-local-model' });
      await waitForEvents(events, (ev) => ev.some((e) => e.name === 'sl_llm_req_guarded' && e.kind === 'LLMStart'));
      const start = events.find((e) => e.name === 'sl_llm_req_guarded' && e.kind === 'LLMStart');
      assert.deepEqual(start.input, {
        headers: { 'X-Scope-Local': 'yes' },
        content: { messages: [], model: 'scope-local-model' },
      });
    } finally {
      scopeDeregisterLlmSanitizeRequestGuardrail(scope.uuid, 'sl_llm_san_req_evt');
      deregisterSubscriber('sl_llm_san_req_evt_sub');
      popScope(scope);
    }
  });

  it('register and deregister scope-local llm sanitize response guardrail', () => {
    const scope = pushScope('sl_llm_guard_resp', ScopeType.Agent, null, null);
    scopeRegisterLlmSanitizeResponseGuardrail(scope.uuid, 'sl_llm_san_resp_1', 10, (response) => {
      response.checked = true;
      return response;
    });
    const removed = scopeDeregisterLlmSanitizeResponseGuardrail(scope.uuid, 'sl_llm_san_resp_1');
    assert.equal(removed, true);
    popScope(scope);
  });

  it('scope-local llm sanitize response guardrail rewrites end event payload', async () => {
    const events = [];
    const scope = pushScope('sl_llm_guard_resp_exec', ScopeType.Agent, null, null);
    registerSubscriber('sl_llm_san_resp_evt_sub', (e) => events.push(e));
    scopeRegisterLlmSanitizeResponseGuardrail(scope.uuid, 'sl_llm_san_resp_evt', 10, (response) => {
      response.scopeChecked = true;
      return response;
    });

    try {
      const result = await llmCallExecute(
        'sl_llm_resp_guarded',
        makeNative(),
        () => ({ ok: true }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { ok: true });
      await waitForEvents(events, (ev) => ev.some((e) => e.name === 'sl_llm_resp_guarded' && e.kind === 'LLMEnd'));
      const end = events.find((e) => e.name === 'sl_llm_resp_guarded' && e.kind === 'LLMEnd');
      assert.deepEqual(end.output, { ok: true, scopeChecked: true });
    } finally {
      scopeDeregisterLlmSanitizeResponseGuardrail(scope.uuid, 'sl_llm_san_resp_evt');
      deregisterSubscriber('sl_llm_san_resp_evt_sub');
      popScope(scope);
    }
  });

  it('register and deregister scope-local llm conditional execution guardrail', () => {
    const scope = pushScope('sl_llm_guard_cond', ScopeType.Agent, null, null);
    scopeRegisterLlmConditionalExecutionGuardrail(scope.uuid, 'sl_llm_cond_1', 10, () => null);
    const removed = scopeDeregisterLlmConditionalExecutionGuardrail(scope.uuid, 'sl_llm_cond_1');
    assert.equal(removed, true);
    popScope(scope);
  });

  it('scope-local llm conditional guardrail blocks execution', async () => {
    const scope = pushScope('sl_llm_guard_block', ScopeType.Agent, null, null);
    scopeRegisterLlmConditionalExecutionGuardrail(scope.uuid, 'sl_llm_block_1', 10, () => 'blocked by scope llm guardrail');
    try {
      await assert.rejects(
        () => llmCallExecute(
          'sl_llm_blocked',
          makeNative(),
          () => ({ should_not: 'run' }),
          null,
          null,
          null,
          null,
          null,
        ),
        /blocked|guardrail|rejected/i,
      );
    } finally {
      scopeDeregisterLlmConditionalExecutionGuardrail(scope.uuid, 'sl_llm_block_1');
      popScope(scope);
    }
  });

  it('duplicate scope-local llm guardrail name fails', () => {
    const scope = pushScope('sl_llm_guard_dup', ScopeType.Agent, null, null);
    scopeRegisterLlmSanitizeRequestGuardrail(scope.uuid, 'sl_llm_dup_guard', 10, (request) => request);
    assert.throws(() => scopeRegisterLlmSanitizeRequestGuardrail(scope.uuid, 'sl_llm_dup_guard', 20, (request) => request));
    scopeDeregisterLlmSanitizeRequestGuardrail(scope.uuid, 'sl_llm_dup_guard');
    popScope(scope);
  });

  it('deregister nonexistent scope-local llm guardrail returns false', () => {
    const scope = pushScope('sl_llm_guard_nx', ScopeType.Agent, null, null);
    const removed = scopeDeregisterLlmSanitizeRequestGuardrail(scope.uuid, 'nonexistent_llm_guard');
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

  it('scope-local llm request intercept is cleaned up when scope is popped', async () => {
    const scope = pushScope('sl_cleanup_llm_int', ScopeType.Agent, null, null);
    scopeRegisterLlmRequestIntercept(scope.uuid, 'sl_cleanup_llm_req_int', 10, false, ({ request, annotated }) => {
      request.content.fromPoppedScope = true;
      return { request, annotated };
    });
    popScope(scope);

    const result = await llmCallExecute(
      'sl_cleanup_llm_int_call',
      makeNative(),
      (request) => ({ sawIntercept: request.content.fromPoppedScope || false }),
      null,
      null,
      null,
      null,
      null,
    );
    assert.equal(result.sawIntercept, false);
  });

  it('scope-local tool execution intercept is cleaned up when scope is popped', async () => {
    const scope = pushScope('sl_cleanup_tool_exec', ScopeType.Agent, null, null);
    scopeRegisterToolExecutionIntercept(scope.uuid, 'sl_cleanup_tool_exec_int', 10, async (args, next) => {
      const result = await next({ ...args, fromPoppedScope: true });
      return { ...result, wrapped: true };
    });
    popScope(scope);

    const result = await toolCallExecute(
      'sl_cleanup_tool_exec_call',
      { original: true },
      (args) => ({ sawIntercept: args.fromPoppedScope || false }),
      null,
      null,
      null,
      null,
    );
    assert.deepEqual(result, { sawIntercept: false });
  });

  it('scope-local llm execution intercept is cleaned up when scope is popped', async () => {
    const scope = pushScope('sl_cleanup_llm_exec', ScopeType.Agent, null, null);
    scopeRegisterLlmExecutionIntercept(scope.uuid, 'sl_cleanup_llm_exec_int', 10, async (request, next) => {
      const updated = {
        ...request,
        content: { ...request.content, fromPoppedScope: true },
      };
      const result = await next(updated);
      return { ...result, wrapped: true };
    });
    popScope(scope);

    const result = await llmCallExecute(
      'sl_cleanup_llm_exec_call',
      makeNative(),
      (request) => ({ sawIntercept: request.content.fromPoppedScope || false }),
      null,
      null,
      null,
      null,
      null,
    );
    assert.equal(result.sawIntercept, false);
  });

  it('scope-local llm stream execution intercept is cleaned up when scope is popped', async () => {
    const scope = pushScope('sl_cleanup_llm_stream_exec', ScopeType.Agent, null, null);
    scopeRegisterLlmStreamExecutionIntercept(scope.uuid, 'sl_cleanup_llm_stream_exec_int', 10, async (request, next) => {
      const updated = {
        ...request,
        content: { ...request.content, fromPoppedScope: true },
      };
      return next(updated);
    });
    popScope(scope);

    const stream = await llmStreamCallExecute(
      'sl_cleanup_llm_stream_exec_call',
      makeNative(),
      (wrapper) => {
        lib.pushStreamChunk(wrapper.__nat_nexus_stream_id, {
          sawIntercept: wrapper.__nat_nexus_native.content.fromPoppedScope || false,
        });
        lib.endStream(wrapper.__nat_nexus_stream_id);
      },
      null,
      null,
      null,
      null,
      null,
      null,
      null,
    );

    const seen = [];
    for (;;) {
      const chunk = await stream.next();
      if (chunk === null) {
        break;
      }
      seen.push(chunk);
    }

    assert.deepEqual(seen, [{ sawIntercept: false }]);
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
    await waitForEvents(events, (ev) => ev.some(e => e.kind === 'ToolStart'));
    deregisterSubscriber('sl_merge_sub');
    const toolStartEvents = events.filter(e => e.kind === 'ToolStart');
    const input = toolStartEvents.length > 0 ? toolStartEvents[0].input : null;
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

  it('global and scope-local llm request intercepts both run with priority ordering', async () => {
    const order = [];

    lib.registerLlmRequestIntercept('sl_llm_merge_global_req', 5, false, ({ request, annotated }) => {
      order.push('global');
      request.content.globalIntercepted = true;
      return { request, annotated };
    });

    const scope = pushScope('sl_llm_merge_req_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmRequestIntercept(scope.uuid, 'sl_llm_merge_local_req', 15, false, ({ request, annotated }) => {
      order.push('scope');
      request.content.scopeIntercepted = true;
      return { request, annotated };
    });

    try {
      const result = await llmCallExecute(
        'sl_llm_merge_req_call',
        makeNative(),
        (request) => ({
          globalIntercepted: request.content.globalIntercepted || false,
          scopeIntercepted: request.content.scopeIntercepted || false,
        }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { globalIntercepted: true, scopeIntercepted: true });
      assert.deepEqual(order, ['global', 'scope']);
    } finally {
      scopeDeregisterLlmRequestIntercept(scope.uuid, 'sl_llm_merge_local_req');
      popScope(scope);
      lib.deregisterLlmRequestIntercept('sl_llm_merge_global_req');
    }
  });

  it('scope-local llm execution intercept and global intercept merge', async () => {
    lib.registerLlmExecutionIntercept('sl_llm_merge_global_exec', 5, async (request, next) => {
      const result = await next({
        ...request,
        content: { ...request.content, fromGlobal: true },
      });
      return { ...result, globalExec: true };
    });

    const scope = pushScope('sl_llm_merge_exec_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmExecutionIntercept(scope.uuid, 'sl_llm_merge_local_exec', 15, async (request, next) => {
      const result = await next({
        ...request,
        content: { ...request.content, fromScope: true },
      });
      return { ...result, scopeExec: true };
    });

    try {
      const result = await llmCallExecute(
        'sl_llm_merge_exec_call',
        makeNative(),
        (request) => ({
          fromGlobal: request.content.fromGlobal || false,
          fromScope: request.content.fromScope || false,
        }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, {
        fromGlobal: true,
        fromScope: true,
        scopeExec: true,
        globalExec: true,
      });
    } finally {
      scopeDeregisterLlmExecutionIntercept(scope.uuid, 'sl_llm_merge_local_exec');
      popScope(scope);
      lib.deregisterLlmExecutionIntercept('sl_llm_merge_global_exec');
    }
  });
});

// ===========================================================================
// Scope-local LLM intercepts
// ===========================================================================

describe('Scope-local LLM intercepts', () => {
  it('register and deregister scope-local llm request intercept', () => {
    const scope = pushScope('sl_llm_req_int_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmRequestIntercept(scope.uuid, 'sl_llm_req_int', 10, false, ({ request, annotated }) => ({ request, annotated }));
    const removed = scopeDeregisterLlmRequestIntercept(scope.uuid, 'sl_llm_req_int');
    assert.equal(removed, true);
    popScope(scope);
  });

  it('scope-local llm request intercept modifies request', async () => {
    const scope = pushScope('sl_llm_req_mod_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmRequestIntercept(scope.uuid, 'sl_llm_req_mod', 10, false, ({ request, annotated }) => {
      request.content.scopeIntercepted = true;
      return { request, annotated };
    });

    try {
      const result = await llmCallExecute(
        'sl_llm_req_mod_call',
        makeNative(),
        (request) => ({ scopeIntercepted: request.content.scopeIntercepted || false }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { scopeIntercepted: true });
    } finally {
      scopeDeregisterLlmRequestIntercept(scope.uuid, 'sl_llm_req_mod');
      popScope(scope);
    }
  });

  it('scope-local llm request intercept rejects malformed return values', async () => {
    const scope = pushScope('sl_llm_req_bad_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmRequestIntercept(scope.uuid, 'sl_llm_req_bad', 10, false, () => null);

    try {
      await assert.rejects(
        () => llmCallExecute(
          'sl_llm_req_bad_call',
          makeNative(),
          (request) => ({ model: request.content.model }),
          null,
          null,
          null,
          null,
          null,
        ),
        /expected object with 'request' and 'annotated' fields/i,
      );
    } finally {
      scopeDeregisterLlmRequestIntercept(scope.uuid, 'sl_llm_req_bad');
      popScope(scope);
    }
  });

  it('register and deregister scope-local llm execution intercept', () => {
    const scope = pushScope('sl_llm_exec_int_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmExecutionIntercept(scope.uuid, 'sl_llm_exec_int', 10, async (request, next) => next(request));
    const removed = scopeDeregisterLlmExecutionIntercept(scope.uuid, 'sl_llm_exec_int');
    assert.equal(removed, true);
    popScope(scope);
  });

  it('scope-local llm execution intercept composes with next', async () => {
    const scope = pushScope('sl_llm_exec_compose_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmExecutionIntercept(scope.uuid, 'sl_llm_exec_compose', 10, async (request, next) => {
      const result = await next({
        ...request,
        content: { ...request.content, touchedByScopeExec: true },
      });
      return { ...result, wrappedByScope: true };
    });

    try {
      const result = await llmCallExecute(
        'sl_llm_exec_compose_call',
        makeNative(),
        (request) => ({ touchedByScopeExec: request.content.touchedByScopeExec || false }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { touchedByScopeExec: true, wrappedByScope: true });
    } finally {
      scopeDeregisterLlmExecutionIntercept(scope.uuid, 'sl_llm_exec_compose');
      popScope(scope);
    }
  });

  it('scope-local llm execution intercept rejects invalid next request payloads', async () => {
    const scope = pushScope('sl_llm_exec_invalid_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmExecutionIntercept(scope.uuid, 'sl_llm_exec_invalid', 10, async (_request, next) => {
      return next({ headers: 1, content: { model: 'broken' } });
    });

    try {
      await assert.rejects(
        () => llmCallExecute(
          'sl_llm_exec_invalid_call',
          makeNative(),
          () => ({ ok: true }),
          null,
          null,
          null,
          null,
          null,
        ),
        /invalid LLMRequest from JS next/i,
      );
    } finally {
      scopeDeregisterLlmExecutionIntercept(scope.uuid, 'sl_llm_exec_invalid');
      popScope(scope);
    }
  });

  it('register and deregister scope-local llm stream execution intercept', () => {
    const scope = pushScope('sl_llm_stream_int_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmStreamExecutionIntercept(scope.uuid, 'sl_llm_stream_int', 10, async (request, next) => next(request));
    const removed = scopeDeregisterLlmStreamExecutionIntercept(scope.uuid, 'sl_llm_stream_int');
    assert.equal(removed, true);
    popScope(scope);
  });

  it('scope-local llm stream execution intercept composes with next', async () => {
    const scope = pushScope('sl_llm_stream_compose_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmStreamExecutionIntercept(scope.uuid, 'sl_llm_stream_compose', 10, async (request, next) => {
      const chunks = await next({
        ...request,
        content: { ...request.content, touchedByScopeStream: true },
      });
      return [...chunks, { wrappedByScopeStream: true }];
    });

    try {
      const stream = await llmStreamCallExecute(
        'sl_llm_stream_compose_call',
        makeNative(),
        (wrapper) => {
          lib.pushStreamChunk(wrapper.__nat_nexus_stream_id, { downstream: wrapper.__nat_nexus_native.content.touchedByScopeStream });
          lib.endStream(wrapper.__nat_nexus_stream_id);
        },
        null,
        null,
        null,
        null,
        null,
        null,
        null,
      );

      const seen = [];
      for (;;) {
        const chunk = await stream.next();
        if (chunk === null) {
          break;
        }
        seen.push(chunk);
      }

      assert.deepEqual(seen, [
        { downstream: true },
        { wrappedByScopeStream: true },
      ]);
    } finally {
      scopeDeregisterLlmStreamExecutionIntercept(scope.uuid, 'sl_llm_stream_compose');
      popScope(scope);
    }
  });

  it('scope-local llm stream execution intercept rejects invalid next request payloads', async () => {
    const scope = pushScope('sl_llm_stream_invalid_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmStreamExecutionIntercept(scope.uuid, 'sl_llm_stream_invalid', 10, async (_request, next) => {
      return next({ headers: 1, content: { model: 'broken' } });
    });

    try {
      await assert.rejects(
        () => llmStreamCallExecute(
          'sl_llm_stream_invalid_call',
          makeNative(),
          (wrapper) => {
            lib.pushStreamChunk(wrapper.__nat_nexus_stream_id, { shouldNotRun: true });
            lib.endStream(wrapper.__nat_nexus_stream_id);
          },
          null,
          null,
          null,
          null,
          null,
          null,
          null,
        ),
        /invalid LLMRequest from JS next/i,
      );
    } finally {
      scopeDeregisterLlmStreamExecutionIntercept(scope.uuid, 'sl_llm_stream_invalid');
      popScope(scope);
    }
  });

  it('duplicate scope-local llm stream execution intercept fails', () => {
    const scope = pushScope('sl_llm_stream_dup_scope', ScopeType.Agent, null, null);
    scopeRegisterLlmStreamExecutionIntercept(scope.uuid, 'sl_llm_stream_dup', 10, async (request, next) => next(request));
    assert.throws(() => {
      scopeRegisterLlmStreamExecutionIntercept(scope.uuid, 'sl_llm_stream_dup', 20, async (request, next) => next(request));
    });
    scopeDeregisterLlmStreamExecutionIntercept(scope.uuid, 'sl_llm_stream_dup');
    popScope(scope);
  });

  it('deregister nonexistent scope-local llm stream execution intercept returns false', () => {
    const scope = pushScope('sl_llm_stream_nx_scope', ScopeType.Agent, null, null);
    const removed = scopeDeregisterLlmStreamExecutionIntercept(scope.uuid, 'missing_scope_stream');
    assert.equal(removed, false);
    popScope(scope);
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
    await waitForEvents(events, (ev) => ev.some(e => e.kind === 'Mark'));

    const markEvents = events.filter(e => e.kind === 'Mark');
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
    assert.ok(typeof captured.kind === 'string', 'Event should have kind string');

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

  it('scope-local subscriber captures llm events from the same scope', async () => {
    const events = [];
    const scope = pushScope('sl_llm_sub_scope', ScopeType.Agent, null, null);
    scopeRegisterSubscriber(scope.uuid, 'sl_llm_scope_sub', (e) => events.push(e));

    try {
      const result = await llmCallExecute(
        'sl_llm_subscribed_call',
        makeNative(),
        () => ({ ok: true }),
        null,
        null,
        null,
        null,
        'scope-local-llm',
      );
      assert.deepEqual(result, { ok: true });
      await waitForEvents(events, (ev) => ev.filter((e) => e.name === 'sl_llm_subscribed_call').length >= 2);
      const start = events.find((e) => e.name === 'sl_llm_subscribed_call' && e.kind === 'LLMStart');
      const end = events.find((e) => e.name === 'sl_llm_subscribed_call' && e.kind === 'LLMEnd');
      assert.equal(start.model_name, 'scope-local-llm');
      assert.equal(end.uuid, start.uuid);
    } finally {
      scopeDeregisterSubscriber(scope.uuid, 'sl_llm_scope_sub');
      popScope(scope);
    }
  });

  it('invalid scope UUID is rejected for scope-local wrappers', () => {
    const calls = [
      () => scopeRegisterToolSanitizeRequestGuardrail('not-a-uuid', 'bad_tool_req', 10, (_name, args) => args),
      () => scopeDeregisterToolSanitizeRequestGuardrail('not-a-uuid', 'bad_tool_req'),
      () => scopeRegisterToolSanitizeResponseGuardrail('not-a-uuid', 'bad_tool_resp', 10, (_name, result) => result),
      () => scopeDeregisterToolSanitizeResponseGuardrail('not-a-uuid', 'bad_tool_resp'),
      () => scopeRegisterToolConditionalExecutionGuardrail('not-a-uuid', 'bad_tool_cond', 10, () => null),
      () => scopeDeregisterToolConditionalExecutionGuardrail('not-a-uuid', 'bad_tool_cond'),
      () => scopeRegisterToolRequestIntercept('not-a-uuid', 'bad_tool_int', 10, false, (_name, args) => args),
      () => scopeDeregisterToolRequestIntercept('not-a-uuid', 'bad_tool_int'),
      () => scopeRegisterToolExecutionIntercept('not-a-uuid', 'bad_tool_exec', 10, async (args, next) => next(args)),
      () => scopeDeregisterToolExecutionIntercept('not-a-uuid', 'bad_tool_exec'),
      () => scopeRegisterLlmSanitizeRequestGuardrail('not-a-uuid', 'bad_llm_req', 10, (request) => request),
      () => scopeDeregisterLlmSanitizeRequestGuardrail('not-a-uuid', 'bad_llm_req'),
      () => scopeRegisterLlmSanitizeResponseGuardrail('not-a-uuid', 'bad_llm_resp', 10, (response) => response),
      () => scopeDeregisterLlmSanitizeResponseGuardrail('not-a-uuid', 'bad_llm_resp'),
      () => scopeRegisterLlmConditionalExecutionGuardrail('not-a-uuid', 'bad_llm_cond', 10, () => null),
      () => scopeDeregisterLlmConditionalExecutionGuardrail('not-a-uuid', 'bad_llm_cond'),
      () => scopeRegisterLlmRequestIntercept('not-a-uuid', 'bad_llm_int', 10, false, ({ request, annotated }) => ({ request, annotated })),
      () => scopeDeregisterLlmRequestIntercept('not-a-uuid', 'bad_llm_int'),
      () => scopeRegisterLlmExecutionIntercept('not-a-uuid', 'bad_llm_exec', 10, async (request, next) => next(request)),
      () => scopeDeregisterLlmExecutionIntercept('not-a-uuid', 'bad_llm_exec'),
      () => scopeRegisterLlmStreamExecutionIntercept('not-a-uuid', 'bad_llm_stream', 10, async (request, next) => next(request)),
      () => scopeDeregisterLlmStreamExecutionIntercept('not-a-uuid', 'bad_llm_stream'),
      () => scopeRegisterSubscriber('not-a-uuid', 'bad_scope_sub', () => {}),
      () => scopeDeregisterSubscriber('not-a-uuid', 'bad_scope_sub'),
    ];

    for (const call of calls) {
      assert.throws(call, /invalid UUID/i);
    }
  });
});
