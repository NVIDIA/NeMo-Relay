// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  pushScope, popScope,
  toolCall, toolCallEnd, toolCallExecute, toolCallExecuteAsync, toolRequestIntercepts, toolConditionalExecution,
  registerToolSanitizeRequestGuardrail, deregisterToolSanitizeRequestGuardrail,
  registerToolSanitizeResponseGuardrail, deregisterToolSanitizeResponseGuardrail,
  registerToolConditionalExecutionGuardrail, deregisterToolConditionalExecutionGuardrail,
  registerToolRequestIntercept, deregisterToolRequestIntercept,
  registerToolExecutionIntercept, deregisterToolExecutionIntercept,
  registerSubscriber, deregisterSubscriber,
  ScopeType, TOOL_ATTR_LOCAL,
} = lib;

// ===========================================================================
// Tool lifecycle
// ===========================================================================

describe('Tool lifecycle', () => {
  it('tool call and end', () => {
    const handle = toolCall('test_tool', { x: 1 }, null, TOOL_ATTR_LOCAL, null, null, 'tool-call-1');
    assert.equal(handle.name, 'test_tool');
    assert.equal(handle.attributes, TOOL_ATTR_LOCAL);
    assert.ok(handle.uuid.length > 0);
    toolCallEnd(handle, { result: 42 }, null, null);
  });

  it('tool call with attributes', () => {
    const handle = toolCall('attr_tool', {}, null, TOOL_ATTR_LOCAL, null, null);
    assert.equal(handle.attributes, TOOL_ATTR_LOCAL);
    toolCallEnd(handle, {}, null, null);
  });

  it('tool call with data/metadata', () => {
    const handle = toolCall('data_tool', {}, null, null, { info: 'test' }, { version: '1.0' });
    toolCallEnd(handle, {}, { done: true }, null);
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
        await new Promise(r => setTimeout(r, 10));
      }
      assert.ok(events.length >= 2, 'Expected at least 2 events');
    } finally {
      deregisterSubscriber('node_tool_evt_sub');
    }
  });

  it('tool call event exposes toolCallId, rootUuid, and payload fields', async () => {
    const events = [];
    const scope = pushScope('tool_event_parent', ScopeType.Agent, null, null);
    registerSubscriber('node_tool_field_sub', (e) => events.push(e));
    try {
      const handle = toolCall(
        'field_tool',
        { x: 1 },
        scope,
        TOOL_ATTR_LOCAL,
        { start: true },
        { meta: true },
        'tool-call-123',
      );
      assert.equal(handle.parentUuid, scope.uuid);
      assert.equal(handle.attributes, TOOL_ATTR_LOCAL);
      toolCallEnd(handle, { result: 42 }, { end: true }, { final: true });

      const deadline = Date.now() + 2000;
      while (events.filter((e) => e.name === 'field_tool').length < 2 && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 10));
      }

      const start = events.find((e) => e.name === 'field_tool' && e.event_type === 0);
      const end = events.find((e) => e.name === 'field_tool' && e.event_type === 1);
      assert.equal(start.tool_call_id, 'tool-call-123');
      assert.ok(start.root_uuid);
      assert.equal(end.root_uuid, start.root_uuid);
      assert.deepEqual(JSON.parse(start.input), { x: 1 });
      assert.deepEqual(JSON.parse(end.output), { result: 42 });
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
    const result = await toolCallExecute('exec_tool', { x: 10 }, (args) => ({ result: args.x + 1 }), null, null, null, null);
    assert.deepEqual(result, { result: 11 });
  });

  it('execute with attributes', async () => {
    const result = await toolCallExecute('exec_attr_tool', {}, () => ({ ok: true }), null, TOOL_ATTR_LOCAL, null, null);
    assert.deepEqual(result, { ok: true });
  });

  it('async execute awaits Promise-returning callbacks', async () => {
    const result = await toolCallExecuteAsync(
      'exec_async_tool',
      { x: 10 },
      async (args) => ({ result: args.x + 2 }),
      null,
      TOOL_ATTR_LOCAL,
      { data: true },
      { meta: true },
    );
    assert.deepEqual(result, { result: 12 });
  });

  it('async execute surfaces plain string rejections', async () => {
    await assert.rejects(
      () => toolCallExecuteAsync('exec_async_tool_reject', { x: 10 }, async () => {
        throw 'string tool error';
      }, null, null, null, null),
      /string tool error/,
    );
  });
});

// ===========================================================================
// Tool guardrails
// ===========================================================================

describe('Tool guardrails', () => {
  it('sanitize request guardrail', () => {
    registerToolSanitizeRequestGuardrail('node_tool_san_req', 10, (name, args) => { args.sanitized = true; return args; });
    deregisterToolSanitizeRequestGuardrail('node_tool_san_req');
  });

  it('sanitize request guardrail rewrites start event payload', async () => {
    const events = [];
    registerSubscriber('node_tool_san_req_evt', (e) => events.push(e));
    registerToolSanitizeRequestGuardrail('node_tool_san_req_evt_guard', 10, (name, args) => ({ ...args, sanitized: true }));
    try {
      const result = await toolCallExecute('san_req_evt_tool', { x: 1 }, (args) => args, null, null, null, null);
      assert.deepEqual(result, { x: 1 });
      const deadline = Date.now() + 2000;
      while (!events.find((e) => e.name === 'san_req_evt_tool' && e.event_type === 0) && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 10));
      }
      const start = events.find((e) => e.name === 'san_req_evt_tool' && e.event_type === 0);
      assert.deepEqual(JSON.parse(start.input), { x: 1, sanitized: true });
    } finally {
      deregisterToolSanitizeRequestGuardrail('node_tool_san_req_evt_guard');
      deregisterSubscriber('node_tool_san_req_evt');
    }
  });

  it('sanitize response guardrail', () => {
    registerToolSanitizeResponseGuardrail('node_tool_san_resp', 10, (name, result) => { result.checked = true; return result; });
    deregisterToolSanitizeResponseGuardrail('node_tool_san_resp');
  });

  it('sanitize response guardrail rewrites end event payload', async () => {
    const events = [];
    registerSubscriber('node_tool_san_resp_evt', (e) => events.push(e));
    registerToolSanitizeResponseGuardrail('node_tool_san_resp_evt_guard', 10, (name, result) => ({ ...result, checked: true }));
    try {
      const result = await toolCallExecute('san_resp_evt_tool', { x: 1 }, () => ({ ok: true }), null, null, null, null);
      assert.deepEqual(result, { ok: true });
      const deadline = Date.now() + 2000;
      while (!events.find((e) => e.name === 'san_resp_evt_tool' && e.event_type === 1) && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 10));
      }
      const end = events.find((e) => e.name === 'san_resp_evt_tool' && e.event_type === 1);
      assert.deepEqual(JSON.parse(end.output), { ok: true, checked: true });
    } finally {
      deregisterToolSanitizeResponseGuardrail('node_tool_san_resp_evt_guard');
      deregisterSubscriber('node_tool_san_resp_evt');
    }
  });

  it('conditional guardrail (allow)', () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond', 10, (name, args) => null);
    deregisterToolConditionalExecutionGuardrail('node_tool_cond');
  });

  it('conditional guardrail (block)', () => {
    registerToolConditionalExecutionGuardrail('node_tool_block', 10, (name, args) => 'blocked');
    deregisterToolConditionalExecutionGuardrail('node_tool_block');
  });

  it('conditional guardrail ignores non-string return values', async () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond_non_string', 10, () => ({ blocked: true }));
    const result = await toolCallExecute('tool_cond_non_string', { ok: true }, (args) => args, null, null, null, null);
    assert.deepEqual(result, { ok: true });
    deregisterToolConditionalExecutionGuardrail('node_tool_cond_non_string');
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
    registerToolRequestIntercept('node_tool_req_int', 10, false, (name, args) => { args.intercepted = true; return args; });
    deregisterToolRequestIntercept('node_tool_req_int');
  });

  it('execution intercept register/deregister', () => {
    registerToolExecutionIntercept('node_tool_exec_int', 10, async (args, next) => next(args));
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
    registerToolRequestIntercept('node_tool_req_mod', 10, false, (name, args) => { args.added = 'yes'; return args; });
    const result = await toolCallExecute('mod_tool', { original: true }, (args) => args, null, null, null, null);
    assert.equal(result.added, 'yes');
    deregisterToolRequestIntercept('node_tool_req_mod');
  });

  it('request intercept returns null on malformed output', async () => {
    registerToolRequestIntercept('node_tool_req_bad', 10, false, () => null);
    try {
      const result = await toolCallExecute('bad_tool', { original: true }, (args) => args, null, null, null, null);
      assert.equal(result, null);
    } finally {
      deregisterToolRequestIntercept('node_tool_req_bad');
    }
  });

  it('execution intercept composes with next', async () => {
    registerToolExecutionIntercept('node_tool_exec_repl', 10, async (args, next) => {
      const result = await next({ ...args, intercepted: true });
      return { ...result, wrapped: true };
    });
    const result = await toolCallExecute('replaced_tool', {}, (args) => ({ original: !args.intercepted }), null, null, null, null);
    assert.equal(result.original, false);
    assert.equal(result.wrapped, true);
    deregisterToolExecutionIntercept('node_tool_exec_repl');
  });

  it('execution intercept propagates Error messages', async () => {
    registerToolExecutionIntercept('node_tool_exec_throw', 10, async () => {
      throw new Error('tool middleware exploded');
    });
    try {
      await assert.rejects(
        () => toolCallExecute('throwing_tool', { value: 1 }, (args) => args, null, null, null, null),
        /tool middleware exploded/,
      );
    } finally {
      deregisterToolExecutionIntercept('node_tool_exec_throw');
    }
  });

  it('async execute falls back to unknown error for primitive rejections', async () => {
    await assert.rejects(
      () => toolCallExecuteAsync('primitive_reject_tool', { value: 1 }, async () => Promise.reject(42), null, null, null, null),
      /unknown error/i,
    );
  });

  it('standalone request intercepts helper applies intercept chain', async () => {
    registerToolRequestIntercept('node_tool_req_helper', 10, false, (name, args) => ({ ...args, helper: true }));
    try {
      const result = await toolRequestIntercepts('helper_tool', { original: true });
      assert.deepEqual(result, { original: true, helper: true });
    } finally {
      deregisterToolRequestIntercept('node_tool_req_helper');
    }
  });

  it('standalone conditional execution helper throws on rejection', async () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond_helper', 10, () => 'blocked by helper');
    try {
      await assert.rejects(() => toolConditionalExecution('helper_tool', { test: true }), /guardrail rejected/i);
    } finally {
      deregisterToolConditionalExecutionGuardrail('node_tool_cond_helper');
    }
  });

  it('standalone conditional execution helper resolves when allowed', async () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond_allow', 10, () => null);
    try {
      await assert.doesNotReject(() => toolConditionalExecution('helper_tool', { test: true }));
    } finally {
      deregisterToolConditionalExecutionGuardrail('node_tool_cond_allow');
    }
  });
});
