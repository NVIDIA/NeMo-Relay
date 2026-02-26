import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  pushScope, popScope,
  toolCall, toolCallEnd, toolCallExecute,
  registerToolSanitizeRequestGuardrail, deregisterToolSanitizeRequestGuardrail,
  registerToolSanitizeResponseGuardrail, deregisterToolSanitizeResponseGuardrail,
  registerToolConditionalExecutionGuardrail, deregisterToolConditionalExecutionGuardrail,
  registerToolRequestIntercept, deregisterToolRequestIntercept,
  registerToolResponseIntercept, deregisterToolResponseIntercept,
  registerToolExecutionIntercept, deregisterToolExecutionIntercept,
  registerSubscriber, deregisterSubscriber,
  ScopeType, TOOL_ATTR_LOCAL,
} = lib;

// ===========================================================================
// Tool lifecycle
// ===========================================================================

describe('Tool lifecycle', () => {
  it('tool call and end', () => {
    const handle = toolCall('test_tool', { x: 1 }, null, null, null, null);
    assert.equal(handle.name, 'test_tool');
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
    const handle = toolCall('evt_tool', {}, null, null, null, null);
    toolCallEnd(handle, {}, null, null);
    await new Promise(r => setTimeout(r, 50));
    assert.ok(events.length >= 2, 'Expected at least 2 events');
    deregisterSubscriber('node_tool_evt_sub');
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
});

// ===========================================================================
// Tool guardrails
// ===========================================================================

describe('Tool guardrails', () => {
  it('sanitize request guardrail', () => {
    registerToolSanitizeRequestGuardrail('node_tool_san_req', 10, (name, args) => { args.sanitized = true; return args; });
    deregisterToolSanitizeRequestGuardrail('node_tool_san_req');
  });

  it('sanitize response guardrail', () => {
    registerToolSanitizeResponseGuardrail('node_tool_san_resp', 10, (name, result) => { result.checked = true; return result; });
    deregisterToolSanitizeResponseGuardrail('node_tool_san_resp');
  });

  it('conditional guardrail (allow)', () => {
    registerToolConditionalExecutionGuardrail('node_tool_cond', 10, (name, args) => null);
    deregisterToolConditionalExecutionGuardrail('node_tool_cond');
  });

  it('conditional guardrail (block)', () => {
    registerToolConditionalExecutionGuardrail('node_tool_block', 10, (name, args) => 'blocked');
    deregisterToolConditionalExecutionGuardrail('node_tool_block');
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

  it('response intercept register/deregister', () => {
    registerToolResponseIntercept('node_tool_resp_int', 10, false, (name, result) => { result.processed = true; return result; });
    deregisterToolResponseIntercept('node_tool_resp_int');
  });

  it('execution intercept register/deregister', () => {
    registerToolExecutionIntercept('node_tool_exec_int', 10, (name, args) => true, (args) => ({ intercepted: true }));
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

  it('response intercept modifies result', async () => {
    registerToolResponseIntercept('node_tool_resp_mod', 10, false, (name, result) => { result.post_processed = true; return result; });
    const result = await toolCallExecute('resp_mod_tool', {}, (args) => ({ value: 42 }), null, null, null, null);
    assert.equal(result.post_processed, true);
    deregisterToolResponseIntercept('node_tool_resp_mod');
  });

  it('execution intercept replaces func', async () => {
    registerToolExecutionIntercept('node_tool_exec_repl', 10, (name, args) => true, (args) => ({ replaced: true }));
    const result = await toolCallExecute('replaced_tool', {}, (args) => ({ original: true }), null, null, null, null);
    assert.equal(result.replaced, true);
    deregisterToolExecutionIntercept('node_tool_exec_repl');
  });
});
