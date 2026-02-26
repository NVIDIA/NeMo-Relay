import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  getHandle, pushScope, popScope, event,
  toolCall, toolCallEnd, toolCallExecute,
  llmCall, llmCallEnd, llmCallExecute,
  registerToolSanitizeRequestGuardrail,
  deregisterToolSanitizeRequestGuardrail,
  registerToolSanitizeResponseGuardrail,
  deregisterToolSanitizeResponseGuardrail,
  registerToolConditionalExecutionGuardrail,
  deregisterToolConditionalExecutionGuardrail,
  registerToolRequestIntercept,
  deregisterToolRequestIntercept,
  registerToolResponseIntercept,
  deregisterToolResponseIntercept,
  registerToolExecutionIntercept,
  deregisterToolExecutionIntercept,
  registerLlmSanitizeRequestGuardrail,
  deregisterLlmSanitizeRequestGuardrail,
  registerLlmSanitizeResponseGuardrail,
  deregisterLlmSanitizeResponseGuardrail,
  registerLlmConditionalExecutionGuardrail,
  deregisterLlmConditionalExecutionGuardrail,
  registerLlmRequestIntercept,
  deregisterLlmRequestIntercept,
  registerLlmResponseIntercept,
  deregisterLlmResponseIntercept,
  registerLlmStreamResponseIntercept,
  deregisterLlmStreamResponseIntercept,
  registerLlmExecutionIntercept,
  deregisterLlmExecutionIntercept,
  registerLlmStreamExecutionIntercept,
  deregisterLlmStreamExecutionIntercept,
  registerSubscriber,
  deregisterSubscriber,
  ScopeType, JsLlmRequest,
  SCOPE_ATTR_PARALLEL, SCOPE_ATTR_RELOCATABLE,
  TOOL_ATTR_LOCAL, LLM_ATTR_STATELESS, LLM_ATTR_STREAMING,
} = lib;

// ===========================================================================
// Type constants
// ===========================================================================

describe('Type constants', () => {
  it('scope type enum values', () => {
    assert.equal(ScopeType.Agent, 0);
    assert.equal(ScopeType.Function, 1);
    assert.equal(ScopeType.Tool, 2);
    assert.equal(ScopeType.Llm, 3);
    assert.equal(ScopeType.Retriever, 4);
    assert.equal(ScopeType.Embedder, 5);
    assert.equal(ScopeType.Reranker, 6);
    assert.equal(ScopeType.Guardrail, 7);
    assert.equal(ScopeType.Evaluator, 8);
    assert.equal(ScopeType.Custom, 9);
    assert.equal(ScopeType.Unknown, 10);
  });

  it('attribute constants', () => {
    assert.equal(SCOPE_ATTR_PARALLEL, 0b01);
    assert.equal(SCOPE_ATTR_RELOCATABLE, 0b10);
    assert.equal(TOOL_ATTR_LOCAL, 0b01);
    assert.equal(LLM_ATTR_STATELESS, 0b01);
    assert.equal(LLM_ATTR_STREAMING, 0b10);
  });
});

// ===========================================================================
// JsLlmRequest
// ===========================================================================

describe('JsLlmRequest', () => {
  it('construction and getters', () => {
    const req = new JsLlmRequest({ method: 'POST', url: 'https://api.test.com', headers: { 'Content-Type': 'application/json' }, body: { model: 'gpt-4' } });
    assert.equal(req.method, 'POST');
    assert.equal(req.url, 'https://api.test.com');
    assert.deepEqual(req.headers, { 'Content-Type': 'application/json' });
    assert.deepEqual(req.body, { model: 'gpt-4' });
  });
});

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
});

// ===========================================================================
// Events
// ===========================================================================

describe('Events', () => {
  it('basic event', () => {
    event('test_event', null, null, null);
  });

  it('event with data', () => {
    event('data_event', null, { key: 'value' }, null);
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
    const scope = pushScope('sub_test', ScopeType.Agent, null, null);
    popScope(scope);
    // Subscriber callbacks are NonBlocking — wait for the event loop to drain
    await new Promise(r => setTimeout(r, 50));
    assert.ok(events.length > 0, 'Expected at least one event');
    deregisterSubscriber('node_event_collector');
  });

  it('subscriber event properties', async () => {
    let captured = null;
    registerSubscriber('node_prop_collector', (e) => { if (!captured) captured = e; });
    const scope = pushScope('prop_test', ScopeType.Function, null, null);
    popScope(scope);
    await new Promise(r => setTimeout(r, 50));
    assert.ok(captured, 'Expected an event');
    assert.ok(typeof captured.uuid === 'string');
    assert.ok(typeof captured.timestamp === 'string');
    assert.ok(typeof captured.event_type === 'number');
    deregisterSubscriber('node_prop_collector');
  });

  it('mark events', async () => {
    const events = [];
    registerSubscriber('node_mark_collector', (e) => events.push(e));
    event('mark_event', null, { marker: 'test' }, null);
    await new Promise(r => setTimeout(r, 50));
    const found = events.some(e => e.event_type === 2); // Mark = 2
    assert.ok(found, 'Expected a Mark event (eventType=2)');
    deregisterSubscriber('node_mark_collector');
  });
});

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

// ===========================================================================
// LLM lifecycle
// ===========================================================================

function makeLLMRequest(method, url) {
  return new JsLlmRequest({ method, url, headers: {}, body: {} });
}

describe('LLM lifecycle', () => {
  it('llm call and end', () => {
    const req = makeLLMRequest('POST', 'https://api.test.com');
    const handle = llmCall('test_llm', req, null, null, null, null);
    assert.equal(handle.name, 'test_llm');
    assert.ok(handle.uuid.length > 0);
    llmCallEnd(handle, { choices: [{ text: 'hello' }] }, null, null);
  });

  it('llm call with attributes', () => {
    const req = makeLLMRequest('POST', 'https://api.test.com');
    const handle = llmCall('attr_llm', req, null, LLM_ATTR_STATELESS | LLM_ATTR_STREAMING, null, null);
    assert.equal(handle.attributes, LLM_ATTR_STATELESS | LLM_ATTR_STREAMING);
    llmCallEnd(handle, {}, null, null);
  });

  it('llm call with parent', () => {
    const scope = pushScope('llm_parent', ScopeType.Agent, null, null);
    const req = makeLLMRequest('POST', 'https://api.test.com');
    const handle = llmCall('parented_llm', req, scope, null, null, null);
    assert.equal(handle.parentUuid, scope.uuid);
    llmCallEnd(handle, {}, null, null);
    popScope(scope);
  });

  it('llm call with data/metadata', () => {
    const req = makeLLMRequest('POST', 'https://api.test.com');
    const handle = llmCall('data_llm', req, null, null, { info: 'llm_test' }, { version: '2.0' });
    llmCallEnd(handle, {}, { tokens: 100 }, null);
  });

  it('llm call generates events', async () => {
    const events = [];
    registerSubscriber('node_llm_evt_sub', (e) => events.push(e));
    const req = makeLLMRequest('POST', 'https://api.test.com');
    const handle = llmCall('evt_llm', req, null, null, null, null);
    llmCallEnd(handle, {}, null, null);
    await new Promise(r => setTimeout(r, 50));
    assert.ok(events.length >= 2, 'Expected at least 2 events');
    deregisterSubscriber('node_llm_evt_sub');
  });
});

// ===========================================================================
// LLM execute
// ===========================================================================

describe('LLM execute', () => {
  it('basic execute', async () => {
    const req = makeLLMRequest('POST', 'https://api.test.com');
    const result = await llmCallExecute('exec_llm', req, (request) => ({ response: 'hello from llm' }), null, null, null, null);
    assert.deepEqual(result, { response: 'hello from llm' });
  });
});

// ===========================================================================
// LLM guardrails
// ===========================================================================

describe('LLM guardrails', () => {
  it('sanitize request guardrail', () => {
    registerLlmSanitizeRequestGuardrail('node_llm_san_req', 10, (request) => { request.url = 'https://sanitized.com'; return request; });
    deregisterLlmSanitizeRequestGuardrail('node_llm_san_req');
  });

  it('sanitize response guardrail', () => {
    registerLlmSanitizeResponseGuardrail('node_llm_san_resp', 10, (response) => { response.sanitized = true; return response; });
    deregisterLlmSanitizeResponseGuardrail('node_llm_san_resp');
  });

  it('conditional guardrail (allow)', () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond', 10, (request) => null);
    deregisterLlmConditionalExecutionGuardrail('node_llm_cond');
  });

  it('conditional guardrail (block)', () => {
    registerLlmConditionalExecutionGuardrail('node_llm_block', 10, (request) => 'blocked');
    deregisterLlmConditionalExecutionGuardrail('node_llm_block');
  });

  it('duplicate guardrail fails', () => {
    registerLlmSanitizeRequestGuardrail('node_llm_dup_guard', 10, (r) => r);
    assert.throws(() => registerLlmSanitizeRequestGuardrail('node_llm_dup_guard', 20, (r) => r));
    deregisterLlmSanitizeRequestGuardrail('node_llm_dup_guard');
  });
});

// ===========================================================================
// LLM intercepts
// ===========================================================================

describe('LLM intercepts', () => {
  it('request intercept', () => {
    registerLlmRequestIntercept('node_llm_req_int', 10, false, (request) => { request.url = 'https://intercepted.com'; return request; });
    deregisterLlmRequestIntercept('node_llm_req_int');
  });

  it('response intercept', () => {
    registerLlmResponseIntercept('node_llm_resp_int', 10, false, (response) => { response.intercepted = true; return response; });
    deregisterLlmResponseIntercept('node_llm_resp_int');
  });

  it('execution intercept', () => {
    registerLlmExecutionIntercept('node_llm_exec_int', 10, (request) => true, (request) => ({ replaced: true }));
    deregisterLlmExecutionIntercept('node_llm_exec_int');
  });

  it('stream response intercept', () => {
    registerLlmStreamResponseIntercept('node_llm_sse_int', 10, false, (evt) => evt);
    deregisterLlmStreamResponseIntercept('node_llm_sse_int');
  });

  it('stream execution intercept', () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_exec', 10, (request) => true, (request) => ({ stream_result: true }));
    deregisterLlmStreamExecutionIntercept('node_llm_stream_exec');
  });

  it('request intercept with break_chain', () => {
    registerLlmRequestIntercept('node_llm_break', 10, true, (request) => request);
    deregisterLlmRequestIntercept('node_llm_break');
  });

  it('duplicate intercept fails', () => {
    registerLlmRequestIntercept('node_llm_dup_int', 10, false, (r) => r);
    assert.throws(() => registerLlmRequestIntercept('node_llm_dup_int', 20, false, (r) => r));
    deregisterLlmRequestIntercept('node_llm_dup_int');
  });

  it('request intercept modifies request', async () => {
    registerLlmRequestIntercept('node_llm_req_mod', 10, false, (request) => { request.url = 'https://modified.com'; return request; });
    const req = makeLLMRequest('POST', 'https://original.com');
    const result = await llmCallExecute('mod_llm', req, (request) => ({ url: request.url }), null, null, null, null);
    assert.equal(result.url, 'https://modified.com');
    deregisterLlmRequestIntercept('node_llm_req_mod');
  });

  it('response intercept modifies response', async () => {
    registerLlmResponseIntercept('node_llm_resp_mod', 10, false, (response) => { response.post_processed = true; return response; });
    const req = makeLLMRequest('POST', 'https://api.test.com');
    const result = await llmCallExecute('resp_mod_llm', req, (request) => ({ value: 'test' }), null, null, null, null);
    assert.equal(result.post_processed, true);
    deregisterLlmResponseIntercept('node_llm_resp_mod');
  });

  it('execution intercept replaces func', async () => {
    registerLlmExecutionIntercept('node_llm_exec_repl', 10, (request) => true, (request) => ({ replaced: true }));
    const req = makeLLMRequest('POST', 'https://api.test.com');
    const result = await llmCallExecute('repl_llm', req, (request) => ({ original: true }), null, null, null, null);
    assert.equal(result.replaced, true);
    deregisterLlmExecutionIntercept('node_llm_exec_repl');
  });
});

// ===========================================================================
// Deregister nonexistent
// ===========================================================================

describe('Deregister nonexistent', () => {
  it('tool guardrails', () => {
    assert.equal(deregisterToolSanitizeRequestGuardrail('nx'), false);
    assert.equal(deregisterToolSanitizeResponseGuardrail('nx'), false);
    assert.equal(deregisterToolConditionalExecutionGuardrail('nx'), false);
  });

  it('tool intercepts', () => {
    assert.equal(deregisterToolRequestIntercept('nx'), false);
    assert.equal(deregisterToolResponseIntercept('nx'), false);
    assert.equal(deregisterToolExecutionIntercept('nx'), false);
  });

  it('llm guardrails', () => {
    assert.equal(deregisterLlmSanitizeRequestGuardrail('nx'), false);
    assert.equal(deregisterLlmSanitizeResponseGuardrail('nx'), false);
    assert.equal(deregisterLlmConditionalExecutionGuardrail('nx'), false);
  });

  it('llm intercepts', () => {
    assert.equal(deregisterLlmRequestIntercept('nx'), false);
    assert.equal(deregisterLlmResponseIntercept('nx'), false);
    assert.equal(deregisterLlmExecutionIntercept('nx'), false);
    assert.equal(deregisterLlmStreamResponseIntercept('nx'), false);
    assert.equal(deregisterLlmStreamExecutionIntercept('nx'), false);
  });

  it('subscriber', () => {
    assert.equal(deregisterSubscriber('nx'), false);
  });
});
