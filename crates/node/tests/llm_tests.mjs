// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  pushScope, popScope,
  llmCall, llmCallEnd, llmCallExecute,
  registerLlmSanitizeRequestGuardrail, deregisterLlmSanitizeRequestGuardrail,
  registerLlmSanitizeResponseGuardrail, deregisterLlmSanitizeResponseGuardrail,
  registerLlmConditionalExecutionGuardrail, deregisterLlmConditionalExecutionGuardrail,
  registerLlmRequestIntercept, deregisterLlmRequestIntercept,
  registerLlmResponseIntercept, deregisterLlmResponseIntercept,
  registerLlmStreamResponseIntercept, deregisterLlmStreamResponseIntercept,
  registerLlmExecutionIntercept, deregisterLlmExecutionIntercept,
  registerLlmStreamExecutionIntercept, deregisterLlmStreamExecutionIntercept,
  registerSubscriber, deregisterSubscriber,
  ScopeType,
  LLM_ATTR_STATELESS, LLM_ATTR_STREAMING,
} = lib;

function makeNative() {
  return { messages: [], model: 'test-model' };
}

// ===========================================================================
// LLM lifecycle
// ===========================================================================

describe('LLM lifecycle', () => {
  it('llm call and end', () => {
    const native = makeNative();
    const handle = llmCall('test_llm', native, null, null, null, null);
    assert.equal(handle.name, 'test_llm');
    assert.ok(handle.uuid.length > 0);
    llmCallEnd(handle, { choices: [{ text: 'hello' }] }, null, null);
  });

  it('llm call with attributes', () => {
    const native = makeNative();
    const handle = llmCall('attr_llm', native, null, LLM_ATTR_STATELESS | LLM_ATTR_STREAMING, null, null);
    assert.equal(handle.attributes, LLM_ATTR_STATELESS | LLM_ATTR_STREAMING);
    llmCallEnd(handle, {}, null, null);
  });

  it('llm call with parent', () => {
    const scope = pushScope('llm_parent', ScopeType.Agent, null, null);
    const native = makeNative();
    const handle = llmCall('parented_llm', native, scope, null, null, null);
    assert.equal(handle.parentUuid, scope.uuid);
    llmCallEnd(handle, {}, null, null);
    popScope(scope);
  });

  it('llm call with data/metadata', () => {
    const native = makeNative();
    const handle = llmCall('data_llm', native, null, null, { info: 'llm_test' }, { version: '2.0' });
    llmCallEnd(handle, {}, { tokens: 100 }, null);
  });

  it('llm call generates events', async () => {
    const events = [];
    registerSubscriber('node_llm_evt_sub', (e) => events.push(e));
    const native = makeNative();
    const handle = llmCall('evt_llm', native, null, null, null, null);
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
    const native = makeNative();
    const result = await llmCallExecute('exec_llm', native, (n) => ({ response: 'hello from llm' }), null, null, null, null);
    assert.deepEqual(result, { response: 'hello from llm' });
  });
});

// ===========================================================================
// LLM guardrails
// ===========================================================================

describe('LLM guardrails', () => {
  it('sanitize request guardrail', () => {
    registerLlmSanitizeRequestGuardrail('node_llm_san_req', 10, (request) => { request.extra = 'sanitized'; return request; });
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
    registerLlmRequestIntercept('node_llm_req_int', 10, false, (native) => { native.intercepted = true; return native; });
    deregisterLlmRequestIntercept('node_llm_req_int');
  });

  it('response intercept', () => {
    registerLlmResponseIntercept('node_llm_resp_int', 10, false, (response) => { response.intercepted = true; return response; });
    deregisterLlmResponseIntercept('node_llm_resp_int');
  });

  it('execution intercept', () => {
    registerLlmExecutionIntercept('node_llm_exec_int', 10, (native) => true, (native) => ({ replaced: true }));
    deregisterLlmExecutionIntercept('node_llm_exec_int');
  });

  it('stream response intercept', () => {
    registerLlmStreamResponseIntercept('node_llm_sse_int', 10, false, (chunk) => chunk);
    deregisterLlmStreamResponseIntercept('node_llm_sse_int');
  });

  it('stream execution intercept', () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_exec', 10, (native) => true, (native) => ({ stream_result: true }));
    deregisterLlmStreamExecutionIntercept('node_llm_stream_exec');
  });

  it('request intercept with break_chain', () => {
    registerLlmRequestIntercept('node_llm_break', 10, true, (native) => native);
    deregisterLlmRequestIntercept('node_llm_break');
  });

  it('duplicate intercept fails', () => {
    registerLlmRequestIntercept('node_llm_dup_int', 10, false, (n) => n);
    assert.throws(() => registerLlmRequestIntercept('node_llm_dup_int', 20, false, (n) => n));
    deregisterLlmRequestIntercept('node_llm_dup_int');
  });

  it('request intercept modifies request', async () => {
    registerLlmRequestIntercept('node_llm_req_mod', 10, false, (native) => { native.intercepted = true; return native; });
    const native = makeNative();
    const result = await llmCallExecute('mod_llm', native, (n) => ({ saw_intercepted: n.intercepted || false }), null, null, null, null);
    assert.equal(result.saw_intercepted, true);
    deregisterLlmRequestIntercept('node_llm_req_mod');
  });

  it('response intercept modifies response', async () => {
    registerLlmResponseIntercept('node_llm_resp_mod', 10, false, (response) => {
      // response is LLMResponse { data: ... }
      response.data.post_processed = true;
      return response;
    });
    const native = makeNative();
    const result = await llmCallExecute('resp_mod_llm', native, (n) => ({ value: 'test' }), null, null, null, null);
    assert.equal(result.post_processed, true);
    deregisterLlmResponseIntercept('node_llm_resp_mod');
  });

  it('execution intercept replaces func', async () => {
    registerLlmExecutionIntercept('node_llm_exec_repl', 10, (native) => true, (native) => ({ replaced: true }));
    const native = makeNative();
    const result = await llmCallExecute('repl_llm', native, (n) => ({ original: true }), null, null, null, null);
    assert.equal(result.replaced, true);
    deregisterLlmExecutionIntercept('node_llm_exec_repl');
  });
});
