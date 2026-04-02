// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  pushScope, popScope,
  llmCall, llmCallEnd, llmCallExecute, llmStreamCallExecute,
  registerLlmSanitizeRequestGuardrail, deregisterLlmSanitizeRequestGuardrail,
  registerLlmSanitizeResponseGuardrail, deregisterLlmSanitizeResponseGuardrail,
  registerLlmConditionalExecutionGuardrail, deregisterLlmConditionalExecutionGuardrail,
  registerLlmRequestIntercept, deregisterLlmRequestIntercept,
  registerLlmExecutionIntercept, deregisterLlmExecutionIntercept,
  registerLlmStreamExecutionIntercept, deregisterLlmStreamExecutionIntercept,
  registerSubscriber, deregisterSubscriber,
  ScopeType,
  LLM_ATTR_STATELESS, LLM_ATTR_STREAMING,
} = lib;

function makeNative() {
  return { headers: {}, content: { messages: [], model: 'test-model' } };
}

// ===========================================================================
// LLM lifecycle
// ===========================================================================

describe('LLM lifecycle', () => {
  it('llm call and end', () => {
    const native = makeNative();
    const handle = llmCall('test_llm', native, null, null, null, null, null);
    assert.equal(handle.name, 'test_llm');
    assert.ok(handle.uuid.length > 0);
    llmCallEnd(handle, { choices: [{ text: 'hello' }] }, null, null);
  });

  it('llm call with attributes', () => {
    const native = makeNative();
    const handle = llmCall('attr_llm', native, null, LLM_ATTR_STATELESS | LLM_ATTR_STREAMING, null, null, null);
    assert.equal(handle.attributes, LLM_ATTR_STATELESS | LLM_ATTR_STREAMING);
    llmCallEnd(handle, {}, null, null);
  });

  it('llm call with parent', () => {
    const scope = pushScope('llm_parent', ScopeType.Agent, null, null);
    const native = makeNative();
    const handle = llmCall('parented_llm', native, scope, null, null, null, null);
    assert.equal(handle.parentUuid, scope.uuid);
    llmCallEnd(handle, {}, null, null);
    popScope(scope);
  });

  it('llm call with data/metadata', () => {
    const native = makeNative();
    const handle = llmCall('data_llm', native, null, null, { info: 'llm_test' }, { version: '2.0' }, null);
    llmCallEnd(handle, {}, { tokens: 100 }, null);
  });

  it('llm call generates events', async () => {
    const events = [];
    registerSubscriber('node_llm_evt_sub', (e) => events.push(e));
    try {
      const native = makeNative();
      const handle = llmCall('evt_llm', native, null, null, null, null, null);
      llmCallEnd(handle, {}, null, null);
      const deadline = Date.now() + 2000;
      while (events.length < 2 && Date.now() < deadline) {
        await new Promise(r => setTimeout(r, 10));
      }
      assert.ok(events.length >= 2, 'Expected at least 2 events');
    } finally {
      deregisterSubscriber('node_llm_evt_sub');
    }
  });
});

// ===========================================================================
// LLM execute
// ===========================================================================

describe('LLM execute', () => {
  it('basic execute', async () => {
    const native = makeNative();
    const result = await llmCallExecute('exec_llm', native, (n) => ({ response: 'hello from llm' }), null, null, null, null, null);
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

  it('execution intercept', () => {
    registerLlmExecutionIntercept('node_llm_exec_int', 10, async (native, next) => next(native));
    deregisterLlmExecutionIntercept('node_llm_exec_int');
  });

  it('stream execution intercept', () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_exec', 10, async (native, next) => next(native));
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
    registerLlmRequestIntercept('node_llm_req_mod', 10, false, (native) => { native.content.intercepted = true; return native; });
    const native = makeNative();
    const result = await llmCallExecute('mod_llm', native, (n) => ({ saw_intercepted: n.content.intercepted || false }), null, null, null, null, null);
    assert.equal(result.saw_intercepted, true);
    deregisterLlmRequestIntercept('node_llm_req_mod');
  });

  it('execution intercept composes with next', async () => {
    registerLlmExecutionIntercept('node_llm_exec_repl', 10, async (native, next) => {
      native.content.intercepted = true;
      const result = await next(native);
      return { ...result, wrapped: true };
    });
    const native = makeNative();
    const result = await llmCallExecute('repl_llm', native, (n) => ({ original: !n.content.intercepted }), null, null, null, null, null);
    assert.equal(result.original, false);
    assert.equal(result.wrapped, true);
    deregisterLlmExecutionIntercept('node_llm_exec_repl');
  });

  it('stream execution intercept composes with next', async () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_exec_repl', 10, async (native, next) => {
      native.content.intercepted = true;
      const chunks = await next(native);
      return [...chunks, { wrapped: native.content.intercepted }];
    });

    const native = makeNative();
    const seen = [];
    const stream = await llmStreamCallExecute(
      'stream_llm',
      native,
      (wrapper) => {
        lib.pushStreamChunk(wrapper.__nat_nexus_stream_id, { chunk: wrapper.__nat_nexus_native.content.intercepted });
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

    for (;;) {
      const chunk = await stream.next();
      if (chunk === null) {
        break;
      }
      seen.push(chunk);
    }

    assert.deepEqual(seen, [{ chunk: true }, { wrapped: true }]);
    deregisterLlmStreamExecutionIntercept('node_llm_stream_exec_repl');
  });
});
