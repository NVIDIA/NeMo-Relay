// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { waitForSubscriberCallbacks } from './test_support.mjs';

const require = createRequire(import.meta.url);
const lib = require('../index.js');
const nodeDir = fileURLToPath(new URL('..', import.meta.url));

const {
  pushScope,
  popScope,
  llmCall,
  llmCallEnd,
  llmCallExecute,
  llmCallExecuteAsync,
  llmStreamCallExecute,
  llmRequestIntercepts,
  llmConditionalExecution,
  registerLlmSanitizeRequestGuardrail,
  deregisterLlmSanitizeRequestGuardrail,
  registerLlmSanitizeResponseGuardrail,
  deregisterLlmSanitizeResponseGuardrail,
  registerLlmConditionalExecutionGuardrail,
  deregisterLlmConditionalExecutionGuardrail,
  registerLlmRequestIntercept,
  deregisterLlmRequestIntercept,
  registerLlmExecutionIntercept,
  deregisterLlmExecutionIntercept,
  registerLlmStreamExecutionIntercept,
  deregisterLlmStreamExecutionIntercept,
  registerSubscriber,
  deregisterSubscriber,
  flushSubscribers,
  clearLastCallbackError,
  getLastCallbackError,
  ScopeType,
} = lib;

const LLM_ATTR_STATELESS = 0b01;
const LLM_ATTR_STREAMING = 0b10;

function rejectWith(value) {
  return Promise.reject(value);
}

async function flushSubscriberCallbacks() {
  await flushSubscribers();
  for (let i = 0; i < 10; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

async function assertCompletesWithin(promise, message) {
  let timeout;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timeout = setTimeout(() => reject(new assert.AssertionError({ message })), 2000);
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

function makeNative() {
  return {
    headers: {},
    content: {
      messages: [],
      model: 'test-model',
    },
  };
}

function sparseArray() {
  const values = new Array(2);
  values[1] = 1;
  return values;
}

function unprintableError() {
  const error = new Error('sanitize request guardrail failed');
  Object.defineProperties(error, {
    message: {
      get() {
        throw new Error('message getter boom');
      },
    },
    toString: {
      value() {
        throw new Error('string conversion boom');
      },
    },
  });
  return error;
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
    llmCallEnd(
      handle,
      {
        choices: [
          {
            text: 'hello',
          },
        ],
      },
      null,
      null,
    );
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
    const handle = llmCall(
      'data_llm',
      native,
      null,
      null,
      {
        info: 'llm_test',
      },
      {
        version: '2.0',
      },
      null,
    );
    llmCallEnd(
      handle,
      {},
      {
        tokens: 100,
      },
      null,
    );
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
        await new Promise((r) => setTimeout(r, 10));
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
    const result = await llmCallExecute(
      'exec_llm',
      native,
      (n) => ({
        response: 'hello from llm',
      }),
      null,
      null,
      null,
      null,
      null,
    );
    assert.deepEqual(result, {
      response: 'hello from llm',
    });
  });

  it('treats implicit undefined llm results as null', async () => {
    const result = await llmCallExecute(
      'exec_llm_undefined',
      makeNative(),
      () => undefined,
      null,
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
        llmCallExecute('exec_llm_throw', makeNative(), () => {
          throw new Error('sync llm callback failed');
        }),
      /sync llm callback failed/,
    );

    const result = await llmCallExecute('exec_llm_after_throw', makeNative(), () => ({
      ok: true,
    }));
    assert.deepEqual(result, {
      ok: true,
    });
  });

  it('sync and async execute reject invalid JSON results without terminating Node', async () => {
    const cases = [
      ['sync_bigint', () => llmCallExecute('exec_llm_bigint', makeNative(), () => ({ value: 1n }))],
      ['async_bigint', () => llmCallExecuteAsync('exec_async_llm_bigint', makeNative(), async () => 1n)],
      ['sync_sparse_array', () => llmCallExecute('exec_llm_sparse_array', makeNative(), sparseArray)],
      [
        'async_sparse_array',
        () => llmCallExecuteAsync('exec_async_llm_sparse_array', makeNative(), async () => sparseArray()),
      ],
    ];

    for (const [kind, execute] of cases) {
      await assert.rejects(execute, /bigint|undefined|json/i);
      const result = await llmCallExecute(`exec_llm_after_${kind}`, makeNative(), () => ({
        ok: true,
      }));
      assert.deepEqual(result, {
        ok: true,
      });
    }
  });

  it('execute records OTEL status metadata on end events', async () => {
    const events = [];
    registerSubscriber('node_llm_status_metadata_sub', (e) => events.push(e));
    try {
      const result = await llmCallExecute(
        'exec_status_ok_llm',
        makeNative(),
        () => ({
          response: 'ok',
        }),
        null,
        null,
        null,
        {
          caller: 'node-llm',
        },
        null,
      );
      assert.deepEqual(result, {
        response: 'ok',
      });

      await assert.rejects(
        () =>
          llmCallExecuteAsync(
            'exec_status_error_llm',
            makeNative(),
            async () => {
              throw new Error('llm status failure');
            },
            null,
            null,
            null,
            {
              caller: 'node-llm-error',
            },
            null,
          ),
        /llm status failure/,
      );

      await waitForSubscriberCallbacks(
        () =>
          events.some((e) => e.name === 'exec_status_ok_llm' && e.scope_category === 'end') &&
          events.some((e) => e.name === 'exec_status_error_llm' && e.scope_category === 'end'),
      );
      const okEnd = events.find(
        (e) =>
          e.name === 'exec_status_ok_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'end',
      );
      const errorEnd = events.find(
        (e) =>
          e.name === 'exec_status_error_llm' &&
          e.kind === 'scope' &&
          e.category === 'llm' &&
          e.scope_category === 'end',
      );
      assert.ok(okEnd, 'expected successful llm end event');
      assert.equal(okEnd.metadata.caller, 'node-llm');
      assert.equal(okEnd.metadata['otel.status_code'], 'OK');
      assert.ok(errorEnd, 'expected failed llm end event');
      assert.equal(errorEnd.metadata.caller, 'node-llm-error');
      assert.equal(errorEnd.metadata['otel.status_code'], 'ERROR');
      assert.match(errorEnd.metadata['otel.status_description'], /llm status failure/);
    } finally {
      deregisterSubscriber('node_llm_status_metadata_sub');
    }
  });

  it('async execute awaits Promise-returning callbacks', async () => {
    const result = await llmCallExecuteAsync(
      'exec_async_llm',
      makeNative(),
      async (request) => ({
        response: request.content.model,
      }),
      null,
      LLM_ATTR_STATELESS,
      {
        data: true,
      },
      {
        meta: true,
      },
      'async-model',
    );
    assert.deepEqual(result, {
      response: 'test-model',
    });
  });

  it('async execute surfaces plain string rejections', async () => {
    await assert.rejects(
      () =>
        llmCallExecuteAsync(
          'exec_async_llm_reject',
          makeNative(),
          async () => rejectWith('string llm error'),
          null,
          null,
          null,
          null,
          null,
        ),
      /string llm error/,
    );
  });
});

// ===========================================================================
// LLM guardrails
// ===========================================================================

describe('LLM guardrails', () => {
  it('contextual sanitizers receive payload first and codec context second', async () => {
    const events = [];
    let requestContextChecked = false;
    let responseContextChecked = false;
    registerSubscriber('node_contextual_llm_sanitize_events', (event) => events.push(event));
    registerLlmSanitizeRequestGuardrail('node_contextual_llm_sanitize_request', 10, (request, context) => {
      assert.deepEqual(context.codec, { kind: 'none' });
      assert.equal(context.resolveCodec(), null);
      requestContextChecked = true;
      return {
        ...request,
        headers: { ...request.headers, 'X-Contextual-Sanitized': 'request' },
      };
    });
    registerLlmSanitizeResponseGuardrail('node_contextual_llm_sanitize_response', 10, (response, context) => {
      assert.deepEqual(context.codec, { kind: 'none' });
      assert.equal(context.resolveCodec(), null);
      responseContextChecked = true;
      return { ...response, contextualSanitized: true };
    });

    try {
      const result = await llmCallExecute('contextual_sanitize_llm', makeNative(), () => ({ ok: true }));
      assert.deepEqual(result, { ok: true });
      assert.equal(requestContextChecked, true);
      assert.equal(responseContextChecked, true);
      await waitForSubscriberCallbacks(
        () =>
          events.some((event) => event.name === 'contextual_sanitize_llm' && event.scope_category === 'start') &&
          events.some((event) => event.name === 'contextual_sanitize_llm' && event.scope_category === 'end'),
      );
      const start = events.find(
        (event) => event.name === 'contextual_sanitize_llm' && event.scope_category === 'start',
      );
      const end = events.find((event) => event.name === 'contextual_sanitize_llm' && event.scope_category === 'end');
      assert.equal(start.data.headers['X-Contextual-Sanitized'], 'request');
      assert.equal(end.data.contextualSanitized, true);
    } finally {
      deregisterLlmSanitizeRequestGuardrail('node_contextual_llm_sanitize_request');
      deregisterLlmSanitizeResponseGuardrail('node_contextual_llm_sanitize_response');
      deregisterSubscriber('node_contextual_llm_sanitize_events');
    }
  });

  it('rejects incomplete request codec callback pairs', () => {
    const decode = (request) => request;
    const encode = ({ annotated }) => annotated;

    assert.throws(
      () => llmCallExecute('partial_codec_execute', makeNative(), () => ({}), null, null, null, null, null, decode),
      /codecDecode and codecEncode must be provided together/,
    );
    assert.throws(
      () =>
        llmCallExecuteAsync(
          'partial_codec_execute_async',
          makeNative(),
          async () => ({}),
          null,
          null,
          null,
          null,
          null,
          null,
          encode,
        ),
      /codecDecode and codecEncode must be provided together/,
    );
    assert.throws(
      () =>
        llmStreamCallExecute(
          'partial_codec_stream',
          makeNative(),
          () => {},
          null,
          null,
          null,
          null,
          null,
          null,
          null,
          decode,
        ),
      /codecDecode and codecEncode must be provided together/,
    );
  });

  it('resolved sanitizer codecs expose directional operations', async () => {
    const codec = new lib.OpenAIChatCodec();
    let requestDecoded = false;
    let responseDecoded = false;
    registerLlmSanitizeRequestGuardrail('node_resolved_llm_request_codec', 10, (request, context) => {
      assert.deepEqual(context.codec, { kind: 'opaque' });
      const resolved = context.resolveCodec();
      assert.notEqual(resolved, null);
      const annotated = resolved.decode(request);
      requestDecoded = annotated.model === 'test-model';
      return resolved.encode(annotated, request);
    });
    registerLlmSanitizeResponseGuardrail('node_resolved_llm_response_codec', 10, (response, context) => {
      assert.deepEqual(context.codec, { kind: 'opaque' });
      const resolved = context.resolveCodec();
      assert.notEqual(resolved, null);
      const annotated = resolved.decodeResponse(response);
      responseDecoded = annotated.model === 'test-model';
      return response;
    });

    try {
      const response = {
        id: 'chatcmpl-test',
        model: 'test-model',
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'ok' },
            finish_reason: 'stop',
          },
        ],
      };
      const result = await llmCallExecute(
        'resolved_sanitizer_codec_llm',
        makeNative(),
        () => response,
        null,
        null,
        null,
        null,
        null,
        codec.decode.bind(codec),
        ({ annotated, original }) => codec.encode(annotated, original),
        codec.decodeResponse.bind(codec),
      );
      assert.deepEqual(result, response);
      assert.equal(requestDecoded, true);
      assert.equal(responseDecoded, true);
    } finally {
      deregisterLlmSanitizeRequestGuardrail('node_resolved_llm_request_codec');
      deregisterLlmSanitizeResponseGuardrail('node_resolved_llm_response_codec');
    }
  });

  it('streaming sanitizers resolve codecs only during callbacks and can omit the final payload', async () => {
    const codec = new lib.OpenAIChatCodec();
    const events = [];
    let requestCodecUsed = false;
    let responseCodecUsed = false;
    registerSubscriber('node_streaming_codec_sanitize_events', (event) => events.push(event));
    registerLlmSanitizeRequestGuardrail('node_streaming_request_codec', 10, (request, context) => {
      const resolved = context.resolveCodec();
      assert.notEqual(resolved, null);
      const annotated = resolved.decode(request);
      requestCodecUsed = true;
      return resolved.encode(annotated, request);
    });
    registerLlmSanitizeResponseGuardrail('node_streaming_response_codec', 10, (response, context) => {
      const resolved = context.resolveCodec();
      assert.notEqual(resolved, null);
      resolved.decodeResponse(response);
      responseCodecUsed = true;
      return null;
    });

    const response = {
      id: 'chatcmpl-stream',
      model: 'test-model',
      choices: [{ index: 0, message: { role: 'assistant', content: 'secret' }, finish_reason: 'stop' }],
    };
    try {
      const stream = await llmStreamCallExecute(
        'streaming_resolved_sanitizer_codec',
        makeNative(),
        (wrapper) => {
          lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, { delta: 'secret' });
          lib.endStream(wrapper.__nemo_relay_stream_id);
        },
        null,
        () => response,
        null,
        null,
        null,
        null,
        null,
        codec.decode.bind(codec),
        ({ annotated, original }) => codec.encode(annotated, original),
        codec.decodeResponse.bind(codec),
      );
      assert.deepEqual(await stream.next(), { delta: 'secret' });
      assert.equal(await stream.next(), null);
      await flushSubscriberCallbacks();
      assert.equal(requestCodecUsed, true);
      assert.equal(responseCodecUsed, true);
      const end = events.find(
        (event) => event.name === 'streaming_resolved_sanitizer_codec' && event.scope_category === 'end',
      );
      assert.equal(end.data, null);
      assert.equal(end.category_profile.annotated_response, undefined);
    } finally {
      deregisterLlmSanitizeRequestGuardrail('node_streaming_request_codec');
      deregisterLlmSanitizeResponseGuardrail('node_streaming_response_codec');
      deregisterSubscriber('node_streaming_codec_sanitize_events');
    }
  });

  it('stream response sanitizers preserve the invocation scope across await', async () => {
    const originalStack = lib.currentScopeStack();
    const invocationStack = lib.createScopeStack();
    const unrelatedStack = lib.createScopeStack();
    const observed = [];
    let invocationScope;

    registerLlmSanitizeResponseGuardrail('node_stream_response_scope', 10, async (response) => {
      observed.push(lib.getHandle().uuid);
      await new Promise((resolve) => setImmediate(resolve));
      observed.push(lib.getHandle().uuid);
      return response;
    });
    try {
      const execution = lib.withScopeStack(invocationStack, () => {
        invocationScope = lib.pushScope('stream-response-scope', lib.ScopeType.Agent);
        return llmStreamCallExecute(
          'stream_response_scope',
          makeNative(),
          (wrapper) => {
            lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, { token: 'done' });
            lib.endStream(wrapper.__nemo_relay_stream_id);
          },
          null,
          () => ({ done: true }),
          null,
          null,
          null,
          null,
          null,
          null,
          null,
          null,
        );
      });
      lib.setThreadScopeStack(unrelatedStack);
      const stream = await execution;
      assert.deepEqual(await stream.next(), { token: 'done' });
      assert.equal(await stream.next(), null);
      assert.deepEqual(observed, [invocationScope.uuid, invocationScope.uuid]);
    } finally {
      lib.withScopeStack(invocationStack, () => lib.popScope(invocationScope));
      lib.setThreadScopeStack(originalStack);
      deregisterLlmSanitizeResponseGuardrail('node_stream_response_scope');
    }
  });

  it('stream response sanitizers can flush subscribers without deadlocking', async () => {
    let responseFlushed = false;
    registerSubscriber('node_stream_flush_subscriber', () => {});
    registerLlmSanitizeResponseGuardrail('node_stream_flush_response', 10, async (response) => {
      await flushSubscribers();
      responseFlushed = true;
      return response;
    });
    try {
      const stream = await llmStreamCallExecute(
        'node_stream_flush',
        makeNative(),
        (wrapper) => {
          lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, { delta: 'ok' });
          lib.endStream(wrapper.__nemo_relay_stream_id);
        },
        null,
        () => ({ response: 'ok' }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(await stream.next(), { delta: 'ok' });
      assert.equal(
        await assertCompletesWithin(stream.next(), 'stream finalization deadlocked inside an async sanitizer'),
        null,
      );
      await flushSubscribers();
    } finally {
      deregisterLlmSanitizeResponseGuardrail('node_stream_flush_response');
      deregisterSubscriber('node_stream_flush_subscriber');
    }
    assert.equal(responseFlushed, true);
  });

  it('releases custom stream codec references safely after early garbage collection', () => {
    const modulePath = path.join(nodeDir, 'index.js');
    const script = `
      import { createRequire } from 'node:module';
      const require = createRequire(${JSON.stringify(path.join(nodeDir, 'package.json'))});
      const lib = require(${JSON.stringify(modulePath)});
      const codec = new lib.OpenAIChatCodec();
      let ended = false;
      lib.registerSubscriber('early_drop_custom_codec_stream_events', (event) => {
        if (
          event.name === 'early_drop_custom_codec_stream' &&
          event.scope_category === 'end'
        ) {
          ended = true;
        }
      });
      let stream = await lib.llmStreamCallExecute(
        'early_drop_custom_codec_stream',
        {
          headers: {},
          content: { model: 'test-model', messages: [] },
        },
        (wrapper) => {
          setTimeout(() => {
            lib.endStream(wrapper.__nemo_relay_stream_id);
          }, 25);
        },
        null,
        () => ({ model: 'test-model', choices: [] }),
        null,
        null,
        null,
        null,
        null,
        codec.decode.bind(codec),
        ({ annotated, original }) => codec.encode(annotated, original),
        codec.decodeResponse.bind(codec),
      );
      const weak = new WeakRef(stream);
      stream = null;
      let collected = false;
      for (let index = 0; index < 100; index += 1) {
        global.gc();
        await new Promise((resolve) => setImmediate(resolve));
        if (weak.deref() === undefined) {
          collected = true;
          break;
        }
        await new Promise((resolve) => setImmediate(resolve));
      }
      if (!collected) {
        throw new Error('unfinished custom-codec stream was not garbage collected');
      }
      for (let index = 0; index < 100 && !ended; index += 1) {
        global.gc();
        await new Promise((resolve) => setImmediate(resolve));
      }
      if (!ended) {
        throw new Error('early-dropped custom-codec stream did not finish cleanup');
      }
      lib.deregisterSubscriber('early_drop_custom_codec_stream_events');
      await new Promise((resolve) => setImmediate(resolve));
    `;
    execFileSync(process.execPath, ['--expose-gc', '--input-type=module', '--eval', script], {
      stdio: 'inherit',
      timeout: 30_000,
    });
  });

  it('sanitize request guardrail', () => {
    registerLlmSanitizeRequestGuardrail('node_llm_san_req', 10, (request) => {
      request.extra = 'sanitized';
      return request;
    });
    deregisterLlmSanitizeRequestGuardrail('node_llm_san_req');
  });

  it('manual async sanitizers can flush subscribers without deadlocking', async () => {
    let requestFlushed = false;
    let responseFlushed = false;
    registerSubscriber('node_manual_flush_subscriber', () => {});
    registerLlmSanitizeRequestGuardrail('node_manual_flush_request', 10, async (request) => {
      await flushSubscribers();
      requestFlushed = true;
      return request;
    });
    registerLlmSanitizeResponseGuardrail('node_manual_flush_response', 10, async (response) => {
      await flushSubscribers();
      responseFlushed = true;
      return response;
    });
    try {
      const handle = llmCall('node_manual_flush', makeNative());
      llmCallEnd(handle, { response: 'ok' });
      await assertCompletesWithin(flushSubscribers(), 'flushSubscribers deadlocked inside an async sanitizer');
    } finally {
      deregisterLlmSanitizeRequestGuardrail('node_manual_flush_request');
      deregisterLlmSanitizeResponseGuardrail('node_manual_flush_response');
      deregisterSubscriber('node_manual_flush_subscriber');
    }
    assert.equal(requestFlushed, true);
    assert.equal(responseFlushed, true);
  });

  it('sanitize request guardrail rewrites start event payload', async () => {
    const events = [];
    registerSubscriber('node_llm_san_req_evt', (e) => events.push(e));
    registerLlmSanitizeRequestGuardrail('node_llm_san_req_evt_guard', 10, (request) => {
      request.headers = {
        ...request.headers,
        'X-Sanitized': 'yes',
      };
      return request;
    });

    try {
      const result = await llmCallExecute(
        'san_req_evt_llm',
        makeNative(),
        (request) => ({
          model: request.content.model,
        }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, {
        model: 'test-model',
      });
      const deadline = Date.now() + 2000;
      while (
        !events.some(
          (e) =>
            e.name === 'san_req_evt_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'start',
        ) &&
        Date.now() < deadline
      ) {
        await new Promise((r) => setTimeout(r, 10));
      }
      const start = events.find(
        (e) =>
          e.name === 'san_req_evt_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'start',
      );
      assert.deepEqual(start.data, {
        headers: {
          'X-Sanitized': 'yes',
        },
        content: {
          messages: [],
          model: 'test-model',
        },
      });
    } finally {
      deregisterLlmSanitizeRequestGuardrail('node_llm_san_req_evt_guard');
      deregisterSubscriber('node_llm_san_req_evt');
    }
  });

  it('sanitize request guardrail can omit the observability payload', async () => {
    registerLlmSanitizeRequestGuardrail('node_llm_san_req_bad', 10, () => null);
    try {
      const result = await llmCallExecute(
        'san_req_bad_llm',
        makeNative(),
        (request) => ({
          model: request.content.model,
          headers: request.headers,
        }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, {
        model: 'test-model',
        headers: {},
      });
    } finally {
      deregisterLlmSanitizeRequestGuardrail('node_llm_san_req_bad');
    }
  });

  it('sanitize request guardrail failures preserve the payload and remain usable', async () => {
    const events = [];
    clearLastCallbackError();
    registerSubscriber('node_llm_san_req_throw_sub', (event) => events.push(event));
    registerLlmSanitizeRequestGuardrail('node_llm_san_req_throw', 10, () => {
      throw unprintableError();
    });
    try {
      const request = makeNative();
      await llmCallExecute('llm_san_req_throw', request, () => ({ ok: true }), null, null, null, null, null);
      await waitForSubscriberCallbacks(() =>
        events.some(
          (event) =>
            event.name === 'llm_san_req_throw' &&
            event.kind === 'scope' &&
            event.category === 'llm' &&
            event.scope_category === 'start',
        ),
      );
      const start = events.find(
        (event) =>
          event.name === 'llm_san_req_throw' &&
          event.kind === 'scope' &&
          event.category === 'llm' &&
          event.scope_category === 'start',
      );
      assert.deepEqual(start.data, { headers: request.headers, content: request.content });
      assert.equal(getLastCallbackError(), 'internal error: unknown error');

      deregisterLlmSanitizeRequestGuardrail('node_llm_san_req_throw');
      const result = await llmCallExecute(
        'llm_after_san_req_throw',
        makeNative(),
        () => ({ ok: true }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { ok: true });
    } finally {
      deregisterLlmSanitizeRequestGuardrail('node_llm_san_req_throw');
      deregisterSubscriber('node_llm_san_req_throw_sub');
      clearLastCallbackError();
    }
  });

  it('conditional guardrail rejects non-string return values', async () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond_non_string', 10, () => ({
      blocked: true,
    }));
    try {
      await assert.rejects(
        () =>
          llmCallExecute(
            'llm_cond_non_string',
            makeNative(),
            () => ({
              ok: true,
            }),
            null,
            null,
            null,
            null,
            null,
          ),
        /expected string or null/i,
      );
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_llm_cond_non_string');
    }
  });

  it('sanitize response guardrail', () => {
    registerLlmSanitizeResponseGuardrail('node_llm_san_resp', 10, (response) => {
      response.sanitized = true;
      return response;
    });
    deregisterLlmSanitizeResponseGuardrail('node_llm_san_resp');
  });

  it('sanitize response guardrail rewrites end event payload', async () => {
    const events = [];
    registerSubscriber('node_llm_san_resp_evt', (e) => events.push(e));
    registerLlmSanitizeResponseGuardrail('node_llm_san_resp_evt_guard', 10, (response) => {
      response.sanitized = true;
      return response;
    });

    try {
      const result = await llmCallExecute(
        'san_resp_evt_llm',
        makeNative(),
        () => ({
          ok: true,
        }),
        null,
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
            e.name === 'san_resp_evt_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'end',
        ) &&
        Date.now() < deadline
      ) {
        await new Promise((r) => setTimeout(r, 10));
      }
      const end = events.find(
        (e) =>
          e.name === 'san_resp_evt_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'end',
      );
      assert.deepEqual(end.data, {
        ok: true,
        sanitized: true,
      });
    } finally {
      deregisterLlmSanitizeResponseGuardrail('node_llm_san_resp_evt_guard');
      deregisterSubscriber('node_llm_san_resp_evt');
    }
  });

  it('sanitize response guardrail failures preserve the payload and remain usable', async () => {
    const events = [];
    clearLastCallbackError();
    registerSubscriber('node_llm_san_resp_throw_sub', (event) => events.push(event));
    registerLlmSanitizeResponseGuardrail('node_llm_san_resp_throw', 10, () => {
      throw new Error('response sanitizer boom');
    });
    try {
      const response = { ok: true };
      await llmCallExecute('llm_san_resp_throw', makeNative(), () => response, null, null, null, null, null);
      await waitForSubscriberCallbacks(() =>
        events.some(
          (event) =>
            event.name === 'llm_san_resp_throw' &&
            event.kind === 'scope' &&
            event.category === 'llm' &&
            event.scope_category === 'end',
        ),
      );
      const end = events.find(
        (event) =>
          event.name === 'llm_san_resp_throw' &&
          event.kind === 'scope' &&
          event.category === 'llm' &&
          event.scope_category === 'end',
      );
      assert.deepEqual(end.data, response);
      assert.match(getLastCallbackError() ?? '', /response sanitizer boom/i);

      deregisterLlmSanitizeResponseGuardrail('node_llm_san_resp_throw');
      const result = await llmCallExecute(
        'llm_after_san_resp_throw',
        makeNative(),
        () => ({ ok: true }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { ok: true });
    } finally {
      deregisterLlmSanitizeResponseGuardrail('node_llm_san_resp_throw');
      deregisterSubscriber('node_llm_san_resp_throw_sub');
      clearLastCallbackError();
    }
  });

  it('conditional guardrail (allow)', () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond', 10, (request) => null);
    deregisterLlmConditionalExecutionGuardrail('node_llm_cond');
  });

  it('conditional guardrail awaits a Promise result', async () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond_promise', 10, async () => {
      await new Promise((resolve) => setImmediate(resolve));
      return null;
    });
    try {
      const result = await llmCallExecute(
        'llm_cond_promise',
        makeNative(),
        () => ({ ok: true }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { ok: true });
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_llm_cond_promise');
    }
  });

  it('conditional guardrail treats implicit undefined as allow', async () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond_undefined', 10, () => undefined);
    try {
      const result = await llmCallExecute(
        'llm_cond_undefined',
        makeNative(),
        () => ({
          ok: true,
        }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, {
        ok: true,
      });
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_llm_cond_undefined');
    }
  });

  it('conditional guardrail throws a catchable error without terminating Node', async () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond_throw', 10, () => {
      throw new Error('llm guardrail boom');
    });
    try {
      await assert.rejects(
        () =>
          llmCallExecute('llm_cond_throw', makeNative(), () => ({ should_not: 'run' }), null, null, null, null, null),
        /llm guardrail boom/i,
      );
      deregisterLlmConditionalExecutionGuardrail('node_llm_cond_throw');

      const result = await llmCallExecute(
        'llm_after_guardrail_throw',
        makeNative(),
        () => ({ ok: true }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { ok: true });
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_llm_cond_throw');
    }
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
    registerLlmRequestIntercept('node_llm_req_int', 10, false, ({ name, request, annotated }) => {
      request.intercepted = true;
      return {
        request,
        annotated,
      };
    });
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
    registerLlmRequestIntercept('node_llm_break', 10, true, ({ name, request, annotated }) => ({
      request,
      annotated,
    }));
    deregisterLlmRequestIntercept('node_llm_break');
  });

  it('duplicate intercept fails', () => {
    registerLlmRequestIntercept('node_llm_dup_int', 10, false, ({ request, annotated }) => ({
      request,
      annotated,
    }));
    assert.throws(() =>
      registerLlmRequestIntercept('node_llm_dup_int', 20, false, ({ request, annotated }) => ({
        request,
        annotated,
      })),
    );
    deregisterLlmRequestIntercept('node_llm_dup_int');
  });

  it('request intercept modifies request', async () => {
    registerLlmRequestIntercept('node_llm_req_mod', 10, false, ({ request, annotated }) => {
      request.content.intercepted = true;
      return {
        request,
        annotated,
      };
    });
    const native = makeNative();
    const result = await llmCallExecute(
      'mod_llm',
      native,
      (n) => ({
        saw_intercepted: n.content.intercepted || false,
      }),
      null,
      null,
      null,
      null,
      null,
    );
    assert.equal(result.saw_intercepted, true);
    deregisterLlmRequestIntercept('node_llm_req_mod');
  });

  it('request intercept awaits a Promise result', async () => {
    registerLlmRequestIntercept('node_llm_req_promise', 10, false, async ({ request, annotated }) => {
      await new Promise((resolve) => setImmediate(resolve));
      return { request: { ...request, content: { ...request.content, promised: true } }, annotated };
    });
    try {
      const result = await llmCallExecute(
        'llm_req_promise',
        makeNative(),
        (request) => ({ promised: request.content.promised }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { promised: true });
    } finally {
      deregisterLlmRequestIntercept('node_llm_req_promise');
    }
  });

  it('request intercept throws a catchable error without terminating Node', async () => {
    registerLlmRequestIntercept('node_llm_req_throw', 10, false, () => {
      throw new Error('llm request intercept boom');
    });
    try {
      await assert.rejects(
        () =>
          llmCallExecute('llm_req_throw', makeNative(), () => ({ should_not: 'run' }), null, null, null, null, null),
        /llm request intercept boom/i,
      );
      deregisterLlmRequestIntercept('node_llm_req_throw');
      const result = await llmCallExecute(
        'llm_after_request_intercept_throw',
        makeNative(),
        () => ({ ok: true }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { ok: true });
    } finally {
      deregisterLlmRequestIntercept('node_llm_req_throw');
    }
  });

  it('request intercept rejects malformed return values', async () => {
    registerLlmRequestIntercept('node_llm_req_bad', 10, false, () => null);
    try {
      await assert.rejects(
        () =>
          llmCallExecute(
            'bad_req_llm',
            makeNative(),
            (n) => ({
              model: n.content.model,
            }),
            null,
            null,
            null,
            null,
            null,
          ),
        /invalid JS LLM request intercept outcome/i,
      );
    } finally {
      deregisterLlmRequestIntercept('node_llm_req_bad');
    }
  });

  it('execution intercept composes with next', async () => {
    registerLlmExecutionIntercept('node_llm_exec_repl', 10, async (native, next) => {
      native.content.intercepted = true;
      const result = await next(native);
      return {
        ...result,
        wrapped: true,
      };
    });
    const native = makeNative();
    const result = await llmCallExecute(
      'repl_llm',
      native,
      (n) => ({
        original: !n.content.intercepted,
      }),
      null,
      null,
      null,
      null,
      null,
    );
    assert.equal(result.original, false);
    assert.equal(result.wrapped, true);
    deregisterLlmExecutionIntercept('node_llm_exec_repl');
  });

  it('execution intercept rejects a detached next call after settlement', async () => {
    let releaseLateNext;
    const lateGate = new Promise((resolve) => {
      releaseLateNext = resolve;
    });
    let lateNext;
    let providerCalls = 0;
    registerLlmExecutionIntercept('node_llm_exec_late_next', 10, async (native, next) => {
      lateNext = lateGate.then(() => next(native));
      return { source: 'intercept' };
    });
    try {
      const result = await llmCallExecute('late_next_llm', makeNative(), () => {
        providerCalls += 1;
        return { source: 'provider' };
      });
      assert.deepEqual(result, { source: 'intercept' });
      releaseLateNext();
      await assert.rejects(lateNext, /execution continuation is no longer active/i);
      assert.equal(providerCalls, 0);
    } finally {
      releaseLateNext?.();
      await lateNext?.catch(() => {});
      deregisterLlmExecutionIntercept('node_llm_exec_late_next');
    }
  });

  it('execution intercept aborts an already-started async provider after settlement', async () => {
    let releaseProvider;
    const providerGate = new Promise((resolve) => {
      releaseProvider = resolve;
    });
    let providerStarted;
    const started = new Promise((resolve) => {
      providerStarted = resolve;
    });
    let providerAborted;
    const aborted = new Promise((resolve) => {
      providerAborted = resolve;
    });
    let downstream;
    let providerSideEffects = 0;
    registerLlmExecutionIntercept('node_llm_exec_abort_started_provider', 10, async (native, next) => {
      downstream = next(native);
      void downstream.catch(() => {});
      await started;
      return { source: 'intercept' };
    });
    try {
      const result = await llmCallExecuteAsync(
        'abort_started_llm',
        makeNative(),
        async (_request, signal) => {
          providerStarted();
          await Promise.race([
            providerGate,
            new Promise((_, reject) => {
              signal.addEventListener(
                'abort',
                () => {
                  providerAborted();
                  reject(new Error('provider aborted'));
                },
                { once: true },
              );
            }),
          ]);
          providerSideEffects += 1;
          return { source: 'provider' };
        },
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, { source: 'intercept' });
      await assert.rejects(downstream, /execution continuation is no longer active/i);
      await assertCompletesWithin(aborted, 'provider did not receive cancellation after continuation revocation');
      releaseProvider();
      assert.equal(providerSideEffects, 0);
    } finally {
      releaseProvider?.();
      await downstream?.catch(() => {});
      deregisterLlmExecutionIntercept('node_llm_exec_abort_started_provider');
    }
  });

  it('execution intercept rejects invalid next request payloads', async () => {
    registerLlmExecutionIntercept('node_llm_exec_invalid_next', 10, async (_native, next) => {
      return next({
        headers: 1,
        content: {
          model: 'broken',
        },
      });
    });
    await assert.rejects(
      () =>
        llmCallExecute(
          'invalid_next_llm',
          makeNative(),
          () => ({
            ok: true,
          }),
          null,
          null,
          null,
          null,
          null,
        ),
      /invalid LlmRequest from JS next/i,
    );
    deregisterLlmExecutionIntercept('node_llm_exec_invalid_next');
  });

  it('execution intercept preserves primitive rejection values', async () => {
    registerLlmExecutionIntercept('node_llm_exec_unknown_err', 10, async () => {
      return rejectWith(42);
    });
    try {
      await assert.rejects(
        () =>
          llmCallExecute(
            'unknown_err_llm',
            makeNative(),
            () => ({
              ok: true,
            }),
            null,
            null,
            null,
            null,
            null,
          ),
        /internal error: 42/i,
      );
    } finally {
      deregisterLlmExecutionIntercept('node_llm_exec_unknown_err');
    }
  });

  it('async execute preserves primitive rejection values', async () => {
    await assert.rejects(
      () =>
        llmCallExecuteAsync(
          'primitive_reject_llm',
          makeNative(),
          async () => rejectWith(42),
          null,
          null,
          null,
          null,
          null,
        ),
      /internal error: 42/i,
    );
  });

  it('execution intercept rejects non-JSON next arguments without aborting Node', async () => {
    registerLlmExecutionIntercept('node_llm_exec_bigint_next', 10, async (_native, next) => next(1n));
    try {
      await assert.rejects(
        () => llmCallExecute('bigint_next_llm', makeNative(), () => ({ ok: true })),
        /unsupported bigint value.*JSON/i,
      );
    } finally {
      deregisterLlmExecutionIntercept('node_llm_exec_bigint_next');
    }
  });

  it('stream execution intercept composes with next', async () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_exec_repl', 10, async (native, next) => {
      native.content.intercepted = true;
      const chunks = await next(native);
      return [
        ...chunks,
        {
          wrapped: native.content.intercepted,
        },
      ];
    });

    const native = makeNative();
    const seen = [];
    const stream = await llmStreamCallExecute(
      'stream_llm',
      native,
      (wrapper) => {
        lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, {
          chunk: wrapper.__nemo_relay_native.content.intercepted,
        });
        lib.endStream(wrapper.__nemo_relay_stream_id);
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

    assert.deepEqual(seen, [
      {
        chunk: true,
      },
      {
        wrapped: true,
      },
    ]);
    deregisterLlmStreamExecutionIntercept('node_llm_stream_exec_repl');
  });

  it('stream execution intercept rejects a detached next call after its output closes', async () => {
    let releaseLateNext;
    const lateGate = new Promise((resolve) => {
      releaseLateNext = resolve;
    });
    let lateNext;
    let providerCalls = 0;
    registerLlmStreamExecutionIntercept('node_llm_stream_late_next', 10, async (native, next) => {
      lateNext = lateGate.then(() => next(native));
      return [{ source: 'intercept' }];
    });
    try {
      const stream = await llmStreamCallExecute('late_next_stream_llm', makeNative(), () => {
        providerCalls += 1;
      });
      assert.deepEqual(await stream.next(), { source: 'intercept' });
      assert.equal(await stream.next(), null);
      releaseLateNext();
      await assert.rejects(lateNext, /execution continuation is no longer active/i);
      assert.equal(providerCalls, 0);
    } finally {
      releaseLateNext?.();
      await lateNext?.catch(() => {});
      deregisterLlmStreamExecutionIntercept('node_llm_stream_late_next');
    }
  });

  it('stream execution next retains the captured scope after the callback resolves', async () => {
    const invocationStack = lib.createScopeStack();
    const invocationScope = lib.withScopeStack(invocationStack, () => lib.getHandle().uuid);
    let retainedNext;
    registerLlmStreamExecutionIntercept('node_llm_stream_retained_next_scope', 10, async (native, next) => {
      retainedNext = () => next(native);
      // Keep the Rust-to-Node forwarding channel backpressured so the
      // interceptor output stream, and therefore its continuation lease,
      // remains active after this callback Promise resolves.
      return Array.from({ length: 64 }, (_, index) => ({ source: 'intercept', index }));
    });
    try {
      const stream = await lib.withScopeStack(invocationStack, () =>
        llmStreamCallExecute('retained_next_scope_stream_llm', makeNative(), (wrapper) => {
          lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, {
            scope: lib.getHandle().uuid,
          });
          lib.endStream(wrapper.__nemo_relay_stream_id);
        }),
      );

      assert.deepEqual(await retainedNext(), [{ scope: invocationScope }]);
      assert.deepEqual(await stream.next(), { source: 'intercept', index: 0 });
      await stream.close();
    } finally {
      deregisterLlmStreamExecutionIntercept('node_llm_stream_retained_next_scope');
    }
  });

  it('default lazy stream preserves the managed parent context', async () => {
    const events = [];
    registerSubscriber('node_default_lazy_stream_context', (event) => events.push(event));
    try {
      const stream = await llmStreamCallExecute('node_default_lazy_stream_context', makeNative(), (wrapper) => {
        setImmediate(() => {
          lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, {
            parentUuid: lib.capturePropagationContext().parentUuid,
          });
          lib.endStream(wrapper.__nemo_relay_stream_id);
        });
      });
      const providerContext = await stream.next();
      assert.equal(await stream.next(), null);
      await flushSubscribers();
      const start = events.find(
        (event) =>
          event.name === 'node_default_lazy_stream_context' &&
          event.scope_category === 'start' &&
          event.category === 'llm',
      );
      assert.ok(start, 'expected managed LLM start event');
      assert.deepEqual(providerContext, { parentUuid: start.uuid });
    } finally {
      deregisterSubscriber('node_default_lazy_stream_context');
    }
  });

  it('stream execution next honors concurrent per-call scope-stack replacements', async () => {
    const firstStack = lib.createScopeStack();
    const secondStack = lib.createScopeStack();
    const firstScope = lib.withScopeStack(firstStack, () => lib.getHandle().uuid);
    const secondScope = lib.withScopeStack(secondStack, () => lib.getHandle().uuid);
    registerLlmStreamExecutionIntercept('node_llm_stream_next_scope_replacements', 10, async (native, next) => {
      const [first, second] = await Promise.all([
        lib.withScopeStack(firstStack, () =>
          next({
            ...native,
            content: { ...native.content, branch: 'first' },
          }),
        ),
        lib.withScopeStack(secondStack, () =>
          next({
            ...native,
            content: { ...native.content, branch: 'second' },
          }),
        ),
      ]);
      return [...first, ...second];
    });
    try {
      const stream = await llmStreamCallExecute('scoped_next_stream_llm', makeNative(), (wrapper) => {
        lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, {
          branch: wrapper.__nemo_relay_native.content.branch,
          scope: lib.getHandle().uuid,
        });
        lib.endStream(wrapper.__nemo_relay_stream_id);
      });
      const chunks = [];
      for (;;) {
        const chunk = await stream.next();
        if (chunk === null) {
          break;
        }
        chunks.push(chunk);
      }
      assert.deepEqual(chunks, [
        { branch: 'first', scope: firstScope },
        { branch: 'second', scope: secondScope },
      ]);
    } finally {
      deregisterLlmStreamExecutionIntercept('node_llm_stream_next_scope_replacements');
    }
  });

  it('stream execution intercept rejects non-JSON next arguments without aborting Node', async () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_bigint_next', 10, async (_native, next) => next(1n));
    try {
      await assert.rejects(
        () =>
          llmStreamCallExecute('bigint_next_stream_llm', makeNative(), (wrapper) => {
            lib.endStream(wrapper.__nemo_relay_stream_id);
          }),
        /unsupported bigint value.*JSON/i,
      );
    } finally {
      deregisterLlmStreamExecutionIntercept('node_llm_stream_bigint_next');
    }
  });

  it('snapshotted stream execution intercept survives deregistration', async () => {
    let blockerEntered;
    const entered = new Promise((resolve) => {
      blockerEntered = resolve;
    });
    let releaseBlocker;
    const release = new Promise((resolve) => {
      releaseBlocker = resolve;
    });

    registerLlmStreamExecutionIntercept('node_llm_stream_snapshot_target', 100, async (request, next) => [
      ...(await next(request)),
      { snapshotted: true },
    ]);
    registerLlmStreamExecutionIntercept('node_llm_stream_snapshot_blocker', -100, async (request, next) => {
      blockerEntered();
      await release;
      return next(request);
    });

    try {
      const streamPromise = llmStreamCallExecute(
        'stream_snapshot_llm',
        makeNative(),
        (wrapper) => {
          lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, { downstream: true });
          lib.endStream(wrapper.__nemo_relay_stream_id);
        },
        null,
        null,
        null,
        null,
        null,
        null,
        null,
      );
      await entered;
      assert.equal(deregisterLlmStreamExecutionIntercept('node_llm_stream_snapshot_target'), true);
      releaseBlocker();
      const stream = await streamPromise;
      const chunks = [];
      for (;;) {
        const chunk = await stream.next();
        if (chunk === null) {
          break;
        }
        chunks.push(chunk);
      }
      assert.deepEqual(chunks, [{ downstream: true }, { snapshotted: true }]);
    } finally {
      releaseBlocker();
      deregisterLlmStreamExecutionIntercept('node_llm_stream_snapshot_blocker');
      deregisterLlmStreamExecutionIntercept('node_llm_stream_snapshot_target');
    }
  });

  it('completed deregistered stream intercepts do not keep Node alive', () => {
    const modulePath = JSON.stringify(path.join(nodeDir, 'index.js'));
    const script = `
      import { createRequire } from 'node:module';
      const require = createRequire(import.meta.url);
      const lib = require(${modulePath});
      const request = {
        headers: {},
        content: { messages: [], model: 'test-model' },
      };
      lib.registerLlmStreamExecutionIntercept('process-exit-stream', 10, async (value, next) => next(value));
      const stream = await lib.llmStreamCallExecute(
        'process-exit-llm',
        request,
        (wrapper) => {
          lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, { done: true });
          lib.endStream(wrapper.__nemo_relay_stream_id);
        },
        null,
        null,
        null,
        null,
        null,
        null,
        null,
      );
      while (await stream.next() !== null) {}
      await stream.close();
      if (!lib.deregisterLlmStreamExecutionIntercept('process-exit-stream')) {
        throw new Error('stream intercept was not deregistered');
      }
    `;

    execFileSync(process.execPath, ['--input-type=module', '--eval', script], {
      stdio: 'inherit',
      timeout: 5_000,
    });
  });

  it('stream execution intercept can return a single scalar chunk', async () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_scalar', 10, async () => ({
      scalar: true,
    }));

    const seen = [];
    const stream = await llmStreamCallExecute(
      'stream_scalar_llm',
      makeNative(),
      () => {
        throw new Error('downstream stream should not be called');
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

    assert.deepEqual(seen, [
      {
        scalar: true,
      },
    ]);
    deregisterLlmStreamExecutionIntercept('node_llm_stream_scalar');
  });

  it('stream execution intercept rejects invalid next request payloads', async () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_invalid_next', 10, async (_native, next) => {
      return next({
        headers: 1,
        content: {
          model: 'broken',
        },
      });
    });

    await assert.rejects(
      () =>
        llmStreamCallExecute(
          'stream_invalid_next_llm',
          makeNative(),
          (wrapper) => {
            lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, {
              chunk: true,
            });
            lib.endStream(wrapper.__nemo_relay_stream_id);
          },
          null,
          null,
          null,
          null,
          null,
          null,
          null,
        ),
      /invalid LlmRequest from JS next/i,
    );

    deregisterLlmStreamExecutionIntercept('node_llm_stream_invalid_next');
  });

  it('standalone request intercepts helper applies intercept chain', async () => {
    const contributionFixture = JSON.parse(
      readFileSync(
        new URL('../../types/tests/fixtures/llm_optimization_contribution_v1.json', import.meta.url),
        'utf8',
      ),
    );
    registerLlmRequestIntercept('node_llm_req_helper', 10, false, ({ request, annotated }) => {
      request.content.helper = true;
      return {
        request,
        annotated,
        pendingMarks: [
          {
            name: 'request.first',
            categoryProfile: { subtype: 'optimizer.saved_tokens' },
            data: { order: 1 },
          },
          { name: 'request.second', metadata: { source: 'node' } },
        ],
        optimizationContributions: [contributionFixture],
      };
    });

    const result = await llmRequestIntercepts('helper_llm', makeNative());
    assert.equal(result.request.content.helper, true);
    assert.equal(result.annotated, null);
    assert.deepEqual(result.pendingMarks, [
      {
        name: 'request.first',
        category: null,
        categoryProfile: { subtype: 'optimizer.saved_tokens' },
        data: { order: 1 },
        metadata: null,
      },
      {
        name: 'request.second',
        category: null,
        categoryProfile: null,
        data: null,
        metadata: { source: 'node' },
      },
    ]);
    assert.deepEqual(result.optimizationContributions, [contributionFixture]);
    deregisterLlmRequestIntercept('node_llm_req_helper');
  });

  it('generated request-intercept declarations reference the canonical open optimization type', () => {
    const declarations = readFileSync(new URL('../index.d.ts', import.meta.url), 'utf8');
    const pluginDeclarations = readFileSync(new URL('../plugin.d.ts', import.meta.url), 'utf8');
    const openKind = "kind: 'input_compression' | 'model_routing' | (string & {})";

    assert.equal(declarations.split(openKind).length - 1, 1);
    assert.equal(pluginDeclarations.split(openKind).length - 1, 1);
    assert.match(declarations, /registerLlmRequestIntercept\([^\n]*import\('\.\/plugin'\)\.LlmRequestInterceptOutcome/);
    assert.match(
      declarations,
      /scopeRegisterLlmRequestIntercept\([^\n]*import\('\.\/plugin'\)\.LlmRequestInterceptOutcome/,
    );
  });

  it('generated LLM sanitizer declarations expose directional codec contexts', () => {
    const declarations = readFileSync(new URL('../index.d.ts', import.meta.url), 'utf8');

    assert.equal(declarations.split("context: import('./plugin').LlmSanitizeRequestContext").length - 1, 2);
    assert.equal(declarations.split("context: import('./plugin').LlmSanitizeResponseContext").length - 1, 2);
    assert.doesNotMatch(declarations, /registerLlmSanitizeRequestGuardrail\([^\n]*\.\.\.args: any\[\]/);
  });

  it('generated middleware declarations expose Promise-aware callback types', () => {
    const declarations = readFileSync(new URL('../index.d.ts', import.meta.url), 'utf8');
    const registrations = [
      'registerToolSanitizeRequestGuardrail',
      'registerToolSanitizeResponseGuardrail',
      'registerToolConditionalExecutionGuardrail',
      'registerToolRequestIntercept',
      'registerToolExecutionIntercept',
      'registerLlmSanitizeRequestGuardrail',
      'registerLlmSanitizeResponseGuardrail',
      'registerLlmConditionalExecutionGuardrail',
      'registerLlmRequestIntercept',
      'registerLlmExecutionIntercept',
      'registerLlmStreamExecutionIntercept',
      'scopeRegisterToolSanitizeRequestGuardrail',
      'scopeRegisterToolSanitizeResponseGuardrail',
      'scopeRegisterToolConditionalExecutionGuardrail',
      'scopeRegisterToolRequestIntercept',
      'scopeRegisterToolExecutionIntercept',
      'scopeRegisterLlmSanitizeRequestGuardrail',
      'scopeRegisterLlmSanitizeResponseGuardrail',
      'scopeRegisterLlmConditionalExecutionGuardrail',
      'scopeRegisterLlmRequestIntercept',
      'scopeRegisterLlmExecutionIntercept',
      'scopeRegisterLlmStreamExecutionIntercept',
    ];

    for (const registration of registrations) {
      const declaration = declarations.match(new RegExp(`export declare function ${registration}\\([^\\n]+`))?.[0];
      assert.ok(declaration, `missing declaration for ${registration}`);
      assert.doesNotMatch(declaration, /\.\.\.args: any\[\]/, `${registration} must not expose an any callback`);
      assert.match(declaration, /Promise</, `${registration} must expose its Promise callback form`);
    }
    assert.equal(
      declarations.split('next: (request: Json) => Promise<Json[]>').length - 1,
      2,
      'global and scope-local stream intercept declarations must expose the buffered next contract',
    );
  });

  it('plugin declarations expose Promise middleware and the implemented stream contract', () => {
    const declarations = readFileSync(new URL('../plugin.d.ts', import.meta.url), 'utf8');

    assert.match(declarations, /registerMarkSanitizeGuardrail\([\s\S]*?Promise<EventSanitizeFields>/);
    assert.match(declarations, /registerScopeSanitizeStartGuardrail\([\s\S]*?Promise<EventSanitizeFields>/);
    assert.match(declarations, /registerScopeSanitizeEndGuardrail\([\s\S]*?Promise<EventSanitizeFields>/);
    assert.match(declarations, /registerToolSanitizeRequestGuardrail\([\s\S]*?Json \| Promise<Json>/);
    assert.match(declarations, /registerToolSanitizeResponseGuardrail\([\s\S]*?Json \| Promise<Json>/);
    assert.match(declarations, /registerLlmSanitizeRequestGuardrail\([\s\S]*?Promise<Json \| null>/);
    assert.match(declarations, /registerLlmSanitizeResponseGuardrail\([\s\S]*?Promise<Json \| null>/);
    assert.match(declarations, /registerLlmConditionalExecutionGuardrail\([\s\S]*?Promise<string \| null>/);
    assert.match(declarations, /registerLlmRequestIntercept\([\s\S]*?Promise<LlmRequestInterceptOutcome>/);
    assert.match(
      declarations,
      /registerLlmStreamExecutionIntercept\([\s\S]*?next: \(request: Json\) => Promise<Json\[\]>/,
    );
    assert.doesNotMatch(declarations, /registerLlmStreamExecutionIntercept\([\s\S]*?AsyncIterable/);
  });

  it('standalone conditional execution helper throws on rejection', async () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond_helper', 10, () => 'llm blocked by helper');
    try {
      await assert.rejects(() => llmConditionalExecution(makeNative()), /guardrail rejected/i);
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_llm_cond_helper');
    }
  });

  it('standalone conditional execution helper resolves when allowed', async () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond_allow', 10, () => null);
    try {
      await assert.doesNotReject(() => llmConditionalExecution(makeNative()));
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_llm_cond_allow');
    }
  });
});

describe('LLM event fields', () => {
  it('subscriber receives modelName and payload fields', async () => {
    const events = [];
    const scope = pushScope('llm_event_parent', ScopeType.Agent, null, null);
    registerSubscriber('node_llm_field_sub', (e) => events.push(e));
    try {
      const handle = llmCall(
        'field_llm',
        makeNative(),
        scope,
        LLM_ATTR_STATELESS,
        {
          start: true,
        },
        {
          meta: true,
        },
        'gpt-field-model',
      );
      assert.equal(handle.attributes, LLM_ATTR_STATELESS);
      assert.equal(handle.parentUuid, scope.uuid);
      llmCallEnd(
        handle,
        {
          ok: true,
        },
        {
          end: true,
        },
        {
          final: true,
        },
      );

      const deadline = Date.now() + 2000;
      while (events.filter((e) => e.name === 'field_llm').length < 2 && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 10));
      }

      const start = events.find(
        (e) => e.name === 'field_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'start',
      );
      const end = events.find(
        (e) => e.name === 'field_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'end',
      );
      assert.equal(start.category_profile.model_name, 'gpt-field-model');
      assert.deepEqual(start.data, {
        headers: {},
        content: {
          messages: [],
          model: 'test-model',
        },
      });
      assert.deepEqual(end.data, {
        ok: true,
      });
    } finally {
      deregisterSubscriber('node_llm_field_sub');
      popScope(scope);
    }
  });
});
