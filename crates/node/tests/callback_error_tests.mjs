// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  __testClosedCollectorCallback,
  __testClosedFinalizerCallback,
  __testClosedLlmResponseCallback,
  __testClosedLlmSanitizeRequestCallback,
  __testClosedPromiseAwareCall,
  __testClosedToolCallback,
  clearLastCallbackError,
  deregisterLlmSanitizeRequestGuardrail,
  getLastCallbackError,
  llmCallExecute,
  registerLlmSanitizeRequestGuardrail,
} = lib;

function makeNative() {
  return {
    headers: {},
    content: {
      messages: [],
      model: 'test-model',
    },
  };
}

describe('callback error helpers', () => {
  it('getLastCallbackError and clearLastCallbackError expose malformed sanitize-request failures', async () => {
    clearLastCallbackError();
    registerLlmSanitizeRequestGuardrail('node_llm_san_req_public_error', 10, () => ({ broken: true }));
    try {
      const result = await llmCallExecute(
        'san_req_public_error_llm',
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
      assert.match(
        getLastCallbackError() ?? '',
        /JS LLM sanitize request callback failed: failed to deserialize LlmRequest/i,
      );
      clearLastCallbackError();
      assert.equal(getLastCallbackError(), null);
    } finally {
      deregisterLlmSanitizeRequestGuardrail('node_llm_san_req_public_error');
      clearLastCallbackError();
    }
  });

  it('closed tool sanitize callbacks preserve the original payload and record the queue failure', () => {
    const args = {
      value: 1,
    };
    const result = __testClosedToolCallback(
      () => ({
        ok: true,
      }),
      'closed_tool',
      args,
    );
    assert.deepEqual(result, args);
    assert.match(getLastCallbackError() ?? '', /failed to queue JS tool callback/i);
    clearLastCallbackError();
  });

  it('closed llm sanitize-request callbacks fall back to the original request and record the queue failure', () => {
    const request = makeNative();
    const result = __testClosedLlmSanitizeRequestCallback(
      () => ({
        broken: true,
      }),
      request,
    );
    assert.deepEqual(result, request);
    assert.match(getLastCallbackError() ?? '', /failed to queue JS LLM sanitize request callback/i);
    clearLastCallbackError();
  });

  it('closed llm sanitize-response callbacks fall back to the original response and record the queue failure', () => {
    const response = {
      ok: true,
    };
    const result = __testClosedLlmResponseCallback(
      () => ({
        rewritten: true,
      }),
      response,
    );
    assert.deepEqual(result, response);
    assert.match(getLastCallbackError() ?? '', /failed to queue JS LLM sanitize response callback/i);
    clearLastCallbackError();
  });

  it('closed collector callbacks surface the queue failure and record it', async () => {
    assert.throws(
      () =>
        __testClosedCollectorCallback(() => undefined, {
          token: 'x',
        }),
      /failed to queue JS collector callback/i,
    );
    assert.match(getLastCallbackError() ?? '', /failed to queue JS collector callback/i);
    clearLastCallbackError();
  });

  it('closed finalizer callbacks fall back to null and record the queue failure', () => {
    const result = __testClosedFinalizerCallback(() => ({
      done: true,
    }));
    assert.equal(result, null);
    assert.match(getLastCallbackError() ?? '', /failed to queue JS finalizer callback/i);
    clearLastCallbackError();
  });

  it('closed PromiseAwareFn calls reject with the closed threadsafe-function error', async () => {
    await assert.rejects(
      () =>
        __testClosedPromiseAwareCall(() => ({
          ok: true,
        })),
      /PromiseAwareFn threadsafe function closed/i,
    );
  });
});
