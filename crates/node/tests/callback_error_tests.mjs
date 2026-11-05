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
  __testClosedPromiseAwareCall,
  clearLastCallbackError,
  deregisterLlmSanitizeRequestGuardrail,
  flushSubscribers,
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

      const { traceparent, ...headers } = result.headers;
      assert.deepEqual({ ...result, headers }, { model: 'test-model', headers: {} });
      assert.match(traceparent, /^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/);
      await flushSubscribers();
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

  it('closed collector callbacks report the environment shutdown without recording it', () => {
    clearLastCallbackError();
    assert.throws(
      () =>
        __testClosedCollectorCallback(() => undefined, {
          token: 'x',
        }),
      /JS collector callback skipped: Node environment is shutting down/,
    );
    assert.equal(getLastCallbackError(), null);
  });

  it('closed finalizer callbacks yield null without recording the shutdown', () => {
    clearLastCallbackError();
    const result = __testClosedFinalizerCallback(() => ({
      done: true,
    }));
    assert.equal(result, null);
    assert.equal(getLastCallbackError(), null);
  });

  it('closed PromiseAwareFn calls reject with the queue failure', async () => {
    await assert.rejects(
      () =>
        __testClosedPromiseAwareCall(() => ({
          ok: true,
        })),
      /failed to queue threadsafe function call: Closing/,
    );
  });

  it('PromiseAwareFn argument conversion failures reject without invoking the callback', async () => {
    let invoked = false;
    await assert.rejects(
      () =>
        __testClosedPromiseAwareCall(() => {
          invoked = true;
          return null;
        }, true),
      /forced PromiseAwareFn conversion failure/i,
    );
    assert.equal(invoked, false);
  });
});
