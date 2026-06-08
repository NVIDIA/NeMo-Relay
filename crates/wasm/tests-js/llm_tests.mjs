// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  currentScope,
  drainStream,
  expectInvalidLlmRequest,
  makeLlmRequest,
  rejectInvalidLlmRequest,
  resetScopeStack,
  unique,
  waitFor,
  wasm,
} from './test_support.mjs';

test('WebAssembly llm handles preserve nullable inputs and handle properties', () => {
  const stack = resetScopeStack();
  const scope = wasm.pushScope('llm_scope', wasm.ScopeType.Function, null, 0, null, null);
  const llmRequest = makeLlmRequest();
  let llmHandle;

  try {
    llmHandle = wasm.llmCall('optional_llm', llmRequest, currentScope(), undefined, null, undefined);
    assert.equal(llmHandle.name, 'optional_llm');
    assert.equal(typeof llmHandle.uuid, 'string');
    assert.equal(llmHandle.attributes, 0);
    assert.equal(llmHandle.parentUuid, scope.uuid);
    wasm.llmCallEnd(
      llmHandle,
      {
        role: 'assistant',
        content: 'done',
        tool_calls: [],
      },
      null,
      undefined,
    );
  } finally {
    if (llmHandle) {
      llmHandle.free();
    }
    wasm.popScope(scope);
    scope.free();
    stack.free();
  }
});

test('WebAssembly llm execute returns results and nulls through the wrapper surface', async () => {
  const llmResult = await wasm.llmCallExecute('optional_llm_exec', makeLlmRequest(), async () => ({
    role: 'assistant',
    content: 'ok',
    tool_calls: [],
  }));
  assert.equal(llmResult.content, 'ok');

  const llmNullResult = await wasm.llmCallExecute('optional_llm_exec_null', makeLlmRequest(), async () => undefined);
  assert.equal(llmNullResult, null);
});

test('WebAssembly llm wrappers reject invalid request shapes', async () => {
  expectInvalidLlmRequest(() =>
    wasm.llmRequestIntercepts('bad_llm', {
      headers: [],
      content: 'bad',
    }),
  );
  expectInvalidLlmRequest(() =>
    wasm.llmConditionalExecution({
      headers: [],
      content: 'bad',
    }),
  );
  await rejectInvalidLlmRequest(
    wasm.llmCallExecute(
      'bad_exec',
      {
        headers: [],
        content: 'bad',
      },
      async () => ({
        role: 'assistant',
      }),
    ),
  );
  await rejectInvalidLlmRequest(
    wasm.llmStreamCallExecute(
      'bad_stream',
      {
        headers: [],
        content: 'bad',
      },
      async () => [],
    ),
  );
});

test('WebAssembly llm stream execution intercept composes with next', async () => {
  const interceptName = unique('wasm_llm_stream_exec');
  let stream;

  wasm.registerLlmStreamExecutionIntercept(interceptName, 10, async (request, next) => {
    const chunks = await next({
      ...request,
      content: {
        ...request.content,
        intercepted: true,
      },
    });
    return [
      ...chunks,
      {
        wrapped: true,
      },
    ];
  });

  try {
    stream = await wasm.llmStreamCallExecute(
      'stream_llm_intercepted',
      makeLlmRequest(),
      async (request) => [
        {
          downstream: request.content.intercepted === true,
        },
      ],
      null,
      null,
      null,
      null,
      null,
      null,
      null,
    );

    assert.deepEqual(await drainStream(stream), [
      {
        downstream: true,
      },
      {
        wrapped: true,
      },
    ]);
  } finally {
    if (stream) {
      stream.free();
    }
    wasm.deregisterLlmStreamExecutionIntercept(interceptName);
  }
});

test('WebAssembly llm stream execution intercept rejects invalid next payloads', async () => {
  const interceptName = unique('wasm_llm_stream_invalid');
  wasm.registerLlmStreamExecutionIntercept(interceptName, 10, async (_request, next) =>
    next({
      headers: 1,
      content: {
        model: 'broken',
      },
    }),
  );

  try {
    await assert.rejects(
      () =>
        wasm.llmStreamCallExecute(
          'stream_llm_invalid_next',
          makeLlmRequest(),
          async () => [
            {
              shouldNotRun: true,
            },
          ],
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
  } finally {
    wasm.deregisterLlmStreamExecutionIntercept(interceptName);
  }
});

test('WebAssembly llm and stream flows work from the generated Node package', async () => {
  const stack = resetScopeStack();
  const events = [];
  const subscriberName = unique('llm_event_subscriber');
  const request = {
    headers: {
      trace: '1',
    },
    content: {
      model: 'demo-model',
      messages: [],
    },
  };
  const llmInterceptName = unique('llm_req');

  wasm.registerSubscriber(subscriberName, (event) => events.push(event));
  wasm.registerLlmRequestIntercept(llmInterceptName, 1, false, ({ request: nextRequest, annotated }) => ({
    request: {
      ...nextRequest,
      content: {
        ...nextRequest.content,
        intercepted: true,
      },
    },
    annotated,
  }));

  try {
    const interceptedRequest = wasm.llmRequestIntercepts('pkg_llm', request);
    assert.equal(interceptedRequest.content.intercepted, true);
    wasm.llmConditionalExecution(request);
  } finally {
    wasm.deregisterLlmRequestIntercept(llmInterceptName);
    wasm.deregisterSubscriber(subscriberName);
    stack.free();
  }
});

test('WebAssembly llm lifecycle flows work from the generated Node package', async () => {
  const request = {
    headers: {
      trace: '1',
    },
    content: {
      model: 'demo-model',
      messages: [],
    },
  };
  let llmHandle;

  try {
    llmHandle = wasm.llmCall(
      'pkg_llm',
      request,
      null,
      1,
      {
        phase: 'start',
      },
      {
        source: 'js',
      },
      'demo-model',
    );
    assert.equal(llmHandle.name, 'pkg_llm');
    wasm.llmCallEnd(
      llmHandle,
      {
        role: 'assistant',
        content: 'done',
        tool_calls: [],
      },
      {
        phase: 'end',
      },
      {
        source: 'js',
      },
    );
  } finally {
    if (llmHandle) {
      llmHandle.free();
    }
  }
});

test('WebAssembly llm execute flow works from the generated Node package', async () => {
  const stack = resetScopeStack();
  const events = [];
  const subscriberName = unique('llm_event_subscriber_exec');
  const request = {
    headers: {
      trace: '1',
    },
    content: {
      model: 'demo-model',
      messages: [],
    },
  };

  wasm.registerSubscriber(subscriberName, (event) => events.push(event));

  try {
    const llmResult = await wasm.llmCallExecute(
      'pkg_llm_exec',
      request,
      async (nextRequest) => ({
        role: 'assistant',
        content: `hello ${nextRequest.content.model}`,
        tool_calls: [],
      }),
      null,
      1,
      {
        from: 'llm',
      },
      {
        layer: 'js',
      },
      'demo-model',
    );
    assert.equal(llmResult.role, 'assistant');
    assert.equal(llmResult.content, 'hello demo-model');
    await waitFor(() => events.some((event) => event.name === 'pkg_llm_exec'));
  } finally {
    wasm.deregisterSubscriber(subscriberName);
    stack.free();
  }
});

test('WebAssembly llmCallExecute adds OTEL status metadata to end events', async () => {
  const stack = resetScopeStack();
  const events = [];
  const subscriberName = unique('wasm_llm_status_sub');

  wasm.registerSubscriber(subscriberName, (event) => events.push(event));

  try {
    await wasm.llmCallExecute(
      'wasm_llm_status_ok',
      makeLlmRequest(),
      async () => ({
        role: 'assistant',
        content: 'ok',
        tool_calls: [],
      }),
      null,
      0,
      null,
      {
        caller: 'wasm-llm',
        'otel.status_code': 'USER',
      },
      'demo-model',
    );

    await assert.rejects(
      () =>
        wasm.llmCallExecute(
          'wasm_llm_status_error',
          makeLlmRequest(),
          async () => {
            throw new Error('wasm llm failure');
          },
          null,
          0,
          null,
          {
            caller: 'wasm-llm-error',
          },
          'demo-model',
        ),
      /wasm llm failure/,
    );

    const okEvent = await waitFor(() =>
      events.find(
        (event) => event.kind === 'scope' && event.scope_category === 'end' && event.name === 'wasm_llm_status_ok',
      ),
    );
    const errorEvent = await waitFor(() =>
      events.find(
        (event) => event.kind === 'scope' && event.scope_category === 'end' && event.name === 'wasm_llm_status_error',
      ),
    );

    assert.equal(okEvent.metadata.caller, 'wasm-llm');
    assert.equal(okEvent.metadata['otel.status_code'], 'OK');
    assert.equal(errorEvent.metadata.caller, 'wasm-llm-error');
    assert.equal(errorEvent.metadata['otel.status_code'], 'ERROR');
    assert.match(errorEvent.metadata['otel.status_description'], /wasm llm failure/);
  } finally {
    wasm.deregisterSubscriber(subscriberName);
    stack.free();
  }
});

test('WebAssembly llm stream flow works from the generated Node package', async () => {
  const request = {
    headers: {
      trace: '1',
    },
    content: {
      model: 'demo-model',
      messages: [],
    },
  };
  const collected = [];
  let finalized = false;
  let stream;

  try {
    stream = await wasm.llmStreamCallExecute(
      'pkg_llm_stream',
      request,
      async () => [
        {
          delta: 'hello',
        },
        {
          delta: 'world',
        },
      ],
      (chunk) => collected.push(chunk),
      () => {
        finalized = true;
        return {
          combined: true,
        };
      },
      null,
      2,
      {
        from: 'stream',
      },
      {
        layer: 'js',
      },
      'demo-model',
    );
    const chunks = await drainStream(stream);
    assert.deepEqual(chunks, [
      [
        {
          delta: 'hello',
        },
        {
          delta: 'world',
        },
      ],
    ]);
    assert.deepEqual(collected, chunks);
    assert.equal(finalized, true);
  } finally {
    if (stream) {
      stream.free();
    }
  }
});
