// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  __testEncodeWithCodec,
  JsonPassthrough,
  typedLlmExecute,
  typedLlmStreamExecute,
  typedToolExecute,
} from '../pkg/typed.js';
import { drainStream, makeLlmRequest, unique, waitFor, wasm } from './test_support.mjs';

test('WebAssembly typed codec helper preserves annotated and original values', () => {
  const annotatedCodec = {
    encode(annotated, original) {
      return {
        annotated,
        original,
      };
    },
  };

  assert.deepEqual(
    __testEncodeWithCodec(annotatedCodec, {
      annotated: {
        model: 'demo',
      },
      original: {
        model: 'raw',
      },
    }),
    {
      annotated: {
        model: 'demo',
      },
      original: {
        model: 'raw',
      },
    },
  );
});

test('WebAssembly typed tool wrappers execute synchronous flows', async () => {
  const passthrough = new JsonPassthrough();

  const syncToolResult = await typedToolExecute(
    'wrapper_tool_sync',
    {
      value: 4,
    },
    (args) => ({
      tripled: args.value * 3,
    }),
    passthrough,
    passthrough,
    {
      attributes: 1,
      data: {
        source: 'sync',
      },
      metadata: {
        kind: 'tool',
      },
    },
  );
  assert.deepEqual(syncToolResult, {
    tripled: 12,
  });
});

test('WebAssembly typed tool wrappers execute asynchronous flows', async () => {
  const passthrough = new JsonPassthrough();
  const asyncToolResult = await typedToolExecute(
    'wrapper_tool_async',
    {
      value: 5,
    },
    async (args) => ({
      quadrupled: args.value * 4,
    }),
    passthrough,
    passthrough,
  );
  assert.deepEqual(asyncToolResult, {
    quadrupled: 20,
  });
});

test('WebAssembly typed llm wrappers support response codecs', async () => {
  const passthrough = new JsonPassthrough();
  const responseEvents = [];
  const subscriberName = unique('wrapper_llm_cost');
  wasm.registerSubscriber(subscriberName, (event) => responseEvents.push(event));

  try {
    const llmResult = await typedLlmExecute(
      'wrapper_llm',
      makeLlmRequest('gpt-4o-mini'),
      () => ({
        response: 'ok',
        model: 'gpt-4o-mini',
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
          cost: {
            total: 0.00001,
            source: 'provider_reported',
            pricing_provider: 'wasm-provider',
            pricing_model: 'gpt-4o-mini',
            pricing_as_of: '2026-06-04',
            pricing_source: 'https://example.test/pricing',
          },
        },
      }),
      passthrough,
      {
        responseCodec: {
          decodeResponse(response) {
            return response;
          },
        },
      },
    );
    assert.deepEqual(llmResult, {
      response: 'ok',
      model: 'gpt-4o-mini',
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
        cost: {
          total: 0.00001,
          source: 'provider_reported',
          pricing_provider: 'wasm-provider',
          pricing_model: 'gpt-4o-mini',
          pricing_as_of: '2026-06-04',
          pricing_source: 'https://example.test/pricing',
        },
      },
    });

    const endEvent = await waitFor(() =>
      responseEvents.find(
        (event) =>
          event.kind === 'scope' &&
          event.category === 'llm' &&
          event.scope_category === 'end' &&
          event.name === 'wrapper_llm',
      ),
    );
    assert.equal(
      endEvent.category_profile.annotated_response.usage.cost.pricing_provider,
      'wasm-provider',
    );
  } finally {
    wasm.deregisterSubscriber(subscriberName);
  }
});

test('WebAssembly typed llm wrappers execute synchronous flows', async () => {
  const passthrough = new JsonPassthrough();
  const syncLlmResult = await typedLlmExecute(
    'wrapper_llm_sync',
    makeLlmRequest('sync-model'),
    () => ({
      response: 'sync',
    }),
    passthrough,
    {
      attributes: 1,
      data: {
        source: 'sync',
      },
      metadata: {
        kind: 'llm',
      },
      modelName: 'sync-model',
    },
  );
  assert.deepEqual(syncLlmResult, {
    response: 'sync',
  });
});

test('WebAssembly typed llm wrappers execute asynchronous flows', async () => {
  const passthrough = new JsonPassthrough();
  const asyncLlmResult = await typedLlmExecute(
    'wrapper_llm_async',
    makeLlmRequest('async-model'),
    async () => ({
      response: 'async',
    }),
    passthrough,
  );
  assert.deepEqual(asyncLlmResult, {
    response: 'async',
  });
});

test('WebAssembly typed llm wrappers preserve falsy metadata/model options and use request event data', async () => {
  const passthrough = new JsonPassthrough();
  const falsyOptionEvents = [];
  const subscriberName = unique('wrapper_falsy_opts');
  wasm.registerSubscriber(subscriberName, (event) => falsyOptionEvents.push(event));

  try {
    await typedLlmExecute(
      'wrapper_falsy_opts_llm',
      makeLlmRequest('falsy-model'),
      () => ({
        ok: true,
      }),
      passthrough,
      {
        metadata: 0,
        modelName: '',
      },
    );

    const startEvent = await waitFor(() =>
      falsyOptionEvents.find(
        (event) =>
          event.kind === 'scope' &&
          event.category === 'llm' &&
          event.scope_category === 'start' &&
          event.name === 'wrapper_falsy_opts_llm',
      ),
    );
    assert.deepEqual(startEvent.data, makeLlmRequest('falsy-model'));
    assert.equal(startEvent.metadata, 0);
    assert.equal(startEvent.category_profile.model_name, '');
  } finally {
    wasm.deregisterSubscriber(subscriberName);
  }
});

test('WebAssembly typed llm stream wrappers collect chunks with hooks', async () => {
  const passthrough = new JsonPassthrough();
  const seen = [];

  const stream = await typedLlmStreamExecute(
    'wrapper_stream',
    makeLlmRequest('test-model'),
    async function* () {
      yield {
        token: 'hello',
      };
      yield {
        token: 'world',
      };
    },
    (chunk) => {
      seen.push(chunk);
    },
    () => ({
      count: seen.length,
    }),
    passthrough,
    passthrough,
  );

  const chunks = await drainStream(stream);
  assert.deepEqual(chunks, [
    [
      {
        token: 'hello',
      },
      {
        token: 'world',
      },
    ],
  ]);
  assert.deepEqual(seen, chunks);
});

test('WebAssembly typed llm stream wrappers collect chunks without hooks', async () => {
  const passthrough = new JsonPassthrough();
  const streamWithoutHooks = await typedLlmStreamExecute(
    'wrapper_stream_no_hooks',
    makeLlmRequest('test-model'),
    async function* () {
      yield {
        token: 'solo',
      };
    },
    () => {},
    () => ({
      count: 1,
    }),
    passthrough,
    passthrough,
    {
      modelName: 'test-model',
    },
  );

  assert.deepEqual(await drainStream(streamWithoutHooks), [
    [
      {
        token: 'solo',
      },
    ],
  ]);
});
