// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');
const adaptive = require('../adaptive.js');

const {
  GeminiGenerateContentCodec,
  ScopeType,
  createScopeStack,
  deregisterLlmConditionalExecutionGuardrail,
  deregisterLlmExecutionIntercept,
  deregisterToolExecutionIntercept,
  llmCallExecute,
  llmCallExecuteAsync,
  llmConditionalExecution,
  llmStreamCallExecute,
  pushScope,
  registerLlmConditionalExecutionGuardrail,
  registerLlmExecutionIntercept,
  registerSubscriber,
  registerToolExecutionIntercept,
  scopeRegisterSubscriber,
  toolCallExecute,
  toolCallExecuteAsync,
  withScopeStack,
} = lib;

class ToolFailure extends Error {
  constructor(message) {
    super(message);
    this.name = 'ToolFailure';
    this.details = { attempt: 1 };
  }
}

function makeNative() {
  return { headers: {}, content: { messages: [], model: 'test-model' } };
}

function same(expected) {
  return (actual) => {
    assert.equal(actual, expected);
    return true;
  };
}

describe('error propagation', () => {
  it('a sync tool callback throw is rejected as the thrown object', async () => {
    const thrown = new ToolFailure('sync tool boom');
    await assert.rejects(
      () =>
        toolCallExecute('node_propagation_sync_tool', { x: 1 }, () => {
          throw thrown;
        }),
      same(thrown),
    );
    assert.ok(thrown instanceof ToolFailure);
    assert.deepEqual(thrown.details, { attempt: 1 });
  });

  it('an async tool callback rejection is rejected as the thrown object', async () => {
    const thrown = new ToolFailure('async tool boom');
    await assert.rejects(
      () => toolCallExecuteAsync('node_propagation_async_tool', { x: 1 }, async () => Promise.reject(thrown)),
      same(thrown),
    );
  });

  it('a rejection travels unchanged through an intercept chain', async () => {
    const thrown = new ToolFailure('downstream boom');
    let observed;
    registerToolExecutionIntercept('node_propagation_intercept', 10, async (context, next) => {
      try {
        return await next(context.args);
      } catch (error) {
        observed = error;
        throw error;
      }
    });
    try {
      await assert.rejects(
        () =>
          toolCallExecuteAsync('node_propagation_chained_tool', { x: 1 }, async () => {
            throw thrown;
          }),
        same(thrown),
      );
      assert.equal(observed, thrown);
    } finally {
      deregisterToolExecutionIntercept('node_propagation_intercept');
    }
  });

  it('a primitive rejection is rejected as that primitive', async () => {
    await assert.rejects(
      () => llmCallExecuteAsync('node_propagation_primitive', makeNative(), async () => Promise.reject(42)),
      same(42),
    );
  });

  it('an LLM execution intercept throw is rejected as the thrown object', async () => {
    const thrown = new TypeError('intercept boom');
    registerLlmExecutionIntercept('node_propagation_llm_intercept', 10, async () => {
      throw thrown;
    });
    try {
      await assert.rejects(
        () => llmCallExecute('node_propagation_llm', makeNative(), () => ({ ok: true })),
        same(thrown),
      );
    } finally {
      deregisterLlmExecutionIntercept('node_propagation_llm_intercept');
    }
  });

  it('a conditional guardrail throw is rejected as the thrown object', async () => {
    const thrown = new ToolFailure('guardrail boom');
    registerLlmConditionalExecutionGuardrail('node_propagation_guardrail_throw', 10, () => {
      throw thrown;
    });
    try {
      await assert.rejects(() => llmConditionalExecution(makeNative()), same(thrown));
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_propagation_guardrail_throw');
    }
  });

  it('collector and finalizer throws surface as the thrown objects', async () => {
    const collectorFailure = new RangeError('collector boom');
    const finalizerFailure = new ToolFailure('finalizer boom');
    const stream = await llmStreamCallExecute(
      'node_propagation_stream',
      makeNative(),
      async function* () {
        yield { token: 'first' };
      },
      () => {
        throw collectorFailure;
      },
      () => {
        throw finalizerFailure;
      },
    );
    await assert.rejects(() => stream.next(), same(collectorFailure));
    await assert.rejects(() => stream.close(), same(finalizerFailure));
  });

  it('a stream close with several failures rejects with an AggregateError of the thrown objects', async () => {
    const cleanupFailure = new RangeError('cleanup boom');
    const finalizerFailure = new ToolFailure('finalizer boom');
    const stream = await llmStreamCallExecute(
      'node_propagation_stream_aggregate',
      makeNative(),
      async function* (_request, signal) {
        try {
          yield { token: 'first' };
          await new Promise((resolve) => signal.addEventListener('abort', resolve, { once: true }));
        } finally {
          throw cleanupFailure;
        }
      },
      undefined,
      () => {
        throw finalizerFailure;
      },
    );
    assert.deepEqual(await stream.next(), { token: 'first' });
    await assert.rejects(
      () => stream.close(),
      (error) => {
        assert.ok(error instanceof AggregateError);
        assert.deepEqual(error.errors, [cleanupFailure, finalizerFailure]);
        assert.equal(error.errors[0], cleanupFailure);
        assert.equal(error.errors[1], finalizerFailure);
        assert.match(error.message, /cleanup boom; stream finalizer also failed: .*finalizer boom/);
        return true;
      },
    );
  });

  it('a sync throw inside withScopeStack is thrown as the thrown object', () => {
    const thrown = new ToolFailure('scoped boom');
    assert.throws(
      () =>
        withScopeStack(createScopeStack(), () => {
          throw thrown;
        }),
      same(thrown),
    );
  });

  it('argument validation throws napi InvalidArg errors', () => {
    assert.throws(() => registerSubscriber('node_propagation_not_a_function', 'not a function'), {
      code: 'InvalidArg',
      message: 'callback must be a function, got String',
    });
    assert.throws(() => pushScope('node_propagation_scope', ScopeType.Agent, null, null, null, null, null, 1.5), {
      code: 'InvalidArg',
      message: /safe integer number of Unix microseconds/,
    });
    assert.throws(() => scopeRegisterSubscriber('not-a-uuid', 'node_propagation_subscriber', () => {}), {
      code: 'InvalidArg',
      message: /invalid UUID/,
    });
    const codec = new GeminiGenerateContentCodec();
    assert.throws(() => codec.encode({ messages: 'not an array' }, { headers: {}, content: { contents: [] } }), {
      code: 'InvalidArg',
      message: /invalid AnnotatedLlmRequest/,
    });
  });

  it('runtime errors are thrown as plain errors carrying the runtime message', async () => {
    const runtime = new adaptive.AdaptiveRuntime(adaptive.defaultConfig());
    await runtime.shutdown();
    assert.throws(() => runtime.deregister(), {
      name: 'Error',
      message: 'adaptive runtime already shut down',
    });
    await assert.rejects(() => runtime.shutdown(), {
      name: 'Error',
      message: 'adaptive runtime already shut down',
    });
    registerLlmConditionalExecutionGuardrail('node_propagation_guardrail', 10, () => 'blocked by propagation test');
    try {
      await assert.rejects(() => llmConditionalExecution(makeNative()), {
        name: 'Error',
        message: /blocked by propagation test/,
      });
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_propagation_guardrail');
    }
  });
});
