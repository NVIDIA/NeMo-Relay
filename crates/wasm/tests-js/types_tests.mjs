// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { resetScopeStack, wasm } from './test_support.mjs';

test('WebAssembly package exports canonical runtime classes and scope type enum values', () => {
  assert.equal(typeof wasm.ScopeStack, 'function');
  assert.equal(typeof wasm.ScopeHandle, 'function');
  assert.equal(typeof wasm.ToolHandle, 'function');
  assert.equal(typeof wasm.LlmRequest, 'function');
  assert.equal(typeof wasm.OpenTelemetrySubscriber, 'function');
  assert.equal(typeof wasm.OpenInferenceSubscriber, 'function');
  assert.equal(typeof wasm.AtifExporter, 'function');
  assert.equal(typeof wasm.AtofExporter, 'undefined');

  assert.equal(wasm.ScopeType.Agent, 0);
  assert.equal(wasm.ScopeType.Function, 1);
  assert.equal(wasm.ScopeType.Tool, 2);
  assert.equal(wasm.ScopeType.Llm, 3);
  assert.equal(wasm.ScopeType.Retriever, 4);
  assert.equal(wasm.ScopeType.Embedder, 5);
  assert.equal(wasm.ScopeType.Reranker, 6);
  assert.equal(wasm.ScopeType.Guardrail, 7);
  assert.equal(wasm.ScopeType.Evaluator, 8);
  assert.equal(wasm.ScopeType.Custom, 9);
  assert.equal(wasm.ScopeType.Unknown, 10);
});

test('WebAssembly LlmRequest getters and setters round-trip JS values', () => {
  const request = new wasm.LlmRequest(
    {
      trace: '1',
    },
    {
      model: 'demo-model',
      messages: [],
    },
  );

  try {
    assert.deepEqual(request.headers, {
      trace: '1',
    });
    assert.deepEqual(request.content, {
      model: 'demo-model',
      messages: [],
    });

    request.headers = {
      trace: '2',
      nested: true,
    };
    request.content = {
      model: 'demo-model',
      messages: [
        {
          role: 'user',
          content: 'hi',
        },
      ],
    };

    assert.equal(request.headers.trace, '2');
    assert.equal(request.content.messages[0].content, 'hi');
  } finally {
    request.free();
  }
});

test('WebAssembly scope stack helpers expose current scope stack instances', () => {
  const stack = resetScopeStack();
  const currentStack = wasm.currentScopeStack();

  assert.ok(currentStack instanceof wasm.ScopeStack);

  currentStack.free();
  stack.free();
});
