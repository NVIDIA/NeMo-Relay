// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  drainStream,
  expectAlreadyExists,
  expectInvalidUuid,
  makeLlmRequest,
  resetScopeStack,
  scopeRegistrationCases,
  unique,
  wasm,
} from './test_support.mjs';

test('scope-local register and deregister wrappers are callable', () => {
  const stack = resetScopeStack();
  const scope = wasm.pushScope('registration_scope', wasm.ScopeType.Function, null, 0, null, null);

  try {
    for (const [prefix, register, deregister, invoke] of scopeRegistrationCases(scope.uuid)) {
      const name = unique(prefix);
      invoke(scope.uuid, name, register);
      assert.equal(deregister(scope.uuid, name), true, `${prefix} should deregister`);
      assert.equal(deregister(scope.uuid, name), false, `${prefix} should not deregister twice`);
    }
  } finally {
    wasm.popScope(scope);
    scope.free();
    stack.free();
  }
});

test('scope-local registration wrappers cover invalid UUID and duplicate errors', () => {
  const stack = resetScopeStack();
  const scope = wasm.pushScope('registration_errors_scope', wasm.ScopeType.Function, null, 0, null, null);

  try {
    for (const [prefix, register, deregister, invoke] of scopeRegistrationCases(scope.uuid)) {
      const name = unique(`${prefix}_scope`);
      expectInvalidUuid(() => invoke('not-a-uuid', name, register));
      expectInvalidUuid(() => deregister('not-a-uuid', name));
      invoke(scope.uuid, name, register);
      expectAlreadyExists(() => invoke(scope.uuid, name, register));
      assert.equal(deregister(scope.uuid, name), true, `${prefix} duplicate scope registration should clean up`);
    }
  } finally {
    wasm.popScope(scope);
    scope.free();
    stack.free();
  }
});

test('WebAssembly scope-local llm stream execution intercept composes with next', async () => {
  const stack = resetScopeStack();
  const scope = wasm.pushScope('scope_stream_compose', wasm.ScopeType.Function, null, 0, null, null);
  const interceptName = unique('scope_llm_stream_exec');

  wasm.scopeRegisterLlmStreamExecutionIntercept(scope.uuid, interceptName, 10, async (request, next) => {
    const chunks = await next({
      ...request,
      content: {
        ...request.content,
        touchedByScope: true,
      },
    });
    return [
      ...chunks,
      {
        wrappedByScope: true,
      },
    ];
  });

  try {
    const stream = await wasm.llmStreamCallExecute(
      'scope_stream_llm',
      makeLlmRequest(),
      async (request) => [
        {
          downstream: request.content.touchedByScope === true,
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
        wrappedByScope: true,
      },
    ]);
    stream.free();
  } finally {
    wasm.scopeDeregisterLlmStreamExecutionIntercept(scope.uuid, interceptName);
    wasm.popScope(scope);
    scope.free();
    stack.free();
  }
});
