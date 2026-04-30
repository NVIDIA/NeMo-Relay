// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';

import { expectClassError, makeLlmRequest, resetScopeStack, testsJsDir, wasm } from './test_support.mjs';

test('WebAssembly JS wrappers reject wrong handle classes', () => {
  const stack = resetScopeStack();
  const scope = wasm.pushScope('assert_scope', wasm.ScopeType.Function, null, 0, null, null);
  const llmHandle = wasm.llmCall('assert_llm', makeLlmRequest(), null, 0, null, null);
  const toolHandle = wasm.toolCall(
    'assert_tool',
    {
      ok: true,
    },
    null,
    0,
    null,
    null,
  );

  try {
    expectClassError(() => wasm.setThreadScopeStack({}));
    expectClassError(() => wasm.popScope({}));
    expectClassError(() => wasm.event('bad_parent', {}, null, null));
    expectClassError(() => wasm.event('bad_parent_map', new Map(), null, null));
    expectClassError(() => wasm.event('bad_parent_error', new Error('boom'), null, null));
    expectClassError(() => wasm.pushScope('bad_push', wasm.ScopeType.Function, {}, 0, null, null));
    expectClassError(() => wasm.toolCall('bad_tool', {}, {}, 0, null, null));
    expectClassError(() => wasm.toolCallEnd({}, {}, null, null));
    expectClassError(() => wasm.llmCall('bad_llm', makeLlmRequest(), {}, 0, null, null));
    expectClassError(() =>
      wasm.llmCallEnd(
        {},
        {
          role: 'assistant',
          content: 'nope',
          tool_calls: [],
        },
        null,
        null,
      ),
    );
    expectClassError(() => wasm.withScope('bad_scope', wasm.ScopeType.Function, () => null, {}, 0, null, null));
  } finally {
    llmHandle.free();
    toolHandle.free();
    wasm.popScope(scope);
    scope.free();
    stack.free();
  }
});

test('WebAssembly JS wrapper tolerates throwing subscriber callbacks', () => {
  const child = spawnSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `
        import assert from 'node:assert/strict';
        import { createRequire } from 'node:module';

        const require = createRequire(import.meta.url);
        const wasm = require('../pkg');
        const stack = wasm.createScopeStack();
        wasm.setThreadScopeStack(stack);

        const scope = wasm.pushScope('subscriber_throw_scope', wasm.ScopeType.Function, null, 0, null, null);
        const name = 'throwing_subscriber';
        wasm.registerSubscriber(name, () => {
          throw new Error('expected subscriber failure');
        });

        assert.equal(wasm.event('subscriber_throw_mark', scope, { ok: true }, null), undefined);
        assert.equal(wasm.deregisterSubscriber(name), true);
      `,
    ],
    {
      cwd: testsJsDir,
      encoding: 'utf8',
      env: process.env,
    },
  );

  assert.equal(child.status, 0, child.stderr || child.stdout);
});
