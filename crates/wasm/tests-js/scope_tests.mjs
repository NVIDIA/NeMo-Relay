// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { currentScope, resetScopeStack, SCOPE_ATTR_PARALLEL, SCOPE_ATTR_RELOCATABLE, wasm } from './test_support.mjs';

test('WebAssembly scope stack exposes the generated root scope handle', () => {
  const stack = resetScopeStack();
  const root = wasm.getHandle();

  try {
    assert.equal(wasm.scopeStackActive(), true);
    assert.equal(root.scopeType, wasm.ScopeType.Agent);
    assert.equal(typeof root.uuid, 'string');
    assert.ok(root.uuid.length > 0);
    assert.equal(root.parentUuid, null);
    assert.equal(root.data, null);
    assert.equal(root.metadata, null);
  } finally {
    root.free();
    stack.free();
  }
});

test('WebAssembly pushScope preserves attributes, data, and metadata', () => {
  const stack = resetScopeStack();
  let scope;

  try {
    scope = wasm.pushScope(
      'pkg_scope',
      wasm.ScopeType.Function,
      null,
      SCOPE_ATTR_PARALLEL,
      {
        scope: true,
      },
      {
        source: 'js',
      },
    );
    assert.equal(scope.name, 'pkg_scope');
    assert.equal(scope.scopeType, wasm.ScopeType.Function);
    assert.equal(scope.attributes, SCOPE_ATTR_PARALLEL);
    assert.deepEqual(scope.data, {
      scope: true,
    });
    assert.deepEqual(scope.metadata, {
      source: 'js',
    });
    assert.equal(typeof scope.parentUuid, 'string');
  } finally {
    if (scope) {
      wasm.popScope(scope);
      scope.free();
    }
    stack.free();
  }
});

test('WebAssembly pushScope supports nullable inputs and root parent handles', () => {
  const stack = resetScopeStack();
  const root = currentScope();
  const rootUuid = root.uuid;
  let scope;

  try {
    root.free();
    scope = wasm.pushScope('optional_scope', wasm.ScopeType.Function, currentScope(), undefined, null, undefined);
    assert.equal(scope.parentUuid, rootUuid);
    assert.equal(scope.data, null);
    assert.equal(scope.metadata, null);
  } finally {
    if (scope) {
      wasm.popScope(scope);
      scope.free();
    }
    stack.free();
  }
});

test('WebAssembly withScope returns callback data for synchronous callbacks', async () => {
  const stack = resetScopeStack();

  try {
    const result = await wasm.withScope(
      'pkg_with_scope',
      wasm.ScopeType.Function,
      (handle) => ({
        name: handle.name,
        type: handle.scopeType,
        uuid: handle.uuid,
      }),
      null,
      SCOPE_ATTR_RELOCATABLE,
      {
        nested: true,
      },
      {
        origin: 'callback',
      },
    );

    assert.equal(result.name, 'pkg_with_scope');
    assert.equal(result.type, wasm.ScopeType.Function);
    assert.equal(typeof result.uuid, 'string');
  } finally {
    stack.free();
  }
});

test('WebAssembly withScope supports async callbacks', async () => {
  const stack = resetScopeStack();

  try {
    const result = await wasm.withScope(
      'async_scope',
      wasm.ScopeType.Function,
      async (handle) => ({
        uuid: handle.uuid,
        type: handle.scopeType,
      }),
      null,
      0,
      null,
      null,
    );

    assert.equal(result.type, wasm.ScopeType.Function);
    assert.equal(typeof result.uuid, 'string');
  } finally {
    stack.free();
  }
});

test('WebAssembly flushSubscribers succeeds as an API-parity no-op', () => {
  const stack = resetScopeStack();

  try {
    assert.equal(wasm.flushSubscribers(), undefined);
  } finally {
    stack.free();
  }
});
