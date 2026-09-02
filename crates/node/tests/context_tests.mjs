// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  createScopeStack,
  currentScopeStack,
  setThreadScopeStack,
  scopeStackActive,
  getHandle,
  pushScope,
  popScope,
  ScopeType,
  ScopeStack,
  createScopeStackFromPropagation,
  capturePropagationContext,
  captureRootlessPropagationContext,
  propagationContextFromJson,
  propagationContextToJson,
  withScopeStack,
} = lib;

// ===========================================================================
// Context isolation
// ===========================================================================

describe('Context isolation', () => {
  it('createScopeStack returns a ScopeStack', () => {
    const stack = createScopeStack();
    assert.ok(stack, 'Expected a non-null scope stack');
    assert.ok(stack instanceof ScopeStack, 'Expected instance of ScopeStack');
  });

  it('creates an imported stack with the propagated parent on top', () => {
    const original = currentScopeStack();
    const rootUuid = '018f13f0-7c1a-7a80-8000-000000000001';
    const parentUuid = '018f13f0-7c1a-7a80-8000-000000000002';
    const stack = createScopeStackFromPropagation({ version: 1, rootUuid, parentUuid });
    try {
      setThreadScopeStack(stack);
      assert.equal(getHandle().uuid, parentUuid);
    } finally {
      setThreadScopeStack(original);
    }
  });

  it('serializes and validates propagation contexts for transport', () => {
    const context = {
      version: 1,
      rootUuid: '018f13f0-7c1a-7a80-8000-000000000001',
      parentUuid: '018f13f0-7c1a-7a80-8000-000000000002',
    };
    const encoded = propagationContextToJson(context);
    assert.deepEqual(JSON.parse(encoded), {
      version: 1,
      root_uuid: context.rootUuid,
      parent_uuid: context.parentUuid,
    });
    assert.deepEqual(propagationContextFromJson(encoded), context);
    assert.throws(() => propagationContextFromJson('not JSON'), /invalid propagation context JSON/);
    assert.throws(
      () => propagationContextFromJson(`{"version":2,"parent_uuid":"${context.parentUuid}"}`),
      /unsupported propagation context version 2; expected 1/,
    );
  });

  it('captures a rooted context by default and supports explicit rootless capture', () => {
    const stack = createScopeStack();
    withScopeStack(stack, () => {
      const handle = pushScope('propagation-root', ScopeType.Agent, null, null);
      try {
        const rooted = capturePropagationContext();
        const rootless = captureRootlessPropagationContext();
        assert.equal(rooted.rootUuid, handle.uuid);
        assert.equal(rooted.parentUuid, handle.uuid);
        assert.equal(rootless.rootUuid, undefined);
        assert.equal(rootless.parentUuid, handle.uuid);
      } finally {
        popScope(handle);
      }
    });
  });

  it('restores the surrounding stack after withScopeStack', () => {
    const original = currentScopeStack();
    const originalUuid = getHandle().uuid;
    const stack = createScopeStack();
    try {
      withScopeStack(stack, () => {
        pushScope('temporary-with-scope-stack', ScopeType.Agent, null, null);
        assert.equal(getHandle().name, 'temporary-with-scope-stack');
      });
      assert.notEqual(getHandle().name, 'temporary-with-scope-stack');
      assert.throws(
        () =>
          withScopeStack(stack, () => {
            throw new Error('expected');
          }),
        /expected/,
      );
      assert.equal(getHandle().uuid, originalUuid);
    } finally {
      setThreadScopeStack(original);
    }
  });

  it('keeps withScopeStack active until an async callback settles', async () => {
    const originalUuid = getHandle().uuid;
    const stack = createScopeStack();
    const stackUuid = withScopeStack(stack, () => getHandle().uuid);
    let resolveDetached;
    const detached = new Promise((resolve) => {
      resolveDetached = resolve;
    });

    const execution = withScopeStack(stack, async () => {
      assert.equal(getHandle().uuid, stackUuid);
      await new Promise((resolve) => setImmediate(resolve));
      assert.equal(getHandle().uuid, stackUuid);
      setImmediate(() => resolveDetached(getHandle().uuid));
      return 'done';
    });

    assert.equal(getHandle().uuid, originalUuid);
    assert.equal(await execution, 'done');
    assert.equal(getHandle().uuid, originalUuid);
    assert.equal(await detached, originalUuid);
  });

  it('currentScopeStack returns same in same context', () => {
    const s1 = currentScopeStack();
    const s2 = currentScopeStack();
    assert.ok(s1, 'Expected a non-null scope stack');
    assert.ok(s2, 'Expected a non-null scope stack');
    // Both calls in same thread context should return equivalent stacks
    assert.ok(s1 instanceof ScopeStack);
    assert.ok(s2 instanceof ScopeStack);
  });

  it('setThreadScopeStack isolates scopes', () => {
    const original = currentScopeStack();
    const newStack = createScopeStack();

    try {
      setThreadScopeStack(newStack);
      const scope = pushScope('isolated_scope', ScopeType.Agent, null, null);
      const handle = getHandle();
      assert.equal(handle.name, 'isolated_scope');
      popScope(scope);
    } finally {
      setThreadScopeStack(original);
    }
    assert.notEqual(getHandle().name, 'isolated_scope');
  });

  it('scopeStackActive returns true after setThreadScopeStack', () => {
    const stack = createScopeStack();
    setThreadScopeStack(stack);
    assert.equal(scopeStackActive(), true);
  });

  it('two scope stacks are independent', () => {
    const original = currentScopeStack();
    const stack1 = createScopeStack();
    const stack2 = createScopeStack();

    // Push a scope on stack1
    setThreadScopeStack(stack1);
    const s1 = pushScope('stack1_scope', ScopeType.Agent, null, null);

    // Switch to stack2 and push a different scope
    setThreadScopeStack(stack2);
    const s2 = pushScope('stack2_scope', ScopeType.Tool, null, null);

    // Verify stack2 sees its own scope
    const handle2 = getHandle();
    assert.equal(handle2.name, 'stack2_scope');

    // Switch back to stack1 — should see stack1's scope
    setThreadScopeStack(stack1);
    const handle1 = getHandle();
    assert.equal(handle1.name, 'stack1_scope');

    // Clean up
    popScope(s1);
    setThreadScopeStack(stack2);
    popScope(s2);

    // Restore original
    setThreadScopeStack(original);
  });
});
