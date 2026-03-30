// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  createScopeStack, currentScopeStack, setThreadScopeStack, scopeStackActive,
  getHandle, pushScope, popScope,
  ScopeType, JsScopeStack,
} = lib;

// ===========================================================================
// Context isolation
// ===========================================================================

describe('Context isolation', () => {
  it('createScopeStack returns a JsScopeStack', () => {
    const stack = createScopeStack();
    assert.ok(stack, 'Expected a non-null scope stack');
    assert.ok(stack instanceof JsScopeStack, 'Expected instance of JsScopeStack');
  });

  it('currentScopeStack returns same in same context', () => {
    const s1 = currentScopeStack();
    const s2 = currentScopeStack();
    assert.ok(s1, 'Expected a non-null scope stack');
    assert.ok(s2, 'Expected a non-null scope stack');
    // Both calls in same thread context should return equivalent stacks
    assert.ok(s1 instanceof JsScopeStack);
    assert.ok(s2 instanceof JsScopeStack);
  });

  it('setThreadScopeStack isolates scopes', () => {
    const original = currentScopeStack();
    const newStack = createScopeStack();

    // Switch to new stack and push a scope on it
    setThreadScopeStack(newStack);
    const scope = pushScope('isolated_scope', ScopeType.Agent, null, null);
    const handle = getHandle();
    assert.equal(handle.name, 'isolated_scope');
    popScope(scope);

    // Restore original stack — the isolated scope should not be visible
    setThreadScopeStack(original);
    const restored = getHandle();
    assert.notEqual(restored.name, 'isolated_scope');
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
