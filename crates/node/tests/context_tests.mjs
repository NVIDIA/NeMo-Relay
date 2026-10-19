// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  createScopeStack,
  currentScopeStack,
  currentScopeStackSnapshot,
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

  it('preserves W3C trace context in the Relay transport payload', () => {
    const context = {
      version: 1,
      rootUuid: '018f13f0-7c1a-7a80-8000-000000000001',
      parentUuid: '018f13f0-7c1a-7a80-8000-000000000002',
      traceparent: '00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01',
      tracestate: 'vendor=value',
    };
    assert.deepEqual(propagationContextFromJson(propagationContextToJson(context)), context);
    const invalid = propagationContextFromJson(
      JSON.stringify({
        version: context.version,
        root_uuid: context.rootUuid,
        parent_uuid: context.parentUuid,
        traceparent: 'invalid',
        tracestate: context.tracestate,
      }),
    );
    assert.equal(invalid.traceparent, undefined);
    assert.equal(invalid.tracestate, undefined);
  });

  it('discards invalid tracestate while retaining the imported traceparent', () => {
    const traceparent = '00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01';
    for (const tracestate of [
      'vendor=bad\r\nX-Injected: yes',
      'vendor=one,vendor=two',
      Array.from({ length: 33 }, (_, i) => `v${i}=x`).join(','),
      'vendor=' + 'z'.repeat(300),
    ]) {
      const context = propagationContextFromJson(
        JSON.stringify({
          version: 1,
          parent_uuid: '018f13f0-7c1a-7a80-8000-000000000002',
          root_uuid: '018f13f0-7c1a-7a80-8000-000000000001',
          traceparent,
          tracestate,
        }),
      );
      assert.equal(context.traceparent, traceparent);
      assert.equal(context.tracestate, undefined);
      const decoded = propagationContextFromJson(propagationContextToJson(context));
      assert.deepEqual(decoded, context);
    }
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
        assert.equal(rootless.traceparent, undefined);
        assert.equal(rootless.tracestate, undefined);
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

describe('scope_stack_snapshot', () => {
  it('returns absence without activating a fresh process stack', () => {
    const execution = spawnSync(
      process.execPath,
      [
        '-e',
        `
      const assert = require('node:assert/strict');
      const library = require(${JSON.stringify(require.resolve('../index.js'))});
      assert.equal(library.currentScopeStackSnapshot(), null);
      assert.equal(library.currentScopeStackSnapshot(), null);
      assert.equal(library.scopeStackActive(), false);
    `,
      ],
      { encoding: 'utf8' },
    );
    assert.equal(execution.status, 0, execution.stderr);
  });

  it('orders actual handles and retains earlier values after pushes and pops', () => {
    withScopeStack(createScopeStack(), () => {
      const root = currentScopeStackSnapshot();
      assert.equal(root.scopes.length, 1);
      const agent = pushScope('agent', ScopeType.Agent, null, null);
      const functionScope = pushScope('function', ScopeType.Function, null, null);
      const tool = pushScope('tool', ScopeType.Tool, null, null);
      try {
        const retained = currentScopeStackSnapshot();
        const expected = [
          { scopeId: root.scopes[0].scopeId, scopeType: ScopeType.Agent },
          { scopeId: agent.uuid, scopeType: ScopeType.Agent },
          { scopeId: functionScope.uuid, scopeType: ScopeType.Function },
          { scopeId: tool.uuid, scopeType: ScopeType.Tool },
        ];
        assert.deepEqual(retained.scopes, expected);
        assert.deepEqual(currentScopeStackSnapshot().scopes, expected);
        popScope(tool);
        assert.deepEqual(currentScopeStackSnapshot().scopes, expected.slice(0, -1));
        const replacement = pushScope('replacement', ScopeType.Llm, null, null);
        try {
          assert.deepEqual(retained.scopes, expected);
        } finally {
          popScope(replacement);
        }
      } finally {
        popScope(functionScope);
        popScope(agent);
      }
    });
  });

  it('isolates awaited siblings and restores nested callback bindings', async () => {
    const inspectSibling = (name) =>
      withScopeStack(createScopeStack(), async () => {
        const agent = pushScope(name, ScopeType.Agent, null, null);
        try {
          await new Promise((resolve) => setImmediate(resolve));
          const before = currentScopeStackSnapshot();
          assert.equal(before.scopes.at(-1).scopeId, agent.uuid);
          await withScopeStack(createScopeStack(), async () => {
            await new Promise((resolve) => setImmediate(resolve));
            const nested = currentScopeStackSnapshot();
            assert.equal(nested.scopes.length, 1);
            assert.notEqual(nested.scopes[0].scopeId, before.scopes[0].scopeId);
          });
          assert.deepEqual(currentScopeStackSnapshot(), before);
          return agent.uuid;
        } finally {
          popScope(agent);
        }
      });
    const [first, second] = await Promise.all([inspectSibling('first'), inspectSibling('second')]);
    assert.notEqual(first, second);
  });
});

describe('scope_stack_snapshot propagation fixture', () => {
  it('preserves canonical UUIDs and synthetic scope types', () => {
    const entries = JSON.parse(
      readFileSync(new URL('../../types/tests/fixtures/scope_stack_snapshot.json', import.meta.url), 'utf8'),
    ).scopes;
    const rootUuid = entries[0].scope_id;
    const parentUuid = entries[3].scope_id;
    const stack = createScopeStackFromPropagation({ version: 1, rootUuid, parentUuid });
    withScopeStack(stack, () => {
      assert.deepEqual(currentScopeStackSnapshot().scopes, [
        { scopeId: rootUuid, scopeType: ScopeType.Agent },
        { scopeId: parentUuid, scopeType: ScopeType.Unknown },
      ]);
    });
  });
});

describe('scope_stack_snapshot failure cleanup', () => {
  it('restores the surrounding binding after an awaited callback rejection', async () => {
    await withScopeStack(createScopeStack(), async () => {
      const before = currentScopeStackSnapshot();
      await assert.rejects(
        withScopeStack(createScopeStack(), async () => {
          const agent = pushScope('failing-agent', ScopeType.Agent, null, null);
          try {
            await new Promise((resolve) => setImmediate(resolve));
            assert.equal(currentScopeStackSnapshot().scopes.at(-1).scopeId, agent.uuid);
            throw new Error('expected callback failure');
          } finally {
            popScope(agent);
          }
        }),
        /expected callback failure/,
      );
      assert.deepEqual(currentScopeStackSnapshot(), before);
    });
  });
});

describe('scope_stack_snapshot managed callbacks', () => {
  it('correlates awaited LLM and tool continuations within isolated Agents sharing call_1', async () => {
    const agentIdentifiers = new Map();
    const observations = new Map();
    const observe = (owner, boundary, toolCallId = null) => {
      const snapshot = currentScopeStackSnapshot();
      const nearestAgent = snapshot.scopes.findLast((entry) => entry.scopeType === ScopeType.Agent);
      assert.equal(nearestAgent.scopeId, agentIdentifiers.get(owner));
      observations.get(owner).push([boundary, toolCallId, nearestAgent.scopeId]);
    };
    lib.registerLlmExecutionIntercept('snapshot-llm', 1, async (request, _context, next) => {
      const owner = request.content.owner;
      observe(owner, 'llm-before');
      await new Promise((resolve) => setImmediate(resolve));
      const result = await next(request);
      observe(owner, 'llm-after');
      return result;
    });
    lib.registerToolExecutionIntercept('snapshot-tool', 1, async (context, next) => {
      const owner = context.args.owner;
      observe(owner, 'tool-before', context.toolCallId);
      await new Promise((resolve) => setImmediate(resolve));
      const result = await next(context.args);
      observe(owner, 'tool-after', context.toolCallId);
      return result;
    });
    const runAgent = (owner) =>
      withScopeStack(createScopeStack(), async () => {
        const agent = pushScope(owner, ScopeType.Agent, null, null);
        agentIdentifiers.set(owner, agent.uuid);
        observations.set(owner, []);
        const workflow = pushScope(`workflow-${owner}`, ScopeType.Function, null, null);
        try {
          const response = await lib.llmCallExecuteAsync(
            'snapshot-llm',
            { headers: {}, content: { messages: [], owner } },
            async () => ({ tool_call_id: 'call_1' }),
          );
          for (let attempt = 0; attempt < 2; attempt++) {
            const result = await lib.toolCallExecuteAsync(
              'snapshot-tool',
              { owner, attempt },
              async (argumentsValue) => ({ result: argumentsValue }),
              null,
              null,
              null,
              null,
              response.tool_call_id,
            );
            assert.deepEqual(result, { result: { owner, attempt } });
          }
          return agent.uuid;
        } finally {
          popScope(workflow);
          popScope(agent);
        }
      });
    try {
      const [first, second] = await Promise.all([runAgent('first'), runAgent('second')]);
      assert.notEqual(first, second);
      for (const owner of ['first', 'second']) {
        const identifier = agentIdentifiers.get(owner);
        assert.deepEqual(observations.get(owner), [
          ['llm-before', null, identifier],
          ['llm-after', null, identifier],
          ['tool-before', 'call_1', identifier],
          ['tool-after', 'call_1', identifier],
          ['tool-before', 'call_1', identifier],
          ['tool-after', 'call_1', identifier],
        ]);
      }
    } finally {
      lib.deregisterLlmExecutionIntercept('snapshot-llm');
      lib.deregisterToolExecutionIntercept('snapshot-tool');
    }
  });
});
