// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

const require = createRequire(import.meta.url);
const lib = require('../dist/index.js');

const {
  createScopeStack,
  currentScopeStack,
  currentScopeStackSnapshot,
  forkScopeStack,
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

describe('scope_stack_fork', { timeout: 5000 }, () => {
  it('does not retain completed callback context in plain stack handles', async () => {
    const events = [];
    lib.registerSubscriber('plain-handle-context', (runtimeEvent) => events.push(runtimeEvent));
    try {
      await withScopeStack(createScopeStack(), async () => {
        const parent = pushScope('plain-handle-parent', ScopeType.Agent);
        const original = currentScopeStackSnapshot().scopes;
        try {
          let plain;
          let forked;
          await lib.toolCallExecuteAsync('plain-handle-tool', {}, async () => {
            plain = currentScopeStack();
            forked = forkScopeStack();
            return { result: { completed: true } };
          });
          withScopeStack(plain, () => lib.event('plain-handle-later'));
          withScopeStack(forked, () => lib.event('plain-handle-explicit-fork'));
          await lib.flushSubscribers();
          const tool = events.find(
            (runtimeEvent) => runtimeEvent.name === 'plain-handle-tool' && runtimeEvent.scope_category === 'start',
          );
          const later = events.find((runtimeEvent) => runtimeEvent.name === 'plain-handle-later');
          const explicit = events.find((runtimeEvent) => runtimeEvent.name === 'plain-handle-explicit-fork');
          assert.ok(tool && later && explicit);
          assert.equal(later.parent_uuid, parent.uuid);
          assert.equal(explicit.parent_uuid, tool.uuid);
          assert.deepEqual(currentScopeStackSnapshot().scopes, original);
          assert.equal(lib.getLastCallbackError(), null);
        } finally {
          popScope(parent);
        }
      });
    } finally {
      await lib.flushSubscribers();
      lib.deregisterSubscriber('plain-handle-context');
    }
  });

  it('rejects direct captured-stack binding without a callback store', () => {
    const execution = spawnSync(
      process.execPath,
      [
        '-e',
        `
      const assert = require('node:assert/strict');
      const relay = require(${JSON.stringify(require.resolve('../dist/index.js'))});
      const source = relay.createScopeStack();
      let agent;
      const copied = relay.withScopeStack(source, () => {
        agent = relay.pushScope('direct-fork-agent', relay.ScopeType.Agent);
        return relay.forkScopeStack();
      });
      try {
        const previous = relay.currentScopeStackSnapshot();
        assert.throws(() => relay.setThreadScopeStack(copied), /use withScopeStack/);
        assert.deepEqual(relay.currentScopeStackSnapshot(), previous);
        relay.withScopeStack(copied, () => {
          assert.equal(relay.capturePropagationContext().parentUuid, agent.uuid);
        });
      } finally {
        relay.withScopeStack(source, () => relay.popScope(agent));
      }
    `,
      ],
      { encoding: 'utf8', timeout: 5000 },
    );
    assert.equal(execution.status, 0, execution.stderr);
  });

  for (const binding of ['withScopeStack', 'setThreadScopeStack']) {
    it(`keeps fork publication independent through ${binding}`, async () => {
      const completionOrder = [];
      const pendingFlushes = [];
      await withScopeStack(createScopeStack(), async () => {
        const agent = pushScope('independent-publication-agent', ScopeType.Agent);
        const original = currentScopeStackSnapshot().scopes;
        const copied = forkScopeStack();
        const delivered = [];
        lib.registerSubscriber('independent-publication-capture', (runtimeEvent) => {
          if (runtimeEvent.name === 'independent-publication-outer') delivered.push(runtimeEvent.name);
        });
        lib.registerMarkSanitizeGuardrail('independent-publication-sanitizer', 0, async (runtimeEvent, fields) => {
          if (runtimeEvent.name === 'independent-publication-outer') {
            const execute = async () => {
              await lib.toolCallExecuteAsync('independent-tool', {}, async () => {
                assert.deepEqual(currentScopeStackSnapshot().scopes, original);
                pendingFlushes.push(lib.flushSubscribers().then(() => completionOrder.push('independent-flushed')));
                return { result: { completed: true } };
              });
            };
            if (binding === 'withScopeStack') {
              await withScopeStack(copied, execute);
            } else {
              await withScopeStack(createScopeStack(), async () => {
                setThreadScopeStack(copied);
                await execute();
              });
            }
            completionOrder.push('sanitizer-finished');
          }
          return fields;
        });
        try {
          lib.event('independent-publication-outer');
          await lib.flushSubscribers();
          await Promise.all(pendingFlushes);
          assert.equal(lib.getLastCallbackError(), null);
          assert.deepEqual(completionOrder, ['sanitizer-finished', 'independent-flushed']);
          assert.deepEqual(delivered, ['independent-publication-outer']);
          assert.deepEqual(currentScopeStackSnapshot().scopes, original);
        } finally {
          lib.deregisterMarkSanitizeGuardrail('independent-publication-sanitizer');
          lib.deregisterSubscriber('independent-publication-capture');
          popScope(agent);
          await lib.flushSubscribers();
        }
      });
    });
  }

  it('rejects capture without an active context', () => {
    const execution = spawnSync(
      process.execPath,
      [
        '-e',
        `
      const assert = require('node:assert/strict');
      const relay = require(${JSON.stringify(require.resolve('../dist/index.js'))});
      assert.equal(relay.scopeStackActive(), false);
      assert.throws(() => relay.forkScopeStack(), /without an active Relay context/);
      assert.equal(relay.scopeStackActive(), false);
    `,
      ],
      { encoding: 'utf8', timeout: 5000 },
    );
    assert.equal(execution.status, 0, execution.stderr);
  });

  it('copies typed ancestry without binding or duplicating scope lifecycles', async () => {
    const events = [];
    lib.registerSubscriber('fork-lifecycles', (runtimeEvent) => events.push(runtimeEvent));
    try {
      await withScopeStack(createScopeStack(), async () => {
        const agent = pushScope('fork-lifecycle-agent', ScopeType.Agent);
        const workflow = pushScope('fork-lifecycle-function', ScopeType.Function);
        try {
          const original = currentScopeStackSnapshot().scopes;
          const copied = forkScopeStack();
          assert.deepEqual(currentScopeStackSnapshot().scopes, original);
          withScopeStack(copied, () => assert.deepEqual(currentScopeStackSnapshot().scopes, original));
          const failure = new Error('fork callback failed');
          assert.throws(
            () =>
              withScopeStack(copied, () => {
                throw failure;
              }),
            (error) => error === failure,
          );
          await assert.rejects(
            withScopeStack(copied, async () => {
              throw failure;
            }),
            (error) => error === failure,
          );
          assert.deepEqual(currentScopeStackSnapshot().scopes, original);
        } finally {
          popScope(workflow);
          popScope(agent);
        }
      });
      await lib.flushSubscribers();
      for (const name of ['fork-lifecycle-agent', 'fork-lifecycle-function']) {
        assert.deepEqual(
          events
            .filter((runtimeEvent) => runtimeEvent.name === name)
            .map((runtimeEvent) => runtimeEvent.scope_category),
          ['start', 'end'],
        );
      }
    } finally {
      await lib.flushSubscribers();
      lib.deregisterSubscriber('fork-lifecycles');
    }
  });

  it('isolates registration membership while retaining shared callback state', async () => {
    await withScopeStack(createScopeStack(), async () => {
      const agent = pushScope('fork-registrations', ScopeType.Agent);
      let invocations = 0;
      lib.scopeRegisterToolRequestIntercept(agent.uuid, 'inherited', 0, false, (_name, argumentsValue) => ({
        ...argumentsValue,
        invocation: ++invocations,
      }));
      const first = forkScopeStack();
      const second = forkScopeStack();
      try {
        lib.scopeRegisterToolRequestIntercept(agent.uuid, 'parent-only', 1, false, (_name, argumentsValue) => ({
          ...argumentsValue,
          parent: true,
        }));
        assert.deepEqual(await lib.toolRequestIntercepts('request', {}), { invocation: 1, parent: true });
        await withScopeStack(first, async () => {
          assert.equal(lib.scopeDeregisterToolRequestIntercept(agent.uuid, 'inherited'), true);
          assert.deepEqual(await lib.toolRequestIntercepts('request', {}), {});
        });
        assert.deepEqual(await withScopeStack(second, () => lib.toolRequestIntercepts('request', {})), {
          invocation: 2,
        });
        assert.equal(lib.scopeDeregisterToolRequestIntercept(agent.uuid, 'inherited'), true);
        assert.deepEqual(await withScopeStack(second, () => lib.toolRequestIntercepts('request', {})), {
          invocation: 3,
        });
        assert.deepEqual(await lib.toolRequestIntercepts('request', {}), { parent: true });
      } finally {
        popScope(agent);
      }
    });
  });

  it('isolates subscriber membership between the parent and forked stacks', async () => {
    const inheritedNames = [];
    const firstNames = [];
    const observeMarks = (names) => (runtimeEvent) => {
      if (runtimeEvent.kind === 'mark') names.push(runtimeEvent.name);
    };
    await withScopeStack(createScopeStack(), async () => {
      const agent = pushScope('fork-subscriber-membership', ScopeType.Agent);
      try {
        lib.scopeRegisterSubscriber(agent.uuid, 'inherited', observeMarks(inheritedNames));
        const first = forkScopeStack();
        const second = forkScopeStack();
        assert.equal(lib.scopeDeregisterSubscriber(agent.uuid, 'inherited'), true);
        lib.event('parent-mark');
        const outcomes = await Promise.allSettled([
          withScopeStack(first, async () => {
            assert.equal(lib.scopeDeregisterSubscriber(agent.uuid, 'inherited'), true);
            lib.scopeRegisterSubscriber(agent.uuid, 'first-only', observeMarks(firstNames));
            lib.event('first-mark');
          }),
          withScopeStack(second, async () => {
            lib.event('second-mark');
          }),
        ]);
        for (const outcome of outcomes) if (outcome.status === 'rejected') throw outcome.reason;
        await lib.flushSubscribers();
        assert.deepEqual(inheritedNames, ['second-mark']);
        assert.deepEqual(firstNames, ['first-mark']);
      } finally {
        popScope(agent);
        await lib.flushSubscribers();
      }
    });
  });

  it('keeps concurrent child scopes independent across awaits', async () => {
    await withScopeStack(createScopeStack(), async () => {
      const agent = pushScope('fork-siblings', ScopeType.Agent);
      try {
        const original = currentScopeStackSnapshot().scopes;
        const stacks = [forkScopeStack(), forkScopeStack()];
        const outcomes = await Promise.allSettled(
          stacks.map((stack, index) =>
            withScopeStack(stack, async () => {
              const child = pushScope(`fork-sibling-${index}`, ScopeType.Function);
              try {
                await new Promise((resolve) => setImmediate(resolve));
                assert.deepEqual(currentScopeStackSnapshot().scopes, [
                  ...original,
                  { scopeId: child.uuid, scopeType: ScopeType.Function },
                ]);
                return child.parentUuid;
              } finally {
                popScope(child);
              }
            }),
          ),
        );
        assert.deepEqual(outcomes, [
          { status: 'fulfilled', value: agent.uuid },
          { status: 'fulfilled', value: agent.uuid },
        ]);
        assert.deepEqual(currentScopeStackSnapshot().scopes, original);
      } finally {
        popScope(agent);
      }
    });
  });

  it('restores the caller after application cancellation', async () => {
    await withScopeStack(createScopeStack(), async () => {
      const original = currentScopeStackSnapshot().scopes;
      const controller = new AbortController();
      const cancellation = new Error('application cancelled');
      const execution = withScopeStack(forkScopeStack(), async () => {
        const child = pushScope('cancelled-fork', ScopeType.Function);
        try {
          await new Promise((resolve) => setImmediate(resolve));
          controller.signal.throwIfAborted();
        } finally {
          popScope(child);
        }
      });
      controller.abort(cancellation);
      await assert.rejects(execution, (error) => error === cancellation);
      assert.deepEqual(currentScopeStackSnapshot().scopes, original);
    });
  });

  it('preserves each Agent in forked model and tool callbacks sharing a tool-call ID', async () => {
    const runAgent = (name) =>
      withScopeStack(createScopeStack(), async () => {
        const agent = pushScope(name, ScopeType.Agent);
        const workflow = pushScope(`${name}-workflow`, ScopeType.Function);
        const observations = [];
        try {
          const original = currentScopeStackSnapshot().scopes;
          const observeFork = async (boundary) => {
            const propagation = capturePropagationContext();
            const traceparent = lib.captureTraceparent();
            await withScopeStack(forkScopeStack(), async () => {
              await new Promise((resolve) => setImmediate(resolve));
              assert.deepEqual(currentScopeStackSnapshot().scopes, original);
              assert.equal(capturePropagationContext().parentUuid, propagation.parentUuid);
              assert.equal(lib.captureTraceparent(), traceparent);
              const nested = pushScope(`${boundary}-nested`, ScopeType.Function);
              try {
                assert.equal(capturePropagationContext().parentUuid, nested.uuid);
              } finally {
                popScope(nested);
              }
              assert.equal(capturePropagationContext().parentUuid, propagation.parentUuid);
              observations.push([boundary, agent.uuid]);
            });
          };
          const response = await withScopeStack(forkScopeStack(), () =>
            lib.llmCallExecuteAsync('fork-model', { headers: {}, content: { messages: [] } }, async () => {
              await observeFork('model');
              return { tool_call_id: 'shared-call' };
            }),
          );
          const result = await withScopeStack(forkScopeStack(), () =>
            lib.toolCallExecuteAsync(
              'fork-tool',
              { tool_call_id: response.tool_call_id },
              async (argumentsValue) => {
                await observeFork('tool');
                return { result: argumentsValue };
              },
              null,
              null,
              null,
              null,
              response.tool_call_id,
            ),
          );
          assert.deepEqual(result.result, { tool_call_id: 'shared-call' });
          assert.deepEqual(observations, [
            ['model', agent.uuid],
            ['tool', agent.uuid],
          ]);
          return agent.uuid;
        } finally {
          popScope(workflow);
          popScope(agent);
        }
      });
    const outcomes = await Promise.allSettled([runAgent('first-fork-agent'), runAgent('second-fork-agent')]);
    for (const outcome of outcomes) if (outcome.status === 'rejected') throw outcome.reason;
    assert.notEqual(outcomes[0].value, outcomes[1].value);
  });

  for (const earlyClose of [false, true]) {
    it(`retains fork context through streaming ${earlyClose ? 'early close' : 'completion'}`, async () => {
      await withScopeStack(createScopeStack(), async () => {
        const agent = pushScope('fork-stream-agent', ScopeType.Agent);
        const original = currentScopeStackSnapshot().scopes;
        let cleanups = 0;
        let finalizations = 0;
        try {
          await withScopeStack(forkScopeStack(), async () => {
            const stream = await lib.llmStreamCallExecute(
              'fork-stream',
              { headers: {}, content: { messages: [] } },
              async function* (_request, signal) {
                try {
                  assert.deepEqual(currentScopeStackSnapshot().scopes, original);
                  yield { index: 0 };
                  if (earlyClose) {
                    await new Promise((resolve) => {
                      if (signal.aborted) resolve();
                      else signal.addEventListener('abort', resolve, { once: true });
                    });
                  } else {
                    await new Promise((resolve) => setImmediate(resolve));
                    assert.deepEqual(currentScopeStackSnapshot().scopes, original);
                    yield { index: 1 };
                  }
                } finally {
                  assert.deepEqual(currentScopeStackSnapshot().scopes, original);
                  cleanups++;
                }
              },
              null,
              () => {
                finalizations++;
                return {};
              },
            );
            try {
              assert.deepEqual(await stream.next(), { index: 0 });
              if (!earlyClose) {
                assert.deepEqual(await stream.next(), { index: 1 });
                assert.equal(await stream.next(), null);
              }
            } finally {
              await stream.close();
            }
          });
          assert.equal(cleanups, 1);
          assert.equal(finalizations, 1);
          assert.deepEqual(currentScopeStackSnapshot().scopes, original);
        } finally {
          popScope(agent);
        }
      });
    });
  }

  it('routes publications from an awaited event-sanitizer fork', async () => {
    const names = [];
    await withScopeStack(createScopeStack(), async () => {
      const agent = pushScope('fork-publication-agent', ScopeType.Agent);
      const original = currentScopeStackSnapshot().scopes;
      lib.registerSubscriber('fork-publications', (runtimeEvent) => {
        if (runtimeEvent.name.startsWith('fork-publication-')) names.push(runtimeEvent.name);
      });
      lib.registerMarkSanitizeGuardrail('fork-publication-sanitizer', 0, async (runtimeEvent, fields) => {
        if (runtimeEvent.name === 'fork-publication-parent') {
          await withScopeStack(forkScopeStack(), async () => {
            await new Promise((resolve) => setImmediate(resolve));
            assert.deepEqual(currentScopeStackSnapshot().scopes, original);
            lib.event('fork-publication-child');
          });
        }
        return fields;
      });
      try {
        lib.event('fork-publication-parent');
        await lib.flushSubscribers();
        assert.deepEqual(names, ['fork-publication-parent', 'fork-publication-child']);
      } finally {
        popScope(agent);
        await lib.flushSubscribers();
        lib.deregisterMarkSanitizeGuardrail('fork-publication-sanitizer');
        lib.deregisterSubscriber('fork-publications');
      }
    });
  });
});

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
      const library = require(${JSON.stringify(require.resolve('../dist/index.js'))});
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
