// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { createRequire } from 'node:module';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const require = createRequire(import.meta.url);
const lib = require('../index.js');
const plugin = require('../plugin.js');

function capture(name) {
  const events = [];
  lib.registerSubscriber(name, (event) => events.push(event));
  return events;
}

async function waitFor(events, count) {
  for (let attempt = 0; attempt < 100 && events.length < count; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.ok(events.length >= count, `expected ${count} events, received ${events.length}`);
}

function assertSanitizerFieldsPreserved(event, expectedData, expectedMetadata = expectedData) {
  assert.deepEqual(event.data, expectedData);
  assert.equal(event.category_profile?.subtype, 'seeded');
  assert.deepEqual(event.metadata, expectedMetadata);
}

function assertSanitizerFieldsCleared(event) {
  assert.equal(event.data, null);
  assert.equal(event.category_profile, null);
  assert.equal(event.metadata, null);
}

async function initializeWithoutDiscoveredPluginConfig(config) {
  const previousDirectory = process.cwd();
  const directory = mkdtempSync(path.join(tmpdir(), 'nemo-relay-node-'));
  try {
    process.chdir(directory);
    return await plugin.initialize(config);
  } finally {
    process.chdir(previousDirectory);
    rmSync(directory, { recursive: true, force: true });
  }
}

describe('event sanitizer registries', () => {
  it('orders mark sanitizers and supports field removal', async () => {
    const events = capture('node-event-sanitize-order-sub');
    const calls = [];
    lib.registerMarkSanitizeGuardrail('node-event-first', 10, (event, fields) => {
      calls.push([event.name, fields.data]);
      return { ...fields, data: { stage: 'first' }, metadata: null };
    });
    lib.registerMarkSanitizeGuardrail('node-event-second', 20, (event, fields) => {
      calls.push([event.kind, fields.data]);
      return { ...fields, data: { stage: 'second' } };
    });
    try {
      lib.event('checkpoint', null, { secret: 'raw' }, { secret: 'raw' });
      await lib.flushSubscribers();
      await waitFor(events, 1);
    } finally {
      lib.deregisterMarkSanitizeGuardrail('node-event-first');
      lib.deregisterMarkSanitizeGuardrail('node-event-second');
      lib.deregisterSubscriber('node-event-sanitize-order-sub');
    }
    const mark = events.at(-1);
    assert.deepEqual(mark.data, { stage: 'second' });
    assert.equal(mark.metadata, null);
    assert.deepEqual(calls, [
      ['checkpoint', { secret: 'raw' }],
      ['mark', { stage: 'first' }],
    ]);
  });

  it('sanitizes scope start/end data, category profile, and metadata', async () => {
    const events = capture('node-event-sanitize-scope-sub');
    const sanitize = (_event, fields) => ({
      data: null,
      categoryProfile: { ...fields.categoryProfile, subtype: 'sanitized' },
      metadata: { safe: true },
    });
    lib.registerScopeSanitizeStartGuardrail('node-scope-start', 0, sanitize);
    lib.registerScopeSanitizeEndGuardrail('node-scope-end', 0, sanitize);
    try {
      const handle = lib.pushScope(
        'generic',
        lib.ScopeType.Custom,
        null,
        null,
        { secret: 'start' },
        { secret: 'start' },
        { secret: 'input' },
      );
      lib.popScope(handle, { secret: 'output' }, null, { secret: 'end' });
      await lib.flushSubscribers();
      await waitFor(events, 2);
    } finally {
      lib.deregisterScopeSanitizeStartGuardrail('node-scope-start');
      lib.deregisterScopeSanitizeEndGuardrail('node-scope-end');
      lib.deregisterSubscriber('node-event-sanitize-scope-sub');
    }
    const lifecycle = events.filter((event) => event.name === 'generic');
    assert.equal(lifecycle.length, 2);
    assert.ok(lifecycle.every((event) => event.data === null));
    assert.ok(lifecycle.every((event) => event.metadata.safe === true));
    assert.ok(lifecycle.every((event) => event.category_profile.subtype === 'sanitized'));
  });

  it('awaits Promise-returning mark sanitizers without making event() asynchronous', async () => {
    const events = capture('node-event-sanitize-promise-sub');
    let settled = false;
    lib.registerMarkSanitizeGuardrail('node-event-promise', 0, async (_event, fields) => {
      await new Promise((resolve) => setImmediate(resolve));
      settled = true;
      return { ...fields, data: { sanitized: true } };
    });
    try {
      const result = lib.event('promise-checkpoint', null, { raw: true });
      assert.equal(result, undefined);
      assert.equal(settled, false);
      await lib.flushSubscribers();
      await waitFor(events, 1);
    } finally {
      lib.deregisterMarkSanitizeGuardrail('node-event-promise');
      lib.deregisterSubscriber('node-event-sanitize-promise-sub');
    }
    assert.equal(settled, true);
    assert.deepEqual(events.at(-1).data, { sanitized: true });
  });

  it('publishes nested Promise sanitizer events before already queued events', async () => {
    const events = capture('node-event-sanitize-nested-order-sub');
    let sanitizerEntered;
    const entered = new Promise((resolve) => {
      sanitizerEntered = resolve;
    });
    let releaseSanitizer;
    const release = new Promise((resolve) => {
      releaseSanitizer = resolve;
    });
    lib.registerMarkSanitizeGuardrail('node-event-sanitize-nested-order', 0, async (event, fields) => {
      if (event.name === 'node-outer-event') {
        sanitizerEntered();
        await release;
        lib.withScopeStack(lib.createScopeStack(), () => lib.event('node-nested-event'));
      }
      return fields;
    });
    try {
      lib.event('node-outer-event');
      await entered;
      lib.event('node-later-event');
      releaseSanitizer();
      await lib.flushSubscribers();
      await waitFor(events, 3);
    } finally {
      lib.deregisterMarkSanitizeGuardrail('node-event-sanitize-nested-order');
      lib.deregisterSubscriber('node-event-sanitize-nested-order-sub');
    }
    assert.deepEqual(
      events.map((event) => event.name),
      ['node-outer-event', 'node-nested-event', 'node-later-event'],
    );
  });

  it('preserves the emitting scope stack across queued sanitizer awaits', async () => {
    const events = capture('node-event-sanitize-scope-context-sub');
    const originalStack = lib.currentScopeStack();
    const emitterStack = lib.createScopeStack();
    const unrelatedStack = lib.createScopeStack();
    const overrideStack = lib.createScopeStack();
    let overrideRootUuid;
    let emitterScopeUuid;
    let emitterScope;
    const observedParents = [];
    const observedOverrides = [];
    let sanitizerEntered;
    const entered = new Promise((resolve) => {
      sanitizerEntered = resolve;
    });
    let releaseSanitizer;
    const release = new Promise((resolve) => {
      releaseSanitizer = resolve;
    });

    lib.registerMarkSanitizeGuardrail('node-event-scope-context', 0, async (event, fields) => {
      if (event.name !== 'scope-context-original') {
        return fields;
      }
      observedParents.push(lib.getHandle().uuid);
      sanitizerEntered();
      await release;
      observedParents.push(lib.getHandle().uuid);
      lib.event('scope-context-nested', null, { originalParent: event.parent_uuid });
      observedOverrides.push(lib.withScopeStack(overrideStack, () => lib.getHandle().uuid));
      lib.setThreadScopeStack(overrideStack);
      observedOverrides.push(lib.getHandle().uuid);
      return fields;
    });

    try {
      overrideRootUuid = lib.withScopeStack(overrideStack, () => lib.getHandle().uuid);
      lib.withScopeStack(emitterStack, () => {
        emitterScope = lib.pushScope('scope-context-emitter', lib.ScopeType.Agent);
        emitterScopeUuid = emitterScope.uuid;
        lib.event('scope-context-original', null, {});
      });
      await entered;
      lib.withScopeStack(emitterStack, () => lib.popScope(emitterScope));
      lib.setThreadScopeStack(unrelatedStack);
      const unrelatedRootUuid = lib.getHandle().uuid;
      releaseSanitizer();

      await lib.flushSubscribers();
      await lib.flushSubscribers();
      await waitFor(events, 2);

      assert.deepEqual(observedParents, [emitterScopeUuid, emitterScopeUuid]);
      assert.deepEqual(observedOverrides, [overrideRootUuid, overrideRootUuid]);
      const nested = events.find((event) => event.name === 'scope-context-nested');
      assert.equal(nested.parent_uuid, emitterScopeUuid);
      assert.notEqual(nested.parent_uuid, unrelatedRootUuid);
    } finally {
      releaseSanitizer();
      lib.setThreadScopeStack(originalStack);
      lib.deregisterMarkSanitizeGuardrail('node-event-scope-context');
      lib.deregisterSubscriber('node-event-sanitize-scope-context-sub');
    }
  });

  it('preserves the ending scope across an async scope-end sanitizer', async () => {
    const events = capture('node-scope-end-context-sub');
    const observed = [];
    lib.registerScopeSanitizeEndGuardrail('node-scope-end-context', 0, async (_event, fields) => {
      observed.push(lib.getHandle().uuid);
      await new Promise((resolve) => setImmediate(resolve));
      observed.push(lib.getHandle().uuid);
      return fields;
    });
    const scope = lib.pushScope('node-ending-scope', lib.ScopeType.Agent);
    try {
      lib.popScope(scope);
      await lib.flushSubscribers();
      await waitFor(events, 2);
    } finally {
      lib.deregisterScopeSanitizeEndGuardrail('node-scope-end-context');
      lib.deregisterSubscriber('node-scope-end-context-sub');
    }
    assert.deepEqual(observed, [scope.uuid, scope.uuid]);
  });

  it('preserves snapshotted sanitizers after deregistration', async () => {
    const events = capture('node-event-sanitize-snapshot-sub');
    let blockerEntered;
    const entered = new Promise((resolve) => {
      blockerEntered = resolve;
    });
    let releaseBlocker;
    const release = new Promise((resolve) => {
      releaseBlocker = resolve;
    });
    lib.registerMarkSanitizeGuardrail('node-event-snapshot-blocker', 0, async (_event, fields) => {
      blockerEntered();
      await release;
      return fields;
    });
    lib.registerMarkSanitizeGuardrail('node-event-snapshot-target', 10, async (_event, fields) => {
      return { ...fields, data: { snapshotted: true } };
    });
    try {
      lib.event('snapshot-checkpoint', null, { raw: true });
      await entered;
      assert.equal(lib.deregisterMarkSanitizeGuardrail('node-event-snapshot-target'), true);
      releaseBlocker();
      await lib.flushSubscribers();
      await waitFor(events, 1);
    } finally {
      releaseBlocker();
      lib.deregisterMarkSanitizeGuardrail('node-event-snapshot-blocker');
      lib.deregisterMarkSanitizeGuardrail('node-event-snapshot-target');
      lib.deregisterSubscriber('node-event-sanitize-snapshot-sub');
    }
    assert.deepEqual(events.at(-1).data, { snapshotted: true });
  });

  it('waits for an in-flight sanitizer when flushed externally', async () => {
    const events = capture('node-event-sanitize-independent-flush-sub');
    let releaseSanitizer;
    let sanitizerEntered;
    const entered = new Promise((resolve) => {
      sanitizerEntered = resolve;
    });
    const release = new Promise((resolve) => {
      releaseSanitizer = resolve;
    });
    lib.registerMarkSanitizeGuardrail('node-event-independent-flush', 0, async (_event, fields) => {
      sanitizerEntered();
      await release;
      return fields;
    });
    try {
      lib.event('independent-flush-checkpoint', null, { raw: true });
      await entered;
      const flush = lib.flushSubscribers();
      const state = await Promise.race([
        flush.then(() => 'flushed'),
        new Promise((resolve) => setImmediate(() => resolve('pending'))),
      ]);
      assert.equal(state, 'pending');
      releaseSanitizer();
      await flush;
      await waitFor(events, 1);
    } finally {
      releaseSanitizer();
      lib.deregisterMarkSanitizeGuardrail('node-event-independent-flush');
      lib.deregisterSubscriber('node-event-sanitize-independent-flush-sub');
    }
  });

  it('queues managed event sanitizers without blocking execution', async () => {
    lib.registerSubscriber('node-event-queued-managed-sub', () => {});
    let blockerEntered;
    const entered = new Promise((resolve) => {
      blockerEntered = resolve;
    });
    let releaseBlocker;
    const release = new Promise((resolve) => {
      releaseBlocker = resolve;
    });
    let inlineSanitizerReturned = false;

    lib.registerMarkSanitizeGuardrail('node-event-queued-managed-blocker', 0, async (_event, fields) => {
      blockerEntered();
      await release;
      return fields;
    });
    lib.registerScopeSanitizeStartGuardrail('node-event-queued-managed', 0, async (_event, fields) => {
      inlineSanitizerReturned = true;
      return fields;
    });

    try {
      lib.event('queued-managed-blocker', null, { raw: true });
      await entered;
      const execution = lib.toolCallExecute('queued-managed-tool', {}, (args) => ({ result: args }));
      const executionState = await Promise.race([
        execution.then(() => 'executed'),
        new Promise((resolve) => setTimeout(() => resolve('blocked'), 250)),
      ]);
      assert.equal(executionState, 'executed');
      assert.equal(inlineSanitizerReturned, false);
      releaseBlocker();
      await lib.flushSubscribers();
      assert.equal(inlineSanitizerReturned, true);
    } finally {
      releaseBlocker();
      lib.deregisterMarkSanitizeGuardrail('node-event-queued-managed-blocker');
      lib.deregisterScopeSanitizeStartGuardrail('node-event-queued-managed');
      lib.deregisterSubscriber('node-event-queued-managed-sub');
    }
  });

  it('fails closed and records invalid sanitizer results', async () => {
    const events = capture('node-event-sanitize-invalid-sub');
    const invalidResults = {
      scalar: () => 'invalid',
      emptyObject: () => ({}),
      array: () => [],
      promise: () => Promise.resolve([]),
    };
    try {
      for (const [kind, sanitizer] of Object.entries(invalidResults)) {
        const name = `node-event-invalid-${kind}`;
        const seedName = `${name}-seed`;
        lib.clearLastCallbackError();
        lib.registerMarkSanitizeGuardrail(seedName, -1, (_event, fields) => ({
          ...fields,
          data: { kept: kind },
          categoryProfile: { subtype: 'seeded' },
          metadata: { kept: kind },
        }));
        lib.registerMarkSanitizeGuardrail(name, 0, sanitizer);
        try {
          lib.event(name, null, { kept: kind }, { kept: kind });
          await lib.flushSubscribers();
          await waitFor(events, Object.keys(invalidResults).indexOf(kind) + 1);
        } finally {
          lib.deregisterMarkSanitizeGuardrail(seedName);
          lib.deregisterMarkSanitizeGuardrail(name);
        }
        assertSanitizerFieldsCleared(events.at(-1));
        assert.match(lib.getLastCallbackError(), /invalid JS event sanitizer result/);
      }
    } finally {
      lib.deregisterSubscriber('node-event-sanitize-invalid-sub');
    }
  });

  it('uses the thread-safe callback path for managed tool events', async () => {
    const events = capture('node-event-sanitize-background-sub');
    lib.registerScopeSanitizeStartGuardrail('node-background-start', 0, (_event, fields) => ({
      ...fields,
      metadata: { background: true },
    }));
    try {
      await lib.toolCallExecute('background-tool', { raw: true }, (args) => ({ result: args }));
      await lib.flushSubscribers();
      await waitFor(events, 2);
    } finally {
      lib.deregisterScopeSanitizeStartGuardrail('node-background-start');
      lib.deregisterSubscriber('node-event-sanitize-background-sub');
    }
    const start = events.find(
      (event) => event.kind === 'scope' && event.name === 'background-tool' && event.scope_category === 'start',
    );
    assert.equal(start.metadata.background, true);
  });

  it('fails closed and records invalid queued sanitizer results', async () => {
    const events = capture('node-event-sanitize-background-invalid-sub');
    const invalidResults = {
      emptyObject: () => ({}),
      array: () => [],
      promise: () => Promise.resolve([]),
    };
    try {
      for (const [kind, sanitizer] of Object.entries(invalidResults)) {
        const name = `node-background-invalid-${kind}`;
        const seedName = `${name}-seed`;
        lib.clearLastCallbackError();
        lib.registerScopeSanitizeStartGuardrail(seedName, -1, (_event, fields) => ({
          ...fields,
          data: { kept: kind },
          categoryProfile: { ...fields.categoryProfile, subtype: 'seeded' },
          metadata: { kept: kind },
        }));
        lib.registerScopeSanitizeStartGuardrail(name, 0, sanitizer);
        try {
          await lib.toolCallExecute(name, { kept: kind }, (args) => ({ result: args }));
          await lib.flushSubscribers();
          await waitFor(events, (Object.keys(invalidResults).indexOf(kind) + 1) * 2);
        } finally {
          lib.deregisterScopeSanitizeStartGuardrail(seedName);
          lib.deregisterScopeSanitizeStartGuardrail(name);
        }
        const start = events.find(
          (event) => event.kind === 'scope' && event.name === name && event.scope_category === 'start',
        );
        assertSanitizerFieldsCleared(start);
        assert.match(lib.getLastCallbackError(), /invalid JS event sanitizer result/);
      }
    } finally {
      lib.deregisterSubscriber('node-event-sanitize-background-invalid-sub');
    }
  });

  it('fails closed when a queued sanitizer throws', async () => {
    const events = capture('node-event-sanitize-background-throw-sub');
    lib.clearLastCallbackError();
    lib.registerScopeSanitizeStartGuardrail('node-background-throw-seed', -1, (_event, fields) => ({
      ...fields,
      data: { kept: true },
      categoryProfile: { ...fields.categoryProfile, subtype: 'seeded' },
      metadata: { kept: true },
    }));
    lib.registerScopeSanitizeStartGuardrail('node-background-throw', 0, () => {
      throw new Error('background sanitizer boom');
    });
    try {
      await lib.toolCallExecute('background-throw-tool', { kept: true }, (args) => ({ result: args }));
      await lib.flushSubscribers();
      await waitFor(events, 2);
      const start = events.find(
        (event) => event.kind === 'scope' && event.name === 'background-throw-tool' && event.scope_category === 'start',
      );
      assertSanitizerFieldsCleared(start);
      assert.match(lib.getLastCallbackError() ?? '', /background sanitizer boom/i);
    } finally {
      lib.deregisterScopeSanitizeStartGuardrail('node-background-throw-seed');
      lib.deregisterScopeSanitizeStartGuardrail('node-background-throw');
      lib.deregisterSubscriber('node-event-sanitize-background-throw-sub');
      lib.clearLastCallbackError();
    }
  });

  it('inherits and cleans up scope-local mark sanitizers', async () => {
    const events = capture('node-event-sanitize-local-sub');
    const owner = lib.pushScope('owner', lib.ScopeType.Agent);
    lib.scopeRegisterMarkSanitizeGuardrail(owner.uuid, 'node-local-mark', 0, (_event, fields) => ({
      ...fields,
      data: { local: true },
    }));
    lib.event('inside', owner, { raw: true });
    const child = lib.pushScope('child', lib.ScopeType.Function, owner);
    lib.event('inherited', child, { raw: true });
    lib.popScope(child);
    lib.popScope(owner);
    lib.event('outside', null, { raw: true });
    await lib.flushSubscribers();
    await waitFor(events, 3);
    lib.deregisterSubscriber('node-event-sanitize-local-sub');
    const marks = Object.fromEntries(
      events.filter((event) => event.kind === 'mark').map((event) => [event.name, event]),
    );
    assert.deepEqual(marks.inside.data, { local: true });
    assert.deepEqual(marks.inherited.data, { local: true });
    assert.deepEqual(marks.outside.data, { raw: true });
  });

  it('cleans up plugin-owned event sanitizers', async () => {
    const kind = `node.test.event-sanitize.${Date.now()}`;
    const events = capture('node-event-sanitize-plugin-sub');
    plugin.register(kind, {
      register(_config, context) {
        context.registerMarkSanitizeGuardrail('mark', 0, (_event, fields) => ({
          ...fields,
          data: { plugin: true },
        }));
      },
    });
    try {
      await initializeWithoutDiscoveredPluginConfig({
        version: 1,
        components: [plugin.ComponentSpec(kind)],
      });
      lib.event('configured', null, { raw: true });
      await lib.flushSubscribers();
      await waitFor(events, 1);
      plugin.clear();
      lib.event('cleared', null, { raw: true });
      await lib.flushSubscribers();
      await waitFor(events, 2);
    } finally {
      plugin.clear();
      plugin.deregister(kind);
      lib.deregisterSubscriber('node-event-sanitize-plugin-sub');
    }
    const marks = Object.fromEntries(
      events.filter((event) => event.kind === 'mark').map((event) => [event.name, event]),
    );
    assert.deepEqual(marks.configured.data, { plugin: true });
    assert.deepEqual(marks.cleared.data, { raw: true });
  });

  it('fails closed when a plugin-owned sanitizer throws', async () => {
    const kind = `node.test.event-sanitize-throw.${Date.now()}`;
    const events = capture('node-event-sanitize-plugin-throw-sub');
    plugin.register(kind, {
      register(_config, context) {
        context.registerMarkSanitizeGuardrail('seed', -1, (_event, fields) => ({
          ...fields,
          data: { raw: true },
          categoryProfile: { subtype: 'seeded' },
          metadata: { raw: true },
        }));
        context.registerMarkSanitizeGuardrail('mark', 0, () => {
          throw new Error('plugin sanitizer boom');
        });
      },
    });
    lib.clearLastCallbackError();
    try {
      await initializeWithoutDiscoveredPluginConfig({
        version: 1,
        components: [plugin.ComponentSpec(kind)],
      });
      lib.event('plugin-throw', null, { raw: true }, { raw: true });
      await lib.flushSubscribers();
      await waitFor(events, 1);
      assertSanitizerFieldsCleared(events.at(-1));
      assert.match(lib.getLastCallbackError() ?? '', /plugin sanitizer boom/i);
    } finally {
      plugin.clear();
      plugin.deregister(kind);
      lib.deregisterSubscriber('node-event-sanitize-plugin-throw-sub');
      lib.clearLastCallbackError();
    }
  });
});
