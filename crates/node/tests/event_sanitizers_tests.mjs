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

  it('does not deadlock when an async sanitizer flushes subscribers', async () => {
    const events = capture('node-event-sanitize-reentrant-flush-sub');
    let flushReturned = false;
    lib.registerMarkSanitizeGuardrail('node-event-reentrant-flush', 0, async (_event, fields) => {
      await lib.flushSubscribers();
      flushReturned = true;
      return fields;
    });
    try {
      lib.event('reentrant-flush-checkpoint', null, { raw: true });
      await lib.flushSubscribers();
      await waitFor(events, 1);
    } finally {
      lib.deregisterMarkSanitizeGuardrail('node-event-reentrant-flush');
      lib.deregisterSubscriber('node-event-sanitize-reentrant-flush-sub');
    }
    assert.equal(flushReturned, true);
  });

  it('does not treat an unrelated flush as sanitizer re-entrancy', async () => {
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

  it('treats inline managed sanitizers as real flush barriers', async () => {
    lib.registerSubscriber('node-event-inline-flush-sub', () => {});
    let blockerEntered;
    const entered = new Promise((resolve) => {
      blockerEntered = resolve;
    });
    let releaseBlocker;
    const release = new Promise((resolve) => {
      releaseBlocker = resolve;
    });
    let inlineFlushReturned = false;

    lib.registerMarkSanitizeGuardrail('node-event-inline-flush-blocker', 0, async (_event, fields) => {
      blockerEntered();
      await release;
      return fields;
    });
    lib.registerScopeSanitizeStartGuardrail('node-event-inline-flush', 0, async (_event, fields) => {
      await lib.flushSubscribers();
      inlineFlushReturned = true;
      return fields;
    });

    try {
      lib.event('inline-flush-blocker', null, { raw: true });
      await entered;
      const execution = lib.toolCallExecute('inline-flush-tool', {}, (args) => args);
      await new Promise((resolve) => setTimeout(resolve, 50));
      assert.equal(inlineFlushReturned, false);
      releaseBlocker();
      await execution;
      assert.equal(inlineFlushReturned, true);
      await lib.flushSubscribers();
    } finally {
      releaseBlocker();
      lib.deregisterMarkSanitizeGuardrail('node-event-inline-flush-blocker');
      lib.deregisterScopeSanitizeStartGuardrail('node-event-inline-flush');
      lib.deregisterSubscriber('node-event-inline-flush-sub');
    }
  });

  it('clears sanitizer re-entrancy in async descendants after settlement', async () => {
    const events = capture('node-event-sanitize-descendant-flush-sub');
    let secondSanitizerEntered;
    const secondEntered = new Promise((resolve) => {
      secondSanitizerEntered = resolve;
    });
    let releaseSecondSanitizer;
    const releaseSecond = new Promise((resolve) => {
      releaseSecondSanitizer = resolve;
    });
    let descendantFlushStarted;
    const flushStarted = new Promise((resolve) => {
      descendantFlushStarted = resolve;
    });
    let descendantFlush;
    const flushed = new Promise((resolve, reject) => {
      descendantFlush = { resolve, reject };
    });
    lib.registerMarkSanitizeGuardrail('node-event-descendant-flush', 0, async (event, fields) => {
      if (event.name === 'descendant-flush-origin') {
        setTimeout(async () => {
          await secondEntered;
          lib.flushSubscribers().then(descendantFlush.resolve, descendantFlush.reject);
          descendantFlushStarted();
        }, 0);
      } else if (event.name === 'descendant-flush-blocked') {
        secondSanitizerEntered();
        await releaseSecond;
      }
      return fields;
    });
    try {
      lib.event('descendant-flush-origin', null, { raw: true });
      lib.event('descendant-flush-blocked', null, { raw: true });
      await secondEntered;
      await flushStarted;
      const state = await Promise.race([
        flushed.then(() => 'flushed'),
        new Promise((resolve) => setTimeout(() => resolve('pending'), 50)),
      ]);
      assert.equal(state, 'pending');
      releaseSecondSanitizer();
      await flushed;
      await waitFor(events, 2);
    } finally {
      releaseSecondSanitizer();
      lib.deregisterMarkSanitizeGuardrail('node-event-descendant-flush');
      lib.deregisterSubscriber('node-event-sanitize-descendant-flush-sub');
    }
  });

  it('fails open and records invalid sanitizer results', async () => {
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
        assertSanitizerFieldsPreserved(events.at(-1), { kept: kind });
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
      await lib.toolCallExecute('background-tool', { raw: true }, (args) => args);
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

  it('fails open and records invalid queued sanitizer results', async () => {
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
          await lib.toolCallExecute(name, { kept: kind }, (args) => args);
          await lib.flushSubscribers();
          await waitFor(events, (Object.keys(invalidResults).indexOf(kind) + 1) * 2);
        } finally {
          lib.deregisterScopeSanitizeStartGuardrail(seedName);
          lib.deregisterScopeSanitizeStartGuardrail(name);
        }
        const start = events.find(
          (event) => event.kind === 'scope' && event.name === name && event.scope_category === 'start',
        );
        assertSanitizerFieldsPreserved(start, { kept: kind });
        assert.match(lib.getLastCallbackError(), /invalid JS event sanitizer result/);
      }
    } finally {
      lib.deregisterSubscriber('node-event-sanitize-background-invalid-sub');
    }
  });

  it('fails open when a queued sanitizer throws', async () => {
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
      await lib.toolCallExecute('background-throw-tool', { kept: true }, (args) => args);
      await lib.flushSubscribers();
      await waitFor(events, 2);
      const start = events.find(
        (event) => event.kind === 'scope' && event.name === 'background-throw-tool' && event.scope_category === 'start',
      );
      assertSanitizerFieldsPreserved(start, { kept: true });
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

  it('fails open when a plugin-owned sanitizer throws', async () => {
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
      assertSanitizerFieldsPreserved(events.at(-1), { raw: true });
      assert.match(lib.getLastCallbackError() ?? '', /plugin sanitizer boom/i);
    } finally {
      plugin.clear();
      plugin.deregister(kind);
      lib.deregisterSubscriber('node-event-sanitize-plugin-throw-sub');
      lib.clearLastCallbackError();
    }
  });
});
