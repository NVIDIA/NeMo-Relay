// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { createRequire } from 'node:module';

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
      lib.flushSubscribers();
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
      lib.flushSubscribers();
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

  it('fails open and records invalid sanitizer results', async () => {
    const events = capture('node-event-sanitize-invalid-sub');
    lib.clearLastCallbackError();
    lib.registerMarkSanitizeGuardrail('node-event-invalid', 0, () => 'invalid');
    try {
      lib.event('invalid-result', null, { kept: true });
      lib.flushSubscribers();
      await waitFor(events, 1);
    } finally {
      lib.deregisterMarkSanitizeGuardrail('node-event-invalid');
      lib.deregisterSubscriber('node-event-sanitize-invalid-sub');
    }
    assert.deepEqual(events.at(-1).data, { kept: true });
    assert.match(lib.getLastCallbackError(), /event sanitizer callback failed/);
  });

  it('uses the thread-safe callback path for managed tool events', async () => {
    const events = capture('node-event-sanitize-background-sub');
    lib.registerScopeSanitizeStartGuardrail('node-background-start', 0, (_event, fields) => ({
      ...fields,
      metadata: { background: true },
    }));
    try {
      await lib.toolCallExecute('background-tool', { raw: true }, async (args) => args);
      lib.flushSubscribers();
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
    lib.flushSubscribers();
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
      await plugin.initialize({ version: 1, components: [plugin.ComponentSpec(kind)] });
      lib.event('configured', null, { raw: true });
      lib.flushSubscribers();
      await waitFor(events, 1);
      plugin.clear();
      lib.event('cleared', null, { raw: true });
      lib.flushSubscribers();
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
});
