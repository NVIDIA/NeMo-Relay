// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';

const configHome = mkdtempSync(path.join(tmpdir(), 'nemo-relay-node-config-'));
process.env.XDG_CONFIG_HOME = configHome;
process.on('exit', () => rmSync(configHome, { recursive: true, force: true }));

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

describe('event metadata injector bindings', () => {
  it('preserves insertion order while isolating callback failures and invalid output', async () => {
    const events = capture('node-event-metadata-global-sub');
    lib.clearLastCallbackError();
    lib.registerEventMetadataInjector('node-event-metadata-sync', 10, (event) => ({
      'node.existing': 'ignored',
      'node.injector.shared': 'sync-first',
      'node.injector.sync': event.name,
    }));
    lib.registerEventMetadataInjector('node-event-metadata-async', 20, async () => ({
      'node.injector.async': true,
      'node.injector.shared': 'async-later',
    }));
    lib.registerEventMetadataInjector('node-event-metadata-failure', 30, () => {
      throw new Error('node injector failure');
    });
    lib.registerEventMetadataInjector('node-event-metadata-after-failure', 40, () => ({
      'node.injector.after_failure': 'added',
    }));
    lib.registerEventMetadataInjector('node-event-metadata-invalid', 50, () => ['not', 'a', 'mapping']);
    try {
      lib.event('node-event-metadata-global', null, null, { 'node.existing': 'preserved' });
      await lib.flushSubscribers();
      await waitFor(events, 1);
    } finally {
      lib.deregisterEventMetadataInjector('node-event-metadata-invalid');
      lib.deregisterEventMetadataInjector('node-event-metadata-after-failure');
      lib.deregisterEventMetadataInjector('node-event-metadata-failure');
      lib.deregisterEventMetadataInjector('node-event-metadata-async');
      lib.deregisterEventMetadataInjector('node-event-metadata-sync');
      lib.deregisterSubscriber('node-event-metadata-global-sub');
    }

    assert.deepEqual(events.at(-1).metadata, {
      'node.existing': 'preserved',
      'node.injector.after_failure': 'added',
      'node.injector.async': true,
      'node.injector.shared': 'sync-first',
      'node.injector.sync': 'node-event-metadata-global',
    });
    assert.match(lib.getLastCallbackError() ?? '', /invalid JavaScript event metadata injector result/i);
    lib.clearLastCallbackError();
  });

  it('applies and deregisters scope-local callbacks', async () => {
    const events = capture('node-event-metadata-scope-sub');
    const owner = lib.pushScope('node-event-metadata-owner', lib.ScopeType.Agent);
    lib.scopeRegisterEventMetadataInjector(owner.uuid, 'node-event-metadata-local-first', 10, (event) => ({
      'node.injector.scope_local': event.name,
      'node.injector.scope_order': 'first',
    }));
    lib.scopeRegisterEventMetadataInjector(owner.uuid, 'node-event-metadata-local-later', 20, () => ({
      'node.injector.scope_order': 'later',
    }));
    lib.event('node-event-metadata-before-deregister', owner);
    assert.equal(lib.scopeDeregisterEventMetadataInjector(owner.uuid, 'node-event-metadata-local-first'), true);
    assert.equal(lib.scopeDeregisterEventMetadataInjector(owner.uuid, 'node-event-metadata-local-first'), false);
    assert.equal(lib.scopeDeregisterEventMetadataInjector(owner.uuid, 'node-event-metadata-local-later'), true);
    lib.event('node-event-metadata-after-deregister', owner);
    lib.popScope(owner);
    await lib.flushSubscribers();
    await waitFor(events, 4);
    lib.deregisterSubscriber('node-event-metadata-scope-sub');

    const marks = Object.fromEntries(
      events.filter((event) => event.kind === 'mark').map((event) => [event.name, event]),
    );
    assert.deepEqual(marks['node-event-metadata-before-deregister'].metadata, {
      'node.injector.scope_local': 'node-event-metadata-before-deregister',
      'node.injector.scope_order': 'first',
    });
    assert.equal(marks['node-event-metadata-after-deregister'].metadata, null);
  });

  it('accepts numeric arrays with integer and fractional values at runtime', async () => {
    const events = capture('node-event-metadata-numeric-sub');
    lib.registerEventMetadataInjector('node-event-metadata-integers', 10, () => ({
      'node.injector.integers': [1, 2],
    }));
    lib.registerEventMetadataInjector('node-event-metadata-doubles', 20, () => ({
      'node.injector.doubles': [1.25, 2.5],
    }));
    lib.registerEventMetadataInjector('node-event-metadata-mixed-numbers', 30, () => ({
      'node.injector.mixed_numbers': [1, 2.5],
    }));
    try {
      lib.event('node-event-metadata-homogeneous-numeric-arrays');
      await lib.flushSubscribers();
      await waitFor(events, 1);
    } finally {
      lib.deregisterEventMetadataInjector('node-event-metadata-mixed-numbers');
      lib.deregisterEventMetadataInjector('node-event-metadata-doubles');
      lib.deregisterEventMetadataInjector('node-event-metadata-integers');
      lib.deregisterSubscriber('node-event-metadata-numeric-sub');
    }

    assert.deepEqual(events.at(-1).metadata, {
      'node.injector.doubles': [1.25, 2.5],
      'node.injector.integers': [1, 2],
      'node.injector.mixed_numbers': [1, 2.5],
    });
  });

  it('cleans up plugin-owned callbacks', async () => {
    const kind = `node.test.event-metadata.${Date.now()}`;
    const events = capture('node-event-metadata-plugin-sub');
    plugin.register(kind, {
      register(config, context) {
        context.registerEventMetadataInjector('configured', 10, () => config.metadata);
      },
    });
    try {
      await initializeWithoutDiscoveredPluginConfig({
        version: 1,
        components: [
          plugin.ComponentSpec(kind, {
            metadata: { 'node.injector.plugin': 'configured' },
          }),
        ],
      });
      lib.event('node-event-metadata-plugin-configured');
      await lib.flushSubscribers();
      await waitFor(events, 1);

      plugin.clear();
      lib.event('node-event-metadata-plugin-cleared');
      await lib.flushSubscribers();
      await waitFor(events, 2);
    } finally {
      plugin.clear();
      plugin.deregister(kind);
      lib.deregisterSubscriber('node-event-metadata-plugin-sub');
    }

    const marks = Object.fromEntries(
      events.filter((event) => event.kind === 'mark').map((event) => [event.name, event]),
    );
    assert.deepEqual(marks['node-event-metadata-plugin-configured'].metadata, {
      'node.injector.plugin': 'configured',
    });
    assert.equal(marks['node-event-metadata-plugin-cleared'].metadata, null);
  });
});
