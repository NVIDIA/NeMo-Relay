// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, readdirSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const plugin = require('../plugin.js');
const observability = require('../observability.js');
const { ScopeType, pushScope, popScope, event } = require('../index.js');

function tempDir(prefix) {
  return mkdtempSync(join(tmpdir(), `nemo-relay-${prefix}-`));
}

describe('observability plugin helpers', () => {
  it('builds defaults and plugin component shape', () => {
    assert.deepEqual(observability.defaultConfig(), { version: 1 });
    assert.deepEqual(observability.atofConfig(), { enabled: false, mode: 'append' });
    assert.deepEqual(observability.atifConfig(), {
      enabled: false,
      agent_name: 'NeMo Relay',
      model_name: 'unknown',
      filename_template: 'nemo-relay-atif-{session_id}.json',
      subagent_export_mode: 'embedded',
    });
    assert.deepEqual(observability.otlpConfig(), {
      enabled: false,
      transport: 'http_binary',
      headers: {},
      resource_attributes: {},
      service_name: 'nemo-relay',
      timeout_millis: 3000,
    });

    const component = observability.ComponentSpec({ version: 1, atof: observability.atofConfig() });
    assert.equal(component.kind, observability.OBSERVABILITY_PLUGIN_KIND);
    assert.equal(component.enabled, true);
  });

  it('lists builtin observability kind and validates bad values', () => {
    assert.equal(plugin.listKinds().includes(observability.OBSERVABILITY_PLUGIN_KIND), true);
    const report = plugin.validate({
      version: 1,
      components: [
        observability.ComponentSpec({
          version: 1,
          atof: observability.atofConfig({ mode: 'bad' }),
          atif: observability.atifConfig({ filename_template: 'missing-placeholder.json' }),
        }),
      ],
    });
    assert.deepEqual(report.diagnostics.map((diagnostic) => diagnostic.field).sort(), ['filename_template', 'mode']);
  });

  it('activates ATOF and ATIF file sinks', async () => {
    const outputDirectory = tempDir('node-observability-plugin');
    const config = {
      version: 1,
      atof: observability.atofConfig({
        enabled: true,
        output_directory: outputDirectory,
        filename: 'events.jsonl',
        mode: 'overwrite',
      }),
      atif: observability.atifConfig({
        enabled: true,
        agent_name: 'node-agent',
        agent_version: '1.2.3',
        model_name: 'node-model',
        tool_definitions: [{ name: 'search' }],
        extra: { binding: 'node' },
        output_directory: outputDirectory,
        filename_template: 'trajectory-{session_id}.json',
      }),
    };

    await plugin.initialize({
      version: 1,
      components: [observability.ComponentSpec(config)],
    });
    let scope = null;
    try {
      scope = pushScope('node-observability-agent', ScopeType.Agent, null, null, null, null, { agent: true });
      event('node-mark', scope, { step: 1 }, null);
      popScope(scope, { done: true });
      scope = null;
    } finally {
      plugin.clear();
      if (scope) {
        popScope(scope, { done: true });
      }
    }

    const records = readFileSync(join(outputDirectory, 'events.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
    assert.deepEqual(
      records.map((record) => record.kind),
      ['scope', 'mark', 'scope'],
    );

    const trajectory = JSON.parse(readFileSync(join(outputDirectory, `trajectory-${records[0].uuid}.json`), 'utf8'));
    assert.equal(trajectory.agent.name, 'node-agent');
    assert.equal(trajectory.agent.version, '1.2.3');
    assert.equal(trajectory.agent.model_name, 'node-model');
    assert.equal(trajectory.agent.tool_definitions[0].name, 'search');
    assert.equal(trajectory.agent.extra.binding, 'node');
    assert.match(JSON.stringify(trajectory.extra), /node-observability-agent/);
  });

  it('splits ATIF files for multiple top-level agent scopes', async () => {
    const outputDirectory = tempDir('node-observability-plugin-multi-agent');
    const config = {
      version: 1,
      atif: observability.atifConfig({
        enabled: true,
        output_directory: outputDirectory,
        filename_template: 'trajectory-{session_id}.json',
      }),
    };

    await plugin.initialize({
      version: 1,
      components: [observability.ComponentSpec(config)],
    });

    let first = null;
    let nested = null;
    let second = null;
    let firstUuid = null;
    let secondUuid = null;
    try {
      first = pushScope('node-first-agent', ScopeType.Agent, null, null, null, null, { agent: 'first' });
      firstUuid = first.uuid;
      event('node-first-mark', first, { agent: 'first' }, null);
      nested = pushScope('node-nested-agent', ScopeType.Agent, null, null, null, null, { agent: 'nested' });
      event('node-nested-mark', nested, { agent: 'nested' }, null);
      popScope(nested, { done: true });
      nested = null;
      popScope(first, { done: true });
      first = null;

      second = pushScope('node-second-agent', ScopeType.Agent, null, null, null, null, { agent: 'second' });
      secondUuid = second.uuid;
      event('node-second-mark', second, { agent: 'second' }, null);
      popScope(second, { done: true });
      second = null;
    } finally {
      plugin.clear();
      if (nested) {
        popScope(nested, { done: true });
      }
      if (first) {
        popScope(first, { done: true });
      }
      if (second) {
        popScope(second, { done: true });
      }
    }

    const files = readdirSync(outputDirectory).filter((name) => name.startsWith('trajectory-'));
    assert.equal(files.length, 2);

    const firstTrajectory = JSON.parse(readFileSync(join(outputDirectory, `trajectory-${firstUuid}.json`), 'utf8'));
    const secondTrajectory = JSON.parse(readFileSync(join(outputDirectory, `trajectory-${secondUuid}.json`), 'utf8'));
    const firstPayload = JSON.stringify(firstTrajectory.extra);
    const secondPayload = JSON.stringify(secondTrajectory.extra);

    assert.match(firstPayload, /node-first-agent/);
    assert.match(firstPayload, /node-nested-agent/);
    assert.doesNotMatch(firstPayload, /node-second-agent/);
    assert.match(secondPayload, /node-second-agent/);
    assert.doesNotMatch(secondPayload, /node-first-agent/);
    assert.doesNotMatch(secondPayload, /node-nested-agent/);
  });
});
