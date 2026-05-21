// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const { AtofExporter, ScopeType, pushScope, popScope, event } = require('../index.js');

function tempDir(prefix) {
  return mkdtempSync(join(tmpdir(), `nemo-relay-${prefix}-`));
}

function lines(path) {
  return readFileSync(path, 'utf8')
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

describe('AtofExporter', () => {
  it('constructs with defaults and rejects invalid mode', () => {
    const exporter = new AtofExporter({ outputDirectory: tempDir('node-atof-defaults') });
    assert.match(exporter.path, /nemo-relay-events-\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.jsonl$/);
    exporter.shutdown();

    assert.throws(() => new AtofExporter({ mode: 'invalid' }), /mode must be/i);
  });

  it('writes raw ATOF JSONL events and supports lifecycle methods', () => {
    const outputDirectory = tempDir('node-atof');
    const exporter = new AtofExporter({
      outputDirectory,
      mode: 'overwrite',
      filename: 'events.jsonl',
    });
    const name = `node_atof_${Date.now()}_${Math.random().toString(16).slice(2)}`;

    exporter.register(name);
    try {
      const scope = pushScope('atof_scope', ScopeType.Agent, null, null, null, null, { scope: true });
      event('atof_mark', scope, { step: 1 }, null);
      popScope(scope, { done: true });
    } finally {
      assert.equal(exporter.deregister(name), true);
      assert.equal(exporter.deregister(name), false);
      exporter.forceFlush();
      exporter.shutdown();
    }

    const records = lines(join(outputDirectory, 'events.jsonl'));
    assert.deepEqual(
      records.map((record) => record.kind),
      ['scope', 'mark', 'scope'],
    );
    assert.equal(records[0].name, 'atof_scope');
    assert.deepEqual(records[1].data, { step: 1 });
    assert.equal(records[2].scope_category, 'end');
  });

  it('supports append and overwrite modes', () => {
    const outputDirectory = tempDir('node-atof-modes');
    const path = join(outputDirectory, 'events.jsonl');
    writeFileSync(path, '{"existing":true}\n');

    const appendExporter = new AtofExporter({ outputDirectory, filename: 'events.jsonl' });
    appendExporter.shutdown();
    assert.equal(readFileSync(path, 'utf8'), '{"existing":true}\n');

    const overwriteExporter = new AtofExporter({
      outputDirectory,
      mode: 'overwrite',
      filename: 'events.jsonl',
    });
    overwriteExporter.shutdown();
    assert.equal(readFileSync(path, 'utf8'), '');
  });
});
