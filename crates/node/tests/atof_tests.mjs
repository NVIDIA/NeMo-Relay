// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
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

function startRejectingServer(status) {
  let markResponded;
  const responded = new Promise((resolve) => {
    markResponded = resolve;
  });
  const server = createServer((request, response) => {
    request.resume();
    request.on('end', () => {
      response.writeHead(status);
      response.end(markResponded);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const origin = `http://127.0.0.1:${server.address().port}`;
      resolve({ origin, responded, close: () => new Promise((done) => server.close(done)) });
    });
  });
}

describe('AtofExporter', () => {
  it('constructs with defaults and rejects invalid mode', () => {
    const exporter = new AtofExporter({ outputDirectory: tempDir('node-atof-defaults') });
    assert.match(exporter.path, /nemo-relay-events-\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.jsonl$/);
    exporter.shutdown();

    assert.throws(() => new AtofExporter({ mode: 'invalid' }), /mode must be/i);
    assert.throws(
      () =>
        new AtofExporter({
          type: 'stream',
          url: 'http://localhost:8080/events',
          transport: 'bogus',
        }),
      /stream transport/i,
    );
    assert.throws(
      () =>
        new AtofExporter({
          type: 'stream',
          url: 'http://localhost:8080/events',
          fieldNamePolicy: 'bogus',
        }),
      /field_name_policy/i,
    );
    assert.throws(
      () =>
        new AtofExporter({
          type: 'stream',
          url: 'http://localhost:8080/events',
          headerEnv: { authorization: '' },
        }),
      /environment variable/i,
    );
  });

  it('returns null path for a stream sink', () => {
    const exporter = new AtofExporter({
      type: 'stream',
      url: 'http://localhost:8080/events',
    });
    try {
      assert.equal(exporter.path, null);
    } finally {
      exporter.shutdown();
    }
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

  it('records nothing for a file sink', () => {
    const outputDirectory = tempDir('node-atof-diagnostics');
    const exporter = new AtofExporter({ outputDirectory, filename: 'events.jsonl' });
    const name = `node_atof_file_diag_${Date.now()}`;
    exporter.register(name);
    try {
      event('atof_file_mark', null, { step: 1 }, null);
    } finally {
      exporter.deregister(name);
      exporter.forceFlush();
      exporter.shutdown();
    }
    assert.deepEqual(exporter.runtimeDiagnostics(), []);
  });

  it('records rejected stream deliveries', async () => {
    const server = await startRejectingServer(503);
    const exporter = new AtofExporter({
      type: 'stream',
      url: `${server.origin}/events`,
      transport: 'http_post',
    });
    const name = `node_atof_rejected_diag_${Date.now()}`;
    try {
      assert.deepEqual(exporter.runtimeDiagnostics(), []);
      exporter.register(name);
      try {
        event('atof_rejected_mark', null, { step: 1 }, null);
      } finally {
        exporter.deregister(name);
      }
      await server.responded;
      exporter.forceFlush();
      assert.deepEqual(exporter.runtimeDiagnostics(), [
        {
          code: 'atof.stream_delivery_failed',
          message: `ATOF http_post stream to ${server.origin} returned status 503 Service Unavailable`,
          count: 1,
        },
      ]);
    } finally {
      exporter.shutdown();
      await server.close();
    }
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
