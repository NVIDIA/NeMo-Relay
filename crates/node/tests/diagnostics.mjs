// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import fs from 'node:fs';
import { test } from 'node:test';
import { createRequire } from 'node:module';
import { Worker } from 'node:worker_threads';
const require = createRequire(import.meta.url);
const diagnostics = require('../diagnostics.js');
const plugin = require('../plugin.js');
const queryContract = require('../diagnostics_query_contract.js');

test('query adapter matches the shared cross-binding contract fixture', () => {
  const fixture = JSON.parse(
    fs.readFileSync(new URL('../../core/tests/fixtures/diagnostics_query_contract.json', import.meta.url), 'utf8'),
  );
  assert.equal(queryContract.MAX_QUERY_BYTES, fixture.max_query_bytes);
  assert.equal(queryContract.MAX_QUERY_NODES, fixture.adapter_limits.max_query_nodes);
  assert.equal(queryContract.MAX_QUERY_DEPTH, fixture.adapter_limits.max_query_depth);
  assert.equal(queryContract.MAX_QUERY_OBJECT_FIELDS, fixture.adapter_limits.max_query_object_fields);
  assert.equal(queryContract.MAX_QUERY_FIELD_NAME_BYTES, fixture.adapter_limits.max_query_field_name_bytes);
  assert.deepEqual([...queryContract.NUMERIC_QUERY_FIELDS], fixture.numeric_query_fields);
});

test('reader preserves native snapshot and metadata failure reasons', () => {
  for (const reason of ['invalid_snapshot', 'invalid_metadata']) {
    const reader = new diagnostics.DiagnosticsReader({
      snapshot() {
        throw new Error(reason);
      },
      snapshotJson() {
        throw new Error(reason);
      },
      info() {
        throw new Error(reason);
      },
    });
    for (const method of ['snapshot', 'snapshotJson', 'info']) {
      assert.throws(
        () => reader[method](),
        (error) => error instanceof diagnostics.DiagnosticsError && error.reason === reason,
      );
    }
  }
});

test('configured inspection survives plugin and Node environment cleanup', async () => {
  assert.throws(
    () => diagnostics.reader(),
    (error) => error.reason === 'not_initialized',
  );
  const config = { diagnostics: { enabled: true, limits: { max_events: 3 } } };
  plugin.validate(config);
  assert.throws(
    () => diagnostics.reader(),
    (error) => error.reason === 'not_initialized',
  );
  const activation = await plugin.initialize(config);
  const reader = diagnostics.reader();
  const memory = reader.inspectMemory();
  assert.equal(memory.schemaVersion, 1);
  assert.equal(typeof memory.aggregate.reservedBytes, 'bigint');
  assert.equal(memory.componentOccupiedBytes + memory.sharedOrUnassignedReservedBytes, memory.aggregate.reservedBytes);
  assert.equal(memory.resource.availability, 'disabled');
  assert.equal(typeof JSON.parse(reader.inspectMemoryJson()).aggregate.reserved_bytes, 'string');
  assert(reader.info().discovery.detailedMemoryInspection);
  const first = reader.snapshot();
  assert.equal(typeof first.sequence, 'bigint');
  assert.equal(typeof first.lifetime.totals.failuresTotal.value, 'bigint');
  assert.equal(typeof JSON.parse(reader.snapshotJson()).sequence, 'string');
  assert.equal(reader.info().limits.maxEvents, 3);
  assert.equal(reader.info().loggingCoverage, 'attached_since_service_start');
  const modulePath = require.resolve('../diagnostics.js');
  const source = await new Promise((resolve, reject) => {
    const worker = new Worker(
      `const {parentPort} = require("node:worker_threads"); const diagnostics = require(${JSON.stringify(modulePath)}); parentPort.postMessage(diagnostics.reader().snapshot().source.instanceId);`,
      { eval: true },
    );
    let instance;
    worker.on('message', (value) => {
      instance = value;
    });
    worker.on('error', reject);
    worker.on('exit', (code) => (code === 0 ? resolve(instance) : reject(new Error(`worker exit ${code}`))));
  });
  assert.equal(source, first.source.instanceId);
  assert.equal(reader.info().loggingCoverage, 'attached_since_service_start');
  await activation.close();
  const resources = await plugin.initialize({
    diagnostics: config.diagnostics,
    components: [
      {
        kind: 'resource_metrics',
        config: {
          measurement_scope: 'application_process',
          polling: { enabled: false },
          units: { memory: { system_used: 'gibibytes' } },
        },
      },
    ],
  });
  assert.equal(reader.inspectMemory().resource.availability, 'not_yet_sampled');
  await require('../index.js').collectResourceMetrics();
  const sample = reader.inspectMemory().resource.sample;
  assert(sample);
  assert.equal(typeof sample.ordinal, 'bigint');
  if (sample.memory.systemUsed !== null) {
    assert.equal(sample.memory.systemUsed.measurementScope, 'system');
    assert.equal(sample.memory.systemUsed.value.kind, 'decimal');
    assert.equal(typeof sample.memory.systemUsed.value.value, 'number');
  }
  await resources.close();
  assert.equal(reader.inspectMemory().resource.availability, 'disabled');
  assert.equal(reader.snapshot().source.instanceId, source);
  assert.throws(
    () => reader.snapshot({ afterSequence: Number.MAX_SAFE_INTEGER + 1 }),
    (error) => error.reason === 'invalid_query',
  );
  assert.throws(
    () => reader.snapshot({ componentKey: 'x'.repeat(4097) }),
    (error) => error.reason === 'invalid_query',
  );
  assert.throws(
    () => reader.snapshot({ unknown: true }),
    (error) => error.reason === 'invalid_query',
  );
  assert.equal(reader.info().discovery.maxQueryEvents, 256n);
  assert(reader.info().discovery.producerSupport.some((p) => p.name === 'otel' && !p.supported));
  assert.equal(reader.snapshot({ schemaVersion: 1 }).schemaVersion, 1);
  for (const [query, reason] of [
    [{ schemaVersion: 2 }, 'unsupported_schema'],
    [{ maxEvents: 4294967295 }, 'invalid_query'],
    [{ maxResponseBytes: 1 }, 'snapshot_capacity'],
    [{ ownership: { unknown: 'restricted-input' } }, 'invalid_query'],
  ]) {
    assert.throws(
      () => reader.snapshot(query),
      (error) => error.reason === reason,
    );
  }
  await assert.rejects(plugin.initialize({ diagnostics: { enabled: false } }), /configuration_conflict/);
  assert.equal(reader.snapshot().source.instanceId, source);
});
