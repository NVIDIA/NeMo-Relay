// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';
import { Worker } from 'node:worker_threads';
const require = createRequire(import.meta.url);
const diagnostics = require('../diagnostics.js');
const plugin = require('../plugin.js');

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
  await assert.rejects(plugin.initialize({ diagnostics: { enabled: false } }), /configuration_conflict/);
  assert.equal(reader.snapshot().source.instanceId, source);
});
