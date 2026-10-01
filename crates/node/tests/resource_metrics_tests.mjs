// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const relay = require('../index.js');
const plugin = require('../plugin.js');

const config = (resourceConfig = {}) => ({
  version: 1,
  components: [{ kind: 'resource_metrics', enabled: true, config: resourceConfig }],
});

function assertMeasurement(measurement) {
  if (measurement === null) return;
  assert.deepEqual(Object.keys(measurement).sort(), ['unit', 'value']);
  assert.notEqual(measurement.value, null);
  assert.notEqual(measurement.unit, null);
}

describe('resource_metrics built-in plugin API', () => {
  it('exports semantic unit enums accepted by the canonical configuration', () => {
    for (const [unitType, category, field] of [
      [relay.DurationUnit, 'cpu', 'user_time'],
      [relay.CapacityUnit, 'memory', 'resident'],
      [relay.CapacityUnit, 'disk', 'filesystem_total_capacity'],
      [relay.DataUnit, 'disk', 'read_data'],
      [relay.BandwidthUnit, 'disk', 'read_throughput'],
      [relay.CpuUnit, 'cpu', 'consumption_rate'],
      [relay.UtilizationUnit, 'gpu', 'device_compute_utilization'],
    ]) {
      for (const unit of Object.values(unitType)) {
        const report = relay.validateExact(config({ units: { [category]: { [field]: unit } } }));
        assert.deepEqual(report.config.diagnostics, [], `${category}.${field}: ${unit}`);
        assert.equal(JSON.stringify({ value: 1, unit }), `{"value":1,"unit":"${unit}"}`);
      }
    }
    assert.deepEqual(
      Object.values(relay.CountUnit).sort(),
      ['processes', 'threads', 'file_descriptors', 'handles', 'events', 'operations', 'packets', 'errors'].sort(),
    );
    const invalid = relay.validateExact(
      config({ units: { cpu: { user_time: relay.BandwidthUnit.MegabitsPerSecond } } }),
    );
    assert.ok(invalid.config.diagnostics.some((diagnostic) => diagnostic.level === 'error'));
  });

  it('rejects asynchronously when the component is disabled', async () => {
    const activation = await plugin.initialize({
      version: 1,
      components: [{ kind: 'resource_metrics', enabled: false, config: {} }],
    });
    try {
      await assert.rejects(relay.collectResourceMetrics(), /requires an active resource_metrics component/);
    } finally {
      await activation.close();
    }
  });

  it('collects a point-in-time categorized snapshot on demand without polling', async () => {
    const activation = await plugin.initialize(config({ polling: { enabled: false } }));
    try {
      const snapshot = await relay.collectResourceMetrics();
      assert.ok(Number.isFinite(Date.parse(snapshot.timestamp)));
      assert.equal(
        snapshot.operatingSystem,
        process.platform === 'darwin' ? 'macos' : process.platform === 'win32' ? 'windows' : 'linux',
      );
      assert.equal(snapshot.measurementScope, 'process_tree');
      assert.ok(snapshot.processSampling.visibleProcesses >= snapshot.processSampling.sampledProcesses);
      assert.ok(snapshot.processSampling.sampledProcesses >= 1);
      assert.ok(Object.hasOwn(snapshot.processSampling.fieldSampledProcesses, 'cpu.total_time'));
      assert.equal(snapshot.processSampling.fieldSampledProcesses['disk.read_throughput'], 0);
      assert.equal(snapshot.processSampling.fieldSampledProcesses['disk.write_throughput'], 0);
      for (const count of Object.values(snapshot.processSampling.fieldSampledProcesses)) {
        assert.ok(count >= 0 && count <= snapshot.processSampling.sampledProcesses);
      }
      assert.notEqual(snapshot.process.activeCount, null);
      assert.equal(snapshot.cpu.userTime?.unit, snapshot.cpu.userTime === null ? null : 'milliseconds');
      assert.equal(snapshot.memory.resident?.unit, snapshot.memory.resident === null ? null : 'kibibytes');
      assert.equal(snapshot.memory.systemTotal?.unit, 'kibibytes');
      for (const measurement of [
        snapshot.cpu.userTime,
        snapshot.cpu.systemTime,
        snapshot.cpu.totalTime,
        snapshot.cpu.consumptionRate,
        snapshot.memory.resident,
        snapshot.memory.private,
        snapshot.memory.systemUsed,
        snapshot.memory.systemTotal,
        snapshot.memory.systemAvailable,
        snapshot.process.activeCount,
        snapshot.process.threadCount,
        snapshot.disk.readData,
        snapshot.disk.writeData,
        snapshot.disk.readOperations,
        snapshot.disk.writeOperations,
      ])
        assertMeasurement(measurement);
      assert.ok(Array.isArray(snapshot.disk.filesystems));
      assert.equal(snapshot.network.measurementScope, 'global');
      assert.ok(Array.isArray(snapshot.network.interfaces));
      assert.ok(snapshot.gpu.deviceMetrics === null || Array.isArray(snapshot.gpu.deviceMetrics));
      assert.ok(snapshot.gpu.processMetrics === null || Array.isArray(snapshot.gpu.processMetrics));
    } finally {
      await activation.close();
    }
  });

  it('collects system memory when global scope is selected', async () => {
    const activation = await plugin.initialize(config({ measurement_scope: 'global', polling: { enabled: false } }));
    try {
      const snapshot = await relay.collectResourceMetrics();
      assert.equal(snapshot.measurementScope, 'global');
      assert.notEqual(snapshot.memory.systemTotal, null);
      assert.equal(snapshot.memory.systemTotal.unit, 'kibibytes');
      assert.equal(snapshot.memory.resident, null);
    } finally {
      await activation.close();
    }
  });

  it('returns disabled categories and groups as null', async () => {
    const activation = await plugin.initialize(
      config({
        polling: { enabled: false },
        cpu: { enabled: false },
        memory: { enabled: false },
        process: { enabled: false },
        disk: { enabled: true, process_io: false, filesystem_paths: [] },
        gpu: { enabled: false },
      }),
    );
    try {
      const snapshot = await relay.collectResourceMetrics();
      assert.equal(snapshot.cpu, null);
      assert.equal(snapshot.memory, null);
      assert.equal(snapshot.process, null);
      assert.equal(snapshot.gpu, null);
      assert.notEqual(snapshot.disk, null);
      assert.equal(snapshot.disk.readData, null);
      assert.equal(snapshot.disk.writeData, null);
      assert.equal(snapshot.disk.readThroughput, null);
      assert.equal(snapshot.disk.writeThroughput, null);
      assert.equal(snapshot.disk.readOperations, null);
      assert.equal(snapshot.disk.writeOperations, null);
      assert.deepEqual(snapshot.disk.filesystems, []);
    } finally {
      await activation.close();
    }
  });

  it('applies per-metric units and labels network traffic global', async () => {
    const activation = await plugin.initialize(
      config({
        polling: { enabled: false },
        units: {
          cpu: { user_time: 'seconds' },
          memory: { system_total: 'mebibytes' },
          network: { system: { received_data: 'megabytes' } },
        },
      }),
    );
    try {
      const snapshot = await relay.collectResourceMetrics();
      assert.equal(snapshot.memory.systemTotal.unit, 'mebibytes');
      assert.equal(snapshot.network.measurementScope, 'global');
      if (snapshot.cpu.userTime !== null) assert.equal(snapshot.cpu.userTime.unit, 'seconds');
      if (snapshot.network.system.receivedData !== null)
        assert.equal(snapshot.network.system.receivedData.unit, 'megabytes');
    } finally {
      await activation.close();
    }
  });

  it('rejects a zero enabled polling interval', () => {
    const report = relay.validateExact(config({ polling: { enabled: true, interval_millis: 0 } }));
    assert.ok(report.config.diagnostics.some((diagnostic) => diagnostic.level === 'error'));
  });
});
