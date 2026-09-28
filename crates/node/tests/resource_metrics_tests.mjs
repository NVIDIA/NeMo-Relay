// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const relay = require('../index.js');

function assertMeasurementState(measurement) {
  assert.deepEqual(Object.keys(measurement).sort(), ['timestamp', 'unit', 'value']);
  assert.ok(Number.isFinite(Date.parse(measurement.timestamp)));
  if (measurement.value === null) {
    assert.equal(measurement.unit, null);
  } else {
    assert.equal(typeof measurement.unit, 'string');
  }
  assert.equal(Object.hasOwn(measurement, 'status'), false);
  assert.equal(Object.hasOwn(measurement, 'reason'), false);
}

function expectedNativeUnits() {
  switch (process.platform) {
    case 'darwin':
      return { cpu: 'nanoseconds', rate: 'nanoseconds_per_second', memory: 'bytes' };
    case 'linux':
      return { cpu: 'clock_ticks', rate: 'clock_ticks_per_second', memory: 'pages' };
    case 'win32':
      return {
        cpu: 'hundred_nanosecond_intervals',
        rate: 'hundred_nanosecond_intervals_per_second',
        memory: 'bytes',
      };
    default:
      return { cpu: 'unknown', rate: 'unknown', memory: 'unknown' };
  }
}

function expectedOperatingSystem() {
  switch (process.platform) {
    case 'darwin':
      return 'macos';
    case 'linux':
      return 'linux';
    case 'win32':
      return 'windows';
    default:
      return 'unsupported';
  }
}

const measurementFields = [
  'cpuUserTime',
  'cpuSystemTime',
  'cpuTotalTime',
  'cpuConsumptionRate',
  'cpuThrottledTime',
  'effectiveCpuLimit',
  'cpuSomePressureStallTime',
  'cpuFullPressureStallTime',
  'residentMemory',
  'privateMemory',
  'physicalFootprint',
  'virtualMemory',
  'peakResidentMemory',
  'memoryLimit',
  'environmentAccountedMemory',
  'memorySomePressureStallTime',
  'memoryFullPressureStallTime',
  'outOfMemoryEventCount',
  'activeProcessCount',
  'descendantProcessCount',
  'threadCount',
  'lifetimeProcessCreationCount',
  'openFileDescriptorCount',
  'windowsHandleCount',
];

const integerMeasurementFields = measurementFields.filter(
  (field) => !['cpuConsumptionRate', 'effectiveCpuLimit'].includes(field),
);

describe('resource metrics API', () => {
  it('returns fresh typed snapshots and retains bounded polling history and file output', async () => {
    const fresh = relay.collectResourceMetrics();
    assert.equal(fresh.operatingSystem, expectedOperatingSystem());
    assert.equal(Object.hasOwn(fresh, 'measurementScope'), false);
    assert.equal(Object.hasOwn(fresh, 'collectedAt'), false);
    assert.equal(Object.hasOwn(fresh, 'sampleIntervalMillis'), false);
    assert.ok(fresh.activeProcessCount.value > 0);
    for (const field of measurementFields) {
      assertMeasurementState(fresh[field]);
    }
    for (const field of integerMeasurementFields) {
      const value = fresh[field].value;
      if (value !== null) {
        assert.equal(
          typeof value,
          value <= Number.MAX_SAFE_INTEGER ? 'number' : 'bigint',
          `${field} must preserve the integer type at the JavaScript safe-integer boundary`,
        );
      }
    }
    assert.equal(fresh.cpuConsumptionRate.value, null);
    const nativeUnits = expectedNativeUnits();
    assert.equal(fresh.cpuUserTime.unit, nativeUnits.cpu);
    assert.equal(fresh.cpuSystemTime.unit, nativeUnits.cpu);
    assert.equal(fresh.cpuTotalTime.unit, nativeUnits.cpu);
    if (fresh.cpuConsumptionRate.value === null) {
      assert.equal(fresh.cpuConsumptionRate.unit, null);
    } else {
      assert.equal(fresh.cpuConsumptionRate.unit, nativeUnits.rate);
    }
    assert.equal(fresh.residentMemory.unit, nativeUnits.memory);
    for (const field of ['cpuUserTime', 'cpuSystemTime', 'cpuTotalTime']) {
      if (fresh[field].value !== null) assert.ok(Number.isInteger(fresh[field].value));
    }
    if (fresh.cpuConsumptionRate.value !== null) {
      assert.equal(typeof fresh.cpuConsumptionRate.value, 'number');
    }
    assert.ok(Array.isArray(fresh.resourceLimitEvents));
    assert.ok(Array.isArray(fresh.acceleratorDevices));
    assert.ok(Array.isArray(fresh.acceleratorProcesses));
    if (process.platform === 'darwin') {
      assert.notEqual(fresh.physicalFootprint.value, null);
      assert.notEqual(fresh.virtualMemory.value, null);
      assert.notEqual(fresh.openFileDescriptorCount.value, null);
      for (const field of [
        'privateMemory',
        'peakResidentMemory',
        'cpuThrottledTime',
        'effectiveCpuLimit',
        'cpuSomePressureStallTime',
        'cpuFullPressureStallTime',
        'memoryLimit',
        'environmentAccountedMemory',
        'memorySomePressureStallTime',
        'memoryFullPressureStallTime',
        'outOfMemoryEventCount',
        'lifetimeProcessCreationCount',
        'windowsHandleCount',
      ]) {
        assert.equal(fresh[field].value, null);
        assert.equal(fresh[field].unit, null);
      }
    }
    assert.equal(relay.latestResourceMetrics(), null);
    assert.deepEqual(relay.resourceMetricsHistory(), []);

    const directory = await mkdtemp(path.join(os.tmpdir(), 'relay-resource-metrics-'));
    const outputPath = path.join(directory, 'resource-metrics.jsonl');
    const runtime = relay.configureResourceMetrics({
      polling: { enabled: true, intervalMillis: 10, retainedSnapshots: 2 },
      file: {
        enabled: true,
        path: outputPath,
        maxFileSizeBytes: 1_000_000,
        retainedFiles: 2,
      },
    });
    try {
      const deadline = Date.now() + 2_000;
      while (relay.resourceMetricsHistory().length < 2 && Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      const snapshots = relay.resourceMetricsHistory();
      assert.equal(snapshots.length, 2);
      assert.deepEqual(relay.latestResourceMetrics(), snapshots.at(-1));
      assert.notEqual(relay.collectResourceMetrics().activeProcessCount.value, null);
      assert.equal(relay.resourceMetricsHistory().length, 2);

      const fileSnapshots = (await readFile(outputPath, 'utf8'))
        .trim()
        .split('\n')
        .map((line) => JSON.parse(line));
      assert.ok(fileSnapshots.length >= 2);
      assert.ok(fileSnapshots.every((snapshot) => snapshot.operating_system === expectedOperatingSystem()));
      assert.ok(fileSnapshots.every((snapshot) => !Object.hasOwn(snapshot, 'measurement_scope')));
      assert.ok(
        fileSnapshots.every(
          (snapshot) =>
            !Object.hasOwn(snapshot, 'collected_at') &&
            !Object.hasOwn(snapshot, 'sample_interval_millis') &&
            Array.isArray(snapshot.resource_limit_events) &&
            Array.isArray(snapshot.accelerator_devices) &&
            Array.isArray(snapshot.accelerator_processes) &&
            measurementFields.every((field) => {
              const snakeField = field.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
              const measurement = snapshot[snakeField];
              assertMeasurementState(measurement);
              return true;
            }),
        ),
      );
    } finally {
      await runtime.close();
      await rm(directory, { recursive: true, force: true });
    }
  });
});
