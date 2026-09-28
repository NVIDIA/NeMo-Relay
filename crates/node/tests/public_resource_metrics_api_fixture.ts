// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  collectResourceMetrics,
  configureResourceMetrics,
  latestResourceMetrics,
  resourceMetricsHistory,
  type AcceleratorDeviceMetrics,
  type AcceleratorProcessMetrics,
  type ResourceInteger,
  type ResourceLimitEventCount,
  type ResourceMeasurement,
  type ResourceMetricsSnapshot,
  type ResourceMeasurementUnit,
  type ResourceOperatingSystem,
} from '../index.js';

const freshSnapshot: ResourceMetricsSnapshot = collectResourceMetrics();
const latestSnapshot: ResourceMetricsSnapshot | null = latestResourceMetrics();
const retainedSnapshots: ResourceMetricsSnapshot[] = resourceMetricsHistory();
const configuredRuntime = configureResourceMetrics({
  polling: { enabled: true, intervalMillis: 1000, retainedSnapshots: 12 },
  file: {
    enabled: true,
    path: 'resource-metrics.jsonl',
    maxFileSizeBytes: 1024,
    retainedFiles: 2,
  },
});

const nullableRateValue: number | null = freshSnapshot.cpuConsumptionRate.value;
const nullableRateUnit: ResourceMeasurementUnit | null = freshSnapshot.cpuConsumptionRate.unit;
const exactLargeIntegerMeasurement: ResourceMeasurement<ResourceInteger> = {
  value: 9_007_199_254_740_992n,
  unit: 'nanoseconds',
  timestamp: freshSnapshot.cpuTotalTime.timestamp,
};
void nullableRateValue;
void nullableRateUnit;
void exactLargeIntegerMeasurement;
if (latestSnapshot !== null) {
  const operatingSystem: ResourceOperatingSystem = latestSnapshot.operatingSystem;
  void operatingSystem;
  const integerMeasurements: ResourceMeasurement<ResourceInteger>[] = [
    latestSnapshot.cpuUserTime,
    latestSnapshot.cpuSystemTime,
    latestSnapshot.cpuTotalTime,
    latestSnapshot.cpuThrottledTime,
    latestSnapshot.cpuSomePressureStallTime,
    latestSnapshot.cpuFullPressureStallTime,
    latestSnapshot.residentMemory,
    latestSnapshot.privateMemory,
    latestSnapshot.physicalFootprint,
    latestSnapshot.virtualMemory,
    latestSnapshot.peakResidentMemory,
    latestSnapshot.memoryLimit,
    latestSnapshot.environmentAccountedMemory,
    latestSnapshot.memorySomePressureStallTime,
    latestSnapshot.memoryFullPressureStallTime,
    latestSnapshot.outOfMemoryEventCount,
    latestSnapshot.activeProcessCount,
    latestSnapshot.descendantProcessCount,
    latestSnapshot.threadCount,
    latestSnapshot.lifetimeProcessCreationCount,
    latestSnapshot.openFileDescriptorCount,
    latestSnapshot.windowsHandleCount,
  ];
  const floatingPointMeasurements: ResourceMeasurement<number>[] = [
    latestSnapshot.cpuConsumptionRate,
    latestSnapshot.effectiveCpuLimit,
  ];
  const limitEvents: ResourceLimitEventCount[] = latestSnapshot.resourceLimitEvents;
  const acceleratorDevices: AcceleratorDeviceMetrics[] = latestSnapshot.acceleratorDevices;
  const acceleratorProcesses: AcceleratorProcessMetrics[] = latestSnapshot.acceleratorProcesses;
  const nullableUnit: ResourceMeasurement<number> = {
    value: null,
    unit: null,
    timestamp: latestSnapshot.cpuTotalTime.timestamp,
  };
  void integerMeasurements;
  void floatingPointMeasurements;
  void limitEvents;
  void acceleratorDevices;
  void acceleratorProcesses;
  void nullableUnit;
  const timestamp: string = latestSnapshot.cpuTotalTime.timestamp;
  const unit: ResourceMeasurementUnit | null = latestSnapshot.cpuTotalTime.unit;
  const cpuTime: ResourceInteger | null = latestSnapshot.cpuTotalTime.value;
  void timestamp;
  void unit;
  void cpuTime;
}
void retainedSnapshots;
configuredRuntime.close();

// @ts-expect-error Snapshots have only per-measurement timestamps.
void freshSnapshot.collectedAt;
// @ts-expect-error Sampling intervals are not part of the public snapshot.
void freshSnapshot.sampleIntervalMillis;
// @ts-expect-error Measurement failures never expose a reason field.
void freshSnapshot.cpuTotalTime.reason;
// @ts-expect-error Measurements have no availability status.
void freshSnapshot.cpuTotalTime.status;
// @ts-expect-error Process scope is private collection ownership, not part of public snapshots.
void freshSnapshot.measurementScope;
