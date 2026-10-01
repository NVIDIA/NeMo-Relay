// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  collectResourceMetrics,
  type AcceleratorDeviceMetrics,
  type AcceleratorProcessMetrics,
  type FilesystemCapacityMetrics,
  type ResourceNumeric,
  type ResourceLimitEventCount,
  type ResourceMeasurement,
  type ResourceMeasurementScope,
  type ResourceMetricsSnapshot,
  DurationUnit,
  CapacityUnit,
  BandwidthUnit,
  DataUnit,
  CpuUnit,
  UtilizationUnit,
  CountUnit,
  type ResourceOperatingSystem,
} from '../index.js';

export const snapshotPromise: Promise<ResourceMetricsSnapshot> = collectResourceMetrics();
declare const snapshot: ResourceMetricsSnapshot;
export const sampledProcesses: ResourceNumeric | undefined = snapshot.processSampling?.sampledProcesses;
export const nullableCpu: ResourceNumeric | null = snapshot.cpu?.totalTime?.value ?? null;
export const timeUnit: DurationUnit | null = snapshot.cpu?.totalTime?.unit ?? null;
export const memoryUnit: CapacityUnit | null = snapshot.memory?.resident?.unit ?? null;
export const fileSystems: FilesystemCapacityMetrics[] = snapshot.disk?.filesystems ?? [];
export const limits: ResourceLimitEventCount[] = snapshot.process?.limitEvents ?? [];
export const devices: AcceleratorDeviceMetrics[] = snapshot.gpu?.deviceMetrics ?? [];
export const processes: AcceleratorProcessMetrics[] = snapshot.gpu?.processMetrics ?? [];
export const nullableUnit: ResourceMeasurement<number, DurationUnit> | null = null;
export const availableMeasurement: ResourceMeasurement<ResourceNumeric, DurationUnit> = {
  value: 1n,
  unit: DurationUnit.Milliseconds,
};
export const operatingSystem: ResourceOperatingSystem = snapshot.operatingSystem;
export const measurementScope: ResourceMeasurementScope = snapshot.measurementScope;

// @ts-expect-error Measurements have only a value and unit.
export const invalidMeasurementTimestamp = availableMeasurement.timestamp;
// @ts-expect-error Measurement failures are represented by null category fields.
export const invalidMeasurementReason = snapshot.cpu?.totalTime?.reason;
// @ts-expect-error Measurements have no availability status.
export const invalidMeasurementStatus = snapshot.cpu?.totalTime?.status;
// @ts-expect-error The snapshot has a single collection timestamp.
export const invalidCpuTimestamp = snapshot.cpu?.totalTime?.timestamp;

export const fieldCoverage: ResourceNumeric | undefined =
  snapshot.processSampling?.fieldSampledProcesses['cpu.total_time'];

export const duration: ResourceMeasurement<number, DurationUnit> = { value: 1, unit: DurationUnit.Seconds };
export const capacity: ResourceMeasurement<number, CapacityUnit> = { value: 1, unit: CapacityUnit.Gibibytes };
export const data: ResourceMeasurement<number, DataUnit> = { value: 1, unit: DataUnit.Megabytes };
export const bandwidth: ResourceMeasurement<number, BandwidthUnit> = {
  value: 1,
  unit: BandwidthUnit.MegabitsPerSecond,
};
export const cpu: ResourceMeasurement<number, CpuUnit> = { value: 1, unit: CpuUnit.Millicores };
export const utilization: ResourceMeasurement<number, UtilizationUnit> = { value: 1, unit: UtilizationUnit.Fraction };
export const count: ResourceMeasurement<number, CountUnit> = { value: 1, unit: CountUnit.Packets };
// @ts-expect-error Durations cannot use capacity units.
export const invalidDuration: ResourceMeasurement<number, DurationUnit> = { value: 1, unit: CapacityUnit.Bytes };
export const invalidCapacity: ResourceMeasurement<number, CapacityUnit> = {
  value: 1,
  // @ts-expect-error Capacity cannot use bandwidth units.
  unit: BandwidthUnit.MegabitsPerSecond,
};
// @ts-expect-error Bandwidth cannot use duration units.
export const invalidBandwidth: ResourceMeasurement<number, BandwidthUnit> = { value: 1, unit: DurationUnit.Seconds };
// @ts-expect-error CPU duration fields cannot receive capacity measurements.
export const invalidCpuTime: NonNullable<NonNullable<ResourceMetricsSnapshot['cpu']>['totalTime']> = capacity;

// @ts-expect-error Transferred data and capacity remain separate semantic types.
export const invalidCapacityData: ResourceMeasurement<number, CapacityUnit> = data;
