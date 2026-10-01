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

const snapshotPromise: Promise<ResourceMetricsSnapshot> = collectResourceMetrics();
void snapshotPromise;
declare const snapshot: ResourceMetricsSnapshot;
const sampledProcesses: ResourceNumeric | undefined = snapshot.processSampling?.sampledProcesses;
void sampledProcesses;
const nullableCpu: ResourceNumeric | null = snapshot.cpu?.totalTime?.value ?? null;
const timeUnit: DurationUnit | null = snapshot.cpu?.totalTime?.unit ?? null;
const memoryUnit: CapacityUnit | null = snapshot.memory?.resident?.unit ?? null;
const fileSystems: FilesystemCapacityMetrics[] = snapshot.disk?.filesystems ?? [];
const limits: ResourceLimitEventCount[] = snapshot.process?.limitEvents ?? [];
const devices: AcceleratorDeviceMetrics[] = snapshot.gpu?.deviceMetrics ?? [];
const processes: AcceleratorProcessMetrics[] = snapshot.gpu?.processMetrics ?? [];
const nullableUnit: ResourceMeasurement<number, DurationUnit> | null = null;
const availableMeasurement: ResourceMeasurement<ResourceNumeric, DurationUnit> = {
  value: 1n,
  unit: DurationUnit.Milliseconds,
};
const operatingSystem: ResourceOperatingSystem = snapshot.operatingSystem;
const measurementScope: ResourceMeasurementScope = snapshot.measurementScope;
void nullableCpu;
void timeUnit;
void memoryUnit;
void fileSystems;
void limits;
void devices;
void processes;
void nullableUnit;
void availableMeasurement;
void operatingSystem;
void measurementScope;

// @ts-expect-error Measurements have only a value and unit.
void availableMeasurement.timestamp;
// @ts-expect-error Measurement failures are represented by null category fields.
void snapshot.cpu?.totalTime?.reason;
// @ts-expect-error Measurements have no availability status.
void snapshot.cpu?.totalTime?.status;
// @ts-expect-error The snapshot has a single collection timestamp.
void snapshot.cpu?.totalTime?.timestamp;

const fieldCoverage: ResourceNumeric | undefined = snapshot.processSampling?.fieldSampledProcesses['cpu.total_time'];
void fieldCoverage;

const duration: ResourceMeasurement<number, DurationUnit> = { value: 1, unit: DurationUnit.Seconds };
const capacity: ResourceMeasurement<number, CapacityUnit> = { value: 1, unit: CapacityUnit.Gibibytes };
const data: ResourceMeasurement<number, DataUnit> = { value: 1, unit: DataUnit.Megabytes };
const bandwidth: ResourceMeasurement<number, BandwidthUnit> = { value: 1, unit: BandwidthUnit.MegabitsPerSecond };
const cpu: ResourceMeasurement<number, CpuUnit> = { value: 1, unit: CpuUnit.Millicores };
const utilization: ResourceMeasurement<number, UtilizationUnit> = { value: 1, unit: UtilizationUnit.Fraction };
const count: ResourceMeasurement<number, CountUnit> = { value: 1, unit: CountUnit.Packets };
void [duration, capacity, data, bandwidth, cpu, utilization, count];
// @ts-expect-error Durations cannot use capacity units.
const invalidDuration: ResourceMeasurement<number, DurationUnit> = { value: 1, unit: CapacityUnit.Bytes };
// @ts-expect-error Capacity cannot use bandwidth units.
const invalidCapacity: ResourceMeasurement<number, CapacityUnit> = { value: 1, unit: BandwidthUnit.MegabitsPerSecond };
// @ts-expect-error Bandwidth cannot use duration units.
const invalidBandwidth: ResourceMeasurement<number, BandwidthUnit> = { value: 1, unit: DurationUnit.Seconds };
// @ts-expect-error CPU duration fields cannot receive capacity measurements.
const invalidCpuTime: NonNullable<NonNullable<ResourceMetricsSnapshot['cpu']>['totalTime']> = capacity;
void [invalidDuration, invalidCapacity, invalidBandwidth, invalidCpuTime];

// @ts-expect-error Transferred data and capacity remain separate semantic types.
const invalidCapacityData: ResourceMeasurement<number, CapacityUnit> = data;
void invalidCapacityData;
