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
  type ResourceMeasurementUnit,
  type ResourceOperatingSystem,
} from '../index.js';

const snapshotPromise: Promise<ResourceMetricsSnapshot> = collectResourceMetrics();
void snapshotPromise;
declare const snapshot: ResourceMetricsSnapshot;
const sampledProcesses: ResourceNumeric | undefined = snapshot.processSampling?.sampledProcesses;
void sampledProcesses;
const nullableCpu: ResourceNumeric | null = snapshot.cpu?.totalTime?.value ?? null;
const timeUnit: ResourceMeasurementUnit | null = snapshot.cpu?.totalTime?.unit ?? null;
const memoryUnit: ResourceMeasurementUnit | null = snapshot.memory?.resident?.unit ?? null;
const fileSystems: FilesystemCapacityMetrics[] = snapshot.disk?.filesystems ?? [];
const limits: ResourceLimitEventCount[] = snapshot.process?.limitEvents ?? [];
const devices: AcceleratorDeviceMetrics[] = snapshot.gpu?.deviceMetrics ?? [];
const processes: AcceleratorProcessMetrics[] = snapshot.gpu?.processMetrics ?? [];
const nullableUnit: ResourceMeasurement<number> | null = null;
const availableMeasurement: ResourceMeasurement<ResourceNumeric> = {
  value: 1n,
  unit: 'milliseconds',
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
