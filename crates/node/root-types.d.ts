// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type ResourceOperatingSystem = 'linux' | 'macos' | 'windows' | 'unsupported';
export type ResourceMeasurementScope = 'global' | 'application_process' | 'process_tree';

/** Measurement values use bigint for exact integers above JavaScript's safe-integer range. */
export type ResourceNumeric = number | bigint;

export interface ResourceMeasurement<
  T extends number | bigint,
  U extends DurationUnit | CapacityUnit | DataUnit | BandwidthUnit | CpuUnit | UtilizationUnit | CountUnit,
> {
  value: T;
  unit: U;
}

export interface ResourceLimitEventCount {
  resource: 'cpu' | 'memory' | 'processes';
  event: 'throttled' | 'high' | 'maximum' | 'out_of_memory' | 'terminated';
  count: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
}

export type AcceleratorVendor = 'nvidia' | 'amd' | 'intel' | 'apple' | 'other';

export interface AcceleratorDeviceMetrics {
  vendor: AcceleratorVendor;
  deviceIdentifier: string;
  deviceIndex: number | null;
  memoryUsed: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  computeUtilization: ResourceMeasurement<number, UtilizationUnit> | null;
}

export interface AcceleratorProcessMetrics {
  vendor: AcceleratorVendor;
  deviceIdentifier: string;
  deviceIndex: number | null;
  processId: number;
  memoryUsed: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  computeUtilization: ResourceMeasurement<number, UtilizationUnit> | null;
}

export interface FilesystemCapacityMetrics {
  path: string;
  totalCapacity: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  availableCapacity: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  freeCapacity: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
}

export interface CpuMetrics {
  userTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  systemTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  totalTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  consumptionRate: ResourceMeasurement<number, CpuUnit> | null;
  throttledTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  effectiveLimit: ResourceMeasurement<number, CpuUnit> | null;
  somePressureStallTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  fullPressureStallTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  limitEvents: ResourceLimitEventCount[];
}

export interface MemoryMetrics {
  systemUsed: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  systemTotal: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  systemAvailable: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  resident: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  private: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  physicalFootprint: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  virtualMemory: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  peakResident: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  limit: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  environmentAccounted: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  somePressureStallTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  fullPressureStallTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  outOfMemoryEventCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  limitEvents: ResourceLimitEventCount[];
}

export interface ProcessMetrics {
  activeCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  descendantCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  threadCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  lifetimeCreationCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  openFileDescriptorCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  windowsHandleCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  limitEvents: ResourceLimitEventCount[];
}

export interface DiskMetrics {
  readData: ResourceMeasurement<ResourceNumeric, DataUnit> | null;
  writeData: ResourceMeasurement<ResourceNumeric, DataUnit> | null;
  readThroughput: ResourceMeasurement<number, BandwidthUnit> | null;
  writeThroughput: ResourceMeasurement<number, BandwidthUnit> | null;
  readOperations: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  writeOperations: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  filesystems: FilesystemCapacityMetrics[];
}

export interface GpuMetrics {
  deviceMetrics: AcceleratorDeviceMetrics[] | null;
  processMetrics: AcceleratorProcessMetrics[] | null;
}

export interface NetworkTrafficMetrics {
  receivedData: ResourceMeasurement<ResourceNumeric, DataUnit> | null;
  transmittedData: ResourceMeasurement<ResourceNumeric, DataUnit> | null;
  receiveThroughput: ResourceMeasurement<number, BandwidthUnit> | null;
  transmitThroughput: ResourceMeasurement<number, BandwidthUnit> | null;
  receivedPackets: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  transmittedPackets: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  receiveErrors: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  transmitErrors: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
}

export interface NetworkInterfaceMetrics {
  name: string;
  traffic: NetworkTrafficMetrics;
}

export interface NetworkMetrics {
  measurementScope: 'global';
  system: NetworkTrafficMetrics;
  interfaces: NetworkInterfaceMetrics[];
}

export interface ProcessSamplingMetadata {
  visibleProcesses: ResourceNumeric;
  sampledProcesses: ResourceNumeric;
  /** Readable process counts keyed by canonical snake_case measurement paths. */
  fieldSampledProcesses: Record<string, ResourceNumeric>;
}

export interface ResourceMetricsSnapshot {
  timestamp: string;
  operatingSystem: ResourceOperatingSystem;
  measurementScope: ResourceMeasurementScope;
  processSampling: ProcessSamplingMetadata | null;
  cpu: CpuMetrics | null;
  memory: MemoryMetrics | null;
  process: ProcessMetrics | null;
  disk: DiskMetrics | null;
  gpu: GpuMetrics | null;
  network: NetworkMetrics | null;
}

/** Codec identity available while a managed LLM event is sanitized. */
export type LlmCodecIdentity =
  | { kind: 'none' }
  | {
      kind: 'builtin';
      id: 'openai_chat' | 'openai_responses' | 'anthropic_messages' | 'oci_genai' | 'gemini_generate_content';
    }
  | { kind: 'runtime'; id: string }
  | { kind: 'opaque' };

/** Codec context available while an LLM request is sanitized. */
export interface LlmSanitizeRequestContext {
  codec: LlmCodecIdentity;
  /** Resolve the active codec for this callback. Do not retain the result after the callback returns. */
  resolveCodec(): import('./typed').LlmCodec | null;
}

/** Codec context available while an LLM response is sanitized. */
export interface LlmSanitizeResponseContext {
  codec: LlmCodecIdentity;
  /** Resolve the active codec for this callback. Do not retain the result after the callback returns. */
  resolveCodec(): import('./typed').LlmResponseCodec | null;
}

/** Schema tag attached to an opaque optimization contribution payload. */
export interface LlmOptimizationDataSchema {
  name: string;
  version: string;
}

/** Model identity retained for counterfactual pricing and downstream repricing. */
export interface LlmOptimizationModel {
  model: string;
  provider?: string;
}

/** Baseline and effective model identities for a routing optimization. */
export interface LlmOptimizationModelTransition {
  baseline?: LlmOptimizationModel;
  effective?: LlmOptimizationModel;
}

/** Explicit token evidence, independent from a pricing catalog. */
export interface LlmOptimizationTokens {
  /** Token counts must be non-negative JavaScript safe integers. */
  prompt_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  completion_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  cache_read_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  cache_write_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  total_tokens?: number;
}

/** Baseline, effective, and saved token evidence for one optimization. */
export interface LlmOptimizationTokenImpact {
  baseline?: LlmOptimizationTokens;
  effective?: LlmOptimizationTokens;
  saved?: LlmOptimizationTokens;
  quality?: 'observed' | 'estimated';
  estimation_method?: string;
}

/**
 * One plugin's optimization evidence.
 *
 * `kind` is deliberately an open string so new optimizer categories round-trip
 * without a Relay release. Unknown top-level fields are retained by the wire
 * contract and represented by this interface's JSON extension surface.
 */
export interface LlmOptimizationContribution {
  id?: string;
  /** Relay ordering must remain within JavaScript's safe-integer range. */
  sequence?: number;
  producer: string;
  kind: 'input_compression' | 'model_routing' | (string & {});
  applied: boolean;
  model_transition?: LlmOptimizationModelTransition;
  token_impact?: LlmOptimizationTokenImpact;
  payload_schema?: LlmOptimizationDataSchema;
  payload?: Json;
  [key: string]: Json | undefined;
}

/** Canonical result returned by an LLM request intercept. */
export interface LlmRequestInterceptOutcome {
  request: Json;
  annotated?: Json | null;
  pendingMarks?: PendingMarkSpec[];
  optimizationContributions?: LlmOptimizationContribution[];
}

/** Scalar value accepted in event metadata additions. */
export type EventMetadataScalar = string | number | boolean;

/**
 * Flat value accepted in event metadata additions. After JSON conversion,
 * numeric arrays must contain only integer values or only floating-point values.
 */
export type EventMetadataValue = EventMetadataScalar | string[] | number[] | boolean[];

/** Metadata additions returned by an event metadata injector. */
export type EventMetadata = Record<string, EventMetadataValue>;
