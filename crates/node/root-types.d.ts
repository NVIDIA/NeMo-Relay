// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type ResourceMeasurementUnit =
  | 'milliseconds'
  | 'bytes'
  | 'kibibytes'
  | 'logical_processors'
  | 'percentage'
  | 'processes'
  | 'threads'
  | 'file_descriptors'
  | 'handles'
  | 'events'
  | 'operations'
  | 'microseconds'
  | 'seconds'
  | 'minutes'
  | 'kilobytes'
  | 'megabytes'
  | 'gigabytes'
  | 'terabytes'
  | 'mebibytes'
  | 'gibibytes'
  | 'tebibytes'
  | 'bytes_per_second'
  | 'kibibytes_per_second'
  | 'mebibytes_per_second'
  | 'gibibytes_per_second'
  | 'bits_per_second'
  | 'megabits_per_second'
  | 'gigabits_per_second'
  | 'millicores'
  | 'fraction'
  | 'packets'
  | 'errors';

export type ResourceOperatingSystem = 'linux' | 'macos' | 'windows' | 'unsupported';
export type ResourceMeasurementScope = 'global' | 'application_process' | 'process_tree';

/** Measurement values use bigint for exact integers above JavaScript's safe-integer range. */
export type ResourceNumeric = number | bigint;

export interface ResourceMeasurement<T extends number | bigint> {
  value: T;
  unit: ResourceMeasurementUnit;
}

export interface ResourceLimitEventCount {
  resource: 'cpu' | 'memory' | 'processes';
  event: 'throttled' | 'high' | 'maximum' | 'out_of_memory' | 'terminated';
  count: ResourceMeasurement<ResourceNumeric> | null;
}

export type AcceleratorVendor = 'nvidia' | 'amd' | 'intel' | 'apple' | 'other';

export interface AcceleratorDeviceMetrics {
  vendor: AcceleratorVendor;
  deviceIdentifier: string;
  deviceIndex: number | null;
  memoryUsed: ResourceMeasurement<ResourceNumeric> | null;
  computeUtilization: ResourceMeasurement<number> | null;
}

export interface AcceleratorProcessMetrics {
  vendor: AcceleratorVendor;
  deviceIdentifier: string;
  deviceIndex: number | null;
  processId: number;
  memoryUsed: ResourceMeasurement<ResourceNumeric> | null;
  computeUtilization: ResourceMeasurement<number> | null;
}

export interface FilesystemCapacityMetrics {
  path: string;
  totalCapacity: ResourceMeasurement<ResourceNumeric> | null;
  availableCapacity: ResourceMeasurement<ResourceNumeric> | null;
  freeCapacity: ResourceMeasurement<ResourceNumeric> | null;
}

export interface CpuMetrics {
  userTime: ResourceMeasurement<ResourceNumeric> | null;
  systemTime: ResourceMeasurement<ResourceNumeric> | null;
  totalTime: ResourceMeasurement<ResourceNumeric> | null;
  consumptionRate: ResourceMeasurement<number> | null;
  throttledTime: ResourceMeasurement<ResourceNumeric> | null;
  effectiveLimit: ResourceMeasurement<number> | null;
  somePressureStallTime: ResourceMeasurement<ResourceNumeric> | null;
  fullPressureStallTime: ResourceMeasurement<ResourceNumeric> | null;
  limitEvents: ResourceLimitEventCount[];
}

export interface MemoryMetrics {
  systemUsed: ResourceMeasurement<ResourceNumeric> | null;
  systemTotal: ResourceMeasurement<ResourceNumeric> | null;
  systemAvailable: ResourceMeasurement<ResourceNumeric> | null;
  resident: ResourceMeasurement<ResourceNumeric> | null;
  private: ResourceMeasurement<ResourceNumeric> | null;
  physicalFootprint: ResourceMeasurement<ResourceNumeric> | null;
  virtualMemory: ResourceMeasurement<ResourceNumeric> | null;
  peakResident: ResourceMeasurement<ResourceNumeric> | null;
  limit: ResourceMeasurement<ResourceNumeric> | null;
  environmentAccounted: ResourceMeasurement<ResourceNumeric> | null;
  somePressureStallTime: ResourceMeasurement<ResourceNumeric> | null;
  fullPressureStallTime: ResourceMeasurement<ResourceNumeric> | null;
  outOfMemoryEventCount: ResourceMeasurement<ResourceNumeric> | null;
  limitEvents: ResourceLimitEventCount[];
}

export interface ProcessMetrics {
  activeCount: ResourceMeasurement<ResourceNumeric> | null;
  descendantCount: ResourceMeasurement<ResourceNumeric> | null;
  threadCount: ResourceMeasurement<ResourceNumeric> | null;
  lifetimeCreationCount: ResourceMeasurement<ResourceNumeric> | null;
  openFileDescriptorCount: ResourceMeasurement<ResourceNumeric> | null;
  windowsHandleCount: ResourceMeasurement<ResourceNumeric> | null;
  limitEvents: ResourceLimitEventCount[];
}

export interface DiskMetrics {
  readData: ResourceMeasurement<ResourceNumeric> | null;
  writeData: ResourceMeasurement<ResourceNumeric> | null;
  readThroughput: ResourceMeasurement<number> | null;
  writeThroughput: ResourceMeasurement<number> | null;
  readOperations: ResourceMeasurement<ResourceNumeric> | null;
  writeOperations: ResourceMeasurement<ResourceNumeric> | null;
  filesystems: FilesystemCapacityMetrics[];
}

export interface GpuMetrics {
  deviceMetrics: AcceleratorDeviceMetrics[] | null;
  processMetrics: AcceleratorProcessMetrics[] | null;
}

export interface NetworkTrafficMetrics {
  receivedData: ResourceMeasurement<ResourceNumeric> | null;
  transmittedData: ResourceMeasurement<ResourceNumeric> | null;
  receiveThroughput: ResourceMeasurement<number> | null;
  transmitThroughput: ResourceMeasurement<number> | null;
  receivedPackets: ResourceMeasurement<ResourceNumeric> | null;
  transmittedPackets: ResourceMeasurement<ResourceNumeric> | null;
  receiveErrors: ResourceMeasurement<ResourceNumeric> | null;
  transmitErrors: ResourceMeasurement<ResourceNumeric> | null;
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
