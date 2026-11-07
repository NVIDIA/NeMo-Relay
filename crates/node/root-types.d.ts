// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type JsonPrimitive = string | number | boolean | null;
export type JsonObject = { [Key in string]: JsonValue };
export type JsonArray = JsonValue[] | readonly JsonValue[];
export type JsonValue = JsonPrimitive | JsonObject | JsonArray;
export type Json = JsonValue;
export type LlmRequestShape = {
  headers: JsonObject;
  content: Json;
};

type NonJsonValue =
  | ((...args: never[]) => unknown)
  | (abstract new (...args: never[]) => unknown)
  | bigint
  | symbol
  | undefined;
type BareObject<V> = V extends unknown
  ? [object] extends [V]
    ? [V] extends [object]
      ? [keyof V] extends [never]
        ? V
        : never
      : never
    : never
  : never;
type Shaped<T, Mapped, Excluded = NonJsonValue> = [Extract<T, Excluded> | BareObject<T>] extends [never]
  ? Mapped
  : never;
type JsonProperty<V> = [unknown] extends [V]
  ? unknown
  : V extends JsonValue
    ? V
    : V extends NonJsonValue
      ? never
      : V extends object
        ? [BareObject<V>] extends [never]
          ? { [K in keyof V]: JsonProperty<V[K]> }
          : never
        : never;
type JsonObjectProperty<V> = V extends readonly unknown[] ? never : V extends object ? JsonProperty<V> : never;
type LlmRequestProperty<V> = V extends object
  ? { [K in keyof V]: K extends 'headers' ? JsonObjectProperty<V[K]> : JsonProperty<V[K]> } & Record<
      'headers' | 'content',
      unknown
    >
  : never;
type PendingMarkSpecProperty<V> = V extends object
  ? {
      [K in keyof V]: K extends 'data' | 'metadata' | 'categoryProfile'
        ? JsonProperty<V[K]>
        : K extends keyof PendingMarkSpec
          ? PendingMarkSpec[K]
          : JsonProperty<V[K]>;
    } & Record<'name', unknown>
  : never;
type PendingMarksProperty<V> = V extends readonly unknown[] ? { [I in keyof V]: PendingMarkSpecProperty<V[I]> } : never;
type EventMetadataProperty<V> = V extends EventMetadataValue ? V : never;
type MetricMeasurementProperty<V> = V extends object
  ? {
      [K in keyof V]: K extends 'attributes'
        ? JsonProperty<V[K]>
        : K extends keyof MetricMeasurement
          ? MetricMeasurement[K]
          : unknown;
    } & Record<'name' | 'kind' | 'valueType' | 'value', unknown>
  : never;
interface LlmRequestOpen {
  headers: JsonObject;
  content: Json;
  [key: string]: JsonValue;
}
interface ToolExecutionResultOpen {
  result: Json;
  annotation?: Json;
  [key: string]: JsonValue | undefined;
}
interface EventSanitizeFieldsOpen {
  data?: Json;
  categoryProfile?: Json;
  metadata?: Json;
  [key: string]: JsonValue | undefined;
}

export type JsonCompatible<T> =
  | JsonValue
  | ([T] extends [{} | null] ? Shaped<T, { [K in keyof T]: JsonProperty<T[K]> }> : never);
export type JsonObjectCompatible<T> =
  | JsonObject
  | ([T] extends [object]
      ? Shaped<T, { [K in keyof T]: JsonProperty<T[K]> }, NonJsonValue | readonly unknown[]>
      : never);
export type LlmRequestCompatible<T> =
  | LlmRequestOpen
  | ([T] extends [object]
      ? Shaped<
          T,
          { [K in keyof T]: K extends 'headers' ? JsonObjectProperty<T[K]> : JsonProperty<T[K]> } & Record<
            'headers' | 'content',
            unknown
          >
        >
      : never);
export type ToolExecutionResultCompatible<T> =
  | ToolExecutionResultOpen
  | ([T] extends [object] ? Shaped<T, { [K in keyof T]: JsonProperty<T[K]> } & Record<'result', unknown>> : never);
export type ToolExecutionInterceptOutcomeCompatible<T> =
  | ToolExecutionInterceptOutcome
  | ([T] extends [object]
      ? Shaped<
          T,
          { [K in keyof T]: K extends 'pendingMarks' ? PendingMarksProperty<T[K]> : JsonProperty<T[K]> } & Record<
            'result',
            unknown
          >
        >
      : never);
export type LlmRequestInterceptOutcomeCompatible<T> =
  | LlmRequestInterceptOutcome
  | ([T] extends [object]
      ? Shaped<
          T,
          {
            [K in keyof T]: K extends 'request'
              ? LlmRequestProperty<T[K]>
              : K extends 'pendingMarks'
                ? PendingMarksProperty<T[K]>
                : K extends 'optimizationContributions'
                  ? LlmOptimizationContribution[]
                  : JsonProperty<T[K]>;
          } & Record<'request', unknown>
        >
      : never);
export type EventSanitizeFieldsCompatible<T> =
  | EventSanitizeFieldsOpen
  | ([T] extends [object] ? Shaped<T, { [K in keyof T]: JsonProperty<T[K]> }> : never);
export type EventMetadataCompatible<T> =
  | EventMetadata
  | ([T] extends [object] ? Shaped<T, { [K in keyof T]: EventMetadataProperty<T[K]> }> : never);
export type MetricMeasurementsCompatible<T> =
  | readonly MetricMeasurement[]
  | ([T] extends [readonly unknown[]] ? { [I in keyof T]: MetricMeasurementProperty<T[I]> } : never);

export type RuntimeRegistrationKindName = `${RuntimeRegistrationKind}`;

export type ResourceMetricsMeasurementScope = 'runtime_default' | ResourceMeasurementScope;

export type ResourceUnitKind =
  | DurationUnit
  | CapacityUnit
  | DataUnit
  | BandwidthUnit
  | CpuUnit
  | UtilizationUnit
  | CountUnit;

export type ResourceUnit<U extends ResourceUnitKind> = U | `${U}`;

export interface ResourceMetricsPollingConfig {
  enabled?: boolean;
  interval_millis?: number;
}

export interface ResourceMemoryInspectionConfig {
  enabled?: boolean;
  stale_after_millis?: number;
}

export interface ResourceMetricsCpuConfig {
  enabled?: boolean;
}

export interface ResourceMetricsMemoryConfig {
  enabled?: boolean;
}

export interface ResourceMetricsProcessConfig {
  enabled?: boolean;
}

export interface ResourceMetricsDiskConfig {
  enabled?: boolean;
  process_io?: boolean;
  filesystem_paths?: string[];
}

export interface ResourceMetricsGpuConfig {
  enabled?: boolean;
  devices?: string[];
  device_metrics?: boolean;
  process_metrics?: boolean;
}

export interface ResourceMetricsNetworkConfig {
  enabled?: boolean;
  interfaces?: string[];
}

export interface ResourceMetricsCpuUnits {
  user_time?: ResourceUnit<DurationUnit>;
  system_time?: ResourceUnit<DurationUnit>;
  total_time?: ResourceUnit<DurationUnit>;
  throttled_time?: ResourceUnit<DurationUnit>;
  some_pressure_stall_time?: ResourceUnit<DurationUnit>;
  full_pressure_stall_time?: ResourceUnit<DurationUnit>;
  consumption_rate?: ResourceUnit<CpuUnit>;
  effective_limit?: ResourceUnit<CpuUnit>;
}

export interface ResourceMetricsMemoryUnits {
  system_used?: ResourceUnit<CapacityUnit>;
  system_total?: ResourceUnit<CapacityUnit>;
  system_available?: ResourceUnit<CapacityUnit>;
  resident?: ResourceUnit<CapacityUnit>;
  private?: ResourceUnit<CapacityUnit>;
  physical_footprint?: ResourceUnit<CapacityUnit>;
  virtual_memory?: ResourceUnit<CapacityUnit>;
  peak_resident?: ResourceUnit<CapacityUnit>;
  limit?: ResourceUnit<CapacityUnit>;
  environment_accounted?: ResourceUnit<CapacityUnit>;
  some_pressure_stall_time?: ResourceUnit<DurationUnit>;
  full_pressure_stall_time?: ResourceUnit<DurationUnit>;
}

export interface ResourceMetricsDiskUnits {
  read_data?: ResourceUnit<DataUnit>;
  write_data?: ResourceUnit<DataUnit>;
  filesystem_total_capacity?: ResourceUnit<CapacityUnit>;
  filesystem_available_capacity?: ResourceUnit<CapacityUnit>;
  filesystem_free_capacity?: ResourceUnit<CapacityUnit>;
  read_throughput?: ResourceUnit<BandwidthUnit>;
  write_throughput?: ResourceUnit<BandwidthUnit>;
}

export interface ResourceMetricsGpuUnits {
  device_memory_used?: ResourceUnit<CapacityUnit>;
  process_memory_used?: ResourceUnit<CapacityUnit>;
  device_compute_utilization?: ResourceUnit<UtilizationUnit>;
  process_compute_utilization?: ResourceUnit<UtilizationUnit>;
}

export interface ResourceMetricsNetworkTrafficUnits {
  received_data?: ResourceUnit<DataUnit>;
  transmitted_data?: ResourceUnit<DataUnit>;
  receive_throughput?: ResourceUnit<BandwidthUnit>;
  transmit_throughput?: ResourceUnit<BandwidthUnit>;
}

export interface ResourceMetricsNetworkUnits {
  system?: ResourceMetricsNetworkTrafficUnits;
  interface?: ResourceMetricsNetworkTrafficUnits;
}

export interface ResourceMetricsUnits {
  cpu?: ResourceMetricsCpuUnits;
  memory?: ResourceMetricsMemoryUnits;
  disk?: ResourceMetricsDiskUnits;
  gpu?: ResourceMetricsGpuUnits;
  network?: ResourceMetricsNetworkUnits;
}

export interface ResourceMetricsConfig {
  measurement_scope?: ResourceMetricsMeasurementScope;
  polling?: ResourceMetricsPollingConfig;
  memory_inspection?: ResourceMemoryInspectionConfig;
  cpu?: ResourceMetricsCpuConfig;
  memory?: ResourceMetricsMemoryConfig;
  process?: ResourceMetricsProcessConfig;
  disk?: ResourceMetricsDiskConfig;
  gpu?: ResourceMetricsGpuConfig;
  network?: ResourceMetricsNetworkConfig;
  units?: ResourceMetricsUnits;
}

export type ResourceOperatingSystem = 'linux' | 'macos' | 'windows' | 'unsupported';
export type ResourceMeasurementScope = 'global' | 'application_process' | 'process_tree';

/** Measurement values use bigint for exact integers above JavaScript's safe-integer range. */
export type ResourceNumeric = number | bigint;

export interface ResourceMeasurement<T extends ResourceNumeric, U extends ResourceUnitKind> {
  readonly value: T;
  readonly unit: U;
}

export interface ResourceLimitEventCount {
  readonly resource: 'cpu' | 'memory' | 'processes';
  readonly event: 'throttled' | 'high' | 'maximum' | 'out_of_memory' | 'terminated';
  readonly count: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
}

export type AcceleratorVendor = 'nvidia' | 'amd' | 'intel' | 'apple' | 'other';

export interface AcceleratorDeviceMetrics {
  readonly vendor: AcceleratorVendor;
  readonly deviceIdentifier: string;
  readonly deviceIndex: number | null;
  readonly memoryUsed: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly computeUtilization: ResourceMeasurement<number, UtilizationUnit> | null;
}

export interface AcceleratorProcessMetrics {
  readonly vendor: AcceleratorVendor;
  readonly deviceIdentifier: string;
  readonly deviceIndex: number | null;
  readonly processId: number;
  readonly memoryUsed: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly computeUtilization: ResourceMeasurement<number, UtilizationUnit> | null;
}

export interface FilesystemCapacityMetrics {
  readonly path: string;
  readonly totalCapacity: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly availableCapacity: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly freeCapacity: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
}

export interface CpuMetrics {
  readonly userTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  readonly systemTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  readonly totalTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  readonly consumptionRate: ResourceMeasurement<number, CpuUnit> | null;
  readonly throttledTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  readonly effectiveLimit: ResourceMeasurement<number, CpuUnit> | null;
  readonly somePressureStallTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  readonly fullPressureStallTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  readonly limitEvents: readonly ResourceLimitEventCount[];
}

export interface MemoryMetrics {
  readonly systemUsed: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly systemTotal: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly systemAvailable: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly resident: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly private: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly physicalFootprint: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly virtualMemory: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly peakResident: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly limit: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly environmentAccounted: ResourceMeasurement<ResourceNumeric, CapacityUnit> | null;
  readonly somePressureStallTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  readonly fullPressureStallTime: ResourceMeasurement<ResourceNumeric, DurationUnit> | null;
  readonly outOfMemoryEventCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly limitEvents: readonly ResourceLimitEventCount[];
}

export interface ProcessMetrics {
  readonly activeCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly descendantCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly threadCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly lifetimeCreationCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly openFileDescriptorCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly windowsHandleCount: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly limitEvents: readonly ResourceLimitEventCount[];
}

export interface DiskMetrics {
  readonly readData: ResourceMeasurement<ResourceNumeric, DataUnit> | null;
  readonly writeData: ResourceMeasurement<ResourceNumeric, DataUnit> | null;
  readonly readThroughput: ResourceMeasurement<number, BandwidthUnit> | null;
  readonly writeThroughput: ResourceMeasurement<number, BandwidthUnit> | null;
  readonly readOperations: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly writeOperations: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly filesystems: readonly FilesystemCapacityMetrics[];
}

export interface GpuMetrics {
  readonly deviceMetrics: readonly AcceleratorDeviceMetrics[] | null;
  readonly processMetrics: readonly AcceleratorProcessMetrics[] | null;
}

export interface NetworkTrafficMetrics {
  readonly receivedData: ResourceMeasurement<ResourceNumeric, DataUnit> | null;
  readonly transmittedData: ResourceMeasurement<ResourceNumeric, DataUnit> | null;
  readonly receiveThroughput: ResourceMeasurement<number, BandwidthUnit> | null;
  readonly transmitThroughput: ResourceMeasurement<number, BandwidthUnit> | null;
  readonly receivedPackets: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly transmittedPackets: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly receiveErrors: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
  readonly transmitErrors: ResourceMeasurement<ResourceNumeric, CountUnit> | null;
}

export interface NetworkInterfaceMetrics {
  readonly name: string;
  readonly traffic: NetworkTrafficMetrics;
}

export interface NetworkMetrics {
  readonly measurementScope: 'global';
  readonly system: NetworkTrafficMetrics;
  readonly interfaces: readonly NetworkInterfaceMetrics[];
}

export interface ProcessSamplingMetadata {
  readonly visibleProcesses: ResourceNumeric;
  readonly sampledProcesses: ResourceNumeric;
  /** Readable process counts keyed by canonical snake_case measurement paths. */
  readonly fieldSampledProcesses: Readonly<Record<string, ResourceNumeric>>;
}

export interface ResourceMetricsSnapshot {
  readonly timestamp: string;
  readonly operatingSystem: ResourceOperatingSystem;
  readonly measurementScope: ResourceMeasurementScope;
  readonly processSampling: ProcessSamplingMetadata | null;
  readonly cpu: CpuMetrics | null;
  readonly memory: MemoryMetrics | null;
  readonly process: ProcessMetrics | null;
  readonly disk: DiskMetrics | null;
  readonly gpu: GpuMetrics | null;
  readonly network: NetworkMetrics | null;
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

/** Request codec context shared by sanitizer and execution callbacks. */
export interface LlmSanitizeRequestContext {
  codec: LlmCodecIdentity;
  /** Resolve the active codec for this callback. Do not retain the result after the callback returns. */
  resolveCodec(): import('./typed').LlmCodec | null;
}

/** Response codec context shared by sanitizer and execution callbacks. */
export interface LlmSanitizeResponseContext {
  codec: LlmCodecIdentity;
  /** Resolve the active codec for this callback. Do not retain the result after the callback returns. */
  resolveCodec(): import('./typed').LlmResponseCodec | null;
}

/** Request codec context exposed to an LLM execution intercept. */
export type LlmRequestContext = LlmSanitizeRequestContext;

/** Response codec context exposed to an LLM execution intercept. */
export type LlmResponseContext = LlmSanitizeResponseContext;

/** Codec capabilities for one managed LLM execution intercept invocation. */
export interface LlmExecutionContext {
  /** Request codec identity plus optional decode and encode capability. */
  requestCodec: LlmRequestContext;
  /** Unary response codec identity plus optional decode capability; `null` for streaming execution. */
  responseCodec: LlmResponseContext | null;
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
  [key: string]:
    | Json
    | LlmOptimizationModelTransition
    | LlmOptimizationTokenImpact
    | LlmOptimizationDataSchema
    | undefined;
}

/** Canonical result returned by an LLM request intercept. */
export interface LlmRequestInterceptOutcome {
  request: LlmRequestShape;
  annotated?: Json | null;
  pendingMarks?: PendingMarkSpec[];
  optimizationContributions?: LlmOptimizationContribution[];
  [key: string]: JsonValue | PendingMarkSpec[] | LlmOptimizationContribution[] | undefined;
}

export interface ConfigDiagnostic {
  level: 'warning' | 'error';
  code: string;
  component?: string;
  field?: string;
  message: string;
}
export interface ToolExecutionInterceptOutcome extends ToolExecutionResult {
  pendingMarks?: PendingMarkSpec[];
  [key: string]: JsonValue | PendingMarkSpec[] | undefined;
}
export interface PluginContext {
  registerConditionalMiddlewareGuardrail(
    name: string,
    kinds: Array<RuntimeRegistrationKind | RuntimeRegistrationKindName>,
    registrationName: string,
    guardrail: (kinds: RuntimeRegistrationKind[], registrationName: string) => string | null,
  ): void;
  registerSubscriber(name: string, callback: (event: JsonObject) => void | Promise<void>): void;
  registerEventMetadataInjector<TMetadata extends EventMetadataCompatible<TMetadata>>(
    name: string,
    priority: number,
    callback: (event: JsonObject) => TMetadata | Promise<TMetadata>,
  ): void;
  registerMarkSanitizeGuardrail<TFields extends EventSanitizeFieldsCompatible<TFields>>(
    name: string,
    priority: number,
    callback: (event: JsonObject, fields: EventSanitizeFields) => TFields | Promise<TFields>,
  ): void;
  registerScopeSanitizeStartGuardrail<TFields extends EventSanitizeFieldsCompatible<TFields>>(
    name: string,
    priority: number,
    callback: (event: JsonObject, fields: EventSanitizeFields) => TFields | Promise<TFields>,
  ): void;
  registerScopeSanitizeEndGuardrail<TFields extends EventSanitizeFieldsCompatible<TFields>>(
    name: string,
    priority: number,
    callback: (event: JsonObject, fields: EventSanitizeFields) => TFields | Promise<TFields>,
  ): void;
  registerToolSanitizeRequestGuardrail<TValue extends JsonCompatible<TValue>>(
    name: string,
    priority: number,
    callback: (name: string, args: Json) => TValue | Promise<TValue>,
  ): void;
  registerToolSanitizeResponseGuardrail<TValue extends JsonCompatible<TValue>>(
    name: string,
    priority: number,
    callback: (name: string, result: Json) => TValue | Promise<TValue>,
  ): void;
  registerToolConditionalExecutionGuardrail(
    name: string,
    priority: number,
    callback: (name: string, args: Json) => string | null | Promise<string | null>,
  ): void;
  registerLlmSanitizeRequestGuardrail<TRequest extends LlmRequestCompatible<TRequest> = LlmRequestShape>(
    name: string,
    priority: number,
    callback: (
      request: LlmRequestShape,
      context: LlmSanitizeRequestContext,
    ) => TRequest | null | Promise<TRequest | null>,
  ): void;
  registerLlmSanitizeResponseGuardrail<TResponse extends JsonCompatible<TResponse> = Json>(
    name: string,
    priority: number,
    callback: (response: Json, context: LlmSanitizeResponseContext) => TResponse | null | Promise<TResponse | null>,
  ): void;
  registerLlmConditionalExecutionGuardrail(
    name: string,
    priority: number,
    callback: (request: LlmRequestShape) => string | null | Promise<string | null>,
  ): void;
  registerLlmRequestIntercept<TOutcome extends LlmRequestInterceptOutcomeCompatible<TOutcome>>(
    name: string,
    priority: number,
    breakChain: boolean,
    callback: (args: {
      name: string;
      request: LlmRequestShape;
      annotated: Json | null;
    }) => TOutcome | Promise<TOutcome>,
  ): void;
  registerLlmExecutionIntercept<TResponse extends JsonCompatible<TResponse>>(
    name: string,
    priority: number,
    callback: (
      request: LlmRequestShape,
      context: LlmExecutionContext,
      next: <TNextRequest extends LlmRequestCompatible<TNextRequest>>(request: TNextRequest) => Json | Promise<Json>,
    ) => TResponse | Promise<TResponse>,
  ): void;
  registerLlmStreamExecutionIntercept<TChunk extends JsonCompatible<TChunk>>(
    name: string,
    priority: number,
    callback: (
      request: LlmRequestShape,
      context: LlmExecutionContext,
      next: <TNextRequest extends LlmRequestCompatible<TNextRequest>>(
        request: TNextRequest,
      ) => Promise<AsyncIterable<Json>>,
    ) => AsyncIterable<TChunk> | Promise<AsyncIterable<TChunk>>,
  ): void;
  registerToolRequestIntercept<TArgs extends JsonCompatible<TArgs>>(
    name: string,
    priority: number,
    breakChain: boolean,
    callback: (name: string, args: Json) => TArgs | Promise<TArgs>,
  ): void;
  registerToolExecutionIntercept<TOutcome extends ToolExecutionInterceptOutcomeCompatible<TOutcome>>(
    name: string,
    priority: number,
    callback: (
      context: ToolExecutionContext,
      next: <TNextArgs extends JsonCompatible<TNextArgs>>(
        args: TNextArgs,
      ) => ToolExecutionResult | Promise<ToolExecutionResult>,
    ) => TOutcome | Promise<TOutcome>,
  ): void;
}

/** Scalar value accepted in event metadata additions. */
export type EventMetadataScalar = string | number | boolean;

/**
 * Flat value accepted in event metadata additions. After JSON conversion,
 * numeric arrays must contain only integer values or only floating-point values.
 */
export type EventMetadataValue = EventMetadataScalar | readonly string[] | readonly number[] | readonly boolean[];

/** Metadata additions returned by an event metadata injector. */
export type EventMetadata = Record<string, EventMetadataValue>;
