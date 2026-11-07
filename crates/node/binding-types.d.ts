// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// <reference lib="esnext.disposable" />

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

/** `bigint` above JavaScript's safe integer range. */
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
  /** Process counts keyed by snake_case measurement path. */
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

/** Codec identity during LLM event sanitization. */
export type LlmCodecIdentity =
  | { kind: 'none' }
  | {
      kind: 'builtin';
      id: 'openai_chat' | 'openai_responses' | 'anthropic_messages' | 'oci_genai' | 'gemini_generate_content';
    }
  | { kind: 'runtime'; id: string }
  | { kind: 'opaque' };

export interface LlmCodec<
  TAnnotated extends JsonCompatible<TAnnotated> = JsonValue,
  TEncoded extends LlmRequestCompatible<TEncoded> = LlmRequestShape,
> {
  decode: (request: LlmRequestShape) => TAnnotated;
  encode: (annotated: JsonValue, original: LlmRequestShape) => TEncoded;
}

export interface LlmResponseCodec<TDecoded extends JsonCompatible<TDecoded> = JsonValue> {
  decodeResponse: (response: JsonValue) => TDecoded;
}

/** Request codec context for sanitizer and execution callbacks. */
export interface LlmSanitizeRequestContext {
  codec: LlmCodecIdentity;
  /** Active codec, valid until the callback returns. */
  resolveCodec(): LlmCodec | null;
}

/** Response codec context for sanitizer and execution callbacks. */
export interface LlmSanitizeResponseContext {
  codec: LlmCodecIdentity;
  /** Active codec, valid until the callback returns. */
  resolveCodec(): LlmResponseCodec | null;
}

/** Request codec context for an LLM execution intercept. */
export type LlmRequestContext = LlmSanitizeRequestContext;

/** Response codec context for an LLM execution intercept. */
export type LlmResponseContext = LlmSanitizeResponseContext;

/** Codecs available to one LLM execution intercept call. */
export interface LlmExecutionContext {
  /** Request codec, with decode and encode when available. */
  requestCodec: LlmRequestContext;
  /** Response codec, with decode when available; `null` for streaming. */
  responseCodec: LlmResponseContext | null;
}

/** Schema tag for an optimization payload. */
export interface LlmOptimizationDataSchema {
  name: string;
  version: string;
}

/** Model identity kept for repricing. */
export interface LlmOptimizationModel {
  model: string;
  provider?: string;
}

/** Baseline and effective models of a routing optimization. */
export interface LlmOptimizationModelTransition {
  baseline?: LlmOptimizationModel;
  effective?: LlmOptimizationModel;
}

/** Token counts reported directly. */
export interface LlmOptimizationTokens {
  /** Non-negative safe integer. */
  prompt_tokens?: number;
  /** Non-negative safe integer. */
  completion_tokens?: number;
  /** Non-negative safe integer. */
  cache_read_tokens?: number;
  /** Non-negative safe integer. */
  cache_write_tokens?: number;
  /** Non-negative safe integer. */
  total_tokens?: number;
}

/** Baseline, effective, and saved tokens of one optimization. */
export interface LlmOptimizationTokenImpact {
  baseline?: LlmOptimizationTokens;
  effective?: LlmOptimizationTokens;
  saved?: LlmOptimizationTokens;
  quality?: 'observed' | 'estimated';
  estimation_method?: string;
}

/** One plugin's optimization evidence. `kind` accepts any string and unknown fields are kept. */
export interface LlmOptimizationContribution {
  id?: string;
  /** Safe integer. */
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

/** Result of an LLM request intercept. */
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

/** Scalar accepted in event metadata. */
export type EventMetadataScalar = string | number | boolean;

/** Value accepted in event metadata; numeric arrays are all integers or all floats. */
export type EventMetadataValue = EventMetadataScalar | readonly string[] | readonly number[] | readonly boolean[];

/** Metadata returned by an event metadata injector. */
export type EventMetadata = Record<string, EventMetadataValue>;

/** Validation or activation report for a plugin config. */
export interface ConfigReport {
  diagnostics: ConfigDiagnostic[];
  runtime_diagnostics?: RuntimeDiagnostic[];
}

/** One aggregated runtime plugin failure. */
export interface RuntimeDiagnostic {
  code: string;
  component: string;
  field?: string;
  message: string;
  session_id?: string;
  count: number;
}

/** How a dynamic plugin runs. */
export type DynamicPluginKind = 'rust_dynamic' | 'worker';

export interface PluginHostReport {
  config: ConfigReport;
  dynamic_plugins: DynamicPluginValidationReport[];
  /** `plugins.toml` files merged into the resolved config. */
  config_paths: string[];
  /** Merged plugin config with sensitive values redacted. */
  resolved_config: Json;
}

export type DynamicPluginCheckState = 'unknown' | 'valid' | 'invalid';

export interface DynamicPluginValidationStatus {
  manifest: DynamicPluginCheckState;
  compatibility: DynamicPluginCheckState;
  integrity: DynamicPluginCheckState;
  environment: DynamicPluginCheckState;
  authenticity: DynamicPluginCheckState;
  policy_satisfied: DynamicPluginCheckState;
  checked_at?: string | null;
  message?: string | null;
}

export interface DynamicPluginFailure {
  phase: string;
  code: string;
  message: string;
}

export interface DynamicPluginValidationReport {
  plugin_id: string;
  manifest_ref: string;
  kind: DynamicPluginKind;
  status: DynamicPluginValidationStatus;
  failure?: DynamicPluginFailure | null;
  selected: boolean;
}

/** Prompt-cache write tokens for one cache duration. */
export interface CacheWriteTokenCounts {
  count: number;
  /** Cache duration in seconds, when the provider reports it. */
  duration?: number;
}

/** Token usage for cache telemetry. */
export interface CacheUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  cache_read_tokens?: number;
  cache_write_tokens?: CacheWriteTokenCounts[];
  cost?: Json;
}

/** Agent identity in cache telemetry. */
export interface AgentIdentity {
  agent_id: string;
  template_version: string;
  toolset_hash: string;
  model_family: string;
  tenant_scope: string;
}

/** Input to `buildCacheRequestFacts`. */
export interface CacheRequestFactsOptions {
  provider: string;
  requestId: string;
  annotatedRequest: Json;
  agentId: string;
  timestamp?: string;
}

/** Request facts that classify cache misses. */
export interface CacheRequestFacts {
  provider: string;
  stable_prefix_length: number;
  stable_prefix_tokens?: number;
  required_min_tokens?: number;
  first_mismatch_span_id?: string;
  first_mismatch_sequence_index?: number;
  expected_hash_prefix?: string;
  actual_hash_prefix?: string;
  retention_window_secs?: number;
  observed_gap_secs?: number;
  missing_facts?: string[];
}

/** Input to `buildCacheTelemetryEvent`. */
export interface CacheTelemetryEventOptions {
  provider: 'anthropic' | 'openai' | string;
  requestId: string;
  usage?: CacheUsage | null;
  requestFacts?: CacheRequestFacts | null;
  agentId: string;
  templateVersion: string;
  toolsetHash: string;
  modelFamily: string;
  tenantScope: string;
  timestamp?: string;
}

/** Cache telemetry event. */
export interface CacheTelemetryEvent {
  request_id: string;
  agent_identity: AgentIdentity;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  total_prompt_tokens: number;
  hit_rate: number;
  miss_reason?: Record<string, Json>;
  miss_diagnosis?: Record<string, Json>;
  provider: string;
  timestamp: string;
}

/** Diagnostic integer, written as a decimal string in canonical JSON. */
export type DiagnosticNumber = bigint;
/** Producer of the evidence. */
export interface SourceIdentity {
  /** Unique id of this process run. */
  instanceId: string;
  /** Operating system process ID, not an identity key. */
  processId: number | null;
  /** Role, such as gateway or worker. */
  processRole: string;
  /** Relay version that produced the evidence. */
  relayVersion: string;
}

/** Component registration. */
export interface ComponentIdentity {
  /** Logical name without request or session ids. */
  key: string;
  /** Unique id of this registration. */
  instanceId: string;
  /** Generation, increased when a logical key is replaced. */
  generation: bigint;
}

/** Owner and configuration of a component. */
export interface ComponentOwnership {
  /** Subscriber registration, when applicable. */
  subscriberInstanceId: string | null;
  /** Plugin registration, when applicable. */
  pluginInstanceId: string | null;
  /** Id of the committed configuration. */
  configurationId: string | null;
  /** Runtime responsibility of the owner. */
  responsibility: string | null;
}

/** Where an observation came from. */
export interface DiagnosticOrigin {
  /** Refresh generation of the observation. */
  refreshGeneration: bigint;
  /** Original owner and configuration. */
  ownership: ComponentOwnership;
}

/** Session and request of an observation. */
export interface DiagnosticCorrelation {
  /** Trajectory session, when known. */
  sessionId?: string | null;
  /** Request, when known. */
  requestId?: string | null;
}

/** Code and cause of an observation. */
export interface Classification {
  /** Built-in name or registered extension name. */
  canonicalCode: string;
  /** Cause name, absent for successes and notices. */
  failureClass: string | null;
}

/** Text attached to an observation. */
export interface DiagnosticDetail {
  /** Unsanitized text cut to the byte capacity. */
  message: string;
  /** Whether the text was cut. */
  truncated: boolean;
}

/** One retained observation. */
export interface DiagnosticEvent {
  /** Unique update sequence, not an occurrence count. */
  sequence: bigint;
  /** Observation time at the source. */
  observedAt: string;
  /** Registration that made the observation. */
  component: ComponentIdentity;
  /** Configuration at observation time. */
  origin: DiagnosticOrigin;
  /** Remote submission origin, absent for local observations. */
  remote?: RemoteDiagnosticOrigin | null;
  /** Event classification. */
  classification: Classification;
  /** Plugin operation facts, excluded from aggregation keys. */
  context: PluginObservationContext;
  /** Severity, independent of outcome. */
  severity: string;
  /** Operation outcome. */
  outcome: string;
  /** Message attached to the observation, or `null`. */
  detail: DiagnosticDetail | null;
  /** Occurrences added by this recording. */
  count: bigint;
  /** Caller correlation, excluded from aggregation identity. */
  correlation: DiagnosticCorrelation;
}

/** Origin of a remote submission. */
export interface RemoteDiagnosticOrigin {
  /** Host-issued session id. */
  sourceInstanceId: string;
  /** Host-issued plugin activation id. */
  activationInstanceId: string;
  /** Submission sequence, not the host event sequence. */
  submissionSequence: bigint;
  /** Producer-reported wall time, if any. */
  observedAt: string | null;
  /** Host wall time at acceptance, used as the event observation time. */
  receivedAt: string;
  /** Host monotonic time at acceptance since context startup. */
  acceptedElapsedMillis: bigint;
}

/** Count that may saturate. */
export interface DiagnosticCount {
  /** Exact count unless saturated. */
  value: bigint;
  /** Whether overflow made the value a lower bound. */
  saturated: boolean;
}

/** Occurrence totals by outcome. */
export interface DiagnosticTotals {
  /** Accepted recording calls. */
  recordedEventsTotal: DiagnosticCount;
  /** Accepted occurrences across outcomes and codes. */
  occurrencesTotal: DiagnosticCount;
  /** Retained events later evicted. */
  historyEvictionsTotal: DiagnosticCount;
  /** Occurrences with a successful outcome. */
  successesTotal: DiagnosticCount;
  /** Occurrences with a failed outcome. */
  failuresTotal: DiagnosticCount;
  /** Occurrences with no outcome. */
  unspecifiedTotal: DiagnosticCount;
}

/** Totals over a lifetime. */
export interface DiagnosticLifetimeSummary {
  /** Totals for the context or component. */
  totals: DiagnosticTotals;
  /** Elapsed lifetime at capture. */
  elapsedMillis: bigint;
  /** Wall-clock time of the latest failure. */
  lastFailureAt: string | null;
  /** Elapsed time since the latest failure, or `null` when none was observed. */
  sinceLastFailureMillis: bigint | null;
}

/** Totals since the last refresh. */
export interface DiagnosticRefreshSummary {
  /** Generation, increased by a successful refresh. */
  generation: bigint;
  /** Wall-clock time of the refresh. */
  startedAt: string;
  /** Elapsed time since the refresh. */
  elapsedMillis: bigint;
  /** Configuration and owner for this period. */
  ownership: ComponentOwnership;
  /** Counts for this period, separate from lifetime totals. */
  totals: DiagnosticTotals;
}

/** Occurrences of one code. */
export interface DiagnosticCounter {
  /** Canonical code of the counted observations. */
  code: string;
  /** Accepted occurrences. */
  occurrences: DiagnosticCount;
  /** Time of the last update. */
  lastObservedAt: string;
}

/** One time bucket of outcomes. */
export interface DiagnosticWindowBucket {
  /** Inclusive start offset from component start. */
  startElapsedMillis: bigint;
  /** Exclusive end offset of the bucket. */
  endElapsedMillis: bigint;
  /** Successful occurrences in this interval. */
  successes: DiagnosticCount;
  /** Failed occurrences in this interval. */
  failures: DiagnosticCount;
  /** Occurrences with no outcome in this interval. */
  unspecified: DiagnosticCount;
}

/** Interval covered by a window. */
export interface DiagnosticWindowCoverage {
  /** Availability of the window series. */
  availability: string;
  /** Earliest covered offset. */
  startElapsedMillis: bigint;
  /** Latest covered offset at capture. */
  endElapsedMillis: bigint;
  /** First bucket is only partly covered. */
  partialStart: boolean;
  /** Last bucket is still open or partly selected. */
  partialEnd: boolean;
  /** Known loss makes the interval incomplete. */
  incomplete: boolean;
}

/** Fixed-bucket outcome series. */
export interface DiagnosticWindow {
  /** Bucket duration. */
  resolutionMillis: bigint;
  /** Configured bucket count. */
  bucketCapacity: number;
  /** Selected buckets, zero when idle. */
  buckets: Array<DiagnosticWindowBucket>;
  /** Covered interval and edge limits. */
  coverage: DiagnosticWindowCoverage;
}

/** Position for the next page. */
export interface DiagnosticCursor {
  /** Source run whose state was selected. */
  sourceInstanceId: string;
  /** Source revision at the time the cursor was issued. */
  sequence: bigint;
  /** Next component position. */
  componentOffset: bigint;
  /** Next event position. */
  eventOffset: bigint;
  /** Fingerprint of the query scope. */
  scope: bigint;
}

/** Selection for a snapshot. */
export interface DiagnosticQuery {
  /** Snapshot schema version to request. */
  schemaVersion?: number | null;
  /** Selects one component key. */
  componentKey?: string | null;
  /** Selects one registration instance. */
  componentInstanceId?: string | null;
  /** Selects recent events by refresh generation. */
  originGeneration?: bigint | null;
  /** Selects one code. */
  code?: string | null;
  /** Selects a catalogue domain. */
  domain?: string | null;
  /** Selects a plugin catalogue family. */
  pluginFamily?: string | null;
  /** Selects event causes. */
  failureClass?: string | null;
  /** Selects components by current ownership and events by original ownership. */
  ownership?: Partial<ComponentOwnership>;
  /** Selects events after this sequence. */
  afterSequence?: bigint | null;
  /** Maximum number of component rows to return. */
  maxComponents?: number | null;
  /** Maximum number of events to return. */
  maxEvents?: number | null;
  /** Maximum encoded response size in bytes. */
  maxResponseBytes?: number | null;
  /** Resumes a selection with the same scope. */
  continuation?: DiagnosticCursor | null;
}

/** History bounds of a selection. */
export interface DiagnosticRetention {
  /** Oldest selected event sequence, if any. */
  oldestEventSequence: bigint | null;
  /** Latest selected event sequence, if any. */
  latestEventSequence: bigint | null;
  /** Returned event count. */
  eventsRetained: bigint;
  /** Returned component count. */
  componentsRetained: bigint;
  /** Whether query limits cut matching evidence. */
  queryTruncated: boolean;
}

/** Cumulative loss counters. */
export interface DiagnosticLoss {
  /** Events evicted from history. */
  eventsEvicted: DiagnosticCount;
  /** Observations accepted without retaining the event. */
  eventsOmitted: DiagnosticCount;
  /** Closed components evicted with their evidence. */
  componentsEvicted: DiagnosticCount;
  /** Component registrations rejected by limits. */
  componentsRejected: DiagnosticCount;
  /** Code registrations rejected by limits. */
  keysRejected: DiagnosticCount;
  /** Observations rejected before counting. */
  observationsRejected: DiagnosticCount;
  /** Detail cut to its byte limit. */
  messagesTruncated: DiagnosticCount;
  /** Detail dropped without evicting an event. */
  detailsOmitted: DiagnosticCount;
  /** Correlation fields cut to their byte limit. */
  correlationsTruncated: DiagnosticCount;
  /** Correlation fields dropped for lack of space. */
  correlationsOmitted: DiagnosticCount;
  /** Metadata registrations rejected by budget. */
  metadataRejected: DiagnosticCount;
  /** Counted occurrences whose classification was not admitted. */
  classificationsOverflowed: DiagnosticCount;
  /** Window series rejected by budget. */
  windowsRejected: DiagnosticCount;
  /** Queries rejected for lack of workspace. */
  queriesRejected: DiagnosticCount;
  /** Recordings for closed or replaced registrations. */
  staleObservations: DiagnosticCount;
  /** Projection deliveries dropped for lack of workspace. */
  projectionDeliveriesOmitted: DiagnosticCount;
  /** Projection callbacks that panicked. */
  projectionsPanicked: DiagnosticCount;
  /** Projection chains suppressed for cycles or depth. */
  projectionsSuppressed: DiagnosticCount;
  /** Sequence numbers are exhausted. */
  sequenceSaturated: boolean;
}

/** One producer integration. */
export interface DiagnosticCapability {
  /** Capability name, such as `otel`. */
  name: string;
  /** Availability of this capability's evidence. */
  availability: string;
  /** Reason, such as `not_integrated`. */
  reason: string | null;
}

/** Limits of a snapshot. */
export interface DiagnosticCoverage {
  /** Availability of this diagnostic source. */
  availability: string;
  /** Scope of component summaries and windows. */
  summaryScope: string;
  /** Selected history bounds and query cuts. */
  retention: DiagnosticRetention;
  /** Cumulative loss counters. */
  loss: DiagnosticLoss;
  /** Producer integrations and whether each is supported. */
  capabilities: Array<DiagnosticCapability>;
}

/** One component row. */
export interface ComponentSnapshot {
  /** Logical name and registration instance. */
  identity: ComponentIdentity;
  /** Registration time. */
  registeredAt: string;
  /** Time the registration closed, or `null` while it is open. */
  closedAt: string | null;
  /** Lifetime totals and elapsed time. */
  lifetime: DiagnosticLifetimeSummary;
  /** Current refresh period with its owner. */
  sinceRefresh: DiagnosticRefreshSummary;
  /** Selected lifetime counters by code. */
  counters: Array<DiagnosticCounter>;
  /** Outcome windows of the component. */
  windows: Array<DiagnosticWindow>;
}

/** One diagnostics page. */
export interface DiagnosticSnapshot {
  /** Schema version of this snapshot. */
  schemaVersion: number;
  /** Revision of the code catalogue. */
  catalogueVersion: number;
  /** Process that produced the snapshot. */
  source: SourceIdentity;
  /** Capture time at the source. */
  capturedAt: string;
  /** Latest update revision. */
  sequence: bigint;
  /** Lifetime totals for the whole context, ignoring selectors. */
  lifetime: DiagnosticLifetimeSummary;
  /** One row per selected component. */
  components: Array<ComponentSnapshot>;
  /** Selected recent events. */
  events: Array<DiagnosticEvent>;
  /** Availability, scope, integration, and loss limits. */
  coverage: DiagnosticCoverage;
  /** Cursor for the next page, or `null` when there is none. */
  continuation: DiagnosticCursor | null;
}

/** Plugin operation facts. */
export interface PluginObservationContext {
  /** Execution name, when known. */
  execution: string | null;
  /** Lifecycle phase or policy gate, when known. */
  phase: string | null;
}

/** Capabilities of the diagnostics core. */
export interface DiagnosticDiscovery {
  discoveryVersion: number;
  snapshotVersion: number;
  catalogueVersion: number;
  source: SourceIdentity;
  collectionEnabled: boolean;
  queriesEnabled: boolean;
  maxComponents: bigint;
  maxEvents: bigint;
  detailPoolBytes: bigint;
  maxQueries: bigint;
  maxSnapshotBytes: bigint;
  maxOwnedBytes: bigint;
  maxQueryComponents: bigint;
  maxQueryEvents: bigint;
  memoryInspection: boolean;
  restrictedDetail: boolean;
  producerSupport: Array<{ name: string; supported: boolean }>;
  detailedMemoryInspection: boolean;
  resourceCorrelation: boolean;
}

export interface DiagnosticsInfo {
  discovery: DiagnosticDiscovery;
  limits: Record<string, number>;
  memory: DiagnosticMemoryUsage;
  bindingCopyConcurrency: number;
  bindingMaxPageBytes: number;
  bindingReservedBytes: number;
  loggingCoverage: string;
}

/** Resource number with explicit precision. */
export type ResourceMemoryNumber =
  | { kind: 'integer'; value: bigint }
  | { kind: 'decimal'; value: number }
  | { kind: 'non_finite' };
export interface ResourceMemoryUnit {
  dimension: 'capacity' | 'duration' | 'count';
  name: string;
}

/** Reserved bytes and occupancy gauges. */
export interface DiagnosticMemoryUsage {
  reservedBytes: bigint;
  components: bigint;
  counters: bigint;
  events: bigint;
  eventSlotBytes: bigint;
  eventCapacity: bigint;
  historyReservedBytes: bigint;
  historyOccupiedBytes: bigint;
  detailBytes: bigint;
  detailCapacityBytes: bigint;
  detailReservedBytes: bigint;
  detailMetadataBytes: bigint;
  detailLogicalBytes: bigint;
  detailSlackBytes: bigint;
  metadataBytes: bigint;
  windowBuckets: bigint;
  extensions: bigint;
  queryWorkspaces: bigint;
  projections: bigint;
  projectionWorkspaces: bigint;
}

/** One event header and its text. */
export interface EntryMemoryFootprint {
  headerBytes: bigint;
  detailPayloadBytes: bigint;
  detailMetadataBytes: bigint;
  logicalTextBytes: bigint;
  detailSlackBytes: bigint;
  occupiedBytes: bigint;
}

/** Storage owned by one registration, excluding plugin heap. */
export interface ComponentMemoryFootprint {
  registrationBytes: bigint;
  counterBytes: bigint;
  windowBytes: bigint;
  extensionBytes: bigint;
  retainedEventCount: bigint;
  retainedEventBytes: bigint;
  occupiedBytes: bigint;
}

/** One event and its footprint. */
export interface EntryMemoryUsage extends EntryMemoryFootprint {
  sequence: bigint;
  component: ComponentIdentity;
}

/** One registration and its footprint. */
export interface ComponentMemoryUsage extends ComponentMemoryFootprint {
  component: ComponentIdentity;
}

/** Row counts and byte totals. */
export interface MemorySelectionTotals {
  components: bigint;
  events: bigint;
  componentOccupiedBytes: bigint;
  entryOccupiedBytes: bigint;
}

/** One memory page with resource context. */
export interface DiagnosticMemoryPage {
  schemaVersion: number;
  source: SourceIdentity;
  capturedAt: string;
  sequence: bigint;
  aggregate: DiagnosticMemoryUsage;
  loss: DiagnosticLoss;
  componentOccupiedBytes: bigint;
  sharedOrUnassignedReservedBytes: bigint;
  selection: MemorySelectionTotals;
  page: MemorySelectionTotals;
  components: Array<ComponentMemoryUsage>;
  entries: Array<EntryMemoryUsage>;
  resource: ResourceMemoryView;
  continuation: DiagnosticCursor | null;
}

/** Resource value with its number kind, unit, scope, and coverage. */
export interface ResourceMemoryMeasurement {
  value: ResourceMemoryNumber;
  unit: ResourceMemoryUnit;
  measurementScope: string;
  sampledProcesses: bigint | null;
}

/** One cumulative limit event. */
export interface MemoryLimitEvent {
  resource: string;
  event: string;
  count: ResourceMemoryMeasurement | null;
}

/** Memory measurements and up to eight limit events. */
export interface ResourceMemoryMetrics {
  systemUsed: ResourceMemoryMeasurement | null;
  systemTotal: ResourceMemoryMeasurement | null;
  systemAvailable: ResourceMemoryMeasurement | null;
  resident: ResourceMemoryMeasurement | null;
  private: ResourceMemoryMeasurement | null;
  physicalFootprint: ResourceMemoryMeasurement | null;
  virtualMemory: ResourceMemoryMeasurement | null;
  peakResident: ResourceMemoryMeasurement | null;
  limit: ResourceMemoryMeasurement | null;
  environmentAccounted: ResourceMemoryMeasurement | null;
  somePressureStallTime: ResourceMemoryMeasurement | null;
  fullPressureStallTime: ResourceMemoryMeasurement | null;
  outOfMemoryEventCount: ResourceMemoryMeasurement | null;
  limitEvents: Array<MemoryLimitEvent | null>;
}

/** Root process identity checked during collection. */
export interface ResourceProcessIdentity {
  processId: number;
  startIdentity: bigint;
}

/** One collection and its memory result. */
export interface ResourceMemorySample {
  collectorInstanceId: string;
  runtimeGeneration: bigint;
  targetGeneration: bigint;
  stream: string;
  ordinal: bigint;
  startedAt: string;
  completedAt: string;
  collectionMillis: bigint;
  operatingSystem: string;
  measurementScope: string;
  root: ResourceProcessIdentity | null;
  visibleProcesses: bigint | null;
  sampledProcesses: bigint | null;
  memory: ResourceMemoryMetrics;
}

/** Collection availability, association, and freshness. */
export interface ResourceMemoryView {
  availability: string;
  sample: ResourceMemorySample | null;
  association: string;
  ageMillis: bigint | null;
  stale: boolean;
  projectionSlotBytes: bigint;
}
