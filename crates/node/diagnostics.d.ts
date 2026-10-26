// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Lossless native inspection values. Canonical JSON remains snake_case with decimal strings. */
export type DiagnosticNumber = bigint;
/** SourceIdentity preserves the shared diagnostic contract. */
export interface SourceIdentity {
  /** Unique incarnation assigned before logging initialization. */
  instanceId: string;
  /** Descriptive operating system process ID, never an identity key. */
  processId: number | null;
  /** Producer-defined role, such as embedded runtime, gateway, broker, or worker. */
  processRole: string;
  /** Relay package version producing the evidence. */
  relayVersion: string;
}

/** ComponentIdentity preserves the shared diagnostic contract. */
export interface ComponentIdentity {
  /** Stable logical name, excluding request and session identifiers. */
  key: string;
  /** Unique registration instance, distinguishing replacement components. */
  instanceId: string;
  /** Source-local registration generation, increasing when a logical key is replaced. */
  generation: bigint;
}

/** ComponentOwnership preserves the shared diagnostic contract. */
export interface ComponentOwnership {
  /** Subscriber registration associated with this attribution, when applicable. */
  subscriberInstanceId: string | null;
  /** Plugin registration associated with this attribution, when applicable. */
  pluginInstanceId: string | null;
  /** Bounded identity of the committed configuration. */
  configurationId: string | null;
  /** Bounded runtime responsibility for an embedded or process owner. */
  responsibility: string | null;
}

/** DiagnosticOrigin preserves the shared diagnostic contract. */
export interface DiagnosticOrigin {
  /** Refresh generation associated with the originating work or observation. */
  refreshGeneration: bigint;
  /** Original bounded owner and configuration attribution. */
  ownership: ComponentOwnership;
}

/** DiagnosticCorrelation preserves the shared diagnostic contract. */
export interface DiagnosticCorrelation {
  /** Trajectory session associated with the observation, when known. */
  sessionId?: string | null;
  /** Request associated with the observation, when known. */
  requestId?: string | null;
}

/** Classification preserves the shared diagnostic contract. */
export interface Classification {
  /** Stable built-in name or registered namespaced extension name. */
  canonicalCode: string;
  /** Cause name; absent for successful observations and notices. */
  failureClass: string | null;
}

/** DiagnosticDetail preserves the shared diagnostic contract. */
export interface DiagnosticDetail {
  /** Text copied within admitted byte capacity; bounding does not sanitize it. */
  message: string;
  /** Whether the original text exceeded its retained byte allowance. */
  truncated: boolean;
}

/** DiagnosticEvent preserves the shared diagnostic contract. */
export interface DiagnosticEvent {
  /** Unique source-local update sequence; it is not an occurrence counter. */
  sequence: bigint;
  /** Source observation time, preserved through projections. */
  observedAt: string;
  /** Registration that owned the original observation. */
  component: ComponentIdentity;
  /** Original configuration attribution, preserved through component refresh. */
  origin: DiagnosticOrigin;
  /** Authenticated remote submission provenance, absent for local observations. */
  remote?: RemoteDiagnosticOrigin | null;
  /** Canonical event classification; never reconstructed from a latest message. */
  classification: Classification;
  /** Bounded plugin-operation wire facts, excluded from aggregation keys. */
  context: PluginObservationContext;
  /** Evidence severity, independent of its observed operation outcome. */
  severity: string;
  /** Explicit observed operation outcome. */
  outcome: string;
  /** Bounded optional evidence; absence does not imply a missing counter. */
  detail: DiagnosticDetail | null;
  /** Input occurrence delta represented by this recording call. */
  count: bigint;
  /** Restricted caller correlation, excluded from aggregation identity. */
  correlation: DiagnosticCorrelation;
}

/** RemoteDiagnosticOrigin preserves the shared diagnostic contract. */
export interface RemoteDiagnosticOrigin {
  /** Host-issued diagnostic session/source incarnation. */
  sourceInstanceId: string;
  /** Host-issued plugin activation incarnation, distinct from the session. */
  activationInstanceId: string;
  /** Ordered terminal submission sequence; not the host event sequence. */
  submissionSequence: bigint;
  /** Optional producer-reported wall time, with unknown clock uncertainty. */
  observedAt: string | null;
  /** Host wall time at acceptance, also used as the event observation time. */
  receivedAt: string;
  /** Host monotonic elapsed time at acceptance, relative to context startup. */
  acceptedElapsedMillis: bigint;
}

/** DiagnosticCount preserves the shared diagnostic contract. */
export interface DiagnosticCount {
  /** Exact count unless saturation has occurred. */
  value: bigint;
  /** Whether an overflowing increment made the value a lower bound. */
  saturated: boolean;
}

/** DiagnosticTotals preserves the shared diagnostic contract. */
export interface DiagnosticTotals {
  /** Accepted recording calls, one per call regardless of occurrence delta. */
  recordedEventsTotal: DiagnosticCount;
  /** Accepted occurrence deltas across all outcomes and codes. */
  occurrencesTotal: DiagnosticCount;
  /** Previously retained events owned by this component or context and evicted. */
  historyEvictionsTotal: DiagnosticCount;
  /** Occurrences whose owning operation explicitly succeeded. */
  successesTotal: DiagnosticCount;
  /** Occurrences whose owning operation explicitly failed. */
  failuresTotal: DiagnosticCount;
  /** Occurrences without an established operation outcome. */
  unspecifiedTotal: DiagnosticCount;
}

/** DiagnosticLifetimeSummary preserves the shared diagnostic contract. */
export interface DiagnosticLifetimeSummary {
  /** Whole-context or whole-component-instance accounting, as named by the view. */
  totals: DiagnosticTotals;
  /** Monotonic elapsed lifetime measured by the source at capture. */
  elapsedMillis: bigint;
  /** Wall-clock correlation time of the latest observed lifetime failure. */
  lastFailureAt: string | null;
  /** Monotonic elapsed time since that failure; absence never means recovery. */
  sinceLastFailureMillis: bigint | null;
}

/** DiagnosticRefreshSummary preserves the shared diagnostic contract. */
export interface DiagnosticRefreshSummary {
  /** Component-local generation increased by a successful refresh. */
  generation: bigint;
  /** Wall-clock correlation anchor for the successful refresh. */
  startedAt: string;
  /** Monotonic elapsed duration since the refresh boundary. */
  elapsedMillis: bigint;
  /** Current configuration and owner attribution for this counting period. */
  ownership: ComponentOwnership;
  /** Counts for this current period, independently reserved from lifetime totals. */
  totals: DiagnosticTotals;
}

/** DiagnosticCounter preserves the shared diagnostic contract. */
export interface DiagnosticCounter {
  /** Canonical code; the component instance comes from the containing row. */
  code: string;
  /** Accepted occurrences, independent of optional recent detail. */
  occurrences: DiagnosticCount;
  /** Source time when this retained counter was last updated. */
  lastObservedAt: string;
}

/** DiagnosticWindowBucket preserves the shared diagnostic contract. */
export interface DiagnosticWindowBucket {
  /** Inclusive elapsed offset from the component's monotonic start. */
  startElapsedMillis: bigint;
  /** Exclusive nominal bucket end; coverage describes the observed portion. */
  endElapsedMillis: bigint;
  /** Explicit successful-operation occurrences observed in this interval. */
  successes: DiagnosticCount;
  /** Explicit failed-operation occurrences observed in this interval. */
  failures: DiagnosticCount;
  /** Occurrences with no established operation outcome in this interval. */
  unspecified: DiagnosticCount;
}

/** DiagnosticWindowCoverage preserves the shared diagnostic contract. */
export interface DiagnosticWindowCoverage {
  /** Availability of the admitted window series. */
  availability: string;
  /** Earliest covered monotonic elapsed offset within this component lifetime. */
  startElapsedMillis: bigint;
  /** Latest covered monotonic elapsed offset at capture time. */
  endElapsedMillis: bigint;
  /** The first returned bucket is only partly inside the covered interval. */
  partialStart: boolean;
  /** The last returned bucket is still being observed or partly selected. */
  partialEnd: boolean;
  /** Known loss prevents treating the aggregate as a complete interval. */
  incomplete: boolean;
}

/** DiagnosticWindow preserves the shared diagnostic contract. */
export interface DiagnosticWindow {
  /** Duration of each fixed bucket. */
  resolutionMillis: bigint;
  /** Configured bucket slots; memory is independent of observation rate. */
  bucketCapacity: number;
  /** Bounded selected aggregates; idle covered intervals may have zero values. */
  buckets: Array<DiagnosticWindowBucket>;
  /** Actual interval and edge limitations. */
  coverage: DiagnosticWindowCoverage;
}

/** DiagnosticCursor preserves the shared diagnostic contract. */
export interface DiagnosticCursor {
  /** Source incarnation whose retained state was selected. */
  sourceInstanceId: string;
  /** Source revision for detecting intervening mutations or reclamation. */
  sequence: bigint;
  /** Next component-instance position in the bounded source ordering. */
  componentOffset: bigint;
  /** Next recent-event position in the bounded source ordering. */
  eventOffset: bigint;
  /** Source-validated fingerprint of the intersected query scope. */
  scope: bigint;
}

/** DiagnosticQuery preserves the shared diagnostic contract. */
export interface DiagnosticQuery {
  /** Optional canonical snapshot version, rejected when unsupported. */
  schemaVersion?: number | null;
  /** Select one logical component key across retained registrations. */
  componentKey?: string | null;
  /** Select one registration incarnation. */
  componentInstanceId?: string | null;
  /** Select recent events by original refresh generation only. Lifetime and current-refresh summaries keep their declared scope. */
  originGeneration?: bigint | null;
  /** Select exact canonical code counters and events. */
  code?: string | null;
  /** Select the authoritative catalogue parent domain. */
  domain?: string | null;
  /** Select the plugin catalogue family derived from code ownership. */
  pluginFamily?: string | null;
  /** Select event causes only; code counters do not retain cause breakdowns. */
  failureClass?: string | null;
  /** Select component rows by current-refresh ownership and events by original ownership. Selected lifetime counters still span the entire component instance. */
  ownership?: Partial<ComponentOwnership>;
  /** Select later recent events; summary and code counters remain cumulative. */
  afterSequence?: bigint | null;
  /** Requested component-instance row cap, rejected above core limits. */
  maxComponents?: number | null;
  /** Requested recent-event cap, rejected above core limits. */
  maxEvents?: number | null;
  /** Requested encoded response cap, rejected above core limits. */
  maxResponseBytes?: number | null;
  /** Resume a compatible bounded selection without granting additional scope. */
  continuation?: DiagnosticCursor | null;
}

/** DiagnosticRetention preserves the shared diagnostic contract. */
export interface DiagnosticRetention {
  /** Oldest sequence in the selected retained event history, if present. */
  oldestEventSequence: bigint | null;
  /** Latest sequence in the selected retained event history, if present. */
  latestEventSequence: bigint | null;
  /** Number of returned recent events. */
  eventsRetained: bigint;
  /** Number of returned component-instance rows. */
  componentsRetained: bigint;
  /** Query limits omitted otherwise matching retained evidence. */
  queryTruncated: boolean;
}

/** DiagnosticLoss preserves the shared diagnostic contract. */
export interface DiagnosticLoss {
  /** Previously retained events reclaimed from the circular history. */
  eventsEvicted: DiagnosticCount;
  /** Accepted observations whose incoming event evidence was not retained. */
  eventsOmitted: DiagnosticCount;
  /** Closed component metadata reclaimed with dependent evidence. */
  componentsEvicted: DiagnosticCount;
  /** Component registrations rejected by admission limits. */
  componentsRejected: DiagnosticCount;
  /** Code reservations or registrations rejected by admission limits. */
  keysRejected: DiagnosticCount;
  /** Observations rejected before canonical counting. */
  observationsRejected: DiagnosticCount;
  /** Detail shortened within its configured byte allowance. */
  messagesTruncated: DiagnosticCount;
  /** Incoming optional detail omitted without evicting a retained event. */
  detailsOmitted: DiagnosticCount;
  /** Correlation fields shortened within their own byte allowance. */
  correlationsTruncated: DiagnosticCount;
  /** Safe correlation fields omitted because detail storage could not fit them. */
  correlationsOmitted: DiagnosticCount;
  /** Metadata registrations rejected by their admission budget. */
  metadataRejected: DiagnosticCount;
  /** Accepted summary-only deltas whose classification could not be admitted. */
  classificationsOverflowed: DiagnosticCount;
  /** Window series rejected by their independent admission budget. */
  windowsRejected: DiagnosticCount;
  /** Query materializations rejected by bounded workspace admission. */
  queriesRejected: DiagnosticCount;
  /** Recording attempts for closed or replaced component registrations. */
  staleObservations: DiagnosticCount;
  /** Optional projection deliveries omitted for lack of bounded workspace. */
  projectionDeliveriesOmitted: DiagnosticCount;
  /** Projection callbacks whose panic was isolated after acceptance. */
  projectionsPanicked: DiagnosticCount;
  /** Cyclic or excessively deep projection chains suppressed. */
  projectionsSuppressed: DiagnosticCount;
  /** Unique sequence allocation has exhausted its representation. */
  sequenceSaturated: boolean;
}

/** DiagnosticCapability preserves the shared diagnostic contract. */
export interface DiagnosticCapability {
  /** Bounded capability name, such as `otel`. */
  name: string;
  /** Availability of evidence from this capability. */
  availability: string;
  /** Bounded explanatory reason, such as `not_integrated`. */
  reason: string | null;
}

/** DiagnosticCoverage preserves the shared diagnostic contract. */
export interface DiagnosticCoverage {
  /** Availability of this diagnostic source context. */
  availability: string;
  /** Scope of component summaries and component outcome windows. */
  summaryScope: string;
  /** Bounded selected history boundaries and query omission. */
  retention: DiagnosticRetention;
  /** Reserved cumulative loss and capacity accounting. */
  loss: DiagnosticLoss;
  /** Producer integrations whose support or absence is explicitly advertised. */
  capabilities: Array<DiagnosticCapability>;
}

/** ComponentSnapshot preserves the shared diagnostic contract. */
export interface ComponentSnapshot {
  /** Logical responsibility and registration incarnation. */
  identity: ComponentIdentity;
  /** Source correlation time when the registration was admitted. */
  registeredAt: string;
  /** Registration closure; retained summaries remain subject to bounded retention. */
  closedAt: string | null;
  /** Whole-component-instance lifetime accounting and source elapsed time. */
  lifetime: DiagnosticLifetimeSummary;
  /** The sole current refresh period, with explicitly scoped current ownership. */
  sinceRefresh: DiagnosticRefreshSummary;
  /** Selected lifetime code counters; no latest-message set is retained per code. */
  counters: Array<DiagnosticCounter>;
  /** Admitted bounded component outcome windows, independent of refresh periods. */
  windows: Array<DiagnosticWindow>;
}

/** DiagnosticSnapshot preserves the shared diagnostic contract. */
export interface DiagnosticSnapshot {
  /** Supported envelope version, independent of the catalogue version. */
  schemaVersion: number;
  /** Stable catalogue revision; unknown names stay opaque in this envelope. */
  catalogueVersion: number;
  /** Source identity preserved through collection and projection. */
  source: SourceIdentity;
  /** Source capture time; it is not a collector receipt timestamp. */
  capturedAt: string;
  /** Latest source update revision; reads do not increment it. */
  sequence: bigint;
  /** Whole-context lifetime accounting, independent of component/event selectors. Hosts must authorize access to this context-wide view before exposing it. */
  lifetime: DiagnosticLifetimeSummary;
  /** Bounded rows, one per selected retained component instance. */
  components: Array<ComponentSnapshot>;
  /** Bounded selected recent evidence. */
  events: Array<DiagnosticEvent>;
  /** Availability, scope, integration, and loss limitations. */
  coverage: DiagnosticCoverage;
  /** Next bounded page, subject to mutation and retention validation. */
  continuation: DiagnosticCursor | null;
}

/** PluginObservationContext preserves the shared diagnostic contract. */
export interface PluginObservationContext {
  /** Observed execution name, when known; never inferred from a code. */
  execution: string | null;
  /** Observed lifecycle phase or policy-gate context, when known. */
  phase: string | null;
}

/** Core capability view; support does not imply active/admitted producers or evidence. */
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

/** Stable reader failure; messages contain no query or retained evidence. */
export class DiagnosticsError extends Error {
  readonly reason: string;
}
/** Context inspection authority. Never expose it to untrusted clients without authorization. */
export class DiagnosticsReader {
  /** Capture a native camelCase page. All diagnostic u64 values are bigint. */
  snapshot(query?: DiagnosticQuery): DiagnosticSnapshot;
  /** Capture portable canonical JSON, retaining decimal-string diagnostic integers. */
  snapshotJson(query?: DiagnosticQuery): string;
  /** Inspect bounded backing and existing scoped resource context. */
  inspectMemory(query?: DiagnosticQuery): DiagnosticMemoryPage;
  /** Canonical memory JSON with lossless integer strings. */
  inspectMemoryJson(query?: DiagnosticQuery): string;
  /** Inspect separate core/copy budgets, occupancy, and logging coverage. */
  info(): {
    discovery: DiagnosticDiscovery;
    limits: Record<string, number>;
    memory: DiagnosticMemoryUsage;
    bindingCopyConcurrency: number;
    bindingMaxPageBytes: number;
    bindingReservedBytes: number;
    loggingCoverage: string;
  };
}
/** Obtain a reader without initializing the service; may throw not_initialized. */
export function reader(): DiagnosticsReader;

/** Resource precision is explicit; non-finite values have no numeric value. */
export type ResourceMemoryNumber =
  | { kind: 'integer'; value: bigint }
  | { kind: 'decimal'; value: number }
  | { kind: 'non_finite' };
export interface ResourceMemoryUnit {
  dimension: 'capacity' | 'duration' | 'count';
  name: string;
}

/** Reserved diagnostic backing and overlapping occupancy gauges. */
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

/** One retained event header and its exclusive packed text chain. */
export interface EntryMemoryFootprint {
  headerBytes: bigint;
  detailPayloadBytes: bigint;
  detailMetadataBytes: bigint;
  logicalTextBytes: bigint;
  detailSlackBytes: bigint;
  occupiedBytes: bigint;
}

/** Registration-owned diagnostic storage, independent of plugin heap usage. */
export interface ComponentMemoryFootprint {
  registrationBytes: bigint;
  counterBytes: bigint;
  windowBytes: bigint;
  extensionBytes: bigint;
  retainedEventCount: bigint;
  retainedEventBytes: bigint;
  occupiedBytes: bigint;
}

/** Identity and backing footprint of a retained observation. */
export interface EntryMemoryUsage extends EntryMemoryFootprint {
  sequence: bigint;
  component: ComponentIdentity;
}

/** Identity and complete attributed backing of a registration. */
export interface ComponentMemoryUsage extends ComponentMemoryFootprint {
  component: ComponentIdentity;
}

/** Matching row counts and overlapping byte totals. */
export interface MemorySelectionTotals {
  components: bigint;
  events: bigint;
  componentOccupiedBytes: bigint;
  entryOccupiedBytes: bigint;
}

/** One bounded attribution page with independent resource context. */
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

/** A resource field with its original numeric kind, unit, scope, and coverage. */
export interface ResourceMemoryMeasurement {
  value: ResourceMemoryNumber;
  unit: ResourceMemoryUnit;
  measurementScope: string;
  sampledProcesses: bigint | null;
}

/** One cumulative environment limit event. */
export interface MemoryLimitEvent {
  resource: string;
  event: string;
  count: ResourceMemoryMeasurement | null;
}

/** Memory-only measurements and at most eight limit-event slots. */
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

/** Root process incarnation verified during resource collection. */
export interface ResourceProcessIdentity {
  processId: number;
  startIdentity: bigint;
}

/** One collection identity, interval, scope, and memory result. */
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

/** Independent collection availability, association, and freshness. */
export interface ResourceMemoryView {
  availability: string;
  sample: ResourceMemorySample | null;
  association: string;
  ageMillis: bigint | null;
  stale: boolean;
  projectionSlotBytes: bigint;
}
