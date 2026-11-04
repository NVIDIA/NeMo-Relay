// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Declarative struct and enum mappings, with collection wrappers.
//!
//! Field modes describe presence and conversion behavior; only renamed fields
//! need both names. Keep semantic reshaping in `variants.rs`.

use super::*;

dto!(nemo_relay_types::api::event::DataSchema => DataSchema {
    name: scalar,
    version: scalar,
});

dto!(nemo_relay_types::api::llm::LlmRequest => LlmRequest {
    headers: required,
    content: required,
});

dto!(nemo_relay_types::codec::request::AnnotatedLlmRequest => AnnotatedLlmRequest {
    messages: repeated,
    instructions: optional,
    model: optional,
    params: optional,
    tools: optional,
    tool_choice: optional,
    store: optional,
    previous_response_id: optional,
    truncation: optional,
    reasoning: optional,
    include: optional,
    user: optional,
    metadata: optional,
    service_tier: optional,
    parallel_tool_calls: optional,
    max_output_tokens: optional,
    max_tool_calls: optional,
    top_logprobs: optional,
    stream: optional,
    api_specific: codec_data,
    extra: map,
});

dto!(nemo_relay_types::codec::request::OpenAiImageUrl => ImageUrl {
    url: scalar,
    detail: optional,
});

dto!(nemo_relay_types::codec::request::ToolCall => ToolCall {
    id: scalar,
    call_type: scalar,
    function: required,
});

dto!(nemo_relay_types::codec::request::FunctionCall => FunctionCall {
    name: scalar,
    arguments: scalar,
});

dto!(nemo_relay_types::codec::request::FunctionDefinition => FunctionDefinition {
    name: scalar,
    description: optional,
    parameters: optional,
    strict: optional,
    extra: map,
});

dto!(nemo_relay_types::codec::request::ToolChoiceFunction => ToolChoiceFunction {
    choice_type: scalar,
    function: required,
});

dto!(nemo_relay_types::codec::request::ToolChoiceFunctionName => ToolChoiceFunctionName {
    name: scalar,
});

dto!(nemo_relay_types::codec::request::ProviderNativeComponent => ProviderNativeComponent {
    provider: scalar,
    kind: scalar,
    value: required,
});

dto!(nemo_relay_types::codec::request::GenerationParams => GenerationParams {
    temperature: optional,
    max_tokens: optional,
    top_p: optional,
    stop: optional,
});

dto!(nemo_relay_types::codec::response::AnnotatedLlmResponse => AnnotatedLlmResponse {
    id: optional,
    model: optional,
    message: optional,
    tool_calls: optional,
    finish_reason: optional,
    usage: optional,
    optimization_summary: optional,
    api_specific: codec_data,
    extra: map,
});

dto!(nemo_relay_types::codec::response::ResponseToolCall => ResponseToolCall {
    id: scalar,
    name: scalar,
    arguments: required,
});

dto!(nemo_relay_types::codec::response::Usage => Usage {
    prompt_tokens: optional,
    completion_tokens: optional,
    total_tokens: optional,
    cache_read_tokens: optional,
    cache_write_tokens: optional,
    cache_write_1h_tokens: optional,
    uncached_input_tokens: optional,
    cost: optional,
});

dto!(nemo_relay_types::codec::response::CostEstimate => CostEstimate {
    total: optional,
    currency: scalar,
    input: optional,
    output: optional,
    cache_read: optional,
    cache_write: optional,
    source: scalar,
    pricing_provider: optional,
    pricing_model: optional,
    pricing_as_of: optional,
    pricing_source: optional,
});

dto!(nemo_relay_types::codec::optimization::LlmOptimizationModel => LlmOptimizationModel {
    model: scalar,
    provider: optional,
});

dto!(nemo_relay_types::codec::optimization::LlmOptimizationModelTransition => LlmOptimizationModelTransition {
    baseline: optional,
    effective: optional,
});

dto!(nemo_relay_types::codec::optimization::LlmOptimizationTokens => LlmOptimizationTokens {
    prompt_tokens: optional,
    completion_tokens: optional,
    cache_read_tokens: optional,
    cache_write_tokens: optional,
    total_tokens: optional,
});

dto!(nemo_relay_types::codec::optimization::LlmOptimizationTokenImpact => LlmOptimizationTokenImpact {
    baseline: optional,
    effective: optional,
    saved: optional,
    quality: optional,
    estimation_method: optional,
});

dto!(nemo_relay_types::codec::optimization::LlmOptimizationContribution => LlmOptimizationContribution {
    id: optional,
    sequence: optional,
    producer: scalar,
    kind: scalar,
    applied: scalar,
    model_transition: optional,
    token_impact: optional,
    payload_schema: optional,
    payload: optional,
    extra: map,
});

dto!(nemo_relay_types::codec::optimization::LlmOptimizationSummary => LlmOptimizationSummary {
    schema_version: scalar,
    calculation_version: scalar,
    status: scalar,
    limitations: repeated,
    baseline_model: optional,
    effective_model: optional,
    effective_usage: optional,
    baseline_usage: optional,
    tokens_saved: required,
    baseline_cost: optional,
    actual_cost: optional,
    estimated_cost_saved: optional,
    currency: optional,
    contributions: repeated,
});

dto!(nemo_relay_types::api::event::BaseEvent => BaseEvent {
    atof_version: scalar,
    parent_uuid: optional,
    propagation_root_uuid: optional,
    propagation_parent_uuid: optional,
    propagation_traceparent: optional,
    propagation_tracestate: optional,
    uuid: scalar,
    timestamp => time: required,
    name: scalar,
    data: optional,
    data_schema: optional,
    metadata: optional,
});

dto!(nemo_relay_types::api::event::ScopeEvent => ScopeEvent {
    base: required,
    scope_category: scalar,
    attributes: repeated,
    category: scalar,
    category_profile: optional,
});

dto!(nemo_relay_types::api::event::MarkEvent => MarkEvent {
    base: required,
    category: optional,
    category_profile: optional,
});

dto!(nemo_relay_types::api::event::CategoryProfile => CategoryProfile {
    model_name: optional,
    tool_call_id: optional,
    subtype: optional,
    tool_result_annotation: annotation,
    extra: map,
    annotated_request: optional,
    annotated_response: optional,
    time_to_first_chunk => time_to_first_chunk_seconds: optional,
});

dto!(nemo_relay_types::api::event::EventSanitizeFields => EventSanitizeFields {
    data: optional,
    category_profile: optional,
    metadata: optional,
});

dto!(nemo_relay_types::api::event::PendingMarkSpec => PendingMarkSpec {
    name: scalar,
    category: optional,
    category_profile: optional,
    data: optional,
    data_schema: optional,
    metadata: optional,
    severity: optional,
});

dto!(nemo_relay_types::plugin::ConfigDiagnostic => ConfigDiagnostic {
    level: scalar,
    code: scalar,
    component: optional,
    field: optional,
    message: scalar,
});

dto!(nemo_relay_types::api::llm::LlmRequestInterceptOutcome => LlmRequestInterceptOutcome {
    request: required,
    annotated_request: optional,
    pending_marks: repeated,
    optimization_contributions: repeated,
});

dto!(nemo_relay_types::api::tool::ToolExecutionResult => ToolExecutionResult {
    result: required,
    annotation: annotation,
});

dto!(nemo_relay_types::api::scope::ScopeStackEntry => ScopeStackEntry {
    scope_id: scalar,
    scope_type: scalar,
});

dto!(nemo_relay_types::api::scope::ScopeStackSnapshot => ScopeStackSnapshot {
    scopes: repeated,
});

dto!(nemo_relay_types::api::registry::RuntimeRegistrationIdentity => RuntimeRegistrationIdentity {
    kind: scalar,
    local_name: scalar,
    effective_name: scalar,
    owner: required,
});

dto!(nemo_relay_types::diagnostics::ResourceMemoryMeasurement => ResourceMemoryMeasurement {
    value: required,
    unit: required,
    measurement_scope: scalar,
    sampled_processes: optional,
});

dto!(nemo_relay_types::diagnostics::MemoryLimitEvent => MemoryLimitEvent {
    resource: scalar,
    event: scalar,
    count: optional,
});

dto!(nemo_relay_types::diagnostics::ResourceProcessIdentity => ResourceProcessIdentity {
    process_id: scalar,
    start_identity: scalar,
});

dto!(nemo_relay_types::diagnostics::ResourceMemorySample => ResourceMemorySample {
    collector_instance_id: scalar,
    runtime_generation: scalar,
    target_generation: scalar,
    stream: scalar,
    ordinal: scalar,
    started_at => start_time: required,
    completed_at => complete_time: required,
    collection_millis: scalar,
    operating_system: scalar,
    measurement_scope: scalar,
    root: optional,
    visible_processes: optional,
    sampled_processes: optional,
    memory: required,
});

dto!(nemo_relay_types::diagnostics::ResourceMemoryView => ResourceMemoryView {
    availability: scalar,
    sample: optional,
    association: scalar,
    age_millis: optional,
    stale: scalar,
    projection_slot_bytes: scalar,
});

dto!(nemo_relay_types::diagnostics::DiagnosticSnapshot => DiagnosticSnapshot {
    schema_version: scalar,
    catalogue_version: scalar,
    source: required,
    captured_at => capture_time: required,
    sequence: scalar,
    lifetime: required,
    components: repeated,
    events: repeated,
    coverage: required,
    continuation: optional,
});

dto!(nemo_relay_types::diagnostics::SourceIdentity => DiagnosticSourceIdentity {
    instance_id: scalar,
    process_id: optional,
    process_role: scalar,
    relay_version: scalar,
});

dto!(nemo_relay_types::diagnostics::ComponentIdentity => DiagnosticComponentIdentity {
    key: scalar,
    instance_id: scalar,
    generation: scalar,
});

dto!(nemo_relay_types::diagnostics::ComponentOwnership => DiagnosticComponentOwnership {
    subscriber_instance_id: optional,
    plugin_instance_id: optional,
    configuration_id: optional,
    responsibility: optional,
});

dto!(nemo_relay_types::diagnostics::DiagnosticOrigin => DiagnosticOrigin {
    refresh_generation: scalar,
    ownership: required,
});

dto!(nemo_relay_types::diagnostics::DiagnosticDetail => DiagnosticDetail {
    message: scalar,
    truncated: scalar,
});

dto!(nemo_relay_types::diagnostics::RemoteDiagnosticOrigin => RemoteDiagnosticOrigin {
    source_instance_id: scalar,
    activation_instance_id: scalar,
    submission_sequence: scalar,
    observed_at => observe_time: optional,
    received_at => receive_time: required,
    accepted_elapsed_millis: scalar,
});

dto!(nemo_relay_types::diagnostics::DiagnosticEvent => DiagnosticEvent {
    sequence: scalar,
    observed_at => observe_time: required,
    component: required,
    origin: required,
    remote: optional,
    classification: required,
    context: required,
    severity: scalar,
    outcome: scalar,
    detail: optional,
    count: scalar,
    correlation: required,
});

dto!(nemo_relay_types::diagnostics::DiagnosticCount => DiagnosticCount {
    value: scalar,
    saturated: scalar,
});

dto!(nemo_relay_types::diagnostics::DiagnosticTotals => DiagnosticTotals {
    recorded_events_total: required,
    occurrences_total: required,
    history_evictions_total: required,
    successes_total: required,
    failures_total: required,
    unspecified_total: required,
});

dto!(nemo_relay_types::diagnostics::DiagnosticLifetimeSummary => DiagnosticLifetimeSummary {
    totals: required,
    elapsed_millis: scalar,
    last_failure_at => last_failure_time: optional,
    since_last_failure_millis: optional,
});

dto!(nemo_relay_types::diagnostics::DiagnosticRefreshSummary => DiagnosticRefreshSummary {
    generation: scalar,
    started_at => start_time: required,
    elapsed_millis: scalar,
    ownership: required,
    totals: required,
});

dto!(nemo_relay_types::diagnostics::DiagnosticCounter => DiagnosticCounter {
    code: scalar,
    occurrences: required,
    last_observed_at => last_observe_time: required,
});

dto!(nemo_relay_types::diagnostics::DiagnosticWindowBucket => DiagnosticWindowBucket {
    start_elapsed_millis: scalar,
    end_elapsed_millis: scalar,
    successes: required,
    failures: required,
    unspecified: required,
});

dto!(nemo_relay_types::diagnostics::DiagnosticWindowCoverage => DiagnosticWindowCoverage {
    availability: scalar,
    start_elapsed_millis: scalar,
    end_elapsed_millis: scalar,
    partial_start: scalar,
    partial_end: scalar,
    incomplete: scalar,
});

dto!(nemo_relay_types::diagnostics::DiagnosticWindow => DiagnosticWindow {
    resolution_millis: scalar,
    bucket_capacity: scalar,
    buckets: repeated,
    coverage: required,
});

dto!(nemo_relay_types::diagnostics::ComponentSnapshot => DiagnosticComponentSnapshot {
    identity: required,
    registered_at => register_time: required,
    closed_at => close_time: optional,
    lifetime: required,
    since_refresh: required,
    counters: repeated,
    windows: repeated,
});

dto!(nemo_relay_types::diagnostics::DiagnosticRetention => DiagnosticRetention {
    oldest_event_sequence: optional,
    latest_event_sequence: optional,
    events_retained: scalar,
    components_retained: scalar,
    query_truncated: scalar,
});

dto!(nemo_relay_types::diagnostics::DiagnosticLoss => DiagnosticLoss {
    events_evicted: required,
    events_omitted: required,
    components_evicted: required,
    components_rejected: required,
    keys_rejected: required,
    observations_rejected: required,
    messages_truncated: required,
    details_omitted: required,
    correlations_truncated: required,
    correlations_omitted: required,
    metadata_rejected: required,
    classifications_overflowed: required,
    windows_rejected: required,
    queries_rejected: required,
    stale_observations: required,
    projection_deliveries_omitted: required,
    projections_panicked: required,
    projections_suppressed: required,
    sequence_saturated: scalar,
});

dto!(nemo_relay_types::diagnostics::DiagnosticCapability => DiagnosticCapability {
    name: scalar,
    availability: scalar,
    reason: optional,
});

dto!(nemo_relay_types::diagnostics::DiagnosticCoverage => DiagnosticCoverage {
    availability: scalar,
    summary_scope: scalar,
    retention: required,
    loss: required,
    capabilities: repeated,
});

dto!(nemo_relay_types::diagnostics::DiagnosticCursor => DiagnosticCursor {
    source_instance_id: scalar,
    sequence: scalar,
    component_offset: scalar,
    event_offset: scalar,
    scope: scalar,
});

dto!(nemo_relay_types::diagnostics::DiagnosticMemoryUsage => DiagnosticMemoryUsage {
    reserved_bytes: scalar,
    components: scalar,
    counters: scalar,
    events: scalar,
    event_slot_bytes: scalar,
    event_capacity: scalar,
    history_reserved_bytes: scalar,
    history_occupied_bytes: scalar,
    detail_bytes: scalar,
    detail_capacity_bytes: scalar,
    detail_reserved_bytes: scalar,
    detail_metadata_bytes: scalar,
    detail_logical_bytes: scalar,
    detail_slack_bytes: scalar,
    metadata_bytes: scalar,
    window_buckets: scalar,
    extensions: scalar,
    query_workspaces: scalar,
    projections: scalar,
    projection_workspaces: scalar,
});

dto!(nemo_relay_types::diagnostics::EntryMemoryFootprint => EntryMemoryFootprint {
    header_bytes: scalar,
    detail_payload_bytes: scalar,
    detail_metadata_bytes: scalar,
    logical_text_bytes: scalar,
    detail_slack_bytes: scalar,
    occupied_bytes: scalar,
});

dto!(nemo_relay_types::diagnostics::ComponentMemoryFootprint => ComponentMemoryFootprint {
    registration_bytes: scalar,
    counter_bytes: scalar,
    window_bytes: scalar,
    extension_bytes: scalar,
    retained_event_count: scalar,
    retained_event_bytes: scalar,
    occupied_bytes: scalar,
});

dto!(nemo_relay_types::diagnostics::EntryMemoryUsage => EntryMemoryUsage {
    sequence: scalar,
    component: required,
    footprint: required,
});

dto!(nemo_relay_types::diagnostics::ComponentMemoryUsage => ComponentMemoryUsage {
    component: required,
    footprint: required,
});

dto!(nemo_relay_types::diagnostics::MemorySelectionTotals => MemorySelectionTotals {
    components: scalar,
    events: scalar,
    component_occupied_bytes: scalar,
    entry_occupied_bytes: scalar,
});

dto!(nemo_relay_types::diagnostics::DiagnosticMemoryPage => DiagnosticMemoryPage {
    schema_version: scalar,
    source: required,
    captured_at => capture_time: required,
    sequence: scalar,
    aggregate: required,
    component_occupied_bytes: scalar,
    shared_or_unassigned_reserved_bytes: scalar,
    loss: required,
    selection: required,
    page: required,
    components: repeated,
    entries: repeated,
    resource: required,
    continuation: optional,
});

dto!(nemo_relay_types::diagnostics::DiagnosticQuery => DiagnosticQuery {
    schema_version: optional,
    component_key: optional,
    component_instance_id: optional,
    origin_generation: optional,
    code: optional,
    domain: optional,
    plugin_family: optional,
    failure_class: optional,
    ownership: required,
    after_sequence: optional,
    max_components: optional,
    max_events: optional,
    max_response_bytes: optional,
    continuation: optional,
});

dto!(nemo_relay_types::diagnostics::DiagnosticProducerSupport => DiagnosticProducerSupport {
    name: scalar,
    supported: scalar,
});

dto!(nemo_relay_types::diagnostics::DiagnosticDiscovery => DiagnosticDiscovery {
    discovery_version: scalar,
    snapshot_version: scalar,
    catalogue_version: scalar,
    source: required,
    collection_enabled: scalar,
    queries_enabled: scalar,
    max_components: scalar,
    max_events: scalar,
    detail_pool_bytes: scalar,
    max_queries: scalar,
    max_snapshot_bytes: scalar,
    max_owned_bytes: scalar,
    memory_inspection: scalar,
    restricted_detail: scalar,
    producer_support => producer_support_entries: repeated,
    max_query_components: scalar,
    max_query_events: scalar,
    detailed_memory_inspection: scalar,
    resource_correlation: scalar,
});

enum_codec!(nemo_relay_types::api::scope::ScopeType => ScopeType {
    Agent => Agent,
    Function => Function,
    Tool => Tool,
    Llm => Llm,
    Retriever => Retriever,
    Embedder => Embedder,
    Reranker => Reranker,
    Guardrail => Guardrail,
    Evaluator => Evaluator,
    Custom => Custom,
    Unknown => Unknown,
});

enum_codec!(nemo_relay_types::api::event::LogSeverity => LogSeverity {
    Trace => Trace,
    Debug => Debug,
    Info => Info,
    Warn => Warn,
    Error => Error,
});

enum_codec!(nemo_relay_types::codec::response::CostSource => CostSource {
    ModelPricing => ModelPricing,
    ProviderReported => ProviderReported,
});

enum_codec!(nemo_relay_types::codec::optimization::LlmOptimizationEvidenceQuality => LlmOptimizationEvidenceQuality {
    Observed => Observed,
    Estimated => Estimated,
});

enum_codec!(nemo_relay_types::codec::optimization::LlmOptimizationSummaryStatus => LlmOptimizationSummaryStatus {
    Complete => Complete,
    Partial => Partial,
});

enum_codec!(nemo_relay_types::api::event::ScopeCategory => ScopeCategory {
    Start => Start,
    End => End,
});

enum_codec!(nemo_relay_types::plugin::DiagnosticLevel => ConfigDiagnosticLevel {
    Warning => Warning,
    Error => Error,
});

enum_codec!(nemo_relay_types::diagnostics::ResourceMemoryAvailability => ResourceMemoryAvailability {
    Disabled => Disabled,
    AwaitingTarget => AwaitingTarget,
    NotYetSampled => NotYetSampled,
    Available => Available,
    Failed => Failed,
    Invalidated => Invalidated,
    CapacityUnavailable => CapacityUnavailable,
    Unsupported => Unsupported,
});

enum_codec!(nemo_relay_types::diagnostics::ResourceSampleStream => ResourceSampleStream {
    Polling => Polling,
    OnDemand => OnDemand,
});

enum_codec!(nemo_relay_types::diagnostics::ResourceAssociation => ResourceAssociation {
    HostProcess => HostProcess,
    HostProcessTree => HostProcessTree,
    OtherProcessTree => OtherProcessTree,
    System => System,
    Unverified => Unverified,
});

enum_codec!(nemo_relay_types::diagnostics::MemoryFieldScope => MemoryFieldScope {
    System => System,
    SelectedProcesses => SelectedProcesses,
    Environment => Environment,
});

enum_codec!(nemo_relay_types::api::resource_metrics::ResourceMeasurementScope => ResourceMeasurementScope {
    Global => Global,
    ApplicationProcess => ApplicationProcess,
    ProcessTree => ProcessTree,
});

enum_codec!(nemo_relay_types::api::resource_metrics::ResourceOperatingSystem => ResourceOperatingSystem {
    Linux => Linux,
    Macos => Macos,
    Windows => Windows,
    Unsupported => Unsupported,
});

enum_codec!(nemo_relay_types::api::resource_metrics::CapacityUnit => CapacityUnit {
    Bytes => Bytes,
    Kilobytes => Kilobytes,
    Megabytes => Megabytes,
    Gigabytes => Gigabytes,
    Terabytes => Terabytes,
    Kibibytes => Kibibytes,
    Mebibytes => Mebibytes,
    Gibibytes => Gibibytes,
    Tebibytes => Tebibytes,
});

enum_codec!(nemo_relay_types::api::resource_metrics::DurationUnit => DurationUnit {
    Microseconds => Microseconds,
    Milliseconds => Milliseconds,
    Seconds => Seconds,
    Minutes => Minutes,
});

enum_codec!(nemo_relay_types::api::resource_metrics::CountUnit => CountUnit {
    Processes => Processes,
    Threads => Threads,
    FileDescriptors => FileDescriptors,
    Handles => Handles,
    Events => Events,
    Operations => Operations,
    Packets => Packets,
    Errors => Errors,
});

enum_codec!(nemo_relay_types::api::resource_metrics::ResourceLimitResource => ResourceLimitResource {
    Cpu => Cpu,
    Memory => Memory,
    Processes => Processes,
});

enum_codec!(nemo_relay_types::api::resource_metrics::ResourceLimitEventKind => ResourceLimitEventKind {
    Throttled => Throttled,
    High => High,
    Maximum => Maximum,
    OutOfMemory => OutOfMemory,
    Terminated => Terminated,
});

enum_codec!(nemo_relay_types::diagnostics::DiagnosticAvailability => DiagnosticAvailability {
    Available => Available,
    Unavailable => Unavailable,
    Unsupported => Unsupported,
    NeverObserved => NeverObserved,
    Disabled => Disabled,
});

enum_codec!(nemo_relay_types::diagnostics::DiagnosticSummaryScope => DiagnosticSummaryScope {
    WholeComponentLifetime => WholeComponentLifetime,
});

enum_codec!(nemo_relay_types::diagnostics::DiagnosticDomain => DiagnosticDomain {
    Daemon => Daemon,
    Cli => Cli,
    Plugin => Plugin,
    Logging => Logging,
});

enum_codec!(nemo_relay_types::diagnostics::PluginDiagnosticFamily => PluginDiagnosticFamily {
    Common => Common,
    Operations => Operations,
});

enum_codec!(nemo_relay_types::diagnostics::DiagnosticReadReason => DiagnosticReadReason {
    Disabled => Disabled,
    QueryDisabled => QueryDisabled,
    UnsupportedSchema => UnsupportedSchema,
    InvalidQuery => InvalidQuery,
    InvalidCursor => InvalidCursor,
    Busy => Busy,
    SnapshotCapacity => SnapshotCapacity,
    InvalidLimits => InvalidLimits,
    AllocationFailed => AllocationFailed,
    Internal => Internal,
});

list_codec!(String => StringList);
list_codec!(Json => JsonArray);
list_codec!(nemo_relay_types::codec::request::ContentPart => ContentPartList);
list_codec!(nemo_relay_types::codec::request::ToolCall => ToolCallList);
list_codec!(nemo_relay_types::codec::request::ToolDefinition => ToolDefinitionList);
list_codec!(nemo_relay_types::codec::response::ResponseToolCall => ResponseToolCallList);

dto!(nemo_relay_types::diagnostics::PluginObservationContext => DiagnosticPluginContext {
    execution: optional,
    phase: optional,
});

enum_codec!(nemo_relay_types::diagnostics::DiagnosticSeverity => DiagnosticObservationSeverity {
    Trace => Trace, Debug => Debug, Info => Info, Warning => Warning, Error => Error,
});
enum_codec!(nemo_relay_types::diagnostics::DiagnosticOutcome => DiagnosticObservationOutcome {
    Unknown => Unspecified, Success => Success, Failure => Failure,
});

enum_codec!(nemo_relay_types::api::registry::RuntimeRegistrationKind => RegistrationSurface {
    Subscriber => Subscriber,
    EventMetadataInjector => EventMetadataInjector,
    MarkSanitizeGuardrail => MarkSanitizeGuardrail,
    ScopeSanitizeStartGuardrail => ScopeSanitizeStartGuardrail,
    ScopeSanitizeEndGuardrail => ScopeSanitizeEndGuardrail,
    ToolSanitizeRequestGuardrail => ToolSanitizeRequestGuardrail,
    ToolSanitizeResponseGuardrail => ToolSanitizeResponseGuardrail,
    ToolConditionalExecutionGuardrail => ToolConditionalExecutionGuardrail,
    ToolRequestIntercept => ToolRequestIntercept,
    ToolExecutionIntercept => ToolExecutionIntercept,
    LlmSanitizeRequestGuardrail => LlmSanitizeRequestGuardrail,
    LlmSanitizeResponseGuardrail => LlmSanitizeResponseGuardrail,
    LlmConditionalExecutionGuardrail => LlmConditionalExecutionGuardrail,
    LlmRequestIntercept => LlmRequestIntercept,
    LlmExecutionIntercept => LlmExecutionIntercept,
    LlmStreamExecutionIntercept => LlmStreamExecutionIntercept,
});

dto!(nemo_relay_types::diagnostics::ResourceMemoryMetrics => ResourceMemoryMetrics {
    system_used: optional,
    system_total: optional,
    system_available: optional,
    resident: optional,
    private: optional,
    physical_footprint: optional,
    virtual_memory: optional,
    peak_resident: optional,
    limit: optional,
    environment_accounted: optional,
    some_pressure_stall_time: optional,
    full_pressure_stall_time: optional,
    out_of_memory_event_count: optional,
    limit_events: fixed_slots,
});
