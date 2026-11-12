// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use chrono::{DateTime, Utc};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value as Json;

use nemo_relay_adaptive::acg::{
    AgentIdentity, CacheRequestFacts, CacheTelemetryEvent, CacheTelemetryProvider,
};
use nemo_relay_adaptive::context_helpers::set_latency_sensitivity as adaptive_set_latency_sensitivity;
use nemo_relay_adaptive::{AdaptiveConfig, AdaptiveRuntime as CoreAdaptiveRuntime};
use nemo_relay_observability::plugin::default_otel_service_name;
use nemo_relay_plugin_host::dynamic::{
    PluginHostActivation as CorePluginHostActivation, PluginHostReport,
    initialize as initialize_impl, validate as validate_impl,
    validate_exact as validate_exact_impl,
};
use nemo_relay_plugin_host::resource_metrics as core_resource_metrics_api;
use nemo_relay_plugin_host::resource_metrics::{
    AcceleratorDeviceMetrics, AcceleratorProcessMetrics, AcceleratorVendor, BandwidthUnit,
    CapacityUnit, CountUnit, CpuMetrics, CpuUnit, DataUnit, DiskMetrics, DurationUnit,
    FilesystemCapacityMetrics, GpuMetrics, MemoryMetrics, NetworkInterfaceMetrics, NetworkMetrics,
    NetworkTrafficMetrics, ProcessMetrics, ProcessSamplingMetadata as CoreProcessSamplingMetadata,
    ResourceLimitEventCount, ResourceLimitEventKind, ResourceLimitResource, ResourceMeasurement,
    ResourceMeasurementScope, ResourceMetricValue,
    ResourceMetricsSnapshot as CoreResourceMetricsSnapshot, ResourceOperatingSystem, ResourceUnit,
    UtilizationUnit,
};
use nemo_relay_plugin_host::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginConfig, PluginError, PluginRegistration,
    PluginRegistrationContext, deregister_plugin as deregister_plugin_impl,
    list_plugin_kinds as list_plugin_kinds_impl, register_plugin as register_plugin_impl,
    rollback_registrations,
};
use nemo_relay_runtime::api::llm as core_llm_api;
use nemo_relay_runtime::api::llm::{LlmAttributes, LlmRequest};
use nemo_relay_runtime::api::registry as core_registry_api;
use nemo_relay_runtime::api::registry::{
    RuntimeRegistrationKind as CoreRuntimeRegistrationKind,
    RuntimeRegistrationOwnerKind as CoreRuntimeRegistrationOwnerKind,
};
use nemo_relay_runtime::api::runtime::subscriber_dispatcher::with_nested_publication_buffer;
use nemo_relay_runtime::api::runtime::{
    ConditionalMiddlewareGuardrailFn, LlmCollector, LlmExecutionNextFn, LlmFinalizer,
    LlmJsonStream, MiddlewareContinuationContext, PropagationContext as CorePropagationContext,
    TASK_SCOPE_STACK, ToolExecutionNextFn,
    capture_propagation_context as capture_propagation_context_handle,
    capture_propagation_context_with_root as capture_propagation_context_with_root_handle,
    capture_rootless_propagation_context as capture_rootless_propagation_context_handle,
    capture_traceparent as capture_traceparent_handle,
    create_scope_stack_from_propagation as create_scope_stack_from_propagation_handle,
    current_scope_stack as current_scope_stack_handle, scope_stack_active as scope_stack_is_active,
    set_thread_scope_stack as bind_thread_scope_stack, task_scope_top,
    with_scope_stack as with_scope_stack_handle,
};
use nemo_relay_runtime::api::scope as core_scope_api;
use nemo_relay_runtime::api::scope::ScopeAttributes;
use nemo_relay_runtime::api::subscriber as core_subscriber_api;
use nemo_relay_runtime::api::tool as core_tool_api;
use nemo_relay_runtime::api::tool::ToolAttributes;
use nemo_relay_runtime::shared_runtime::initialize_shared_runtime_binding;
use nemo_relay_types::codec::request::AnnotatedLlmRequest;
use nemo_relay_types::codec::response::Usage;
use nemo_relay_types::error::{FlowError, Result as FlowResult};

use crate::callable;
use crate::callback_factory::{self, CallbackScope, Wrapper};
use crate::convert::{
    BoxFuture, Callback, JAVASCRIPT_MAX_SAFE_INTEGER, JsArg, JsArgs2,
    clear_last_callback_error as clear_recorded_callback_error, closure_function, detach,
    get_last_callback_error as get_recorded_callback_error, js_value, lock, opt_json, parse_json,
    parse_timestamp_micros, record_callback_error, settled,
};
use crate::environment::spawn_promise;
use crate::error::{Error, Result};
use crate::promise_call::{Invocation, PromiseAwareFn, PublicationContext};
use crate::stream::LlmStream;
use crate::types::{
    DataSchema, LlmHandle, LogSeverity, MetricMeasurement, MetricTemporality, ScopeHandle,
    ScopeStack, ScopeType, ToolExecutionResult, ToolHandle, mirror_fields, mirror_variants,
};

#[cfg(test)]
mod llm_stack_tests;

/// Choose which middleware stages run for one call. Omitted fields run their stages.
/// Relay still sanitizes event data, uses codecs, and handles call completion.
/// Nested calls use their own options.
#[napi(object)]
pub struct ExecutionOptions {
    pub conditional_execution: Option<bool>,
    pub request_intercepts: Option<bool>,
    pub execution_intercepts: Option<bool>,
}

impl From<ExecutionOptions> for nemo_relay_runtime::api::execution::ExecutionOptions {
    fn from(options: ExecutionOptions) -> Self {
        Self {
            conditional_execution: options.conditional_execution.unwrap_or(true),
            request_intercepts: options.request_intercepts.unwrap_or(true),
            execution_intercepts: options.execution_intercepts.unwrap_or(true),
        }
    }
}

pub(crate) const LLM_STREAM_BRIDGE_CAPACITY: usize = 32;

static NODE_ENVIRONMENT_COUNT: AtomicUsize = AtomicUsize::new(0);
static NODE_ENVIRONMENT_LIFECYCLE_LOCK: StdMutex<()> = StdMutex::new(());

fn register_node_environment() -> FlowResult<()> {
    let _guard = lock(&NODE_ENVIRONMENT_LIFECYCLE_LOCK);
    nemo_relay_logging::initialize_default_logging()?;
    NODE_ENVIRONMENT_COUNT.fetch_add(1, Ordering::AcqRel);
    Ok(())
}

fn cleanup_node_environment() {
    let _guard = lock(&NODE_ENVIRONMENT_LIFECYCLE_LOCK);
    if NODE_ENVIRONMENT_COUNT.fetch_sub(1, Ordering::AcqRel) == 1
        && let Err(error) = nemo_relay_logging::shutdown_default_logging()
    {
        eprintln!("nemo-relay: operational logging shutdown failed: {error}");
    }
}

/// Emit a structured operational log record through Relay's configured sinks.
#[napi(
    ts_generic_types = "TFields extends JsonObjectCompatible<TFields> = JsonObject",
    catch_unwind
)]
pub fn log(
    level: String,
    message: String,
    target: Option<String>,
    #[napi(ts_arg_type = "TFields | null")] fields: Option<Json>,
) -> Result<()> {
    let fields = match fields {
        None => serde_json::Map::new(),
        Some(Json::Object(fields)) => fields,
        Some(_) => return Err(Error::invalid_argument("log fields must be an object")),
    };
    let target = target.unwrap_or_default();
    let target = if target.is_empty() {
        "nemo_relay.node".into()
    } else {
        format!("nemo_relay.node.{target}")
    };
    nemo_relay_logging::emit_str(&level, &target, &message, fields).map_err(Error::from)
}

macro_rules! log_levels {
    ($($level:ident: $doc:literal,)*) => {$(
        #[doc = $doc]
        #[napi(
            ts_generic_types = "TFields extends JsonObjectCompatible<TFields> = JsonObject",
            catch_unwind
        )]
        pub fn $level(
            message: String,
            target: Option<String>,
            #[napi(ts_arg_type = "TFields | null")] fields: Option<Json>,
        ) -> Result<()> {
            log(stringify!($level).into(), message, target, fields)
        }
    )*};
}

log_levels! {
    trace: " Emit a trace-level operational log record through Relay's configured sinks.",
    debug: " Emit a debug-level operational log record through Relay's configured sinks.",
    info: " Emit an info-level operational log record through Relay's configured sinks.",
    warn: " Emit a warn-level operational log record through Relay's configured sinks.",
    error: " Emit an error-level operational log record through Relay's configured sinks.",
}

/// Acquire a fresh resource metrics snapshot without requiring polling.
/// Collection runs outside the JavaScript event loop.
#[napi(ts_return_type = "Promise<ResourceMetricsSnapshot>", catch_unwind)]
pub fn collect_resource_metrics(env: &Env) -> Result<PromiseRaw<'_, Unknown<'_>>> {
    spawn_promise(
        env,
        async {
            core_resource_metrics_api::collect()
                .await
                .map_err(Error::from)
        },
        |env, snapshot| {
            js_value(
                env,
                env.to_js_value(&NodeResourceMetricsSnapshot::node_from(snapshot))?,
            )
        },
    )
}

trait NodeFrom<T> {
    fn node_from(value: T) -> Self;
}

impl<T: NodeFrom<U>, U> NodeFrom<Option<U>> for Option<T> {
    fn node_from(value: Option<U>) -> Self {
        value.map(T::node_from)
    }
}

impl<T: NodeFrom<U>, U> NodeFrom<Vec<U>> for Vec<T> {
    fn node_from(value: Vec<U>) -> Self {
        value.into_iter().map(T::node_from).collect()
    }
}

impl<T: NodeFrom<U>, U> NodeFrom<BTreeMap<String, U>> for BTreeMap<String, T> {
    fn node_from(value: BTreeMap<String, U>) -> Self {
        value
            .into_iter()
            .map(|(key, value)| (key, T::node_from(value)))
            .collect()
    }
}

macro_rules! node_identity {
    ($($ty:ty),* $(,)?) => {$(
        impl NodeFrom<$ty> for $ty {
            fn node_from(value: $ty) -> Self {
                value
            }
        }
    )*};
}

node_identity!(
    String,
    u32,
    DateTime<Utc>,
    AcceleratorVendor,
    ResourceLimitResource,
    ResourceLimitEventKind,
    ResourceMeasurementScope,
    ResourceOperatingSystem,
);

impl<U: ResourceUnit> NodeFrom<ResourceMeasurement<f64, U>> for ResourceMeasurement<f64, U> {
    fn node_from(value: ResourceMeasurement<f64, U>) -> Self {
        value
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum NodeResourceMetricInteger {
    Number(f64),
    BigInt(u64),
}

impl NodeFrom<u64> for NodeResourceMetricInteger {
    fn node_from(value: u64) -> Self {
        if value <= JAVASCRIPT_MAX_SAFE_INTEGER {
            Self::Number(value as f64)
        } else {
            Self::BigInt(value)
        }
    }
}

#[cfg(test)]
#[path = "../../tests/rust/api/resource_metric_integer_tests.rs"]
mod node_resource_metric_integer_tests;

impl Serialize for NodeResourceMetricInteger {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Number(value) => serializer.serialize_f64(*value),
            Self::BigInt(value) => serializer.serialize_u64(*value),
        }
    }
}

#[derive(Serialize)]
struct NodeIntegerResourceMeasurement<U> {
    value: NodeResourceMetricInteger,
    unit: U,
}

impl<U: ResourceUnit> NodeFrom<ResourceMeasurement<ResourceMetricValue, U>>
    for NodeIntegerResourceMeasurement<U>
{
    fn node_from(measurement: ResourceMeasurement<ResourceMetricValue, U>) -> Self {
        Self {
            value: match measurement.value {
                ResourceMetricValue::Integer(value) => NodeResourceMetricInteger::node_from(value),
                ResourceMetricValue::Decimal(value) => NodeResourceMetricInteger::Number(value),
            },
            unit: measurement.unit,
        }
    }
}

macro_rules! node_metrics {
    ($($name:ident: $core:ty { $($field:ident: $ty:ty),* $(,)? })*) => {$(
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct $name {
            $($field: $ty,)*
        }

        impl NodeFrom<$core> for $name {
            fn node_from(value: $core) -> Self {
                Self {
                    $($field: NodeFrom::node_from(value.$field),)*
                }
            }
        }
    )*};
}

node_metrics! {
    NodeResourceLimitEventCount: ResourceLimitEventCount {
        resource: ResourceLimitResource,
        event: ResourceLimitEventKind,
        count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    }
    NodeAcceleratorDeviceMetrics: AcceleratorDeviceMetrics {
        vendor: AcceleratorVendor,
        device_identifier: String,
        device_index: Option<u32>,
        memory_used: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        compute_utilization: Option<ResourceMeasurement<f64, UtilizationUnit>>,
    }
    NodeAcceleratorProcessMetrics: AcceleratorProcessMetrics {
        vendor: AcceleratorVendor,
        device_identifier: String,
        device_index: Option<u32>,
        process_id: u32,
        memory_used: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        compute_utilization: Option<ResourceMeasurement<f64, UtilizationUnit>>,
    }
    NodeFilesystemCapacityMetrics: FilesystemCapacityMetrics {
        path: String,
        total_capacity: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        available_capacity: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        free_capacity: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
    }
    NodeCpuMetrics: CpuMetrics {
        user_time: Option<NodeIntegerResourceMeasurement<DurationUnit>>,
        system_time: Option<NodeIntegerResourceMeasurement<DurationUnit>>,
        total_time: Option<NodeIntegerResourceMeasurement<DurationUnit>>,
        consumption_rate: Option<ResourceMeasurement<f64, CpuUnit>>,
        throttled_time: Option<NodeIntegerResourceMeasurement<DurationUnit>>,
        effective_limit: Option<ResourceMeasurement<f64, CpuUnit>>,
        some_pressure_stall_time: Option<NodeIntegerResourceMeasurement<DurationUnit>>,
        full_pressure_stall_time: Option<NodeIntegerResourceMeasurement<DurationUnit>>,
        limit_events: Vec<NodeResourceLimitEventCount>,
    }
    NodeMemoryMetrics: MemoryMetrics {
        system_used: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        system_total: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        system_available: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        resident: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        private: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        physical_footprint: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        virtual_memory: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        peak_resident: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        limit: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        environment_accounted: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
        some_pressure_stall_time: Option<NodeIntegerResourceMeasurement<DurationUnit>>,
        full_pressure_stall_time: Option<NodeIntegerResourceMeasurement<DurationUnit>>,
        out_of_memory_event_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        limit_events: Vec<NodeResourceLimitEventCount>,
    }
    NodeProcessMetrics: ProcessMetrics {
        active_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        descendant_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        thread_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        lifetime_creation_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        open_file_descriptor_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        windows_handle_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        limit_events: Vec<NodeResourceLimitEventCount>,
    }
    NodeDiskMetrics: DiskMetrics {
        read_data: Option<NodeIntegerResourceMeasurement<DataUnit>>,
        write_data: Option<NodeIntegerResourceMeasurement<DataUnit>>,
        read_throughput: Option<ResourceMeasurement<f64, BandwidthUnit>>,
        write_throughput: Option<ResourceMeasurement<f64, BandwidthUnit>>,
        read_operations: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        write_operations: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        filesystems: Vec<NodeFilesystemCapacityMetrics>,
    }
    NodeGpuMetrics: GpuMetrics {
        device_metrics: Option<Vec<NodeAcceleratorDeviceMetrics>>,
        process_metrics: Option<Vec<NodeAcceleratorProcessMetrics>>,
    }
    NodeNetworkTrafficMetrics: NetworkTrafficMetrics {
        received_data: Option<NodeIntegerResourceMeasurement<DataUnit>>,
        transmitted_data: Option<NodeIntegerResourceMeasurement<DataUnit>>,
        receive_throughput: Option<ResourceMeasurement<f64, BandwidthUnit>>,
        transmit_throughput: Option<ResourceMeasurement<f64, BandwidthUnit>>,
        received_packets: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        transmitted_packets: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        receive_errors: Option<NodeIntegerResourceMeasurement<CountUnit>>,
        transmit_errors: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    }
    NodeNetworkInterfaceMetrics: NetworkInterfaceMetrics {
        name: String,
        traffic: NodeNetworkTrafficMetrics,
    }
    NodeNetworkMetrics: NetworkMetrics {
        measurement_scope: ResourceMeasurementScope,
        system: NodeNetworkTrafficMetrics,
        interfaces: Vec<NodeNetworkInterfaceMetrics>,
    }
    NodeProcessSamplingMetadata: CoreProcessSamplingMetadata {
        visible_processes: NodeResourceMetricInteger,
        sampled_processes: NodeResourceMetricInteger,
        field_sampled_processes: BTreeMap<String, NodeResourceMetricInteger>,
    }
    NodeResourceMetricsSnapshot: CoreResourceMetricsSnapshot {
        timestamp: DateTime<Utc>,
        operating_system: ResourceOperatingSystem,
        measurement_scope: ResourceMeasurementScope,
        process_sampling: Option<NodeProcessSamplingMetadata>,
        cpu: Option<NodeCpuMetrics>,
        memory: Option<NodeMemoryMetrics>,
        process: Option<NodeProcessMetrics>,
        disk: Option<NodeDiskMetrics>,
        gpu: Option<NodeGpuMetrics>,
        network: Option<NodeNetworkMetrics>,
    }
}

fn effective_scope(env: &Env) -> Result<CallbackScope> {
    Ok(
        callback_factory::callback_scope(env)?.unwrap_or_else(|| CallbackScope {
            scope_stack: current_scope_stack_handle(),
            publication_buffer: None,
            propagation: None,
            execution_context: None,
            publication_context_id: None,
            publication_context_captured: false,
        }),
    )
}

fn with_callback_scope<CallbackResult>(
    scope: &CallbackScope,
    callback: impl FnOnce() -> CallbackResult,
) -> CallbackResult {
    match &scope.execution_context {
        Some(context) => context.run_sync(callback),
        None => with_scope_stack_handle(scope.scope_stack.clone(), || {
            with_nested_publication_buffer(scope.publication_buffer.clone(), callback)
        }),
    }
}

fn with_effective_scope_stack<T>(env: &Env, callback: impl FnOnce() -> T) -> Result<T> {
    Ok(with_callback_scope(&effective_scope(env)?, callback))
}

fn parse_uuid(value: &str, what: &str) -> Result<uuid::Uuid> {
    uuid::Uuid::parse_str(value)
        .map_err(|error| Error::invalid_argument(format!("invalid {what}: {error}")))
}

struct CallContext {
    publication_context_id: Option<String>,
    scope: CallbackScope,
}

impl CallContext {
    fn capture(env: &Env) -> Result<Self> {
        let scope = effective_scope(env)?;
        let publication_context_id = if scope.publication_context_captured {
            scope.publication_context_id.clone()
        } else {
            scope
                .publication_context_id
                .clone()
                .or(callback_factory::event_sanitizer_callback_context_id(env)?)
        };
        Ok(Self {
            publication_context_id,
            scope,
        })
    }

    fn parent(&self, handle: Option<&ScopeHandle>) -> core_scope_api::ScopeHandle {
        match handle {
            Some(handle) => handle.inner.clone(),
            None => with_scope_stack_handle(self.scope.scope_stack.clone(), task_scope_top),
        }
    }

    fn enter<T>(&self, callback: impl FnOnce() -> T) -> T {
        with_callback_scope(&self.scope, callback)
    }

    fn publication(&self) -> PublicationContext {
        PublicationContext {
            context_id: self.publication_context_id.clone(),
            publication_buffer: self.scope.publication_buffer.clone(),
        }
    }

    fn run<F>(self, future: F) -> impl Future<Output = F::Output> + Send + 'static
    where
        F: Future + Send + 'static,
    {
        // Keep the pipeline future out of the context wrappers' inline state.
        let future = Box::pin(future);
        let publication = self.publication();
        async move {
            match self.scope.execution_context {
                Some(context) => publication.run(context.run(future)).await,
                None => {
                    publication
                        .run(TASK_SCOPE_STACK.scope(self.scope.scope_stack, future))
                        .await
                }
            }
        }
    }
}

#[napi_derive::module_init]
fn init() {
    nemo_relay_builtins::initialize_runtime();
    initialize_shared_runtime_binding("node")
        .expect("node runtime ownership initialization should succeed");
    nemo_relay_builtins::plugin::ensure_builtin_plugins_registered()
        .expect("node built-in plugin registration should succeed");
}

#[napi(module_exports)]
fn initialize_node_environment(env: Env) -> napi::Result<()> {
    register_node_environment().map_err(|error| napi::Error::from_reason(error.to_string()))?;
    if let Err(error) = env.add_env_cleanup_hook((), |_| cleanup_node_environment()) {
        cleanup_node_environment();
        return Err(error);
    }
    crate::environment::mark_exiting_on_process_exit(&env)?;
    Ok(())
}

fn parse_string_map(value: Option<Json>, field_name: &str) -> Result<HashMap<String, String>> {
    let Some(value) = value else {
        return Ok(HashMap::new());
    };
    let Json::Object(map) = value else {
        return Err(Error::invalid_argument(format!(
            "{field_name} must be an object of string values",
        )));
    };
    let mut out = HashMap::with_capacity(map.len());
    for (key, value) in map {
        let Json::String(value) = value else {
            return Err(Error::invalid_argument(format!(
                "{field_name} must be an object of string values",
            )));
        };
        out.insert(key, value);
    }
    Ok(out)
}

fn otel_status_metadata(status_code: &'static str, status_message: Option<String>) -> Json {
    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "otel.status_code".to_string(),
        Json::String(status_code.to_string()),
    );
    if let Some(status_message) = status_message {
        metadata.insert(
            "otel.status_description".to_string(),
            Json::String(status_message),
        );
    }
    Json::Object(metadata)
}

fn parse_otel_type(value: &str) -> Result<nemo_relay_observability::OpenTelemetryType> {
    match value {
        "full" => Ok(nemo_relay_observability::OpenTelemetryType::Full),
        "gen_ai" => Ok(nemo_relay_observability::OpenTelemetryType::GenAi),
        "openinference" => Ok(nemo_relay_observability::OpenTelemetryType::OpenInference),
        other => Err(Error::invalid_argument(format!(
            "type must be 'full', 'gen_ai', or 'openinference', got {other:?}",
        ))),
    }
}

fn parse_otel_transport(value: &str) -> Result<nemo_relay_observability::otel::OtlpTransport> {
    match value {
        "http_binary" => Ok(nemo_relay_observability::otel::OtlpTransport::HttpBinary),
        "grpc" => Ok(nemo_relay_observability::otel::OtlpTransport::Grpc),
        other => Err(Error::invalid_argument(format!(
            "transport must be 'http_binary' or 'grpc', got {other:?}",
        ))),
    }
}

fn parse_mark_projection(value: String) -> Result<nemo_relay_observability::MarkProjection> {
    serde_json::from_value(Json::String(value))
        .map_err(|error| Error::invalid_argument(error.to_string()))
}

fn parse_attribute_mappings(
    value: Json,
) -> Result<Vec<nemo_relay_observability::OtlpAttributeMapping>> {
    let mappings: Vec<_> = serde_json::from_value(value)
        .map_err(|error| Error::invalid_argument(format!("attributeMappings: {error}")))?;
    nemo_relay_observability::validate_attribute_mappings(&mappings)
        .map_err(Error::invalid_argument)?;
    Ok(mappings)
}

fn parse_otel_file_sink(
    output_directory: &str,
    filename: Option<&str>,
    format: Option<&str>,
    mode: Option<&str>,
) -> Result<nemo_relay_observability::otel::OtlpFileSinkSettings> {
    nemo_relay_observability::otel::OtlpFileSinkSettings::from_parts(
        output_directory,
        filename,
        format,
        mode,
    )
    .map_err(Error::invalid_argument)
}

fn nonblank(value: &str, what: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() {
        return Err(Error::invalid_argument(format!(
            "{what} must be a nonblank string"
        )));
    }
    Ok(value.to_owned())
}

fn with_string_map<C>(
    mut config: C,
    value: Option<Json>,
    field: &str,
    apply: fn(C, String, String) -> C,
) -> Result<C> {
    for (key, value) in parse_string_map(value, field)? {
        config = apply(config, key, value);
    }
    Ok(config)
}

fn completed_span_context_ttl(value: BigInt) -> Result<Duration> {
    let (negative, value, lossless) = value.get_u64();
    if negative || !lossless {
        return Err(Error::invalid_argument(
            "completedSpanContextTtlMillis must be a nonnegative u64 BigInt",
        ));
    }
    if value == 0 {
        return Err(Error::invalid_argument(
            "completedSpanContextTtlMillis must be greater than 0",
        ));
    }
    Ok(Duration::from_millis(value))
}

macro_rules! option_setters {
    ($config:ident, $options:ident; $($field:ident($value:ident) => $setter:ident($argument:expr);)*) => {$(
        if let Some($value) = $options.$field {
            $config = $config.$setter($argument);
        }
    )*};
}

macro_rules! endpoint_options {
    ($config:ident, $options:ident) => {
        option_setters!($config, $options;
            transport(transport) => with_transport(parse_otel_transport(&transport)?);
            timeout_millis(timeout) => with_timeout(Duration::from_millis(timeout.into()));
            tls(tls) => with_tls(tls.into());
            allow_insecure(allow_insecure) => with_allow_insecure(allow_insecure);
        );
        $config = with_string_map(
            $config,
            $options.headers,
            "headers",
            |config, key, value| config.with_header(key, value),
        )?;
        $config = with_string_map(
            $config,
            $options.header_env,
            "headerEnv",
            |config, key, variable| config.with_header_env(key, variable),
        )?;
    };
}

macro_rules! service_options {
    ($config:ident, $options:ident) => {
        option_setters!($config, $options;
            service_namespace(namespace) => with_service_namespace(namespace);
            service_version(version) => with_service_version(version);
            instrumentation_scope(scope) => with_instrumentation_scope(scope);
        );
        $config = with_string_map(
            $config,
            $options.resource_attributes,
            "resourceAttributes",
            |config, key, value| config.with_resource_attribute(key, value),
        )?;
    };
}

macro_rules! trace_options {
    ($config:ident, $options:ident) => {
        option_setters!($config, $options;
            completed_span_context_ttl_millis(ttl) => with_completed_span_context_ttl(completed_span_context_ttl(ttl)?);
            mark_projection(projection) => with_mark_projection(parse_mark_projection(projection)?);
            mark_exclude_names(names) => with_mark_exclude_names(names);
            attribute_mappings(mappings) => with_attribute_mappings(parse_attribute_mappings(mappings)?);
            promote_metadata_prefixes(prefixes) => with_promote_metadata_prefixes(prefixes);
            promote_resource_metadata_prefixes(prefixes) => with_promote_resource_metadata_prefixes(prefixes);
        );
    };
}

fn build_otel_config(
    options: OpenTelemetryConfig,
) -> Result<nemo_relay_observability::otel::OpenTelemetryConfig> {
    let mut config = nemo_relay_observability::otel::OpenTelemetryConfig::new(
        parse_otel_type(&options.r#type)?,
        nonblank(&options.endpoint, "endpoint")?,
    )
    .with_service_name(
        options
            .service_name
            .unwrap_or_else(default_otel_service_name),
    );
    endpoint_options!(config, options);
    service_options!(config, options);
    trace_options!(config, options);
    Ok(config)
}

fn build_otel_log_config(
    options: OpenTelemetryLogConfig,
) -> Result<nemo_relay_observability::otel_logs::OpenTelemetryLogConfig> {
    let mut config = nemo_relay_observability::otel_logs::OpenTelemetryLogConfig::new(nonblank(
        &options.endpoint,
        "endpoint",
    )?)
    .with_service_name(
        options
            .service_name
            .unwrap_or_else(default_otel_service_name),
    );
    option_setters!(config, options;
        max_queue_size(size) => with_max_queue_size(size as usize);
        max_export_batch_size(size) => with_max_export_batch_size(size as usize);
        scheduled_delay_millis(delay) => with_scheduled_delay(Duration::from_millis(delay.into()));
        completed_span_context_ttl_millis(ttl) => with_completed_span_context_ttl(completed_span_context_ttl(ttl)?);
        minimum_severity(severity) => with_minimum_severity(severity.into());
    );
    endpoint_options!(config, options);
    service_options!(config, options);
    Ok(config)
}

fn build_otel_metric_config(
    options: OpenTelemetryMetricConfig,
) -> Result<nemo_relay_observability::otel_metrics::OpenTelemetryMetricConfig> {
    let mut config = nemo_relay_observability::otel_metrics::OpenTelemetryMetricConfig::new(
        nonblank(&options.endpoint, "endpoint")?,
    )
    .with_service_name(
        options
            .service_name
            .unwrap_or_else(default_otel_service_name),
    );
    option_setters!(config, options;
        export_interval_millis(interval) => with_export_interval(Duration::from_millis(interval.into()));
        max_instruments(limit) => with_max_instruments(limit as usize);
        cardinality_limit(limit) => with_cardinality_limit(limit as usize);
        temporality(temporality) => with_temporality(temporality.into());
    );
    endpoint_options!(config, options);
    service_options!(config, options);
    Ok(config)
}

fn exporter_runtime_diagnostics_json(
    diagnostics: nemo_relay_observability::ExporterRuntimeDiagnostics,
) -> Json {
    Json::Array(
        diagnostics
            .entries()
            .iter()
            .map(|diagnostic| {
                serde_json::json!({
                    "code": diagnostic.code,
                    "message": diagnostic.message,
                    "count": diagnostic.count,
                })
            })
            .collect(),
    )
}

fn build_atof_config(
    options: Option<AtofExporterConfig>,
) -> Result<nemo_relay_observability::atof::AtofExporterConfig> {
    let options = options.unwrap_or_default();
    match options.r#type.as_deref().unwrap_or("file") {
        "file" => {
            let mut config = nemo_relay_observability::atof::AtofExporterConfig::new();
            option_setters!(config, options;
                output_directory(directory) => with_output_directory(PathBuf::from(directory));
                filename(filename) => with_filename(filename);
                mode(mode) => with_mode(
                    nemo_relay_observability::atof::AtofExporterMode::parse(&mode).ok_or_else(|| {
                        Error::invalid_argument("mode must be 'append' or 'overwrite'")
                    })?
                );
            );
            Ok(config)
        }
        "stream" => {
            let url = options
                .url
                .ok_or_else(|| Error::invalid_argument("stream sink requires url"))?;
            let transport = nemo_relay_observability::atof::AtofEndpointTransport::parse(
                options.transport.as_deref().unwrap_or("http_post"),
            )
            .ok_or_else(|| {
                Error::invalid_argument(
                    "stream transport must be 'http_post', 'websocket', or 'ndjson'",
                )
            })?;
            let mut sink =
                nemo_relay_observability::atof::AtofStreamSinkConfig::new(url, transport);
            option_setters!(sink, options;
                timeout_millis(timeout_millis) => with_timeout_millis(timeout_millis.into());
                allow_insecure(allow_insecure) => with_allow_insecure(allow_insecure);
                field_name_policy(policy) => with_field_name_policy(
                    nemo_relay_observability::atof::AtofEndpointFieldNamePolicy::parse(&policy)
                        .ok_or_else(|| {
                            Error::invalid_argument(
                                "stream field_name_policy must be 'preserve' or 'replace_dots'",
                            )
                        })?
                );
            );
            sink = with_string_map(sink, options.headers, "headers", |sink, key, value| {
                sink.with_header(key, value)
            })?;
            sink = with_string_map(
                sink,
                options.header_env,
                "headerEnv",
                |sink, key, variable| sink.with_header_env(key, variable),
            )?;
            Ok(nemo_relay_observability::atof::AtofExporterConfig::new().with_stream_sink(sink))
        }
        _ => Err(Error::invalid_argument(
            "ATOF sink type must be 'file' or 'stream'",
        )),
    }
}

pub(crate) fn llm_stream_from_rust_stream(
    rust_stream: LlmJsonStream,
    publication: PublicationContext,
) -> LlmStream {
    LlmStream::new(
        rust_stream,
        Default::default(),
        Default::default(),
        publication,
    )
}

struct ScopedJsonCall {
    value: Json,
    scope: CallbackScope,
}

impl callable::JsCall for ScopedJsonCall {
    type Args = JsArgs2;

    fn args(self, env: &Env) -> napi::Result<JsArgs2> {
        Ok(FnArgs::from((
            js_value(env, self.value)?,
            self.scope.into_js(env)?,
        )))
    }
}

fn serialize_llm_request(request: &LlmRequest) -> FlowResult<Json> {
    serde_json::to_value(request)
        .map_err(|error| FlowError::Internal(format!("failed to serialize LlmRequest: {error}")))
}

type ScopedFn<A, R> = Arc<dyn Fn(A) -> BoxFuture<FlowResult<R>> + Send + Sync>;

fn scoped_fn<A: Send + 'static, R: Send + 'static>(
    env: &Env,
    func: &Function<JsArg, JsArg>,
    error_prefix: &'static str,
    encode: fn(A) -> FlowResult<Json>,
    decode: fn(Json) -> FlowResult<R>,
) -> napi::Result<ScopedFn<A, R>> {
    let func: Function<JsArg, JsArg> = callback_factory::wrap(env, Wrapper::Execution, func)?;
    let scoped: Function<JsArgs2, JsArg> = callback_factory::wrap(env, Wrapper::Scoped, &func)?;
    let func: Arc<callable::AsyncReplyTsfn<ScopedJsonCall>> =
        Arc::new(callable::reply_tsfn(env, &scoped, error_prefix)?);
    Ok(Arc::new(move |value| {
        let func = Arc::clone(&func);
        let scope = CallbackScope::capture();
        Box::pin(async move {
            let call = ScopedJsonCall {
                value: encode(value)?,
                scope,
            };
            decode(callable::call_tsfn(&func, call, error_prefix).await?)
        })
    }))
}

type PluginRegistrations = Arc<StdMutex<Vec<PluginRegistration>>>;

struct PluginContext<'env> {
    env: &'env Env,
    namespace_prefix: String,
    registrations: PluginRegistrations,
    object: Object<'env>,
}

impl PluginContext<'_> {
    fn register_method(
        &mut self,
        property: &str,
        label: &'static str,
        deregister: fn(&str) -> FlowResult<bool>,
        register: impl 'static + Fn(&FunctionCallContext, &str) -> Result<()>,
    ) -> napi::Result<()> {
        let namespace_prefix = self.namespace_prefix.clone();
        let registrations = Arc::clone(&self.registrations);
        let function = closure_function(self.env, property, move |ctx| {
            let name = format!("{namespace_prefix}{}", ctx.get::<String>(0)?);
            register(&ctx, &name)?;
            let registered_name = name.clone();
            lock(&registrations).push(PluginRegistration::new(
                "plugin",
                name,
                Box::new(move || {
                    deregister(&registered_name).map(|_| ()).map_err(|error| {
                        PluginError::RegistrationFailed(format!(
                            "{label} deregistration failed: {error}"
                        ))
                    })
                }),
            ));
            Ok(())
        })?;
        self.object.set_named_property(property, function)
    }

    fn method<F: 'static>(
        &mut self,
        property: &str,
        label: &'static str,
        make: fn(&Env, &Function<JsArg, JsArg>) -> napi::Result<F>,
        register: fn(&str, i32, F) -> FlowResult<()>,
        deregister: fn(&str) -> FlowResult<bool>,
    ) -> napi::Result<()> {
        self.register_method(property, label, deregister, move |ctx, name| {
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<Callback>(2)?;
            register(name, priority, make(ctx.env, &callback)?).map_err(Error::from)
        })
    }

    fn chain_method<F: 'static>(
        &mut self,
        property: &str,
        label: &'static str,
        make: fn(&Env, &Function<JsArg, JsArg>) -> napi::Result<F>,
        register: fn(&str, i32, bool, F) -> FlowResult<()>,
        deregister: fn(&str) -> FlowResult<bool>,
    ) -> napi::Result<()> {
        self.register_method(property, label, deregister, move |ctx, name| {
            let priority = ctx.get::<i32>(1)?;
            let break_chain = ctx.get::<bool>(2)?;
            let callback = ctx.get::<Callback>(3)?;
            register(name, priority, break_chain, make(ctx.env, &callback)?).map_err(Error::from)
        })
    }

    fn subscriber_method(&mut self) -> napi::Result<()> {
        self.register_method(
            "registerSubscriber",
            "subscriber",
            core_subscriber_api::deregister_subscriber,
            |ctx, name| {
                let callback = ctx.get::<Callback>(1)?;
                core_subscriber_api::register_subscriber(
                    name,
                    callable::wrap_js_event_subscriber(ctx.env, name.to_owned(), &callback)?,
                )
                .map_err(Error::from)
            },
        )
    }

    fn conditional_gate_method(&mut self) -> napi::Result<()> {
        self.register_method(
            "registerConditionalMiddlewareGuardrail",
            "conditional middleware guardrail",
            core_registry_api::deregister_conditional_middleware_guardrail,
            |ctx, name| {
                let kinds = ctx
                    .get::<Vec<RuntimeRegistrationKind>>(1)?
                    .into_iter()
                    .map(Into::into)
                    .collect::<BTreeSet<_>>();
                let registration_name = ctx.get::<String>(2)?;
                let guardrail = ctx.get::<Callback>(3)?;
                core_registry_api::register_conditional_middleware_guardrail(
                    name,
                    kinds,
                    &registration_name,
                    node_conditional_middleware_guardrail(ctx.env, &guardrail)?,
                )
                .map_err(Error::from)
            },
        )
    }
}

fn build_plugin_context<'env>(
    env: &'env Env,
    namespace_prefix: String,
    registrations: PluginRegistrations,
) -> napi::Result<Object<'env>> {
    let mut context = PluginContext {
        env,
        namespace_prefix,
        registrations,
        object: Object::new(env)?,
    };
    context.conditional_gate_method()?;
    context.subscriber_method()?;
    install_plugin_methods(&mut context)?;
    Ok(context.object)
}

struct NodePluginRegisterCall {
    plugin_config: Json,
    namespace_prefix: String,
    registrations: PluginRegistrations,
}

impl callable::JsCall for NodePluginRegisterCall {
    type Args = JsArgs2;

    fn args(self, env: &Env) -> napi::Result<JsArgs2> {
        let plugin_config = js_value(env, self.plugin_config)?;
        let plugin_context = build_plugin_context(env, self.namespace_prefix, self.registrations)?;
        Ok(FnArgs::from((plugin_config, detach(env, &plugin_context))))
    }
}

const CONDITIONAL_GATE_PREFIX: &str = "conditional middleware guardrail";

const CONDITIONAL_GATE_CALLBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

struct ConditionalGateCall {
    kinds: Vec<String>,
    registration_name: String,
}

impl callable::JsCall for ConditionalGateCall {
    type Args = JsArgs2;

    fn args(self, env: &Env) -> napi::Result<JsArgs2> {
        Ok(FnArgs::from((
            js_value(env, self.kinds)?,
            js_value(env, self.registration_name)?,
        )))
    }
}

fn decode_conditional_gate_result(value: FlowResult<Json>) -> Result<Option<String>> {
    serde_json::from_value(value?).map_err(|_| {
        Error::invalid_argument("conditional middleware guardrail must return a string or null")
    })
}

fn node_conditional_middleware_guardrail(
    env: &Env,
    guardrail: &Function<JsArg, JsArg>,
) -> napi::Result<ConditionalMiddlewareGuardrailFn> {
    let callback = callable::SyncJsCallback::new(
        env,
        callback_factory::wrap::<JsArgs2, JsArg>(env, Wrapper::ConditionalGate, guardrail)?,
        CONDITIONAL_GATE_PREFIX,
        Some(CONDITIONAL_GATE_CALLBACK_TIMEOUT),
    )?;
    Ok(Arc::new(move |kinds, effective_name| {
        let call = ConditionalGateCall {
            kinds: kinds.iter().map(|kind| kind.as_str().to_string()).collect(),
            registration_name: effective_name.to_string(),
        };
        decode_conditional_gate_result(callback.call(call)).unwrap_or_else(|error| {
            record_callback_error(format!(
                "Node conditional middleware guardrail failed open: {error}"
            ));
            None
        })
    }))
}

fn node_llm_codec(
    env: &Env,
    decode: Option<&Function<JsArg, JsArg>>,
    encode: Option<&Function<JsArg, JsArg>>,
) -> Result<Option<Arc<dyn nemo_relay_types::codec::traits::LlmCodec>>> {
    let (decode, encode) = match (decode, encode) {
        (Some(decode), Some(encode)) => (decode, encode),
        (None, None) => return Ok(None),
        _ => {
            return Err(Error::invalid_argument(
                "codecDecode and codecEncode must be provided together",
            ));
        }
    };
    Ok(Some(Arc::new(callable::NapiCodec {
        decode: callable::js_codec_callback(env, decode, "JS codec decode callback failed")?,
        encode: callable::js_codec_callback(env, encode, "JS codec encode callback failed")?,
    })))
}

fn node_llm_response_codec(
    env: &Env,
    decode: Option<&Function<JsArg, JsArg>>,
) -> Result<Option<Arc<dyn nemo_relay_types::codec::traits::LlmResponseCodec>>> {
    let Some(decode) = decode else {
        return Ok(None);
    };
    let decode_response =
        callable::js_codec_callback(env, decode, "JS response codec decode callback failed")?;
    Ok(Some(Arc::new(callable::NapiResponseCodec {
        decode_response,
    })))
}

const PLUGIN_VALIDATE_PREFIX: &str = "JS plugin validate failed";

const PLUGIN_REGISTER_PREFIX: &str = "JS plugin register callback failed";

struct NodePlugin {
    plugin_kind: String,
    validate: Option<callable::SyncJsCallback<Json>>,
    register: callable::AsyncReplyTsfn<NodePluginRegisterCall>,
}

impl Plugin for NodePlugin {
    fn plugin_kind(&self) -> &str {
        &self.plugin_kind
    }

    fn validate(
        &self,
        plugin_config: &serde_json::Map<String, serde_json::Value>,
    ) -> Vec<ConfigDiagnostic> {
        let Some(validate) = &self.validate else {
            return vec![];
        };
        let failed = |message: String| {
            vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "plugin.validate_failed".into(),
                component: Some(self.plugin_kind.clone()),
                field: None,
                message,
            }]
        };
        match validate.call(Json::Object(plugin_config.clone())) {
            Ok(Json::Null) => vec![],
            Ok(value) => {
                serde_json::from_value::<Vec<ConfigDiagnostic>>(value).unwrap_or_else(|error| {
                    failed(format!(
                        "JS plugin validate returned invalid diagnostics: {error}"
                    ))
                })
            }
            Err(error) => failed(error.to_string()),
        }
    }

    fn register<'a>(
        &'a self,
        plugin_config: &serde_json::Map<String, serde_json::Value>,
        ctx: &'a mut dyn PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), PluginError>> + Send + 'a>> {
        let namespace_prefix = ctx.qualify_name("");
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            let registrations = Arc::new(StdMutex::new(Vec::<PluginRegistration>::new()));
            let payload = NodePluginRegisterCall {
                plugin_config: Json::Object(plugin_config),
                namespace_prefix,
                registrations: registrations.clone(),
            };
            if let Err(error) =
                callable::call_tsfn(&self.register, payload, PLUGIN_REGISTER_PREFIX).await
            {
                rollback_registrations(&mut lock(&registrations));
                return Err(PluginError::RegistrationFailed(error.to_string()));
            }

            let drained = std::mem::take(&mut *lock(&registrations));
            ctx.extend_registrations(drained);
            Ok(())
        })
    }
}

/// Transport-neutral Relay causal context for application-managed transport.
#[napi(object)]
pub struct PropagationContext {
    pub version: u32,
    pub root_uuid: Option<String>,
    pub parent_uuid: String,
    pub traceparent: Option<String>,
    pub tracestate: Option<String>,
}

impl TryFrom<PropagationContext> for CorePropagationContext {
    type Error = Error;

    fn try_from(context: PropagationContext) -> Result<Self> {
        let root_uuid = context
            .root_uuid
            .as_deref()
            .map(|value| parse_uuid(value, "root UUID"))
            .transpose()?;
        let parent_uuid = parse_uuid(&context.parent_uuid, "parent UUID")?;
        let version = u16::try_from(context.version)
            .map_err(|_| Error::invalid_argument("propagation context version is out of range"))?;
        let context = Self {
            version,
            root_uuid,
            parent_uuid,
            traceparent: context.traceparent,
            tracestate: context.tracestate,
        };
        context.validate()?;
        Ok(context)
    }
}

impl From<CorePropagationContext> for PropagationContext {
    fn from(context: CorePropagationContext) -> Self {
        Self {
            version: u32::from(context.version),
            root_uuid: context.root_uuid.map(|uuid| uuid.to_string()),
            parent_uuid: context.parent_uuid.to_string(),
            traceparent: context.traceparent,
            tracestate: context.tracestate,
        }
    }
}

fn callback_propagation_context(
    scope: &CallbackScope,
    mut context: CorePropagationContext,
) -> Result<CorePropagationContext> {
    let stack = scope
        .scope_stack
        .read()
        .map_err(|error| Error::generic_failure(error.to_string()))?;
    let has_propagated_parent = stack
        .scopes()
        .iter()
        .any(|scope| stack.is_propagated_parent(scope.uuid));
    drop(stack);
    if !has_propagated_parent {
        context.root_uuid = context
            .traceparent
            .as_deref()
            .map(ToOwned::to_owned)
            .or_else(|| with_callback_scope(scope, capture_traceparent_handle).ok())
            .and_then(|traceparent| {
                traceparent
                    .get(3..35)
                    .and_then(|value| uuid::Uuid::parse_str(value).ok())
            })
            .or(Some(context.parent_uuid));
    }
    Ok(context)
}

fn rootless_callback_propagation_context(
    scope: &CallbackScope,
    context: CorePropagationContext,
) -> Result<CorePropagationContext> {
    let mut context = callback_propagation_context(scope, context)?;
    context.root_uuid = None;
    let rootless = with_callback_scope(scope, capture_rootless_propagation_context_handle)
        .map_err(Error::from)?;
    if rootless.traceparent.is_none() {
        context.traceparent = None;
        context.tracestate = None;
    }
    Ok(context)
}

/// Creates a new isolated scope stack.
#[napi(catch_unwind)]
pub fn create_scope_stack() -> ScopeStack {
    ScopeStack::new()
}

fn captured<T>(
    env: &Env,
    from_callback: impl FnOnce(&CallbackScope, CorePropagationContext) -> Result<T>,
    from_stack: impl FnOnce() -> FlowResult<T>,
) -> Result<T> {
    let scope = effective_scope(env)?;
    match scope.propagation.clone() {
        Some(context) => from_callback(&scope, context),
        None => with_callback_scope(&scope, from_stack).map_err(Error::from),
    }
}

/// Capture the current Relay causal parent for application-managed transport.
#[napi(catch_unwind)]
pub fn capture_propagation_context(env: Env) -> Result<PropagationContext> {
    captured(
        &env,
        callback_propagation_context,
        capture_propagation_context_handle,
    )
    .map(Into::into)
}

/// Capture the current Relay causal parent without a propagation root.
#[napi(catch_unwind)]
pub fn capture_rootless_propagation_context(env: Env) -> Result<PropagationContext> {
    captured(
        &env,
        rootless_callback_propagation_context,
        capture_rootless_propagation_context_handle,
    )
    .map(Into::into)
}

/// Capture the current parent with an optional stable application session root.
#[napi(catch_unwind)]
pub fn capture_propagation_context_with_root(
    env: Env,
    root_uuid: Option<String>,
) -> Result<PropagationContext> {
    let root_uuid = root_uuid
        .as_deref()
        .map(|value| parse_uuid(value, "root UUID"))
        .transpose()?;
    captured(
        &env,
        |scope, context| {
            let mut context = rootless_callback_propagation_context(scope, context)?;
            context.root_uuid = root_uuid;
            Ok(context)
        },
        || capture_propagation_context_with_root_handle(root_uuid),
    )
    .map(Into::into)
}

/// Capture the current Relay context as a W3C `traceparent` value.
#[napi(catch_unwind)]
pub fn capture_traceparent(env: Env) -> Result<String> {
    captured(
        &env,
        |scope, context| {
            let context = callback_propagation_context(scope, context)?;
            match context.traceparent {
                Some(traceparent) => Ok(traceparent),
                None => context.to_traceparent().map_err(Error::from),
            }
        },
        capture_traceparent_handle,
    )
}
/// Serialize a Relay causal context to the JSON wire format.
#[napi(catch_unwind)]
pub fn propagation_context_to_json(context: PropagationContext) -> Result<String> {
    CorePropagationContext::try_from(context)?
        .to_json()
        .map_err(Error::from)
}

/// Convert a Relay propagation context to a W3C `traceparent` value.
#[napi(catch_unwind)]
pub fn propagation_context_to_traceparent(context: PropagationContext) -> Result<String> {
    CorePropagationContext::try_from(context)?
        .to_traceparent()
        .map_err(Error::from)
}

/// Deserialize and validate a Relay causal context from the JSON wire format.
#[napi(catch_unwind)]
pub fn propagation_context_from_json(value: String) -> Result<PropagationContext> {
    CorePropagationContext::from_json(&value)
        .map(Into::into)
        .map_err(Error::from)
}

/// Create an isolated scope stack seeded from a received propagation context.
#[napi(catch_unwind)]
pub fn create_scope_stack_from_propagation(context: PropagationContext) -> Result<ScopeStack> {
    create_scope_stack_from_propagation_handle(&CorePropagationContext::try_from(context)?)
        .map(ScopeStack::from)
        .map_err(Error::from)
}

/// Copy the active Relay scope stack without changing the caller's binding.
///
/// Preserves scope UUIDs/types, scope-local registrations, and managed event,
/// trace, and publication context. Use one fork per concurrent operation with
/// `withScopeStack`, and finish the operation before its inherited scopes close.
#[napi(catch_unwind)]
pub fn fork_scope_stack(env: Env) -> Result<ScopeStack> {
    if !scope_stack_active(env)? {
        return Err(Error::invalid_argument(
            "cannot fork scope stack without an active Relay context",
        ));
    }
    let mut scope = effective_scope(&env)?;
    let context = with_callback_scope(&scope, MiddlewareContinuationContext::capture)
        .isolated_with_scope_stack(&scope.scope_stack)
        .map_err(Error::from)?;
    scope.scope_stack = context.scope_stack();
    scope.execution_context = Some(context);
    scope.propagation = None;
    if !scope.publication_context_captured {
        scope.publication_context_id = scope
            .publication_context_id
            .clone()
            .or(callback_factory::event_sanitizer_callback_context_id(&env)?);
    }
    scope.publication_context_captured = true;
    Ok(ScopeStack {
        inner: scope.scope_stack.clone(),
        publication_buffer: scope.publication_buffer.clone(),
        captured_scope: Some(scope),
    })
}

/// Run a callback with an isolated scope stack installed.
///
/// The caller's stack is restored immediately after callback invocation. When
/// the callback returns a Promise, the requested stack remains active for that
/// Promise until it settles and is then expired for inherited detached work.
/// Use this helper instead of `setThreadScopeStack` to isolate concurrent async
/// branches.
#[napi(ts_generic_types = "T", ts_return_type = "T", catch_unwind)]
pub fn with_scope_stack<'env>(
    env: &'env Env,
    stack: &ScopeStack,
    #[napi(ts_arg_type = "() => T")] callback: Callback,
) -> Result<Unknown<'env>> {
    let scope = match &stack.captured_scope {
        Some(scope) => scope.clone(),
        None => CallbackScope {
            scope_stack: stack.inner.clone(),
            publication_buffer: effective_scope(env)?
                .publication_buffer
                .or_else(|| stack.publication_buffer.clone()),
            propagation: None,
            execution_context: None,
            publication_context_id: None,
            publication_context_captured: false,
        },
    };
    Ok(callback_factory::with_callback_scope(
        env, scope, &callback,
    )?)
}

/// Return owned scope identities, or null without an explicitly active binding.
#[napi(catch_unwind)]
pub fn current_scope_stack_snapshot(env: Env) -> Result<Option<crate::types::ScopeStackSnapshot>> {
    let snapshot = match callback_factory::callback_scope(&env)? {
        Some(scope) => {
            nemo_relay_runtime::api::runtime::scope_stack_snapshot(&scope.scope_stack).map(Some)
        }
        None => nemo_relay_runtime::api::runtime::current_scope_stack_snapshot(),
    };
    snapshot
        .map(|snapshot| snapshot.map(Into::into))
        .map_err(Error::from)
}

/// Returns the current execution context's scope stack handle.
#[napi(catch_unwind)]
pub fn current_scope_stack(env: Env) -> Result<ScopeStack> {
    if let Some(scope) = callback_factory::callback_scope(&env)? {
        return Ok(ScopeStack {
            inner: scope.scope_stack.clone(),
            publication_buffer: scope.publication_buffer.clone(),
            captured_scope: scope.execution_context.as_ref().map(|_| scope.clone()),
        });
    }
    Ok(ScopeStack::from(current_scope_stack_handle()))
}

/// Binds a scope stack to the current thread or async resource.
///
/// This mutates the current execution resource. Use `withScopeStack` when
/// concurrent asynchronous branches need isolated stack replacements.
/// Captured stacks require an active JavaScript callback context; otherwise
/// this returns an error and callers must use `withScopeStack`.
#[napi(catch_unwind)]
pub fn set_thread_scope_stack(env: Env, stack: &ScopeStack) -> Result<()> {
    let scope = match &stack.captured_scope {
        Some(scope) => scope.clone(),
        None => CallbackScope {
            scope_stack: stack.inner.clone(),
            publication_buffer: effective_scope(&env)?
                .publication_buffer
                .or_else(|| stack.publication_buffer.clone()),
            propagation: None,
            execution_context: None,
            publication_context_id: None,
            publication_context_captured: false,
        },
    };
    if callback_factory::set_callback_scope(&env, scope)? {
        return Ok(());
    }
    if stack.captured_scope.is_some() {
        return Err(Error::invalid_argument(
            "cannot bind a captured scope stack without an active JavaScript callback context; use withScopeStack()",
        ));
    }
    bind_thread_scope_stack(stack.inner.clone());
    Ok(())
}

/// Returns whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns `true` if `setThreadScopeStack` has been called on the current
/// thread, or the caller is inside a task-local scope. Returns `false` when
/// only the auto-created default is present.
#[napi(catch_unwind)]
pub fn scope_stack_active(env: Env) -> Result<bool> {
    if callback_factory::callback_scope(&env)?.is_some() {
        return Ok(true);
    }
    Ok(scope_stack_is_active())
}
/// Returns the most recent callback error that could not be surfaced through a direct exception.
///
/// This is primarily used for sanitize callback paths that omit observability
/// payloads and cannot surface their errors directly.
#[napi(catch_unwind)]
pub fn get_last_callback_error() -> Option<String> {
    get_recorded_callback_error()
}

/// Clears the most recent callback error recorded by the Node binding.
#[napi(catch_unwind)]
pub fn clear_last_callback_error() {
    clear_recorded_callback_error();
}

fn settled_now<T>(mut future: BoxFuture<T>) -> Result<T> {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(value) => Ok(value),
        std::task::Poll::Pending => Err(Error::generic_failure(
            "stream callback did not settle synchronously",
        )),
    }
}

/// Internal test helper: route streams through the Tokio forwarding task.
#[napi(js_name = "__testForceStreamForwarding", catch_unwind)]
pub fn test_force_stream_forwarding(enabled: bool) {
    crate::stream::set_force_forwarding(enabled);
}

/// Internal test helper: invoke a closed JS collector wrapper and surface the queue failure.
#[napi(js_name = "__testClosedCollectorCallback", catch_unwind)]
pub fn test_closed_collector_callback(
    env: Env,
    #[napi(ts_arg_type = "(...args: any[]) => any")] callback: Callback,
    chunk: Json,
) -> Result<()> {
    clear_recorded_callback_error();
    let callback = callable::JsStreamCallback::<callable::JsCollectorCall>::new(&env, &callback)?;
    callback.abort_tsfn()?;
    let LlmCollector::Async(mut wrapped) = callable::wrap_js_collector_fn(callback) else {
        return Err(Error::generic_failure("JS collectors are asynchronous"));
    };
    settled_now(wrapped(chunk))?.map_err(Error::from)
}

/// Internal test helper: invoke a closed JS finalizer wrapper and return the fallback value.
#[napi(js_name = "__testClosedFinalizerCallback", catch_unwind)]
pub fn test_closed_finalizer_callback(
    env: Env,
    #[napi(ts_arg_type = "(...args: any[]) => any")] callback: Callback,
) -> Result<Json> {
    clear_recorded_callback_error();
    let callback = callable::JsStreamCallback::<callable::JsFinalizerCall>::new(&env, &callback)?;
    callback.abort_tsfn()?;
    let LlmFinalizer::Async(wrapped) = callable::wrap_js_finalizer_fn(callback, Default::default())
    else {
        return Err(Error::generic_failure("JS finalizers are asynchronous"));
    };
    Ok(settled_now(wrapped())?.unwrap_or(Json::Null))
}

/// Internal test helper: exercise PromiseAwareFn queue and conversion failures.
#[napi(
    js_name = "__testClosedPromiseAwareCall",
    ts_return_type = "Promise<unknown>",
    catch_unwind
)]
pub fn test_closed_promise_aware_call<'env>(
    env: &'env Env,
    #[napi(ts_arg_type = "(...args: any[]) => any")] func: Callback,
    force_conversion_failure: Option<bool>,
) -> Result<PromiseRaw<'env, Json>> {
    let promise_aware = Arc::new(PromiseAwareFn::new(env, &func)?);
    let force_conversion_failure = force_conversion_failure.unwrap_or(false);
    if !force_conversion_failure {
        promise_aware.abort_tsfn()?;
    }

    spawn_promise(
        env,
        async move {
            let invocation = if force_conversion_failure {
                Invocation::built(|_| {
                    Err(napi::Error::from_reason(
                        "forced PromiseAwareFn conversion failure",
                    ))
                })
            } else {
                Invocation::single(Json::Null)
            };
            promise_aware.call(invocation).await.map_err(Error::from)
        },
        js_value,
    )
}

/// Get the handle for the current top-of-stack execution scope.
///
/// Returns the `ScopeHandle` for the innermost active scope on the current task's scope stack.
/// Throws if the scope stack is empty.
#[napi(catch_unwind)]
pub fn get_handle(env: Env) -> Result<ScopeHandle> {
    with_effective_scope_stack(&env, core_scope_api::get_handle)?
        .map(ScopeHandle::from)
        .map_err(Error::from)
}

/// Push a new execution scope onto the scope stack.
///
/// Creates a child scope with the given `name` and `scopeType`. If `handle` is provided,
/// the new scope is parented to that scope; otherwise it is parented to the current top scope.
/// Optional `attributes` is a bitfield of scope attribute flags.
/// Optional `data` is a JSON application payload stored on the scope handle.
/// Optional `metadata` is a JSON metadata payload recorded on the scope start event.
/// Optional `input` is a semantic JSON payload exported on the scope start event.
/// Optional `timestamp` is a Unix timestamp in microseconds recorded as the handle
/// start time and start event timestamp. It must be a safe integer number; omit it
/// to use the current runtime time.
/// Returns the handle for the newly created scope.
#[napi(
    ts_generic_types = "TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json, TInput extends JsonCompatible<TInput> = Json",
    catch_unwind
)]
#[allow(clippy::too_many_arguments)]
pub fn push_scope(
    env: Env,
    name: String,
    scope_type: ScopeType,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    #[napi(ts_arg_type = "TInput | null")] input: Option<Json>,
    timestamp: Option<f64>,
) -> Result<ScopeHandle> {
    let attrs = ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let timestamp = parse_timestamp_micros(timestamp)?;
    with_effective_scope_stack(&env, || {
        core_scope_api::push_scope(
            core_scope_api::PushScopeParams::builder()
                .name(name.as_str())
                .scope_type(scope_type.into())
                .parent_opt(handle.map(|h| &h.inner))
                .attributes(attrs)
                .data_opt(opt_json(data))
                .metadata_opt(opt_json(metadata))
                .input_opt(opt_json(input))
                .timestamp_opt(timestamp)
                .build(),
        )
    })?
    .map(ScopeHandle::from)
    .map_err(Error::from)
}

/// Pop an execution scope from the scope stack.
///
/// Removes the scope identified by `handle` from the stack and emits an end event.
/// Optional `output` is a semantic JSON payload exported on the scope end event.
/// Optional `timestamp` is a Unix timestamp in microseconds recorded on the end event.
/// It must be a safe integer number; omit it to use the runtime default end timestamp.
/// Optional `metadata` is a JSON metadata payload recorded on the scope end event.
/// Throws if the handle does not match the current top scope.
#[napi(
    ts_generic_types = "TOutput extends JsonCompatible<TOutput> = Json, TMetadata extends JsonCompatible<TMetadata> = Json",
    catch_unwind
)]
pub fn pop_scope(
    env: Env,
    handle: &ScopeHandle,
    #[napi(ts_arg_type = "TOutput | null")] output: Option<Json>,
    timestamp: Option<f64>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    with_effective_scope_stack(&env, || {
        core_scope_api::pop_scope(
            core_scope_api::PopScopeParams::builder()
                .handle_uuid(&handle.inner.uuid)
                .output_opt(opt_json(output))
                .timestamp_opt(timestamp)
                .metadata_opt(opt_json(metadata))
                .build(),
        )
    })?
    .map_err(Error::from)?;
    Ok(())
}

/// Push a scope, run a callback, then pop the scope automatically.
///
/// Creates a child scope with the given `name` and `scopeType`, invokes the
/// `callback` with the new scope handle, and guarantees that the scope is popped
/// when the callback completes (whether it returns normally, throws, or returns a
/// rejected Promise). Supports both synchronous and async (Promise-returning)
/// callbacks.
///
/// Optional `handle` sets the parent scope; `attributes` is a bitfield of scope
/// attribute flags; `data` is stored on the scope handle; `metadata` is recorded
/// on the start event; and `input` is exported as the semantic start-event payload.
///
/// Returns a Promise that resolves with the callback's return value.
#[allow(clippy::too_many_arguments)]
#[napi(
    ts_generic_types = "T extends JsonCompatible<T> | void = Json, TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json, TInput extends JsonCompatible<TInput> = Json",
    ts_return_type = "Promise<T extends void ? null : T>",
    catch_unwind
)]
pub fn with_scope<'env>(
    env: &'env Env,
    name: String,
    scope_type: ScopeType,
    #[napi(ts_arg_type = "(handle: ScopeHandle) => T | Promise<T>")] callback: Callback,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    #[napi(ts_arg_type = "TInput | null")] input: Option<Json>,
) -> Result<PromiseRaw<'env, Json>> {
    let attrs = ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let context = CallContext::capture(env)?;
    let pa_fn = Arc::new(PromiseAwareFn::new(env, &callback)?);
    let scope_handle = context
        .enter(|| {
            core_scope_api::push_scope(
                core_scope_api::PushScopeParams::builder()
                    .name(name.as_str())
                    .scope_type(scope_type.into())
                    .parent_opt(handle.map(|h| &h.inner))
                    .attributes(attrs)
                    .data_opt(opt_json(data))
                    .metadata_opt(opt_json(metadata))
                    .input_opt(opt_json(input))
                    .build(),
            )
        })
        .map_err(Error::from)?;
    let scope_uuid = scope_handle.uuid;

    spawn_promise(
        env,
        context.run(async move {
            let result = pa_fn
                .call(Invocation::built(move |env| {
                    js_value(env, ScopeHandle::from(scope_handle))
                }))
                .await;
            let metadata = match &result {
                Ok(_) => otel_status_metadata("OK", None),
                Err(error) => otel_status_metadata("ERROR", Some(error.to_string())),
            };
            let popped = core_scope_api::pop_scope(
                core_scope_api::PopScopeParams::builder()
                    .handle_uuid(&scope_uuid)
                    .metadata_opt(Some(metadata))
                    .build(),
            );
            match (result, popped) {
                (Ok(value), Ok(())) => Ok(value),
                (Ok(_), Err(pop_error)) => Err(Error::from(pop_error)),
                (Err(error), popped) => {
                    if let Err(pop_error) = popped {
                        record_callback_error(format!(
                            "nemo_relay: failed to pop scope after a withScope callback failed: {pop_error}"
                        ));
                    }
                    Err(Error::from(error))
                }
            }
        }),
        js_value,
    )
}

/// Emit a custom mark event on the current scope.
///
/// Emits a named event with optional `data` and `metadata` payloads. If `handle` is provided,
/// the event is associated with that scope; otherwise it uses the current top scope.
/// Optional `timestamp` is a Unix timestamp in microseconds recorded on the mark event.
/// It must be a safe integer number; omit it to use the current runtime time.
/// Optional `dataSchema` identifies the shape of `data`; `severity` supplies the
/// typed severity used by OpenTelemetry log exporters.
#[napi(
    ts_generic_types = "TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json",
    catch_unwind
)]
#[allow(clippy::too_many_arguments)]
pub fn event(
    env: Env,
    name: String,
    handle: Option<&ScopeHandle>,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    timestamp: Option<f64>,
    data_schema: Option<DataSchema>,
    severity: Option<LogSeverity>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    with_effective_scope_stack(&env, || {
        core_scope_api::event(
            core_scope_api::EmitMarkEventParams::builder()
                .name(&name)
                .parent_opt(handle.map(|h| &h.inner))
                .data_opt(opt_json(data))
                .data_schema_opt(data_schema.map(Into::into))
                .metadata_opt(opt_json(metadata))
                .severity_opt(severity.map(Into::into))
                .timestamp_opt(timestamp)
                .build(),
        )
    })?
    .map_err(Error::from)
}

/// Emit an atomic, validated group of OpenTelemetry metric recording operations.
///
/// `name` is the metric mark name. Every measurement is validated before any
/// event is published. Optional `handle`, `metadata`, and microsecond `timestamp`
/// have the same behavior as the corresponding `event` parameters.
#[napi(
    ts_generic_types = "TMeasurements extends MetricMeasurementsCompatible<TMeasurements>, TMetadata extends JsonCompatible<TMetadata> = Json",
    catch_unwind
)]
pub fn metric(
    env: Env,
    name: String,
    #[napi(ts_arg_type = "TMeasurements")] measurements: Vec<MetricMeasurement>,
    handle: Option<&ScopeHandle>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    timestamp: Option<f64>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    with_effective_scope_stack(&env, || {
        core_scope_api::metric(
            core_scope_api::EmitMetricEventParams::builder()
                .name(&name)
                .measurements(measurements.into_iter().map(Into::into).collect())
                .parent_opt(handle.map(|handle| &handle.inner))
                .metadata_opt(opt_json(metadata))
                .timestamp_opt(timestamp)
                .build(),
        )
    })?
    .map_err(Error::from)
}

/// Begin a manual tool call lifecycle span.
///
/// Registers a tool invocation with the given `name` and `args`. Sanitize-request
/// guardrails are applied to the emitted start-event payload; request and execution
/// intercepts run only through `toolCallExecute`. Returns a `ToolHandle` that must
/// be passed to `toolCallEnd()` when the tool finishes. Optional `handle` specifies
/// the parent scope; `attributes` is a bitfield; `data` is stored on the handle;
/// `metadata` is recorded on the start event; and `toolCallId` is recorded in the
/// tool event category profile. Optional `timestamp` is a Unix timestamp in
/// microseconds recorded as the handle start time and start event timestamp. It must
/// be a safe integer number; omit it to use the current runtime time.
#[napi(
    ts_generic_types = "TArgs extends JsonCompatible<TArgs>, TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json",
    catch_unwind
)]
#[allow(clippy::too_many_arguments)]
pub fn tool_call(
    env: Env,
    name: String,
    #[napi(ts_arg_type = "TArgs")] args: Json,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    tool_call_id: Option<String>,
    timestamp: Option<f64>,
) -> Result<ToolHandle> {
    let attrs = ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let timestamp = parse_timestamp_micros(timestamp)?;
    with_effective_scope_stack(&env, || {
        core_tool_api::tool_call(
            core_tool_api::ToolCallParams::builder()
                .name(name.as_str())
                .args(args)
                .parent_opt(handle.map(|h| &h.inner))
                .attributes(attrs)
                .data_opt(opt_json(data))
                .metadata_opt(opt_json(metadata))
                .tool_call_id_opt(tool_call_id)
                .timestamp_opt(timestamp)
                .build(),
        )
    })?
    .map(ToolHandle::from)
    .map_err(Error::from)
}

/// End a manual tool call lifecycle span.
///
/// Signals that the tool call identified by `handle` has completed with the given `result`.
/// Sanitize-response guardrails are applied to the emitted end-event payload; response
/// intercepts run only through `toolCallExecute`. Optional `data` is used when the
/// sanitized result is JSON null, and optional `metadata` is recorded on the end event.
/// Optional `timestamp` is a Unix timestamp in microseconds recorded on the end event.
/// It must be a safe integer number; omit it to use the runtime default end timestamp.
#[napi(
    ts_generic_types = "TResult extends ToolExecutionResultCompatible<TResult>, TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json",
    catch_unwind
)]
pub fn tool_call_end(
    env: Env,
    handle: &ToolHandle,
    #[napi(ts_arg_type = "TResult")] result: ToolExecutionResult,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    timestamp: Option<f64>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    with_effective_scope_stack(&env, || {
        core_tool_api::tool_call_end(
            core_tool_api::ToolCallEndParams::builder()
                .handle(&handle.inner)
                .execution_result(result.into())
                .data_opt(opt_json(data))
                .metadata_opt(opt_json(metadata))
                .timestamp_opt(timestamp)
                .build(),
        )
    })?
    .map_err(Error::from)
}

/// Execute a tool call end-to-end with full lifecycle management.
///
/// By default, this call runs these stages:
///
/// 1. Check conditional-execution guardrails on the original input.
/// 2. Apply request intercepts.
/// 3. Sanitize the request copy in the `Start` event.
/// 4. Run execution intercepts around `func`.
/// 5. Sanitize the response copy in the `End` event.
///
/// If a guardrail rejects the call, Relay emits a standalone Mark event.
/// It does not emit a Start/End pair. The call returns `GuardrailRejected`.
/// Sanitizers change event data. They do not change the result returned to the caller.
/// An optional `toolCallId` is recorded in both lifecycle events.
/// Use the last `executionOptions` argument to choose which stages run for this call.
#[allow(clippy::too_many_arguments)]
#[napi(
    ts_generic_types = "TArgs extends JsonCompatible<TArgs>, TResult extends ToolExecutionResultCompatible<TResult>, TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json",
    ts_return_type = "Promise<ToolExecutionResult>",
    catch_unwind
)]
pub fn tool_call_execute<'env>(
    env: &'env Env,
    name: String,
    #[napi(ts_arg_type = "TArgs")] args: Json,
    #[napi(ts_arg_type = "(arg: Json) => TResult")] func: Callback,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    tool_call_id: Option<String>,
    execution_options: Option<ExecutionOptions>,
) -> Result<PromiseRaw<'env, ToolExecutionResult>> {
    let func = scoped_fn(
        env,
        &func,
        "JS tool execution callback failed",
        Ok,
        callable::parse_tool_execution_result,
    )?;
    execute_tool_call(
        env,
        name,
        args,
        func,
        handle,
        attributes,
        data,
        metadata,
        tool_call_id,
        execution_options,
    )
}

#[expect(clippy::too_many_arguments)]
fn execute_tool_call<'env>(
    env: &'env Env,
    name: String,
    args: Json,
    func: ToolExecutionNextFn,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    tool_call_id: Option<String>,
    execution_options: Option<ExecutionOptions>,
) -> Result<PromiseRaw<'env, ToolExecutionResult>> {
    let execution_options = execution_options.map(Into::into).unwrap_or_default();
    let attrs = ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let context = CallContext::capture(env)?;
    let parent = context.parent(handle);
    spawn_promise(
        env,
        context.run(async move {
            core_tool_api::tool_call_execute(
                core_tool_api::ToolCallExecuteParams::builder()
                    .name(name)
                    .args(args)
                    .func(func)
                    .execution_options(execution_options)
                    .parent(parent)
                    .attributes(attrs)
                    .data_opt(opt_json(data))
                    .metadata_opt(opt_json(metadata))
                    .tool_call_id_opt(tool_call_id)
                    .build(),
            )
            .await
            .map(ToolExecutionResult::from)
            .map_err(Error::from)
        }),
        js_value,
    )
}

/// Execute a tool call end-to-end with a callback that returns a value or a Promise.
///
/// Runs the same lifecycle as `toolCallExecute`. An optional `toolCallId` is
/// recorded in both lifecycle events.
/// Use the last `executionOptions` argument to choose which stages run for this call.
#[allow(clippy::too_many_arguments)]
#[napi(
    ts_generic_types = "TArgs extends JsonCompatible<TArgs>, TResult extends ToolExecutionResultCompatible<TResult>, TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json",
    ts_return_type = "Promise<ToolExecutionResult>",
    catch_unwind
)]
pub fn tool_call_execute_async<'env>(
    env: &'env Env,
    name: String,
    #[napi(ts_arg_type = "TArgs")] args: Json,
    #[napi(ts_arg_type = "(arg: Json, signal: AbortSignal) => TResult | Promise<TResult>")]
    func: Callback,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    tool_call_id: Option<String>,
    execution_options: Option<ExecutionOptions>,
) -> Result<PromiseRaw<'env, ToolExecutionResult>> {
    let pa_fn = Arc::new(PromiseAwareFn::new(env, &func)?);
    let func: ToolExecutionNextFn = Arc::new(move |args| {
        let pa_fn = pa_fn.clone();
        Box::pin(async move {
            let result = pa_fn.call(Invocation::single(args)).await?;
            callable::parse_tool_execution_result(result)
        })
    });
    execute_tool_call(
        env,
        name,
        args,
        func,
        handle,
        attributes,
        data,
        metadata,
        tool_call_id,
        execution_options,
    )
}

/// Begin a manual LLM call lifecycle span.
///
/// Registers an LLM invocation with the given provider `name` and request payload.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LlmRequest` schema. Returns an `LlmHandle` that must be passed to `llmCallEnd()`
/// when the response is received. Sanitize-request guardrails are applied to the emitted
/// start-event payload; request and execution intercepts run only through `llmCallExecute`.
/// Optional `handle` specifies the parent scope; `attributes` is a bitfield; `data` is
/// stored on the handle; `metadata` is recorded on the start event; and `modelName` is
/// recorded in the LLM event category profile. Optional `timestamp` is a Unix timestamp
/// in microseconds recorded as the handle start time and start event timestamp. It must
/// be a safe integer number; omit it to use the current runtime time.
/// Optional `codecDecode` and `codecEncode` callbacks help sanitize and annotate
/// the telemetry copy. Supply both callbacks together. The caller's request stays
/// unchanged. `codecEncode` receives an object with `annotated` and `original` fields.
/// If `codecDecode` throws or returns an invalid value, Relay skips the annotation, still
/// publishes the event, and records the error for `getLastCallbackError()`.
#[allow(clippy::too_many_arguments)]
#[napi(
    ts_generic_types = "TRequest extends LlmRequestCompatible<TRequest>, TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json, TAnnotated extends JsonCompatible<TAnnotated> = Json, TEncoded extends LlmRequestCompatible<TEncoded> = LlmRequestShape",
    catch_unwind
)]
pub fn llm_call(
    env: Env,
    name: String,
    #[napi(ts_arg_type = "TRequest")] request: Json,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    model_name: Option<String>,
    timestamp: Option<f64>,
    #[napi(ts_arg_type = "(request: LlmRequestShape) => TAnnotated")] codec_decode: Option<
        Callback,
    >,
    #[napi(ts_arg_type = "(arg: { annotated: Json; original: LlmRequestShape }) => TEncoded")]
    codec_encode: Option<Callback>,
) -> Result<LlmHandle> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let timestamp = parse_timestamp_micros(timestamp)?;
    let llm_request: LlmRequest = parse_json(request, "LlmRequest")?;
    let codec = node_llm_codec(&env, codec_decode.as_deref(), codec_encode.as_deref())?;
    let params = core_llm_api::LlmCallParams::builder()
        .name(&name)
        .request(&llm_request)
        .parent_opt(handle.map(|h| &h.inner))
        .attributes(attrs)
        .data_opt(opt_json(data))
        .metadata_opt(opt_json(metadata))
        .model_name_opt(model_name)
        .timestamp_opt(timestamp)
        .codec_opt(codec)
        .build();
    with_effective_scope_stack(&env, || core_llm_api::llm_call(params))?
        .map(LlmHandle::from)
        .map_err(Error::from)
}

/// End a manual LLM call lifecycle span.
///
/// Signals that the LLM call identified by `handle` has completed with the given `response`.
/// Sanitize-response guardrails are applied to the emitted end-event payload; response
/// intercepts run only through `llmCallExecute`. Optional `data` is used when the
/// sanitized response is JSON null, and optional `metadata` is recorded on the end event.
/// Optional `timestamp` is a Unix timestamp in microseconds recorded on the end event.
/// It must be a safe integer number; omit it to use the runtime default end timestamp.
#[napi(
    ts_generic_types = "TResponse extends JsonCompatible<TResponse>, TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json",
    catch_unwind
)]
pub fn llm_call_end(
    env: Env,
    handle: &LlmHandle,
    #[napi(ts_arg_type = "TResponse")] response: Json,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    timestamp: Option<f64>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    with_effective_scope_stack(&env, || {
        core_llm_api::llm_call_end(
            core_llm_api::LlmCallEndParams::builder()
                .handle(&handle.inner)
                .response(response)
                .data_opt(opt_json(data))
                .metadata_opt(opt_json(metadata))
                .timestamp_opt(timestamp)
                .build(),
        )
    })?
    .map_err(Error::from)
}

/// Execute an LLM call end-to-end with full lifecycle management.
///
/// By default, this call runs these stages:
///
/// 1. Check conditional-execution guardrails on the original input.
/// 2. Apply request intercepts.
/// 3. Sanitize the request copy in the `Start` event.
/// 4. Run execution intercepts around `func`.
/// 5. Sanitize the response copy in the `End` event.
///
/// If a guardrail rejects the call, Relay emits a standalone Mark event.
/// It does not emit a Start/End pair. The call returns `GuardrailRejected`.
/// Sanitizers change event data. They do not change the result returned to the caller.
/// Pass a JSON object with `headers` and `content` fields in `request`.
/// Supply `codecDecode` and `codecEncode` together or omit both.
/// Use the last `executionOptions` argument to choose which stages run for this call.
#[allow(clippy::too_many_arguments)]
#[napi(
    ts_generic_types = "TRequest extends LlmRequestCompatible<TRequest>, TResponse extends JsonCompatible<TResponse>, TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json, TAnnotated extends JsonCompatible<TAnnotated> = Json, TEncoded extends LlmRequestCompatible<TEncoded> = LlmRequestShape, TDecodedResponse extends JsonCompatible<TDecodedResponse> = Json",
    ts_return_type = "Promise<Json>",
    catch_unwind
)]
pub fn llm_call_execute<'env>(
    env: &'env Env,
    name: String,
    #[napi(ts_arg_type = "TRequest")] request: Json,
    #[napi(ts_arg_type = "(request: LlmRequestShape) => TResponse")] func: Callback,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    model_name: Option<String>,
    #[napi(ts_arg_type = "(request: LlmRequestShape) => TAnnotated")] codec_decode: Option<
        Callback,
    >,
    #[napi(ts_arg_type = "(arg: { annotated: Json; original: LlmRequestShape }) => TEncoded")]
    codec_encode: Option<Callback>,
    #[napi(ts_arg_type = "(response: Json) => TDecodedResponse")] response_codec_decode: Option<
        Callback,
    >,
    execution_options: Option<ExecutionOptions>,
) -> Result<PromiseRaw<'env, Json>> {
    let request: LlmRequest = parse_json(request, "LlmRequest")?;
    let func = scoped_fn(
        env,
        &func,
        "JS LLM execution callback failed",
        |request| serialize_llm_request(&request),
        Ok,
    )?;
    execute_llm_call(
        env,
        name,
        request,
        func,
        handle,
        attributes,
        data,
        metadata,
        model_name,
        codec_decode,
        codec_encode,
        response_codec_decode,
        execution_options,
    )
}

#[expect(clippy::too_many_arguments)]
fn execute_llm_call<'env>(
    env: &'env Env,
    name: String,
    request: LlmRequest,
    func: LlmExecutionNextFn,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    codec_decode: Option<Callback>,
    codec_encode: Option<Callback>,
    response_codec_decode: Option<Callback>,
    execution_options: Option<ExecutionOptions>,
) -> Result<PromiseRaw<'env, Json>> {
    let execution_options = execution_options.map(Into::into).unwrap_or_default();
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let context = CallContext::capture(env)?;
    let parent = context.parent(handle);
    let codec = node_llm_codec(env, codec_decode.as_deref(), codec_encode.as_deref())?;
    let response_codec = node_llm_response_codec(env, response_codec_decode.as_deref())?;
    spawn_promise(
        env,
        context.run(async move {
            let params = core_llm_api::LlmCallExecuteParams::builder()
                .name(name)
                .request(request)
                .func(func)
                .execution_options(execution_options)
                .parent(parent)
                .attributes(attrs)
                .data_opt(opt_json(data))
                .metadata_opt(opt_json(metadata))
                .model_name_opt(model_name)
                .codec_opt(codec)
                .response_codec_opt(response_codec)
                .build();
            Box::pin(core_llm_api::llm_call_execute(params))
                .await
                .map_err(Error::from)
        }),
        js_value,
    )
}

/// Execute an LLM call end-to-end with a callback that returns a value or a Promise.
///
/// Runs the same lifecycle as `llmCallExecute`. Supply `codecDecode` and `codecEncode`
/// together or omit both.
/// Use the last `executionOptions` argument to choose which stages run for this call.
#[allow(clippy::too_many_arguments)]
#[napi(
    ts_generic_types = "TRequest extends LlmRequestCompatible<TRequest>, TResponse extends JsonCompatible<TResponse>, TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json, TAnnotated extends JsonCompatible<TAnnotated> = Json, TEncoded extends LlmRequestCompatible<TEncoded> = LlmRequestShape, TDecodedResponse extends JsonCompatible<TDecodedResponse> = Json",
    ts_return_type = "Promise<Json>",
    catch_unwind
)]
pub fn llm_call_execute_async<'env>(
    env: &'env Env,
    name: String,
    #[napi(ts_arg_type = "TRequest")] request: Json,
    #[napi(
        ts_arg_type = "(request: LlmRequestShape, signal: AbortSignal) => TResponse | Promise<TResponse>"
    )]
    func: Callback,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    model_name: Option<String>,
    #[napi(ts_arg_type = "(request: LlmRequestShape) => TAnnotated")] codec_decode: Option<
        Callback,
    >,
    #[napi(ts_arg_type = "(arg: { annotated: Json; original: LlmRequestShape }) => TEncoded")]
    codec_encode: Option<Callback>,
    #[napi(ts_arg_type = "(response: Json) => TDecodedResponse")] response_codec_decode: Option<
        Callback,
    >,
    execution_options: Option<ExecutionOptions>,
) -> Result<PromiseRaw<'env, Json>> {
    let request: LlmRequest = parse_json(request, "LlmRequest")?;
    let pa_fn = Arc::new(PromiseAwareFn::new(env, &func)?);
    let func: LlmExecutionNextFn = Arc::new(move |request| {
        let pa_fn = pa_fn.clone();
        Box::pin(async move {
            pa_fn
                .call(Invocation::single(serialize_llm_request(&request)?))
                .await
        })
    });
    execute_llm_call(
        env,
        name,
        request,
        func,
        handle,
        attributes,
        data,
        metadata,
        model_name,
        codec_decode,
        codec_encode,
        response_codec_decode,
        execution_options,
    )
}

/// Execute a streaming LLM call end-to-end with full lifecycle management.
///
/// By default, conditional-execution guardrails run first on the raw request.
/// Sanitize-request guardrails only affect the emitted `Start` event payload, and
/// sanitize-response guardrails only affect the aggregated `End` event payload.
/// Returns an `LlmStream` whose `next()` method yields response chunks incrementally.
/// The `func` callback receives the intercepted request as JSON and an `AbortSignal`,
/// and returns an `AsyncIterable` of JSON chunks.
/// Stream-level intercepts are applied to each chunk.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LlmRequest` schema.
///
/// The optional `collector` callback is invoked with each intercepted chunk as JSON,
/// allowing the caller to accumulate chunks for aggregation. The optional `finalizer`
/// callback is invoked once when the stream is exhausted or closed early and
/// must return a JSON value representing the aggregated response. Both may return a
/// Promise. If the collector throws or rejects, the stream fails with that error. If
/// the finalizer throws or rejects, the aggregated response is `null`. Consumers that
/// stop reading early must await `stream.close()`, which aborts the signal and waits
/// for producer cleanup. Supply `codecDecode` and `codecEncode` together or omit both.
/// Use the last `executionOptions` argument to choose which stages run for this call.
#[allow(clippy::too_many_arguments)]
#[napi(
    ts_generic_types = "TRequest extends LlmRequestCompatible<TRequest>, TChunk extends JsonCompatible<TChunk>, TAggregate extends JsonCompatible<TAggregate> = Json, TData extends JsonCompatible<TData> = Json, TMetadata extends JsonCompatible<TMetadata> = Json, TAnnotated extends JsonCompatible<TAnnotated> = Json, TEncoded extends LlmRequestCompatible<TEncoded> = LlmRequestShape, TDecodedResponse extends JsonCompatible<TDecodedResponse> = Json",
    ts_return_type = "Promise<LlmStream>",
    catch_unwind
)]
pub fn llm_stream_call_execute<'env>(
    env: &'env Env,
    name: String,
    #[napi(ts_arg_type = "TRequest")] request: Json,
    #[napi(
        ts_arg_type = "(request: LlmRequestShape, signal: AbortSignal) => AsyncIterable<TChunk> | Promise<AsyncIterable<TChunk>>"
    )]
    func: Callback,
    #[napi(ts_arg_type = "(chunk: Json) => void | Promise<void>")] collector: Option<Callback>,
    #[napi(ts_arg_type = "() => TAggregate | Promise<TAggregate>")] finalizer: Option<Callback>,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    #[napi(ts_arg_type = "TData | null")] data: Option<Json>,
    #[napi(ts_arg_type = "TMetadata | null")] metadata: Option<Json>,
    model_name: Option<String>,
    #[napi(ts_arg_type = "(request: LlmRequestShape) => TAnnotated")] codec_decode: Option<
        Callback,
    >,
    #[napi(ts_arg_type = "(arg: { annotated: Json; original: LlmRequestShape }) => TEncoded")]
    codec_encode: Option<Callback>,
    #[napi(ts_arg_type = "(response: Json) => TDecodedResponse")] response_codec_decode: Option<
        Callback,
    >,
    execution_options: Option<ExecutionOptions>,
) -> Result<PromiseRaw<'env, LlmStream>> {
    let execution_options = execution_options.map(Into::into).unwrap_or_default();
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let context = CallContext::capture(env)?;
    let parent = context.parent(handle);
    let llm_request: LlmRequest = parse_json(request, "LlmRequest")?;

    let wrapped_collector = match collector {
        Some(cb) => callable::wrap_js_collector_fn(callable::JsStreamCallback::new(env, &cb)?),
        None => LlmCollector::from(|_: Json| Ok(())),
    };

    let finalizer_error = callable::FinalizerError::default();
    let wrapped_finalizer = match finalizer {
        Some(cb) => callable::wrap_js_finalizer_fn(
            callable::JsStreamCallback::new(env, &cb)?,
            finalizer_error.clone(),
        ),
        None => LlmFinalizer::from(|| Json::Null),
    };

    let producers = crate::promise_call::StreamCancellation::default();
    let default_fn = callable::wrap_js_llm_stream_fn(env, &func, producers.clone())?;

    let codec = node_llm_codec(env, codec_decode.as_deref(), codec_encode.as_deref())?;
    let response_codec = node_llm_response_codec(env, response_codec_decode.as_deref())?;
    let publication = context.publication();
    spawn_promise(
        env,
        context.run(async move {
            let params = core_llm_api::LlmStreamCallExecuteParams::builder()
                .name(name)
                .request(llm_request)
                .func(default_fn)
                .execution_options(execution_options)
                .collector(wrapped_collector)
                .finalizer(wrapped_finalizer)
                .parent(parent)
                .attributes(attrs)
                .data_opt(opt_json(data))
                .metadata_opt(opt_json(metadata))
                .model_name_opt(model_name)
                .codec_opt(codec)
                .response_codec_opt(response_codec)
                .build();
            let rust_stream = Box::pin(core_llm_api::llm_stream_call_execute(params))
                .await
                .map_err(Error::from)?;
            Ok::<_, Error>(LlmStream::new(
                rust_stream,
                finalizer_error,
                producers,
                publication,
            ))
        }),
        js_value,
    )
}

/// A global runtime registration surface that a conditional middleware guardrail can select.
#[napi(string_enum = "snake_case")]
pub enum RuntimeRegistrationKind {
    Subscriber,
    EventMetadataInjector,
    MarkSanitizeGuardrail,
    ScopeSanitizeStartGuardrail,
    ScopeSanitizeEndGuardrail,
    ToolSanitizeRequestGuardrail,
    ToolSanitizeResponseGuardrail,
    ToolConditionalExecutionGuardrail,
    ToolRequestIntercept,
    ToolExecutionIntercept,
    LlmSanitizeRequestGuardrail,
    LlmSanitizeResponseGuardrail,
    LlmConditionalExecutionGuardrail,
    LlmRequestIntercept,
    LlmExecutionIntercept,
    LlmStreamExecutionIntercept,
}

mirror_variants!(RuntimeRegistrationKind <=> CoreRuntimeRegistrationKind {
    Subscriber,
    EventMetadataInjector,
    MarkSanitizeGuardrail,
    ScopeSanitizeStartGuardrail,
    ScopeSanitizeEndGuardrail,
    ToolSanitizeRequestGuardrail,
    ToolSanitizeResponseGuardrail,
    ToolConditionalExecutionGuardrail,
    ToolRequestIntercept,
    ToolExecutionIntercept,
    LlmSanitizeRequestGuardrail,
    LlmSanitizeResponseGuardrail,
    LlmConditionalExecutionGuardrail,
    LlmRequestIntercept,
    LlmExecutionIntercept,
    LlmStreamExecutionIntercept,
});

/// The owner category reported for a global runtime registration.
#[napi(string_enum = "snake_case")]
pub enum RuntimeRegistrationOwnerKind {
    Core,
    GlobalApi,
    Plugin,
}

mirror_variants!(RuntimeRegistrationOwnerKind <=> CoreRuntimeRegistrationOwnerKind {
    Core,
    GlobalApi,
    Plugin,
});

/// Discovery metadata describing the owner of a runtime registration.
#[napi(object)]
pub struct RuntimeRegistrationOwner {
    pub kind: RuntimeRegistrationOwnerKind,
    pub plugin_kind: Option<String>,
    pub component_ordinal: Option<u32>,
}

/// Structured identity for one global gateable runtime registration.
#[napi(object)]
pub struct RuntimeRegistrationIdentity {
    pub kind: RuntimeRegistrationKind,
    pub local_name: String,
    pub effective_name: String,
    pub owner: RuntimeRegistrationOwner,
}

mirror_fields!(
    RuntimeRegistrationOwner
        <= core_registry_api::RuntimeRegistrationOwner {
            kind,
            plugin_kind,
            component_ordinal,
        }
);

mirror_fields!(
    RuntimeRegistrationIdentity
        <= core_registry_api::RuntimeRegistrationIdentity {
            kind,
            local_name,
            effective_name,
            owner,
        }
);

/// Register a global eligibility gate for runtime registrations matching kind and name.
#[napi(catch_unwind)]
pub fn register_conditional_middleware_guardrail(
    env: Env,
    name: String,
    #[napi(ts_arg_type = "Array<RuntimeRegistrationKind | RuntimeRegistrationKindName>")]
    kinds: Vec<RuntimeRegistrationKind>,
    registration_name: String,
    #[napi(
        ts_arg_type = "(kinds: RuntimeRegistrationKind[], registrationName: string) => string | null"
    )]
    guardrail: Callback,
) -> Result<()> {
    let selected = kinds.into_iter().map(Into::into).collect::<BTreeSet<_>>();
    core_registry_api::register_conditional_middleware_guardrail(
        &name,
        selected,
        &registration_name,
        node_conditional_middleware_guardrail(&env, &guardrail)?,
    )
    .map_err(Error::from)
}

/// Deregister a global runtime-registration eligibility gate by name.
#[napi(catch_unwind)]
pub fn deregister_conditional_middleware_guardrail(name: String) -> Result<bool> {
    core_registry_api::deregister_conditional_middleware_guardrail(&name).map_err(Error::from)
}

/// List global gateable runtime registrations.
#[napi(
    ts_generic_types = "K extends RuntimeRegistrationKind | RuntimeRegistrationKindName = RuntimeRegistrationKind",
    ts_return_type = "Array<RuntimeRegistrationIdentity & { kind: K extends RuntimeRegistrationKind ? K : Extract<RuntimeRegistrationKind, K> }>",
    catch_unwind
)]
pub fn list_runtime_registrations(
    #[napi(ts_arg_type = "Array<K> | undefined | null")] kinds: Option<
        Vec<RuntimeRegistrationKind>,
    >,
) -> Result<Vec<RuntimeRegistrationIdentity>> {
    let selected = kinds.map(|kinds| kinds.into_iter().map(Into::into).collect::<BTreeSet<_>>());
    let registrations =
        core_registry_api::list_runtime_registrations(selected.as_ref()).map_err(Error::from)?;
    Ok(registrations.into_iter().map(Into::into).collect())
}

macro_rules! registration_function {
    ([] $item:item) => {
        #[napi(catch_unwind)]
        $item
    };
    ([$generics:literal] $item:item) => {
        #[napi(ts_generic_types = $generics, catch_unwind)]
        $item
    };
}

macro_rules! registration_api {
    ($(
        $(#[$register_meta:meta])*
        fn $register:ident(
            $callback:ident: $ts_arg_type:literal
            $(, generics: $generics:literal)?
            $(, scope: $scope_uuid:ident)?
            $(, chain: $break_chain:ident)?
            $(, plugin: $plugin_method:ident($property:literal, $label:literal))?
        ) = $wrap:path => $core_register:path;
        $(#[$deregister_meta:meta])*
        fn $deregister:ident => $core_deregister:path;
    )*) => {
        $(
            registration_function! {
                [$($generics)?]
                $(#[$register_meta])*
                pub fn $register(
                    env: Env,
                    $($scope_uuid: String,)?
                    name: String,
                    priority: i32,
                    $($break_chain: bool,)?
                    #[napi(ts_arg_type = $ts_arg_type)] $callback: Callback,
                ) -> Result<()> {
                    with_effective_scope_stack(&env, || {
                        $core_register(
                            $(&parse_uuid(&$scope_uuid, "UUID")?,)?
                            &name,
                            priority,
                            $($break_chain,)?
                            $wrap(&env, &$callback)?,
                        )
                        .map_err(Error::from)
                    })?
                }
            }

            $(#[$deregister_meta])*
            #[napi(catch_unwind)]
            pub fn $deregister(env: Env, $($scope_uuid: String,)? name: String) -> Result<bool> {
                with_effective_scope_stack(&env, || {
                    $core_deregister($(&parse_uuid(&$scope_uuid, "UUID")?,)? &name)
                        .map_err(Error::from)
                })?
            }
        )*

        fn install_plugin_methods(context: &mut PluginContext<'_>) -> napi::Result<()> {
            $($(
                context.$plugin_method($property, $label, $wrap, $core_register, $core_deregister)?;
            )?)*
            Ok(())
        }
    };
}

registration_api! {
    #[doc = " Register a global event metadata injector."]
    #[doc = ""]
    #[doc = " The callback receives an immutable event snapshot and may return additions"]
    #[doc = " directly or in a Promise. Relay validates and merges accepted additions"]
    #[doc = " before event sanitizers run."]
    fn register_event_metadata_injector(injector: "(event: JsonObject) => TMetadata | Promise<TMetadata>", generics: "TMetadata extends EventMetadataCompatible<TMetadata>", plugin: method("registerEventMetadataInjector", "event metadata injector"))
        = callable::wrap_js_event_metadata_injector_promise_fn => core_registry_api::register_event_metadata_injector;
    #[doc = " Deregister a global event metadata injector by name."]
    fn deregister_event_metadata_injector => core_registry_api::deregister_event_metadata_injector;

    #[doc = " Register an event sanitize guardrail."]
    #[doc = ""]
    #[doc = " The callback may return fields directly or in a Promise. Scope and mark"]
    #[doc = " calls queue the event and return synchronously; publication resumes after"]
    #[doc = " the Promise settles. Callback, serialization, conversion, or invalid-result"]
    #[doc = " failures clear the emitted event fields and record the error for"]
    #[doc = " `getLastCallbackError()`. Await `flushSubscribers()` before inspecting"]
    #[doc = " either the delivered event or that error."]
    fn register_mark_sanitize_guardrail(guardrail: "(event: JsonObject, fields: EventSanitizeFields) => TFields | Promise<TFields>", generics: "TFields extends EventSanitizeFieldsCompatible<TFields>", plugin: method("registerMarkSanitizeGuardrail", "mark sanitize guardrail"))
        = callable::wrap_js_event_sanitize_promise_fn => core_registry_api::register_mark_sanitize_guardrail;
    fn deregister_mark_sanitize_guardrail => core_registry_api::deregister_mark_sanitize_guardrail;

    #[doc = " Register an event sanitize guardrail."]
    #[doc = ""]
    #[doc = " The callback may return fields directly or in a Promise. Scope and mark"]
    #[doc = " calls queue the event and return synchronously; publication resumes after"]
    #[doc = " the Promise settles. Callback, serialization, conversion, or invalid-result"]
    #[doc = " failures clear the emitted event fields and record the error for"]
    #[doc = " `getLastCallbackError()`. Await `flushSubscribers()` before inspecting"]
    #[doc = " either the delivered event or that error."]
    fn register_scope_sanitize_start_guardrail(guardrail: "(event: JsonObject, fields: EventSanitizeFields) => TFields | Promise<TFields>", generics: "TFields extends EventSanitizeFieldsCompatible<TFields>", plugin: method("registerScopeSanitizeStartGuardrail", "scope start sanitize guardrail"))
        = callable::wrap_js_event_sanitize_promise_fn => core_registry_api::register_scope_sanitize_start_guardrail;
    fn deregister_scope_sanitize_start_guardrail => core_registry_api::deregister_scope_sanitize_start_guardrail;

    #[doc = " Register an event sanitize guardrail."]
    #[doc = ""]
    #[doc = " The callback may return fields directly or in a Promise. Scope and mark"]
    #[doc = " calls queue the event and return synchronously; publication resumes after"]
    #[doc = " the Promise settles. Callback, serialization, conversion, or invalid-result"]
    #[doc = " failures clear the emitted event fields and record the error for"]
    #[doc = " `getLastCallbackError()`. Await `flushSubscribers()` before inspecting"]
    #[doc = " either the delivered event or that error."]
    fn register_scope_sanitize_end_guardrail(guardrail: "(event: JsonObject, fields: EventSanitizeFields) => TFields | Promise<TFields>", generics: "TFields extends EventSanitizeFieldsCompatible<TFields>", plugin: method("registerScopeSanitizeEndGuardrail", "scope end sanitize guardrail"))
        = callable::wrap_js_event_sanitize_promise_fn => core_registry_api::register_scope_sanitize_end_guardrail;
    fn deregister_scope_sanitize_end_guardrail => core_registry_api::deregister_scope_sanitize_end_guardrail;

    #[doc = " Register a guardrail that sanitizes tool request arguments before execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(toolName, args)` and must return sanitized args."]
    #[doc = " Higher `priority` values run first. Throws if a guardrail with the same `name` already exists."]
    #[doc = " If the callback throws, Relay omits the emitted payload and records the error"]
    #[doc = " for `getLastCallbackError()`."]
    fn register_tool_sanitize_request_guardrail(guardrail: "(toolName: string, value: Json) => TValue | Promise<TValue>", generics: "TValue extends JsonCompatible<TValue>", plugin: method("registerToolSanitizeRequestGuardrail", "tool sanitize request guardrail"))
        = callable::wrap_js_tool_sanitize_promise_fn => core_registry_api::register_tool_sanitize_request_guardrail;
    #[doc = " Deregister a tool request sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed."]
    fn deregister_tool_sanitize_request_guardrail => core_registry_api::deregister_tool_sanitize_request_guardrail;

    #[doc = " Register a guardrail that sanitizes tool response data after execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(toolName, result)` and must return sanitized result."]
    #[doc = " Higher `priority` values run first. Throws if a guardrail with the same `name` already exists."]
    #[doc = " If the callback throws, Relay omits the emitted payload and records the error"]
    #[doc = " for `getLastCallbackError()`."]
    fn register_tool_sanitize_response_guardrail(guardrail: "(toolName: string, value: Json) => TValue | Promise<TValue>", generics: "TValue extends JsonCompatible<TValue>", plugin: method("registerToolSanitizeResponseGuardrail", "tool sanitize response guardrail"))
        = callable::wrap_js_tool_sanitize_promise_fn => core_registry_api::register_tool_sanitize_response_guardrail;
    #[doc = " Deregister a tool response sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed."]
    fn deregister_tool_sanitize_response_guardrail => core_registry_api::deregister_tool_sanitize_response_guardrail;

    #[doc = " Register a guardrail that conditionally gates tool execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(toolName, args)` and must return `null` to allow"]
    #[doc = " execution or a rejection reason string to block it. Higher `priority` values run first."]
    #[doc = " If the callback throws, the managed call rejects and the protected callback does not run."]
    fn register_tool_conditional_execution_guardrail(guardrail: "(toolName: string, args: Json) => string | null | Promise<string | null>", plugin: method("registerToolConditionalExecutionGuardrail", "tool conditional execution guardrail"))
        = callable::wrap_js_tool_conditional_promise_fn => core_registry_api::register_tool_conditional_execution_guardrail;
    #[doc = " Deregister a tool conditional execution guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed."]
    fn deregister_tool_conditional_execution_guardrail => core_registry_api::deregister_tool_conditional_execution_guardrail;

    #[doc = " Register an intercept that transforms tool request arguments."]
    #[doc = ""]
    #[doc = " The `callable` receives `(toolName, args)` and returns transformed args. If `breakChain`"]
    #[doc = " is `true`, no lower-priority intercepts run after this one. Higher `priority` values run first."]
    #[doc = " If the callback throws, the managed call rejects and later middleware does not run."]
    fn register_tool_request_intercept(callable: "(toolName: string, args: Json) => TArgs | Promise<TArgs>", generics: "TArgs extends JsonCompatible<TArgs>", chain: break_chain, plugin: chain_method("registerToolRequestIntercept", "tool request intercept"))
        = callable::wrap_js_tool_request_intercept_promise_fn => core_registry_api::register_tool_request_intercept;
    #[doc = " Deregister a tool request intercept by name."]
    #[doc = ""]
    #[doc = " Returns `true` if an intercept with that name was found and removed."]
    fn deregister_tool_request_intercept => core_registry_api::deregister_tool_request_intercept;

    #[doc = " Register a tool execution intercept following the middleware chain pattern."]
    #[doc = ""]
    #[doc = " The `callable` receives `ToolExecutionContext` and `next`. The context"]
    #[doc = " exposes `toolName`, `args`, and `toolCallId`; `toolCallId` is `null` when"]
    #[doc = " the managed call has no provider-issued correlation identifier. Call"]
    #[doc = " `next(context.args)` to continue the chain, or return an outcome directly"]
    #[doc = " to short-circuit execution."]
    fn register_tool_execution_intercept(callable: "(context: ToolExecutionContext, next: <TNextArgs extends JsonCompatible<TNextArgs>>(args: TNextArgs) => ToolExecutionResult | Promise<ToolExecutionResult>) => TOutcome | Promise<TOutcome>", generics: "TOutcome extends ToolExecutionInterceptOutcomeCompatible<TOutcome>", plugin: method("registerToolExecutionIntercept", "tool execution intercept"))
        = callable::wrap_js_tool_exec_intercept_fn => core_registry_api::register_tool_execution_intercept;
    #[doc = " Deregister a tool execution intercept by name."]
    #[doc = ""]
    #[doc = " Removes an intercept registered through either registration shape. Returns"]
    #[doc = " `true` if an intercept with that name was found and removed."]
    fn deregister_tool_execution_intercept => core_registry_api::deregister_tool_execution_intercept;

    #[doc = " Register a guardrail that sanitizes LLM request data before execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(request, context)` and must return the sanitized request,"]
    #[doc = " or `null` to omit the observability payload. Lower `priority` values run first. Throws if a"]
    #[doc = " guardrail with the same `name` already exists. If the callback throws, Relay omits the payload"]
    #[doc = " and annotation, continues publication, and records the error for `getLastCallbackError()`."]
    fn register_llm_sanitize_request_guardrail(guardrail: "(request: LlmRequestShape, context: LlmSanitizeRequestContext) => TRequest | null | Promise<TRequest | null>", generics: "TRequest extends LlmRequestCompatible<TRequest> = LlmRequestShape", plugin: method("registerLlmSanitizeRequestGuardrail", "llm sanitize request guardrail"))
        = callable::wrap_js_llm_sanitize_request_promise_fn => core_registry_api::register_llm_sanitize_request_guardrail;
    #[doc = " Deregister an LLM request sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed."]
    fn deregister_llm_sanitize_request_guardrail => core_registry_api::deregister_llm_sanitize_request_guardrail;

    #[doc = " Register a guardrail that sanitizes LLM response data after execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(response, context)` and must return the sanitized response,"]
    #[doc = " or `null` to omit the observability payload. Lower `priority` values run first. Throws if a"]
    #[doc = " guardrail with the same `name` already exists. If the callback throws, Relay omits the payload"]
    #[doc = " and annotation, continues publication, and records the error for `getLastCallbackError()`."]
    fn register_llm_sanitize_response_guardrail(guardrail: "(response: Json, context: LlmSanitizeResponseContext) => TResponse | null | Promise<TResponse | null>", generics: "TResponse extends JsonCompatible<TResponse> = Json", plugin: method("registerLlmSanitizeResponseGuardrail", "llm sanitize response guardrail"))
        = callable::wrap_js_llm_sanitize_response_promise_fn => core_registry_api::register_llm_sanitize_response_guardrail;
    #[doc = " Deregister an LLM response sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed."]
    fn deregister_llm_sanitize_response_guardrail => core_registry_api::deregister_llm_sanitize_response_guardrail;

    #[doc = " Register a guardrail that conditionally gates LLM execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives the LLM request as JSON and must return `null` to allow"]
    #[doc = " execution or a rejection reason string to block it. Higher `priority` values run first."]
    #[doc = " If the callback throws, the managed call rejects and the protected callback does not run."]
    fn register_llm_conditional_execution_guardrail(guardrail: "(request: LlmRequestShape) => string | null | Promise<string | null>", plugin: method("registerLlmConditionalExecutionGuardrail", "llm conditional execution guardrail"))
        = callable::wrap_js_llm_conditional_promise_fn => core_registry_api::register_llm_conditional_execution_guardrail;
    #[doc = " Deregister an LLM conditional execution guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed."]
    fn deregister_llm_conditional_execution_guardrail => core_registry_api::deregister_llm_conditional_execution_guardrail;

    #[doc = " Register an intercept that transforms LLM request data."]
    #[doc = ""]
    #[doc = " The `callable` receives the `LlmRequest` (as JSON) and returns a transformed request."]
    #[doc = " If `breakChain` is `true`, no lower-priority intercepts run after this one."]
    #[doc = " Higher `priority` values run first."]
    #[doc = " If the callback throws, the managed call rejects and later middleware does not run."]
    fn register_llm_request_intercept(callable: "(args: { name: string; request: LlmRequestShape; annotated: Json | null }) => TOutcome | Promise<TOutcome>", generics: "TOutcome extends LlmRequestInterceptOutcomeCompatible<TOutcome>", chain: break_chain, plugin: chain_method("registerLlmRequestIntercept", "llm request intercept"))
        = callable::wrap_js_llm_request_intercept_promise_fn => core_registry_api::register_llm_request_intercept;
    #[doc = " Deregister an LLM request intercept by name."]
    #[doc = ""]
    #[doc = " Returns `true` if an intercept with that name was found and removed."]
    fn deregister_llm_request_intercept => core_registry_api::deregister_llm_request_intercept;

    #[doc = " Register an LLM execution intercept following the middleware chain pattern."]
    #[doc = ""]
    #[doc = " The `callable` receives the request, codec context, and a `next` function. Call"]
    #[doc = " `next(request)` to invoke the next intercept or original implementation; skip calling"]
    #[doc = " `next` to short-circuit the chain. `next` may be called repeatedly or concurrently while"]
    #[doc = " `callable` is pending; each call receives an isolated scope-stack branch, and"]
    #[doc = " unfinished or later calls reject after `callable` settles."]
    fn register_llm_execution_intercept(callable: "(request: LlmRequestShape, context: LlmExecutionContext, next: <TNextRequest extends LlmRequestCompatible<TNextRequest>>(request: TNextRequest) => Json | Promise<Json>) => TResponse | Promise<TResponse>", generics: "TResponse extends JsonCompatible<TResponse>", plugin: method("registerLlmExecutionIntercept", "llm execution intercept"))
        = callable::wrap_js_llm_exec_intercept_fn => core_registry_api::register_llm_execution_intercept;
    #[doc = " Deregister an LLM execution intercept by name."]
    #[doc = ""]
    #[doc = " Returns `true` if an intercept with that name was found and removed."]
    fn deregister_llm_execution_intercept => core_registry_api::deregister_llm_execution_intercept;

    #[doc = " Register a streaming LLM execution intercept following the middleware chain pattern."]
    #[doc = ""]
    #[doc = " The `callable` receives the request, request-codec context, and a `next` function. The"]
    #[doc = " response codec is `null` because streaming execution has no complete-response codec."]
    #[doc = " Call `next(request)` to"]
    #[doc = " invoke the next intercept or original streaming implementation; in Node the"]
    #[doc = " returned promise resolves to a lazy `AsyncIterable`. Return it directly or wrap it"]
    #[doc = " `next` to short-circuit the chain. `next` may be called repeatedly or concurrently"]
    #[doc = " while `callable` is pending; each call receives an isolated scope-stack branch,"]
    #[doc = " and unfinished or later calls reject after the returned interceptor stream settles."]
    #[doc = " A downstream stream returned successfully by `next` keeps its normal lifetime."]
    fn register_llm_stream_execution_intercept(callable: "(request: LlmRequestShape, context: LlmExecutionContext, next: <TNextRequest extends LlmRequestCompatible<TNextRequest>>(request: TNextRequest) => Promise<AsyncIterable<Json>>) => AsyncIterable<TChunk> | Promise<AsyncIterable<TChunk>>", generics: "TChunk extends JsonCompatible<TChunk>", plugin: method("registerLlmStreamExecutionIntercept", "llm stream execution intercept"))
        = callable::wrap_js_llm_stream_exec_intercept_fn => core_registry_api::register_llm_stream_execution_intercept;
    #[doc = " Deregister an LLM stream execution intercept by name."]
    #[doc = ""]
    #[doc = " Returns `true` if an intercept with that name was found and removed."]
    fn deregister_llm_stream_execution_intercept => core_registry_api::deregister_llm_stream_execution_intercept;

    #[doc = " Register an event metadata injector owned by an active scope."]
    fn scope_register_event_metadata_injector(injector: "(event: JsonObject) => TMetadata | Promise<TMetadata>", generics: "TMetadata extends EventMetadataCompatible<TMetadata>", scope: scope_uuid)
        = callable::wrap_js_event_metadata_injector_promise_fn => core_registry_api::scope_register_event_metadata_injector;
    #[doc = " Deregister a scope-local event metadata injector by name."]
    fn scope_deregister_event_metadata_injector => core_registry_api::scope_deregister_event_metadata_injector;

    #[doc = " Register a scope-local event sanitize guardrail."]
    #[doc = ""]
    #[doc = " The callback may return fields directly or in a Promise. Scope and mark"]
    #[doc = " calls queue the event and return synchronously; publication resumes after"]
    #[doc = " the Promise settles. Callback, serialization, conversion, or invalid-result"]
    #[doc = " failures clear the emitted event fields and record the error for"]
    #[doc = " `getLastCallbackError()`. Await `flushSubscribers()` before inspecting"]
    #[doc = " either the delivered event or that error."]
    fn scope_register_mark_sanitize_guardrail(guardrail: "(event: JsonObject, fields: EventSanitizeFields) => TFields | Promise<TFields>", generics: "TFields extends EventSanitizeFieldsCompatible<TFields>", scope: scope_uuid)
        = callable::wrap_js_event_sanitize_promise_fn => core_registry_api::scope_register_mark_sanitize_guardrail;
    fn scope_deregister_mark_sanitize_guardrail => core_registry_api::scope_deregister_mark_sanitize_guardrail;

    #[doc = " Register a scope-local event sanitize guardrail."]
    #[doc = ""]
    #[doc = " The callback may return fields directly or in a Promise. Scope and mark"]
    #[doc = " calls queue the event and return synchronously; publication resumes after"]
    #[doc = " the Promise settles. Callback, serialization, conversion, or invalid-result"]
    #[doc = " failures clear the emitted event fields and record the error for"]
    #[doc = " `getLastCallbackError()`. Await `flushSubscribers()` before inspecting"]
    #[doc = " either the delivered event or that error."]
    fn scope_register_scope_sanitize_start_guardrail(guardrail: "(event: JsonObject, fields: EventSanitizeFields) => TFields | Promise<TFields>", generics: "TFields extends EventSanitizeFieldsCompatible<TFields>", scope: scope_uuid)
        = callable::wrap_js_event_sanitize_promise_fn => core_registry_api::scope_register_scope_sanitize_start_guardrail;
    fn scope_deregister_scope_sanitize_start_guardrail => core_registry_api::scope_deregister_scope_sanitize_start_guardrail;

    #[doc = " Register a scope-local event sanitize guardrail."]
    #[doc = ""]
    #[doc = " The callback may return fields directly or in a Promise. Scope and mark"]
    #[doc = " calls queue the event and return synchronously; publication resumes after"]
    #[doc = " the Promise settles. Callback, serialization, conversion, or invalid-result"]
    #[doc = " failures clear the emitted event fields and record the error for"]
    #[doc = " `getLastCallbackError()`. Await `flushSubscribers()` before inspecting"]
    #[doc = " either the delivered event or that error."]
    fn scope_register_scope_sanitize_end_guardrail(guardrail: "(event: JsonObject, fields: EventSanitizeFields) => TFields | Promise<TFields>", generics: "TFields extends EventSanitizeFieldsCompatible<TFields>", scope: scope_uuid)
        = callable::wrap_js_event_sanitize_promise_fn => core_registry_api::scope_register_scope_sanitize_end_guardrail;
    fn scope_deregister_scope_sanitize_end_guardrail => core_registry_api::scope_deregister_scope_sanitize_end_guardrail;

    #[doc = " Register a scope-local guardrail that sanitizes tool request arguments before execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(toolName, args)` and must return sanitized args."]
    #[doc = " Higher `priority` values run first. Throws if a guardrail with the same `name` already exists"]
    #[doc = " on the specified scope."]
    #[doc = " If the callback throws, Relay omits the emitted payload and records the error"]
    #[doc = " for `getLastCallbackError()`."]
    fn scope_register_tool_sanitize_request_guardrail(guardrail: "(toolName: string, value: Json) => TValue | Promise<TValue>", generics: "TValue extends JsonCompatible<TValue>", scope: scope_uuid)
        = callable::wrap_js_tool_sanitize_promise_fn => core_registry_api::scope_register_tool_sanitize_request_guardrail;
    #[doc = " Deregister a scope-local tool request sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed from the specified scope."]
    fn scope_deregister_tool_sanitize_request_guardrail => core_registry_api::scope_deregister_tool_sanitize_request_guardrail;

    #[doc = " Register a scope-local guardrail that sanitizes tool response data after execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(toolName, result)` and must return sanitized result."]
    #[doc = " Higher `priority` values run first. Throws if a guardrail with the same `name` already exists"]
    #[doc = " on the specified scope."]
    #[doc = " If the callback throws, Relay omits the emitted payload and records the error"]
    #[doc = " for `getLastCallbackError()`."]
    fn scope_register_tool_sanitize_response_guardrail(guardrail: "(toolName: string, value: Json) => TValue | Promise<TValue>", generics: "TValue extends JsonCompatible<TValue>", scope: scope_uuid)
        = callable::wrap_js_tool_sanitize_promise_fn => core_registry_api::scope_register_tool_sanitize_response_guardrail;
    #[doc = " Deregister a scope-local tool response sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed from the specified scope."]
    fn scope_deregister_tool_sanitize_response_guardrail => core_registry_api::scope_deregister_tool_sanitize_response_guardrail;

    #[doc = " Register a scope-local guardrail that conditionally gates tool execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(toolName, args)` and must return `null` to allow"]
    #[doc = " execution or a rejection reason string to block it. Higher `priority` values run first."]
    #[doc = " If the callback throws, the managed call rejects and the protected callback does not run."]
    fn scope_register_tool_conditional_execution_guardrail(guardrail: "(toolName: string, args: Json) => string | null | Promise<string | null>", scope: scope_uuid)
        = callable::wrap_js_tool_conditional_promise_fn => core_registry_api::scope_register_tool_conditional_execution_guardrail;
    #[doc = " Deregister a scope-local tool conditional execution guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed from the specified scope."]
    fn scope_deregister_tool_conditional_execution_guardrail => core_registry_api::scope_deregister_tool_conditional_execution_guardrail;

    #[doc = " Register a scope-local intercept that transforms tool request arguments."]
    #[doc = ""]
    #[doc = " The `callable` receives `(toolName, args)` and returns transformed args. If `breakChain`"]
    #[doc = " is `true`, no lower-priority intercepts run after this one. Higher `priority` values run first."]
    #[doc = " If the callback throws, the managed call rejects and later middleware does not run."]
    fn scope_register_tool_request_intercept(callable: "(toolName: string, args: Json) => TArgs | Promise<TArgs>", generics: "TArgs extends JsonCompatible<TArgs>", scope: scope_uuid, chain: break_chain)
        = callable::wrap_js_tool_request_intercept_promise_fn => core_registry_api::scope_register_tool_request_intercept;
    #[doc = " Deregister a scope-local tool request intercept by name."]
    #[doc = ""]
    #[doc = " Returns `true` if an intercept with that name was found and removed from the specified scope."]
    fn scope_deregister_tool_request_intercept => core_registry_api::scope_deregister_tool_request_intercept;

    #[doc = " Register a scope-local tool execution intercept following the middleware chain pattern."]
    #[doc = ""]
    #[doc = " The `callable` receives `ToolExecutionContext` and `next`. The context"]
    #[doc = " exposes `toolName`, `args`, and `toolCallId`; `toolCallId` is `null` when"]
    #[doc = " the managed call has no provider-issued correlation identifier. Call"]
    #[doc = " `next(context.args)` to continue the chain, or return an outcome directly"]
    #[doc = " to short-circuit execution."]
    fn scope_register_tool_execution_intercept(callable: "(context: ToolExecutionContext, next: <TNextArgs extends JsonCompatible<TNextArgs>>(args: TNextArgs) => ToolExecutionResult | Promise<ToolExecutionResult>) => TOutcome | Promise<TOutcome>", generics: "TOutcome extends ToolExecutionInterceptOutcomeCompatible<TOutcome>", scope: scope_uuid)
        = callable::wrap_js_tool_exec_intercept_fn => core_registry_api::scope_register_tool_execution_intercept;
    #[doc = " Deregister a scope-local tool execution intercept by name."]
    #[doc = ""]
    #[doc = " Removes an intercept registered through either registration shape. Returns"]
    #[doc = " `true` if an intercept with that name was found and removed from the"]
    #[doc = " specified scope."]
    fn scope_deregister_tool_execution_intercept => core_registry_api::scope_deregister_tool_execution_intercept;

    #[doc = " Register a scope-local guardrail that sanitizes LLM request data before execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(request, context)` and must return the sanitized request,"]
    #[doc = " or `null` to omit the observability payload. Lower `priority` values run first. Throws if a"]
    #[doc = " guardrail with the same `name` already exists on the specified scope. If the callback throws,"]
    #[doc = " Relay omits the payload and annotation, continues publication, and records the error for"]
    #[doc = " `getLastCallbackError()`."]
    fn scope_register_llm_sanitize_request_guardrail(guardrail: "(request: LlmRequestShape, context: LlmSanitizeRequestContext) => TRequest | null | Promise<TRequest | null>", generics: "TRequest extends LlmRequestCompatible<TRequest> = LlmRequestShape", scope: scope_uuid)
        = callable::wrap_js_llm_sanitize_request_promise_fn => core_registry_api::scope_register_llm_sanitize_request_guardrail;
    #[doc = " Deregister a scope-local LLM request sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed from the specified scope."]
    fn scope_deregister_llm_sanitize_request_guardrail => core_registry_api::scope_deregister_llm_sanitize_request_guardrail;

    #[doc = " Register a scope-local guardrail that sanitizes LLM response data after execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(response, context)` and must return the sanitized response,"]
    #[doc = " or `null` to omit the observability payload. Lower `priority` values run first. Throws if a"]
    #[doc = " guardrail with the same `name` already exists on the specified scope. If the callback throws,"]
    #[doc = " Relay omits the payload and annotation, continues publication, and records the error for"]
    #[doc = " `getLastCallbackError()`."]
    fn scope_register_llm_sanitize_response_guardrail(guardrail: "(response: Json, context: LlmSanitizeResponseContext) => TResponse | null | Promise<TResponse | null>", generics: "TResponse extends JsonCompatible<TResponse> = Json", scope: scope_uuid)
        = callable::wrap_js_llm_sanitize_response_promise_fn => core_registry_api::scope_register_llm_sanitize_response_guardrail;
    #[doc = " Deregister a scope-local LLM response sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed from the specified scope."]
    fn scope_deregister_llm_sanitize_response_guardrail => core_registry_api::scope_deregister_llm_sanitize_response_guardrail;

    #[doc = " Register a scope-local guardrail that conditionally gates LLM execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives the LLM request as JSON and must return `null` to allow"]
    #[doc = " execution or a rejection reason string to block it. Higher `priority` values run first."]
    #[doc = " If the callback throws, the managed call rejects and the protected callback does not run."]
    fn scope_register_llm_conditional_execution_guardrail(guardrail: "(request: LlmRequestShape) => string | null | Promise<string | null>", scope: scope_uuid)
        = callable::wrap_js_llm_conditional_promise_fn => core_registry_api::scope_register_llm_conditional_execution_guardrail;
    #[doc = " Deregister a scope-local LLM conditional execution guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed from the specified scope."]
    fn scope_deregister_llm_conditional_execution_guardrail => core_registry_api::scope_deregister_llm_conditional_execution_guardrail;

    #[doc = " Register a scope-local intercept that transforms LLM request data."]
    #[doc = ""]
    #[doc = " The `callable` receives the `LlmRequest` (as JSON) and returns a transformed request."]
    #[doc = " If `breakChain` is `true`, no lower-priority intercepts run after this one."]
    #[doc = " Higher `priority` values run first."]
    #[doc = " If the callback throws, the managed call rejects and later middleware does not run."]
    fn scope_register_llm_request_intercept(callable: "(args: { name: string; request: LlmRequestShape; annotated: Json | null }) => TOutcome | Promise<TOutcome>", generics: "TOutcome extends LlmRequestInterceptOutcomeCompatible<TOutcome>", scope: scope_uuid, chain: break_chain)
        = callable::wrap_js_llm_request_intercept_promise_fn => core_registry_api::scope_register_llm_request_intercept;
    #[doc = " Deregister a scope-local LLM request intercept by name."]
    #[doc = ""]
    #[doc = " Returns `true` if an intercept with that name was found and removed from the specified scope."]
    fn scope_deregister_llm_request_intercept => core_registry_api::scope_deregister_llm_request_intercept;

    #[doc = " Register a scope-local LLM execution intercept following the middleware chain pattern."]
    #[doc = ""]
    #[doc = " The `callable` receives the request, codec context, and a `next` function. Call"]
    #[doc = " `next(request)` to invoke the next intercept or original implementation; skip calling `next` to"]
    #[doc = " short-circuit the chain. `next` may be called repeatedly or concurrently while"]
    #[doc = " `callable` is pending; each call receives an isolated scope-stack branch, and"]
    #[doc = " unfinished or later calls reject after `callable` settles."]
    fn scope_register_llm_execution_intercept(callable: "(request: LlmRequestShape, context: LlmExecutionContext, next: <TNextRequest extends LlmRequestCompatible<TNextRequest>>(request: TNextRequest) => Json | Promise<Json>) => TResponse | Promise<TResponse>", generics: "TResponse extends JsonCompatible<TResponse>", scope: scope_uuid)
        = callable::wrap_js_llm_exec_intercept_fn => core_registry_api::scope_register_llm_execution_intercept;
    #[doc = " Deregister a scope-local LLM execution intercept by name."]
    #[doc = ""]
    #[doc = " Returns `true` if an intercept with that name was found and removed from the specified scope."]
    fn scope_deregister_llm_execution_intercept => core_registry_api::scope_deregister_llm_execution_intercept;

    #[doc = " Register a scope-local streaming LLM execution intercept following the middleware chain pattern."]
    #[doc = ""]
    #[doc = " The `callable` receives the request, request-codec context, and a `next` function. The"]
    #[doc = " response codec is `null` because streaming execution has no complete-response codec."]
    #[doc = " Call `next(request)` to"]
    #[doc = " invoke the next intercept or original streaming implementation; in Node the"]
    #[doc = " returned promise resolves to a lazy `AsyncIterable`. Return it directly or wrap it"]
    #[doc = " `next` to short-circuit the chain. `next` may be called repeatedly or concurrently"]
    #[doc = " while `callable` is pending; each call receives an isolated scope-stack branch,"]
    #[doc = " and unfinished or later calls reject after the returned interceptor stream settles."]
    #[doc = " A downstream stream returned successfully by `next` keeps its normal lifetime."]
    fn scope_register_llm_stream_execution_intercept(callable: "(request: LlmRequestShape, context: LlmExecutionContext, next: <TNextRequest extends LlmRequestCompatible<TNextRequest>>(request: TNextRequest) => Promise<AsyncIterable<Json>>) => AsyncIterable<TChunk> | Promise<AsyncIterable<TChunk>>", generics: "TChunk extends JsonCompatible<TChunk>", scope: scope_uuid)
        = callable::wrap_js_llm_stream_exec_intercept_fn => core_registry_api::scope_register_llm_stream_execution_intercept;
    #[doc = " Deregister a scope-local LLM stream execution intercept by name."]
    #[doc = ""]
    #[doc = " Returns `true` if an intercept with that name was found and removed from the specified scope."]
    fn scope_deregister_llm_stream_execution_intercept => core_registry_api::scope_deregister_llm_stream_execution_intercept;
}

/// Register a named event subscriber that receives all lifecycle events.
///
/// The `callback` receives each event as the canonical JSON event object and may return a
/// Promise. Events are delivered asynchronously and non-blocking. Callback failures are
/// isolated, reported to stderr and `getLastCallbackError()`, and do not reject
/// `flushSubscribers()`. Throws if a subscriber with the same `name` already exists.
#[napi(catch_unwind)]
pub fn register_subscriber(
    env: Env,
    name: String,
    #[napi(ts_arg_type = "(event: JsonObject) => void | Promise<void>")] callback: Callback,
) -> Result<()> {
    let callback = callable::wrap_js_event_subscriber(&env, name.clone(), &callback)?;
    core_subscriber_api::register_subscriber(&name, callback).map_err(Error::from)
}

/// Deregister an event subscriber by name.
///
/// Future emissions stop seeing the subscriber. Already queued event snapshots may
/// still run. Returns `true` if a subscriber with that name was found and removed.
#[napi(catch_unwind)]
pub fn deregister_subscriber(name: String) -> Result<bool> {
    core_subscriber_api::deregister_subscriber(&name).map_err(Error::from)
}

/// Return a Promise that resolves when native and JavaScript subscriber callbacks and
/// managed terminal publications registered before this call finish.
///
/// Call this function outside subscribers, event sanitizers, conditional
/// guardrails, and request or execution intercepts. A queued tool or LLM
/// observability sanitizer may call it, but the Promise resolves without
/// waiting for its own publication.
///
/// Awaiting this Promise does not block the Node event loop while Promise-returning event
/// sanitizers settle or queued JavaScript subscriber callbacks run. Native events emitted by a
/// JavaScript subscriber are separate publications and may require another flush.
///
/// The Promise rejects if the blocking task fails or the core subscriber flush returns an error.
/// Callers should handle errors when awaiting it.
#[napi(ts_return_type = "Promise<void>", catch_unwind)]
pub fn flush_subscribers<'env>(env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
    let reentrant = crate::callback_factory::publication_callback_active(env)?;
    spawn_promise(
        env,
        async move {
            if reentrant {
                return Ok(());
            }
            tokio::task::spawn_blocking(|| {
                core_subscriber_api::flush_subscribers()
                    .map(|()| callable::flush_js_subscriber_callbacks())
            })
            .await
            .map_err(|error| Error::from(FlowError::Internal(error.to_string())))?
            .map_err(Error::from)
        },
        |env, _| js_value(env, ()),
    )
}

/// Register a scope-local named event subscriber that receives lifecycle events
/// for the specified scope.
///
/// The `callback` receives each event as the canonical JSON event object and may return a
/// Promise. Events are delivered asynchronously and non-blocking. Callback failures are
/// isolated, reported to stderr and `getLastCallbackError()`, and do not reject
/// `flushSubscribers()`. Throws if a subscriber with the same `name` already exists on the
/// specified scope.
#[napi(catch_unwind)]
pub fn scope_register_subscriber(
    env: Env,
    scope_uuid: String,
    name: String,
    #[napi(ts_arg_type = "(event: JsonObject) => void | Promise<void>")] callback: Callback,
) -> Result<()> {
    let uuid = parse_uuid(&scope_uuid, "UUID")?;
    let callback = callable::wrap_js_event_subscriber(&env, name.clone(), &callback)?;
    with_effective_scope_stack(&env, || {
        core_subscriber_api::scope_register_subscriber(&uuid, &name, callback).map_err(Error::from)
    })?
}

/// Deregister a scope-local event subscriber by name.
///
/// Returns `true` if a subscriber with that name was found and removed from the specified scope.
#[napi(catch_unwind)]
pub fn scope_deregister_subscriber(env: Env, scope_uuid: String, name: String) -> Result<bool> {
    let uuid = parse_uuid(&scope_uuid, "UUID")?;
    with_effective_scope_stack(&env, || {
        core_subscriber_api::scope_deregister_subscriber(&uuid, &name).map_err(Error::from)
    })?
}

/// Run the registered tool request intercept chain on the given arguments.
/// Returns the transformed arguments.
#[napi(
    ts_generic_types = "TArgs extends JsonCompatible<TArgs>",
    ts_return_type = "Promise<Json>",
    catch_unwind
)]
pub fn tool_request_intercepts<'env>(
    env: &'env Env,
    name: String,
    #[napi(ts_arg_type = "TArgs")] args: Json,
) -> Result<PromiseRaw<'env, Json>> {
    let context = CallContext::capture(env)?;
    spawn_promise(
        env,
        context.run(async move {
            core_tool_api::tool_request_intercepts(&name, args)
                .await
                .map_err(Error::from)
        }),
        js_value,
    )
}

/// Run the registered tool conditional execution guardrail chain.
/// Throws if any guardrail rejects.
#[napi(
    ts_generic_types = "TArgs extends JsonCompatible<TArgs>",
    ts_return_type = "Promise<void>",
    catch_unwind
)]
pub fn tool_conditional_execution<'env>(
    env: &'env Env,
    name: String,
    #[napi(ts_arg_type = "TArgs")] args: Json,
) -> Result<PromiseRaw<'env, ()>> {
    let context = CallContext::capture(env)?;
    spawn_promise(
        env,
        context.run(async move {
            core_tool_api::tool_conditional_execution(&name, &args)
                .await
                .map_err(Error::from)
        }),
        |env, _| js_value(env, ()),
    )
}

/// Run the registered LLM request intercept chain on the given request.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LlmRequest` schema. Returns the transformed request as JSON.
#[napi(
    ts_generic_types = "TRequest extends LlmRequestCompatible<TRequest>",
    ts_return_type = "Promise<LlmRequestInterceptOutcome>",
    catch_unwind
)]
pub fn llm_request_intercepts<'env>(
    env: &'env Env,
    name: String,
    #[napi(ts_arg_type = "TRequest")] request: Json,
) -> Result<PromiseRaw<'env, Json>> {
    let llm_request: LlmRequest = parse_json(request, "LlmRequest")?;
    let context = CallContext::capture(env)?;
    spawn_promise(
        env,
        context.run(async move {
            Box::pin(core_llm_api::llm_request_intercepts(&name, llm_request))
                .await
                .map(|r| {
                    serde_json::json!({
                        "request": r.request,
                        "annotated": r.annotated_request,
                        "pendingMarks": callable::js_pending_marks(r.pending_marks),
                        "optimizationContributions": r.optimization_contributions,
                    })
                })
                .map_err(Error::from)
        }),
        js_value,
    )
}

/// Run the registered LLM conditional execution guardrail chain.
/// Throws if any guardrail rejects. The `request` should be a JSON object with `headers`
/// and `content` fields matching the `LlmRequest` schema.
#[napi(
    ts_generic_types = "TRequest extends LlmRequestCompatible<TRequest>",
    ts_return_type = "Promise<void>",
    catch_unwind
)]
pub fn llm_conditional_execution<'env>(
    env: &'env Env,
    #[napi(ts_arg_type = "TRequest")] request: Json,
) -> Result<PromiseRaw<'env, ()>> {
    let llm_request: LlmRequest = parse_json(request, "LlmRequest")?;
    let context = CallContext::capture(env)?;
    spawn_promise(
        env,
        context.run(async move {
            core_llm_api::llm_conditional_execution(&llm_request)
                .await
                .map_err(Error::from)
        }),
        |env, _| js_value(env, ()),
    )
}

/// An Agent Trajectory Interchange Format (ATIF) exporter that collects lifecycle
/// events and exports them as a structured trajectory.
///
/// Create an instance with session and agent metadata, then register it as an event subscriber.
/// When ready, call `exportJson()` to serialize the collected trajectory.
#[napi]
pub struct AtifExporter {
    inner: nemo_relay_observability::atif::AtifExporter,
}

#[napi]
impl AtifExporter {
    /// Create a new ATIF exporter.
    ///
    /// `sessionId` identifies the session. `agentName` and `agentVersion` describe the agent.
    /// Optional `modelName` records the LLM model used.
    #[napi(constructor, catch_unwind)]
    pub fn new(
        session_id: String,
        agent_name: String,
        agent_version: String,
        model_name: Option<String>,
    ) -> Result<Self> {
        let agent_info = nemo_relay_observability::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: None,
            extra: None,
        };
        Ok(Self {
            inner: nemo_relay_plugin_host::exporter::atif(session_id, agent_info),
        })
    }

    /// Register this exporter as an event subscriber with the given name.
    ///
    /// Throws if a subscriber with the same `name` already exists.
    #[napi(catch_unwind)]
    pub fn register(&self, name: String) -> Result<()> {
        let subscriber = self.inner.subscriber();
        core_subscriber_api::register_subscriber(&name, subscriber).map_err(Error::from)
    }

    /// Deregister this exporter's event subscriber by name.
    ///
    /// Returns `true` if a subscriber with that name was found and removed.
    #[napi(catch_unwind)]
    pub fn deregister(&self, name: String) -> Result<bool> {
        core_subscriber_api::deregister_subscriber(&name).map_err(Error::from)
    }

    /// Export the collected trajectory as a JSON string.
    ///
    /// Returns a JSON-serialized `AtifTrajectory`.
    #[napi(catch_unwind)]
    pub fn export_json(&self) -> Result<String> {
        let trajectory = self.inner.try_export().map_err(Error::from)?;
        serde_json::to_string(&trajectory).map_err(Error::from)
    }

    /// Clear all collected events from the exporter.
    #[napi(catch_unwind)]
    pub fn clear(&self) {
        self.inner.clear();
    }
}

/// One tagged sink configuration for `AtofExporter`.
#[napi(object)]
#[derive(Default)]
pub struct AtofExporterConfig {
    /// Sink type: `"file"` (default) or `"stream"`.
    pub r#type: Option<String>,
    /// Output directory. Defaults to the current working directory.
    pub output_directory: Option<String>,
    /// `"append"` (default) or `"overwrite"`.
    pub mode: Option<String>,
    /// Output filename. Defaults to `nemo-relay-events-YYYY-MM-DD-HH.MM.SS.jsonl`.
    pub filename: Option<String>,
    /// Stream endpoint URL. Required when `type` is `"stream"`.
    pub url: Option<String>,
    /// `"http_post"` (default), `"websocket"`, or `"ndjson"`.
    pub transport: Option<String>,
    /// Extra stream headers as string key/value pairs.
    pub headers: Option<Json>,
    /// Header names mapped to environment variables that supply their values.
    pub header_env: Option<Json>,
    /// Per-stream timeout in milliseconds.
    pub timeout_millis: Option<u32>,
    /// Field name policy applied before sending stream events.
    pub field_name_policy: Option<String>,
    /// Allow `http://` or `ws://` to any host; events and configured headers travel unencrypted.
    pub allow_insecure: Option<bool>,
}

/// Single-sink Agent Trajectory Observability Format (ATOF) exporter.
#[napi]
pub struct AtofExporter {
    inner: nemo_relay_observability::atof::AtofExporter,
}

#[napi]
impl AtofExporter {
    /// Create a new Agent Trajectory Observability Format (ATOF) JSONL exporter
    /// from a config object.
    #[napi(constructor, catch_unwind)]
    pub fn new(config: Option<AtofExporterConfig>) -> Result<Self> {
        let inner = nemo_relay_plugin_host::exporter::atof(build_atof_config(config)?)
            .map_err(Error::from)?;
        Ok(Self { inner })
    }

    /// Return the JSONL output path, or `null` for a stream sink.
    #[napi(getter, catch_unwind)]
    pub fn path(&self) -> Option<String> {
        self.inner
            .path()
            .map(|path| path.to_string_lossy().into_owned())
    }

    /// Register this exporter globally with the given name.
    #[napi(catch_unwind)]
    pub fn register(&self, name: String) -> Result<()> {
        self.inner.register(&name).map_err(Error::from)
    }

    /// Deregister a subscriber by name.
    #[napi(catch_unwind)]
    pub fn deregister(&self, name: String) -> Result<bool> {
        self.inner.deregister(&name).map_err(Error::from)
    }

    /// Outside subscriber and middleware callbacks, wait for queued subscriber delivery, then
    /// flush the file sink or ask the stream sink to drain for up to its timeout. A stream timeout
    /// is logged and does not by itself return an error.
    #[napi(catch_unwind)]
    pub fn force_flush(&self) -> Result<()> {
        self.inner.force_flush().map_err(Error::from)
    }

    /// Outside subscriber and middleware callbacks, wait for queued subscriber delivery, then
    /// flush the file sink or ask the stream sink to drain and close up to its timeout. A stream
    /// timeout is logged and does not by itself return an error.
    #[napi(catch_unwind)]
    pub fn shutdown(&self) -> Result<()> {
        self.inner.shutdown().map_err(Error::from)
    }

    /// Return delivery diagnostics recorded by this exporter's stream sink.
    #[napi(
        ts_return_type = "Array<{ code: string; message: string; count: number }>",
        catch_unwind
    )]
    pub fn runtime_diagnostics(&self) -> Json {
        exporter_runtime_diagnostics_json(self.inner.runtime_diagnostics())
    }
}

/// Mutable configuration object for `OpenTelemetrySubscriber`.
#[napi(object)]
pub struct OpenTelemetryConfig {
    /// `"full"`, `"gen_ai"`, or `"openinference"`.
    #[napi(ts_type = "\"full\" | \"gen_ai\" | \"openinference\"")]
    pub r#type: String,
    /// `"http_binary"` (default) or `"grpc"`.
    pub transport: Option<String>,
    /// OTLP endpoint, such as `http://localhost:4318/v1/traces`.
    pub endpoint: String,
    /// TLS certificate and private key file paths for this endpoint.
    pub tls: Option<OpenTelemetryTlsConfig>,
    /// Extra exporter headers/metadata as string key/value pairs.
    pub headers: Option<Json>,
    /// Header names mapped to environment variables resolved during subscriber activation.
    #[napi(ts_type = "Record<string, string>")]
    pub header_env: Option<Json>,
    /// Allow plain HTTP to any host; data and configured headers travel unencrypted.
    pub allow_insecure: Option<bool>,
    /// Extra OpenTelemetry resource attributes as string key/value pairs.
    pub resource_attributes: Option<Json>,
    /// `service.name` resource attribute. Defaults to `"unknown_service"`.
    pub service_name: Option<String>,
    /// Optional `service.namespace` resource attribute.
    pub service_namespace: Option<String>,
    /// Optional `service.version` resource attribute.
    pub service_version: Option<String>,
    /// Instrumentation scope name. Defaults to `"opentelemetry"`.
    pub instrumentation_scope: Option<String>,
    /// Export timeout in milliseconds. Defaults to `3000`.
    pub timeout_millis: Option<u32>,
    /// Completed scope lineage retention in milliseconds as a `bigint`. Defaults to `60000`.
    pub completed_span_context_ttl_millis: Option<BigInt>,
    /// Mark projection for full and OpenInference exporters. Defaults to `"inherit"`.
    #[napi(ts_type = "\"inherit\" | \"event\" | \"tool\"")]
    pub mark_projection: Option<String>,
    /// Mark names excluded from full and OpenInference projections.
    pub mark_exclude_names: Option<Vec<String>>,
    /// Attribute aliases for full and OpenInference projections.
    pub attribute_mappings: Option<Json>,
    /// Literal Event metadata prefixes copied to top-level OTLP attributes.
    pub promote_metadata_prefixes: Option<Vec<String>>,
    /// Literal root-scope Event metadata prefixes copied to OTLP resource attributes.
    pub promote_resource_metadata_prefixes: Option<Vec<String>>,
}

/// Paths to PEM certificate and private key files for one OTLP endpoint.
#[napi(object)]
pub struct OpenTelemetryTlsConfig {
    /// CA bundle that replaces the system's trusted CAs.
    pub certificate: Option<String>,
    /// Client certificate chain. Also set `privateKey`.
    pub client_certificate: Option<String>,
    /// Client private key. Also set `clientCertificate`.
    pub private_key: Option<String>,
}

mirror_fields!(OpenTelemetryTlsConfig => nemo_relay_observability::OpenTelemetryTlsConfig {
    certificate, client_certificate, private_key,
});

/// Configuration for a subscriber that writes OTLP to a local file.
///
/// Carries no endpoint, transport, headers, or timeout: a file destination has
/// no use for them.
#[napi(object)]
pub struct OpenTelemetryFileSinkConfig {
    /// `"full"`, `"gen_ai"`, or `"openinference"`.
    #[napi(ts_type = "\"full\" | \"gen_ai\" | \"openinference\"")]
    pub r#type: String,
    /// Directory containing the output file.
    pub output_directory: String,
    /// Output filename. Defaults to a name derived from `format`.
    pub filename: Option<String>,
    /// `"json_lines"` (default) or `"proto"`.
    #[napi(ts_type = "\"json_lines\" | \"proto\"")]
    pub format: Option<String>,
    /// `"overwrite"` (default) or `"append"`.
    #[napi(ts_type = "\"append\" | \"overwrite\"")]
    pub mode: Option<String>,
    /// Extra OpenTelemetry resource attributes as string key/value pairs.
    pub resource_attributes: Option<Json>,
    /// `service.name` resource attribute. Defaults to `"unknown_service"`.
    pub service_name: Option<String>,
    /// Optional `service.namespace` resource attribute.
    pub service_namespace: Option<String>,
    /// Optional `service.version` resource attribute.
    pub service_version: Option<String>,
    /// Instrumentation scope name. Defaults to `"opentelemetry"`.
    pub instrumentation_scope: Option<String>,
    /// Completed scope lineage retention in milliseconds as a `bigint`. Defaults to `60000`.
    pub completed_span_context_ttl_millis: Option<BigInt>,
    /// Mark projection for full and OpenInference exporters. Defaults to `"inherit"`.
    #[napi(ts_type = "\"inherit\" | \"event\" | \"tool\"")]
    pub mark_projection: Option<String>,
    /// Mark names excluded from full and OpenInference projections.
    pub mark_exclude_names: Option<Vec<String>>,
    /// Attribute aliases for full and OpenInference projections.
    pub attribute_mappings: Option<Json>,
    /// Literal Event metadata prefixes copied to top-level OTLP attributes.
    pub promote_metadata_prefixes: Option<Vec<String>>,
    /// Literal root-scope Event metadata prefixes copied to OTLP resource attributes.
    pub promote_resource_metadata_prefixes: Option<Vec<String>>,
}

fn build_otel_file_sink_config(
    options: OpenTelemetryFileSinkConfig,
) -> Result<nemo_relay_observability::otel::OpenTelemetryFileSinkConfig> {
    let otel_type = parse_otel_type(&options.r#type)?;
    let directory = nonblank(&options.output_directory, "outputDirectory")?;
    let sink = parse_otel_file_sink(
        &directory,
        options.filename.as_deref(),
        options.format.as_deref(),
        options.mode.as_deref(),
    )?;
    let mut config =
        nemo_relay_observability::otel::OpenTelemetryFileSinkConfig::new(otel_type, sink);
    if let Some(service_name) = options.service_name {
        config = config.with_service_name(service_name);
    }
    service_options!(config, options);
    trace_options!(config, options);
    Ok(config)
}

/// OpenTelemetry-backed event subscriber.
#[napi]
pub struct OpenTelemetrySubscriber {
    inner: nemo_relay_observability::otel::OpenTelemetrySubscriber,
}

#[napi]
impl OpenTelemetrySubscriber {
    /// Create a new OpenTelemetry subscriber from a config object.
    #[napi(constructor, catch_unwind)]
    pub fn new(config: OpenTelemetryConfig) -> Result<Self> {
        let inner = nemo_relay_plugin_host::exporter::opentelemetry(build_otel_config(config)?)
            .map_err(Error::from)?;
        Ok(Self { inner })
    }

    /// Create a subscriber that writes OTLP to a local file.
    #[napi(factory, catch_unwind)]
    pub fn file_sink(config: OpenTelemetryFileSinkConfig) -> Result<Self> {
        let inner = nemo_relay_plugin_host::exporter::opentelemetry_file(
            build_otel_file_sink_config(config)?,
        )
        .map_err(Error::from)?;
        Ok(Self { inner })
    }
}

/// Configuration object for `OpenTelemetryLogSubscriber`.
#[napi(object)]
pub struct OpenTelemetryLogConfig {
    /// OTLP endpoint. Bare HTTP origins append `/v1/logs`.
    pub endpoint: String,
    /// TLS certificate and private key file paths for this endpoint.
    pub tls: Option<OpenTelemetryTlsConfig>,
    /// `"http_binary"` (default) or `"grpc"`.
    #[napi(ts_type = "\"http_binary\" | \"grpc\"")]
    pub transport: Option<String>,
    /// Extra exporter headers/metadata as string key/value pairs.
    #[napi(ts_type = "Record<string, string>")]
    pub headers: Option<Json>,
    /// Header names mapped to environment variables resolved during subscriber activation.
    #[napi(ts_type = "Record<string, string>")]
    pub header_env: Option<Json>,
    /// Allow plain HTTP to any host; data and configured headers travel unencrypted.
    pub allow_insecure: Option<bool>,
    /// Extra OpenTelemetry resource attributes as string key/value pairs.
    #[napi(ts_type = "Record<string, string>")]
    pub resource_attributes: Option<Json>,
    /// `service.name` resource attribute. Defaults to `"unknown_service"`.
    pub service_name: Option<String>,
    /// Optional `service.namespace` resource attribute.
    pub service_namespace: Option<String>,
    /// Optional `service.version` resource attribute.
    pub service_version: Option<String>,
    /// Instrumentation scope name. Defaults to `"opentelemetry"`.
    pub instrumentation_scope: Option<String>,
    /// Export timeout in milliseconds. Defaults to `3000`.
    pub timeout_millis: Option<u32>,
    /// Minimum severity exported. Defaults to `LogSeverity.Info`.
    pub minimum_severity: Option<LogSeverity>,
    /// Maximum queued log records. Defaults to `2048`.
    pub max_queue_size: Option<u32>,
    /// Maximum records per export batch. Defaults to `512`.
    pub max_export_batch_size: Option<u32>,
    /// Maximum delay before a partial batch is exported. Defaults to `1000`.
    pub scheduled_delay_millis: Option<u32>,
    /// Completed scope lineage retention in milliseconds as a `bigint`. Defaults to `60000`.
    pub completed_span_context_ttl_millis: Option<BigInt>,
}

/// Subscriber that exports severity-tagged marks through OTLP logs.
#[napi]
pub struct OpenTelemetryLogSubscriber {
    inner: nemo_relay_observability::otel_logs::OpenTelemetryLogSubscriber,
}

#[napi]
impl OpenTelemetryLogSubscriber {
    /// Create a log subscriber from a config object.
    #[napi(constructor, catch_unwind)]
    pub fn new(config: OpenTelemetryLogConfig) -> Result<Self> {
        let inner =
            nemo_relay_plugin_host::exporter::opentelemetry_logs(build_otel_log_config(config)?)
                .map_err(Error::from)?;
        Ok(Self { inner })
    }
}

/// Configuration object for `OpenTelemetryMetricSubscriber`.
#[napi(object)]
pub struct OpenTelemetryMetricConfig {
    /// OTLP endpoint. Bare HTTP origins append `/v1/metrics`.
    pub endpoint: String,
    /// TLS certificate and private key file paths for this endpoint.
    pub tls: Option<OpenTelemetryTlsConfig>,
    /// `"http_binary"` (default) or `"grpc"`.
    #[napi(ts_type = "\"http_binary\" | \"grpc\"")]
    pub transport: Option<String>,
    /// Extra exporter headers/metadata as string key/value pairs.
    #[napi(ts_type = "Record<string, string>")]
    pub headers: Option<Json>,
    /// Header names mapped to environment variables resolved during subscriber activation.
    #[napi(ts_type = "Record<string, string>")]
    pub header_env: Option<Json>,
    /// Allow plain HTTP to any host; data and configured headers travel unencrypted.
    pub allow_insecure: Option<bool>,
    /// Extra OpenTelemetry resource attributes as string key/value pairs.
    #[napi(ts_type = "Record<string, string>")]
    pub resource_attributes: Option<Json>,
    /// `service.name` resource attribute. Defaults to `"unknown_service"`.
    pub service_name: Option<String>,
    /// Optional `service.namespace` resource attribute.
    pub service_namespace: Option<String>,
    /// Optional `service.version` resource attribute.
    pub service_version: Option<String>,
    /// Instrumentation scope name. Defaults to `"opentelemetry"`.
    pub instrumentation_scope: Option<String>,
    /// Export timeout in milliseconds. Defaults to `3000`.
    pub timeout_millis: Option<u32>,
    /// Collection interval in milliseconds. Defaults to `60000`.
    pub export_interval_millis: Option<u32>,
    /// Preferred aggregation temporality. Defaults to cumulative.
    pub temporality: Option<MetricTemporality>,
    /// Maximum number of retained instrument descriptors. Defaults to `256`.
    pub max_instruments: Option<u32>,
    /// Maximum series cardinality per instrument. Defaults to `2000`.
    pub cardinality_limit: Option<u32>,
}

/// Subscriber that records metric marks and exports them through OTLP metrics.
#[napi]
pub struct OpenTelemetryMetricSubscriber {
    inner: nemo_relay_observability::otel_metrics::OpenTelemetryMetricSubscriber,
}

#[napi]
impl OpenTelemetryMetricSubscriber {
    /// Create a metric subscriber from a config object.
    #[napi(constructor, catch_unwind)]
    pub fn new(config: OpenTelemetryMetricConfig) -> Result<Self> {
        let inner = nemo_relay_plugin_host::exporter::opentelemetry_metrics(
            build_otel_metric_config(config)?,
        )
        .map_err(Error::from)?;
        Ok(Self { inner })
    }
}

macro_rules! otel_subscriber_methods {
    ($(
        $subscriber:ident {
            $(#[$flush_meta:meta])* force_flush;
            $(#[$shutdown_meta:meta])* shutdown;
        }
    )*) => {$(
        #[napi]
        impl $subscriber {
            /// Register this subscriber globally with the given name.
            #[napi(catch_unwind)]
            pub fn register(&self, name: String) -> Result<()> {
                self.inner.register(&name).map_err(Error::from)
            }

            /// Deregister a subscriber by name.
            #[napi(catch_unwind)]
            pub fn deregister(&self, name: String) -> Result<bool> {
                self.inner.deregister(&name).map_err(Error::from)
            }

            $(#[$flush_meta])*
            #[napi(catch_unwind)]
            pub fn force_flush(&self) -> Result<()> {
                self.inner.force_flush().map_err(Error::from)
            }

            /// Return bounded runtime diagnostics recorded by this subscriber.
            #[napi(
                ts_return_type = "Array<{ code: string; message: string; count: number }>",
                catch_unwind
            )]
            pub fn runtime_diagnostics(&self) -> Json {
                exporter_runtime_diagnostics_json(self.inner.runtime_diagnostics())
            }

            $(#[$shutdown_meta])*
            #[napi(catch_unwind)]
            pub fn shutdown(&self) -> Result<()> {
                self.inner.shutdown().map_err(Error::from)
            }
        }
    )*};
}

otel_subscriber_methods! {
    OpenTelemetrySubscriber {
        #[doc = " Force a flush of finished spans through the exporter."]
        #[doc = ""]
        #[doc = " A successful flush updates `runtimeDiagnostics()` with queue drops observed so far."]
        force_flush;
        #[doc = " Shut down the underlying tracer provider."]
        shutdown;
    }
    OpenTelemetryLogSubscriber {
        #[doc = " Flush queued Relay events and the OTLP log processor."]
        #[doc = ""]
        #[doc = " A successful flush updates `runtimeDiagnostics()` with queue drops observed so far."]
        force_flush;
        #[doc = " Shut down the underlying logger provider."]
        shutdown;
    }
    OpenTelemetryMetricSubscriber {
        #[doc = " Collect and export current metric aggregates immediately."]
        force_flush;
        #[doc = " Shut down the underlying meter provider."]
        shutdown;
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CacheTelemetryOptions {
    provider: String,
    #[serde(alias = "request_id")]
    request_id: String,
    usage: Option<Json>,
    #[serde(default, alias = "request_facts")]
    request_facts: Option<Json>,
    #[serde(alias = "agent_id")]
    agent_id: String,
    #[serde(alias = "template_version")]
    template_version: String,
    #[serde(alias = "toolset_hash")]
    toolset_hash: String,
    #[serde(alias = "model_family")]
    model_family: String,
    #[serde(alias = "tenant_scope")]
    tenant_scope: String,
    timestamp: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CacheRequestFactsOptions {
    provider: String,
    #[serde(alias = "request_id")]
    request_id: String,
    #[serde(alias = "annotated_request")]
    annotated_request: Json,
    #[serde(alias = "agent_id")]
    agent_id: String,
    timestamp: Option<String>,
}

enum NodeAdaptiveRuntimeState {
    Pending {
        config: AdaptiveConfig,
        report: nemo_relay_plugin_host::ConfigReport,
    },
    Ready(CoreAdaptiveRuntime),
}

const ADAPTIVE_RUNTIME_LOCKED: &str =
    "adaptive runtime is locked by an async operation; try again after await completes";
const ADAPTIVE_RUNTIME_SHUT_DOWN: &str = "adaptive runtime already shut down";
const ADAPTIVE_RUNTIME_LIFECYCLE_STOPPED: &str = "adaptive runtime lifecycle task stopped";

type SharedAdaptiveRuntimeState = Arc<tokio::sync::Mutex<Option<NodeAdaptiveRuntimeState>>>;

type LifecycleReply = tokio::sync::oneshot::Sender<Result<()>>;

enum LifecycleCommand {
    Register(LifecycleReply),
    Shutdown(LifecycleReply),
}

async fn run_adaptive_lifecycle(
    state: SharedAdaptiveRuntimeState,
    mut commands: tokio::sync::mpsc::UnboundedReceiver<LifecycleCommand>,
) {
    while let Some(command) = commands.recv().await {
        let (reply, result) = {
            let mut state = state.lock().await;
            match command {
                LifecycleCommand::Register(reply) => {
                    (reply, register_adaptive_runtime(&mut state).await)
                }
                LifecycleCommand::Shutdown(reply) => {
                    (reply, shutdown_adaptive_runtime(&mut state).await)
                }
            }
        };
        let _ = reply.send(result);
    }
}

async fn register_adaptive_runtime(state: &mut Option<NodeAdaptiveRuntimeState>) -> Result<()> {
    match state {
        None => Err(Error::generic_failure(ADAPTIVE_RUNTIME_SHUT_DOWN)),
        Some(NodeAdaptiveRuntimeState::Pending { config, .. }) => {
            let mut runtime = CoreAdaptiveRuntime::new(config.clone()).await?;
            let registered = runtime.register().await.map_err(Error::from);
            *state = Some(NodeAdaptiveRuntimeState::Ready(runtime));
            registered
        }
        Some(NodeAdaptiveRuntimeState::Ready(runtime)) => {
            runtime.register().await.map_err(Error::from)
        }
    }
}

async fn shutdown_adaptive_runtime(state: &mut Option<NodeAdaptiveRuntimeState>) -> Result<()> {
    match state.take() {
        None => Err(Error::generic_failure(ADAPTIVE_RUNTIME_SHUT_DOWN)),
        Some(NodeAdaptiveRuntimeState::Pending { .. }) => Ok(()),
        Some(NodeAdaptiveRuntimeState::Ready(runtime)) => {
            runtime.shutdown().await.map_err(Error::from)
        }
    }
}

/// Owned adaptive runtime that can register adaptive features outside the plugin system.
#[napi]
pub struct AdaptiveRuntime {
    state: SharedAdaptiveRuntimeState,
    commands: tokio::sync::mpsc::UnboundedSender<LifecycleCommand>,
}

impl AdaptiveRuntime {
    fn state(&self) -> Result<tokio::sync::MappedMutexGuard<'_, NodeAdaptiveRuntimeState>> {
        let guard = self
            .state
            .try_lock()
            .map_err(|_| Error::generic_failure(ADAPTIVE_RUNTIME_LOCKED))?;
        tokio::sync::MutexGuard::try_map(guard, Option::as_mut)
            .map_err(|_| Error::generic_failure(ADAPTIVE_RUNTIME_SHUT_DOWN))
    }

    fn ready(
        &self,
        action: &str,
    ) -> Result<tokio::sync::MappedMutexGuard<'_, CoreAdaptiveRuntime>> {
        tokio::sync::MappedMutexGuard::try_map(self.state()?, |state| match state {
            NodeAdaptiveRuntimeState::Ready(runtime) => Some(runtime),
            NodeAdaptiveRuntimeState::Pending { .. } => None,
        })
        .map_err(|_| {
            Error::generic_failure(format!(
                "adaptive runtime must be registered before {action}"
            ))
        })
    }

    fn lifecycle<'env>(
        &self,
        env: &'env Env,
        command: fn(LifecycleReply) -> LifecycleCommand,
    ) -> Result<PromiseRaw<'env, ()>> {
        let (reply, completed) = tokio::sync::oneshot::channel();
        self.commands
            .send(command(reply))
            .map_err(|_| Error::generic_failure(ADAPTIVE_RUNTIME_LIFECYCLE_STOPPED))?;
        spawn_promise(
            env,
            async move {
                completed
                    .await
                    .map_err(|_| Error::generic_failure(ADAPTIVE_RUNTIME_LIFECYCLE_STOPPED))?
            },
            |env, ()| js_value(env, ()),
        )
    }
}

#[napi]
impl AdaptiveRuntime {
    /// Create an adaptive runtime wrapper from config.
    ///
    /// The runtime is constructed lazily when `register()` is awaited.
    #[napi(
        factory,
        js_name = "create",
        ts_generic_types = "TConfig extends JsonCompatible<TConfig>",
        catch_unwind
    )]
    pub fn create(#[napi(ts_arg_type = "TConfig")] config: Json) -> Result<Self> {
        let config: AdaptiveConfig = serde_json::from_value(config)
            .map_err(|error| Error::invalid_argument(error.to_string()))?;
        let report = validate_adaptive_config_or_err(&config)?;
        let state = Arc::new(tokio::sync::Mutex::new(Some(
            NodeAdaptiveRuntimeState::Pending { config, report },
        )));
        let (commands, receiver) = tokio::sync::mpsc::unbounded_channel();
        within_runtime_if_available(|| {
            tokio::spawn(run_adaptive_lifecycle(Arc::clone(&state), receiver))
        });
        Ok(Self { state, commands })
    }

    /// Register all configured adaptive runtime features.
    ///
    /// `register()` and `shutdown()` calls run one at a time in call order.
    #[napi(ts_return_type = "Promise<void>", catch_unwind)]
    pub fn register<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        self.lifecycle(env, LifecycleCommand::Register)
    }

    /// Deregister all previously registered adaptive runtime features.
    #[napi(catch_unwind)]
    pub fn deregister(&self) -> Result<()> {
        match &mut *self.state()? {
            NodeAdaptiveRuntimeState::Pending { .. } => Ok(()),
            NodeAdaptiveRuntimeState::Ready(runtime) => runtime.deregister().map_err(Error::from),
        }
    }

    /// Shut down the adaptive runtime and consume its Rust runtime state.
    #[napi(ts_return_type = "Promise<void>", catch_unwind)]
    pub fn shutdown<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        self.lifecycle(env, LifecycleCommand::Shutdown)
    }

    /// Block until the telemetry drain has processed pending events.
    #[napi(js_name = "waitForIdle", catch_unwind)]
    pub fn wait_for_idle(&self) -> Result<()> {
        match &*self.state()? {
            NodeAdaptiveRuntimeState::Pending { .. } => Ok(()),
            NodeAdaptiveRuntimeState::Ready(runtime) => {
                runtime.wait_for_idle();
                Ok(())
            }
        }
    }

    /// Return the validation report captured during runtime construction.
    #[napi(ts_return_type = "ConfigReport", catch_unwind)]
    pub fn report(&self) -> Result<Json> {
        let state = self.state()?;
        let report = match &*state {
            NodeAdaptiveRuntimeState::Pending { report, .. } => report,
            NodeAdaptiveRuntimeState::Ready(runtime) => runtime.report(),
        };
        serde_json::to_value(report).map_err(Error::from)
    }

    /// Bind the runtime's ACG request rewrite to a scope.
    #[napi(js_name = "bindScope", catch_unwind)]
    pub fn bind_scope(&self, scope_handle: &ScopeHandle) -> Result<()> {
        self.ready("binding ACG request intercepts")?
            .bind_scope(scope_handle.inner.uuid)
            .map_err(Error::from)
    }

    /// Build cache request facts for an annotated LLM request.
    #[napi(
        js_name = "buildCacheRequestFacts",
        ts_generic_types = "TOptions extends JsonCompatible<TOptions>",
        ts_return_type = "CacheRequestFacts | null",
        catch_unwind
    )]
    pub fn build_cache_request_facts(
        &self,
        #[napi(ts_arg_type = "TOptions")] options: Json,
    ) -> Result<Option<Json>> {
        let options: CacheRequestFactsOptions = serde_json::from_value(options)
            .map_err(|error| Error::invalid_argument(error.to_string()))?;
        let _request_id = parse_uuid(&options.request_id, "requestId UUID")?;
        let _timestamp = parse_cache_telemetry_timestamp(options.timestamp.as_deref())?;
        let annotated_request: AnnotatedLlmRequest =
            parse_json(options.annotated_request, "annotatedRequest")?;
        self.ready("building cache request facts")?
            .build_cache_request_facts(&options.agent_id, &options.provider, &annotated_request)
            .map(|facts| serde_json::to_value(&facts))
            .transpose()
            .map_err(Error::from)
    }
}

fn validate_adaptive_config_or_err(
    config: &AdaptiveConfig,
) -> Result<nemo_relay_plugin_host::ConfigReport> {
    let report = CoreAdaptiveRuntime::validate_config(config);
    if report.has_errors() {
        let joined = report
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.level == DiagnosticLevel::Error)
            .map(|diagnostic| diagnostic.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(Error::invalid_argument(joined));
    }
    Ok(report)
}

fn parse_cache_telemetry_provider(provider: &str) -> Result<CacheTelemetryProvider> {
    match provider {
        "anthropic" => Ok(CacheTelemetryProvider::Anthropic),
        "openai" => Ok(CacheTelemetryProvider::OpenAI),
        other => Err(Error::invalid_argument(format!(
            "unsupported provider: {other}",
        ))),
    }
}

fn parse_cache_telemetry_timestamp(timestamp: Option<&str>) -> Result<DateTime<Utc>> {
    match timestamp {
        Some(value) => DateTime::parse_from_rfc3339(value)
            .map(|value| value.with_timezone(&Utc))
            .map_err(|error| Error::invalid_argument(format!("invalid timestamp: {error}"))),
        None => Ok(Utc::now()),
    }
}

fn parse_cache_request_facts(value: Option<Json>) -> Result<Option<CacheRequestFacts>> {
    value
        .map(|value| parse_json(value, "requestFacts"))
        .transpose()
}

/// Validate an adaptive config document without constructing a runtime.
#[napi(
    js_name = "validateAdaptiveConfig",
    ts_generic_types = "TConfig extends JsonCompatible<TConfig>",
    ts_return_type = "ConfigReport",
    catch_unwind
)]
pub fn validate_adaptive_config(#[napi(ts_arg_type = "TConfig")] config: Json) -> Result<Json> {
    let config: AdaptiveConfig = serde_json::from_value(config)
        .map_err(|error| Error::invalid_argument(error.to_string()))?;
    serde_json::to_value(CoreAdaptiveRuntime::validate_config(&config)).map_err(Error::from)
}

/// Build one adaptive cache telemetry event from normalized usage.
#[napi(
    js_name = "buildCacheTelemetryEvent",
    ts_generic_types = "TOptions extends JsonCompatible<TOptions>",
    ts_return_type = "CacheTelemetryEvent | null",
    catch_unwind
)]
pub fn build_cache_telemetry_event(
    #[napi(ts_arg_type = "TOptions")] options: Json,
) -> Result<Option<Json>> {
    let options: CacheTelemetryOptions = serde_json::from_value(options)
        .map_err(|error| Error::invalid_argument(error.to_string()))?;
    let provider = parse_cache_telemetry_provider(&options.provider)?;
    let request_id = parse_uuid(&options.request_id, "requestId UUID")?;
    let timestamp = parse_cache_telemetry_timestamp(options.timestamp.as_deref())?;
    let Some(usage_json) = options.usage else {
        return Ok(None);
    };
    let usage: Usage = parse_json(usage_json, "usage")?;
    let request_facts = parse_cache_request_facts(options.request_facts)?;
    let agent_identity = AgentIdentity {
        agent_id: options.agent_id,
        template_version: options.template_version,
        toolset_hash: options.toolset_hash,
        model_family: options.model_family,
        tenant_scope: options.tenant_scope,
    };
    CacheTelemetryEvent::from_usage(
        request_id,
        agent_identity,
        provider,
        &usage,
        timestamp,
        request_facts.as_ref(),
    )
    .map(|event| serde_json::to_value(&event))
    .transpose()
    .map_err(Error::from)
}

/// Set manual latency sensitivity on the current scope.
#[napi(js_name = "setLatencySensitivity", catch_unwind)]
pub fn set_latency_sensitivity(value: u32) -> Result<()> {
    if value == 0 {
        return Err(Error::invalid_argument(
            "sensitivity must be positive (> 0)",
        ));
    }
    adaptive_set_latency_sensitivity(value).map_err(Error::generic_failure)
}

/// Register a plugin backed by JavaScript callbacks.
///
/// `validate` receives `(pluginConfig)` and should return a diagnostics array.
/// `register` receives `(pluginConfig, context)` and should use the context methods
/// to attach subscribers or intercepts. Both callbacks must be synchronous.
#[napi(catch_unwind)]
pub fn register_plugin(
    env: Env,
    plugin_kind: String,
    #[napi(
        ts_arg_type = "((pluginConfig: JsonObject) => ConfigDiagnostic[] | null | undefined) | undefined | null"
    )]
    validate: Option<Callback>,
    #[napi(ts_arg_type = "(pluginConfig: JsonObject, context: PluginContext) => void")]
    register: Callback,
) -> Result<()> {
    let validate = validate
        .map(|func| {
            callable::SyncJsCallback::new(
                &env,
                callback_factory::wrap(&env, Wrapper::Middleware, &func)?,
                PLUGIN_VALIDATE_PREFIX,
                None,
            )
        })
        .transpose()?;
    let register: Function<JsArgs2, JsArg> =
        callback_factory::wrap(&env, Wrapper::Middleware, &register)?;
    let register = callable::reply_tsfn(&env, &register, PLUGIN_REGISTER_PREFIX)?;

    register_plugin_impl(Arc::new(NodePlugin {
        plugin_kind,
        validate,
        register,
    }))
    .map_err(Error::from)
}

/// Deregister a plugin by kind.
#[napi(catch_unwind)]
pub fn deregister_plugin(plugin_kind: String) -> bool {
    deregister_plugin_impl(&plugin_kind)
}

/// Owned dynamic plugin activation.
///
/// Keep this object alive while code may invoke callbacks registered by the
/// dynamic plugins. Call `close()` for deterministic cleanup; garbage
/// collection performs the same cleanup as a defensive fallback.
#[napi]
pub struct PluginHostActivation {
    close_state: Arc<DynamicPluginCloseState>,
    teardown: std::sync::mpsc::Sender<Box<CorePluginHostActivation>>,
}

type DynamicPluginTeardownResult = std::result::Result<(), Arc<Error>>;

enum DynamicPluginTeardownTrigger {
    Close,
    Finalizer,
}

fn run_dynamic_plugin_teardown(
    state: Arc<DynamicPluginCloseState>,
    requests: std::sync::mpsc::Receiver<Box<CorePluginHostActivation>>,
) {
    while let Ok(activation) = requests.recv() {
        if !state.close(activation, DynamicPluginTeardownTrigger::Close) {
            return;
        }
    }
    if let Some(activation) = state.take_active() {
        state.close(activation, DynamicPluginTeardownTrigger::Finalizer);
    }
}

enum DynamicPluginCloseStatus {
    Active(Box<CorePluginHostActivation>),
    Closing,
    Closed,
}

struct DynamicPluginCloseState {
    status: StdMutex<DynamicPluginCloseStatus>,
    report: StdMutex<PluginHostReport>,
    completion: tokio::sync::watch::Sender<Option<DynamicPluginTeardownResult>>,
}

impl DynamicPluginCloseState {
    fn new(activation: CorePluginHostActivation) -> Self {
        let (completion, _) = tokio::sync::watch::channel(None);
        let report = activation.report();
        Self {
            status: StdMutex::new(DynamicPluginCloseStatus::Active(Box::new(activation))),
            report: StdMutex::new(report),
            completion,
        }
    }

    fn report(&self) -> PluginHostReport {
        let latest = match &*lock(&self.status) {
            DynamicPluginCloseStatus::Active(activation) => Some(activation.report()),
            DynamicPluginCloseStatus::Closing | DynamicPluginCloseStatus::Closed => None,
        };
        let mut report = lock(&self.report);
        if let Some(latest) = latest {
            *report = latest;
        }
        report.clone()
    }

    fn active(&self) -> bool {
        matches!(&*lock(&self.status), DynamicPluginCloseStatus::Active(_))
    }

    fn take_active(&self) -> Option<Box<CorePluginHostActivation>> {
        let mut status = lock(&self.status);
        let activation = match std::mem::replace(&mut *status, DynamicPluginCloseStatus::Closing) {
            DynamicPluginCloseStatus::Active(activation) => activation,
            other => {
                *status = other;
                return None;
            }
        };
        *lock(&self.report) = activation.report();
        self.completion.send_replace(None);
        Some(activation)
    }

    fn request_close(&self, teardown: &std::sync::mpsc::Sender<Box<CorePluginHostActivation>>) {
        if let Some(activation) = self.take_active() {
            teardown
                .send(activation)
                .expect("plugin teardown thread outlives its handles");
        }
    }

    fn close(
        &self,
        mut activation: Box<CorePluginHostActivation>,
        trigger: DynamicPluginTeardownTrigger,
    ) -> bool {
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = activation.close();
            (result, activation.report())
        })) {
            Ok((result, report)) => {
                *lock(&self.report) = report;
                result.map_err(Error::from)
            }
            Err(_) => Err(Error::generic_failure("plugin host teardown task panicked")),
        }
        .map_err(Arc::new);
        if matches!(trigger, DynamicPluginTeardownTrigger::Finalizer)
            && let Err(error) = &result
        {
            eprintln!("nemo_relay: plugin host finalizer teardown failed: {error}");
        }
        self.finish(activation, result)
    }

    fn finish(
        &self,
        activation: Box<CorePluginHostActivation>,
        result: DynamicPluginTeardownResult,
    ) -> bool {
        self.finish_with_hook(activation, result, || {})
    }

    fn finish_with_hook(
        &self,
        activation: Box<CorePluginHostActivation>,
        result: DynamicPluginTeardownResult,
        before_publish: impl FnOnce(),
    ) -> bool {
        let mut status = lock(&self.status);
        let retryable = result.is_err() && activation.is_active();
        *status = if retryable {
            DynamicPluginCloseStatus::Active(activation)
        } else {
            DynamicPluginCloseStatus::Closed
        };
        before_publish();
        self.completion.send_replace(Some(result));
        retryable
    }

    async fn wait_for_close(&self) -> DynamicPluginTeardownResult {
        settled(self.completion.subscribe())
            .await
            .unwrap_or_else(|| {
                Err(Arc::new(Error::generic_failure(
                    "dynamic plugin teardown result channel closed unexpectedly",
                )))
            })
    }
}

#[cfg(test)]
#[path = "../../tests/rust/api/dynamic_plugin_close_state_tests.rs"]
mod dynamic_plugin_close_state_tests;

#[napi]
impl PluginHostActivation {
    /// Return the validation report produced by activation.
    #[napi(getter, ts_return_type = "PluginHostReport", catch_unwind)]
    pub fn report(&self) -> Result<Json> {
        serde_json::to_value(self.close_state.report()).map_err(Error::from)
    }

    /// Return whether this activation handle has not begun teardown.
    ///
    /// Failed teardown leaves the activation active so `close()` can retry.
    #[napi(getter, catch_unwind)]
    pub fn is_active(&self) -> Result<bool> {
        Ok(self.close_state.active())
    }

    /// Clear plugin callbacks before unloading libraries and workers.
    ///
    /// This method is idempotent, including when concurrent callers race to
    /// close the same activation.
    #[napi(ts_return_type = "Promise<void>", catch_unwind)]
    pub fn close<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        self.close_state.request_close(&self.teardown);
        let close_state = Arc::clone(&self.close_state);
        spawn_promise(
            env,
            async move {
                close_state
                    .wait_for_close()
                    .await
                    .map_err(|error| Error::Napi(error.to_napi()))
            },
            |env, _| js_value(env, ()),
        )
    }

    /// Delegate structured `await using` cleanup to `close()`.
    #[napi(
        well_known_symbol = "asyncDispose",
        ts_return_type = "Promise<void>",
        catch_unwind
    )]
    pub fn async_dispose<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        self.close(env)
    }
}

/// Initialize the core-owned static and dynamic plugin host from layered files.
#[napi(
    ts_generic_types = "TConfig extends JsonCompatible<TConfig>",
    catch_unwind
)]
pub fn initialize<'env>(
    env: &'env Env,
    #[napi(ts_arg_type = "TConfig")] config: Json,
    additional_plugins_toml: Option<String>,
) -> Result<PromiseRaw<'env, PluginHostActivation>> {
    spawn_promise(
        env,
        async move {
            let config: nemo_relay_plugin_host::dynamic::PluginHostConfig =
                parse_json(config, "plugin config")?;
            let activation = initialize_impl(config, additional_plugins_toml.map(Into::into))
                .await
                .map_err(Error::from)?;
            let close_state = Arc::new(DynamicPluginCloseState::new(activation));
            let (teardown, requests) = std::sync::mpsc::channel();
            let worker_state = Arc::clone(&close_state);
            std::thread::Builder::new()
                .name("nemo-relay-node-plugin-teardown".into())
                .spawn(move || run_dynamic_plugin_teardown(worker_state, requests))
                .map_err(|error| {
                    Error::generic_failure(format!(
                        "failed to spawn plugin teardown thread: {error}"
                    ))
                })?;
            Ok::<_, Error>(PluginHostActivation {
                close_state,
                teardown,
            })
        },
        js_value,
    )
}
/// Validate dynamic plugins without loading any plugin code.
#[napi(
    ts_generic_types = "TConfig extends JsonCompatible<TConfig>",
    ts_return_type = "PluginHostReport",
    catch_unwind
)]
pub fn validate(
    #[napi(ts_arg_type = "TConfig")] config: Json,
    additional_plugins_toml: Option<String>,
) -> Result<Json> {
    let config: nemo_relay_plugin_host::dynamic::PluginHostConfig =
        parse_json(config, "plugin config")?;
    let report =
        validate_impl(config, additional_plugins_toml.map(Into::into)).map_err(Error::from)?;
    serde_json::to_value(report).map_err(Error::from)
}

/// Validate only the supplied static plugin configuration.
#[napi(
    ts_generic_types = "TConfig extends JsonCompatible<TConfig>",
    ts_return_type = "PluginHostReport",
    catch_unwind
)]
pub fn validate_exact(#[napi(ts_arg_type = "TConfig")] config: Json) -> Result<Json> {
    if config.get("diagnostics").is_some() {
        return Err(Error::invalid_argument(
            "diagnostics is a host-only setting; use layered validation",
        ));
    }
    let config: PluginConfig = parse_json(config, "plugin config")?;
    serde_json::to_value(validate_exact_impl(config)).map_err(Error::from)
}

/// List registered plugin kinds.
#[napi(catch_unwind)]
pub fn list_plugin_kinds() -> Vec<String> {
    list_plugin_kinds_impl()
}
