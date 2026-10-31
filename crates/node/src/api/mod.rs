// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public NAPI API functions for the NeMo Relay Node.js bindings.
//!
//! This module exposes the full agent runtime API to JavaScript/TypeScript:
//! scope stack management, tool and LLM lifecycle operations, guardrail and
//! intercept registration/deregistration, and event subscriber management.
//! All functions are annotated with `#[napi]` and their doc comments appear
//! in the generated `index.d.ts` TypeScript definitions.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use chrono::{DateTime, Utc};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeCallContext, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value as Json;

use nemo_relay::api::llm as core_llm_api;
use nemo_relay::api::llm::{LlmAttributes, LlmRequest};
use nemo_relay::api::registry as core_registry_api;
use nemo_relay::api::resource_metrics as core_resource_metrics_api;
use nemo_relay::api::resource_metrics::{
    AcceleratorDeviceMetrics, AcceleratorProcessMetrics, BandwidthUnit, CapacityUnit, CountUnit,
    CpuMetrics, CpuUnit, DataUnit, DiskMetrics, DurationUnit, FilesystemCapacityMetrics,
    GpuMetrics, MemoryMetrics, NetworkInterfaceMetrics, NetworkMetrics, NetworkTrafficMetrics,
    ProcessMetrics, ProcessSamplingMetadata as CoreProcessSamplingMetadata,
    ResourceLimitEventCount, ResourceLimitEventKind, ResourceLimitResource, ResourceMeasurement,
    ResourceMeasurementScope, ResourceMetricValue,
    ResourceMetricsSnapshot as CoreResourceMetricsSnapshot, ResourceOperatingSystem, ResourceUnit,
    UtilizationUnit,
};
use nemo_relay::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, with_nested_publication_buffer,
};
use nemo_relay::api::runtime::{
    EventMetadataInjectorFn, EventSanitizeFn, LlmCollector, LlmExecutionNextFn, LlmFinalizer,
    LlmJsonStream, ToolExecutionNextFn,
};
use nemo_relay::api::runtime::{
    TASK_SCOPE_STACK, capture_propagation_context as capture_propagation_context_handle,
    capture_propagation_context_with_root as capture_propagation_context_with_root_handle,
    capture_rootless_propagation_context as capture_rootless_propagation_context_handle,
    capture_traceparent as capture_traceparent_handle,
    create_scope_stack as create_scope_stack_handle,
    create_scope_stack_from_propagation as create_scope_stack_from_propagation_handle,
    current_scope_stack as current_scope_stack_handle, scope_stack_active as scope_stack_is_active,
    set_thread_scope_stack as bind_thread_scope_stack, task_scope_top,
    with_scope_stack as with_scope_stack_handle,
};
use nemo_relay::api::scope as core_scope_api;
use nemo_relay::api::scope::ScopeAttributes;
use nemo_relay::api::subscriber as core_subscriber_api;
use nemo_relay::api::tool as core_tool_api;
use nemo_relay::api::tool::ToolAttributes;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::response::Usage;
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::plugin::dynamic::{
    PluginHostActivation as CorePluginHostActivation, PluginHostReport,
    initialize as initialize_impl, validate as validate_impl,
    validate_exact as validate_exact_impl,
};
use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginConfig, PluginError, PluginRegistration,
    PluginRegistrationContext, deregister_plugin as deregister_plugin_impl,
    list_plugin_kinds as list_plugin_kinds_impl, register_plugin as register_plugin_impl,
    rollback_registrations,
};
use nemo_relay::shared_runtime::initialize_shared_runtime_binding;
use nemo_relay_adaptive::acg::{
    AgentIdentity, CacheRequestFacts, CacheTelemetryEvent, CacheTelemetryProvider,
};
use nemo_relay_adaptive::context_helpers::set_latency_sensitivity as adaptive_set_latency_sensitivity;
use nemo_relay_adaptive::plugin_component::register_adaptive_component;
use nemo_relay_adaptive::{AdaptiveConfig, AdaptiveRuntime as CoreAdaptiveRuntime};
use nemo_relay_pii_redaction::component::register_pii_redaction_component;

pub(crate) const LLM_STREAM_BRIDGE_CAPACITY: usize = 32;

use crate::callable;
use crate::callback_factory::{self, CallbackScope};
use crate::convert::{
    Callback, JsArg, JsArgs2, PersistentJsFunction, Tsfn, abort_tsfn,
    clear_last_callback_error as clear_recorded_callback_error, closure_function, detach,
    get_last_callback_error as get_recorded_callback_error, js_value, opt_json,
    parse_timestamp_micros, record_callback_error,
};
use crate::environment::spawn_promise;
use crate::error::{Error, Result};
use crate::promise_call::PromiseAwareFn;
use crate::promise_call::with_publication_callback_context;
use crate::stream::LlmStream;
use crate::types::{
    DataSchema, LlmHandle, LogSeverity, MetricMeasurement, MetricTemporality, ScopeHandle,
    ScopeStack, ScopeType, ToolExecutionResult, ToolHandle,
};

#[cfg(test)]
mod llm_stack_tests;

static NODE_ENVIRONMENT_COUNT: AtomicUsize = AtomicUsize::new(0);
static NODE_ENVIRONMENT_LIFECYCLE_LOCK: StdMutex<()> = StdMutex::new(());

fn register_node_environment() -> FlowResult<()> {
    let _guard = NODE_ENVIRONMENT_LIFECYCLE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    nemo_relay::logging::initialize_default_logging()?;
    NODE_ENVIRONMENT_COUNT.fetch_add(1, Ordering::AcqRel);
    Ok(())
}

fn cleanup_node_environment() {
    let _guard = NODE_ENVIRONMENT_LIFECYCLE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if NODE_ENVIRONMENT_COUNT.fetch_sub(1, Ordering::AcqRel) == 1
        && let Err(error) = nemo_relay::logging::shutdown_default_logging()
    {
        eprintln!("nemo-relay: operational logging shutdown failed: {error}");
    }
}

/// Emit a structured operational log record through Relay's configured sinks.
#[napi(catch_unwind)]
pub fn log(
    level: String,
    message: String,
    target: Option<String>,
    fields: Option<Json>,
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
    nemo_relay::logging::emit_str(&level, &target, &message, fields).map_err(Error::from)
}

#[napi(catch_unwind)]
pub fn trace(message: String, target: Option<String>, fields: Option<Json>) -> Result<()> {
    log("trace".into(), message, target, fields)
}
#[napi(catch_unwind)]
pub fn debug(message: String, target: Option<String>, fields: Option<Json>) -> Result<()> {
    log("debug".into(), message, target, fields)
}
#[napi(catch_unwind)]
pub fn info(message: String, target: Option<String>, fields: Option<Json>) -> Result<()> {
    log("info".into(), message, target, fields)
}
#[napi(catch_unwind)]
pub fn warn(message: String, target: Option<String>, fields: Option<Json>) -> Result<()> {
    log("warn".into(), message, target, fields)
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
                env.to_js_value(&NodeResourceMetricsSnapshot::from(snapshot))?,
            )
        },
    )
}

const JAVASCRIPT_MAX_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;

#[derive(Debug, Clone, Copy, PartialEq)]
enum NodeResourceMetricInteger {
    Number(f64),
    BigInt(u64),
}

impl From<u64> for NodeResourceMetricInteger {
    fn from(value: u64) -> Self {
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

impl<U: ResourceUnit> From<ResourceMeasurement<ResourceMetricValue, U>>
    for NodeIntegerResourceMeasurement<U>
{
    fn from(measurement: ResourceMeasurement<ResourceMetricValue, U>) -> Self {
        Self {
            value: match measurement.value {
                ResourceMetricValue::Integer(value) => NodeResourceMetricInteger::from(value),
                ResourceMetricValue::Decimal(value) => NodeResourceMetricInteger::Number(value),
            },
            unit: measurement.unit,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeResourceLimitEventCount {
    resource: ResourceLimitResource,
    event: ResourceLimitEventKind,
    count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
}

impl From<ResourceLimitEventCount> for NodeResourceLimitEventCount {
    fn from(event: ResourceLimitEventCount) -> Self {
        Self {
            resource: event.resource,
            event: event.event,
            count: event.count.map(Into::into),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeAcceleratorDeviceMetrics {
    vendor: nemo_relay::api::resource_metrics::AcceleratorVendor,
    device_identifier: String,
    device_index: Option<u32>,
    memory_used: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
    compute_utilization: Option<ResourceMeasurement<f64, UtilizationUnit>>,
}

impl From<AcceleratorDeviceMetrics> for NodeAcceleratorDeviceMetrics {
    fn from(device: AcceleratorDeviceMetrics) -> Self {
        Self {
            vendor: device.vendor,
            device_identifier: device.device_identifier,
            device_index: device.device_index,
            memory_used: device.memory_used.map(Into::into),
            compute_utilization: device.compute_utilization,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeAcceleratorProcessMetrics {
    vendor: nemo_relay::api::resource_metrics::AcceleratorVendor,
    device_identifier: String,
    device_index: Option<u32>,
    process_id: u32,
    memory_used: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
    compute_utilization: Option<ResourceMeasurement<f64, UtilizationUnit>>,
}

impl From<AcceleratorProcessMetrics> for NodeAcceleratorProcessMetrics {
    fn from(process: AcceleratorProcessMetrics) -> Self {
        Self {
            vendor: process.vendor,
            device_identifier: process.device_identifier,
            device_index: process.device_index,
            process_id: process.process_id,
            memory_used: process.memory_used.map(Into::into),
            compute_utilization: process.compute_utilization,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeFilesystemCapacityMetrics {
    path: String,
    total_capacity: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
    available_capacity: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
    free_capacity: Option<NodeIntegerResourceMeasurement<CapacityUnit>>,
}

impl From<FilesystemCapacityMetrics> for NodeFilesystemCapacityMetrics {
    fn from(filesystem: FilesystemCapacityMetrics) -> Self {
        Self {
            path: filesystem.path,
            total_capacity: filesystem.total_capacity.map(Into::into),
            available_capacity: filesystem.available_capacity.map(Into::into),
            free_capacity: filesystem.free_capacity.map(Into::into),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeCpuMetrics {
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

impl From<CpuMetrics> for NodeCpuMetrics {
    fn from(cpu: CpuMetrics) -> Self {
        Self {
            user_time: cpu.user_time.map(Into::into),
            system_time: cpu.system_time.map(Into::into),
            total_time: cpu.total_time.map(Into::into),
            consumption_rate: cpu.consumption_rate,
            throttled_time: cpu.throttled_time.map(Into::into),
            effective_limit: cpu.effective_limit,
            some_pressure_stall_time: cpu.some_pressure_stall_time.map(Into::into),
            full_pressure_stall_time: cpu.full_pressure_stall_time.map(Into::into),
            limit_events: cpu.limit_events.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeMemoryMetrics {
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

impl From<MemoryMetrics> for NodeMemoryMetrics {
    fn from(memory: MemoryMetrics) -> Self {
        Self {
            system_used: memory.system_used.map(Into::into),
            system_total: memory.system_total.map(Into::into),
            system_available: memory.system_available.map(Into::into),
            resident: memory.resident.map(Into::into),
            private: memory.private.map(Into::into),
            physical_footprint: memory.physical_footprint.map(Into::into),
            virtual_memory: memory.virtual_memory.map(Into::into),
            peak_resident: memory.peak_resident.map(Into::into),
            limit: memory.limit.map(Into::into),
            environment_accounted: memory.environment_accounted.map(Into::into),
            some_pressure_stall_time: memory.some_pressure_stall_time.map(Into::into),
            full_pressure_stall_time: memory.full_pressure_stall_time.map(Into::into),
            out_of_memory_event_count: memory.out_of_memory_event_count.map(Into::into),
            limit_events: memory.limit_events.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeProcessMetrics {
    active_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    descendant_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    thread_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    lifetime_creation_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    open_file_descriptor_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    windows_handle_count: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    limit_events: Vec<NodeResourceLimitEventCount>,
}

impl From<ProcessMetrics> for NodeProcessMetrics {
    fn from(process: ProcessMetrics) -> Self {
        Self {
            active_count: process.active_count.map(Into::into),
            descendant_count: process.descendant_count.map(Into::into),
            thread_count: process.thread_count.map(Into::into),
            lifetime_creation_count: process.lifetime_creation_count.map(Into::into),
            open_file_descriptor_count: process.open_file_descriptor_count.map(Into::into),
            windows_handle_count: process.windows_handle_count.map(Into::into),
            limit_events: process.limit_events.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeDiskMetrics {
    read_data: Option<NodeIntegerResourceMeasurement<DataUnit>>,
    write_data: Option<NodeIntegerResourceMeasurement<DataUnit>>,
    read_throughput: Option<ResourceMeasurement<f64, BandwidthUnit>>,
    write_throughput: Option<ResourceMeasurement<f64, BandwidthUnit>>,
    read_operations: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    write_operations: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    filesystems: Vec<NodeFilesystemCapacityMetrics>,
}

impl From<DiskMetrics> for NodeDiskMetrics {
    fn from(disk: DiskMetrics) -> Self {
        Self {
            read_data: disk.read_data.map(Into::into),
            write_data: disk.write_data.map(Into::into),
            read_throughput: disk.read_throughput,
            write_throughput: disk.write_throughput,
            read_operations: disk.read_operations.map(Into::into),
            write_operations: disk.write_operations.map(Into::into),
            filesystems: disk.filesystems.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeGpuMetrics {
    device_metrics: Option<Vec<NodeAcceleratorDeviceMetrics>>,
    process_metrics: Option<Vec<NodeAcceleratorProcessMetrics>>,
}

impl From<GpuMetrics> for NodeGpuMetrics {
    fn from(gpu: GpuMetrics) -> Self {
        Self {
            device_metrics: gpu
                .device_metrics
                .map(|devices| devices.into_iter().map(Into::into).collect()),
            process_metrics: gpu
                .process_metrics
                .map(|processes| processes.into_iter().map(Into::into).collect()),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeNetworkTrafficMetrics {
    received_data: Option<NodeIntegerResourceMeasurement<DataUnit>>,
    transmitted_data: Option<NodeIntegerResourceMeasurement<DataUnit>>,
    receive_throughput: Option<ResourceMeasurement<f64, BandwidthUnit>>,
    transmit_throughput: Option<ResourceMeasurement<f64, BandwidthUnit>>,
    received_packets: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    transmitted_packets: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    receive_errors: Option<NodeIntegerResourceMeasurement<CountUnit>>,
    transmit_errors: Option<NodeIntegerResourceMeasurement<CountUnit>>,
}
impl From<NetworkTrafficMetrics> for NodeNetworkTrafficMetrics {
    fn from(traffic: NetworkTrafficMetrics) -> Self {
        Self {
            received_data: traffic.received_data.map(Into::into),
            transmitted_data: traffic.transmitted_data.map(Into::into),
            receive_throughput: traffic.receive_throughput,
            transmit_throughput: traffic.transmit_throughput,
            received_packets: traffic.received_packets.map(Into::into),
            transmitted_packets: traffic.transmitted_packets.map(Into::into),
            receive_errors: traffic.receive_errors.map(Into::into),
            transmit_errors: traffic.transmit_errors.map(Into::into),
        }
    }
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeNetworkInterfaceMetrics {
    name: String,
    traffic: NodeNetworkTrafficMetrics,
}
impl From<NetworkInterfaceMetrics> for NodeNetworkInterfaceMetrics {
    fn from(interface: NetworkInterfaceMetrics) -> Self {
        Self {
            name: interface.name,
            traffic: interface.traffic.into(),
        }
    }
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeNetworkMetrics {
    measurement_scope: ResourceMeasurementScope,
    system: NodeNetworkTrafficMetrics,
    interfaces: Vec<NodeNetworkInterfaceMetrics>,
}
impl From<NetworkMetrics> for NodeNetworkMetrics {
    fn from(network: NetworkMetrics) -> Self {
        Self {
            measurement_scope: network.measurement_scope,
            system: network.system.into(),
            interfaces: network.interfaces.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeProcessSamplingMetadata {
    visible_processes: NodeResourceMetricInteger,
    sampled_processes: NodeResourceMetricInteger,
    field_sampled_processes: BTreeMap<String, NodeResourceMetricInteger>,
}

impl From<CoreProcessSamplingMetadata> for NodeProcessSamplingMetadata {
    fn from(metadata: CoreProcessSamplingMetadata) -> Self {
        Self {
            visible_processes: metadata.visible_processes.into(),
            sampled_processes: metadata.sampled_processes.into(),
            field_sampled_processes: metadata
                .field_sampled_processes
                .into_iter()
                .map(|(field, count)| (field, count.into()))
                .collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeResourceMetricsSnapshot {
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

impl From<CoreResourceMetricsSnapshot> for NodeResourceMetricsSnapshot {
    fn from(snapshot: CoreResourceMetricsSnapshot) -> Self {
        Self {
            timestamp: snapshot.timestamp,
            operating_system: snapshot.operating_system,
            measurement_scope: snapshot.measurement_scope,
            process_sampling: snapshot.process_sampling.map(Into::into),
            cpu: snapshot.cpu.map(Into::into),
            memory: snapshot.memory.map(Into::into),
            process: snapshot.process.map(Into::into),
            disk: snapshot.disk.map(Into::into),
            gpu: snapshot.gpu.map(Into::into),
            network: snapshot.network.map(Into::into),
        }
    }
}
#[napi(catch_unwind)]
pub fn error(message: String, target: Option<String>, fields: Option<Json>) -> Result<()> {
    log("error".into(), message, target, fields)
}

fn effective_scope(env: &Env) -> Result<CallbackScope> {
    Ok(
        callback_factory::callback_scope(env)?.unwrap_or_else(|| CallbackScope {
            scope_stack: current_scope_stack_handle(),
            publication_buffer: None,
            propagation: None,
        }),
    )
}

fn effective_scope_context(
    env: &Env,
) -> Result<(
    nemo_relay::api::runtime::ScopeStackHandle,
    Option<PublicationBuffer>,
)> {
    effective_scope(env).map(|scope| (scope.scope_stack, scope.publication_buffer))
}

fn with_callback_scope<T>(scope: &CallbackScope, callback: impl FnOnce() -> T) -> T {
    with_scope_stack_handle(scope.scope_stack.clone(), || {
        with_nested_publication_buffer(scope.publication_buffer.clone(), callback)
    })
}

fn with_effective_scope_stack<T>(env: &Env, callback: impl FnOnce() -> T) -> Result<T> {
    Ok(with_callback_scope(&effective_scope(env)?, callback))
}

fn effective_scope_top(
    scope_stack: &nemo_relay::api::runtime::ScopeStackHandle,
) -> nemo_relay::api::scope::ScopeHandle {
    with_scope_stack_handle(scope_stack.clone(), task_scope_top)
}

#[napi_derive::module_init]
fn init() {
    initialize_shared_runtime_binding("node")
        .expect("node runtime ownership initialization should succeed");
    register_adaptive_component()
        .expect("node adaptive plugin component registration should succeed");
    register_pii_redaction_component()
        .expect("node pii redaction plugin component registration should succeed");
}

#[cfg(not(test))]
#[napi(module_exports)]
fn initialize_node_environment(env: Env) -> napi::Result<()> {
    register_node_environment().map_err(|error| napi::Error::from_reason(error.to_string()))?;
    if let Err(error) = env.add_env_cleanup_hook((), |_| cleanup_node_environment()) {
        cleanup_node_environment();
        return Err(error);
    }
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

fn parse_otel_type(value: &str) -> Result<nemo_relay::observability::OpenTelemetryType> {
    match value {
        "full" => Ok(nemo_relay::observability::OpenTelemetryType::Full),
        "gen_ai" => Ok(nemo_relay::observability::OpenTelemetryType::GenAi),
        "openinference" => Ok(nemo_relay::observability::OpenTelemetryType::OpenInference),
        other => Err(Error::invalid_argument(format!(
            "type must be 'full', 'gen_ai', or 'openinference', got {other:?}",
        ))),
    }
}

fn parse_otel_transport(
    value: Option<String>,
) -> Result<nemo_relay::observability::otel::OtlpTransport> {
    match value.as_deref().unwrap_or("http_binary") {
        "http_binary" => Ok(nemo_relay::observability::otel::OtlpTransport::HttpBinary),
        "grpc" => Ok(nemo_relay::observability::otel::OtlpTransport::Grpc),
        other => Err(Error::invalid_argument(format!(
            "transport must be 'http_binary' or 'grpc', got {other:?}",
        ))),
    }
}

fn parse_mark_projection(
    value: Option<String>,
) -> Result<nemo_relay::observability::MarkProjection> {
    serde_json::from_value(Json::String(value.unwrap_or_else(|| "inherit".to_string())))
        .map_err(|error| Error::invalid_argument(error.to_string()))
}

fn parse_attribute_mappings(
    value: Option<Json>,
) -> Result<Vec<nemo_relay::observability::OtlpAttributeMapping>> {
    let mappings = match value {
        Some(value) => serde_json::from_value(value)
            .map_err(|error| Error::invalid_argument(format!("attributeMappings: {error}")))?,
        None => Vec::new(),
    };
    nemo_relay::observability::validate_attribute_mappings(&mappings)
        .map_err(Error::invalid_argument)?;
    Ok(mappings)
}

fn parse_otel_file_sink(
    output_directory: &str,
    filename: Option<&str>,
    format: Option<&str>,
    mode: Option<&str>,
) -> Result<nemo_relay::observability::otel::OtlpFileSinkSettings> {
    nemo_relay::observability::otel::OtlpFileSinkSettings::from_parts(
        output_directory,
        filename,
        format,
        mode,
    )
    .map_err(Error::invalid_argument)
}

fn build_otel_config(
    options: OpenTelemetryConfig,
) -> Result<nemo_relay::observability::otel::OpenTelemetryConfig> {
    let otel_type = parse_otel_type(&options.r#type)?;
    let endpoint = options.endpoint.trim().to_string();
    if endpoint.is_empty() {
        return Err(Error::invalid_argument(
            "endpoint must be a nonblank string",
        ));
    }
    let transport = parse_otel_transport(options.transport)?;
    let service_name = options
        .service_name
        .unwrap_or_else(|| "unknown_service".to_string());
    let instrumentation_scope = options
        .instrumentation_scope
        .unwrap_or_else(|| "opentelemetry".to_string());
    let timeout_millis = options.timeout_millis.unwrap_or(3_000);
    let completed_span_context_ttl_millis =
        parse_completed_span_context_ttl_millis(options.completed_span_context_ttl_millis)?;

    let mut config = nemo_relay::observability::otel::OpenTelemetryConfig::new(otel_type, endpoint)
        .with_transport(transport)
        .with_timeout(std::time::Duration::from_millis(timeout_millis.into()))
        .with_service_name(service_name)
        .with_instrumentation_scope(instrumentation_scope)
        .with_completed_span_context_ttl(std::time::Duration::from_millis(
            completed_span_context_ttl_millis,
        ));

    if let Some(namespace) = options.service_namespace {
        config = config.with_service_namespace(namespace);
    }
    if let Some(version) = options.service_version {
        config = config.with_service_version(version);
    }
    for (key, value) in parse_string_map(options.headers, "headers")? {
        config = config.with_header(key, value);
    }
    for (key, variable) in parse_string_map(options.header_env, "headerEnv")? {
        config = config.with_header_env(key, variable);
    }
    if let Some(tls) = options.tls {
        config = config.with_tls(tls.into());
    }
    if let Some(allow_insecure) = options.allow_insecure {
        config = config.with_allow_insecure(allow_insecure);
    }
    for (key, value) in parse_string_map(options.resource_attributes, "resourceAttributes")? {
        config = config.with_resource_attribute(key, value);
    }
    config = config
        .with_mark_projection(parse_mark_projection(options.mark_projection)?)
        .with_mark_exclude_names(
            options
                .mark_exclude_names
                .unwrap_or_else(nemo_relay::observability::default_mark_exclude_names),
        )
        .with_attribute_mappings(parse_attribute_mappings(options.attribute_mappings)?)
        .with_promote_metadata_prefixes(options.promote_metadata_prefixes.unwrap_or_default())
        .with_promote_resource_metadata_prefixes(
            options
                .promote_resource_metadata_prefixes
                .unwrap_or_default(),
        );
    Ok(config)
}

/// Parses the optional completed-span-context TTL shared by every trace config.
fn parse_completed_span_context_ttl_millis(value: Option<BigInt>) -> Result<u64> {
    let parsed = value
        .map(|ttl| {
            let (negative, value, lossless) = ttl.get_u64();
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
            Ok(value)
        })
        .transpose()?
        .unwrap_or_else(|| {
            u64::try_from(
                nemo_relay::observability::otel::DEFAULT_COMPLETED_SPAN_CONTEXT_TTL.as_millis(),
            )
            .expect("the default completed span context TTL fits in u64 milliseconds")
        });
    Ok(parsed)
}

fn build_otel_log_config(
    options: OpenTelemetryLogConfig,
) -> Result<nemo_relay::observability::otel_logs::OpenTelemetryLogConfig> {
    let endpoint = options.endpoint.trim().to_string();
    if endpoint.is_empty() {
        return Err(Error::invalid_argument(
            "endpoint must be a nonblank string",
        ));
    }
    let completed_span_context_ttl_millis =
        parse_completed_span_context_ttl_millis(options.completed_span_context_ttl_millis)?;
    let mut config = nemo_relay::observability::otel_logs::OpenTelemetryLogConfig::new(endpoint)
        .with_transport(parse_otel_transport(options.transport)?)
        .with_service_name(
            options
                .service_name
                .unwrap_or_else(|| "unknown_service".to_string()),
        )
        .with_instrumentation_scope(
            options
                .instrumentation_scope
                .unwrap_or_else(|| "opentelemetry".to_string()),
        )
        .with_timeout(std::time::Duration::from_millis(
            options.timeout_millis.unwrap_or(3_000).into(),
        ))
        .with_max_queue_size(options.max_queue_size.unwrap_or(2_048) as usize)
        .with_max_export_batch_size(options.max_export_batch_size.unwrap_or(512) as usize)
        .with_scheduled_delay(std::time::Duration::from_millis(
            options.scheduled_delay_millis.unwrap_or(1_000).into(),
        ))
        .with_completed_span_context_ttl(std::time::Duration::from_millis(
            completed_span_context_ttl_millis,
        ));
    if let Some(severity) = options.minimum_severity {
        config = config.with_minimum_severity(severity.into());
    }
    if let Some(namespace) = options.service_namespace {
        config = config.with_service_namespace(namespace);
    }
    if let Some(version) = options.service_version {
        config = config.with_service_version(version);
    }
    for (key, value) in parse_string_map(options.headers, "headers")? {
        config = config.with_header(key, value);
    }
    for (key, variable) in parse_string_map(options.header_env, "headerEnv")? {
        config = config.with_header_env(key, variable);
    }
    if let Some(tls) = options.tls {
        config = config.with_tls(tls.into());
    }
    if let Some(allow_insecure) = options.allow_insecure {
        config = config.with_allow_insecure(allow_insecure);
    }
    for (key, value) in parse_string_map(options.resource_attributes, "resourceAttributes")? {
        config = config.with_resource_attribute(key, value);
    }
    Ok(config)
}

fn build_otel_metric_config(
    options: OpenTelemetryMetricConfig,
) -> Result<nemo_relay::observability::otel_metrics::OpenTelemetryMetricConfig> {
    let endpoint = options.endpoint.trim().to_string();
    if endpoint.is_empty() {
        return Err(Error::invalid_argument(
            "endpoint must be a nonblank string",
        ));
    }
    let mut config =
        nemo_relay::observability::otel_metrics::OpenTelemetryMetricConfig::new(endpoint)
            .with_transport(parse_otel_transport(options.transport)?)
            .with_service_name(
                options
                    .service_name
                    .unwrap_or_else(|| "unknown_service".to_string()),
            )
            .with_instrumentation_scope(
                options
                    .instrumentation_scope
                    .unwrap_or_else(|| "opentelemetry".to_string()),
            )
            .with_timeout(std::time::Duration::from_millis(
                options.timeout_millis.unwrap_or(3_000).into(),
            ))
            .with_export_interval(std::time::Duration::from_millis(
                options.export_interval_millis.unwrap_or(60_000).into(),
            ))
            .with_max_instruments(options.max_instruments.unwrap_or(256) as usize)
            .with_cardinality_limit(options.cardinality_limit.unwrap_or(2_000) as usize);
    if let Some(temporality) = options.temporality {
        config = config.with_temporality(temporality.into());
    }
    if let Some(namespace) = options.service_namespace {
        config = config.with_service_namespace(namespace);
    }
    if let Some(version) = options.service_version {
        config = config.with_service_version(version);
    }
    for (key, value) in parse_string_map(options.headers, "headers")? {
        config = config.with_header(key, value);
    }
    for (key, variable) in parse_string_map(options.header_env, "headerEnv")? {
        config = config.with_header_env(key, variable);
    }
    if let Some(tls) = options.tls {
        config = config.with_tls(tls.into());
    }
    if let Some(allow_insecure) = options.allow_insecure {
        config = config.with_allow_insecure(allow_insecure);
    }
    for (key, value) in parse_string_map(options.resource_attributes, "resourceAttributes")? {
        config = config.with_resource_attribute(key, value);
    }
    Ok(config)
}

fn exporter_runtime_diagnostics_json(
    diagnostics: nemo_relay::observability::ExporterRuntimeDiagnostics,
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
) -> Result<nemo_relay::observability::atof::AtofExporterConfig> {
    let options = options.unwrap_or_default();
    match options.r#type.as_deref().unwrap_or("file") {
        "file" => {
            let mut config = nemo_relay::observability::atof::AtofExporterConfig::new();
            if let Some(output_directory) = options.output_directory {
                config = config.with_output_directory(PathBuf::from(output_directory));
            }
            if let Some(filename) = options.filename {
                config = config.with_filename(filename);
            }
            if let Some(mode) = options.mode {
                let Some(mode) = nemo_relay::observability::atof::AtofExporterMode::parse(&mode)
                else {
                    return Err(Error::invalid_argument(
                        "mode must be 'append' or 'overwrite'",
                    ));
                };
                config = config.with_mode(mode);
            }
            Ok(config)
        }
        "stream" => {
            let url = options
                .url
                .ok_or_else(|| Error::invalid_argument("stream sink requires url"))?;
            let transport = options.transport.unwrap_or_else(|| "http_post".to_string());
            let Some(transport) =
                nemo_relay::observability::atof::AtofEndpointTransport::parse(&transport)
            else {
                return Err(Error::invalid_argument(
                    "stream transport must be 'http_post', 'websocket', or 'ndjson'",
                ));
            };
            let mut sink =
                nemo_relay::observability::atof::AtofStreamSinkConfig::new(url, transport);
            if let Some(timeout_millis) = options.timeout_millis {
                sink = sink.with_timeout_millis(timeout_millis.into());
            }
            if let Some(allow_insecure) = options.allow_insecure {
                sink = sink.with_allow_insecure(allow_insecure);
            }
            if let Some(field_name_policy) = options.field_name_policy {
                let Some(field_name_policy) =
                    nemo_relay::observability::atof::AtofEndpointFieldNamePolicy::parse(
                        &field_name_policy,
                    )
                else {
                    return Err(Error::invalid_argument(
                        "stream field_name_policy must be 'preserve' or 'replace_dots'",
                    ));
                };
                sink = sink.with_field_name_policy(field_name_policy);
            }
            for (key, value) in parse_string_map(options.headers, "headers")? {
                sink = sink.with_header(key, value);
            }
            for (key, variable) in parse_string_map(options.header_env, "headerEnv")? {
                sink = sink.with_header_env(key, variable);
            }
            Ok(nemo_relay::observability::atof::AtofExporterConfig::new().with_stream_sink(sink))
        }
        _ => Err(Error::invalid_argument(
            "ATOF sink type must be 'file' or 'stream'",
        )),
    }
}

pub(crate) fn llm_stream_from_rust_stream(
    rust_stream: LlmJsonStream,
    publication_context_id: Option<String>,
) -> LlmStream {
    LlmStream::new(
        rust_stream,
        Default::default(),
        Default::default(),
        publication_context_id,
        None,
    )
}

struct ScopedJsonCall {
    value: Json,
    scope: CallbackScope,
}

fn scoped_json_callback_tsfn_from_wrapper(
    callback: Function<JsArg, JsArg>,
) -> Result<Tsfn<ScopedJsonCall, JsArgs2, JsArg>> {
    Ok(callback
        .build_threadsafe_function::<ScopedJsonCall>()
        .weak::<true>()
        .build_callback(|ctx: ThreadsafeCallContext<ScopedJsonCall>| {
            let value = js_value(&ctx.env, ctx.value.value)?;
            Ok(FnArgs::from((value, ctx.value.scope.into_js(&ctx.env)?)))
        })?)
}

fn scoped_json_callback_tsfn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> Result<Tsfn<ScopedJsonCall, JsArgs2, JsArg>> {
    scoped_json_callback_tsfn_from_wrapper(callback_factory::wrap_scoped_callback(env, func)?)
}

fn scoped_tool_execution_fn(func: Tsfn<ScopedJsonCall, JsArgs2, JsArg>) -> ToolExecutionNextFn {
    let func = Arc::new(func);
    Arc::new(move |args: Json| {
        let func = func.clone();
        let scope = CallbackScope::capture();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let status = func.call_with_return_value(
                ScopedJsonCall { value: args, scope },
                ThreadsafeFunctionCallMode::NonBlocking,
                move |value: napi::Result<JsArg>, env| {
                    let _ = tx.send(callable::tsfn_envelope(
                        &env,
                        value,
                        "JS tool execution callback failed",
                    ));
                    Ok(())
                },
            );
            if status != napi::Status::Ok {
                return Err(FlowError::Internal(format!(
                    "failed to queue JS tool execution callback: {status:?}",
                )));
            }
            let result = rx
                .await
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            callable::parse_tool_execution_result(result?)
        })
    })
}

fn scoped_llm_execution_fn(func: Tsfn<ScopedJsonCall, JsArgs2, JsArg>) -> LlmExecutionNextFn {
    let func = Arc::new(func);
    Arc::new(move |request: LlmRequest| {
        let func = func.clone();
        let scope = CallbackScope::capture();
        let request = serde_json::to_value(request).unwrap_or(Json::Null);
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let status = func.call_with_return_value(
                ScopedJsonCall {
                    value: request,
                    scope,
                },
                ThreadsafeFunctionCallMode::NonBlocking,
                move |value: napi::Result<JsArg>, env| {
                    let _ = tx.send(callable::tsfn_envelope(
                        &env,
                        value,
                        "JS LLM execution callback failed",
                    ));
                    Ok(())
                },
            );
            if status != napi::Status::Ok {
                return Err(FlowError::Internal(format!(
                    "failed to queue JS LLM execution callback: {status:?}",
                )));
            }
            rx.await
                .map_err(|error| FlowError::Internal(error.to_string()))?
        })
    })
}

fn middleware_tool_callback_tsfn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> Result<Tsfn<(String, Json), JsArgs2, JsArg>> {
    let callback: Function<JsArgs2, JsArg> = callback_factory::wrap_middleware_callback(env, func)?;
    let tsfn = callback
        .build_threadsafe_function::<(String, Json)>()
        .weak::<true>()
        .build_callback(|ctx: ThreadsafeCallContext<(String, Json)>| {
            let name = ctx.env.create_string_from_std(ctx.value.0)?;
            let args = js_value(&ctx.env, ctx.value.1)?;
            Ok(FnArgs::from((detach(&ctx.env, &name), args)))
        })?;
    Ok(tsfn)
}

fn middleware_json_callback_tsfn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> Result<Tsfn<Json, Json, JsArg>> {
    let callback: Function<Json, JsArg> = callback_factory::wrap_middleware_callback(env, func)?;
    let tsfn = callback
        .build_threadsafe_function::<Json>()
        .weak::<true>()
        .build_callback(|ctx: ThreadsafeCallContext<Json>| Ok(ctx.value))?;
    Ok(tsfn)
}

#[allow(clippy::too_many_arguments)]
fn add_plugin_event_sanitizer(
    env: &Env,
    context: &mut Object,
    property: &str,
    namespace_prefix: String,
    registrations: Arc<StdMutex<Vec<PluginRegistration>>>,
    register: fn(&str, i32, EventSanitizeFn) -> FlowResult<()>,
    deregister: fn(&str) -> FlowResult<bool>,
    label: &'static str,
) -> napi::Result<()> {
    let function = closure_function(env, property, move |ctx| {
        let name = format!("{}{}", namespace_prefix, ctx.get::<String>(0)?);
        let priority = ctx.get::<i32>(1)?;
        let callback = ctx.get::<Callback>(2)?;
        register(&name, priority, node_event_sanitize_fn(ctx.env, &callback)?)
            .map_err(Error::from)?;
        let name_clone = name.clone();
        registrations.lock().unwrap().push(PluginRegistration::new(
            "plugin",
            name_clone.clone(),
            Box::new(move || {
                deregister(&name_clone).map(|_| ()).map_err(|error| {
                    PluginError::RegistrationFailed(format!(
                        "{label} deregistration failed: {error}"
                    ))
                })
            }),
        ));
        Ok(())
    })?;
    context.set_named_property(property, function)
}

fn add_plugin_event_metadata_injector(
    env: &Env,
    context: &mut Object,
    namespace_prefix: String,
    registrations: Arc<StdMutex<Vec<PluginRegistration>>>,
) -> napi::Result<()> {
    let function = closure_function(
        env,
        "__nemo_relay_plugin_register_event_metadata_injector",
        move |ctx| {
            let name = format!("{}{}", namespace_prefix, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<Callback>(2)?;
            core_registry_api::register_event_metadata_injector(
                &name,
                priority,
                node_event_metadata_injector_fn(ctx.env, &callback)?,
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            registrations.lock().unwrap().push(PluginRegistration::new(
                "plugin",
                name_clone.clone(),
                Box::new(move || {
                    core_registry_api::deregister_event_metadata_injector(&name_clone)
                        .map(|_| ())
                        .map_err(|error| {
                            PluginError::RegistrationFailed(format!(
                                "event metadata injector deregistration failed: {error}"
                            ))
                        })
                }),
            ));
            Ok(())
        },
    )?;
    context.set_named_property("registerEventMetadataInjector", function)
}

fn build_plugin_context(
    env: &Env,
    namespace_prefix: String,
    registrations: Arc<StdMutex<Vec<PluginRegistration>>>,
) -> napi::Result<Object<'_>> {
    let mut context = Object::new(env)?;

    let conditional_gate_regs = registrations.clone();
    let conditional_gate_namespace = namespace_prefix.clone();
    let register_conditional_gate = closure_function(
        env,
        "__nemo_relay_plugin_register_conditional_middleware_guardrail",
        move |ctx| {
            let name = format!("{}{}", conditional_gate_namespace, ctx.get::<String>(0)?);
            let kinds = ctx
                .get::<Vec<RuntimeRegistrationKind>>(1)?
                .into_iter()
                .map(Into::into)
                .collect::<BTreeSet<_>>();
            let registration_name = ctx.get::<String>(2)?;
            let guardrail = ctx.get::<Callback>(3)?;
            let callback = node_conditional_middleware_guardrail(ctx.env, &guardrail)?;
            core_registry_api::register_conditional_middleware_guardrail(
                &name,
                kinds,
                &registration_name,
                Arc::new(move |kinds, effective_name| {
                    callback
                        .call(
                            kinds.iter().map(|kind| kind.as_str().to_string()).collect(),
                            effective_name.to_string(),
                        )
                        .unwrap_or_else(|error| {
                            record_callback_error(format!(
                                "Node conditional middleware guardrail failed open: {error}"
                            ));
                            None
                        })
                }),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            conditional_gate_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_conditional_middleware_guardrail(&name_clone)
                            .map(|_| ())
                            .map_err(|error| {
                                PluginError::RegistrationFailed(format!(
                                    "conditional middleware guardrail deregistration failed: {error}"
                                ))
                            })
                    }),
                ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerConditionalMiddlewareGuardrail",
        register_conditional_gate,
    )?;

    let subscriber_regs = registrations.clone();
    let subscriber_namespace = namespace_prefix.clone();
    let register_subscriber = closure_function(
        env,
        "__nemo_relay_adaptive_register_subscriber",
        move |ctx| {
            let name = format!("{}{}", subscriber_namespace, ctx.get::<String>(0)?);
            let callback = ctx.get::<Callback>(1)?;
            core_subscriber_api::register_subscriber(
                &name,
                callable::wrap_js_event_subscriber(ctx.env, name.clone(), &callback)?,
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            subscriber_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_subscriber_api::deregister_subscriber(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "subscriber deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            Ok(())
        },
    )?;
    context.set_named_property("registerSubscriber", register_subscriber)?;

    add_plugin_event_metadata_injector(
        env,
        &mut context,
        namespace_prefix.clone(),
        registrations.clone(),
    )?;

    add_plugin_event_sanitizer(
        env,
        &mut context,
        "registerMarkSanitizeGuardrail",
        namespace_prefix.clone(),
        registrations.clone(),
        core_registry_api::register_mark_sanitize_guardrail,
        core_registry_api::deregister_mark_sanitize_guardrail,
        "mark sanitize guardrail",
    )?;
    add_plugin_event_sanitizer(
        env,
        &mut context,
        "registerScopeSanitizeStartGuardrail",
        namespace_prefix.clone(),
        registrations.clone(),
        core_registry_api::register_scope_sanitize_start_guardrail,
        core_registry_api::deregister_scope_sanitize_start_guardrail,
        "scope start sanitize guardrail",
    )?;
    add_plugin_event_sanitizer(
        env,
        &mut context,
        "registerScopeSanitizeEndGuardrail",
        namespace_prefix.clone(),
        registrations.clone(),
        core_registry_api::register_scope_sanitize_end_guardrail,
        core_registry_api::deregister_scope_sanitize_end_guardrail,
        "scope end sanitize guardrail",
    )?;

    let tool_sanitize_request_regs = registrations.clone();
    let tool_sanitize_request_namespace = namespace_prefix.clone();
    let register_tool_sanitize_request_guardrail = closure_function(
        env,
        "__nemo_relay_plugin_register_tool_sanitize_request_guardrail",
        move |ctx| {
            let name = format!(
                "{}{}",
                tool_sanitize_request_namespace,
                ctx.get::<String>(0)?
            );
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<Callback>(2)?;
            core_registry_api::register_tool_sanitize_request_guardrail(
                &name,
                priority,
                callable::wrap_js_tool_sanitize_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            tool_sanitize_request_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_tool_sanitize_request_guardrail(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "tool sanitize request guardrail deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerToolSanitizeRequestGuardrail",
        register_tool_sanitize_request_guardrail,
    )?;

    let tool_sanitize_response_regs = registrations.clone();
    let tool_sanitize_response_namespace = namespace_prefix.clone();
    let register_tool_sanitize_response_guardrail = closure_function(
        env,
        "__nemo_relay_plugin_register_tool_sanitize_response_guardrail",
        move |ctx| {
            let name = format!(
                "{}{}",
                tool_sanitize_response_namespace,
                ctx.get::<String>(0)?
            );
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<Callback>(2)?;
            core_registry_api::register_tool_sanitize_response_guardrail(
                &name,
                priority,
                callable::wrap_js_tool_sanitize_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            tool_sanitize_response_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_tool_sanitize_response_guardrail(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "tool sanitize response guardrail deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerToolSanitizeResponseGuardrail",
        register_tool_sanitize_response_guardrail,
    )?;

    let tool_conditional_regs = registrations.clone();
    let tool_conditional_namespace = namespace_prefix.clone();
    let register_tool_conditional_execution_guardrail = closure_function(
        env,
        "__nemo_relay_plugin_register_tool_conditional_execution_guardrail",
        move |ctx| {
            let name = format!("{}{}", tool_conditional_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<Callback>(2)?;
            core_registry_api::register_tool_conditional_execution_guardrail(
                &name,
                priority,
                callable::wrap_js_tool_conditional_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            tool_conditional_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_tool_conditional_execution_guardrail(
                            &name_clone,
                        )
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "tool conditional execution guardrail deregistration failed: {e}"
                            ))
                        })
                    }),
                ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerToolConditionalExecutionGuardrail",
        register_tool_conditional_execution_guardrail,
    )?;

    let llm_sanitize_request_regs = registrations.clone();
    let llm_sanitize_request_namespace = namespace_prefix.clone();
    let register_llm_sanitize_request_guardrail = closure_function(
        env,
        "__nemo_relay_plugin_register_llm_sanitize_request_guardrail",
        move |ctx| {
            let name = format!(
                "{}{}",
                llm_sanitize_request_namespace,
                ctx.get::<String>(0)?
            );
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<Callback>(2)?;
            core_registry_api::register_llm_sanitize_request_guardrail(
                &name,
                priority,
                callable::wrap_js_llm_sanitize_request_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            llm_sanitize_request_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_llm_sanitize_request_guardrail(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "llm sanitize request guardrail deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerLlmSanitizeRequestGuardrail",
        register_llm_sanitize_request_guardrail,
    )?;

    let llm_sanitize_response_regs = registrations.clone();
    let llm_sanitize_response_namespace = namespace_prefix.clone();
    let register_llm_sanitize_response_guardrail = closure_function(
        env,
        "__nemo_relay_plugin_register_llm_sanitize_response_guardrail",
        move |ctx| {
            let name = format!(
                "{}{}",
                llm_sanitize_response_namespace,
                ctx.get::<String>(0)?
            );
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<Callback>(2)?;
            core_registry_api::register_llm_sanitize_response_guardrail(
                &name,
                priority,
                callable::wrap_js_llm_sanitize_response_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            llm_sanitize_response_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_llm_sanitize_response_guardrail(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "llm sanitize response guardrail deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerLlmSanitizeResponseGuardrail",
        register_llm_sanitize_response_guardrail,
    )?;

    let llm_conditional_regs = registrations.clone();
    let llm_conditional_namespace = namespace_prefix.clone();
    let register_llm_conditional_execution_guardrail = closure_function(
        env,
        "__nemo_relay_plugin_register_llm_conditional_execution_guardrail",
        move |ctx| {
            let name = format!("{}{}", llm_conditional_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<Callback>(2)?;
            core_registry_api::register_llm_conditional_execution_guardrail(
                &name,
                priority,
                callable::wrap_js_llm_conditional_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            llm_conditional_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_llm_conditional_execution_guardrail(
                            &name_clone,
                        )
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "llm conditional execution guardrail deregistration failed: {e}"
                            ))
                        })
                    }),
                ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerLlmConditionalExecutionGuardrail",
        register_llm_conditional_execution_guardrail,
    )?;

    let llm_regs = registrations.clone();
    let llm_request_namespace = namespace_prefix.clone();
    let register_llm_request_intercept = closure_function(
        env,
        "__nemo_relay_adaptive_register_llm_request_intercept",
        move |ctx| {
            let name = format!("{}{}", llm_request_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let break_chain = ctx.get::<bool>(2)?;
            let callback = ctx.get::<Callback>(3)?;
            core_registry_api::register_llm_request_intercept(
                &name,
                priority,
                break_chain,
                callable::wrap_js_llm_request_intercept_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            llm_regs.lock().unwrap().push(PluginRegistration::new(
                "plugin",
                name_clone.clone(),
                Box::new(move || {
                    core_registry_api::deregister_llm_request_intercept(&name_clone)
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "llm request intercept deregistration failed: {e}"
                            ))
                        })
                }),
            ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerLlmRequestIntercept",
        register_llm_request_intercept,
    )?;

    let llm_exec_regs = registrations.clone();
    let llm_exec_namespace = namespace_prefix.clone();
    let register_llm_execution_intercept = closure_function(
        env,
        "__nemo_relay_adaptive_register_llm_execution_intercept",
        move |ctx| {
            let name = format!("{}{}", llm_exec_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<Callback>(2)?;
            core_registry_api::register_llm_execution_intercept(
                &name,
                priority,
                callable::wrap_js_llm_exec_intercept_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            llm_exec_regs.lock().unwrap().push(PluginRegistration::new(
                "plugin",
                name_clone.clone(),
                Box::new(move || {
                    core_registry_api::deregister_llm_execution_intercept(&name_clone)
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "llm execution intercept deregistration failed: {e}"
                            ))
                        })
                }),
            ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerLlmExecutionIntercept",
        register_llm_execution_intercept,
    )?;

    let llm_stream_exec_regs = registrations.clone();
    let llm_stream_namespace = namespace_prefix.clone();
    let register_llm_stream_execution_intercept = closure_function(
        env,
        "__nemo_relay_adaptive_register_llm_stream_execution_intercept",
        move |ctx| {
            let name = format!("{}{}", llm_stream_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<Callback>(2)?;
            core_registry_api::register_llm_stream_execution_intercept(
                &name,
                priority,
                callable::wrap_js_llm_stream_exec_intercept_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            llm_stream_exec_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_llm_stream_execution_intercept(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "llm stream execution intercept deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerLlmStreamExecutionIntercept",
        register_llm_stream_execution_intercept,
    )?;

    let tool_request_regs = registrations.clone();
    let tool_request_namespace = namespace_prefix.clone();
    let register_tool_request_intercept = closure_function(
        env,
        "__nemo_relay_adaptive_register_tool_request_intercept",
        move |ctx| {
            let name = format!("{}{}", tool_request_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let break_chain = ctx.get::<bool>(2)?;
            let callback = ctx.get::<Callback>(3)?;
            core_registry_api::register_tool_request_intercept(
                &name,
                priority,
                break_chain,
                callable::wrap_js_tool_request_intercept_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            tool_request_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_tool_request_intercept(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "tool request intercept deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerToolRequestIntercept",
        register_tool_request_intercept,
    )?;

    let tool_regs = registrations.clone();
    let tool_exec_namespace = namespace_prefix;
    let register_tool_execution_intercept = closure_function(
        env,
        "__nemo_relay_adaptive_register_tool_execution_intercept",
        move |ctx| {
            let name = format!("{}{}", tool_exec_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<Callback>(2)?;
            core_registry_api::register_tool_execution_intercept(
                &name,
                priority,
                callable::wrap_js_tool_exec_intercept_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(Error::from)?;

            let name_clone = name.clone();
            tool_regs.lock().unwrap().push(PluginRegistration::new(
                "plugin",
                name_clone.clone(),
                Box::new(move || {
                    core_registry_api::deregister_tool_execution_intercept(&name_clone)
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "tool execution intercept deregistration failed: {e}"
                            ))
                        })
                }),
            ));
            Ok(())
        },
    )?;
    context.set_named_property(
        "registerToolExecutionIntercept",
        register_tool_execution_intercept,
    )?;

    Ok(context)
}

struct NodePluginRegisterCall {
    plugin_config: Json,
    namespace_prefix: String,
    registrations: Arc<StdMutex<Vec<PluginRegistration>>>,
}

const CONDITIONAL_GATE_PREFIX: &str = "conditional middleware guardrail";

fn decode_conditional_gate_result(value: FlowResult<Json>) -> Result<Option<String>> {
    serde_json::from_value(value?).map_err(|_| {
        Error::invalid_argument("conditional middleware guardrail must return a string or null")
    })
}

struct NodeConditionalMiddlewareGuardrail {
    direct: PersistentJsFunction<FnArgs<(Vec<String>, String)>, JsArg>,
    thread_safe: Tsfn<(Vec<String>, String), JsArgs2, JsArg>,
    registration_thread: std::thread::ThreadId,
}

const CONDITIONAL_GATE_CALLBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

fn recv_conditional_gate_result(
    rx: std::sync::mpsc::Receiver<Result<Option<String>>>,
    timeout: std::time::Duration,
) -> Result<Option<String>> {
    rx.recv_timeout(timeout).map_err(|error| match error {
        std::sync::mpsc::RecvTimeoutError::Timeout => Error::generic_failure(format!(
            "conditional gate callback timed out after {} seconds",
            timeout.as_secs()
        )),
        std::sync::mpsc::RecvTimeoutError::Disconnected => {
            Error::generic_failure("conditional gate callback channel disconnected")
        }
    })?
}

impl NodeConditionalMiddlewareGuardrail {
    fn call(&self, kinds: Vec<String>, registration_name: String) -> Result<Option<String>> {
        if std::thread::current().id() == self.registration_thread {
            let env = self.direct.env();
            let value = self
                .direct
                .function()?
                .call((kinds, registration_name).into())?;
            let envelope = unsafe { value.into_unknown(&env) };
            return decode_conditional_gate_result(callable::unwrap_envelope(
                envelope,
                CONDITIONAL_GATE_PREFIX,
            ));
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let status = self.thread_safe.call_with_return_value(
            (kinds, registration_name),
            ThreadsafeFunctionCallMode::NonBlocking,
            move |value: napi::Result<JsArg>, env| {
                let _ = tx.send(decode_conditional_gate_result(callable::tsfn_envelope(
                    &env,
                    value,
                    CONDITIONAL_GATE_PREFIX,
                )));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            return Err(Error::generic_failure(format!(
                "conditional gate callback scheduling failed: {status:?}"
            )));
        }
        recv_conditional_gate_result(rx, CONDITIONAL_GATE_CALLBACK_TIMEOUT)
    }
}

fn node_conditional_middleware_guardrail(
    env: &Env,
    guardrail: &Function<JsArg, JsArg>,
) -> Result<Arc<NodeConditionalMiddlewareGuardrail>> {
    let guardrail: Function<FnArgs<(Vec<String>, String)>, JsArg> =
        callback_factory::wrap_conditional_gate_callback(env, guardrail)?;
    let direct = PersistentJsFunction::new(env, &guardrail)?;
    let registration_thread = std::thread::current().id();
    let thread_safe = guardrail
        .build_threadsafe_function::<(Vec<String>, String)>()
        .weak::<true>()
        .build_callback(|ctx: ThreadsafeCallContext<(Vec<String>, String)>| {
            let kinds = js_value(&ctx.env, ctx.value.0)?;
            let registration_name = ctx.env.create_string_from_std(ctx.value.1)?;
            Ok(FnArgs::from((kinds, detach(&ctx.env, &registration_name))))
        })?;
    Ok(Arc::new(NodeConditionalMiddlewareGuardrail {
        direct,
        thread_safe,
        registration_thread,
    }))
}

#[cfg(test)]
#[path = "../../tests/rust/api/conditional_gate_tests.rs"]
mod conditional_gate_tests;

fn node_event_sanitize_fn(env: &Env, func: &Function<JsArg, JsArg>) -> Result<EventSanitizeFn> {
    // The registry and queued snapshots own the only callback references.
    // PromiseAwareFn releases its TSFN on the last drop, so deregistration
    // preserves already-snapshotted publication while still cleaning up
    // deterministically once that work finishes.
    let callback = Arc::new(crate::promise_call::PromiseAwareFn::new(env, func)?);
    Ok(callable::wrap_js_event_sanitize_promise_fn(callback))
}

fn node_event_metadata_injector_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> Result<EventMetadataInjectorFn> {
    let callback = Arc::new(crate::promise_call::PromiseAwareFn::new(env, func)?);
    Ok(callable::wrap_js_event_metadata_injector_promise_fn(
        callback,
    ))
}

fn node_llm_codec(
    env: &Env,
    decode: Option<&Function<JsArg, JsArg>>,
    encode: Option<&Function<JsArg, JsArg>>,
) -> Result<Option<Arc<dyn nemo_relay::codec::traits::LlmCodec>>> {
    let (decode, encode) = match (decode, encode) {
        (Some(decode), Some(encode)) => (decode, encode),
        (None, None) => return Ok(None),
        _ => {
            return Err(Error::invalid_argument(
                "codecDecode and codecEncode must be provided together",
            ));
        }
    };
    let decode = callable::JsCodecCallback::new(env, decode, "JS codec decode callback failed")?;
    let encode = callable::JsCodecCallback::new(env, encode, "JS codec encode callback failed")?;
    Ok(Some(callable::wrap_js_codec(decode, encode)))
}

fn node_llm_response_codec(
    env: &Env,
    decode: Option<&Function<JsArg, JsArg>>,
) -> Result<Option<Arc<dyn nemo_relay::codec::traits::LlmResponseCodec>>> {
    Ok(decode
        .map(|decode| {
            callable::JsCodecCallback::new(env, decode, "JS response codec decode callback failed")
                .map(callable::wrap_js_response_codec)
        })
        .transpose()?)
}

const PLUGIN_VALIDATE_PREFIX: &str = "JS plugin validate callback failed";

struct NodePluginValidateCallback {
    direct: PersistentJsFunction<Json, JsArg>,
    thread_safe: Tsfn<Json, Json, JsArg>,
    registration_thread: std::thread::ThreadId,
}

impl NodePluginValidateCallback {
    fn call(&self, plugin_config: Json) -> Result<Json> {
        if std::thread::current().id() == self.registration_thread {
            let env = self.direct.env();
            let value = self.direct.function()?.call(plugin_config)?;
            let envelope = unsafe { value.into_unknown(&env) };
            return Ok(callable::unwrap_envelope(envelope, PLUGIN_VALIDATE_PREFIX)?);
        }

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let status = self.thread_safe.call_with_return_value(
            plugin_config,
            ThreadsafeFunctionCallMode::NonBlocking,
            move |value: napi::Result<JsArg>, env| {
                let _ = tx.send(
                    callable::tsfn_envelope(&env, value, PLUGIN_VALIDATE_PREFIX)
                        .map_err(Error::from),
                );
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            return Err(Error::generic_failure(format!(
                "failed to queue JS plugin validate callback: {status:?}"
            )));
        }
        rx.recv()
            .map_err(|_| Error::generic_failure("JS plugin validate completion channel closed"))?
    }
}

struct NodePlugin {
    plugin_kind: String,
    validate: Option<NodePluginValidateCallback>,
    register: Tsfn<NodePluginRegisterCall, JsArgs2, JsArg>,
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
        match validate.call(Json::Object(plugin_config.clone())) {
            Ok(Json::Null) => vec![],
            Ok(value) => {
                serde_json::from_value::<Vec<ConfigDiagnostic>>(value).unwrap_or_else(|e| {
                    vec![ConfigDiagnostic {
                        level: DiagnosticLevel::Error,
                        code: "plugin.validate_failed".into(),
                        component: Some(self.plugin_kind.clone()),
                        field: None,
                        message: format!("JS plugin validate returned invalid diagnostics: {e}"),
                    }]
                })
            }
            Err(e) => vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "plugin.validate_failed".into(),
                component: Some(self.plugin_kind.clone()),
                field: None,
                message: format!("JS plugin validate failed: {e}"),
            }],
        }
    }

    fn register<'a>(
        &'a self,
        plugin_config: &serde_json::Map<String, serde_json::Value>,
        ctx: &'a mut PluginRegistrationContext,
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
            let (tx, rx) = std::sync::mpsc::sync_channel::<FlowResult<()>>(1);
            let status = self.register.call_with_return_value(
                payload,
                ThreadsafeFunctionCallMode::NonBlocking,
                move |value: napi::Result<JsArg>, env| {
                    let _ = tx.send(
                        callable::tsfn_envelope(&env, value, "JS plugin register callback failed")
                            .map(|_| ()),
                    );
                    Ok(())
                },
            );
            let result = if status != napi::Status::Ok {
                Err(PluginError::RegistrationFailed(format!(
                    "failed to queue JS plugin register callback: {status:?}"
                )))
            } else {
                rx.recv()
                    .map_err(|_| {
                        PluginError::RegistrationFailed(
                            "JS plugin register completion channel closed".into(),
                        )
                    })?
                    .map_err(|error| PluginError::RegistrationFailed(error.to_string()))
            };
            if let Err(error) = result {
                let mut registrations = registrations
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                rollback_registrations(&mut registrations);
                return Err(error);
            }

            let drained = std::mem::take(&mut *registrations.lock().map_err(|e| {
                PluginError::RegistrationFailed(format!("plugin registrations lock poisoned: {e}"))
            })?);
            ctx.extend_registrations(drained);
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Transport-neutral Relay causal context for application-managed transport.
#[napi(object)]
pub struct PropagationContext {
    pub version: u32,
    pub root_uuid: Option<String>,
    pub parent_uuid: String,
    pub traceparent: Option<String>,
    pub tracestate: Option<String>,
}

fn propagation_context_from_napi(
    context: PropagationContext,
) -> Result<nemo_relay::api::runtime::PropagationContext> {
    let root_uuid = context
        .root_uuid
        .as_deref()
        .map(uuid::Uuid::parse_str)
        .transpose()
        .map_err(|error| Error::invalid_argument(format!("invalid root UUID: {error}")))?;
    let parent_uuid = uuid::Uuid::parse_str(&context.parent_uuid)
        .map_err(|error| Error::invalid_argument(format!("invalid parent UUID: {error}")))?;
    let version = u16::try_from(context.version)
        .map_err(|_| Error::invalid_argument("propagation context version is out of range"))?;
    let context = nemo_relay::api::runtime::PropagationContext {
        version,
        root_uuid,
        parent_uuid,
        traceparent: context.traceparent,
        tracestate: context.tracestate,
    };
    context.validate().map_err(Error::from)?;
    Ok(context)
}

pub(crate) fn propagation_context_to_napi(
    context: nemo_relay::api::runtime::PropagationContext,
) -> PropagationContext {
    PropagationContext {
        version: u32::from(context.version),
        root_uuid: context.root_uuid.map(|uuid| uuid.to_string()),
        parent_uuid: context.parent_uuid.to_string(),
        traceparent: context.traceparent,
        tracestate: context.tracestate,
    }
}

fn callback_propagation_context(
    scope: &CallbackScope,
    mut context: nemo_relay::api::runtime::PropagationContext,
) -> Result<nemo_relay::api::runtime::PropagationContext> {
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
    context: nemo_relay::api::runtime::PropagationContext,
) -> Result<nemo_relay::api::runtime::PropagationContext> {
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
    ScopeStack {
        inner: create_scope_stack_handle(),
        publication_buffer: None,
    }
}

/// Capture the current Relay causal parent for application-managed transport.
#[napi(catch_unwind)]
pub fn capture_propagation_context(env: Env) -> Result<PropagationContext> {
    let scope = effective_scope(&env)?;
    if let Some(context) = scope.propagation.clone() {
        return callback_propagation_context(&scope, context).map(propagation_context_to_napi);
    }
    with_callback_scope(&scope, capture_propagation_context_handle)
        .map(propagation_context_to_napi)
        .map_err(Error::from)
}

/// Capture the current Relay causal parent without a propagation root.
#[napi(catch_unwind)]
pub fn capture_rootless_propagation_context(env: Env) -> Result<PropagationContext> {
    let scope = effective_scope(&env)?;
    if let Some(context) = scope.propagation.clone() {
        return rootless_callback_propagation_context(&scope, context)
            .map(propagation_context_to_napi);
    }
    with_callback_scope(&scope, capture_rootless_propagation_context_handle)
        .map(propagation_context_to_napi)
        .map_err(Error::from)
}

/// Capture the current parent with an optional stable application session root.
#[napi(catch_unwind)]
pub fn capture_propagation_context_with_root(
    env: Env,
    root_uuid: Option<String>,
) -> Result<PropagationContext> {
    let root_uuid = root_uuid
        .as_deref()
        .map(uuid::Uuid::parse_str)
        .transpose()
        .map_err(|error| Error::invalid_argument(format!("invalid root UUID: {error}")))?;
    let scope = effective_scope(&env)?;
    if let Some(context) = scope.propagation.clone() {
        let mut context = rootless_callback_propagation_context(&scope, context)?;
        context.root_uuid = root_uuid;
        return Ok(propagation_context_to_napi(context));
    }
    with_callback_scope(&scope, || {
        capture_propagation_context_with_root_handle(root_uuid)
    })
    .map(propagation_context_to_napi)
    .map_err(Error::from)
}

/// Capture the current Relay context as a W3C `traceparent` value.
#[napi(catch_unwind)]
pub fn capture_traceparent(env: Env) -> Result<String> {
    let scope = effective_scope(&env)?;
    if let Some(context) = scope.propagation.clone() {
        let context = callback_propagation_context(&scope, context)?;
        if let Some(traceparent) = context.traceparent {
            return Ok(traceparent);
        }
        return context.to_traceparent().map_err(Error::from);
    }
    with_callback_scope(&scope, capture_traceparent_handle).map_err(Error::from)
}
/// Serialize a Relay causal context to the JSON wire format.
#[napi(catch_unwind)]
pub fn propagation_context_to_json(context: PropagationContext) -> Result<String> {
    propagation_context_from_napi(context)?
        .to_json()
        .map_err(Error::from)
}

/// Convert a Relay propagation context to a W3C `traceparent` value.
#[napi(catch_unwind)]
pub fn propagation_context_to_traceparent(context: PropagationContext) -> Result<String> {
    propagation_context_from_napi(context)?
        .to_traceparent()
        .map_err(Error::from)
}

/// Deserialize and validate a Relay causal context from the JSON wire format.
#[napi(catch_unwind)]
pub fn propagation_context_from_json(value: String) -> Result<PropagationContext> {
    nemo_relay::api::runtime::PropagationContext::from_json(&value)
        .map(propagation_context_to_napi)
        .map_err(Error::from)
}

/// Create an isolated scope stack seeded from a received propagation context.
#[napi(catch_unwind)]
pub fn create_scope_stack_from_propagation(context: PropagationContext) -> Result<ScopeStack> {
    create_scope_stack_from_propagation_handle(&propagation_context_from_napi(context)?)
        .map(ScopeStack::from)
        .map_err(Error::from)
}

/// Run a callback with an isolated scope stack installed.
///
/// The caller's stack is restored immediately after callback invocation. When
/// the callback returns a Promise, the requested stack remains active for that
/// Promise until it settles and is then expired for inherited detached work.
/// Use this helper instead of `setThreadScopeStack` to isolate concurrent async
/// branches.
#[napi(catch_unwind)]
pub fn with_scope_stack<'env>(
    env: &'env Env,
    stack: &ScopeStack,
    #[napi(ts_arg_type = "(...args: any[]) => any")] callback: Callback,
) -> Result<Unknown<'env>> {
    let scope = CallbackScope {
        scope_stack: stack.inner.clone(),
        publication_buffer: effective_scope(env)?
            .publication_buffer
            .or_else(|| stack.publication_buffer.clone()),
        propagation: None,
    };
    Ok(callback_factory::with_callback_scope(
        env, scope, &callback,
    )?)
}

/// Return owned scope identities, or null without an explicitly active binding.
#[napi(catch_unwind)]
pub fn current_scope_stack_snapshot(env: Env) -> Result<Option<crate::types::ScopeStackSnapshot>> {
    let snapshot = match callback_factory::callback_scope(&env)? {
        Some(scope) => nemo_relay::api::runtime::scope_stack_snapshot(&scope.scope_stack).map(Some),
        None => nemo_relay::api::runtime::current_scope_stack_snapshot(),
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
            inner: scope.scope_stack,
            publication_buffer: scope.publication_buffer,
        });
    }
    Ok(ScopeStack::from(current_scope_stack_handle()))
}

/// Binds a scope stack to the current thread or async resource.
///
/// This mutates the current execution resource. Use `withScopeStack` when
/// concurrent asynchronous branches need isolated stack replacements.
#[napi(catch_unwind)]
pub fn set_thread_scope_stack(env: Env, stack: &ScopeStack) -> Result<()> {
    let scope = CallbackScope {
        scope_stack: stack.inner.clone(),
        publication_buffer: effective_scope(&env)?
            .publication_buffer
            .or_else(|| stack.publication_buffer.clone()),
        propagation: None,
    };
    if callback_factory::set_callback_scope(&env, scope)? {
        return Ok(());
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

/// Internal test helper: invoke a closed JS tool callback wrapper and return the fallback value.
#[napi(js_name = "__testClosedToolCallback", catch_unwind)]
pub fn test_closed_tool_callback<'env>(
    env: &'env Env,
    #[napi(ts_arg_type = "(arg0: string, arg1: Json) => any")] callback: Callback,
    name: String,
    args: Json,
) -> Result<PromiseRaw<'env, Json>> {
    clear_recorded_callback_error();
    let tsfn = middleware_tool_callback_tsfn(env, &callback)?;
    abort_tsfn(&tsfn)?;
    let wrapped = callable::wrap_js_tool_fn(tsfn);
    let fallback = args.clone();
    spawn_promise::<_, _, Error, _, _>(
        env,
        async move {
            match wrapped(name, args).await {
                Ok(value) => Ok(value),
                Err(error) => {
                    record_callback_error(error.to_string());
                    Ok(fallback)
                }
            }
        },
        js_value,
    )
}

/// Internal test helper: model a closed JS LLM request sanitizer.
#[napi(js_name = "__testClosedLlmSanitizeRequestCallback", catch_unwind)]
pub fn test_closed_llm_sanitize_request_callback(
    env: &Env,
    #[napi(ts_arg_type = "(arg: Json) => any")] callback: Callback,
    request: Json,
) -> Result<Option<Json>> {
    clear_recorded_callback_error();
    abort_tsfn(&middleware_json_callback_tsfn(env, &callback)?)?;
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| Error::invalid_argument(format!("invalid LlmRequest: {e}")))?;
    drop(llm_request);
    record_callback_error("nemo_relay: failed to queue JS LLM sanitize request callback");
    Ok(None)
}

/// Internal test helper: model a closed JS LLM response sanitizer.
#[napi(js_name = "__testClosedLlmResponseCallback", catch_unwind)]
pub fn test_closed_llm_response_callback(
    env: &Env,
    #[napi(ts_arg_type = "(arg: Json) => any")] callback: Callback,
    response: Json,
) -> Result<Option<Json>> {
    clear_recorded_callback_error();
    abort_tsfn(&middleware_json_callback_tsfn(env, &callback)?)?;
    drop(response);
    record_callback_error("nemo_relay: failed to queue JS LLM sanitize response callback");
    Ok(None)
}

/// Resolve a stream callback future whose queue step already failed.
fn settled_now<T>(mut future: Pin<Box<dyn Future<Output = T> + Send>>) -> Result<T> {
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
    let callback = callable::js_collector_tsfn(&env, &callback)?.without_direct();
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
    let callback = callable::js_finalizer_tsfn(&env, &callback)?.without_direct();
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
    let promise_aware = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(env, &func)
            .map_err(|e| Error::generic_failure(format!("failed to create PromiseAwareFn: {e}")))?,
    );
    if !force_conversion_failure.unwrap_or(false) {
        promise_aware.close()?;
    }

    spawn_promise(
        env,
        async move {
            if force_conversion_failure.unwrap_or(false) {
                promise_aware
                    .call_with_arg0(Box::new(|_| {
                        Err(napi::Error::from_reason(
                            "forced PromiseAwareFn conversion failure",
                        ))
                    }))
                    .await
            } else {
                promise_aware.call(Json::Null).await
            }
            .map_err(Error::from)
        },
        js_value,
    )
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

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
#[napi(catch_unwind)]
#[allow(clippy::too_many_arguments)]
pub fn push_scope(
    env: Env,
    name: String,
    scope_type: ScopeType,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    input: Option<Json>,
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
#[napi(catch_unwind)]
pub fn pop_scope(
    env: Env,
    handle: &ScopeHandle,
    output: Option<Json>,
    timestamp: Option<f64>,
    metadata: Option<Json>,
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
#[napi(ts_return_type = "Promise<unknown>", catch_unwind)]
pub fn with_scope<'env>(
    env: &'env Env,
    name: String,
    scope_type: ScopeType,
    #[napi(ts_arg_type = "(...args: any[]) => any")] callback: Callback,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    input: Option<Json>,
) -> Result<PromiseRaw<'env, Json>> {
    let attrs = ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(env)?;
    let scope_handle = with_scope_stack_handle(scope_stack.clone(), || {
        with_nested_publication_buffer(publication_buffer.clone(), || {
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
    })
    .map(ScopeHandle::from)
    .map_err(Error::from)?;

    let scope_uuid = scope_handle.inner.uuid;
    // Hand the callback a real `ScopeHandle` instance, matching the Rust,
    // Python bindings, so it can be passed back into `event`,
    // `toolCallExecute`, and `llmCallExecute`. The instance is materialized on
    // the JS thread because a `napi_wrap`'d handle cannot cross the
    // threadsafe-function boundary as plain JSON.
    let callback_handle = scope_handle.inner.clone();

    // Create a promise-aware wrapper so we handle both sync and async callbacks.
    let error_publication_buffer = publication_buffer.clone();
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(env, &callback).map_err(|e| {
            let status_message = format!("failed to create PromiseAwareFn: {e}");
            let _ = with_scope_stack_handle(scope_stack.clone(), || {
                with_nested_publication_buffer(error_publication_buffer.clone(), || {
                    core_scope_api::pop_scope(
                        core_scope_api::PopScopeParams::builder()
                            .handle_uuid(&scope_uuid)
                            .metadata_opt(Some(otel_status_metadata(
                                "ERROR",
                                Some(status_message.clone()),
                            )))
                            .build(),
                    )
                })
            });
            Error::generic_failure(status_message)
        })?,
    );

    spawn_promise(
        env,
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            let build_handle: crate::promise_call::Arg0Builder =
                                Box::new(move |env: &Env| {
                                    js_value(env, ScopeHandle::from(callback_handle))
                                });

                            let result = pa_fn.call_with_arg0(build_handle).await;
                            let metadata = match &result {
                                Ok(_) => otel_status_metadata("OK", None),
                                Err(error) => {
                                    otel_status_metadata("ERROR", Some(error.to_string()))
                                }
                            };
                            // Always pop the scope, even on error.
                            let _ = core_scope_api::pop_scope(
                                core_scope_api::PopScopeParams::builder()
                                    .handle_uuid(&scope_uuid)
                                    .metadata_opt(Some(metadata))
                                    .build(),
                            );
                            result.map_err(Error::from)
                        })
                        .await
                },
            )
            .await
        },
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
#[napi(catch_unwind)]
#[allow(clippy::too_many_arguments)]
pub fn event(
    env: Env,
    name: String,
    handle: Option<&ScopeHandle>,
    data: Option<Json>,
    metadata: Option<Json>,
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
#[napi(catch_unwind)]
pub fn metric(
    env: Env,
    name: String,
    measurements: Vec<MetricMeasurement>,
    handle: Option<&ScopeHandle>,
    metadata: Option<Json>,
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

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

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
#[napi(catch_unwind)]
#[allow(clippy::too_many_arguments)]
pub fn tool_call(
    env: Env,
    name: String,
    args: Json,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
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
#[napi(catch_unwind)]
pub fn tool_call_end(
    env: Env,
    handle: &ToolHandle,
    result: ToolExecutionResult,
    data: Option<Json>,
    metadata: Option<Json>,
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
/// Runs conditional-execution guardrails (on raw args) → request intercepts →
/// sanitize-request guardrails for the emitted `Start` event payload →
/// execution intercepts → `func` → sanitize-response guardrails for the emitted
/// `End` event payload. On rejection, only a standalone Mark event is emitted
/// (no Start/End pair) and `GuardrailRejected` is returned. Returns the final
/// execution result; sanitize guardrails do not rewrite the caller-visible value.
/// An optional trailing `toolCallId` is recorded in both lifecycle events.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<ToolExecutionResult>", catch_unwind)]
pub fn tool_call_execute<'env>(
    env: &'env Env,
    name: String,
    args: Json,
    #[napi(ts_arg_type = "(arg: Json) => ToolExecutionResult")] func: Callback,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    tool_call_id: Option<String>,
) -> Result<PromiseRaw<'env, ToolExecutionResult>> {
    let attrs = ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(env)?;
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(|| effective_scope_top(&scope_stack));
    let callback: Function<JsArg, JsArg> = callback_factory::wrap_execution_callback(env, &func)?;
    let default_fn = scoped_tool_execution_fn(scoped_json_callback_tsfn(env, &callback)?);

    spawn_promise(
        env,
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            core_tool_api::tool_call_execute(
                                core_tool_api::ToolCallExecuteParams::builder()
                                    .name(name)
                                    .args(args)
                                    .func(default_fn)
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
                        })
                        .await
                },
            )
            .await
        },
        js_value,
    )
}

/// Execute a tool call end-to-end, supporting both sync and async (Promise-returning) callbacks.
///
/// Same lifecycle as `toolCallExecute` (guardrails → intercepts → func → response processing),
/// but transparently handles JS callbacks that return Promises. Uses `napi_is_promise` to detect
/// Promise return values and resolves them before continuing the pipeline.
/// An optional trailing `toolCallId` is recorded in both lifecycle events.
///
/// Accepts a raw `Function` instead of `ThreadsafeFunction` so it can create a
/// promise-aware wrapper with access to `Env`.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<ToolExecutionResult>", catch_unwind)]
pub fn tool_call_execute_async<'env>(
    env: &'env Env,
    name: String,
    args: Json,
    #[napi(
        ts_arg_type = "(arg: Json, signal: AbortSignal) => ToolExecutionResult | Promise<ToolExecutionResult>"
    )]
    func: Callback,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    tool_call_id: Option<String>,
) -> Result<PromiseRaw<'env, ToolExecutionResult>> {
    let attrs = ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(env)?;
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(|| effective_scope_top(&scope_stack));

    // Create promise-aware wrapper — this must happen on the JS thread (we have Env).
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(env, &func)
            .map_err(|e| Error::generic_failure(format!("failed to create PromiseAwareFn: {e}")))?,
    );

    let exec_fn: ToolExecutionNextFn = std::sync::Arc::new(move |args| {
        let pa_fn = pa_fn.clone();
        Box::pin(async move {
            let result = pa_fn.call(args).await?;
            callable::parse_tool_execution_result(result)
        })
    });

    spawn_promise(
        env,
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            core_tool_api::tool_call_execute(
                                core_tool_api::ToolCallExecuteParams::builder()
                                    .name(name)
                                    .args(args)
                                    .func(exec_fn)
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
                        })
                        .await
                },
            )
            .await
        },
        js_value,
    )
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

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
#[napi(catch_unwind)]
pub fn llm_call(
    env: Env,
    name: String,
    request: Json,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    timestamp: Option<f64>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_decode: Option<Callback>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_encode: Option<Callback>,
) -> Result<LlmHandle> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let timestamp = parse_timestamp_micros(timestamp)?;
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| Error::invalid_argument(format!("invalid LlmRequest: {e}")))?;
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
#[napi(catch_unwind)]
pub fn llm_call_end(
    env: Env,
    handle: &LlmHandle,
    response: Json,
    data: Option<Json>,
    metadata: Option<Json>,
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
/// Runs conditional-execution guardrails (on raw request) → request intercepts →
/// sanitize-request guardrails for the emitted `Start` event payload →
/// execution intercepts → `func` → sanitize-response guardrails for the emitted
/// `End` event payload. On rejection, only a standalone Mark event is emitted
/// (no Start/End pair) and `GuardrailRejected` is returned. The `request`
/// should be a JSON object with `headers` and `content` fields matching the
/// `LlmRequest` schema. Returns the final execution response; sanitize
/// guardrails do not rewrite the caller-visible value.
/// Supply `codecDecode` and `codecEncode` together or omit both.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<unknown>", catch_unwind)]
pub fn llm_call_execute<'env>(
    env: &'env Env,
    name: String,
    request: Json,
    #[napi(ts_arg_type = "(arg: Json) => any")] func: Callback,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_decode: Option<Callback>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_encode: Option<Callback>,
    #[napi(ts_arg_type = "(arg: Json) => any")] response_codec_decode: Option<Callback>,
) -> Result<PromiseRaw<'env, Json>> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(env)?;
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(|| effective_scope_top(&scope_stack));
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| Error::invalid_argument(format!("invalid LlmRequest: {e}")))?;
    let callback: Function<JsArg, JsArg> = callback_factory::wrap_execution_callback(env, &func)?;
    let default_fn = scoped_llm_execution_fn(scoped_json_callback_tsfn(env, &callback)?);
    let codec = node_llm_codec(env, codec_decode.as_deref(), codec_encode.as_deref())?;
    let response_codec = node_llm_response_codec(env, response_codec_decode.as_deref())?;
    spawn_promise(
        env,
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            let params = core_llm_api::LlmCallExecuteParams::builder()
                                .name(name)
                                .request(llm_request)
                                .func(default_fn)
                                .parent(parent)
                                .attributes(attrs)
                                .data_opt(opt_json(data))
                                .metadata_opt(opt_json(metadata))
                                .model_name_opt(model_name)
                                .codec_opt(codec)
                                .response_codec_opt(response_codec)
                                .build();
                            // Construct the large core future when Tokio polls this
                            // bridge, keeping it out of the JavaScript caller's stack.
                            Box::pin(core_llm_api::llm_call_execute(params))
                                .await
                                .map_err(Error::from)
                        })
                        .await
                },
            )
            .await
        },
        js_value,
    )
}

/// Execute an LLM call end-to-end, supporting both sync and async (Promise-returning) callbacks.
///
/// Same lifecycle as `llmCallExecute` (guardrails → intercepts → func → response processing),
/// but transparently handles JS callbacks that return Promises.
/// Supply `codecDecode` and `codecEncode` together or omit both.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<unknown>", catch_unwind)]
pub fn llm_call_execute_async<'env>(
    env: &'env Env,
    name: String,
    request: Json,
    #[napi(ts_arg_type = "(...args: any[]) => any")] func: Callback,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_decode: Option<Callback>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_encode: Option<Callback>,
    #[napi(ts_arg_type = "(arg: Json) => any")] response_codec_decode: Option<Callback>,
) -> Result<PromiseRaw<'env, Json>> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(env)?;
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(|| effective_scope_top(&scope_stack));
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| Error::invalid_argument(format!("invalid LlmRequest: {e}")))?;
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(env, &func)
            .map_err(|e| Error::generic_failure(format!("failed to create PromiseAwareFn: {e}")))?,
    );

    let exec_fn: LlmExecutionNextFn = std::sync::Arc::new(move |req| {
        let pa_fn = pa_fn.clone();
        let req_json = serde_json::to_value(&req).unwrap_or(Json::Null);
        Box::pin(async move { pa_fn.call(req_json).await })
    });

    let codec = node_llm_codec(env, codec_decode.as_deref(), codec_encode.as_deref())?;
    let response_codec = node_llm_response_codec(env, response_codec_decode.as_deref())?;

    spawn_promise(
        env,
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            let params = core_llm_api::LlmCallExecuteParams::builder()
                                .name(name)
                                .request(llm_request)
                                .func(exec_fn)
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
                        })
                        .await
                },
            )
            .await
        },
        js_value,
    )
}

/// Execute a streaming LLM call end-to-end with full lifecycle management.
///
/// Like `llmCallExecute`, conditional-execution guardrails run first on the raw request.
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
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<LlmStream>", catch_unwind)]
pub fn llm_stream_call_execute<'env>(
    env: &'env Env,
    name: String,
    request: Json,
    #[napi(
        ts_arg_type = "(request: Json, signal: AbortSignal) => AsyncIterable<Json> | Promise<AsyncIterable<Json>>"
    )]
    func: Callback,
    #[napi(ts_arg_type = "(chunk: Json) => void | Promise<void>")] collector: Option<Callback>,
    #[napi(ts_arg_type = "() => Json | Promise<Json>")] finalizer: Option<Callback>,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_decode: Option<Callback>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_encode: Option<Callback>,
    #[napi(ts_arg_type = "(arg: Json) => any")] response_codec_decode: Option<Callback>,
) -> Result<PromiseRaw<'env, LlmStream>> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(env)?;
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(|| effective_scope_top(&scope_stack));
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| Error::invalid_argument(format!("invalid LlmRequest: {e}")))?;

    let wrapped_collector = match collector {
        Some(cb) => callable::wrap_js_collector_fn(callable::js_collector_tsfn(env, &cb)?),
        None => LlmCollector::from(|_: Json| Ok(())),
    };

    let finalizer_error = callable::FinalizerError::default();
    let wrapped_finalizer = match finalizer {
        Some(cb) => callable::wrap_js_finalizer_fn(
            callable::js_finalizer_tsfn(env, &cb)?,
            finalizer_error.clone(),
        ),
        None => LlmFinalizer::from(|| Json::Null),
    };

    let provider = Arc::new(PromiseAwareFn::new(env, &func)?);
    let producers = crate::promise_call::StreamCancellation::default();
    let default_fn = callable::wrap_js_llm_stream_fn(provider, producers.clone());

    let codec = node_llm_codec(env, codec_decode.as_deref(), codec_encode.as_deref())?;
    let response_codec = node_llm_response_codec(env, response_codec_decode.as_deref())?;
    spawn_promise::<_, _, Error, _, _>(
        env,
        async move {
            with_publication_callback_context(
                publication_context_id.clone(),
                publication_buffer.clone(),
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            let params = core_llm_api::LlmStreamCallExecuteParams::builder()
                                .name(name)
                                .request(llm_request)
                                .func(default_fn)
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
                            let rust_stream =
                                Box::pin(core_llm_api::llm_stream_call_execute(params))
                                    .await
                                    .map_err(Error::from)?;

                            Ok(LlmStream::new(
                                rust_stream,
                                finalizer_error,
                                producers,
                                publication_context_id,
                                publication_buffer,
                            ))
                        })
                        .await
                },
            )
            .await
        },
        js_value,
    )
}

// ---------------------------------------------------------------------------
// Event metadata injector and event guardrail registrations
// ---------------------------------------------------------------------------

/// Register a global event metadata injector.
///
/// The callback receives an immutable event snapshot and may return additions
/// directly or in a Promise. Relay validates and merges accepted additions
/// before event sanitizers run.
#[napi(catch_unwind)]
pub fn register_event_metadata_injector(
    env: Env,
    name: String,
    priority: i32,
    #[napi(ts_arg_type = "(event: Json) => EventMetadata | Promise<EventMetadata>")]
    injector: Callback,
) -> Result<()> {
    core_registry_api::register_event_metadata_injector(
        &name,
        priority,
        node_event_metadata_injector_fn(&env, &injector)?,
    )
    .map_err(Error::from)
}

/// Deregister a global event metadata injector by name.
#[napi(catch_unwind)]
pub fn deregister_event_metadata_injector(name: String) -> Result<bool> {
    core_registry_api::deregister_event_metadata_injector(&name).map_err(Error::from)
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

impl From<RuntimeRegistrationKind> for core_registry_api::RuntimeRegistrationKind {
    fn from(kind: RuntimeRegistrationKind) -> Self {
        match kind {
            RuntimeRegistrationKind::Subscriber => Self::Subscriber,
            RuntimeRegistrationKind::EventMetadataInjector => Self::EventMetadataInjector,
            RuntimeRegistrationKind::MarkSanitizeGuardrail => Self::MarkSanitizeGuardrail,
            RuntimeRegistrationKind::ScopeSanitizeStartGuardrail => {
                Self::ScopeSanitizeStartGuardrail
            }
            RuntimeRegistrationKind::ScopeSanitizeEndGuardrail => Self::ScopeSanitizeEndGuardrail,
            RuntimeRegistrationKind::ToolSanitizeRequestGuardrail => {
                Self::ToolSanitizeRequestGuardrail
            }
            RuntimeRegistrationKind::ToolSanitizeResponseGuardrail => {
                Self::ToolSanitizeResponseGuardrail
            }
            RuntimeRegistrationKind::ToolConditionalExecutionGuardrail => {
                Self::ToolConditionalExecutionGuardrail
            }
            RuntimeRegistrationKind::ToolRequestIntercept => Self::ToolRequestIntercept,
            RuntimeRegistrationKind::ToolExecutionIntercept => Self::ToolExecutionIntercept,
            RuntimeRegistrationKind::LlmSanitizeRequestGuardrail => {
                Self::LlmSanitizeRequestGuardrail
            }
            RuntimeRegistrationKind::LlmSanitizeResponseGuardrail => {
                Self::LlmSanitizeResponseGuardrail
            }
            RuntimeRegistrationKind::LlmConditionalExecutionGuardrail => {
                Self::LlmConditionalExecutionGuardrail
            }
            RuntimeRegistrationKind::LlmRequestIntercept => Self::LlmRequestIntercept,
            RuntimeRegistrationKind::LlmExecutionIntercept => Self::LlmExecutionIntercept,
            RuntimeRegistrationKind::LlmStreamExecutionIntercept => {
                Self::LlmStreamExecutionIntercept
            }
        }
    }
}

impl From<core_registry_api::RuntimeRegistrationKind> for RuntimeRegistrationKind {
    fn from(kind: core_registry_api::RuntimeRegistrationKind) -> Self {
        match kind {
            core_registry_api::RuntimeRegistrationKind::Subscriber => Self::Subscriber,
            core_registry_api::RuntimeRegistrationKind::EventMetadataInjector => {
                Self::EventMetadataInjector
            }
            core_registry_api::RuntimeRegistrationKind::MarkSanitizeGuardrail => {
                Self::MarkSanitizeGuardrail
            }
            core_registry_api::RuntimeRegistrationKind::ScopeSanitizeStartGuardrail => {
                Self::ScopeSanitizeStartGuardrail
            }
            core_registry_api::RuntimeRegistrationKind::ScopeSanitizeEndGuardrail => {
                Self::ScopeSanitizeEndGuardrail
            }
            core_registry_api::RuntimeRegistrationKind::ToolSanitizeRequestGuardrail => {
                Self::ToolSanitizeRequestGuardrail
            }
            core_registry_api::RuntimeRegistrationKind::ToolSanitizeResponseGuardrail => {
                Self::ToolSanitizeResponseGuardrail
            }
            core_registry_api::RuntimeRegistrationKind::ToolConditionalExecutionGuardrail => {
                Self::ToolConditionalExecutionGuardrail
            }
            core_registry_api::RuntimeRegistrationKind::ToolRequestIntercept => {
                Self::ToolRequestIntercept
            }
            core_registry_api::RuntimeRegistrationKind::ToolExecutionIntercept => {
                Self::ToolExecutionIntercept
            }
            core_registry_api::RuntimeRegistrationKind::LlmSanitizeRequestGuardrail => {
                Self::LlmSanitizeRequestGuardrail
            }
            core_registry_api::RuntimeRegistrationKind::LlmSanitizeResponseGuardrail => {
                Self::LlmSanitizeResponseGuardrail
            }
            core_registry_api::RuntimeRegistrationKind::LlmConditionalExecutionGuardrail => {
                Self::LlmConditionalExecutionGuardrail
            }
            core_registry_api::RuntimeRegistrationKind::LlmRequestIntercept => {
                Self::LlmRequestIntercept
            }
            core_registry_api::RuntimeRegistrationKind::LlmExecutionIntercept => {
                Self::LlmExecutionIntercept
            }
            core_registry_api::RuntimeRegistrationKind::LlmStreamExecutionIntercept => {
                Self::LlmStreamExecutionIntercept
            }
        }
    }
}

/// The owner category reported for a global runtime registration.
#[napi(string_enum = "snake_case")]
pub enum RuntimeRegistrationOwnerKind {
    Core,
    GlobalApi,
    Plugin,
}

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

impl From<core_registry_api::RuntimeRegistrationIdentity> for RuntimeRegistrationIdentity {
    fn from(identity: core_registry_api::RuntimeRegistrationIdentity) -> Self {
        let owner_kind = match identity.owner.kind {
            core_registry_api::RuntimeRegistrationOwnerKind::Core => {
                RuntimeRegistrationOwnerKind::Core
            }
            core_registry_api::RuntimeRegistrationOwnerKind::GlobalApi => {
                RuntimeRegistrationOwnerKind::GlobalApi
            }
            core_registry_api::RuntimeRegistrationOwnerKind::Plugin => {
                RuntimeRegistrationOwnerKind::Plugin
            }
        };
        Self {
            kind: identity.kind.into(),
            local_name: identity.local_name,
            effective_name: identity.effective_name,
            owner: RuntimeRegistrationOwner {
                kind: owner_kind,
                plugin_kind: identity.owner.plugin_kind,
                component_ordinal: identity.owner.component_ordinal,
            },
        }
    }
}

/// Register a global eligibility gate for runtime registrations matching kind and name.
#[napi(catch_unwind)]
pub fn register_conditional_middleware_guardrail(
    env: Env,
    name: String,
    kinds: Vec<RuntimeRegistrationKind>,
    registration_name: String,
    #[napi(
        ts_arg_type = "(kinds: RuntimeRegistrationKind[], registrationName: string) => string | null"
    )]
    guardrail: Callback,
) -> Result<()> {
    let selected = kinds.into_iter().map(Into::into).collect::<BTreeSet<_>>();
    let callback = node_conditional_middleware_guardrail(&env, &guardrail)?;
    core_registry_api::register_conditional_middleware_guardrail(
        &name,
        selected,
        &registration_name,
        Arc::new(move |kinds, effective_name| {
            callback
                .call(
                    kinds.iter().map(|kind| kind.as_str().to_string()).collect(),
                    effective_name.to_string(),
                )
                .unwrap_or_else(|error| {
                    record_callback_error(format!(
                        "Node conditional middleware guardrail failed open: {error}"
                    ));
                    None
                })
        }),
    )
    .map_err(Error::from)
}

/// Deregister a global runtime-registration eligibility gate by name.
#[napi(catch_unwind)]
pub fn deregister_conditional_middleware_guardrail(name: String) -> Result<bool> {
    core_registry_api::deregister_conditional_middleware_guardrail(&name).map_err(Error::from)
}

/// List global gateable runtime registrations.
#[napi(catch_unwind)]
pub fn list_runtime_registrations(
    kinds: Option<Vec<RuntimeRegistrationKind>>,
) -> Result<Vec<RuntimeRegistrationIdentity>> {
    let selected = kinds.map(|kinds| kinds.into_iter().map(Into::into).collect::<BTreeSet<_>>());
    let registrations =
        core_registry_api::list_runtime_registrations(selected.as_ref()).map_err(Error::from)?;
    Ok(registrations.into_iter().map(Into::into).collect())
}

macro_rules! napi_event_guardrail_api {
    ($register_name:ident, $deregister_name:ident, $core_register:path, $core_deregister:path) => {
        /// Register an event sanitize guardrail.
        ///
        /// The callback may return fields directly or in a Promise. Scope and mark
        /// calls queue the event and return synchronously; publication resumes after
        /// the Promise settles. Callback, serialization, conversion, or invalid-result
        /// failures clear the emitted event fields and record the error for
        /// `getLastCallbackError()`. Await `flushSubscribers()` before inspecting
        /// either the delivered event or that error.
        #[napi(catch_unwind)]
        pub fn $register_name(
            env: Env,
            name: String,
            priority: i32,
            #[napi(
                ts_arg_type = "(event: Json, fields: EventSanitizeFields) => EventSanitizeFields | Promise<EventSanitizeFields>"
            )]
            guardrail: Callback,
        ) -> Result<()> {
            $core_register(&name, priority, node_event_sanitize_fn(&env, &guardrail)?)
                .map_err(Error::from)
        }

        #[napi(catch_unwind)]
        pub fn $deregister_name(name: String) -> Result<bool> {
            $core_deregister(&name).map_err(Error::from)
        }
    };
}

napi_event_guardrail_api!(
    register_mark_sanitize_guardrail,
    deregister_mark_sanitize_guardrail,
    core_registry_api::register_mark_sanitize_guardrail,
    core_registry_api::deregister_mark_sanitize_guardrail
);
napi_event_guardrail_api!(
    register_scope_sanitize_start_guardrail,
    deregister_scope_sanitize_start_guardrail,
    core_registry_api::register_scope_sanitize_start_guardrail,
    core_registry_api::deregister_scope_sanitize_start_guardrail
);
napi_event_guardrail_api!(
    register_scope_sanitize_end_guardrail,
    deregister_scope_sanitize_end_guardrail,
    core_registry_api::register_scope_sanitize_end_guardrail,
    core_registry_api::deregister_scope_sanitize_end_guardrail
);

// Written as `#[doc]` attributes: a `///` comment forwarded through `macro_rules!` reaches
// napi-derive as a raw string literal and leaks `r"` into generated declarations.
macro_rules! napi_guardrail_tool_api {
    ($(#[doc = $reg_doc:expr_2021])* $register_name:ident,
     $(#[doc = $dereg_doc:expr_2021])* $deregister_name:ident,
     $core_register:path, $core_deregister:path) => {
        $(#[doc = $reg_doc])*
        #[napi(catch_unwind)]
        pub fn $register_name(
            env: Env,
            name: String,
            priority: i32,
            #[napi(
                ts_arg_type = "(toolName: string, value: Json) => Json | Promise<Json>"
            )]
            guardrail: Callback,
        ) -> Result<()> {
            let callback = Arc::new(PromiseAwareFn::new(&env, &guardrail)?);
            $core_register(
                &name,
                priority,
                callable::wrap_js_tool_sanitize_promise_fn(callback),
            )
            .map_err(Error::from)
        }

        $(#[doc = $dereg_doc])*
        #[napi(catch_unwind)]
        pub fn $deregister_name(name: String) -> Result<bool> {
            $core_deregister(&name).map_err(Error::from)
        }
    };
}

napi_guardrail_tool_api!(
    #[doc = " Register a guardrail that sanitizes tool request arguments before execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(toolName, args)` and must return sanitized args."]
    #[doc = " Higher `priority` values run first. Throws if a guardrail with the same `name` already exists."]
    #[doc = " If the callback throws, Relay omits the emitted payload and records the error"]
    #[doc = " for `getLastCallbackError()`."]
    register_tool_sanitize_request_guardrail,
    #[doc = " Deregister a tool request sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed."]
    deregister_tool_sanitize_request_guardrail,
    core_registry_api::register_tool_sanitize_request_guardrail,
    core_registry_api::deregister_tool_sanitize_request_guardrail
);

napi_guardrail_tool_api!(
    #[doc = " Register a guardrail that sanitizes tool response data after execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(toolName, result)` and must return sanitized result."]
    #[doc = " Higher `priority` values run first. Throws if a guardrail with the same `name` already exists."]
    #[doc = " If the callback throws, Relay omits the emitted payload and records the error"]
    #[doc = " for `getLastCallbackError()`."]
    register_tool_sanitize_response_guardrail,
    #[doc = " Deregister a tool response sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed."]
    deregister_tool_sanitize_response_guardrail,
    core_registry_api::register_tool_sanitize_response_guardrail,
    core_registry_api::deregister_tool_sanitize_response_guardrail
);

/// Register a guardrail that conditionally gates tool execution.
///
/// The `guardrail` callback receives `(toolName, args)` and must return `null` to allow
/// execution or a rejection reason string to block it. Higher `priority` values run first.
/// If the callback throws, the managed call rejects and the protected callback does not run.
#[napi(catch_unwind)]
pub fn register_tool_conditional_execution_guardrail(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(toolName: string, args: Json) => string | null | Promise<string | null>"
    )]
    guardrail: Callback,
) -> Result<()> {
    let callback = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &guardrail).map_err(|error| {
            Error::generic_failure(format!("failed to create PromiseAwareFn: {error}"))
        })?,
    );
    core_registry_api::register_tool_conditional_execution_guardrail(
        &name,
        priority,
        callable::wrap_js_tool_conditional_promise_fn(callback),
    )
    .map_err(Error::from)
}

/// Deregister a tool conditional execution guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed.
#[napi(catch_unwind)]
pub fn deregister_tool_conditional_execution_guardrail(name: String) -> Result<bool> {
    core_registry_api::deregister_tool_conditional_execution_guardrail(&name).map_err(Error::from)
}

// ---------------------------------------------------------------------------
// Tool intercept registrations
// ---------------------------------------------------------------------------

macro_rules! napi_intercept_tool_api {
    ($(#[doc = $reg_doc:expr_2021])* $register_name:ident,
     $(#[doc = $dereg_doc:expr_2021])* $deregister_name:ident,
     $core_register:path, $core_deregister:path) => {
        $(#[doc = $reg_doc])*
        #[napi(catch_unwind)]
        pub fn $register_name(
            env: Env,
            name: String,
            priority: i32,
            break_chain: bool,
            #[napi(
                ts_arg_type = "(toolName: string, args: Json) => Json | Promise<Json>"
            )]
            callable: Callback,
        ) -> Result<()> {
            let callback = std::sync::Arc::new(
                crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|error| {
                    Error::generic_failure(format!("failed to create PromiseAwareFn: {error}"))
                })?,
            );
            $core_register(
                &name,
                priority,
                break_chain,
                callable::wrap_js_tool_request_intercept_promise_fn(callback),
            )
            .map_err(Error::from)
        }

        $(#[doc = $dereg_doc])*
        #[napi(catch_unwind)]
        pub fn $deregister_name(name: String) -> Result<bool> {
            $core_deregister(&name).map_err(Error::from)
        }
    };
}

napi_intercept_tool_api!(
    #[doc = " Register an intercept that transforms tool request arguments."]
    #[doc = ""]
    #[doc = " The `callable` receives `(toolName, args)` and returns transformed args. If `breakChain`"]
    #[doc = " is `true`, no lower-priority intercepts run after this one. Higher `priority` values run first."]
    #[doc = " If the callback throws, the managed call rejects and later middleware does not run."]
    register_tool_request_intercept,
    #[doc = " Deregister a tool request intercept by name."]
    #[doc = ""]
    #[doc = " Returns `true` if an intercept with that name was found and removed."]
    deregister_tool_request_intercept,
    core_registry_api::register_tool_request_intercept,
    core_registry_api::deregister_tool_request_intercept
);

/// Register a tool execution intercept following the middleware chain pattern.
///
/// The `callable` receives `ToolExecutionContext` and `next`. The context
/// exposes `toolName`, `args`, and `toolCallId`; `toolCallId` is `null` when
/// the managed call has no provider-issued correlation identifier. Call
/// `next(context.args)` to continue the chain, or return an outcome directly
/// to short-circuit execution.
#[napi(catch_unwind)]
pub fn register_tool_execution_intercept(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(context: ToolExecutionContext, next: (args: Json) => ToolExecutionResult | Promise<ToolExecutionResult>) => { result: Json; annotation?: Json; pendingMarks?: Array<PendingMarkSpec> } | Promise<{ result: Json; annotation?: Json; pendingMarks?: Array<PendingMarkSpec> }>"
    )]
    callable: Callback,
) -> Result<()> {
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable)
            .map_err(|e| Error::generic_failure(format!("failed to create PromiseAwareFn: {e}")))?,
    );
    core_registry_api::register_tool_execution_intercept(
        &name,
        priority,
        callable::wrap_js_tool_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(Error::from)?;
    Ok(())
}

/// Deregister a tool execution intercept by name.
///
/// Removes an intercept registered through either registration shape. Returns
/// `true` if an intercept with that name was found and removed.
#[napi(catch_unwind)]
pub fn deregister_tool_execution_intercept(name: String) -> Result<bool> {
    core_registry_api::deregister_tool_execution_intercept(&name).map_err(Error::from)
}

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register a guardrail that sanitizes LLM request data before execution.
///
/// The `guardrail` callback receives `(request, context)` and must return the sanitized request,
/// or `null` to omit the observability payload. Lower `priority` values run first. Throws if a
/// guardrail with the same `name` already exists. If the callback throws, Relay omits the payload
/// and annotation, continues publication, and records the error for `getLastCallbackError()`.
#[napi(catch_unwind)]
pub fn register_llm_sanitize_request_guardrail(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, context: LlmSanitizeRequestContext) => Json | null | Promise<Json | null>"
    )]
    guardrail: Callback,
) -> Result<()> {
    core_registry_api::register_llm_sanitize_request_guardrail(
        &name,
        priority,
        callable::wrap_js_llm_sanitize_request_promise_fn(Arc::new(PromiseAwareFn::new(
            &env, &guardrail,
        )?)),
    )
    .map_err(Error::from)
}

/// Deregister an LLM request sanitization guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed.
#[napi(catch_unwind)]
pub fn deregister_llm_sanitize_request_guardrail(name: String) -> Result<bool> {
    core_registry_api::deregister_llm_sanitize_request_guardrail(&name).map_err(Error::from)
}

/// Register a guardrail that sanitizes LLM response data after execution.
///
/// The `guardrail` callback receives `(response, context)` and must return the sanitized response,
/// or `null` to omit the observability payload. Lower `priority` values run first. Throws if a
/// guardrail with the same `name` already exists. If the callback throws, Relay omits the payload
/// and annotation, continues publication, and records the error for `getLastCallbackError()`.
#[napi(catch_unwind)]
pub fn register_llm_sanitize_response_guardrail(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(response: Json, context: LlmSanitizeResponseContext) => Json | null | Promise<Json | null>"
    )]
    guardrail: Callback,
) -> Result<()> {
    core_registry_api::register_llm_sanitize_response_guardrail(
        &name,
        priority,
        callable::wrap_js_llm_sanitize_response_promise_fn(Arc::new(PromiseAwareFn::new(
            &env, &guardrail,
        )?)),
    )
    .map_err(Error::from)
}

/// Deregister an LLM response sanitization guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed.
#[napi(catch_unwind)]
pub fn deregister_llm_sanitize_response_guardrail(name: String) -> Result<bool> {
    core_registry_api::deregister_llm_sanitize_response_guardrail(&name).map_err(Error::from)
}

/// Register a guardrail that conditionally gates LLM execution.
///
/// The `guardrail` callback receives the LLM request as JSON and must return `null` to allow
/// execution or a rejection reason string to block it. Higher `priority` values run first.
/// If the callback throws, the managed call rejects and the protected callback does not run.
#[napi(catch_unwind)]
pub fn register_llm_conditional_execution_guardrail(
    env: Env,
    name: String,
    priority: i32,
    #[napi(ts_arg_type = "(request: Json) => string | null | Promise<string | null>")]
    guardrail: Callback,
) -> Result<()> {
    let callback = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &guardrail).map_err(|error| {
            Error::generic_failure(format!("failed to create PromiseAwareFn: {error}"))
        })?,
    );
    core_registry_api::register_llm_conditional_execution_guardrail(
        &name,
        priority,
        callable::wrap_js_llm_conditional_promise_fn(callback),
    )
    .map_err(Error::from)
}

/// Deregister an LLM conditional execution guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed.
#[napi(catch_unwind)]
pub fn deregister_llm_conditional_execution_guardrail(name: String) -> Result<bool> {
    core_registry_api::deregister_llm_conditional_execution_guardrail(&name).map_err(Error::from)
}

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register an intercept that transforms LLM request data.
///
/// The `callable` receives the `LlmRequest` (as JSON) and returns a transformed request.
/// If `breakChain` is `true`, no lower-priority intercepts run after this one.
/// Higher `priority` values run first.
/// If the callback throws, the managed call rejects and later middleware does not run.
#[napi(catch_unwind)]
pub fn register_llm_request_intercept(
    env: Env,
    name: String,
    priority: i32,
    break_chain: bool,
    #[napi(
        ts_arg_type = "(args: { name: string; request: Json; annotated: Json | null }) => LlmRequestInterceptOutcome | Promise<LlmRequestInterceptOutcome>"
    )]
    callable: Callback,
) -> Result<()> {
    let callback = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|error| {
            Error::generic_failure(format!("failed to create PromiseAwareFn: {error}"))
        })?,
    );
    core_registry_api::register_llm_request_intercept(
        &name,
        priority,
        break_chain,
        callable::wrap_js_llm_request_intercept_promise_fn(callback),
    )
    .map_err(Error::from)
}

/// Deregister an LLM request intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi(catch_unwind)]
pub fn deregister_llm_request_intercept(name: String) -> Result<bool> {
    core_registry_api::deregister_llm_request_intercept(&name).map_err(Error::from)
}

/// Register an LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request, codec context, and a `next` function. Call
/// `next(request)` to invoke the next intercept or original implementation; skip calling
/// `next` to short-circuit the chain. `next` may be called repeatedly or concurrently while
/// `callable` is pending; each call receives an isolated scope-stack branch, and
/// unfinished or later calls reject after `callable` settles.
#[napi(catch_unwind)]
pub fn register_llm_execution_intercept(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, context: LlmExecutionContext, next: (request: Json) => Json | Promise<Json>) => Json | Promise<Json>"
    )]
    callable: Callback,
) -> Result<()> {
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable)
            .map_err(|e| Error::generic_failure(format!("failed to create PromiseAwareFn: {e}")))?,
    );
    core_registry_api::register_llm_execution_intercept(
        &name,
        priority,
        callable::wrap_js_llm_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(Error::from)?;
    Ok(())
}

/// Deregister an LLM execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi(catch_unwind)]
pub fn deregister_llm_execution_intercept(name: String) -> Result<bool> {
    core_registry_api::deregister_llm_execution_intercept(&name).map_err(Error::from)
}

/// Register a streaming LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request, request-codec context, and a `next` function. The
/// response codec is `null` because streaming execution has no complete-response codec.
/// Call `next(request)` to
/// invoke the next intercept or original streaming implementation; in Node the
/// returned promise resolves to a lazy `AsyncIterable`. Return it directly or wrap it
/// `next` to short-circuit the chain. `next` may be called repeatedly or concurrently
/// while `callable` is pending; each call receives an isolated scope-stack branch,
/// and unfinished or later calls reject after the returned interceptor stream settles.
/// A downstream stream returned successfully by `next` keeps its normal lifetime.
#[napi(catch_unwind)]
pub fn register_llm_stream_execution_intercept(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, context: LlmExecutionContext, next: (request: Json) => Promise<AsyncIterable<Json>>) => AsyncIterable<Json> | Promise<AsyncIterable<Json>>"
    )]
    callable: Callback,
) -> Result<()> {
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable)
            .map_err(|e| Error::generic_failure(format!("failed to create PromiseAwareFn: {e}")))?,
    );
    core_registry_api::register_llm_stream_execution_intercept(
        &name,
        priority,
        callable::wrap_js_llm_stream_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(Error::from)?;
    Ok(())
}

/// Deregister an LLM stream execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi(catch_unwind)]
pub fn deregister_llm_stream_execution_intercept(name: String) -> Result<bool> {
    core_registry_api::deregister_llm_stream_execution_intercept(&name).map_err(Error::from)
}

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

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
    #[napi(ts_arg_type = "(event: Json) => void | Promise<void>")] callback: Callback,
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
                core_subscriber_api::flush_subscribers()?;
                callable::flush_js_subscriber_callbacks()
            })
            .await
            .map_err(|error| Error::from(FlowError::Internal(error.to_string())))?
            .map_err(Error::from)
        },
        |env, _| js_value(env, ()),
    )
}

// ---------------------------------------------------------------------------
// Scope-local event metadata injector and event guardrail registrations
// ---------------------------------------------------------------------------

/// Register an event metadata injector owned by an active scope.
#[napi(catch_unwind)]
pub fn scope_register_event_metadata_injector(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(ts_arg_type = "(event: Json) => EventMetadata | Promise<EventMetadata>")]
    injector: Callback,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|error| Error::invalid_argument(format!("invalid UUID: {error}")))?;
    core_registry_api::scope_register_event_metadata_injector(
        &uuid,
        &name,
        priority,
        node_event_metadata_injector_fn(&env, &injector)?,
    )
    .map_err(Error::from)
}

/// Deregister a scope-local event metadata injector by name.
#[napi(catch_unwind)]
pub fn scope_deregister_event_metadata_injector(scope_uuid: String, name: String) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|error| Error::invalid_argument(format!("invalid UUID: {error}")))?;
    core_registry_api::scope_deregister_event_metadata_injector(&uuid, &name).map_err(Error::from)
}

macro_rules! napi_scope_event_guardrail_api {
    ($register_name:ident, $deregister_name:ident, $core_register:path, $core_deregister:path) => {
        /// Register a scope-local event sanitize guardrail.
        ///
        /// The callback may return fields directly or in a Promise. Scope and mark
        /// calls queue the event and return synchronously; publication resumes after
        /// the Promise settles. Callback, serialization, conversion, or invalid-result
        /// failures clear the emitted event fields and record the error for
        /// `getLastCallbackError()`. Await `flushSubscribers()` before inspecting
        /// either the delivered event or that error.
        #[napi(catch_unwind)]
        pub fn $register_name(
            env: Env,
            scope_uuid: String,
            name: String,
            priority: i32,
            #[napi(
                ts_arg_type = "(event: Json, fields: EventSanitizeFields) => EventSanitizeFields | Promise<EventSanitizeFields>"
            )]
            guardrail: Callback,
        ) -> Result<()> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
            $core_register(
                &uuid,
                &name,
                priority,
                node_event_sanitize_fn(&env, &guardrail)?,
            )
            .map_err(Error::from)
        }

        #[napi(catch_unwind)]
        pub fn $deregister_name(scope_uuid: String, name: String) -> Result<bool> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
            $core_deregister(&uuid, &name).map_err(Error::from)
        }
    };
}

napi_scope_event_guardrail_api!(
    scope_register_mark_sanitize_guardrail,
    scope_deregister_mark_sanitize_guardrail,
    core_registry_api::scope_register_mark_sanitize_guardrail,
    core_registry_api::scope_deregister_mark_sanitize_guardrail
);
napi_scope_event_guardrail_api!(
    scope_register_scope_sanitize_start_guardrail,
    scope_deregister_scope_sanitize_start_guardrail,
    core_registry_api::scope_register_scope_sanitize_start_guardrail,
    core_registry_api::scope_deregister_scope_sanitize_start_guardrail
);
napi_scope_event_guardrail_api!(
    scope_register_scope_sanitize_end_guardrail,
    scope_deregister_scope_sanitize_end_guardrail,
    core_registry_api::scope_register_scope_sanitize_end_guardrail,
    core_registry_api::scope_deregister_scope_sanitize_end_guardrail
);

macro_rules! napi_scope_guardrail_tool_api {
    ($(#[doc = $reg_doc:expr_2021])* $register_name:ident,
     $(#[doc = $dereg_doc:expr_2021])* $deregister_name:ident,
     $core_register:path, $core_deregister:path) => {
        $(#[doc = $reg_doc])*
        #[napi(catch_unwind)]
        pub fn $register_name(
            env: Env,
            scope_uuid: String,
            name: String,
            priority: i32,
            #[napi(
                ts_arg_type = "(toolName: string, value: Json) => Json | Promise<Json>"
            )]
            guardrail: Callback,
        ) -> Result<()> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
            let callback = Arc::new(PromiseAwareFn::new(&env, &guardrail)?);
            $core_register(
                &uuid,
                &name,
                priority,
                callable::wrap_js_tool_sanitize_promise_fn(callback),
            )
            .map_err(Error::from)
        }

        $(#[doc = $dereg_doc])*
        #[napi(catch_unwind)]
        pub fn $deregister_name(scope_uuid: String, name: String) -> Result<bool> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
            $core_deregister(&uuid, &name).map_err(Error::from)
        }
    };
}

napi_scope_guardrail_tool_api!(
    #[doc = " Register a scope-local guardrail that sanitizes tool request arguments before execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(toolName, args)` and must return sanitized args."]
    #[doc = " Higher `priority` values run first. Throws if a guardrail with the same `name` already exists"]
    #[doc = " on the specified scope."]
    #[doc = " If the callback throws, Relay omits the emitted payload and records the error"]
    #[doc = " for `getLastCallbackError()`."]
    scope_register_tool_sanitize_request_guardrail,
    #[doc = " Deregister a scope-local tool request sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed from the specified scope."]
    scope_deregister_tool_sanitize_request_guardrail,
    core_registry_api::scope_register_tool_sanitize_request_guardrail,
    core_registry_api::scope_deregister_tool_sanitize_request_guardrail
);

napi_scope_guardrail_tool_api!(
    #[doc = " Register a scope-local guardrail that sanitizes tool response data after execution."]
    #[doc = ""]
    #[doc = " The `guardrail` callback receives `(toolName, result)` and must return sanitized result."]
    #[doc = " Higher `priority` values run first. Throws if a guardrail with the same `name` already exists"]
    #[doc = " on the specified scope."]
    #[doc = " If the callback throws, Relay omits the emitted payload and records the error"]
    #[doc = " for `getLastCallbackError()`."]
    scope_register_tool_sanitize_response_guardrail,
    #[doc = " Deregister a scope-local tool response sanitization guardrail by name."]
    #[doc = ""]
    #[doc = " Returns `true` if a guardrail with that name was found and removed from the specified scope."]
    scope_deregister_tool_sanitize_response_guardrail,
    core_registry_api::scope_register_tool_sanitize_response_guardrail,
    core_registry_api::scope_deregister_tool_sanitize_response_guardrail
);

/// Register a scope-local guardrail that conditionally gates tool execution.
///
/// The `guardrail` callback receives `(toolName, args)` and must return `null` to allow
/// execution or a rejection reason string to block it. Higher `priority` values run first.
/// If the callback throws, the managed call rejects and the protected callback does not run.
#[napi(catch_unwind)]
pub fn scope_register_tool_conditional_execution_guardrail(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(toolName: string, args: Json) => string | null | Promise<string | null>"
    )]
    guardrail: Callback,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_register_tool_conditional_execution_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_tool_conditional_promise_fn(std::sync::Arc::new(
            crate::promise_call::PromiseAwareFn::new(&env, &guardrail).map_err(|error| {
                Error::generic_failure(format!("failed to create PromiseAwareFn: {error}"))
            })?,
        )),
    )
    .map_err(Error::from)
}

/// Deregister a scope-local tool conditional execution guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed from the specified scope.
#[napi(catch_unwind)]
pub fn scope_deregister_tool_conditional_execution_guardrail(
    scope_uuid: String,
    name: String,
) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_deregister_tool_conditional_execution_guardrail(&uuid, &name)
        .map_err(Error::from)
}

// ---------------------------------------------------------------------------
// Scope-local intercept registrations — Tool
// ---------------------------------------------------------------------------

macro_rules! napi_scope_intercept_tool_api {
    ($(#[doc = $reg_doc:expr_2021])* $register_name:ident,
     $(#[doc = $dereg_doc:expr_2021])* $deregister_name:ident,
     $core_register:path, $core_deregister:path) => {
        $(#[doc = $reg_doc])*
        #[napi(catch_unwind)]
        pub fn $register_name(
            env: Env,
            scope_uuid: String,
            name: String,
            priority: i32,
            break_chain: bool,
            #[napi(
                ts_arg_type = "(toolName: string, args: Json) => Json | Promise<Json>"
            )]
            callable: Callback,
        ) -> Result<()> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
            let callback = std::sync::Arc::new(
                crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|error| {
                    Error::generic_failure(format!("failed to create PromiseAwareFn: {error}"))
                })?,
            );
            $core_register(
                &uuid,
                &name,
                priority,
                break_chain,
                callable::wrap_js_tool_request_intercept_promise_fn(callback),
            )
            .map_err(Error::from)
        }

        $(#[doc = $dereg_doc])*
        #[napi(catch_unwind)]
        pub fn $deregister_name(scope_uuid: String, name: String) -> Result<bool> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
            $core_deregister(&uuid, &name).map_err(Error::from)
        }
    };
}

napi_scope_intercept_tool_api!(
    #[doc = " Register a scope-local intercept that transforms tool request arguments."]
    #[doc = ""]
    #[doc = " The `callable` receives `(toolName, args)` and returns transformed args. If `breakChain`"]
    #[doc = " is `true`, no lower-priority intercepts run after this one. Higher `priority` values run first."]
    #[doc = " If the callback throws, the managed call rejects and later middleware does not run."]
    scope_register_tool_request_intercept,
    #[doc = " Deregister a scope-local tool request intercept by name."]
    #[doc = ""]
    #[doc = " Returns `true` if an intercept with that name was found and removed from the specified scope."]
    scope_deregister_tool_request_intercept,
    core_registry_api::scope_register_tool_request_intercept,
    core_registry_api::scope_deregister_tool_request_intercept
);

/// Register a scope-local tool execution intercept following the middleware chain pattern.
///
/// The `callable` receives `ToolExecutionContext` and `next`. The context
/// exposes `toolName`, `args`, and `toolCallId`; `toolCallId` is `null` when
/// the managed call has no provider-issued correlation identifier. Call
/// `next(context.args)` to continue the chain, or return an outcome directly
/// to short-circuit execution.
#[napi(catch_unwind)]
pub fn scope_register_tool_execution_intercept(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(context: ToolExecutionContext, next: (args: Json) => ToolExecutionResult | Promise<ToolExecutionResult>) => { result: Json; annotation?: Json; pendingMarks?: Array<PendingMarkSpec> } | Promise<{ result: Json; annotation?: Json; pendingMarks?: Array<PendingMarkSpec> }>"
    )]
    callable: Callback,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable)
            .map_err(|e| Error::generic_failure(format!("failed to create PromiseAwareFn: {e}")))?,
    );
    core_registry_api::scope_register_tool_execution_intercept(
        &uuid,
        &name,
        priority,
        callable::wrap_js_tool_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(Error::from)?;
    Ok(())
}

/// Deregister a scope-local tool execution intercept by name.
///
/// Removes an intercept registered through either registration shape. Returns
/// `true` if an intercept with that name was found and removed from the
/// specified scope.
#[napi(catch_unwind)]
pub fn scope_deregister_tool_execution_intercept(scope_uuid: String, name: String) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_deregister_tool_execution_intercept(&uuid, &name).map_err(Error::from)
}

// ---------------------------------------------------------------------------
// Scope-local guardrail registrations — LLM
// ---------------------------------------------------------------------------

/// Register a scope-local guardrail that sanitizes LLM request data before execution.
///
/// The `guardrail` callback receives `(request, context)` and must return the sanitized request,
/// or `null` to omit the observability payload. Lower `priority` values run first. Throws if a
/// guardrail with the same `name` already exists on the specified scope. If the callback throws,
/// Relay omits the payload and annotation, continues publication, and records the error for
/// `getLastCallbackError()`.
#[napi(catch_unwind)]
pub fn scope_register_llm_sanitize_request_guardrail(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, context: LlmSanitizeRequestContext) => Json | null | Promise<Json | null>"
    )]
    guardrail: Callback,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_register_llm_sanitize_request_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_sanitize_request_promise_fn(Arc::new(PromiseAwareFn::new(
            &env, &guardrail,
        )?)),
    )
    .map_err(Error::from)
}

/// Deregister a scope-local LLM request sanitization guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed from the specified scope.
#[napi(catch_unwind)]
pub fn scope_deregister_llm_sanitize_request_guardrail(
    scope_uuid: String,
    name: String,
) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_deregister_llm_sanitize_request_guardrail(&uuid, &name)
        .map_err(Error::from)
}

/// Register a scope-local guardrail that sanitizes LLM response data after execution.
///
/// The `guardrail` callback receives `(response, context)` and must return the sanitized response,
/// or `null` to omit the observability payload. Lower `priority` values run first. Throws if a
/// guardrail with the same `name` already exists on the specified scope. If the callback throws,
/// Relay omits the payload and annotation, continues publication, and records the error for
/// `getLastCallbackError()`.
#[napi(catch_unwind)]
pub fn scope_register_llm_sanitize_response_guardrail(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(response: Json, context: LlmSanitizeResponseContext) => Json | null | Promise<Json | null>"
    )]
    guardrail: Callback,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_register_llm_sanitize_response_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_sanitize_response_promise_fn(Arc::new(PromiseAwareFn::new(
            &env, &guardrail,
        )?)),
    )
    .map_err(Error::from)
}

/// Deregister a scope-local LLM response sanitization guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed from the specified scope.
#[napi(catch_unwind)]
pub fn scope_deregister_llm_sanitize_response_guardrail(
    scope_uuid: String,
    name: String,
) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_deregister_llm_sanitize_response_guardrail(&uuid, &name)
        .map_err(Error::from)
}

/// Register a scope-local guardrail that conditionally gates LLM execution.
///
/// The `guardrail` callback receives the LLM request as JSON and must return `null` to allow
/// execution or a rejection reason string to block it. Higher `priority` values run first.
/// If the callback throws, the managed call rejects and the protected callback does not run.
#[napi(catch_unwind)]
pub fn scope_register_llm_conditional_execution_guardrail(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(ts_arg_type = "(request: Json) => string | null | Promise<string | null>")]
    guardrail: Callback,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_register_llm_conditional_execution_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_conditional_promise_fn(std::sync::Arc::new(
            crate::promise_call::PromiseAwareFn::new(&env, &guardrail).map_err(|error| {
                Error::generic_failure(format!("failed to create PromiseAwareFn: {error}"))
            })?,
        )),
    )
    .map_err(Error::from)
}

/// Deregister a scope-local LLM conditional execution guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed from the specified scope.
#[napi(catch_unwind)]
pub fn scope_deregister_llm_conditional_execution_guardrail(
    scope_uuid: String,
    name: String,
) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_deregister_llm_conditional_execution_guardrail(&uuid, &name)
        .map_err(Error::from)
}

// ---------------------------------------------------------------------------
// Scope-local intercept registrations — LLM
// ---------------------------------------------------------------------------

/// Register a scope-local intercept that transforms LLM request data.
///
/// The `callable` receives the `LlmRequest` (as JSON) and returns a transformed request.
/// If `breakChain` is `true`, no lower-priority intercepts run after this one.
/// Higher `priority` values run first.
/// If the callback throws, the managed call rejects and later middleware does not run.
#[napi(catch_unwind)]
pub fn scope_register_llm_request_intercept(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    break_chain: bool,
    #[napi(
        ts_arg_type = "(args: { name: string; request: Json; annotated: Json | null }) => LlmRequestInterceptOutcome | Promise<LlmRequestInterceptOutcome>"
    )]
    callable: Callback,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    let callback = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|error| {
            Error::generic_failure(format!("failed to create PromiseAwareFn: {error}"))
        })?,
    );
    core_registry_api::scope_register_llm_request_intercept(
        &uuid,
        &name,
        priority,
        break_chain,
        callable::wrap_js_llm_request_intercept_promise_fn(callback),
    )
    .map_err(Error::from)
}

/// Deregister a scope-local LLM request intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi(catch_unwind)]
pub fn scope_deregister_llm_request_intercept(scope_uuid: String, name: String) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_deregister_llm_request_intercept(&uuid, &name).map_err(Error::from)
}

/// Register a scope-local LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request, codec context, and a `next` function. Call
/// `next(request)` to invoke the next intercept or original implementation; skip calling `next` to
/// short-circuit the chain. `next` may be called repeatedly or concurrently while
/// `callable` is pending; each call receives an isolated scope-stack branch, and
/// unfinished or later calls reject after `callable` settles.
#[napi(catch_unwind)]
pub fn scope_register_llm_execution_intercept(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, context: LlmExecutionContext, next: (request: Json) => Json | Promise<Json>) => Json | Promise<Json>"
    )]
    callable: Callback,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable)
            .map_err(|e| Error::generic_failure(format!("failed to create PromiseAwareFn: {e}")))?,
    );
    core_registry_api::scope_register_llm_execution_intercept(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(Error::from)?;
    Ok(())
}

/// Deregister a scope-local LLM execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi(catch_unwind)]
pub fn scope_deregister_llm_execution_intercept(scope_uuid: String, name: String) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_deregister_llm_execution_intercept(&uuid, &name).map_err(Error::from)
}

/// Register a scope-local streaming LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request, request-codec context, and a `next` function. The
/// response codec is `null` because streaming execution has no complete-response codec.
/// Call `next(request)` to
/// invoke the next intercept or original streaming implementation; in Node the
/// returned promise resolves to a lazy `AsyncIterable`. Return it directly or wrap it
/// `next` to short-circuit the chain. `next` may be called repeatedly or concurrently
/// while `callable` is pending; each call receives an isolated scope-stack branch,
/// and unfinished or later calls reject after the returned interceptor stream settles.
/// A downstream stream returned successfully by `next` keeps its normal lifetime.
#[napi(catch_unwind)]
pub fn scope_register_llm_stream_execution_intercept(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, context: LlmExecutionContext, next: (request: Json) => Promise<AsyncIterable<Json>>) => AsyncIterable<Json> | Promise<AsyncIterable<Json>>"
    )]
    callable: Callback,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable)
            .map_err(|e| Error::generic_failure(format!("failed to create PromiseAwareFn: {e}")))?,
    );
    core_registry_api::scope_register_llm_stream_execution_intercept(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_stream_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(Error::from)?;
    Ok(())
}

/// Deregister a scope-local LLM stream execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi(catch_unwind)]
pub fn scope_deregister_llm_stream_execution_intercept(
    scope_uuid: String,
    name: String,
) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_deregister_llm_stream_execution_intercept(&uuid, &name)
        .map_err(Error::from)
}

// ---------------------------------------------------------------------------
// Scope-local subscriber registrations
// ---------------------------------------------------------------------------

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
    #[napi(ts_arg_type = "(event: Json) => void | Promise<void>")] callback: Callback,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    let callback = callable::wrap_js_event_subscriber(&env, name.clone(), &callback)?;
    core_subscriber_api::scope_register_subscriber(&uuid, &name, callback).map_err(Error::from)
}

/// Deregister a scope-local event subscriber by name.
///
/// Returns `true` if a subscriber with that name was found and removed from the specified scope.
#[napi(catch_unwind)]
pub fn scope_deregister_subscriber(scope_uuid: String, name: String) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| Error::invalid_argument(format!("invalid UUID: {e}")))?;
    core_subscriber_api::scope_deregister_subscriber(&uuid, &name).map_err(Error::from)
}

// ---------------------------------------------------------------------------
// Standalone middleware chains
// ---------------------------------------------------------------------------

/// Run the registered tool request intercept chain on the given arguments.
/// Returns the transformed arguments.
#[napi(ts_return_type = "Promise<unknown>", catch_unwind)]
pub fn tool_request_intercepts<'env>(
    env: &'env Env,
    name: String,
    args: Json,
) -> Result<PromiseRaw<'env, Json>> {
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(env)?;
    spawn_promise(
        env,
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            core_tool_api::tool_request_intercepts(&name, args)
                                .await
                                .map_err(Error::from)
                        })
                        .await
                },
            )
            .await
        },
        js_value,
    )
}

/// Run the registered tool conditional execution guardrail chain.
/// Throws if any guardrail rejects.
#[napi(ts_return_type = "Promise<void>", catch_unwind)]
pub fn tool_conditional_execution<'env>(
    env: &'env Env,
    name: String,
    args: Json,
) -> Result<PromiseRaw<'env, ()>> {
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(env)?;
    spawn_promise(
        env,
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            core_tool_api::tool_conditional_execution(&name, &args)
                                .await
                                .map_err(Error::from)
                        })
                        .await
                },
            )
            .await
        },
        |env, _| js_value(env, ()),
    )
}

/// Run the registered LLM request intercept chain on the given request.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LlmRequest` schema. Returns the transformed request as JSON.
#[napi(ts_return_type = "Promise<LlmRequestInterceptOutcome>", catch_unwind)]
pub fn llm_request_intercepts<'env>(
    env: &'env Env,
    name: String,
    request: Json,
) -> Result<PromiseRaw<'env, Json>> {
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| Error::invalid_argument(format!("invalid LlmRequest: {e}")))?;
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(env)?;
    spawn_promise(
        env,
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
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
                        })
                        .await
                },
            )
            .await
        },
        js_value,
    )
}

/// Run the registered LLM conditional execution guardrail chain.
/// Throws if any guardrail rejects. The `request` should be a JSON object with `headers`
/// and `content` fields matching the `LlmRequest` schema.
#[napi(ts_return_type = "Promise<void>", catch_unwind)]
pub fn llm_conditional_execution<'env>(
    env: &'env Env,
    request: Json,
) -> Result<PromiseRaw<'env, ()>> {
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| Error::invalid_argument(format!("invalid LlmRequest: {e}")))?;
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(env)?;
    spawn_promise(
        env,
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            core_llm_api::llm_conditional_execution(&llm_request)
                                .await
                                .map_err(Error::from)
                        })
                        .await
                },
            )
            .await
        },
        |env, _| js_value(env, ()),
    )
}

// ---------------------------------------------------------------------------
// Agent Trajectory Interchange Format (ATIF) Exporter
// ---------------------------------------------------------------------------

/// An Agent Trajectory Interchange Format (ATIF) exporter that collects lifecycle
/// events and exports them as a structured trajectory.
///
/// Create an instance with session and agent metadata, then register it as an event subscriber.
/// When ready, call `exportJson()` to serialize the collected trajectory.
#[napi]
pub struct AtifExporter {
    inner: nemo_relay::observability::atif::AtifExporter,
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
        let agent_info = nemo_relay::observability::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: None,
            extra: None,
        };
        Ok(Self {
            inner: nemo_relay::observability::atif::AtifExporter::new(session_id, agent_info),
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
    inner: nemo_relay::observability::atof::AtofExporter,
}

#[napi]
impl AtofExporter {
    /// Create a new Agent Trajectory Observability Format (ATOF) JSONL exporter
    /// from a config object.
    #[napi(constructor, catch_unwind)]
    pub fn new(config: Option<AtofExporterConfig>) -> Result<Self> {
        let inner = nemo_relay::observability::atof::AtofExporter::new(build_atof_config(config)?)
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
#[derive(Default)]
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
#[derive(Default)]
pub struct OpenTelemetryTlsConfig {
    /// CA bundle that replaces the system's trusted CAs.
    pub certificate: Option<String>,
    /// Client certificate chain. Also set `privateKey`.
    pub client_certificate: Option<String>,
    /// Client private key. Also set `clientCertificate`.
    pub private_key: Option<String>,
}

impl From<OpenTelemetryTlsConfig> for nemo_relay::observability::OpenTelemetryTlsConfig {
    fn from(value: OpenTelemetryTlsConfig) -> Self {
        Self {
            certificate: value.certificate,
            client_certificate: value.client_certificate,
            private_key: value.private_key,
        }
    }
}

/// Configuration for a subscriber that writes OTLP to a local file.
///
/// Carries no endpoint, transport, headers, or timeout: a file destination has
/// no use for them.
#[napi(object)]
#[derive(Default)]
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
) -> Result<nemo_relay::observability::otel::OpenTelemetryFileSinkConfig> {
    let otel_type = parse_otel_type(&options.r#type)?;
    let directory = options.output_directory.trim();
    if directory.is_empty() {
        return Err(Error::invalid_argument(
            "outputDirectory must be a nonblank string",
        ));
    }
    let sink = parse_otel_file_sink(
        directory,
        options.filename.as_deref(),
        options.format.as_deref(),
        options.mode.as_deref(),
    )?;
    let mut config =
        nemo_relay::observability::otel::OpenTelemetryFileSinkConfig::new(otel_type, sink)
            .with_instrumentation_scope(
                options
                    .instrumentation_scope
                    .unwrap_or_else(|| "opentelemetry".to_string()),
            );
    // An omitted service name stays unset so the SDK can detect it from
    // OTEL_SERVICE_NAME, matching the plugin configuration path.
    if let Some(service_name) = options.service_name {
        config = config.with_service_name(service_name);
    }
    if let Some(namespace) = options.service_namespace {
        config = config.with_service_namespace(namespace);
    }
    if let Some(version) = options.service_version {
        config = config.with_service_version(version);
    }
    for (key, value) in parse_string_map(options.resource_attributes, "resourceAttributes")? {
        config = config.with_resource_attribute(key, value);
    }
    config = config
        .with_completed_span_context_ttl(std::time::Duration::from_millis(
            parse_completed_span_context_ttl_millis(options.completed_span_context_ttl_millis)?,
        ))
        .with_mark_projection(parse_mark_projection(options.mark_projection)?)
        .with_mark_exclude_names(
            options
                .mark_exclude_names
                .unwrap_or_else(nemo_relay::observability::default_mark_exclude_names),
        )
        .with_attribute_mappings(parse_attribute_mappings(options.attribute_mappings)?)
        .with_promote_metadata_prefixes(options.promote_metadata_prefixes.unwrap_or_default())
        .with_promote_resource_metadata_prefixes(
            options
                .promote_resource_metadata_prefixes
                .unwrap_or_default(),
        );
    Ok(config)
}

/// OpenTelemetry-backed event subscriber.
#[napi]
pub struct OpenTelemetrySubscriber {
    inner: nemo_relay::observability::otel::OpenTelemetrySubscriber,
}

#[napi]
impl OpenTelemetrySubscriber {
    /// Create a new OpenTelemetry subscriber from a config object.
    #[napi(constructor, catch_unwind)]
    pub fn new(config: OpenTelemetryConfig) -> Result<Self> {
        let inner = nemo_relay::observability::otel::OpenTelemetrySubscriber::new(
            build_otel_config(config)?,
        )
        .map_err(Error::from)?;
        Ok(Self { inner })
    }

    /// Create a subscriber that writes OTLP to a local file.
    #[napi(factory, catch_unwind)]
    pub fn file_sink(config: OpenTelemetryFileSinkConfig) -> Result<Self> {
        let inner = nemo_relay::observability::otel::OpenTelemetrySubscriber::new_file_sink(
            build_otel_file_sink_config(config)?,
        )
        .map_err(Error::from)?;
        Ok(Self { inner })
    }

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

    /// Force a flush of finished spans through the exporter.
    ///
    /// A successful flush updates `runtimeDiagnostics()` with queue drops observed so far.
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

    /// Shut down the underlying tracer provider.
    #[napi(catch_unwind)]
    pub fn shutdown(&self) -> Result<()> {
        self.inner.shutdown().map_err(Error::from)
    }
}

/// Configuration object for `OpenTelemetryLogSubscriber`.
#[napi(object)]
#[derive(Default)]
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
    inner: nemo_relay::observability::otel_logs::OpenTelemetryLogSubscriber,
}

#[napi]
impl OpenTelemetryLogSubscriber {
    /// Create a log subscriber from a config object.
    #[napi(constructor, catch_unwind)]
    pub fn new(config: OpenTelemetryLogConfig) -> Result<Self> {
        let inner = nemo_relay::observability::otel_logs::OpenTelemetryLogSubscriber::new(
            build_otel_log_config(config)?,
        )
        .map_err(Error::from)?;
        Ok(Self { inner })
    }

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

    /// Flush queued Relay events and the OTLP log processor.
    ///
    /// A successful flush updates `runtimeDiagnostics()` with queue drops observed so far.
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

    /// Shut down the underlying logger provider.
    #[napi(catch_unwind)]
    pub fn shutdown(&self) -> Result<()> {
        self.inner.shutdown().map_err(Error::from)
    }
}

/// Configuration object for `OpenTelemetryMetricSubscriber`.
#[napi(object)]
#[derive(Default)]
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
    inner: nemo_relay::observability::otel_metrics::OpenTelemetryMetricSubscriber,
}

#[napi]
impl OpenTelemetryMetricSubscriber {
    /// Create a metric subscriber from a config object.
    #[napi(constructor, catch_unwind)]
    pub fn new(config: OpenTelemetryMetricConfig) -> Result<Self> {
        let inner = nemo_relay::observability::otel_metrics::OpenTelemetryMetricSubscriber::new(
            build_otel_metric_config(config)?,
        )
        .map_err(Error::from)?;
        Ok(Self { inner })
    }

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

    /// Collect and export current metric aggregates immediately.
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

    /// Shut down the underlying meter provider.
    #[napi(catch_unwind)]
    pub fn shutdown(&self) -> Result<()> {
        self.inner.shutdown().map_err(Error::from)
    }
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
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
        report: nemo_relay::plugin::ConfigReport,
    },
    Ready(CoreAdaptiveRuntime),
}

/// Owned adaptive runtime that can register adaptive features outside the plugin system.
#[napi]
pub struct AdaptiveRuntime {
    inner: Arc<tokio::sync::Mutex<Option<NodeAdaptiveRuntimeState>>>,
}

#[napi]
impl AdaptiveRuntime {
    /// Create an adaptive runtime wrapper from config.
    ///
    /// The runtime is constructed lazily when `register()` is awaited.
    #[napi(constructor, catch_unwind)]
    pub fn new(config: Json) -> Result<Self> {
        let config: AdaptiveConfig = serde_json::from_value(config)
            .map_err(|error| Error::invalid_argument(error.to_string()))?;
        let report = validate_adaptive_config_or_err(&config)?;
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(Some(
                NodeAdaptiveRuntimeState::Pending { config, report },
            ))),
        })
    }

    /// Register all configured adaptive runtime features.
    ///
    /// `register()` and `shutdown()` both temporarily take ownership of the
    /// runtime state, so concurrent calls are mutually exclusive by design.
    /// Once either operation takes the state, another concurrent registration or
    /// shutdown attempt fails with "adaptive runtime already shut down". This
    /// prevents double-registration and shutdown-during-registration races.
    #[napi(ts_return_type = "Promise<void>", catch_unwind)]
    pub fn register<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        let inner = self.inner.clone();
        spawn_promise(
            env,
            async move {
                let state = {
                    let mut guard = inner.lock().await;
                    guard.take().ok_or_else(|| {
                        Error::generic_failure("adaptive runtime already shut down")
                    })?
                };

                let (result, next_state) = match state {
                    NodeAdaptiveRuntimeState::Pending { config, report } => {
                        match CoreAdaptiveRuntime::new(config.clone()).await {
                            Ok(mut runtime) => {
                                let result = runtime.register().await.map_err(Error::from);
                                (result, Some(NodeAdaptiveRuntimeState::Ready(runtime)))
                            }
                            Err(error) => (
                                Err(Error::from(error)),
                                Some(NodeAdaptiveRuntimeState::Pending { config, report }),
                            ),
                        }
                    }
                    NodeAdaptiveRuntimeState::Ready(mut runtime) => {
                        let result = runtime.register().await.map_err(Error::from);
                        (result, Some(NodeAdaptiveRuntimeState::Ready(runtime)))
                    }
                };

                let mut guard = inner.lock().await;
                *guard = next_state;
                result
            },
            |env, _| js_value(env, ()),
        )
    }

    /// Deregister all previously registered adaptive runtime features.
    #[napi(catch_unwind)]
    pub fn deregister(&self) -> Result<()> {
        let mut guard = self.inner.try_lock().map_err(|_| {
            Error::generic_failure(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard
            .as_mut()
            .ok_or_else(|| Error::generic_failure("adaptive runtime already shut down"))?;
        match state {
            NodeAdaptiveRuntimeState::Pending { .. } => Ok(()),
            NodeAdaptiveRuntimeState::Ready(runtime) => runtime.deregister().map_err(Error::from),
        }
    }

    /// Shut down the adaptive runtime and consume its Rust runtime state.
    #[napi(ts_return_type = "Promise<void>", catch_unwind)]
    pub fn shutdown<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        let inner = self.inner.clone();
        spawn_promise(
            env,
            async move {
                let state = {
                    let mut guard = inner.lock().await;
                    guard.take().ok_or_else(|| {
                        Error::generic_failure("adaptive runtime already shut down")
                    })?
                };
                match state {
                    NodeAdaptiveRuntimeState::Pending { .. } => Ok(()),
                    NodeAdaptiveRuntimeState::Ready(runtime) => {
                        runtime.shutdown().await.map_err(Error::from)
                    }
                }
            },
            |env, _| js_value(env, ()),
        )
    }

    /// Block until the telemetry drain has processed pending events.
    #[napi(js_name = "waitForIdle", catch_unwind)]
    pub fn wait_for_idle(&self) -> Result<()> {
        let guard = self.inner.try_lock().map_err(|_| {
            Error::generic_failure(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard
            .as_ref()
            .ok_or_else(|| Error::generic_failure("adaptive runtime already shut down"))?;
        match state {
            NodeAdaptiveRuntimeState::Pending { .. } => Ok(()),
            NodeAdaptiveRuntimeState::Ready(runtime) => {
                runtime.wait_for_idle();
                Ok(())
            }
        }
    }

    /// Return the validation report captured during runtime construction.
    #[napi(catch_unwind)]
    pub fn report(&self) -> Result<Json> {
        let guard = self.inner.try_lock().map_err(|_| {
            Error::generic_failure(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard
            .as_ref()
            .ok_or_else(|| Error::generic_failure("adaptive runtime already shut down"))?;
        let report = match state {
            NodeAdaptiveRuntimeState::Pending { report, .. } => report,
            NodeAdaptiveRuntimeState::Ready(runtime) => runtime.report(),
        };
        serde_json::to_value(report).map_err(Error::from)
    }

    /// Bind the runtime's ACG request rewrite to a scope.
    #[napi(js_name = "bindScope", catch_unwind)]
    pub fn bind_scope(&self, scope_handle: &ScopeHandle) -> Result<()> {
        let mut guard = self.inner.try_lock().map_err(|_| {
            Error::generic_failure(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard
            .as_mut()
            .ok_or_else(|| Error::generic_failure("adaptive runtime already shut down"))?;
        match state {
            NodeAdaptiveRuntimeState::Pending { .. } => Err(Error::generic_failure(
                "adaptive runtime must be registered before binding ACG request intercepts",
            )),
            NodeAdaptiveRuntimeState::Ready(runtime) => runtime
                .bind_scope(scope_handle.inner.uuid)
                .map_err(Error::from),
        }
    }

    /// Build cache request facts for an annotated LLM request.
    #[napi(js_name = "buildCacheRequestFacts", catch_unwind)]
    pub fn build_cache_request_facts(&self, options: Json) -> Result<Option<Json>> {
        let options: CacheRequestFactsOptions = serde_json::from_value(options)
            .map_err(|error| Error::invalid_argument(error.to_string()))?;
        let _request_id = parse_cache_telemetry_request_id(&options.request_id)?;
        let _timestamp = parse_cache_telemetry_timestamp(options.timestamp.as_deref())?;
        let annotated_request: AnnotatedLlmRequest =
            serde_json::from_value(options.annotated_request).map_err(|error| {
                Error::invalid_argument(format!("invalid annotatedRequest: {error}"))
            })?;
        let guard = self.inner.try_lock().map_err(|_| {
            Error::generic_failure(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard
            .as_ref()
            .ok_or_else(|| Error::generic_failure("adaptive runtime already shut down"))?;
        match state {
            NodeAdaptiveRuntimeState::Pending { .. } => Err(Error::generic_failure(
                "adaptive runtime must be registered before building cache request facts",
            )),
            NodeAdaptiveRuntimeState::Ready(runtime) => Ok(runtime
                .build_cache_request_facts(&options.agent_id, &options.provider, &annotated_request)
                .map(|facts| serde_json::to_value(&facts))
                .transpose()
                .map_err(Error::from)?),
        }
    }
}

fn validate_adaptive_config_or_err(
    config: &AdaptiveConfig,
) -> Result<nemo_relay::plugin::ConfigReport> {
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

fn parse_cache_telemetry_request_id(request_id: &str) -> Result<uuid::Uuid> {
    uuid::Uuid::parse_str(request_id)
        .map_err(|error| Error::invalid_argument(format!("invalid requestId UUID: {error}")))
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
    let Some(value) = value else {
        return Ok(None);
    };
    serde_json::from_value(value)
        .map(Some)
        .map_err(|error| Error::invalid_argument(format!("invalid requestFacts: {error}")))
}

/// Validate an adaptive config document without constructing a runtime.
#[napi(js_name = "validateAdaptiveConfig", catch_unwind)]
pub fn validate_adaptive_config(config: Json) -> Result<Json> {
    let config: AdaptiveConfig = serde_json::from_value(config)
        .map_err(|error| Error::invalid_argument(error.to_string()))?;
    serde_json::to_value(CoreAdaptiveRuntime::validate_config(&config)).map_err(Error::from)
}

/// Build one adaptive cache telemetry event from normalized usage.
#[napi(js_name = "buildCacheTelemetryEvent", catch_unwind)]
pub fn build_cache_telemetry_event(options: Json) -> Result<Option<Json>> {
    let options: CacheTelemetryOptions = serde_json::from_value(options)
        .map_err(|error| Error::invalid_argument(error.to_string()))?;
    let provider = parse_cache_telemetry_provider(&options.provider)?;
    let request_id = parse_cache_telemetry_request_id(&options.request_id)?;
    let timestamp = parse_cache_telemetry_timestamp(options.timestamp.as_deref())?;
    let Some(usage_json) = options.usage else {
        return Ok(None);
    };
    let usage: Usage = serde_json::from_value(usage_json)
        .map_err(|error| Error::invalid_argument(format!("invalid usage: {error}")))?;
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
    #[napi(ts_arg_type = "(...args: any[]) => any | undefined | null")] validate: Option<Callback>,
    #[napi(ts_arg_type = "(...args: any[]) => any")] register: Callback,
) -> Result<()> {
    let validate_callback = match validate {
        Some(func) => {
            let callback: Function<Json, JsArg> =
                callback_factory::wrap_middleware_callback(&env, &func)?;
            let direct = PersistentJsFunction::new(&env, &callback)?;
            let thread_safe = callback
                .build_threadsafe_function::<Json>()
                .weak::<true>()
                .build_callback(|ctx: ThreadsafeCallContext<Json>| Ok(ctx.value))?;
            Some(NodePluginValidateCallback {
                direct,
                thread_safe,
                registration_thread: std::thread::current().id(),
            })
        }
        None => None,
    };
    let register: Function<JsArgs2, JsArg> =
        callback_factory::wrap_middleware_callback(&env, &register)?;
    let register_tsfn = register
        .build_threadsafe_function::<NodePluginRegisterCall>()
        .weak::<true>()
        .build_callback(move |ctx: ThreadsafeCallContext<NodePluginRegisterCall>| {
            let plugin_config = js_value(&ctx.env, ctx.value.plugin_config)?;
            let plugin_context = build_plugin_context(
                &ctx.env,
                ctx.value.namespace_prefix,
                ctx.value.registrations,
            )?;
            Ok(FnArgs::from((
                plugin_config,
                detach(&ctx.env, &plugin_context),
            )))
        })?;

    register_plugin_impl(Arc::new(NodePlugin {
        plugin_kind,
        validate: validate_callback,
        register: register_tsfn,
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
}

type DynamicPluginTeardownResult = std::result::Result<(), Arc<Error>>;

enum DynamicPluginCloseStatus {
    Active(Box<Option<CorePluginHostActivation>>),
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
            status: StdMutex::new(DynamicPluginCloseStatus::Active(Box::new(Some(activation)))),
            report: StdMutex::new(report),
            completion,
        }
    }

    fn report(&self) -> PluginHostReport {
        let latest = {
            let status = self
                .status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match &*status {
                DynamicPluginCloseStatus::Active(activation) => activation
                    .as_ref()
                    .as_ref()
                    .map(CorePluginHostActivation::report),
                DynamicPluginCloseStatus::Closing | DynamicPluginCloseStatus::Closed => None,
            }
        };
        let mut report = self
            .report
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(latest) = latest {
            *report = latest;
        }
        report.clone()
    }

    fn active(&self) -> bool {
        let status = self
            .status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match &*status {
            DynamicPluginCloseStatus::Active(activation) => activation.as_ref().is_some(),
            DynamicPluginCloseStatus::Closing | DynamicPluginCloseStatus::Closed => false,
        }
    }

    fn begin_close(self: &Arc<Self>, log_finalizer_error: bool) {
        let activation = {
            let mut status = self
                .status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match &mut *status {
                DynamicPluginCloseStatus::Active(activation) => {
                    if let Some(current) = activation.as_ref().as_ref() {
                        *self
                            .report
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()) = current.report();
                    }
                    let activation = activation.as_mut().take();
                    if activation.is_some() {
                        self.completion.send_replace(None);
                    }
                    *status = DynamicPluginCloseStatus::Closing;
                    activation
                }
                DynamicPluginCloseStatus::Closing | DynamicPluginCloseStatus::Closed => None,
            }
        };
        let Some(activation) = activation else {
            return;
        };

        // Keep the activation outside the spawned closure so a thread-spawn
        // failure cannot drop it and synchronously run teardown on the JS thread.
        let activation = Arc::new(StdMutex::new(Some(activation)));
        let worker_activation = Arc::clone(&activation);
        let close_state = Arc::clone(self);
        let spawn = std::thread::Builder::new()
            .name("nemo-relay-node-plugin-teardown".into())
            .spawn(move || {
                let activation = worker_activation
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take();
                let (activation, result) = match activation {
                    Some(mut activation) => {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            activation.close()
                        }))
                        .map_err(|_| Error::generic_failure("plugin host teardown task panicked"))
                        .and_then(|result| result.map_err(Error::from))
                        .map_err(Arc::new);
                        *close_state
                            .report
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()) = activation.report();
                        (Some(activation), result)
                    }
                    None => (
                        None,
                        Err(Arc::new(Error::generic_failure(
                            "plugin host teardown task lost its activation",
                        ))),
                    ),
                };
                if log_finalizer_error && let Err(error) = &result {
                    eprintln!("nemo_relay: plugin host finalizer teardown failed: {error}");
                }
                close_state.finish(activation, result);
            });

        if let Err(error) = spawn {
            let activation = activation
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            let error = Arc::new(Error::generic_failure(format!(
                "failed to start dynamic plugin teardown task: {error}"
            )));
            if log_finalizer_error {
                // There is no caller left to retry, and teardown must not run
                // synchronously on the JS finalizer thread.
                if let Some(activation) = activation {
                    std::mem::forget(activation);
                }
                eprintln!("nemo_relay: dynamic plugin finalizer teardown failed: {error}");
                self.finish(None, Err(error));
            } else {
                self.finish(activation, Err(error));
            }
        }
    }

    fn finish(
        &self,
        activation: Option<CorePluginHostActivation>,
        result: DynamicPluginTeardownResult,
    ) {
        self.finish_with_hook(activation, result, || {});
    }

    fn finish_with_hook(
        &self,
        activation: Option<CorePluginHostActivation>,
        result: DynamicPluginTeardownResult,
        before_publish: impl FnOnce(),
    ) {
        let retryable = result.is_err()
            && activation
                .as_ref()
                .is_some_and(CorePluginHostActivation::is_active);
        let mut status = self
            .status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *status = if retryable {
            DynamicPluginCloseStatus::Active(Box::new(activation))
        } else {
            DynamicPluginCloseStatus::Closed
        };
        before_publish();
        self.completion.send_replace(Some(result));
    }

    async fn wait_for_close(&self) -> DynamicPluginTeardownResult {
        let mut completion = self.completion.subscribe();
        loop {
            if let Some(result) = completion.borrow().clone() {
                return result;
            }
            if completion.changed().await.is_err() {
                return Err(Arc::new(Error::generic_failure(
                    "dynamic plugin teardown result channel closed unexpectedly",
                )));
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/rust/api/dynamic_plugin_close_state_tests.rs"]
mod dynamic_plugin_close_state_tests;

#[napi]
impl PluginHostActivation {
    /// Return the validation report produced by activation.
    #[napi(getter, catch_unwind)]
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
        let close_state = Arc::clone(&self.close_state);
        close_state.begin_close(false);
        spawn_promise(
            env,
            async move { close_state.wait_for_close().await },
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

impl Drop for PluginHostActivation {
    fn drop(&mut self) {
        self.close_state.begin_close(true);
    }
}

/// Initialize the core-owned static and dynamic plugin host from layered files.
#[napi(catch_unwind)]
pub fn initialize<'env>(
    env: &'env Env,
    config: Json,
    additional_plugins_toml: Option<String>,
) -> Result<PromiseRaw<'env, PluginHostActivation>> {
    spawn_promise::<_, _, Error, _, _>(
        env,
        async move {
            let config: nemo_relay::plugin::dynamic::PluginHostConfig =
                serde_json::from_value(config).map_err(|error| {
                    Error::invalid_argument(format!("invalid plugin config: {error}"))
                })?;
            let activation = initialize_impl(config, additional_plugins_toml.map(Into::into))
                .await
                .map_err(Error::from)?;
            Ok(PluginHostActivation {
                close_state: Arc::new(DynamicPluginCloseState::new(activation)),
            })
        },
        js_value,
    )
}
/// Validate dynamic plugins without loading any plugin code.
#[napi(catch_unwind)]
pub fn validate(config: Json, additional_plugins_toml: Option<String>) -> Result<Json> {
    let config: nemo_relay::plugin::dynamic::PluginHostConfig = serde_json::from_value(config)
        .map_err(|error| Error::invalid_argument(format!("invalid plugin config: {error}")))?;
    let report =
        validate_impl(config, additional_plugins_toml.map(Into::into)).map_err(Error::from)?;
    serde_json::to_value(report).map_err(Error::from)
}

/// Validate only the supplied static plugin configuration.
#[napi(catch_unwind)]
pub fn validate_exact(config: Json) -> Result<Json> {
    if config.get("diagnostics").is_some() {
        return Err(Error::invalid_argument(
            "diagnostics is a host-only setting; use layered validation",
        ));
    }
    let config: PluginConfig = serde_json::from_value(config)
        .map_err(|error| Error::invalid_argument(format!("invalid plugin config: {error}")))?;
    serde_json::to_value(validate_exact_impl(config)).map_err(Error::from)
}

/// List registered plugin kinds.
#[napi(catch_unwind)]
pub fn list_plugin_kinds() -> Vec<String> {
    list_plugin_kinds_impl()
}
