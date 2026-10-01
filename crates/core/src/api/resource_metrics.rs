// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fresh and managed system resource metric observation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock, Weak};

use nemo_relay_types::api::event::{MetricKind, MetricMeasurement, MetricValueType};
pub use nemo_relay_types::api::resource_metrics::{
    AcceleratorDeviceMetrics, AcceleratorProcessMetrics, AcceleratorVendor, BandwidthUnit,
    CapacityUnit, CountUnit, CpuMetrics, CpuUnit, DataUnit, DiskMetrics, DurationUnit,
    FilesystemCapacityMetrics, GpuMetrics, MemoryMetrics, NetworkInterfaceMetrics, NetworkMetrics,
    NetworkTrafficMetrics, ProcessMetrics, ProcessSamplingMetadata, ResourceLimitEventCount,
    ResourceLimitEventKind, ResourceLimitResource, ResourceMeasurement, ResourceMeasurementScope,
    ResourceMetricValue, ResourceMetricsSnapshot, ResourceOperatingSystem, ResourceUnit,
    UtilizationUnit,
};
use serde_json::{Value, json};

use crate::api::runtime::scope_stack::{ScopeStack, ScopeStackHandle};
use crate::api::scope::{self, ScopeHandle};
use crate::error::Result;
pub use crate::plugins::resource_metrics::config::{
    ResourceMetricsConfig, ResourceMetricsCpuConfig, ResourceMetricsCpuUnits,
    ResourceMetricsDiskConfig, ResourceMetricsDiskUnits, ResourceMetricsGpuConfig,
    ResourceMetricsGpuUnits, ResourceMetricsMeasurementScope, ResourceMetricsMemoryConfig,
    ResourceMetricsMemoryUnits, ResourceMetricsNetworkConfig, ResourceMetricsNetworkTrafficUnits,
    ResourceMetricsNetworkUnits, ResourceMetricsPollingConfig, ResourceMetricsProcessConfig,
    ResourceMetricsUnits,
};
#[doc(hidden)]
pub use crate::resource_metrics::manager::CliResourceMetricsLaunchGuard;
pub use crate::resource_metrics::manager::ResourceMetricsTargetGuard;

use uuid::Uuid;

/// Canonical event name for resource metric marks.
pub const RESOURCE_METRICS_EVENT_NAME: &str = "nemo.relay.resource_metrics";

type AgentScopeRegistry = HashMap<Uuid, Weak<RwLock<ScopeStack>>>;

static AGENT_SCOPES: LazyLock<Mutex<AgentScopeRegistry>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static AGENT_SCOPE_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);

/// Acquire a fresh snapshot without requiring managed polling.
///
/// Await this operation on a Tokio runtime. OS and vendor queries run on a
/// blocking worker. Dropping the future stops waiting; a query already running
/// may still finish and update the on-demand sampling baseline.
pub async fn collect() -> Result<ResourceMetricsSnapshot> {
    crate::resource_metrics::manager::collect_on_demand().await
}

/// Prepare a CLI launch before activating a process-tree resource metrics component.
///
/// The guard keeps process-tree polling paused until the launched child becomes
/// the collection target. The CLI holds it until gateway shutdown.
#[doc(hidden)]
pub fn prepare_cli_resource_metrics_process_tree() -> Result<CliResourceMetricsLaunchGuard> {
    crate::resource_metrics::manager::prepare_cli_owned_process_tree()
}

/// Direct the active resource metrics plugin to a Relay-owned child process tree.
///
/// This lifecycle hook is used by the CLI while supervising a launched agent.
/// The returned guard restores application-process collection when supervision
/// ends. Plugin configuration remains exclusively on the plugin-host API.
#[doc(hidden)]
#[cfg(not(windows))]
pub fn target_resource_metrics_to_owned_process_tree(
    process_id: u32,
) -> Result<Option<ResourceMetricsTargetGuard>> {
    crate::resource_metrics::manager::target_owned_process_tree(process_id)
}

/// Direct the active resource metrics plugin to a Relay-owned Windows Job Object.
///
/// # Safety
/// `job_handle` must be a live Job Object handle owned by this process. Relay
/// duplicates the handle before returning and never closes the caller's handle.
#[doc(hidden)]
#[cfg(windows)]
pub unsafe fn target_resource_metrics_to_owned_process_tree(
    process_id: u32,
    job_handle: isize,
) -> Result<Option<ResourceMetricsTargetGuard>> {
    crate::resource_metrics::manager::target_owned_process_tree_with_job_handle(
        process_id, job_handle,
    )
}

pub(crate) fn register_agent_scope(scope: ScopeHandle, stack: ScopeStackHandle) {
    let mut scopes = AGENT_SCOPES
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    // Bound abandoned entries even without polling, without scanning on every push.
    if AGENT_SCOPE_REGISTRATIONS
        .fetch_add(1, Ordering::Relaxed)
        .is_multiple_of(64)
    {
        scopes.retain(|_, stack| stack.strong_count() > 0);
    }
    scopes.insert(scope.uuid, Arc::downgrade(&stack));
}

pub(crate) fn unregister_agent_scope(uuid: &Uuid) {
    AGENT_SCOPES
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .remove(uuid);
}

pub(crate) fn emit_resource_metrics_snapshot(
    snapshot: &ResourceMetricsSnapshot,
    fallback_scope_stack: &ScopeStackHandle,
) {
    let measurements = metric_measurements(snapshot);
    let timestamp = snapshot.timestamp;
    let targets = {
        let mut scopes = AGENT_SCOPES
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut targets = Vec::new();
        scopes.retain(|uuid, stack| {
            if let Some(stack) = stack.upgrade() {
                let scope = stack
                    .read()
                    .unwrap_or_else(|error| error.into_inner())
                    .find(uuid)
                    .cloned();
                if let Some(scope) = scope {
                    targets.push((scope, stack));
                    true
                } else {
                    false
                }
            } else {
                false
            }
        });
        targets.sort_by_key(|(scope, _)| scope.uuid);
        targets
    };
    let mut emitted_for_agent = false;
    let mut only_closed_scopes = true;
    for (parent, stack) in targets {
        match scope::metric_on_scope_stack(
            RESOURCE_METRICS_EVENT_NAME,
            measurements.clone(),
            &parent,
            timestamp,
            stack,
        ) {
            Ok(true) => emitted_for_agent = true,
            Ok(false) => unregister_agent_scope(&parent.uuid),
            Err(error) => {
                only_closed_scopes = false;
                log_collection_failure(&error);
            }
        }
    }
    if emitted_for_agent || !only_closed_scopes {
        return;
    }
    let root_stack =
        match crate::api::runtime::scope_stack::root_scope_stack_snapshot(fallback_scope_stack) {
            Ok(stack) => stack,
            Err(error) => {
                log_collection_failure(&error);
                return;
            }
        };
    let parent = root_stack
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .scopes()
        .first()
        .expect("scope stack should contain its implicit root")
        .clone();
    if let Err(error) = scope::metric_on_scope_stack(
        RESOURCE_METRICS_EVENT_NAME,
        measurements,
        &parent,
        timestamp,
        root_stack,
    ) {
        log_collection_failure(&error);
    }
}

fn metric_measurements(snapshot: &ResourceMetricsSnapshot) -> Vec<MetricMeasurement> {
    let operating_system = snapshot.operating_system.as_str();
    let measurement_scope = snapshot.measurement_scope.as_str();
    let mut measurements = Vec::new();
    // Every successful poll needs a valid metric mark, even if its selected
    // categories have no available measurements on this host.
    let sample = ResourceMeasurement::new(1_u64, CountUnit::Events);
    push_integer(
        &mut measurements,
        "nemo.relay.resource.sample_count",
        "Resource metric samples collected in this poll",
        operating_system,
        measurement_scope,
        Some(&sample),
    );

    macro_rules! integer {
        ($name:literal, $description:literal, $measurement:expr) => {
            push_integer(
                &mut measurements,
                $name,
                $description,
                operating_system,
                measurement_scope,
                $measurement.as_ref(),
            );
        };
    }
    macro_rules! float {
        ($name:literal, $description:literal, $measurement:expr) => {
            push_float(
                &mut measurements,
                $name,
                $description,
                operating_system,
                measurement_scope,
                $measurement.as_ref(),
            );
        };
    }

    if let Some(cpu) = &snapshot.cpu {
        integer!(
            "nemo.relay.resource.cpu.user_time",
            "User-mode CPU time of currently visible processes",
            cpu.user_time
        );
        integer!(
            "nemo.relay.resource.cpu.system_time",
            "System-mode CPU time of currently visible processes",
            cpu.system_time
        );
        integer!(
            "nemo.relay.resource.cpu.total_time",
            "CPU time of currently visible processes",
            cpu.total_time
        );
        float!(
            "nemo.relay.resource.cpu.consumption_rate",
            "Average CPU capacity consumed over the sampling interval",
            cpu.consumption_rate
        );
        integer!(
            "nemo.relay.resource.cpu.throttled_time",
            "Cumulative CPU throttled time",
            cpu.throttled_time
        );
        float!(
            "nemo.relay.resource.cpu.effective_limit",
            "Effective CPU capacity",
            cpu.effective_limit
        );
        integer!(
            "nemo.relay.resource.cpu.pressure.some_stall_time",
            "Cumulative CPU pressure time with some work stalled",
            cpu.some_pressure_stall_time
        );
        integer!(
            "nemo.relay.resource.cpu.pressure.full_stall_time",
            "Cumulative CPU pressure time with all work stalled",
            cpu.full_pressure_stall_time
        );
        for event in &cpu.limit_events {
            push_integer_with_attributes(
                &mut measurements,
                "nemo.relay.resource.limit.event_count",
                "Cumulative resource-limit event count",
                operating_system,
                measurement_scope,
                event.count.as_ref(),
                json!({
                    "nemo_relay.resource.limit.resource": event.resource,
                    "nemo_relay.resource.limit.event": event.event,
                }),
            );
        }
    }
    if let Some(memory) = &snapshot.memory {
        integer!(
            "nemo.relay.resource.memory.system.used",
            "System memory used",
            memory.system_used
        );
        integer!(
            "nemo.relay.resource.memory.system.total",
            "Total system memory",
            memory.system_total
        );
        integer!(
            "nemo.relay.resource.memory.system.available",
            "Available system memory",
            memory.system_available
        );
        integer!(
            "nemo.relay.resource.memory.resident",
            "Resident memory",
            memory.resident
        );
        integer!(
            "nemo.relay.resource.memory.private",
            "Private memory",
            memory.private
        );
        integer!(
            "nemo.relay.resource.memory.physical_footprint",
            "Physical memory footprint",
            memory.physical_footprint
        );
        integer!(
            "nemo.relay.resource.memory.virtual",
            "Virtual memory",
            memory.virtual_memory
        );
        integer!(
            "nemo.relay.resource.memory.peak_resident",
            "Peak resident memory",
            memory.peak_resident
        );
        integer!(
            "nemo.relay.resource.memory.limit",
            "Effective memory limit",
            memory.limit
        );
        integer!(
            "nemo.relay.resource.memory.environment_accounted",
            "Environment-accounted memory",
            memory.environment_accounted
        );
        integer!(
            "nemo.relay.resource.memory.pressure.some_stall_time",
            "Cumulative memory pressure time with some work stalled",
            memory.some_pressure_stall_time
        );
        integer!(
            "nemo.relay.resource.memory.pressure.full_stall_time",
            "Cumulative memory pressure time with all work stalled",
            memory.full_pressure_stall_time
        );
        integer!(
            "nemo.relay.resource.memory.out_of_memory_event_count",
            "Cumulative out-of-memory event count",
            memory.out_of_memory_event_count
        );
        for event in &memory.limit_events {
            push_integer_with_attributes(
                &mut measurements,
                "nemo.relay.resource.limit.event_count",
                "Cumulative resource-limit event count",
                operating_system,
                measurement_scope,
                event.count.as_ref(),
                json!({
                    "nemo_relay.resource.limit.resource": event.resource,
                    "nemo_relay.resource.limit.event": event.event,
                }),
            );
        }
    }
    if let Some(process) = &snapshot.process {
        integer!(
            "nemo.relay.resource.process.active_count",
            "Active process count",
            process.active_count
        );
        integer!(
            "nemo.relay.resource.process.descendant_count",
            "Descendant process count",
            process.descendant_count
        );
        integer!(
            "nemo.relay.resource.thread.count",
            "Thread count",
            process.thread_count
        );
        integer!(
            "nemo.relay.resource.process.lifetime_creation_count",
            "Lifetime process creation count",
            process.lifetime_creation_count
        );
        integer!(
            "nemo.relay.resource.process.open_file_descriptor_count",
            "Open file descriptor count",
            process.open_file_descriptor_count
        );
        integer!(
            "nemo.relay.resource.process.windows_handle_count",
            "Windows handle count",
            process.windows_handle_count
        );
        for event in &process.limit_events {
            push_integer_with_attributes(
                &mut measurements,
                "nemo.relay.resource.limit.event_count",
                "Cumulative resource-limit event count",
                operating_system,
                measurement_scope,
                event.count.as_ref(),
                json!({
                    "nemo_relay.resource.limit.resource": event.resource,
                    "nemo_relay.resource.limit.event": event.event,
                }),
            );
        }
    }
    if let Some(disk) = &snapshot.disk {
        integer!(
            "nemo.relay.resource.disk.read_data",
            "Lifetime data read by currently visible processes",
            disk.read_data
        );
        integer!(
            "nemo.relay.resource.disk.write_data",
            "Lifetime data written by currently visible processes",
            disk.write_data
        );
        float!(
            "nemo.relay.resource.disk.read_throughput",
            "Average disk read transfer rate since the preceding sample",
            disk.read_throughput
        );
        float!(
            "nemo.relay.resource.disk.write_throughput",
            "Average disk write transfer rate since the preceding sample",
            disk.write_throughput
        );
        integer!(
            "nemo.relay.resource.disk.read.operations",
            "Lifetime read operations by currently visible processes",
            disk.read_operations
        );
        integer!(
            "nemo.relay.resource.disk.write.operations",
            "Lifetime write operations by currently visible processes",
            disk.write_operations
        );
        for filesystem in &disk.filesystems {
            let attributes = json!({"nemo_relay.resource.disk.filesystem_path": filesystem.path});
            push_integer_with_attributes(
                &mut measurements,
                "nemo.relay.resource.disk.filesystem.total_capacity",
                "Total filesystem capacity",
                operating_system,
                measurement_scope,
                filesystem.total_capacity.as_ref(),
                attributes.clone(),
            );
            push_integer_with_attributes(
                &mut measurements,
                "nemo.relay.resource.disk.filesystem.available_capacity",
                "Filesystem capacity available to unprivileged callers",
                operating_system,
                measurement_scope,
                filesystem.available_capacity.as_ref(),
                attributes.clone(),
            );
            push_integer_with_attributes(
                &mut measurements,
                "nemo.relay.resource.disk.filesystem.free_capacity",
                "Total free filesystem capacity, including reserved space",
                operating_system,
                measurement_scope,
                filesystem.free_capacity.as_ref(),
                attributes,
            );
        }
    }
    if let Some(gpu) = &snapshot.gpu {
        for device in gpu.device_metrics.as_deref().unwrap_or_default() {
            let attributes = accelerator_attributes(
                &device.vendor,
                &device.device_identifier,
                device.device_index,
                None,
            );
            push_integer_with_attributes(
                &mut measurements,
                "nemo.relay.resource.accelerator.device.memory_used",
                "Accelerator device memory in use",
                operating_system,
                measurement_scope,
                device.memory_used.as_ref(),
                attributes.clone(),
            );
            push_float_with_attributes(
                &mut measurements,
                "nemo.relay.resource.accelerator.device.compute_utilization",
                "Accelerator device compute utilization",
                operating_system,
                measurement_scope,
                device.compute_utilization.as_ref(),
                attributes,
            );
        }
        for process in gpu.process_metrics.as_deref().unwrap_or_default() {
            let attributes = accelerator_attributes(
                &process.vendor,
                &process.device_identifier,
                process.device_index,
                Some(process.process_id),
            );
            push_integer_with_attributes(
                &mut measurements,
                "nemo.relay.resource.accelerator.process.memory_used",
                "Accelerator memory attributed to an owned process",
                operating_system,
                measurement_scope,
                process.memory_used.as_ref(),
                attributes.clone(),
            );
            push_float_with_attributes(
                &mut measurements,
                "nemo.relay.resource.accelerator.process.compute_utilization",
                "Accelerator compute utilization attributed to an owned process",
                operating_system,
                measurement_scope,
                process.compute_utilization.as_ref(),
                attributes,
            );
        }
    }
    if let Some(network) = &snapshot.network {
        push_network_traffic(
            &mut measurements,
            "nemo.relay.resource.network.system",
            operating_system,
            &network.system,
            json!({}),
        );
        for interface in &network.interfaces {
            push_network_traffic(
                &mut measurements,
                "nemo.relay.resource.network.interface",
                operating_system,
                &interface.traffic,
                json!({"nemo_relay.resource.network.interface_name": interface.name}),
            );
        }
    }
    if let Some(coverage) = &snapshot.process_sampling {
        let visible = ResourceMeasurement::new(coverage.visible_processes, CountUnit::Processes);
        let sampled = ResourceMeasurement::new(coverage.sampled_processes, CountUnit::Processes);
        push_integer(
            &mut measurements,
            "nemo.relay.resource.process_sampling.visible_count",
            "Processes found before process sampling",
            operating_system,
            measurement_scope,
            Some(&visible),
        );
        push_integer(
            &mut measurements,
            "nemo.relay.resource.process_sampling.sampled_count",
            "Processes whose base query succeeded",
            operating_system,
            measurement_scope,
            Some(&sampled),
        );
        for (field, count) in &coverage.field_sampled_processes {
            let count = ResourceMeasurement::new(*count, CountUnit::Processes);
            push_integer_with_attributes(
                &mut measurements,
                "nemo.relay.resource.process_sampling.field_sampled_count",
                "Processes supplying the specified counter",
                operating_system,
                measurement_scope,
                Some(&count),
                json!({"nemo_relay.resource.field": field}),
            );
        }
    }
    measurements
}

fn push_network_traffic(
    measurements: &mut Vec<MetricMeasurement>,
    prefix: &str,
    operating_system: &str,
    traffic: &NetworkTrafficMetrics,
    attributes: Value,
) {
    macro_rules! integer {
        ($field:literal, $value:expr) => {
            push_integer_with_attributes(
                measurements,
                &format!("{prefix}.{}", $field),
                "System network interface counter",
                operating_system,
                ResourceMeasurementScope::Global.as_str(),
                $value.as_ref(),
                attributes.clone(),
            );
        };
    }
    integer!("received_data", traffic.received_data);
    integer!("transmitted_data", traffic.transmitted_data);
    integer!("received_packets", traffic.received_packets);
    integer!("transmitted_packets", traffic.transmitted_packets);
    integer!("receive_errors", traffic.receive_errors);
    integer!("transmit_errors", traffic.transmit_errors);
    for (field, value) in [
        ("receive_throughput", &traffic.receive_throughput),
        ("transmit_throughput", &traffic.transmit_throughput),
    ] {
        push_float_with_attributes(
            measurements,
            &format!("{prefix}.{field}"),
            "Average system network transfer rate since the preceding sample",
            operating_system,
            ResourceMeasurementScope::Global.as_str(),
            value.as_ref(),
            attributes.clone(),
        );
    }
}

fn push_float<U: ResourceUnit>(
    measurements: &mut Vec<MetricMeasurement>,
    name: &str,
    description: &str,
    operating_system: &str,
    measurement_scope: &str,
    measurement: Option<&ResourceMeasurement<f64, U>>,
) {
    push_float_with_attributes(
        measurements,
        name,
        description,
        operating_system,
        measurement_scope,
        measurement,
        json!({}),
    );
}

fn push_float_with_attributes<U: ResourceUnit>(
    measurements: &mut Vec<MetricMeasurement>,
    name: &str,
    description: &str,
    operating_system: &str,
    measurement_scope: &str,
    measurement: Option<&ResourceMeasurement<f64, U>>,
    additional_attributes: Value,
) {
    let Some(measurement) = measurement else {
        return;
    };
    let value = measurement.value;
    if !value.is_finite() {
        return;
    }
    measurements.push(
        MetricMeasurement::builder()
            .name(name)
            .kind(MetricKind::Gauge)
            .value_type(MetricValueType::F64)
            .value(json!(value))
            .unit(measurement.unit.as_str())
            .description(description)
            .attributes(metric_attributes(
                operating_system,
                measurement_scope,
                additional_attributes,
            ))
            .build(),
    );
}

fn push_integer<U: ResourceUnit>(
    measurements: &mut Vec<MetricMeasurement>,
    name: &str,
    description: &str,
    operating_system: &str,
    measurement_scope: &str,
    measurement: Option<&ResourceMeasurement<ResourceMetricValue, U>>,
) {
    push_integer_with_attributes(
        measurements,
        name,
        description,
        operating_system,
        measurement_scope,
        measurement,
        json!({}),
    );
}

fn push_integer_with_attributes<U: ResourceUnit>(
    measurements: &mut Vec<MetricMeasurement>,
    name: &str,
    description: &str,
    operating_system: &str,
    measurement_scope: &str,
    measurement: Option<&ResourceMeasurement<ResourceMetricValue, U>>,
    additional_attributes: Value,
) {
    let Some(measurement) = measurement else {
        return;
    };
    // OpenTelemetry keeps one numeric type per instrument name. A unit that can
    // produce a fractional conversion must use f64 even when this poll's value
    // happens to be whole, or the next poll can reject the entire metric mark.
    let fractional_unit = match measurement.unit.as_str() {
        "seconds" | "minutes" | "kilobytes" | "megabytes" | "gigabytes" | "terabytes"
        | "mebibytes" | "gibibytes" | "tebibytes" => true,
        "kibibytes" => {
            name.starts_with("nemo.relay.resource.disk.")
                || name.starts_with("nemo.relay.resource.network.")
        }
        _ => false,
    };
    let (value_type, value) = match measurement.value {
        ResourceMetricValue::Integer(value) if fractional_unit => {
            (MetricValueType::F64, json!(value as f64))
        }
        ResourceMetricValue::Integer(value) if value <= i64::MAX as u64 => {
            (MetricValueType::U64, json!(value))
        }
        ResourceMetricValue::Decimal(value) if value.is_finite() => {
            (MetricValueType::F64, json!(value))
        }
        _ => return,
    };
    measurements.push(
        MetricMeasurement::builder()
            .name(name)
            .kind(MetricKind::Gauge)
            .value_type(value_type)
            .value(value)
            .unit(measurement.unit.as_str())
            .description(description)
            .attributes(metric_attributes(
                operating_system,
                measurement_scope,
                additional_attributes,
            ))
            .build(),
    );
}

fn metric_attributes(
    operating_system: &str,
    measurement_scope: &str,
    additional_attributes: Value,
) -> Value {
    let mut attributes = additional_attributes
        .as_object()
        .cloned()
        .unwrap_or_default();
    attributes.insert(
        "nemo_relay.resource.operating_system".into(),
        json!(operating_system),
    );
    attributes.insert(
        "nemo_relay.resource.measurement_scope".into(),
        json!(measurement_scope),
    );
    // The mark carries the sample timestamp. Adding it as an attribute would
    // create a new OpenTelemetry time series on every poll.
    Value::Object(attributes)
}

fn accelerator_attributes(
    vendor: &AcceleratorVendor,
    device_identifier: &str,
    device_index: Option<u32>,
    process_id: Option<u32>,
) -> Value {
    let mut attributes = json!({
        "nemo_relay.resource.accelerator.vendor": vendor,
        "nemo_relay.resource.accelerator.device_identifier": device_identifier,
    });
    if let Some(device_index) = device_index {
        attributes["nemo_relay.resource.accelerator.device_index"] = json!(device_index);
    }
    if let Some(process_id) = process_id {
        attributes["nemo_relay.resource.process_id"] = json!(process_id);
    }
    attributes
}

fn log_collection_failure(error: &impl std::fmt::Display) {
    log::warn!(
        target: "nemo_relay.plugin",
        event = "resource_metrics_agent_event_failed",
        error_kind = "collection";
        "Resource metrics acquisition did not affect the Agent event: {error}"
    );
}

#[cfg(test)]
#[path = "../../tests/unit/resource_metrics_tests.rs"]
mod tests;
