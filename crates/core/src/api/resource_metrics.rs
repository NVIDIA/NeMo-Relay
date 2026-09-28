// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fresh and managed system resource metric observation.

use nemo_relay_types::api::event::{MetricKind, MetricMeasurement, MetricValueType};
pub use nemo_relay_types::api::resource_metrics::{
    AcceleratorDeviceMetrics, AcceleratorProcessMetrics, AcceleratorVendor,
    ResourceLimitEventCount, ResourceLimitEventKind, ResourceLimitResource, ResourceMeasurement,
    ResourceMeasurementUnit, ResourceMetricsSnapshot, ResourceOperatingSystem,
};
use serde_json::{Value, json};

use crate::api::scope::{self, EmitMetricEventParams, ScopeHandle};
use crate::error::Result;
pub use crate::resource_metrics::config::{
    ResourceMetricsConfig, ResourceMetricsFileConfig, ResourceMetricsPollingConfig,
};
pub use crate::resource_metrics::manager::ResourceMetricsRuntime;
use crate::resource_metrics::manager::{SamplingSeries, global_manager};

/// Canonical event name for resource metric marks.
pub const RESOURCE_METRICS_EVENT_NAME: &str = "nemo.relay.resource_metrics";

impl ResourceMetricsRuntime {
    /// Configure optional polling and structured-file output for the containing process.
    pub fn configure(config: ResourceMetricsConfig) -> Result<Self> {
        Self::configure_current_process(config)
    }

    /// Configure an owned direct child process tree.
    ///
    /// This internal integration entry point rejects unrelated PIDs. The requested root must be a
    /// direct child of the current Relay process when configuration occurs.
    #[doc(hidden)]
    pub fn configure_owned_process_tree(
        config: ResourceMetricsConfig,
        process_id: u32,
    ) -> Result<Self> {
        Self::configure_owned_target(config, process_id)
    }

    /// Configure an owned child process tree and its existing Windows Job Object.
    ///
    /// # Safety
    ///
    /// `job_handle` must be a live Job Object handle owned by this process. Relay duplicates it
    /// before returning and never closes the caller's handle.
    #[cfg(windows)]
    #[doc(hidden)]
    pub unsafe fn configure_owned_process_tree_with_job_handle(
        config: ResourceMetricsConfig,
        process_id: u32,
        job_handle: isize,
    ) -> Result<Self> {
        Self::configure_owned_target_with_job_handle(config, process_id, job_handle)
    }
}

/// Acquire a fresh snapshot without requiring managed polling.
pub fn collect_resource_metrics() -> Result<ResourceMetricsSnapshot> {
    Ok(global_manager()?.collect(SamplingSeries::OnDemand))
}

/// Return the latest successful managed-polling snapshot.
pub fn latest_resource_metrics() -> Result<Option<ResourceMetricsSnapshot>> {
    Ok(global_manager()?.latest())
}

/// Return successful managed-polling snapshots from oldest to newest.
pub fn resource_metrics_history() -> Result<Vec<ResourceMetricsSnapshot>> {
    Ok(global_manager()?.history())
}

pub(crate) fn emit_agent_event_resource_metrics(parent: &ScopeHandle) {
    let manager = match global_manager() {
        Ok(manager) => manager,
        Err(error) => {
            log_collection_failure(&error);
            return;
        }
    };
    let snapshot = manager.collect(SamplingSeries::AgentEvent);
    let measurements = metric_measurements(&snapshot);
    if measurements.is_empty() {
        log::warn!(
            target: "nemo_relay.resource_metrics",
            event = "resource_metrics_mark_skipped",
            reason = "no_available_measurements";
            "Resource metrics mark was skipped because no numerical measurements were available"
        );
        return;
    }
    if let Err(error) = scope::metric(
        EmitMetricEventParams::builder()
            .name(RESOURCE_METRICS_EVENT_NAME)
            .measurements(measurements)
            .parent(parent)
            .timestamp(latest_measurement_timestamp(&snapshot))
            .build(),
    ) {
        log_collection_failure(&error);
    }
}

fn metric_measurements(snapshot: &ResourceMetricsSnapshot) -> Vec<MetricMeasurement> {
    let operating_system = snapshot.operating_system.as_str();
    let mut measurements = Vec::new();
    push_integer(
        &mut measurements,
        "nemo.relay.resource.cpu.user_time",
        "Cumulative user-mode CPU time",
        operating_system,
        &snapshot.cpu_user_time,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.cpu.system_time",
        "Cumulative system-mode CPU time",
        operating_system,
        &snapshot.cpu_system_time,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.cpu.total_time",
        "Cumulative CPU time",
        operating_system,
        &snapshot.cpu_total_time,
    );
    push_float(
        &mut measurements,
        "nemo.relay.resource.cpu.consumption_rate",
        "Native CPU-time units consumed per wall-clock second",
        operating_system,
        &snapshot.cpu_consumption_rate,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.cpu.throttled_time",
        "Cumulative CPU throttled time",
        operating_system,
        &snapshot.cpu_throttled_time,
    );
    push_float(
        &mut measurements,
        "nemo.relay.resource.cpu.effective_limit",
        "Effective CPU capacity",
        operating_system,
        &snapshot.effective_cpu_limit,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.cpu.pressure.some_stall_time",
        "Cumulative CPU pressure time with some work stalled",
        operating_system,
        &snapshot.cpu_some_pressure_stall_time,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.cpu.pressure.full_stall_time",
        "Cumulative CPU pressure time with all work stalled",
        operating_system,
        &snapshot.cpu_full_pressure_stall_time,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.memory.resident",
        "Resident memory",
        operating_system,
        &snapshot.resident_memory,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.memory.private",
        "Private memory",
        operating_system,
        &snapshot.private_memory,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.memory.physical_footprint",
        "Physical memory footprint",
        operating_system,
        &snapshot.physical_footprint,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.memory.virtual",
        "Virtual memory",
        operating_system,
        &snapshot.virtual_memory,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.memory.peak_resident",
        "Peak resident memory",
        operating_system,
        &snapshot.peak_resident_memory,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.memory.limit",
        "Effective memory limit",
        operating_system,
        &snapshot.memory_limit,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.memory.environment_accounted",
        "Environment-accounted memory",
        operating_system,
        &snapshot.environment_accounted_memory,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.memory.pressure.some_stall_time",
        "Cumulative memory pressure time with some work stalled",
        operating_system,
        &snapshot.memory_some_pressure_stall_time,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.memory.pressure.full_stall_time",
        "Cumulative memory pressure time with all work stalled",
        operating_system,
        &snapshot.memory_full_pressure_stall_time,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.memory.out_of_memory_event_count",
        "Cumulative out-of-memory event count",
        operating_system,
        &snapshot.out_of_memory_event_count,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.process.active_count",
        "Active process count",
        operating_system,
        &snapshot.active_process_count,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.process.descendant_count",
        "Descendant process count",
        operating_system,
        &snapshot.descendant_process_count,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.thread.count",
        "Thread count",
        operating_system,
        &snapshot.thread_count,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.process.lifetime_creation_count",
        "Lifetime process creation count",
        operating_system,
        &snapshot.lifetime_process_creation_count,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.process.open_file_descriptor_count",
        "Open file descriptor count",
        operating_system,
        &snapshot.open_file_descriptor_count,
    );
    push_integer(
        &mut measurements,
        "nemo.relay.resource.process.windows_handle_count",
        "Windows handle count",
        operating_system,
        &snapshot.windows_handle_count,
    );
    for event in &snapshot.resource_limit_events {
        push_integer_with_attributes(
            &mut measurements,
            "nemo.relay.resource.limit.event_count",
            "Cumulative resource-limit event count",
            operating_system,
            &event.count,
            json!({
                "nemo_relay.resource.limit.resource": event.resource,
                "nemo_relay.resource.limit.event": event.event,
            }),
        );
    }
    for device in &snapshot.accelerator_devices {
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
            &device.memory_used,
            attributes.clone(),
        );
        push_float_with_attributes(
            &mut measurements,
            "nemo.relay.resource.accelerator.device.compute_utilization",
            "Accelerator device compute utilization",
            operating_system,
            &device.compute_utilization,
            attributes,
        );
    }
    for process in &snapshot.accelerator_processes {
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
            &process.memory_used,
            attributes.clone(),
        );
        push_float_with_attributes(
            &mut measurements,
            "nemo.relay.resource.accelerator.process.compute_utilization",
            "Accelerator compute utilization attributed to an owned process",
            operating_system,
            &process.compute_utilization,
            attributes,
        );
    }
    measurements
}

fn push_float(
    measurements: &mut Vec<MetricMeasurement>,
    name: &str,
    description: &str,
    operating_system: &str,
    measurement: &ResourceMeasurement<f64>,
) {
    push_float_with_attributes(
        measurements,
        name,
        description,
        operating_system,
        measurement,
        json!({}),
    );
}

fn push_float_with_attributes(
    measurements: &mut Vec<MetricMeasurement>,
    name: &str,
    description: &str,
    operating_system: &str,
    measurement: &ResourceMeasurement<f64>,
    additional_attributes: Value,
) {
    let Some(value) = measurement
        .value()
        .copied()
        .filter(|value| value.is_finite())
    else {
        return;
    };
    let Some(unit) = measurement.unit else {
        return;
    };
    measurements.push(
        MetricMeasurement::builder()
            .name(name)
            .kind(MetricKind::Gauge)
            .value_type(MetricValueType::F64)
            .value(json!(value))
            .unit(unit.as_str())
            .description(description)
            .attributes(metric_attributes(
                operating_system,
                measurement,
                additional_attributes,
            ))
            .build(),
    );
}

fn push_integer(
    measurements: &mut Vec<MetricMeasurement>,
    name: &str,
    description: &str,
    operating_system: &str,
    measurement: &ResourceMeasurement<u64>,
) {
    push_integer_with_attributes(
        measurements,
        name,
        description,
        operating_system,
        measurement,
        json!({}),
    );
}

fn push_integer_with_attributes(
    measurements: &mut Vec<MetricMeasurement>,
    name: &str,
    description: &str,
    operating_system: &str,
    measurement: &ResourceMeasurement<u64>,
    additional_attributes: Value,
) {
    let Some(value) = measurement
        .value()
        .copied()
        .filter(|value| *value <= i64::MAX as u64)
    else {
        return;
    };
    let Some(unit) = measurement.unit else {
        return;
    };
    measurements.push(
        MetricMeasurement::builder()
            .name(name)
            .kind(MetricKind::Gauge)
            .value_type(MetricValueType::U64)
            .value(json!(value))
            .unit(unit.as_str())
            .description(description)
            .attributes(metric_attributes(
                operating_system,
                measurement,
                additional_attributes,
            ))
            .build(),
    );
}

fn metric_attributes<T>(
    operating_system: &str,
    measurement: &ResourceMeasurement<T>,
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
        "nemo_relay.resource.measurement_timestamp".into(),
        json!(measurement.timestamp.to_rfc3339()),
    );
    Value::Object(attributes)
}

fn accelerator_attributes(
    vendor: &AcceleratorVendor,
    device_identifier: &str,
    device_index: Option<u32>,
    process_id: Option<u32>,
) -> Value {
    json!({
        "nemo_relay.resource.accelerator.vendor": vendor,
        "nemo_relay.resource.accelerator.device_identifier": device_identifier,
        "nemo_relay.resource.accelerator.device_index": device_index,
        "nemo_relay.resource.process_id": process_id,
    })
}

fn latest_measurement_timestamp(
    snapshot: &ResourceMetricsSnapshot,
) -> chrono::DateTime<chrono::Utc> {
    let mut timestamps = vec![
        snapshot.cpu_user_time.timestamp,
        snapshot.cpu_system_time.timestamp,
        snapshot.cpu_total_time.timestamp,
        snapshot.cpu_consumption_rate.timestamp,
        snapshot.cpu_throttled_time.timestamp,
        snapshot.effective_cpu_limit.timestamp,
        snapshot.cpu_some_pressure_stall_time.timestamp,
        snapshot.cpu_full_pressure_stall_time.timestamp,
        snapshot.resident_memory.timestamp,
        snapshot.private_memory.timestamp,
        snapshot.physical_footprint.timestamp,
        snapshot.virtual_memory.timestamp,
        snapshot.peak_resident_memory.timestamp,
        snapshot.memory_limit.timestamp,
        snapshot.environment_accounted_memory.timestamp,
        snapshot.memory_some_pressure_stall_time.timestamp,
        snapshot.memory_full_pressure_stall_time.timestamp,
        snapshot.out_of_memory_event_count.timestamp,
        snapshot.active_process_count.timestamp,
        snapshot.descendant_process_count.timestamp,
        snapshot.thread_count.timestamp,
        snapshot.lifetime_process_creation_count.timestamp,
        snapshot.open_file_descriptor_count.timestamp,
        snapshot.windows_handle_count.timestamp,
    ];
    timestamps.extend(
        snapshot
            .resource_limit_events
            .iter()
            .map(|event| event.count.timestamp),
    );
    timestamps.extend(snapshot.accelerator_devices.iter().flat_map(|device| {
        [
            device.memory_used.timestamp,
            device.compute_utilization.timestamp,
        ]
    }));
    timestamps.extend(snapshot.accelerator_processes.iter().flat_map(|process| {
        [
            process.memory_used.timestamp,
            process.compute_utilization.timestamp,
        ]
    }));
    timestamps
        .into_iter()
        .max()
        .expect("resource metrics snapshots contain measurements")
}

fn log_collection_failure(error: &impl std::fmt::Display) {
    log::warn!(
        target: "nemo_relay.resource_metrics",
        event = "resource_metrics_agent_event_failed",
        error_kind = "collection";
        "Resource metrics acquisition did not affect the Agent event: {error}"
    );
}

#[cfg(test)]
#[path = "../../tests/unit/resource_metrics_tests.rs"]
mod tests;
