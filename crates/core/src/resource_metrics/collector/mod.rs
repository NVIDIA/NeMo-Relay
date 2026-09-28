// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io;

use chrono::{DateTime, Utc};
use nemo_relay_types::api::resource_metrics::{
    AcceleratorDeviceMetrics, AcceleratorProcessMetrics, ResourceLimitEventCount,
    ResourceLimitEventKind, ResourceLimitResource, ResourceMeasurement, ResourceMeasurementScope,
    ResourceMeasurementUnit, ResourceMetricsSnapshot,
};

use crate::error::{FlowError, Result};

#[cfg(any(target_os = "linux", windows))]
mod accelerator;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
use linux as platform;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos as platform;
#[cfg(windows)]
mod windows;
#[cfg(windows)]
use windows as platform;
#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod unsupported;
#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
use unsupported as platform;

#[derive(Debug, Clone)]
pub(crate) struct CollectionTarget {
    pub(crate) process_id: u32,
    pub(crate) start_identity: u64,
    pub(crate) measurement_scope: ResourceMeasurementScope,
    #[cfg(windows)]
    job_handle: Option<std::sync::Arc<platform::OwnedJobHandle>>,
}

#[derive(Debug, Clone)]
pub(crate) struct ProcessSample {
    pub(crate) process_id: u32,
    pub(crate) start_identity: u64,
    pub(crate) user_cpu_time: Option<u64>,
    pub(crate) system_cpu_time: Option<u64>,
    pub(crate) total_cpu_time: Option<u64>,
    pub(crate) resident_memory: Option<u64>,
    pub(crate) private_memory: Option<u64>,
    pub(crate) physical_footprint: Option<u64>,
    pub(crate) virtual_memory: Option<u64>,
    pub(crate) peak_resident_memory: Option<u64>,
    pub(crate) thread_count: Option<u64>,
    pub(crate) open_file_descriptor_count: Option<u64>,
    pub(crate) windows_handle_count: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct IntegerSample {
    pub(super) value: u64,
    pub(super) unit: ResourceMeasurementUnit,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FloatSample {
    pub(super) value: f64,
    pub(super) unit: ResourceMeasurementUnit,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LimitEventSample {
    pub(super) resource: ResourceLimitResource,
    pub(super) event: ResourceLimitEventKind,
    pub(super) count: u64,
}

#[derive(Debug, Default)]
pub(super) struct EnvironmentSample {
    pub(super) cpu_throttled_time: Option<IntegerSample>,
    pub(super) effective_cpu_limit: Option<FloatSample>,
    pub(super) cpu_some_pressure_stall_time: Option<IntegerSample>,
    pub(super) cpu_full_pressure_stall_time: Option<IntegerSample>,
    pub(super) memory_limit: Option<IntegerSample>,
    pub(super) environment_accounted_memory: Option<IntegerSample>,
    pub(super) memory_some_pressure_stall_time: Option<IntegerSample>,
    pub(super) memory_full_pressure_stall_time: Option<IntegerSample>,
    pub(super) out_of_memory_event_count: Option<IntegerSample>,
    pub(super) lifetime_process_creation_count: Option<IntegerSample>,
    pub(super) resource_limit_events: Vec<LimitEventSample>,
}

#[derive(Debug, Default)]
pub(super) struct AcceleratorSample {
    pub(super) devices: Vec<AcceleratorDeviceMetrics>,
    pub(super) processes: Vec<AcceleratorProcessMetrics>,
}

#[derive(Debug)]
pub(crate) struct CollectedSnapshot {
    pub(crate) snapshot: ResourceMetricsSnapshot,
    pub(crate) completed_at: DateTime<Utc>,
    pub(crate) successful: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CollectionIssue {
    Unsupported,
    PermissionDenied,
    TargetTerminated,
    TargetIdentityChanged,
    SourceUnavailable,
}

pub(crate) fn current_process_target() -> Result<CollectionTarget> {
    let process_id = std::process::id();
    let start_identity = platform::process_identity(process_id).map_err(|error| {
        FlowError::Internal(format!(
            "failed to identify current process for resource metrics: {error}"
        ))
    })?;
    Ok(CollectionTarget {
        process_id,
        start_identity,
        measurement_scope: ResourceMeasurementScope::ApplicationProcess,
        #[cfg(windows)]
        job_handle: None,
    })
}

pub(crate) fn owned_process_tree_target(process_id: u32) -> Result<CollectionTarget> {
    let current_process_id = std::process::id();
    let parent_process_id = platform::parent_process_id(process_id).map_err(platform_error)?;
    if parent_process_id != Some(current_process_id) {
        return Err(FlowError::InvalidArgument(format!(
            "resource metrics target {process_id} is not a direct child of Relay process {current_process_id}"
        )));
    }
    Ok(CollectionTarget {
        process_id,
        start_identity: platform::process_identity(process_id).map_err(platform_error)?,
        measurement_scope: ResourceMeasurementScope::OwnedProcessTree,
        #[cfg(windows)]
        job_handle: None,
    })
}

#[cfg(windows)]
pub(crate) fn owned_process_tree_target_with_job_handle(
    process_id: u32,
    job_handle: isize,
) -> Result<CollectionTarget> {
    let mut target = owned_process_tree_target(process_id)?;
    target.job_handle = Some(std::sync::Arc::new(
        platform::OwnedJobHandle::duplicate(job_handle).map_err(platform_error)?,
    ));
    Ok(target)
}

pub(crate) fn collect(target: &CollectionTarget) -> CollectedSnapshot {
    let process_ids = match target.measurement_scope {
        ResourceMeasurementScope::ApplicationProcess => Ok(vec![target.process_id]),
        ResourceMeasurementScope::OwnedProcessTree => platform::process_tree_ids(target.process_id),
    };
    let process_ids = match process_ids {
        Ok(process_ids) => process_ids,
        Err(error) => {
            log::warn!(
                target: "nemo_relay.resource_metrics",
                event = "resource_metrics_collection_failed",
                error_kind = "source_unavailable";
                "Resource metrics collection failed: {error}"
            );
            return unavailable_snapshot();
        }
    };
    let mut samples = Vec::with_capacity(process_ids.len());
    let mut expected_process_count = process_ids.len();
    let mut collection_issue = None;
    for process_id in &process_ids {
        let expected_start_identity = if *process_id == target.process_id {
            target.start_identity
        } else {
            match platform::process_identity(*process_id) {
                Ok(start_identity) => start_identity,
                Err(error) if is_process_terminated_error(&error) => {
                    expected_process_count = expected_process_count.saturating_sub(1);
                    continue;
                }
                Err(error) => {
                    record_collection_issue(&mut collection_issue, reason_for_io_error(&error));
                    continue;
                }
            }
        };
        match platform::process_sample(*process_id) {
            Ok(sample) if sample.start_identity == expected_start_identity => samples.push(sample),
            Ok(_) if *process_id == target.process_id => {
                return unavailable_snapshot();
            }
            Ok(_) => record_collection_issue(
                &mut collection_issue,
                CollectionIssue::TargetIdentityChanged,
            ),
            Err(_) if *process_id == target.process_id => return unavailable_snapshot(),
            Err(error) if is_process_terminated_error(&error) => {
                expected_process_count = expected_process_count.saturating_sub(1);
            }
            Err(error) => {
                record_collection_issue(&mut collection_issue, reason_for_io_error(&error))
            }
        }
    }
    let Some(root) = samples
        .iter()
        .find(|sample| sample.process_id == target.process_id)
    else {
        return unavailable_snapshot();
    };
    if root.start_identity != target.start_identity {
        return unavailable_snapshot();
    }
    match platform::process_identity(target.process_id) {
        Ok(start_identity) if start_identity == target.start_identity => {}
        _ => return unavailable_snapshot(),
    }

    let complete = collection_issue.is_none() && samples.len() == expected_process_count;
    let included = samples.iter().collect::<Vec<_>>();
    let live_process_ids = samples
        .iter()
        .map(|sample| sample.process_id)
        .collect::<Vec<_>>();
    let active_process_count = included.len() as u64;
    let descendant_process_count = active_process_count.saturating_sub(1);
    let environment = if complete {
        platform::environment_sample(target, &live_process_ids).unwrap_or_default()
    } else {
        EnvironmentSample::default()
    };
    let accelerators = if complete {
        platform::accelerator_sample(&live_process_ids)
    } else {
        AcceleratorSample::default()
    };

    let snapshot = ResourceMetricsSnapshot {
        operating_system: platform::OPERATING_SYSTEM,
        cpu_user_time: sum_integer_measurements(
            &included,
            complete,
            platform::CPU_TIME_UNIT,
            |sample| sample.user_cpu_time,
        ),
        cpu_system_time: sum_integer_measurements(
            &included,
            complete,
            platform::CPU_TIME_UNIT,
            |sample| sample.system_cpu_time,
        ),
        cpu_total_time: sum_integer_measurements(
            &included,
            complete,
            platform::CPU_TIME_UNIT,
            |sample| sample.total_cpu_time,
        ),
        cpu_consumption_rate: unavailable(),
        cpu_throttled_time: integer_measurement(environment.cpu_throttled_time),
        effective_cpu_limit: float_measurement(environment.effective_cpu_limit),
        cpu_some_pressure_stall_time: integer_measurement(environment.cpu_some_pressure_stall_time),
        cpu_full_pressure_stall_time: integer_measurement(environment.cpu_full_pressure_stall_time),
        resident_memory: sum_integer_measurements(
            &included,
            complete,
            platform::RESIDENT_MEMORY_UNIT,
            |sample| sample.resident_memory,
        ),
        private_memory: sum_integer_measurements(
            &included,
            complete,
            platform::PRIVATE_MEMORY_UNIT,
            |sample| sample.private_memory,
        ),
        physical_footprint: sum_integer_measurements(
            &included,
            complete,
            platform::PHYSICAL_FOOTPRINT_UNIT,
            |sample| sample.physical_footprint,
        ),
        virtual_memory: sum_integer_measurements(
            &included,
            complete,
            platform::VIRTUAL_MEMORY_UNIT,
            |sample| sample.virtual_memory,
        ),
        peak_resident_memory: if target.measurement_scope
            == ResourceMeasurementScope::ApplicationProcess
        {
            sum_integer_measurements(
                &included,
                complete,
                platform::PEAK_RESIDENT_MEMORY_UNIT,
                |sample| sample.peak_resident_memory,
            )
        } else {
            unavailable()
        },
        memory_limit: integer_measurement(environment.memory_limit),
        environment_accounted_memory: integer_measurement(environment.environment_accounted_memory),
        memory_some_pressure_stall_time: integer_measurement(
            environment.memory_some_pressure_stall_time,
        ),
        memory_full_pressure_stall_time: integer_measurement(
            environment.memory_full_pressure_stall_time,
        ),
        out_of_memory_event_count: integer_measurement(environment.out_of_memory_event_count),
        active_process_count: count_measurement(
            active_process_count,
            complete,
            ResourceMeasurementUnit::Processes,
        ),
        descendant_process_count: count_measurement(
            descendant_process_count,
            complete,
            ResourceMeasurementUnit::Processes,
        ),
        thread_count: sum_integer_measurements(
            &included,
            complete,
            ResourceMeasurementUnit::Threads,
            |sample| sample.thread_count,
        ),
        lifetime_process_creation_count: integer_measurement(
            environment.lifetime_process_creation_count,
        ),
        open_file_descriptor_count: sum_integer_measurements(
            &included,
            complete,
            ResourceMeasurementUnit::FileDescriptors,
            |sample| sample.open_file_descriptor_count,
        ),
        windows_handle_count: sum_integer_measurements(
            &included,
            complete,
            ResourceMeasurementUnit::Handles,
            |sample| sample.windows_handle_count,
        ),
        resource_limit_events: environment
            .resource_limit_events
            .into_iter()
            .map(|sample| ResourceLimitEventCount {
                resource: sample.resource,
                event: sample.event,
                count: ResourceMeasurement::available(
                    Utc::now(),
                    sample.count,
                    ResourceMeasurementUnit::Events,
                ),
            })
            .collect(),
        accelerator_devices: accelerators.devices,
        accelerator_processes: accelerators.processes,
    };
    CollectedSnapshot {
        snapshot,
        completed_at: Utc::now(),
        successful: true,
    }
}

fn sum_integer_measurements(
    samples: &[&ProcessSample],
    complete: bool,
    unit: ResourceMeasurementUnit,
    value: impl Fn(&ProcessSample) -> Option<u64>,
) -> ResourceMeasurement<u64> {
    if !complete {
        return unavailable();
    }
    let Some(total) = samples.iter().try_fold(0_u64, |total, sample| {
        value(sample).and_then(|value| total.checked_add(value))
    }) else {
        return unavailable();
    };
    ResourceMeasurement::available(Utc::now(), total, unit)
}

fn count_measurement(
    value: u64,
    complete: bool,
    unit: ResourceMeasurementUnit,
) -> ResourceMeasurement<u64> {
    if complete {
        ResourceMeasurement::available(Utc::now(), value, unit)
    } else {
        unavailable()
    }
}

fn integer_measurement(sample: Option<IntegerSample>) -> ResourceMeasurement<u64> {
    sample.map_or_else(unavailable, |sample| {
        ResourceMeasurement::available(Utc::now(), sample.value, sample.unit)
    })
}

fn float_measurement(sample: Option<FloatSample>) -> ResourceMeasurement<f64> {
    sample.map_or_else(unavailable, |sample| {
        ResourceMeasurement::available(Utc::now(), sample.value, sample.unit)
    })
}

fn record_collection_issue(current: &mut Option<CollectionIssue>, next: CollectionIssue) {
    if current.is_none() || next == CollectionIssue::PermissionDenied {
        *current = Some(next);
    }
}

fn unavailable_snapshot() -> CollectedSnapshot {
    CollectedSnapshot {
        snapshot: ResourceMetricsSnapshot {
            operating_system: platform::OPERATING_SYSTEM,
            cpu_user_time: unavailable(),
            cpu_system_time: unavailable(),
            cpu_total_time: unavailable(),
            cpu_consumption_rate: unavailable(),
            cpu_throttled_time: unavailable(),
            effective_cpu_limit: unavailable(),
            cpu_some_pressure_stall_time: unavailable(),
            cpu_full_pressure_stall_time: unavailable(),
            resident_memory: unavailable(),
            private_memory: unavailable(),
            physical_footprint: unavailable(),
            virtual_memory: unavailable(),
            peak_resident_memory: unavailable(),
            memory_limit: unavailable(),
            environment_accounted_memory: unavailable(),
            memory_some_pressure_stall_time: unavailable(),
            memory_full_pressure_stall_time: unavailable(),
            out_of_memory_event_count: unavailable(),
            active_process_count: unavailable(),
            descendant_process_count: unavailable(),
            thread_count: unavailable(),
            lifetime_process_creation_count: unavailable(),
            open_file_descriptor_count: unavailable(),
            windows_handle_count: unavailable(),
            resource_limit_events: Vec::new(),
            accelerator_devices: Vec::new(),
            accelerator_processes: Vec::new(),
        },
        completed_at: Utc::now(),
        successful: false,
    }
}

fn unavailable<T>() -> ResourceMeasurement<T> {
    ResourceMeasurement::unavailable(Utc::now())
}

fn platform_error(error: io::Error) -> FlowError {
    FlowError::Internal(format!(
        "failed to inspect resource metrics target: {error}"
    ))
}

fn reason_for_io_error(error: &io::Error) -> CollectionIssue {
    match error.kind() {
        io::ErrorKind::NotFound => CollectionIssue::TargetTerminated,
        io::ErrorKind::PermissionDenied => CollectionIssue::PermissionDenied,
        io::ErrorKind::Unsupported => CollectionIssue::Unsupported,
        _ => CollectionIssue::SourceUnavailable,
    }
}

fn is_process_terminated_error(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::NotFound {
        return true;
    }
    #[cfg(unix)]
    {
        error.raw_os_error() == Some(3) // ESRCH
    }
    #[cfg(windows)]
    {
        error.raw_os_error() == Some(87) // ERROR_INVALID_PARAMETER for an exited PID
    }
    #[cfg(not(any(unix, windows)))]
    {
        false
    }
}
