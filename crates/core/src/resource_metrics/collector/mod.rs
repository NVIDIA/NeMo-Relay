// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::io;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use chrono::Utc;
use nemo_relay_types::api::resource_metrics::{
    AcceleratorDeviceMetrics, AcceleratorProcessMetrics, CpuMetrics, DiskMetrics,
    FilesystemCapacityMetrics, GpuMetrics, MemoryMetrics, ProcessMetrics, ProcessSamplingMetadata,
    ResourceLimitEventCount, ResourceLimitEventKind, ResourceLimitResource, ResourceMeasurement,
    ResourceMeasurementScope, ResourceMeasurementUnit, ResourceMetricValue,
    ResourceMetricsSnapshot,
};

use crate::error::{FlowError, Result};
use crate::plugins::resource_metrics::config::ResourceMetricsConfig;

#[cfg(any(target_os = "linux", windows))]
mod accelerator;
#[cfg(target_os = "macos")]
mod accelerator_macos;
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
    pub(crate) disk_read_bytes: Option<u64>,
    pub(crate) disk_write_bytes: Option<u64>,
    pub(crate) disk_read_operations: Option<u64>,
    pub(crate) disk_write_operations: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ProcessSampleConfig {
    pub(super) cpu: bool,
    pub(super) memory: bool,
    pub(super) process: bool,
    pub(super) disk_io: bool,
}

impl ProcessSampleConfig {
    fn any(self) -> bool {
        self.cpu || self.memory || self.process || self.disk_io
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ProcessCpuSample {
    pub(crate) process_id: u32,
    pub(crate) start_identity: u64,
    pub(crate) total_cpu_time_millis: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ProcessIoSample {
    pub(crate) process_id: u32,
    pub(crate) start_identity: u64,
    pub(crate) read_bytes: Option<u64>,
    pub(crate) write_bytes: Option<u64>,
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
    pub(super) devices: Option<Vec<AcceleratorDeviceMetrics>>,
    pub(super) processes: Option<Vec<AcceleratorProcessMetrics>>,
}

#[derive(Default)]
pub(crate) struct SamplingState {
    #[cfg(any(target_os = "linux", windows))]
    accelerator: accelerator::SamplingState,
    #[cfg(target_os = "macos")]
    accelerator_macos: accelerator_macos::SamplingState,
}

#[derive(Debug)]
pub(crate) struct CollectedSnapshot {
    pub(crate) snapshot: ResourceMetricsSnapshot,
    pub(crate) process_sampled_instant: Instant,
    pub(crate) successful: bool,
    pub(crate) process_cpu_samples: Vec<ProcessCpuSample>,
    pub(crate) process_io_samples: Vec<ProcessIoSample>,
}

#[derive(Debug, Default, Clone)]
struct SystemMemorySample {
    used: Option<ResourceMeasurement<ResourceMetricValue>>,
    total: Option<ResourceMeasurement<ResourceMetricValue>>,
    available: Option<ResourceMeasurement<ResourceMetricValue>>,
}

pub(crate) struct GlobalCpuSampler {
    system: sysinfo::System,
    last_sampled_at: Instant,
}

static SYSTEM_MEMORY_SAMPLER: LazyLock<Mutex<sysinfo::System>> = LazyLock::new(|| {
    Mutex::new(sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    ))
});

impl GlobalCpuSampler {
    pub(crate) fn sample(sampler: &mut Option<Self>) -> Option<ResourceMeasurement<f64>> {
        if sampler.is_none() {
            let system = sysinfo::System::new_with_specifics(
                sysinfo::RefreshKind::nothing()
                    .with_cpu(sysinfo::CpuRefreshKind::nothing().with_cpu_usage()),
            );
            *sampler = Some(Self {
                system,
                last_sampled_at: Instant::now(),
            });
            return None;
        }
        let sampler = sampler.as_mut()?;
        let now = Instant::now();
        if now.duration_since(sampler.last_sampled_at) < sysinfo::MINIMUM_CPU_UPDATE_INTERVAL {
            return None;
        }
        sampler.system.refresh_cpu_usage();
        sampler.last_sampled_at = now;
        let cpus = sampler.system.cpus();
        if cpus.is_empty() {
            return None;
        }
        let rate = cpus
            .iter()
            .map(|cpu| f64::from(cpu.cpu_usage()) / 100.0)
            .sum::<f64>();
        rate.is_finite()
            .then(|| ResourceMeasurement::new(rate, ResourceMeasurementUnit::LogicalProcessors))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CollectionIssue {
    Unsupported,
    PermissionDenied,
    TargetTerminated,
    TargetIdentityChanged,
    SourceUnavailable,
}

pub(crate) fn current_process_target(
    measurement_scope: ResourceMeasurementScope,
) -> Result<CollectionTarget> {
    let process_id = std::process::id();
    let start_identity = platform::process_identity(process_id).map_err(|error| {
        FlowError::Internal(format!(
            "failed to identify current process for resource metrics: {error}"
        ))
    })?;
    Ok(CollectionTarget {
        process_id,
        start_identity,
        measurement_scope,
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
        measurement_scope: ResourceMeasurementScope::ProcessTree,
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

#[cfg(test)]
pub(crate) fn collect(
    target: &CollectionTarget,
    config: &ResourceMetricsConfig,
) -> CollectedSnapshot {
    collect_with_state(target, config, &mut SamplingState::default())
}

#[allow(clippy::cognitive_complexity)] // Keep the single collection pass and its coverage checks together.
pub(crate) fn collect_with_state(
    target: &CollectionTarget,
    config: &ResourceMetricsConfig,
    sampling_state: &mut SamplingState,
) -> CollectedSnapshot {
    let sample_config = ProcessSampleConfig {
        cpu: config.cpu.enabled,
        memory: config.memory.enabled
            && target.measurement_scope != ResourceMeasurementScope::Global,
        process: config.process.enabled,
        disk_io: config.disk.enabled && config.disk.process_io,
    };
    let needs_process_ids =
        sample_config.any() || (config.gpu.enabled && config.gpu.process_metrics);
    let process_ids = if !needs_process_ids {
        Ok(Vec::new())
    } else {
        match target.measurement_scope {
            ResourceMeasurementScope::Global => platform::all_process_ids(),
            ResourceMeasurementScope::ApplicationProcess => Ok(vec![target.process_id]),
            ResourceMeasurementScope::ProcessTree => {
                #[cfg(windows)]
                {
                    if let Some(job) = target.job_handle.as_deref() {
                        platform::job_process_ids(job)
                    } else {
                        platform::process_tree_ids(target.process_id)
                    }
                }
                #[cfg(not(windows))]
                {
                    platform::process_tree_ids(target.process_id)
                }
            }
        }
    };
    let (process_ids, process_ids_unavailable) = match process_ids {
        Ok(process_ids) => (process_ids, false),
        Err(error) => {
            log::warn!(
                target: "nemo_relay.plugin",
                event = "resource_metrics_collection_failed",
                error_kind = "source_unavailable";
                "Resource metrics collection failed: {error}"
            );
            (Vec::new(), true)
        }
    };
    if needs_process_ids
        && !process_ids_unavailable
        && !sample_config.any()
        && target.measurement_scope != ResourceMeasurementScope::Global
        && platform::process_identity(target.process_id).ok() != Some(target.start_identity)
    {
        return unavailable_snapshot(target, config);
    }
    let mut samples = Vec::with_capacity(process_ids.len());
    let mut collection_issue = None;
    #[cfg(windows)]
    let thread_counts = if sample_config.process {
        platform::thread_counts().ok()
    } else {
        None
    };
    for process_id in process_ids.iter().filter(|_| sample_config.any()) {
        if target.measurement_scope == ResourceMeasurementScope::Global {
            match platform::process_sample(
                *process_id,
                sample_config,
                #[cfg(windows)]
                thread_counts.as_ref(),
            ) {
                Ok(sample) => samples.push(sample),
                Err(error) => {
                    record_collection_issue(&mut collection_issue, reason_for_io_error(&error))
                }
            }
            continue;
        }
        let expected_start_identity = if *process_id == target.process_id {
            target.start_identity
        } else {
            match platform::process_identity(*process_id) {
                Ok(start_identity) => start_identity,
                Err(error) => {
                    record_collection_issue(&mut collection_issue, reason_for_io_error(&error));
                    continue;
                }
            }
        };
        match platform::process_sample(
            *process_id,
            sample_config,
            #[cfg(windows)]
            thread_counts.as_ref(),
        ) {
            Ok(sample) if sample.start_identity == expected_start_identity => samples.push(sample),
            Ok(_) if *process_id == target.process_id => {
                return unavailable_snapshot(target, config);
            }
            Ok(_) => record_collection_issue(
                &mut collection_issue,
                CollectionIssue::TargetIdentityChanged,
            ),
            Err(_) if *process_id == target.process_id => {
                return unavailable_snapshot(target, config);
            }
            Err(error) => {
                record_collection_issue(&mut collection_issue, reason_for_io_error(&error))
            }
        }
    }
    if sample_config.any()
        && !process_ids_unavailable
        && target.measurement_scope != ResourceMeasurementScope::Global
    {
        let Some(root) = samples
            .iter()
            .find(|sample| sample.process_id == target.process_id)
        else {
            return unavailable_snapshot(target, config);
        };
        if root.start_identity != target.start_identity {
            return unavailable_snapshot(target, config);
        }
        match platform::process_identity(target.process_id) {
            Ok(start_identity) if start_identity == target.start_identity => {}
            _ => return unavailable_snapshot(target, config),
        }
    }
    let process_sampled_instant = Instant::now();

    let complete = !process_ids_unavailable
        && collection_issue.is_none()
        && (!sample_config.any() || samples.len() == process_ids.len());
    let included = samples.iter().collect::<Vec<_>>();
    let mut process_sampling =
        (sample_config.any() && !process_ids_unavailable).then_some(ProcessSamplingMetadata {
            visible_processes: process_ids.len() as u64,
            sampled_processes: samples.len() as u64,
            field_sampled_processes: BTreeMap::new(),
        });
    // A global snapshot may report a partial process sum when coverage is explicit.
    // Other scopes still require every selected process to be sampled.
    let aggregation_complete = complete
        || (target.measurement_scope == ResourceMeasurementScope::Global && !samples.is_empty());
    let mut aggregation = ProcessAggregation {
        samples: &included,
        complete: aggregation_complete,
        allow_partial: target.measurement_scope == ResourceMeasurementScope::Global,
        field_sampled_processes: BTreeMap::new(),
    };
    let process_cpu_samples = if complete
        && config.cpu.enabled
        && target.measurement_scope != ResourceMeasurementScope::Global
    {
        included
            .iter()
            .filter_map(|sample| {
                Some(ProcessCpuSample {
                    process_id: sample.process_id,
                    start_identity: sample.start_identity,
                    total_cpu_time_millis: sample.total_cpu_time?,
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    let process_io_samples =
        if aggregation_complete && config.disk.enabled && config.disk.process_io {
            included
                .iter()
                .map(|sample| ProcessIoSample {
                    process_id: sample.process_id,
                    start_identity: sample.start_identity,
                    read_bytes: sample.disk_read_bytes,
                    write_bytes: sample.disk_write_bytes,
                })
                .collect()
        } else {
            Vec::new()
        };
    let active_process_count = if target.measurement_scope == ResourceMeasurementScope::Global {
        process_ids.len() as u64
    } else {
        included.len() as u64
    };
    let descendant_process_count = active_process_count.saturating_sub(1);
    let environment = if !(config.cpu.enabled || config.memory.enabled || config.process.enabled) {
        EnvironmentSample::default()
    } else if target.measurement_scope == ResourceMeasurementScope::Global {
        platform::global_environment_sample(config).unwrap_or_default()
    } else if complete {
        platform::environment_sample(target, &process_ids, config).unwrap_or_default()
    } else {
        EnvironmentSample::default()
    };
    let mut gpu_config = config.gpu.clone();
    gpu_config.process_metrics &= !process_ids_unavailable
        && (complete || target.measurement_scope == ResourceMeasurementScope::Global);
    let accelerators =
        if gpu_config.enabled && (gpu_config.device_metrics || gpu_config.process_metrics) {
            platform::accelerator_sample(&process_ids, &gpu_config, sampling_state)
        } else {
            AcceleratorSample::default()
        };
    if needs_process_ids
        && !process_ids_unavailable
        && !sample_config.any()
        && target.measurement_scope != ResourceMeasurementScope::Global
        && platform::process_identity(target.process_id).ok() != Some(target.start_identity)
    {
        return unavailable_snapshot(target, config);
    }
    let filesystems = if config.disk.enabled {
        config
            .disk
            .filesystem_paths
            .iter()
            .map(|path| match platform::filesystem_capacity(path) {
                Ok((total, available, free)) => FilesystemCapacityMetrics {
                    path: path.to_string_lossy().into_owned(),
                    total_capacity: available_measurement(total, ResourceMeasurementUnit::Bytes),
                    available_capacity: available_measurement(
                        available,
                        ResourceMeasurementUnit::Bytes,
                    ),
                    free_capacity: available_measurement(free, ResourceMeasurementUnit::Bytes),
                },
                Err(_) => FilesystemCapacityMetrics {
                    path: path.to_string_lossy().into_owned(),
                    total_capacity: unavailable(),
                    available_capacity: unavailable(),
                    free_capacity: unavailable(),
                },
            })
            .collect()
    } else {
        Vec::new()
    };

    let limit_events = environment
        .resource_limit_events
        .into_iter()
        .map(|sample| ResourceLimitEventCount {
            resource: sample.resource,
            event: sample.event,
            count: available_measurement(sample.count, ResourceMeasurementUnit::Events),
        })
        .collect::<Vec<_>>();
    let cpu = config.cpu.enabled.then(|| CpuMetrics {
        user_time: aggregation.sum("cpu.user_time", platform::CPU_TIME_UNIT, |sample| {
            sample.user_cpu_time
        }),
        system_time: aggregation.sum("cpu.system_time", platform::CPU_TIME_UNIT, |sample| {
            sample.system_cpu_time
        }),
        total_time: aggregation.sum("cpu.total_time", platform::CPU_TIME_UNIT, |sample| {
            sample.total_cpu_time
        }),
        consumption_rate: None,
        throttled_time: integer_measurement(environment.cpu_throttled_time),
        effective_limit: float_measurement(environment.effective_cpu_limit),
        some_pressure_stall_time: integer_measurement(environment.cpu_some_pressure_stall_time),
        full_pressure_stall_time: integer_measurement(environment.cpu_full_pressure_stall_time),
        limit_events: limit_events
            .iter()
            .filter(|event| event.resource == ResourceLimitResource::Cpu)
            .cloned()
            .collect(),
    });
    let system_memory = config
        .memory
        .enabled
        .then(system_memory_sample)
        .flatten()
        .unwrap_or_default();
    let process_scope = target.measurement_scope != ResourceMeasurementScope::Global;
    let memory = config.memory.enabled.then(|| MemoryMetrics {
        system_used: system_memory.used,
        system_total: system_memory.total,
        system_available: system_memory.available,
        resident: process_scope
            .then(|| {
                aggregation.sum(
                    "memory.resident",
                    platform::RESIDENT_MEMORY_UNIT,
                    |sample| sample.resident_memory,
                )
            })
            .flatten(),
        private: process_scope
            .then(|| {
                aggregation.sum("memory.private", platform::PRIVATE_MEMORY_UNIT, |sample| {
                    sample.private_memory
                })
            })
            .flatten(),
        physical_footprint: process_scope
            .then(|| {
                aggregation.sum(
                    "memory.physical_footprint",
                    platform::PHYSICAL_FOOTPRINT_UNIT,
                    |sample| sample.physical_footprint,
                )
            })
            .flatten(),
        virtual_memory: process_scope
            .then(|| {
                aggregation.sum(
                    "memory.virtual_memory",
                    platform::VIRTUAL_MEMORY_UNIT,
                    |sample| sample.virtual_memory,
                )
            })
            .flatten(),
        peak_resident: if target.measurement_scope == ResourceMeasurementScope::ApplicationProcess {
            aggregation.sum(
                "memory.peak_resident",
                platform::PEAK_RESIDENT_MEMORY_UNIT,
                |sample| sample.peak_resident_memory,
            )
        } else {
            None
        },
        limit: process_scope
            .then(|| integer_measurement(environment.memory_limit))
            .flatten(),
        environment_accounted: process_scope
            .then(|| integer_measurement(environment.environment_accounted_memory))
            .flatten(),
        some_pressure_stall_time: integer_measurement(environment.memory_some_pressure_stall_time),
        full_pressure_stall_time: integer_measurement(environment.memory_full_pressure_stall_time),
        out_of_memory_event_count: process_scope
            .then(|| integer_measurement(environment.out_of_memory_event_count))
            .flatten(),
        limit_events: limit_events
            .iter()
            .filter(|event| event.resource == ResourceLimitResource::Memory)
            .cloned()
            .collect(),
    });
    let process = config.process.enabled.then(|| ProcessMetrics {
        active_count: count_measurement(
            active_process_count,
            !process_ids_unavailable
                && (target.measurement_scope == ResourceMeasurementScope::Global || complete),
            ResourceMeasurementUnit::Processes,
        ),
        descendant_count: process_scope
            .then(|| {
                count_measurement(
                    descendant_process_count,
                    complete,
                    ResourceMeasurementUnit::Processes,
                )
            })
            .flatten(),
        thread_count: aggregation.sum(
            "process.thread_count",
            ResourceMeasurementUnit::Threads,
            |sample| sample.thread_count,
        ),
        lifetime_creation_count: integer_measurement(environment.lifetime_process_creation_count),
        open_file_descriptor_count: aggregation.sum(
            "process.open_file_descriptor_count",
            ResourceMeasurementUnit::FileDescriptors,
            |sample| sample.open_file_descriptor_count,
        ),
        windows_handle_count: aggregation.sum(
            "process.windows_handle_count",
            ResourceMeasurementUnit::Handles,
            |sample| sample.windows_handle_count,
        ),
        limit_events: limit_events
            .iter()
            .filter(|event| event.resource == ResourceLimitResource::Processes)
            .cloned()
            .collect(),
    });
    let disk = config.disk.enabled.then(|| DiskMetrics {
        read_data: config
            .disk
            .process_io
            .then(|| {
                aggregation.sum("disk.read_data", ResourceMeasurementUnit::Bytes, |sample| {
                    sample.disk_read_bytes
                })
            })
            .flatten(),
        write_data: config
            .disk
            .process_io
            .then(|| {
                aggregation.sum(
                    "disk.write_data",
                    ResourceMeasurementUnit::Bytes,
                    |sample| sample.disk_write_bytes,
                )
            })
            .flatten(),
        read_throughput: None,
        write_throughput: None,
        read_operations: config
            .disk
            .process_io
            .then(|| {
                aggregation.sum(
                    "disk.read_operations",
                    ResourceMeasurementUnit::Operations,
                    |sample| sample.disk_read_operations,
                )
            })
            .flatten(),
        write_operations: config
            .disk
            .process_io
            .then(|| {
                aggregation.sum(
                    "disk.write_operations",
                    ResourceMeasurementUnit::Operations,
                    |sample| sample.disk_write_operations,
                )
            })
            .flatten(),
        filesystems,
    });
    let gpu = config.gpu.enabled.then(|| GpuMetrics {
        device_metrics: config
            .gpu
            .device_metrics
            .then_some(accelerators.devices)
            .flatten(),
        process_metrics: config
            .gpu
            .process_metrics
            .then_some(accelerators.processes)
            .flatten(),
    });
    let timestamp = Utc::now();
    if let Some(coverage) = &mut process_sampling {
        coverage.field_sampled_processes = aggregation.field_sampled_processes;
    }
    let snapshot = ResourceMetricsSnapshot {
        timestamp,
        operating_system: platform::OPERATING_SYSTEM,
        measurement_scope: target.measurement_scope,
        process_sampling,
        cpu,
        memory,
        process,
        disk,
        gpu,
        network: None,
    };
    CollectedSnapshot {
        snapshot,
        process_sampled_instant,
        successful: true,
        process_cpu_samples,
        process_io_samples,
    }
}

#[cfg(any(target_os = "linux", windows, test))]
pub(super) fn accelerator_device_is_selected(
    config: &crate::plugins::resource_metrics::config::ResourceMetricsGpuConfig,
    identifier: &str,
    index: u32,
) -> bool {
    config.devices.is_empty()
        || config
            .devices
            .iter()
            .any(|selector| selector == identifier || selector == &index.to_string())
}

struct ProcessAggregation<'a> {
    samples: &'a [&'a ProcessSample],
    complete: bool,
    allow_partial: bool,
    field_sampled_processes: BTreeMap<String, u64>,
}

impl ProcessAggregation<'_> {
    fn sum(
        &mut self,
        field: &str,
        unit: ResourceMeasurementUnit,
        value: impl Fn(&ProcessSample) -> Option<u64>,
    ) -> Option<ResourceMeasurement<ResourceMetricValue>> {
        let mut supplied = 0_u64;
        let mut total = Some(0_u64);
        for sample in self.samples {
            if let Some(value) = value(sample) {
                supplied += 1;
                total = total.and_then(|total| total.checked_add(value));
            }
        }
        self.field_sampled_processes.insert(field.into(), supplied);
        if !self.complete
            || supplied == 0
            || (!self.allow_partial && supplied != self.samples.len() as u64)
        {
            return unavailable();
        }
        available_measurement(total?, unit)
    }
}

fn count_measurement(
    value: u64,
    complete: bool,
    unit: ResourceMeasurementUnit,
) -> Option<ResourceMeasurement<ResourceMetricValue>> {
    if complete {
        available_measurement(value, unit)
    } else {
        unavailable()
    }
}

fn system_memory_sample() -> Option<SystemMemorySample> {
    let mut system = SYSTEM_MEMORY_SAMPLER
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    system.refresh_memory();
    let total = system.total_memory();
    if total == 0 {
        return None;
    }
    let used = system.used_memory();
    let available = system.available_memory();
    Some(SystemMemorySample {
        used: available_measurement(bytes_to_kibibytes(used), ResourceMeasurementUnit::Kibibytes),
        total: available_measurement(
            bytes_to_kibibytes(total),
            ResourceMeasurementUnit::Kibibytes,
        ),
        available: available_measurement(
            bytes_to_kibibytes(available),
            ResourceMeasurementUnit::Kibibytes,
        ),
    })
}

fn integer_measurement(
    sample: Option<IntegerSample>,
) -> Option<ResourceMeasurement<ResourceMetricValue>> {
    sample.map_or_else(unavailable, |sample| {
        available_measurement(sample.value, sample.unit)
    })
}

fn float_measurement(sample: Option<FloatSample>) -> Option<ResourceMeasurement<f64>> {
    sample.map_or_else(unavailable, |sample| {
        available_measurement(sample.value, sample.unit)
    })
}

fn available_measurement<T>(
    value: impl Into<T>,
    unit: ResourceMeasurementUnit,
) -> Option<ResourceMeasurement<T>> {
    Some(ResourceMeasurement::new(value, unit))
}

fn record_collection_issue(current: &mut Option<CollectionIssue>, next: CollectionIssue) {
    if current.is_none() || next == CollectionIssue::PermissionDenied {
        *current = Some(next);
    }
}

fn unavailable_snapshot(
    target: &CollectionTarget,
    config: &ResourceMetricsConfig,
) -> CollectedSnapshot {
    let completed_at = Utc::now();
    CollectedSnapshot {
        snapshot: ResourceMetricsSnapshot {
            timestamp: completed_at,
            operating_system: platform::OPERATING_SYSTEM,
            measurement_scope: target.measurement_scope,
            process_sampling: None,
            cpu: config.cpu.enabled.then(|| CpuMetrics {
                user_time: None,
                system_time: None,
                total_time: None,
                consumption_rate: None,
                throttled_time: None,
                effective_limit: None,
                some_pressure_stall_time: None,
                full_pressure_stall_time: None,
                limit_events: Vec::new(),
            }),
            memory: config.memory.enabled.then(|| MemoryMetrics {
                system_used: None,
                system_total: None,
                system_available: None,
                resident: None,
                private: None,
                physical_footprint: None,
                virtual_memory: None,
                peak_resident: None,
                limit: None,
                environment_accounted: None,
                some_pressure_stall_time: None,
                full_pressure_stall_time: None,
                out_of_memory_event_count: None,
                limit_events: Vec::new(),
            }),
            process: config.process.enabled.then(|| ProcessMetrics {
                active_count: None,
                descendant_count: None,
                thread_count: None,
                lifetime_creation_count: None,
                open_file_descriptor_count: None,
                windows_handle_count: None,
                limit_events: Vec::new(),
            }),
            disk: config.disk.enabled.then(|| DiskMetrics {
                read_data: None,
                write_data: None,
                read_throughput: None,
                write_throughput: None,
                read_operations: None,
                write_operations: None,
                filesystems: Vec::new(),
            }),
            gpu: config.gpu.enabled.then_some(GpuMetrics {
                device_metrics: None,
                process_metrics: None,
            }),
            network: None,
        },
        process_sampled_instant: Instant::now(),
        successful: false,
        process_cpu_samples: Vec::new(),
        process_io_samples: Vec::new(),
    }
}

fn unavailable<T>() -> Option<ResourceMeasurement<T>> {
    None
}

/// Convert an integral number of nanoseconds to whole milliseconds (truncate).
#[cfg(test)]
pub(crate) const fn nanoseconds_to_milliseconds(value: u64) -> u64 {
    value / 1_000_000
}

/// Convert Mach ticks to whole milliseconds using the platform timebase (truncate).
#[cfg(any(target_os = "macos", test))]
pub(crate) fn mach_ticks_to_milliseconds(
    ticks: u64,
    numerator: u32,
    denominator: u32,
) -> Option<u64> {
    if numerator == 0 || denominator == 0 {
        return None;
    }
    let milliseconds =
        (u128::from(ticks) * u128::from(numerator)) / (u128::from(denominator) * 1_000_000);
    u64::try_from(milliseconds).ok()
}

/// Convert a Windows FILETIME value (100 ns intervals) to whole milliseconds (truncate).
#[cfg(any(windows, test))]
pub(crate) const fn filetime_intervals_to_milliseconds(value: u64) -> u64 {
    value / 10_000
}

/// Convert microseconds to whole milliseconds (truncate).
#[cfg(test)]
pub(crate) const fn microseconds_to_milliseconds(value: u64) -> u64 {
    value / 1_000
}

/// Convert byte counts to whole KiB (truncate).
pub(crate) const fn bytes_to_kibibytes(value: u64) -> u64 {
    value / 1_024
}

/// Convert process-accounting ticks to whole milliseconds using the runtime tick rate.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn clock_ticks_to_milliseconds(value: u64, ticks_per_second: u64) -> Option<u64> {
    (ticks_per_second > 0)
        .then(|| (u128::from(value) * 1_000) / u128::from(ticks_per_second))
        .and_then(|milliseconds| u64::try_from(milliseconds).ok())
}

#[cfg(test)]
#[path = "../../../tests/unit/resource_metrics/collector_tests.rs"]
mod conversion_tests;

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
