// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structured system resource metric snapshot types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// The Relay-owned process boundary measured by a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceMeasurementScope {
    /// The process containing the Relay runtime.
    ApplicationProcess,
    /// A direct child process and every descendant still owned by that child.
    OwnedProcessTree,
}

/// Operating system that produced a resource metrics snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceOperatingSystem {
    /// Linux.
    Linux,
    /// macOS.
    Macos,
    /// Windows.
    Windows,
    /// A platform without a supported resource metrics collector.
    Unsupported,
}

impl ResourceOperatingSystem {
    /// Return the stable wire name used by metric projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Linux => "linux",
            Self::Macos => "macos",
            Self::Windows => "windows",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Native or mathematically defined unit for one resource measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceMeasurementUnit {
    /// Nanoseconds.
    Nanoseconds,
    /// Microseconds.
    Microseconds,
    /// Linux process-accounting clock ticks.
    ClockTicks,
    /// Windows `FILETIME` intervals of one hundred nanoseconds.
    HundredNanosecondIntervals,
    /// Native macOS CPU nanoseconds consumed per wall-clock second.
    NanosecondsPerSecond,
    /// Native Linux CPU clock ticks consumed per wall-clock second.
    ClockTicksPerSecond,
    /// Native Windows hundred-nanosecond CPU intervals consumed per wall-clock second.
    HundredNanosecondIntervalsPerSecond,
    /// Bytes.
    Bytes,
    /// Kibibytes as reported by Linux procfs.
    Kibibytes,
    /// Native operating-system memory pages.
    Pages,
    /// Logical processor equivalents, including fractional processors.
    LogicalProcessors,
    /// Percentage from zero through one hundred.
    Percentage,
    /// Process count.
    Processes,
    /// Thread count.
    Threads,
    /// Open file-descriptor count.
    FileDescriptors,
    /// Windows kernel-handle count.
    Handles,
    /// Resource-limit event count.
    Events,
}

impl ResourceMeasurementUnit {
    /// Return the stable wire name used by metric projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nanoseconds => "nanoseconds",
            Self::Microseconds => "microseconds",
            Self::ClockTicks => "clock_ticks",
            Self::HundredNanosecondIntervals => "hundred_nanosecond_intervals",
            Self::NanosecondsPerSecond => "nanoseconds_per_second",
            Self::ClockTicksPerSecond => "clock_ticks_per_second",
            Self::HundredNanosecondIntervalsPerSecond => "hundred_nanosecond_intervals_per_second",
            Self::Bytes => "bytes",
            Self::Kibibytes => "kibibytes",
            Self::Pages => "pages",
            Self::LogicalProcessors => "logical_processors",
            Self::Percentage => "percentage",
            Self::Processes => "processes",
            Self::Threads => "threads",
            Self::FileDescriptors => "file_descriptors",
            Self::Handles => "handles",
            Self::Events => "events",
        }
    }
}

/// One timestamped resource measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceMeasurement<T> {
    /// Acquired value, or no value when it cannot be acquired accurately.
    pub value: Option<T>,
    /// Unit for the acquired value. This is null whenever `value` is null.
    pub unit: Option<ResourceMeasurementUnit>,
    /// UTC time at which Relay finalized this individual measurement.
    pub timestamp: DateTime<Utc>,
}

impl<T> ResourceMeasurement<T> {
    /// Construct an acquired measurement.
    pub const fn available(
        timestamp: DateTime<Utc>,
        value: T,
        unit: ResourceMeasurementUnit,
    ) -> Self {
        Self {
            value: Some(value),
            unit: Some(unit),
            timestamp,
        }
    }

    /// Construct an unavailable measurement.
    pub const fn unavailable(timestamp: DateTime<Utc>) -> Self {
        Self {
            value: None,
            unit: None,
            timestamp,
        }
    }

    /// Return the acquired value.
    pub const fn value(&self) -> Option<&T> {
        self.value.as_ref()
    }
}

/// Resource whose configured boundary produced a limit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceLimitResource {
    /// CPU capacity.
    Cpu,
    /// Memory capacity.
    Memory,
    /// Process capacity.
    Processes,
}

/// Direct operating-system resource-limit event kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceLimitEventKind {
    /// CPU execution was throttled.
    Throttled,
    /// A high threshold was crossed.
    High,
    /// A configured maximum was reached.
    Maximum,
    /// An out-of-memory condition occurred.
    OutOfMemory,
    /// A process was terminated by the resource boundary.
    Terminated,
}

/// One typed cumulative resource-limit event counter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceLimitEventCount {
    /// Limited resource.
    pub resource: ResourceLimitResource,
    /// Event reported by the operating system.
    pub event: ResourceLimitEventKind,
    /// Cumulative event count.
    pub count: ResourceMeasurement<u64>,
}

/// Accelerator vendor associated with a metric record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorVendor {
    /// NVIDIA accelerator.
    Nvidia,
    /// AMD accelerator.
    Amd,
    /// Intel accelerator.
    Intel,
    /// Apple accelerator.
    Apple,
    /// Another vendor exposed through a supported operating-system interface.
    Other,
}

/// Device-wide accelerator measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorDeviceMetrics {
    /// Device vendor.
    pub vendor: AcceleratorVendor,
    /// Stable device identifier supplied by the acquisition API.
    pub device_identifier: String,
    /// API-local device index, when supplied by the acquisition API.
    pub device_index: Option<u32>,
    /// Device memory currently in use.
    pub memory_used: ResourceMeasurement<u64>,
    /// Device compute-engine utilization.
    pub compute_utilization: ResourceMeasurement<f64>,
}

/// Accelerator measurements attributed to one Relay-owned process.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorProcessMetrics {
    /// Device vendor.
    pub vendor: AcceleratorVendor,
    /// Stable device identifier supplied by the acquisition API.
    pub device_identifier: String,
    /// API-local device index, when supplied by the acquisition API.
    pub device_index: Option<u32>,
    /// Relay-owned process identifier.
    pub process_id: u32,
    /// Device memory attributed to this process.
    pub memory_used: ResourceMeasurement<u64>,
    /// Compute-engine utilization attributed to this process.
    pub compute_utilization: ResourceMeasurement<f64>,
}

/// One observation of Relay-owned system resources.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceMetricsSnapshot {
    /// Operating system that produced this snapshot.
    pub operating_system: ResourceOperatingSystem,
    /// Cumulative user-mode CPU time in the platform's native unit.
    pub cpu_user_time: ResourceMeasurement<u64>,
    /// Cumulative system-mode CPU time in the platform's native unit.
    pub cpu_system_time: ResourceMeasurement<u64>,
    /// Cumulative user plus system CPU time in the platform's native unit.
    pub cpu_total_time: ResourceMeasurement<u64>,
    /// Native CPU-time units consumed per wall-clock second.
    pub cpu_consumption_rate: ResourceMeasurement<f64>,
    /// Cumulative time during which the owned environment was CPU-throttled.
    pub cpu_throttled_time: ResourceMeasurement<u64>,
    /// Effective CPU capacity in logical processor equivalents.
    pub effective_cpu_limit: ResourceMeasurement<f64>,
    /// Cumulative CPU pressure time during which some work was stalled.
    pub cpu_some_pressure_stall_time: ResourceMeasurement<u64>,
    /// Cumulative CPU pressure time during which all work was stalled.
    pub cpu_full_pressure_stall_time: ResourceMeasurement<u64>,
    /// Resident memory in the platform's native unit.
    pub resident_memory: ResourceMeasurement<u64>,
    /// Private memory in the platform's native unit.
    pub private_memory: ResourceMeasurement<u64>,
    /// Physical footprint in the platform's native unit.
    pub physical_footprint: ResourceMeasurement<u64>,
    /// Virtual memory in the platform's native unit.
    pub virtual_memory: ResourceMeasurement<u64>,
    /// Lifetime peak resident memory in the platform's native unit.
    pub peak_resident_memory: ResourceMeasurement<u64>,
    /// Effective environment memory limit in the platform's native unit.
    pub memory_limit: ResourceMeasurement<u64>,
    /// Memory charged to the owned environment in the platform's native unit.
    pub environment_accounted_memory: ResourceMeasurement<u64>,
    /// Cumulative memory pressure time during which some work was stalled.
    pub memory_some_pressure_stall_time: ResourceMeasurement<u64>,
    /// Cumulative memory pressure time during which all work was stalled.
    pub memory_full_pressure_stall_time: ResourceMeasurement<u64>,
    /// Cumulative out-of-memory event count.
    pub out_of_memory_event_count: ResourceMeasurement<u64>,
    /// Number of active processes in the measured scope.
    pub active_process_count: ResourceMeasurement<u64>,
    /// Number of active descendants below the measured root process.
    pub descendant_process_count: ResourceMeasurement<u64>,
    /// Number of threads across the measured scope.
    pub thread_count: ResourceMeasurement<u64>,
    /// Lifetime count of processes created inside the owned environment.
    pub lifetime_process_creation_count: ResourceMeasurement<u64>,
    /// Number of open file descriptors across the measured scope.
    pub open_file_descriptor_count: ResourceMeasurement<u64>,
    /// Number of Windows kernel handles across the measured scope.
    pub windows_handle_count: ResourceMeasurement<u64>,
    /// Typed cumulative resource-limit event counters.
    pub resource_limit_events: Vec<ResourceLimitEventCount>,
    /// Device-wide accelerator observations.
    pub accelerator_devices: Vec<AcceleratorDeviceMetrics>,
    /// Accelerator observations attributed only to Relay-owned processes.
    pub accelerator_processes: Vec<AcceleratorProcessMetrics>,
}
