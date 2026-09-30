// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structured system resource metric snapshot types.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// System or process boundary measured by a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceMeasurementScope {
    /// All system resources visible to the running environment.
    Global,
    /// The process containing the Relay runtime.
    ApplicationProcess,
    /// A root process and every descendant under that process.
    ProcessTree,
}

impl ResourceMeasurementScope {
    /// Return the stable wire name used by metric projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::ApplicationProcess => "application_process",
            Self::ProcessTree => "process_tree",
        }
    }
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
    /// Microseconds.
    Microseconds,
    /// Milliseconds.
    Milliseconds,
    /// Seconds.
    Seconds,
    /// Minutes.
    Minutes,
    /// Bytes.
    Bytes,
    /// Kilobytes.
    Kilobytes,
    /// Megabytes.
    Megabytes,
    /// Gigabytes.
    Gigabytes,
    /// Terabytes.
    Terabytes,
    /// Kibibytes.
    Kibibytes,
    /// Mebibytes.
    Mebibytes,
    /// Gibibytes.
    Gibibytes,
    /// Tebibytes.
    Tebibytes,
    /// Bytes Per Second.
    BytesPerSecond,
    /// Kibibytes Per Second.
    KibibytesPerSecond,
    /// Mebibytes Per Second.
    MebibytesPerSecond,
    /// Gibibytes Per Second.
    GibibytesPerSecond,
    /// Bits Per Second.
    BitsPerSecond,
    /// Megabits Per Second.
    MegabitsPerSecond,
    /// Gigabits Per Second.
    GigabitsPerSecond,
    /// Logical processor equivalents, including fractional processors.
    LogicalProcessors,
    /// Millicores.
    Millicores,
    /// Percentage from zero through one hundred.
    Percentage,
    /// Fraction.
    Fraction,
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
    /// Disk operation count.
    Operations,
    /// Network packet count.
    Packets,
    /// Network interface error count.
    Errors,
}

impl ResourceMeasurementUnit {
    /// Return the stable wire name used by metric projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Microseconds => "microseconds",
            Self::Milliseconds => "milliseconds",
            Self::Seconds => "seconds",
            Self::Minutes => "minutes",
            Self::Bytes => "bytes",
            Self::Kilobytes => "kilobytes",
            Self::Megabytes => "megabytes",
            Self::Gigabytes => "gigabytes",
            Self::Terabytes => "terabytes",
            Self::Kibibytes => "kibibytes",
            Self::Mebibytes => "mebibytes",
            Self::Gibibytes => "gibibytes",
            Self::Tebibytes => "tebibytes",
            Self::BytesPerSecond => "bytes_per_second",
            Self::KibibytesPerSecond => "kibibytes_per_second",
            Self::MebibytesPerSecond => "mebibytes_per_second",
            Self::GibibytesPerSecond => "gibibytes_per_second",
            Self::BitsPerSecond => "bits_per_second",
            Self::MegabitsPerSecond => "megabits_per_second",
            Self::GigabitsPerSecond => "gigabits_per_second",
            Self::LogicalProcessors => "logical_processors",
            Self::Millicores => "millicores",
            Self::Percentage => "percentage",
            Self::Fraction => "fraction",
            Self::Processes => "processes",
            Self::Threads => "threads",
            Self::FileDescriptors => "file_descriptors",
            Self::Handles => "handles",
            Self::Events => "events",
            Self::Operations => "operations",
            Self::Packets => "packets",
            Self::Errors => "errors",
        }
    }
}

/// Exact integral values remain integers; converted fractional values are decimals.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResourceMetricValue {
    /// Integer.
    Integer(u64),
    /// Decimal.
    Decimal(f64),
}

impl From<u64> for ResourceMetricValue {
    fn from(value: u64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for ResourceMetricValue {
    fn from(value: f64) -> Self {
        Self::Decimal(value)
    }
}

impl ResourceMetricValue {
    fn compare_to_u64(&self, other: u64) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match self {
            Self::Integer(value) => Some(value.cmp(&other)),
            Self::Decimal(value) if value.is_nan() => None,
            Self::Decimal(value) if *value < 0.0 => Some(Ordering::Less),
            // `u64::MAX as f64` rounds to 2^64, which is greater than every u64.
            Self::Decimal(value) if *value >= u64::MAX as f64 => Some(Ordering::Greater),
            Self::Decimal(value) => {
                let whole = value.trunc() as u64;
                match whole.cmp(&other) {
                    Ordering::Equal if value.fract() > 0.0 => Some(Ordering::Greater),
                    ordering => Some(ordering),
                }
            }
        }
    }
}

impl PartialEq for ResourceMetricValue {
    fn eq(&self, other: &Self) -> bool {
        use std::cmp::Ordering;

        match (self, other) {
            (Self::Integer(left), Self::Integer(right)) => left == right,
            (Self::Decimal(left), Self::Decimal(right)) => left == right,
            (Self::Decimal(_), Self::Integer(right)) => {
                self.compare_to_u64(*right) == Some(Ordering::Equal)
            }
            (Self::Integer(left), Self::Decimal(_)) => {
                other.compare_to_u64(*left) == Some(Ordering::Equal)
            }
        }
    }
}

impl PartialEq<u64> for ResourceMetricValue {
    fn eq(&self, other: &u64) -> bool {
        self.compare_to_u64(*other) == Some(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd<u64> for ResourceMetricValue {
    fn partial_cmp(&self, other: &u64) -> Option<std::cmp::Ordering> {
        self.compare_to_u64(*other)
    }
}

/// One available resource measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceMeasurement<T> {
    /// Acquired value. The containing category field is `None` when unavailable.
    pub value: T,
    /// Unit for the acquired value.
    pub unit: ResourceMeasurementUnit,
}

impl<T> ResourceMeasurement<T> {
    /// Construct an available measurement.
    pub fn new(value: impl Into<T>, unit: ResourceMeasurementUnit) -> Self {
        Self {
            value: value.into(),
            unit,
        }
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
    pub count: Option<ResourceMeasurement<ResourceMetricValue>>,
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
    /// Device identifier supplied by the collection backend.
    pub device_identifier: String,
    /// API-local device index, when supplied by the acquisition API.
    pub device_index: Option<u32>,
    /// Device memory currently in use.
    pub memory_used: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Device compute-engine utilization.
    pub compute_utilization: Option<ResourceMeasurement<f64>>,
}

/// Accelerator measurements attributed to one process in the selected scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorProcessMetrics {
    /// Device vendor.
    pub vendor: AcceleratorVendor,
    /// Device identifier supplied by the collection backend.
    pub device_identifier: String,
    /// API-local device index, when supplied by the acquisition API.
    pub device_index: Option<u32>,
    /// Process identifier.
    pub process_id: u32,
    /// Device memory attributed to this process.
    pub memory_used: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Compute-engine utilization attributed to this process.
    pub compute_utilization: Option<ResourceMeasurement<f64>>,
}

/// Filesystem capacity for one configured path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilesystemCapacityMetrics {
    /// Configured path identifying the filesystem.
    pub path: String,
    /// Total filesystem capacity in the configured unit.
    pub total_capacity: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Capacity available to the calling user, including any applicable quota.
    pub available_capacity: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Total free capacity, including space reserved from ordinary callers.
    pub free_capacity: Option<ResourceMeasurement<ResourceMetricValue>>,
}

/// Coverage of process queries used for process-summed measurements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessSamplingMetadata {
    /// Process IDs found when the collection began.
    pub visible_processes: u64,
    /// Process IDs for which the requested base process query succeeded.
    pub sampled_processes: u64,
    /// Process counts for readable counters or usable disk-rate intervals, keyed by measurement path.
    /// Zero means no process supplied a usable value; absent keys were not attempted.
    pub field_sampled_processes: BTreeMap<String, u64>,
}

/// CPU measurements and CPU limit events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CpuMetrics {
    /// Cumulative user-mode CPU time, in milliseconds by default.
    pub user_time: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Cumulative system-mode CPU time, in milliseconds by default.
    pub system_time: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Cumulative user plus system CPU time, in milliseconds by default.
    pub total_time: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// CPU capacity consumed, in logical processors by default.
    pub consumption_rate: Option<ResourceMeasurement<f64>>,
    /// Cumulative time during which the owned environment was CPU-throttled.
    pub throttled_time: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Effective CPU capacity, in logical processors by default.
    pub effective_limit: Option<ResourceMeasurement<f64>>,
    /// Cumulative CPU pressure time during which some work was stalled.
    pub some_pressure_stall_time: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Cumulative CPU pressure time during which all work was stalled.
    pub full_pressure_stall_time: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// CPU resource-limit event counters.
    pub limit_events: Vec<ResourceLimitEventCount>,
}

/// Memory measurements and memory limit events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// System memory currently in use, in KiB by default, in every scope.
    pub system_used: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Total system memory, in KiB by default, in every scope.
    pub system_total: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// System memory available for use, in KiB by default, in every scope.
    pub system_available: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Resident memory, in KiB by default.
    pub resident: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Private memory, in KiB by default.
    pub private: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Physical footprint, in KiB by default.
    pub physical_footprint: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Virtual memory, in KiB by default.
    pub virtual_memory: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Lifetime peak resident memory, in KiB by default.
    pub peak_resident: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Effective environment memory limit, in KiB by default.
    pub limit: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Memory charged to the owned environment, in KiB by default.
    pub environment_accounted: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Cumulative memory pressure time during which some work was stalled.
    pub some_pressure_stall_time: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Cumulative memory pressure time during which all work was stalled.
    pub full_pressure_stall_time: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Cumulative out-of-memory event count.
    pub out_of_memory_event_count: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Memory resource-limit event counters.
    pub limit_events: Vec<ResourceLimitEventCount>,
}

/// Process counts, process limits, handles, and descriptors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessMetrics {
    /// Number of active processes in the measured scope.
    pub active_count: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Number of active descendants below the measured root process.
    pub descendant_count: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Number of threads across the measured scope.
    pub thread_count: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Lifetime count of processes created inside the owned environment.
    pub lifetime_creation_count: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Number of open file descriptors across the measured scope.
    pub open_file_descriptor_count: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Number of Windows kernel handles across the measured scope.
    pub windows_handle_count: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Process resource-limit event counters.
    pub limit_events: Vec<ResourceLimitEventCount>,
}

/// Disk I/O for the measured process scope and filesystem capacity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiskMetrics {
    /// Cumulative disk data read by processes in the selected scope.
    pub read_data: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Cumulative disk data written by processes in the selected scope.
    pub write_data: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Read transfer rate since the previous sample.
    pub read_throughput: Option<ResourceMeasurement<f64>>,
    /// Write transfer rate since the previous sample.
    pub write_throughput: Option<ResourceMeasurement<f64>>,
    /// Cumulative read operations in the measured process scope.
    pub read_operations: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Cumulative write operations in the measured process scope.
    pub write_operations: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Filesystem-capacity observations for configured paths.
    pub filesystems: Vec<FilesystemCapacityMetrics>,
}

/// Network counters for one interface or the selected system aggregate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkTrafficMetrics {
    /// Received data.
    pub received_data: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Transmitted data.
    pub transmitted_data: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Receive throughput.
    pub receive_throughput: Option<ResourceMeasurement<f64>>,
    /// Transmit throughput.
    pub transmit_throughput: Option<ResourceMeasurement<f64>>,
    /// Received packets.
    pub received_packets: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Transmitted packets.
    pub transmitted_packets: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Receive errors.
    pub receive_errors: Option<ResourceMeasurement<ResourceMetricValue>>,
    /// Transmit errors.
    pub transmit_errors: Option<ResourceMeasurement<ResourceMetricValue>>,
}

/// Network traffic for one visible interface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkInterfaceMetrics {
    /// Operating-system interface name.
    pub name: String,
    /// Counters and rates for this interface.
    pub traffic: NetworkTrafficMetrics,
}

/// System-wide network traffic, regardless of the process measurement scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Global measurement scope, independent of the snapshot's process scope.
    pub measurement_scope: ResourceMeasurementScope,
    /// Aggregate of the selected interfaces.
    pub system: NetworkTrafficMetrics,
    /// One record per selected visible interface.
    pub interfaces: Vec<NetworkInterfaceMetrics>,
}

/// Accelerator (GPU) device-wide and process-attributed measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// Device-wide accelerator observations.
    pub device_metrics: Option<Vec<AcceleratorDeviceMetrics>>,
    /// Accelerator observations attributed to processes in the selected scope.
    pub process_metrics: Option<Vec<AcceleratorProcessMetrics>>,
}

/// One point-in-time observation of the selected system and process resources.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceMetricsSnapshot {
    /// UTC time at which collection completed.
    pub timestamp: DateTime<Utc>,
    /// Operating system that produced this snapshot.
    pub operating_system: ResourceOperatingSystem,
    /// Boundary for process-attributed values; system and device values keep their own scope.
    pub measurement_scope: ResourceMeasurementScope,
    /// Coverage for process-summed values, or `None` when no query was requested or listing failed.
    pub process_sampling: Option<ProcessSamplingMetadata>,
    /// CPU category, or `None` when CPU collection is disabled.
    pub cpu: Option<CpuMetrics>,
    /// Memory category, or `None` when memory collection is disabled.
    pub memory: Option<MemoryMetrics>,
    /// Process category, or `None` when process collection is disabled.
    pub process: Option<ProcessMetrics>,
    /// Disk category, or `None` when disk collection is disabled.
    pub disk: Option<DiskMetrics>,
    /// GPU category, or `None` when GPU collection is disabled.
    pub gpu: Option<GpuMetrics>,
    /// System-wide network category, or `None` when disabled.
    pub network: Option<NetworkMetrics>,
}
