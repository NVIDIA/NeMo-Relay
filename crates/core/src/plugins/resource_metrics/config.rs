// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{FlowError, Result};

/// Measurement boundary selected for resource metric collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResourceMetricsMeasurementScope {
    /// Choose the default for the runtime: global in the CLI, or the current process tree in APIs.
    #[default]
    RuntimeDefault,
    /// Collect resources for the full system visible to the running environment.
    Global,
    /// Collect resources for the process containing the Relay runtime.
    ApplicationProcess,
    /// Collect resources for the selected root process and all descendants.
    ProcessTree,
}

impl ResourceMetricsMeasurementScope {
    /// Stable value used by the configuration editor and serialized config.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RuntimeDefault => "runtime_default",
            Self::Global => "global",
            Self::ApplicationProcess => "application_process",
            Self::ProcessTree => "process_tree",
        }
    }
}

/// Periodic collection controlled by the `resource_metrics` plugin component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMetricsPollingConfig {
    /// Whether the plugin should collect and emit a mark periodically.
    pub enabled: bool,
    /// Delay between polling ticks, in milliseconds.
    pub interval_millis: u64,
}

impl Default for ResourceMetricsPollingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_millis: 5_000,
        }
    }
}

/// CPU observations, limits, and pressure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMetricsCpuConfig {
    /// Whether CPU data is included in snapshots.
    pub enabled: bool,
}

impl Default for ResourceMetricsCpuConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Memory observations, limits, pressure, and OOM events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMetricsMemoryConfig {
    /// Whether memory data is included in snapshots.
    pub enabled: bool,
}

impl Default for ResourceMetricsMemoryConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Process, thread, handle, descriptor, and process-limit observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMetricsProcessConfig {
    /// Whether process data is included in snapshots.
    pub enabled: bool,
}

impl Default for ResourceMetricsProcessConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Disk I/O for the selected process scope and filesystem capacity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMetricsDiskConfig {
    /// Whether disk data is included in snapshots.
    pub enabled: bool,
    /// Collect lifetime read and write counters for processes visible in the selected scope.
    pub process_io: bool,
    /// Filesystem paths for which capacity should be reported. Empty means none.
    pub filesystem_paths: Vec<PathBuf>,
}

impl Default for ResourceMetricsDiskConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            process_io: true,
            filesystem_paths: Vec::new(),
        }
    }
}

/// Accelerator observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMetricsGpuConfig {
    /// Whether accelerator data is included in snapshots.
    pub enabled: bool,
    /// Device identifiers to include. Empty means all supported devices.
    pub devices: Vec<String>,
    /// Collect device-wide measurements.
    pub device_metrics: bool,
    /// Collect measurements attributed to processes in the selected scope.
    pub process_metrics: bool,
}

impl Default for ResourceMetricsGpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            devices: Vec::new(),
            device_metrics: true,
            process_metrics: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
/// Time units available for resource measurements.
pub enum TimeUnit {
    /// Microseconds.
    Microseconds,
    /// Milliseconds.
    #[default]
    Milliseconds,
    /// Seconds.
    Seconds,
    /// Minutes.
    Minutes,
}
impl From<TimeUnit> for nemo_relay_types::api::resource_metrics::ResourceMeasurementUnit {
    fn from(unit: TimeUnit) -> Self {
        match unit {
            TimeUnit::Microseconds => Self::Microseconds,
            TimeUnit::Milliseconds => Self::Milliseconds,
            TimeUnit::Seconds => Self::Seconds,
            TimeUnit::Minutes => Self::Minutes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
/// Storage units for memory and GPU readings.
pub enum MemoryUnit {
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
    #[default]
    Kibibytes,
    /// Mebibytes.
    Mebibytes,
    /// Gibibytes.
    Gibibytes,
    /// Tebibytes.
    Tebibytes,
}
impl From<MemoryUnit> for nemo_relay_types::api::resource_metrics::ResourceMeasurementUnit {
    fn from(unit: MemoryUnit) -> Self {
        match unit {
            MemoryUnit::Bytes => Self::Bytes,
            MemoryUnit::Kilobytes => Self::Kilobytes,
            MemoryUnit::Megabytes => Self::Megabytes,
            MemoryUnit::Gigabytes => Self::Gigabytes,
            MemoryUnit::Terabytes => Self::Terabytes,
            MemoryUnit::Kibibytes => Self::Kibibytes,
            MemoryUnit::Mebibytes => Self::Mebibytes,
            MemoryUnit::Gibibytes => Self::Gibibytes,
            MemoryUnit::Tebibytes => Self::Tebibytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
/// Storage units for disk and network readings.
pub enum DataUnit {
    /// Bytes.
    #[default]
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
}
impl From<DataUnit> for nemo_relay_types::api::resource_metrics::ResourceMeasurementUnit {
    fn from(unit: DataUnit) -> Self {
        match unit {
            DataUnit::Bytes => Self::Bytes,
            DataUnit::Kilobytes => Self::Kilobytes,
            DataUnit::Megabytes => Self::Megabytes,
            DataUnit::Gigabytes => Self::Gigabytes,
            DataUnit::Terabytes => Self::Terabytes,
            DataUnit::Kibibytes => Self::Kibibytes,
            DataUnit::Mebibytes => Self::Mebibytes,
            DataUnit::Gibibytes => Self::Gibibytes,
            DataUnit::Tebibytes => Self::Tebibytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
/// Transfer-rate units for disk and network readings.
pub enum ThroughputUnit {
    /// Bytes per second.
    #[default]
    BytesPerSecond,
    /// Kibibytes per second.
    KibibytesPerSecond,
    /// Mebibytes per second.
    MebibytesPerSecond,
    /// Gibibytes per second.
    GibibytesPerSecond,
    /// Bits per second.
    BitsPerSecond,
    /// Megabits per second.
    MegabitsPerSecond,
    /// Gigabits per second.
    GigabitsPerSecond,
}
impl From<ThroughputUnit> for nemo_relay_types::api::resource_metrics::ResourceMeasurementUnit {
    fn from(unit: ThroughputUnit) -> Self {
        match unit {
            ThroughputUnit::BytesPerSecond => Self::BytesPerSecond,
            ThroughputUnit::KibibytesPerSecond => Self::KibibytesPerSecond,
            ThroughputUnit::MebibytesPerSecond => Self::MebibytesPerSecond,
            ThroughputUnit::GibibytesPerSecond => Self::GibibytesPerSecond,
            ThroughputUnit::BitsPerSecond => Self::BitsPerSecond,
            ThroughputUnit::MegabitsPerSecond => Self::MegabitsPerSecond,
            ThroughputUnit::GigabitsPerSecond => Self::GigabitsPerSecond,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
/// Units for CPU use and capacity.
pub enum CpuUnit {
    /// Logical processors.
    #[default]
    LogicalProcessors,
    /// Millicores.
    Millicores,
}
impl From<CpuUnit> for nemo_relay_types::api::resource_metrics::ResourceMeasurementUnit {
    fn from(unit: CpuUnit) -> Self {
        match unit {
            CpuUnit::LogicalProcessors => Self::LogicalProcessors,
            CpuUnit::Millicores => Self::Millicores,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
/// Units for GPU utilization.
pub enum UtilizationUnit {
    /// Percentage.
    #[default]
    Percentage,
    /// Fraction.
    Fraction,
}
impl From<UtilizationUnit> for nemo_relay_types::api::resource_metrics::ResourceMeasurementUnit {
    fn from(unit: UtilizationUnit) -> Self {
        match unit {
            UtilizationUnit::Percentage => Self::Percentage,
            UtilizationUnit::Fraction => Self::Fraction,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
/// CPU measurement unit choices.
pub struct ResourceMetricsCpuUnits {
    /// User time.
    pub user_time: TimeUnit,
    /// System time.
    pub system_time: TimeUnit,
    /// Total time.
    pub total_time: TimeUnit,
    /// Throttled time.
    pub throttled_time: TimeUnit,
    /// Some pressure stall time.
    pub some_pressure_stall_time: TimeUnit,
    /// Full pressure stall time.
    pub full_pressure_stall_time: TimeUnit,
    /// Consumption rate.
    pub consumption_rate: CpuUnit,
    /// Effective limit.
    pub effective_limit: CpuUnit,
}

crate::editor_config! {
    impl ResourceMetricsCpuUnits {
        user_time => { label: "user_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        system_time => { label: "system_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        total_time => { label: "total_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        throttled_time => { label: "throttled_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        some_pressure_stall_time => { label: "some_pressure_stall_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        full_pressure_stall_time => { label: "full_pressure_stall_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        consumption_rate => { label: "consumption_rate", kind: Enum, values: ["logical_processors", "millicores"] },
        effective_limit => { label: "effective_limit", kind: Enum, values: ["logical_processors", "millicores"] },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
/// Memory measurement unit choices.
pub struct ResourceMetricsMemoryUnits {
    /// System used.
    pub system_used: MemoryUnit,
    /// System total.
    pub system_total: MemoryUnit,
    /// System available.
    pub system_available: MemoryUnit,
    /// Resident.
    pub resident: MemoryUnit,
    /// Private.
    pub private: MemoryUnit,
    /// Physical footprint.
    pub physical_footprint: MemoryUnit,
    /// Virtual memory.
    pub virtual_memory: MemoryUnit,
    /// Peak resident.
    pub peak_resident: MemoryUnit,
    /// Limit.
    pub limit: MemoryUnit,
    /// Environment accounted.
    pub environment_accounted: MemoryUnit,
    /// Some pressure stall time.
    pub some_pressure_stall_time: TimeUnit,
    /// Full pressure stall time.
    pub full_pressure_stall_time: TimeUnit,
}

crate::editor_config! {
    impl ResourceMetricsMemoryUnits {
        system_used => { label: "system_used", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        system_total => { label: "system_total", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        system_available => { label: "system_available", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        resident => { label: "resident", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        private => { label: "private", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        physical_footprint => { label: "physical_footprint", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        virtual_memory => { label: "virtual_memory", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        peak_resident => { label: "peak_resident", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        limit => { label: "limit", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        environment_accounted => { label: "environment_accounted", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        some_pressure_stall_time => { label: "some_pressure_stall_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        full_pressure_stall_time => { label: "full_pressure_stall_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
/// Disk measurement unit choices.
pub struct ResourceMetricsDiskUnits {
    /// Read data.
    pub read_data: DataUnit,
    /// Write data.
    pub write_data: DataUnit,
    /// Filesystem total capacity.
    pub filesystem_total_capacity: DataUnit,
    /// Filesystem available capacity.
    pub filesystem_available_capacity: DataUnit,
    /// Filesystem free capacity.
    pub filesystem_free_capacity: DataUnit,
    /// Read throughput.
    pub read_throughput: ThroughputUnit,
    /// Write throughput.
    pub write_throughput: ThroughputUnit,
}

crate::editor_config! {
    impl ResourceMetricsDiskUnits {
        read_data => { label: "read_data", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        write_data => { label: "write_data", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        filesystem_total_capacity => { label: "filesystem_total_capacity", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        filesystem_available_capacity => { label: "filesystem_available_capacity", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        filesystem_free_capacity => { label: "filesystem_free_capacity", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        read_throughput => { label: "read_throughput", kind: Enum, values: ["bytes_per_second", "kibibytes_per_second", "mebibytes_per_second", "gibibytes_per_second", "bits_per_second", "megabits_per_second", "gigabits_per_second"] },
        write_throughput => { label: "write_throughput", kind: Enum, values: ["bytes_per_second", "kibibytes_per_second", "mebibytes_per_second", "gibibytes_per_second", "bits_per_second", "megabits_per_second", "gigabits_per_second"] },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
/// GPU measurement unit choices.
pub struct ResourceMetricsGpuUnits {
    /// Device memory used.
    pub device_memory_used: MemoryUnit,
    /// Process memory used.
    pub process_memory_used: MemoryUnit,
    /// Device compute utilization.
    pub device_compute_utilization: UtilizationUnit,
    /// Process compute utilization.
    pub process_compute_utilization: UtilizationUnit,
}

crate::editor_config! {
    impl ResourceMetricsGpuUnits {
        device_memory_used => { label: "device_memory_used", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        process_memory_used => { label: "process_memory_used", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        device_compute_utilization => { label: "device_compute_utilization", kind: Enum, values: ["percentage", "fraction"] },
        process_compute_utilization => { label: "process_compute_utilization", kind: Enum, values: ["percentage", "fraction"] },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
/// Unit choices for one group of network traffic.
pub struct ResourceMetricsNetworkTrafficUnits {
    /// Received data.
    pub received_data: DataUnit,
    /// Transmitted data.
    pub transmitted_data: DataUnit,
    /// Receive throughput.
    pub receive_throughput: ThroughputUnit,
    /// Transmit throughput.
    pub transmit_throughput: ThroughputUnit,
}

crate::editor_config! {
    impl ResourceMetricsNetworkTrafficUnits {
        received_data => { label: "received_data", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        transmitted_data => { label: "transmitted_data", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        receive_throughput => { label: "receive_throughput", kind: Enum, values: ["bytes_per_second", "kibibytes_per_second", "mebibytes_per_second", "gibibytes_per_second", "bits_per_second", "megabits_per_second", "gigabits_per_second"] },
        transmit_throughput => { label: "transmit_throughput", kind: Enum, values: ["bytes_per_second", "kibibytes_per_second", "mebibytes_per_second", "gibibytes_per_second", "bits_per_second", "megabits_per_second", "gigabits_per_second"] },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
/// Unit choices for system and interface traffic.
pub struct ResourceMetricsNetworkUnits {
    /// System.
    pub system: ResourceMetricsNetworkTrafficUnits,
    /// Interface.
    pub interface: ResourceMetricsNetworkTrafficUnits,
}
crate::editor_config! {
    impl ResourceMetricsNetworkUnits {
        system => { label: "System", kind: Section, nested: ResourceMetricsNetworkTrafficUnits, default: ResourceMetricsNetworkTrafficUnits },
        interface => { label: "Interface", kind: Section, nested: ResourceMetricsNetworkTrafficUnits, default: ResourceMetricsNetworkTrafficUnits },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
/// Output unit choices for resource metrics.
pub struct ResourceMetricsUnits {
    /// CPU measurements.
    pub cpu: ResourceMetricsCpuUnits,
    /// Memory measurements.
    pub memory: ResourceMetricsMemoryUnits,
    /// Disk measurements.
    pub disk: ResourceMetricsDiskUnits,
    /// GPU measurements.
    pub gpu: ResourceMetricsGpuUnits,
    /// Network measurements.
    pub network: ResourceMetricsNetworkUnits,
}
crate::editor_config! {
    impl ResourceMetricsUnits {
        cpu => { label: "CPU", kind: Section, nested: ResourceMetricsCpuUnits, default: ResourceMetricsCpuUnits },
        memory => { label: "Memory", kind: Section, nested: ResourceMetricsMemoryUnits, default: ResourceMetricsMemoryUnits },
        disk => { label: "Disk", kind: Section, nested: ResourceMetricsDiskUnits, default: ResourceMetricsDiskUnits },
        gpu => { label: "GPU", kind: Section, nested: ResourceMetricsGpuUnits, default: ResourceMetricsGpuUnits },
        network => { label: "Network", kind: Section, nested: ResourceMetricsNetworkUnits, default: ResourceMetricsNetworkUnits },
    }
}

/// System interface counters. Empty `interfaces` selects every visible interface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMetricsNetworkConfig {
    /// Include system network measurements.
    pub enabled: bool,
    /// Interface names to include, or an empty list for all visible interfaces.
    pub interfaces: Vec<String>,
}
impl Default for ResourceMetricsNetworkConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interfaces: Vec::new(),
        }
    }
}
crate::editor_config! {
    impl ResourceMetricsNetworkConfig {
        enabled => { label: "enabled", kind: Boolean },
        interfaces => { label: "interfaces", kind: List, list: &crate::config_editor::STRING_LIST_ITEM },
    }
}

/// Configuration for the built-in `resource_metrics` plugin component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMetricsConfig {
    /// Resource boundary to collect. The runtime default is global in the CLI
    /// and the current process tree in embedded APIs.
    pub measurement_scope: ResourceMetricsMeasurementScope,
    /// Periodic polling behavior.
    pub polling: ResourceMetricsPollingConfig,
    /// CPU observation category.
    pub cpu: ResourceMetricsCpuConfig,
    /// Memory observation category.
    pub memory: ResourceMetricsMemoryConfig,
    /// Process observation category.
    pub process: ResourceMetricsProcessConfig,
    /// Disk observation category.
    pub disk: ResourceMetricsDiskConfig,
    /// GPU observation category.
    pub gpu: ResourceMetricsGpuConfig,
    /// System interface counter selection.
    pub network: ResourceMetricsNetworkConfig,
    /// Output unit choices for measurements with convertible dimensions.
    pub units: ResourceMetricsUnits,
}

crate::editor_config! {
    impl ResourceMetricsPollingConfig {
        enabled => { label: "enabled", kind: Boolean },
        interval_millis => { label: "interval_millis", kind: Integer },
    }
}

crate::editor_config! {
    impl ResourceMetricsCpuConfig {
        enabled => { label: "enabled", kind: Boolean },
    }
}

crate::editor_config! {
    impl ResourceMetricsMemoryConfig {
        enabled => { label: "enabled", kind: Boolean },
    }
}

crate::editor_config! {
    impl ResourceMetricsProcessConfig {
        enabled => { label: "enabled", kind: Boolean },
    }
}

crate::editor_config! {
    impl ResourceMetricsDiskConfig {
        enabled => { label: "enabled", kind: Boolean },
        process_io => { label: "process_io", kind: Boolean },
        filesystem_paths => {
            label: "filesystem_paths",
            kind: List,
            list: &crate::config_editor::STRING_LIST_ITEM,
        },
    }
}

crate::editor_config! {
    impl ResourceMetricsGpuConfig {
        enabled => { label: "enabled", kind: Boolean },
        devices => {
            label: "devices",
            kind: List,
            list: &crate::config_editor::STRING_LIST_ITEM,
        },
        device_metrics => { label: "device_metrics", kind: Boolean },
        process_metrics => { label: "process_metrics", kind: Boolean },
    }
}

crate::editor_config! {
    impl ResourceMetricsConfig {
        measurement_scope => {
            label: "measurement_scope",
            kind: Enum,
            values: ["runtime_default", "global", "application_process", "process_tree"],
        },
        polling => {
            label: "polling",
            kind: Section,
            nested: ResourceMetricsPollingConfig,
            default: ResourceMetricsPollingConfig,
        },
        cpu => {
            label: "CPU",
            kind: Section,
            nested: ResourceMetricsCpuConfig,
            default: ResourceMetricsCpuConfig,
        },
        memory => {
            label: "Memory",
            kind: Section,
            nested: ResourceMetricsMemoryConfig,
            default: ResourceMetricsMemoryConfig,
        },
        process => {
            label: "Process",
            kind: Section,
            nested: ResourceMetricsProcessConfig,
            default: ResourceMetricsProcessConfig,
        },
        disk => {
            label: "Disk",
            kind: Section,
            nested: ResourceMetricsDiskConfig,
            default: ResourceMetricsDiskConfig,
        },
        gpu => {
            label: "GPU",
            kind: Section,
            nested: ResourceMetricsGpuConfig,
            default: ResourceMetricsGpuConfig,
        },
        network => { label: "Network", kind: Section, nested: ResourceMetricsNetworkConfig, default: ResourceMetricsNetworkConfig },
        units => { label: "Units", kind: Section, nested: ResourceMetricsUnits, default: ResourceMetricsUnits },
    }
}

impl ResourceMetricsConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.polling.interval_millis == 0 {
            return Err(FlowError::InvalidArgument(
                "resource_metrics.polling.interval_millis must be greater than zero".into(),
            ));
        }
        for (index, path) in self.disk.filesystem_paths.iter().enumerate() {
            if path.as_os_str().is_empty() {
                return Err(FlowError::InvalidArgument(format!(
                    "resource_metrics.disk.filesystem_paths[{index}] must not be empty"
                )));
            }
            if !path.is_absolute() {
                return Err(FlowError::InvalidArgument(format!(
                    "resource_metrics.disk.filesystem_paths[{index}] must be absolute"
                )));
            }
        }
        if self
            .network
            .interfaces
            .iter()
            .any(|name| name.trim().is_empty())
        {
            return Err(FlowError::InvalidArgument(
                "resource_metrics.network.interfaces must not contain blank names".into(),
            ));
        }
        if self
            .gpu
            .devices
            .iter()
            .any(|device| device.trim().is_empty())
        {
            return Err(FlowError::InvalidArgument(
                "resource_metrics.gpu.devices must not contain empty identifiers".into(),
            ));
        }
        Ok(())
    }
}
