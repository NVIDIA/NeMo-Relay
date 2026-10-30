// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{Error as FlowError, Result};

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

pub use nemo_relay_types::api::resource_metrics::{
    BandwidthUnit, CapacityUnit, CpuUnit, DataUnit, DurationUnit, UtilizationUnit,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
/// CPU measurement unit choices.
pub struct ResourceMetricsCpuUnits {
    /// User time.
    pub user_time: DurationUnit,
    /// System time.
    pub system_time: DurationUnit,
    /// Total time.
    pub total_time: DurationUnit,
    /// Throttled time.
    pub throttled_time: DurationUnit,
    /// Some pressure stall time.
    pub some_pressure_stall_time: DurationUnit,
    /// Full pressure stall time.
    pub full_pressure_stall_time: DurationUnit,
    /// Consumption rate.
    pub consumption_rate: CpuUnit,
    /// Effective limit.
    pub effective_limit: CpuUnit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
/// Memory measurement unit choices.
pub struct ResourceMetricsMemoryUnits {
    /// System used.
    pub system_used: CapacityUnit,
    /// System total.
    pub system_total: CapacityUnit,
    /// System available.
    pub system_available: CapacityUnit,
    /// Resident.
    pub resident: CapacityUnit,
    /// Private.
    pub private: CapacityUnit,
    /// Physical footprint.
    pub physical_footprint: CapacityUnit,
    /// Virtual memory.
    pub virtual_memory: CapacityUnit,
    /// Peak resident.
    pub peak_resident: CapacityUnit,
    /// Limit.
    pub limit: CapacityUnit,
    /// Environment accounted.
    pub environment_accounted: CapacityUnit,
    /// Some pressure stall time.
    pub some_pressure_stall_time: DurationUnit,
    /// Full pressure stall time.
    pub full_pressure_stall_time: DurationUnit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
/// Disk measurement unit choices.
pub struct ResourceMetricsDiskUnits {
    /// Read data.
    pub read_data: DataUnit,
    /// Write data.
    pub write_data: DataUnit,
    /// Filesystem total capacity.
    pub filesystem_total_capacity: CapacityUnit,
    /// Filesystem available capacity.
    pub filesystem_available_capacity: CapacityUnit,
    /// Filesystem free capacity.
    pub filesystem_free_capacity: CapacityUnit,
    /// Read throughput.
    pub read_throughput: BandwidthUnit,
    /// Write throughput.
    pub write_throughput: BandwidthUnit,
}

impl Default for ResourceMetricsDiskUnits {
    fn default() -> Self {
        Self {
            read_data: DataUnit::Bytes,
            write_data: DataUnit::Bytes,
            filesystem_total_capacity: CapacityUnit::Bytes,
            filesystem_available_capacity: CapacityUnit::Bytes,
            filesystem_free_capacity: CapacityUnit::Bytes,
            read_throughput: BandwidthUnit::BytesPerSecond,
            write_throughput: BandwidthUnit::BytesPerSecond,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
/// GPU measurement unit choices.
pub struct ResourceMetricsGpuUnits {
    /// Device memory used.
    pub device_memory_used: CapacityUnit,
    /// Process memory used.
    pub process_memory_used: CapacityUnit,
    /// Device compute utilization.
    pub device_compute_utilization: UtilizationUnit,
    /// Process compute utilization.
    pub process_compute_utilization: UtilizationUnit,
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
    pub receive_throughput: BandwidthUnit,
    /// Transmit throughput.
    pub transmit_throughput: BandwidthUnit,
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

/// Read-only latest memory projection. Does not enable polling or collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMemoryInspectionConfig {
    /// Retain one fixed memory sample from existing collections.
    pub enabled: bool,
    /// Mark a retained sample stale after this completion age, in milliseconds.
    pub stale_after_millis: u64,
}
impl Default for ResourceMemoryInspectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            stale_after_millis: 30_000,
        }
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
    /// Fixed latest-memory projection policy, independent of polling.
    pub memory_inspection: ResourceMemoryInspectionConfig,
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

impl ResourceMetricsConfig {
    #[doc(hidden)]
    pub fn validate(&self) -> Result<()> {
        if self.memory_inspection.stale_after_millis == 0 {
            return Err(FlowError::InvalidArgument(
                "resource_metrics.memory_inspection.stale_after_millis must be greater than zero"
                    .into(),
            ));
        }
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
