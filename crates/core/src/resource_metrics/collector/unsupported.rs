// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io;

use nemo_relay_types::api::resource_metrics::{ResourceMeasurementUnit, ResourceOperatingSystem};

use super::{
    AcceleratorSample, CollectionTarget, EnvironmentSample, ProcessSample, ProcessSampleConfig,
};

pub(super) const OPERATING_SYSTEM: ResourceOperatingSystem = ResourceOperatingSystem::Unsupported;
pub(super) const CPU_TIME_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Milliseconds;
pub(super) const RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Kibibytes;
pub(super) const PRIVATE_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Kibibytes;
pub(super) const PHYSICAL_FOOTPRINT_UNIT: ResourceMeasurementUnit =
    ResourceMeasurementUnit::Kibibytes;
pub(super) const VIRTUAL_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Kibibytes;
pub(super) const PEAK_RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit =
    ResourceMeasurementUnit::Kibibytes;

pub(super) fn process_identity(_process_id: u32) -> io::Result<u64> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "resource metrics are unsupported on this platform",
    ))
}

pub(super) fn parent_process_id(_process_id: u32) -> io::Result<Option<u32>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "resource metrics are unsupported on this platform",
    ))
}

pub(super) fn process_tree_ids(_root_process_id: u32) -> io::Result<Vec<u32>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "resource metrics are unsupported on this platform",
    ))
}

pub(super) fn all_process_ids() -> io::Result<Vec<u32>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "resource metrics are unsupported on this platform",
    ))
}

pub(super) fn process_sample(
    _process_id: u32,
    _config: ProcessSampleConfig,
) -> io::Result<ProcessSample> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "resource metrics are unsupported on this platform",
    ))
}

pub(super) fn environment_sample(
    _target: &CollectionTarget,
    _process_ids: &[u32],
    _config: &crate::plugins::resource_metrics::config::ResourceMetricsConfig,
) -> io::Result<EnvironmentSample> {
    Ok(EnvironmentSample::default())
}

pub(super) fn global_environment_sample(
    _config: &crate::plugins::resource_metrics::config::ResourceMetricsConfig,
) -> io::Result<EnvironmentSample> {
    Ok(EnvironmentSample::default())
}

pub(super) fn accelerator_sample(
    _process_ids: &[u32],
    _config: &crate::plugins::resource_metrics::config::ResourceMetricsGpuConfig,
    _sampling_state: &mut super::SamplingState,
) -> AcceleratorSample {
    AcceleratorSample::default()
}

pub(super) fn filesystem_capacity(_path: &std::path::Path) -> io::Result<(u64, u64, u64)> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "filesystem metrics are unsupported on this platform",
    ))
}
