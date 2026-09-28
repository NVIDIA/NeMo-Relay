// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io;

use nemo_relay_types::api::resource_metrics::{ResourceMeasurementUnit, ResourceOperatingSystem};

use super::{AcceleratorSample, CollectionTarget, EnvironmentSample, ProcessSample};

pub(super) const OPERATING_SYSTEM: ResourceOperatingSystem = ResourceOperatingSystem::Unsupported;
pub(super) const CPU_TIME_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Nanoseconds;
pub(super) const RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Bytes;
pub(super) const PRIVATE_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Bytes;
pub(super) const PHYSICAL_FOOTPRINT_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Bytes;
pub(super) const VIRTUAL_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Bytes;
pub(super) const PEAK_RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit =
    ResourceMeasurementUnit::Bytes;

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

pub(super) fn process_sample(_process_id: u32) -> io::Result<ProcessSample> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "resource metrics are unsupported on this platform",
    ))
}

pub(super) fn environment_sample(
    _target: &CollectionTarget,
    _process_ids: &[u32],
) -> io::Result<EnvironmentSample> {
    Ok(EnvironmentSample::default())
}

pub(super) fn accelerator_sample(_process_ids: &[u32]) -> AcceleratorSample {
    AcceleratorSample::default()
}
