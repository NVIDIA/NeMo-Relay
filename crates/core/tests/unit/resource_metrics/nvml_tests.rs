// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

unsafe extern "C" fn device_count(out: *mut u32) -> i32 {
    unsafe { out.write(1) };
    NVML_SUCCESS
}

unsafe extern "C" fn device_handle(_: u32, out: *mut NvmlDevice) -> i32 {
    unsafe { out.write(std::ptr::dangling_mut()) };
    NVML_SUCCESS
}

unsafe extern "C" fn device_uuid(_: NvmlDevice, out: *mut c_char, size: u32) -> i32 {
    let uuid = b"GPU-fixture\0";
    assert!(size as usize >= uuid.len());
    unsafe { std::ptr::copy_nonoverlapping(uuid.as_ptr().cast(), out, uuid.len()) };
    NVML_SUCCESS
}

unsafe extern "C" fn device_memory(_: NvmlDevice, out: *mut NvmlMemory) -> i32 {
    unsafe { (*out).used = 4_194_304 };
    NVML_SUCCESS
}

unsafe extern "C" fn device_utilization(_: NvmlDevice, out: *mut NvmlUtilization) -> i32 {
    unsafe { (*out).gpu = 25 };
    NVML_SUCCESS
}

unsafe extern "C" fn processes(_: NvmlDevice, count: *mut u32, out: *mut NvmlProcessInfo) -> i32 {
    unsafe { count.write(2) };
    if out.is_null() {
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }
    unsafe {
        out.write(NvmlProcessInfo {
            process_id: 42,
            used_gpu_memory: 4096,
            ..Default::default()
        });
        out.add(1).write(NvmlProcessInfo {
            process_id: 99,
            used_gpu_memory: NVML_VALUE_NOT_AVAILABLE,
            ..Default::default()
        });
    }
    NVML_SUCCESS
}

unsafe extern "C" fn process_utilization(
    _: NvmlDevice,
    out: *mut NvmlProcessUtilization,
    count: *mut u32,
    last_seen: u64,
) -> i32 {
    unsafe { count.write(1) };
    if out.is_null() {
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }
    unsafe {
        out.write(NvmlProcessUtilization {
            process_id: 42,
            timestamp: last_seen + 1,
            streaming_multiprocessor_utilization: 30,
            ..Default::default()
        })
    };
    NVML_SUCCESS
}

fn session() -> NvmlSession {
    // Keep an ordinary system library alive; all NVML calls use the ABI fixtures above.
    #[cfg(target_os = "linux")]
    let library = unsafe { Library::new("libc.so.6") }.unwrap();
    #[cfg(windows)]
    let library = unsafe { Library::new("kernel32.dll") }.unwrap();
    NvmlSession {
        _library: library,
        device_count,
        device_by_index: device_handle,
        device_uuid,
        device_memory,
        device_utilization,
        compute_processes: Some(processes),
        graphics_processes: Some(processes),
        process_utilization: Some(process_utilization),
    }
}

#[test]
fn nvml_snapshot_normalizes_memory_filters_processes_and_uses_a_fresh_utilization_baseline() {
    let session = session();
    let config = ResourceMetricsGpuConfig::default();
    let mut state = SamplingState::default();
    let first = collect_nvml_with_session(&session, &[42], &config, &mut state).unwrap();
    let devices = first.devices.unwrap();
    assert_eq!(devices[0].device_identifier, "GPU-fixture");
    assert_eq!(devices[0].memory_used.as_ref().unwrap().value, 4096_u64);
    assert_eq!(
        devices[0].memory_used.as_ref().unwrap().unit,
        ResourceMeasurementUnit::Kibibytes
    );
    assert_eq!(devices[0].compute_utilization.as_ref().unwrap().value, 25.0);
    let processes = first.processes.unwrap();
    assert_eq!(processes.len(), 1);
    assert_eq!(processes[0].process_id, 42);
    assert_eq!(processes[0].memory_used.as_ref().unwrap().value, 4_u64);
    assert!(processes[0].compute_utilization.is_none());
    let second = collect_nvml_with_session(&session, &[42], &config, &mut state).unwrap();
    assert_eq!(
        second.processes.unwrap()[0]
            .compute_utilization
            .as_ref()
            .unwrap()
            .value,
        30.0
    );
}

#[test]
fn nvml_selection_and_disabled_groups_do_not_publish_measurements() {
    let session = session();
    let mut state = SamplingState::default();
    for (device_metrics, process_metrics) in [(true, false), (false, true), (false, false)] {
        let config = ResourceMetricsGpuConfig {
            device_metrics,
            process_metrics,
            ..Default::default()
        };
        let sample = collect_nvml_with_session(&session, &[42], &config, &mut state).unwrap();
        assert_eq!(sample.devices.is_some(), device_metrics);
        assert_eq!(sample.processes.is_some(), process_metrics);
    }
    let config = ResourceMetricsGpuConfig {
        devices: vec!["other-device".to_owned()],
        ..Default::default()
    };
    let sample = collect_nvml_with_session(&session, &[42], &config, &mut state).unwrap();
    assert!(sample.devices.unwrap().is_empty());
    assert!(sample.processes.unwrap().is_empty());
}

#[test]
fn nvml_unavailable_process_memory_remains_unavailable() {
    let session = session();
    let sample = collect_nvml_with_session(
        &session,
        &[99],
        &ResourceMetricsGpuConfig::default(),
        &mut SamplingState::default(),
    )
    .unwrap();
    let process = &sample.processes.unwrap()[0];
    assert!(process.memory_used.is_none());
    assert!(process.compute_utilization.is_none());
}
unsafe extern "C" fn unavailable_count(_: *mut u32) -> i32 {
    1
}
unsafe extern "C" fn unavailable_handle(_: u32, _: *mut NvmlDevice) -> i32 {
    1
}
unsafe extern "C" fn unavailable_uuid(_: NvmlDevice, _: *mut c_char, _: u32) -> i32 {
    1
}
unsafe extern "C" fn unavailable_memory(_: NvmlDevice, _: *mut NvmlMemory) -> i32 {
    1
}
unsafe extern "C" fn unavailable_utilization(_: NvmlDevice, _: *mut NvmlUtilization) -> i32 {
    1
}

#[test]
fn nvml_query_failures_leave_individual_measurements_unavailable() {
    let config = ResourceMetricsGpuConfig::default();
    let mut state = SamplingState::default();
    let mut backend = session();
    backend.device_count = unavailable_count;
    assert!(collect_nvml_with_session(&backend, &[42], &config, &mut state).is_none());
    backend.device_count = device_count;
    backend.device_by_index = unavailable_handle;
    assert!(
        collect_nvml_with_session(&backend, &[42], &config, &mut state)
            .unwrap()
            .devices
            .unwrap()
            .is_empty()
    );
    backend.device_by_index = device_handle;
    backend.device_uuid = unavailable_uuid;
    backend.device_memory = unavailable_memory;
    backend.device_utilization = unavailable_utilization;
    backend.compute_processes = None;
    backend.graphics_processes = None;
    backend.process_utilization = None;
    let sample = collect_nvml_with_session(&backend, &[42], &config, &mut state).unwrap();
    let device = &sample.devices.unwrap()[0];
    assert_eq!(device.device_identifier, "nvidia:0");
    assert!(device.memory_used.is_none());
    assert!(device.compute_utilization.is_none());
    assert!(sample.processes.unwrap().is_empty());
}
