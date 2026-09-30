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

#[test]
fn nvml_library_loading_validates_required_symbols_initialization_and_optional_queries() {
    let directory = tempfile::tempdir().unwrap();
    // Windows ARM runners may use an x64 compiler host, so build for the test process.
    #[cfg(windows)]
    let target = format!("{}-pc-windows-msvc", std::env::consts::ARCH);
    #[cfg(target_os = "linux")]
    let target = format!("{}-unknown-linux-gnu", std::env::consts::ARCH);
    let output = std::process::Command::new("rustc")
        .args([
            "--crate-type",
            "cdylib",
            "--edition",
            "2024",
            "--crate-name",
            "nvml_fixture",
            "--target",
            &target,
            "--out-dir",
        ])
        .arg(directory.path())
        .arg(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/resource_metrics/nvml_fixture.rs"),
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let path = directory.path().join(format!(
        "{}nvml_fixture{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));
    // SAFETY: The fixture exports the NVML C ABI and remains alive while symbols are used.
    let library = unsafe { Library::new(&path) }.unwrap();
    let backend = load_nvml_session_from_library(library).expect("required symbols should load");
    assert!(backend.compute_processes.is_none());
    assert!(backend.graphics_processes.is_none());
    assert!(backend.process_utilization.is_none());
    let result = collect_nvml_with_session(
        &backend,
        &[42],
        &ResourceMetricsGpuConfig::default(),
        &mut SamplingState::default(),
    )
    .unwrap();
    assert!(result.devices.unwrap().is_empty());
    assert!(result.processes.unwrap().is_empty());
    // SAFETY: The setter has the fixture's documented ABI; the loaded session owns the library.
    unsafe {
        symbol::<extern "C" fn(i32)>(&backend._library, b"fixture_initialize_result\0").unwrap()(1)
    };
    let library = unsafe { Library::new(&path) }.unwrap();
    assert!(load_nvml_session_from_library(library).is_none());
    assert!(load_nvml_session_from_library(session()._library).is_none());
}

unsafe extern "C" fn process_query_fails_after_size(
    _: NvmlDevice,
    count: *mut u32,
    out: *mut NvmlProcessInfo,
) -> i32 {
    unsafe { count.write(1) };
    if out.is_null() {
        NVML_ERROR_INSUFFICIENT_SIZE
    } else {
        1
    }
}

unsafe extern "C" fn utilization_query_fails_after_size(
    _: NvmlDevice,
    out: *mut NvmlProcessUtilization,
    count: *mut u32,
    _: u64,
) -> i32 {
    unsafe { count.write(1) };
    if out.is_null() {
        NVML_ERROR_INSUFFICIENT_SIZE
    } else {
        1
    }
}

unsafe extern "C" fn utilization_samples_out_of_order(
    _: NvmlDevice,
    out: *mut NvmlProcessUtilization,
    count: *mut u32,
    _: u64,
) -> i32 {
    unsafe { count.write(3) };
    if out.is_null() {
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }
    for (index, (timestamp, value)) in [(1, 10), (3, 30), (2, 20)].into_iter().enumerate() {
        unsafe {
            out.add(index).write(NvmlProcessUtilization {
                process_id: 42,
                timestamp,
                streaming_multiprocessor_utilization: value,
                ..Default::default()
            })
        };
    }
    NVML_SUCCESS
}

#[test]
fn nvml_process_queries_reject_second_call_failures_and_use_the_newest_sample() {
    let owned = BTreeSet::from([42]);
    // SAFETY: ABI fixtures use only the allocated buffers and ignore the dummy device.
    unsafe {
        assert!(
            read_nvml_processes(process_query_fails_after_size, std::ptr::dangling_mut())
                .is_empty()
        );
        let (values, newest) = read_nvml_process_utilization(
            utilization_query_fails_after_size,
            std::ptr::dangling_mut(),
            &owned,
            0,
        );
        assert!(values.is_empty());
        assert!(newest.is_none());
        let (values, newest) = read_nvml_process_utilization(
            utilization_samples_out_of_order,
            std::ptr::dangling_mut(),
            &owned,
            0,
        );
        assert_eq!(values[&42], 30.0);
        assert_eq!(newest, Some(3));
    }
}

#[test]
fn combined_vendor_samples_apply_selectors_and_disabled_groups_consistently() {
    for selectors in [
        vec![],
        vec!["amd:1".into()],
        vec!["1".into()],
        vec!["missing".into()],
    ] {
        for (device_metrics, process_metrics) in
            [(true, true), (false, true), (true, false), (false, false)]
        {
            let sample = AcceleratorSample {
                devices: Some(vec![AcceleratorDeviceMetrics {
                    vendor: AcceleratorVendor::Amd,
                    device_identifier: "amd:1".into(),
                    device_index: Some(1),
                    memory_used: None,
                    compute_utilization: None,
                }]),
                processes: Some(vec![AcceleratorProcessMetrics {
                    vendor: AcceleratorVendor::Amd,
                    device_identifier: "amd:1".into(),
                    device_index: Some(1),
                    process_id: 42,
                    memory_used: None,
                    compute_utilization: None,
                }]),
            };
            let config = ResourceMetricsGpuConfig {
                devices: selectors.clone(),
                device_metrics,
                process_metrics,
                ..Default::default()
            };
            let selected = select_measurement_groups(sample, &config);
            let expected_count = usize::from(
                selectors
                    .first()
                    .is_none_or(|selector| selector != "missing"),
            );
            assert_eq!(
                selected.devices.as_ref().map(Vec::len),
                device_metrics.then_some(expected_count)
            );
            assert_eq!(
                selected.processes.as_ref().map(Vec::len),
                process_metrics.then_some(expected_count)
            );
        }
    }
    let sample = collect(
        &[],
        &ResourceMetricsGpuConfig {
            device_metrics: false,
            process_metrics: false,
            ..Default::default()
        },
        &mut SamplingState::default(),
    );
    assert!(sample.devices.is_none());
    assert!(sample.processes.is_none());
}

unsafe extern "C" fn unsupported_process_query(
    _: NvmlDevice,
    count: *mut u32,
    _: *mut NvmlProcessInfo,
) -> i32 {
    unsafe { count.write(1) };
    1
}

unsafe extern "C" fn unsupported_utilization_query(
    _: NvmlDevice,
    _: *mut NvmlProcessUtilization,
    count: *mut u32,
    _: u64,
) -> i32 {
    unsafe { count.write(1) };
    1
}

unsafe extern "C" fn unknown_process_memory(
    _: NvmlDevice,
    count: *mut u32,
    out: *mut NvmlProcessInfo,
) -> i32 {
    unsafe { count.write(1) };
    if out.is_null() {
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }
    unsafe {
        out.write(NvmlProcessInfo {
            process_id: 42,
            used_gpu_memory: NVML_VALUE_NOT_AVAILABLE,
            ..Default::default()
        })
    };
    NVML_SUCCESS
}

#[test]
fn nvml_unsupported_process_queries_remain_unavailable_and_do_not_erase_known_memory() {
    let mut backend = session();
    backend.compute_processes = Some(unsupported_process_query);
    backend.graphics_processes = None;
    backend.process_utilization = Some(unsupported_utilization_query);
    let sample = collect_nvml_with_session(
        &backend,
        &[42],
        &ResourceMetricsGpuConfig::default(),
        &mut SamplingState::default(),
    )
    .unwrap();
    assert!(sample.processes.unwrap().is_empty());
    assert!(sample.devices.unwrap()[0].memory_used.is_some());
    backend.compute_processes = Some(processes);
    backend.graphics_processes = Some(unknown_process_memory);
    let sample = collect_nvml_with_session(
        &backend,
        &[42],
        &ResourceMetricsGpuConfig::default(),
        &mut SamplingState::default(),
    )
    .unwrap();
    let process = &sample.processes.unwrap()[0];
    assert_eq!(process.memory_used.as_ref().unwrap().value, 4_u64);
    assert!(process.compute_utilization.is_none());
}
