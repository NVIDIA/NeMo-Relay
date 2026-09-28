// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, c_char, c_void};

use chrono::Utc;
use libloading::Library;
use nemo_relay_types::api::resource_metrics::{
    AcceleratorDeviceMetrics, AcceleratorProcessMetrics, AcceleratorVendor, ResourceMeasurement,
    ResourceMeasurementUnit,
};

use super::AcceleratorSample;

const NVML_SUCCESS: i32 = 0;
const NVML_ERROR_INSUFFICIENT_SIZE: i32 = 7;
const NVML_VALUE_NOT_AVAILABLE: u64 = u64::MAX;

type NvmlDevice = *mut c_void;
type NvmlInit = unsafe extern "C" fn() -> i32;
type NvmlShutdown = unsafe extern "C" fn() -> i32;
type NvmlDeviceGetCount = unsafe extern "C" fn(*mut u32) -> i32;
type NvmlDeviceGetHandleByIndex = unsafe extern "C" fn(u32, *mut NvmlDevice) -> i32;
type NvmlDeviceGetUuid = unsafe extern "C" fn(NvmlDevice, *mut c_char, u32) -> i32;
type NvmlDeviceGetMemoryInfo = unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> i32;
type NvmlDeviceGetUtilization = unsafe extern "C" fn(NvmlDevice, *mut NvmlUtilization) -> i32;
type NvmlDeviceGetProcesses =
    unsafe extern "C" fn(NvmlDevice, *mut u32, *mut NvmlProcessInfo) -> i32;
type NvmlDeviceGetProcessUtilization =
    unsafe extern "C" fn(NvmlDevice, *mut NvmlProcessUtilization, *mut u32, u64) -> i32;

#[repr(C)]
#[derive(Default)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

#[repr(C)]
#[derive(Default)]
struct NvmlUtilization {
    gpu: u32,
    memory: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvmlProcessInfo {
    process_id: u32,
    used_gpu_memory: u64,
    gpu_instance_id: u32,
    compute_instance_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvmlProcessUtilization {
    process_id: u32,
    timestamp: u64,
    streaming_multiprocessor_utilization: u32,
    memory_utilization: u32,
    encoder_utilization: u32,
    decoder_utilization: u32,
}

pub(super) fn collect(process_ids: &[u32]) -> AcceleratorSample {
    let mut sample = collect_nvml(process_ids).unwrap_or_default();
    #[cfg(target_os = "linux")]
    sample.processes.extend(collect_linux_drm(process_ids));
    sample
}

fn collect_nvml(process_ids: &[u32]) -> Option<AcceleratorSample> {
    let library = load_nvml()?;
    // SAFETY: Every function pointer is loaded from the live NVML library with its documented ABI.
    unsafe {
        let initialize: NvmlInit = symbol(&library, b"nvmlInit_v2\0")?;
        let shutdown: NvmlShutdown = symbol(&library, b"nvmlShutdown\0")?;
        if initialize() != NVML_SUCCESS {
            return None;
        }
        let _guard = NvmlShutdownGuard(shutdown);
        let device_count: NvmlDeviceGetCount = symbol(&library, b"nvmlDeviceGetCount_v2\0")?;
        let device_by_index: NvmlDeviceGetHandleByIndex =
            symbol(&library, b"nvmlDeviceGetHandleByIndex_v2\0")?;
        let device_uuid: NvmlDeviceGetUuid = symbol(&library, b"nvmlDeviceGetUUID\0")?;
        let device_memory: NvmlDeviceGetMemoryInfo =
            symbol(&library, b"nvmlDeviceGetMemoryInfo\0")?;
        let device_utilization: NvmlDeviceGetUtilization =
            symbol(&library, b"nvmlDeviceGetUtilizationRates\0")?;
        let compute_processes: Option<NvmlDeviceGetProcesses> =
            symbol(&library, b"nvmlDeviceGetComputeRunningProcesses_v3\0");
        let graphics_processes: Option<NvmlDeviceGetProcesses> =
            symbol(&library, b"nvmlDeviceGetGraphicsRunningProcesses_v3\0");
        let process_utilization: Option<NvmlDeviceGetProcessUtilization> =
            symbol(&library, b"nvmlDeviceGetProcessUtilization\0");

        let mut count = 0_u32;
        if device_count(&mut count) != NVML_SUCCESS {
            return None;
        }
        let owned = process_ids.iter().copied().collect::<BTreeSet<_>>();
        let mut result = AcceleratorSample::default();
        for index in 0..count {
            let mut device = std::ptr::null_mut();
            if device_by_index(index, &mut device) != NVML_SUCCESS || device.is_null() {
                continue;
            }
            let identifier =
                read_nvml_uuid(device_uuid, device).unwrap_or_else(|| format!("nvidia:{index}"));
            let mut memory = NvmlMemory::default();
            let memory_used = if device_memory(device, &mut memory) == NVML_SUCCESS {
                ResourceMeasurement::available(
                    Utc::now(),
                    memory.used,
                    ResourceMeasurementUnit::Bytes,
                )
            } else {
                ResourceMeasurement::unavailable(Utc::now())
            };
            let mut utilization = NvmlUtilization::default();
            let compute_utilization =
                if device_utilization(device, &mut utilization) == NVML_SUCCESS {
                    ResourceMeasurement::available(
                        Utc::now(),
                        f64::from(utilization.gpu),
                        ResourceMeasurementUnit::Percentage,
                    )
                } else {
                    ResourceMeasurement::unavailable(Utc::now())
                };
            result.devices.push(AcceleratorDeviceMetrics {
                vendor: AcceleratorVendor::Nvidia,
                device_identifier: identifier.clone(),
                device_index: Some(index),
                memory_used,
                compute_utilization,
            });

            let mut process_memory = BTreeMap::<u32, Option<u64>>::new();
            for query in [compute_processes, graphics_processes]
                .into_iter()
                .flatten()
            {
                for process in read_nvml_processes(query, device) {
                    if owned.contains(&process.process_id) {
                        let memory = (process.used_gpu_memory != NVML_VALUE_NOT_AVAILABLE)
                            .then_some(process.used_gpu_memory);
                        process_memory
                            .entry(process.process_id)
                            .and_modify(|existing| {
                                *existing = match (*existing, memory) {
                                    (Some(current), Some(next)) => Some(current.max(next)),
                                    (None, next) => next,
                                    (current, None) => current,
                                }
                            })
                            .or_insert(memory);
                    }
                }
            }
            let process_utilization = process_utilization
                .map(|query| read_nvml_process_utilization(query, device, &owned))
                .unwrap_or_default();
            let observed_processes = process_memory
                .keys()
                .chain(process_utilization.keys())
                .copied()
                .collect::<BTreeSet<_>>();
            for process_id in observed_processes {
                result.processes.push(AcceleratorProcessMetrics {
                    vendor: AcceleratorVendor::Nvidia,
                    device_identifier: identifier.clone(),
                    device_index: Some(index),
                    process_id,
                    memory_used: process_memory
                        .get(&process_id)
                        .copied()
                        .flatten()
                        .map_or_else(
                            || ResourceMeasurement::unavailable(Utc::now()),
                            |value| {
                                ResourceMeasurement::available(
                                    Utc::now(),
                                    value,
                                    ResourceMeasurementUnit::Bytes,
                                )
                            },
                        ),
                    compute_utilization: process_utilization.get(&process_id).copied().map_or_else(
                        || ResourceMeasurement::unavailable(Utc::now()),
                        |value| {
                            ResourceMeasurement::available(
                                Utc::now(),
                                value,
                                ResourceMeasurementUnit::Percentage,
                            )
                        },
                    ),
                });
            }
        }
        Some(result)
    }
}

struct NvmlShutdownGuard(NvmlShutdown);

impl Drop for NvmlShutdownGuard {
    fn drop(&mut self) {
        // SAFETY: NVML was initialized before this guard was created and is shut down once.
        unsafe {
            (self.0)();
        }
    }
}

fn load_nvml() -> Option<Library> {
    #[cfg(target_os = "linux")]
    let names = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        .into_iter()
        .map(std::path::PathBuf::from)
        .collect::<Vec<_>>();
    #[cfg(windows)]
    let names = {
        let mut paths = Vec::new();
        if let Some(system_root) = std::env::var_os("SystemRoot") {
            paths.push(
                std::path::PathBuf::from(system_root)
                    .join("System32")
                    .join("nvml.dll"),
            );
        }
        if let Some(program_files) = std::env::var_os("ProgramW6432") {
            paths.push(
                std::path::PathBuf::from(program_files)
                    .join("NVIDIA Corporation")
                    .join("NVSMI")
                    .join("nvml.dll"),
            );
        }
        paths
    };
    names.into_iter().find_map(|name| {
        // SAFETY: Symbols are used only while the returned library remains alive.
        unsafe { Library::new(name).ok() }
    })
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Option<T> {
    // SAFETY: The caller supplies the documented NVML symbol type and keeps `library` alive.
    unsafe { library.get::<T>(name).ok().map(|symbol| *symbol) }
}

unsafe fn read_nvml_uuid(query: NvmlDeviceGetUuid, device: NvmlDevice) -> Option<String> {
    let mut buffer = [0_i8; 96];
    // SAFETY: `buffer` is writable for its reported length and `device` is a live NVML handle.
    if unsafe { query(device, buffer.as_mut_ptr(), buffer.len() as u32) } != NVML_SUCCESS {
        return None;
    }
    // SAFETY: NVML guarantees a null-terminated UUID on success.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_str()
        .ok()
        .map(str::to_owned)
}

unsafe fn read_nvml_processes(
    query: NvmlDeviceGetProcesses,
    device: NvmlDevice,
) -> Vec<NvmlProcessInfo> {
    let mut count = 0_u32;
    // SAFETY: A null output buffer is the documented size-query form.
    let first = unsafe { query(device, &mut count, std::ptr::null_mut()) };
    if count == 0 || !matches!(first, NVML_SUCCESS | NVML_ERROR_INSUFFICIENT_SIZE) {
        return Vec::new();
    }
    let mut processes = vec![NvmlProcessInfo::default(); count as usize];
    // SAFETY: `processes` has capacity for the count supplied back to NVML.
    if unsafe { query(device, &mut count, processes.as_mut_ptr()) } != NVML_SUCCESS {
        return Vec::new();
    }
    processes.truncate(count as usize);
    processes
}

unsafe fn read_nvml_process_utilization(
    query: NvmlDeviceGetProcessUtilization,
    device: NvmlDevice,
    owned: &BTreeSet<u32>,
) -> BTreeMap<u32, f64> {
    let mut count = 0_u32;
    // SAFETY: A null output buffer is the documented size-query form.
    let first = unsafe { query(device, std::ptr::null_mut(), &mut count, 0) };
    if count == 0 || !matches!(first, NVML_SUCCESS | NVML_ERROR_INSUFFICIENT_SIZE) {
        return BTreeMap::new();
    }
    let mut samples = vec![NvmlProcessUtilization::default(); count as usize];
    // SAFETY: `samples` has capacity for the count supplied back to NVML.
    if unsafe { query(device, samples.as_mut_ptr(), &mut count, 0) } != NVML_SUCCESS {
        return BTreeMap::new();
    }
    samples.truncate(count as usize);
    let mut latest = BTreeMap::<u32, (u64, f64)>::new();
    for sample in samples {
        if owned.contains(&sample.process_id) {
            let candidate = (
                sample.timestamp,
                f64::from(sample.streaming_multiprocessor_utilization),
            );
            latest
                .entry(sample.process_id)
                .and_modify(|current| {
                    if candidate.0 > current.0 {
                        *current = candidate;
                    }
                })
                .or_insert(candidate);
        }
    }
    latest
        .into_iter()
        .map(|(process_id, (_, utilization))| (process_id, utilization))
        .collect()
}

#[cfg(target_os = "linux")]
fn collect_linux_drm(process_ids: &[u32]) -> Vec<AcceleratorProcessMetrics> {
    use std::fs;

    let mut records = Vec::new();
    for process_id in process_ids {
        let Ok(entries) = fs::read_dir(format!("/proc/{process_id}/fdinfo")) else {
            continue;
        };
        let mut clients = BTreeMap::<String, DrmClient>::new();
        for entry in entries.filter_map(Result::ok) {
            let Ok(contents) = fs::read_to_string(entry.path()) else {
                continue;
            };
            let Some(client) = parse_drm_client(&contents) else {
                continue;
            };
            if client.vendor == AcceleratorVendor::Nvidia {
                continue;
            }
            clients.entry(client.identity.clone()).or_insert(client);
        }
        let mut by_device = BTreeMap::<String, Vec<DrmClient>>::new();
        for client in clients.into_values() {
            by_device
                .entry(client.device_identifier.clone())
                .or_default()
                .push(client);
        }
        for (device_identifier, clients) in by_device {
            let vendor = clients[0].vendor;
            let memory = aggregate_drm_memory(&clients);
            let utilization = clients
                .iter()
                .filter_map(|client| client.compute_utilization)
                .reduce(f64::max);
            records.push(AcceleratorProcessMetrics {
                vendor,
                device_identifier,
                device_index: None,
                process_id: *process_id,
                memory_used: memory.map_or_else(
                    || ResourceMeasurement::unavailable(Utc::now()),
                    |(value, unit)| ResourceMeasurement::available(Utc::now(), value, unit),
                ),
                compute_utilization: utilization.map_or_else(
                    || ResourceMeasurement::unavailable(Utc::now()),
                    |value| {
                        ResourceMeasurement::available(
                            Utc::now(),
                            value,
                            ResourceMeasurementUnit::Percentage,
                        )
                    },
                ),
            });
        }
    }
    records
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
struct DrmClient {
    identity: String,
    vendor: AcceleratorVendor,
    device_identifier: String,
    memory_regions: Vec<(u64, ResourceMeasurementUnit)>,
    compute_utilization: Option<f64>,
}

#[cfg(target_os = "linux")]
fn parse_drm_client(contents: &str) -> Option<DrmClient> {
    let values = contents
        .lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim(), value.trim()))
        .collect::<BTreeMap<_, _>>();
    let driver = *values.get("drm-driver")?;
    let client_id = *values.get("drm-client-id")?;
    let device_identifier = values
        .get("drm-pdev")
        .or_else(|| values.get("drm-minor"))
        .map(|value| (*value).to_owned())
        .unwrap_or_else(|| driver.to_owned());
    let vendor = match driver.to_ascii_lowercase().as_str() {
        "amdgpu" | "radeon" => AcceleratorVendor::Amd,
        "i915" | "xe" => AcceleratorVendor::Intel,
        "nouveau" | "nvidia" => AcceleratorVendor::Nvidia,
        _ => AcceleratorVendor::Other,
    };
    let memory_regions = values
        .iter()
        .filter(|(key, _)| {
            key.starts_with("drm-memory-")
                && !key.ends_with("-resident")
                && !key.ends_with("-purgeable")
                && !key.ends_with("-active")
        })
        .filter_map(|(_, value)| parse_drm_memory(value))
        .collect();
    let compute_utilization = values
        .iter()
        .filter_map(|(key, cycles)| {
            let engine = key.strip_prefix("drm-cycles-")?;
            let total = values.get(format!("drm-total-cycles-{engine}").as_str())?;
            let cycles = cycles.split_whitespace().next()?.parse::<f64>().ok()?;
            let total = total.split_whitespace().next()?.parse::<f64>().ok()?;
            (total > 0.0).then_some((cycles / total * 100.0).clamp(0.0, 100.0))
        })
        .reduce(f64::max);
    Some(DrmClient {
        identity: format!("{driver}:{device_identifier}:{client_id}"),
        vendor,
        device_identifier,
        memory_regions,
        compute_utilization,
    })
}

#[cfg(target_os = "linux")]
fn parse_drm_memory(value: &str) -> Option<(u64, ResourceMeasurementUnit)> {
    let mut fields = value.split_whitespace();
    let value = fields.next()?.parse().ok()?;
    let unit = match fields.next()? {
        "KiB" => ResourceMeasurementUnit::Kibibytes,
        "bytes" | "B" => ResourceMeasurementUnit::Bytes,
        _ => return None,
    };
    Some((value, unit))
}

#[cfg(target_os = "linux")]
fn aggregate_drm_memory(clients: &[DrmClient]) -> Option<(u64, ResourceMeasurementUnit)> {
    let mut unit = None;
    let mut total = 0_u64;
    let mut present = false;
    for (value, candidate_unit) in clients
        .iter()
        .flat_map(|client| client.memory_regions.iter().copied())
    {
        if unit.is_some_and(|unit| unit != candidate_unit) {
            return None;
        }
        unit = Some(candidate_unit);
        total = total.checked_add(value)?;
        present = true;
    }
    present.then(|| (total, unit.expect("present DRM memory has a unit")))
}
