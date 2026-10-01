// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, c_char, c_void};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use libloading::Library;
use nemo_relay_types::api::resource_metrics::{
    AcceleratorDeviceMetrics, AcceleratorProcessMetrics, AcceleratorVendor, CapacityUnit,
    ResourceMeasurement, UtilizationUnit,
};

use super::{AcceleratorSample, accelerator_device_is_selected};
use crate::plugins::resource_metrics::config::ResourceMetricsGpuConfig;

const NVML_SUCCESS: i32 = 0;
const NVML_ERROR_INSUFFICIENT_SIZE: i32 = 7;
const NVML_VALUE_NOT_AVAILABLE: u64 = u64::MAX;

static NVML_SESSION: LazyLock<Mutex<NvmlSessionCache>> =
    LazyLock::new(|| Mutex::new(NvmlSessionCache::default()));

#[derive(Default)]
struct NvmlSessionCache {
    session: Option<NvmlSession>,
    retry_after: Option<Instant>,
}

#[derive(Default)]
pub(super) struct SamplingState {
    nvml_process_timestamps: BTreeMap<String, u64>,
    #[cfg(target_os = "linux")]
    drm_counter_baselines: BTreeMap<(u32, u64, String, String), (DrmEngineCounter, Instant)>,
}

type NvmlDevice = *mut c_void;
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

struct NvmlSession {
    _library: Library,
    device_count: NvmlDeviceGetCount,
    device_by_index: NvmlDeviceGetHandleByIndex,
    device_uuid: NvmlDeviceGetUuid,
    device_memory: NvmlDeviceGetMemoryInfo,
    device_utilization: NvmlDeviceGetUtilization,
    compute_processes: Option<NvmlDeviceGetProcesses>,
    graphics_processes: Option<NvmlDeviceGetProcesses>,
    process_utilization: Option<NvmlDeviceGetProcessUtilization>,
}

pub(super) fn collect(
    process_ids: &[u32],
    config: &ResourceMetricsGpuConfig,
    sampling_state: &mut SamplingState,
) -> AcceleratorSample {
    let sample = collect_nvml(process_ids, config, sampling_state).unwrap_or_default();
    #[cfg(target_os = "linux")]
    let mut sample = sample;
    #[cfg(target_os = "linux")]
    if config.device_metrics {
        sample
            .devices
            .get_or_insert_with(Vec::new)
            .extend(collect_linux_drm_devices(config));
    }
    #[cfg(target_os = "linux")]
    if config.process_metrics {
        sample
            .processes
            .get_or_insert_with(Vec::new)
            .extend(collect_linux_drm(process_ids, sampling_state));
    }
    select_measurement_groups(sample, config)
}

fn select_measurement_groups(
    mut sample: AcceleratorSample,
    config: &ResourceMetricsGpuConfig,
) -> AcceleratorSample {
    if !config.device_metrics {
        sample.devices = None;
    }
    if !config.process_metrics {
        sample.processes = None;
    }
    if !config.devices.is_empty() {
        if let Some(devices) = &mut sample.devices {
            devices.retain(|device| {
                config.devices.iter().any(|selector| {
                    selector == &device.device_identifier
                        || device
                            .device_index
                            .is_some_and(|index| selector == &index.to_string())
                })
            });
        }
        if let Some(processes) = &mut sample.processes {
            processes.retain(|process| {
                config.devices.iter().any(|selector| {
                    selector == &process.device_identifier
                        || process
                            .device_index
                            .is_some_and(|index| selector == &index.to_string())
                })
            });
        }
    }
    sample
}

#[cfg(target_os = "linux")]
fn collect_linux_drm_devices(config: &ResourceMetricsGpuConfig) -> Vec<AcceleratorDeviceMetrics> {
    collect_linux_drm_devices_from(std::path::Path::new("/sys/class/drm"), config)
}

#[cfg(target_os = "linux")]
fn collect_linux_drm_devices_from(
    directory: &std::path::Path,
    config: &ResourceMetricsGpuConfig,
) -> Vec<AcceleratorDeviceMetrics> {
    use std::fs;

    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut devices = Vec::new();
    let mut seen = BTreeSet::new();
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name();
        let Some(index) = name
            .to_str()
            .and_then(|name| name.strip_prefix("card"))
            .filter(|index| !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()))
        else {
            continue;
        };
        let device = entry.path().join("device");
        let vendor = match fs::read_to_string(device.join("vendor"))
            .ok()
            .as_deref()
            .map(str::trim)
        {
            Some("0x1002") => AcceleratorVendor::Amd,
            Some("0x8086") => AcceleratorVendor::Intel,
            _ => continue,
        };
        let identifier = fs::canonicalize(&device)
            .ok()
            .and_then(|path| {
                path.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| format!("card{index}"));
        if !seen.insert(identifier.clone())
            || (!config.devices.is_empty() && !config.devices.contains(&identifier))
        {
            continue;
        }
        let memory_used = fs::read_to_string(device.join("mem_info_vram_used"))
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .map(|bytes| ResourceMeasurement::new(bytes / 1_024, CapacityUnit::Kibibytes));
        let compute_utilization = (vendor == AcceleratorVendor::Amd)
            .then(|| fs::read_to_string(device.join("gpu_busy_percent")).ok())
            .flatten()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .filter(|percent| *percent <= 100)
            .map(|percent| {
                ResourceMeasurement::new(f64::from(percent), UtilizationUnit::Percentage)
            });
        if memory_used.is_none() && compute_utilization.is_none() {
            continue;
        }
        devices.push(AcceleratorDeviceMetrics {
            vendor,
            device_identifier: identifier,
            device_index: None,
            memory_used,
            compute_utilization,
        });
    }
    devices
}

fn collect_nvml(
    process_ids: &[u32],
    config: &ResourceMetricsGpuConfig,
    sampling_state: &mut SamplingState,
) -> Option<AcceleratorSample> {
    let mut cache = NVML_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if cache.session.is_none() {
        if cache
            .retry_after
            .is_some_and(|retry_after| Instant::now() < retry_after)
        {
            return None;
        }
        cache.session = load_nvml_session();
        if cache.session.is_none() {
            cache.retry_after = Some(Instant::now() + Duration::from_secs(60));
            return None;
        }
        cache.retry_after = None;
    }
    let session = cache.session.as_ref()?;
    collect_nvml_with_session(session, process_ids, config, sampling_state)
}

fn collect_nvml_with_session(
    session: &NvmlSession,
    process_ids: &[u32],
    config: &ResourceMetricsGpuConfig,
    sampling_state: &mut SamplingState,
) -> Option<AcceleratorSample> {
    // SAFETY: The session keeps the library loaded and NVML initialized for this process.
    unsafe {
        let mut count = 0_u32;
        if (session.device_count)(&mut count) != NVML_SUCCESS {
            return None;
        }
        let owned = process_ids.iter().copied().collect::<BTreeSet<_>>();
        let mut result = AcceleratorSample {
            devices: config.device_metrics.then(Vec::new),
            processes: config.process_metrics.then(Vec::new),
        };
        for index in 0..count {
            let mut device = std::ptr::null_mut();
            if (session.device_by_index)(index, &mut device) != NVML_SUCCESS || device.is_null() {
                continue;
            }
            let identifier = read_nvml_uuid(session.device_uuid, device)
                .unwrap_or_else(|| format!("nvidia:{index}"));
            if !accelerator_device_is_selected(config, &identifier, index) {
                continue;
            }
            let mut memory = NvmlMemory::default();
            let memory_used = (config.device_metrics
                && (session.device_memory)(device, &mut memory) == NVML_SUCCESS)
                .then(|| ResourceMeasurement::new(memory.used / 1_024, CapacityUnit::Kibibytes));
            let mut utilization = NvmlUtilization::default();
            let compute_utilization = (config.device_metrics
                && (session.device_utilization)(device, &mut utilization) == NVML_SUCCESS)
                .then(|| {
                    ResourceMeasurement::new(
                        f64::from(utilization.gpu),
                        UtilizationUnit::Percentage,
                    )
                });
            if config.device_metrics {
                result
                    .devices
                    .as_mut()
                    .expect("enabled device metrics")
                    .push(AcceleratorDeviceMetrics {
                        vendor: AcceleratorVendor::Nvidia,
                        device_identifier: identifier.clone(),
                        device_index: Some(index),
                        memory_used,
                        compute_utilization,
                    });
            }

            let mut process_memory = BTreeMap::<u32, Option<u64>>::new();
            for query in [session.compute_processes, session.graphics_processes]
                .into_iter()
                .flatten()
                .filter(|_| config.process_metrics)
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
            let process_utilization = session
                .process_utilization
                .filter(|_| config.process_metrics)
                .map(|query| {
                    let timestamps = &mut sampling_state.nvml_process_timestamps;
                    let has_previous_sample = timestamps.contains_key(&identifier);
                    let last_seen = timestamps.get(&identifier).copied().unwrap_or(0);
                    let (samples, newest_timestamp) =
                        read_nvml_process_utilization(query, device, &owned, last_seen);
                    if let Some(timestamp) = newest_timestamp {
                        timestamps.insert(identifier.clone(), timestamp);
                    }
                    if has_previous_sample {
                        samples
                    } else {
                        BTreeMap::new()
                    }
                })
                .unwrap_or_default();
            let observed_processes = process_memory
                .keys()
                .chain(process_utilization.keys())
                .copied()
                .collect::<BTreeSet<_>>();
            for process_id in observed_processes {
                result
                    .processes
                    .as_mut()
                    .expect("enabled process metrics")
                    .push(AcceleratorProcessMetrics {
                        vendor: AcceleratorVendor::Nvidia,
                        device_identifier: identifier.clone(),
                        device_index: Some(index),
                        process_id,
                        memory_used: process_memory.get(&process_id).copied().flatten().map(
                            |value| {
                                ResourceMeasurement::new(value / 1_024, CapacityUnit::Kibibytes)
                            },
                        ),
                        compute_utilization: process_utilization.get(&process_id).copied().map(
                            |value| ResourceMeasurement::new(value, UtilizationUnit::Percentage),
                        ),
                    });
            }
        }
        Some(result)
    }
}

fn load_nvml_session() -> Option<NvmlSession> {
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
        let library = unsafe { Library::new(name).ok()? };
        load_nvml_session_from_library(library)
    })
}

fn load_nvml_session_from_library(library: Library) -> Option<NvmlSession> {
    // SAFETY: Each symbol is loaded with its documented NVML ABI.
    unsafe {
        let initialize: unsafe extern "C" fn() -> i32 = symbol(&library, b"nvmlInit_v2\0")?;
        let device_count = symbol(&library, b"nvmlDeviceGetCount_v2\0")?;
        let device_by_index = symbol(&library, b"nvmlDeviceGetHandleByIndex_v2\0")?;
        let device_uuid = symbol(&library, b"nvmlDeviceGetUUID\0")?;
        let device_memory = symbol(&library, b"nvmlDeviceGetMemoryInfo\0")?;
        let device_utilization = symbol(&library, b"nvmlDeviceGetUtilizationRates\0")?;
        let compute_processes = symbol(&library, b"nvmlDeviceGetComputeRunningProcesses_v3\0");
        let graphics_processes = symbol(&library, b"nvmlDeviceGetGraphicsRunningProcesses_v3\0");
        let process_utilization = symbol(&library, b"nvmlDeviceGetProcessUtilization\0");
        if initialize() != NVML_SUCCESS {
            return None;
        }
        Some(NvmlSession {
            device_count,
            device_by_index,
            device_uuid,
            device_memory,
            device_utilization,
            compute_processes,
            graphics_processes,
            process_utilization,
            _library: library,
        })
    }
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Option<T> {
    // SAFETY: The caller supplies the documented NVML symbol type and keeps `library` alive.
    unsafe { library.get::<T>(name).ok().map(|symbol| *symbol) }
}

unsafe fn read_nvml_uuid(query: NvmlDeviceGetUuid, device: NvmlDevice) -> Option<String> {
    let mut buffer: [c_char; 96] = [0; 96];
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
    last_seen_timestamp: u64,
) -> (BTreeMap<u32, f64>, Option<u64>) {
    let mut count = 0_u32;
    // SAFETY: A null output buffer is the documented size-query form.
    let first = unsafe {
        query(
            device,
            std::ptr::null_mut(),
            &mut count,
            last_seen_timestamp,
        )
    };
    if count == 0 || !matches!(first, NVML_SUCCESS | NVML_ERROR_INSUFFICIENT_SIZE) {
        return (BTreeMap::new(), None);
    }
    let mut samples = vec![NvmlProcessUtilization::default(); count as usize];
    // SAFETY: `samples` has capacity for the count supplied back to NVML.
    if unsafe {
        query(
            device,
            samples.as_mut_ptr(),
            &mut count,
            last_seen_timestamp,
        )
    } != NVML_SUCCESS
    {
        return (BTreeMap::new(), None);
    }
    samples.truncate(count as usize);
    let newest_timestamp = samples.iter().map(|sample| sample.timestamp).max();
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
    let latest = latest
        .into_iter()
        .map(|(process_id, (_, utilization))| (process_id, utilization))
        .collect();
    (latest, newest_timestamp)
}

#[cfg(target_os = "linux")]
fn collect_linux_drm(
    process_ids: &[u32],
    sampling_state: &mut SamplingState,
) -> Vec<AcceleratorProcessMetrics> {
    use std::fs;

    let sampled_at = Instant::now();
    let mut records = Vec::new();
    let mut live_counters = BTreeSet::new();
    for process_id in process_ids {
        let Ok(start_identity) = super::linux::process_identity(*process_id) else {
            continue;
        };
        let Ok(entries) = fs::read_dir(format!("/proc/{process_id}/fdinfo")) else {
            continue;
        };
        let descriptors = std::path::PathBuf::from(format!("/proc/{process_id}/fd"));
        let mut clients = BTreeMap::<String, DrmClient>::new();
        for entry in entries.filter_map(Result::ok) {
            // Stat the underlying descriptor before reading fdinfo. Device numbers
            // also recognize DRM nodes opened through aliases outside /dev/dri.
            let descriptor = descriptors.join(entry.file_name());
            if !is_drm_descriptor(&descriptor) {
                continue;
            }
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
        if super::linux::process_identity(*process_id).ok() != Some(start_identity) {
            continue;
        }
        records.extend(drm_process_records(
            *process_id,
            start_identity,
            clients,
            sampled_at,
            sampling_state,
            &mut live_counters,
        ));
    }
    sampling_state
        .drm_counter_baselines
        .retain(|key, _| live_counters.contains(key));
    records
}

#[cfg(target_os = "linux")]
fn drm_process_records(
    process_id: u32,
    start_identity: u64,
    clients: BTreeMap<String, DrmClient>,
    sampled_at: Instant,
    sampling_state: &mut SamplingState,
    live_counters: &mut BTreeSet<(u32, u64, String, String)>,
) -> Vec<AcceleratorProcessMetrics> {
    let mut records = Vec::new();
    let mut by_device = BTreeMap::<String, Vec<DrmClient>>::new();
    for client in clients.into_values() {
        live_counters.extend(client.engine_counters.iter().map(|(engine, _)| {
            (
                process_id,
                start_identity,
                client.identity.clone(),
                engine.clone(),
            )
        }));
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
            .filter_map(|client| {
                drm_client_utilization(
                    process_id,
                    start_identity,
                    client,
                    sampling_state,
                    sampled_at,
                )
            })
            .reduce(f64::max);
        records.push(AcceleratorProcessMetrics {
            vendor,
            device_identifier,
            device_index: None,
            process_id,
            memory_used: memory.map(|(value, unit)| ResourceMeasurement::new(value, unit)),
            compute_utilization: utilization
                .map(|value| ResourceMeasurement::new(value, UtilizationUnit::Percentage)),
        });
    }
    records
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
struct DrmClient {
    identity: String,
    vendor: AcceleratorVendor,
    device_identifier: String,
    memory_regions: Vec<(u64, CapacityUnit)>,
    engine_counters: Vec<(String, DrmEngineCounter)>,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug)]
enum DrmEngineCounter {
    Cycles { busy: u64, total: u64 },
    BusyNanoseconds { busy: u64, capacity: u64 },
}

#[cfg(target_os = "linux")]
fn is_drm_descriptor(path: &std::path::Path) -> bool {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};

    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    metadata.file_type().is_char_device() && is_drm_device_number(metadata.rdev())
}

#[cfg(target_os = "linux")]
fn is_drm_device_number(device: u64) -> bool {
    // Linux dev_t encodes the major number in two disjoint bit ranges.
    let major = ((device >> 8) & 0xfff) | ((device >> 32) & 0xffff_f000);
    major == 226
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
    let mut memory_regions = BTreeMap::new();
    for (key, value) in &values {
        if let Some(region) = key.strip_prefix("drm-resident-")
            && let Some(memory) = parse_drm_memory(value)
        {
            memory_regions.insert(region, memory);
        }
    }
    let mut engine_counters = BTreeMap::new();
    for (key, value) in &values {
        let Some(engine) = key.strip_prefix("drm-cycles-") else {
            continue;
        };
        let Some(total) = values.get(format!("drm-total-cycles-{engine}").as_str()) else {
            continue;
        };
        if let (Some(busy), Some(total)) = (
            value
                .split_whitespace()
                .next()
                .and_then(|value| value.parse().ok()),
            total
                .split_whitespace()
                .next()
                .and_then(|value| value.parse().ok()),
        ) {
            engine_counters.insert(engine.to_owned(), DrmEngineCounter::Cycles { busy, total });
        }
    }
    for (key, value) in &values {
        let Some(engine) = key.strip_prefix("drm-engine-") else {
            continue;
        };
        if engine.starts_with("capacity-") || engine_counters.contains_key(engine) {
            continue;
        }
        let mut parts = value.split_whitespace();
        let (Some(busy), Some("ns")) = (
            parts.next().and_then(|value| value.parse().ok()),
            parts.next(),
        ) else {
            continue;
        };
        let capacity = values
            .get(format!("drm-engine-capacity-{engine}").as_str())
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1);
        engine_counters.insert(
            engine.to_owned(),
            DrmEngineCounter::BusyNanoseconds { busy, capacity },
        );
    }
    Some(DrmClient {
        identity: format!("{driver}:{device_identifier}:{client_id}"),
        vendor,
        device_identifier,
        memory_regions: memory_regions.into_values().collect(),
        engine_counters: engine_counters.into_iter().collect(),
    })
}

#[cfg(target_os = "linux")]
fn drm_client_utilization(
    process_id: u32,
    start_identity: u64,
    client: &DrmClient,
    sampling_state: &mut SamplingState,
    sampled_at: Instant,
) -> Option<f64> {
    let baselines = &mut sampling_state.drm_counter_baselines;
    client
        .engine_counters
        .iter()
        .filter_map(|(engine, counter)| {
            let key = (
                process_id,
                start_identity,
                client.identity.clone(),
                engine.clone(),
            );
            match baselines.get(&key).copied() {
                Some((
                    DrmEngineCounter::Cycles {
                        busy: previous_busy,
                        total: previous_total,
                    },
                    _,
                )) if let DrmEngineCounter::Cycles { busy, total } = counter
                    && *busy >= previous_busy
                    && *total >= previous_total =>
                {
                    baselines.insert(key, (*counter, sampled_at));
                    let total_delta = total - previous_total;
                    (total_delta > 0).then(|| {
                        ((*busy - previous_busy) as f64 / total_delta as f64 * 100.0)
                            .clamp(0.0, 100.0)
                    })
                }
                Some((
                    DrmEngineCounter::BusyNanoseconds {
                        busy: previous_busy,
                        ..
                    },
                    previous_at,
                )) if let DrmEngineCounter::BusyNanoseconds { busy, capacity } = counter
                    && *busy >= previous_busy =>
                {
                    baselines.insert(key, (*counter, sampled_at));
                    let elapsed_nanos = sampled_at.duration_since(previous_at).as_nanos();
                    elapsed_nanos
                        .checked_mul(u128::from(*capacity))
                        .filter(|denominator| *denominator > 0)
                        .map(|denominator| {
                            ((*busy - previous_busy) as f64 / denominator as f64 * 100.0)
                                .clamp(0.0, 100.0)
                        })
                }
                Some((previous, _))
                    if std::mem::discriminant(&previous) == std::mem::discriminant(counter) =>
                {
                    baselines.insert(key, (*counter, sampled_at));
                    None
                }
                None => {
                    baselines.insert(key, (*counter, sampled_at));
                    None
                }
                Some(_) => {
                    baselines.insert(key, (*counter, sampled_at));
                    None
                }
            }
        })
        .reduce(f64::max)
}

#[cfg(target_os = "linux")]
fn parse_drm_memory(value: &str) -> Option<(u64, CapacityUnit)> {
    let mut fields = value.split_whitespace();
    let value = fields.next()?.parse::<u64>().ok()?;
    let unit = match fields.next() {
        Some("KiB") => CapacityUnit::Kibibytes,
        Some("MiB") => {
            return Some((value.checked_mul(1_024)?, CapacityUnit::Kibibytes));
        }
        Some("bytes" | "B") | None => {
            return Some((value / 1_024, CapacityUnit::Kibibytes));
        }
        _ => return None,
    };
    Some((value, unit))
}

#[cfg(target_os = "linux")]
fn aggregate_drm_memory(clients: &[DrmClient]) -> Option<(u64, CapacityUnit)> {
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

#[cfg(all(test, target_os = "linux"))]
#[path = "../../../tests/unit/resource_metrics/accelerator_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../tests/unit/resource_metrics/nvml_tests.rs"]
mod nvml_tests;
