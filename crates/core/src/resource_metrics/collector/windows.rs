// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::io;
use std::mem::size_of;
use std::os::windows::ffi::OsStrExt;

use nemo_relay_types::api::resource_metrics::{ResourceMeasurementUnit, ResourceOperatingSystem};
use windows_sys::Win32::Foundation::{
    CloseHandle, DUPLICATE_SAME_ACCESS, DuplicateHandle, ERROR_NO_MORE_FILES, FILETIME, HANDLE,
    INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPPROCESS,
    TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX,
};
use windows_sys::Win32::System::Threading::{
    ALL_PROCESSOR_GROUPS, GetActiveProcessorCount, GetCurrentProcess, GetProcessHandleCount,
    GetProcessIoCounters, GetProcessTimes, IO_COUNTERS, OpenProcess, PROCESS_QUERY_INFORMATION,
    PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ,
};

use super::{
    AcceleratorSample, CollectionTarget, EnvironmentSample, FloatSample, IntegerSample,
    ProcessSample, ProcessSampleConfig, accelerator, bytes_to_kibibytes,
    filetime_intervals_to_milliseconds,
};

pub(super) const OPERATING_SYSTEM: ResourceOperatingSystem = ResourceOperatingSystem::Windows;
pub(super) const CPU_TIME_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Milliseconds;
pub(super) const RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Kibibytes;
pub(super) const PRIVATE_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Kibibytes;
pub(super) const PHYSICAL_FOOTPRINT_UNIT: ResourceMeasurementUnit =
    ResourceMeasurementUnit::Kibibytes;
pub(super) const VIRTUAL_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Kibibytes;
pub(super) const PEAK_RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit =
    ResourceMeasurementUnit::Kibibytes;

#[derive(Debug)]
pub(super) struct OwnedJobHandle(HANDLE);

// SAFETY: Windows Job Object handles may be queried from any thread and this wrapper only closes
// its own duplicate.
unsafe impl Send for OwnedJobHandle {}
// SAFETY: QueryInformationJobObject is thread-safe for a live Job Object handle.
unsafe impl Sync for OwnedJobHandle {}

impl OwnedJobHandle {
    pub(super) fn duplicate(source: isize) -> io::Result<Self> {
        let source = source as HANDLE;
        let current_process = unsafe { GetCurrentProcess() };
        let mut duplicate = std::ptr::null_mut();
        // SAFETY: Both process handles refer to this process, `source` is borrowed from the live
        // supervisor, and `duplicate` is writable storage for the resulting owned handle.
        if unsafe {
            DuplicateHandle(
                current_process,
                source,
                current_process,
                &mut duplicate,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        } == 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(Self(duplicate))
        }
    }
}

impl Drop for OwnedJobHandle {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns the duplicated Job Object handle.
        unsafe { CloseHandle(self.0) };
    }
}

struct OwnedHandle(windows_sys::Win32::Foundation::HANDLE);

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        // SAFETY: This handle was returned by a successful Windows open or snapshot call and is
        // closed exactly once by this owner.
        unsafe { CloseHandle(self.0) };
    }
}

pub(super) fn process_identity(process_id: u32) -> io::Result<u64> {
    let handle = open_process(process_id, PROCESS_QUERY_LIMITED_INFORMATION)?;
    process_times(handle.0).map(|(identity, _, _)| identity)
}

pub(super) fn parent_process_id(process_id: u32) -> io::Result<Option<u32>> {
    Ok(process_entries()?
        .into_iter()
        .find_map(|(candidate, parent)| (candidate == process_id).then_some(parent)))
}

pub(super) fn process_tree_ids(root_process_id: u32) -> io::Result<Vec<u32>> {
    let entries = process_entries()?;
    let mut process_ids = vec![root_process_id];
    let mut identities = HashMap::from([(root_process_id, process_identity(root_process_id)?)]);
    let mut visited = HashSet::from([root_process_id]);
    let mut next = 0;
    while next < process_ids.len() {
        let parent = process_ids[next];
        next += 1;
        let parent_identity = identities[&parent];
        for &(process_id, parent_process_id) in &entries {
            if parent_process_id != parent || visited.contains(&process_id) {
                continue;
            }
            // A child cannot predate its parent. This also rejects stale parent PIDs
            // after Windows reuses a process ID.
            let identity = match process_identity(process_id) {
                Ok(identity) => identity,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::NotFound | io::ErrorKind::InvalidInput
                    ) =>
                {
                    continue;
                }
                Err(error) => return Err(error),
            };
            if identity < parent_identity {
                continue;
            }
            visited.insert(process_id);
            identities.insert(process_id, identity);
            process_ids.push(process_id);
        }
    }
    Ok(process_ids)
}

pub(super) fn job_process_ids(job: &OwnedJobHandle) -> io::Result<Vec<u32>> {
    use windows_sys::Win32::Foundation::ERROR_MORE_DATA;
    use windows_sys::Win32::System::JobObjects::{
        JOBOBJECT_BASIC_PROCESS_ID_LIST, JobObjectBasicProcessIdList, QueryInformationJobObject,
    };

    let mut capacity = 64_usize;
    loop {
        let bytes = size_of::<JOBOBJECT_BASIC_PROCESS_ID_LIST>()
            .checked_add((capacity - 1).saturating_mul(size_of::<usize>()))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Job Object list is too large")
            })?;
        let buffer_size = u32::try_from(bytes).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Job Object list is too large")
        })?;
        let mut buffer = vec![0_usize; bytes.div_ceil(size_of::<usize>())];
        // SAFETY: The aligned buffer is writable for `buffer_size` bytes and the Job Object
        // handle remains owned by the collection target throughout this query.
        let succeeded = unsafe {
            QueryInformationJobObject(
                job.0,
                JobObjectBasicProcessIdList,
                buffer.as_mut_ptr().cast(),
                buffer_size,
                std::ptr::null_mut(),
            )
        } != 0;
        if !succeeded {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(ERROR_MORE_DATA as i32) {
                capacity = capacity.checked_mul(2).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Job Object list is too large")
                })?;
                continue;
            }
            return Err(error);
        }
        // SAFETY: A successful query initialized the fixed header and at most `capacity`
        // process IDs in the supplied buffer.
        let list = unsafe { &*buffer.as_ptr().cast::<JOBOBJECT_BASIC_PROCESS_ID_LIST>() };
        let count = list.NumberOfProcessIdsInList as usize;
        let assigned = list.NumberOfAssignedProcesses as usize;
        if count > capacity {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Job Object returned more process IDs than the buffer holds",
            ));
        }
        if count < assigned {
            capacity = assigned.max(capacity.checked_mul(2).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Job Object list is too large")
            })?);
            continue;
        }
        // SAFETY: The length was checked against the allocated flexible-array capacity.
        let ids = unsafe { std::slice::from_raw_parts(list.ProcessIdList.as_ptr(), count) };
        return ids
            .iter()
            .map(|id| {
                u32::try_from(*id).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Job Object process ID is too large",
                    )
                })
            })
            .collect();
    }
}

fn process_entries() -> io::Result<Vec<(u32, u32)>> {
    // SAFETY: The flags and zero process ID request a documented system process snapshot.
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    let snapshot = OwnedHandle(snapshot);
    let mut entry = PROCESSENTRY32W {
        dwSize: size_of::<PROCESSENTRY32W>() as u32,
        ..Default::default()
    };
    let mut entries = Vec::new();
    // SAFETY: `entry` has the documented size and remains valid for the complete iteration.
    if unsafe { Process32FirstW(snapshot.0, &mut entry) } == 0 {
        return Err(io::Error::last_os_error());
    }
    loop {
        entries.push((entry.th32ProcessID, entry.th32ParentProcessID));
        // SAFETY: The snapshot and output entry remain valid.
        if unsafe { Process32NextW(snapshot.0, &mut entry) } == 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(ERROR_NO_MORE_FILES as i32) {
                break;
            }
            return Err(error);
        }
    }
    Ok(entries)
}

pub(super) fn all_process_ids() -> io::Result<Vec<u32>> {
    let process_ids = process_entries()?
        .into_iter()
        .map(|(process_id, _)| process_id)
        .filter(|process_id| *process_id != 0)
        .collect::<Vec<_>>();
    if process_ids.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no processes were visible to the collector",
        ));
    }
    Ok(process_ids)
}

pub(super) fn process_sample(
    process_id: u32,
    config: ProcessSampleConfig,
    thread_counts: Option<&HashMap<u32, u64>>,
) -> io::Result<ProcessSample> {
    let handle = open_process(process_id, PROCESS_QUERY_LIMITED_INFORMATION)?;
    let (start_identity, user_cpu_time, system_cpu_time) = process_times(handle.0)?;
    let memory = if config.memory {
        open_process(process_id, PROCESS_QUERY_INFORMATION | PROCESS_VM_READ)
            .and_then(|handle| process_memory(handle.0))
            .ok()
    } else {
        None
    };
    let windows_handle_count = config
        .process
        .then(|| process_handle_count(handle.0).ok())
        .flatten();
    let io = config
        .disk_io
        .then(|| process_io_counters(handle.0).ok())
        .flatten();
    Ok(ProcessSample {
        process_id,
        start_identity,
        user_cpu_time: config
            .cpu
            .then(|| filetime_intervals_to_milliseconds(user_cpu_time)),
        system_cpu_time: config
            .cpu
            .then(|| filetime_intervals_to_milliseconds(system_cpu_time)),
        total_cpu_time: config
            .cpu
            .then(|| {
                user_cpu_time
                    .checked_add(system_cpu_time)
                    .map(filetime_intervals_to_milliseconds)
            })
            .flatten(),
        resident_memory: memory.map(|memory| bytes_to_kibibytes(memory.WorkingSetSize as u64)),
        private_memory: memory.map(|memory| bytes_to_kibibytes(memory.PrivateUsage as u64)),
        physical_footprint: None,
        virtual_memory: None,
        peak_resident_memory: memory
            .map(|memory| bytes_to_kibibytes(memory.PeakWorkingSetSize as u64)),
        thread_count: config
            .process
            .then(|| thread_counts.and_then(|counts| counts.get(&process_id).copied()))
            .flatten(),
        open_file_descriptor_count: None,
        windows_handle_count,
        disk_read_bytes: io.map(|io| io.ReadTransferCount),
        disk_write_bytes: io.map(|io| io.WriteTransferCount),
        disk_read_operations: io.map(|io| io.ReadOperationCount),
        disk_write_operations: io.map(|io| io.WriteOperationCount),
    })
}

pub(super) fn environment_sample(
    target: &CollectionTarget,
    _process_ids: &[u32],
    config: &crate::plugins::resource_metrics::config::ResourceMetricsConfig,
) -> io::Result<EnvironmentSample> {
    use windows_sys::Win32::System::JobObjects::{
        JOB_OBJECT_CPU_RATE_CONTROL_ENABLE, JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
        JOB_OBJECT_CPU_RATE_CONTROL_MIN_MAX_RATE, JOB_OBJECT_LIMIT_JOB_MEMORY,
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_CPU_RATE_CONTROL_INFORMATION,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectBasicAccountingInformation,
        JobObjectCpuRateControlInformation, JobObjectExtendedLimitInformation,
    };

    let Some(job) = target.job_handle.as_ref() else {
        return Ok(EnvironmentSample::default());
    };
    let accounting = config
        .process
        .enabled
        .then(|| {
            query_job_information::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>(
                job.0,
                JobObjectBasicAccountingInformation,
            )
            .ok()
        })
        .flatten();
    let limits = config
        .memory
        .enabled
        .then(|| {
            query_job_information::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>(
                job.0,
                JobObjectExtendedLimitInformation,
            )
            .ok()
        })
        .flatten();
    let cpu = config
        .cpu
        .enabled
        .then(|| {
            query_job_information::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>(
                job.0,
                JobObjectCpuRateControlInformation,
            )
            .ok()
        })
        .flatten();
    let effective_cpu_limit = cpu.and_then(|cpu| {
        if cpu.ControlFlags & JOB_OBJECT_CPU_RATE_CONTROL_ENABLE == 0 {
            return None;
        }
        let maximum_rate = if cpu.ControlFlags & JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP != 0 {
            // SAFETY: HARD_CAP selects the `CpuRate` union member.
            unsafe { cpu.Anonymous.CpuRate }
        } else if cpu.ControlFlags & JOB_OBJECT_CPU_RATE_CONTROL_MIN_MAX_RATE != 0 {
            // SAFETY: MIN_MAX_RATE selects the `Anonymous.MaxRate` union member.
            unsafe { u32::from(cpu.Anonymous.Anonymous.MaxRate) }
        } else {
            return None;
        };
        // SAFETY: ALL_PROCESSOR_GROUPS is the documented whole-system query.
        let processors = unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) };
        (processors > 0).then_some(FloatSample {
            value: f64::from(maximum_rate) / 10_000.0 * f64::from(processors),
            unit: ResourceMeasurementUnit::LogicalProcessors,
        })
    });
    let memory_limit = limits.and_then(|limits| {
        (limits.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_JOB_MEMORY != 0).then_some(
            IntegerSample {
                value: bytes_to_kibibytes(limits.JobMemoryLimit as u64),
                unit: ResourceMeasurementUnit::Kibibytes,
            },
        )
    });
    Ok(EnvironmentSample {
        effective_cpu_limit,
        memory_limit,
        lifetime_process_creation_count: accounting.map(|accounting| IntegerSample {
            value: u64::from(accounting.TotalProcesses),
            unit: ResourceMeasurementUnit::Processes,
        }),
        ..EnvironmentSample::default()
    })
}

pub(super) fn global_environment_sample(
    _config: &crate::plugins::resource_metrics::config::ResourceMetricsConfig,
) -> io::Result<EnvironmentSample> {
    Ok(EnvironmentSample::default())
}

pub(super) fn accelerator_sample(
    process_ids: &[u32],
    config: &crate::plugins::resource_metrics::config::ResourceMetricsGpuConfig,
    sampling_state: &mut super::SamplingState,
) -> AcceleratorSample {
    accelerator::collect(process_ids, config, &mut sampling_state.accelerator)
}

pub(super) fn filesystem_capacity(path: &std::path::Path) -> io::Result<(u64, u64, u64)> {
    let mut wide = path.as_os_str().encode_wide().collect::<Vec<_>>();
    wide.push(0);
    let mut available = 0_u64;
    let mut total = 0_u64;
    let mut free = 0_u64;
    // SAFETY: `wide` is nul-terminated and each output pointer references writable storage.
    if unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut available, &mut total, &mut free) } == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok((total, available, free))
    }
}

fn process_io_counters(handle: HANDLE) -> io::Result<IO_COUNTERS> {
    let mut counters = IO_COUNTERS::default();
    // SAFETY: `handle` is live and `counters` is writable storage for the API result.
    if unsafe { GetProcessIoCounters(handle, &mut counters) } == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(counters)
    }
}

fn open_process(process_id: u32, access: u32) -> io::Result<OwnedHandle> {
    // SAFETY: The process ID comes from Relay ownership or a system snapshot. No handle is
    // inherited, and the returned handle is wrapped immediately for deterministic close.
    let handle = unsafe { OpenProcess(access, 0, process_id) };
    if handle.is_null() {
        Err(io::Error::last_os_error())
    } else {
        Ok(OwnedHandle(handle))
    }
}

fn process_times(handle: windows_sys::Win32::Foundation::HANDLE) -> io::Result<(u64, u64, u64)> {
    let mut creation = FILETIME::default();
    let mut exit = FILETIME::default();
    let mut kernel = FILETIME::default();
    let mut user = FILETIME::default();
    // SAFETY: All pointers reference initialized FILETIME values and `handle` is live.
    if unsafe { GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((
        filetime_value(creation),
        filetime_value(user),
        filetime_value(kernel),
    ))
}

fn process_memory(handle: HANDLE) -> io::Result<PROCESS_MEMORY_COUNTERS_EX> {
    let mut counters = PROCESS_MEMORY_COUNTERS_EX {
        cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    // SAFETY: `counters` has its documented size and `handle` is live.
    if unsafe {
        K32GetProcessMemoryInfo(
            handle,
            std::ptr::from_mut(&mut counters).cast(),
            size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        )
    } == 0
    {
        Err(io::Error::last_os_error())
    } else {
        Ok(counters)
    }
}

fn process_handle_count(handle: HANDLE) -> io::Result<u64> {
    let mut count = 0_u32;
    // SAFETY: `handle` is live and `count` is writable storage for the returned count.
    if unsafe { GetProcessHandleCount(handle, &mut count) } == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(u64::from(count))
    }
}

fn query_job_information<T: Default>(handle: HANDLE, class: i32) -> io::Result<T> {
    use windows_sys::Win32::System::JobObjects::QueryInformationJobObject;

    let mut value = T::default();
    // SAFETY: `value` is correctly sized writable storage for the requested information class.
    if unsafe {
        QueryInformationJobObject(
            handle,
            class,
            std::ptr::from_mut(&mut value).cast(),
            size_of::<T>() as u32,
            std::ptr::null_mut(),
        )
    } == 0
    {
        Err(io::Error::last_os_error())
    } else {
        Ok(value)
    }
}

pub(super) fn thread_counts() -> io::Result<HashMap<u32, u64>> {
    // SAFETY: The flags and zero process ID request a documented system thread snapshot.
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    let snapshot = OwnedHandle(snapshot);
    let mut entry = THREADENTRY32 {
        dwSize: size_of::<THREADENTRY32>() as u32,
        ..Default::default()
    };
    let mut counts = HashMap::<u32, u64>::new();
    // SAFETY: `entry` has the documented size and remains valid for the complete iteration.
    if unsafe { Thread32First(snapshot.0, &mut entry) } == 0 {
        return Err(io::Error::last_os_error());
    }
    loop {
        let count = counts.entry(entry.th32OwnerProcessID).or_default();
        *count = count.saturating_add(1);
        // SAFETY: The snapshot and output entry remain valid.
        if unsafe { Thread32Next(snapshot.0, &mut entry) } == 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(ERROR_NO_MORE_FILES as i32) {
                break;
            }
            return Err(error);
        }
    }
    Ok(counts)
}

fn filetime_value(value: FILETIME) -> u64 {
    (u64::from(value.dwHighDateTime) << 32) | u64::from(value.dwLowDateTime)
}
#[cfg(test)]
#[path = "../../../tests/unit/resource_metrics/windows_tests.rs"]
mod tests;
