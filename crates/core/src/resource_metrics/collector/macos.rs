// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::io;
use std::mem::{MaybeUninit, size_of};
use std::sync::LazyLock;

use nemo_relay_types::api::resource_metrics::{
    CapacityUnit, DurationUnit, ResourceOperatingSystem,
};

use super::{
    AcceleratorSample, CollectionTarget, EnvironmentSample, ProcessSample, ProcessSampleConfig,
    accelerator_macos, bytes_to_kibibytes, mach_ticks_to_milliseconds,
};

pub(super) const OPERATING_SYSTEM: ResourceOperatingSystem = ResourceOperatingSystem::Macos;
pub(super) const CPU_TIME_UNIT: DurationUnit = DurationUnit::Milliseconds;
pub(super) const RESIDENT_MEMORY_UNIT: CapacityUnit = CapacityUnit::Kibibytes;
pub(super) const PRIVATE_MEMORY_UNIT: CapacityUnit = CapacityUnit::Kibibytes;
pub(super) const PHYSICAL_FOOTPRINT_UNIT: CapacityUnit = CapacityUnit::Kibibytes;
pub(super) const VIRTUAL_MEMORY_UNIT: CapacityUnit = CapacityUnit::Kibibytes;
pub(super) const PEAK_RESIDENT_MEMORY_UNIT: CapacityUnit = CapacityUnit::Kibibytes;

const PROC_PIDLISTFDS: i32 = 1;
const PROC_PIDTBSDINFO: i32 = 3;
const PROC_PIDTASKINFO: i32 = 4;
const PROC_ALL_PIDS: u32 = 1;
const RUSAGE_INFO_V4: i32 = 4;

#[repr(C)]
#[derive(Default)]
struct MachTimebaseInfo {
    numerator: u32,
    denominator: u32,
}

static MACH_TIMEBASE: LazyLock<Option<MachTimebaseInfo>> = LazyLock::new(|| {
    let mut timebase = MachTimebaseInfo::default();
    // SAFETY: the pointer refers to writable storage matching mach_timebase_info_data_t.
    let status = unsafe { mach_timebase_info(&mut timebase) };
    (status == 0 && timebase.numerator != 0 && timebase.denominator != 0).then_some(timebase)
});

unsafe extern "C" {
    fn __error() -> *mut i32;
    fn mach_timebase_info(timebase: *mut MachTimebaseInfo) -> i32;
}

fn cpu_time_milliseconds(ticks: u64) -> Option<u64> {
    let timebase = MACH_TIMEBASE.as_ref()?;
    mach_ticks_to_milliseconds(ticks, timebase.numerator, timebase.denominator)
}

#[repr(C)]
struct ProcBsdInfo {
    _flags: u32,
    _status: u32,
    _exit_status: u32,
    process_id: u32,
    parent_process_id: u32,
    _user_id: u32,
    _group_id: u32,
    _real_user_id: u32,
    _real_group_id: u32,
    _saved_user_id: u32,
    _saved_group_id: u32,
    _reserved: u32,
    _command: [u8; 16],
    _name: [u8; 32],
    _allocated_file_slots: u32,
    _process_group_id: u32,
    _job_control_count: u32,
    _controlling_device: u32,
    _terminal_process_group_id: u32,
    _nice: i32,
    start_time_seconds: u64,
    start_time_microseconds: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcFdInfo {
    _descriptor: i32,
    _descriptor_type: u32,
}

#[repr(C)]
struct ProcTaskInfo {
    virtual_size: u64,
    resident_size: u64,
    total_user_ticks: u64,
    total_system_ticks: u64,
    _live_threads_user_ticks: u64,
    _live_threads_system_ticks: u64,
    _policy: i32,
    _fault_count: i32,
    _page_in_count: i32,
    _copy_on_write_fault_count: i32,
    _messages_sent: i32,
    _messages_received: i32,
    _mach_system_calls: i32,
    _unix_system_calls: i32,
    _context_switches: i32,
    thread_count: i32,
    _running_thread_count: i32,
    _priority: i32,
}

#[repr(C)]
#[derive(Default)]
struct RusageInfoV4 {
    uuid: [u8; 16],
    user_time: u64,
    system_time: u64,
    package_idle_wakeups: u64,
    interrupt_wakeups: u64,
    page_ins: u64,
    wired_size: u64,
    resident_size: u64,
    physical_footprint: u64,
    process_start_absolute_time: u64,
    process_exit_absolute_time: u64,
    child_user_time: u64,
    child_system_time: u64,
    child_package_idle_wakeups: u64,
    child_interrupt_wakeups: u64,
    child_page_ins: u64,
    child_elapsed_absolute_time: u64,
    disk_bytes_read: u64,
    disk_bytes_written: u64,
    cpu_time_quality_of_service_default: u64,
    cpu_time_quality_of_service_maintenance: u64,
    cpu_time_quality_of_service_background: u64,
    cpu_time_quality_of_service_utility: u64,
    cpu_time_quality_of_service_legacy: u64,
    cpu_time_quality_of_service_user_initiated: u64,
    cpu_time_quality_of_service_user_interactive: u64,
    billed_system_time: u64,
    serviced_system_time: u64,
    logical_writes: u64,
    lifetime_max_physical_footprint: u64,
    instructions: u64,
    cycles: u64,
    billed_energy: u64,
    serviced_energy: u64,
    interval_max_physical_footprint: u64,
    runnable_time: u64,
}

#[link(name = "proc")]
unsafe extern "C" {
    fn proc_listpids(
        process_type: u32,
        type_info: u32,
        buffer: *mut c_void,
        buffer_size: i32,
    ) -> i32;
    fn proc_listchildpids(parent_process_id: i32, buffer: *mut c_void, buffer_size: i32) -> i32;
    fn proc_pidinfo(
        process_id: i32,
        flavor: i32,
        argument: u64,
        buffer: *mut c_void,
        buffer_size: i32,
    ) -> i32;
    fn proc_pid_rusage(process_id: i32, flavor: i32, buffer: *mut c_void) -> i32;
}

pub(super) fn process_identity(process_id: u32) -> io::Result<u64> {
    let info = bsd_info(process_id)?;
    Ok((info.start_time_seconds << 20) | info.start_time_microseconds)
}

pub(super) fn parent_process_id(process_id: u32) -> io::Result<Option<u32>> {
    Ok(Some(bsd_info(process_id)?.parent_process_id))
}

pub(super) fn process_tree_ids(root_process_id: u32) -> io::Result<Vec<u32>> {
    let mut process_ids = vec![root_process_id];
    let mut discovered = BTreeSet::from([root_process_id]);
    let mut next = 0;
    while next < process_ids.len() {
        let process_id = process_ids[next];
        next += 1;
        let children = match child_process_ids(process_id) {
            Ok(children) => children,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        for child_process_id in children {
            if discovered.insert(child_process_id) {
                process_ids.push(child_process_id);
            }
        }
    }
    Ok(process_ids)
}

pub(super) fn all_process_ids() -> io::Result<Vec<u32>> {
    let mut capacity = 1_024_usize;
    loop {
        let mut process_ids = vec![0_u32; capacity];
        let buffer_size =
            i32::try_from(capacity.saturating_mul(size_of::<u32>())).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "process list is too large")
            })?;
        // SAFETY: `process_ids` is writable storage with `buffer_size` bytes for the documented
        // `PROC_ALL_PIDS` result. The API writes process IDs as `pid_t` values.
        let written = unsafe {
            proc_listpids(
                PROC_ALL_PIDS,
                0,
                process_ids.as_mut_ptr().cast(),
                buffer_size,
            )
        };
        if written < 0 {
            return Err(io::Error::last_os_error());
        }
        let written_ids = (written as usize) / size_of::<u32>();
        if written_ids < capacity {
            process_ids.truncate(written_ids);
            process_ids.retain(|process_id| *process_id != 0);
            if process_ids.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "no processes were visible to the collector",
                ));
            }
            process_ids.sort_unstable();
            return Ok(process_ids);
        }
        capacity = capacity.checked_mul(2).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "process list size overflow")
        })?;
    }
}

pub(super) fn process_sample(
    process_id: u32,
    config: ProcessSampleConfig,
) -> io::Result<ProcessSample> {
    let bsd_info = bsd_info(process_id)?;
    let task_info = (config.cpu || config.memory || config.process)
        .then(|| read_process_info::<ProcTaskInfo>(process_id, PROC_PIDTASKINFO).ok())
        .flatten();
    let resource_usage = (config.memory || config.disk_io)
        .then(|| process_resource_usage(process_id).ok())
        .flatten();
    Ok(ProcessSample {
        process_id: bsd_info.process_id,
        start_identity: (bsd_info.start_time_seconds << 20) | bsd_info.start_time_microseconds,
        user_cpu_time: task_info
            .as_ref()
            .filter(|_| config.cpu)
            .and_then(|task| cpu_time_milliseconds(task.total_user_ticks)),
        system_cpu_time: task_info
            .as_ref()
            .filter(|_| config.cpu)
            .and_then(|task| cpu_time_milliseconds(task.total_system_ticks)),
        total_cpu_time: task_info.as_ref().filter(|_| config.cpu).and_then(|task| {
            task.total_user_ticks
                .checked_add(task.total_system_ticks)
                .and_then(cpu_time_milliseconds)
        }),
        resident_memory: task_info
            .as_ref()
            .filter(|_| config.memory)
            .map(|task| bytes_to_kibibytes(task.resident_size)),
        private_memory: None,
        physical_footprint: resource_usage
            .as_ref()
            .filter(|_| config.memory)
            .map(|usage| bytes_to_kibibytes(usage.physical_footprint)),
        virtual_memory: task_info
            .as_ref()
            .filter(|_| config.memory)
            .map(|task| bytes_to_kibibytes(task.virtual_size)),
        // macOS exposes peak physical footprint, not peak resident memory. Keep this field
        // unavailable rather than returning a different memory concept under the same name.
        peak_resident_memory: None,
        thread_count: task_info
            .as_ref()
            .filter(|_| config.process)
            .and_then(|task| u64::try_from(task.thread_count).ok()),
        open_file_descriptor_count: config
            .process
            .then(|| open_file_descriptor_count(process_id).ok())
            .flatten(),
        windows_handle_count: None,
        disk_read_bytes: resource_usage
            .as_ref()
            .filter(|_| config.disk_io)
            .map(|usage| usage.disk_bytes_read),
        disk_write_bytes: resource_usage
            .as_ref()
            .filter(|_| config.disk_io)
            .map(|usage| usage.disk_bytes_written),
        disk_read_operations: None,
        disk_write_operations: None,
    })
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
    process_ids: &[u32],
    config: &crate::plugins::resource_metrics::config::ResourceMetricsGpuConfig,
    sampling_state: &mut super::SamplingState,
) -> AcceleratorSample {
    accelerator_macos::collect(process_ids, config, &mut sampling_state.accelerator_macos)
}

pub(super) fn filesystem_capacity(path: &std::path::Path) -> io::Result<(u64, u64, u64)> {
    let stats = rustix::fs::statvfs(path).map_err(io::Error::from)?;
    let block_size = if stats.f_frsize != 0 {
        stats.f_frsize
    } else {
        stats.f_bsize
    };
    if block_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "filesystem reported no block size",
        ));
    }
    let capacity = |blocks: u64| {
        blocks.checked_mul(block_size).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "filesystem capacity exceeds u64",
            )
        })
    };
    Ok((
        capacity(stats.f_blocks)?,
        capacity(stats.f_bavail)?,
        capacity(stats.f_bfree)?,
    ))
}

fn process_resource_usage(process_id: u32) -> io::Result<RusageInfoV4> {
    let process_id = i32::try_from(process_id)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "process ID is too large"))?;
    let mut usage = RusageInfoV4::default();
    // SAFETY: `usage` is writable storage with the exact layout required by `RUSAGE_INFO_V4`.
    if unsafe {
        proc_pid_rusage(
            process_id,
            RUSAGE_INFO_V4,
            std::ptr::from_mut(&mut usage).cast(),
        )
    } != 0
    {
        Err(io::Error::last_os_error())
    } else {
        Ok(usage)
    }
}

fn open_file_descriptor_count(process_id: u32) -> io::Result<u64> {
    let process_id = i32::try_from(process_id)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "process ID is too large"))?;
    let mut descriptors = vec![ProcFdInfo::default(); 64];
    loop {
        let buffer_size = descriptors
            .len()
            .checked_mul(size_of::<ProcFdInfo>())
            .and_then(|bytes| i32::try_from(bytes).ok())
            .ok_or_else(|| io::Error::other("file descriptor list is too large"))?;
        // SAFETY: __error returns this thread's writable errno storage. Clear it so an
        // empty successful list can be distinguished from a failed query.
        unsafe {
            *__error() = 0;
        }
        // SAFETY: the buffer contains writable proc_fdinfo records and buffer_size is its
        // exact byte length. PROC_PIDLISTFDS returns the number of bytes written.
        let written = unsafe {
            proc_pidinfo(
                process_id,
                PROC_PIDLISTFDS,
                0,
                descriptors.as_mut_ptr().cast(),
                buffer_size,
            )
        };
        if written <= 0 {
            let error = io::Error::last_os_error();
            if written == 0 && error.raw_os_error() == Some(0) {
                return Ok(0);
            }
            return Err(error);
        }
        if written > buffer_size || !(written as usize).is_multiple_of(size_of::<ProcFdInfo>()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid file descriptor list",
            ));
        }
        if written < buffer_size {
            return Ok(written as u64 / size_of::<ProcFdInfo>() as u64);
        }
        let capacity = descriptors
            .len()
            .checked_mul(2)
            .ok_or_else(|| io::Error::other("file descriptor list size overflow"))?;
        descriptors.resize(capacity, ProcFdInfo::default());
    }
}

fn bsd_info(process_id: u32) -> io::Result<ProcBsdInfo> {
    read_process_info(process_id, PROC_PIDTBSDINFO)
}

fn read_process_info<T>(process_id: u32, flavor: i32) -> io::Result<T> {
    let process_id = i32::try_from(process_id)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "process ID is too large"))?;
    let buffer_size = i32::try_from(size_of::<T>()).expect("process info size fits in i32");
    let mut value = MaybeUninit::<T>::zeroed();
    // SAFETY: `value` points to writable storage of exactly `buffer_size` bytes. The flavor
    // determines `T`, and a value is assumed initialized only after libproc reports a full write.
    let written = unsafe {
        proc_pidinfo(
            process_id,
            flavor,
            0,
            value.as_mut_ptr().cast(),
            buffer_size,
        )
    };
    if written == 0 {
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::PermissionDenied {
            Err(error)
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "process information is unavailable",
            ))
        }
    } else if written != buffer_size {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "incomplete process information",
        ))
    } else {
        // SAFETY: A successful `proc_pidinfo` call above initialized the complete value.
        Ok(unsafe { value.assume_init() })
    }
}

fn child_process_ids(parent_process_id: u32) -> io::Result<Vec<u32>> {
    let parent_process_id = i32::try_from(parent_process_id)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "process ID is too large"))?;
    let mut child_process_ids = vec![0_i32; 64];
    loop {
        let buffer_size = i32::try_from(child_process_ids.len() * size_of::<i32>())
            .map_err(|_| io::Error::other("child process list is too large"))?;
        // SAFETY: `child_process_ids` exposes writable storage for exactly `buffer_size` bytes.
        let written = unsafe {
            proc_listchildpids(
                parent_process_id,
                child_process_ids.as_mut_ptr().cast(),
                buffer_size,
            )
        };
        if written < 0 {
            return Err(io::Error::last_os_error());
        }
        // proc_listchildpids returns a PID count, although its buffer size is in bytes.
        if (written as usize) < child_process_ids.len() {
            child_process_ids.truncate(written as usize);
            return Ok(child_process_ids
                .into_iter()
                .filter_map(|process_id| u32::try_from(process_id).ok())
                .collect());
        }
        child_process_ids.resize(child_process_ids.len() * 2, 0);
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/resource_metrics/macos_tests.rs"]
mod tests;
