// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::io;
use std::mem::{MaybeUninit, size_of};

use nemo_relay_types::api::resource_metrics::{ResourceMeasurementUnit, ResourceOperatingSystem};

use super::{AcceleratorSample, CollectionTarget, EnvironmentSample, ProcessSample};

pub(super) const OPERATING_SYSTEM: ResourceOperatingSystem = ResourceOperatingSystem::Macos;
pub(super) const CPU_TIME_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Nanoseconds;
pub(super) const RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Bytes;
pub(super) const PRIVATE_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Bytes;
pub(super) const PHYSICAL_FOOTPRINT_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Bytes;
pub(super) const VIRTUAL_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Bytes;
pub(super) const PEAK_RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit =
    ResourceMeasurementUnit::Bytes;

const PROC_PIDTBSDINFO: i32 = 3;
const PROC_PIDTASKINFO: i32 = 4;
const RUSAGE_INFO_V4: i32 = 4;

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
    open_file_count: u32,
    _process_group_id: u32,
    _job_control_count: u32,
    _controlling_device: u32,
    _terminal_process_group_id: u32,
    _nice: i32,
    start_time_seconds: u64,
    start_time_microseconds: u64,
}

#[repr(C)]
struct ProcTaskInfo {
    virtual_size: u64,
    resident_size: u64,
    total_user_nanoseconds: u64,
    total_system_nanoseconds: u64,
    _live_threads_user_nanoseconds: u64,
    _live_threads_system_nanoseconds: u64,
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
        for child_process_id in child_process_ids(process_id)? {
            if discovered.insert(child_process_id) {
                process_ids.push(child_process_id);
            }
        }
    }
    Ok(process_ids)
}

pub(super) fn process_sample(process_id: u32) -> io::Result<ProcessSample> {
    let bsd_info = bsd_info(process_id)?;
    let task_info = read_process_info::<ProcTaskInfo>(process_id, PROC_PIDTASKINFO)?;
    let resource_usage = process_resource_usage(process_id).ok();
    Ok(ProcessSample {
        process_id: bsd_info.process_id,
        start_identity: (bsd_info.start_time_seconds << 20) | bsd_info.start_time_microseconds,
        user_cpu_time: Some(task_info.total_user_nanoseconds),
        system_cpu_time: Some(task_info.total_system_nanoseconds),
        total_cpu_time: Some(
            task_info
                .total_user_nanoseconds
                .saturating_add(task_info.total_system_nanoseconds),
        ),
        resident_memory: Some(task_info.resident_size),
        private_memory: None,
        physical_footprint: resource_usage.map(|usage| usage.physical_footprint),
        virtual_memory: Some(task_info.virtual_size),
        peak_resident_memory: None,
        thread_count: u64::try_from(task_info.thread_count).ok(),
        open_file_descriptor_count: Some(u64::from(bsd_info.open_file_count)),
        windows_handle_count: None,
    })
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
        if written < buffer_size {
            child_process_ids.truncate(written as usize / size_of::<i32>());
            return Ok(child_process_ids
                .into_iter()
                .filter_map(|process_id| u32::try_from(process_id).ok())
                .collect());
        }
        child_process_ids.resize(child_process_ids.len() * 2, 0);
    }
}
