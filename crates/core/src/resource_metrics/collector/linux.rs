// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use nemo_relay_types::api::resource_metrics::{
    ResourceLimitEventKind, ResourceLimitResource, ResourceMeasurementUnit, ResourceOperatingSystem,
};

use super::{
    AcceleratorSample, CollectionTarget, EnvironmentSample, FloatSample, IntegerSample,
    LimitEventSample, ProcessSample, ProcessSampleConfig, accelerator,
    clock_ticks_to_milliseconds as ticks_to_millis,
};

pub(super) const OPERATING_SYSTEM: ResourceOperatingSystem = ResourceOperatingSystem::Linux;
pub(super) const CPU_TIME_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Milliseconds;
pub(super) const RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Kibibytes;
pub(super) const PRIVATE_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Kibibytes;
pub(super) const PHYSICAL_FOOTPRINT_UNIT: ResourceMeasurementUnit =
    ResourceMeasurementUnit::Kibibytes;
pub(super) const VIRTUAL_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Kibibytes;
pub(super) const PEAK_RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit =
    ResourceMeasurementUnit::Kibibytes;

static CLOCK_TICKS_PER_SECOND: LazyLock<Option<u64>> = LazyLock::new(|| {
    let rate = rustix::param::clock_ticks_per_second();
    (rate > 0).then_some(rate)
});

pub(super) fn process_identity(process_id: u32) -> io::Result<u64> {
    let stat = fs::read_to_string(format!("/proc/{process_id}/stat"))?;
    parse_stat(&stat)
        .map(|sample| sample.start_identity)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid /proc process stat"))
}

pub(super) fn parent_process_id(process_id: u32) -> io::Result<Option<u32>> {
    let stat = fs::read_to_string(format!("/proc/{process_id}/stat"))?;
    let fields = stat_fields(&stat)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid /proc process stat"))?;
    Ok(fields.get(1).and_then(|parent| parent.parse().ok()))
}

pub(super) fn process_tree_ids(root_process_id: u32) -> io::Result<Vec<u32>> {
    let mut process_ids = vec![root_process_id];
    let mut discovered = BTreeSet::from([root_process_id]);
    let mut next = 0;
    while next < process_ids.len() {
        let process_id = process_ids[next];
        next += 1;
        let tasks = match fs::read_dir(format!("/proc/{process_id}/task")) {
            Ok(tasks) => tasks,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        for task in tasks {
            let task = task?;
            let Some(thread_id) = task
                .file_name()
                .to_str()
                .and_then(|id| id.parse::<u32>().ok())
            else {
                continue;
            };
            let children_path = format!("/proc/{process_id}/task/{thread_id}/children");
            let children = match fs::read_to_string(children_path) {
                Ok(children) => children,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            for child in children
                .split_whitespace()
                .filter_map(|child| child.parse::<u32>().ok())
            {
                if discovered.insert(child) {
                    process_ids.push(child);
                }
            }
        }
    }
    Ok(process_ids)
}

pub(super) fn all_process_ids() -> io::Result<Vec<u32>> {
    let mut process_ids = Vec::new();
    for entry in fs::read_dir("/proc")? {
        let entry = entry?;
        if let Some(process_id) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
            .filter(|process_id| *process_id != 0)
        {
            process_ids.push(process_id);
        }
    }
    process_ids.sort_unstable();
    if process_ids.is_empty() {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no processes were visible under /proc",
        ))
    } else {
        Ok(process_ids)
    }
}

pub(super) fn process_sample(
    process_id: u32,
    config: ProcessSampleConfig,
) -> io::Result<ProcessSample> {
    let stat = fs::read_to_string(format!("/proc/{process_id}/stat"))?;
    let mut sample = parse_stat(&stat)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid /proc process stat"))?;
    if !config.cpu {
        sample.user_cpu_time = None;
        sample.system_cpu_time = None;
        sample.total_cpu_time = None;
    }
    if !config.process {
        sample.thread_count = None;
    }
    if !config.memory {
        sample.resident_memory = None;
    } else {
        let status = fs::read_to_string(format!("/proc/{process_id}/status")).ok();
        sample.virtual_memory = status
            .as_deref()
            .and_then(|status| proc_status_kibibytes(status, "VmSize"));
        sample.peak_resident_memory = status
            .as_deref()
            .and_then(|status| proc_status_kibibytes(status, "VmHWM"));
        sample.private_memory = fs::read_to_string(format!("/proc/{process_id}/smaps_rollup"))
            .ok()
            .and_then(|rollup| private_memory_kibibytes(&rollup));
    }
    if config.process {
        sample.open_file_descriptor_count = fs::read_dir(format!("/proc/{process_id}/fd"))
            .ok()
            .and_then(|entries| {
                entries
                    .collect::<io::Result<Vec<_>>>()
                    .ok()
                    .map(|entries| entries.len() as u64)
            });
    }
    if config.disk_io {
        let io = fs::read_to_string(format!("/proc/{process_id}/io")).ok();
        sample.disk_read_bytes = io.as_deref().and_then(|io| proc_io_count(io, "read_bytes"));
        sample.disk_write_bytes = io
            .as_deref()
            .and_then(|io| proc_io_count(io, "write_bytes"));
        sample.disk_read_operations = io.as_deref().and_then(|io| proc_io_count(io, "syscr"));
        sample.disk_write_operations = io.as_deref().and_then(|io| proc_io_count(io, "syscw"));
    }
    Ok(sample)
}

pub(super) fn environment_sample(
    target: &CollectionTarget,
    process_ids: &[u32],
    config: &crate::plugins::resource_metrics::config::ResourceMetricsConfig,
) -> io::Result<EnvironmentSample> {
    let Some((directory, mount_point)) =
        exclusive_cgroup_directory(target.process_id, process_ids)?
    else {
        return Ok(EnvironmentSample::default());
    };
    let cpu_stat = config
        .cpu
        .enabled
        .then(|| key_value_file(directory.join("cpu.stat")))
        .flatten();
    let memory_events = config
        .memory
        .enabled
        .then(|| key_value_file(directory.join("memory.events")))
        .flatten();
    let process_events = config
        .process
        .enabled
        .then(|| key_value_file(directory.join("pids.events")))
        .flatten();
    let cpu_pressure = if config.cpu.enabled {
        pressure_totals(directory.join("cpu.pressure"))
    } else {
        (None, None)
    };
    let memory_pressure = if config.memory.enabled {
        pressure_totals(directory.join("memory.pressure"))
    } else {
        (None, None)
    };
    let throttled_count = cpu_stat
        .as_ref()
        .and_then(|values| values.get("nr_throttled"))
        .copied();
    let memory_high_count = memory_events
        .as_ref()
        .and_then(|values| values.get("high"))
        .copied();
    let memory_maximum_count = memory_events
        .as_ref()
        .and_then(|values| values.get("max"))
        .copied();
    let out_of_memory_count = memory_events
        .as_ref()
        .and_then(|values| values.get("oom"))
        .copied();
    let process_maximum_count = process_events
        .as_ref()
        .and_then(|values| values.get("max"))
        .copied();
    let mut resource_limit_events = Vec::new();
    append_limit_event(
        &mut resource_limit_events,
        ResourceLimitResource::Cpu,
        ResourceLimitEventKind::Throttled,
        throttled_count,
    );
    append_limit_event(
        &mut resource_limit_events,
        ResourceLimitResource::Memory,
        ResourceLimitEventKind::High,
        memory_high_count,
    );
    append_limit_event(
        &mut resource_limit_events,
        ResourceLimitResource::Memory,
        ResourceLimitEventKind::Maximum,
        memory_maximum_count,
    );
    append_limit_event(
        &mut resource_limit_events,
        ResourceLimitResource::Memory,
        ResourceLimitEventKind::OutOfMemory,
        out_of_memory_count,
    );
    append_limit_event(
        &mut resource_limit_events,
        ResourceLimitResource::Processes,
        ResourceLimitEventKind::Maximum,
        process_maximum_count,
    );
    Ok(EnvironmentSample {
        cpu_throttled_time: cpu_stat
            .and_then(|values| values.get("throttled_usec").copied())
            .map(|value| IntegerSample {
                value: value / 1_000,
                unit: ResourceMeasurementUnit::Milliseconds,
            }),
        effective_cpu_limit: config
            .cpu
            .enabled
            .then(|| effective_cpu_limit(&directory, &mount_point))
            .flatten(),
        cpu_some_pressure_stall_time: cpu_pressure.0.map(milliseconds),
        cpu_full_pressure_stall_time: cpu_pressure.1.map(milliseconds),
        memory_limit: config
            .memory
            .enabled
            .then(|| effective_memory_limit(&directory, &mount_point))
            .flatten()
            .map(kibibytes),
        environment_accounted_memory: config
            .memory
            .enabled
            .then(|| read_u64(directory.join("memory.current")))
            .flatten()
            .map(kibibytes),
        memory_some_pressure_stall_time: memory_pressure.0.map(milliseconds),
        memory_full_pressure_stall_time: memory_pressure.1.map(milliseconds),
        out_of_memory_event_count: out_of_memory_count.map(events),
        lifetime_process_creation_count: None,
        resource_limit_events,
    })
}

pub(super) fn global_environment_sample(
    config: &crate::plugins::resource_metrics::config::ResourceMetricsConfig,
) -> io::Result<EnvironmentSample> {
    let cpu_pressure = if config.cpu.enabled {
        pressure_totals(PathBuf::from("/proc/pressure/cpu"))
    } else {
        (None, None)
    };
    let memory_pressure = if config.memory.enabled {
        pressure_totals(PathBuf::from("/proc/pressure/memory"))
    } else {
        (None, None)
    };
    Ok(EnvironmentSample {
        cpu_some_pressure_stall_time: cpu_pressure.0.map(milliseconds),
        cpu_full_pressure_stall_time: cpu_pressure.1.map(milliseconds),
        memory_some_pressure_stall_time: memory_pressure.0.map(milliseconds),
        memory_full_pressure_stall_time: memory_pressure.1.map(milliseconds),
        ..EnvironmentSample::default()
    })
}

pub(super) fn accelerator_sample(
    process_ids: &[u32],
    config: &crate::plugins::resource_metrics::config::ResourceMetricsGpuConfig,
    sampling_state: &mut super::SamplingState,
) -> AcceleratorSample {
    accelerator::collect(process_ids, config, &mut sampling_state.accelerator)
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

fn parse_stat(stat: &str) -> Option<ProcessSample> {
    let (process_id, rest) = stat.split_once(' ')?;
    let process_id = process_id.parse().ok()?;
    let fields = stat_fields(rest)?;
    let user_cpu_ticks = fields.get(11)?.parse::<u64>().ok()?;
    let system_cpu_ticks = fields.get(12)?.parse::<u64>().ok()?;
    Some(ProcessSample {
        process_id,
        start_identity: fields.get(19)?.parse().ok()?,
        user_cpu_time: clock_ticks_to_milliseconds(user_cpu_ticks),
        system_cpu_time: clock_ticks_to_milliseconds(system_cpu_ticks),
        total_cpu_time: user_cpu_ticks
            .checked_add(system_cpu_ticks)
            .and_then(clock_ticks_to_milliseconds),
        resident_memory: fields
            .get(21)?
            .parse::<u64>()
            .ok()
            .and_then(pages_to_kibibytes),
        private_memory: None,
        physical_footprint: None,
        virtual_memory: None,
        peak_resident_memory: None,
        thread_count: fields.get(17)?.parse().ok(),
        open_file_descriptor_count: None,
        windows_handle_count: None,
        disk_read_bytes: None,
        disk_write_bytes: None,
        disk_read_operations: None,
        disk_write_operations: None,
    })
}

fn stat_fields(stat: &str) -> Option<Vec<&str>> {
    Some(stat.rsplit_once(')')?.1.split_whitespace().collect())
}

fn proc_status_kibibytes(status: &str, key: &str) -> Option<u64> {
    status.lines().find_map(|line| {
        let (candidate, value) = line.split_once(':')?;
        (candidate == key)
            .then(|| value.split_whitespace().next()?.parse().ok())
            .flatten()
    })
}

fn private_memory_kibibytes(rollup: &str) -> Option<u64> {
    let clean = proc_status_kibibytes(rollup, "Private_Clean")?;
    let dirty = proc_status_kibibytes(rollup, "Private_Dirty")?;
    clean
        .checked_add(dirty)?
        .checked_add(proc_status_kibibytes(rollup, "Private_Hugetlb").unwrap_or(0))
}

fn exclusive_cgroup_directory(
    process_id: u32,
    process_ids: &[u32],
) -> io::Result<Option<(PathBuf, PathBuf)>> {
    let relative = fs::read_to_string(format!("/proc/{process_id}/cgroup"))?
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(str::to_owned);
    let Some(relative) = relative else {
        return Ok(None);
    };
    let Some((mount_root, mount_point)) = cgroup2_mount()? else {
        return Ok(None);
    };
    let Ok(within_mount) = Path::new(&relative).strip_prefix(&mount_root) else {
        return Ok(None);
    };
    let directory = mount_point.join(within_mount);
    let owned = process_ids.iter().copied().collect::<BTreeSet<_>>();
    let mut members = BTreeSet::new();
    let matches = cgroup_members_match_owned(&directory, &owned, &mut members)?;
    Ok((matches && members == owned).then_some((directory, mount_point)))
}

fn cgroup2_mount() -> io::Result<Option<(PathBuf, PathBuf)>> {
    let mount_info = fs::read_to_string("/proc/self/mountinfo")?;
    Ok(mount_info.lines().find_map(|line| {
        let (before, after) = line.split_once(" - ")?;
        if !after.starts_with("cgroup2 ") {
            return None;
        }
        let mut fields = before.split_whitespace();
        let root = fields.nth(3)?;
        let point = fields.next()?;
        Some((
            PathBuf::from(root.replace("\\040", " ")),
            PathBuf::from(point.replace("\\040", " ")),
        ))
    }))
}

fn cgroup_members_match_owned(
    directory: &Path,
    owned: &BTreeSet<u32>,
    members: &mut BTreeSet<u32>,
) -> io::Result<bool> {
    let processes = fs::read_to_string(directory.join("cgroup.procs"))?;
    for process_id in processes
        .split_whitespace()
        .filter_map(|value| value.parse::<u32>().ok())
    {
        if !owned.contains(&process_id) {
            return Ok(false);
        }
        members.insert(process_id);
    }
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && !cgroup_members_match_owned(&entry.path(), owned, members)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
#[path = "../../../tests/unit/resource_metrics/linux_tests.rs"]
mod tests;

fn key_value_file(path: PathBuf) -> Option<BTreeMap<String, u64>> {
    let contents = fs::read_to_string(path).ok()?;
    Some(
        contents
            .lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                Some((fields.next()?.to_owned(), fields.next()?.parse().ok()?))
            })
            .collect(),
    )
}

fn pressure_totals(path: PathBuf) -> (Option<u64>, Option<u64>) {
    let Ok(contents) = fs::read_to_string(path) else {
        return (None, None);
    };
    let total = |kind: &str| {
        contents.lines().find_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next()? == kind)
                .then(|| fields.find_map(|field| field.strip_prefix("total=")?.parse().ok()))?
        })
    };
    (total("some"), total("full"))
}

// Ancestor limits constrain descendants even when the leaf's limit is `max`. Stop at
// the visible cgroup mount, and do not report an effective limit after a read failure.
fn cgroup_limit_files(directory: &Path, mount_point: &Path, name: &str) -> Option<Vec<String>> {
    directory.strip_prefix(mount_point).ok()?;
    let mut values = Vec::new();
    for ancestor in directory.ancestors() {
        match fs::read_to_string(ancestor.join(name)) {
            Ok(value) => values.push(value),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(_) => return None,
        }
        if ancestor == mount_point {
            return Some(values);
        }
    }
    None
}

fn effective_memory_limit(directory: &Path, mount_point: &Path) -> Option<u64> {
    let mut limit: Option<u64> = None;
    for value in cgroup_limit_files(directory, mount_point, "memory.max")? {
        if value.trim() != "max" {
            let value = value.trim().parse::<u64>().ok()?;
            limit = Some(limit.map_or(value, |previous| previous.min(value)));
        }
    }
    limit
}

fn effective_cpu_limit(directory: &Path, mount_point: &Path) -> Option<FloatSample> {
    let mut quota_limit: Option<f64> = None;
    for value in cgroup_limit_files(directory, mount_point, "cpu.max")? {
        let mut fields = value.split_whitespace();
        let quota = fields.next()?;
        let period = fields.next()?.parse::<f64>().ok()?;
        if !period.is_finite() || period <= 0.0 {
            return None;
        }
        if quota != "max" {
            let quota = quota.parse::<f64>().ok()? / period;
            if !quota.is_finite() || quota < 0.0 {
                return None;
            }
            quota_limit = Some(quota_limit.map_or(quota, |previous| previous.min(quota)));
        }
    }
    let processor_set_limit = fs::read_to_string(directory.join("cpuset.cpus.effective"))
        .ok()
        .and_then(|value| processor_set_count(value.trim()).map(|count| count as f64));
    let value = match (quota_limit, processor_set_limit) {
        (Some(quota), Some(processors)) => quota.min(processors),
        (Some(quota), None) => quota,
        (None, Some(processors)) => processors,
        (None, None) => return None,
    };
    Some(FloatSample {
        value,
        unit: ResourceMeasurementUnit::LogicalProcessors,
    })
}

fn processor_set_count(value: &str) -> Option<u64> {
    value.split(',').try_fold(0_u64, |total, segment| {
        let count = if let Some((start, end)) = segment.split_once('-') {
            let start = start.parse::<u64>().ok()?;
            let end = end.parse::<u64>().ok()?;
            end.checked_sub(start)?.checked_add(1)?
        } else {
            segment.parse::<u64>().ok()?;
            1
        };
        total.checked_add(count)
    })
}

fn read_u64(path: PathBuf) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn append_limit_event(
    events: &mut Vec<LimitEventSample>,
    resource: ResourceLimitResource,
    event: ResourceLimitEventKind,
    count: Option<u64>,
) {
    if let Some(count) = count {
        events.push(LimitEventSample {
            resource,
            event,
            count,
        });
    }
}

fn milliseconds(value: u64) -> IntegerSample {
    IntegerSample {
        value: value / 1_000,
        unit: ResourceMeasurementUnit::Milliseconds,
    }
}

fn kibibytes(value: u64) -> IntegerSample {
    IntegerSample {
        value: value / 1_024,
        unit: ResourceMeasurementUnit::Kibibytes,
    }
}

fn pages_to_kibibytes(value: u64) -> Option<u64> {
    u64::try_from((u128::from(value) * rustix::param::page_size() as u128) / 1_024).ok()
}

fn clock_ticks_to_milliseconds(value: u64) -> Option<u64> {
    (*CLOCK_TICKS_PER_SECOND).and_then(|rate| ticks_to_millis(value, rate))
}

fn proc_io_count(contents: &str, key: &str) -> Option<u64> {
    contents.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == key).then(|| value.trim().parse().ok()).flatten()
    })
}

fn events(value: u64) -> IntegerSample {
    IntegerSample {
        value,
        unit: ResourceMeasurementUnit::Events,
    }
}
