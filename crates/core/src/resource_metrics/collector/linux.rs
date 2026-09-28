// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use nemo_relay_types::api::resource_metrics::{
    ResourceLimitEventKind, ResourceLimitResource, ResourceMeasurementUnit, ResourceOperatingSystem,
};

use super::{
    AcceleratorSample, CollectionTarget, EnvironmentSample, FloatSample, IntegerSample,
    LimitEventSample, ProcessSample, accelerator,
};

pub(super) const OPERATING_SYSTEM: ResourceOperatingSystem = ResourceOperatingSystem::Linux;
pub(super) const CPU_TIME_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::ClockTicks;
pub(super) const RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Pages;
pub(super) const PRIVATE_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Kibibytes;
pub(super) const PHYSICAL_FOOTPRINT_UNIT: ResourceMeasurementUnit =
    ResourceMeasurementUnit::Kibibytes;
pub(super) const VIRTUAL_MEMORY_UNIT: ResourceMeasurementUnit = ResourceMeasurementUnit::Kibibytes;
pub(super) const PEAK_RESIDENT_MEMORY_UNIT: ResourceMeasurementUnit =
    ResourceMeasurementUnit::Kibibytes;

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
        let children_path = format!("/proc/{process_id}/task/{process_id}/children");
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
    Ok(process_ids)
}

pub(super) fn process_sample(process_id: u32) -> io::Result<ProcessSample> {
    let stat = fs::read_to_string(format!("/proc/{process_id}/stat"))?;
    let mut sample = parse_stat(&stat)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid /proc process stat"))?;
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
    sample.open_file_descriptor_count = fs::read_dir(format!("/proc/{process_id}/fd"))
        .ok()
        .and_then(|entries| {
            entries
                .collect::<io::Result<Vec<_>>>()
                .ok()
                .map(|entries| entries.len() as u64)
        });
    Ok(sample)
}

pub(super) fn environment_sample(
    target: &CollectionTarget,
    process_ids: &[u32],
) -> io::Result<EnvironmentSample> {
    let Some(directory) = exclusive_cgroup_directory(target.process_id, process_ids)? else {
        return Ok(EnvironmentSample::default());
    };
    let cpu_stat = key_value_file(directory.join("cpu.stat"));
    let memory_events = key_value_file(directory.join("memory.events"));
    let process_events = key_value_file(directory.join("pids.events"));
    let cpu_pressure = pressure_totals(directory.join("cpu.pressure"));
    let memory_pressure = pressure_totals(directory.join("memory.pressure"));
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
                value,
                unit: ResourceMeasurementUnit::Microseconds,
            }),
        effective_cpu_limit: effective_cpu_limit(&directory),
        cpu_some_pressure_stall_time: cpu_pressure.0.map(microseconds),
        cpu_full_pressure_stall_time: cpu_pressure.1.map(microseconds),
        memory_limit: read_limit(directory.join("memory.max")).map(bytes),
        environment_accounted_memory: read_u64(directory.join("memory.current")).map(bytes),
        memory_some_pressure_stall_time: memory_pressure.0.map(microseconds),
        memory_full_pressure_stall_time: memory_pressure.1.map(microseconds),
        out_of_memory_event_count: out_of_memory_count.map(events),
        lifetime_process_creation_count: None,
        resource_limit_events,
    })
}

pub(super) fn accelerator_sample(process_ids: &[u32]) -> AcceleratorSample {
    accelerator::collect(process_ids)
}

fn parse_stat(stat: &str) -> Option<ProcessSample> {
    let (process_id, rest) = stat.split_once(' ')?;
    let process_id = process_id.parse().ok()?;
    let fields = stat_fields(rest)?;
    Some(ProcessSample {
        process_id,
        start_identity: fields.get(19)?.parse().ok()?,
        user_cpu_time: fields.get(11)?.parse().ok(),
        system_cpu_time: fields.get(12)?.parse().ok(),
        total_cpu_time: fields
            .get(11)?
            .parse::<u64>()
            .ok()?
            .checked_add(fields.get(12)?.parse().ok()?),
        resident_memory: fields.get(21)?.parse().ok(),
        private_memory: None,
        physical_footprint: None,
        virtual_memory: None,
        peak_resident_memory: None,
        thread_count: fields.get(17)?.parse().ok(),
        open_file_descriptor_count: None,
        windows_handle_count: None,
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

fn exclusive_cgroup_directory(process_id: u32, process_ids: &[u32]) -> io::Result<Option<PathBuf>> {
    let relative = fs::read_to_string(format!("/proc/{process_id}/cgroup"))?
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(str::to_owned);
    let Some(relative) = relative else {
        return Ok(None);
    };
    let Some(mount) = cgroup2_mount_point()? else {
        return Ok(None);
    };
    let directory = mount.join(relative.trim_start_matches('/'));
    let mut members = BTreeSet::new();
    collect_cgroup_members(&directory, &mut members)?;
    let owned = process_ids.iter().copied().collect::<BTreeSet<_>>();
    Ok((members == owned).then_some(directory))
}

fn cgroup2_mount_point() -> io::Result<Option<PathBuf>> {
    let mount_info = fs::read_to_string("/proc/self/mountinfo")?;
    Ok(mount_info.lines().find_map(|line| {
        let (before, after) = line.split_once(" - ")?;
        after.starts_with("cgroup2 ").then(|| {
            before
                .split_whitespace()
                .nth(4)
                .map(|path| PathBuf::from(path.replace("\\040", " ")))
        })?
    }))
}

fn collect_cgroup_members(directory: &Path, members: &mut BTreeSet<u32>) -> io::Result<()> {
    if let Ok(processes) = fs::read_to_string(directory.join("cgroup.procs")) {
        members.extend(
            processes
                .split_whitespace()
                .filter_map(|value| value.parse().ok()),
        );
    }
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            collect_cgroup_members(&entry.path(), members)?;
        }
    }
    Ok(())
}

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

fn effective_cpu_limit(directory: &Path) -> Option<FloatSample> {
    let quota_limit = fs::read_to_string(directory.join("cpu.max"))
        .ok()
        .and_then(|value| {
            let mut fields = value.split_whitespace();
            let quota = fields.next()?;
            let period = fields.next()?.parse::<f64>().ok()?;
            (quota != "max").then(|| quota.parse::<f64>().ok().map(|quota| quota / period))?
        });
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

fn read_limit(path: PathBuf) -> Option<u64> {
    let value = fs::read_to_string(path).ok()?;
    (value.trim() != "max").then(|| value.trim().parse().ok())?
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

fn microseconds(value: u64) -> IntegerSample {
    IntegerSample {
        value,
        unit: ResourceMeasurementUnit::Microseconds,
    }
}

fn bytes(value: u64) -> IntegerSample {
    IntegerSample {
        value,
        unit: ResourceMeasurementUnit::Bytes,
    }
}

fn events(value: u64) -> IntegerSample {
    IntegerSample {
        value,
        unit: ResourceMeasurementUnit::Events,
    }
}
