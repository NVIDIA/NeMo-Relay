// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::plugins::resource_metrics::config::ResourceMetricsConfig;
use std::os::windows::io::AsRawHandle;
use std::process::{Child, Command, Stdio};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_CPU_RATE_CONTROL_ENABLE,
    JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP, JOB_OBJECT_LIMIT_JOB_MEMORY,
    JOBOBJECT_CPU_RATE_CONTROL_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JobObjectCpuRateControlInformation, JobObjectExtendedLimitInformation, SetInformationJobObject,
    TerminateJobObject,
};

struct ChildJob {
    child: Child,
    job: OwnedJobHandle,
}

impl Drop for ChildJob {
    fn drop(&mut self) {
        // SAFETY: the fixture owns this live Job Object and all of its processes.
        unsafe { TerminateJobObject(self.job.0, 0) };
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn owned_job_collection_reads_limits_processes_and_normalized_counters() {
    // SAFETY: null arguments request default security and an unnamed Job Object.
    let job = OwnedJobHandle(unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) });
    assert!(!job.0.is_null());
    let child = Command::new("cmd.exe")
        .args(["/C", "ping -n 60 127.0.0.1 >nul"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let fixture = ChildJob { child, job };
    // SAFETY: both handles are live and owned by the fixture.
    assert_ne!(
        unsafe { AssignProcessToJobObject(fixture.job.0, fixture.child.as_raw_handle()) },
        0
    );
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY;
    limits.JobMemoryLimit = 1024 * 1024 * 1024;
    let mut cpu = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION {
        ControlFlags: JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
        ..Default::default()
    };
    cpu.Anonymous.CpuRate = 5000;
    // SAFETY: the live job receives correctly sized information structures.
    unsafe {
        assert_ne!(
            SetInformationJobObject(
                fixture.job.0,
                JobObjectExtendedLimitInformation,
                std::ptr::from_ref(&limits).cast(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32
            ),
            0
        );
        assert_ne!(
            SetInformationJobObject(
                fixture.job.0,
                JobObjectCpuRateControlInformation,
                std::ptr::from_ref(&cpu).cast(),
                size_of::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>() as u32
            ),
            0
        );
    }
    let target = super::super::owned_process_tree_target_with_job_handle(
        fixture.child.id(),
        fixture.job.0 as isize,
    )
    .unwrap();
    let members = job_process_ids(target.job_handle.as_ref().unwrap()).unwrap();
    assert!(members.contains(&fixture.child.id()));
    assert_eq!(
        parent_process_id(fixture.child.id()).unwrap(),
        Some(std::process::id())
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let descendants = loop {
        let tree = process_tree_ids(fixture.child.id()).unwrap();
        if tree.len() > 1 {
            break tree;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "cmd should launch its ping child"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    assert!(descendants.contains(&fixture.child.id()));
    for pid in descendants.iter().filter(|pid| **pid != fixture.child.id()) {
        assert_eq!(parent_process_id(*pid).unwrap(), Some(fixture.child.id()));
    }
    let config = ResourceMetricsConfig::default();
    let environment = environment_sample(&target, &members, &config).unwrap();
    let memory = environment.memory_limit.unwrap();
    assert_eq!(memory.value, 1_048_576);
    assert_eq!(memory.unit, ResourceMeasurementUnit::Kibibytes);
    assert!(environment.effective_cpu_limit.unwrap().value > 0.0);
    assert!(environment.lifetime_process_creation_count.unwrap().value >= 1);
    let snapshot = super::super::collect(&target, &config).snapshot;
    assert!(snapshot.cpu.unwrap().total_time.is_some());
    assert!(snapshot.process.unwrap().windows_handle_count.is_some());
    let mut disabled = config;
    disabled.cpu.enabled = false;
    disabled.memory.enabled = false;
    disabled.process.enabled = false;
    let environment = environment_sample(&target, &members, &disabled).unwrap();
    assert!(environment.effective_cpu_limit.is_none());
    assert!(environment.memory_limit.is_none());
    assert!(environment.lifetime_process_creation_count.is_none());
    assert!(OwnedJobHandle::duplicate(0).is_err());
    assert!(process_identity(u32::MAX).is_err());
    assert!(
        filesystem_capacity(std::path::Path::new(
            "Z:\\resource-metrics-nonexistent-fixture"
        ))
        .is_err()
    );
}

#[test]
fn windows_queries_reject_invalid_process_handles_and_missing_paths() {
    let invalid = INVALID_HANDLE_VALUE;
    assert!(process_times(invalid).is_err());
    assert!(process_io_counters(invalid).is_err());
    assert!(process_memory(invalid).is_err());
    assert!(process_handle_count(invalid).is_err());
    assert!(
        query_job_information::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>(
            invalid,
            JobObjectExtendedLimitInformation,
        )
        .is_err()
    );
    assert!(open_process(u32::MAX, PROCESS_QUERY_LIMITED_INFORMATION).is_err());
    let directory = tempfile::tempdir().unwrap();
    assert!(filesystem_capacity(&directory.path().join("missing")).is_err());
}

#[test]
fn job_cpu_limits_distinguish_default_weight_based_and_min_max_control() {
    use windows_sys::Win32::System::JobObjects::{
        JOB_OBJECT_CPU_RATE_CONTROL_MIN_MAX_RATE, JOB_OBJECT_CPU_RATE_CONTROL_WEIGHT_BASED,
    };
    let job = OwnedJobHandle(unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) });
    assert!(!job.0.is_null());
    let target = CollectionTarget {
        process_id: std::process::id(),
        start_identity: process_identity(std::process::id()).unwrap(),
        measurement_scope:
            nemo_relay_types::api::resource_metrics::ResourceMeasurementScope::ProcessTree,
        job_handle: Some(std::sync::Arc::new(job)),
    };
    let handle = target.job_handle.as_ref().unwrap().0;
    let config = ResourceMetricsConfig::default();
    assert!(
        environment_sample(&target, &[], &config)
            .unwrap()
            .effective_cpu_limit
            .is_none()
    );
    for flags in [
        JOB_OBJECT_CPU_RATE_CONTROL_WEIGHT_BASED,
        JOB_OBJECT_CPU_RATE_CONTROL_MIN_MAX_RATE,
    ] {
        let disabled = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION::default();
        let mut cpu = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION {
            ControlFlags: JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | flags,
            ..Default::default()
        };
        if flags == JOB_OBJECT_CPU_RATE_CONTROL_WEIGHT_BASED {
            cpu.Anonymous.Weight = 5;
        } else {
            cpu.Anonymous.Anonymous.MinRate = 1;
            cpu.Anonymous.Anonymous.MaxRate = 5000;
        }
        // SAFETY: the fixture owns the live job and supplies correctly sized control structures.
        unsafe {
            assert_ne!(
                SetInformationJobObject(
                    handle,
                    JobObjectCpuRateControlInformation,
                    std::ptr::from_ref(&disabled).cast(),
                    size_of::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>() as u32
                ),
                0
            );
            assert_ne!(
                SetInformationJobObject(
                    handle,
                    JobObjectCpuRateControlInformation,
                    std::ptr::from_ref(&cpu).cast(),
                    size_of::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>() as u32
                ),
                0
            );
        }
        let environment = environment_sample(&target, &[], &config).unwrap();
        if flags == JOB_OBJECT_CPU_RATE_CONTROL_WEIGHT_BASED {
            assert!(environment.effective_cpu_limit.is_none());
        } else {
            let limit = environment.effective_cpu_limit.unwrap();
            let processors = unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) };
            assert_eq!(limit.value, f64::from(processors) / 2.0);
            assert_eq!(limit.unit, ResourceMeasurementUnit::LogicalProcessors);
        }
    }
}
