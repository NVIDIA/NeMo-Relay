// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::fs;

use super::cgroup_members_match_owned;

#[test]
fn cgroup_members_stop_at_an_unowned_process() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("cgroup.procs"), "100\n200\n").unwrap();
    fs::create_dir(root.path().join("unreadable-child")).unwrap();

    let owned = BTreeSet::from([100]);
    let mut members = BTreeSet::new();
    assert!(!cgroup_members_match_owned(root.path(), &owned, &mut members).unwrap());
    assert_eq!(members, BTreeSet::from([100]));
}

#[test]
fn cgroup_members_include_owned_descendants() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("cgroup.procs"), "100\n").unwrap();
    let child = root.path().join("child");
    fs::create_dir(&child).unwrap();
    fs::write(child.join("cgroup.procs"), "200\n").unwrap();

    let owned = BTreeSet::from([100, 200]);
    let mut members = BTreeSet::new();
    assert!(cgroup_members_match_owned(root.path(), &owned, &mut members).unwrap());
    assert_eq!(members, owned);
}

#[test]
fn effective_limits_include_ancestors_and_stop_at_the_mount() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("cpu.max"), "1000 100000").unwrap();
    fs::write(root.path().join("memory.max"), "1024").unwrap();
    let mount = root.path().join("mount");
    let parent = mount.join("parent");
    let leaf = parent.join("leaf");
    fs::create_dir_all(&leaf).unwrap();
    fs::write(mount.join("cpu.max"), "200000 100000").unwrap();
    fs::write(mount.join("memory.max"), "2147483648").unwrap();
    fs::write(parent.join("cpu.max"), "50000 100000").unwrap();
    fs::write(parent.join("memory.max"), "1073741824").unwrap();
    fs::write(leaf.join("cpu.max"), "max 100000").unwrap();
    fs::write(leaf.join("memory.max"), "max").unwrap();
    fs::write(leaf.join("cpuset.cpus.effective"), "0-7").unwrap();

    assert_eq!(
        super::effective_cpu_limit(&leaf, &mount).unwrap().value,
        0.5
    );
    assert_eq!(
        super::effective_memory_limit(&leaf, &mount),
        Some(1073741824)
    );

    // Tighter leaf limits win, and cpuset restrictions also constrain CPU capacity.
    fs::write(leaf.join("cpu.max"), "25000 100000").unwrap();
    fs::write(leaf.join("memory.max"), "536870912").unwrap();
    assert_eq!(
        super::effective_cpu_limit(&leaf, &mount).unwrap().value,
        0.25
    );
    assert_eq!(
        super::effective_memory_limit(&leaf, &mount),
        Some(536870912)
    );
    fs::write(parent.join("cpu.max"), "max 100000").unwrap();
    fs::write(leaf.join("cpu.max"), "max 100000").unwrap();
    fs::write(leaf.join("cpuset.cpus.effective"), "0").unwrap();
    assert_eq!(
        super::effective_cpu_limit(&leaf, &mount).unwrap().value,
        1.0
    );
}

#[test]
fn effective_limits_are_unavailable_for_invalid_or_unreadable_ancestors() {
    let root = tempfile::tempdir().unwrap();
    let leaf = root.path().join("leaf");
    fs::create_dir(&leaf).unwrap();
    fs::write(leaf.join("cpu.max"), "50000 100000").unwrap();
    fs::write(leaf.join("cpuset.cpus.effective"), "0-7").unwrap();
    fs::write(leaf.join("memory.max"), "1024").unwrap();
    fs::write(root.path().join("cpu.max"), "invalid").unwrap();
    fs::write(root.path().join("memory.max"), "invalid").unwrap();
    assert!(super::effective_cpu_limit(&leaf, root.path()).is_none());
    assert!(super::effective_memory_limit(&leaf, root.path()).is_none());
    fs::remove_file(root.path().join("cpu.max")).unwrap();
    fs::create_dir(root.path().join("cpu.max")).unwrap();
    fs::remove_file(root.path().join("memory.max")).unwrap();
    fs::create_dir(root.path().join("memory.max")).unwrap();
    assert!(super::effective_cpu_limit(&leaf, root.path()).is_none());
    assert!(super::effective_memory_limit(&leaf, root.path()).is_none());
    assert!(super::effective_memory_limit(root.path(), &leaf).is_none());
}
#[test]
fn cgroup_sample_reads_selected_categories_and_normalizes_units() {
    use crate::plugins::resource_metrics::config::ResourceMetricsConfig;
    use nemo_relay_types::api::resource_metrics::ResourceMeasurementUnit;
    let directory = tempfile::tempdir().unwrap();
    for (name, value) in [
        ("cpu.stat", "nr_throttled 7\nthrottled_usec 2000999\n"),
        ("cpu.max", "50000 100000"),
        ("cpuset.cpus.effective", "0-3"),
        ("memory.events", "high 2\nmax 3\noom 4\n"),
        ("memory.max", "4096"),
        ("memory.current", "2048"),
        ("pids.events", "max 5\n"),
        (
            "cpu.pressure",
            "some avg10=0 total=3000999\nfull avg10=0 total=1000999\n",
        ),
        (
            "memory.pressure",
            "some avg10=0 total=4000999\nfull avg10=0 total=2000999\n",
        ),
    ] {
        fs::write(directory.path().join(name), value).unwrap();
    }
    let sample = super::environment_sample_from_cgroup(
        directory.path(),
        directory.path(),
        &ResourceMetricsConfig::default(),
    )
    .unwrap();
    assert_eq!(sample.cpu_throttled_time.unwrap().value, 2000);
    assert_eq!(sample.effective_cpu_limit.unwrap().value, 0.5);
    assert_eq!(sample.memory_limit.unwrap().value, 4);
    let memory = sample.environment_accounted_memory.unwrap();
    assert_eq!(memory.value, 2);
    assert_eq!(memory.unit, ResourceMeasurementUnit::Kibibytes);
    assert_eq!(sample.cpu_some_pressure_stall_time.unwrap().value, 3000);
    assert_eq!(sample.memory_full_pressure_stall_time.unwrap().value, 2000);
    assert_eq!(sample.out_of_memory_event_count.unwrap().value, 4);
    assert_eq!(sample.resource_limit_events.len(), 5);
    let mut config = ResourceMetricsConfig::default();
    config.cpu.enabled = false;
    config.memory.enabled = false;
    config.process.enabled = false;
    let sample =
        super::environment_sample_from_cgroup(directory.path(), directory.path(), &config).unwrap();
    assert!(sample.cpu_throttled_time.is_none());
    assert!(sample.memory_limit.is_none());
    assert!(sample.resource_limit_events.is_empty());
}
#[test]
fn linux_status_io_and_processor_set_parsers_reject_invalid_counts() {
    assert_eq!(
        super::proc_status_kibibytes("VmSize: 4096 kB\n", "VmSize"),
        Some(4096)
    );
    assert_eq!(
        super::proc_status_kibibytes("VmSize: 4096 kB\n", "Missing"),
        None
    );
    assert_eq!(
        super::proc_status_kibibytes("VmSize: invalid kB\n", "VmSize"),
        None
    );
    assert_eq!(
        super::private_memory_kibibytes(
            "Private_Clean: 3 kB\nPrivate_Dirty: 5 kB\nPrivate_Hugetlb: 7 kB\n"
        ),
        Some(15)
    );
    assert_eq!(
        super::private_memory_kibibytes(
            "Private_Clean: 18446744073709551615 kB\nPrivate_Dirty: 1 kB\n"
        ),
        None
    );
    assert_eq!(
        super::proc_io_count("read_bytes: 1024\n", "read_bytes"),
        Some(1024)
    );
    assert_eq!(
        super::proc_io_count("read_bytes: invalid\n", "read_bytes"),
        None
    );
    for (value, expected) in [
        ("0-3,8,10-11", Some(7)),
        ("3-1", None),
        ("invalid", None),
        ("0-18446744073709551615", None),
    ] {
        assert_eq!(super::processor_set_count(value), expected);
    }
    assert!(super::parse_stat("invalid").is_none());
    assert!(super::parse_stat("123 (short process) S 1").is_none());
}

#[test]
fn process_tree_scans_skip_only_expected_exit_races() {
    use std::io;
    assert!(super::is_exit_race(&io::Error::from(
        io::ErrorKind::NotFound
    )));
    assert!(super::is_exit_race(&io::Error::from_raw_os_error(
        rustix::io::Errno::SRCH.raw_os_error(),
    )));
    assert!(!super::is_exit_race(&io::Error::from(
        io::ErrorKind::PermissionDenied
    )));
    assert!(!super::is_exit_race(&io::Error::from(
        io::ErrorKind::InvalidData
    )));
}

#[test]
fn cpu_limits_reject_invalid_quotas_and_keep_available_quota_without_cpuset() {
    let directory = tempfile::tempdir().unwrap();
    for invalid in [
        "1000 0",
        "1000 -1",
        "1000 NaN",
        "1000 inf",
        "-1 100000",
        "inf 100000",
    ] {
        fs::write(directory.path().join("cpu.max"), invalid).unwrap();
        assert!(
            super::effective_cpu_limit(directory.path(), directory.path()).is_none(),
            "{invalid}"
        );
    }
    fs::write(directory.path().join("cpu.max"), "50000 100000").unwrap();
    assert_eq!(
        super::effective_cpu_limit(directory.path(), directory.path())
            .unwrap()
            .value,
        0.5
    );
    fs::write(directory.path().join("cpu.max"), "max 100000").unwrap();
    assert!(super::effective_cpu_limit(directory.path(), directory.path()).is_none());
}
