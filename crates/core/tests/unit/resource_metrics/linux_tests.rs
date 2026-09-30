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
