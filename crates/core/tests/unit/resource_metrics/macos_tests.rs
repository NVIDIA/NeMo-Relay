// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::{Child, Command};

#[derive(Default)]
struct SleepingChildren(Vec<Child>);

impl SleepingChildren {
    fn spawn(&mut self) {
        self.0
            .push(Command::new("/bin/sleep").arg("60").spawn().unwrap());
    }

    fn assert_all_are_listed(&self) {
        let ids = super::child_process_ids(std::process::id()).unwrap();
        for child in &self.0 {
            assert!(
                ids.contains(&child.id()),
                "missing child {} in {ids:?}",
                child.id()
            );
        }
    }
}

impl Drop for SleepingChildren {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[test]
fn child_process_query_keeps_one_two_and_three_children() {
    let mut children = SleepingChildren::default();
    for _ in 0..3 {
        children.spawn();
        children.assert_all_are_listed();
    }
    let tree = super::process_tree_ids(std::process::id()).unwrap();
    assert!(tree.contains(&std::process::id()));
    for child in &children.0 {
        assert!(tree.contains(&child.id()));
    }
}

#[test]
fn child_process_query_grows_when_pid_capacity_is_full() {
    let mut children = SleepingChildren::default();
    // The production query initially allocates room for 64 PIDs.
    for _ in 0..65 {
        children.spawn();
    }
    children.assert_all_are_listed();
}

#[test]
fn descriptor_count_reports_open_files_and_grows_the_query_buffer() {
    use std::io::{Read, Write};
    use std::os::unix::process::CommandExt;
    use std::process::Stdio;

    unsafe extern "C" {
        fn dup(descriptor: i32) -> i32;
    }

    let mut baseline = None;
    for extra_descriptors in [0, 100] {
        let mut command = Command::new("/bin/cat");
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        // SAFETY: the child hook only calls the async-signal-safe dup syscall.
        unsafe {
            command.pre_exec(move || {
                for _ in 0..extra_descriptors {
                    if dup(0) < 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        let mut children = SleepingChildren(vec![command.spawn().unwrap()]);
        // Wait for the child's main loop so temporary loader descriptors have closed.
        children.0[0]
            .stdin
            .as_mut()
            .unwrap()
            .write_all(b"ready\n")
            .unwrap();
        let mut ready = [0; 6];
        children.0[0]
            .stdout
            .as_mut()
            .unwrap()
            .read_exact(&mut ready)
            .unwrap();
        assert_eq!(&ready, b"ready\n");
        let count = super::open_file_descriptor_count(children.0[0].id()).unwrap();
        // Test runners and coverage tools can leave inherited descriptors open.
        // Measure those along with stdin/stdout/stderr before adding duplicates.
        if extra_descriptors == 0 {
            assert!(count >= 3);
            baseline = Some(count);
        } else {
            assert_eq!(count, baseline.unwrap() + extra_descriptors);
        }
        let sample = super::process_sample(
            children.0[0].id(),
            super::ProcessSampleConfig {
                process: true,
                cpu: false,
                memory: false,
                disk_io: false,
            },
        )
        .unwrap();
        assert_eq!(sample.open_file_descriptor_count, Some(count));
        let sample = super::process_sample(
            children.0[0].id(),
            super::ProcessSampleConfig {
                process: false,
                cpu: false,
                memory: false,
                disk_io: false,
            },
        )
        .unwrap();
        assert!(sample.open_file_descriptor_count.is_none());
    }
}
