// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn rotating_writer_reports_missing_file_state() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("events.log");
    let mut writer = SizeRotatingFileWriter::new(path, 1, 1).unwrap();
    writer.file = None;
    writer.current_size = 1;

    let rotate_error = writer
        .write(b"next")
        .expect_err("rotation requires an open active file");
    assert_eq!(rotate_error.kind(), io::ErrorKind::Other);

    writer.current_size = 0;
    let write_error = writer
        .write(b"next")
        .expect_err("writing requires an open active file");
    assert_eq!(write_error.kind(), io::ErrorKind::Other);

    let flush_error = writer
        .flush()
        .expect_err("flushing requires an open active file");
    assert_eq!(flush_error.kind(), io::ErrorKind::Other);
}

#[test]
fn rotation_helpers_handle_relative_paths_and_missing_generations() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("relay");
    assert_eq!(rotated_log_path(&base, 3), temp.path().join("relay.3"));

    rotate_files(&base, 3).unwrap();
    assert!(!rotated_log_path(&base, 1).exists());

    create_parent_directory(Path::new("relay.log")).unwrap();
}

#[test]
fn failed_rotation_reopens_the_active_file() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("relay.log");
    let backup = rotated_log_path(&base, 1);
    fs::create_dir(&backup).unwrap();

    let mut writer = SizeRotatingFileWriter::new(base.clone(), 1, 1).unwrap();
    writer.write_all(b"first").unwrap();
    let _error = writer
        .write(b"second")
        .expect_err("an existing backup directory prevents rotation");
    assert!(writer.file.is_some());
    assert_eq!(writer.current_size, fs::metadata(base).unwrap().len());
}
