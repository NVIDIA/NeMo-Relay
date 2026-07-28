// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn rotating_writer_rotates_retains_and_reports_closed_file_errors() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("nested").join("relay.log");
    let mut writer = SizeRotatingFileWriter::new(path.clone(), 4, 2).unwrap();
    assert_eq!(writer.write(b"abcd").unwrap(), 4);
    writer.flush().unwrap();
    assert_eq!(writer.write(b"e").unwrap(), 1);
    writer.flush().unwrap();

    assert_eq!(std::fs::read(rotated_log_path(&path, 1)).unwrap(), b"abcd");
    assert_eq!(std::fs::read(&path).unwrap(), b"e");

    writer.file = None;
    assert!(writer.write(b"x").is_err());
    assert!(writer.flush().is_err());
}

#[test]
fn rotation_helpers_handle_empty_and_missing_files() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("missing.log");
    rotate_files(&path, 2).unwrap();
    assert_eq!(
        rotated_log_path(&path, 2),
        directory.path().join("missing.2.log")
    );
    create_parent_directory(std::path::Path::new("plain.log")).unwrap();
}
