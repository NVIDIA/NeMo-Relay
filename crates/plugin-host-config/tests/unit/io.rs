// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::OpenOptions;

use tempfile::tempdir;

use super::*;

#[test]
fn oversized_regular_file_is_rejected_from_metadata_without_allocation() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("oversized.toml");
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .unwrap();
    file.set_len(MAX_BOUNDED_FILE_BYTES + 1).unwrap();

    let error = read_bounded_regular_file(&path, "plugin configuration file").unwrap_err();
    assert!(error.to_string().contains("exceeds"));
    assert!(error.to_string().contains("byte limit"));
}
