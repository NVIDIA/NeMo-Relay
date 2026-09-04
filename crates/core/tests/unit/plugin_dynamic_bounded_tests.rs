// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;

use super::*;

#[test]
fn bounded_reader_streams_regular_files_across_multiple_chunks() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("artifact.bin");
    let expected = vec![0x5a; 128 * 1024 + 17];
    fs::write(&path, &expected).unwrap();

    assert_eq!(
        read_bounded_regular_file(&path, "fixture artifact").unwrap(),
        expected
    );

    let mut streamed = Vec::new();
    let mut chunks = 0usize;
    stream_bounded_regular_file(&path, "fixture artifact", |chunk| {
        chunks += 1;
        streamed.extend_from_slice(chunk)
    })
    .unwrap();
    assert_eq!(streamed, expected);
    assert!(chunks > 1, "expected more than one chunk, got {chunks}");
}

#[test]
fn bounded_reader_rejects_missing_nonregular_and_oversized_files() {
    let temp = tempfile::tempdir().unwrap();

    let missing = read_bounded_regular_file(&temp.path().join("missing"), "fixture").unwrap_err();
    assert!(missing.contains("failed to inspect fixture"), "{missing}");

    let directory = read_bounded_regular_file(temp.path(), "fixture").unwrap_err();
    assert!(directory.contains("must be a regular file"), "{directory}");

    let oversized = temp.path().join("oversized.bin");
    let file = fs::File::create(&oversized).unwrap();
    file.set_len(MAX_DYNAMIC_PLUGIN_FILE_BYTES + 1).unwrap();
    let error = read_bounded_regular_file(&oversized, "fixture").unwrap_err();
    assert!(error.contains("exceeds the"), "{error}");
}

#[cfg(unix)]
#[test]
fn bounded_reader_rejects_symlinks_without_following_them() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("target.bin");
    let link = temp.path().join("link.bin");
    fs::write(&target, b"secret").unwrap();
    symlink(&target, &link).unwrap();

    let error = read_bounded_regular_file(&link, "fixture").unwrap_err();
    assert!(error.contains("must be a regular file"), "{error}");
}
