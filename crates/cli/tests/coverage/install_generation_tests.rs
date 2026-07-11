// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::{File, OpenOptions};

use tempfile::tempdir;

use super::*;

#[test]
fn generation_markers_have_one_canonical_encoding() {
    let active = GenerationMarker::Active("generation-a".into());
    let retired = GenerationMarker::Retired("generation-a".into());

    assert_eq!(active.token(), "generation-a");
    assert_eq!(retired.token(), "generation-a");
    assert_eq!(active.encoded(), "generation-a\n");
    assert_eq!(retired.encoded(), "retired:generation-a\n");
    assert!(!active.is_retired());
    assert!(retired.is_retired());
}

#[test]
fn retirement_without_invalidation_only_releases_the_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let mut retirement = GenerationRetirement::acquire(&path).unwrap().unwrap();

    retirement.restore_after_rollback().unwrap();

    assert!(retirement.lock.is_none());
    assert!(!retirement.changed);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn active_generation_guard_fences_retirement_until_startup_finishes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    let generation = InstallGeneration::capture(path.clone()).unwrap();
    let guard = generation.guard_current().unwrap();

    let error = match GenerationRetirement::acquire_with_timeout(&path, Duration::from_millis(20)) {
        Err(error) => error,
        Ok(_) => panic!("retirement must wait for the active startup guard"),
    };
    assert!(error.contains("timed out waiting"), "{error}");

    drop(guard);
    assert!(
        GenerationRetirement::acquire_with_timeout(&path, Duration::from_secs(1))
            .unwrap()
            .is_some()
    );
}

#[test]
fn rollback_can_restore_with_the_original_lock_still_held() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    std::fs::write(&path, "retired:generation-a\n").unwrap();
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let mut retirement = GenerationRetirement {
        lock: Some(lock),
        path: path.clone(),
        original: GenerationMarker::Active("generation-a".into()),
        changed: true,
    };

    retirement.restore_after_rollback().unwrap();

    assert_eq!(std::fs::read_to_string(path).unwrap(), "generation-a\n");
    assert!(!retirement.changed);
}

#[test]
fn invalidation_requires_a_live_exclusive_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    let mut retirement = GenerationRetirement {
        lock: None,
        path: path.clone(),
        original: GenerationMarker::Active("generation-a".into()),
        changed: false,
    };

    let error = retirement.invalidate_for_replacement().unwrap_err();

    assert!(error.contains("is not locked"), "{error}");
    assert!(error.contains(&path.display().to_string()), "{error}");
}

#[test]
fn rollback_reports_when_an_invalidated_marker_cannot_be_reopened() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("missing-generation");
    let mut retirement = GenerationRetirement {
        lock: None,
        path: path.clone(),
        original: GenerationMarker::Active("generation-a".into()),
        changed: true,
    };

    let error = retirement.restore_after_rollback().unwrap_err();

    assert!(error.contains("failed to reopen"), "{error}");
    assert!(error.contains(&path.display().to_string()), "{error}");
}

#[test]
fn marker_replacement_preserves_the_operation_in_io_errors() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    std::fs::write(&path, "generation-a\n").unwrap();
    let mut read_only = File::open(&path).unwrap();

    let error = replace_generation_marker(
        &mut read_only,
        &path,
        &GenerationMarker::Retired("generation-a".into()),
        "invalidate",
    )
    .unwrap_err();

    assert!(error.contains("failed to invalidate"), "{error}");
    assert!(error.contains(&path.display().to_string()), "{error}");
}

#[test]
fn malformed_retirement_requires_a_token() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    std::fs::write(&path, "retired:\n").unwrap();

    let error = match GenerationRetirement::acquire(&path) {
        Err(error) => error,
        Ok(_) => panic!("a retired marker without a token must be rejected"),
    };

    assert!(error.contains("retired marker without a token"), "{error}");
}

#[test]
fn generation_creation_reports_an_invalid_parent() {
    let dir = tempdir().unwrap();
    let parent = dir.path().join("not-a-directory");
    std::fs::write(&parent, "file").unwrap();

    let error = write_new_generation(&parent.join(GENERATION_FILE_NAME)).unwrap_err();

    assert!(error.contains("failed to create"), "{error}");
}

#[test]
fn generation_creation_reports_an_unwritable_target_shape() {
    let dir = tempdir().unwrap();

    let error = write_new_generation(dir.path()).unwrap_err();

    assert!(error.contains("failed to write"), "{error}");
}

#[cfg(unix)]
#[test]
fn generation_reader_reports_directory_read_errors() {
    let dir = tempdir().unwrap();
    let directory = File::open(dir.path()).unwrap();

    let error = read_generation_marker(&directory, dir.path()).unwrap_err();

    assert!(
        error.contains("failed to read MCP install generation"),
        "{error}"
    );
}
