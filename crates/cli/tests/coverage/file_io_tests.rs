// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::OpenOptions;

use tempfile::tempdir;

use super::*;

#[test]
fn lock_attempts_distinguish_contention_from_errors() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("advisory.lock");
    let owner = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let waiter = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();

    assert_eq!(try_lock_exclusive(&owner).unwrap(), LockAttempt::Acquired);
    assert_eq!(try_lock_exclusive(&waiter).unwrap(), LockAttempt::Contended);
    assert_eq!(try_lock_shared(&waiter).unwrap(), LockAttempt::Contended);

    fs2::FileExt::unlock(&owner).unwrap();
    assert_eq!(try_lock_shared(&waiter).unwrap(), LockAttempt::Acquired);
    fs2::FileExt::unlock(&waiter).unwrap();
}

#[cfg(windows)]
#[test]
fn windows_lock_violation_is_normalized_as_contention() {
    let error = std::io::Error::from_raw_os_error(
        windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION as i32,
    );

    assert_eq!(
        normalize_lock_attempt(Err(error)).unwrap(),
        LockAttempt::Contended
    );
}
