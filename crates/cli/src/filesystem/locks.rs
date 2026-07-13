// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Nonblocking advisory file-lock primitives.

use std::fs::File;
use std::io;

use fs2::FileExt;

/// Result of one nonblocking advisory-file-lock attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LockAttempt {
    Acquired,
    Contended,
}

/// Attempt an exclusive advisory lock without waiting.
pub(crate) fn try_lock_exclusive(file: &File) -> io::Result<LockAttempt> {
    normalize_lock_attempt(FileExt::try_lock_exclusive(file))
}

/// Attempt a shared advisory lock without waiting.
pub(crate) fn try_lock_shared(file: &File) -> io::Result<LockAttempt> {
    normalize_lock_attempt(FileExt::try_lock_shared(file))
}

/// Release an advisory lock acquired through the helpers above.
pub(crate) fn unlock_file(file: &File) -> io::Result<()> {
    FileExt::unlock(file)
}

pub(crate) fn normalize_lock_attempt(result: io::Result<()>) -> io::Result<LockAttempt> {
    match result {
        Ok(()) => Ok(LockAttempt::Acquired),
        Err(error) if lock_is_contended(&error) => Ok(LockAttempt::Contended),
        Err(error) => Err(error),
    }
}

fn lock_is_contended(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::WouldBlock {
        return true;
    }
    #[cfg(windows)]
    {
        error.raw_os_error() == Some(windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION as i32)
    }
    #[cfg(not(windows))]
    {
        false
    }
}
