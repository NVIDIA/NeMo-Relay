// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Platform-aware filesystem primitives shared by CLI subsystems.

mod atomic;
pub(crate) mod bounded;
mod locks;
mod snapshots;

#[cfg(test)]
pub(crate) use atomic::fail_next_atomic_write;
#[cfg(all(test, windows))]
pub(crate) use atomic::windows_wide;
pub(crate) use atomic::{atomic_write, atomic_write_private, atomic_write_system_readable};
#[cfg(windows)]
pub(crate) use atomic::{
    atomic_write_with_windows_dacl, open_private_windows_file, protect_private_windows_path,
    read_windows_dacl, windows_path_is_private,
};
#[cfg(all(test, windows))]
pub(crate) use locks::normalize_lock_attempt;
pub(crate) use locks::{LockAttempt, try_lock_exclusive, try_lock_shared, unlock_file};
pub(crate) use snapshots::{
    FileSnapshot, atomic_write_preserving_symlink, backup, backup_path, ensure_symlink_path,
    remove_backup, remove_file_preserving_symlink, restore_file_snapshot, snapshot_optional_file,
};

#[cfg(test)]
#[path = "../../tests/coverage/shared/file_io_tests.rs"]
mod tests;
