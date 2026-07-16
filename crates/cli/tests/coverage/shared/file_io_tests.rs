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

#[test]
fn bounded_file_reads_stream_regular_content_and_reject_directories() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("payload.json");
    std::fs::write(&path, b"relay payload").unwrap();

    assert_eq!(
        crate::filesystem::bounded::read_bounded_regular_file(&path, "payload").unwrap(),
        b"relay payload"
    );

    let mut chunks = Vec::new();
    crate::filesystem::bounded::stream_bounded_regular_file(&path, "payload", |chunk| {
        chunks.extend_from_slice(chunk);
    })
    .unwrap();
    assert_eq!(chunks, b"relay payload");

    let error = crate::filesystem::bounded::read_bounded_regular_file(directory.path(), "payload")
        .unwrap_err();
    assert!(error.contains("must be a regular file"), "{error}");
}

#[test]
fn backups_and_snapshots_restore_original_or_missing_files() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("relay.toml");
    std::fs::write(&path, b"original").unwrap();

    backup(&path).unwrap();
    std::fs::write(&path, b"changed").unwrap();
    backup(&path).unwrap();
    assert_eq!(std::fs::read(backup_path(&path)).unwrap(), b"original");
    remove_backup(&path).unwrap();
    remove_backup(&path).unwrap();

    let existing = snapshot_optional_file(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    restore_file_snapshot(&existing).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"changed");

    let missing = directory.path().join("missing.toml");
    let absent = snapshot_optional_file(&missing).unwrap();
    std::fs::write(&missing, b"temporary").unwrap();
    restore_file_snapshot(&absent).unwrap();
    assert!(!missing.exists());
}

#[cfg(unix)]
#[test]
fn private_atomic_write_ignores_a_permissive_umask() {
    use std::os::unix::fs::PermissionsExt;

    struct UmaskGuard(libc::mode_t);
    impl Drop for UmaskGuard {
        fn drop(&mut self) {
            // SAFETY: Restores the process umask while the environment-test mutex is held.
            unsafe { libc::umask(self.0) };
        }
    }

    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    // SAFETY: The process-global umask is serialized by the environment-test mutex.
    let previous = unsafe { libc::umask(0) };
    let _guard = UmaskGuard(previous);
    let directory = tempdir().unwrap();
    let path = directory.path().join("secret.toml");

    atomic_write_private(&path, b"secret\n").unwrap();

    assert_eq!(
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
        0o600
    );
}

#[cfg(unix)]
#[test]
fn ordinary_atomic_write_preserves_existing_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempdir().unwrap();
    let path = directory.path().join("config.toml");
    std::fs::write(&path, b"old\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();

    atomic_write(&path, b"new\n").unwrap();

    assert_eq!(
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
        0o640
    );
}

#[cfg(windows)]
#[test]
fn private_atomic_write_does_not_inherit_a_broad_parent_dacl() {
    let directory = tempdir().unwrap();
    set_windows_dacl(directory.path(), "D:P(A;;FA;;;WD)");
    let path = directory.path().join("secret.toml");

    atomic_write_private(&path, b"old-secret\n").unwrap();
    atomic_write_private(&path, b"new-secret\n").unwrap();

    let parent = windows_sddl(directory.path());
    let file = windows_sddl(&path);
    assert!(
        parent.contains("WD"),
        "parent DACL was not broadly readable: {parent}"
    );
    assert!(file.contains("D:P"), "file DACL is not protected: {file}");
    assert!(
        file.contains("OW") || file.contains("S-1-3-4"),
        "file DACL does not grant its owner access: {file}"
    );
    assert!(
        !file.contains("WD"),
        "file inherited Everyone access: {file}"
    );
}

#[cfg(windows)]
#[test]
fn failed_windows_atomic_replacement_keeps_the_original_target() {
    use std::os::windows::fs::OpenOptionsExt;

    let directory = tempdir().unwrap();
    let path = directory.path().join("config.toml");
    std::fs::write(&path, b"original\n").unwrap();
    let held = OpenOptions::new()
        .read(true)
        .share_mode(0)
        .open(&path)
        .unwrap();

    let error = atomic_write(&path, b"replacement\n").unwrap_err();
    drop(held);

    assert!(error.contains("failed to replace"), "{error}");
    assert_eq!(std::fs::read(&path).unwrap(), b"original\n");
    assert!(std::fs::read_dir(directory.path()).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")
    }));
}

#[cfg(windows)]
fn set_windows_dacl(path: &std::path::Path, sddl: &str) {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::{
        DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR,
        SetFileSecurityW,
    };

    let sddl = windows_wide(sddl);
    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
    // SAFETY: The SDDL is NUL-terminated and the output pointer is valid.
    assert_ne!(
        unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                std::ptr::null_mut(),
            )
        },
        0,
        "{}",
        std::io::Error::last_os_error()
    );
    let path = windows_wide(path.as_os_str());
    // SAFETY: The path and descriptor are valid for the duration of the call.
    let result = unsafe {
        SetFileSecurityW(
            path.as_ptr(),
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            descriptor,
        )
    };
    // SAFETY: The descriptor was allocated by ConvertStringSecurityDescriptor... above.
    unsafe { LocalFree(descriptor.cast()) };
    assert_ne!(result, 0, "{}", std::io::Error::last_os_error());
}

#[cfg(windows)]
fn windows_sddl(path: &std::path::Path) -> String {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Authorization::{
        ConvertSecurityDescriptorToStringSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::{DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR};

    let mut descriptor = read_windows_dacl(path).unwrap();
    let mut rendered = std::ptr::null_mut();
    let mut rendered_len = 0;
    // SAFETY: The self-relative descriptor buffer is valid, and both output pointers reference
    // writable storage. The returned UTF-16 allocation is released below.
    assert_ne!(
        unsafe {
            ConvertSecurityDescriptorToStringSecurityDescriptorW(
                descriptor.as_mut_ptr().cast::<std::ffi::c_void>() as PSECURITY_DESCRIPTOR,
                SDDL_REVISION_1,
                DACL_SECURITY_INFORMATION,
                &mut rendered,
                &mut rendered_len,
            )
        },
        0,
        "{}",
        std::io::Error::last_os_error()
    );
    // SAFETY: The API returned `rendered_len` initialized UTF-16 code units.
    let value = String::from_utf16_lossy(unsafe {
        std::slice::from_raw_parts(rendered, rendered_len as usize)
    });
    // SAFETY: `rendered` was allocated by ConvertSecurityDescriptor... above.
    unsafe { LocalFree(rendered.cast()) };
    value
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
