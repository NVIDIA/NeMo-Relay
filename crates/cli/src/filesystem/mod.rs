// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small, platform-aware filesystem primitives shared by CLI subsystems.

pub(crate) mod bounded;
mod locks;
pub(crate) mod temp;

pub(crate) use locks::{LockAttempt, try_lock_exclusive, try_lock_shared, unlock_file};

use std::fs::{self, File, OpenOptions, Permissions};
use std::io::{self, Write};
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

/// Atomically replace `path` with `bytes`, creating its parent directory when needed.
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let permissions = fs::metadata(path)
        .ok()
        .filter(|metadata| metadata.is_file())
        .map(|metadata| metadata.permissions());
    atomic_write_with_permissions(path, bytes, permissions.as_ref())
}

/// Atomically replace a secret-bearing file with owner-only access.
///
/// The restriction is applied to the temporary file at creation, before its name is visible to
/// another process. This avoids both a permissive umask on Unix and inherited broad directory
/// access-control entries on Windows.
pub(crate) fn atomic_write_private(path: &Path, bytes: &[u8]) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        atomic_write_impl(
            path,
            bytes,
            Some(&Permissions::from_mode(0o600)),
            AtomicWritePrivacy::Private,
            None,
        )
    }
    #[cfg(windows)]
    {
        atomic_write_impl(path, bytes, None, AtomicWritePrivacy::Private, None)
    }
    #[cfg(not(any(unix, windows)))]
    {
        atomic_write_impl(path, bytes, None, AtomicWritePrivacy::Standard, None)
    }
}

/// Atomically replace `path` while applying `permissions` before the new bytes become visible.
pub(crate) fn atomic_write_with_permissions(
    path: &Path,
    bytes: &[u8],
    permissions: Option<&Permissions>,
) -> Result<(), String> {
    atomic_write_impl(path, bytes, permissions, AtomicWritePrivacy::Standard, None)
}

/// Atomically restore bytes with an exact Windows discretionary access-control descriptor.
#[cfg(windows)]
pub(crate) fn atomic_write_with_windows_dacl(
    path: &Path,
    bytes: &[u8],
    dacl: &[u8],
) -> Result<(), String> {
    atomic_write_impl(path, bytes, None, AtomicWritePrivacy::Standard, Some(dacl))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AtomicWritePrivacy {
    Standard,
    Private,
}

fn atomic_write_impl(
    path: &Path,
    bytes: &[u8],
    permissions: Option<&Permissions>,
    privacy: AtomicWritePrivacy,
    windows_dacl: Option<&[u8]>,
) -> Result<(), String> {
    #[cfg(test)]
    if take_injected_atomic_write_failure(path) {
        return Err(format!(
            "failed to write {}: injected test failure",
            path.display()
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("nemo-relay");
    let tmp = path.with_file_name(format!(".{file_name}.{}.tmp", uuid::Uuid::now_v7()));
    let result = (|| {
        let mut file = open_atomic_temp(&tmp, path, permissions, privacy, windows_dacl)
            .map_err(|error| format!("failed to create {}: {error}", tmp.display()))?;
        file.write_all(bytes)
            .map_err(|error| format!("failed to write {}: {error}", tmp.display()))?;
        file.sync_all()
            .map_err(|error| format!("failed to sync {}: {error}", tmp.display()))?;
        if let Some(permissions) = permissions {
            fs::set_permissions(&tmp, permissions.clone()).map_err(|error| {
                format!("failed to set permissions on {}: {error}", tmp.display())
            })?;
        }
        drop(file);
        replace_file(&tmp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(unix)]
fn open_atomic_temp(
    tmp: &Path,
    _target: &Path,
    permissions: Option<&Permissions>,
    _privacy: AtomicWritePrivacy,
    _windows_dacl: Option<&[u8]>,
) -> io::Result<File> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    if let Some(permissions) = permissions {
        options.mode(permissions.mode() & 0o7777);
    }
    options.open(tmp)
}

#[cfg(windows)]
fn open_atomic_temp(
    tmp: &Path,
    target: &Path,
    _permissions: Option<&Permissions>,
    privacy: AtomicWritePrivacy,
    windows_dacl: Option<&[u8]>,
) -> io::Result<File> {
    if let Some(descriptor) = windows_dacl {
        return create_windows_file(tmp, descriptor.as_ptr().cast_mut().cast());
    }
    if privacy == AtomicWritePrivacy::Private {
        return create_private_windows_file(tmp);
    }
    if target.exists() {
        let mut descriptor = read_windows_dacl(target)?;
        return create_windows_file(tmp, descriptor.as_mut_ptr().cast());
    }
    OpenOptions::new().create_new(true).write(true).open(tmp)
}

#[cfg(not(any(unix, windows)))]
fn open_atomic_temp(
    tmp: &Path,
    _target: &Path,
    _permissions: Option<&Permissions>,
    _privacy: AtomicWritePrivacy,
    _windows_dacl: Option<&[u8]>,
) -> io::Result<File> {
    OpenOptions::new().create_new(true).write(true).open(tmp)
}

#[cfg(windows)]
fn create_private_windows_file(path: &Path) -> io::Result<File> {
    with_private_windows_descriptor(|descriptor| create_windows_file(path, descriptor))
}

/// Opens or creates a secret-bearing file without inheriting a broad Windows DACL.
///
/// The protected owner/System descriptor is applied by `CreateFileW` when the file is created and
/// repaired before an existing file is returned to the caller. The containing directory must be
/// protected separately before this function is called.
#[cfg(windows)]
pub(crate) fn open_private_windows_file(path: &Path) -> io::Result<File> {
    use windows_sys::Win32::Foundation::{GENERIC_READ, GENERIC_WRITE};
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_ALWAYS,
    };

    let file = with_private_windows_descriptor(|descriptor| {
        open_windows_file(
            path,
            descriptor,
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            OPEN_ALWAYS,
        )
    })?;
    protect_private_windows_path(path)?;
    Ok(file)
}

/// Applies and verifies the protected owner/System DACL used for secret-bearing Windows paths.
#[cfg(windows)]
pub(crate) fn protect_private_windows_path(path: &Path) -> io::Result<()> {
    use windows_sys::Win32::Security::{
        DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION, SetFileSecurityW,
    };

    if !windows_path_owned_by_current_user(path)? {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("{} is not owned by the current user", path.display()),
        ));
    }
    let path_wide = windows_wide(path.as_os_str());
    with_private_windows_descriptor(|descriptor| {
        // SAFETY: The path and descriptor remain valid for the duration of the call.
        if unsafe {
            SetFileSecurityW(
                path_wide.as_ptr(),
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                descriptor,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    })?;
    if !windows_path_is_private(path)? {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "failed to verify protected owner/System access on {}",
                path.display()
            ),
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn with_private_windows_descriptor<T>(
    operation: impl FnOnce(windows_sys::Win32::Security::PSECURITY_DESCRIPTOR) -> io::Result<T>,
) -> io::Result<T> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::PSECURITY_DESCRIPTOR;

    let descriptor_sddl = windows_wide("D:P(A;;FA;;;OW)(A;;FA;;;SY)");
    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
    // SAFETY: The SDDL string is NUL-terminated and `descriptor` points to writable storage. The
    // returned allocation is released with LocalFree below.
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            descriptor_sddl.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    let result = operation(descriptor);
    // SAFETY: `descriptor` was allocated by ConvertStringSecurityDescriptor... and has not been
    // freed or transferred.
    unsafe { LocalFree(descriptor.cast()) };
    result
}

#[cfg(windows)]
fn windows_path_owned_by_current_user(path: &Path) -> io::Result<bool> {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::Security::{
        EqualSid, GetSecurityDescriptorOwner, GetTokenInformation, OWNER_SECURITY_INFORMATION,
        PSID, TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    let mut descriptor = read_windows_security_descriptor(path, OWNER_SECURITY_INFORMATION)?;
    let mut owner: PSID = std::ptr::null_mut();
    let mut defaulted = 0;
    // SAFETY: The self-relative descriptor buffer is valid and both outputs point to writable
    // storage for the duration of the call.
    if unsafe {
        GetSecurityDescriptorOwner(descriptor.as_mut_ptr().cast(), &mut owner, &mut defaulted)
    } == 0
        || owner.is_null()
    {
        return Err(io::Error::last_os_error());
    }

    let mut token: HANDLE = std::ptr::null_mut();
    // SAFETY: GetCurrentProcess returns a valid pseudo-handle and `token` is writable.
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let result = (|| {
        let mut required = 0;
        // SAFETY: This sizing call intentionally supplies a null output buffer.
        unsafe { GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut required) };
        if required == 0 {
            return Err(io::Error::last_os_error());
        }
        let word = std::mem::size_of::<usize>();
        let mut buffer = vec![0_usize; (required as usize).div_ceil(word)];
        // SAFETY: The aligned buffer has at least `required` writable bytes.
        if unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                buffer.as_mut_ptr().cast(),
                required,
                &mut required,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: GetTokenInformation initialized a TOKEN_USER at the aligned buffer address.
        let user = unsafe { &*buffer.as_ptr().cast::<TOKEN_USER>() };
        // SAFETY: Both SID pointers remain valid while their backing buffers are alive.
        Ok(unsafe { EqualSid(owner, user.User.Sid) != 0 })
    })();
    // SAFETY: `token` is an owned handle returned by OpenProcessToken.
    unsafe { CloseHandle(token) };
    result
}

#[cfg(windows)]
pub(crate) fn windows_path_is_private(path: &Path) -> io::Result<bool> {
    use windows_sys::Win32::Security::{DACL_SECURITY_INFORMATION, OWNER_SECURITY_INFORMATION};

    if !windows_path_owned_by_current_user(path)? {
        return Ok(false);
    }
    let mut actual = read_windows_security_descriptor(
        path,
        OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
    )?;
    let actual = windows_dacl_sddl(actual.as_mut_ptr().cast())?;
    with_private_windows_descriptor(|expected| Ok(actual == windows_dacl_sddl(expected)?))
}

#[cfg(windows)]
fn windows_dacl_sddl(
    descriptor: windows_sys::Win32::Security::PSECURITY_DESCRIPTOR,
) -> io::Result<String> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Authorization::{
        ConvertSecurityDescriptorToStringSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::DACL_SECURITY_INFORMATION;

    let mut rendered = std::ptr::null_mut();
    let mut rendered_len = 0;
    // SAFETY: The descriptor is valid and both output pointers reference writable storage.
    if unsafe {
        ConvertSecurityDescriptorToStringSecurityDescriptorW(
            descriptor,
            SDDL_REVISION_1,
            DACL_SECURITY_INFORMATION,
            &mut rendered,
            &mut rendered_len,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: The API returned `rendered_len` initialized UTF-16 code units.
    let value = String::from_utf16_lossy(unsafe {
        std::slice::from_raw_parts(rendered, rendered_len as usize)
    })
    .trim_end_matches('\0')
    .to_string();
    // SAFETY: `rendered` was allocated by ConvertSecurityDescriptor... above.
    unsafe { LocalFree(rendered.cast()) };
    Ok(value)
}

#[cfg(windows)]
pub(crate) fn read_windows_dacl(path: &Path) -> io::Result<Vec<u8>> {
    use windows_sys::Win32::Security::DACL_SECURITY_INFORMATION;

    read_windows_security_descriptor(path, DACL_SECURITY_INFORMATION)
}

#[cfg(windows)]
fn read_windows_security_descriptor(
    path: &Path,
    information: windows_sys::Win32::Security::OBJECT_SECURITY_INFORMATION,
) -> io::Result<Vec<u8>> {
    use windows_sys::Win32::Security::GetFileSecurityW;

    let path = windows_wide(path.as_os_str());
    let mut required = 0;
    // SAFETY: This sizing call intentionally supplies a null output buffer and valid length
    // pointer, as required by GetFileSecurityW.
    unsafe {
        GetFileSecurityW(
            path.as_ptr(),
            information,
            std::ptr::null_mut(),
            0,
            &mut required,
        )
    };
    if required == 0 {
        return Err(io::Error::last_os_error());
    }
    let mut descriptor = vec![0_u8; required as usize];
    // SAFETY: The path is NUL-terminated and the allocated output buffer is `required` bytes.
    if unsafe {
        GetFileSecurityW(
            path.as_ptr(),
            information,
            descriptor.as_mut_ptr().cast(),
            required,
            &mut required,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(descriptor)
}

#[cfg(windows)]
fn create_windows_file(
    path: &Path,
    descriptor: windows_sys::Win32::Security::PSECURITY_DESCRIPTOR,
) -> io::Result<File> {
    use windows_sys::Win32::Foundation::GENERIC_WRITE;
    use windows_sys::Win32::Storage::FileSystem::CREATE_NEW;

    open_windows_file(path, descriptor, GENERIC_WRITE, 0, CREATE_NEW)
}

#[cfg(windows)]
fn open_windows_file(
    path: &Path,
    descriptor: windows_sys::Win32::Security::PSECURITY_DESCRIPTOR,
    desired_access: u32,
    share_mode: u32,
    creation_disposition: u32,
) -> io::Result<File> {
    use std::os::windows::io::FromRawHandle;
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
    use windows_sys::Win32::Storage::FileSystem::{CreateFileW, FILE_ATTRIBUTE_NORMAL};

    let path = windows_wide(path.as_os_str());
    let attributes = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: descriptor,
        bInheritHandle: 0,
    };
    // SAFETY: The path and security descriptor remain valid for the call, and a successful owned
    // handle is transferred to File.
    let handle = unsafe {
        CreateFileW(
            path.as_ptr(),
            desired_access,
            share_mode,
            &attributes,
            creation_disposition,
            FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `handle` is a newly created, valid, owned file handle.
    Ok(unsafe { File::from_raw_handle(handle) })
}

#[cfg(windows)]
fn windows_wide(value: impl AsRef<std::ffi::OsStr>) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;

    value.as_ref().encode_wide().chain(Some(0)).collect()
}

#[cfg(test)]
pub(crate) fn fail_next_atomic_write(path: &Path) {
    injected_atomic_write_failures()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(path.to_path_buf());
}

#[cfg(test)]
fn take_injected_atomic_write_failure(path: &Path) -> bool {
    injected_atomic_write_failures()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .remove(path)
}

#[cfg(test)]
fn injected_atomic_write_failures() -> &'static std::sync::Mutex<std::collections::HashSet<PathBuf>>
{
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};

    static FAILURES: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    FAILURES.get_or_init(Default::default)
}

#[cfg(not(windows))]
fn replace_file(tmp: &Path, path: &Path) -> Result<(), String> {
    fs::rename(tmp, path).map_err(|error| format!("failed to replace {}: {error}", path.display()))
}

#[cfg(windows)]
fn replace_file(tmp: &Path, path: &Path) -> Result<(), String> {
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    let tmp = windows_wide(tmp.as_os_str());
    let path_wide = windows_wide(path.as_os_str());
    // SAFETY: Both paths are NUL-terminated and remain valid for the call. The files share a
    // directory, so Windows performs one replace-existing rename without a missing-target window.
    if unsafe {
        MoveFileExW(
            tmp.as_ptr(),
            path_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    } == 0
    {
        return Err(format!(
            "failed to replace {}: {}",
            path.display(),
            io::Error::last_os_error()
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/file_io_tests.rs"]
mod tests;
