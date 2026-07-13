// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal cross-platform process detachment for the shared gateway.

use std::process::{Child, Command};

#[cfg(windows)]
use std::sync::Mutex;

#[cfg(windows)]
static SIDECAR_SPAWN_LOCK: Mutex<()> = Mutex::new(());

#[cfg(windows)]
struct HandleInheritanceGuard {
    handles: Vec<windows_sys::Win32::Foundation::HANDLE>,
}

#[cfg(windows)]
impl HandleInheritanceGuard {
    fn suppress(
        handles: impl IntoIterator<Item = windows_sys::Win32::Foundation::HANDLE>,
    ) -> std::io::Result<Self> {
        use windows_sys::Win32::Foundation::{
            GetHandleInformation, HANDLE_FLAG_INHERIT, INVALID_HANDLE_VALUE, SetHandleInformation,
        };

        let mut guard = Self {
            handles: Vec::new(),
        };
        for handle in handles {
            if handle.is_null() || handle == INVALID_HANDLE_VALUE || guard.handles.contains(&handle)
            {
                continue;
            }
            let mut flags = 0;
            // SAFETY: handle is live and flags is writable storage.
            if unsafe { GetHandleInformation(handle, &mut flags) } == 0 {
                return Err(std::io::Error::last_os_error());
            }
            if flags & HANDLE_FLAG_INHERIT == 0 {
                continue;
            }
            // SAFETY: The mask changes only the inheritance bit on this live handle.
            if unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0) } == 0 {
                return Err(std::io::Error::last_os_error());
            }
            guard.handles.push(handle);
        }
        Ok(guard)
    }

    fn restore(&mut self) -> std::io::Result<()> {
        use windows_sys::Win32::Foundation::{HANDLE_FLAG_INHERIT, SetHandleInformation};

        let mut failed = Vec::new();
        let mut first_error = None;
        for handle in self.handles.drain(..).rev() {
            // SAFETY: Each handle was live and inheritable when this guard suppressed it.
            if unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) }
                == 0
            {
                failed.push(handle);
                first_error.get_or_insert_with(std::io::Error::last_os_error);
            }
        }
        self.handles = failed;
        first_error.map_or(Ok(()), Err)
    }
}

#[cfg(windows)]
impl Drop for HandleInheritanceGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

#[cfg(windows)]
pub(crate) fn spawn_detached(command: &mut Command) -> std::io::Result<Child> {
    use std::os::windows::io::AsRawHandle;

    let _spawn_guard = SIDECAR_SPAWN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let standard_handles = [
        std::io::stdin().as_raw_handle().cast(),
        std::io::stdout().as_raw_handle().cast(),
        std::io::stderr().as_raw_handle().cast(),
    ];
    let mut inheritance = HandleInheritanceGuard::suppress(standard_handles)?;
    let spawned = command.spawn();
    let restored = inheritance.restore();
    match (spawned, restored) {
        (Ok(child), Ok(())) => Ok(child),
        (Err(error), Ok(())) => Err(error),
        (Ok(mut child), Err(error)) => {
            let _ = child.kill();
            let _ = child.wait();
            Err(error)
        }
        (Err(spawn_error), Err(restore_error)) => Err(std::io::Error::new(
            spawn_error.kind(),
            format!("{spawn_error}; additionally, {restore_error}"),
        )),
    }
}

#[cfg(not(windows))]
pub(crate) fn spawn_detached(command: &mut Command) -> std::io::Result<Child> {
    command.spawn()
}

#[cfg(unix)]
pub(crate) fn configure_detached(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    // SAFETY: setsid is async-signal-safe and runs in the post-fork child before exec.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

#[cfg(any(test, windows))]
pub(crate) const WINDOWS_CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
#[cfg(any(test, windows))]
pub(crate) const WINDOWS_CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
#[cfg(any(test, windows))]
pub(crate) const WINDOWS_CREATE_NO_WINDOW: u32 = 0x0800_0000;
#[cfg(any(test, windows))]
pub(crate) const WINDOWS_JOB_OBJECT_LIMIT_BREAKAWAY_OK: u32 = 0x0000_0800;
#[cfg(any(test, windows))]
pub(crate) const WINDOWS_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK: u32 = 0x0000_1000;

#[cfg(any(test, windows))]
pub(crate) fn windows_creation_flags(in_job: bool, job_limit_flags: Option<u32>) -> (u32, bool) {
    let base = WINDOWS_CREATE_NEW_PROCESS_GROUP | WINDOWS_CREATE_NO_WINDOW;
    if !in_job {
        return (base, false);
    }
    match job_limit_flags {
        Some(flags) if flags & WINDOWS_JOB_OBJECT_LIMIT_BREAKAWAY_OK != 0 => {
            (base | WINDOWS_CREATE_BREAKAWAY_FROM_JOB, false)
        }
        Some(flags) if flags & WINDOWS_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK != 0 => (base, false),
        Some(_) | None => (base, true),
    }
}

#[cfg(windows)]
fn current_windows_job_limits() -> (bool, Option<u32>) {
    use windows_sys::Win32::System::JobObjects::{
        IsProcessInJob, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        QueryInformationJobObject,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut in_job = 0;
    // SAFETY: The pseudo current-process handle and null current-job handle are valid here.
    if unsafe { IsProcessInJob(GetCurrentProcess(), std::ptr::null_mut(), &mut in_job) } == 0 {
        return (true, None);
    }
    if in_job == 0 {
        return (false, Some(0));
    }
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    // SAFETY: The output buffer matches the requested information class.
    let queried = unsafe {
        QueryInformationJobObject(
            std::ptr::null_mut(),
            JobObjectExtendedLimitInformation,
            std::ptr::from_mut(&mut limits).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            std::ptr::null_mut(),
        )
    };
    if queried == 0 {
        (true, None)
    } else {
        (true, Some(limits.BasicLimitInformation.LimitFlags))
    }
}

#[cfg(windows)]
pub(crate) fn configure_detached(command: &mut Command) {
    use std::os::windows::process::CommandExt;

    let (in_job, limits) = current_windows_job_limits();
    let (flags, limited_lifetime) = windows_creation_flags(in_job, limits);
    if limited_lifetime {
        eprintln!(
            "warning: the current Windows Job Object does not permit process breakaway; the shared Relay gateway lifetime is limited to the host job"
        );
    }
    command.creation_flags(flags);
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn configure_detached(_command: &mut Command) {}

pub(crate) fn terminate_tree(child: &mut Child) {
    #[cfg(unix)]
    {
        let process_group = -(child.id() as i32);
        // SAFETY: Detached gateways call setsid, so the child PID is the process-group ID.
        if unsafe { libc::kill(process_group, libc::SIGKILL) } == -1 {
            let _ = child.kill();
        }
    }
    #[cfg(windows)]
    {
        let status = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .status();
        if !status.is_ok_and(|status| status.success()) {
            let _ = child.kill();
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = child.kill();
    }
    let _ = child.wait();
}
