// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-platform process detachment and tree cleanup for sidecars.

#[cfg(windows)]
use std::env;
use std::process::{Child, Command};
#[cfg(windows)]
use std::sync::OnceLock;

#[cfg(windows)]
const SIDECAR_JOB_NAME_ENV: &str = "NEMO_RELAY_SIDECAR_JOB_NAME";
#[cfg(windows)]
static RETAINED_SIDECAR_JOB: OnceLock<SidecarJob> = OnceLock::new();

pub(super) struct DetachedSidecarProcess {
    child: Child,
    #[cfg(windows)]
    job: Option<SidecarJob>,
}

impl DetachedSidecarProcess {
    pub(super) fn new(child: Child, #[cfg(windows)] prepared_job: Option<SidecarJob>) -> Self {
        #[cfg(windows)]
        let job = prepared_job;
        Self {
            child,
            #[cfg(windows)]
            job,
        }
    }

    pub(super) fn id(&self) -> u32 {
        self.child.id()
    }

    pub(super) fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }

    pub(super) fn terminate(&mut self) {
        #[cfg(windows)]
        if let Some(job) = self.job.as_ref() {
            job.terminate();
        }
        terminate_sidecar_process_tree(&mut self.child);
    }

    pub(super) fn terminate_retained_descendants(&self) {
        #[cfg(windows)]
        if let Some(job) = self.job.as_ref() {
            job.terminate();
        }
    }
}

#[cfg(windows)]
pub(crate) struct SidecarJob {
    handle: windows_sys::Win32::Foundation::HANDLE,
    name: String,
}

// SAFETY: Job Object handles can be used from any thread, and this wrapper uniquely owns it.
#[cfg(windows)]
unsafe impl Send for SidecarJob {}
// SAFETY: Windows Job Object operations are thread-safe for a live kernel handle.
#[cfg(windows)]
unsafe impl Sync for SidecarJob {}

#[cfg(windows)]
impl SidecarJob {
    pub(crate) fn create() -> Result<Self, String> {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::JobObjects::{
            CreateJobObjectW, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JobObjectExtendedLimitInformation, SetInformationJobObject,
        };

        let name = format!("Local\\NeMoRelaySidecar-{}", uuid::Uuid::now_v7().simple());
        let wide = std::ffi::OsStr::new(&name)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        // SAFETY: Null security attributes select defaults and `wide` is NUL-terminated.
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), wide.as_ptr()) };
        if handle.is_null() {
            return Err(format!(
                "failed to create detached sidecar Job Object: {}",
                std::io::Error::last_os_error()
            ));
        }
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_CLOSE;
        // SAFETY: `handle` is live and `limits` is correctly sized for the requested class.
        let configured = unsafe {
            SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                std::ptr::from_ref(&limits).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if configured == 0 {
            let error = std::io::Error::last_os_error();
            // SAFETY: `handle` was created above and has not been transferred.
            unsafe { CloseHandle(handle) };
            return Err(format!(
                "failed to configure detached sidecar Job Object cleanup: {error}"
            ));
        }
        Ok(Self { handle, name })
    }

    #[cfg(test)]
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn configure_child(&self, command: &mut Command) {
        command.env(SIDECAR_JOB_NAME_ENV, &self.name);
    }

    #[cfg(test)]
    pub(crate) fn assign(&self, child: &Child) -> Result<(), String> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::AssignProcessToJobObject;

        // SAFETY: Both handles are live kernel handles owned by this process.
        let assigned =
            unsafe { AssignProcessToJobObject(self.handle, child.as_raw_handle().cast()) };
        if assigned == 0 {
            return Err(format!(
                "failed to assign detached sidecar {} to its Job Object: {}",
                child.id(),
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }

    pub(crate) fn terminate(&self) {
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;

        // SAFETY: The retained handle owns the Job Object assigned to this sidecar process tree.
        let _ = unsafe { TerminateJobObject(self.handle, 1) };
    }
}

#[cfg(windows)]
pub(crate) fn join_sidecar_job_from_env() -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{AssignProcessToJobObject, OpenJobObjectW};
    use windows_sys::Win32::System::SystemServices::{
        JOB_OBJECT_ASSIGN_PROCESS, JOB_OBJECT_TERMINATE,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let Some(name) = env::var_os(SIDECAR_JOB_NAME_ENV) else {
        return Ok(());
    };
    // SAFETY: Windows environment mutation is synchronized by the operating system. Remove the
    // private handoff value before any plugin worker can inherit it.
    unsafe { env::remove_var(SIDECAR_JOB_NAME_ENV) };
    let wide = name
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    // SAFETY: `wide` is NUL-terminated and requests only assignment/termination rights.
    let job = unsafe {
        OpenJobObjectW(
            JOB_OBJECT_ASSIGN_PROCESS | JOB_OBJECT_TERMINATE,
            0,
            wide.as_ptr(),
        )
    };
    if job.is_null() {
        return Err(format!(
            "failed to open detached sidecar Job Object; persistent bootstrap cannot guarantee process-tree cleanup: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: `job` is live and the pseudo current-process handle is valid.
    let assigned = unsafe { AssignProcessToJobObject(job, GetCurrentProcess()) };
    if assigned == 0 {
        let error = std::io::Error::last_os_error();
        // SAFETY: `job` was opened above and has not been transferred.
        unsafe { CloseHandle(job) };
        return Err(format!(
            "failed to join detached sidecar Job Object; the current Windows Job Object may reject nested assignment, so persistent bootstrap cannot guarantee process-tree cleanup: {error}"
        ));
    }
    let retained = SidecarJob {
        handle: job,
        name: name.to_string_lossy().into_owned(),
    };
    RETAINED_SIDECAR_JOB.set(retained).map_err(|_| {
        "detached sidecar Job Object was initialized more than once in one process".to_string()
    })
}

#[cfg(not(windows))]
pub(crate) fn join_sidecar_job_from_env() -> Result<(), String> {
    Ok(())
}

#[cfg(windows)]
impl Drop for SidecarJob {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;

        // SAFETY: `handle` is uniquely owned by this wrapper and closed exactly once.
        unsafe { CloseHandle(self.handle) };
    }
}

#[cfg(unix)]
pub(crate) fn terminate_sidecar_process_tree(child: &mut Child) {
    let process_group = -(child.id() as i32);
    // SAFETY: The detached sidecar calls `setsid` before exec, so its PID is also the process-group
    // ID. A negative PID targets that complete group and does not dereference memory.
    if unsafe { libc::kill(process_group, libc::SIGKILL) } == -1 {
        let _ = child.kill();
    }
    let _ = child.wait();
}

#[cfg(windows)]
pub(crate) fn terminate_sidecar_process_tree(child: &mut Child) {
    let status = Command::new("taskkill")
        .args(["/PID", &child.id().to_string(), "/T", "/F"])
        .status();
    if !status.is_ok_and(|status| status.success()) {
        let _ = child.kill();
    }
    let _ = child.wait();
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn terminate_sidecar_process_tree(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(unix)]
pub(crate) fn configure_detached_sidecar(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    // SAFETY: `setsid` is async-signal-safe and has no memory-safety preconditions. It runs in the
    // post-fork child before exec so the shared sidecar is outside the MCP client's session and
    // process group.
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
pub(crate) const WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_CLOSE: u32 = 0x0000_2000;

#[cfg(any(test, windows))]
pub(crate) fn windows_sidecar_creation_flags(
    in_job: bool,
    job_limit_flags: Option<u32>,
) -> (u32, bool) {
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
    // SAFETY: The pseudo current-process handle and null current-job handle are valid for this
    // query, and `in_job` points to writable storage for the result.
    if unsafe { IsProcessInJob(GetCurrentProcess(), std::ptr::null_mut(), &mut in_job) } == 0 {
        return (true, None);
    }
    if in_job == 0 {
        return (false, Some(0));
    }
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    // SAFETY: A null job handle queries the job associated with the current process. The buffer is
    // correctly sized and aligned for the requested information class.
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
pub(crate) fn configure_detached_sidecar(command: &mut Command) {
    use std::os::windows::process::CommandExt;

    let (in_job, limits) = current_windows_job_limits();
    let (flags, limited_lifetime) = windows_sidecar_creation_flags(in_job, limits);
    if limited_lifetime {
        eprintln!(
            "warning: the current Windows Job Object does not permit process breakaway; the shared Relay gateway lifetime is limited to the host job"
        );
    }
    command.creation_flags(flags);
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn configure_detached_sidecar(_command: &mut Command) {}

#[cfg(test)]
pub(crate) fn terminate_unready_sidecar(
    mut child: Child,
    pid_path: &std::path::Path,
    url: &str,
) -> Result<(), String> {
    match child.try_wait() {
        Ok(Some(status)) => {
            let _ = std::fs::remove_file(pid_path);
            return Err(format!(
                "nemo-relay sidecar exited before becoming ready at {url}: {status}"
            ));
        }
        Ok(None) => {}
        Err(error) => {
            let _ = std::fs::remove_file(pid_path);
            return Err(format!(
                "failed to inspect nemo-relay sidecar process: {error}"
            ));
        }
    }
    if let Err(error) = child.kill() {
        let _ = std::fs::remove_file(pid_path);
        return Err(format!(
            "nemo-relay sidecar did not become ready at {url}; failed to terminate startup process: {error}"
        ));
    }
    let _ = child.wait();
    let _ = std::fs::remove_file(pid_path);
    Err(format!(
        "nemo-relay sidecar did not become ready at {url}; terminated startup process"
    ))
}
