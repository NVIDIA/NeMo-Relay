// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private containment for one daemon worker and its plugin processes.

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JobObjectExtendedLimitInformation, SetInformationJobObject, TerminateJobObject,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

pub(super) struct ProcessTree {
    job: HANDLE,
}

// SAFETY: Job Object handles may be used by any thread. This wrapper owns the
// handle until Drop; callers keep the wrapper alive while operating on it.
unsafe impl Send for ProcessTree {}
// SAFETY: Windows synchronizes operations on a live Job Object handle.
unsafe impl Sync for ProcessTree {}

impl ProcessTree {
    pub(super) fn current() -> std::io::Result<Self> {
        // SAFETY: Null security attributes and name create a private, unnamed,
        // non-inheritable Job Object rather than opening a launcher-owned job.
        let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if job.is_null() {
            return Err(last_windows_error(
                "failed to create daemon worker cleanup Job Object",
            ));
        }
        let tree = Self { job };
        // No breakaway flags: plugin processes inherit this job. No kill-on-close:
        // the worker itself is a member, so releasing a healthy guard must not
        // terminate it. Only an explicit failed-cleanup decision terminates it.
        let limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        // SAFETY: The owned handle is live and the input matches the selected class.
        if unsafe {
            SetInformationJobObject(
                tree.job,
                JobObjectExtendedLimitInformation,
                std::ptr::from_ref(&limits).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        } == 0
        {
            return Err(last_windows_error(
                "failed to configure daemon worker cleanup Job Object",
            ));
        }
        // SAFETY: GetCurrentProcess is a live pseudo-handle with the required
        // access rights. Assignment precedes plugin loading and child creation.
        if unsafe { AssignProcessToJobObject(tree.job, GetCurrentProcess()) } == 0 {
            return Err(last_windows_error(
                "failed to contain daemon worker in its private cleanup Job Object; the inherited Windows job may reject nested assignment",
            ));
        }
        Ok(tree)
    }

    pub(super) fn terminate(&self) -> std::io::Result<()> {
        // SAFETY: This private job contains this worker and its inherited plugin
        // children, never the launcher's inherited job. The watchdog retains its
        // owner while invoking this kernel operation.
        if unsafe { TerminateJobObject(self.job, 1) } == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

impl Drop for ProcessTree {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns the handle. Kill-on-close is disabled,
        // so closing it does not terminate the current process on successful exit.
        unsafe { CloseHandle(self.job) };
    }
}

fn last_windows_error(context: &str) -> std::io::Error {
    let source = std::io::Error::last_os_error();
    std::io::Error::new(source.kind(), format!("{context}: {source}"))
}
