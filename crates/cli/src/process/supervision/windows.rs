// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Race-free Windows Job Object ownership for coding-agent wrapper trees.

use std::process::ExitStatus;

pub(super) struct ProcessTree {
    job: AgentJob,
}

pub(super) async fn spawn(
    command: &mut tokio::process::Command,
) -> std::io::Result<(tokio::process::Child, ProcessTree)> {
    let job = AgentJob::create()?;
    command.creation_flags(windows_sys::Win32::System::Threading::CREATE_SUSPENDED);
    let mut child = command.spawn()?;
    if let Err(error) = job.assign(&child) {
        abort_spawn(&job, &mut child).await;
        return Err(error);
    }
    if let Err(error) = resume_suspended_process(child.id().ok_or_else(|| {
        std::io::Error::other("coding-agent process exited before Relay could resume it")
    })?) {
        abort_spawn(&job, &mut child).await;
        return Err(error);
    }
    Ok((child, ProcessTree { job }))
}

pub(super) async fn wait(
    _tree: &mut ProcessTree,
    child: &mut tokio::process::Child,
) -> std::io::Result<ExitStatus> {
    child.wait().await
}

impl ProcessTree {
    pub(super) fn restore_terminal(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    pub(super) fn terminate(&mut self, _child: &mut tokio::process::Child) -> std::io::Result<()> {
        self.job.terminate()
    }
}

struct AgentJob {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

const WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_CLOSE: u32 = 0x0000_2000;

// SAFETY: Job Object handles can be used from any thread, and this wrapper uniquely owns it.
unsafe impl Send for AgentJob {}
// SAFETY: Windows Job Object operations are thread-safe for a live kernel handle.
unsafe impl Sync for AgentJob {}

impl AgentJob {
    fn create() -> std::io::Result<Self> {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::JobObjects::{
            CreateJobObjectW, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JobObjectExtendedLimitInformation, SetInformationJobObject,
        };

        // SAFETY: Null security attributes and name request a private, unnamed Job Object.
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if handle.is_null() {
            return Err(last_windows_error(
                "failed to create coding-agent Job Object",
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
            let error = last_windows_error("failed to configure coding-agent Job Object cleanup");
            // SAFETY: `handle` was created above and has not been transferred.
            unsafe { CloseHandle(handle) };
            return Err(error);
        }
        Ok(Self { handle })
    }

    fn assign(&self, child: &tokio::process::Child) -> std::io::Result<()> {
        use windows_sys::Win32::System::JobObjects::AssignProcessToJobObject;

        let child_handle = child.raw_handle().ok_or_else(|| {
            std::io::Error::other("coding-agent process exited before Job Object assignment")
        })?;
        // SAFETY: Both handles are live kernel handles owned by this process.
        if unsafe { AssignProcessToJobObject(self.handle, child_handle.cast()) } == 0 {
            Err(last_windows_error(&format!(
                "failed to assign coding-agent process {} to its Job Object; the current Windows Job Object may reject nested assignment",
                child.id().unwrap_or_default()
            )))
        } else {
            Ok(())
        }
    }

    fn terminate(&self) -> std::io::Result<()> {
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;

        // SAFETY: This handle owns the Job Object assigned to the coding-agent process tree.
        if unsafe { TerminateJobObject(self.handle, 1) } == 0 {
            Err(last_windows_error(
                "failed to terminate coding-agent Job Object",
            ))
        } else {
            Ok(())
        }
    }
}

async fn abort_spawn(job: &AgentJob, child: &mut tokio::process::Child) {
    // The process is still suspended when assignment or resume fails, so no descendant can escape
    // before the Job Object and direct-child fallbacks terminate it.
    let _ = job.terminate();
    let _ = child.start_kill();
    let _ = child.wait().await;
}

fn resume_suspended_process(process_id: u32) -> std::io::Result<()> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
    };
    use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};

    // SAFETY: A system-wide thread snapshot does not borrow caller memory.
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return Err(last_windows_error(
            "failed to enumerate the suspended coding-agent thread",
        ));
    }
    let mut entry = THREADENTRY32 {
        dwSize: std::mem::size_of::<THREADENTRY32>() as u32,
        ..THREADENTRY32::default()
    };
    // SAFETY: `snapshot` is live and `entry` is correctly sized writable storage.
    let mut has_entry = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while has_entry {
        if entry.th32OwnerProcessID == process_id {
            // SAFETY: The snapshot supplied this live thread identifier and only resume access is
            // requested.
            let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) };
            if thread.is_null() {
                let error =
                    last_windows_error("failed to open the suspended coding-agent primary thread");
                // SAFETY: `snapshot` is uniquely owned and closed exactly once on this path.
                unsafe { CloseHandle(snapshot) };
                return Err(error);
            }
            // CREATE_SUSPENDED starts the primary thread with a suspend count of one. Resume until
            // that count reaches zero, while rejecting a zero count that would imply the process
            // had already run before Job Object assignment.
            // SAFETY: `thread` is live and was opened with THREAD_SUSPEND_RESUME.
            let mut previous_count = unsafe { ResumeThread(thread) };
            while previous_count > 1 && previous_count != u32::MAX {
                // SAFETY: The same live thread handle remains owned by this function.
                previous_count = unsafe { ResumeThread(thread) };
            }
            let resume_error = if previous_count == u32::MAX {
                Some(last_windows_error(
                    "failed to resume the Job-owned coding-agent process",
                ))
            } else if previous_count == 0 {
                Some(std::io::Error::other(
                    "coding-agent primary thread was not suspended before Job Object assignment",
                ))
            } else {
                None
            };
            // SAFETY: Both handles are uniquely owned and closed exactly once on this path.
            unsafe {
                CloseHandle(thread);
                CloseHandle(snapshot);
            }
            return resume_error.map_or(Ok(()), Err);
        }
        // SAFETY: `snapshot` and `entry` remain valid for the next enumeration result.
        has_entry = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: `snapshot` is uniquely owned and closed exactly once on this path.
    unsafe { CloseHandle(snapshot) };
    Err(std::io::Error::other(format!(
        "could not find the suspended primary thread for coding-agent process {process_id}"
    )))
}

fn last_windows_error(context: &str) -> std::io::Error {
    let source = std::io::Error::last_os_error();
    std::io::Error::new(source.kind(), format!("{context}: {source}"))
}

impl Drop for AgentJob {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;

        // SAFETY: `handle` is uniquely owned by this wrapper and closed exactly once. The Job
        // Object's kill-on-close limit provides a final descendant-cleanup guarantee.
        unsafe { CloseHandle(self.handle) };
    }
}
