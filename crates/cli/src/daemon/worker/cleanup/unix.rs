// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The existing detached launch gives each daemon worker its own Unix session.

pub(super) struct ProcessTree {
    process_group: libc::pid_t,
}

impl ProcessTree {
    pub(super) fn current() -> std::io::Result<Self> {
        // SAFETY: These calls inspect the current process without borrowing memory.
        let process = unsafe { libc::getpid() };
        let group = unsafe { libc::getpgrp() };
        let session = unsafe { libc::getsid(0) };
        if process != group || process != session {
            return Err(std::io::Error::other(
                "daemon worker cleanup requires its own detached process group and session",
            ));
        }
        // SAFETY: Signal 0 only checks permission for this verified, private group.
        if unsafe { libc::kill(-group, 0) } != 0 {
            return Err(std::io::Error::other(format!(
                "daemon worker cannot signal its cleanup process group: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self {
            process_group: group,
        })
    }

    pub(super) fn terminate(&self) -> std::io::Result<()> {
        // SAFETY: The current worker still owns this group. Its PID cannot be
        // reused while the watchdog is alive; another generation has its own group.
        if unsafe { libc::kill(-self.process_group, libc::SIGKILL) } == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}
