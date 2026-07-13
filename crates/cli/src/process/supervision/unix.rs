// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unix process-group ownership with foreground terminal job control.

use std::process::ExitStatus;
use std::time::Duration;

use super::combine_cleanup_results;

const TERMINATION_GRACE_PERIOD: Duration = Duration::from_secs(5);

pub(super) struct ProcessTree {
    process_group: i32,
    terminal: Option<TerminalForeground>,
    signals: TerminationSignals,
}

pub(super) async fn spawn(
    command: &mut tokio::process::Command,
) -> std::io::Result<(tokio::process::Child, ProcessTree)> {
    // Register before spawning so a signal cannot terminate Relay in the interval between child
    // creation and supervision. Tokio retains the OS handlers process-wide, which is appropriate:
    // a transparent run exits immediately after this child finishes.
    let signals = TerminationSignals::new()?;
    let terminal_owner = terminal_foreground_owner()?;
    command.process_group(0);
    let mut child = command.spawn()?;
    let process_group = child.id().ok_or_else(|| {
        std::io::Error::other("coding-agent process exited before Relay could supervise it")
    })? as i32;
    let terminal = match terminal_owner {
        Some(owner) => match TerminalForeground::acquire(owner, process_group) {
            Ok(terminal) => Some(terminal),
            Err(error) => {
                terminate_group(process_group, &mut child);
                let _ = child.wait().await;
                return Err(std::io::Error::new(
                    error.kind(),
                    format!(
                        "failed to give the coding agent foreground terminal ownership: {error}"
                    ),
                ));
            }
        },
        None => None,
    };
    Ok((
        child,
        ProcessTree {
            process_group,
            terminal,
            signals,
        },
    ))
}

pub(super) async fn wait(
    tree: &mut ProcessTree,
    child: &mut tokio::process::Child,
) -> std::io::Result<ExitStatus> {
    let mut termination_deadline = None;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if tree.resume_foreground_child()? {
            // SIGCONT changes the child group out of a stopped state in the kernel. Yield once so
            // a following WNOWAIT probe cannot observe the stop transition being cleared.
            tokio::task::yield_now().await;
            continue;
        }
        if tree.terminal.is_some() && child_is_stopped(tree.process_group)? {
            tree.restore_terminal()?;
            tree.stop_supervisor_group()?;
            if let Some(status) = child.try_wait()? {
                return Ok(status);
            }
            if let Some(terminal) = tree.terminal.as_mut() {
                terminal.resume_after_supervisor()?;
            }
        }
        if termination_deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
            tree.terminate(child)?;
            return child.wait().await;
        }
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(25)) => {}
            signal = tree.signals.recv() => {
                let signal = signal?;
                if termination_deadline.is_some() {
                    tree.terminate(child)?;
                    return child.wait().await;
                }
                tree.forward_signal(signal, child)?;
                termination_deadline = Some(tokio::time::Instant::now() + TERMINATION_GRACE_PERIOD);
            }
        }
    }
}

impl ProcessTree {
    pub(super) fn restore_terminal(&mut self) -> std::io::Result<()> {
        self.terminal
            .as_mut()
            .map_or(Ok(()), TerminalForeground::restore)
    }

    pub(super) fn terminate(&mut self, child: &mut tokio::process::Child) -> std::io::Result<()> {
        // SAFETY: The child was spawned with `process_group(0)`, so its PID is the process-group
        // ID. A negative PID targets the complete group and does not dereference memory.
        if unsafe { libc::kill(-self.process_group, libc::SIGKILL) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            // The group can disappear between the wrapper exit and cleanup. If the wrapper moved
            // itself out of the group, retain the direct-child guarantee as a safe fallback.
            let _ = child.start_kill();
            Ok(())
        } else {
            Err(error)
        }
    }

    fn forward_signal(
        &mut self,
        signal: i32,
        child: &mut tokio::process::Child,
    ) -> std::io::Result<()> {
        // SAFETY: The child was placed in this independently owned process group before it ran.
        if unsafe { libc::kill(-self.process_group, signal) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            // If a wrapper moved itself out of the group, retain the stronger no-orphan guarantee.
            child.start_kill()
        } else {
            Err(error)
        }
    }

    fn stop_supervisor_group(&self) -> std::io::Result<()> {
        let Some(terminal) = &self.terminal else {
            return Ok(());
        };
        // The shell owns and resumes foreground jobs by process group. Stopping only the Relay PID
        // would leave a non-exec wrapper or pipeline sibling running and prevent correct job-state
        // reporting by the shell.
        // SAFETY: `owner_process_group` was read from getpgrp for this live foreground job.
        if unsafe { libc::kill(-terminal.owner_process_group, libc::SIGSTOP) } == -1 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn resume_foreground_child(&mut self) -> std::io::Result<bool> {
        let Some(terminal) = self.terminal.as_mut() else {
            return Ok(false);
        };
        if terminal.active || terminal_process_group()? != terminal.owner_process_group {
            return Ok(false);
        }
        // A background agent does not have to read the terminal. If a shell later runs `fg` while
        // that agent is still running, Relay receives no new SIGCONT to wake a dedicated handler;
        // observe the terminal handoff here and complete it for the child group.
        terminal.activate()?;
        Ok(true)
    }
}

struct TerminalForeground {
    owner_process_group: i32,
    child_process_group: i32,
    active: bool,
}

impl TerminalForeground {
    fn acquire(owner_process_group: i32, child_process_group: i32) -> std::io::Result<Self> {
        let mut terminal = Self {
            owner_process_group,
            child_process_group,
            active: false,
        };
        terminal.activate()?;
        Ok(terminal)
    }

    fn activate(&mut self) -> std::io::Result<()> {
        set_terminal_process_group(self.child_process_group)?;
        self.active = true;
        // The child can race to read before the parent transfers the terminal and stop with
        // SIGTTIN. Continuing the whole group after the transfer closes that standard job-control
        // race and also resumes a user-stopped agent after Relay itself is continued.
        if let Err(error) = self.continue_child() {
            return combine_cleanup_results([
                ("continue foreground coding-agent group", Err(error)),
                ("restore foreground terminal", self.restore()),
            ]);
        }
        Ok(())
    }

    fn resume_after_supervisor(&mut self) -> std::io::Result<()> {
        // A shell's `fg` first returns Relay's group to the foreground and then continues it. `bg`
        // only continues Relay while the shell stays foreground. Preserve that distinction: a
        // background agent may run, or stop naturally with SIGTTIN if it attempts terminal input.
        if terminal_process_group()? == self.owner_process_group {
            self.activate()
        } else {
            self.continue_child()
        }
    }

    fn continue_child(&self) -> std::io::Result<()> {
        // SAFETY: A negative PID targets the process group created for this child.
        if unsafe { libc::kill(-self.child_process_group, libc::SIGCONT) } == -1 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error);
            }
        }
        Ok(())
    }

    fn restore(&mut self) -> std::io::Result<()> {
        if !self.active {
            return Ok(());
        }
        set_terminal_process_group(self.owner_process_group)?;
        self.active = false;
        Ok(())
    }
}

struct TerminationSignals {
    hangup: tokio::signal::unix::Signal,
    interrupt: tokio::signal::unix::Signal,
    quit: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

impl TerminationSignals {
    fn new() -> std::io::Result<Self> {
        use tokio::signal::unix::{SignalKind, signal};

        Ok(Self {
            hangup: signal(SignalKind::hangup())?,
            interrupt: signal(SignalKind::interrupt())?,
            quit: signal(SignalKind::quit())?,
            terminate: signal(SignalKind::terminate())?,
        })
    }

    async fn recv(&mut self) -> std::io::Result<i32> {
        let signal = tokio::select! {
            signal = self.hangup.recv() => signal.map(|()| libc::SIGHUP),
            signal = self.interrupt.recv() => signal.map(|()| libc::SIGINT),
            signal = self.quit.recv() => signal.map(|()| libc::SIGQUIT),
            signal = self.terminate.recv() => signal.map(|()| libc::SIGTERM),
        };
        signal.ok_or_else(|| std::io::Error::other("transparent-run signal receiver closed"))
    }
}

fn terminal_foreground_owner() -> std::io::Result<Option<i32>> {
    // SAFETY: STDIN_FILENO is a process-owned descriptor. `isatty` does not modify it.
    if unsafe { libc::isatty(libc::STDIN_FILENO) } == 0 {
        return Ok(None);
    }
    // SAFETY: `getpgrp` has no preconditions and cannot fail.
    let owner_process_group = unsafe { libc::getpgrp() };
    let foreground_process_group = terminal_process_group()?;
    if foreground_process_group != owner_process_group {
        return Err(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            "Relay is not the terminal foreground process; bring the transparent run to the foreground or redirect its standard input",
        ));
    }
    Ok(Some(owner_process_group))
}

fn terminal_process_group() -> std::io::Result<i32> {
    // SAFETY: STDIN_FILENO was verified as a terminal during process-tree preparation.
    let foreground_process_group = unsafe { libc::tcgetpgrp(libc::STDIN_FILENO) };
    if foreground_process_group == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(foreground_process_group)
    }
}

fn set_terminal_process_group(process_group: i32) -> std::io::Result<()> {
    let mut blocked = std::mem::MaybeUninit::<libc::sigset_t>::uninit();
    let mut previous = std::mem::MaybeUninit::<libc::sigset_t>::uninit();
    // SAFETY: Both pointers reference valid sigset_t storage. Blocking SIGTTOU on this thread lets
    // the background supervisor reclaim the foreground terminal without stopping itself.
    let mask_result = unsafe {
        libc::sigemptyset(blocked.as_mut_ptr());
        libc::sigaddset(blocked.as_mut_ptr(), libc::SIGTTOU);
        libc::pthread_sigmask(libc::SIG_BLOCK, blocked.as_ptr(), previous.as_mut_ptr())
    };
    if mask_result != 0 {
        return Err(std::io::Error::from_raw_os_error(mask_result));
    }
    // SAFETY: The descriptor is a controlling terminal checked during acquisition, and the target
    // is either the original foreground group or the supervised child group in the same session.
    let foreground_result = unsafe { libc::tcsetpgrp(libc::STDIN_FILENO, process_group) };
    let foreground_error = (foreground_result == -1).then(std::io::Error::last_os_error);
    // SAFETY: `previous` was initialized by the successful pthread_sigmask call above.
    let restore_result = unsafe {
        libc::pthread_sigmask(
            libc::SIG_SETMASK,
            previous.assume_init_ref(),
            std::ptr::null_mut(),
        )
    };
    combine_cleanup_results([
        (
            "set terminal foreground process group",
            foreground_error.map_or(Ok(()), Err),
        ),
        (
            "restore supervisor signal mask",
            if restore_result == 0 {
                Ok(())
            } else {
                Err(std::io::Error::from_raw_os_error(restore_result))
            },
        ),
    ])
}

fn child_is_stopped(pid: i32) -> std::io::Result<bool> {
    let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
    // SAFETY: `info` points to writable siginfo_t storage. WNOWAIT observes only stop state and
    // leaves the eventual exit status for Tokio to reap.
    let result = unsafe {
        libc::waitid(
            libc::P_PID,
            pid as _,
            info.as_mut_ptr(),
            libc::WSTOPPED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    if result == -1 {
        let error = std::io::Error::last_os_error();
        return if error.raw_os_error() == Some(libc::ECHILD) {
            Ok(false)
        } else {
            Err(error)
        };
    }
    // SAFETY: waitid initialized `info` on success; a zero si_pid means no state was available.
    Ok(unsafe { info.assume_init().si_pid() } == pid)
}

fn terminate_group(process_group: i32, child: &mut tokio::process::Child) {
    // SAFETY: The negative PID targets the child process group and does not dereference memory.
    if unsafe { libc::kill(-process_group, libc::SIGKILL) } == -1 {
        let _ = child.start_kill();
    }
}
