// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level fallback for cleanup that cannot safely interrupt plugin code.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[cfg(unix)]
#[path = "cleanup/unix.rs"]
mod unix;
#[cfg(windows)]
#[path = "cleanup/windows.rs"]
mod windows;
#[cfg(unix)]
use unix::ProcessTree;
#[cfg(windows)]
use windows::ProcessTree;

use super::super::common::control::DRAIN_LIFETIME_MS;

enum Signal {
    Arm(Instant),
    Complete,
}

struct TriggerState {
    sender: mpsc::Sender<Signal>,
    armed: AtomicBool,
}

#[derive(Clone)]
pub(super) struct CleanupTrigger(Arc<TriggerState>);

impl CleanupTrigger {
    pub(super) fn arm(&self) {
        let deadline = Instant::now() + Duration::from_millis(DRAIN_LIFETIME_MS);
        if !self.0.armed.swap(true, Ordering::AcqRel) {
            let _ = self.0.sender.send(Signal::Arm(deadline));
        }
    }
}

/// Only the dedicated worker executable may acquire authority to terminate itself.
pub(super) struct CleanupGuard {
    trigger: CleanupTrigger,
    tree: Arc<ProcessTree>,
    watchdog: Option<JoinHandle<()>>,
    completed: bool,
}

impl CleanupGuard {
    pub(super) fn new() -> std::io::Result<Self> {
        let tree = Arc::new(ProcessTree::current()?);
        let (sender, receiver) = mpsc::channel();
        let owned_tree = Arc::clone(&tree);
        // Create the thread before admission or plugins. Native callbacks can
        // block a runtime executor, so a Tokio timeout is not a cleanup bound.
        let watchdog = std::thread::Builder::new()
            .name("worker-cleanup".into())
            .spawn(move || {
                if let Ok(Signal::Arm(deadline)) = receiver.recv()
                    && !matches!(
                        receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())),
                        Ok(Signal::Complete)
                    )
                {
                    terminate(&owned_tree);
                }
            })?;
        Ok(Self {
            trigger: CleanupTrigger(Arc::new(TriggerState {
                sender,
                armed: AtomicBool::new(false),
            })),
            tree,
            watchdog: Some(watchdog),
            completed: false,
        })
    }

    pub(super) fn trigger(&self) -> CleanupTrigger {
        self.trigger.clone()
    }

    /// Only checked, successful teardown permits the process to leave normally.
    pub(super) fn complete(mut self) {
        let _ = self.trigger.0.sender.send(Signal::Complete);
        if let Some(watchdog) = self.watchdog.take() {
            let _ = watchdog.join();
        }
        self.completed = true;
    }
}

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        if !self.completed {
            // Error paths can intentionally retain unsafe plugin state. Exiting
            // just the host would strand its children; terminate their owner too.
            terminate(&self.tree);
        }
    }
}

fn terminate(tree: &ProcessTree) -> ! {
    // Do not take logging/plugin locks or run destructors while native code may
    // hold them. Whole-process termination does not abort/unload its callback.
    let exit_code = match tree.terminate() {
        Ok(()) => 1,
        // A distinct failure status avoids treating an OS refusal as confirmed
        // descendant cleanup. Do not risk blocking stderr I/O on this path.
        Err(_) => 70,
    };
    #[cfg(unix)]
    // SAFETY: The dedicated process is exiting after a failed cleanup. _exit
    // avoids atexit handlers that could wait on the blocked plugin callback.
    unsafe {
        libc::_exit(exit_code)
    }
    #[cfg(windows)]
    {
        // SAFETY: If job termination failed, terminate only the current worker.
        unsafe {
            windows_sys::Win32::System::Threading::TerminateProcess(
                windows_sys::Win32::System::Threading::GetCurrentProcess(),
                exit_code as u32,
            )
        };
        std::process::abort()
    }
}
