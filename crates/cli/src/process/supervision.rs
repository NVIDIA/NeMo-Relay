// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ownership and cleanup for coding-agent wrapper process trees.

use std::process::ExitStatus;

#[cfg(not(any(unix, windows)))]
mod fallback;
#[cfg(not(any(unix, windows)))]
use fallback as platform;
#[cfg(unix)]
mod unix;
#[cfg(unix)]
use unix as platform;
#[cfg(windows)]
mod windows;
#[cfg(windows)]
use windows as platform;

/// A coding-agent process whose complete wrapper tree is owned by Relay.
///
/// Transparent runs accept wrapper commands such as `npx codex`. Killing only the immediate
/// wrapper can leave the real agent running after its private gateway and hook configuration have
/// been removed, so the platform implementation owns and cleans up the entire tree.
pub(crate) struct SupervisedChild {
    child: tokio::process::Child,
    tree: platform::ProcessTree,
    tree_active: bool,
}

impl SupervisedChild {
    /// Spawns a command into an independently terminable process tree.
    pub(crate) async fn spawn(command: &mut tokio::process::Command) -> std::io::Result<Self> {
        command.kill_on_drop(true);
        let (child, tree) = platform::spawn(command).await?;
        Ok(Self {
            child,
            tree,
            tree_active: true,
        })
    }

    /// Waits for the wrapper and terminates any descendants it left behind.
    pub(crate) async fn wait(&mut self) -> std::io::Result<ExitStatus> {
        let status = match platform::wait(&mut self.tree, &mut self.child).await {
            Ok(status) => status,
            Err(wait_error) => return Err(self.clean_up_wait_error(wait_error).await),
        };
        let terminal_result = self.tree.restore_terminal();
        let tree_result = self.tree.terminate(&mut self.child);
        if tree_result.is_ok() {
            self.tree_active = false;
        }
        combine_cleanup_results([
            ("restore foreground terminal", terminal_result),
            ("terminate remaining coding-agent descendants", tree_result),
        ])?;
        Ok(status)
    }

    #[cfg(all(test, unix))]
    pub(super) async fn inject_wait_error_for_test(
        &mut self,
        error: std::io::Error,
    ) -> std::io::Result<ExitStatus> {
        Err(self.clean_up_wait_error(error).await)
    }

    async fn clean_up_wait_error(&mut self, error: std::io::Error) -> std::io::Error {
        let cleanup_error = self.terminate().await.err();
        let detail = cleanup_error.map_or_else(String::new, |cleanup_error| {
            format!("; additionally failed to terminate the coding-agent tree: {cleanup_error}")
        });
        std::io::Error::new(
            error.kind(),
            format!("failed while supervising the coding-agent tree: {error}{detail}"),
        )
    }

    /// Terminates and reaps the complete supervised process tree.
    pub(crate) async fn terminate(&mut self) -> std::io::Result<()> {
        let terminal_result = self.tree.restore_terminal();
        let tree_result = self.tree.terminate(&mut self.child);
        if tree_result.is_err() {
            // Preserve direct-child cleanup even if the platform tree primitive failed. The
            // original tree error remains authoritative because descendants may still be alive.
            let _ = self.child.start_kill();
        }
        let wait_result = self.child.wait().await.map(|_| ());
        if tree_result.is_ok() {
            self.tree_active = false;
        }
        combine_cleanup_results([
            ("restore foreground terminal", terminal_result),
            ("terminate coding-agent process tree", tree_result),
            ("reap coding-agent wrapper", wait_result),
        ])
    }
}

impl Drop for SupervisedChild {
    fn drop(&mut self) {
        let _ = self.tree.restore_terminal();
        if self.tree_active {
            let _ = self.tree.terminate(&mut self.child);
        }
    }
}

pub(super) fn combine_cleanup_results<const N: usize>(
    results: [(&str, std::io::Result<()>); N],
) -> std::io::Result<()> {
    let errors = results
        .into_iter()
        .filter_map(|(operation, result)| result.err().map(|error| format!("{operation}: {error}")))
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(std::io::Error::other(errors.join("; ")))
    }
}
