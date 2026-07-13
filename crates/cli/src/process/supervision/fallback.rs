// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct-child fallback for platforms without process-group or Job Object support.

use std::process::ExitStatus;

pub(super) struct ProcessTree;

pub(super) async fn spawn(
    command: &mut tokio::process::Command,
) -> std::io::Result<(tokio::process::Child, ProcessTree)> {
    command.spawn().map(|child| (child, ProcessTree))
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

    pub(super) fn terminate(&mut self, child: &mut tokio::process::Child) -> std::io::Result<()> {
        child.start_kill()
    }
}
