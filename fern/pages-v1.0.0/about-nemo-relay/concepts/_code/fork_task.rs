// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::context::{fork_scope_stack, with_task_scope_stack};

async fn spawn_worker() -> nemo_relay::Result<tokio::task::JoinHandle<()>> {
    let stack = fork_scope_stack()?;
    Ok(tokio::spawn(with_task_scope_stack(stack, async {
        // Relay work in an isolated child task.
    })))
}
