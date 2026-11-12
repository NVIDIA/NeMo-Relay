// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::context::{create_scope_stack, set_thread_scope_stack, with_task_scope_stack};
use nemo_relay::scope::{EmitMarkEventParams, event};
use serde_json::json;

/// Use an explicit scope stack across execution boundaries.
pub async fn run_example() -> Result<(), Box<dyn std::error::Error>> {
    let stack = create_scope_stack();
    with_task_scope_stack(stack.clone(), async {
        event(
            EmitMarkEventParams::builder()
                .name("started")
                .data(json!({"ok": true}))
                .build(),
        )
    })
    .await?;

    std::thread::spawn(move || {
        set_thread_scope_stack(stack);
        // NeMo Relay calls in this thread attach to the same explicit stack.
    })
    .join()
    .unwrap();
    Ok(())
}
