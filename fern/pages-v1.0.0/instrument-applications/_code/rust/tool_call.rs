// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::scope::{self, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeType};
use nemo_relay::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::tool::{ToolCallExecuteParams, ToolExecutionResult, tool_call_execute};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    register_subscriber(
        "instrumentation-check",
        Arc::new(|event| {
            println!("{} {}", event.kind(), event.name());
        }),
    )?;

    let handle = scope::push_scope(
        PushScopeParams::builder()
            .name("agent-run")
            .scope_type(ScopeType::Agent)
            .attributes(ScopeAttributes::empty())
            .data(json!({"example": "instrument-tool"}))
            .build(),
    )?;

    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("search")
            .args(json!({"query": "runtime instrumentation"}))
            .func(Arc::new(|tool_arguments| {
                Box::pin(async move {
                    Ok(ToolExecutionResult::new(json!({
                        "query": tool_arguments["query"],
                        "hits": [{"title": "NeMo Relay"}]
                    })))
                })
            }))
            .parent(handle.clone())
            .build(),
    )
    .await;

    scope::pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build())?;
    flush_subscribers()?;
    let _ = deregister_subscriber("instrumentation-check")?;
    println!("{}", result?.result);
    Ok(())
}
