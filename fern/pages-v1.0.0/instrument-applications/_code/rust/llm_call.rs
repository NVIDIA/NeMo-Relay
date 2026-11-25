// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay::scope::{self, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeType};
use nemo_relay::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    register_subscriber(
        "llm-check",
        Arc::new(|event| {
            println!("{} {}", event.kind(), event.name());
        }),
    )?;

    let handle = scope::push_scope(
        PushScopeParams::builder()
            .name("agent-run")
            .scope_type(ScopeType::Agent)
            .attributes(ScopeAttributes::empty())
            .build(),
    )?;

    let request = LlmRequest {
        headers: Default::default(),
        content: json!({"messages": [{"role": "user", "content": "hello"}]}),
    };
    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("demo-provider")
            .request(request)
            .func(Arc::new(|request| {
                Box::pin(async move {
                    Ok(json!({
                        "text": "hello",
                        "messages": request.content["messages"]
                    }))
                })
            }))
            .parent(handle.clone())
            .model_name("demo-model")
            .build(),
    )
    .await;

    scope::pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build())?;
    flush_subscribers()?;
    let _ = deregister_subscriber("llm-check")?;
    println!("{}", result?);
    Ok(())
}
