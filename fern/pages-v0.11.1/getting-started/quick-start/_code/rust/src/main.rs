// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::api::llm::{llm_call_execute, LlmCallExecuteParams, LlmRequest};
use nemo_relay::api::scope::{
    self, EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeType,
};
use nemo_relay::api::subscriber::flush_subscribers;
use nemo_relay::api::tool::{tool_call_execute, ToolCallExecuteParams, ToolExecutionResult};
use nemo_relay::plugin::dynamic::initialize;
use nemo_relay::plugin::PluginConfig;
use serde_json::json;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

/// Emit a local agent workflow and close its scope even if a callback fails.
async fn run_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let handle = scope::push_scope(
        PushScopeParams::builder()
            .name("demo-agent")
            .scope_type(ScopeType::Agent)
            .attributes(ScopeAttributes::empty())
            .data(json!({"binding": "rust"}))
            .build(),
    )?;

    let workflow_result: Result<(), Box<dyn std::error::Error>> = async {
        scope::event(
            EmitMarkEventParams::builder()
                .name("initialized")
                .data(json!({"ok": true}))
                .build(),
        )?;

        let tool_result = tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("emit_marker")
                .args(json!({"marker": "RELAY_QUICKSTART_TOOL_OK"}))
                .func(Arc::new(|args| Box::pin(async move {
                    Ok(ToolExecutionResult::new(json!({"marker": args["marker"]})))
                })))
                .build(),
        )
        .await?;

        let llm_result = llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("demo-provider")
                .request(LlmRequest {
                    headers: Default::default(),
                    content: json!({
                        "messages": [{
                            "role": "user",
                            "content": "RELAY_QUICKSTART_RESPONSE_OK"
                        }]
                    }),
                })
                .func(Arc::new(|_| Box::pin(async move {
                    Ok(json!({"content": "RELAY_QUICKSTART_RESPONSE_OK"}))
                })))
                .model_name("demo-model")
                .build(),
        )
        .await?;

        println!(
            "{}",
            tool_result.result["marker"]
                .as_str()
                .ok_or_else(|| io::Error::other("tool marker is missing"))?
        );
        println!(
            "{}",
            llm_result["content"]
                .as_str()
                .ok_or_else(|| io::Error::other("response marker is missing"))?
        );
        Ok(())
    }
    .await;

    // Pop before propagating errors so a failed callback cannot leave the agent scope open.
    let pop_result = scope::pop_scope(
        PopScopeParams::builder().handle_uuid(&handle.uuid).build(),
    );
    workflow_result?;
    pop_result?;
    Ok(())
}

#[tokio::main]
/// Run the local example and verify its exported lifecycle records.
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut activation = initialize(
        PluginConfig::default(),
        Some(PathBuf::from("plugins.toml")),
    )
    .await?;

    // Attempt both cleanup operations even when the workflow or subscriber flush fails.
    // Applying `?` immediately would skip later cleanup and could lose diagnostic events.
    let work_result = run_workflow().await;
    let flush_result = flush_subscribers();
    let close_result = activation.close();
    work_result?;
    flush_result?;
    close_result?;

    let event_text = fs::read_to_string("logs/events.jsonl")?;
    let events: Vec<serde_json::Value> = event_text.lines()
        .map(serde_json::from_str).collect::<Result<_, _>>()?;
    if events.is_empty() || events.iter().any(|event|
        event["atof_version"] != "0.1" ||
        !matches!(event["kind"].as_str(), Some("scope" | "mark"))
    ) {
        return Err(io::Error::other("Expected nonempty ATOF 0.1 records").into());
    }
    for category in ["agent", "tool", "llm"] {
        let paired = events.iter().any(|end|
            end["kind"] == "scope" && end["scope_category"] == "end" &&
            end["category"] == category && end["uuid"].is_string() &&
            events.iter().any(|start|
                start["kind"] == "scope" && start["scope_category"] == "start" &&
                ["uuid", "name", "category", "parent_uuid"].iter()
                    .all(|key| start[key] == end[key])
            )
        );
        if !paired {
            return Err(io::Error::other(format!("Missing paired {category} records")).into());
        }
    }
    println!("RELAY_QUICKSTART_EVENTS_OK");
    Ok(())
}
