// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use serde_json::json;
use std::sync::Arc;

/// Execute the local provider callback through Relay.
pub async fn run_example() -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let request = LlmRequest {
        headers: Default::default(),
        content: json!({"messages": [{"role": "user", "content": "hello"}]}),
    };

    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("demo-provider")
            .request(request)
            .func(Arc::new(|provider_request| {
                Box::pin(
                    async move { Ok(json!({"text": "hi", "request": provider_request.content})) },
                )
            }))
            .model_name("demo-model")
            .build(),
    )
    .await?;
    Ok(response)
}
