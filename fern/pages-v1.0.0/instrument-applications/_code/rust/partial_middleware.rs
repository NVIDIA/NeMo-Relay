// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::llm::{LlmRequest, llm_conditional_execution, llm_request_intercepts};
use nemo_relay::tool::{tool_conditional_execution, tool_request_intercepts};
use serde_json::json;

/// Apply selected middleware without invoking a callback.
pub async fn run_example()
-> Result<(serde_json::Value, nemo_relay::llm::LlmRequest), Box<dyn std::error::Error>> {
    let tool_arguments = tool_request_intercepts("search", json!({"query": "weather"})).await?;
    tool_conditional_execution("search", &tool_arguments).await?;

    let request = LlmRequest {
        headers: Default::default(),
        content: json!({"messages": [{"role": "user", "content": "hello"}]}),
    };
    let outcome = llm_request_intercepts("demo-provider", request).await?;
    llm_conditional_execution(&outcome.request).await?;
    Ok((tool_arguments, outcome.request))
}
