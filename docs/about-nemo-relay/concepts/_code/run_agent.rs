// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::context::capture_traceparent;
use nemo_relay::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay::middleware::LlmExecutionNextFn;
use nemo_relay::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use nemo_relay::{Json, Result};

async fn run_agent(request: LlmRequest, provider: LlmExecutionNextFn) -> Result<(Json, String)> {
    let agent = push_scope(
        PushScopeParams::builder()
            .name("agent-run")
            .scope_type(ScopeType::Agent)
            .build(),
    )?;
    let outcome = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("provider")
            .request(request)
            .func(provider)
            .build(),
    )
    .await
    .and_then(|result| {
        // The LLM scope has closed. The Agent scope is still active.
        let agent_traceparent = capture_traceparent()?;
        Ok((result, agent_traceparent))
    });
    // Close the Agent scope even if execution or capture returned an error.
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build())?;
    outcome
}
