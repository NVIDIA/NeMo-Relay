// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool status and cleanup when Codex skips an MCP completion hook.

use super::*;

/// Add error or interruption fields without changing the reported status.
pub(super) fn tool_end_metadata(metadata: Value, status: Option<&str>) -> Value {
    let mut completion = json!({ "status": status });
    let normalized = status.map(str::to_ascii_lowercase);
    if matches!(
        normalized.as_deref(),
        Some("error" | "failed" | "failure" | "denied" | "timeout")
    ) {
        completion["otel.status_code"] = json!("ERROR");
        completion["error.type"] = json!("tool_execution_failed");
    }
    if matches!(
        normalized.as_deref(),
        Some("interrupted" | "cancelled" | "canceled")
    ) {
        completion["tool_completion_outcome"] = json!("interrupted");
    }
    merge_metadata(metadata, completion)
}

/// Mark a call as incomplete when cleanup closes it without a completion signal.
pub(super) fn incomplete_tool_metadata(reason: &str) -> Value {
    json!({
        "status": reason,
        "tool_completion_source": "lifecycle",
        "tool_completion_outcome": "incomplete",
        "tool_completion_timing": "upper_bound",
    })
}

impl Session {
    /// Codex skips `PostToolUse` for failed MCP calls. A result in the next Responses
    /// request shows that the call has ended. Its exact end time and success status
    /// are unknown. Match the call ID and owning agent. Do not guess failure from
    /// output text. Shell tools can leave a process running after they return output.
    pub(super) fn close_codex_mcp_results(
        &mut self,
        start: &LlmGatewayStart,
        owner: Option<&str>,
    ) -> Result<(), CliError> {
        if self.agent_kind != AgentKind::Codex || start.provider != "openai.responses" {
            return Ok(());
        }
        let Some(input) = start.request.content.get("input").and_then(Value::as_array) else {
            return Ok(());
        };
        for item in input {
            if item.get("type").and_then(Value::as_str) != Some("function_call_output") {
                continue;
            }
            let (Some(id), Some(output)) = (
                item.get("call_id").and_then(Value::as_str),
                item.get("output"),
            ) else {
                continue;
            };
            if !(output.is_string() || output.is_array()) {
                continue;
            }
            if !self.tools.get(id).is_some_and(|tool| {
                tool.name.starts_with("mcp__") && tool.owner_subagent_id.as_deref() == owner
            }) {
                continue;
            }
            let tool = self.tools.remove(id).expect("matched active MCP tool");
            let mut metadata = self.event_identity_metadata(json!({
                "status": "unknown",
                "tool_completion_source": "gateway_request",
                "tool_completion_outcome": "unknown",
                "tool_completion_timing": "upper_bound",
            }));
            metadata["source"] = json!("gateway");
            self.apply_tool_execution_metadata(&mut metadata, owner);
            tool_call_end(
                ToolCallEndParams::builder()
                    .handle(&tool.handle)
                    .execution_result(output.clone().into())
                    .metadata(metadata)
                    .build(),
            )?;
            self.recovered_tool_completions
                .record(completion::recovered_tool_key(&self.session_id, id));
            self.touch_subagent_activity(owner);
        }
        Ok(())
    }
}
