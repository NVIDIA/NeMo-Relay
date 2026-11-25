// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::tool::{
    ToolCallEndParams, ToolCallParams, ToolExecutionResult, tool_call, tool_call_end,
};
use serde_json::{Value, json};

fn search(query: &str) -> nemo_relay::Result<Value> {
    if query.is_empty() {
        return Err(nemo_relay::FlowError::InvalidArgument(
            "query is required".into(),
        ));
    }
    Ok(json!({"hits": 2}))
}

/// Complete the manual tool lifecycle and preserve operation errors.
pub fn run_example(query: &str) -> nemo_relay::Result<()> {
    let handle = tool_call(
        ToolCallParams::builder()
            .name("search")
            .args(json!({"query": query}))
            .data(json!({"attempt": 1}))
            .build(),
    )?;

    let (result, execution_error) = match search(query) {
        Ok(result) => (result, None),
        Err(error) => (json!({"error": error.to_string()}), Some(error)),
    };
    let completion_result = tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .execution_result(ToolExecutionResult::new(result))
            .build(),
    );

    if let Some(execution_error) = execution_error {
        if let Err(completion_error) = completion_result {
            eprintln!("Relay could not record tool completion: {completion_error}");
        }
        return Err(execution_error);
    }
    completion_result
}
