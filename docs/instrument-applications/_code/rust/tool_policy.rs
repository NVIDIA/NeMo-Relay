// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::registry::{
    register_tool_conditional_execution_guardrail, register_tool_execution_intercept,
    register_tool_sanitize_request_guardrail,
};
use nemo_relay::tool::ToolExecutionInterceptOutcome;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

pub fn register_tool_policy() -> nemo_relay::Result<()> {
    register_tool_sanitize_request_guardrail(
        "search.redact_api_key",
        10,
        Arc::new(|_tool_name, mut tool_arguments| {
            Box::pin(async move {
                if let Some(object) = tool_arguments.as_object_mut()
                    && object.contains_key("api_key")
                {
                    object.insert("api_key".into(), json!("<redacted>"));
                }
                Ok(tool_arguments)
            })
        }),
    )?;

    register_tool_conditional_execution_guardrail(
        "search.require_query",
        20,
        Arc::new(|_tool_name, tool_arguments| {
            Box::pin(async move {
                Ok(
                    match tool_arguments.get("query").and_then(|value| value.as_str()) {
                        Some(query) if !query.is_empty() => None,
                        _ => Some("search.query is required".into()),
                    },
                )
            })
        }),
    )?;

    register_tool_execution_intercept(
        "search.measure",
        30,
        Arc::new(|context, next_call| {
            let name = context.tool_name().to_owned();
            Box::pin(async move {
                let started = Instant::now();
                let result = next_call(context.into_args()).await;
                println!("{name} completed in {:?}", started.elapsed());
                Ok(ToolExecutionInterceptOutcome::from(result?))
            })
        }),
    )?;
    Ok(())
}
