// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::registry::register_tool_conditional_execution_guardrail;
use std::sync::Arc;

pub fn register_query_policy() -> nemo_relay::Result<()> {
    register_tool_conditional_execution_guardrail(
        "search.require_query",
        20,
        Arc::new(|tool_name, arguments| {
            Box::pin(async move {
                if tool_name != "search" {
                    return Ok(None);
                }
                Ok(
                    match arguments.get("query").and_then(|value| value.as_str()) {
                        Some(query) if !query.is_empty() => None,
                        _ => Some("search.query is required".into()),
                    },
                )
            })
        }),
    )?;
    Ok(())
}
