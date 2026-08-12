// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::StreamExt;
use nemo_relay_plugin::{
    EventCategory, Json, LlmJsonAsyncStream, PendingMarkSpec, PluginContext,
    ToolExecutionInterceptOutcome,
};
use serde_json::json;

use crate::config::ExampleConfig;

pub(crate) fn register(
    context: &mut PluginContext<'_>,
    config: &ExampleConfig,
) -> nemo_relay_plugin::Result<()> {
    if !config.execution.enabled {
        return Ok(());
    }

    context.register_tool_execution_intercept(
        "documentation_tool_execution",
        config.execution.priority,
        {
            let emit_pending_marks = config.execution.emit_pending_marks;
            move |_name, request, next| async move {
                let result = next.call(request).await?;
                let mut outcome = ToolExecutionInterceptOutcome::new(result);
                if emit_pending_marks {
                    outcome = outcome.with_pending_mark(
                        PendingMarkSpec::builder()
                            .name("example.native.tool_execution")
                            .category(EventCategory::custom())
                            .data(json!({ "source": "documentation" }))
                            .build(),
                    );
                }
                Ok(outcome)
            }
        },
    )?;

    context.register_llm_execution_intercept(
        "documentation_llm_execution",
        config.execution.priority,
        move |_name, request, next| async move {
            if request
                .content
                .get("repeat_downstream")
                .and_then(Json::as_bool)
                .unwrap_or(false)
            {
                let repeated = next.clone();
                let (first, second) =
                    tokio::join!(repeated.call(request.clone()), next.call(request));
                let response = first?;
                second?;
                Ok(response)
            } else {
                next.call(request).await
            }
        },
    )?;

    context.register_llm_stream_execution_intercept(
        "documentation_llm_stream_execution",
        config.execution.priority,
        move |_name, request, next| async move {
            let stream = next.call(request).await?;
            let stream: LlmJsonAsyncStream = Box::pin(stream.map(|chunk| {
                chunk.map(|chunk| match chunk {
                    Json::Object(mut object) => {
                        object.insert("plugin_stream".into(), Json::Bool(true));
                        Json::Object(object)
                    }
                    other => other,
                })
            }));
            Ok(stream)
        },
    )?;

    Ok(())
}
