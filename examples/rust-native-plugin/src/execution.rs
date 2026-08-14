// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::StreamExt;
use nemo_relay_plugin::{
    EventCategory, Json, LlmJsonAsyncStream, PendingMarkSpec, PluginContext, PluginRuntime,
    ToolExecutionInterceptOutcome,
};
use serde_json::json;

use crate::config::ExampleConfig;
use crate::runtime::emit_configured_runtime_events;

pub(crate) fn register(
    context: &mut PluginContext<'_>,
    config: &ExampleConfig,
    runtime: &PluginRuntime,
) -> nemo_relay_plugin::Result<()> {
    if config.runtime.emit_marks || config.runtime.emit_isolated_scope {
        context.register_tool_execution_intercept("documentation_runtime_events", 0, {
            let tag = config.tag.clone();
            let runtime = runtime.clone();
            let runtime_config = config.runtime.clone();
            move |_name, request, next| {
                let tag = tag.clone();
                let runtime = runtime.clone();
                let runtime_config = runtime_config.clone();
                async move {
                    emit_configured_runtime_events(&runtime, &tag, &runtime_config)?;
                    Ok(ToolExecutionInterceptOutcome::from(next.call(request).await?))
                }
            }
        })?;
    }

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
                let mut outcome = ToolExecutionInterceptOutcome::from(result);
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
