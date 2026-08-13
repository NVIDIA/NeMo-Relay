// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_plugin::{
    EventCategory, Json, LlmOptimizationContribution, LlmRequestInterceptOutcome, PendingMarkSpec,
    PluginContext,
};
use serde_json::json;

use crate::config::ExampleConfig;
use crate::observe::add_header;

pub(crate) fn register(
    context: &mut PluginContext<'_>,
    config: &ExampleConfig,
) -> nemo_relay_plugin::Result<()> {
    if !config.requests.enabled {
        return Ok(());
    }

    context.register_tool_conditional_execution_guardrail("documentation_tool_policy", 10, {
        let mode = config.requests.mode.clone();
        let blocked = config.requests.blocked_tools.clone();
        move |name, _args| {
            let mode = mode.clone();
            let blocked = blocked.clone();
            async move {
                Ok((mode == "enforce" && blocked.contains(&name))
                    .then(|| format!("tool '{name}' is blocked by documentation policy")))
            }
        }
    })?;
    context.register_tool_request_intercept(
        "documentation_tool_request",
        config.requests.priority,
        config.requests.break_chain,
        {
            let tag = config.tag.clone();
            move |name, value| {
                let tag = tag.clone();
                async move { Ok(tag_tool_request(value, &name, &tag)) }
            }
        },
    )?;

    context.register_llm_conditional_execution_guardrail("documentation_llm_policy", 10, {
        let mode = config.requests.mode.clone();
        let blocked = config.requests.blocked_models.clone();
        move |request| {
            let mode = mode.clone();
            let blocked = blocked.clone();
            async move {
                let model = request
                    .content
                    .get("model")
                    .and_then(Json::as_str)
                    .unwrap_or_default();
                Ok(
                    (mode == "enforce" && blocked.iter().any(|candidate| candidate == model))
                        .then(|| format!("model '{model}' is blocked by documentation policy")),
                )
            }
        }
    })?;
    context.register_llm_request_intercept(
        "documentation_llm_request",
        config.requests.priority,
        config.requests.break_chain,
        {
            let header_name = config.requests.header_name.clone();
            let header_value = config.requests.header_value.clone();
            let emit_pending_marks = config.execution.emit_pending_marks;
            move |_name, request, annotated| {
                let header_name = header_name.clone();
                let header_value = header_value.clone();
                async move {
                    let mut outcome = LlmRequestInterceptOutcome::new(
                        add_header(request, &header_name, &header_value),
                        annotated,
                    )
                    .with_optimization_contribution(
                        LlmOptimizationContribution::new(
                            "examples.rust_native_policy",
                            "request_rewrite",
                        ),
                    );
                    if emit_pending_marks {
                        outcome = outcome.with_pending_mark(
                            PendingMarkSpec::builder()
                                .name("example.native.llm_request")
                                .category(EventCategory::custom())
                                .data(json!({ "header": header_name }))
                                .build(),
                        );
                    }
                    Ok(outcome)
                }
            }
        },
    )?;

    Ok(())
}

fn tag_tool_request(value: Json, name: &str, tag: &str) -> Json {
    match value {
        Json::Object(mut object) => {
            object.insert("plugin_tag".into(), Json::String(tag.into()));
            object.insert("plugin_tool".into(), Json::String(name.into()));
            Json::Object(object)
        }
        other => other,
    }
}
