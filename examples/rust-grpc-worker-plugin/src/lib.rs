// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod config;

use std::collections::BTreeMap;

use futures_util::StreamExt;
use nemo_relay_worker::{
    ConfigDiagnostic, DiagnosticLevel, EventSanitizeFields, Json, JsonStream,
    LlmOptimizationContribution, LlmRequestInterceptOutcome, PendingMarkSpec, PluginContext,
    PluginRuntime, Result, ScopeType, ToolExecutionInterceptOutcome, WorkerPlugin, WorkerSdkError,
};
use serde_json::json;

use config::ExampleConfig;

/// Complete worker implementation used by the plugin authoring documentation.
pub struct DocumentationWorker;

impl WorkerPlugin for DocumentationWorker {
    fn plugin_id(&self) -> &str {
        "examples.rust_grpc_worker"
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, config: &Json) -> Vec<ConfigDiagnostic> {
        config::validate(config)
    }

    fn register(&self, context: &mut PluginContext, config: &Json) -> Result<()> {
        if let Some(diagnostic) = config::validate(config)
            .into_iter()
            .find(|diagnostic| diagnostic.level == DiagnosticLevel::Error)
        {
            let location = diagnostic.field.unwrap_or(diagnostic.code);
            return Err(WorkerSdkError::InvalidInput(format!(
                "{location}: {}",
                diagnostic.message
            )));
        }
        let config = ExampleConfig::parse(config).map_err(WorkerSdkError::InvalidInput)?;
        if config.registration_control.enabled {
            let reason = config.registration_control.reason.clone();
            context.register_conditional_middleware_guardrail(
                "documentation_registration_control",
                config.registration_control.kinds.iter().copied().collect(),
                &config.registration_control.registration_name,
                move |_, registration_name| {
                    let decision = registration_name
                        .starts_with("documentation-controlled-")
                        .then(|| reason.clone());
                    async move { Ok(decision) }
                },
            );
        }
        if config.observe.enabled {
            register_observation(context, &config);
        }
        if config.requests.enabled {
            register_requests(context, &config);
        }
        register_execution(context, &config);
        Ok(())
    }
}

/// Validate the shared example configuration without starting the worker server.
pub fn validate_example_config(config: &Json) -> Vec<ConfigDiagnostic> {
    config::validate(config)
}

/// Return the configuration defaults as the JSON document accepted by this worker.
///
/// The integration tests compare this document with the checked JSON Schema so the
/// documented defaults cannot drift from the runtime defaults.
pub fn default_example_config() -> Json {
    serde_json::to_value(ExampleConfig::default()).expect("example configuration must serialize")
}

fn register_observation(context: &mut PluginContext, config: &ExampleConfig) {
    context.register_event_metadata_injector(
        "documentation_event_metadata_injector",
        5,
        |_event| async move {
            Ok(BTreeMap::from([(
                "external.injector.transport".into(),
                Json::String("rust_grpc_worker".into()),
            )]))
        },
    );
    context.register_subscriber("documentation_subscriber", |_event| {});

    let sanitize_fields = |mut fields: EventSanitizeFields, keys: &[String], tag: &str| {
        fields.data = fields.data.map(|value| redact(value, keys));
        fields.metadata = Some(tag_metadata(fields.metadata, tag, keys));
        if let Some(profile) = fields.category_profile.take() {
            fields.category_profile = serde_json::to_value(&profile)
                .map(|value| redact(value, keys))
                .and_then(serde_json::from_value)
                .ok()
                .or(Some(profile));
        }
        fields
    };
    context.register_mark_sanitize_guardrail("documentation_mark_sanitizer", 10, {
        let keys = config.observe.redact_keys.clone();
        let tag = config.tag.clone();
        move |_event, fields| {
            let fields = sanitize_fields(fields, &keys, &tag);
            async move { Ok(fields) }
        }
    });
    context.register_scope_sanitize_start_guardrail("documentation_scope_start_sanitizer", 10, {
        let keys = config.observe.redact_keys.clone();
        let tag = config.tag.clone();
        move |_event, fields| {
            let fields = sanitize_fields(fields, &keys, &tag);
            async move { Ok(fields) }
        }
    });
    context.register_scope_sanitize_end_guardrail("documentation_scope_end_sanitizer", 10, {
        let keys = config.observe.redact_keys.clone();
        let tag = config.tag.clone();
        move |_event, fields| {
            let fields = sanitize_fields(fields, &keys, &tag);
            async move { Ok(fields) }
        }
    });
    context.register_tool_sanitize_request_guardrail("documentation_tool_request_sanitizer", 10, {
        let keys = config.observe.redact_keys.clone();
        move |_name, value| {
            let keys = keys.clone();
            async move { Ok(redact(value, &keys)) }
        }
    });
    context.register_tool_sanitize_response_guardrail(
        "documentation_tool_response_sanitizer",
        10,
        {
            let keys = config.observe.redact_keys.clone();
            move |_name, value| {
                let keys = keys.clone();
                async move { Ok(redact(value, &keys)) }
            }
        },
    );
    context.register_llm_sanitize_request_guardrail("documentation_llm_request_sanitizer", 10, {
        let keys = config.observe.redact_keys.clone();
        move |mut request, codec_context| {
            let keys = keys.clone();
            async move {
                if let Some(codec) = codec_context.resolve_codec() {
                    let annotated = codec.decode(&request).await?;
                    let annotated = serde_json::to_value(annotated)
                        .map(|value| redact(value, &keys))
                        .and_then(serde_json::from_value)
                        .map_err(|error| WorkerSdkError::InvalidInput(error.to_string()))?;
                    request = codec.encode(&annotated, &request).await?;
                }
                request.content = redact(request.content, &keys);
                Ok(Some(request))
            }
        }
    });
    context.register_llm_sanitize_response_guardrail("documentation_llm_response_sanitizer", 10, {
        let keys = config.observe.redact_keys.clone();
        move |response, codec_context| {
            let keys = keys.clone();
            async move {
                if let Some(codec) = codec_context.resolve_codec() {
                    let _annotated = codec.decode(&response).await?;
                }
                Ok(Some(redact(response, &keys)))
            }
        }
    });
}

fn register_requests(context: &mut PluginContext, config: &ExampleConfig) {
    context.register_tool_conditional_execution_guardrail("documentation_tool_policy", 10, {
        let mode = config.requests.mode.clone();
        let blocked = config.requests.blocked_tools.clone();
        move |name, _value| {
            let mode = mode.clone();
            let blocked = blocked.clone();
            async move {
                Ok((mode == "enforce" && blocked.contains(&name))
                    .then(|| format!("tool '{name}' is blocked by documentation policy")))
            }
        }
    });
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
    );
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
    });
    context.register_llm_request_intercept(
        "documentation_llm_request",
        config.requests.priority,
        config.requests.break_chain,
        {
            let header_name = config.requests.header_name.clone();
            let header_value = config.requests.header_value.clone();
            let tag = config.tag.clone();
            let emit_marks = config.execution.emit_pending_marks;
            move |_model, mut request, annotated| {
                let header_name = header_name.clone();
                let header_value = header_value.clone();
                let tag = tag.clone();
                async move {
                    request
                        .headers
                        .insert(header_name.clone(), Json::String(header_value));
                    let mut outcome = LlmRequestInterceptOutcome::new(request, annotated)
                        .with_optimization_contribution(LlmOptimizationContribution::new(
                            "examples.rust_grpc_worker",
                            "request_rewrite",
                        ));
                    if emit_marks {
                        outcome = outcome.with_pending_mark(
                            PendingMarkSpec::builder()
                                .name("example.rust_worker.llm_request")
                                .data(json!({ "tag": tag }))
                                .build(),
                        );
                    }
                    Ok(outcome)
                }
            }
        },
    );
}

fn register_execution(context: &mut PluginContext, config: &ExampleConfig) {
    if (config.runtime.emit_marks || config.runtime.emit_isolated_scope)
        && let Some(runtime) = context.runtime()
    {
        context.register_tool_execution_intercept("documentation_runtime_events", 0, {
            let tag = config.tag.clone();
            let runtime_config = config.runtime.clone();
            move |_name, value, next| {
                let runtime = runtime.clone();
                let tag = tag.clone();
                let runtime_config = runtime_config.clone();
                async move {
                    emit_runtime_events(&runtime, &tag, &runtime_config).await?;
                    Ok(ToolExecutionInterceptOutcome::from(next.call(value).await?))
                }
            }
        });
    }

    if !config.execution.enabled {
        return;
    }

    context.register_tool_execution_intercept(
        "documentation_tool_execution",
        config.execution.priority,
        {
            let emit_marks = config.execution.emit_pending_marks;
            move |_name, value, next| async move {
                let result = next.call(value).await?;
                let mut outcome = ToolExecutionInterceptOutcome::from(result);
                if emit_marks {
                    outcome = outcome.with_pending_mark(
                        PendingMarkSpec::builder()
                            .name("example.rust_worker.tool_execution")
                            .build(),
                    );
                }
                Ok(outcome)
            }
        },
    );
    context.register_llm_execution_intercept(
        "documentation_llm_execution",
        config.execution.priority,
        move |_model, request, next| async move {
            if request
                .content
                .get("repeat_downstream")
                .and_then(Json::as_bool)
                .unwrap_or(false)
            {
                let repeated = next.clone();
                let (first, _second) =
                    tokio::join!(repeated.call(request.clone()), next.call(request));
                first
            } else {
                next.call(request).await
            }
        },
    );
    context.register_llm_stream_execution_intercept(
        "documentation_llm_stream_execution",
        config.execution.priority,
        move |_model, request, next| async move {
            let stream = next.call(request).await?;
            let mapped: JsonStream = Box::pin(stream.map(|chunk| {
                chunk.map(|chunk| match chunk {
                    Json::Object(mut object) => {
                        object.insert("plugin_stream".into(), Json::Bool(true));
                        Json::Object(object)
                    }
                    other => other,
                })
            }));
            Ok(mapped)
        },
    );
}

async fn emit_runtime_events(
    runtime: &PluginRuntime,
    tag: &str,
    config: &config::RuntimeConfig,
) -> Result<()> {
    let handle = runtime
        .push_scope(
            None,
            "example.rust_worker.request",
            ScopeType::Custom,
            Some(json!({ "tag": tag })),
            None,
            None,
        )
        .await?;
    let work = if config.emit_marks {
        runtime
            .emit_mark(
                "example.rust_worker.request.seen",
                Some(json!({ "tag": tag })),
                None,
            )
            .await
    } else {
        Ok(())
    };
    match work {
        Ok(()) => {
            runtime
                .pop_scope(&handle, Some(json!({ "done": true })), None)
                .await?;
        }
        Err(error) => {
            let _ = runtime
                .pop_scope(&handle, None, Some(json!({ "failed": true })))
                .await;
            return Err(error);
        }
    }

    if config.emit_isolated_scope {
        let stack = runtime.create_scope_stack().await?;
        let emitted = runtime
            .with_scope_stack(&stack, || async {
                runtime
                    .emit_mark(
                        "example.rust_worker.isolated.mark",
                        Some(json!({ "tag": tag })),
                        None,
                    )
                    .await
            })
            .await;
        let dropped = runtime.drop_scope_stack(&stack).await;
        emitted?;
        dropped?;
    }
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

fn redact(value: Json, keys: &[String]) -> Json {
    match value {
        Json::Object(mut object) => {
            for (key, value) in &mut object {
                if keys.iter().any(|candidate| candidate == key) {
                    *value = Json::String("[REDACTED]".into());
                } else {
                    *value = redact(value.take(), keys);
                }
            }
            Json::Object(object)
        }
        Json::Array(values) => Json::Array(
            values
                .into_iter()
                .map(|value| redact(value, keys))
                .collect(),
        ),
        other => other,
    }
}

fn tag_metadata(metadata: Option<Json>, tag: &str, keys: &[String]) -> Json {
    let mut object = match redact(metadata.unwrap_or_else(|| json!({})), keys) {
        Json::Object(object) => object,
        other => serde_json::Map::from_iter([("original".into(), other)]),
    };
    object.insert("plugin_tag".into(), Json::String(tag.into()));
    Json::Object(object)
}
