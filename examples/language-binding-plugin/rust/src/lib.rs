// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod config;

use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::{StreamExt, stream};
use nemo_relay::api::event::{Event, EventSanitizeFields, PendingMarkSpec};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmRequestInterceptOutcome, LlmStreamCallExecuteParams,
    llm_call_execute, llm_stream_call_execute,
};
use nemo_relay::api::runtime::callbacks::LlmJsonStream;
use nemo_relay::api::runtime::{create_scope_stack, with_scope_stack};
use nemo_relay::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeType, event, pop_scope, push_scope,
};
use nemo_relay::api::subscriber::flush_subscribers;
use nemo_relay::api::tool::{
    ToolCallExecuteParams, ToolExecutionInterceptOutcome, ToolExecutionResult, tool_call_execute,
};
use nemo_relay::plugin::{
    ConfigDiagnostic, ConfigPolicy, Plugin, PluginComponentSpec, PluginConfig,
    PluginRegistrationContext, Result as PluginResult, clear_plugin_configuration,
    deregister_plugin, initialize_plugins_exact, list_plugin_kinds, register_plugin,
    validate_plugin_config,
};
use serde_json::{Map, Value as Json, json};

pub struct DocumentationPlugin;

static OBSERVED_EVENTS: AtomicUsize = AtomicUsize::new(0);
static OBSERVED_EVENT_DETAILS: Mutex<Vec<Event>> = Mutex::new(Vec::new());

impl Plugin for DocumentationPlugin {
    fn plugin_kind(&self) -> &str {
        "documentation-plugin"
    }

    fn validate(&self, config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        config::validate(config, &ConfigPolicy::default())
    }

    fn validate_with_policy(
        &self,
        config: &Map<String, Json>,
        policy: &ConfigPolicy,
    ) -> Vec<ConfigDiagnostic> {
        config::validate(config, policy)
    }

    fn register<'a>(
        &'a self,
        config: &Map<String, Json>,
        context: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        let config = config.clone();
        Box::pin(async move {
            let settings =
                config::parse(&config).map_err(nemo_relay::plugin::PluginError::InvalidConfig)?;
            if settings.registration_control.enabled {
                let reason = settings.registration_control.reason.clone();
                context.register_conditional_middleware_guardrail(
                    "registration-control",
                    settings.registration_control.kinds.iter().copied().collect::<BTreeSet<_>>(),
                    &settings.registration_control.registration_name,
                    Arc::new(move |_, _| Some(reason.clone())),
                )?;
            }
            if settings.observe.enabled {
                context.register_subscriber(
                    "events",
                    Arc::new(|event| {
                        OBSERVED_EVENTS.fetch_add(1, Ordering::Relaxed);
                        OBSERVED_EVENT_DETAILS
                            .lock()
                            .expect("event details lock should not be poisoned")
                            .push(event.clone());
                        println!("event: {}", event.name());
                    }),
                )?;
                context.register_mark_sanitize_guardrail(
                    "mark-redaction",
                    10,
                    Arc::new({
                        let redact_keys = settings.observe.redact_keys.clone();
                        move |_event, fields| {
                            let redact_keys = redact_keys.clone();
                            Box::pin(async move { Ok(redact_event_fields(fields, &redact_keys)) })
                        }
                    }),
                )?;
                context.register_scope_sanitize_start_guardrail(
                    "scope-start-redaction",
                    10,
                    Arc::new({
                        let redact_keys = settings.observe.redact_keys.clone();
                        move |_event, fields| {
                            let redact_keys = redact_keys.clone();
                            Box::pin(async move { Ok(redact_event_fields(fields, &redact_keys)) })
                        }
                    }),
                )?;
                context.register_scope_sanitize_end_guardrail(
                    "scope-end-redaction",
                    10,
                    Arc::new({
                        let redact_keys = settings.observe.redact_keys.clone();
                        move |_event, fields| {
                            let redact_keys = redact_keys.clone();
                            Box::pin(async move { Ok(redact_event_fields(fields, &redact_keys)) })
                        }
                    }),
                )?;
                context.register_tool_sanitize_request_guardrail(
                    "tool-request-redaction",
                    10,
                    Arc::new({
                        let redact_keys = settings.observe.redact_keys.clone();
                        move |_name, value| {
                            let redact_keys = redact_keys.clone();
                            Box::pin(async move { Ok(redact_json(value, &redact_keys)) })
                        }
                    }),
                )?;
                context.register_tool_sanitize_response_guardrail(
                    "tool-response-redaction",
                    10,
                    Arc::new({
                        let redact_keys = settings.observe.redact_keys.clone();
                        move |_name, value| {
                            let redact_keys = redact_keys.clone();
                            Box::pin(async move { Ok(redact_json(value, &redact_keys)) })
                        }
                    }),
                )?;
                context.register_llm_sanitize_request_guardrail(
                    "llm-request-redaction",
                    10,
                    Arc::new({
                        let redact_keys = settings.observe.redact_keys.clone();
                        move |mut request, _context| {
                            let redact_keys = redact_keys.clone();
                            Box::pin(async move {
                                request.content = redact_json(request.content, &redact_keys);
                                Ok(Some(request))
                            })
                        }
                    }),
                )?;
                context.register_llm_sanitize_response_guardrail(
                    "llm-response-redaction",
                    10,
                    Arc::new({
                        let redact_keys = settings.observe.redact_keys.clone();
                        move |response, _context| {
                            let redact_keys = redact_keys.clone();
                            Box::pin(async move { Ok(Some(redact_json(response, &redact_keys))) })
                        }
                    }),
                )?;
            }
            if settings.requests.enabled {
                context.register_tool_conditional_execution_guardrail(
                    "tool-policy",
                    10,
                    Arc::new({
                        let mode = settings.requests.mode.clone();
                        let blocked = settings.requests.blocked_tools.clone();
                        move |name, _args| {
                            let mode = mode.clone();
                            let blocked = blocked.clone();
                            Box::pin(async move {
                                Ok((mode == "enforce" && blocked.contains(&name))
                                    .then(|| format!("tool '{name}' is blocked")))
                            })
                        }
                    }),
                )?;
                context.register_tool_request_intercept(
                    "tool-request",
                    settings.requests.priority,
                    settings.requests.break_chain,
                    Arc::new({
                        let tag = settings.tag.clone();
                        move |_name, mut args| {
                            let tag = tag.clone();
                            Box::pin(async move {
                                if let Some(object) = args.as_object_mut() {
                                    object.insert("plugin_tag".into(), Json::String(tag));
                                }
                                Ok(args)
                            })
                        }
                    }),
                )?;
                context.register_llm_conditional_execution_guardrail(
                    "llm-policy",
                    10,
                    Arc::new({
                        let mode = settings.requests.mode.clone();
                        let blocked = settings.requests.blocked_models.clone();
                        move |request| {
                            let mode = mode.clone();
                            let blocked = blocked.clone();
                            Box::pin(async move {
                                let model = request
                                    .content
                                    .get("model")
                                    .and_then(Json::as_str)
                                    .unwrap_or_default();
                                Ok((mode == "enforce"
                                    && blocked.iter().any(|candidate| candidate == model))
                                .then(|| format!("model '{model}' is blocked")))
                            })
                        }
                    }),
                )?;
                context.register_llm_request_intercept(
                    "llm-request",
                    settings.requests.priority,
                    settings.requests.break_chain,
                    Arc::new({
                        let header_name = settings.requests.header_name.clone();
                        let header_value = settings.requests.header_value.clone();
                        move |_name, mut request, annotated| {
                            let header_name = header_name.clone();
                            let header_value = header_value.clone();
                            Box::pin(async move {
                                request
                                    .headers
                                    .insert(header_name, Json::String(header_value));
                                Ok(LlmRequestInterceptOutcome::new(request, annotated))
                            })
                        }
                    }),
                )?;
            }
            if settings.runtime.emit_marks || settings.runtime.emit_isolated_scope {
                context.register_tool_execution_intercept(
                    "runtime-events",
                    0,
                    Arc::new({
                        let tag = settings.tag.clone();
                        let runtime = settings.runtime.clone();
                        move |_name, args, next| {
                            let tag = tag.clone();
                            let runtime = runtime.clone();
                            Box::pin(async move {
                                emit_runtime_events(&tag, &runtime)?;
                                Ok(ToolExecutionInterceptOutcome::from(next(args).await?))
                            })
                        }
                    }),
                )?;
            }
            if settings.execution.enabled {
                context.register_tool_execution_intercept(
                    "tool-pending-mark",
                    settings.execution.priority,
                    Arc::new({
                        let emit_pending_marks = settings.execution.emit_pending_marks;
                        move |_name, args, next| {
                            Box::pin(async move {
                                let result = next(args).await?;
                                let outcome = ToolExecutionInterceptOutcome::from(result);
                                Ok(if emit_pending_marks {
                                    outcome.with_pending_mark(
                                        PendingMarkSpec::builder()
                                            .name("documentation-plugin.tool-complete")
                                            .data(json!({ "source": "documentation" }))
                                            .build(),
                                    )
                                } else {
                                    outcome
                                })
                            })
                        }
                    }),
                )?;
                context.register_llm_execution_intercept(
                    "llm-execution",
                    settings.execution.priority,
                    Arc::new(move |_name, request, next| {
                        Box::pin(async move { next(request).await })
                    }),
                )?;
                context.register_llm_stream_execution_intercept(
                    "llm-stream",
                    settings.execution.priority,
                    Arc::new(move |_name, request, next| {
                        Box::pin(async move {
                            let stream = next(request).await?;
                            Ok(LlmJsonStream::new(stream.map(|chunk| {
                                chunk.map(|mut chunk| {
                                    if let Some(object) = chunk.as_object_mut() {
                                        object.insert("plugin_stream".into(), Json::Bool(true));
                                    }
                                    chunk
                                })
                            })))
                        })
                    }),
                )?;
            }
            Ok(())
        })
    }
}

pub fn config_with_enabled(mode: &str, enabled: bool) -> PluginConfig {
    let mut component = PluginComponentSpec::new("documentation-plugin");
    component.enabled = enabled;
    component.config = json!({
        "tag": "documentation",
        "observe": { "enabled": true, "redact_keys": ["secret"] },
        "requests": {
            "enabled": true,
            "mode": mode,
            "blocked_tools": ["dangerous_tool"],
            "blocked_models": ["restricted-model"],
            "header_name": "x-nemo-relay-plugin",
            "header_value": "documentation",
            "priority": 20,
            "break_chain": false
        },
        "execution": { "enabled": true, "priority": 30, "emit_pending_marks": true },
        "runtime": { "emit_marks": true, "emit_isolated_scope": true },
        "registration_control": {
            "enabled": false,
            "kinds": ["subscriber"],
            "registration_name": "documentation-controlled-subscriber",
            "reason": "disabled by documentation plugin"
        }
    })
    .as_object()
    .expect("config object")
    .clone();
    PluginConfig {
        components: vec![component],
        ..PluginConfig::default()
    }
}

pub fn config(mode: &str) -> PluginConfig {
    config_with_enabled(mode, true)
}

pub fn reset_observed_events() {
    OBSERVED_EVENTS.store(0, Ordering::Relaxed);
    OBSERVED_EVENT_DETAILS
        .lock()
        .expect("event details lock should not be poisoned")
        .clear();
}

pub fn observed_event_count() -> usize {
    OBSERVED_EVENTS.load(Ordering::Relaxed)
}

pub fn observed_events() -> Vec<Event> {
    OBSERVED_EVENT_DETAILS
        .lock()
        .expect("event details lock should not be poisoned")
        .clone()
}

fn redact_event_fields(
    mut fields: EventSanitizeFields,
    redact_keys: &[String],
) -> EventSanitizeFields {
    fields.data = fields.data.map(|value| redact_json(value, redact_keys));
    fields.metadata = fields.metadata.map(|value| redact_json(value, redact_keys));
    fields
}

fn redact_json(value: Json, redact_keys: &[String]) -> Json {
    match value {
        Json::Object(mut object) => {
            for (key, value) in &mut object {
                if redact_keys.iter().any(|candidate| candidate == key) {
                    *value = Json::String("[REDACTED]".into());
                } else {
                    *value = redact_json(value.take(), redact_keys);
                }
            }
            Json::Object(object)
        }
        Json::Array(values) => Json::Array(
            values
                .into_iter()
                .map(|value| redact_json(value, redact_keys))
                .collect(),
        ),
        other => other,
    }
}

fn emit_runtime_events(tag: &str, runtime: &config::Runtime) -> nemo_relay::error::Result<()> {
    if runtime.emit_marks {
        event(
            EmitMarkEventParams::builder()
                .name("documentation-plugin.request")
                .data(json!({ "tag": tag, "secret": "application-value" }))
                .build(),
        )?;
    }
    if runtime.emit_isolated_scope {
        let isolated_stack = create_scope_stack();
        with_scope_stack(isolated_stack, || {
            let scope = push_scope(
                PushScopeParams::builder()
                    .name("documentation-plugin.isolated")
                    .scope_type(ScopeType::Custom)
                    .build(),
            )?;
            pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build())
        })?;
    }
    Ok(())
}

pub async fn run_workflow() -> Result<(), Box<dyn std::error::Error>> {
    reset_observed_events();
    register_plugin(Arc::new(DocumentationPlugin))?;
    println!("registered: {:?}", list_plugin_kinds());
    let invalid = validate_plugin_config(&config("invalid"));
    assert_eq!(
        invalid.diagnostics[0].code,
        "documentation-plugin.unsupported_mode"
    );
    let disabled_invalid = validate_plugin_config(&config_with_enabled("invalid", false));
    assert_eq!(
        disabled_invalid.diagnostics[0].code,
        "documentation-plugin.unsupported_mode"
    );
    println!("invalid: {:?}", invalid.diagnostics);
    let report = initialize_plugins_exact(config("enforce")).await?;
    println!("active: {report:?}");

    let tool = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"value": 1}))
            .func(Arc::new(|args| {
                Box::pin(async move {
                    Ok(ToolExecutionResult::annotated(
                        args,
                        json!({"source": "application"}),
                    ))
                })
            }))
            .build(),
    )
    .await?;
    assert_eq!(tool.result, json!({"value": 1, "plugin_tag": "documentation"}));
    assert_eq!(tool.annotation, Some(json!({"source": "application"})));
    println!("tool: {tool:?}");

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "allowed-model"}),
    };
    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("allowed-model")
            .request(request.clone())
            .func(Arc::new(|request| {
                Box::pin(async move { Ok(json!({"headers": request.headers})) })
            }))
            .build(),
    )
    .await?;
    assert_eq!(response["headers"]["x-nemo-relay-plugin"], "documentation");
    println!("llm: {response}");

    let mut output = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("allowed-model")
            .request(request)
            .func(Arc::new(|_request| {
                Box::pin(async {
                    Ok(LlmJsonStream::new(stream::iter(vec![
                        Ok(json!({"chunk": 1})),
                        Ok(json!({"chunk": 2})),
                    ])))
                })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({"done": true})))
            .build(),
    )
    .await?;
    let mut chunks = Vec::new();
    while let Some(chunk) = output.next().await {
        let chunk = chunk?;
        println!("stream: {chunk}");
        chunks.push(chunk);
    }
    assert_eq!(chunks.len(), 2);
    assert!(chunks.iter().all(|chunk| chunk["plugin_stream"] == true));
    flush_subscribers()?;
    assert!(OBSERVED_EVENTS.load(Ordering::Relaxed) > 0);

    clear_plugin_configuration()?;
    assert!(deregister_plugin("documentation-plugin"));
    assert!(
        !list_plugin_kinds()
            .iter()
            .any(|kind| kind == "documentation-plugin")
    );
    println!("teardown: complete");
    Ok(())
}
