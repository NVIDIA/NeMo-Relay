// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod config;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::{StreamExt, stream};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmRequestInterceptOutcome, LlmStreamCallExecuteParams,
    llm_call_execute, llm_stream_call_execute,
};
use nemo_relay::api::runtime::callbacks::LlmJsonStream;
use nemo_relay::api::tool::{ToolCallExecuteParams, tool_call_execute};
use nemo_relay::plugin::{
    ConfigDiagnostic, ConfigPolicy, Plugin, PluginComponentSpec, PluginConfig,
    PluginRegistrationContext, Result as PluginResult, clear_plugin_configuration,
    deregister_plugin, initialize_plugins_exact, list_plugin_kinds, register_plugin,
    validate_plugin_config,
};
use serde_json::{Map, Value as Json, json};

pub struct DocumentationPlugin;

static OBSERVED_EVENTS: AtomicUsize = AtomicUsize::new(0);

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
            let _documented_controls = (
                &settings.observe.redact_keys,
                settings.execution.emit_pending_marks,
                settings.runtime.emit_marks,
                settings.runtime.emit_isolated_scope,
            );
            if settings.observe.enabled {
                context.register_subscriber(
                    "events",
                    Arc::new(|event| {
                        OBSERVED_EVENTS.fetch_add(1, Ordering::Relaxed);
                        println!("event: {}", event.name());
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
            if settings.execution.enabled {
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
        "runtime": { "emit_marks": true, "emit_isolated_scope": true }
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
}

pub fn observed_event_count() -> usize {
    OBSERVED_EVENTS.load(Ordering::Relaxed)
}

pub async fn run_workflow() -> Result<(), Box<dyn std::error::Error>> {
    OBSERVED_EVENTS.store(0, Ordering::Relaxed);
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
            .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
            .build(),
    )
    .await?;
    assert_eq!(tool, json!({"value": 1, "plugin_tag": "documentation"}));
    println!("tool: {tool}");

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
