// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};

use nemo_flow_adaptive::{
    AdaptiveConfig, AdaptiveHintsComponentConfig, BackendSpec, ComponentSpec as AdaptiveComponent,
    ConfigPolicy, StateConfig, TelemetryComponentConfig, ToolParallelismComponentConfig,
    UnsupportedBehavior, register_adaptive_component,
};
use nemo_flow_core::{
    ConfigDiagnostic, DiagnosticLevel, LLMAttributes, LLMRequest, LlmExecutionNextFn,
    LlmStreamExecutionNextFn, NemoFlowContextState, PluginComponentSpec, PluginConfig, PluginError,
    PluginHandler, PluginRegistrationContext, ToolAttributes, ToolExecutionNextFn,
    clear_plugin_configuration, deregister_plugin_handler, global_context, initialize_plugins,
    nemo_flow_llm_call_execute, nemo_flow_llm_request_intercepts,
    nemo_flow_llm_stream_call_execute, nemo_flow_register_subscriber, nemo_flow_tool_call_execute,
    register_plugin_handler, validate_plugin_config,
};
use serde_json::{Map, Value as Json, json};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

static TEST_MUTEX: Mutex<()> = Mutex::const_new(());

fn reset_global() {
    let _ = clear_plugin_configuration();
    let _ = deregister_plugin_handler("test.header_plugin");
    let _ = deregister_plugin_handler("test.failing_plugin");

    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoFlowContextState::new();
}

#[tokio::test]
async fn test_adaptive_plugin_registers_and_passes_calls_through() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();

    let report = initialize_plugins(PluginConfig {
        components: vec![
            AdaptiveComponent::new(AdaptiveConfig {
                state: Some(StateConfig {
                    backend: BackendSpec::in_memory(),
                }),
                telemetry: Some(TelemetryComponentConfig {
                    subscriber_name: Some("adaptive_test_subscriber".into()),
                    learners: vec!["latency_sensitivity".into()],
                }),
                adaptive_hints: Some(AdaptiveHintsComponentConfig::default()),
                tool_parallelism: Some(ToolParallelismComponentConfig::default()),
                ..AdaptiveConfig::default()
            })
            .into(),
        ],
        ..PluginConfig::default()
    })
    .await
    .unwrap();
    assert!(report.diagnostics.is_empty());

    let request = nemo_flow_llm_request_intercepts(
        "test-model",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        },
    )
    .unwrap();
    assert_eq!(request.content["messages"], json!([]));

    let llm_func: LlmExecutionNextFn =
        Arc::new(|_req: LLMRequest| Box::pin(async { Ok(json!({"response": "ok"})) }));
    let llm_result = nemo_flow_llm_call_execute(
        "test-llm",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        },
        llm_func,
        None,
        LLMAttributes::empty(),
        None,
        None,
        Some("gpt-4".into()),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(llm_result, json!({"response": "ok"}));

    let tool_func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let tool_result = nemo_flow_tool_call_execute(
        "search",
        json!({"query": "test"}),
        tool_func,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(tool_result, json!({"query": "test"}));

    clear_plugin_configuration().unwrap();
}

#[test]
fn test_adaptive_plugin_validation_reports_missing_state_and_unknown_fields() {
    register_adaptive_component().unwrap();

    let report = validate_plugin_config(&PluginConfig {
        components: vec![PluginComponentSpec {
            kind: "adaptive".into(),
            enabled: true,
            config: Map::from_iter([
                ("version".into(), json!(1)),
                (
                    "telemetry".into(),
                    json!({"learners": ["latency_sensitivity"]}),
                ),
                (
                    "adaptive_hints".into(),
                    json!({"inject_header": true, "unknown_flag": true}),
                ),
            ]),
        }],
        ..PluginConfig::default()
    });

    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "adaptive.section_disabled_missing_state")
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "adaptive.unknown_field")
    );
}

#[tokio::test]
async fn test_adaptive_plugin_rejects_unsupported_mode_with_strict_policy() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();

    let err = initialize_plugins(PluginConfig {
        components: vec![
            AdaptiveComponent::new(AdaptiveConfig {
                policy: ConfigPolicy {
                    unsupported_value: UnsupportedBehavior::Error,
                    ..ConfigPolicy::default()
                },
                tool_parallelism: Some(ToolParallelismComponentConfig {
                    priority: 100,
                    mode: "broken".into(),
                }),
                ..AdaptiveConfig::default()
            })
            .into(),
        ],
        ..PluginConfig::default()
    })
    .await
    .unwrap_err();

    assert!(err.to_string().contains("unsupported"));
}

struct HeaderPluginHandler;

impl PluginHandler for HeaderPluginHandler {
    fn plugin_kind(&self) -> &str {
        "test.header_plugin"
    }

    fn allows_multiple_components(&self) -> bool {
        true
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn std::future::Future<Output = std::result::Result<(), PluginError>> + Send + 'a>>
    {
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            let priority = plugin_config
                .get("priority")
                .and_then(|value| value.as_i64())
                .unwrap_or(42) as i32;
            ctx.register_llm_request_intercept(
                "header_plugin",
                priority,
                false,
                Box::new(|_name, mut request, annotated| {
                    request
                        .headers
                        .insert("x-hosted-plugin".into(), json!("set"));
                    Ok((request, annotated))
                }),
            )?;
            ctx.register_tool_request_intercept(
                "tool_request_plugin",
                priority,
                false,
                Box::new(|_name, mut args| {
                    if let Json::Object(ref mut map) = args {
                        map.insert("x-hosted-tool-plugin".into(), json!(true));
                    }
                    Ok(args)
                }),
            )?;
            ctx.register_llm_execution_intercept(
                "llm_exec_plugin",
                priority,
                Arc::new(|_name, request, next| {
                    Box::pin(async move {
                        let mut response = next(request).await?;
                        if let Json::Object(ref mut map) = response {
                            map.insert("x-hosted-llm-exec".into(), json!(true));
                        }
                        Ok(response)
                    })
                }),
            )?;
            ctx.register_llm_stream_execution_intercept(
                "llm_stream_exec_plugin",
                priority,
                Arc::new(|_name, request, next| {
                    Box::pin(async move {
                        let mut stream = next(request).await?;
                        let mut chunks = Vec::new();
                        while let Some(item) = stream.next().await {
                            let mut chunk = item?;
                            if let Json::Object(ref mut map) = chunk {
                                map.insert("x-hosted-llm-stream-exec".into(), json!(true));
                            }
                            chunks.push(Ok(chunk));
                        }
                        let stream = Box::pin(tokio_stream::iter(chunks))
                            as Pin<
                                Box<
                                    dyn tokio_stream::Stream<Item = nemo_flow_core::Result<Json>>
                                        + Send,
                                >,
                            >;
                        Ok(stream)
                    })
                }),
            )?;
            Ok(())
        })
    }
}

#[tokio::test]
async fn test_top_level_plugin_registers_request_and_execution_intercepts() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();
    register_plugin_handler(Arc::new(HeaderPluginHandler)).unwrap();

    initialize_plugins(PluginConfig {
        components: vec![
            AdaptiveComponent::new(AdaptiveConfig {
                adaptive_hints: Some(AdaptiveHintsComponentConfig::default()),
                ..AdaptiveConfig::default()
            })
            .into(),
            PluginComponentSpec {
                kind: "test.header_plugin".into(),
                enabled: true,
                config: Map::from_iter([("priority".into(), json!(7))]),
            },
        ],
        ..PluginConfig::default()
    })
    .await
    .unwrap();

    let request = nemo_flow_llm_request_intercepts(
        "test-model",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        },
    )
    .unwrap();
    assert_eq!(request.headers.get("x-hosted-plugin"), Some(&json!("set")));

    let tool_func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let tool_result = nemo_flow_tool_call_execute(
        "search",
        json!({"query": "test"}),
        tool_func,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(tool_result["x-hosted-tool-plugin"], json!(true));

    let llm_func: LlmExecutionNextFn =
        Arc::new(|_req: LLMRequest| Box::pin(async move { Ok(json!({"response": "ok"})) }));
    let llm_result = nemo_flow_llm_call_execute(
        "test-llm",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        },
        llm_func,
        None,
        LLMAttributes::empty(),
        None,
        None,
        Some("gpt-4".into()),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(llm_result["x-hosted-llm-exec"], json!(true));

    let llm_stream_func: LlmStreamExecutionNextFn = Arc::new(|_req: LLMRequest| {
        Box::pin(async move {
            let chunks = vec![Ok(json!({"streamed": true}))];
            Ok(Box::pin(tokio_stream::iter(chunks))
                as Pin<
                    Box<dyn tokio_stream::Stream<Item = nemo_flow_core::Result<Json>> + Send>,
                >)
        })
    });
    let collected = Arc::new(StdMutex::new(Vec::new()));
    let collected_for_closure = collected.clone();
    let mut stream = nemo_flow_llm_stream_call_execute(
        "test-stream-llm",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        },
        llm_stream_func,
        Box::new(move |chunk| {
            collected_for_closure.lock().unwrap().push(chunk);
            Ok(())
        }),
        Box::new(|| json!({"final": true})),
        None,
        LLMAttributes::empty(),
        None,
        None,
        Some("gpt-4".into()),
        None,
        None,
    )
    .await
    .unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first["x-hosted-llm-stream-exec"], json!(true));
    assert_eq!(
        collected.lock().unwrap()[0]["x-hosted-llm-stream-exec"],
        json!(true)
    );

    clear_plugin_configuration().unwrap();
    assert!(deregister_plugin_handler("test.header_plugin"));
}

struct FailingPluginHandler;

impl PluginHandler for FailingPluginHandler {
    fn plugin_kind(&self) -> &str {
        "test.failing_plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "plugin.test_warning".into(),
            component: Some("test.failing_plugin".into()),
            field: None,
            message: "plugin validation executed".into(),
        }]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn std::future::Future<Output = std::result::Result<(), PluginError>> + Send + 'a>>
    {
        Box::pin(async move {
            ctx.register_subscriber("failing_plugin_subscriber", Arc::new(|_| {}))?;
            Err(PluginError::RegistrationFailed(
                "simulated hosted plugin failure".into(),
            ))
        })
    }
}

#[tokio::test]
async fn test_top_level_plugin_registration_rolls_back_partial_work() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();

    register_plugin_handler(Arc::new(FailingPluginHandler)).unwrap();

    let err = initialize_plugins(PluginConfig {
        components: vec![PluginComponentSpec {
            kind: "test.failing_plugin".into(),
            enabled: true,
            config: Map::new(),
        }],
        ..PluginConfig::default()
    })
    .await
    .unwrap_err();
    assert!(err.to_string().contains("simulated hosted plugin failure"));

    nemo_flow_register_subscriber("failing_plugin_subscriber", Arc::new(|_| {})).unwrap();
    nemo_flow_core::nemo_flow_deregister_subscriber("failing_plugin_subscriber").unwrap();

    assert!(deregister_plugin_handler("test.failing_plugin"));
}
