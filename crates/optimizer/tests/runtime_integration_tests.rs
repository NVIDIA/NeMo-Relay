// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};

use nemo_flow_core::{
    LLMAttributes, LLMRequest, LlmExecutionNextFn, LlmStreamExecutionNextFn, NemoFlowContextState,
    ToolAttributes, ToolExecutionNextFn, global_context, nemo_flow_llm_call_execute,
    nemo_flow_llm_request_intercepts, nemo_flow_llm_stream_call_execute,
    nemo_flow_register_subscriber, nemo_flow_tool_call_execute,
};
use nemo_flow_optimizer::{
    BackendSpec, BuildContext, ComponentSpec, ConfigDiagnostic, ConfigPolicy, DiagnosticLevel,
    DynamoHintsComponentConfig, HostedPluginHandler, HostedRegistrationContext, OptimizerComponent,
    OptimizerComponentFactory, OptimizerConfig, OptimizerError, OptimizerRuntime,
    RegistrationContext, Result, StateConfig, TelemetryComponentConfig,
    ToolParallelismComponentConfig, UnsupportedBehavior, ValidationContext,
    deregister_component_factory, deregister_hosted_plugin_handler, register_component_factory,
    register_hosted_plugin_handler,
};
use serde_json::{Map, Value as Json, json};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

static TEST_MUTEX: Mutex<()> = Mutex::const_new(());

fn reset_global() {
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoFlowContextState::new();
}

#[tokio::test]
async fn test_runtime_registers_and_passes_calls_through() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = OptimizerRuntime::new(OptimizerConfig {
        state: Some(StateConfig {
            backend: BackendSpec::in_memory(),
        }),
        components: vec![
            TelemetryComponentConfig {
                subscriber_name: Some("optimizer_test_subscriber".into()),
                learners: vec!["latency_sensitivity".into()],
            }
            .into(),
            DynamoHintsComponentConfig::default().into(),
            ToolParallelismComponentConfig::default().into(),
        ],
        ..OptimizerConfig::default()
    })
    .await
    .unwrap();

    runtime.register().await.unwrap();

    let llm_func: LlmExecutionNextFn =
        Arc::new(|_req: LLMRequest| Box::pin(async { Ok(serde_json::json!({"response": "ok"})) }));
    let llm_result = nemo_flow_llm_call_execute(
        "test-llm",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({"messages": []}),
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
    assert_eq!(llm_result, serde_json::json!({"response": "ok"}));

    let tool_func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let tool_result = nemo_flow_tool_call_execute(
        "search",
        serde_json::json!({"query": "test"}),
        tool_func,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(tool_result, serde_json::json!({"query": "test"}));

    runtime.deregister().unwrap();
}

#[tokio::test]
async fn test_runtime_validate_unknown_component_policy() {
    let report = OptimizerRuntime::validate_config(&OptimizerConfig {
        components: vec![ComponentSpec::new("future_component")],
        ..OptimizerConfig::default()
    });
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "optimizer.unknown_component")
    );

    let err = OptimizerRuntime::new(OptimizerConfig {
        policy: ConfigPolicy {
            unknown_component: UnsupportedBehavior::Error,
            ..ConfigPolicy::default()
        },
        components: vec![ComponentSpec::new("future_component")],
        ..OptimizerConfig::default()
    })
    .await
    .unwrap_err();
    assert!(err.to_string().contains("unsupported"));
}

#[test]
fn test_external_component_warns_when_plugin_kind_is_unknown() {
    let report = OptimizerRuntime::validate_config(&OptimizerConfig {
        components: vec![ComponentSpec {
            kind: "external_component".into(),
            enabled: true,
            config: Map::from_iter([
                ("plugin_kind".into(), json!("missing.plugin")),
                ("instance_id".into(), json!("plugin-1")),
                ("plugin_config".into(), json!({})),
            ]),
        }],
        ..OptimizerConfig::default()
    });
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "optimizer.unknown_plugin_kind")
    );
}

struct NativeTestFactory;

struct NativeTestComponent {
    name: String,
}

impl OptimizerComponentFactory for NativeTestFactory {
    fn kind(&self) -> &'static str {
        "native_test_component"
    }

    fn validate(
        &self,
        _spec: &ComponentSpec,
        _policy: &ConfigPolicy,
        _ctx: &ValidationContext,
    ) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn build(
        &self,
        _spec: &ComponentSpec,
        _ctx: &BuildContext,
    ) -> Result<Box<dyn OptimizerComponent>> {
        Ok(Box::new(NativeTestComponent {
            name: "native_test_component_intercept".into(),
        }))
    }
}

impl OptimizerComponent for NativeTestComponent {
    fn kind(&self) -> &'static str {
        "native_test_component"
    }

    fn register<'a>(
        &'a mut self,
        ctx: &'a mut RegistrationContext<'_>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            ctx.register_llm_request_intercept(
                &self.name,
                10,
                false,
                Box::new(|_name, mut request, annotated| {
                    request
                        .headers
                        .insert("x-native-component".into(), json!(true));
                    Ok((request, annotated))
                }),
            )
        })
    }
}

#[tokio::test]
async fn test_custom_component_factory_registers_runtime_behavior() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();

    register_component_factory(Arc::new(NativeTestFactory)).unwrap();

    let mut runtime = OptimizerRuntime::new(OptimizerConfig {
        components: vec![ComponentSpec::new("native_test_component")],
        ..OptimizerConfig::default()
    })
    .await
    .unwrap();

    runtime.register().await.unwrap();
    let request = nemo_flow_llm_request_intercepts(
        "test-model",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        },
    )
    .unwrap();
    assert_eq!(
        request.headers.get("x-native-component"),
        Some(&json!(true))
    );

    runtime.deregister().unwrap();
    assert!(deregister_component_factory("native_test_component"));
}

struct HeaderPluginHandler;

impl HostedPluginHandler for HeaderPluginHandler {
    fn plugin_kind(&self) -> &str {
        "test.header_plugin"
    }

    fn validate(
        &self,
        _instance_id: &str,
        _plugin_config: &Map<String, Json>,
    ) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register(
        &self,
        instance_id: &str,
        plugin_config: &Map<String, Json>,
        ctx: &mut HostedRegistrationContext,
    ) -> Result<()> {
        let priority = plugin_config
            .get("priority")
            .and_then(|value| value.as_i64())
            .unwrap_or(42) as i32;
        let name = format!("{instance_id}.header_plugin");
        ctx.register_llm_request_intercept(
            &name,
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
            &format!("{instance_id}.tool_request_plugin"),
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
            &format!("{instance_id}.llm_exec_plugin"),
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
            &format!("{instance_id}.llm_stream_exec_plugin"),
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
        )
    }
}

#[tokio::test]
async fn test_hosted_plugin_registers_request_and_execution_intercepts() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();

    register_hosted_plugin_handler(Arc::new(HeaderPluginHandler)).unwrap();

    let mut runtime = OptimizerRuntime::new(OptimizerConfig {
        components: vec![ComponentSpec {
            kind: "external_component".into(),
            enabled: true,
            config: Map::from_iter([
                ("plugin_kind".into(), json!("test.header_plugin")),
                ("instance_id".into(), json!("plugin-1")),
                ("plugin_config".into(), json!({"priority": 7})),
            ]),
        }],
        ..OptimizerConfig::default()
    })
    .await
    .unwrap();

    runtime.register().await.unwrap();
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

    runtime.deregister().unwrap();
    assert!(deregister_hosted_plugin_handler("test.header_plugin"));
}

struct FailingPluginHandler;

impl HostedPluginHandler for FailingPluginHandler {
    fn plugin_kind(&self) -> &str {
        "test.failing_plugin"
    }

    fn validate(
        &self,
        _instance_id: &str,
        _plugin_config: &Map<String, Json>,
    ) -> Vec<ConfigDiagnostic> {
        vec![ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "optimizer.test_warning".into(),
            component: Some("external_component".into()),
            field: None,
            message: "plugin validation executed".into(),
        }]
    }

    fn register(
        &self,
        _instance_id: &str,
        _plugin_config: &Map<String, Json>,
        ctx: &mut HostedRegistrationContext,
    ) -> Result<()> {
        ctx.register_subscriber("failing_plugin_subscriber", Arc::new(|_| {}))?;
        Err(OptimizerError::RegistrationFailed(
            "simulated hosted plugin failure".into(),
        ))
    }
}

#[tokio::test]
async fn test_hosted_plugin_registration_rolls_back_partial_work() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();

    register_hosted_plugin_handler(Arc::new(FailingPluginHandler)).unwrap();

    let mut runtime = OptimizerRuntime::new(OptimizerConfig {
        components: vec![ComponentSpec {
            kind: "external_component".into(),
            enabled: true,
            config: Map::from_iter([
                ("plugin_kind".into(), json!("test.failing_plugin")),
                ("instance_id".into(), json!("plugin-2")),
                ("plugin_config".into(), json!({})),
            ]),
        }],
        ..OptimizerConfig::default()
    })
    .await
    .unwrap();

    let err = runtime.register().await.unwrap_err();
    assert!(err.to_string().contains("simulated hosted plugin failure"));

    nemo_flow_register_subscriber("failing_plugin_subscriber", Arc::new(|_| {})).unwrap();
    nemo_flow_core::nemo_flow_deregister_subscriber("failing_plugin_subscriber").unwrap();

    assert!(deregister_hosted_plugin_handler("test.failing_plugin"));
}
