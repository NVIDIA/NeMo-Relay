// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for runtime integration in the NeMo Relay adaptive crate.

use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex, RwLock};

use chrono::Utc;
use nemo_relay::api::event::Event;
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmStreamCallExecuteParams, llm_call_execute, llm_request_intercepts,
    llm_stream_call_execute,
};
use nemo_relay::api::runtime::NemoRelayContextState;
use nemo_relay::api::runtime::global_context;
use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::tool_call_execute;
use nemo_relay::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use nemo_relay::codec::response::AnnotatedLlmResponse;
use nemo_relay::codec::traits::LlmResponseCodec;
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginComponentSpec, PluginConfig, PluginError,
    PluginRegistrationContext, clear_plugin_configuration, deregister_plugin, initialize_plugins,
    register_plugin, validate_plugin_config,
};
use nemo_relay::plugin::{ConfigPolicy, UnsupportedBehavior};
use nemo_relay_adaptive::acg::{StabilityThresholds, analyze_stability, build_prompt_ir};
use nemo_relay_adaptive::acg_learner::AcgLearner;
use nemo_relay_adaptive::cache_diagnostics::{CacheDiagnosticsTracker, build_cache_request_facts};
use nemo_relay_adaptive::config::{
    AdaptiveConfig, AdaptiveHintsComponentConfig, BackendSpec, StateConfig,
    TelemetryComponentConfig, ToolParallelismComponentConfig,
};
use nemo_relay_adaptive::learner::traits::Learner;
use nemo_relay_adaptive::plugin_component::{
    ComponentSpec as AdaptiveComponent, register_adaptive_component,
};
use nemo_relay_adaptive::types::cache::HotCache;
use nemo_relay_adaptive::types::records::{CallKind, CallRecord, RunRecord};
use nemo_relay_adaptive::{InMemoryBackend, StorageBackendDyn};
use serde_json::{Map, Value as Json, json};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use uuid::Uuid;

static TEST_MUTEX: Mutex<()> = Mutex::const_new(());

fn short_hash(value: &str) -> &str {
    value.get(..16).unwrap_or(value)
}

fn reset_global() {
    let _ = clear_plugin_configuration();
    let _ = deregister_plugin("test.header_plugin");
    let _ = deregister_plugin("test.failing_plugin");

    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

fn sample_annotated_request(model: &str) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are a careful planner".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Summarize the latest findings".to_string()),
                name: None,
            },
        ],
        model: Some(model.to_string()),
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: Map::new(),
    }
}

fn sample_growing_chat_requests(model: &str) -> Vec<AnnotatedLlmRequest> {
    vec![
        AnnotatedLlmRequest {
            messages: vec![
                Message::System {
                    content: MessageContent::Text("You are a careful planner".to_string()),
                    name: None,
                },
                Message::User {
                    content: MessageContent::Text("Summarize the latest findings".to_string()),
                    name: None,
                },
            ],
            model: Some(model.to_string()),
            params: None,
            tools: None,
            tool_choice: None,
            store: None,
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user: None,
            metadata: None,
            service_tier: None,
            parallel_tool_calls: None,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            stream: None,
            extra: Map::new(),
        },
        AnnotatedLlmRequest {
            messages: vec![
                Message::System {
                    content: MessageContent::Text("You are a careful planner".to_string()),
                    name: None,
                },
                Message::User {
                    content: MessageContent::Text("Summarize the latest findings".to_string()),
                    name: None,
                },
                Message::Assistant {
                    content: Some(MessageContent::Text(
                        "The findings are stable so far.".to_string(),
                    )),
                    tool_calls: None,
                    name: None,
                },
                Message::User {
                    content: MessageContent::Text("Continue with the next batch.".to_string()),
                    name: None,
                },
            ],
            model: Some(model.to_string()),
            params: None,
            tools: None,
            tool_choice: None,
            store: None,
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user: None,
            metadata: None,
            service_tier: None,
            parallel_tool_calls: None,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            stream: None,
            extra: Map::new(),
        },
        AnnotatedLlmRequest {
            messages: vec![
                Message::System {
                    content: MessageContent::Text("You are a careful planner".to_string()),
                    name: None,
                },
                Message::User {
                    content: MessageContent::Text("Summarize the latest findings".to_string()),
                    name: None,
                },
                Message::Assistant {
                    content: Some(MessageContent::Text(
                        "The findings are stable so far.".to_string(),
                    )),
                    tool_calls: None,
                    name: None,
                },
                Message::User {
                    content: MessageContent::Text("Continue with the next batch.".to_string()),
                    name: None,
                },
                Message::Assistant {
                    content: Some(MessageContent::Text(
                        "The second batch matches the first.".to_string(),
                    )),
                    tool_calls: None,
                    name: None,
                },
                Message::User {
                    content: MessageContent::Text("Prepare the final summary.".to_string()),
                    name: None,
                },
            ],
            model: Some(model.to_string()),
            params: None,
            tools: None,
            tool_choice: None,
            store: None,
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user: None,
            metadata: None,
            service_tier: None,
            parallel_tool_calls: None,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            stream: None,
            extra: Map::new(),
        },
    ]
}

fn empty_hot_cache() -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }))
}

fn sample_run_with_request(agent_id: &str, annotated_request: AnnotatedLlmRequest) -> RunRecord {
    let started_at = Utc::now();

    RunRecord {
        id: Uuid::now_v7(),
        agent_id: agent_id.to_string(),
        calls: vec![CallRecord {
            kind: CallKind::Llm,
            name: "planner".to_string(),
            started_at,
            ended_at: Some(started_at),
            metadata_snapshot: None,
            output_tokens: Some(128),
            prompt_tokens: Some(32),
            total_tokens: Some(160),
            model_name: Some("claude-3-5-sonnet".to_string()),
            tool_call_count: None,
            annotated_request: Some(Arc::new(annotated_request)),
            annotated_response: None,
        }],
        started_at,
        ended_at: Some(started_at),
    }
}

fn sample_run_with_requests(
    agent_id: &str,
    annotated_requests: Vec<AnnotatedLlmRequest>,
) -> RunRecord {
    let started_at = Utc::now();

    RunRecord {
        id: Uuid::now_v7(),
        agent_id: agent_id.to_string(),
        calls: annotated_requests
            .into_iter()
            .enumerate()
            .map(|(index, annotated_request)| CallRecord {
                kind: CallKind::Llm,
                name: format!("planner-{index}"),
                started_at,
                ended_at: Some(started_at),
                metadata_snapshot: None,
                output_tokens: Some(128),
                prompt_tokens: Some(32),
                total_tokens: Some(160),
                model_name: Some("claude-3-5-sonnet".to_string()),
                tool_call_count: None,
                annotated_request: Some(Arc::new(annotated_request)),
                annotated_response: None,
            })
            .collect(),
        started_at,
        ended_at: Some(started_at),
    }
}

#[test]
fn acg_component_source_resolves_request_surfaces_instead_of_decoding_from_provider() {
    let source = include_str!("../../src/acg_component.rs");

    assert!(
        source.contains("resolve_request_surface_from_request"),
        "acg_component should resolve request surfaces independently from provider selection",
    );
    assert!(
        !source.contains("decode_request_for_provider"),
        "acg_component should no longer decode semantic requests from the configured provider",
    );
}

#[test]
fn runtime_integration_acg_component_source_keeps_response_codecs_observability_only() {
    let source = include_str!("../../src/acg_component.rs");

    assert!(
        source.contains("Response codecs stay on the observability path after provider execution."),
        "acg_component should document the request-hint vs response-observability boundary",
    );
    assert!(
        !source.contains("response_codec"),
        "acg_component request hint translation should not branch on response codecs",
    );
}

#[test]
fn runtime_integration_acg_component_source_passes_through_on_request_surface_mismatch() {
    let source = include_str!("../../src/acg_component.rs");

    assert!(
        source.contains("request surface"),
        "acg_component should keep an explicit request surface compatibility boundary",
    );
    assert!(
        source.contains("request surface mismatches pass through unchanged"),
        "acg_component should document that request surface mismatches pass through unchanged in the production fallback path",
    );
    assert!(
        source.contains("unwrap_or(request)"),
        "runtime mismatch fallback should preserve the original request unchanged",
    );
}

struct FailingResponseCodec;

impl LlmResponseCodec for FailingResponseCodec {
    fn decode_response(
        &self,
        _response: &nemo_relay::json::Json,
    ) -> FlowResult<AnnotatedLlmResponse> {
        Err(FlowError::Internal(
            "response annotation intentionally failed for test".to_string(),
        ))
    }
}

#[tokio::test(flavor = "current_thread")]
async fn runtime_integration_response_codec_decode_failure_keeps_annotations_optional() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();

    let captured_events = Arc::new(StdMutex::new(Vec::<Event>::new()));
    let captured_for_subscriber = captured_events.clone();
    register_subscriber(
        "response-codec-optional",
        Arc::new(move |event| {
            captured_for_subscriber.lock().unwrap().push(event.clone());
        }),
    )
    .unwrap();

    let llm_func: LlmExecutionNextFn = Arc::new(|request: LlmRequest| {
        Box::pin(async move { Ok(json!({"response": "ok", "echo": request.content})) })
    });

    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test-llm-response-codec")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": [{"role": "user", "content": "hello"}]}),
            })
            .func(llm_func)
            .model_name("gpt-4o")
            .response_codec(Arc::new(FailingResponseCodec))
            .build(),
    )
    .await
    .expect("response codec failures should not fail execution");

    assert_eq!(response["response"], json!("ok"));

    flush_subscribers().unwrap();
    let end_event = captured_events
        .lock()
        .unwrap()
        .iter()
        .find(|event| {
            event.scope_type() == Some(nemo_relay::api::scope::ScopeType::Llm)
                && event.scope_category() == Some(nemo_relay::api::event::ScopeCategory::End)
        })
        .cloned()
        .expect("llm end event should still emit");
    assert!(
        end_event.annotated_response().is_none(),
        "failed response decode should leave annotated_response optional",
    );

    deregister_subscriber("response-codec-optional").unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn runtime_integration_in_memory_backend_round_trips_canonical_acg_payloads() {
    let backend = InMemoryBackend::new();
    let request = sample_annotated_request("claude-3-5-sonnet");
    let prompt_ir = build_prompt_ir(&request).expect("request should build canonical PromptIR");
    let stability = analyze_stability(
        std::slice::from_ref(&prompt_ir),
        &StabilityThresholds::default(),
    );

    backend
        .store_observations("agent-memory-roundtrip", std::slice::from_ref(&prompt_ir))
        .await
        .expect("canonical PromptIR should store in-memory");
    backend
        .store_stability("agent-memory-roundtrip", &stability)
        .await
        .expect("canonical stability should store in-memory");

    let loaded_observations = backend
        .load_observations("agent-memory-roundtrip")
        .await
        .expect("canonical PromptIR should load from in-memory");
    let loaded_stability = backend
        .load_stability("agent-memory-roundtrip")
        .await
        .expect("canonical stability should load from in-memory");

    assert_eq!(loaded_observations, Some(vec![prompt_ir]));
    assert_eq!(loaded_stability, Some(stability));
}

#[tokio::test(flavor = "current_thread")]
async fn runtime_integration_acg_learner_persists_agent_seed_state_for_runtime_hydration() {
    let agent_id = "agent-runtime-hydration";
    let backend = InMemoryBackend::new();
    let hot_cache = empty_hot_cache();
    let request = sample_annotated_request("claude-3-5-sonnet");
    let learner = AcgLearner::new(agent_id, 8, StabilityThresholds::default());

    learner
        .process_run(
            &sample_run_with_request(agent_id, request.clone()),
            &backend,
            &hot_cache,
        )
        .await
        .expect("ACG learner should persist canonical state");

    let persisted_observations = backend
        .load_observations(agent_id)
        .await
        .expect("runtime seed observations should load by agent id")
        .expect("runtime seeding requires an aggregate observations entry");
    let persisted_stability = backend
        .load_stability(agent_id)
        .await
        .expect("runtime seed stability should load by agent id")
        .expect("runtime seeding requires an aggregate stability entry");

    let seeded_hot_cache = empty_hot_cache();
    {
        let mut guard = seeded_hot_cache.write().unwrap();
        guard.acg_stability = Some(persisted_stability.clone());
        guard.acg_observation_count = persisted_observations.len() as u32;
    }

    let tracker = Arc::new(RwLock::new(CacheDiagnosticsTracker::default()));
    let facts = build_cache_request_facts(
        agent_id,
        "passthrough",
        &request,
        &seeded_hot_cache,
        &tracker,
    )
    .expect("hydrated hot cache should produce cache request facts");

    assert_eq!(
        facts.stable_prefix_length,
        persisted_stability.stable_prefix_length
    );
    assert!(
        !facts
            .missing_facts
            .contains(&"acg_stability_unavailable".to_string()),
        "hydrated hot cache should expose persisted ACG stability"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn runtime_integration_acg_learner_reuses_learning_buckets_across_growing_chat_turns() {
    let agent_id = "agent-growing-chat";
    let backend = InMemoryBackend::new();
    let hot_cache = empty_hot_cache();
    let requests = sample_growing_chat_requests("claude-3-5-sonnet");
    let learner = AcgLearner::new(agent_id, 8, StabilityThresholds::default());
    let learning_key = format!(
        "{agent_id}::model=claude-3-5-sonnet::seed={}::system={}::tools=no-tools",
        short_hash(&format!(
            "user:{}",
            nemo_relay_adaptive::acg::sha256_hex("Summarize the latest findings")
        )),
        short_hash(&nemo_relay_adaptive::acg::sha256_hex(
            "You are a careful planner"
        )),
    );

    learner
        .process_run(
            &sample_run_with_requests(agent_id, requests),
            &backend,
            &hot_cache,
        )
        .await
        .expect("ACG learner should aggregate growing chat turns into one learning bucket");

    let persisted_observations = backend
        .load_observations(&learning_key)
        .await
        .expect("learning-bucket observations should load")
        .expect("learning-bucket observations should be stored");
    let persisted_stability = backend
        .load_stability(&learning_key)
        .await
        .expect("learning-bucket stability should load")
        .expect("learning-bucket stability should be stored");

    assert_eq!(persisted_observations.len(), 3);
    assert_eq!(persisted_stability.total_observations, 3);

    let guard = hot_cache.read().unwrap();
    assert_eq!(
        guard.acg_profile_observation_counts.get(&learning_key),
        Some(&3)
    );
    assert_eq!(guard.acg_observation_count, 3);
    assert_eq!(
        guard
            .acg_profiles
            .get(&learning_key)
            .map(|stability| stability.total_observations),
        Some(3)
    );
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

    let request = llm_request_intercepts(
        "test-model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        },
    )
    .unwrap();
    assert_eq!(request.content["messages"], json!([]));

    let llm_func: LlmExecutionNextFn =
        Arc::new(|_req: LlmRequest| Box::pin(async { Ok(json!({"response": "ok"})) }));
    let llm_result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test-llm")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": []}),
            })
            .func(llm_func)
            .model_name("gpt-4")
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(llm_result, json!({"response": "ok"}));

    let tool_func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let tool_result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("search")
            .args(json!({"query": "test"}))
            .func(tool_func)
            .build(),
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

struct HeaderPlugin;

impl Plugin for HeaderPlugin {
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
                Arc::new(|_name, mut request, annotated| {
                    request.headers.insert("x-plugin".into(), json!("set"));
                    Ok((request, annotated))
                }),
            )?;
            ctx.register_tool_request_intercept(
                "tool_request_plugin",
                priority,
                false,
                Arc::new(|_name, mut args| {
                    if let Json::Object(ref mut map) = args {
                        map.insert("x-tool-plugin".into(), json!(true));
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
                            map.insert("x-plugin-llm-exec".into(), json!(true));
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
                                map.insert("x-plugin-llm-stream-exec".into(), json!(true));
                            }
                            chunks.push(Ok(chunk));
                        }
                        let stream = Box::pin(tokio_stream::iter(chunks))
                            as Pin<Box<dyn tokio_stream::Stream<Item = FlowResult<Json>> + Send>>;
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
    register_plugin(Arc::new(HeaderPlugin)).unwrap();

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

    let request = llm_request_intercepts(
        "test-model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        },
    )
    .unwrap();
    assert_eq!(request.headers.get("x-plugin"), Some(&json!("set")));

    let tool_func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let tool_result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("search")
            .args(json!({"query": "test"}))
            .func(tool_func)
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(tool_result["x-tool-plugin"], json!(true));

    let llm_func: LlmExecutionNextFn =
        Arc::new(|_req: LlmRequest| Box::pin(async move { Ok(json!({"response": "ok"})) }));
    let llm_result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test-llm")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": []}),
            })
            .func(llm_func)
            .model_name("gpt-4")
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(llm_result["x-plugin-llm-exec"], json!(true));

    let llm_stream_func: LlmStreamExecutionNextFn = Arc::new(|_req: LlmRequest| {
        Box::pin(async move {
            let chunks = vec![Ok(json!({"streamed": true}))];
            Ok(Box::pin(tokio_stream::iter(chunks))
                as Pin<
                    Box<dyn tokio_stream::Stream<Item = FlowResult<Json>> + Send>,
                >)
        })
    });
    let collected = Arc::new(StdMutex::new(Vec::new()));
    let collected_for_closure = collected.clone();
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("test-stream-llm")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": []}),
            })
            .func(llm_stream_func)
            .collector(Box::new(move |chunk| {
                collected_for_closure.lock().unwrap().push(chunk);
                Ok(())
            }))
            .finalizer(Box::new(|| json!({"final": true})))
            .model_name("gpt-4")
            .build(),
    )
    .await
    .unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first["x-plugin-llm-stream-exec"], json!(true));
    assert_eq!(
        collected.lock().unwrap()[0]["x-plugin-llm-stream-exec"],
        json!(true)
    );

    clear_plugin_configuration().unwrap();
    assert!(deregister_plugin("test.header_plugin"));
}

struct FailingPlugin;

impl Plugin for FailingPlugin {
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
                "simulated plugin failure".into(),
            ))
        })
    }
}

#[tokio::test]
async fn test_top_level_plugin_registration_rolls_back_partial_work() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();

    register_plugin(Arc::new(FailingPlugin)).unwrap();

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
    assert!(err.to_string().contains("simulated plugin failure"));

    register_subscriber("failing_plugin_subscriber", Arc::new(|_| {})).unwrap();
    deregister_subscriber("failing_plugin_subscriber").unwrap();

    assert!(deregister_plugin("test.failing_plugin"));
}
