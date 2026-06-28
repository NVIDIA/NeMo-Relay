// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for runtime features in the NeMo Relay adaptive crate.

use super::*;

use std::sync::Arc;

use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::SpanId;
use crate::acg::stability::StabilityAnalysisResult;
use crate::config::{BackendSpec, StateConfig};
use crate::intercepts::AGENT_HINTS_HEADER_KEY;
use crate::test_support::GLOBAL_RUNTIME_TEST_MUTEX;
use crate::trie::accumulator::AccumulatorState;
use crate::trie::serialization::TrieEnvelope;
use crate::types::metadata::{AgentHints, MetadataEnvelope, ParallelHint};
use crate::types::plan::{ExecutionPlan, ParallelGroup};
use crate::types::records::RunRecord;
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmStreamCallExecuteParams, llm_call_execute,
    llm_request_intercepts, llm_stream_call_execute,
};
use nemo_relay::api::registry::{
    deregister_llm_execution_intercept, deregister_llm_request_intercept,
    deregister_llm_stream_execution_intercept, deregister_tool_execution_intercept,
    register_llm_execution_intercept, register_llm_request_intercept,
    register_llm_stream_execution_intercept, register_tool_execution_intercept,
};
use nemo_relay::api::runtime::ToolExecutionNextFn;
use nemo_relay::api::runtime::global_context;
use nemo_relay::api::runtime::{
    LlmExecutionNextFn, LlmStreamExecutionNextFn, NemoRelayContextState,
};
use nemo_relay::api::subscriber::{deregister_subscriber, register_subscriber};
use nemo_relay::api::tool::tool_call_execute;
use nemo_relay::codec::anthropic::AnthropicMessagesCodec;
use nemo_relay::codec::traits::LlmCodec;
use nemo_relay::error::FlowError;
use nemo_relay::plugin::{ConfigPolicy, DiagnosticLevel, UnsupportedBehavior};
use nemo_relay::plugin::{clear_plugin_configuration, rollback_registrations};
use serde_json::json;
use tokio_stream::StreamExt;

fn reset_global() {
    let _ = clear_plugin_configuration();
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

fn sample_plan(agent_id: &str) -> ExecutionPlan {
    ExecutionPlan {
        agent_id: agent_id.to_string(),
        parallel_groups: vec![ParallelGroup {
            group_id: "group-a".to_string(),
            tool_names: vec!["search".to_string()],
        }],
        metadata_template: MetadataEnvelope {
            run_id: Uuid::now_v7(),
            agent_id: agent_id.to_string(),
            parallel_hints: vec![ParallelHint {
                tool_name: "search".to_string(),
                group_id: "group-a".to_string(),
                explicit: true,
            }],
            extensions: json!({}),
        },
    }
}

fn long_text(token_count: usize) -> String {
    "x".repeat(token_count * 4)
}

fn layered_acg_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4-20250514",
            "system": long_text(1400),
            "messages": [
                {"role": "user", "content": long_text(1500)},
                {"role": "user", "content": long_text(1600)}
            ]
        }),
    }
}

fn layered_acg_stability_result(observation_count: u32) -> StabilityAnalysisResult {
    let annotated_request = AnthropicMessagesCodec
        .decode(&layered_acg_request())
        .expect("layered request should decode");
    let prompt_ir = crate::acg::ir_builder::build_prompt_ir(&annotated_request)
        .expect("layered request should build PromptIR");

    StabilityAnalysisResult {
        scores: vec![
            BlockStabilityScore {
                span_id: SpanId("block-0".to_string()),
                classification: StabilityClass::Stable,
                score: 0.99,
                confidence: 0.95,
                observation_count,
            },
            BlockStabilityScore {
                span_id: SpanId("block-1".to_string()),
                classification: StabilityClass::Stable,
                score: 0.99,
                confidence: 0.9,
                observation_count,
            },
            BlockStabilityScore {
                span_id: SpanId("block-2".to_string()),
                classification: StabilityClass::Stable,
                score: 0.99,
                confidence: 0.85,
                observation_count,
            },
        ],
        stable_prefix_length: 3,
        stable_prefix_fingerprint: crate::acg::stability::prompt_prefix_fingerprint(&prompt_ir, 3),
        total_observations: observation_count,
        converged: false,
    }
}

fn assert_already_registered(result: nemo_relay::error::Result<()>, name: &str) {
    match result {
        Err(FlowError::AlreadyExists(message)) => assert!(message.contains(name)),
        other => panic!("expected {name} to be registered, got {other:?}"),
    }
}

fn assert_subscriber_registered(name: &str) {
    assert_already_registered(register_subscriber(name, Arc::new(|_event| {})), name);
}

fn assert_subscriber_absent(name: &str) {
    register_subscriber(name, Arc::new(|_event| {})).unwrap();
    deregister_subscriber(name).unwrap();
}

fn assert_llm_request_intercept_registered(name: &str) {
    assert_already_registered(
        register_llm_request_intercept(
            name,
            i32::MAX,
            false,
            Arc::new(|_name, request, annotated| Ok((request, annotated))),
        ),
        name,
    );
}

fn assert_llm_request_intercept_absent(name: &str) {
    register_llm_request_intercept(
        name,
        i32::MAX,
        false,
        Arc::new(|_name, request, annotated| Ok((request, annotated))),
    )
    .unwrap();
    deregister_llm_request_intercept(name).unwrap();
}

fn assert_llm_execution_intercept_registered(name: &str) {
    assert_already_registered(
        register_llm_execution_intercept(
            name,
            i32::MAX,
            Arc::new(|_name, request, next| next(request)),
        ),
        name,
    );
}

fn assert_llm_execution_intercept_absent(name: &str) {
    register_llm_execution_intercept(
        name,
        i32::MAX,
        Arc::new(|_name, request, next| next(request)),
    )
    .unwrap();
    deregister_llm_execution_intercept(name).unwrap();
}

fn assert_llm_stream_execution_intercept_registered(name: &str) {
    assert_already_registered(
        register_llm_stream_execution_intercept(
            name,
            i32::MAX,
            Arc::new(|_name, request, next| next(request)),
        ),
        name,
    );
}

fn assert_llm_stream_execution_intercept_absent(name: &str) {
    register_llm_stream_execution_intercept(
        name,
        i32::MAX,
        Arc::new(|_name, request, next| next(request)),
    )
    .unwrap();
    deregister_llm_stream_execution_intercept(name).unwrap();
}

fn assert_tool_execution_intercept_registered(name: &str) {
    assert_already_registered(
        register_tool_execution_intercept(name, i32::MAX, Arc::new(|_name, args, next| next(args))),
        name,
    );
}

fn assert_tool_execution_intercept_absent(name: &str) {
    register_tool_execution_intercept(name, i32::MAX, Arc::new(|_name, args, next| next(args)))
        .unwrap();
    deregister_tool_execution_intercept(name).unwrap();
}

struct SeedFailBackend;

impl StorageBackendDyn for SeedFailBackend {
    fn store_run_dyn<'a>(
        &'a self,
        _record: &'a RunRecord,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn load_plan_dyn<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ExecutionPlan>>> + Send + 'a>> {
        Box::pin(async { Err(AdaptiveError::Storage("seed failed".into())) })
    }

    fn list_runs_dyn<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RunRecord>>> + Send + 'a>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn store_trie<'a>(
        &'a self,
        _agent_id: &'a str,
        _envelope: &'a TrieEnvelope,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn load_trie<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<TrieEnvelope>>> + Send + 'a>> {
        Box::pin(async { Ok(None) })
    }

    fn store_accumulators<'a>(
        &'a self,
        _agent_id: &'a str,
        _state: &'a AccumulatorState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn load_accumulators<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<AccumulatorState>>> + Send + 'a>> {
        Box::pin(async { Ok(None) })
    }
}

struct PartiallyFailingFeature;

impl AdaptiveFeature for PartiallyFailingFeature {
    fn register<'a>(
        &'a mut self,
        ctx: &'a mut RegistrationContext<'_>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            ctx.register_subscriber("partial_feature", Arc::new(|_event| {}))?;
            Err(AdaptiveError::Internal("feature boom".into()))
        })
    }
}

#[test]
fn build_learners_filters_unknown_entries() {
    let learners = build_learners(
        "agent-a",
        &["latency_sensitivity".to_string(), "unknown".to_string()],
        None,
        None,
        None,
    );
    assert_eq!(learners.len(), 1);
}

#[test]
fn adaptive_runtime_validate_config_covers_supported_warning_error_and_ignore_branches() {
    for (mode, provider) in [
        ("observe_only", "passthrough"),
        ("inject_hints", "anthropic"),
        ("schedule", "openai"),
    ] {
        let report = AdaptiveRuntime::validate_config(&AdaptiveConfig {
            state: Some(StateConfig {
                backend: BackendSpec::in_memory(),
            }),
            tool_parallelism: Some(ToolParallelismComponentConfig {
                mode: mode.to_string(),
                ..ToolParallelismComponentConfig::default()
            }),
            acg: Some(AcgComponentConfig {
                provider: provider.to_string(),
                ..AcgComponentConfig::default()
            }),
            ..AdaptiveConfig::default()
        });
        assert!(
            report.diagnostics.is_empty(),
            "{mode}/{provider} should not emit diagnostics: {:?}",
            report.diagnostics
        );
    }

    let missing_state = AdaptiveRuntime::validate_config(&AdaptiveConfig {
        telemetry: Some(TelemetryComponentConfig::default()),
        acg: Some(AcgComponentConfig::default()),
        ..AdaptiveConfig::default()
    });
    assert_eq!(
        missing_state
            .diagnostics
            .iter()
            .filter(
                |diag| diag.code == "adaptive.section_disabled_missing_state"
                    && diag.level == DiagnosticLevel::Warning
            )
            .count(),
        2
    );

    let errors = AdaptiveRuntime::validate_config(&AdaptiveConfig {
        version: 99,
        state: Some(StateConfig {
            backend: BackendSpec {
                kind: "unknown-backend".to_string(),
                config: serde_json::Map::new(),
            },
        }),
        tool_parallelism: Some(ToolParallelismComponentConfig {
            mode: "unsupported".to_string(),
            ..ToolParallelismComponentConfig::default()
        }),
        acg: Some(AcgComponentConfig {
            provider: "custom-provider".to_string(),
            ..AcgComponentConfig::default()
        }),
        policy: ConfigPolicy {
            unknown_component: UnsupportedBehavior::Error,
            unsupported_value: UnsupportedBehavior::Error,
            ..ConfigPolicy::default()
        },
        ..AdaptiveConfig::default()
    });
    assert!(errors.has_errors());
    assert!(errors.diagnostics.iter().any(
        |diag| diag.code == "adaptive.unknown_backend" && diag.level == DiagnosticLevel::Error
    ));
    assert!(
        errors
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("mode")
                && diag.level == DiagnosticLevel::Error)
    );
    assert!(
        errors
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("provider")
                && diag.level == DiagnosticLevel::Error)
    );

    let ignored = AdaptiveRuntime::validate_config(&AdaptiveConfig {
        version: 99,
        state: Some(StateConfig {
            backend: BackendSpec {
                kind: "unknown-backend".to_string(),
                config: serde_json::Map::new(),
            },
        }),
        tool_parallelism: Some(ToolParallelismComponentConfig {
            mode: "unsupported".to_string(),
            ..ToolParallelismComponentConfig::default()
        }),
        acg: Some(AcgComponentConfig {
            provider: "custom-provider".to_string(),
            ..AcgComponentConfig::default()
        }),
        policy: ConfigPolicy {
            unknown_component: UnsupportedBehavior::Ignore,
            unsupported_value: UnsupportedBehavior::Ignore,
            ..ConfigPolicy::default()
        },
        ..AdaptiveConfig::default()
    });
    assert!(ignored.diagnostics.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_new_rejects_invalid_configs_with_joined_errors() {
    let err = AdaptiveRuntime::new(AdaptiveConfig {
        version: 2,
        telemetry: Some(TelemetryComponentConfig::default()),
        policy: ConfigPolicy {
            unsupported_value: UnsupportedBehavior::Error,
            ..ConfigPolicy::default()
        },
        ..AdaptiveConfig::default()
    })
    .await
    .unwrap_err();

    match err {
        AdaptiveError::InvalidConfig(message) => assert!(!message.is_empty()),
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn registration_context_take_event_receiver_only_allows_one_consumer() {
    let _lock = GLOBAL_RUNTIME_TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig::default())
        .await
        .unwrap();
    let mut ctx = RegistrationContext::new(&mut runtime);

    assert!(ctx.take_event_receiver().is_ok());
    let err = ctx.take_event_receiver().unwrap_err();
    assert!(
        matches!(err, AdaptiveError::Internal(message) if message.contains("telemetry already registered"))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn telemetry_feature_registers_subscriber_and_starts_drain_task() {
    let _lock = GLOBAL_RUNTIME_TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig {
        state: Some(StateConfig {
            backend: BackendSpec::in_memory(),
        }),
        ..AdaptiveConfig::default()
    })
    .await
    .unwrap();
    let mut feature = TelemetryFeature::new(
        TelemetryComponentConfig {
            subscriber_name: Some("adaptive_feature_test_subscriber".into()),
            learners: vec!["latency_sensitivity".into()],
        },
        "agent-telemetry".into(),
        Uuid::now_v7(),
        None,
        None,
        None,
    );
    let name = feature.subscriber_name.clone();

    let mut registrations = {
        let mut ctx = RegistrationContext::new(&mut runtime);
        feature.register(&mut ctx).await.unwrap();
        ctx.finish()
    };
    assert!(runtime.drain_handle.is_some());
    assert_subscriber_registered(&name);

    rollback_registrations(&mut registrations);
    assert_subscriber_absent(&name);

    if let Some(handle) = runtime.drain_handle.take() {
        handle.abort();
    }
}

#[tokio::test(flavor = "current_thread")]
async fn telemetry_feature_requires_backend() {
    let _lock = GLOBAL_RUNTIME_TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig::default())
        .await
        .unwrap();
    let mut feature = TelemetryFeature::new(
        TelemetryComponentConfig::default(),
        "agent-telemetry".into(),
        Uuid::now_v7(),
        None,
        None,
        None,
    );
    let mut ctx = RegistrationContext::new(&mut runtime);

    let err = feature.register(&mut ctx).await.unwrap_err();
    assert!(
        matches!(err, AdaptiveError::InvalidConfig(message) if message.contains("telemetry requires state backend"))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_hints_feature_registers_request_intercept() {
    let _lock = GLOBAL_RUNTIME_TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig::default())
        .await
        .unwrap();
    runtime.hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: Some(AgentHints {
            osl: 10,
            iat: 20,
            priority: 3,
            latency_sensitivity: 2.0,
            prefix_id: "agent-a-d0".to_string(),
            total_requests: 4,
        }),
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));

    let mut feature = AdaptiveHintsFeature::new(
        AdaptiveHintsComponentConfig {
            priority: 7,
            break_chain: true,
            ..AdaptiveHintsComponentConfig::default()
        },
        runtime.hot_cache.clone(),
        "agent-a".into(),
        Uuid::now_v7(),
    );
    let name = feature.name.clone();

    let mut ctx = RegistrationContext::new(&mut runtime);
    feature.register(&mut ctx).await.unwrap();
    assert_llm_request_intercept_registered(&name);

    let request = llm_request_intercepts(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({}),
        },
    )
    .unwrap();
    assert!(request.headers.contains_key(AGENT_HINTS_HEADER_KEY));

    let mut registrations = ctx.finish();
    rollback_registrations(&mut registrations);
    assert_llm_request_intercept_absent(&name);
}

#[tokio::test(flavor = "current_thread")]
async fn tool_parallelism_feature_registers_execution_intercept() {
    let _lock = GLOBAL_RUNTIME_TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig::default())
        .await
        .unwrap();
    runtime.hot_cache = Arc::new(RwLock::new(HotCache {
        plan: Some(sample_plan("agent-tools")),
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));

    let mut feature = ToolParallelismFeature::new(
        ToolParallelismComponentConfig {
            priority: 11,
            ..ToolParallelismComponentConfig::default()
        },
        runtime.hot_cache.clone(),
        Uuid::now_v7(),
    );
    let name = feature.name.clone();

    let mut ctx = RegistrationContext::new(&mut runtime);
    feature.register(&mut ctx).await.unwrap();
    assert_tool_execution_intercept_registered(&name);

    let next: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("search")
            .args(json!({"query": "coverage"}))
            .func(next)
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result["query"], json!("coverage"));

    let mut registrations = ctx.finish();
    rollback_registrations(&mut registrations);
    assert_tool_execution_intercept_absent(&name);
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_register_survives_hot_cache_seed_failures() {
    let _lock = GLOBAL_RUNTIME_TEST_MUTEX.lock().await;
    reset_global();

    let config = AdaptiveConfig {
        adaptive_hints: Some(AdaptiveHintsComponentConfig::default()),
        ..AdaptiveConfig::default()
    };
    let report = validate_config(&config);
    let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut runtime = AdaptiveRuntime {
        config,
        report,
        registered_agent_id: None,
        backend: Some(Arc::new(SeedFailBackend)),
        hot_cache: Arc::new(RwLock::new(HotCache {
            plan: None,
            trie: None,
            agent_hints_default: None,
            acg_profiles: std::collections::HashMap::new(),
            acg_profile_observation_counts: std::collections::HashMap::new(),
            acg_stability: None,
            acg_observation_count: 0,
        })),
        cache_diagnostics_tracker: Arc::new(RwLock::new(CacheDiagnosticsTracker::default())),
        pending_events: Arc::new(AtomicUsize::new(0)),
        event_tx,
        event_rx: Some(event_rx),
        drain_handle: None,
        registered: false,
        runtime_id: Uuid::now_v7(),
        bound_scopes: Arc::new(RwLock::new(HashSet::new())),
        registrations: vec![],
    };

    runtime.register().await.unwrap();
    assert!(runtime.registered);
    assert!(!runtime.registrations.is_empty());
    runtime.deregister().unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_register_is_idempotent_for_active_features() {
    let _lock = GLOBAL_RUNTIME_TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig {
        adaptive_hints: Some(AdaptiveHintsComponentConfig::default()),
        tool_parallelism: Some(ToolParallelismComponentConfig::default()),
        ..AdaptiveConfig::default()
    })
    .await
    .unwrap();

    runtime.register().await.unwrap();
    let registrations_after_first = runtime.registrations.len();
    runtime.register().await.unwrap();

    assert_eq!(registrations_after_first, 2);
    assert_eq!(runtime.registrations.len(), registrations_after_first);

    runtime.deregister().unwrap();
    assert!(!runtime.registered);
    assert!(runtime.registrations.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_register_rolls_back_when_telemetry_receiver_is_missing() {
    let _lock = GLOBAL_RUNTIME_TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig {
        state: Some(StateConfig {
            backend: BackendSpec::in_memory(),
        }),
        telemetry: Some(TelemetryComponentConfig::default()),
        ..AdaptiveConfig::default()
    })
    .await
    .unwrap();
    runtime.event_rx = None;

    let err = runtime.register().await.unwrap_err();
    assert!(
        matches!(err, AdaptiveError::Internal(message) if message.contains("telemetry already registered"))
    );
    assert!(!runtime.registered);
    assert!(runtime.drain_handle.is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn registration_context_registers_all_supported_callback_types() {
    let _lock = GLOBAL_RUNTIME_TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig::default())
        .await
        .unwrap();
    let mut ctx = RegistrationContext::new(&mut runtime);

    ctx.register_subscriber("adaptive_test_subscriber", Arc::new(|_event| {}))
        .unwrap();
    ctx.register_llm_request_intercept(
        "adaptive_test_request",
        5,
        false,
        Arc::new(|_name, request, annotated| Ok((request, annotated))),
    )
    .unwrap();
    ctx.register_llm_execution_intercept(
        "adaptive_test_execution",
        6,
        Arc::new(|_name, request, _next| Box::pin(async move { Ok(request.content) })),
    )
    .unwrap();
    ctx.register_llm_stream_execution_intercept(
        "adaptive_test_stream",
        7,
        Arc::new(|_name, request, _next| {
            Box::pin(async move {
                Ok(Box::pin(tokio_stream::iter(vec![Ok(request.content)]))
                    as Pin<
                        Box<
                            dyn tokio_stream::Stream<
                                    Item = nemo_relay::error::Result<nemo_relay::json::Json>,
                                > + Send,
                        >,
                    >)
            })
        }),
    )
    .unwrap();
    ctx.register_tool_execution_intercept(
        "adaptive_test_tool",
        8,
        Arc::new(|_name, args, _next| Box::pin(async move { Ok(args) })),
    )
    .unwrap();

    let mut registrations = ctx.finish();
    assert_subscriber_registered("adaptive_test_subscriber");
    assert_llm_request_intercept_registered("adaptive_test_request");
    assert_llm_execution_intercept_registered("adaptive_test_execution");
    assert_llm_stream_execution_intercept_registered("adaptive_test_stream");
    assert_tool_execution_intercept_registered("adaptive_test_tool");

    rollback_registrations(&mut registrations);
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_helper_methods_cover_report_wait_for_idle_and_feature_filtering() {
    let config = AdaptiveConfig {
        agent_id: Some("explicit-agent".into()),
        telemetry: Some(TelemetryComponentConfig {
            learners: vec!["tool_parallelism".into(), "acg".into()],
            ..TelemetryComponentConfig::default()
        }),
        adaptive_hints: Some(AdaptiveHintsComponentConfig::default()),
        tool_parallelism: Some(ToolParallelismComponentConfig::default()),
        acg: Some(AcgComponentConfig::default()),
        ..AdaptiveConfig::default()
    };
    let runtime_without_backend = AdaptiveRuntime::new(config.clone()).await.unwrap();

    assert_eq!(runtime_without_backend.agent_id(), "explicit-agent");
    assert!(!runtime_without_backend.report().has_errors());
    assert_eq!(runtime_without_backend.pending_features("agent-a").len(), 2);
    assert_eq!(
        build_learners(
            "agent-a",
            &["tool_parallelism".to_string(), "acg".to_string()],
            config.tool_parallelism.as_ref(),
            config.acg.as_ref(),
            config.convergence.as_ref(),
        )
        .len(),
        2
    );

    runtime_without_backend
        .pending_events
        .store(1, Ordering::SeqCst);
    let pending = runtime_without_backend.pending_events.clone();
    let waiter = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        pending.store(0, Ordering::SeqCst);
    });
    runtime_without_backend.wait_for_idle();
    waiter.join().unwrap();

    let runtime_with_backend = AdaptiveRuntime::new(AdaptiveConfig {
        state: Some(StateConfig {
            backend: BackendSpec::in_memory(),
        }),
        ..config
    })
    .await
    .unwrap();
    assert_eq!(runtime_with_backend.pending_features("agent-a").len(), 4);
}

#[tokio::test(flavor = "current_thread")]
async fn acg_feature_registers_execution_and_stream_intercepts() {
    let _lock = GLOBAL_RUNTIME_TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig::default())
        .await
        .unwrap();
    runtime.hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: Some(layered_acg_stability_result(6)),
        acg_observation_count: 6,
    }));
    let mut feature = AcgFeature::new(
        AcgComponentConfig {
            provider: "anthropic".into(),
            priority: 13,
            ..AcgComponentConfig::default()
        },
        runtime.hot_cache.clone(),
        runtime.bound_scopes.clone(),
        "agent-acg".into(),
        Uuid::now_v7(),
    );

    let execution_name = feature.execution_name.clone();
    let stream_name = feature.stream_name.clone();
    let bound_scopes = runtime.bound_scopes.clone();
    let mut ctx = RegistrationContext::new(&mut runtime);
    feature.register(&mut ctx).await.unwrap();

    assert_llm_execution_intercept_registered(&execution_name);
    assert_llm_stream_execution_intercept_registered(&stream_name);

    let next: LlmExecutionNextFn = Arc::new(|request| Box::pin(async move { Ok(request.content) }));
    let rewritten = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("anthropic")
            .request(layered_acg_request())
            .func(next.clone())
            .model_name("claude-sonnet-4-20250514")
            .build(),
    )
    .await
    .unwrap();
    assert!(rewritten["system"][0]["cache_control"].is_object());

    let stream_next: LlmStreamExecutionNextFn = Arc::new(|request| {
        Box::pin(async move {
            let stream: nemo_relay::api::runtime::LlmJsonStream =
                Box::pin(tokio_stream::iter(vec![Ok(request.content)]));
            Ok(stream)
        })
    });
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("anthropic")
            .request(layered_acg_request())
            .func(stream_next.clone())
            .collector(Box::new(|_chunk| Ok(())))
            .finalizer(Box::new(|| json!({"done": true})))
            .model_name("claude-sonnet-4-20250514")
            .build(),
    )
    .await
    .unwrap();
    let stream_rewritten = stream.next().await.unwrap().unwrap();
    assert!(stream_rewritten["system"][0]["cache_control"].is_object());

    bound_scopes.write().unwrap().insert(Uuid::now_v7());
    let passthrough = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("anthropic")
            .request(layered_acg_request())
            .func(next)
            .model_name("claude-sonnet-4-20250514")
            .build(),
    )
    .await
    .unwrap();
    assert!(passthrough["system"].is_string());

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("anthropic")
            .request(layered_acg_request())
            .func(stream_next)
            .collector(Box::new(|_chunk| Ok(())))
            .finalizer(Box::new(|| json!({"done": true})))
            .model_name("claude-sonnet-4-20250514")
            .build(),
    )
    .await
    .unwrap();
    let stream_passthrough = stream.next().await.unwrap().unwrap();
    assert!(stream_passthrough["system"].is_string());

    let mut registrations = ctx.finish();
    rollback_registrations(&mut registrations);
    assert_llm_execution_intercept_absent(&execution_name);
    assert_llm_stream_execution_intercept_absent(&stream_name);
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_register_feature_rolls_back_partial_registrations_and_abort_handle() {
    let _lock = GLOBAL_RUNTIME_TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig::default())
        .await
        .unwrap();
    {
        let mut ctx = RegistrationContext::new(&mut runtime);
        ctx.register_subscriber("existing_feature", Arc::new(|_event| {}))
            .unwrap();
        runtime.registrations = ctx.finish();
    }
    runtime.drain_handle = Some(tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(60)).await;
    }));
    runtime.registered = true;

    let mut feature: Box<dyn AdaptiveFeature> = Box::new(PartiallyFailingFeature);
    let err = runtime.register_feature(&mut feature).await.unwrap_err();

    assert!(matches!(err, AdaptiveError::Internal(message) if message.contains("feature boom")));
    assert!(!runtime.registered);
    assert!(runtime.drain_handle.is_none());
    assert!(runtime.registrations.is_empty());
    assert_subscriber_absent("existing_feature");
    assert_subscriber_absent("partial_feature");
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_shutdown_is_a_clean_noop_after_deregister() {
    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig::default())
        .await
        .unwrap();
    runtime.deregister().unwrap();
    runtime.shutdown().await.unwrap();
}
