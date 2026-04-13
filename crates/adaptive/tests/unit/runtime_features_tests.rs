// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use std::sync::Arc;

use nemo_flow::api::llm::llm_request_intercepts;
use nemo_flow::api::tool::tool_call_execute;
use nemo_flow::context::callbacks::ToolExecutionNextFn;
use nemo_flow::context::global::global_context;
use nemo_flow::context::state::NemoFlowContextState;
use nemo_flow::plugin::{ConfigPolicy, UnsupportedBehavior};
use nemo_flow::plugin::{clear_plugin_configuration, rollback_registrations};
use nemo_flow::types::llm::LLMRequest;
use nemo_flow::types::tool::ToolAttributes;
use serde_json::json;
use tokio::sync::Mutex;

use crate::config::{BackendSpec, StateConfig};
use crate::intercepts::AGENT_HINTS_HEADER_KEY;
use crate::trie::accumulator::AccumulatorState;
use crate::trie::serialization::TrieEnvelope;
use crate::types::metadata::{AgentHints, MetadataEnvelope, ParallelHint};
use crate::types::plan::{ExecutionPlan, ParallelGroup};
use crate::types::records::RunRecord;

static TEST_MUTEX: Mutex<()> = Mutex::const_new(());

fn reset_global() {
    let _ = clear_plugin_configuration();
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoFlowContextState::new();
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

#[test]
fn build_learners_filters_unknown_entries() {
    let learners = build_learners(
        "agent-a",
        &["latency_sensitivity".to_string(), "unknown".to_string()],
    );
    assert_eq!(learners.len(), 1);
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
    let _lock = TEST_MUTEX.lock().await;
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
    let _lock = TEST_MUTEX.lock().await;
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
    );
    let name = feature.subscriber_name.clone();

    let mut registrations = {
        let mut ctx = RegistrationContext::new(&mut runtime);
        feature.register(&mut ctx).await.unwrap();
        ctx.finish()
    };
    assert!(runtime.drain_handle.is_some());
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key(&name)
    );

    rollback_registrations(&mut registrations);
    assert!(
        !global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key(&name)
    );

    if let Some(handle) = runtime.drain_handle.take() {
        handle.abort();
    }
}

#[tokio::test(flavor = "current_thread")]
async fn telemetry_feature_requires_backend() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig::default())
        .await
        .unwrap();
    let mut feature = TelemetryFeature::new(
        TelemetryComponentConfig::default(),
        "agent-telemetry".into(),
        Uuid::now_v7(),
    );
    let mut ctx = RegistrationContext::new(&mut runtime);

    let err = feature.register(&mut ctx).await.unwrap_err();
    assert!(
        matches!(err, AdaptiveError::InvalidConfig(message) if message.contains("telemetry requires state backend"))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_hints_feature_registers_request_intercept() {
    let _lock = TEST_MUTEX.lock().await;
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
    assert!(
        global_context()
            .read()
            .unwrap()
            .llm_request_intercepts
            .contains(&name)
    );

    let request = llm_request_intercepts(
        "model",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({}),
        },
    )
    .unwrap();
    assert!(request.headers.contains_key(AGENT_HINTS_HEADER_KEY));

    let mut registrations = ctx.finish();
    rollback_registrations(&mut registrations);
    assert!(
        !global_context()
            .read()
            .unwrap()
            .llm_request_intercepts
            .contains(&name)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn tool_parallelism_feature_registers_execution_intercept() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();

    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig::default())
        .await
        .unwrap();
    runtime.hot_cache = Arc::new(RwLock::new(HotCache {
        plan: Some(sample_plan("agent-tools")),
        trie: None,
        agent_hints_default: None,
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
    assert!(
        global_context()
            .read()
            .unwrap()
            .tool_execution_intercepts
            .contains(&name)
    );

    let next: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let result = tool_call_execute(
        "search",
        json!({"query": "coverage"}),
        next,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(result["query"], json!("coverage"));

    let mut registrations = ctx.finish();
    rollback_registrations(&mut registrations);
    assert!(
        !global_context()
            .read()
            .unwrap()
            .tool_execution_intercepts
            .contains(&name)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_register_survives_hot_cache_seed_failures() {
    let _lock = TEST_MUTEX.lock().await;
    reset_global();

    let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut runtime = AdaptiveRuntime {
        config: AdaptiveConfig {
            adaptive_hints: Some(AdaptiveHintsComponentConfig::default()),
            ..AdaptiveConfig::default()
        },
        backend: Some(Arc::new(SeedFailBackend)),
        hot_cache: Arc::new(RwLock::new(HotCache {
            plan: None,
            trie: None,
            agent_hints_default: None,
        })),
        event_tx,
        event_rx: Some(event_rx),
        drain_handle: None,
        registered: false,
        runtime_id: Uuid::now_v7(),
        registrations: vec![],
    };

    runtime.register().await.unwrap();
    assert!(runtime.registered);
    assert!(!runtime.registrations.is_empty());
    runtime.deregister().unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_register_is_idempotent_for_active_features() {
    let _lock = TEST_MUTEX.lock().await;
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
    let _lock = TEST_MUTEX.lock().await;
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
