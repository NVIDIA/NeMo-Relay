// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for runtime in the NeMo Relay adaptive crate.

use nemo_relay::api::llm::{LlmRequest, llm_request_intercepts};
use nemo_relay::api::runtime::{
    NemoRelayContextState, create_scope_stack, global_context, set_thread_scope_stack,
};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use serde_json::{Map, Value as Json};

use crate::config::{
    AcgComponentConfig, AdaptiveConfig, BackendSpec, StateConfig, TelemetryComponentConfig,
    ToolParallelismComponentConfig,
};
use crate::error::AdaptiveError;
use crate::runtime::backend::build_backend;
use crate::runtime::features::AdaptiveRuntime;
use crate::runtime::validation::validate_config;
use nemo_relay::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use nemo_relay::plugin::{ConfigPolicy, UnsupportedBehavior};

#[cfg(feature = "redis-backend")]
const REDIS_TEST_ENV: &str = "NEMO_RELAY_RUN_REDIS_TESTS";

fn reset_runtime_context() {
    let context = global_context();
    let mut state = context.write().unwrap();
    *state = NemoRelayContextState::new();
    set_thread_scope_stack(create_scope_stack());
}

fn sample_annotated_request(model: Option<&str>) -> AnnotatedLlmRequest {
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
        model: model.map(str::to_string),
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

fn sample_layered_request(model: Option<&str>, language_guide: &str) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are a careful planner".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text(language_guide.to_string()),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "Acknowledged. I will apply the stable review lens.".to_string(),
                )),
                tool_calls: None,
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Bundle contents go here".to_string()),
                name: None,
            },
        ],
        model: model.map(str::to_string),
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

#[tokio::test(flavor = "current_thread")]
async fn build_backend_supports_in_memory_and_rejects_unknown_kinds() {
    let backend = build_backend(&BackendSpec::in_memory()).await.unwrap();
    assert!(backend.list_runs_dyn("agent").await.unwrap().is_empty());

    let invalid_backend = build_backend(&BackendSpec {
        kind: "bogus".to_string(),
        config: serde_json::Map::<String, Json>::new(),
    })
    .await;
    match invalid_backend {
        Err(AdaptiveError::InvalidConfig(message)) => {
            assert!(message.contains("unsupported backend"));
        }
        Err(other) => panic!("unexpected backend error: {other}"),
        Ok(_) => panic!("expected invalid backend to fail"),
    }
}

#[cfg(feature = "redis-backend")]
#[tokio::test(flavor = "current_thread")]
async fn build_backend_redis_requires_url_and_maps_invalid_client_urls() {
    let missing_url = build_backend(&BackendSpec {
        kind: "redis".to_string(),
        config: serde_json::Map::<String, Json>::new(),
    })
    .await;
    match missing_url {
        Err(AdaptiveError::InvalidConfig(message)) => {
            assert!(message.contains("missing url"));
        }
        Err(other) => panic!("unexpected missing-url error: {other}"),
        Ok(_) => panic!("expected missing redis url to fail"),
    }

    let invalid_url = build_backend(&BackendSpec {
        kind: "redis".to_string(),
        config: serde_json::Map::from_iter([(
            "url".to_string(),
            Json::String("not-a-redis-url".to_string()),
        )]),
    })
    .await;
    match invalid_url {
        Err(AdaptiveError::Storage(message)) => {
            assert!(message.contains("redis client"));
        }
        Err(other) => panic!("unexpected invalid-url error: {other}"),
        Ok(_) => panic!("expected invalid redis url to fail"),
    }
}

#[cfg(feature = "redis-backend")]
#[tokio::test(flavor = "current_thread")]
async fn build_backend_redis_supports_success_path_when_server_is_available() {
    if std::env::var_os(REDIS_TEST_ENV).is_none() {
        eprintln!("SKIP: set {REDIS_TEST_ENV}=1 to run Redis-backed tests");
        return;
    }

    if crate::redis::RedisBackend::new("redis://127.0.0.1/", "probe:".to_string())
        .await
        .is_err()
    {
        eprintln!("SKIP: Redis not available at 127.0.0.1:6379");
        return;
    }

    let backend = build_backend(&BackendSpec {
        kind: "redis".to_string(),
        config: serde_json::Map::from_iter([
            (
                "url".to_string(),
                Json::String("redis://127.0.0.1/".to_string()),
            ),
            (
                "key_prefix".to_string(),
                Json::String("runtime-success:".to_string()),
            ),
        ]),
    })
    .await
    .expect("expected redis backend to build");

    let runs = backend
        .list_runs_dyn("runtime-success-agent")
        .await
        .expect("expected empty run listing");
    assert!(runs.is_empty());
}

#[cfg(not(feature = "redis-backend"))]
#[tokio::test(flavor = "current_thread")]
async fn build_backend_redis_reports_feature_disabled_when_compiled_out() {
    let disabled = build_backend(&BackendSpec {
        kind: "redis".to_string(),
        config: serde_json::Map::from_iter([(
            "url".to_string(),
            Json::String("redis://127.0.0.1/".to_string()),
        )]),
    })
    .await;

    match disabled {
        Err(AdaptiveError::InvalidConfig(message)) => {
            assert!(message.contains("not enabled"));
        }
        Err(other) => panic!("unexpected feature-disabled error: {other}"),
        Ok(_) => panic!("expected redis backend to be disabled in this build"),
    }
}

#[test]
fn validate_config_reports_version_mode_and_telemetry_gaps() {
    let report = validate_config(&AdaptiveConfig {
        version: 2,
        telemetry: Some(TelemetryComponentConfig::default()),
        tool_parallelism: Some(ToolParallelismComponentConfig {
            mode: "invalid".to_string(),
            ..ToolParallelismComponentConfig::default()
        }),
        policy: ConfigPolicy {
            unsupported_value: UnsupportedBehavior::Error,
            ..ConfigPolicy::default()
        },
        ..AdaptiveConfig::default()
    });

    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "adaptive.unsupported_config_version")
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "adaptive.unsupported_value")
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "adaptive.section_disabled_missing_state")
    );
}

#[test]
fn validate_config_reports_unknown_backend_and_acg_provider_per_policy() {
    let warn_report = validate_config(&AdaptiveConfig {
        state: Some(StateConfig {
            backend: BackendSpec {
                kind: "mystery".to_string(),
                config: Map::new(),
            },
        }),
        acg: Some(AcgComponentConfig {
            provider: "custom".to_string(),
            ..AcgComponentConfig::default()
        }),
        policy: ConfigPolicy {
            unknown_component: UnsupportedBehavior::Warn,
            unsupported_value: UnsupportedBehavior::Warn,
            ..ConfigPolicy::default()
        },
        ..AdaptiveConfig::default()
    });
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "adaptive.unknown_backend"
                && diag.level == nemo_relay::plugin::DiagnosticLevel::Warning)
    );
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "adaptive.unsupported_value"
                && diag.field.as_deref() == Some("provider"))
    );

    let ignore_report = validate_config(&AdaptiveConfig {
        state: Some(StateConfig {
            backend: BackendSpec {
                kind: "mystery".to_string(),
                config: Map::new(),
            },
        }),
        policy: ConfigPolicy {
            unknown_component: UnsupportedBehavior::Ignore,
            unsupported_value: UnsupportedBehavior::Ignore,
            ..ConfigPolicy::default()
        },
        acg: Some(AcgComponentConfig {
            provider: "custom".to_string(),
            ..AcgComponentConfig::default()
        }),
        ..AdaptiveConfig::default()
    });
    assert!(ignore_report.diagnostics.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_new_accepts_valid_in_memory_configuration() {
    let runtime = AdaptiveRuntime::new(AdaptiveConfig {
        state: Some(StateConfig {
            backend: BackendSpec::in_memory(),
        }),
        ..AdaptiveConfig::default()
    })
    .await
    .unwrap();

    let rendered = format!("{runtime:?}");
    assert!(rendered.contains("AdaptiveRuntime"));
    assert!(rendered.contains("registered"));
}

#[test]
fn adaptive_owned_runtime_sources_use_canonical_acg_module_paths() {
    let owned_sources: [(&str, &str, &[&str]); 7] = [
        (
            "src/config.rs",
            include_str!("../../src/config.rs"),
            &["crate::acg::stability::StabilityThresholds"],
        ),
        (
            "src/runtime/features.rs",
            include_str!("../../src/runtime/features.rs"),
            &["use crate::acg::CacheRequestFacts;"],
        ),
        (
            "src/acg_component.rs",
            include_str!("../../src/acg_component.rs"),
            &["use crate::acg::plugin::{PluginInput, ProviderPlugin};"],
        ),
        (
            "src/acg_learner.rs",
            include_str!("../../src/acg_learner.rs"),
            &["use crate::acg::ir_builder::build_prompt_ir;"],
        ),
        (
            "src/acg_profile.rs",
            include_str!("../../src/acg_profile.rs"),
            &["use crate::acg::canonicalize::{canonicalize_value, sha256_hex};"],
        ),
        (
            "src/cache_diagnostics.rs",
            include_str!("../../src/cache_diagnostics.rs"),
            &[
                "use crate::acg::canonicalize::sha256_hex;",
                "use crate::acg::ir_builder::build_prompt_ir;",
            ],
        ),
        (
            "src/tool_parallelism_learner.rs",
            include_str!("../../src/tool_parallelism_learner.rs"),
            &["use crate::acg::canonicalize::sha256_hex;"],
        ),
    ];

    for (path, source, canonical_patterns) in owned_sources {
        assert!(
            !source.contains("nemo_relay_acg::"),
            "{path} should not fall back to the compatibility shim",
        );
        for canonical_pattern in canonical_patterns {
            assert!(
                source.contains(canonical_pattern),
                "{path} should import ACG through `{canonical_pattern}`",
            );
        }
    }
}

#[test]
fn adaptive_acg_defaults_and_profile_key_behavior_stay_stable() {
    let config = AdaptiveConfig::default();
    assert!(config.acg.is_none());

    let acg = AcgComponentConfig::default();
    assert_eq!(acg.provider, "passthrough");
    assert_eq!(acg.observation_window, 100);
    assert_eq!(acg.priority, 50);
    assert_eq!(
        acg.stability_thresholds,
        crate::acg::stability::StabilityThresholds::default()
    );

    let profile_key = crate::acg_profile::derive_acg_profile_key(
        "agent-1",
        &sample_annotated_request(Some("claude-sonnet-4")),
    );
    assert_eq!(
        profile_key,
        "agent-1::model=claude-sonnet-4::roles=system.user::system=sha256:97f793c76::anchor=no-anchor::tools=no-tools"
    );
    let learning_key = crate::acg_profile::derive_acg_learning_key(
        "agent-1",
        &sample_annotated_request(Some("claude-sonnet-4")),
    );
    let expected_learning_key = format!(
        "agent-1::model=claude-sonnet-4::seed=stable-scaffold::system={}::tools=no-tools",
        crate::acg::sha256_hex("You are a careful planner"),
    );
    assert_eq!(learning_key, expected_learning_key,);

    let grown_chat_request = AnnotatedLlmRequest {
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
                    "I found several stable observations.".to_string(),
                )),
                tool_calls: None,
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Continue with the next batch.".to_string()),
                name: None,
            },
        ],
        model: Some("claude-sonnet-4".to_string()),
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
    };
    assert_eq!(
        crate::acg_profile::derive_acg_learning_key("agent-1", &grown_chat_request),
        learning_key,
        "growing direct chats should reuse the same learning bucket",
    );
    assert_ne!(
        crate::acg_profile::derive_acg_profile_key("agent-1", &grown_chat_request),
        profile_key,
        "diagnostic keys should still reflect the exact live role shape",
    );

    let rust_key = crate::acg_profile::derive_acg_profile_key(
        "agent-1",
        &sample_layered_request(Some("claude-sonnet-4"), "Rust review guide"),
    );
    let python_key = crate::acg_profile::derive_acg_profile_key(
        "agent-1",
        &sample_layered_request(Some("claude-sonnet-4"), "Python review guide"),
    );
    assert_ne!(
        rust_key, python_key,
        "layered requests should separate profiles when the stable guide layer differs",
    );
    let rust_learning_key = crate::acg_profile::derive_acg_learning_key(
        "agent-1",
        &sample_layered_request(Some("claude-sonnet-4"), "Rust review guide"),
    );
    let python_learning_key = crate::acg_profile::derive_acg_learning_key(
        "agent-1",
        &sample_layered_request(Some("claude-sonnet-4"), "Python review guide"),
    );
    assert_eq!(
        rust_learning_key, python_learning_key,
        "variable first-user guide content should share the stable system scaffold",
    );

    let rust_bundle_variant = AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are a careful planner".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Rust review guide".to_string()),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "Acknowledged. I will apply the stable review lens.".to_string(),
                )),
                tool_calls: None,
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Different bundle contents go here".to_string()),
                name: None,
            },
        ],
        model: Some("claude-sonnet-4".to_string()),
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
    };
    let rust_bundle_variant_key =
        crate::acg_profile::derive_acg_profile_key("agent-1", &rust_bundle_variant);
    assert_eq!(
        rust_key, rust_bundle_variant_key,
        "layered requests should keep the same profile when only later bundle or turn content changes",
    );
    let rust_bundle_variant_learning_key =
        crate::acg_profile::derive_acg_learning_key("agent-1", &rust_bundle_variant);
    assert_eq!(
        rust_learning_key, rust_bundle_variant_learning_key,
        "layered requests should keep the same learning bucket when only later bundle or turn content changes",
    );
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_build_cache_request_facts_keeps_missing_stability_semantics() {
    let runtime = AdaptiveRuntime::new(AdaptiveConfig::default())
        .await
        .expect("default adaptive runtime should construct");

    let facts = runtime
        .build_cache_request_facts(
            "agent-1",
            "anthropic",
            &sample_annotated_request(Some("claude-sonnet-4")),
        )
        .expect("runtime should still emit request facts without stability state");

    assert_eq!(facts.provider, "anthropic");
    assert_eq!(facts.stable_prefix_length, 0);
    assert_eq!(facts.stable_prefix_tokens, None);
    assert_eq!(facts.required_min_tokens, None);
    assert_eq!(facts.missing_facts, vec!["acg_stability_unavailable"]);
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_runtime_bind_scope_requires_registration_and_passes_through_without_state() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime_context();
    let mut runtime = AdaptiveRuntime::new(AdaptiveConfig {
        agent_id: Some("agent-1".to_string()),
        state: Some(StateConfig {
            backend: BackendSpec::in_memory(),
        }),
        acg: Some(AcgComponentConfig::default()),
        ..AdaptiveConfig::default()
    })
    .await
    .expect("adaptive runtime with acg should construct");
    let scope = push_scope(
        PushScopeParams::builder()
            .name("adaptive-runtime-scope")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .expect("scope push should succeed");

    let registration_err = match runtime.bind_scope(scope.uuid) {
        Ok(_) => panic!("expected scope binding to require registration"),
        Err(err) => err,
    };
    assert!(matches!(
        registration_err,
        AdaptiveError::RegistrationFailed(message)
            if message.contains("must be registered before binding ACG request intercepts")
    ));

    runtime
        .register()
        .await
        .expect("adaptive runtime should register");

    runtime
        .bind_scope(scope.uuid)
        .expect("registered runtime should bind acg to the active scope");
    let request = LlmRequest {
        headers: Map::new(),
        content: serde_json::json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are helpful.",
            "model": "claude-sonnet-4-20250514",
        }),
    };

    let translated = llm_request_intercepts("anthropic", request.clone())
        .expect("request intercept chain should pass through when no hot-cache state exists");

    assert_eq!(translated.request.content, request.content);
    pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build())
        .expect("scope pop should succeed");
}
