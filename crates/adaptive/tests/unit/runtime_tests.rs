// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value as Json;

use crate::config::{
    AdaptiveConfig, BackendSpec, StateConfig, TelemetryComponentConfig,
    ToolParallelismComponentConfig,
};
use crate::error::AdaptiveError;
use crate::runtime::backend::build_backend;
use crate::runtime::features::AdaptiveRuntime;
use crate::runtime::validation::validate_config;
use nemo_flow::plugin::{ConfigPolicy, UnsupportedBehavior};

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
