// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3 storage integration tests for the ATIF observability exporter.
//!
//! These tests require an S3-compatible object store reachable through the
//! standard AWS environment variables (`AWS_ACCESS_KEY_ID`,
//! `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, optionally `AWS_ENDPOINT_URL` and
//! `AWS_ALLOW_HTTP`). They only execute when `NEMO_RELAY_RUN_S3_TESTS=1` is
//! set. The destination bucket must be supplied via
//! `NEMO_RELAY_S3_TEST_BUCKET`. A test-scoped key prefix containing a UUID is
//! used so concurrent runs cannot collide; an override is available via
//! `NEMO_RELAY_S3_TEST_KEY_PREFIX` for environments that pin a prefix.

#![cfg(feature = "atif-storage")]

use std::time::Duration;

use nemo_relay::api::runtime::{
    NemoRelayContextState, create_scope_stack, global_context, set_thread_scope_stack,
};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use nemo_relay::observability::plugin_component::OBSERVABILITY_PLUGIN_KIND;
use nemo_relay::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins,
};
use object_store::ObjectStore;
use serde_json::{Value as Json, json};
use uuid::Uuid;

const RUN_ENV: &str = "NEMO_RELAY_RUN_S3_TESTS";
const BUCKET_ENV: &str = "NEMO_RELAY_S3_TEST_BUCKET";
const KEY_PREFIX_ENV: &str = "NEMO_RELAY_S3_TEST_KEY_PREFIX";

fn env_value_is_truthy(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some(value) if !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    )
}

fn run_tests_enabled() -> bool {
    let raw =
        std::env::var_os(RUN_ENV).map(|value| value.to_string_lossy().into_owned());
    env_value_is_truthy(raw.as_deref())
}

fn read_bucket() -> Option<String> {
    let value = std::env::var(BUCKET_ENV).ok()?;
    if value.trim().is_empty() { None } else { Some(value) }
}

fn build_test_key_prefix() -> String {
    let base = std::env::var(KEY_PREFIX_ENV)
        .unwrap_or_else(|_| "nemo-relay-atif-integration/".to_string());
    let trimmed = base.trim_end_matches('/');
    let run_id = Uuid::now_v7();
    if trimmed.is_empty() {
        format!("{run_id}/")
    } else {
        format!("{trimmed}/{run_id}/")
    }
}

fn reset_runtime() {
    let _ = clear_plugin_configuration();
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

fn build_observability_config(bucket: &str, key_prefix: &str) -> PluginConfig {
    let Json::Object(component_config) = json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": {
                "type": "s3",
                "bucket": bucket,
                "key_prefix": key_prefix,
            }
        }
    }) else {
        unreachable!("config builder produced non-object root")
    };
    PluginConfig {
        version: 1,
        components: vec![PluginComponentSpec {
            kind: OBSERVABILITY_PLUGIN_KIND.to_string(),
            enabled: true,
            config: component_config,
        }],
        policy: Default::default(),
    }
}

fn read_object_with_retries(
    runtime: &tokio::runtime::Runtime,
    store: &dyn ObjectStore,
    key: &str,
) -> Vec<u8> {
    runtime.block_on(async {
        let path = object_store::path::Path::from(key);
        // The dispatcher uploads from a different runtime thread; allow a brief
        // grace window for S3-compatible backends with eventual consistency.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            match store.get(&path).await {
                Ok(result) => {
                    return result
                        .bytes()
                        .await
                        .expect("uploaded payload should be readable")
                        .to_vec();
                }
                Err(err) if std::time::Instant::now() < deadline => {
                    eprintln!("waiting for upload to settle: {err}");
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Err(err) => panic!("failed to read uploaded ATIF object '{key}': {err}"),
            }
        }
    })
}

fn build_verification_store(
    bucket: &str,
) -> (tokio::runtime::Runtime, Box<dyn ObjectStore>) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("verification runtime should build");
    let store = object_store::aws::AmazonS3Builder::from_env()
        .with_bucket_name(bucket)
        .build()
        .expect("verification S3 client should build");
    (runtime, Box::new(store))
}

fn cleanup_prefix(
    runtime: &tokio::runtime::Runtime,
    store: &dyn ObjectStore,
    key_prefix: &str,
) {
    use futures::stream::StreamExt;
    runtime.block_on(async {
        let prefix_path = object_store::path::Path::from(key_prefix.trim_end_matches('/'));
        let mut listing = store.list(Some(&prefix_path));
        while let Some(entry) = listing.next().await {
            match entry {
                Ok(meta) => {
                    if let Err(err) = store.delete(&meta.location).await {
                        eprintln!("cleanup: failed to delete {}: {err}", meta.location);
                    }
                }
                Err(err) => {
                    eprintln!("cleanup: list error: {err}");
                    break;
                }
            }
        }
    });
}

#[test]
fn atif_storage_uploads_trajectory_to_s3() {
    if !run_tests_enabled() {
        eprintln!(
            "SKIP: set {RUN_ENV} to a truthy value (for example, {RUN_ENV}=1) to run ATIF S3 storage tests"
        );
        return;
    }
    let Some(bucket) = read_bucket() else {
        eprintln!("SKIP: set {BUCKET_ENV} to the destination bucket for ATIF S3 storage tests");
        return;
    };

    let key_prefix = build_test_key_prefix();
    reset_runtime();

    let config = build_observability_config(&bucket, &key_prefix);
    futures::executor::block_on(initialize_plugins(config))
        .expect("observability plugin should initialize with S3 storage");

    let handle = push_scope(
        PushScopeParams::builder()
            .name("atif-storage-integration")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .expect("push agent scope");
    let session_id = handle.uuid;
    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .build(),
    )
    .expect("pop agent scope");

    clear_plugin_configuration().expect("plugin teardown should flush the trajectory");

    let key = format!("{key_prefix}trajectory-{session_id}.json");
    let (runtime, store) = build_verification_store(&bucket);
    let body = read_object_with_retries(&runtime, store.as_ref(), &key);
    let value: Json = serde_json::from_slice(&body).expect("uploaded payload should be JSON");
    assert_eq!(
        value["schema_version"].as_str(),
        Some("ATIF-v1.6"),
        "uploaded artifact should be an ATIF trajectory"
    );
    let expected_session_id = session_id.to_string();
    assert_eq!(value["session_id"].as_str(), Some(expected_session_id.as_str()));

    cleanup_prefix(&runtime, store.as_ref(), &key_prefix);
}

#[test]
fn s3_test_env_truthy_parsing() {
    assert!(!env_value_is_truthy(None));
    assert!(!env_value_is_truthy(Some("")));
    assert!(!env_value_is_truthy(Some("   ")));
    assert!(!env_value_is_truthy(Some("0")));
    assert!(!env_value_is_truthy(Some(" false ")));
    assert!(!env_value_is_truthy(Some("FALSE")));
    assert!(env_value_is_truthy(Some("1")));
    assert!(env_value_is_truthy(Some("true")));
    assert!(env_value_is_truthy(Some("yes")));
}
