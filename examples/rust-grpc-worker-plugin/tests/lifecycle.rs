// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic transport coverage for the documented Rust grpc-v1 worker.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};

use nemo_relay::api::event::Event;
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::{ToolCallExecuteParams, ToolExecutionResult, tool_call_execute};
use nemo_relay::plugin::PluginConfig;
use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationSpec, DynamicPluginKind, PluginHostActivation,
};
use serde_json::{Map, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::sync::Mutex as AsyncMutex;

static TEST_LOCK: AsyncMutex<()> = AsyncMutex::const_new(());
const PLUGIN_ID: &str = "examples.rust_grpc_worker";
const SUBSCRIBER: &str = "rust_grpc_worker_example_lifecycle_events";
const CONTROLLED_SUBSCRIBER: &str = "documentation-controlled-subscriber";
const ALLOWED_SUBSCRIBER: &str = "documentation-observed-subscriber";

#[tokio::test(flavor = "multi_thread")]
async fn built_worker_validates_registers_executes_and_shuts_down() {
    let _guard = TEST_LOCK.lock().await;
    let (_build_dir, worker) = build_worker();
    let manifest_dir = TempDir::new().expect("manifest directory should be created");
    let manifest = write_manifest(manifest_dir.path(), &worker);
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        SUBSCRIBER,
        Arc::new(move |event| {
            captured
                .lock()
                .expect("event lock should not be poisoned")
                .push(event.clone());
        }),
    )
    .expect("test subscriber should register");
    let controlled_events = Arc::new(AtomicUsize::new(0));
    let captured_controlled_events = Arc::clone(&controlled_events);
    register_subscriber(
        CONTROLLED_SUBSCRIBER,
        Arc::new(move |_| {
            captured_controlled_events.fetch_add(1, Ordering::SeqCst);
        }),
    )
    .expect("controlled subscriber should register");
    let allowed_events = Arc::new(AtomicUsize::new(0));
    let captured_allowed_events = Arc::clone(&allowed_events);
    register_subscriber(
        ALLOWED_SUBSCRIBER,
        Arc::new(move |_| {
            captured_allowed_events.fetch_add(1, Ordering::SeqCst);
        }),
    )
    .expect("allowed subscriber should register");

    let (activation, report) = PluginHostActivation::activate(
        PluginConfig::default(),
        [DynamicPluginActivationSpec {
            plugin_id: PLUGIN_ID.into(),
            kind: DynamicPluginKind::Worker,
            manifest_ref: manifest.to_string_lossy().into_owned(),
            environment_ref: None,
            config: documented_config(),
        }],
    )
    .await
    .expect("the materialized worker manifest should activate");
    assert!(report.diagnostics.is_empty(), "{report:?}");
    flush_subscribers().expect("activation events should flush");
    let controlled_baseline = controlled_events.load(Ordering::SeqCst);

    let result = tool_call_execute(
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
    .await
    .expect("worker middleware should execute through grpc-v1");
    assert_eq!(result.result["plugin_tag"], "documentation");
    assert_eq!(result.result["plugin_tool"], "safe_tool");
    assert_eq!(result.annotation, Some(json!({"source": "application"})));

    flush_subscribers().expect("worker events should flush");
    assert_eq!(controlled_events.load(Ordering::SeqCst), controlled_baseline);
    assert!(
        events
            .lock()
            .expect("event lock should not be poisoned")
            .iter()
            .any(|event| event.name() == "example.rust_worker.request.seen")
    );
    assert!(events.lock().expect("event lock should not be poisoned").iter().any(|event| {
        event
            .metadata()
            .and_then(|metadata| metadata.get("external.injector.transport"))
            == Some(&json!("rust_grpc_worker"))
    }));

    activation
        .clear()
        .expect("worker shutdown should follow callback cleanup");

    let (allowed_activation, report) = PluginHostActivation::activate(
        PluginConfig::default(),
        [DynamicPluginActivationSpec {
            plugin_id: PLUGIN_ID.into(),
            kind: DynamicPluginKind::Worker,
            manifest_ref: manifest.to_string_lossy().into_owned(),
            environment_ref: None,
            config: allowed_config(),
        }],
    )
    .await
    .expect("the allow-path worker configuration should activate");
    assert!(report.diagnostics.is_empty(), "{report:?}");
    flush_subscribers().expect("allow-path activation events should flush");
    let allowed_baseline = allowed_events.load(Ordering::SeqCst);
    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("allowed_tool")
            .args(json!({}))
            .func(Arc::new(|args| Box::pin(async move { Ok(ToolExecutionResult::new(args)) })))
            .build(),
    )
    .await
    .expect("a None gate decision should leave the matching subscriber enabled");
    flush_subscribers().expect("allow-path subscriber events should flush");
    assert!(allowed_events.load(Ordering::SeqCst) > allowed_baseline);
    allowed_activation
        .clear()
        .expect("allow-path worker should shut down cleanly");

    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("restored_tool")
            .args(json!({}))
            .func(Arc::new(|args| Box::pin(async move { Ok(ToolExecutionResult::new(args)) })))
            .build(),
    )
    .await
    .expect("managed execution should continue after worker clear");
    flush_subscribers().expect("restored subscriber events should flush");
    assert!(controlled_events.load(Ordering::SeqCst) > controlled_baseline);
    deregister_subscriber(CONTROLLED_SUBSCRIBER)
        .expect("controlled subscriber should deregister");
    deregister_subscriber(ALLOWED_SUBSCRIBER).expect("allowed subscriber should deregister");
    deregister_subscriber(SUBSCRIBER).expect("test subscriber should deregister");
}

fn documented_config() -> Map<String, serde_json::Value> {
    json!({
        "tag": "documentation",
        "observe": { "enabled": true, "redact_keys": ["secret"] },
        "requests": {
            "enabled": true,
            "mode": "enforce",
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
            "enabled": true,
            "kinds": ["subscriber"],
            "registration_name": "documentation-controlled-subscriber",
            "reason": "disabled by documentation plugin"
        }
    })
    .as_object()
    .expect("documented configuration is an object")
    .clone()
}

fn allowed_config() -> Map<String, serde_json::Value> {
    let mut config = documented_config();
    config
        .get_mut("registration_control")
        .and_then(serde_json::Value::as_object_mut)
        .expect("registration control config should be an object")
        .insert(
            "registration_name".into(),
            json!(ALLOWED_SUBSCRIBER),
        );
    config
}

fn build_worker() -> (TempDir, PathBuf) {
    let target = TempDir::new().expect("build target directory should be created");
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let status = Command::new(env!("CARGO"))
        .args(["build", "--manifest-path"])
        .arg(manifest)
        .arg("--target-dir")
        .arg(target.path())
        .status()
        .expect("cargo build should start");
    assert!(
        status.success(),
        "cargo build should produce the worker executable"
    );
    let worker = target.path().join("debug").join(format!(
        "nemo-relay-rust-grpc-worker-plugin-example{}",
        std::env::consts::EXE_SUFFIX
    ));
    assert!(
        worker.exists(),
        "cargo build should produce the expected worker executable"
    );
    (target, worker)
}

fn write_manifest(directory: &Path, worker: &Path) -> PathBuf {
    let digest = digest(worker);
    let worker = toml_basic_string(&worker.to_string_lossy());
    let manifest = directory.join("relay-plugin.toml");
    std::fs::write(
        &manifest,
        format!(
            r#"manifest_version = 1

[plugin]
id = "{PLUGIN_ID}"
kind = "worker"

[compat]
relay = ">=0.8.0,<1.0"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[integrity]
sha256 = "{digest}"

[load]
runtime = "rust"
entrypoint = {worker}
"#,
        ),
    )
    .expect("materialized manifest should write");
    manifest
}

fn toml_basic_string(value: &str) -> String {
    format!("{value:?}")
}

#[test]
fn toml_basic_string_escapes_windows_worker_paths() {
    assert_eq!(
        toml_basic_string(r"C:\Users\relay\worker.exe"),
        r#""C:\\Users\\relay\\worker.exe""#
    );
}

fn digest(path: &Path) -> String {
    let digest = Sha256::digest(std::fs::read(path).expect("read artifact"));
    format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}
