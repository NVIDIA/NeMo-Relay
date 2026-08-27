// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic dynamic-host coverage for the documented native plugin.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};

use nemo_relay::api::event::Event;
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::{ToolCallExecuteParams, ToolExecutionResult, tool_call_execute};
use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationSpec, DynamicPluginKind, PluginHostActivation,
};
use nemo_relay::plugin::{PluginConfig, list_plugin_kinds};
use serde_json::{Map, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::sync::Mutex as AsyncMutex;

static TEST_LOCK: AsyncMutex<()> = AsyncMutex::const_new(());
const PLUGIN_ID: &str = "examples.rust_native_policy";
const SUBSCRIBER: &str = "rust_native_example_lifecycle_events";
const CONTROLLED_SUBSCRIBER: &str = "documentation-controlled-subscriber";
const ALLOWED_SUBSCRIBER: &str = "documentation-observed-subscriber";

#[tokio::test]
async fn built_cdylib_validates_activates_runs_and_unloads() {
    let _guard = TEST_LOCK.lock().await;
    let (_build_dir, build) = build_cdylib();
    let manifest_dir = TempDir::new().expect("manifest directory should be created");
    let manifest = write_manifest(manifest_dir.path(), &build);
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

    let config = documented_config();
    let (activation, report) = PluginHostActivation::activate(
        PluginConfig::default(),
        [DynamicPluginActivationSpec {
            plugin_id: PLUGIN_ID.into(),
            kind: DynamicPluginKind::RustDynamic,
            manifest_ref: manifest.to_string_lossy().into_owned(),
            environment_ref: None,
            config,
        }],
    )
    .await
    .expect("the materialized native manifest should activate");
    assert!(report.diagnostics.is_empty(), "{report:?}");
    flush_subscribers().expect("activation events should flush");
    let controlled_baseline = controlled_events.load(Ordering::SeqCst);
    let allowed_baseline = allowed_events.load(Ordering::SeqCst);

    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"secret": "application-value"}))
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
    .expect("native tool middleware should execute");
    assert_eq!(result.result["secret"], "application-value");
    assert!(result.result.get("plugin_tag").is_none());
    assert_eq!(result.annotation, Some(json!({"source": "application"})));

    flush_subscribers().expect("native events should flush");
    assert_eq!(
        controlled_events.load(Ordering::SeqCst),
        controlled_baseline,
        "the activation-owned gate should suppress future subscriber snapshots"
    );
    assert!(
        allowed_events.load(Ordering::SeqCst) > allowed_baseline,
        "a None gate decision should leave the matching subscriber enabled"
    );
    assert!(
        events
            .lock()
            .expect("event lock should not be poisoned")
            .iter()
            .any(|event| event.name() == "example.native.request.seen")
    );
    assert!(events.lock().expect("event lock should not be poisoned").iter().any(|event| {
        event
            .metadata()
            .and_then(|metadata| metadata.get("external.injector.transport"))
            == Some(&json!("rust_native_plugin"))
    }));

    activation
        .clear()
        .expect("callbacks should clear before the library unloads");
    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("restored_tool")
            .args(json!({}))
            .func(Arc::new(|args| Box::pin(async move { Ok(ToolExecutionResult::new(args)) })))
            .build(),
    )
    .await
    .expect("managed execution should continue after plugin clear");
    flush_subscribers().expect("restored subscriber events should flush");
    assert!(controlled_events.load(Ordering::SeqCst) > controlled_baseline);
    deregister_subscriber(CONTROLLED_SUBSCRIBER)
        .expect("controlled subscriber should deregister");
    deregister_subscriber(ALLOWED_SUBSCRIBER).expect("allowed subscriber should deregister");
    deregister_subscriber(SUBSCRIBER).expect("test subscriber should deregister");
    assert!(!list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));
}

fn documented_config() -> Map<String, serde_json::Value> {
    json!({
        "tag": "documentation",
        "observe": { "enabled": true, "redact_keys": ["secret"] },
        "requests": {
            "enabled": false,
            "mode": "enforce",
            "blocked_tools": ["dangerous_tool"],
            "blocked_models": ["restricted-model"],
            "header_name": "x-nemo-relay-plugin",
            "header_value": "documentation",
            "priority": 20,
            "break_chain": false
        },
        "execution": { "enabled": false, "priority": 30, "emit_pending_marks": true },
        "runtime": { "emit_marks": true, "emit_isolated_scope": true },
        "registration_control": {
            "enabled": true,
            "kinds": ["subscriber"],
            "registration_name": "documentation-controlled-subscriber",
            "allowed_registration_name": "documentation-observed-subscriber",
            "reason": "disabled by documentation plugin"
        },
        "executor": { "worker_threads": 2 }
    })
    .as_object()
    .expect("documented configuration is an object")
    .clone()
}

fn build_cdylib() -> (TempDir, PathBuf) {
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
        "cargo build should produce the native library"
    );
    let library = target.path().join("debug").join(library_name());
    assert!(
        library.exists(),
        "cargo build should produce the expected library"
    );
    (target, library)
}

fn write_manifest(directory: &Path, library: &Path) -> PathBuf {
    let digest = digest(library);
    let library = toml_basic_string(&library.to_string_lossy());
    let manifest = directory.join("relay-plugin.toml");
    std::fs::write(
        &manifest,
        format!(
            r#"manifest_version = 1

[plugin]
id = "{PLUGIN_ID}"
kind = "rust_dynamic"

[compat]
relay = ">=0.8.0,<1.0"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[integrity]
sha256 = "{digest}"

[load]
library = {library}
symbol = "nemo_relay_register_plugin"
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
fn toml_basic_string_escapes_windows_library_paths() {
    assert_eq!(
        toml_basic_string(r"C:\Users\relay\plugin.dll"),
        r#""C:\\Users\\relay\\plugin.dll""#
    );
}

fn library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "nemo_relay_rust_native_plugin_example.dll"
    } else if cfg!(target_os = "macos") {
        "libnemo_relay_rust_native_plugin_example.dylib"
    } else {
        "libnemo_relay_rust_native_plugin_example.so"
    }
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
