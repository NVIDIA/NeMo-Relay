// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for worker-backed local-model PII redaction.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

use nemo_relay::api::event::Event;
use nemo_relay::api::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay::api::runtime::LlmExecutionNextFn;
use nemo_relay::api::scope::{EmitMarkEventParams, event};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::codec::traits::LlmResponseCodec;
use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationSpec, DynamicPluginKind, PluginHostActivation,
};
use nemo_relay::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, local_model_provider,
};
use nemo_relay_pii_redaction::component::{
    PII_REDACTION_PLUGIN_KIND, register_pii_redaction_component,
};
use serde_json::{Map, json};
use tempfile::TempDir;

static WORKER_PII_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test(flavor = "multi_thread")]
async fn worker_provider_sanitizes_events_and_is_removed_after_host_clear() {
    let _guard = WORKER_PII_TEST_LOCK.lock().await;
    let _ = clear_plugin_configuration();
    register_pii_redaction_component().expect("PII component should register");
    let worker_binary = build_fixture_worker();
    let (_manifest_dir, manifest_ref) = write_worker_manifest(&worker_binary);

    let plugin_config = PluginConfig {
        version: 1,
        components: vec![PluginComponentSpec {
            kind: PII_REDACTION_PLUGIN_KIND.into(),
            enabled: true,
            config: Map::from_iter([
                ("mode".into(), json!("local_model")),
                ("codec".into(), json!("openai_chat")),
                ("input".into(), json!(true)),
                ("output".into(), json!(true)),
                ("tool_input".into(), json!(false)),
                ("tool_output".into(), json!(false)),
                ("mark".into(), json!(true)),
                (
                    "local".into(),
                    json!({
                        "backend": "fixture_worker/fixture_local_model",
                        "target_paths": ["/message"],
                        "target_path_patterns": [
                            "/messages/*/content",
                            "/message"
                        ]
                    }),
                ),
            ]),
        }],
        policy: Default::default(),
    };
    let (activation, report) = PluginHostActivation::activate(
        plugin_config,
        [DynamicPluginActivationSpec {
            plugin_id: "fixture_worker".into(),
            kind: DynamicPluginKind::Worker,
            manifest_ref: manifest_ref.to_string_lossy().into_owned(),
            environment_ref: None,
            config: Map::from_iter([("provider_only".into(), json!(true))]),
        }],
    )
    .await
    .expect("worker and PII component should activate together");
    assert!(!report.has_errors());
    assert!(local_model_provider("fixture_worker/fixture_local_model").is_ok());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    let subscriber_name = "worker-pii-e2e";
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .expect("test subscriber should register");

    let callback_value = json!({
        "message": "keep PRIVATE hidden",
        "unselected": "PRIVATE remains outside the configured path"
    });
    event(
        EmitMarkEventParams::builder()
            .name("worker-pii")
            .data(callback_value.clone())
            .build(),
    )
    .expect("mark should emit");
    flush_subscribers().expect("sanitized event should flush");

    {
        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(
            captured[0].data(),
            Some(&json!({
                "message": "keep [REDACTED] hidden",
                "unselected": "PRIVATE remains outside the configured path"
            }))
        );
    }
    assert_eq!(
        callback_value["message"], "keep PRIVATE hidden",
        "sanitization must not mutate caller-owned JSON"
    );

    let callback_request = Arc::new(Mutex::new(None));
    let observed_request = Arc::clone(&callback_request);
    let response = json!({
        "id": "chatcmpl-PRIVATE",
        "model": "model-PRIVATE",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "answer for PRIVATE"
            },
            "finish_reason": "stop"
        }]
    });
    let callback_response = response.clone();
    let callback: LlmExecutionNextFn = Arc::new(move |request| {
        *observed_request.lock().unwrap() = Some(request.clone());
        let response = callback_response.clone();
        Box::pin(async move { Ok(response) })
    });
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "model-PRIVATE",
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "question from PRIVATE"}
            ],
            "vendor_trace": "trace-PRIVATE"
        }),
    };
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let returned = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(request.clone())
            .func(callback)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .expect("LLM callback should complete");
    flush_subscribers().expect("LLM events should flush");

    assert_eq!(
        returned, response,
        "sanitize guardrails must not rewrite callback values"
    );
    assert_eq!(
        callback_request.lock().unwrap().as_ref(),
        Some(&request),
        "sanitize guardrails must not rewrite provider requests"
    );
    let captured = events.lock().unwrap();
    assert_eq!(captured.len(), 3);
    assert_eq!(
        captured[1].input().unwrap()["content"]["messages"][1]["content"],
        "question from [REDACTED]"
    );
    assert_eq!(
        captured[1].input().unwrap()["content"]["model"],
        "model-PRIVATE",
        "model identifiers are outside the selected content paths"
    );
    assert_eq!(
        captured[1].input().unwrap()["content"]["vendor_trace"],
        "trace-PRIVATE",
        "provider metadata is outside the selected content paths"
    );
    assert_eq!(
        captured[2].output().unwrap()["choices"][0]["message"]["content"],
        "answer for [REDACTED]"
    );
    assert_eq!(captured[2].output().unwrap()["id"], "chatcmpl-PRIVATE");
    assert_eq!(captured[2].output().unwrap()["model"], "model-PRIVATE");
    drop(captured);

    deregister_subscriber(subscriber_name).expect("test subscriber should deregister");
    activation.clear().expect("plugin host should clear");
    assert!(
        local_model_provider("fixture_worker/fixture_local_model").is_err(),
        "worker provider should not outlive its host activation"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_exit_during_sanitization_fails_closed_and_removes_provider() {
    let _guard = WORKER_PII_TEST_LOCK.lock().await;
    let _ = clear_plugin_configuration();
    register_pii_redaction_component().expect("PII component should register");
    let worker_binary = build_fixture_worker();
    let (_manifest_dir, manifest_ref) = write_worker_manifest(&worker_binary);
    let plugin_config = PluginConfig {
        version: 1,
        components: vec![PluginComponentSpec {
            kind: PII_REDACTION_PLUGIN_KIND.into(),
            enabled: true,
            config: Map::from_iter([
                ("mode".into(), json!("local_model")),
                ("input".into(), json!(false)),
                ("output".into(), json!(false)),
                ("tool_input".into(), json!(false)),
                ("tool_output".into(), json!(false)),
                ("mark".into(), json!(true)),
                (
                    "local".into(),
                    json!({
                        "backend": "fixture_worker/fixture_local_model",
                        "target_paths": ["/message"]
                    }),
                ),
            ]),
        }],
        policy: Default::default(),
    };
    let (activation, report) = PluginHostActivation::activate(
        plugin_config,
        [DynamicPluginActivationSpec {
            plugin_id: "fixture_worker".into(),
            kind: DynamicPluginKind::Worker,
            manifest_ref: manifest_ref.to_string_lossy().into_owned(),
            environment_ref: None,
            config: Map::from_iter([
                ("provider_only".into(), json!(true)),
                ("exit_in_local_model".into(), json!(true)),
            ]),
        }],
    )
    .await
    .expect("worker and PII component should activate together");
    assert!(!report.has_errors());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    let subscriber_name = "worker-pii-exit";
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .expect("test subscriber should register");

    event(
        EmitMarkEventParams::builder()
            .name("worker-pii-exit")
            .data(json!({
                "message": "PRIVATE",
                "unselected": "PRIVATE"
            }))
            .build(),
    )
    .expect("worker failure must not block event emission");
    flush_subscribers().expect("fail-closed event should flush");
    assert_eq!(
        events.lock().unwrap()[0].data(),
        Some(&json!({
            "message": "[REDACTED]",
            "unselected": "PRIVATE"
        }))
    );

    deregister_subscriber(subscriber_name).expect("test subscriber should deregister");
    let error = activation
        .clear()
        .expect_err("stopped worker shutdown should be reported")
        .to_string();
    assert!(error.contains("shutdown"), "{error}");
    assert!(
        local_model_provider("fixture_worker/fixture_local_model").is_err(),
        "failed worker provider should not survive host teardown"
    );
}

fn build_fixture_worker() -> PathBuf {
    static FIXTURE_BINARY: OnceLock<PathBuf> = OnceLock::new();
    FIXTURE_BINARY
        .get_or_init(|| {
            let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../core/tests/fixtures/worker_plugin/Cargo.toml");
            let target_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/worker-plugin-fixture/target");
            let status = Command::new("cargo")
                .arg("build")
                .arg("--quiet")
                .arg("--locked")
                .arg("--manifest-path")
                .arg(&manifest)
                .arg("--target-dir")
                .arg(&target_dir)
                .status()
                .expect("fixture worker build should start");
            assert!(status.success(), "fixture worker build should succeed");
            let binary = target_dir.join("debug").join(format!(
                "nemo-relay-worker-plugin-fixture{}",
                std::env::consts::EXE_SUFFIX
            ));
            assert!(binary.exists(), "fixture worker binary should exist");
            binary
        })
        .clone()
}

fn write_worker_manifest(binary: &Path) -> (TempDir, PathBuf) {
    let temp = TempDir::new().expect("manifest directory should be created");
    let manifest = temp.path().join("relay-plugin.toml");
    let contents = format!(
        r#"
manifest_version = 1

[plugin]
id = "fixture_worker"
kind = "worker"

[compat]
relay = "={version}"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "rust"
entrypoint = {entrypoint:?}
"#,
        version = env!("CARGO_PKG_VERSION"),
        entrypoint = binary.to_string_lossy(),
    );
    std::fs::write(&manifest, contents).expect("worker manifest should be written");
    (temp, manifest)
}
