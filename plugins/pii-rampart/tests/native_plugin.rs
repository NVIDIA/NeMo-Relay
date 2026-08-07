// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live coverage for the packaged Rampart native plugin.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nemo_relay::api::event::Event;
use nemo_relay::api::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay::api::runtime::{TASK_SCOPE_STACK, create_scope_stack};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::{ToolCallExecuteParams, tool_call_execute};
use nemo_relay::codec::gemini_generate_content::GeminiGenerateContentCodec;
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::plugin::PluginConfig;
use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationSpec, DynamicPluginKind, PluginHostActivation,
};
use serde_json::{Map, Value as Json, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const SUBSCRIBER_NAME: &str = "pii-rampart-native-live";
const REQUEST_EMAIL: &str = "alex.fournier@example.com";
const RESPONSE_EMAIL: &str = "reviewer@example.org";

struct TestCleanup {
    activation: Option<PluginHostActivation>,
}

impl Drop for TestCleanup {
    fn drop(&mut self) {
        let _ = deregister_subscriber(SUBSCRIBER_NAME);
        if let Some(activation) = self.activation.take() {
            let _ = activation.clear();
        }
    }
}

#[test]
fn config_schema_matches_activation_invariants() {
    let schema: Json = serde_json::from_slice(
        &std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("config.schema.json"))
            .expect("read Rampart config schema"),
    )
    .expect("parse Rampart config schema");
    jsonschema::draft202012::meta::validate(&schema).expect("validate Rampart config schema");
    let validator = jsonschema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .build(&schema)
        .expect("compile Rampart config schema");

    for config in [
        json!({"model_path": "/model", "preset": "trajectory_context"}),
        json!({"model_path": "/model", "target_paths": ["/content"]}),
        json!({
            "model_path": "/model",
            "target_path_patterns": ["/messages/*/content"],
            "codec": "gemini_generate_content"
        }),
        json!({
            "model_path": "/model",
            "preset": "trajectory_context",
            "target_paths": []
        }),
    ] {
        assert!(
            validator.is_valid(&config),
            "valid config rejected: {config}"
        );
    }

    for config in [
        json!({"model_path": "/model"}),
        json!({
            "model_path": "/model",
            "preset": "trajectory_context",
            "target_paths": ["/content"]
        }),
        json!({
            "model_path": "/model",
            "target_paths": [],
            "target_path_patterns": []
        }),
        json!({
            "model_path": "/model",
            "target_paths": ["/content"],
            "custom_mark_payload_policy": "redact_all_leaves"
        }),
        json!({
            "model_path": "/model",
            "target_paths": ["/content"],
            "input": false,
            "output": false,
            "mark": false,
            "tool_input": false,
            "tool_output": false
        }),
    ] {
        assert!(
            !validator.is_valid(&config),
            "invalid config accepted: {config}"
        );
    }
}

#[test]
#[ignore = "requires a built native library and the pinned Rampart model snapshot"]
fn native_plugin_redacts_observability_without_mutating_calls() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build live-test runtime")
        .block_on(run_live_test());
}

async fn run_live_test() {
    let library = plugin_library();
    let model = model_snapshot();
    let manifest_dir = TempDir::new().expect("create manifest directory");
    let manifest = write_manifest(manifest_dir.path(), &library);
    let config = json!({
        "model_path": model,
        "preset": "trajectory_context",
        "custom_mark_payload_policy": "redact_all_leaves"
    })
    .as_object()
    .expect("plugin config is an object")
    .clone();

    let (activation, report) = PluginHostActivation::activate(
        PluginConfig::default(),
        [DynamicPluginActivationSpec {
            plugin_id: "pii_rampart".into(),
            kind: DynamicPluginKind::RustDynamic,
            manifest_ref: manifest.to_string_lossy().into_owned(),
            environment_ref: None,
            config,
        }],
    )
    .await
    .expect("activate Rampart native plugin");
    assert!(
        !report.has_errors(),
        "Rampart activation diagnostics: {:?}",
        report.diagnostics
    );
    let mut cleanup = TestCleanup {
        activation: Some(activation),
    };

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        SUBSCRIBER_NAME,
        Arc::new(move |event| captured.lock().expect("event lock").push(event.clone())),
    )
    .expect("register event subscriber");

    let tool_arguments_seen = Arc::new(Mutex::new(None::<Json>));
    let tool_arguments_capture = Arc::clone(&tool_arguments_seen);
    let llm_request_seen = Arc::new(Mutex::new(None::<LlmRequest>));
    let llm_request_capture = Arc::clone(&llm_request_seen);

    let stack = create_scope_stack();
    let gemini_request_seen = Arc::new(Mutex::new(None::<LlmRequest>));
    let gemini_request_capture = Arc::clone(&gemini_request_seen);

    let (tool_result, llm_result, gemini_result) = TASK_SCOPE_STACK
        .scope(stack, async move {
            let scope = push_scope(
                PushScopeParams::builder()
                    .name("rampart-native-live")
                    .scope_type(ScopeType::Agent)
                    .input(json!({"contact": REQUEST_EMAIL, "safe": "scope-input"}))
                    .build(),
            )
            .expect("push live-test scope");

            let tool_result = tool_call_execute(
                ToolCallExecuteParams::builder()
                    .name("lookup_contact")
                    .args(json!({"email": REQUEST_EMAIL, "safe": "tool-input"}))
                    .func(Arc::new(move |arguments| {
                        *tool_arguments_capture.lock().expect("tool capture lock") =
                            Some(arguments);
                        Box::pin(async {
                            Ok(json!({"email": RESPONSE_EMAIL, "safe": "tool-output"}))
                        })
                    }))
                    .build(),
            )
            .await
            .expect("execute managed tool call");

            let request = LlmRequest {
                headers: Map::new(),
                content: json!({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": format!("Email {REQUEST_EMAIL}")}]
                }),
            };
            let response = json!({
                "id": "chatcmpl-rampart-live",
                "object": "chat.completion",
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": format!("Email {RESPONSE_EMAIL}")},
                    "finish_reason": "stop"
                }]
            });
            let callback_response = response.clone();
            let codec = Arc::new(OpenAIChatCodec);
            let llm_result = llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name("openai")
                    .request(request)
                    .func(Arc::new(move |request| {
                        *llm_request_capture.lock().expect("LLM capture lock") = Some(request);
                        let response = callback_response.clone();
                        Box::pin(async move { Ok(response) })
                    }))
                    .codec(codec.clone())
                    .response_codec(codec)
                    .build(),
            )
            .await
            .expect("execute managed LLM call");

            let gemini_request = LlmRequest {
                headers: Map::new(),
                content: json!({
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": format!("Email {REQUEST_EMAIL}; safe-gemini-input")}]
                    }]
                }),
            };
            let gemini_response = json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": format!("Email {RESPONSE_EMAIL}; safe-gemini-output")}]
                    },
                    "finishReason": "STOP"
                }],
                "modelVersion": "gemini-test-model"
            });
            let callback_response = gemini_response.clone();
            let codec = Arc::new(GeminiGenerateContentCodec);
            let gemini_result = llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name("gemini")
                    .request(gemini_request)
                    .func(Arc::new(move |request| {
                        *gemini_request_capture.lock().expect("Gemini capture lock") =
                            Some(request);
                        let response = callback_response.clone();
                        Box::pin(async move { Ok(response) })
                    }))
                    .codec(codec.clone())
                    .response_codec(codec)
                    .build(),
            )
            .await
            .expect("execute managed Gemini call");

            pop_scope(
                PopScopeParams::builder()
                    .handle_uuid(&scope.uuid)
                    .output(json!({"contact": RESPONSE_EMAIL, "safe": "scope-output"}))
                    .build(),
            )
            .expect("pop live-test scope");
            (tool_result, llm_result, gemini_result)
        })
        .await;

    assert_eq!(
        tool_arguments_seen
            .lock()
            .expect("tool capture lock")
            .as_ref()
            .expect("tool callback ran")["email"],
        REQUEST_EMAIL
    );
    assert_eq!(tool_result["email"], RESPONSE_EMAIL);
    assert!(
        llm_request_seen
            .lock()
            .expect("LLM capture lock")
            .as_ref()
            .expect("LLM callback ran")
            .content
            .to_string()
            .contains(REQUEST_EMAIL)
    );
    assert!(llm_result.to_string().contains(RESPONSE_EMAIL));
    assert!(
        gemini_request_seen
            .lock()
            .expect("Gemini capture lock")
            .as_ref()
            .expect("Gemini callback ran")
            .content
            .to_string()
            .contains(REQUEST_EMAIL)
    );
    assert!(gemini_result.to_string().contains(RESPONSE_EMAIL));

    let concurrent_emails = run_concurrent_tool_calls(8).await;

    flush_subscribers().expect("flush live-test events");
    let serialized = serde_json::to_string(&*events.lock().expect("event lock"))
        .expect("serialize captured events");
    assert!(!serialized.contains(REQUEST_EMAIL), "request PII leaked");
    assert!(!serialized.contains(RESPONSE_EMAIL), "response PII leaked");
    for email in &concurrent_emails {
        assert!(
            !serialized.contains(email),
            "concurrent PII leaked: {email}"
        );
    }
    assert!(serialized.contains("[REDACTED]"));
    for retained in [
        "scope-input",
        "scope-output",
        "tool-input",
        "tool-output",
        "safe-gemini-input",
        "safe-gemini-output",
    ] {
        assert!(
            serialized.contains(retained),
            "missing safe value {retained}"
        );
    }

    abort_in_flight_sanitization().await;
    deregister_subscriber(SUBSCRIBER_NAME).expect("deregister event subscriber");
    let teardown_started = Instant::now();
    cleanup
        .activation
        .take()
        .expect("activation remains live")
        .clear()
        .expect("clear Rampart native plugin");
    assert!(
        teardown_started.elapsed() < Duration::from_secs(2),
        "plugin teardown waited on cancelled inference"
    );
    tokio::time::sleep(Duration::from_millis(750)).await;
}

async fn run_concurrent_tool_calls(count: usize) -> Vec<String> {
    let mut tasks = Vec::with_capacity(count);
    let mut emails = Vec::with_capacity(count * 2);
    for index in 0..count {
        let input_email = format!("fanout-input-{index}@example.com");
        let output_email = format!("fanout-output-{index}@example.org");
        emails.push(input_email.clone());
        emails.push(output_email.clone());
        tasks.push(tokio::spawn(async move {
            TASK_SCOPE_STACK
                .scope(create_scope_stack(), async move {
                    let callback_output = output_email.clone();
                    let result = tokio::time::timeout(
                        Duration::from_secs(10),
                        tool_call_execute(
                            ToolCallExecuteParams::builder()
                                .name("concurrent_lookup")
                                .args(json!({"email": input_email, "safe": index}))
                                .func(Arc::new(move |arguments| {
                                    assert_eq!(arguments["safe"], index);
                                    let output = callback_output.clone();
                                    Box::pin(
                                        async move { Ok(json!({"email": output, "safe": index})) },
                                    )
                                }))
                                .build(),
                        ),
                    )
                    .await
                    .expect("concurrent tool call timed out")
                    .expect("concurrent tool call failed");
                    assert_eq!(result["email"], output_email);
                })
                .await;
        }));
    }
    for task in tasks {
        task.await.expect("concurrent tool task failed");
    }
    emails
}

async fn abort_in_flight_sanitization() {
    let selected_text = format!(
        "{} cancellation@example.com",
        "private context ".repeat(256)
    );
    let task = tokio::spawn(async move {
        TASK_SCOPE_STACK
            .scope(create_scope_stack(), async move {
                let _ = tool_call_execute(
                    ToolCallExecuteParams::builder()
                        .name("cancelled_lookup")
                        .args(json!({"content": selected_text}))
                        .func(Arc::new(|arguments| Box::pin(async move { Ok(arguments) })))
                        .build(),
                )
                .await;
            })
            .await;
    });
    tokio::time::sleep(Duration::from_millis(2)).await;
    task.abort();
    assert!(
        task.await
            .expect_err("cancelled call unexpectedly completed")
            .is_cancelled(),
        "aborted call should report task cancellation"
    );
}

fn plugin_library() -> PathBuf {
    std::env::var_os("NEMO_RELAY_TEST_RAMPART_LIBRARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join("release")
                .join(platform_library_name())
        })
}

fn model_snapshot() -> PathBuf {
    if let Some(path) = std::env::var_os("NEMO_RELAY_TEST_RAMPART_MODEL") {
        return PathBuf::from(path);
    }
    let home = std::env::var_os("HOME").expect("set NEMO_RELAY_TEST_RAMPART_MODEL");
    PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nationaldesignstudio--rampart/snapshots")
        .join("b1993e4e68b082835b80ffc65acc03325ea2e501")
}

fn platform_library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "nemo_relay_pii_rampart_plugin.dll"
    } else if cfg!(target_os = "macos") {
        "libnemo_relay_pii_rampart_plugin.dylib"
    } else {
        "libnemo_relay_pii_rampart_plugin.so"
    }
}

fn write_manifest(directory: &Path, library: &Path) -> PathBuf {
    assert!(
        library.is_file(),
        "plugin library does not exist: {library:?}"
    );
    let digest = Sha256::digest(std::fs::read(library).expect("read plugin library"))
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let manifest = format!(
        r#"manifest_version = 1

[plugin]
id = "pii_rampart"
kind = "rust_dynamic"

[compat]
relay = ">=0.8,<1.0"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[integrity]
sha256 = "sha256:{digest}"

[load]
library = {library}
symbol = "nemo_relay_register_plugin"
"#,
        library = serde_json::to_string(&library.to_string_lossy()).expect("quote library path"),
    );
    let path = directory.join("relay-plugin.toml");
    std::fs::write(&path, manifest).expect("write native plugin manifest");
    path
}
