// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live coverage for the packaged Rampart native plugin.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay::api::runtime::{LlmCodecIdentity, TASK_SCOPE_STACK, create_scope_stack};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::{ToolCallExecuteParams, ToolExecutionResult, tool_call_execute};
use nemo_relay::codec::gemini_generate_content::GeminiGenerateContentCodec;
use nemo_relay::codec::oci_genai::OCIGenAIChatCodec;
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::response::AnnotatedLlmResponse;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::Result as RelayResult;
use nemo_relay::plugin::PluginConfig;
use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationSpec, DynamicPluginKind, PluginHostActivation,
};
use serde_json::{Map, Value as Json, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const SUBSCRIBER_NAME: &str = "pii-rampart-native-live";
const EXPLICIT_SUBSCRIBER_NAME: &str = "pii-rampart-native-v4-codec-live";
const REQUEST_EMAIL: &str = "alex.fournier@example.com";
const RESPONSE_EMAIL: &str = "reviewer@example.org";
const TOOL_ANNOTATION_EMAIL: &str = "annotation-owner@example.net";
const RUNTIME_CODEC_ID: &str = "test.rampart.openai.v4";
static NATIVE_TEST_LOCK: Mutex<()> = Mutex::new(());

struct TestCleanup {
    activation: Option<PluginHostActivation>,
    subscriber_name: &'static str,
}

impl Drop for TestCleanup {
    fn drop(&mut self) {
        let _ = deregister_subscriber(self.subscriber_name);
        if let Some(activation) = self.activation.take() {
            let _ = activation.clear();
        }
    }
}

/// OpenAI wire codec with a runtime identity.
///
/// The v3 identity-only bridge cannot resolve this codec. A retained and
/// sanitized request therefore proves that the rebuilt plugin received and
/// used the typed ABI-v4 codec capability, rather than falling back to a
/// built-in codec selected from an identity string.
struct RuntimeOpenAiCodec;

impl LlmCodec for RuntimeOpenAiCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::Runtime(RUNTIME_CODEC_ID.into())
    }

    fn decode(&self, request: &LlmRequest) -> RelayResult<AnnotatedLlmRequest> {
        LlmCodec::decode(&OpenAIChatCodec, request)
    }

    fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> RelayResult<LlmRequest> {
        LlmCodec::encode(&OpenAIChatCodec, annotated, original)
    }
}

impl LlmResponseCodec for RuntimeOpenAiCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::Runtime(RUNTIME_CODEC_ID.into())
    }

    fn decode_response(&self, response: &Json) -> RelayResult<AnnotatedLlmResponse> {
        OpenAIChatCodec.decode_response(response)
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
        json!({
            "model_path": "/model",
            "target_paths": ["/message"],
            "codec": "openai_chat",
            "executor": {"worker_threads": 2}
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
        json!({
            "model_path": "/model",
            "target_paths": ["/content"],
            "executor": {"worker_threads": 0}
        }),
        json!({
            "model_path": "/model",
            "target_paths": ["/message"],
            "codec": "oci_genai"
        }),
        json!({
            "model_path": "/model",
            "target_paths": ["/content"],
            "executor": {"worker_threads": 2, "queue_depth": 8}
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
    let _guard = NATIVE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build live-test runtime")
        .block_on(run_live_test());
}

#[test]
#[ignore = "requires a built native library and the pinned Rampart model snapshot"]
fn native_plugin_v4_codec_capability_obeys_normalized_projection_boundaries() {
    let _guard = NATIVE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build v4 codec live-test runtime")
        .block_on(run_explicit_codec_live_test());
}

async fn run_explicit_codec_live_test() {
    const OPENAI_REQUEST_EMAIL: &str = "openai.request@example.com";
    const OPENAI_RESPONSE_EMAIL: &str = "openai.response@example.org";
    const RUNTIME_REQUEST_EMAIL: &str = "runtime.request@example.com";
    const RUNTIME_RESPONSE_EMAIL: &str = "runtime.response@example.org";

    let library = plugin_library();
    let model = model_snapshot();
    let manifest_dir = TempDir::new().expect("create v4 codec manifest directory");
    let manifest = write_manifest(manifest_dir.path(), &library);
    let config = json!({
        "model_path": model,
        "target_paths": ["/message"],
        "target_path_patterns": ["/messages/*/content"],
        "mark": false,
        "tool_input": false,
        "tool_output": false,
        "executor": {"worker_threads": 1}
    })
    .as_object()
    .expect("explicit codec plugin config is an object")
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
    .expect("activate Rampart native plugin for v4 codec coverage");
    assert!(
        !report.has_errors(),
        "Rampart v4 codec activation diagnostics: {:?}",
        report.diagnostics
    );
    let mut cleanup = TestCleanup {
        activation: Some(activation),
        subscriber_name: EXPLICIT_SUBSCRIBER_NAME,
    };

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        EXPLICIT_SUBSCRIBER_NAME,
        Arc::new(move |event| {
            captured
                .lock()
                .expect("v4 codec event lock")
                .push(event.clone())
        }),
    )
    .expect("register v4 codec event subscriber");

    let openai_request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "openai-test-model",
            "messages": [{
                "role": "user",
                "content": format!("Email {OPENAI_REQUEST_EMAIL}; safe-openai-v4-request")
            }]
        }),
    };
    let openai_response = json!({
        "id": "chatcmpl-openai-v4",
        "object": "chat.completion",
        "model": "openai-test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": format!("Email {OPENAI_RESPONSE_EMAIL}; safe-openai-v4-response")
            },
            "finish_reason": "stop"
        }]
    });
    let runtime_request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "runtime-openai-test-model",
            "messages": [{
                "role": "user",
                "content": format!("Email {RUNTIME_REQUEST_EMAIL}; safe-runtime-v4-request")
            }]
        }),
    };
    let runtime_response = json!({
        "id": "chatcmpl-runtime-v4",
        "object": "chat.completion",
        "model": "runtime-openai-test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": format!("Email {RUNTIME_RESPONSE_EMAIL}; safe-runtime-v4-response")
            },
            "finish_reason": "stop"
        }]
    });
    let expected_openai_request = openai_request.clone();
    let expected_openai_response = openai_response.clone();
    let expected_runtime_request = runtime_request.clone();
    let expected_runtime_response = runtime_response.clone();

    let openai_request_seen = Arc::new(Mutex::new(None::<LlmRequest>));
    let openai_request_capture = Arc::clone(&openai_request_seen);
    let runtime_request_seen = Arc::new(Mutex::new(None::<LlmRequest>));
    let runtime_request_capture = Arc::clone(&runtime_request_seen);

    let (openai_result, runtime_result) = TASK_SCOPE_STACK
        .scope(create_scope_stack(), async move {
            let callback_response = openai_response.clone();
            let codec = Arc::new(OpenAIChatCodec);
            let openai_result = llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name("openai-v4-codec")
                    .request(openai_request)
                    .func(Arc::new(move |request| {
                        *openai_request_capture
                            .lock()
                            .expect("OpenAI v4 request capture lock") = Some(request);
                        let response = callback_response.clone();
                        Box::pin(async move { Ok(response) })
                    }))
                    .codec(codec.clone())
                    .response_codec(codec)
                    .build(),
            )
            .await
            .expect("execute managed OpenAI v4 codec call");

            let callback_response = runtime_response.clone();
            let codec = Arc::new(RuntimeOpenAiCodec);
            let runtime_result = llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name("runtime-openai-v4-codec")
                    .request(runtime_request)
                    .func(Arc::new(move |request| {
                        *runtime_request_capture
                            .lock()
                            .expect("runtime v4 request capture lock") = Some(request);
                        let response = callback_response.clone();
                        Box::pin(async move { Ok(response) })
                    }))
                    .codec(codec.clone())
                    .response_codec(codec)
                    .build(),
            )
            .await
            .expect("execute managed runtime v4 codec call");

            (openai_result, runtime_result)
        })
        .await;

    let openai_request_seen = openai_request_seen
        .lock()
        .expect("OpenAI v4 request capture lock");
    assert_eq!(
        openai_request_seen
            .as_ref()
            .expect("OpenAI v4 provider callback ran")
            .content,
        expected_openai_request.content,
        "Rampart must not mutate the provider-visible OpenAI request content"
    );
    assert_eq!(openai_result, expected_openai_response);
    let runtime_request_seen = runtime_request_seen
        .lock()
        .expect("runtime v4 request capture lock");
    assert_eq!(
        runtime_request_seen
            .as_ref()
            .expect("runtime v4 provider callback ran")
            .content,
        expected_runtime_request.content,
        "Rampart must not mutate the provider-visible runtime-codec request content"
    );
    assert_eq!(runtime_result, expected_runtime_response);
    flush_subscribers().expect("flush explicit v4 codec events");
    let events = events.lock().expect("v4 codec event lock");
    let openai_start = scope_event(&events, "openai-v4-codec", ScopeCategory::Start);
    let openai_end = scope_event(&events, "openai-v4-codec", ScopeCategory::End);
    let runtime_start = scope_event(&events, "runtime-openai-v4-codec", ScopeCategory::Start);
    let runtime_end = scope_event(&events, "runtime-openai-v4-codec", ScopeCategory::End);

    let openai_start = openai_start
        .input()
        .expect("supported OpenAI input remains observable");
    assert!(openai_start.to_string().contains("safe-openai-v4-request"));
    assert!(openai_start.to_string().contains("[REDACTED]"));
    let openai_end = openai_end
        .output()
        .expect("supported OpenAI output remains observable");
    assert!(openai_end.to_string().contains("safe-openai-v4-response"));
    assert!(openai_end.to_string().contains("[REDACTED]"));

    let runtime_start = runtime_start
        .input()
        .expect("typed ABI-v4 runtime request capability remains observable");
    assert!(
        runtime_start
            .to_string()
            .contains("safe-runtime-v4-request")
    );
    assert!(runtime_start.to_string().contains("[REDACTED]"));
    assert!(
        runtime_end.output().is_none(),
        "runtime response projection must fail closed until a lossless output overlay exists"
    );
    assert!(runtime_end.annotated_response().is_none());

    let serialized = serde_json::to_string(&*events).expect("serialize explicit v4 codec events");
    for email in [
        OPENAI_REQUEST_EMAIL,
        OPENAI_RESPONSE_EMAIL,
        RUNTIME_REQUEST_EMAIL,
        RUNTIME_RESPONSE_EMAIL,
    ] {
        assert!(!serialized.contains(email), "v4 codec PII leaked: {email}");
    }
    for retained in [
        "safe-openai-v4-request",
        "safe-openai-v4-response",
        // A retained runtime-codec request is the capability proof: the v3
        // identity-only bridge cannot resolve and project this request.
        "safe-runtime-v4-request",
    ] {
        assert!(
            serialized.contains(retained),
            "sanitized v4 codec copy lost safe marker {retained}: {serialized}"
        );
    }
    assert!(!serialized.contains("safe-runtime-v4-response"));
    assert!(serialized.contains("[REDACTED]"));
    drop(events);

    deregister_subscriber(EXPLICIT_SUBSCRIBER_NAME).expect("deregister v4 codec event subscriber");
    cleanup
        .activation
        .take()
        .expect("v4 codec activation remains live")
        .clear()
        .expect("clear Rampart v4 codec plugin");
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
        subscriber_name: SUBSCRIBER_NAME,
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
    let oci_request_seen = Arc::new(Mutex::new(None::<LlmRequest>));
    let oci_request_capture = Arc::clone(&oci_request_seen);

    let (tool_result, llm_result, gemini_result, oci_result) = TASK_SCOPE_STACK
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
                            Ok(ToolExecutionResult::annotated(
                                json!({"email": RESPONSE_EMAIL, "safe": "tool-output"}),
                                json!({
                                    "email": TOOL_ANNOTATION_EMAIL,
                                    "safe": "tool-annotation"
                                }),
                            ))
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

            let oci_request = LlmRequest {
                headers: Map::new(),
                content: json!({
                    "modelId": "oci-test-model",
                    "vendorEnvelope": {"revision": 7},
                    "chatRequest": {
                        "apiFormat": "GENERIC",
                        "messages": [{
                            "role": "USER",
                            "content": [
                                {"type": "TEXT", "text": format!("Email {REQUEST_EMAIL}; ordinary first request segment")},
                                {"type": "TEXT", "text": format!("Contact {REQUEST_EMAIL}; ordinary second request segment")}
                            ]
                        }]
                    }
                }),
            };
            let oci_response = json!({
                "modelId": "oci-test-model",
                "vendorEnvelope": {"revision": 9},
                "chatResponse": {
                    "apiFormat": "GENERIC",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "ASSISTANT",
                                "content": [
                                    {"type": "TEXT", "text": format!("Email {RESPONSE_EMAIL}; ordinary first response segment")},
                                    {"type": "TEXT", "text": format!("Contact {RESPONSE_EMAIL}; ordinary second response segment")}
                                ]
                            },
                            "finishReason": "stop"
                        },
                        {
                            "index": 1,
                            "message": {
                                "role": "ASSISTANT",
                                "content": [{"type": "TEXT", "text": format!("Alternate {RESPONSE_EMAIL}; ordinary alternate response segment")}]
                            },
                            "finishReason": "stop"
                        }
                    ]
                }
            });
            let callback_response = oci_response.clone();
            let codec = Arc::new(OCIGenAIChatCodec);
            let oci_result = llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name("oci")
                    .request(oci_request)
                    .func(Arc::new(move |request| {
                        *oci_request_capture.lock().expect("OCI capture lock") = Some(request);
                        let response = callback_response.clone();
                        Box::pin(async move { Ok(response) })
                    }))
                    .codec(codec.clone())
                    .response_codec(codec)
                    .build(),
            )
            .await
            .expect("execute managed OCI call");

            pop_scope(
                PopScopeParams::builder()
                    .handle_uuid(&scope.uuid)
                    .output(json!({"contact": RESPONSE_EMAIL, "safe": "scope-output"}))
                    .build(),
            )
            .expect("pop live-test scope");
            (tool_result, llm_result, gemini_result, oci_result)
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
    assert_eq!(tool_result.result["email"], RESPONSE_EMAIL);
    assert_eq!(
        tool_result
            .annotation
            .as_ref()
            .expect("tool annotation remains visible to the caller")["email"],
        TOOL_ANNOTATION_EMAIL
    );
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
    {
        let oci_request_seen = oci_request_seen.lock().expect("OCI capture lock");
        let oci_request_seen = oci_request_seen.as_ref().expect("OCI callback ran");
        assert!(oci_request_seen.content.to_string().contains(REQUEST_EMAIL));
        assert_eq!(
            oci_request_seen.content["chatRequest"]["messages"][0]["content"]
                .as_array()
                .expect("provider OCI multipart request remains an array")
                .len(),
            2
        );
    }
    assert!(oci_result.to_string().contains(RESPONSE_EMAIL));
    assert_eq!(
        oci_result["chatResponse"]["choices"]
            .as_array()
            .expect("provider OCI response choices remain an array")
            .len(),
        2
    );

    let concurrent_emails = run_concurrent_tool_calls(8).await;

    flush_subscribers().expect("flush live-test events");
    let serialized = serde_json::to_string(&*events.lock().expect("event lock"))
        .expect("serialize captured events");
    assert!(!serialized.contains(REQUEST_EMAIL), "request PII leaked");
    assert!(!serialized.contains(RESPONSE_EMAIL), "response PII leaked");
    assert!(
        !serialized.contains(TOOL_ANNOTATION_EMAIL),
        "tool annotation PII leaked"
    );
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
        "tool-annotation",
        "safe-gemini-input",
        "safe-gemini-output",
        "ordinary first request segment",
        "ordinary second request segment",
        "ordinary first response segment",
        "ordinary second response segment",
        "ordinary alternate response segment",
    ] {
        assert!(
            serialized.contains(retained),
            "missing safe value {retained}"
        );
    }

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
        "plugin teardown exceeded two seconds after the subscriber flush barrier"
    );
    tokio::time::sleep(Duration::from_millis(750)).await;
}

fn scope_event<'a>(events: &'a [Event], name: &str, category: ScopeCategory) -> &'a Event {
    events
        .iter()
        .find(|event| event.name() == name && event.scope_category() == Some(category))
        .unwrap_or_else(|| panic!("missing {category:?} event for {name}"))
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
                                    Box::pin(async move {
                                        Ok(ToolExecutionResult::new(
                                            json!({"email": output, "safe": index}),
                                        ))
                                    })
                                }))
                                .build(),
                        ),
                    )
                    .await
                    .expect("concurrent tool call timed out")
                    .expect("concurrent tool call failed");
                    assert_eq!(result.result["email"], output_email);
                })
                .await;
        }));
    }
    for task in tasks {
        task.await.expect("concurrent tool task failed");
    }
    emails
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
relay = ">=0.8.0,<1.0"
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
