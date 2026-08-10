// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the native Rampart callback adapters.

use super::*;
use nemo_relay::error::FlowError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

fn block_on<T>(future: impl Future<Output = T>) -> T {
    Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(future)
}

#[test]
fn codec_context_preserves_supported_identities_and_fails_unknown_values_to_opaque() {
    let cases = [
        ("none", None, LlmCodecIdentity::None),
        (
            "builtin",
            Some("openai_chat"),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat),
        ),
        (
            "builtin",
            Some("openai_responses"),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiResponses),
        ),
        (
            "builtin",
            Some("anthropic_messages"),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::AnthropicMessages),
        ),
        (
            "builtin",
            Some("gemini_generate_content"),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::GeminiGenerateContent),
        ),
        (
            "runtime",
            Some("third_party_codec"),
            LlmCodecIdentity::Runtime("third_party_codec".into()),
        ),
        ("builtin", None, LlmCodecIdentity::Opaque),
        ("builtin", Some("unknown"), LlmCodecIdentity::Opaque),
        ("runtime", None, LlmCodecIdentity::Opaque),
        ("unknown", Some("codec"), LlmCodecIdentity::Opaque),
    ];

    for (codec_kind, codec_id, expected) in cases {
        assert_eq!(
            CodecContext {
                codec_kind: codec_kind.into(),
                codec_id: codec_id.map(str::to_string),
            }
            .into_identity(),
            expected
        );
    }
}

#[test]
fn tool_callback_forwards_values_and_propagates_errors() {
    let observed = Arc::new(Mutex::new(None));
    let callback: ToolSanitizeFn = {
        let observed = Arc::clone(&observed);
        Arc::new(move |name, value| {
            let observed = Arc::clone(&observed);
            Box::pin(async move {
                *observed.lock().unwrap() = Some((name, value));
                Ok(serde_json::json!({"sanitized": true}))
            })
        })
    };
    let output = block_on(tool_callback(callback)(serde_json::json!({
        "name": "read_file",
        "value": {"path": "/tmp/private"}
    })))
    .unwrap();

    assert_eq!(output, serde_json::json!({"sanitized": true}));
    assert_eq!(
        *observed.lock().unwrap(),
        Some(("read_file".into(), serde_json::json!({"path": "/tmp/private"})))
    );

    let failing: ToolSanitizeFn = Arc::new(|_, _| {
        Box::pin(async { Err(FlowError::Internal("sanitizer failed".into())) })
    });
    let error = block_on(tool_callback(failing)(
        serde_json::json!({"name": "tool", "value": {}}),
    ))
    .unwrap_err();
    assert!(error.contains("sanitizer failed"));
}

#[test]
fn llm_callbacks_preserve_codec_identity_and_encode_omission_as_null() {
    let request_identity = Arc::new(Mutex::new(None));
    let request_callback: LlmSanitizeRequestFn = {
        let request_identity = Arc::clone(&request_identity);
        Arc::new(move |_request, context| {
            let request_identity = Arc::clone(&request_identity);
            Box::pin(async move {
                *request_identity.lock().unwrap() = Some(context.codec().clone());
                Ok(None)
            })
        })
    };
    let request_output = block_on(llm_request_callback(request_callback)(serde_json::json!({
        "request": {"headers": {}, "content": {"messages": []}},
        "context": {"codec_kind": "runtime", "codec_id": "custom"}
    })))
    .unwrap();
    assert_eq!(request_output, Json::Null);
    assert_eq!(
        *request_identity.lock().unwrap(),
        Some(LlmCodecIdentity::Runtime("custom".into()))
    );

    let response_identity = Arc::new(Mutex::new(None));
    let response_callback: LlmSanitizeResponseFn = {
        let response_identity = Arc::clone(&response_identity);
        Arc::new(move |_response, context| {
            let response_identity = Arc::clone(&response_identity);
            Box::pin(async move {
                *response_identity.lock().unwrap() = Some(context.codec().clone());
                Ok(None)
            })
        })
    };
    let response_output = block_on(llm_response_callback(response_callback)(serde_json::json!({
        "response": {"choices": []},
        "context": {"codec_kind": "builtin", "codec_id": "openai_chat"}
    })))
    .unwrap();
    assert_eq!(response_output, Json::Null);
    assert_eq!(
        *response_identity.lock().unwrap(),
        Some(LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat))
    );
}

#[test]
fn callback_adapters_reject_malformed_invocations_before_user_code_runs() {
    let called = Arc::new(AtomicBool::new(false));
    let callback: ToolSanitizeFn = {
        let called = Arc::clone(&called);
        Arc::new(move |_, value| {
            called.store(true, Ordering::Relaxed);
            Box::pin(async move { Ok(value) })
        })
    };

    let error = block_on(tool_callback(callback)(serde_json::json!({
        "name": "missing-value"
    })))
    .unwrap_err();

    assert!(error.contains("invalid Rampart invocation"));
    assert!(!called.load(Ordering::Relaxed));
}
