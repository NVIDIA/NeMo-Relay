// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the native Rampart typed callback adapters.

use super::*;
use serde_json::json;

#[test]
fn codec_identity_bridge_preserves_every_supported_identity() {
    let cases = [
        (NativeLlmCodecIdentity::None, LlmCodecIdentity::None),
        (
            NativeLlmCodecIdentity::BuiltIn(NativeBuiltinLlmCodec::OpenAiChat),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat),
        ),
        (
            NativeLlmCodecIdentity::BuiltIn(NativeBuiltinLlmCodec::OpenAiResponses),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiResponses),
        ),
        (
            NativeLlmCodecIdentity::BuiltIn(NativeBuiltinLlmCodec::AnthropicMessages),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::AnthropicMessages),
        ),
        (
            NativeLlmCodecIdentity::BuiltIn(NativeBuiltinLlmCodec::OCIGenAI),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OCIGenAI),
        ),
        (
            NativeLlmCodecIdentity::BuiltIn(NativeBuiltinLlmCodec::GeminiGenerateContent),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::GeminiGenerateContent),
        ),
        (
            NativeLlmCodecIdentity::Runtime("third_party_codec".into()),
            LlmCodecIdentity::Runtime("third_party_codec".into()),
        ),
        (NativeLlmCodecIdentity::Opaque, LlmCodecIdentity::Opaque),
    ];

    for (native, expected) in cases {
        assert_eq!(core_codec_identity(&native), expected);
    }
}

#[test]
fn sdk_executor_defaults_to_one_thread_and_accepts_component_override() {
    let plugin = RampartNativePlugin;
    assert_eq!(plugin.executor_config().worker_threads, 1);

    let config = json!({"executor": {"worker_threads": 4}})
        .as_object()
        .unwrap()
        .clone();
    assert_eq!(
        plugin
            .executor_config_for_component(&config)
            .unwrap()
            .worker_threads,
        4
    );
}

#[test]
fn executor_config_is_validated_separately_from_rampart_config() {
    let plugin = RampartNativePlugin;
    let config = json!({
        "model_path": "/model",
        "target_paths": ["/content"],
        "executor": {"worker_threads": 0}
    })
    .as_object()
    .unwrap()
    .clone();

    let rampart_only = rampart_config(&config);
    assert!(!rampart_only.contains_key("executor"));
    assert_eq!(rampart_only["model_path"], "/model");

    let diagnostics = plugin.validate(&config);
    let executor = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "pii_rampart.invalid_executor")
        .expect("invalid executor must produce a diagnostic");
    assert_eq!(executor.level, DiagnosticLevel::Error);
    assert_eq!(executor.field.as_deref(), Some("executor.worker_threads"));
    assert!(executor.message.contains("greater than zero"));
    assert!(
        diagnostics
            .iter()
            .all(|diagnostic| diagnostic.field.as_deref() != Some("executor")),
        "the SDK-owned executor field must not be reported as an unknown Rampart field"
    );
}

#[test]
fn unknown_executor_fields_are_rejected() {
    let plugin = RampartNativePlugin;
    let config = json!({
        "model_path": "/model",
        "target_paths": ["/content"],
        "executor": {"worker_threads": 2, "queue_depth": 8}
    })
    .as_object()
    .unwrap()
    .clone();

    let diagnostics = plugin.validate(&config);
    let unknown = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "pii_rampart.unknown_executor_field")
        .expect("unknown executor field must produce a diagnostic");
    assert_eq!(unknown.level, DiagnosticLevel::Error);
    assert_eq!(unknown.field.as_deref(), Some("executor.queue_depth"));
}

#[test]
fn bridge_errors_distinguish_request_and_response_capabilities() {
    assert!(
        missing_request_codec()
            .to_string()
            .contains("request codec capability")
    );
    assert!(
        missing_response_codec()
            .to_string()
            .contains("response codec capability")
    );
}
