// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn known_builtin_codec_identities_decode_as_builtins() {
    use wire::LlmCodecReference as ProtoIdentity;

    for codec in [
        BuiltinLlmCodec::OpenAiChat,
        BuiltinLlmCodec::OpenAiResponses,
        BuiltinLlmCodec::AnthropicMessages,
        BuiltinLlmCodec::OCIGenAI,
        BuiltinLlmCodec::GeminiGenerateContent,
    ] {
        let proto = ProtoIdentity {
            kind: "builtin".into(),
            id: Some(codec.id().to_owned()),
        };
        assert_eq!(
            codec_identity_from_proto(Some(&proto)).unwrap(),
            LlmCodecIdentity::BuiltIn(codec),
        );
    }
}

#[test]
fn unknown_codec_identity_does_not_prevent_capability_access() {
    use wire::LlmCodecReference as ProtoIdentity;

    let proto = ProtoIdentity {
        kind: "builtin".into(),
        id: Some("future_provider".to_owned()),
    };
    assert_eq!(
        codec_identity_from_proto(Some(&proto)).unwrap(),
        LlmCodecIdentity::Runtime("future_provider".into())
    );
}

#[tokio::test]
/// Allows retrying callback registration after a failed host connection.
async fn conditional_callback_name_is_released_when_host_connection_fails() {
    let runtime = PluginRuntime {
        activation_id: "activation".into(),
        auth_token: "token".into(),
        host_endpoint: "unsupported://host".into(),
        host_channel: Arc::new(OnceCell::new()),
        max_message_bytes: nemo_relay_worker_proto::WorkerCallbackMessageLimits::default(),
        callback_name: None,
        diagnostics: Arc::new(OnceCell::new()),
        conditional_middleware_callbacks: Arc::new(Mutex::new(HashMap::new())),
    };

    for _ in 0..2 {
        let error = runtime
            .register_conditional_middleware_guardrail(
                "retryable-gate",
                BTreeSet::from([RuntimeRegistrationKind::Subscriber]),
                "target",
                |_, _| async { Ok(None) },
            )
            .await
            .expect_err("unsupported host endpoint should fail");
        assert!(matches!(error, WorkerSdkError::InvalidInput(_)));
    }

    assert!(
        runtime
            .conditional_middleware_callbacks
            .lock()
            .expect("conditional callback lock")
            .is_empty()
    );
}
