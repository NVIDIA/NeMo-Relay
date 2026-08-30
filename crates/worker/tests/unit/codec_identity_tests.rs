// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn known_builtin_codec_identities_decode_as_builtins() {
    use nemo_relay_worker_proto::v1::LlmCodecIdentity as ProtoIdentity;
    use nemo_relay_worker_proto::v1::LlmCodecKind;

    for codec in [
        BuiltinLlmCodec::OpenAiChat,
        BuiltinLlmCodec::OpenAiResponses,
        BuiltinLlmCodec::AnthropicMessages,
        BuiltinLlmCodec::OCIGenAI,
        BuiltinLlmCodec::GeminiGenerateContent,
    ] {
        let proto = ProtoIdentity {
            kind: LlmCodecKind::Builtin as i32,
            id: Some(codec.id().to_owned()),
        };
        assert_eq!(
            codec_identity_from_proto(Some(&proto)),
            LlmCodecIdentity::BuiltIn(codec),
        );
    }
}

#[test]
fn unknown_builtin_codec_identity_decodes_as_opaque() {
    use nemo_relay_worker_proto::v1::LlmCodecIdentity as ProtoIdentity;
    use nemo_relay_worker_proto::v1::LlmCodecKind;

    let proto = ProtoIdentity {
        kind: LlmCodecKind::Builtin as i32,
        id: Some("future_provider".to_owned()),
    };
    assert_eq!(
        codec_identity_from_proto(Some(&proto)),
        LlmCodecIdentity::Opaque
    );
}

#[tokio::test]
async fn conditional_callback_name_is_released_when_host_connection_fails() {
    let runtime = PluginRuntime {
        activation_id: "activation".into(),
        auth_token: "token".into(),
        host_endpoint: "unsupported://host".into(),
        host_channel: Arc::new(OnceCell::new()),
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
