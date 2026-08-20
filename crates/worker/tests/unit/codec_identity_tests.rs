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
