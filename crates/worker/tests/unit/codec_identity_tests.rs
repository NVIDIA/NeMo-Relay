// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_gemini_codec_identity_decoded_as_builtin_not_opaque() {
    use nemo_relay_worker_proto::v1::LlmCodecIdentity as ProtoIdentity;
    use nemo_relay_worker_proto::v1::LlmCodecKind;

    let proto = ProtoIdentity {
        kind: LlmCodecKind::Builtin as i32,
        id: Some("gemini_generate_content".to_string()),
    };
    let identity = codec_identity_from_proto(Some(&proto));
    assert_eq!(
        identity,
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::GeminiGenerateContent),
        "Gemini generateContent codec id must decode to BuiltIn(GeminiGenerateContent), not Opaque"
    );
}
