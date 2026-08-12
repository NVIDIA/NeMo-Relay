// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility tests for shared LLM codec identities.

use nemo_relay_types::codec::identity::{BuiltinLlmCodec, LlmCodecIdentity};
use serde_json::json;

#[test]
fn builtin_codec_ids_round_trip() {
    let cases = [
        (BuiltinLlmCodec::OpenAiChat, "openai_chat"),
        (BuiltinLlmCodec::OpenAiResponses, "openai_responses"),
        (BuiltinLlmCodec::AnthropicMessages, "anthropic_messages"),
        (BuiltinLlmCodec::OCIGenAI, "oci_genai"),
        (
            BuiltinLlmCodec::GeminiGenerateContent,
            "gemini_generate_content",
        ),
    ];

    for (codec, id) in cases {
        assert_eq!(codec.id(), id);
        assert_eq!(BuiltinLlmCodec::from_id(id), Some(codec));
    }
}

#[test]
fn builtin_codec_ids_reject_unknown_values() {
    for id in ["", "openai-chat", "OpenAI_chat", "unknown"] {
        assert_eq!(BuiltinLlmCodec::from_id(id), None);
    }
}

#[test]
fn codec_identity_preserves_the_native_plugin_json_contract() {
    let cases = [
        (LlmCodecIdentity::None, json!({"kind": "none"})),
        (
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat),
            json!({"kind": "builtin", "id": "openai_chat"}),
        ),
        (
            LlmCodecIdentity::Runtime("com.example.chat.v1".to_owned()),
            json!({"kind": "runtime", "id": "com.example.chat.v1"}),
        ),
        (LlmCodecIdentity::Opaque, json!({"kind": "opaque"})),
    ];

    for (identity, expected) in cases {
        assert_eq!(serde_json::to_value(&identity).unwrap(), expected);
        assert_eq!(
            serde_json::from_value::<LlmCodecIdentity>(expected).unwrap(),
            identity
        );
    }
}
