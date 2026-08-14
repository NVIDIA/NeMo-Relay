// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility tests for shared LLM codec identities.

use nemo_relay_types::codec::identity::{BuiltinLlmCodec, LlmCodecIdentity};
use serde_json::json;

#[test]
fn builtin_codec_ids_round_trip() {
    for &codec in BuiltinLlmCodec::ALL {
        assert_eq!(BuiltinLlmCodec::from_id(codec.id()), Some(codec));
        assert_eq!(serde_json::to_value(codec).unwrap(), json!(codec.id()));
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
