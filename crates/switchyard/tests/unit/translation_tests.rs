// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;
use switchyard_translation::{
    FileSource, MediaSource, ToolResult, TranslationDiagnostic, TranslationError,
};

fn request(content: Json) -> LlmRequest {
    LlmRequest {
        headers: Map::new(),
        content,
    }
}

#[test]
fn wire_formats_and_message_helpers_preserve_portable_requests() {
    assert_eq!(
        wire_format(WireProtocol::OpenaiChat),
        WireFormat::OpenAiChat
    );
    assert_eq!(
        wire_format(WireProtocol::OpenaiResponses),
        WireFormat::OpenAiResponses
    );
    assert_eq!(
        wire_format(WireProtocol::AnthropicMessages),
        WireFormat::AnthropicMessages
    );

    let engine = translation_engine();
    let decoded = decode_request(
        &engine,
        WireProtocol::OpenaiChat,
        &request(json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "latest"}
            ]
        })),
    )
    .unwrap();
    assert_eq!(latest_user_prompt(&decoded).as_deref(), Some("latest"));
    assert_eq!(recent_message_window(&decoded, 2).messages.len(), 2);
    assert_eq!(recent_message_window(&decoded, 20).messages.len(), 3);
    assert!(
        validate_portable_request(
            &engine,
            WireProtocol::OpenaiChat,
            &request(json!({"model": "test", "messages": [{"role": "user", "content": "ok"}]})),
        )
        .is_ok()
    );
}

#[test]
fn portability_guards_reject_provider_specific_request_fields() {
    let engine = translation_engine();
    for key in [
        "cache_control",
        "audio",
        "thinking",
        "computer_use",
        "server_tool_use",
    ] {
        let mut content = serde_json::Map::from_iter([
            ("model".into(), json!("test")),
            (
                "messages".into(),
                json!([{"role": "user", "content": "ok"}]),
            ),
        ]);
        content.insert(key.into(), json!(true));
        assert!(
            validate_portable_request(
                &engine,
                WireProtocol::OpenaiChat,
                &request(Json::Object(content)),
            )
            .is_err()
        );
    }
    assert!(
        validate_portable_request(
            &engine,
            WireProtocol::OpenaiResponses,
            &request(json!({
                "model": "test",
                "input": "ok",
                "stream_options": {"include_usage": "yes"}
            })),
        )
        .is_err()
    );
}

#[test]
fn anthropic_image_sources_must_be_complete_and_known() {
    let engine = translation_engine();
    assert!(
        validate_portable_request(
            &engine,
            WireProtocol::AnthropicMessages,
            &request(json!({
                "model": "test",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": [{"type": "image"}]}]
            })),
        )
        .is_err()
    );
    for source in [
        json!({}),
        json!({"type": "url", "url": ""}),
        json!({"type": "base64", "media_type": "image/png", "data": ""}),
        json!({"type": "asset", "id": "image-1"}),
    ] {
        assert!(contains_invalid_anthropic_image_source(&json!({
            "messages": [{"content": [{"type": "image", "source": source}]}]
        })));
    }
    assert!(!contains_invalid_anthropic_image_source(&json!({
        "messages": [{"content": [{"type": "image", "source": {"type": "url", "url": "https://example.test/image.png"}}]}]
    })));
    assert!(!contains_invalid_anthropic_image_source(
        &json!({"text": ["plain", "values"]})
    ));
}

#[test]
fn response_portability_allows_only_shared_content() {
    assert!(
        ensure_portable_response(
            WireProtocol::OpenaiChat,
            &json!({"choices": [{"message": {"content": "ok"}}]}),
        )
        .is_ok()
    );
    for message in [json!({"audio": {}}), json!({"reasoning_content": "hidden"})] {
        assert!(
            ensure_portable_response(
                WireProtocol::OpenaiChat,
                &json!({"choices": [{"message": message}]}),
            )
            .is_err()
        );
    }

    assert!(
        ensure_portable_response(
            WireProtocol::OpenaiResponses,
            &json!({"output": [{"type": "message"}]}),
        )
        .is_ok()
    );
    for kind in [
        "reasoning",
        "computer_call",
        "computer_call_output",
        "web_search_call",
    ] {
        assert!(
            ensure_portable_response(
                WireProtocol::OpenaiResponses,
                &json!({"output": [{"type": kind}]}),
            )
            .is_err()
        );
    }

    assert!(
        ensure_portable_response(
            WireProtocol::AnthropicMessages,
            &json!({"content": [{"type": "text"}, {"type": "tool_use"}]}),
        )
        .is_ok()
    );
    assert!(
        ensure_portable_response(
            WireProtocol::AnthropicMessages,
            &json!({"content": [{"type": "thinking"}]}),
        )
        .is_err()
    );
}

#[test]
fn data_uri_and_recursive_key_helpers_distinguish_valid_shapes() {
    assert_eq!(
        base64_data_uri_parts("data:image/png;base64,Zm9v"),
        Some(("image/png", "Zm9v"))
    );
    for value in [
        "https://example.test/image.png",
        "data:image/png,Zm9v",
        "data:;base64,Zm9v",
        "data:image/png;base64,",
    ] {
        assert!(base64_data_uri_parts(value).is_none());
    }
    assert!(contains_any_key_recursive(
        &json!({"outer": [{"forbidden": true}]}),
        &["forbidden"],
    ));
    assert!(!contains_any_key_recursive(
        &json!({"outer": [{"allowed": true}]}),
        &["forbidden"],
    ));
}

#[test]
fn unsupported_content_and_image_sources_are_rejected_before_translation() {
    assert!(!unsupported_content_block(&ContentBlock::Text {
        text: "ok".into(),
    }));
    assert!(unsupported_content_block(&ContentBlock::Reasoning {
        text: "hidden".into(),
        signature: None,
    }));
    assert!(unsupported_content_block(&ContentBlock::Audio {
        source: MediaSource::Raw(json!({})),
    }));
    assert!(unsupported_content_block(&ContentBlock::Video {
        source: MediaSource::Raw(json!({})),
    }));
    assert!(unsupported_content_block(&ContentBlock::File {
        source: FileSource::Raw(json!({})),
    }));
    assert!(unsupported_content_block(&ContentBlock::Unknown {
        provider: WireFormat::OpenAiChat.into(),
        raw: json!({}),
    }));
    assert!(unsupported_content_block(&ContentBlock::ToolResult(
        ToolResult {
            tool_call_id: "call-1".into(),
            content: vec![ContentBlock::Audio {
                source: MediaSource::Raw(json!({})),
            }],
            is_error: None,
        },
    )));

    assert!(!invalid_image_source(&ImageSource::Url {
        url: "https://example.test/image.png".into(),
        detail: None,
    }));
    assert!(unsupported_content_block(&ContentBlock::Image {
        source: ImageSource::Raw(json!({})),
    }));
    assert!(!invalid_image_source(&ImageSource::Url {
        url: "data:image/png;base64,Zm9v".into(),
        detail: None,
    }));
    assert!(invalid_image_source(&ImageSource::Url {
        url: "data:image/png,Zm9v".into(),
        detail: None,
    }));
    assert!(!invalid_image_source(&ImageSource::Base64 {
        media_type: Some("image/png".into()),
        data: "Zm9v".into(),
    }));
    assert!(invalid_image_source(&ImageSource::Base64 {
        media_type: None,
        data: "Zm9v".into(),
    }));
    assert!(invalid_image_source(&ImageSource::Base64 {
        media_type: Some("image/png".into()),
        data: String::new(),
    }));
    assert!(invalid_image_source(&ImageSource::Raw(json!({}))));
}

#[test]
fn translation_helpers_cover_diagnostics_and_cross_protocol_responses() {
    assert!(ensure_no_diagnostics(&[]).is_ok());
    assert!(
        ensure_no_diagnostics(&[TranslationDiagnostic::warning("lossy", "not portable")]).is_err()
    );
    assert!(matches!(
        translation_error(TranslationError::Other("invalid".into())),
        FlowError::InvalidArgument(message) if message.contains("invalid")
    ));
    assert!(portable_stream_options(&json!({"include_usage": true})));
    assert!(!portable_stream_options(
        &json!({"include_usage": true, "extra": false})
    ));
    assert!(!portable_stream_options(&json!(true)));

    let response = json!({
        "id": "chat-1",
        "object": "chat.completion",
        "model": "test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }]
    });
    let engine = translation_engine();
    assert_eq!(
        translate_response(
            &engine,
            WireProtocol::OpenaiChat,
            WireProtocol::OpenaiChat,
            &response,
        )
        .unwrap(),
        response
    );
    let translated = translate_response(
        &engine,
        WireProtocol::OpenaiChat,
        WireProtocol::AnthropicMessages,
        &response,
    )
    .unwrap();
    assert_eq!(translated["type"], "message");
    assert_eq!(translated["role"], "assistant");
    assert_eq!(
        translated["content"],
        json!([{ "type": "text", "text": "ok" }])
    );
}
