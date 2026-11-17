// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structured refusals must survive buffered and streamed normalization.

use nemo_relay_codecs::openai_chat::{OpenAIChatCodec, OpenAIChatStreamingCodec};
use nemo_relay_codecs::openai_responses::{OpenAIResponsesCodec, OpenAIResponsesStreamingCodec};
use nemo_relay_types::codec::content::ContentKind;
use nemo_relay_types::codec::request::{ContentPart, MessageContent};
use nemo_relay_types::codec::response::{ApiSpecificResponse, FinishReason};
use nemo_relay_types::codec::streaming::StreamingCodec;
use nemo_relay_types::codec::traits::LlmResponseCodec;
use serde_json::json;

#[test]
fn chat_refusal_is_typed_and_preserves_response_metadata_and_tools() {
    let raw = json!({
        "id":"r", "model":"m", "choices":[{"message":{
            "content":null, "refusal":"Cannot help.\nMore detail.",
            "tool_calls":[{"id":"c", "function":{"name":"f", "arguments":"{}"}}]
        }, "finish_reason":"stop"}],
        "usage":{"prompt_tokens":3, "completion_tokens":4, "total_tokens":7}
    });
    let response = OpenAIChatCodec.decode_response(&raw).unwrap();
    assert_eq!(
        response.response_text().as_deref(),
        Some("Cannot help.\nMore detail.")
    );
    assert!(matches!(
        response.output().parts().next().unwrap().kind,
        ContentKind::Refusal(_)
    ));
    assert_eq!(response.id.as_deref(), Some("r"));
    assert_eq!(response.model.as_deref(), Some("m"));
    assert_eq!(response.finish_reason, Some(FinishReason::Complete));
    assert_eq!(response.usage.unwrap().total_tokens, Some(7));
    assert_eq!(response.tool_calls.unwrap()[0].name, "f");
}

#[test]
fn chat_mixed_text_and_refusal_preserve_empty_strings_and_part_kinds() {
    for text in ["prefix\nline", ""] {
        for refusal in ["no\nthanks", ""] {
            let response = OpenAIChatCodec
                .decode_response(&json!({"choices":[{"message":{
                    "content":text, "refusal":refusal
                }}]}))
                .unwrap();
            assert_eq!(
                response.output().text_parts().collect::<Vec<_>>(),
                [text, refusal]
            );
            let Some(MessageContent::Parts(parts)) = response.message else {
                panic!("typed parts expected")
            };
            assert!(matches!(parts[0], ContentPart::Text { .. }));
            assert!(matches!(parts[1], ContentPart::Refusal { .. }));
        }
    }
}

#[test]
fn chat_null_or_absent_refusals_keep_the_legacy_text_shape() {
    for message in [
        json!({"content":"text"}),
        json!({"content":"text", "refusal":null}),
    ] {
        let response = OpenAIChatCodec
            .decode_response(&json!({"choices":[{"message":message}]}))
            .unwrap();
        assert_eq!(response.message, Some(MessageContent::Text("text".into())));
    }
}

#[test]
fn responses_refusals_preserve_mixed_block_order_and_wire_items() {
    let raw = json!({"status":"completed", "output":[
        {"type":"reasoning", "summary":[{"type":"summary_text", "text":"excluded"}]},
        {"type":"message", "content":[
            {"type":"output_text", "text":"first\nline"},
            {"type":"refusal", "refusal":"no\nthanks", "provider_extra":true},
            {"type":"output_text", "text":"last"}
        ]},
        {"type":"function_call", "call_id":"c", "name":"f", "arguments":"{}"}
    ]});
    let response = OpenAIResponsesCodec.decode_response(&raw).unwrap();
    assert_eq!(
        response.output().text_parts().collect::<Vec<_>>(),
        ["first\nline", "no\nthanks", "last"]
    );
    let Some(MessageContent::Parts(parts)) = &response.message else {
        panic!("typed parts expected")
    };
    assert!(matches!(parts[1], ContentPart::Refusal { .. }));
    assert_eq!(response.tool_calls.as_ref().unwrap()[0].name, "f");
    let Some(ApiSpecificResponse::OpenAIResponses { output_items, .. }) = response.api_specific
    else {
        panic!("retained wire items expected")
    };
    assert_eq!(output_items.as_ref(), raw["output"].as_array());
}

#[test]
fn responses_refusal_only_including_empty_text_is_not_absent() {
    for refusal in ["no", ""] {
        let response = OpenAIResponsesCodec
            .decode_response(&json!({"output":[{
                "type":"message", "content":[{"type":"refusal", "refusal":refusal}]
            }]}))
            .unwrap();
        assert_eq!(response.response_text().as_deref(), Some(refusal));
        assert!(matches!(
            response.output().parts().next().unwrap().kind,
            ContentKind::Refusal(_)
        ));
    }
}

#[test]
fn chat_stream_accumulates_refusals_independently_per_choice() {
    let codec = OpenAIChatStreamingCodec::new();
    let mut collect = codec.borrowed_collector().unwrap();
    collect(&json!({"choices":[
        {"index":0, "delta":{"role":"assistant", "content":"prefix", "refusal":"Cannot "}},
        {"index":1, "delta":{"refusal":"other"}}
    ]}))
    .unwrap();
    collect(&json!({"choices":[{"index":0, "delta":{"refusal":"help"}}]})).unwrap();
    collect(
        &json!({"choices":[{"index":0, "delta":{"refusal":null}, "finish_reason":"stop"}],
        "usage":{"prompt_tokens":1, "completion_tokens":2}}),
    )
    .unwrap();
    let aggregate = codec.finalizer()();
    assert_eq!(aggregate["choices"][0]["message"]["refusal"], "Cannot help");
    assert_eq!(aggregate["choices"][1]["message"]["refusal"], "other");
    assert_eq!(aggregate["choices"][1]["message"]["content"], json!(null));
    let response = OpenAIChatCodec.decode_response(&aggregate).unwrap();
    assert_eq!(
        response.response_text().as_deref(),
        Some("prefix\nCannot help")
    );
    assert_eq!(response.finish_reason, Some(FinishReason::Complete));
}

#[test]
fn chat_stream_distinguishes_empty_refusal_from_missing_or_null() {
    for delta in [json!({}), json!({"refusal":null}), json!({"refusal":""})] {
        let codec = OpenAIChatStreamingCodec::new();
        codec.collector()(json!({"choices":[{"index":0,"delta":delta}]})).unwrap();
        let aggregate = codec.finalizer()();
        assert_eq!(
            aggregate["choices"][0]["message"].get("refusal"),
            delta.get("refusal").filter(|v| v.is_string())
        );
    }
}

#[test]
fn responses_streamed_refusal_snapshots_are_normalized_once() {
    for terminal_has_output in [false, true] {
        let codec = OpenAIResponsesStreamingCodec::new();
        let mut collect = codec.collector();
        let item = json!({"type":"message", "role":"assistant", "content":[{"type":"refusal","refusal":"no"}]});
        collect(json!({"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"no"})).unwrap();
        collect(json!({"type":"response.output_item.done","output_index":0,"item":item})).unwrap();
        let mut terminal = json!({"id":"r", "status":"completed"});
        if terminal_has_output {
            terminal["output"] = json!([item]);
        }
        collect(json!({"type":"response.completed","response":terminal})).unwrap();
        let response = OpenAIResponsesCodec
            .decode_response(&codec.finalizer()())
            .unwrap();
        assert_eq!(response.response_text().as_deref(), Some("no"));
        assert_eq!(response.output().parts().count(), 1);
    }
}
