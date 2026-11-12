// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral coverage of borrowed normalized content views.

use std::borrow::Cow;

use nemo_relay_types::codec::content::{ContentField, ContentKind, ContentRoot, PayloadRef};
use nemo_relay_types::codec::request::{
    AnnotatedLlmRequest, ContentPart, FunctionCall, Message, MessageContent, ToolCall,
};
use nemo_relay_types::codec::response::{AnnotatedLlmResponse, ResponseToolCall};
use serde_json::{Value, json};

fn content(value: Value) -> MessageContent {
    serde_json::from_value(value).unwrap()
}

fn message(value: Value) -> Message {
    serde_json::from_value(value).unwrap()
}

#[test]
fn joins_only_text_and_refusals_preserving_empty_parts_and_whitespace() {
    let content = content(json!([
        {"type": "text", "text": " first "},
        {"type": "refusal", "refusal": ""},
        {"type": "image", "image": {"text": "hidden"}},
        {"type": "text", "text": "last\n"}
    ]));
    let before = serde_json::to_value(&content).unwrap();
    let view = content.content_view();
    assert_eq!(
        view.text_parts().collect::<Vec<_>>(),
        [" first ", "", "last\n"]
    );
    assert!(matches!(view.text(), Some(Cow::Owned(ref s)) if s == " first \n\nlast\n"));
    assert_eq!(serde_json::to_value(&content).unwrap(), before);

    for text in ["", "one"] {
        let content = MessageContent::Text(text.into());
        let Some(Cow::Borrowed(borrowed)) = content.content_view().text() else {
            panic!("one part must borrow");
        };
        let MessageContent::Text(original) = &content else {
            unreachable!()
        };
        assert_eq!(borrowed.as_ptr(), original.as_ptr());
    }
    assert!(
        MessageContent::Parts(vec![])
            .content_view()
            .text()
            .is_none()
    );
}

#[test]
fn selects_exact_final_item_and_reports_request_indices() {
    let mut request = AnnotatedLlmRequest::default();
    assert!(request.last_message().is_none());
    request
        .messages
        .push(message(json!({"role":"user", "content":"old"})));
    request
        .messages
        .push(message(json!({"role":"assistant", "content":[
            {"type":"text", "text":"answer"}, {"type":"refusal", "refusal":"refused"}
        ]})));
    let entries = request.last_message().unwrap().parts().collect::<Vec<_>>();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].location.root, ContentRoot::RequestMessage(1));
    assert_eq!(entries[1].location.field, ContentField::Content);
    assert_eq!(entries[1].location.index, Some(1));
    assert_eq!(
        request.last_message().unwrap().text().as_deref(),
        Some("answer\nrefused")
    );
    request.messages.push(message(json!({"role":"assistant"})));
    assert!(request.last_message().unwrap().text().is_none());
    assert_eq!(request.last_message().unwrap().parts().count(), 0);
}

#[test]
fn tool_call_representations_are_borrowed_and_follow_content() {
    let assistant = Message::Assistant {
        content: Some(MessageContent::Text("calling".into())),
        name: None,
        tool_calls: Some(vec![ToolCall {
            id: "call-1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "lookup".into(),
                arguments: "{not parsed}".into(),
            },
        }]),
    };
    let entries = assistant.content_view().parts().collect::<Vec<_>>();
    assert!(matches!(entries[0].kind, ContentKind::Text("calling")));
    assert_eq!(entries[1].location.field, ContentField::ToolCalls);
    assert_eq!(entries[1].location.index, Some(0));
    let ContentKind::ToolCall(call) = entries[1].kind else {
        panic!("tool call expected")
    };
    assert_eq!(call.id, Some("call-1"));
    assert_eq!(call.name, "lookup");
    assert!(matches!(call.arguments, PayloadRef::String("{not parsed}")));
    assert_eq!(assistant.content_view().text().as_deref(), Some("calling"));

    let response = AnnotatedLlmResponse {
        message: Some(content(json!([{"type":"text", "text":"done"}]))),
        tool_calls: Some(vec![ResponseToolCall {
            id: "call-2".into(),
            name: "lookup".into(),
            arguments: json!({"q":"value"}),
        }]),
        ..Default::default()
    };
    let before = serde_json::to_value(&response).unwrap();
    let entries = response.output().parts().collect::<Vec<_>>();
    assert_eq!(entries[0].location.field, ContentField::Message);
    assert_eq!(entries[1].location.root, ContentRoot::Response);
    assert_eq!(entries[1].location.field, ContentField::ToolCalls);
    let ContentKind::ToolCall(call) = entries[1].kind else {
        panic!("tool call expected")
    };
    let PayloadRef::Json(args) = call.arguments else {
        panic!("JSON arguments expected")
    };
    assert!(std::ptr::eq(
        args,
        &response.tool_calls.as_ref().unwrap()[0].arguments
    ));
    assert_eq!(serde_json::to_value(&response).unwrap(), before);

    let tool_only = AnnotatedLlmResponse {
        message: None,
        ..response
    };
    assert_eq!(tool_only.output().parts().count(), 1);
    assert!(tool_only.response_text().is_none());
    assert_eq!(AnnotatedLlmResponse::default().output().parts().count(), 0);
}

#[test]
fn top_level_tools_and_legacy_results_stay_typed() {
    let cases = [
        json!({"role":"tool", "tool_call_id":"c", "content":"result"}),
        json!({"role":"tool", "tool_call_id":"c", "content":[{"type":"text","text":"result"}]}),
        json!({"role":"function", "name":"f", "content":"result"}),
        json!({"role":"function", "name":"f", "content":null}),
        json!({"role":"tool_result", "call_id":"c", "output":"result"}),
        json!({"role":"tool_result", "call_id":"c", "output":{"text":"result"}}),
    ];
    for value in cases {
        let item = message(value);
        let before = serde_json::to_value(&item).unwrap();
        let entries = item.content_view().parts().collect::<Vec<_>>();
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0].kind, ContentKind::ToolResult(_)));
        assert!(item.content_view().text().is_none());
        assert_eq!(serde_json::to_value(&item).unwrap(), before);
    }
    let item = message(
        json!({"role":"tool_call", "id":"item", "call_id":"c", "name":"f", "arguments":{"x":1}}),
    );
    let entry = item.content_view().parts().next().unwrap();
    assert_eq!(entry.location.field, ContentField::ToolCallItem);
    let ContentKind::ToolCall(call) = entry.kind else {
        panic!("tool call expected")
    };
    assert_eq!(call.id, Some("item"));
    assert_eq!(call.call_id, Some("c"));
    assert!(matches!(call.arguments, PayloadRef::Json(_)));
    assert!(item.content_view().text().is_none());
}

#[test]
fn mixed_parts_preserve_types_locations_and_multiple_tool_results() {
    let value = content(json!([
        {"type":"text", "text":"hello"},
        {"type":"tool_use", "id":"c", "name":"lookup", "input":{"text":"arg"}},
        {"type":"tool_result", "tool_use_id":"c", "content":"string result", "is_error":true},
        {"type":"tool_result", "tool_use_id":"d", "content":{"text":"JSON result"}},
        {"type":"image_url", "image_url":{"url":"https://example.com/image"}},
        {"type":"image", "image":{}},
        {"type":"audio", "audio":{}},
        {"type":"file", "file":{}},
        {"type":"provider_native", "provider":"custom", "kind":"unknown", "value":{"nested":{"text":"hidden"}}}
    ]));
    let entries = value.content_view().parts().collect::<Vec<_>>();
    assert_eq!(entries.len(), 9);
    for (index, entry) in entries.iter().enumerate() {
        assert_eq!(entry.location.root, ContentRoot::Content);
        assert_eq!(entry.location.index, Some(index));
    }
    assert!(matches!(entries[1].kind, ContentKind::ToolCall(_)));
    assert!(
        matches!(entries[2].kind, ContentKind::ToolResult(result) if result.is_error == Some(true) && result.call_id == Some("c"))
    );
    assert!(matches!(entries[3].kind, ContentKind::ToolResult(_)));
    assert!(matches!(
        entries[8].kind,
        ContentKind::ProviderNative { .. }
    ));
    assert_eq!(value.content_view().text().as_deref(), Some("hello"));
}

#[test]
fn helper_selection_does_not_skip_textless_items() {
    let mut request = AnnotatedLlmRequest {
        instructions: Some(MessageContent::Parts(vec![])),
        messages: vec![
            message(json!({"role":"system", "content":"fallback"})),
            message(json!({"role":"user", "content":"old"})),
            message(json!({"role":"user", "content":[]})),
        ],
        ..Default::default()
    };
    assert!(request.system_prompt().is_none());
    assert!(request.last_user_message().is_none());
    request.instructions = None;
    request.messages[0] = message(json!({"role":"system", "content":[]}));
    request
        .messages
        .push(message(json!({"role":"developer", "content":"later"})));
    assert!(request.system_prompt().is_none());
    request.messages[2] = message(json!({"role":"user", "content":[
        {"type":"text", "text":"a"}, {"type":"refusal", "refusal":"b"}
    ]}));
    assert_eq!(request.last_user_message().as_deref(), Some("a\nb"));
    request.instructions = Some(content(json!([
        {"type":"text", "text":"a"}, {"type":"text", "text":"b"}
    ])));
    assert_eq!(request.system_prompt().as_deref(), Some("a\nb"));
}

#[test]
fn native_fallback_is_shallow_and_retains_native_payloads() {
    let item = message(
        json!({"role":"provider_native", "provider":"openai_responses", "kind":"message",
        "value":{"role":"user", "content":[
            {"type":"input_text", "text":"a"}, {"type":"refusal", "refusal":"b"},
            {"type":"thinking", "text":"hidden"}, {"type":"unknown", "nested":{"text":"hidden"}}
        ]}}),
    );
    assert_eq!(item.content_view().text().as_deref(), Some("a\nb"));
    let entries = item.content_view().parts().collect::<Vec<_>>();
    assert_eq!(entries.len(), 4);
    assert_eq!(entries[1].location.field, ContentField::NativeContent);
    assert_eq!(entries[1].location.index, Some(1));
    assert!(matches!(
        entries[1].kind,
        ContentKind::ProviderNative {
            kind: "refusal",
            ..
        }
    ));
    let mut request = AnnotatedLlmRequest {
        messages: vec![item],
        ..Default::default()
    };
    assert_eq!(request.last_user_message().as_deref(), Some("a\nb"));
    request.messages.push(message(
        json!({"role":"provider_native", "provider":"openai_responses", "kind":"message",
        "value":{"role":"user", "content":[]}}),
    ));
    assert!(request.last_user_message().is_none());

    for (kind, value, expected) in [
        ("unknown", json!({"text":"direct"}), Some("direct")),
        ("refusal", json!({"refusal":"no"}), Some("no")),
        ("unknown", json!({"nested":{"text":"hidden"}}), None),
        ("thinking", json!({"text":"hidden"}), None),
        ("image", json!({"text":"hidden"}), None),
        ("unknown", json!({"text":"hidden", "thought":true}), None),
        ("tool_result", json!({"text":"hidden"}), None),
    ] {
        let content = MessageContent::Parts(vec![ContentPart::ProviderNative {
            provider: "custom".into(),
            kind: kind.into(),
            value,
        }]);
        let response = AnnotatedLlmResponse {
            message: Some(content),
            ..Default::default()
        };
        assert_eq!(response.response_text().as_deref(), expected);
    }
}

#[test]
fn media_parts_remain_distinct_and_are_not_text() {
    let value = content(json!([
        {"type":"image_url", "image_url":{"url":"https://example.com/image"}},
        {"type":"image", "image":{}},
        {"type":"audio", "audio":{}},
        {"type":"file", "file":{}}
    ]));
    let entries = value.content_view().parts().collect::<Vec<_>>();
    assert!(matches!(entries[0].kind, ContentKind::ImageUrl(_)));
    assert!(matches!(entries[1].kind, ContentKind::Image(_)));
    assert!(matches!(entries[2].kind, ContentKind::Audio(_)));
    assert!(matches!(entries[3].kind, ContentKind::File(_)));
    assert!(value.content_view().text().is_none());
}
