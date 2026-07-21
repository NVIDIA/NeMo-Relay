// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Map, Value as Json};

use nemo_relay::codec::request::{ContentPart, MessageContent};
use nemo_relay::codec::resolve::ProviderSurface;
use nemo_relay::codec::response::{AnnotatedLlmResponse, FinishReason, ResponseToolCall};

#[derive(Clone, Copy)]
pub(crate) enum BuiltinCodecName {
    OpenAIChat,
    OpenAIResponses,
    AnthropicMessages,
}

impl BuiltinCodecName {
    pub(crate) fn from_provider_surface(surface: ProviderSurface) -> Self {
        match surface {
            ProviderSurface::OpenAIChat => Self::OpenAIChat,
            ProviderSurface::OpenAIResponses => Self::OpenAIResponses,
            ProviderSurface::AnthropicMessages => Self::AnthropicMessages,
        }
    }

    pub(crate) fn overlay_response_payload(
        self,
        payload: Json,
        annotated: &AnnotatedLlmResponse,
    ) -> Json {
        match self {
            Self::OpenAIChat => overlay_openai_chat_response(payload, annotated),
            Self::OpenAIResponses => overlay_openai_responses_response(payload, annotated),
            Self::AnthropicMessages => overlay_anthropic_response(payload, annotated),
        }
    }
}

fn overlay_openai_chat_response(mut payload: Json, annotated: &AnnotatedLlmResponse) -> Json {
    let Some(root) = payload.as_object_mut() else {
        return payload;
    };
    set_optional_string_field(root, "id", annotated.id.as_deref());
    set_optional_string_field(root, "model", annotated.model.as_deref());

    let Some(choice) = root
        .get_mut("choices")
        .and_then(Json::as_array_mut)
        .and_then(|choices| choices.first_mut())
        .and_then(Json::as_object_mut)
    else {
        return payload;
    };

    set_optional_string_field(
        choice,
        "finish_reason",
        annotated
            .finish_reason
            .as_ref()
            .map(openai_chat_finish_reason),
    );

    let Some(message) = choice.get_mut("message").and_then(Json::as_object_mut) else {
        return payload;
    };
    set_optional_string_field(
        message,
        "content",
        annotated_message_text(annotated.message.as_ref()).as_deref(),
    );
    overlay_openai_chat_tool_calls(message, annotated.tool_calls.as_deref());
    payload
}

fn overlay_openai_responses_response(mut payload: Json, annotated: &AnnotatedLlmResponse) -> Json {
    let Some(root) = payload.as_object_mut() else {
        return payload;
    };
    set_optional_string_field(root, "id", annotated.id.as_deref());
    set_optional_string_field(root, "model", annotated.model.as_deref());
    set_optional_string_field(
        root,
        "status",
        annotated
            .finish_reason
            .as_ref()
            .map(openai_responses_status),
    );

    if let Some(items) = root.get_mut("output").and_then(Json::as_array_mut) {
        overlay_output_text_blocks(items, annotated_message_text(annotated.message.as_ref()));
        overlay_openai_responses_tool_calls(items, annotated.tool_calls.as_deref());
    }
    payload
}

fn overlay_anthropic_response(mut payload: Json, annotated: &AnnotatedLlmResponse) -> Json {
    let Some(root) = payload.as_object_mut() else {
        return payload;
    };
    set_optional_string_field(root, "id", annotated.id.as_deref());
    set_optional_string_field(root, "model", annotated.model.as_deref());
    set_optional_string_field(
        root,
        "stop_reason",
        annotated.finish_reason.as_ref().map(anthropic_stop_reason),
    );

    if let Some(blocks) = root.get_mut("content").and_then(Json::as_array_mut) {
        overlay_anthropic_text_blocks(blocks, annotated_message_text(annotated.message.as_ref()));
        overlay_anthropic_tool_calls(blocks, annotated.tool_calls.as_deref());
    }
    payload
}

fn overlay_openai_chat_tool_calls(
    message: &mut Map<String, Json>,
    tool_calls: Option<&[ResponseToolCall]>,
) {
    let Some(raw_calls) = message.get_mut("tool_calls").and_then(Json::as_array_mut) else {
        return;
    };
    let Some(tool_calls) = tool_calls else {
        message.remove("tool_calls");
        return;
    };

    raw_calls.truncate(tool_calls.len());

    for (raw_call, sanitized_call) in raw_calls.iter_mut().zip(tool_calls.iter()) {
        let Some(raw_call) = raw_call.as_object_mut() else {
            message.remove("tool_calls");
            return;
        };
        set_optional_string_field(raw_call, "id", Some(sanitized_call.id.as_str()));
        let Some(function) = raw_call.get_mut("function").and_then(Json::as_object_mut) else {
            message.remove("tool_calls");
            return;
        };
        set_optional_string_field(function, "name", Some(sanitized_call.name.as_str()));
        set_optional_string_field(
            function,
            "arguments",
            Some(json_string(&sanitized_call.arguments).as_str()),
        );
    }
}

fn overlay_openai_responses_tool_calls(
    items: &mut Vec<Json>,
    tool_calls: Option<&[ResponseToolCall]>,
) {
    let Some(tool_calls) = tool_calls else {
        items.retain(|item| item.get("type").and_then(Json::as_str) != Some("function_call"));
        return;
    };

    let mut sanitized_calls = tool_calls.iter();
    items.retain_mut(|item| {
        let Some(item_type) = item.get("type").and_then(Json::as_str) else {
            return true;
        };
        if item_type != "function_call" {
            return true;
        }
        let Some(raw_call) = item.as_object_mut() else {
            return false;
        };
        let Some(sanitized_call) = sanitized_calls.next() else {
            return false;
        };
        set_optional_string_field(raw_call, "call_id", Some(sanitized_call.id.as_str()));
        set_optional_string_field(raw_call, "name", Some(sanitized_call.name.as_str()));
        set_optional_string_field(
            raw_call,
            "arguments",
            Some(json_string(&sanitized_call.arguments).as_str()),
        );
        true
    });
}

fn overlay_anthropic_tool_calls(blocks: &mut Vec<Json>, tool_calls: Option<&[ResponseToolCall]>) {
    let Some(tool_calls) = tool_calls else {
        blocks.retain(|block| block.get("type").and_then(Json::as_str) != Some("tool_use"));
        return;
    };

    let mut sanitized_calls = tool_calls.iter();
    blocks.retain_mut(|block| {
        let Some(block_type) = block.get("type").and_then(Json::as_str) else {
            return true;
        };
        if block_type != "tool_use" {
            return true;
        }
        let Some(raw_call) = block.as_object_mut() else {
            return false;
        };
        let Some(sanitized_call) = sanitized_calls.next() else {
            return false;
        };
        set_optional_string_field(raw_call, "id", Some(sanitized_call.id.as_str()));
        set_optional_string_field(raw_call, "name", Some(sanitized_call.name.as_str()));
        raw_call.insert("input".into(), sanitized_call.arguments.clone());
        true
    });
}

fn overlay_output_text_blocks(items: &mut [Json], message_text: Option<String>) {
    let text_items = items.iter_mut().filter_map(|item| {
        (item.get("type").and_then(Json::as_str) == Some("message"))
            .then_some(item.get_mut("content"))
            .flatten()
            .and_then(Json::as_array_mut)
    });
    let Some(text) = message_text else {
        for content in text_items {
            for block in content.iter_mut() {
                if block.get("type").and_then(Json::as_str) == Some("output_text")
                    && let Some(block) = block.as_object_mut()
                {
                    block.remove("text");
                }
            }
        }
        return;
    };

    let parts: Vec<&str> = text.split('\n').collect();
    for content in text_items {
        let output_text_count = content
            .iter()
            .filter(|block| block.get("type").and_then(Json::as_str) == Some("output_text"))
            .count();
        let mut text_blocks = content.iter_mut().filter_map(|block| {
            (block.get("type").and_then(Json::as_str) == Some("output_text"))
                .then_some(block.as_object_mut())
                .flatten()
        });

        if output_text_count <= 1 {
            if let Some(block) = text_blocks.next() {
                set_optional_string_field(block, "text", Some(text.as_str()));
            }
            continue;
        }

        for (index, block) in text_blocks.by_ref().enumerate() {
            let part = parts
                .get(index)
                .copied()
                .or_else(|| (index == 0).then_some(text.as_str()));
            set_optional_string_field(block, "text", part);
        }
    }
}

fn overlay_anthropic_text_blocks(blocks: &mut [Json], message_text: Option<String>) {
    let text_block_count = blocks
        .iter()
        .filter(|block| block.get("type").and_then(Json::as_str) == Some("text"))
        .count();
    let parts = message_text
        .as_deref()
        .map(|text| text.split('\n').collect::<Vec<_>>());
    let mut text_block_index = 0usize;

    for block in blocks {
        if block.get("type").and_then(Json::as_str) != Some("text") {
            continue;
        }
        let Some(block) = block.as_object_mut() else {
            continue;
        };
        if text_block_count <= 1 {
            set_optional_string_field(block, "text", message_text.as_deref());
            text_block_index += 1;
            continue;
        }
        let part = parts
            .as_ref()
            .and_then(|parts| parts.get(text_block_index).copied())
            .or_else(|| {
                (text_block_index == 0)
                    .then_some(message_text.as_deref())
                    .flatten()
            });
        set_optional_string_field(block, "text", part);
        text_block_index += 1;
    }
}

fn annotated_message_text(message: Option<&MessageContent>) -> Option<String> {
    match message? {
        MessageContent::Text(text) => Some(text.clone()),
        MessageContent::Parts(parts) => {
            let text_parts: Vec<&str> = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text, .. } => Some(text.as_str()),
                    ContentPart::Refusal { refusal, .. } => Some(refusal.as_str()),
                    ContentPart::ProviderNative { value, .. } => value
                        .get("text")
                        .and_then(Json::as_str)
                        .or_else(|| value.get("refusal").and_then(Json::as_str)),
                    ContentPart::ImageUrl { .. }
                    | ContentPart::Image { .. }
                    | ContentPart::Audio { .. }
                    | ContentPart::File { .. }
                    | ContentPart::ToolUse { .. }
                    | ContentPart::ToolResult { .. } => None,
                })
                .collect();
            (!text_parts.is_empty()).then(|| text_parts.join("\n"))
        }
    }
}

fn set_optional_string_field(object: &mut Map<String, Json>, key: &str, value: Option<&str>) {
    match value {
        Some(value) => {
            object.insert(key.to_string(), Json::String(value.to_string()));
        }
        None => {
            object.remove(key);
        }
    }
}

fn json_string(value: &Json) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn openai_chat_finish_reason(reason: &FinishReason) -> &str {
    match reason {
        FinishReason::Complete => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolUse => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Unknown(other) => other.as_str(),
    }
}

fn openai_responses_status(reason: &FinishReason) -> &str {
    match reason {
        FinishReason::Complete => "completed",
        FinishReason::Length | FinishReason::ContentFilter => "incomplete",
        FinishReason::ToolUse => "completed",
        FinishReason::Unknown(other) => other.as_str(),
    }
}

fn anthropic_stop_reason(reason: &FinishReason) -> &str {
    match reason {
        FinishReason::Complete => "end_turn",
        FinishReason::Length => "max_tokens",
        FinishReason::ToolUse => "tool_use",
        FinishReason::ContentFilter => "refusal",
        FinishReason::Unknown(other) => other.as_str(),
    }
}

#[cfg(test)]
#[path = "../tests/coverage/overlay_tests.rs"]
mod tests;
