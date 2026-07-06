// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::codec::anthropic::AnthropicMessagesCodec;
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::codec::openai_responses::OpenAIResponsesCodec;
use nemo_relay::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use nemo_relay::codec::response::{AnnotatedLlmResponse, FinishReason, ResponseToolCall, Usage};
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec, LlmResponseEncoder};
use nemo_relay::error::{FlowError, Result};
use serde_json::{Map, Value as Json, json};

use crate::component::WireProtocol;

pub(crate) fn decode_request(
    protocol: WireProtocol,
    request: &LlmRequest,
) -> Result<AnnotatedLlmRequest> {
    let normalized = normalize_multimodal_for_decode(protocol, request);
    let mut annotated = match protocol {
        WireProtocol::OpenaiChat => OpenAIChatCodec.decode(&normalized),
        WireProtocol::OpenaiResponses => OpenAIResponsesCodec.decode(&normalized),
        WireProtocol::AnthropicMessages => AnthropicMessagesCodec.decode(&normalized),
    }?;
    if protocol == WireProtocol::AnthropicMessages {
        annotated.stream = request.content.get("stream").and_then(Json::as_bool);
        annotated.extra.remove("stream");
    }
    Ok(annotated)
}

pub(crate) fn encode_request(
    protocol: WireProtocol,
    annotated: &AnnotatedLlmRequest,
    headers: Map<String, Json>,
) -> Result<LlmRequest> {
    ensure_portable_request(annotated)?;
    let template = LlmRequest {
        headers,
        content: json!({}),
    };
    let mut encoded = match protocol {
        WireProtocol::OpenaiChat => OpenAIChatCodec.encode(annotated, &template),
        WireProtocol::OpenaiResponses => OpenAIResponsesCodec.encode(annotated, &template),
        WireProtocol::AnthropicMessages => AnthropicMessagesCodec.encode(annotated, &template),
    }?;
    if protocol == WireProtocol::AnthropicMessages
        && let Some(stream) = annotated.stream
        && let Some(body) = encoded.content.as_object_mut()
    {
        body.insert("stream".into(), json!(stream));
    }
    encode_provider_multimodal(protocol, &mut encoded);
    Ok(encoded)
}

fn normalize_multimodal_for_decode(protocol: WireProtocol, request: &LlmRequest) -> LlmRequest {
    let mut normalized = request.clone();
    match protocol {
        WireProtocol::OpenaiChat => {}
        WireProtocol::OpenaiResponses => normalize_responses_input(&mut normalized.content),
        WireProtocol::AnthropicMessages => normalize_anthropic_messages(&mut normalized.content),
    }
    normalized
}

fn encode_provider_multimodal(protocol: WireProtocol, request: &mut LlmRequest) {
    match protocol {
        WireProtocol::OpenaiChat => {}
        WireProtocol::OpenaiResponses => encode_responses_input(&mut request.content),
        WireProtocol::AnthropicMessages => encode_anthropic_messages(&mut request.content),
    }
}

fn normalize_responses_input(body: &mut Json) {
    if let Some(tools) = body.get_mut("tools").and_then(Json::as_array_mut) {
        for tool in tools {
            if tool.get("function").is_none() && tool["type"] == "function" {
                let function = json!({
                    "name": tool["name"],
                    "description": tool.get("description").cloned(),
                    "parameters": tool.get("parameters").cloned(),
                });
                *tool = json!({"type": "function", "function": function});
            }
        }
    }
    if let Some(choice) = body.get_mut("tool_choice")
        && choice["type"] == "function"
        && choice.get("function").is_none()
    {
        *choice = json!({"type": "function", "function": {"name": choice["name"]}});
    }
    let Some(input) = body.get_mut("input").and_then(Json::as_array_mut) else {
        return;
    };
    let mut messages = Vec::new();
    for mut item in std::mem::take(input) {
        match item.get("type").and_then(Json::as_str) {
            Some("function_call") => messages.push(json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or(Json::Null),
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": item.get("arguments").cloned().unwrap_or_else(|| json!("{}")),
                    }
                }]
            })),
            Some("function_call_output") => messages.push(json!({
                "role": "tool",
                "tool_call_id": item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or(Json::Null),
                "content": item.get("output").cloned().unwrap_or(Json::Null),
            })),
            _ if item.get("role").is_some() => {
                normalize_content_parts(&mut item["content"], WireProtocol::OpenaiResponses);
                messages.push(item);
            }
            _ => messages.push(item),
        }
    }
    *input = messages;
}

fn normalize_anthropic_messages(body: &mut Json) {
    let Some(input) = body.get_mut("messages").and_then(Json::as_array_mut) else {
        return;
    };
    let mut messages = Vec::new();
    for item in std::mem::take(input) {
        let role = item.get("role").and_then(Json::as_str).unwrap_or("user");
        let Some(blocks) = item.get("content").and_then(Json::as_array) else {
            messages.push(item);
            continue;
        };
        let mut content = Vec::new();
        let mut tool_calls = Vec::new();
        let mut tool_results = Vec::new();
        for block in blocks {
            match block.get("type").and_then(Json::as_str) {
                Some("text") => content.push(json!({"type": "text", "text": block["text"]})),
                Some("image") if block["source"]["type"] == "url" => content.push(json!({
                    "type": "image_url", "image_url": {"url": block["source"]["url"]}
                })),
                Some("tool_use") => tool_calls.push(json!({
                    "id": block["id"], "type": "function",
                    "function": {"name": block["name"], "arguments": block["input"].to_string()}
                })),
                Some("tool_result") => tool_results.push(json!({
                    "role": "tool", "tool_call_id": block["tool_use_id"],
                    "content": anthropic_tool_result_text(block.get("content"))
                })),
                _ => content.push(block.clone()),
            }
        }
        if role == "assistant" {
            messages.push(json!({
                "role": "assistant",
                "content": (!content.is_empty()).then_some(Json::Array(content)),
                "tool_calls": (!tool_calls.is_empty()).then_some(Json::Array(tool_calls)),
            }));
        } else if !content.is_empty() {
            messages.push(json!({"role": role, "content": content}));
        }
        messages.extend(tool_results);
    }
    *input = messages;
}

fn anthropic_tool_result_text(content: Option<&Json>) -> String {
    match content {
        Some(Json::String(value)) => value.clone(),
        Some(Json::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Json::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        Some(value) => value.to_string(),
        None => String::new(),
    }
}

fn normalize_content_parts(content: &mut Json, protocol: WireProtocol) {
    let Some(parts) = content.as_array_mut() else {
        return;
    };
    for part in parts {
        match protocol {
            WireProtocol::OpenaiResponses if part["type"] == "input_text" => {
                part["type"] = json!("text");
            }
            WireProtocol::OpenaiResponses if part["type"] == "input_image" => {
                let url = part.get("image_url").and_then(|value| {
                    value.as_str().map(ToOwned::to_owned).or_else(|| {
                        value
                            .get("url")
                            .and_then(Json::as_str)
                            .map(ToOwned::to_owned)
                    })
                });
                if let Some(url) = url {
                    *part = json!({"type": "image_url", "image_url": {"url": url}});
                }
            }
            _ => {}
        }
    }
}

fn encode_responses_input(body: &mut Json) {
    if let Some(tools) = body.get_mut("tools").and_then(Json::as_array_mut) {
        for tool in tools {
            if let Some(function) = tool.get("function") {
                *tool = json!({
                    "type": "function",
                    "name": function["name"],
                    "description": function.get("description").cloned(),
                    "parameters": function.get("parameters").cloned(),
                });
            }
        }
    }
    if let Some(choice) = body.get_mut("tool_choice")
        && let Some(function) = choice.get("function")
    {
        *choice = json!({"type": "function", "name": function["name"]});
    }
    let Some(input) = body.get_mut("input").and_then(Json::as_array_mut) else {
        return;
    };
    let mut items = Vec::new();
    for mut message in std::mem::take(input) {
        match message.get("role").and_then(Json::as_str) {
            Some("assistant") => {
                if !message["content"].is_null() {
                    encode_content_parts(&mut message["content"], WireProtocol::OpenaiResponses);
                    let content = message["content"].clone();
                    items.push(json!({"type": "message", "role": "assistant", "content": content}));
                }
                for call in message["tool_calls"].as_array().into_iter().flatten() {
                    items.push(json!({
                        "type": "function_call", "call_id": call["id"], "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"]
                    }));
                }
            }
            Some("tool") => items.push(json!({
                "type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]
            })),
            _ => {
                encode_content_parts(&mut message["content"], WireProtocol::OpenaiResponses);
                items.push(message);
            }
        }
    }
    *input = items;
}

fn encode_anthropic_messages(body: &mut Json) {
    let Some(input) = body.get_mut("messages").and_then(Json::as_array_mut) else {
        return;
    };
    let mut messages = Vec::new();
    for mut message in std::mem::take(input) {
        let role = message
            .get("role")
            .and_then(Json::as_str)
            .unwrap_or("user")
            .to_string();
        if role == "tool" {
            messages.push(json!({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message["content"]
            }]}));
            continue;
        }
        encode_content_parts(&mut message["content"], WireProtocol::AnthropicMessages);
        let mut content = match message.get("content") {
            Some(Json::Array(parts)) => parts.clone(),
            Some(Json::String(text)) => vec![json!({"type": "text", "text": text})],
            _ => Vec::new(),
        };
        for call in message["tool_calls"].as_array().into_iter().flatten() {
            let arguments = call["function"]["arguments"]
                .as_str()
                .and_then(|value| serde_json::from_str::<Json>(value).ok())
                .unwrap_or_else(|| call["function"]["arguments"].clone());
            content.push(json!({
                "type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": arguments
            }));
        }
        messages.push(json!({"role": role, "content": content}));
    }
    *input = messages;
}

fn encode_content_parts(content: &mut Json, protocol: WireProtocol) {
    let Some(parts) = content.as_array_mut() else {
        return;
    };
    for part in parts {
        match protocol {
            WireProtocol::OpenaiResponses if part["type"] == "text" => {
                part["type"] = json!("input_text");
            }
            WireProtocol::OpenaiResponses if part["type"] == "image_url" => {
                let image_url = part["image_url"]["url"].clone();
                *part = json!({"type": "input_image", "image_url": image_url});
            }
            WireProtocol::AnthropicMessages if part["type"] == "image_url" => {
                let url = part["image_url"]["url"].clone();
                *part = json!({"type": "image", "source": {"type": "url", "url": url}});
            }
            _ => {}
        }
    }
}

fn ensure_portable_request(request: &AnnotatedLlmRequest) -> Result<()> {
    if request.reasoning.is_some() || request.include.is_some() || !request.extra.is_empty() {
        return Err(FlowError::InvalidArgument(
            "request uses provider-specific fields that cannot be translated safely".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_portable_request(
    protocol: WireProtocol,
    request: &LlmRequest,
) -> Result<()> {
    let annotated = decode_request(protocol, request)?;
    ensure_portable_request(&annotated)?;
    if contains_key_recursive(&request.content, "cache_control")
        || contains_key_recursive(&request.content, "audio")
        || contains_key_recursive(&request.content, "thinking")
        || contains_key_recursive(&request.content, "computer_use")
        || contains_key_recursive(&request.content, "server_tool_use")
    {
        return Err(FlowError::InvalidArgument(
            "request uses a provider-specific extension that requires same-protocol fail-open"
                .into(),
        ));
    }
    Ok(())
}

fn contains_key_recursive(value: &Json, key: &str) -> bool {
    match value {
        Json::Object(object) => {
            object.contains_key(key)
                || object
                    .values()
                    .any(|value| contains_key_recursive(value, key))
        }
        Json::Array(items) => items.iter().any(|value| contains_key_recursive(value, key)),
        _ => false,
    }
}

pub(crate) fn latest_user_prompt(annotated: &AnnotatedLlmRequest) -> Option<String> {
    annotated
        .messages
        .iter()
        .rev()
        .find_map(|message| match message {
            Message::User { content, .. } => content_text(content),
            _ => None,
        })
}

pub(crate) fn recent_message_window(
    annotated: &AnnotatedLlmRequest,
    count: usize,
) -> AnnotatedLlmRequest {
    let mut output = annotated.clone();
    let split = output.messages.len().saturating_sub(count);
    let mut messages = output.messages.split_off(split);
    if let Some(system) = annotated
        .messages
        .iter()
        .find(|message| matches!(message, Message::System { .. }))
        .cloned()
        && !matches!(messages.first(), Some(Message::System { .. }))
    {
        messages.insert(0, system);
    }
    output.messages = messages;
    output
}

fn content_text(content: &MessageContent) -> Option<String> {
    match content {
        MessageContent::Text(text) => Some(text.clone()),
        MessageContent::Parts(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| match part {
                    nemo_relay::codec::request::ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
    }
}

pub(crate) fn translate_response(
    source: WireProtocol,
    target: WireProtocol,
    response: &Json,
) -> Result<Json> {
    if source == target {
        return Ok(response.clone());
    }
    ensure_portable_response(source, response)?;
    let annotated = decode_response(source, response)?;
    ProviderResponseEncoder(target).encode_response(&annotated)
}

struct ProviderResponseEncoder(WireProtocol);

impl LlmResponseEncoder for ProviderResponseEncoder {
    fn encode_response(&self, response: &AnnotatedLlmResponse) -> Result<Json> {
        encode_response(self.0, response)
    }
}

fn ensure_portable_response(protocol: WireProtocol, response: &Json) -> Result<()> {
    let unsupported =
        match protocol {
            WireProtocol::OpenaiChat => {
                response["choices"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .any(|choice| {
                        choice["message"].get("audio").is_some()
                            || choice["message"].get("reasoning_content").is_some()
                    })
            }
            WireProtocol::OpenaiResponses => response["output"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|item| {
                    matches!(
                        item.get("type").and_then(Json::as_str),
                        Some(
                            "reasoning"
                                | "computer_call"
                                | "computer_call_output"
                                | "web_search_call"
                        )
                    )
                }),
            WireProtocol::AnthropicMessages => response["content"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|block| {
                    !matches!(
                        block.get("type").and_then(Json::as_str),
                        Some("text" | "tool_use")
                    )
                }),
        };
    if unsupported {
        return Err(FlowError::InvalidArgument(
            "response uses provider-specific fields that cannot be translated safely".into(),
        ));
    }
    Ok(())
}

pub(crate) fn decode_response(
    protocol: WireProtocol,
    response: &Json,
) -> Result<AnnotatedLlmResponse> {
    match protocol {
        WireProtocol::OpenaiChat => OpenAIChatCodec.decode_response(response),
        WireProtocol::OpenaiResponses => OpenAIResponsesCodec.decode_response(response),
        WireProtocol::AnthropicMessages => AnthropicMessagesCodec.decode_response(response),
    }
}

pub(crate) fn encode_response(
    protocol: WireProtocol,
    response: &AnnotatedLlmResponse,
) -> Result<Json> {
    match protocol {
        WireProtocol::OpenaiChat => Ok(encode_openai_chat_response(response)),
        WireProtocol::OpenaiResponses => Ok(encode_openai_responses_response(response)),
        WireProtocol::AnthropicMessages => Ok(encode_anthropic_response(response)),
    }
}

fn encode_openai_chat_response(response: &AnnotatedLlmResponse) -> Json {
    let mut message = Map::from_iter([("role".into(), json!("assistant"))]);
    message.insert("content".into(), content_json(response.message.as_ref()));
    if let Some(tool_calls) = &response.tool_calls {
        message.insert(
            "tool_calls".into(),
            Json::Array(tool_calls.iter().map(openai_tool_call).collect()),
        );
    }
    json!({
        "id": response.id.clone().unwrap_or_else(|| "chatcmpl-relay".into()),
        "object": "chat.completion",
        "model": response.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": openai_finish(response.finish_reason.as_ref()),
        }],
        "usage": openai_usage(response.usage.as_ref()),
    })
}

fn encode_anthropic_response(response: &AnnotatedLlmResponse) -> Json {
    let mut content = Vec::new();
    if let Some(message) = &response.message {
        content.push(json!({"type": "text", "text": content_text(message).unwrap_or_default()}));
    }
    if let Some(tool_calls) = &response.tool_calls {
        content.extend(tool_calls.iter().map(|call| {
            json!({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
        }));
    }
    json!({
        "id": response.id.clone().unwrap_or_else(|| "msg_relay".into()),
        "type": "message",
        "role": "assistant",
        "model": response.model,
        "content": content,
        "stop_reason": anthropic_finish(response.finish_reason.as_ref()),
        "stop_sequence": null,
        "usage": anthropic_usage(response.usage.as_ref()),
    })
}

fn encode_openai_responses_response(response: &AnnotatedLlmResponse) -> Json {
    let mut output = Vec::new();
    if let Some(message) = &response.message {
        output.push(json!({
            "id": "msg_relay",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": content_text(message).unwrap_or_default(), "annotations": []}],
        }));
    }
    if let Some(tool_calls) = &response.tool_calls {
        output.extend(tool_calls.iter().map(|call| {
            json!({
                "id": call.id,
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments.to_string(),
                "status": "completed",
            })
        }));
    }
    json!({
        "id": response.id.clone().unwrap_or_else(|| "resp_relay".into()),
        "object": "response",
        "status": responses_status(response.finish_reason.as_ref()),
        "model": response.model,
        "output": output,
        "usage": responses_usage(response.usage.as_ref()),
    })
}

fn content_json(content: Option<&MessageContent>) -> Json {
    match content {
        Some(MessageContent::Text(text)) => json!(text),
        Some(MessageContent::Parts(parts)) => serde_json::to_value(parts).unwrap_or(Json::Null),
        None => Json::Null,
    }
}

fn openai_tool_call(call: &ResponseToolCall) -> Json {
    json!({
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments.to_string()},
    })
}

fn openai_finish(reason: Option<&FinishReason>) -> Json {
    json!(match reason {
        Some(FinishReason::Complete) => "stop",
        Some(FinishReason::Length) => "length",
        Some(FinishReason::ToolUse) => "tool_calls",
        Some(FinishReason::ContentFilter) => "content_filter",
        Some(FinishReason::Unknown(value)) => value.as_str(),
        None => "stop",
    })
}

fn anthropic_finish(reason: Option<&FinishReason>) -> Json {
    json!(match reason {
        Some(FinishReason::Length) => "max_tokens",
        Some(FinishReason::ToolUse) => "tool_use",
        Some(FinishReason::Unknown(value)) => value.as_str(),
        _ => "end_turn",
    })
}

fn responses_status(reason: Option<&FinishReason>) -> &'static str {
    match reason {
        Some(FinishReason::Length | FinishReason::ContentFilter) => "incomplete",
        _ => "completed",
    }
}

fn openai_usage(usage: Option<&Usage>) -> Json {
    match usage {
        Some(usage) => json!({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }),
        None => Json::Null,
    }
}

fn anthropic_usage(usage: Option<&Usage>) -> Json {
    match usage {
        Some(usage) => json!({
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cache_read_input_tokens": usage.cache_read_tokens,
            "cache_creation_input_tokens": usage.cache_write_tokens,
        }),
        None => json!({"input_tokens": 0, "output_tokens": 0}),
    }
}

fn responses_usage(usage: Option<&Usage>) -> Json {
    match usage {
        Some(usage) => json!({
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }),
        None => Json::Null,
    }
}

#[cfg(test)]
pub(crate) fn response_to_stream_chunks(protocol: WireProtocol, response: &Json) -> Vec<Json> {
    match protocol {
        WireProtocol::OpenaiChat => openai_chat_stream_chunks(response),
        WireProtocol::AnthropicMessages => anthropic_stream_chunks(response),
        WireProtocol::OpenaiResponses => responses_stream_chunks(response),
    }
}

#[cfg(test)]
fn openai_chat_stream_chunks(response: &Json) -> Vec<Json> {
    let choice = &response["choices"][0];
    vec![json!({
        "id": response["id"], "object": "chat.completion.chunk", "model": response["model"],
        "choices": [{"index": 0, "delta": choice["message"], "finish_reason": choice["finish_reason"]}],
        "usage": response["usage"],
    })]
}

#[cfg(test)]
fn anthropic_stream_chunks(response: &Json) -> Vec<Json> {
    let mut chunks = vec![json!({
        "type": "message_start",
        "message": {"id": response["id"], "type": "message", "role": "assistant", "model": response["model"], "content": [], "stop_reason": null, "usage": response["usage"]}
    })];
    for (index, block) in response["content"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        chunks.push(json!({"type": "content_block_start", "index": index, "content_block": block}));
        if block["type"] == "text" {
            chunks.push(json!({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block["text"]}}));
        }
        chunks.push(json!({"type": "content_block_stop", "index": index}));
    }
    chunks.push(json!({"type": "message_delta", "delta": {"stop_reason": response["stop_reason"]}, "usage": response["usage"]}));
    chunks.push(json!({"type": "message_stop"}));
    chunks
}

#[cfg(test)]
fn responses_stream_chunks(response: &Json) -> Vec<Json> {
    let mut chunks = vec![json!({"type": "response.created", "response": response})];
    for (index, item) in response["output"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        chunks.push(
            json!({"type": "response.output_item.added", "output_index": index, "item": item}),
        );
        if item["type"] == "message"
            && let Some(text) = item["content"][0]["text"].as_str()
        {
            chunks.push(json!({"type": "response.output_text.delta", "output_index": index, "content_index": 0, "delta": text}));
            chunks.push(json!({"type": "response.output_text.done", "output_index": index, "content_index": 0, "text": text}));
        }
        chunks.push(
            json!({"type": "response.output_item.done", "output_index": index, "item": item}),
        );
    }
    chunks.push(json!({"type": "response.completed", "response": response}));
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use nemo_relay::codec::anthropic::AnthropicMessagesStreamingCodec;
    use nemo_relay::codec::openai_chat::OpenAIChatStreamingCodec;
    use nemo_relay::codec::openai_responses::OpenAIResponsesStreamingCodec;
    use nemo_relay::codec::request::{ContentPart, Message};
    use nemo_relay::codec::streaming::StreamingCodec;

    fn request(protocol: WireProtocol) -> LlmRequest {
        let content = match protocol {
            WireProtocol::OpenaiChat => json!({
                "model": "source-model",
                "messages": [
                    {"role": "system", "content": "Be concise"},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "https://example.test/cat.png"}}
                    ]},
                    {"role": "assistant", "content": null, "tool_calls": [{
                        "id": "call-1", "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"key\":\"cat\"}"}
                    }]},
                    {"role": "tool", "tool_call_id": "call-1", "content": "found"}
                ],
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
                "tool_choice": "auto",
                "temperature": 0.2,
                "max_tokens": 64,
                "stream": true
            }),
            WireProtocol::OpenaiResponses => json!({
                "model": "source-model",
                "instructions": "Be concise",
                "input": [
                    {"type": "message", "role": "user", "content": [
                        {"type": "input_text", "text": "Describe this"},
                        {"type": "input_image", "image_url": "https://example.test/cat.png"}
                    ]},
                    {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{\"key\":\"cat\"}"},
                    {"type": "function_call_output", "call_id": "call-1", "output": "found"}
                ],
                "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
                "tool_choice": "auto",
                "temperature": 0.2,
                "max_output_tokens": 64,
                "stream": true
            }),
            WireProtocol::AnthropicMessages => json!({
                "model": "source-model",
                "system": "Be concise",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image", "source": {"type": "url", "url": "https://example.test/cat.png"}}
                    ]},
                    {"role": "assistant", "content": [{
                        "type": "tool_use", "id": "call-1", "name": "lookup", "input": {"key": "cat"}
                    }]},
                    {"role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": "call-1", "content": "found"
                    }]}
                ],
                "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "auto"},
                "temperature": 0.2,
                "max_tokens": 64,
                "stream": true
            }),
        };
        LlmRequest {
            headers: Map::new(),
            content,
        }
    }

    fn response(protocol: WireProtocol) -> Json {
        match protocol {
            WireProtocol::OpenaiChat => json!({
                "id": "chat-1", "object": "chat.completion", "model": "served-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "done", "tool_calls": [{
                    "id": "call-2", "type": "function", "function": {"name": "finish", "arguments": "{\"ok\":true}"}
                }]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
            }),
            WireProtocol::OpenaiResponses => json!({
                "id": "resp-1", "object": "response", "status": "completed", "model": "served-model",
                "output": [
                    {"id": "msg-1", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "done", "annotations": []}]},
                    {"id": "fc-1", "type": "function_call", "call_id": "call-2", "name": "finish", "arguments": "{\"ok\":true}", "status": "completed"}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
            }),
            WireProtocol::AnthropicMessages => json!({
                "id": "msg-1", "type": "message", "role": "assistant", "model": "served-model",
                "content": [
                    {"type": "text", "text": "done"},
                    {"type": "tool_use", "id": "call-2", "name": "finish", "input": {"ok": true}}
                ],
                "stop_reason": "tool_use", "stop_sequence": null,
                "usage": {"input_tokens": 10, "output_tokens": 4}
            }),
        }
    }

    fn protocols() -> [WireProtocol; 3] {
        [
            WireProtocol::OpenaiChat,
            WireProtocol::OpenaiResponses,
            WireProtocol::AnthropicMessages,
        ]
    }

    fn has_image(messages: &[Message]) -> bool {
        messages.iter().any(|message| {
            let content = match message {
                Message::System { content, .. }
                | Message::User { content, .. }
                | Message::Tool { content, .. } => Some(content),
                Message::Assistant { content, .. } => content.as_ref(),
            };
            matches!(content, Some(MessageContent::Parts(parts)) if parts.iter().any(|part| matches!(part, ContentPart::ImageUrl { .. })))
        })
    }

    fn tool_message_count(messages: &[Message]) -> usize {
        messages
            .iter()
            .filter(|message| {
                matches!(message, Message::Tool { .. })
                    || matches!(message, Message::Assistant { tool_calls: Some(calls), .. } if !calls.is_empty())
            })
            .count()
    }

    #[test]
    fn request_translation_matrix_preserves_normalized_core() {
        for source in protocols() {
            let source_request = request(source);
            let source_annotated = decode_request(source, &source_request).unwrap();
            assert!(has_image(&source_annotated.messages), "source={source:?}");
            assert_eq!(
                tool_message_count(&source_annotated.messages),
                2,
                "source={source:?}"
            );
            for target in protocols() {
                let encoded = encode_request(target, &source_annotated, Map::new()).unwrap();
                let roundtrip = decode_request(target, &encoded).unwrap();
                assert!(
                    has_image(&roundtrip.messages),
                    "source={source:?} target={target:?}"
                );
                assert_eq!(
                    tool_message_count(&roundtrip.messages),
                    2,
                    "source={source:?} target={target:?} body={}",
                    encoded.content
                );
                assert_eq!(roundtrip.params, source_annotated.params);
                assert_eq!(roundtrip.stream, Some(true));
            }
        }
    }

    #[test]
    fn buffered_response_translation_matrix_preserves_core_output() {
        for source in protocols() {
            for target in protocols() {
                let translated = translate_response(source, target, &response(source)).unwrap();
                let annotated = decode_response(target, &translated).unwrap();
                assert_eq!(
                    content_text(annotated.message.as_ref().unwrap()).as_deref(),
                    Some("done")
                );
                assert_eq!(annotated.tool_calls.as_ref().map(Vec::len), Some(1));
                assert_eq!(
                    annotated
                        .usage
                        .as_ref()
                        .and_then(|usage| usage.total_tokens),
                    Some(14)
                );
            }
        }
    }

    #[test]
    fn streaming_translation_matrix_emits_collectable_terminal_sequences() {
        for source in protocols() {
            for target in protocols() {
                let translated = translate_response(source, target, &response(source)).unwrap();
                let codec: Box<dyn StreamingCodec> = match target {
                    WireProtocol::OpenaiChat => Box::new(OpenAIChatStreamingCodec::new()),
                    WireProtocol::OpenaiResponses => Box::new(OpenAIResponsesStreamingCodec::new()),
                    WireProtocol::AnthropicMessages => {
                        Box::new(AnthropicMessagesStreamingCodec::new())
                    }
                };
                let mut collector = codec.collector();
                let finalizer = codec.finalizer();
                for chunk in response_to_stream_chunks(target, &translated) {
                    collector(chunk).unwrap();
                }
                let aggregate = finalizer();
                let annotated = decode_response(target, &aggregate).unwrap();
                assert!(
                    annotated.message.is_some(),
                    "source={source:?} target={target:?}"
                );
            }
        }
    }

    #[test]
    fn cross_protocol_translation_rejects_reasoning_extensions() {
        let response = json!({
            "id": "resp-1", "object": "response", "status": "completed", "model": "m",
            "output": [{"id": "r-1", "type": "reasoning", "summary": []}]
        });
        assert!(
            translate_response(
                WireProtocol::OpenaiResponses,
                WireProtocol::OpenaiChat,
                &response
            )
            .is_err()
        );
        assert_eq!(
            translate_response(
                WireProtocol::OpenaiResponses,
                WireProtocol::OpenaiResponses,
                &response
            )
            .unwrap(),
            response
        );
    }
}
