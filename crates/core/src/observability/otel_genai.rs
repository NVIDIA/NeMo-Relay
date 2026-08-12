// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry GenAI semantic-convention projection.

#![allow(deprecated)] // Generated GenAI constants are retained for the pinned v1.42-era schema.

use crate::api::event::{Event, EventNormalizationExt};
use crate::api::scope::ScopeType;
use crate::codec::request::{ApiSpecificRequest, ContentPart, Message, MessageContent};
use crate::codec::response::{AnnotatedLlmResponse, FinishReason};
use crate::json::Json;
use opentelemetry::KeyValue;
use opentelemetry::trace::SpanKind;
use opentelemetry_semantic_conventions::attribute as semconv;
use serde_json::{Map, Value};

const OPERATION_CHAT: &str = "chat";
const OPERATION_EMBEDDINGS: &str = "embeddings";
const OPERATION_EXECUTE_TOOL: &str = "execute_tool";
const OPERATION_GENERATE_CONTENT: &str = "generate_content";
const OPERATION_INVOKE_AGENT: &str = "invoke_agent";
const OPERATION_RETRIEVAL: &str = "retrieval";
const OPERATION_TEXT_COMPLETION: &str = "text_completion";

// OpenTelemetry Rust 0.32 still predates generated constants for these
// development attributes. Keep the missing keys in one projection-local block
// until the generated crate exposes them.
const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";
const GEN_AI_INPUT_MESSAGES: &str = "gen_ai.input.messages";
const GEN_AI_OUTPUT_MESSAGES: &str = "gen_ai.output.messages";
const GEN_AI_SYSTEM_INSTRUCTIONS: &str = "gen_ai.system_instructions";
const GEN_AI_RETRIEVAL_TOP_K: &str = "gen_ai.retrieval.top_k";
const GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS: &str = "gen_ai.usage.cache_creation.input_tokens";
const GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";

fn has_gen_ai_semantics(event: &Event) -> bool {
    matches!(
        event.scope_type(),
        Some(
            ScopeType::Agent
                | ScopeType::Llm
                | ScopeType::Tool
                | ScopeType::Embedder
                | ScopeType::Retriever
        )
    )
}

pub(super) fn span_name(event: &Event) -> String {
    if !has_gen_ai_semantics(event) {
        return event.name().to_string();
    }
    let operation = operation_name(event);
    let qualifier = match event.scope_type() {
        Some(ScopeType::Agent) => Some(agent_name(event)),
        Some(ScopeType::Tool) => Some(tool_name(event)),
        Some(ScopeType::Retriever) => data_source_id(event),
        Some(ScopeType::Llm | ScopeType::Embedder) => request_model(event),
        _ => None,
    };
    qualifier.filter(|value| !value.is_empty()).map_or_else(
        || operation.to_string(),
        |value| format!("{operation} {value}"),
    )
}

pub(super) fn span_kind(event: &Event) -> SpanKind {
    match event.scope_type() {
        Some(ScopeType::Agent | ScopeType::Tool) => SpanKind::Internal,
        Some(ScopeType::Llm | ScopeType::Embedder | ScopeType::Retriever) => SpanKind::Client,
        _ => SpanKind::Internal,
    }
}

pub(super) fn start_attributes(event: &Event) -> Vec<KeyValue> {
    if !has_gen_ai_semantics(event) {
        return Vec::new();
    }
    let mut attributes = Vec::new();
    attributes.push(KeyValue::new(
        semconv::GEN_AI_OPERATION_NAME,
        operation_name(event),
    ));

    match event.scope_type() {
        Some(ScopeType::Agent) => {
            push_conversation_attribute(&mut attributes, event);
            push_agent_attributes(&mut attributes, event);
        }
        Some(ScopeType::Llm) => {
            push_provider_and_server_attributes(&mut attributes, event);
            push_conversation_attribute(&mut attributes, event);
            push_llm_request_attributes(&mut attributes, event);
        }
        Some(ScopeType::Tool) => push_tool_attributes(&mut attributes, event),
        Some(ScopeType::Retriever) => {
            push_provider_and_server_attributes(&mut attributes, event);
            push_retrieval_attributes(&mut attributes, event);
        }
        Some(ScopeType::Embedder) => {
            push_provider_and_server_attributes(&mut attributes, event);
            push_model_attribute(&mut attributes, event);
        }
        _ => {}
    }
    attributes
}

pub(super) fn end_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = Vec::new();
    push_error_attributes(&mut attributes, event);
    match event.scope_type() {
        Some(ScopeType::Llm) => push_llm_response_attributes(&mut attributes, event),
        Some(ScopeType::Embedder) => push_embedding_response_attributes(&mut attributes, event),
        _ => {}
    }
    attributes
}

fn push_embedding_response_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(value) = scalar_string(event, &["gen_ai.response.model", "response_model", "model"])
    {
        attributes.push(KeyValue::new("gen_ai.response.model", value));
    }
    if let Some(value) = scalar_i64(
        event,
        &[
            semconv::GEN_AI_USAGE_INPUT_TOKENS,
            "input_tokens",
            "prompt_tokens",
        ],
    ) {
        attributes.push(KeyValue::new(semconv::GEN_AI_USAGE_INPUT_TOKENS, value));
    }
}

fn operation_name(event: &Event) -> &'static str {
    match event.scope_type() {
        Some(ScopeType::Agent) => OPERATION_INVOKE_AGENT,
        Some(ScopeType::Tool) => OPERATION_EXECUTE_TOOL,
        Some(ScopeType::Embedder) => OPERATION_EMBEDDINGS,
        Some(ScopeType::Retriever) => OPERATION_RETRIEVAL,
        Some(ScopeType::Llm) => llm_operation_name(event),
        _ => OPERATION_CHAT,
    }
}

fn llm_operation_name(event: &Event) -> &'static str {
    let name = event.name().to_ascii_lowercase();
    if name.contains("generate_content") || name.contains("generatecontent") {
        OPERATION_GENERATE_CONTENT
    } else if name.contains("completion") && !name.contains("chat") {
        OPERATION_TEXT_COMPLETION
    } else {
        OPERATION_CHAT
    }
}

fn push_provider_and_server_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(provider) = provider_name(event) {
        attributes.push(KeyValue::new(GEN_AI_PROVIDER_NAME, provider));
    }
    if let Some(address) = scalar_string(event, &[semconv::SERVER_ADDRESS, "server_address"]) {
        attributes.push(KeyValue::new(semconv::SERVER_ADDRESS, address));
    }
    if let Some(port) = scalar_i64(event, &[semconv::SERVER_PORT, "server_port"]) {
        attributes.push(KeyValue::new(semconv::SERVER_PORT, port));
    }
}

fn push_conversation_attribute(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(conversation_id) = scalar_string(
        event,
        &[
            "gen_ai.conversation.id",
            "conversation_id",
            "session_id",
            "thread_id",
        ],
    ) {
        attributes.push(KeyValue::new("gen_ai.conversation.id", conversation_id));
    }
}

fn push_agent_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    attributes.push(KeyValue::new("gen_ai.agent.name", agent_name(event)));
    if let Some(value) = scalar_string(event, &["gen_ai.agent.description", "agent_description"]) {
        attributes.push(KeyValue::new("gen_ai.agent.description", value));
    }
    push_model_attribute(attributes, event);
}

fn push_model_attribute(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(model) = request_model(event) {
        attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_MODEL, model));
    }
}

fn push_llm_request_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    let Some(request) = event.normalized_llm_request() else {
        push_model_attribute(attributes, event);
        return;
    };
    let request = request.as_ref();
    if let Some(model) = request
        .model
        .clone()
        .or_else(|| event.model_name().map(ToOwned::to_owned))
    {
        attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_MODEL, model));
    }
    if let Some(params) = request.params.as_ref() {
        if let Some(value) = params.temperature {
            attributes.push(KeyValue::new("gen_ai.request.temperature", value));
        }
        if request.max_output_tokens.is_none()
            && let Some(value) = params.max_tokens.and_then(to_i64)
        {
            attributes.push(KeyValue::new("gen_ai.request.max_tokens", value));
        }
        if let Some(value) = params.top_p {
            attributes.push(KeyValue::new("gen_ai.request.top_p", value));
        }
        if let Some(value) = params.stop.as_ref() {
            attributes.push(KeyValue::new(
                "gen_ai.request.stop_sequences",
                string_array(value.iter().cloned()),
            ));
        }
    }
    if let Some(value) = request.max_output_tokens.and_then(to_i64) {
        attributes.push(KeyValue::new("gen_ai.request.max_tokens", value));
    }
    if request.stream == Some(true) {
        attributes.push(KeyValue::new("gen_ai.request.stream", true));
    }
    if let Some(instructions) = request
        .instructions
        .as_ref()
        .and_then(system_instructions_json)
    {
        attributes.push(KeyValue::new(GEN_AI_SYSTEM_INSTRUCTIONS, instructions));
    }
    if let Some(messages) = input_messages_json(&request.messages) {
        attributes.push(KeyValue::new(GEN_AI_INPUT_MESSAGES, messages));
    }
    push_api_specific_request_attributes(attributes, request.api_specific.as_ref());
}

fn push_api_specific_request_attributes(
    attributes: &mut Vec<KeyValue>,
    api_specific: Option<&ApiSpecificRequest>,
) {
    match api_specific {
        Some(ApiSpecificRequest::AnthropicMessages { top_k, .. }) => {
            if let Some(value) = top_k.and_then(to_i64) {
                attributes.push(KeyValue::new("gen_ai.request.top_k", value));
            }
        }
        Some(ApiSpecificRequest::OpenAIChat {
            frequency_penalty,
            n,
            presence_penalty,
            seed,
            ..
        }) => {
            if let Some(value) = frequency_penalty {
                attributes.push(KeyValue::new("gen_ai.request.frequency_penalty", *value));
            }
            if let Some(value) = n.filter(|value| *value != 1).and_then(to_i64) {
                attributes.push(KeyValue::new("gen_ai.request.choice.count", value));
            }
            if let Some(value) = presence_penalty {
                attributes.push(KeyValue::new("gen_ai.request.presence_penalty", *value));
            }
            if let Some(value) = seed {
                attributes.push(KeyValue::new("gen_ai.request.seed", *value));
            }
        }
        _ => {}
    }
}

fn push_llm_response_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    let Some(response) = event.normalized_llm_response() else {
        return;
    };
    let response = response.as_ref();
    if let Some(value) = response.id.as_ref() {
        attributes.push(KeyValue::new("gen_ai.response.id", value.clone()));
    }
    if let Some(value) = response.model.as_ref() {
        attributes.push(KeyValue::new("gen_ai.response.model", value.clone()));
    }
    if let Some(value) = response.finish_reason.as_ref() {
        attributes.push(KeyValue::new(
            "gen_ai.response.finish_reasons",
            string_array([finish_reason(value).to_string()]),
        ));
    }
    if let Some(messages) = output_messages_json(response) {
        attributes.push(KeyValue::new(GEN_AI_OUTPUT_MESSAGES, messages));
    }
    if let Some(usage) = response.usage.as_ref() {
        // Anthropic reports uncached, cache-read, and cache-creation input
        // tokens separately. Other providers such as OpenAI include cache
        // reads in their prompt count, so only combine known Anthropic usage.
        if let Some(input_tokens) = gen_ai_input_tokens(event, response) {
            attributes.push(KeyValue::new(
                semconv::GEN_AI_USAGE_INPUT_TOKENS,
                input_tokens,
            ));
        }
        if let Some(value) = usage.completion_tokens.and_then(to_i64) {
            attributes.push(KeyValue::new("gen_ai.usage.output_tokens", value));
        }
        if let Some(value) = usage.cache_read_tokens.and_then(to_i64) {
            attributes.push(KeyValue::new(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, value));
        }
        if let Some(value) = usage.cache_write_tokens.and_then(to_i64) {
            attributes.push(KeyValue::new(
                GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                value,
            ));
        }
    }
}

fn gen_ai_input_tokens(event: &Event, response: &AnnotatedLlmResponse) -> Option<i64> {
    let usage = response.usage.as_ref()?;
    let provider = provider_name(event);
    super::input_tokens_including_cache(provider.as_deref(), Some(response), usage).and_then(to_i64)
}

fn input_messages_json(messages: &[Message]) -> Option<String> {
    if messages.is_empty() {
        return None;
    }
    let messages = messages.iter().map(input_message).collect::<Vec<_>>();
    serde_json::to_string(&messages).ok()
}

fn system_instructions_json(instructions: &MessageContent) -> Option<String> {
    let parts = content_parts(instructions);
    if parts.is_empty() {
        return None;
    }
    serde_json::to_string(&parts).ok()
}

fn input_message(message: &Message) -> Json {
    let (role, name, mut parts) = match message {
        Message::System { content, name } => ("system", name.as_ref(), content_parts(content)),
        Message::Developer { content, name } => {
            ("developer", name.as_ref(), content_parts(content))
        }
        Message::User { content, name } => (
            if is_tool_result_message(content) {
                "tool"
            } else {
                "user"
            },
            name.as_ref(),
            content_parts(content),
        ),
        Message::Assistant {
            content,
            tool_calls,
            name,
        } => {
            let mut parts = content.as_ref().map_or_else(Vec::new, content_parts);
            if let Some(tool_calls) = tool_calls {
                parts.extend(tool_calls.iter().map(|call| {
                    let arguments = serde_json::from_str(&call.function.arguments)
                        .unwrap_or_else(|_| Json::String(call.function.arguments.clone()));
                    serde_json::json!({
                        "type": "tool_call",
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": arguments,
                    })
                }));
            }
            ("assistant", name.as_ref(), parts)
        }
        Message::Tool {
            content,
            tool_call_id,
        } => (
            "tool",
            None,
            vec![serde_json::json!({
                "type": "tool_call_response",
                "id": tool_call_id,
                "response": message_content_value(content),
            })],
        ),
        Message::Function { content, name } => (
            "tool",
            Some(name),
            vec![serde_json::json!({
                "type": "tool_call_response",
                "response": content,
            })],
        ),
        Message::ToolCallItem {
            call_id,
            name,
            arguments,
            ..
        } => (
            "assistant",
            None,
            vec![serde_json::json!({
                "type": "tool_call",
                "id": call_id,
                "name": name,
                "arguments": arguments,
            })],
        ),
        Message::ToolResultItem {
            call_id, output, ..
        } => (
            "tool",
            None,
            vec![serde_json::json!({
                "type": "tool_call_response",
                "id": call_id,
                "response": output,
            })],
        ),
        Message::ProviderNative { kind, value, .. } => (
            value
                .get("role")
                .and_then(Json::as_str)
                .unwrap_or("provider_native"),
            None,
            vec![generic_part(kind, value)],
        ),
    };
    let mut object = Map::from_iter([
        ("role".to_string(), Json::String(role.to_string())),
        ("parts".to_string(), Json::Array(std::mem::take(&mut parts))),
    ]);
    if let Some(name) = name {
        object.insert("name".to_string(), Json::String(name.clone()));
    }
    Json::Object(object)
}

fn is_tool_result_message(content: &MessageContent) -> bool {
    matches!(
        content,
        MessageContent::Parts(parts)
            if !parts.is_empty()
                && parts
                    .iter()
                    .all(|part| matches!(part, ContentPart::ToolResult { .. }))
    )
}

fn output_messages_json(response: &AnnotatedLlmResponse) -> Option<String> {
    let mut parts = response
        .message
        .as_ref()
        .map_or_else(Vec::new, content_parts);
    if let Some(tool_calls) = response.tool_calls.as_ref() {
        parts.extend(tool_calls.iter().map(|call| {
            serde_json::json!({
                "type": "tool_call",
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            })
        }));
    }
    if parts.is_empty() {
        return None;
    }
    let object = Map::from_iter([
        ("role".to_string(), Json::String("assistant".to_string())),
        ("parts".to_string(), Json::Array(parts)),
        (
            "finish_reason".to_string(),
            Json::String(
                response
                    .finish_reason
                    .as_ref()
                    .map_or("unknown", finish_reason)
                    .to_string(),
            ),
        ),
    ]);
    serde_json::to_string(&[Json::Object(object)]).ok()
}

fn content_parts(content: &MessageContent) -> Vec<Json> {
    match content {
        MessageContent::Text(content) => vec![text_part(content)],
        MessageContent::Parts(parts) => parts.iter().map(content_part).collect(),
    }
}

fn content_part(part: &ContentPart) -> Json {
    match part {
        ContentPart::Text { text, .. } => text_part(text),
        ContentPart::Refusal { refusal, .. } => text_part(refusal),
        ContentPart::ToolUse {
            id, name, input, ..
        } => serde_json::json!({
            "type": "tool_call",
            "id": id,
            "name": name,
            "arguments": input,
        }),
        ContentPart::ToolResult {
            tool_use_id,
            content,
            ..
        } => serde_json::json!({
            "type": "tool_call_response",
            "id": tool_use_id,
            "response": content,
        }),
        ContentPart::ProviderNative { kind, value, .. } => generic_part(kind, value),
        ContentPart::ImageUrl { .. } => serialized_part("image_url", part),
        ContentPart::Image { .. } => serialized_part("image", part),
        ContentPart::Audio { .. } => serialized_part("audio", part),
        ContentPart::File { .. } => serialized_part("file", part),
    }
}

fn serialized_part(kind: &str, part: &ContentPart) -> Json {
    generic_part(kind, &serde_json::to_value(part).unwrap_or(Json::Null))
}

fn text_part(content: &str) -> Json {
    serde_json::json!({"type": "text", "content": content})
}

fn generic_part(kind: &str, value: &Json) -> Json {
    let mut object = value
        .as_object()
        .cloned()
        .unwrap_or_else(|| Map::from_iter([("content".to_string(), value.clone())]));
    object.insert("type".to_string(), Json::String(kind.to_string()));
    Json::Object(object)
}

fn message_content_value(content: &MessageContent) -> Json {
    match content {
        MessageContent::Text(text) => Json::String(text.clone()),
        MessageContent::Parts(parts) => serde_json::to_value(parts).unwrap_or(Json::Null),
    }
}

fn push_tool_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    attributes.push(KeyValue::new(semconv::GEN_AI_TOOL_NAME, tool_name(event)));
    if let Some(value) = scalar_string(event, &["gen_ai.tool.type", "tool_type"]) {
        attributes.push(KeyValue::new("gen_ai.tool.type", value));
    }
    if let Some(value) = event
        .tool_call_id()
        .map(ToOwned::to_owned)
        .or_else(|| scalar_string(event, &["gen_ai.tool.call.id", "tool_call_id"]))
    {
        attributes.push(KeyValue::new("gen_ai.tool.call.id", value));
    }
    if let Some(value) = scalar_string(
        event,
        &["gen_ai.tool.description", "tool_description", "description"],
    ) {
        attributes.push(KeyValue::new("gen_ai.tool.description", value));
    }
}

fn push_retrieval_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(value) = data_source_id(event) {
        attributes.push(KeyValue::new("gen_ai.data_source.id", value));
    }
    push_model_attribute(attributes, event);
    if let Some(value) = scalar_i64(event, &[GEN_AI_RETRIEVAL_TOP_K, "top_k"]) {
        attributes.push(KeyValue::new(GEN_AI_RETRIEVAL_TOP_K, value));
    }
}

fn finish_reason(reason: &FinishReason) -> &str {
    match reason {
        FinishReason::Complete => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolUse => "tool_call",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Unknown(value) => value,
    }
}

fn request_model(event: &Event) -> Option<String> {
    event
        .normalized_llm_request()
        .and_then(|request| request.as_ref().model.clone())
        .or_else(|| event.model_name().map(ToOwned::to_owned))
        .or_else(|| {
            scalar_string(
                event,
                &[semconv::GEN_AI_REQUEST_MODEL, "model", "model_name"],
            )
        })
}

fn provider_name(event: &Event) -> Option<String> {
    scalar_string(event, &[GEN_AI_PROVIDER_NAME, "provider_name", "provider"])
        .or_else(|| provider_from_event_name(event))
        .or_else(|| provider_from_normalized_request(event).map(str::to_string))
}

fn provider_from_event_name(event: &Event) -> Option<String> {
    let name = event.name().to_ascii_lowercase();
    [
        ("azure_ai_inference", "azure.ai.inference"),
        ("azure ai inference", "azure.ai.inference"),
        ("azure_openai", "azure.ai.openai"),
        ("azure openai", "azure.ai.openai"),
        ("anthropic", "anthropic"),
        ("claude", "anthropic"),
        ("bedrock", "aws.bedrock"),
        ("cohere", "cohere"),
        ("deepseek", "deepseek"),
        ("gemini", "gcp.gemini"),
        ("vertex", "gcp.vertex_ai"),
        ("groq", "groq"),
        ("mistral", "mistral_ai"),
        ("openai", "openai"),
        ("gpt", "openai"),
        ("perplexity", "perplexity"),
    ]
    .into_iter()
    .find_map(|(needle, provider)| name.contains(needle).then(|| provider.to_string()))
}

fn provider_from_normalized_request(event: &Event) -> Option<&'static str> {
    let request = event.normalized_llm_request()?;
    match request.as_ref().api_specific.as_ref()? {
        ApiSpecificRequest::AnthropicMessages { .. } => Some("anthropic"),
        ApiSpecificRequest::OpenAIChat { .. } | ApiSpecificRequest::OpenAIResponses { .. } => {
            Some("openai")
        }
        // Not an OTel well-known value yet; follows the dotted cloud-provider
        // convention (`aws.bedrock`, `gcp.gemini`).
        ApiSpecificRequest::OCIGenAI { .. } => Some("oci.genai"),
        ApiSpecificRequest::Custom { .. } => None,
    }
}

fn agent_name(event: &Event) -> String {
    scalar_string(event, &["gen_ai.agent.name"]).unwrap_or_else(|| event.name().to_string())
}

fn tool_name(event: &Event) -> String {
    scalar_string(event, &[semconv::GEN_AI_TOOL_NAME]).unwrap_or_else(|| event.name().to_string())
}

fn data_source_id(event: &Event) -> Option<String> {
    scalar_string(
        event,
        &[
            "gen_ai.data_source.id",
            "data_source_id",
            "index_name",
            "collection_name",
        ],
    )
}

fn push_error_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    let is_error = event
        .metadata()
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("otel.status_code"))
        .and_then(Value::as_str)
        .is_some_and(|status| status.eq_ignore_ascii_case("ERROR"));
    if !is_error {
        return;
    }
    let error_type = scalar_string(event, &[semconv::ERROR_TYPE, "error_type"])
        .unwrap_or_else(|| "_OTHER".to_string());
    attributes.push(KeyValue::new(semconv::ERROR_TYPE, error_type));
}

fn scalar_string(event: &Event, keys: &[&str]) -> Option<String> {
    find_scalar(event, keys, |value| {
        value
            .as_str()
            .map(str::to_string)
            .or_else(|| (value.is_number() || value.is_boolean()).then(|| value.to_string()))
    })
}

fn scalar_i64(event: &Event, keys: &[&str]) -> Option<i64> {
    find_scalar(event, keys, |value| {
        value.as_i64().or_else(|| value.as_u64().and_then(to_i64))
    })
}

fn find_scalar<T>(event: &Event, keys: &[&str], convert: impl Fn(&Json) -> Option<T>) -> Option<T> {
    let profile_value = event.category_profile().and_then(|profile| {
        keys.iter()
            .find_map(|key| profile.extra.get(*key).and_then(&convert))
    });
    profile_value.or_else(|| {
        event_objects(event).into_iter().find_map(|object| {
            keys.iter()
                .find_map(|key| object_value(object, key).and_then(&convert))
        })
    })
}

fn object_value<'a>(object: &'a Map<String, Json>, key: &str) -> Option<&'a Json> {
    object.get(key).or_else(|| {
        ["usage", "request", "response"]
            .into_iter()
            .filter_map(|container| object.get(container).and_then(Value::as_object))
            .find_map(|nested| nested.get(key))
    })
}

fn event_objects(event: &Event) -> Vec<&Map<String, Json>> {
    let mut objects = Vec::new();
    if let Some(value) = event.metadata().and_then(Value::as_object) {
        objects.push(value);
    }
    if let Some(value) = event.data().and_then(Value::as_object) {
        objects.push(value);
    }
    objects
}

fn string_array(values: impl IntoIterator<Item = String>) -> opentelemetry::Value {
    opentelemetry::Value::Array(opentelemetry::Array::String(
        values.into_iter().map(Into::into).collect(),
    ))
}

fn to_i64(value: u64) -> Option<i64> {
    i64::try_from(value).ok()
}
