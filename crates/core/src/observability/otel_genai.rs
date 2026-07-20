// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry GenAI semantic-convention projection.

use crate::api::event::{Event, EventNormalizationExt};
use crate::api::scope::ScopeType;
use crate::codec::request::{ContentPart, Message, MessageContent};
use crate::codec::response::{AnnotatedLlmResponse, FinishReason};
use crate::json::Json;
use opentelemetry::KeyValue;
use opentelemetry::trace::SpanKind;
use opentelemetry_semantic_conventions::attribute as semconv;
use serde_json::{Map, Value, json};

const SEMANTIC_CONVENTION_VERSION: &str = "1.37+";
const OPERATION_CHAT: &str = "chat";
const OPERATION_EMBEDDINGS: &str = "embeddings";
const OPERATION_EXECUTE_TOOL: &str = "execute_tool";
const OPERATION_GENERATE_CONTENT: &str = "generate_content";
const OPERATION_INVOKE_AGENT: &str = "invoke_agent";
const OPERATION_RERANK: &str = "rerank";
const OPERATION_RETRIEVAL: &str = "retrieval";
const OPERATION_TEXT_COMPLETION: &str = "text_completion";

// OpenTelemetry Rust 0.31 matches Relay's SDK version but predates generated
// constants for these 1.37+ development attributes. Keep the missing keys in
// one projection-local block until the generated crate exposes them without a
// major SDK upgrade.
const GEN_AI_AGENT_VERSION: &str = "gen_ai.agent.version";
const GEN_AI_INPUT_MESSAGES: &str = "gen_ai.input.messages";
const GEN_AI_OUTPUT_MESSAGES: &str = "gen_ai.output.messages";
const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";
const GEN_AI_RETRIEVAL_DOCUMENTS: &str = "gen_ai.retrieval.documents";
const GEN_AI_RETRIEVAL_QUERY_TEXT: &str = "gen_ai.retrieval.query.text";
const GEN_AI_SYSTEM_INSTRUCTIONS: &str = "gen_ai.system_instructions";
const GEN_AI_TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";
const GEN_AI_TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";
const GEN_AI_TOOL_DEFINITIONS: &str = "gen_ai.tool.definitions";
const GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS: &str = "gen_ai.usage.cache_creation.input_tokens";
const GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";

pub(super) fn supports(event: &Event) -> bool {
    matches!(
        event.scope_type(),
        Some(
            ScopeType::Agent
                | ScopeType::Llm
                | ScopeType::Tool
                | ScopeType::Embedder
                | ScopeType::Retriever
                | ScopeType::Reranker
        )
    )
}

pub(super) fn span_name(event: &Event) -> String {
    let operation = operation_name(event);
    let qualifier = match event.scope_type() {
        Some(ScopeType::Agent | ScopeType::Tool) => Some(event.name().to_string()),
        Some(ScopeType::Retriever) => data_source_id(event),
        Some(ScopeType::Llm | ScopeType::Embedder | ScopeType::Reranker) => request_model(event),
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
        _ => SpanKind::Client,
    }
}

pub(super) fn start_attributes(event: &Event, capture_content: bool) -> Vec<KeyValue> {
    let mut attributes = super::otel::common_attributes(event);
    super::push_serialized_top_level_attributes(
        &mut attributes,
        "nemo_relay.handle_attributes",
        event.attributes(),
    );
    attributes.push(KeyValue::new(
        "nemo_relay.otel.semantic_convention",
        SEMANTIC_CONVENTION_VERSION,
    ));
    attributes.push(KeyValue::new(
        semconv::GEN_AI_OPERATION_NAME,
        operation_name(event),
    ));

    push_common_attributes(&mut attributes, event);
    match event.scope_type() {
        Some(ScopeType::Agent) => push_agent_attributes(&mut attributes, event, capture_content),
        Some(ScopeType::Llm) => {
            push_llm_request_attributes(&mut attributes, event, capture_content)
        }
        Some(ScopeType::Tool) => push_tool_attributes(&mut attributes, event, capture_content),
        Some(ScopeType::Retriever | ScopeType::Reranker) => {
            push_retrieval_attributes(&mut attributes, event, capture_content)
        }
        Some(ScopeType::Embedder) => push_model_attribute(&mut attributes, event),
        _ => {}
    }
    attributes
}

pub(super) fn end_attributes(event: &Event, capture_content: bool) -> Vec<KeyValue> {
    let mut attributes = Vec::new();
    push_error_attributes(&mut attributes, event);
    match event.scope_type() {
        Some(ScopeType::Llm) => {
            push_llm_response_attributes(&mut attributes, event, capture_content)
        }
        Some(ScopeType::Tool) if capture_content => {
            if let Some(result) = wrapped_content_or_value(event.output(), &["result", "output"]) {
                attributes.push(KeyValue::new(GEN_AI_TOOL_CALL_RESULT, json_string(result)));
            }
        }
        Some(ScopeType::Embedder | ScopeType::Reranker) => {
            push_non_llm_response_attributes(&mut attributes, event);
            if capture_content && event.scope_type() == Some(ScopeType::Reranker) {
                push_retrieval_content(&mut attributes, event);
            }
        }
        Some(ScopeType::Retriever) if capture_content => {
            push_retrieval_content(&mut attributes, event);
        }
        _ => {}
    }
    attributes
}

fn push_non_llm_response_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(value) = scalar_string(
        event,
        &[semconv::GEN_AI_RESPONSE_MODEL, "response_model", "model"],
    ) {
        attributes.push(KeyValue::new(semconv::GEN_AI_RESPONSE_MODEL, value));
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
    if let Some(value) = scalar_i64(
        event,
        &[
            semconv::GEN_AI_USAGE_OUTPUT_TOKENS,
            "output_tokens",
            "completion_tokens",
        ],
    ) {
        attributes.push(KeyValue::new(semconv::GEN_AI_USAGE_OUTPUT_TOKENS, value));
    }
}

fn operation_name(event: &Event) -> &'static str {
    match event.scope_type() {
        Some(ScopeType::Agent) => OPERATION_INVOKE_AGENT,
        Some(ScopeType::Tool) => OPERATION_EXECUTE_TOOL,
        Some(ScopeType::Embedder) => OPERATION_EMBEDDINGS,
        Some(ScopeType::Retriever) => OPERATION_RETRIEVAL,
        Some(ScopeType::Reranker) => OPERATION_RERANK,
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

fn push_common_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(provider) = provider_name(event) {
        attributes.push(KeyValue::new(GEN_AI_PROVIDER_NAME, provider));
    }
    if let Some(conversation_id) = scalar_string(
        event,
        &[
            semconv::GEN_AI_CONVERSATION_ID,
            "conversation_id",
            "session_id",
            "thread_id",
        ],
    ) {
        attributes.push(KeyValue::new(
            semconv::GEN_AI_CONVERSATION_ID,
            conversation_id,
        ));
    }
    if let Some(address) = scalar_string(event, &[semconv::SERVER_ADDRESS, "server_address"]) {
        attributes.push(KeyValue::new(semconv::SERVER_ADDRESS, address));
    }
    if let Some(port) = scalar_i64(event, &[semconv::SERVER_PORT, "server_port"]) {
        attributes.push(KeyValue::new(semconv::SERVER_PORT, port));
    }
}

fn push_agent_attributes(attributes: &mut Vec<KeyValue>, event: &Event, capture_content: bool) {
    attributes.push(KeyValue::new(
        semconv::GEN_AI_AGENT_NAME,
        event.name().to_string(),
    ));
    for (key, candidates) in [
        (
            semconv::GEN_AI_AGENT_ID,
            &["gen_ai.agent.id", "agent_id"][..],
        ),
        (
            GEN_AI_AGENT_VERSION,
            &["gen_ai.agent.version", "agent_version"][..],
        ),
    ] {
        if let Some(value) = scalar_string(event, candidates) {
            attributes.push(KeyValue::new(key, value));
        }
    }
    if capture_content
        && let Some(value) =
            scalar_string(event, &["gen_ai.agent.description", "agent_description"])
    {
        attributes.push(KeyValue::new(semconv::GEN_AI_AGENT_DESCRIPTION, value));
    }
    push_model_attribute(attributes, event);
    if capture_content {
        push_tool_definitions(attributes, event);
    }
}

fn push_model_attribute(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(model) = request_model(event) {
        attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_MODEL, model));
    }
}

fn push_llm_request_attributes(
    attributes: &mut Vec<KeyValue>,
    event: &Event,
    capture_content: bool,
) {
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
            attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_TEMPERATURE, value));
        }
        if request.max_output_tokens.is_none()
            && let Some(value) = params.max_tokens.and_then(to_i64)
        {
            attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_MAX_TOKENS, value));
        }
        if let Some(value) = params.top_p {
            attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_TOP_P, value));
        }
        if let Some(value) = params.stop.as_ref() {
            attributes.push(KeyValue::new(
                semconv::GEN_AI_REQUEST_STOP_SEQUENCES,
                string_array(value.iter().cloned()),
            ));
        }
    }
    if let Some(value) = request.max_output_tokens.and_then(to_i64) {
        attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_MAX_TOKENS, value));
    }
    if !capture_content {
        return;
    }
    let (instructions, messages) = request_messages(&request.messages);
    if !instructions.is_empty() {
        attributes.push(KeyValue::new(
            GEN_AI_SYSTEM_INSTRUCTIONS,
            json_string(&Value::Array(instructions)),
        ));
    }
    if !messages.is_empty() {
        attributes.push(KeyValue::new(
            GEN_AI_INPUT_MESSAGES,
            json_string(&Value::Array(messages)),
        ));
    }
    push_tool_definitions(attributes, event);
}

fn push_llm_response_attributes(
    attributes: &mut Vec<KeyValue>,
    event: &Event,
    capture_content: bool,
) {
    let Some(response) = event.normalized_llm_response() else {
        return;
    };
    let response = response.as_ref();
    if let Some(value) = response.id.as_ref() {
        attributes.push(KeyValue::new(semconv::GEN_AI_RESPONSE_ID, value.clone()));
    }
    if let Some(value) = response.model.as_ref() {
        attributes.push(KeyValue::new(semconv::GEN_AI_RESPONSE_MODEL, value.clone()));
    }
    if let Some(value) = response.finish_reason.as_ref() {
        attributes.push(KeyValue::new(
            semconv::GEN_AI_RESPONSE_FINISH_REASONS,
            string_array([finish_reason(value).to_string()]),
        ));
    }
    if let Some(usage) = response.usage.as_ref() {
        if let Some(value) = usage.prompt_tokens.and_then(to_i64) {
            attributes.push(KeyValue::new(semconv::GEN_AI_USAGE_INPUT_TOKENS, value));
        }
        if let Some(value) = usage.completion_tokens.and_then(to_i64) {
            attributes.push(KeyValue::new(semconv::GEN_AI_USAGE_OUTPUT_TOKENS, value));
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
    if capture_content && let Some(message) = output_message(response) {
        attributes.push(KeyValue::new(
            GEN_AI_OUTPUT_MESSAGES,
            json_string(&json!([message])),
        ));
    }
}

fn push_tool_attributes(attributes: &mut Vec<KeyValue>, event: &Event, capture_content: bool) {
    attributes.push(KeyValue::new(
        semconv::GEN_AI_TOOL_NAME,
        event.name().to_string(),
    ));
    if let Some(value) = scalar_string(event, &[semconv::GEN_AI_TOOL_TYPE, "tool_type"]) {
        attributes.push(KeyValue::new(semconv::GEN_AI_TOOL_TYPE, value));
    }
    if let Some(value) = event
        .tool_call_id()
        .map(ToOwned::to_owned)
        .or_else(|| scalar_string(event, &[semconv::GEN_AI_TOOL_CALL_ID, "tool_call_id"]))
    {
        attributes.push(KeyValue::new(semconv::GEN_AI_TOOL_CALL_ID, value));
    }
    if capture_content {
        if let Some(value) = scalar_string(
            event,
            &[
                semconv::GEN_AI_TOOL_DESCRIPTION,
                "tool_description",
                "description",
            ],
        ) {
            attributes.push(KeyValue::new(semconv::GEN_AI_TOOL_DESCRIPTION, value));
        }
        if let Some(arguments) = wrapped_content_or_value(event.input(), &["arguments", "input"]) {
            attributes.push(KeyValue::new(
                GEN_AI_TOOL_CALL_ARGUMENTS,
                json_string(arguments),
            ));
        }
    }
}

fn push_retrieval_attributes(attributes: &mut Vec<KeyValue>, event: &Event, capture_content: bool) {
    if let Some(value) = data_source_id(event) {
        attributes.push(KeyValue::new(semconv::GEN_AI_DATA_SOURCE_ID, value));
    }
    push_model_attribute(attributes, event);
    if let Some(value) = scalar_f64(event, &[semconv::GEN_AI_REQUEST_TOP_K, "top_k"]) {
        attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_TOP_K, value));
    }
    if capture_content {
        push_retrieval_content(attributes, event);
    }
}

fn push_retrieval_content(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(value) = content_value(event.data(), &["query", "query_text"]) {
        let query = value
            .as_str()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| json_string(value));
        attributes.push(KeyValue::new(GEN_AI_RETRIEVAL_QUERY_TEXT, query));
    }
    if let Some(value) = content_value(event.data(), &["documents", "results"]) {
        attributes.push(KeyValue::new(
            GEN_AI_RETRIEVAL_DOCUMENTS,
            json_string(value),
        ));
    }
}

fn push_tool_definitions(attributes: &mut Vec<KeyValue>, event: &Event) {
    let definitions = if let Some(request) = event.normalized_llm_request()
        && let Some(tools) = request.tools.as_ref()
    {
        tools
            .iter()
            .map(|tool| {
                let mut definition = Map::new();
                definition.insert("type".to_string(), Value::String(tool.tool_type.clone()));
                definition.insert(
                    "name".to_string(),
                    Value::String(tool.function.name.clone()),
                );
                if let Some(value) = tool.function.description.as_ref() {
                    definition.insert("description".to_string(), Value::String(value.clone()));
                }
                if let Some(value) = tool.function.parameters.as_ref() {
                    definition.insert("parameters".to_string(), value.clone());
                }
                Value::Object(definition)
            })
            .collect::<Vec<_>>()
    } else if let Some(value) = content_value(event.data(), &["tool_definitions", "tools"])
        && let Some(values) = value.as_array()
    {
        values.clone()
    } else {
        return;
    };
    attributes.push(KeyValue::new(
        GEN_AI_TOOL_DEFINITIONS,
        json_string(&Value::Array(definitions)),
    ));
}

fn request_messages(messages: &[Message]) -> (Vec<Value>, Vec<Value>) {
    let mut instructions = Vec::new();
    let mut input = Vec::new();
    for message in messages {
        match message {
            Message::System { content, .. } => instructions.extend(content_parts(content)),
            Message::User { content, name } => input.push(message_value(
                "user",
                name.as_deref(),
                content_parts(content),
            )),
            Message::Assistant {
                content,
                tool_calls,
                name,
            } => {
                let mut parts = content.as_ref().map(content_parts).unwrap_or_default();
                if let Some(tool_calls) = tool_calls {
                    parts.extend(tool_calls.iter().map(|call| {
                        let arguments = serde_json::from_str(&call.function.arguments)
                            .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));
                        json!({
                            "type": "tool_call",
                            "id": call.id,
                            "name": call.function.name,
                            "arguments": arguments,
                        })
                    }));
                }
                input.push(message_value("assistant", name.as_deref(), parts));
            }
            Message::Tool {
                content,
                tool_call_id,
            } => input.push(json!({
                "role": "tool",
                "parts": [{
                    "type": "tool_call_response",
                    "id": tool_call_id,
                    "result": content_result(content),
                }],
            })),
        }
    }
    (instructions, input)
}

fn output_message(response: &AnnotatedLlmResponse) -> Option<Value> {
    let mut parts = response
        .message
        .as_ref()
        .map(content_parts)
        .unwrap_or_default();
    if let Some(tool_calls) = response.tool_calls.as_ref() {
        parts.extend(tool_calls.iter().map(|call| {
            json!({
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
    let mut message = Map::new();
    message.insert("role".to_string(), Value::String("assistant".to_string()));
    message.insert("parts".to_string(), Value::Array(parts));
    if let Some(reason) = response.finish_reason.as_ref() {
        message.insert(
            "finish_reason".to_string(),
            Value::String(finish_reason(reason).to_string()),
        );
    }
    Some(Value::Object(message))
}

fn message_value(role: &str, name: Option<&str>, parts: Vec<Value>) -> Value {
    let mut message = Map::new();
    message.insert("role".to_string(), Value::String(role.to_string()));
    message.insert("parts".to_string(), Value::Array(parts));
    if let Some(name) = name {
        message.insert("name".to_string(), Value::String(name.to_string()));
    }
    Value::Object(message)
}

fn content_parts(content: &MessageContent) -> Vec<Value> {
    match content {
        MessageContent::Text(content) => vec![json!({"type": "text", "content": content})],
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => json!({"type": "text", "content": text}),
                ContentPart::ImageUrl { image_url } => {
                    json!({"type": "uri", "uri": image_url.url})
                }
            })
            .collect(),
    }
}

fn content_result(content: &MessageContent) -> Value {
    match content {
        MessageContent::Text(value) => Value::String(value.clone()),
        MessageContent::Parts(_) => Value::Array(content_parts(content)),
    }
}

fn finish_reason(reason: &FinishReason) -> &str {
    match reason {
        FinishReason::Complete => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolUse => "tool_calls",
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
    scalar_string(event, &[GEN_AI_PROVIDER_NAME, "provider_name", "provider"]).or_else(|| {
        let name = event.name().to_ascii_lowercase();
        [
            ("azure", "azure.ai.openai"),
            ("anthropic", "anthropic"),
            ("bedrock", "aws.bedrock"),
            ("cohere", "cohere"),
            ("deepseek", "deepseek"),
            ("gemini", "gcp.gemini"),
            ("vertex", "gcp.vertex_ai"),
            ("groq", "groq"),
            ("mistral", "mistral_ai"),
            ("openai", "openai"),
            ("perplexity", "perplexity"),
        ]
        .into_iter()
        .find_map(|(needle, provider)| name.contains(needle).then(|| provider.to_string()))
    })
}

fn data_source_id(event: &Event) -> Option<String> {
    scalar_string(
        event,
        &[
            semconv::GEN_AI_DATA_SOURCE_ID,
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
    if let Some(profile) = event.category_profile() {
        for key in keys {
            if let Some(value) = profile.extra.get(*key) {
                if let Some(value) = value.as_str() {
                    return Some(value.to_string());
                }
                if value.is_number() || value.is_boolean() {
                    return Some(value.to_string());
                }
            }
        }
    }
    for object in event_objects(event) {
        for key in keys {
            if let Some(value) = object_value(object, key) {
                if let Some(value) = value.as_str() {
                    return Some(value.to_string());
                }
                if value.is_number() || value.is_boolean() {
                    return Some(value.to_string());
                }
            }
        }
    }
    None
}

fn scalar_i64(event: &Event, keys: &[&str]) -> Option<i64> {
    if let Some(profile) = event.category_profile() {
        for key in keys {
            if let Some(value) = profile.extra.get(*key) {
                if let Some(value) = value.as_i64() {
                    return Some(value);
                }
                if let Some(value) = value.as_u64().and_then(to_i64) {
                    return Some(value);
                }
            }
        }
    }
    for object in event_objects(event) {
        for key in keys {
            if let Some(value) = object_value(object, key) {
                if let Some(value) = value.as_i64() {
                    return Some(value);
                }
                if let Some(value) = value.as_u64().and_then(to_i64) {
                    return Some(value);
                }
            }
        }
    }
    None
}

fn scalar_f64(event: &Event, keys: &[&str]) -> Option<f64> {
    if let Some(profile) = event.category_profile() {
        for key in keys {
            if let Some(value) = profile.extra.get(*key).and_then(Value::as_f64) {
                return Some(value);
            }
        }
    }
    for object in event_objects(event) {
        for key in keys {
            if let Some(value) = object_value(object, key).and_then(Value::as_f64) {
                return Some(value);
            }
        }
    }
    None
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

fn content_value<'a>(value: Option<&'a Json>, keys: &[&str]) -> Option<&'a Json> {
    let value = value?;
    if let Some(object) = value.as_object() {
        for key in keys {
            if let Some(value) = object.get(*key) {
                return Some(value);
            }
        }
        return None;
    }
    Some(value)
}

fn wrapped_content_or_value<'a>(value: Option<&'a Json>, keys: &[&str]) -> Option<&'a Json> {
    let value = value?;
    content_value(Some(value), keys).or(Some(value))
}

fn json_string(value: &Value) -> String {
    serde_json::to_string(value).expect("serializing a JSON value cannot fail")
}

fn string_array(values: impl IntoIterator<Item = String>) -> opentelemetry::Value {
    opentelemetry::Value::Array(opentelemetry::Array::String(
        values.into_iter().map(Into::into).collect(),
    ))
}

fn to_i64(value: u64) -> Option<i64> {
    i64::try_from(value).ok()
}
