// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry GenAI semantic-convention projection.

#![allow(deprecated)] // Generated GenAI constants are retained for the pinned v1.42-era schema.

use crate::api::event::{Event, EventNormalizationExt};
use crate::api::scope::ScopeType;
use crate::codec::request::{
    ApiSpecificRequest, ContentPart, Message, MessageContent, tool_definition_identities,
};
use crate::codec::response::{AnnotatedLlmResponse, ApiSpecificResponse, FinishReason};
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

// The pinned OpenTelemetry crate does not yet generate these development
// attributes. Keep only those keys projection-local until it does.
const GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID: &str = "gen_ai.request.previous_response.id";
const GEN_AI_REQUEST_REASONING_LEVEL: &str = "gen_ai.request.reasoning.level";
const GEN_AI_RETRIEVAL_TOP_K: &str = "gen_ai.retrieval.top_k";
const GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS: &str = "gen_ai.usage.cache_write.input_tokens";

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
    ) || is_agent_turn(event)
}

fn is_agent_turn(event: &Event) -> bool {
    event.scope_type() == Some(ScopeType::Custom)
        && event
            .metadata()
            .and_then(Json::as_object)
            .and_then(|metadata| metadata.get("nemo_relay_scope_role"))
            .and_then(Json::as_str)
            == Some("turn")
}

pub(super) fn span_name(event: &Event) -> String {
    if !has_gen_ai_semantics(event) {
        return event.name().to_string();
    }
    let operation = operation_name(event);
    let qualifier = match event.scope_type() {
        Some(ScopeType::Agent) => Some(agent_name(event)),
        Some(ScopeType::Custom) if is_agent_turn(event) => {
            semantic_string(event, semconv::GEN_AI_AGENT_NAME)
        }
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
        Some(ScopeType::Custom) if is_agent_turn(event) => {
            push_conversation_attribute(&mut attributes, event);
            push_explicit_internal_agent_attributes(&mut attributes, event);
        }
        Some(ScopeType::Llm) => {
            push_provider_and_server_attributes(&mut attributes, event);
            push_conversation_attribute(&mut attributes, event);
            push_llm_request_attributes(&mut attributes, event);
            push_tool_definitions(&mut attributes, event);
        }
        Some(ScopeType::Tool) => {
            push_conversation_attribute(&mut attributes, event);
            push_tool_attributes(&mut attributes, event);
            push_tool_content(
                &mut attributes,
                semconv::GEN_AI_TOOL_CALL_ARGUMENTS,
                event.input(),
            );
        }
        Some(ScopeType::Retriever) => {
            push_provider_and_server_attributes(&mut attributes, event);
            push_retrieval_attributes(&mut attributes, event);
        }
        Some(ScopeType::Embedder) => {
            push_provider_and_server_attributes(&mut attributes, event);
            push_embedding_request_attributes(&mut attributes, event);
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
        Some(ScopeType::Tool) => {
            // Correlation can become available only at completion. Never replace
            // the start name with an end-event label.
            push_tool_metadata(&mut attributes, event);
            if !attributes.iter().any(|a| a.key.as_str() == "error.type") {
                push_tool_content(
                    &mut attributes,
                    semconv::GEN_AI_TOOL_CALL_RESULT,
                    event.output(),
                );
            }
        }
        _ => {}
    }
    attributes
}

/// Build the low-cardinality dimensions required by the standard GenAI client
/// metrics. Returns `None` when Relay cannot determine the required provider.
pub(super) fn client_metric_attributes(event: &Event) -> Option<Json> {
    let provider = provider_name(event)?;
    let mut attributes = Map::new();
    attributes.insert(
        semconv::GEN_AI_OPERATION_NAME.to_string(),
        Json::String(operation_name(event).to_string()),
    );
    attributes.insert(
        semconv::GEN_AI_PROVIDER_NAME.to_string(),
        Json::String(provider),
    );
    if let Some(model) = request_model(event) {
        attributes.insert(
            semconv::GEN_AI_REQUEST_MODEL.to_string(),
            Json::String(model),
        );
    }
    if let Some(model) = event
        .annotated_response()
        .and_then(|response| response.model.clone())
    {
        attributes.insert(
            semconv::GEN_AI_RESPONSE_MODEL.to_string(),
            Json::String(model),
        );
    }
    if let Some(address) = scalar_string(event, &[semconv::SERVER_ADDRESS, "server_address"]) {
        attributes.insert(semconv::SERVER_ADDRESS.to_string(), Json::String(address));
    }
    if let Some(port) = scalar_i64(event, &[semconv::SERVER_PORT, "server_port"]) {
        attributes.insert(semconv::SERVER_PORT.to_string(), Json::from(port));
    }
    Some(Json::Object(attributes))
}

fn push_embedding_response_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
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
}

fn operation_name(event: &Event) -> &'static str {
    match event.scope_type() {
        Some(ScopeType::Agent) => OPERATION_INVOKE_AGENT,
        Some(ScopeType::Custom) if is_agent_turn(event) => OPERATION_INVOKE_AGENT,
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
        attributes.push(KeyValue::new(semconv::GEN_AI_PROVIDER_NAME, provider));
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
}

fn push_agent_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    attributes.push(KeyValue::new(semconv::GEN_AI_AGENT_NAME, agent_name(event)));
    if let Some(value) = scalar_string(
        event,
        &[semconv::GEN_AI_AGENT_DESCRIPTION, "agent_description"],
    ) {
        attributes.push(KeyValue::new(semconv::GEN_AI_AGENT_DESCRIPTION, value));
    }
    // Internal agent spans may report a model only when the agent is known to
    // use a single configured model. Require the instrumentation source to make
    // that assertion through the canonical key rather than inferring it from a
    // generic `model` label.
    if let Some(value) = semantic_string(event, semconv::GEN_AI_REQUEST_MODEL) {
        attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_MODEL, value));
    }
}

fn push_explicit_internal_agent_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    for key in [
        semconv::GEN_AI_AGENT_NAME,
        semconv::GEN_AI_AGENT_DESCRIPTION,
    ] {
        if let Some(value) = semantic_string(event, key) {
            attributes.push(KeyValue::new(key, value));
        }
    }
}

fn push_model_attribute(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(model) = request_model(event) {
        attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_MODEL, model));
    }
}

fn push_embedding_request_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    push_model_attribute(attributes, event);
    if let Some(value) = scalar_i64(
        event,
        &[semconv::GEN_AI_EMBEDDINGS_DIMENSION_COUNT, "dimensions"],
    )
    .filter(|value| *value > 0)
    {
        attributes.push(KeyValue::new(
            semconv::GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
            value,
        ));
    }
    if let Some(values) = string_values(
        event,
        &[
            semconv::GEN_AI_REQUEST_ENCODING_FORMATS,
            "encoding_formats",
            "encoding_format",
        ],
    ) {
        attributes.push(KeyValue::new(
            semconv::GEN_AI_REQUEST_ENCODING_FORMATS,
            string_array(values),
        ));
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
    if request.stream == Some(true) {
        attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_STREAM, true));
    }
    if let Some(value) = request
        .previous_response_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        attributes.push(KeyValue::new(
            GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID,
            value.to_string(),
        ));
    }
    if let Some(value) = request_reasoning_level(event, request) {
        attributes.push(KeyValue::new(GEN_AI_REQUEST_REASONING_LEVEL, value));
    }
    if let Some(value) = requested_output_type(event, request.api_specific.as_ref()) {
        attributes.push(KeyValue::new(semconv::GEN_AI_OUTPUT_TYPE, value));
    }
    if let Some(instructions) = request
        .instructions
        .as_ref()
        .and_then(system_instructions_json)
    {
        attributes.push(KeyValue::new(
            semconv::GEN_AI_SYSTEM_INSTRUCTIONS,
            instructions,
        ));
    }
    if let Some(messages) = input_messages_json(&request.messages) {
        attributes.push(KeyValue::new(semconv::GEN_AI_INPUT_MESSAGES, messages));
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
                attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_TOP_K, value));
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
                attributes.push(KeyValue::new(
                    semconv::GEN_AI_REQUEST_FREQUENCY_PENALTY,
                    *value,
                ));
            }
            if let Some(value) = n.filter(|value| *value != 1).and_then(to_i64) {
                attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_CHOICE_COUNT, value));
            }
            if let Some(value) = presence_penalty {
                attributes.push(KeyValue::new(
                    semconv::GEN_AI_REQUEST_PRESENCE_PENALTY,
                    *value,
                ));
            }
            if let Some(value) = seed {
                attributes.push(KeyValue::new(semconv::GEN_AI_REQUEST_SEED, *value));
            }
        }
        _ => {}
    }
}

fn push_llm_response_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(value) = event.time_to_first_chunk() {
        attributes.push(KeyValue::new(
            semconv::GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
            value,
        ));
    }
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
    if let Some(messages) = output_messages_json(response) {
        attributes.push(KeyValue::new(semconv::GEN_AI_OUTPUT_MESSAGES, messages));
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
            attributes.push(KeyValue::new(semconv::GEN_AI_USAGE_OUTPUT_TOKENS, value));
        }
        if let Some(value) = usage.cache_read_tokens.and_then(to_i64) {
            attributes.push(KeyValue::new(
                semconv::GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                value,
            ));
        }
        if let Some(value) = usage.cache_write_tokens.and_then(to_i64) {
            attributes.push(KeyValue::new(GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS, value));
        }
    }
    if let Some(value) = reasoning_output_tokens(event, response) {
        attributes.push(KeyValue::new(
            semconv::GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
            value,
        ));
    }
}

fn request_reasoning_level(
    event: &Event,
    request: &crate::codec::request::AnnotatedLlmRequest,
) -> Option<String> {
    semantic_string(event, GEN_AI_REQUEST_REASONING_LEVEL).or_else(|| {
        match request.api_specific.as_ref() {
            Some(ApiSpecificRequest::OpenAIChat {
                reasoning_effort, ..
            }) => reasoning_effort
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .map(ToOwned::to_owned),
            _ => request
                .reasoning
                .as_ref()
                .and_then(Json::as_object)
                .and_then(|reasoning| reasoning.get("effort"))
                .and_then(Json::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(ToOwned::to_owned),
        }
    })
}

fn requested_output_type(
    event: &Event,
    api_specific: Option<&ApiSpecificRequest>,
) -> Option<String> {
    semantic_string(event, semconv::GEN_AI_OUTPUT_TYPE).or_else(|| match api_specific {
        Some(ApiSpecificRequest::OpenAIChat {
            modalities,
            response_format,
            ..
        }) => response_format
            .as_ref()
            .and_then(json_output_type)
            .or_else(|| openai_chat_output_type(modalities.as_deref())),
        Some(ApiSpecificRequest::OpenAIResponses { text, .. }) => text
            .as_ref()
            .and_then(Json::as_object)
            .and_then(|text| text.get("format"))
            .and_then(json_output_type),
        _ => None,
    })
}

fn json_output_type(format: &Json) -> Option<String> {
    matches!(
        format.as_object()?.get("type")?.as_str()?,
        "json_object" | "json_schema"
    )
    .then(|| "json".to_string())
}

fn openai_chat_output_type(modalities: Option<&[String]>) -> Option<String> {
    match modalities? {
        [modality] if modality == "audio" => Some("speech".to_string()),
        [modality] if modality == "text" => Some("text".to_string()),
        _ => None,
    }
}

fn reasoning_output_tokens(event: &Event, response: &AnnotatedLlmResponse) -> Option<i64> {
    scalar_i64(event, &[semconv::GEN_AI_USAGE_REASONING_OUTPUT_TOKENS])
        .filter(|value| *value >= 0)
        .or_else(|| match response.api_specific.as_ref() {
            Some(ApiSpecificResponse::OpenAIResponses {
                output_tokens_details: Some(details),
                ..
            }) => details
                .get("reasoning_tokens")
                .and_then(Json::as_u64)
                .and_then(to_i64),
            Some(ApiSpecificResponse::GeminiGenerateContent {
                thoughts_tokens, ..
            }) => thoughts_tokens.and_then(to_i64),
            _ => None,
        })
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
    push_tool_metadata(attributes, event);
}

fn push_tool_metadata(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(value) = tool_metadata_string(event, &[semconv::GEN_AI_TOOL_TYPE, "tool_type"]) {
        attributes.push(KeyValue::new(semconv::GEN_AI_TOOL_TYPE, value));
    }
    if let Some(value) = event
        .tool_call_id()
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| tool_metadata_string(event, &[semconv::GEN_AI_TOOL_CALL_ID, "tool_call_id"]))
    {
        attributes.push(KeyValue::new(semconv::GEN_AI_TOOL_CALL_ID, value));
    }
    if let Some(value) = tool_metadata_string(
        event,
        &[
            semconv::GEN_AI_TOOL_DESCRIPTION,
            "tool_description",
            "description",
        ],
    ) {
        attributes.push(KeyValue::new(semconv::GEN_AI_TOOL_DESCRIPTION, value));
    }
    if let Some(value) = tool_metadata_string(event, &[semconv::GEN_AI_AGENT_NAME, "agent_name"]) {
        attributes.push(KeyValue::new(semconv::GEN_AI_AGENT_NAME, value));
    }
}

// Tool data is arbitrary user content: a payload's `description`, `tool_type`,
// or even a dotted semantic key must not masquerade as instrumentation metadata.
fn tool_metadata_string(event: &Event, keys: &[&str]) -> Option<String> {
    tool_metadata_with_origin(event, keys).map(|(value, _)| value)
}

// Explicit aliases and typed profile metadata outrank inferred canonical keys.
fn tool_metadata_with_origin(event: &Event, keys: &[&str]) -> Option<(String, bool)> {
    let mut fallback = None;
    for key in keys {
        let profile = event
            .category_profile()
            .and_then(|profile| profile.extra.get(*key));
        let metadata = event.metadata().and_then(|metadata| metadata.get(*key));
        let inferred = event
            .metadata()
            .and_then(|metadata| metadata.get("nemo_relay.tool.execution.defaults"))
            .and_then(|defaults| defaults.get(*key));
        for (value, is_inferred) in [
            (profile, false),
            (metadata, metadata.is_some() && metadata == inferred),
        ] {
            let Some(value) = value
                .and_then(Json::as_str)
                .filter(|value| !value.trim().is_empty())
            else {
                continue;
            };
            if !is_inferred {
                return Some((value.to_string(), false));
            }
            fallback.get_or_insert_with(|| (value.to_string(), true));
        }
    }
    fallback
}

pub(super) fn inferred_tool_attribute_keys(event: &Event) -> std::collections::HashSet<String> {
    [
        (
            semconv::GEN_AI_TOOL_TYPE,
            [semconv::GEN_AI_TOOL_TYPE, "tool_type"],
        ),
        (
            semconv::GEN_AI_AGENT_NAME,
            [semconv::GEN_AI_AGENT_NAME, "agent_name"],
        ),
    ]
    .into_iter()
    .filter_map(|(key, aliases)| {
        tool_metadata_with_origin(event, &aliases)
            .and_then(|(_, inferred)| inferred.then(|| key.to_string()))
    })
    .collect()
}

fn push_tool_content(attributes: &mut Vec<KeyValue>, key: &'static str, value: Option<&Json>) {
    let Some(value) = value else { return };
    let parsed = value
        .as_str()
        .and_then(|text| serde_json::from_str::<Json>(text).ok());
    let value = parsed.as_ref().unwrap_or(value);
    // Span attributes cannot carry structured JSON directly, so preserve every
    // non-null sanitized payload as canonical JSON text without inventing a
    // wrapper that changes the tool's arguments or result.
    if !value.is_null() {
        attributes.push(KeyValue::new(key, value.to_string()));
    }
}

fn push_tool_definitions(attributes: &mut Vec<KeyValue>, event: &Event) {
    let Some(request) = event.normalized_llm_request() else {
        return;
    };
    let Some(tools) = request.as_ref().tools.as_ref() else {
        return;
    };
    let definitions: Vec<Json> = tools
        .iter()
        .flat_map(tool_definition_identities)
        .map(|(tool_type, name)| serde_json::json!({"type": tool_type, "name": name}))
        .collect();
    // Only required schema properties; provider wrappers and optional schemas
    // can be large and need not accompany every inference.
    if !definitions.is_empty() {
        attributes.push(KeyValue::new(
            semconv::GEN_AI_TOOL_DEFINITIONS,
            Json::Array(definitions).to_string(),
        ));
    }
}

fn push_retrieval_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(value) = data_source_id(event) {
        attributes.push(KeyValue::new(semconv::GEN_AI_DATA_SOURCE_ID, value));
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
    scalar_string(
        event,
        &[semconv::GEN_AI_PROVIDER_NAME, "provider_name", "provider"],
    )
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
    scalar_string(event, &[semconv::GEN_AI_AGENT_NAME]).unwrap_or_else(|| event.name().to_string())
}

fn tool_name(event: &Event) -> String {
    tool_metadata_string(event, &[semconv::GEN_AI_TOOL_NAME])
        .unwrap_or_else(|| event.name().to_string())
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

fn semantic_string(event: &Event, key: &str) -> Option<String> {
    find_scalar(event, &[key], |value| {
        value
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .map(ToOwned::to_owned)
    })
}

fn string_values(event: &Event, keys: &[&str]) -> Option<Vec<String>> {
    find_scalar(event, keys, |value| match value {
        Json::String(value) if !value.trim().is_empty() => Some(vec![value.clone()]),
        Json::Array(values) => {
            let values = values
                .iter()
                .map(Json::as_str)
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .filter(|value| !value.trim().is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            (!values.is_empty()).then_some(values)
        }
        _ => None,
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
