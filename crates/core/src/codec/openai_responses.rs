// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the OpenAI Responses API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the OpenAI Responses API format.
//!
//! The Responses API differs significantly from Chat Completions:
//! - **Response**: Heterogeneous `output` array (message, function_call, reasoning)
//!   instead of `choices[0].message`.
//! - **Finish reason**: Derived from `status` + `incomplete_details.reason`
//!   instead of `finish_reason` field.
//! - **Request**: Uses `input` (string or array) instead of `messages`, and
//!   `instructions` (top-level) instead of system message.
//! - **Max tokens**: `max_output_tokens` instead of `max_tokens`.

use serde::Deserialize;

use crate::api::llm::LlmRequest;
use crate::error::{FlowError, Result};
use crate::json::Json;

use super::request::{
    AnnotatedLlmRequest, GenerationParams, Message, MessageContent, ToolChoice, ToolChoiceFunction,
    ToolChoiceFunctionName, ToolDefinition,
};
use super::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, FinishReason, RawUsageCost, ResponseToolCall, Usage,
    estimate_cost_for_provider, infer_model_provider, provider_reported_cost,
};
use super::traits::{LlmCodec, LlmResponseCodec};

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Responses API.
pub struct OpenAIResponsesCodec;

// ---------------------------------------------------------------------------
// Private intermediate serde structs for response decode
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawResponsesResponse {
    id: Option<String>,
    model: Option<String>,
    status: Option<String>,
    output: Option<Vec<Json>>,
    usage: Option<RawResponsesUsage>,
    incomplete_details: Option<Json>,
    previous_response_id: Option<String>,
    store: Option<bool>,
    service_tier: Option<String>,
    truncation: Option<Json>,
    reasoning: Option<Json>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawResponsesUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
    input_tokens_details: Option<RawInputTokensDetails>,
    output_tokens_details: Option<RawOutputTokensDetails>,
    #[serde(rename = "cost_usd")]
    provider_cost: Option<f64>,
    cost: Option<RawUsageCost>,
}

#[derive(Deserialize, Clone)]
struct RawInputTokensDetails {
    cached_tokens: Option<u64>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize, Clone)]
struct RawOutputTokensDetails {
    reasoning_tokens: Option<u64>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Map Responses API `status` + `incomplete_details` to normalized [`FinishReason`].
fn map_responses_finish_reason(
    status: Option<&str>,
    incomplete_details: Option<&Json>,
) -> Option<FinishReason> {
    let incomplete_reason = incomplete_details
        .and_then(|d| d.get("reason"))
        .and_then(|r| r.as_str());

    match status {
        Some("completed") => Some(FinishReason::Complete),
        Some("incomplete") => match incomplete_reason {
            Some("max_output_tokens") => Some(FinishReason::Length),
            Some("content_filter") => Some(FinishReason::ContentFilter),
            Some(other) => Some(FinishReason::Unknown(other.to_string())),
            None => Some(FinishReason::Unknown("incomplete".to_string())),
        },
        Some(other) => Some(FinishReason::Unknown(other.to_string())),
        None => None,
    }
}

/// Parse OpenAI tool call arguments from JSON string to [`Json`] value.
///
/// Falls back to [`Json::String`] if parsing fails (malformed model output).
fn parse_arguments(arguments: &str) -> Json {
    serde_json::from_str(arguments).unwrap_or_else(|_| Json::String(arguments.to_string()))
}

fn input_tokens_details_to_json(details: &RawInputTokensDetails) -> Json {
    let mut obj = serde_json::Map::new();
    if let Some(cached_tokens) = details.cached_tokens {
        obj.insert("cached_tokens".into(), Json::from(cached_tokens));
    }
    obj.extend(details.extra.clone());
    Json::Object(obj)
}

fn output_tokens_details_to_json(details: &RawOutputTokensDetails) -> Json {
    let mut obj = serde_json::Map::new();
    if let Some(reasoning_tokens) = details.reasoning_tokens {
        obj.insert("reasoning_tokens".into(), Json::from(reasoning_tokens));
    }
    obj.extend(details.extra.clone());
    Json::Object(obj)
}

/// Keys that are modeled in [`AnnotatedLlmRequest`] and should NOT go into `extra`.
const MODELED_REQUEST_KEYS: &[&str] = &[
    "input",
    "instructions",
    "model",
    "max_output_tokens",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "store",
    "previous_response_id",
    "truncation",
    "reasoning",
    "include",
    "user",
    "metadata",
    "service_tier",
    "parallel_tool_calls",
    "max_tool_calls",
    "top_logprobs",
    "stream",
];
const UNPARSED_INPUT_ITEMS_KEY: &str = "_openai_responses_unparsed_input_items";

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

fn collect_output_parts(items: Option<&[Json]>) -> (Vec<String>, Vec<ResponseToolCall>) {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    if let Some(items) = items {
        for item in items {
            collect_output_item(item, &mut text_parts, &mut tool_calls);
        }
    }

    (text_parts, tool_calls)
}

fn collect_output_item(
    item: &Json,
    text_parts: &mut Vec<String>,
    tool_calls: &mut Vec<ResponseToolCall>,
) {
    match item
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("")
    {
        "message" => collect_message_text_parts(item, text_parts),
        "output_text" => {
            if let Some(text) = output_text_block(item) {
                text_parts.push(text);
            }
        }
        "function_call" => tool_calls.push(parse_function_call(item)),
        _ => {}
    }
}

fn collect_message_text_parts(item: &Json, text_parts: &mut Vec<String>) {
    let Some(content) = item.get("content").and_then(|value| value.as_array()) else {
        return;
    };

    for block in content {
        if let Some(text) = output_text_block(block) {
            text_parts.push(text);
        }
    }
}

fn output_text_block(block: &Json) -> Option<String> {
    (block.get("type").and_then(|value| value.as_str()) == Some("output_text"))
        .then(|| block.get("text").and_then(|value| value.as_str()))
        .flatten()
        .map(str::to_string)
}

fn parse_function_call(item: &Json) -> ResponseToolCall {
    ResponseToolCall {
        id: item
            .get("call_id")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string(),
        name: item
            .get("name")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string(),
        arguments: item
            .get("arguments")
            .and_then(|value| value.as_str())
            .map(parse_arguments)
            .unwrap_or(Json::Object(serde_json::Map::new())),
    }
}

fn message_from_text_parts(text_parts: Vec<String>) -> Option<MessageContent> {
    match text_parts.as_slice() {
        [] => None,
        [text] => Some(MessageContent::Text(text.clone())),
        _ => Some(MessageContent::Text(text_parts.join("\n"))),
    }
}

fn top_level_output_text(response: &Json) -> Option<MessageContent> {
    response
        .get("output_text")
        .and_then(|value| value.as_str())
        .filter(|text| !text.is_empty())
        .map(|text| MessageContent::Text(text.to_string()))
}

fn optional_vec<T>(items: Vec<T>) -> Option<Vec<T>> {
    (!items.is_empty()).then_some(items)
}

fn split_system_and_input_messages(messages: &[Message]) -> (Option<String>, Vec<&Message>) {
    let mut system_text = None;
    let mut input_messages = Vec::new();

    for msg in messages {
        match msg {
            Message::System { content, .. } => {
                if let MessageContent::Text(text) = content {
                    system_text = Some(text.clone());
                }
            }
            other => input_messages.push(other),
        }
    }

    (system_text, input_messages)
}

fn set_or_remove_string(obj: &mut serde_json::Map<String, Json>, key: &str, value: Option<String>) {
    if let Some(value) = value {
        obj.insert(key.into(), Json::String(value));
    } else {
        obj.remove(key);
    }
}

fn insert_serialized<T: serde::Serialize>(
    obj: &mut serde_json::Map<String, Json>,
    key: &str,
    value: &T,
    context: &str,
) -> Result<()> {
    let json = serde_json::to_value(value)
        .map_err(|e| FlowError::Internal(format!("OpenAI Responses {context} encode: {e}")))?;
    obj.insert(key.into(), json);
    Ok(())
}

fn overlay_generation_params(obj: &mut serde_json::Map<String, Json>, params: &GenerationParams) {
    if let Some(temp) = params.temperature {
        obj.insert("temperature".into(), json_f64(temp));
    }
    if let Some(top_p) = params.top_p {
        obj.insert("top_p".into(), json_f64(top_p));
    }
    if let Some(max_tokens) = params.max_tokens {
        obj.insert("max_output_tokens".into(), Json::from(max_tokens));
        obj.remove("max_tokens");
    }
}

fn encode_openai_responses_input(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
) -> Result<()> {
    let (system_text, input_messages) = split_system_and_input_messages(&annotated.messages);
    set_or_remove_string(obj, "instructions", system_text);
    if let Some(raw_input_items) = annotated.extra.get(UNPARSED_INPUT_ITEMS_KEY) {
        obj.insert("input".into(), raw_input_items.clone());
    } else {
        insert_serialized(obj, "input", &input_messages, "input")?;
    }
    Ok(())
}

fn encode_openai_responses_tools(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
) -> Result<()> {
    if let Some(ref tools) = annotated.tools {
        insert_serialized(obj, "tools", tools, "tools")?;
    }
    if let Some(ref tool_choice) = annotated.tool_choice {
        insert_serialized(obj, "tool_choice", tool_choice, "tool_choice")?;
    }
    Ok(())
}

fn overlay_openai_responses_fields(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
) {
    if let Some(ref model) = annotated.model {
        obj.insert("model".into(), Json::String(model.clone()));
    }
    overlay_openai_responses_json_fields(obj, annotated);
    overlay_openai_responses_string_fields(obj, annotated);
    overlay_openai_responses_bool_fields(obj, annotated);
    overlay_openai_responses_u64_fields(obj, annotated);
}

fn overlay_openai_responses_json_fields(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
) {
    for (key, value) in [
        ("truncation", &annotated.truncation),
        ("reasoning", &annotated.reasoning),
        ("include", &annotated.include),
        ("metadata", &annotated.metadata),
    ] {
        if let Some(value) = value {
            obj.insert(key.into(), value.clone());
        }
    }
}

fn overlay_openai_responses_string_fields(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
) {
    for (key, value) in [
        ("previous_response_id", &annotated.previous_response_id),
        ("user", &annotated.user),
        ("service_tier", &annotated.service_tier),
    ] {
        if let Some(value) = value {
            obj.insert(key.into(), Json::String(value.clone()));
        }
    }
}

fn overlay_openai_responses_bool_fields(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
) {
    for (key, value) in [
        ("store", annotated.store),
        ("parallel_tool_calls", annotated.parallel_tool_calls),
        ("stream", annotated.stream),
    ] {
        if let Some(value) = value {
            obj.insert(key.into(), Json::Bool(value));
        }
    }
}

fn overlay_openai_responses_u64_fields(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
) {
    for (key, value) in [
        ("max_output_tokens", annotated.max_output_tokens),
        ("max_tool_calls", annotated.max_tool_calls),
        ("top_logprobs", annotated.top_logprobs),
    ] {
        if let Some(value) = value {
            obj.insert(key.into(), Json::from(value));
        }
    }
}

fn merge_openai_responses_extra_fields(
    obj: &mut serde_json::Map<String, Json>,
    extra: &serde_json::Map<String, Json>,
) {
    for (k, v) in extra {
        if k != UNPARSED_INPUT_ITEMS_KEY {
            obj.insert(k.clone(), v.clone());
        }
    }
}

fn decode_openai_or_anthropic_tool_choice(value: &Json) -> Option<ToolChoice> {
    if let Ok(parsed) = serde_json::from_value::<ToolChoice>(value.clone()) {
        return Some(parsed);
    }

    let obj = value.as_object()?;
    match obj.get("type").and_then(|v| v.as_str()) {
        Some("auto") => Some(ToolChoice::Auto),
        Some("any") => Some(ToolChoice::Required),
        Some("none") => Some(ToolChoice::None),
        Some("tool") => {
            let name = obj.get("name").and_then(|v| v.as_str())?.to_string();
            Some(ToolChoice::Specific(ToolChoiceFunction {
                choice_type: "function".to_string(),
                function: ToolChoiceFunctionName { name },
            }))
        }
        _ => None,
    }
}

fn decode_openai_or_anthropic_parallel_tool_calls(
    obj: &serde_json::Map<String, Json>,
) -> Option<bool> {
    if let Some(value) = obj.get("parallel_tool_calls").and_then(|v| v.as_bool()) {
        return Some(value);
    }
    let tool_choice = obj.get("tool_choice")?.as_object()?;
    tool_choice
        .get("disable_parallel_tool_use")
        .and_then(|v| v.as_bool())
        .map(|disabled| !disabled)
}

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for OpenAIResponsesCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let raw: RawResponsesResponse = serde_json::from_value(response.clone())
            .map_err(|e| FlowError::Internal(format!("OpenAI Responses response decode: {e}")))?;

        let all_output_items = raw.output.clone();
        let (text_parts, tool_calls) = collect_output_parts(raw.output.as_deref());
        let message =
            message_from_text_parts(text_parts).or_else(|| top_level_output_text(response));
        let tool_calls = optional_vec(tool_calls);

        // Map finish reason from status + incomplete_details.
        let finish_reason =
            map_responses_finish_reason(raw.status.as_deref(), raw.incomplete_details.as_ref());

        let input_tokens_details = raw.usage.as_ref().and_then(|u| {
            u.input_tokens_details
                .as_ref()
                .map(input_tokens_details_to_json)
        });
        let output_tokens_details = raw.usage.as_ref().and_then(|u| {
            u.output_tokens_details
                .as_ref()
                .map(output_tokens_details_to_json)
        });

        // Map usage.
        let model_for_pricing = raw.model.as_deref();
        let model_provider = infer_model_provider("openai", model_for_pricing);
        let usage = raw.usage.map(|u| {
            let mut usage = Usage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.total_tokens,
                cache_read_tokens: u
                    .input_tokens_details
                    .as_ref()
                    .and_then(|d| d.cached_tokens),
                cache_write_tokens: None,
                cost: provider_reported_cost(u.provider_cost, u.cost),
            };
            if usage.cost.is_none() {
                usage.cost = model_for_pricing.and_then(|model| {
                    estimate_cost_for_provider(model_provider.as_deref(), model, &usage)
                });
            }
            usage
        });

        // Build API-specific fields.
        let api_specific = Some(ApiSpecificResponse::OpenAIResponses {
            output_items: all_output_items,
            status: raw.status,
            incomplete_details: raw.incomplete_details,
            previous_response_id: raw.previous_response_id,
            store: raw.store,
            service_tier: raw.service_tier,
            truncation: raw.truncation,
            reasoning: raw.reasoning,
            input_tokens_details,
            output_tokens_details,
        });

        Ok(AnnotatedLlmResponse {
            id: raw.id,
            model: raw.model,
            message,
            tool_calls,
            finish_reason,
            usage,
            api_specific,
            extra: raw.extra,
        })
    }
}

// ---------------------------------------------------------------------------
// LlmCodec implementation
// ---------------------------------------------------------------------------

impl LlmCodec for OpenAIResponsesCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;

        let mut messages: Vec<Message> = Vec::new();
        let mut preserved_unparsed_input: Option<Json> = None;

        // Extract instructions -> system message (first).
        if let Some(instructions) = obj.get("instructions").and_then(|v| v.as_str()) {
            messages.push(Message::System {
                content: MessageContent::Text(instructions.to_string()),
                name: None,
            });
        }

        // Extract input.
        if let Some(input) = obj.get("input") {
            if let Some(s) = input.as_str() {
                // Input is a simple string -> single User message.
                messages.push(Message::User {
                    content: MessageContent::Text(s.to_string()),
                    name: None,
                });
            } else if input.is_array() {
                // Strict-first parse to avoid partial normalized state.
                match serde_json::from_value::<Vec<Message>>(input.clone()) {
                    Ok(input_messages) => messages.extend(input_messages),
                    Err(_) => {
                        // Preserve full original array for lossless handling.
                        preserved_unparsed_input = Some(input.clone());
                    }
                }
            }
        }

        // Extract model.
        let model = obj.get("model").and_then(|v| v.as_str()).map(String::from);

        // Extract generation params.
        let temperature = obj.get("temperature").and_then(|v| v.as_f64());
        let top_p = obj.get("top_p").and_then(|v| v.as_f64());
        let max_tokens = obj.get("max_output_tokens").and_then(|v| v.as_u64());
        // Responses API does not support stop sequences.

        let params = if temperature.is_some() || max_tokens.is_some() || top_p.is_some() {
            Some(GenerationParams {
                temperature,
                max_tokens,
                top_p,
                stop: None,
            })
        } else {
            None
        };

        // Extract tools.
        let tools: Option<Vec<ToolDefinition>> = obj
            .get("tools")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| FlowError::Internal(format!("OpenAI Responses tools decode: {e}")))?;

        // Extract tool_choice.
        let tool_choice: Option<ToolChoice> = obj
            .get("tool_choice")
            .and_then(decode_openai_or_anthropic_tool_choice);

        // Collect extra fields (keys not in MODELED_REQUEST_KEYS).
        let mut extra: serde_json::Map<String, Json> = obj
            .iter()
            .filter(|(k, _)| !MODELED_REQUEST_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if let Some(input_items) = preserved_unparsed_input {
            extra.insert(UNPARSED_INPUT_ITEMS_KEY.into(), input_items);
        }

        Ok(AnnotatedLlmRequest {
            messages,
            model,
            params,
            tools,
            tool_choice,
            store: obj.get("store").and_then(|v| v.as_bool()),
            previous_response_id: obj
                .get("previous_response_id")
                .and_then(|v| v.as_str())
                .map(String::from),
            truncation: obj.get("truncation").cloned(),
            reasoning: obj.get("reasoning").cloned(),
            include: obj.get("include").cloned(),
            user: obj.get("user").and_then(|v| v.as_str()).map(String::from),
            metadata: obj.get("metadata").cloned(),
            service_tier: obj
                .get("service_tier")
                .and_then(|v| v.as_str())
                .map(String::from),
            parallel_tool_calls: decode_openai_or_anthropic_parallel_tool_calls(obj),
            max_output_tokens: obj.get("max_output_tokens").and_then(|v| v.as_u64()),
            max_tool_calls: obj.get("max_tool_calls").and_then(|v| v.as_u64()),
            top_logprobs: obj.get("top_logprobs").and_then(|v| v.as_u64()),
            stream: obj.get("stream").and_then(|v| v.as_bool()),
            extra,
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        let mut content = original.content.clone();
        let obj = content
            .as_object_mut()
            .ok_or_else(|| FlowError::Internal("original content is not an object".into()))?;

        encode_openai_responses_input(obj, annotated)?;
        if let Some(ref params) = annotated.params {
            overlay_generation_params(obj, params);
        }
        encode_openai_responses_tools(obj, annotated)?;
        overlay_openai_responses_fields(obj, annotated);
        merge_openai_responses_extra_fields(obj, &annotated.extra);

        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

// ---------------------------------------------------------------------------
// Streaming codec
// ---------------------------------------------------------------------------

/// Streaming counterpart to [`OpenAIResponsesCodec`].
///
/// Replays the OpenAI Responses SSE event sequence into the same JSON shape the API returns for a
/// non-streaming request (`{id, model, status, output, usage, incomplete_details, ...}`). Once
/// finalized, the assembled JSON can be fed back through [`OpenAIResponsesCodec::decode_response`]
/// to produce the canonical [`AnnotatedLlmResponse`].
///
/// # Strategy
///
/// The Responses API is a relatively forgiving streaming target because every event carries
/// either the full `response` snapshot (`response.created`, `response.in_progress`,
/// `response.completed`, `response.failed`, `response.incomplete`) or the final-state output item
/// (`response.output_item.done`). We:
///
/// 1. Track the latest `response` snapshot — terminal events (`completed`/`failed`/`incomplete`)
///    typically carry the complete state including `output`, so we prefer those when present.
/// 2. Track output items by `output_index` — `output_item.done` events deliver the final per-item
///    state, used as a fallback when the terminal `response.output` is missing or empty.
/// 3. Per-token `output_text.delta` and `function_call_arguments.delta` events are ignored
///    because their content is redelivered in the matching `output_item.done` event. Skipping
///    deltas keeps the codec resilient to schema additions and avoids double-accumulation.
///
/// Internal state lives behind `Arc<Mutex<...>>` so the `&self`-produced collector and finalizer
/// closures share access. Each instance is single-use because [`LlmFinalizerFn`] consumes the
/// finalize step.
///
/// [`AnnotatedLlmResponse`]: crate::codec::response::AnnotatedLlmResponse
/// [`LlmFinalizerFn`]: crate::api::runtime::LlmFinalizerFn
pub struct OpenAIResponsesStreamingCodec {
    state: std::sync::Arc<std::sync::Mutex<OpenAIResponsesStreamingState>>,
}

impl OpenAIResponsesStreamingCodec {
    /// Creates a fresh streaming codec with empty accumulator state.
    pub fn new() -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(
                OpenAIResponsesStreamingState::default(),
            )),
        }
    }
}

impl Default for OpenAIResponsesStreamingCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl super::streaming::StreamingCodec for OpenAIResponsesStreamingCodec {
    fn collector(&self) -> crate::api::runtime::LlmCollectorFn {
        let state = std::sync::Arc::clone(&self.state);
        Box::new(move |event: Json| -> Result<()> {
            let mut guard = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.observe(&event);
            Ok(())
        })
    }

    fn finalizer(&self) -> crate::api::runtime::LlmFinalizerFn {
        let state = std::sync::Arc::clone(&self.state);
        Box::new(move || -> Json {
            let mut guard = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::take(&mut *guard).finalize()
        })
    }
}

#[derive(Debug, Default)]
struct OpenAIResponsesStreamingState {
    /// Latest `response` snapshot from any event that carries one. Last write wins, so terminal
    /// events with the complete state will end up here when they fire.
    response: Option<serde_json::Map<String, Json>>,
    /// Items keyed by `output_index`. Captured from `response.output_item.added` (initial) and
    /// replaced on `response.output_item.done` (final). Used as a fallback for `output` when the
    /// terminal `response` snapshot lacks it.
    items: std::collections::BTreeMap<usize, Json>,
}

impl OpenAIResponsesStreamingState {
    fn observe(&mut self, event: &Json) {
        let event_type = event.get("type").and_then(Json::as_str).unwrap_or("");
        match event_type {
            "response.created"
            | "response.in_progress"
            | "response.completed"
            | "response.failed"
            | "response.incomplete" => self.observe_response_snapshot(event),
            "response.output_item.added" | "response.output_item.done" => {
                self.observe_output_item(event);
            }
            // response.output_text.delta, response.function_call_arguments.delta,
            // response.content_part.added/done — content is redelivered in output_item.done, so we
            // don't accumulate deltas. Unknown events are ignored.
            _ => {}
        }
    }

    fn observe_response_snapshot(&mut self, event: &Json) {
        let Some(response) = event.get("response") else {
            return;
        };
        if let Json::Object(map) = response {
            self.response = Some(map.clone());
        }
    }

    fn observe_output_item(&mut self, event: &Json) {
        let Some(index) = event.get("output_index").and_then(Json::as_u64) else {
            return;
        };
        let Some(item) = event.get("item") else {
            return;
        };
        self.items.insert(index as usize, item.clone());
    }

    fn finalize(self) -> Json {
        let mut output = self.response.unwrap_or_default();
        // If the latest snapshot lacked `output` (or has an empty array because it came from an
        // early `response.created` event), backfill from per-item accumulator. Terminal events
        // typically carry the complete output, so this branch is a safety net for truncated
        // streams or schemas that drop output from terminal events.
        let snapshot_output_empty = output
            .get("output")
            .and_then(Json::as_array)
            .map(|arr| arr.is_empty())
            .unwrap_or(true);
        if snapshot_output_empty && !self.items.is_empty() {
            let items: Vec<Json> = self.items.into_values().collect();
            output.insert("output".to_string(), Json::Array(items));
        }
        Json::Object(output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/codec/openai_responses_tests.rs"]
mod tests;
