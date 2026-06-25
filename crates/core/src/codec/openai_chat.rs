// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the OpenAI Chat Completions API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the OpenAI Chat Completions format.

use serde::Deserialize;

use crate::api::llm::LlmRequest;
use crate::error::{FlowError, Result};
use crate::json::Json;

use super::request::{AnnotatedLlmRequest, GenerationParams, Message, ToolChoice, ToolDefinition};
use super::resolve::{ProviderSurface, SurfaceDescriptor};
use super::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, FinishReason, RawUsageCost, ResponseToolCall, Usage,
    estimate_cost_for_provider, infer_model_provider, provider_reported_cost,
};
use super::traits::{LlmCodec, LlmResponseCodec};

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Chat Completions API.
pub struct OpenAIChatCodec;

// ---------------------------------------------------------------------------
// Built-in surface descriptor (codec-owned detection, registered in resolve)
// ---------------------------------------------------------------------------

pub(crate) const SURFACE_DESCRIPTOR: SurfaceDescriptor = SurfaceDescriptor {
    surface: ProviderSurface::OpenAIChat,
    detect_request: |obj, _| obj.contains_key("messages"),
    detect_response: |obj| obj.get("choices").is_some_and(Json::is_array),
    decode_request: |request| OpenAIChatCodec.decode(request),
    decode_response: |raw| OpenAIChatCodec.decode_response(raw),
};

// ---------------------------------------------------------------------------
// Private intermediate serde structs for response decode
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawChatCompletion {
    id: Option<String>,
    model: Option<String>,
    choices: Option<Vec<RawChoice>>,
    usage: Option<RawChatUsage>,
    system_fingerprint: Option<String>,
    service_tier: Option<String>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawChoice {
    message: Option<RawMessage>,
    finish_reason: Option<String>,
    logprobs: Option<Json>,
}

#[derive(Deserialize)]
struct RawMessage {
    content: Option<String>,
    tool_calls: Option<Vec<RawToolCall>>,
}

#[derive(Deserialize)]
struct RawToolCall {
    id: Option<String>,
    function: Option<RawFunction>,
}

#[derive(Deserialize)]
struct RawFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct RawChatUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    prompt_tokens_details: Option<RawPromptTokensDetails>,
    #[serde(rename = "cost_usd")]
    provider_cost: Option<f64>,
    cost: Option<RawUsageCost>,
}

#[derive(Deserialize)]
struct RawPromptTokensDetails {
    cached_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Map OpenAI Chat finish_reason string to normalized [`FinishReason`].
fn map_chat_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Complete,
        "length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolUse,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Unknown(other.to_string()),
    }
}

/// Parse OpenAI tool call arguments from JSON string to [`Json`] value.
///
/// Falls back to [`Json::String`] if parsing fails (malformed model output).
fn parse_arguments(arguments: &str) -> Json {
    serde_json::from_str(arguments).unwrap_or_else(|_| Json::String(arguments.to_string()))
}

/// Keys that are modeled in [`AnnotatedLlmRequest`] and should NOT go into `extra`.
const MODELED_REQUEST_KEYS: &[&str] = &[
    "messages",
    "model",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "stop",
    "tools",
    "tool_choice",
    "store",
    "user",
    "metadata",
    "service_tier",
    "parallel_tool_calls",
    "top_logprobs",
    "stream",
];

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for OpenAIChatCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let raw: RawChatCompletion = serde_json::from_value(response.clone())
            .map_err(|e| FlowError::Internal(format!("OpenAI Chat response decode: {e}")))?;

        // Extract first choice (if any).
        let choice = raw.choices.as_ref().and_then(|c| c.first());

        // Map message content.
        let message = choice
            .and_then(|c| c.message.as_ref())
            .and_then(|m| m.content.as_ref())
            .map(|s| super::request::MessageContent::Text(s.clone()));

        // Map tool calls, skipping entries that lack a usable function body.
        // Some providers (proxies, vLLM, NIM) may return partial tool_calls
        // entries where `function` or `function.name` is absent or null.
        let tool_calls = choice
            .and_then(|c| c.message.as_ref())
            .and_then(|m| m.tool_calls.as_ref())
            .map(|tcs| {
                tcs.iter()
                    .filter_map(|tc| {
                        let func = tc.function.as_ref()?;
                        let name = func.name.as_ref()?;
                        Some(ResponseToolCall {
                            id: tc.id.clone().unwrap_or_default(),
                            name: name.clone(),
                            arguments: func
                                .arguments
                                .as_deref()
                                .map(parse_arguments)
                                .unwrap_or(Json::Object(Default::default())),
                        })
                    })
                    .collect::<Vec<_>>()
            });

        // Map finish reason.
        let finish_reason = choice
            .and_then(|c| c.finish_reason.as_deref())
            .map(map_chat_finish_reason);

        // Map usage.
        let model_for_pricing = raw.model.as_deref();
        let model_provider = infer_model_provider("openai", model_for_pricing);
        let usage = raw.usage.map(|u| {
            let mut usage = Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                cache_read_tokens: u.prompt_tokens_details.and_then(|d| d.cached_tokens),
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
        let logprobs = choice.and_then(|c| c.logprobs.clone());
        let api_specific = Some(ApiSpecificResponse::OpenAIChat {
            logprobs,
            system_fingerprint: raw.system_fingerprint,
            service_tier: raw.service_tier,
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

impl LlmCodec for OpenAIChatCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;

        // Extract messages (default to empty vec if absent).
        let messages: Vec<Message> = obj
            .get("messages")
            .map(|v| serde_json::from_value(v.clone()).unwrap_or_default())
            .unwrap_or_default();

        // Extract model.
        let model = obj.get("model").and_then(|v| v.as_str()).map(String::from);

        // Extract generation params.
        let temperature = obj.get("temperature").and_then(|v| v.as_f64());
        let top_p = obj.get("top_p").and_then(|v| v.as_f64());
        let stop = obj
            .get("stop")
            .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok());

        // max_completion_tokens takes priority over max_tokens (newer API key).
        let max_tokens = obj
            .get("max_completion_tokens")
            .and_then(|v| v.as_u64())
            .or_else(|| obj.get("max_tokens").and_then(|v| v.as_u64()));

        let params =
            if temperature.is_some() || max_tokens.is_some() || top_p.is_some() || stop.is_some() {
                Some(GenerationParams {
                    temperature,
                    max_tokens,
                    top_p,
                    stop,
                })
            } else {
                None
            };

        // Extract tools.
        let tools: Option<Vec<ToolDefinition>> = obj
            .get("tools")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| FlowError::Internal(format!("OpenAI Chat tools decode: {e}")))?;

        // Extract tool_choice.
        let tool_choice: Option<ToolChoice> = obj
            .get("tool_choice")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| FlowError::Internal(format!("OpenAI Chat tool_choice decode: {e}")))?;

        // Collect extra fields (keys not in MODELED_REQUEST_KEYS).
        let extra: serde_json::Map<String, Json> = obj
            .iter()
            .filter(|(k, _)| !MODELED_REQUEST_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        Ok(AnnotatedLlmRequest {
            messages,
            model,
            params,
            tools,
            tool_choice,
            store: obj.get("store").and_then(|v| v.as_bool()),
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user: obj.get("user").and_then(|v| v.as_str()).map(String::from),
            metadata: obj.get("metadata").cloned(),
            service_tier: obj
                .get("service_tier")
                .and_then(|v| v.as_str())
                .map(String::from),
            parallel_tool_calls: obj.get("parallel_tool_calls").and_then(|v| v.as_bool()),
            max_output_tokens: None,
            max_tool_calls: None,
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

        insert_serialized(obj, "messages", &annotated.messages, "messages")?;

        if let Some(ref model) = annotated.model {
            obj.insert("model".into(), Json::String(model.clone()));
        }

        if let Some(ref params) = annotated.params {
            overlay_generation_params(obj, params)?;
        }

        if let Some(ref tools) = annotated.tools {
            insert_serialized(obj, "tools", tools, "tools")?;
        }

        if let Some(ref tool_choice) = annotated.tool_choice {
            insert_serialized(obj, "tool_choice", tool_choice, "tool_choice")?;
        }

        if let Some(store) = annotated.store {
            obj.insert("store".into(), Json::Bool(store));
        }
        if let Some(ref user) = annotated.user {
            obj.insert("user".into(), Json::String(user.clone()));
        }
        if let Some(ref metadata) = annotated.metadata {
            obj.insert("metadata".into(), metadata.clone());
        }
        if let Some(ref service_tier) = annotated.service_tier {
            obj.insert("service_tier".into(), Json::String(service_tier.clone()));
        }
        if let Some(parallel_tool_calls) = annotated.parallel_tool_calls {
            obj.insert(
                "parallel_tool_calls".into(),
                Json::Bool(parallel_tool_calls),
            );
        }
        if let Some(top_logprobs) = annotated.top_logprobs {
            obj.insert("top_logprobs".into(), Json::from(top_logprobs));
        }
        if let Some(stream) = annotated.stream {
            obj.insert("stream".into(), Json::Bool(stream));
        }

        for (k, v) in &annotated.extra {
            obj.insert(k.clone(), v.clone());
        }

        // Force `stream_options.include_usage` when the caller did not set it.
        //
        // Rationale: OpenAI-compatible backends only emit the terminal chunk
        // containing `usage` (prompt/completion/total tokens) when this flag
        // is true. Without it, Phoenix spans show `token_count=0` for every
        // LLM call even though the provider knows the real counts. The
        // observability exporter (OpenInference) reads usage off the
        // annotated response, so the flag has to be set at the request level
        // before bytes go on the wire.
        //
        // Guarded on `stream == true` per the OpenAI Chat Completions spec,
        // which restricts `stream_options` to streaming requests. Caller-
        // provided `stream_options` are preserved verbatim (including
        // explicit opt-outs such as `include_usage: false`).
        let is_streaming = obj.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
        if is_streaming && !obj.contains_key("stream_options") {
            obj.insert(
                "stream_options".into(),
                serde_json::json!({"include_usage": true}),
            );
        }

        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

fn insert_serialized<T: serde::Serialize>(
    obj: &mut serde_json::Map<String, Json>,
    key: &str,
    value: &T,
    context: &str,
) -> Result<()> {
    let json = serde_json::to_value(value)
        .map_err(|e| FlowError::Internal(format!("OpenAI Chat {context} encode: {e}")))?;
    obj.insert(key.into(), json);
    Ok(())
}

fn overlay_generation_params(
    obj: &mut serde_json::Map<String, Json>,
    params: &GenerationParams,
) -> Result<()> {
    if let Some(temp) = params.temperature {
        obj.insert("temperature".into(), json_f64(temp));
    }
    if let Some(top_p) = params.top_p {
        obj.insert("top_p".into(), json_f64(top_p));
    }
    if let Some(ref stop) = params.stop {
        insert_serialized(obj, "stop", stop, "stop")?;
    }
    if let Some(max_tokens) = params.max_tokens {
        let key = if obj.contains_key("max_completion_tokens") {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        obj.insert(key.into(), Json::from(max_tokens));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Streaming codec
// ---------------------------------------------------------------------------

/// Streaming counterpart to [`OpenAIChatCodec`].
///
/// Replays the OpenAI Chat Completions SSE chunk sequence into the same JSON shape returned for a
/// non-streaming request (`{id, object, created, model, choices: [{message, finish_reason}],
/// usage}`). Once finalized, the assembled JSON can be fed back through
/// [`OpenAIChatCodec::decode_response`] to produce the canonical
/// [`AnnotatedLlmResponse`].
///
/// # Strategy
///
/// Chat Completions streams untyped SSE chunks of `{choices: [{index, delta: {...},
/// finish_reason: ...}]}`. Each delta may carry a `role` (typically only on the first chunk),
/// incremental `content` text, or partial `tool_calls` whose `function.arguments` stream as a
/// JSON-encoded string fragment-by-fragment. Top-level fields (`id`, `model`, `created`) are
/// repeated on every chunk; we capture them once. Final-chunk `usage` is preserved when emitted
/// (only sent when `stream_options.include_usage` is set on the request).
///
/// The OpenAI `[DONE]` end-of-stream sentinel is dropped by the SSE event decoder before
/// reaching the collector, so this codec never sees it.
///
/// Internal state lives behind `Arc<Mutex<...>>` so the `&self`-produced collector and finalizer
/// closures share access. Each instance is single-use because [`LlmFinalizerFn`] consumes the
/// finalize step.
///
/// [`LlmFinalizerFn`]: crate::api::runtime::LlmFinalizerFn
pub struct OpenAIChatStreamingCodec {
    state: std::sync::Arc<std::sync::Mutex<OpenAIChatStreamingState>>,
}

impl OpenAIChatStreamingCodec {
    /// Creates a fresh streaming codec with empty accumulator state.
    pub fn new() -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(OpenAIChatStreamingState::default())),
        }
    }
}

impl Default for OpenAIChatStreamingCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl super::streaming::StreamingCodec for OpenAIChatStreamingCodec {
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
struct OpenAIChatStreamingState {
    id: Option<String>,
    object: Option<String>,
    created: Option<u64>,
    model: Option<String>,
    /// Per-choice accumulator keyed by `choice.index`. BTreeMap so finalize emits choices in
    /// stable order.
    choices: std::collections::BTreeMap<u64, ChoiceState>,
    /// Top-level usage from the final chunk (when `stream_options.include_usage` is set).
    usage: Option<Json>,
}

#[derive(Debug, Default)]
struct ChoiceState {
    role: Option<String>,
    content: String,
    has_content: bool,
    /// Tool calls keyed by their `index` within the choice. Each tool call's `arguments` is
    /// streamed as a JSON-encoded string accumulated fragment-by-fragment.
    tool_calls: std::collections::BTreeMap<u64, ToolCallState>,
    finish_reason: Option<String>,
}

#[derive(Debug, Default)]
struct ToolCallState {
    id: Option<String>,
    type_: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl OpenAIChatStreamingState {
    fn observe(&mut self, chunk: &Json) {
        // Top-level fields (id, object, created, model) are repeated on every chunk; capture once
        // each so unrelated later chunks can't overwrite the canonical values.
        if self.id.is_none()
            && let Some(id) = chunk.get("id").and_then(Json::as_str)
        {
            self.id = Some(id.to_string());
        }
        if self.object.is_none()
            && let Some(obj) = chunk.get("object").and_then(Json::as_str)
        {
            self.object = Some(obj.to_string());
        }
        if self.created.is_none()
            && let Some(c) = chunk.get("created").and_then(Json::as_u64)
        {
            self.created = Some(c);
        }
        if self.model.is_none()
            && let Some(m) = chunk.get("model").and_then(Json::as_str)
        {
            self.model = Some(m.to_string());
        }
        if let Some(usage) = chunk.get("usage") {
            // Some streams emit `usage: null` on every chunk and the real usage only on the
            // final chunk; only capture non-null usage objects.
            if !usage.is_null() {
                self.usage = Some(usage.clone());
            }
        }
        let Some(choices) = chunk.get("choices").and_then(Json::as_array) else {
            return;
        };
        for choice in choices {
            self.observe_choice(choice);
        }
    }

    fn observe_choice(&mut self, choice: &Json) {
        let index = choice.get("index").and_then(Json::as_u64).unwrap_or(0);
        let entry = self.choices.entry(index).or_default();
        entry.observe_finish_reason(choice);
        entry.observe_delta(choice.get("delta"));
    }

    fn finalize(self) -> Json {
        let mut output = serde_json::Map::new();
        if let Some(id) = self.id {
            output.insert("id".to_string(), Json::String(id));
        }
        // After streaming, the final shape is `chat.completion`, not `chat.completion.chunk`.
        // Strip the `.chunk` suffix so the assembled JSON round-trips through
        // OpenAIChatCodec::decode_response with the same `object` field a non-streaming response
        // would carry.
        if let Some(object) = self.object {
            let normalized = object
                .strip_suffix(".chunk")
                .map(str::to_string)
                .unwrap_or(object);
            output.insert("object".to_string(), Json::String(normalized));
        }
        if let Some(created) = self.created {
            output.insert("created".to_string(), Json::Number(created.into()));
        }
        if let Some(model) = self.model {
            output.insert("model".to_string(), Json::String(model));
        }
        let choices: Vec<Json> = self
            .choices
            .into_iter()
            .map(|(index, choice)| choice.finalize(index))
            .collect();
        output.insert("choices".to_string(), Json::Array(choices));
        if let Some(usage) = self.usage {
            output.insert("usage".to_string(), usage);
        }
        Json::Object(output)
    }
}

impl ChoiceState {
    fn observe_finish_reason(&mut self, choice: &Json) {
        if let Some(reason) = choice.get("finish_reason").and_then(Json::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
    }

    fn observe_delta(&mut self, delta: Option<&Json>) {
        let Some(delta) = delta else {
            return;
        };
        if let Some(role) = delta.get("role").and_then(Json::as_str) {
            self.role = Some(role.to_string());
        }
        if let Some(content) = delta.get("content").and_then(Json::as_str) {
            self.content.push_str(content);
            self.has_content = true;
        }
        self.observe_tool_calls(delta);
    }

    fn observe_tool_calls(&mut self, delta: &Json) {
        if let Some(tool_calls) = delta.get("tool_calls").and_then(Json::as_array) {
            for tool_call in tool_calls {
                self.observe_tool_call(tool_call);
            }
        }
    }

    fn observe_tool_call(&mut self, tool_call: &Json) {
        let index = tool_call.get("index").and_then(Json::as_u64).unwrap_or(0);
        let state = self.tool_calls.entry(index).or_default();
        if let Some(id) = tool_call.get("id").and_then(Json::as_str) {
            state.id = Some(id.to_string());
        }
        if let Some(type_) = tool_call.get("type").and_then(Json::as_str) {
            state.type_ = Some(type_.to_string());
        }
        if let Some(function) = tool_call.get("function") {
            state.observe_function(function);
        }
    }

    fn finalize(self, index: u64) -> Json {
        let mut message = serde_json::Map::new();
        message.insert(
            "role".to_string(),
            Json::String(self.role.unwrap_or_else(|| "assistant".to_string())),
        );
        // OpenAI's wire format uses `content: null` when the model only emitted tool calls.
        // Preserve that distinction: empty-string content when the model said something, null
        // when it didn't.
        if self.has_content {
            message.insert("content".to_string(), Json::String(self.content));
        } else {
            message.insert("content".to_string(), Json::Null);
        }
        if !self.tool_calls.is_empty() {
            let tool_calls: Vec<Json> = self
                .tool_calls
                .into_values()
                .map(ToolCallState::finalize)
                .collect();
            message.insert("tool_calls".to_string(), Json::Array(tool_calls));
        }
        let mut choice = serde_json::Map::new();
        choice.insert("index".to_string(), Json::Number(index.into()));
        choice.insert("message".to_string(), Json::Object(message));
        if let Some(reason) = self.finish_reason {
            choice.insert("finish_reason".to_string(), Json::String(reason));
        } else {
            choice.insert("finish_reason".to_string(), Json::Null);
        }
        Json::Object(choice)
    }
}

impl ToolCallState {
    fn observe_function(&mut self, function: &Json) {
        if let Some(name) = function.get("name").and_then(Json::as_str) {
            self.name = Some(name.to_string());
        }
        if let Some(args) = function.get("arguments").and_then(Json::as_str) {
            self.arguments.push_str(args);
        }
    }

    fn finalize(self) -> Json {
        let mut function = serde_json::Map::new();
        function.insert(
            "name".to_string(),
            Json::String(self.name.unwrap_or_default()),
        );
        function.insert("arguments".to_string(), Json::String(self.arguments));
        let mut call = serde_json::Map::new();
        if let Some(id) = self.id {
            call.insert("id".to_string(), Json::String(id));
        }
        call.insert(
            "type".to_string(),
            Json::String(self.type_.unwrap_or_else(|| "function".to_string())),
        );
        call.insert("function".to_string(), Json::Object(function));
        Json::Object(call)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/codec/openai_chat_tests.rs"]
mod tests;
