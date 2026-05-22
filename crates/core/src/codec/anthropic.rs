// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the Anthropic Messages API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the Anthropic Messages API format.
//!
//! # Anthropic-specific patterns handled
//!
//! - **Content blocks**: Heterogeneous array of `text`, `tool_use`, `thinking`,
//!   `redacted_thinking`, `mcp_tool_use`, `server_tool_use` blocks
//! - **Top-level system**: System prompt is a top-level field, not inside messages
//! - **stop_reason**: Maps to [`FinishReason`] (not `finish_reason`)
//! - **Tool definitions**: Uses `input_schema` instead of `parameters`
//! - **Tool choice**: `{"type":"auto"}` / `{"type":"any"}` / `{"type":"tool","name":"..."}`
//! - **Cache tokens**: `cache_read_input_tokens` / `cache_creation_input_tokens`

use serde::Deserialize;

use crate::api::llm::LlmRequest;
use crate::error::{FlowError, Result};
use crate::json::Json;

use super::request::{
    AnnotatedLlmRequest, FunctionDefinition, GenerationParams, Message, MessageContent, ToolChoice,
    ToolChoiceFunction, ToolChoiceFunctionName, ToolDefinition,
};
use super::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, FinishReason, ResponseToolCall, Usage,
};
use super::traits::{LlmCodec, LlmResponseCodec};

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the Anthropic Messages API.
pub struct AnthropicMessagesCodec;

// ---------------------------------------------------------------------------
// Private intermediate serde structs for response decode
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawAnthropicResponse {
    id: Option<String>,
    #[serde(rename = "type")]
    object_type: Option<String>,
    role: Option<String>,
    model: Option<String>,
    content: Option<Vec<Json>>,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    service_tier: Option<String>,
    container: Option<Json>,
    usage: Option<RawAnthropicUsage>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawAnthropicUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Map Anthropic `stop_reason` string to normalized [`FinishReason`].
fn map_anthropic_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" => FinishReason::Complete,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolUse,
        other => FinishReason::Unknown(other.to_string()),
    }
}

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

/// Keys that are modeled in [`AnnotatedLlmRequest`] and should NOT go into `extra`.
const MODELED_REQUEST_KEYS: &[&str] = &[
    "system",
    "messages",
    "model",
    "max_tokens",
    "temperature",
    "top_p",
    "stop_sequences",
    "tools",
    "tool_choice",
    "metadata",
    "service_tier",
];

/// Decode the Anthropic `tool_choice` JSON value into a normalized [`ToolChoice`].
///
/// Anthropic format:
/// - `{"type": "auto"}` -> `ToolChoice::Auto`
/// - `{"type": "any"}` -> `ToolChoice::Required`
/// - `{"type": "none"}` -> `ToolChoice::None`
/// - `{"type": "tool", "name": "X"}` -> `ToolChoice::Specific`
fn decode_anthropic_tool_choice(val: &Json) -> Option<ToolChoice> {
    let obj = val.as_object()?;
    let tc_type = obj.get("type")?.as_str()?;
    match tc_type {
        "auto" => Some(ToolChoice::Auto),
        "any" => Some(ToolChoice::Required),
        "none" => Some(ToolChoice::None),
        "tool" => {
            let name = obj.get("name")?.as_str()?.to_string();
            Some(ToolChoice::Specific(ToolChoiceFunction {
                choice_type: "function".into(),
                function: ToolChoiceFunctionName { name },
            }))
        }
        _ => None,
    }
}

/// Extract Anthropic `disable_parallel_tool_use` from tool_choice and map
/// to normalized `parallel_tool_calls` semantics.
fn decode_parallel_tool_calls(val: &Json) -> Option<bool> {
    let obj = val.as_object()?;
    obj.get("disable_parallel_tool_use")
        .and_then(|v| v.as_bool())
        .map(|disabled| !disabled)
}

/// Encode a normalized [`ToolChoice`] back into Anthropic JSON format.
fn encode_anthropic_tool_choice(tc: &ToolChoice) -> Json {
    match tc {
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::Required => serde_json::json!({"type": "any"}),
        ToolChoice::None => serde_json::json!({"type": "none"}),
        ToolChoice::Specific(func) => {
            serde_json::json!({"type": "tool", "name": func.function.name})
        }
    }
}

fn encode_tool_choice_with_parallel_hint(
    tc: &ToolChoice,
    parallel_tool_calls: Option<bool>,
) -> Json {
    let mut value = encode_anthropic_tool_choice(tc);
    if let (Some(parallel), Some(obj)) = (parallel_tool_calls, value.as_object_mut()) {
        obj.insert("disable_parallel_tool_use".into(), Json::Bool(!parallel));
    }
    value
}

/// Extract the system prompt from an Anthropic top-level `system` field.
///
/// Handles both string and array-of-content-blocks formats.
fn extract_system_message(system_val: &Json) -> Option<Message> {
    if let Some(s) = system_val.as_str() {
        Some(Message::System {
            content: MessageContent::Text(s.to_string()),
            name: None,
        })
    } else if let Some(arr) = system_val.as_array() {
        // Array of content blocks -- extract text from each "text" block.
        let texts: Vec<&str> = arr
            .iter()
            .filter_map(|block| {
                let block_type = block.get("type")?.as_str()?;
                if block_type == "text" {
                    block.get("text")?.as_str()
                } else {
                    None
                }
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(Message::System {
                content: MessageContent::Text(texts.join("\n")),
                name: None,
            })
        }
    } else {
        None
    }
}

/// Extract system text from a [`Message::System`] for encoding back to top-level.
fn extract_system_text(msg: &Message) -> Option<String> {
    match msg {
        Message::System {
            content: MessageContent::Text(s),
            ..
        } => Some(s.clone()),
        Message::System {
            content: MessageContent::Parts(parts),
            ..
        } => {
            let texts: Vec<&str> = parts
                .iter()
                .filter_map(|p| match p {
                    super::request::ContentPart::Text { text } => Some(text.as_str()),
                    super::request::ContentPart::ImageUrl { .. } => None,
                })
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n"))
            }
        }
        _ => None,
    }
}

fn split_system_and_messages(messages: &[Message]) -> (Option<String>, Vec<&Message>) {
    let mut system_text = None;
    let mut non_system_messages = Vec::new();

    for msg in messages {
        if let Some(text) = extract_system_text(msg) {
            system_text = Some(text);
        } else {
            non_system_messages.push(msg);
        }
    }

    (system_text, non_system_messages)
}

fn insert_serialized<T: serde::Serialize>(
    obj: &mut serde_json::Map<String, Json>,
    key: &str,
    value: &T,
    context: &str,
) -> Result<()> {
    let json = serde_json::to_value(value)
        .map_err(|e| FlowError::Internal(format!("Anthropic Messages {context} encode: {e}")))?;
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
        obj.insert("max_tokens".into(), Json::from(max_tokens));
    }
}

fn encode_anthropic_tools(tools: &[ToolDefinition]) -> Vec<Json> {
    tools
        .iter()
        .map(|td| {
            let mut tool = serde_json::Map::new();
            tool.insert("name".into(), Json::String(td.function.name.clone()));
            if let Some(ref desc) = td.function.description {
                tool.insert("description".into(), Json::String(desc.clone()));
            }
            if let Some(ref params) = td.function.parameters {
                tool.insert("input_schema".into(), params.clone());
            }
            Json::Object(tool)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for AnthropicMessagesCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let raw: RawAnthropicResponse = serde_json::from_value(response.clone())
            .map_err(|e| FlowError::Internal(format!("Anthropic Messages response decode: {e}")))?;

        // Process content blocks.
        let content_blocks = raw.content.as_ref();

        // Extract text from all "text" blocks, concatenated with newline.
        let text_parts: Vec<&str> = content_blocks
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|block| {
                        let block_type = block.get("type")?.as_str()?;
                        if block_type == "text" {
                            block.get("text")?.as_str()
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let message = if text_parts.is_empty() {
            None
        } else {
            Some(MessageContent::Text(text_parts.join("\n")))
        };

        // Extract tool_use blocks (only "tool_use" type, NOT mcp_tool_use or server_tool_use).
        let tool_calls: Vec<ResponseToolCall> = content_blocks
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|block| {
                        let block_type = block.get("type")?.as_str()?;
                        if block_type == "tool_use" {
                            let id = block.get("id")?.as_str()?.to_string();
                            let name = block.get("name")?.as_str()?.to_string();
                            // CRITICAL: input is already parsed JSON -- clone directly.
                            let arguments = block.get("input")?.clone();
                            Some(ResponseToolCall {
                                id,
                                name,
                                arguments,
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let tool_calls = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        };

        // Map stop_reason to FinishReason.
        let finish_reason = raw.stop_reason.as_deref().map(map_anthropic_stop_reason);

        // Map usage.
        let usage = raw.usage.map(|u| {
            let prompt = u.input_tokens;
            let completion = u.output_tokens;
            Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                // Anthropic does not supply total_tokens; compute it.
                total_tokens: match (prompt, completion) {
                    (Some(p), Some(c)) => Some(p + c),
                    _ => None,
                },
                cache_read_tokens: u.cache_read_input_tokens,
                cache_write_tokens: u.cache_creation_input_tokens,
            }
        });

        // Build API-specific fields: all content blocks + stop_sequence.
        let api_specific_content_blocks = raw.content.clone();
        let api_specific = Some(ApiSpecificResponse::AnthropicMessages {
            object_type: raw.object_type,
            role: raw.role,
            stop_reason: raw.stop_reason,
            stop_sequence: raw.stop_sequence,
            service_tier: raw.service_tier,
            container: raw.container,
            content_blocks: api_specific_content_blocks,
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

impl LlmCodec for AnthropicMessagesCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;

        // Extract system from top-level field.
        let system_msg = obj.get("system").and_then(extract_system_message);

        // Extract messages (default to empty vec if absent).
        let mut messages: Vec<Message> = obj
            .get("messages")
            .map(|v| serde_json::from_value(v.clone()).unwrap_or_default())
            .unwrap_or_default();

        // Prepend system message if present.
        if let Some(sys) = system_msg {
            messages.insert(0, sys);
        }

        // Extract model.
        let model = obj.get("model").and_then(|v| v.as_str()).map(String::from);

        // Extract generation params.
        let temperature = obj.get("temperature").and_then(|v| v.as_f64());
        let top_p = obj.get("top_p").and_then(|v| v.as_f64());
        let max_tokens = obj.get("max_tokens").and_then(|v| v.as_u64());
        // Anthropic uses stop_sequences (not stop).
        let stop = obj
            .get("stop_sequences")
            .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok());

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

        // Extract tools: Anthropic uses flat structure (name, description, input_schema).
        // Normalize to ToolDefinition { type: "function", function: { name, description, parameters } }.
        let tools: Option<Vec<ToolDefinition>> = obj.get("tools").and_then(|v| {
            let arr = v.as_array()?;
            let defs: Vec<ToolDefinition> = arr
                .iter()
                .filter_map(|tool| {
                    let name = tool.get("name")?.as_str()?.to_string();
                    let description = tool
                        .get("description")
                        .and_then(|d| d.as_str())
                        .map(String::from);
                    let parameters = tool.get("input_schema").cloned();
                    Some(ToolDefinition {
                        tool_type: "function".into(),
                        function: FunctionDefinition {
                            name,
                            description,
                            parameters,
                        },
                    })
                })
                .collect();
            if defs.is_empty() { None } else { Some(defs) }
        });

        // Extract tool_choice: Anthropic format.
        let tool_choice = obj
            .get("tool_choice")
            .and_then(decode_anthropic_tool_choice);
        let parallel_tool_calls = obj.get("tool_choice").and_then(decode_parallel_tool_calls);

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
            store: None,
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user: None,
            metadata: obj.get("metadata").cloned(),
            service_tier: obj
                .get("service_tier")
                .and_then(|v| v.as_str())
                .map(String::from),
            parallel_tool_calls,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            stream: None,
            extra,
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        let mut content = original.content.clone();
        let obj = content
            .as_object_mut()
            .ok_or_else(|| FlowError::Internal("original content is not an object".into()))?;

        let (system_text, non_system_messages) = split_system_and_messages(&annotated.messages);

        if let Some(text) = system_text {
            obj.insert("system".into(), Json::String(text));
        }

        // Overlay messages (non-system only).
        insert_serialized(obj, "messages", &non_system_messages, "messages")?;

        // Overlay model if present.
        if let Some(ref model) = annotated.model {
            obj.insert("model".into(), Json::String(model.clone()));
        }

        // Overlay generation params.
        if let Some(ref params) = annotated.params {
            overlay_generation_params(obj, params);
            // Write stop_sequences (Anthropic key name, not "stop").
            if let Some(ref stop) = params.stop {
                insert_serialized(obj, "stop_sequences", stop, "stop_sequences")?;
            }
        }

        // Overlay tools in Anthropic format: { name, description, input_schema }.
        // Denormalize from ToolDefinition (drop type/function wrapper, rename parameters -> input_schema).
        if let Some(ref tools) = annotated.tools {
            let anthropic_tools = encode_anthropic_tools(tools);
            insert_serialized(obj, "tools", &anthropic_tools, "tools")?;
        }

        // Overlay tool_choice in Anthropic format.
        if let Some(ref tool_choice) = annotated.tool_choice {
            obj.insert(
                "tool_choice".into(),
                encode_tool_choice_with_parallel_hint(tool_choice, annotated.parallel_tool_calls),
            );
        }

        if let Some(ref metadata) = annotated.metadata {
            obj.insert("metadata".into(), metadata.clone());
        }
        if let Some(ref service_tier) = annotated.service_tier {
            obj.insert("service_tier".into(), Json::String(service_tier.clone()));
        }

        // Merge extra fields back.
        for (k, v) in &annotated.extra {
            obj.insert(k.clone(), v.clone());
        }

        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

// ---------------------------------------------------------------------------
// Streaming codec
// ---------------------------------------------------------------------------

/// Streaming counterpart to [`AnthropicMessagesCodec`].
///
/// Replays the Anthropic Messages SSE event sequence into the same JSON shape Anthropic returns
/// for a non-streaming request (`{id, type, role, model, content, stop_reason, stop_sequence,
/// usage}`). Once finalized, the assembled JSON can be fed back through
/// [`AnthropicMessagesCodec::decode_response`] to produce an
/// [`AnnotatedLlmResponse`] — meaning streaming and
/// non-streaming Anthropic requests converge on the same observability output.
///
/// Internal state lives behind `Arc<Mutex<...>>` so the `&self`-produced collector and finalizer
/// closures share access. Each instance is single-use because [`LlmFinalizerFn`] consumes the
/// finalize step.
///
/// [`LlmFinalizerFn`]: crate::api::runtime::LlmFinalizerFn
pub struct AnthropicMessagesStreamingCodec {
    state: std::sync::Arc<std::sync::Mutex<AnthropicMessagesStreamingState>>,
}

impl AnthropicMessagesStreamingCodec {
    /// Creates a fresh streaming codec with empty accumulator state.
    pub fn new() -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(
                AnthropicMessagesStreamingState::default(),
            )),
        }
    }
}

impl Default for AnthropicMessagesStreamingCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl super::streaming::StreamingCodec for AnthropicMessagesStreamingCodec {
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
            // Move state out so finalize can consume it; the codec is single-use, so leaving a
            // default behind is intentional and never observed by another caller.
            std::mem::take(&mut *guard).finalize()
        })
    }
}

#[derive(Debug, Default)]
struct AnthropicMessagesStreamingState {
    id: Option<String>,
    type_: Option<String>,
    role: Option<String>,
    model: Option<String>,
    /// Latest usage snapshot. `message_start` carries an initial value (input tokens, zero output
    /// so far); `message_delta` updates it cumulatively. Last write wins.
    usage: Option<Json>,
    stop_reason: Option<String>,
    /// Stored as raw `Json` to preserve `null` (Anthropic's wire shape) versus omitted.
    stop_sequence: Option<Json>,
    /// Indexed by the SSE event's `index` field. `None` slots accommodate sparse indices though
    /// Anthropic emits them in order today.
    blocks: Vec<Option<StreamingBlock>>,
}

#[derive(Debug, Default, Clone)]
struct StreamingBlock {
    /// The `content_block` JSON captured at `content_block_start`. Deltas mutate fields directly
    /// for blocks Anthropic delivers incrementally (text, tool_use input, citations); other block
    /// types (server_tool_use results) ship complete at start and pass through unchanged.
    skeleton: serde_json::Map<String, Json>,
    text: String,
    has_text: bool,
    partial_json: String,
    has_partial_json: bool,
    citations: Vec<Json>,
    has_citations: bool,
}

impl AnthropicMessagesStreamingState {
    fn observe(&mut self, event: &Json) {
        let event_type = event.get("type").and_then(Json::as_str).unwrap_or("");
        match event_type {
            "message_start" => self.observe_message_start(event),
            "content_block_start" => self.observe_content_block_start(event),
            "content_block_delta" => self.observe_content_block_delta(event),
            "message_delta" => self.observe_message_delta(event),
            // content_block_stop, message_stop, ping, and any unknown event type carry no
            // accumulator-relevant payload. Unknown types are ignored rather than erroring so a
            // future Anthropic event addition does not break observability.
            _ => {}
        }
    }

    fn observe_message_start(&mut self, event: &Json) {
        let Some(message) = event.get("message") else {
            return;
        };
        if let Some(id) = message.get("id").and_then(Json::as_str) {
            self.id = Some(id.to_string());
        }
        if let Some(model) = message.get("model").and_then(Json::as_str) {
            self.model = Some(model.to_string());
        }
        if let Some(role) = message.get("role").and_then(Json::as_str) {
            self.role = Some(role.to_string());
        }
        if let Some(t) = message.get("type").and_then(Json::as_str) {
            self.type_ = Some(t.to_string());
        }
        if let Some(usage) = message.get("usage") {
            self.usage = Some(usage.clone());
        }
    }

    fn observe_content_block_start(&mut self, event: &Json) {
        let Some(index) = event.get("index").and_then(Json::as_u64) else {
            return;
        };
        let Some(content_block) = event.get("content_block") else {
            return;
        };
        let skeleton = match content_block {
            Json::Object(map) => map.clone(),
            _ => return,
        };
        let index = index as usize;
        while self.blocks.len() <= index {
            self.blocks.push(None);
        }
        self.blocks[index] = Some(StreamingBlock {
            skeleton,
            ..StreamingBlock::default()
        });
    }

    fn observe_content_block_delta(&mut self, event: &Json) {
        let Some(index) = event.get("index").and_then(Json::as_u64) else {
            return;
        };
        let index = index as usize;
        let Some(delta) = event.get("delta") else {
            return;
        };
        let delta_type = delta.get("type").and_then(Json::as_str).unwrap_or("");
        let Some(slot) = self.blocks.get_mut(index) else {
            return;
        };
        let Some(block) = slot.as_mut() else { return };
        match delta_type {
            "text_delta" => {
                if let Some(text) = delta.get("text").and_then(Json::as_str) {
                    block.text.push_str(text);
                    block.has_text = true;
                }
            }
            "input_json_delta" => {
                if let Some(partial) = delta.get("partial_json").and_then(Json::as_str) {
                    block.partial_json.push_str(partial);
                    block.has_partial_json = true;
                }
            }
            "citations_delta" => {
                if let Some(citation) = delta.get("citation") {
                    block.citations.push(citation.clone());
                    block.has_citations = true;
                }
            }
            // thinking_delta, signature_delta, and any future delta types fall through; the block
            // skeleton retains whatever shape was set at content_block_start.
            _ => {}
        }
    }

    fn observe_message_delta(&mut self, event: &Json) {
        if let Some(delta) = event.get("delta") {
            if let Some(reason) = delta.get("stop_reason").and_then(Json::as_str) {
                self.stop_reason = Some(reason.to_string());
            }
            if let Some(seq) = delta.get("stop_sequence") {
                self.stop_sequence = Some(seq.clone());
            }
        }
        if let Some(usage) = event.get("usage") {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> Json {
        let mut output = serde_json::Map::new();
        if let Some(id) = self.id {
            output.insert("id".to_string(), Json::String(id));
        }
        if let Some(t) = self.type_ {
            output.insert("type".to_string(), Json::String(t));
        }
        if let Some(role) = self.role {
            output.insert("role".to_string(), Json::String(role));
        }
        if let Some(model) = self.model {
            output.insert("model".to_string(), Json::String(model));
        }
        let content: Vec<Json> = self
            .blocks
            .into_iter()
            .filter_map(|block| block.map(StreamingBlock::finalize))
            .collect();
        output.insert("content".to_string(), Json::Array(content));
        if let Some(reason) = self.stop_reason {
            output.insert("stop_reason".to_string(), Json::String(reason));
        }
        if let Some(seq) = self.stop_sequence {
            output.insert("stop_sequence".to_string(), seq);
        }
        if let Some(usage) = self.usage {
            output.insert("usage".to_string(), usage);
        }
        Json::Object(output)
    }
}

impl StreamingBlock {
    fn finalize(mut self) -> Json {
        if self.has_text {
            self.skeleton
                .insert("text".to_string(), Json::String(self.text));
        }
        if self.has_partial_json {
            // Concatenated `partial_json` fragments are expected to parse as a JSON object — that's
            // the assembled tool input. If parsing fails (Anthropic emits malformed deltas, stream
            // truncated mid-block), surface the raw concatenation so observability still captures
            // something rather than dropping the call.
            let parsed = match serde_json::from_str::<Json>(&self.partial_json) {
                Ok(value) => value,
                Err(_) => Json::String(self.partial_json),
            };
            self.skeleton.insert("input".to_string(), parsed);
        }
        if self.has_citations {
            self.skeleton
                .insert("citations".to_string(), Json::Array(self.citations));
        }
        Json::Object(self.skeleton)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/codec/anthropic_tests.rs"]
mod tests;
