// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM request codec types and trait.
//!
//! This module defines the [`AnnotatedLlmRequest`] type system for structured
//! LLM request representation and the [`crate::codec::traits::LlmCodec`] trait
//! for bidirectional translation between opaque [`crate::api::llm::LlmRequest`]
//! payloads and typed form.

use serde::{Deserialize, Serialize};

use crate::Json;

// ---------------------------------------------------------------------------
// AnnotatedLlmRequest type hierarchy
// ---------------------------------------------------------------------------

/// Structured view of an LLM request, produced by a Codec from opaque
/// [`LlmRequest`](crate::api::llm::LlmRequest) content.
///
/// The `extra` field captures unknown future top-level keys. Modeled
/// provider-specific controls belong in [`AnnotatedLlmRequest::api_specific`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedLlmRequest {
    /// Parsed conversation messages.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Provider-level instructions that are not part of the conversation array.
    ///
    /// Anthropic encodes this as `system`; OpenAI Responses uses `instructions`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<MessageContent>,
    /// Model identifier (e.g., `"gpt-4"`, `"claude-sonnet-4-20250514"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Common generation parameters, normalized.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<GenerationParams>,
    /// Tool definitions (function schemas) available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Tool choice control.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// OpenAI Responses: whether to persist response state server-side.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// OpenAI Responses: prior response to continue from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// OpenAI Responses: context truncation behavior.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Json>,
    /// OpenAI Responses: reasoning configuration object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Json>,
    /// OpenAI Responses: include filter for additional output/state items.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Json>,
    /// OpenAI user identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// OpenAI metadata map/object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Json>,
    /// OpenAI service tier preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// OpenAI tool parallelism toggle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// OpenAI Responses max output token limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// OpenAI Responses max tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u64>,
    /// OpenAI logprob fanout count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u64>,
    /// OpenAI streaming toggle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// API-specific request data that does not have portable semantics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_specific: Option<ApiSpecificRequest>,
    /// Unknown future top-level fields.
    ///
    /// Baseline-aware codecs remove deleted keys and overlay changed or added
    /// keys without rebuilding untouched provider JSON.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Json>,
}

/// A single message in a conversation, tagged by role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// A system instruction message.
    System {
        /// The message content.
        content: MessageContent,
        /// Optional sender name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// A user message.
    User {
        /// The message content.
        content: MessageContent,
        /// Optional sender name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// A developer instruction message used by OpenAI APIs.
    Developer {
        /// The message content.
        content: MessageContent,
        /// Optional sender name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// An assistant response, optionally containing tool calls.
    Assistant {
        /// The message content (optional — may be absent when tool calls are present).
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// Tool calls requested by the assistant.
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
        /// Optional sender name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// A tool result message.
    Tool {
        /// The tool execution result.
        content: MessageContent,
        /// The ID of the tool call this result corresponds to.
        tool_call_id: String,
    },
    /// A legacy OpenAI function-result message.
    Function {
        /// The function result. OpenAI permits an explicit null value.
        content: Option<String>,
        /// Function name.
        name: String,
    },
    /// A portable top-level tool-call item, primarily used by OpenAI Responses.
    #[serde(rename = "tool_call")]
    ToolCallItem {
        /// Optional provider item ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Provider call ID used to correlate the result.
        call_id: String,
        /// Tool/function name.
        name: String,
        /// Parsed tool arguments.
        arguments: Json,
        /// Provider fields without portable semantics.
        #[serde(default, flatten)]
        extra: serde_json::Map<String, Json>,
    },
    /// A portable top-level tool-result item, primarily used by OpenAI Responses.
    #[serde(rename = "tool_result")]
    ToolResultItem {
        /// Optional provider item ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Provider call ID used to correlate the result.
        call_id: String,
        /// Provider result payload.
        output: Json,
        /// Provider fields without portable semantics.
        #[serde(default, flatten)]
        extra: serde_json::Map<String, Json>,
    },
    /// Lossless provider-native request item.
    #[serde(rename = "provider_native")]
    ProviderNative {
        /// Provider surface that owns the value.
        provider: String,
        /// Native discriminator or a descriptive fallback.
        kind: String,
        /// Exact provider JSON value.
        value: Json,
    },
}

/// Message content: either a plain string or multimodal parts array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Multimodal content parts.
    Parts(Vec<ContentPart>),
}

/// A single content part within a multimodal message.
///
/// v1 supports text only. Future versions may add `ImageUrl`, `Audio`, etc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A text content part.
    Text {
        /// The text content.
        text: String,
        /// Provider fields without portable semantics.
        #[serde(default, flatten)]
        extra: serde_json::Map<String, Json>,
    },
    /// An image URL content part.
    ImageUrl {
        /// Image URL payload.
        image_url: OpenAiImageUrl,
        /// Provider fields without portable semantics.
        #[serde(default, flatten)]
        extra: serde_json::Map<String, Json>,
    },
    /// Portable image input payload.
    Image {
        /// Provider-neutral image data object.
        image: Json,
        /// Provider fields without portable semantics.
        #[serde(default, flatten)]
        extra: serde_json::Map<String, Json>,
    },
    /// Portable audio input payload.
    Audio {
        /// Provider-neutral audio data object.
        audio: Json,
        /// Provider fields without portable semantics.
        #[serde(default, flatten)]
        extra: serde_json::Map<String, Json>,
    },
    /// Portable file or document input payload.
    File {
        /// Provider-neutral file data object.
        file: Json,
        /// Provider fields without portable semantics.
        #[serde(default, flatten)]
        extra: serde_json::Map<String, Json>,
    },
    /// Assistant refusal content.
    Refusal {
        /// Refusal text.
        refusal: String,
        /// Provider fields without portable semantics.
        #[serde(default, flatten)]
        extra: serde_json::Map<String, Json>,
    },
    /// Tool call embedded in a provider content-block array.
    ToolUse {
        /// Tool call identifier.
        id: String,
        /// Tool name.
        name: String,
        /// Parsed arguments.
        input: Json,
        /// Provider fields without portable semantics.
        #[serde(default, flatten)]
        extra: serde_json::Map<String, Json>,
    },
    /// Tool result embedded in a provider content-block array.
    ToolResult {
        /// Tool call identifier.
        tool_use_id: String,
        /// Tool result payload.
        content: Json,
        /// Whether the tool failed.
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        /// Provider fields without portable semantics.
        #[serde(default, flatten)]
        extra: serde_json::Map<String, Json>,
    },
    /// Lossless provider-native content block.
    ProviderNative {
        /// Provider surface that owns the value.
        provider: String,
        /// Native block discriminator or a descriptive fallback.
        kind: String,
        /// Exact provider JSON value.
        value: Json,
    },
}

/// OpenAI image URL payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiImageUrl {
    /// URL for the image.
    pub url: String,
    /// Optional provider-specific detail hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A tool call requested by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// The type of tool call (typically `"function"`).
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function to call.
    pub function: FunctionCall,
}

/// A function call within a tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The name of the function to call.
    pub name: String,
    /// The function arguments as a JSON string (per OpenAI convention).
    pub arguments: String,
}

/// A tool definition available to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDefinition {
    /// Portable function tool.
    Function {
        /// The function definition.
        function: FunctionDefinition,
        /// Provider fields on the function-tool wrapper.
        #[serde(default, flatten)]
        extra: serde_json::Map<String, Json>,
    },
    /// Lossless provider-native tool definition.
    ProviderNative {
        /// Provider surface that owns the value.
        provider: String,
        /// Native tool discriminator or a descriptive fallback.
        kind: String,
        /// Exact provider JSON value.
        value: Json,
    },
}

/// A function definition within a tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// The name of the function.
    pub name: String,
    /// A description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON Schema for the function parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Json>,
    /// Whether the provider should enforce the parameter schema strictly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// Provider fields without portable semantics.
    #[serde(default, flatten)]
    pub extra: serde_json::Map<String, Json>,
}

/// Tool choice control: how the model should use available tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    /// Let the model decide whether to call a tool.
    Auto,
    /// Do not call any tools.
    None,
    /// The model must call at least one tool.
    Required,
    /// Force a specific function by name.
    #[serde(untagged)]
    Specific(ToolChoiceFunction),
    /// Lossless provider-native tool choice.
    #[serde(untagged)]
    ProviderNative(ProviderNativeComponent),
}

/// Lossless provider-native component embedded in the annotated request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderNativeComponent {
    /// Provider surface that owns the value.
    pub provider: String,
    /// Native discriminator or a descriptive fallback.
    pub kind: String,
    /// Exact provider JSON value.
    pub value: Json,
}

/// API-specific request fields that do not have portable semantics.
#[allow(
    clippy::large_enum_variant,
    reason = "provider wire-schema fields stay directly mutable on each public variant"
)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "api")]
pub enum ApiSpecificRequest {
    /// Anthropic Messages-specific request fields.
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages {
        /// Top-level prompt cache control.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<Json>,
        /// Reusable container identifier.
        #[serde(skip_serializing_if = "Option::is_none")]
        container: Option<String>,
        /// Requested inference geography.
        #[serde(skip_serializing_if = "Option::is_none")]
        inference_geo: Option<String>,
        /// Provider output configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        output_config: Option<Json>,
        /// Extended-thinking configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking: Option<Json>,
        /// Top-k sampling limit.
        #[serde(skip_serializing_if = "Option::is_none")]
        top_k: Option<u64>,
        /// User profile attribution identifier.
        #[serde(skip_serializing_if = "Option::is_none")]
        user_profile_id: Option<String>,
    },
    /// OpenAI Chat Completions-specific request fields.
    #[serde(rename = "openai_chat")]
    OpenAIChat {
        /// Audio output configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<Json>,
        /// Frequency penalty.
        #[serde(skip_serializing_if = "Option::is_none")]
        frequency_penalty: Option<f64>,
        /// Deprecated function-call control.
        #[serde(skip_serializing_if = "Option::is_none")]
        function_call: Option<Json>,
        /// Deprecated function definitions.
        #[serde(skip_serializing_if = "Option::is_none")]
        functions: Option<Vec<Json>>,
        /// Token logit bias map.
        #[serde(skip_serializing_if = "Option::is_none")]
        logit_bias: Option<Json>,
        /// Whether token log probabilities are requested.
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<bool>,
        /// Requested output modalities.
        #[serde(skip_serializing_if = "Option::is_none")]
        modalities: Option<Vec<String>>,
        /// Request moderation configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        moderation: Option<Json>,
        /// Number of completion choices.
        #[serde(skip_serializing_if = "Option::is_none")]
        n: Option<u64>,
        /// Predicted output content.
        #[serde(skip_serializing_if = "Option::is_none")]
        prediction: Option<Json>,
        /// Presence penalty.
        #[serde(skip_serializing_if = "Option::is_none")]
        presence_penalty: Option<f64>,
        /// Prompt cache routing key.
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_key: Option<String>,
        /// Prompt cache configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_options: Option<Json>,
        /// Deprecated prompt cache retention policy.
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_retention: Option<String>,
        /// Requested reasoning effort.
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_effort: Option<String>,
        /// Structured response format configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        response_format: Option<Json>,
        /// Stable safety identifier.
        #[serde(skip_serializing_if = "Option::is_none")]
        safety_identifier: Option<String>,
        /// Best-effort deterministic sampling seed.
        #[serde(skip_serializing_if = "Option::is_none")]
        seed: Option<i64>,
        /// Streaming response configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        stream_options: Option<Json>,
        /// Requested response verbosity.
        #[serde(skip_serializing_if = "Option::is_none")]
        verbosity: Option<String>,
        /// Web-search configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        web_search_options: Option<Json>,
    },
    /// OpenAI Responses-specific request fields.
    #[serde(rename = "openai_responses")]
    OpenAIResponses {
        /// Whether the response should run in the background.
        #[serde(skip_serializing_if = "Option::is_none")]
        background: Option<bool>,
        /// Context-management entries.
        #[serde(skip_serializing_if = "Option::is_none")]
        context_management: Option<Json>,
        /// Conversation identifier or object.
        #[serde(skip_serializing_if = "Option::is_none")]
        conversation: Option<Json>,
        /// Request moderation configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        moderation: Option<Json>,
        /// Reusable prompt template reference.
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt: Option<Json>,
        /// Prompt cache routing key.
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_key: Option<String>,
        /// Prompt cache configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_options: Option<Json>,
        /// Deprecated prompt cache retention policy.
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_retention: Option<String>,
        /// Stable safety identifier.
        #[serde(skip_serializing_if = "Option::is_none")]
        safety_identifier: Option<String>,
        /// Streaming response configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        stream_options: Option<Json>,
        /// Text output configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<Json>,
    },
    /// Custom provider request fields.
    #[serde(rename = "custom")]
    Custom {
        /// Custom API identifier.
        api_name: String,
        /// Opaque custom API data.
        data: Json,
    },
}

/// A specific tool choice that forces a named function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    /// The type (typically `"function"`).
    #[serde(rename = "type")]
    pub choice_type: String,
    /// The function to call.
    pub function: ToolChoiceFunctionName,
}

/// The name component of a specific tool choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolChoiceFunctionName {
    /// The function name.
    pub name: String,
}

/// Normalized generation parameters across providers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GenerationParams {
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Helper methods
// ---------------------------------------------------------------------------

impl AnnotatedLlmRequest {
    /// Extract the text content of the first system message, if any.
    ///
    /// For [`MessageContent::Text`], returns the string directly.
    /// For [`MessageContent::Parts`], returns the text of the first
    /// [`ContentPart::Text`] part.
    pub fn system_prompt(&self) -> Option<&str> {
        if let Some(text) = self.instructions.as_ref().and_then(first_content_text) {
            return Some(text);
        }
        self.messages.iter().find_map(|m| match m {
            Message::System { content, .. } | Message::Developer { content, .. } => {
                first_content_text(content)
            }
            _ => None,
        })
    }

    /// Get the text content of the last user message, if any.
    ///
    /// Searches messages in reverse order and returns the first user
    /// message found. For [`MessageContent::Parts`], returns the text of
    /// the first [`ContentPart::Text`] part.
    pub fn last_user_message(&self) -> Option<&str> {
        self.messages.iter().rev().find_map(|m| match m {
            Message::User { content, .. } => first_content_text(content),
            Message::ProviderNative {
                provider, value, ..
            } if provider == "openai_responses"
                && value.get("role").and_then(Json::as_str) == Some("user") =>
            {
                native_message_text(value)
            }
            _ => None,
        })
    }

    /// Check if any assistant message in the conversation contains tool calls.
    ///
    /// Returns `true` if at least one [`Message::Assistant`] variant has a
    /// non-empty `tool_calls` field.
    pub fn has_tool_calls(&self) -> bool {
        self.messages.iter().any(|m| match m {
            Message::Assistant {
                tool_calls: Some(calls),
                content,
                ..
            } => !calls.is_empty() || content.as_ref().is_some_and(content_has_tool_use),
            Message::Assistant {
                content: Some(content),
                ..
            } => content_has_tool_use(content),
            Message::ToolCallItem { .. } => true,
            Message::ProviderNative { value, .. } => matches!(
                value.get("type").and_then(Json::as_str),
                Some("function_call" | "custom_tool_call" | "tool_use")
            ),
            _ => false,
        })
    }
}

fn first_content_text(content: &MessageContent) -> Option<&str> {
    match content {
        MessageContent::Text(text) => Some(text.as_str()),
        MessageContent::Parts(parts) => parts.iter().find_map(|part| match part {
            ContentPart::Text { text, .. } => Some(text.as_str()),
            ContentPart::ProviderNative { value, .. } => value
                .get("text")
                .and_then(Json::as_str)
                .or_else(|| value.get("refusal").and_then(Json::as_str)),
            ContentPart::ImageUrl { .. }
            | ContentPart::Image { .. }
            | ContentPart::Audio { .. }
            | ContentPart::File { .. }
            | ContentPart::Refusal { .. }
            | ContentPart::ToolUse { .. }
            | ContentPart::ToolResult { .. } => None,
        }),
    }
}

fn content_has_tool_use(content: &MessageContent) -> bool {
    match content {
        MessageContent::Text(_) => false,
        MessageContent::Parts(parts) => parts.iter().any(|part| match part {
            ContentPart::ToolUse { .. } => true,
            ContentPart::ProviderNative { value, .. } => matches!(
                value.get("type").and_then(Json::as_str),
                Some("tool_use" | "mcp_tool_use" | "server_tool_use")
            ),
            _ => false,
        }),
    }
}

fn native_message_text(value: &Json) -> Option<&str> {
    match value.get("content")? {
        Json::String(text) => Some(text.as_str()),
        Json::Array(parts) => parts.iter().find_map(|part| {
            part.get("text")
                .and_then(Json::as_str)
                .or_else(|| part.get("refusal").and_then(Json::as_str))
        }),
        _ => None,
    }
}
