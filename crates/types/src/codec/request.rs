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
/// The `extra` field captures any provider-specific keys not modeled by the
/// known fields, ensuring lossless round-trip through `decode`/`encode`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedLlmRequest {
    /// Parsed conversation messages.
    pub messages: Vec<Message>,
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
    /// Extensible key-value pairs for unmodeled provider-specific fields.
    /// Merged back into the request body during encode via `serde(flatten)`.
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
    },
    /// An image URL content part.
    ImageUrl {
        /// Image URL payload.
        image_url: OpenAiImageUrl,
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

/// A tool definition (function schema) available to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The type of tool (typically `"function"`).
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition.
    pub function: FunctionDefinition,
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
        self.messages.iter().find_map(|m| match m {
            Message::System { content, .. } => match content {
                MessageContent::Text(s) => Some(s.as_str()),
                MessageContent::Parts(parts) => parts.iter().find_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                }),
            },
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
            Message::User { content, .. } => match content {
                MessageContent::Text(s) => Some(s.as_str()),
                MessageContent::Parts(parts) => parts.iter().find_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                }),
            },
            _ => None,
        })
    }

    /// Check if any assistant message in the conversation contains tool calls.
    ///
    /// Returns `true` if at least one [`Message::Assistant`] variant has a
    /// non-empty `tool_calls` field.
    pub fn has_tool_calls(&self) -> bool {
        self.messages.iter().any(|m| {
            matches!(
                m,
                Message::Assistant { tool_calls: Some(calls), .. } if !calls.is_empty()
            )
        })
    }
}
