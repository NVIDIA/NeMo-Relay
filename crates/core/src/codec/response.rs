// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalized LLM response types produced by response codecs.
//!
//! This module defines [`AnnotatedLLMResponse`] and its supporting types
//! for structured, API-agnostic access to LLM response data.

use serde::{Deserialize, Serialize};

use crate::json::Json;

use super::request::MessageContent;

// ---------------------------------------------------------------------------
// AnnotatedLLMResponse type hierarchy
// ---------------------------------------------------------------------------

/// Structured view of an LLM response, produced by a response codec from
/// raw JSON API output.
///
/// The `extra` field captures any top-level keys not modeled by the known
/// fields, ensuring lossless round-trip through serde.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedLLMResponse {
    /// Response ID from the API (e.g., "chatcmpl-abc123", "resp_abc123", "msg_abc123").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// The model that actually served the request (may differ from requested model).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// The assistant's response content, reusing [`MessageContent`] from request types.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<MessageContent>,

    /// Tool calls requested by the model, normalized across APIs.
    ///
    /// Uses [`ResponseToolCall`] (arguments as [`Json`]) NOT the request-side
    /// `ToolCall` (arguments as `String`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,

    /// Why generation stopped, normalized across APIs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// API-specific response data that cannot be normalized across providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_specific: Option<ApiSpecificResponse>,

    /// Catch-all for unmodeled top-level fields, ensuring lossless round-trip.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Json>,
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// Token usage statistics from an LLM API response.
///
/// All fields are `Option<u64>` because not every provider supplies every
/// field. For example, cache token counts are only available from providers
/// that support prompt caching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Tokens consumed by the prompt/input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Tokens generated in the completion/output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Total tokens (prompt + completion).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Tokens served from prompt cache (read).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    /// Tokens written to prompt cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// FinishReason
// ---------------------------------------------------------------------------

/// Normalized reason why the model stopped generating.
///
/// Maps from provider-specific stop reasons:
/// - **Complete**: OpenAI Chat `"stop"`, Anthropic `"end_turn"`, Responses `"completed"`
/// - **Length**: OpenAI Chat `"length"`, Anthropic `"max_tokens"`, Responses incomplete+max_output_tokens
/// - **ToolUse**: OpenAI Chat `"tool_calls"`, Anthropic `"tool_use"`
/// - **ContentFilter**: OpenAI Chat `"content_filter"`, Responses incomplete+content_filter
/// - **Unknown**: Forward-compatible catch-all for unrecognized reasons
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Model naturally completed its response.
    Complete,
    /// Maximum token limit reached.
    Length,
    /// Model requested a tool call.
    ToolUse,
    /// Content was filtered by safety systems.
    ContentFilter,
    /// Unknown or forward-compatible reason.
    Unknown(String),
}

impl FinishReason {
    /// Returns `true` if the model naturally completed its response.
    ///
    /// Only the [`FinishReason::Complete`] variant returns `true`.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self, FinishReason::Complete)
    }
}

// ---------------------------------------------------------------------------
// ResponseToolCall
// ---------------------------------------------------------------------------

/// A tool call requested by the model in its response.
///
/// Unlike the request-side `ToolCall` (which stores arguments as a JSON
/// string per OpenAI convention), response tool calls store arguments as
/// parsed [`Json`]. Codecs parse OpenAI's string arguments during decode;
/// Anthropic's `input` is already parsed JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// The function/tool name.
    pub name: String,
    /// The arguments as parsed JSON (not a string).
    pub arguments: Json,
}

// ---------------------------------------------------------------------------
// ApiSpecificResponse
// ---------------------------------------------------------------------------

/// API-specific response data that cannot be normalized across providers.
///
/// Each variant captures fields unique to a particular LLM API, stored via
/// internal tagging on the `"api"` key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "api")]
pub enum ApiSpecificResponse {
    /// OpenAI Chat Completions-specific fields.
    #[serde(rename = "openai_chat")]
    OpenAIChat {
        /// Token-level log probabilities (raw JSON, too complex to normalize).
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<Json>,
        /// System fingerprint for reproducibility.
        #[serde(skip_serializing_if = "Option::is_none")]
        system_fingerprint: Option<String>,
        /// Processing tier used (e.g., "default").
        #[serde(skip_serializing_if = "Option::is_none")]
        service_tier: Option<String>,
    },

    /// OpenAI Responses API-specific fields.
    #[serde(rename = "openai_responses")]
    OpenAIResponses {
        /// Full output items array for direct access.
        #[serde(skip_serializing_if = "Option::is_none")]
        output_items: Option<Vec<Json>>,
        /// Response status (e.g., "completed", "incomplete").
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        /// Details about why the response is incomplete.
        #[serde(skip_serializing_if = "Option::is_none")]
        incomplete_details: Option<Json>,
    },

    /// Anthropic Messages API-specific fields.
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages {
        /// Which stop sequence was matched (if any).
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_sequence: Option<String>,
        /// Full content blocks array for direct access.
        #[serde(skip_serializing_if = "Option::is_none")]
        content_blocks: Option<Vec<Json>>,
    },

    /// Custom/unknown API -- catch-all for user-implemented codecs.
    #[serde(rename = "custom")]
    Custom {
        /// API identifier.
        api_name: String,
        /// Opaque API-specific data.
        data: Json,
    },
}

// ---------------------------------------------------------------------------
// Helper methods
// ---------------------------------------------------------------------------

impl AnnotatedLLMResponse {
    /// Extract the text content of the response message.
    ///
    /// For [`MessageContent::Text`], returns the string directly.
    /// For [`MessageContent::Parts`], returns the text of the first
    /// [`ContentPart::Text`] part.
    /// Returns `None` if `message` is `None`.
    #[must_use]
    pub fn response_text(&self) -> Option<&str> {
        match self.message.as_ref()? {
            MessageContent::Text(s) => Some(s.as_str()),
            MessageContent::Parts(parts) => parts
                .iter()
                .map(|p| {
                    let super::request::ContentPart::Text { text } = p;
                    text.as_str()
                })
                .next(),
        }
    }

    /// Check if the response contains any tool calls.
    ///
    /// Returns `true` if `tool_calls` is `Some` with at least one element.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/codec/response_tests.rs"]
mod tests;
