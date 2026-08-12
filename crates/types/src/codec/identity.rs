// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable LLM codec identities shared across Relay runtime and SDK boundaries.
//!
//! These types identify a codec selected by the Relay runtime. They do not
//! provide codec implementations, provider detection, or runtime capabilities.

use serde::{Deserialize, Serialize};

/// Relay's built-in LLM codec identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinLlmCodec {
    /// OpenAI Chat Completions request and response payloads.
    #[serde(rename = "openai_chat")]
    OpenAiChat,
    /// OpenAI Responses request and response payloads.
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    /// Anthropic Messages request and response payloads.
    AnthropicMessages,
    /// OCI Generative AI chat request and response payloads.
    OCIGenAI,
    /// Gemini generateContent request and response payloads.
    GeminiGenerateContent,
}

impl BuiltinLlmCodec {
    /// Stable identifier used in configuration and SDK transport boundaries.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::OpenAiChat => "openai_chat",
            Self::OpenAiResponses => "openai_responses",
            Self::AnthropicMessages => "anthropic_messages",
            Self::OCIGenAI => "oci_genai",
            Self::GeminiGenerateContent => "gemini_generate_content",
        }
    }

    /// Resolve a stable built-in codec identifier.
    #[must_use]
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "openai_chat" => Some(Self::OpenAiChat),
            "openai_responses" => Some(Self::OpenAiResponses),
            "anthropic_messages" => Some(Self::AnthropicMessages),
            "oci_genai" => Some(Self::OCIGenAI),
            "gemini_generate_content" => Some(Self::GeminiGenerateContent),
            _ => None,
        }
    }
}

/// Per-call LLM codec identity supplied to sanitizer and SDK callbacks.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum LlmCodecIdentity {
    /// No codec was active for this payload direction.
    #[default]
    None,
    /// A Relay built-in codec was active.
    #[serde(rename = "builtin")]
    BuiltIn(BuiltinLlmCodec),
    /// A runtime-registered codec was active, identified by its stable ID.
    Runtime(String),
    /// A codec was active but does not expose a registered identity.
    Opaque,
}
