// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable LLM codec identities shared across Relay runtime and SDK boundaries.
//!
//! These types identify a codec selected by the Relay runtime. They do not
//! provide codec implementations, provider detection, or runtime capabilities.

use serde::{Deserialize, Serialize};

macro_rules! builtin_llm_codecs {
    ($( $(#[$meta:meta])* $variant:ident => $id:literal, )+) => {
        /// Relay's built-in LLM codec identities.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        pub enum BuiltinLlmCodec {
            $(
                $(#[$meta])*
                #[serde(rename = $id)]
                $variant,
            )+
        }

        impl BuiltinLlmCodec {
            /// Every built-in codec identity in stable ID order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// Stable identifier used in configuration and SDK transport boundaries.
            #[must_use]
            pub const fn id(self) -> &'static str {
                match self {
                    $(Self::$variant => $id),+
                }
            }

            /// Resolve a stable built-in codec identifier.
            #[must_use]
            pub fn from_id(id: &str) -> Option<Self> {
                match id {
                    $($id => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

builtin_llm_codecs! {
    /// OpenAI Chat Completions request and response payloads.
    OpenAiChat => "openai_chat",
    /// OpenAI Responses request and response payloads.
    OpenAiResponses => "openai_responses",
    /// Anthropic Messages request and response payloads.
    AnthropicMessages => "anthropic_messages",
    /// OCI Generative AI chat request and response payloads.
    OCIGenAI => "oci_genai",
    /// Gemini generateContent request and response payloads.
    GeminiGenerateContent => "gemini_generate_content",
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
