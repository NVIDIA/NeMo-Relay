// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalized LLM response types produced by response codecs.
//!
//! This module defines [`AnnotatedLlmResponse`] and its supporting types
//! for structured, API-agnostic access to LLM response data.

use serde::{Deserialize, Serialize};

use crate::json::Json;

pub use super::pricing::{
    CacheReadAccounting, ModelPricing, PricingCatalog, PricingCatalogError, PricingConfig,
    PricingResolver, PricingSource, PricingSourceConfig, PricingUnit, PromptCachePricing,
    TokenPricingRates, active_pricing_resolver, attach_estimated_cost,
    attach_estimated_cost_for_provider, estimate_cost, estimate_cost_for_provider,
    estimate_cost_with_catalog, estimate_cost_with_provider, infer_model_provider,
    pricing_for_model, pricing_for_provider, reset_active_pricing_resolver,
    set_active_pricing_resolver,
};
use super::request::MessageContent;

// ---------------------------------------------------------------------------
// AnnotatedLlmResponse type hierarchy
// ---------------------------------------------------------------------------

/// Structured view of an LLM response, produced by a response codec from
/// raw JSON API output.
///
/// The `extra` field captures any top-level keys not modeled by the known
/// fields, ensuring lossless round-trip through serde.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedLlmResponse {
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
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
    /// Optional cost reported by provider data or estimated from Relay pricing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<CostEstimate>,
}

/// Source of a normalized cost value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostSource {
    /// Cost was estimated by applying Relay's model pricing table to usage.
    ModelPricing,
    /// Cost was reported directly by a provider or framework payload.
    ProviderReported,
}

/// Normalized LLM response cost.
///
/// Provider-reported cost is preserved as-is. Model-pricing estimates include
/// source and as-of metadata so downstream systems can audit stale pricing
/// tables without losing a usable estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Total cost in `currency`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
    /// ISO 4217 currency code for the cost fields.
    #[serde(default = "default_cost_currency")]
    pub currency: String,
    /// Uncached prompt/input token cost in `currency`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<f64>,
    /// Completion/output token cost in `currency`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<f64>,
    /// Prompt cache read cost in `currency`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    /// Prompt cache write cost in `currency`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
    /// Origin of this cost value.
    pub source: CostSource,
    /// Provider associated with the cost or pricing estimate, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing_provider: Option<String>,
    /// Model ID associated with the cost or pricing estimate, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing_model: Option<String>,
    /// Date the pricing value was last verified, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing_as_of: Option<String>,
    /// Source URL or label for the pricing value, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing_source: Option<String>,
}

impl CostEstimate {
    /// Returns the explicit total, or the sum of component costs when no total was supplied.
    #[must_use]
    pub fn total_or_component_sum(&self) -> Option<f64> {
        self.total.or_else(|| {
            let (has_component, total) =
                [self.input, self.output, self.cache_read, self.cache_write]
                    .into_iter()
                    .flatten()
                    .fold((false, 0.0), |(_, total), value| (true, total + value));
            has_component.then_some(total)
        })
    }

    /// Returns the total only when it is denominated in the requested currency.
    #[must_use]
    pub fn total_for_currency(&self, currency: &str) -> Option<f64> {
        self.currency
            .eq_ignore_ascii_case(currency)
            .then_some(self.total)
            .flatten()
    }

    /// Returns the explicit or component-derived total in the requested currency.
    #[must_use]
    pub fn total_or_component_sum_for_currency(&self, currency: &str) -> Option<f64> {
        self.currency
            .eq_ignore_ascii_case(currency)
            .then(|| self.total_or_component_sum())
            .flatten()
    }
}

/// Provider/framework cost object accepted by built-in response codecs.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RawUsageCost {
    /// Normalized total cost in the supplied currency.
    pub total: Option<f64>,
    /// Uncached prompt/input token cost in the supplied currency.
    pub input: Option<f64>,
    /// Completion/output token cost in the supplied currency.
    pub output: Option<f64>,
    /// Prompt cache read cost in the supplied currency.
    pub cache_read: Option<f64>,
    /// Prompt cache write cost in the supplied currency.
    pub cache_write: Option<f64>,
    /// Optional currency override from provider data.
    pub currency: Option<String>,
    /// Optional provider provenance.
    pub pricing_provider: Option<String>,
    /// Optional model provenance.
    pub pricing_model: Option<String>,
    /// Optional as-of provenance.
    pub pricing_as_of: Option<String>,
    /// Optional source provenance.
    pub pricing_source: Option<String>,
}

pub(crate) fn provider_reported_cost(
    provider_total_cost: Option<f64>,
    cost: Option<RawUsageCost>,
) -> Option<CostEstimate> {
    let cost = cost.unwrap_or_default();
    let input = cost.input;
    let output = cost.output;
    let cache_read = cost.cache_read;
    let cache_write = cost.cache_write;
    let has_currency_native_amount = cost.total.is_some()
        || cost.input.is_some()
        || cost.output.is_some()
        || cost.cache_read.is_some()
        || cost.cache_write.is_some();
    let component_total = [input, output, cache_read, cache_write]
        .into_iter()
        .flatten()
        .sum();
    let has_component_cost =
        input.is_some() || output.is_some() || cache_read.is_some() || cache_write.is_some();
    let total = provider_total_cost
        .or(cost.total)
        .or_else(|| has_component_cost.then_some(component_total));

    if total.is_none()
        && input.is_none()
        && output.is_none()
        && cache_read.is_none()
        && cache_write.is_none()
    {
        return None;
    }

    Some(CostEstimate {
        total,
        currency: if provider_total_cost.is_some() {
            default_cost_currency()
        } else if has_currency_native_amount {
            cost.currency.unwrap_or_else(default_cost_currency)
        } else {
            default_cost_currency()
        },
        input,
        output,
        cache_read,
        cache_write,
        source: CostSource::ProviderReported,
        pricing_provider: cost.pricing_provider,
        pricing_model: cost.pricing_model,
        pricing_as_of: cost.pricing_as_of,
        pricing_source: cost.pricing_source,
    })
}

fn default_cost_currency() -> String {
    "USD".into()
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
        /// Echoed previous response ID for conversation continuation.
        #[serde(skip_serializing_if = "Option::is_none")]
        previous_response_id: Option<String>,
        /// Whether this response is marked for server-side storage.
        #[serde(skip_serializing_if = "Option::is_none")]
        store: Option<bool>,
        /// Service tier used for the response.
        #[serde(skip_serializing_if = "Option::is_none")]
        service_tier: Option<String>,
        /// Truncation behavior metadata.
        #[serde(skip_serializing_if = "Option::is_none")]
        truncation: Option<Json>,
        /// Reasoning configuration/result metadata.
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<Json>,
        /// Raw input token details payload.
        #[serde(skip_serializing_if = "Option::is_none")]
        input_tokens_details: Option<Json>,
        /// Raw output token details payload.
        #[serde(skip_serializing_if = "Option::is_none")]
        output_tokens_details: Option<Json>,
    },

    /// Anthropic Messages API-specific fields.
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages {
        /// Anthropic object type (typically `"message"`).
        #[serde(skip_serializing_if = "Option::is_none")]
        object_type: Option<String>,
        /// Anthropic response role (typically `"assistant"`).
        #[serde(skip_serializing_if = "Option::is_none")]
        role: Option<String>,
        /// Raw Anthropic stop_reason.
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
        /// Which stop sequence was matched (if any).
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_sequence: Option<String>,
        /// Anthropic response service tier when present.
        #[serde(skip_serializing_if = "Option::is_none")]
        service_tier: Option<String>,
        /// Anthropic container payload when present.
        #[serde(skip_serializing_if = "Option::is_none")]
        container: Option<Json>,
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

impl AnnotatedLlmResponse {
    /// Extract the text content of the response message.
    ///
    /// For [`MessageContent::Text`], returns the string directly.
    /// For [`MessageContent::Parts`], returns the text of the first
    /// [`super::request::ContentPart::Text`] part.
    /// Returns `None` if `message` is `None`.
    #[must_use]
    pub fn response_text(&self) -> Option<&str> {
        match self.message.as_ref()? {
            MessageContent::Text(s) => Some(s.as_str()),
            MessageContent::Parts(parts) => parts.iter().find_map(|p| match p {
                super::request::ContentPart::Text { text } => Some(text.as_str()),
                super::request::ContentPart::ImageUrl { .. } => None,
            }),
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
