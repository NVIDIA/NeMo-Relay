// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin-neutral evidence and summaries for LLM optimizations.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::Json;
use crate::api::event::DataSchema;

use super::response::{CostEstimate, Usage};

/// Open, forward-compatible optimization classification.
///
/// This is intentionally a string-backed newtype rather than a closed enum:
/// third-party optimization kinds must deserialize and round-trip losslessly
/// before Relay knows about them. Standard constants and constructors provide
/// enum-like ergonomics without making new producers wait for a core release.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LlmOptimizationKind(String);

impl LlmOptimizationKind {
    /// A request transformation that reduces input tokens.
    pub const INPUT_COMPRESSION: &'static str = "input_compression";
    /// A routing decision that changes the model serving a request.
    pub const MODEL_ROUTING: &'static str = "model_routing";

    /// Preserve an arbitrary producer-defined kind on the wire.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Return the exact wire value.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Construct the standard input-compression kind.
    #[must_use]
    pub fn input_compression() -> Self {
        Self::new(Self::INPUT_COMPRESSION)
    }

    /// Construct the standard model-routing kind.
    #[must_use]
    pub fn model_routing() -> Self {
        Self::new(Self::MODEL_ROUTING)
    }
}

impl From<String> for LlmOptimizationKind {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for LlmOptimizationKind {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

/// Model identity used for counterfactual and effective pricing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmOptimizationModel {
    /// Model identifier understood by Relay's pricing resolver.
    pub model: String,
    /// Optional pricing-provider namespace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

impl LlmOptimizationModel {
    /// Construct a model identity without a provider namespace.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            provider: None,
        }
    }

    /// Add the pricing-provider namespace.
    #[must_use]
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }
}

/// A model change proposed or applied by an optimizer.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmOptimizationModelTransition {
    /// Counterfactual model that would otherwise have served the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<LlmOptimizationModel>,
    /// Model selected by the optimization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective: Option<LlmOptimizationModel>,
}

/// Token quantities retained independently from pricing arithmetic.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmOptimizationTokens {
    /// Input/prompt tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Output/completion tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Tokens read from a provider prompt cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    /// Tokens written to a provider prompt cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    /// Total tokens when supplied by the producer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

impl LlmOptimizationTokens {
    /// Construct explicit prompt and total token savings.
    #[must_use]
    pub fn saved_prompt(prompt_tokens: u64) -> Self {
        Self {
            prompt_tokens: Some(prompt_tokens),
            total_tokens: Some(prompt_tokens),
            ..Self::default()
        }
    }
}

/// Whether token evidence was observed directly or estimated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmOptimizationEvidenceQuality {
    /// Directly observed token counts.
    Observed,
    /// Counts produced by a tokenizer or estimator.
    Estimated,
}

/// Shared token evidence that Relay can aggregate without understanding a plugin payload.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmOptimizationTokenImpact {
    /// Token counts before the optimization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<LlmOptimizationTokens>,
    /// Token counts after the optimization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective: Option<LlmOptimizationTokens>,
    /// Explicit reduction retained for downstream repricing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved: Option<LlmOptimizationTokens>,
    /// Evidence quality for these counts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<LlmOptimizationEvidenceQuality>,
    /// Tokenizer, counter, or estimation method.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimation_method: Option<String>,
}

/// Trait implemented by typed plugin payloads embedded in an optimization contribution.
pub trait LlmOptimizationPayload: Serialize {
    /// Stable schema name for the payload.
    const SCHEMA_NAME: &'static str;
    /// Schema version for the payload.
    const SCHEMA_VERSION: &'static str;
}

/// One optimizer's evidence for a change to an LLM call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmOptimizationContribution {
    /// Relay-assigned contribution identifier. Producer values are replaced on ingestion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Uuid>,
    /// Relay-assigned order within the LLM call. Producer values are replaced on ingestion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<u64>,
    /// Stable producer or plugin identity.
    pub producer: String,
    /// Open optimization classification.
    pub kind: LlmOptimizationKind,
    /// Whether the optimization affected the executed call.
    #[serde(default)]
    pub applied: bool,
    /// Optional shared model transition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_transition: Option<LlmOptimizationModelTransition>,
    /// Optional shared token evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_impact: Option<LlmOptimizationTokenImpact>,
    /// Schema of `payload`; required whenever `payload` is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_schema: Option<DataSchema>,
    /// Opaque producer payload retained for audit and future consumers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<Json>,
    /// Unknown top-level fields from future producers.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Json>,
}

impl LlmOptimizationContribution {
    /// Construct an applied contribution with no model, token, or custom evidence.
    #[must_use]
    pub fn new(producer: impl Into<String>, kind: impl Into<LlmOptimizationKind>) -> Self {
        Self {
            id: None,
            sequence: None,
            producer: producer.into(),
            kind: kind.into(),
            applied: true,
            model_transition: None,
            token_impact: None,
            payload_schema: None,
            payload: None,
            extra: BTreeMap::new(),
        }
    }

    /// Attach a schema-tagged custom payload.
    pub fn with_payload<T: LlmOptimizationPayload>(
        mut self,
        payload: &T,
    ) -> Result<Self, serde_json::Error> {
        self.payload_schema = Some(DataSchema {
            name: T::SCHEMA_NAME.to_string(),
            version: T::SCHEMA_VERSION.to_string(),
        });
        self.payload = Some(serde_json::to_value(payload)?);
        Ok(self)
    }
}

/// Completeness of close-time optimization accounting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmOptimizationSummaryStatus {
    /// All requested token and monetary calculations were available.
    Complete,
    /// Evidence remains useful, but one or more calculations were unavailable.
    Partial,
}

/// Close-time accounting attached to the normalized LLM response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmOptimizationSummary {
    /// Wire schema version.
    pub schema_version: String,
    /// Arithmetic implementation version.
    pub calculation_version: String,
    /// Whether all accounting inputs were available.
    pub status: LlmOptimizationSummaryStatus,
    /// Machine-readable reasons for partial accounting.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub limitations: Vec<String>,
    /// Counterfactual model used for baseline pricing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_model: Option<LlmOptimizationModel>,
    /// Model that actually served the terminal response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_model: Option<LlmOptimizationModel>,
    /// Usage observed on the effective response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_usage: Option<Usage>,
    /// Counterfactual usage derived from observed usage plus explicit savings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_usage: Option<Usage>,
    /// Explicit aggregate token reductions, retained independently from pricing.
    pub tokens_saved: LlmOptimizationTokens,
    /// Estimated cost for baseline model/usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_cost: Option<CostEstimate>,
    /// Provider-reported or Relay-estimated actual cost.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_cost: Option<CostEstimate>,
    /// Baseline cost minus actual cost when both share a currency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_cost_saved: Option<f64>,
    /// Currency for `estimated_cost_saved`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub currency: Option<String>,
    /// Ordered, bounded source evidence used by the calculation.
    pub contributions: Vec<LlmOptimizationContribution>,
}
