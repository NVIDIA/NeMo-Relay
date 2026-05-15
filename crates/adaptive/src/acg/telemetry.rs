// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache performance telemetry types for the Adaptive Cache Governor (ACG)
//! system.
//!
//! These types normalize provider-specific cache metrics (Anthropic
//! `cache_read_input_tokens`/`cache_creation_input_tokens`, OpenAI
//! `cached_tokens`) into a common schema for uniform measurement.
//! Populated by provider-specific normalization logic in Phase 9.

use chrono::{DateTime, Utc};
use nemo_flow::codec::response::Usage;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::types::AgentIdentity;

// ===================================================================
// Cache miss diagnosis contract
// ===================================================================

/// Request-time facts used to classify a cache miss without leaking prompt text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheRequestFacts {
    /// Canonical provider string associated with the request facts.
    pub provider: String,
    /// Number of stable prefix blocks observed in the request.
    pub stable_prefix_length: usize,
    /// Token count for the stable prefix when it can be measured safely.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub stable_prefix_tokens: Option<u32>,
    /// Minimum provider threshold required for cache reuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub required_min_tokens: Option<u32>,
    /// Span ID of the first stable block that mismatched the retained exemplar.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub first_mismatch_span_id: Option<String>,
    /// Sequence index of the first mismatching stable block.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub first_mismatch_sequence_index: Option<u32>,
    /// Expected short SHA-256 hash prefix for the first mismatching block.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub expected_hash_prefix: Option<String>,
    /// Actual short SHA-256 hash prefix for the first mismatching block.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub actual_hash_prefix: Option<String>,
    /// Active cache retention window in seconds when provider semantics expose one.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub retention_window_secs: Option<f64>,
    /// Observed elapsed time since the same stable prefix was last seen.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub observed_gap_secs: Option<f64>,
    /// Facts that were unavailable when the runtime attempted diagnosis.
    #[serde(default)]
    pub missing_facts: Vec<String>,
}

/// Structured diagnosis for a cache miss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheMissDiagnosis {
    /// Single-line bounded explanation of the miss.
    pub summary: String,
    /// Single actionable follow-up for the caller.
    pub recommendation: String,
    /// Evidence supporting the diagnosis.
    pub evidence: CacheMissEvidence,
}

/// Typed evidence for a cache miss diagnosis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CacheMissEvidence {
    /// Stable prefix diverged from the retained exemplar.
    PrefixMismatch {
        /// Span ID of the first mismatching stable block.
        first_mismatch_span_id: String,
        /// Zero-based sequence index of the mismatching block.
        sequence_index: u32,
        /// Expected short SHA-256 hash prefix.
        expected_hash_prefix: String,
        /// Actual short SHA-256 hash prefix.
        actual_hash_prefix: String,
    },
    /// Stable prefix is too short for provider cache reuse.
    BelowMinimumThreshold {
        /// Observed stable prefix tokens.
        observed_prefix_tokens: u32,
        /// Required minimum tokens for cache reuse.
        required_min_tokens: u32,
        /// Source of the token estimate.
        estimation_source: String,
    },
    /// Stable prefix likely aged out of the provider retention window.
    RetentionExpired {
        /// Observed gap between requests with the same stable prefix.
        observed_gap_secs: f64,
        /// Provider retention window in seconds.
        retention_window_secs: f64,
        /// Human-readable provider semantics summary.
        provider_semantics: String,
    },
    /// Diagnosis could not be justified from the available facts.
    Unknown {
        /// List of facts that were unavailable at classification time.
        missing_facts: Vec<String>,
    },
}

// ===================================================================
// Cache miss reason taxonomy
// ===================================================================

/// Reason why a cache miss occurred.
///
/// Covers 8 determinable reasons plus an extensible `Other` variant.
/// Uses internally-tagged JSON representation (`"reason"` field) so
/// each variant serializes as `{"reason": "snake_case"}` and the
/// `Other` variant additionally carries a `description` field.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum CacheMissReason {
    /// Prompt prefix didn't match cached prefix.
    PrefixMismatch,
    /// Stable prefix shorter than provider minimum for caching.
    BelowMinimumThreshold,
    /// Cached prefix retention window elapsed.
    RetentionExpired,
    /// Request routed to different worker/pool.
    RoutingMismatch,
    /// Cache evicted due to capacity pressure.
    Evicted,
    /// Backend/model doesn't support caching.
    UnsupportedFeature,
    /// First request for this prefix (no prior cache entry).
    ColdStart,
    /// Reason could not be determined from provider response.
    Unknown,
    /// Extensible escape hatch for reasons not yet in the enum.
    Other {
        /// Human-readable description of the miss reason.
        description: String,
    },
}

// ===================================================================
// Cache hit rate (aggregated)
// ===================================================================

/// Aggregated cache hit rate over a time window.
///
/// Used for dashboard metrics and trend analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheHitRate {
    /// Hit rate in the range `[0.0, 1.0]`.
    pub hit_rate: f64,
    /// Number of requests in the measurement window.
    pub sample_count: u32,
    /// Duration of the measurement window in seconds.
    pub window_duration_secs: f64,
}

// ===================================================================
// Cache telemetry event (per-call)
// ===================================================================

/// Cache telemetry provider identity for canonical `Usage` normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheTelemetryProvider {
    /// Anthropic Messages cache telemetry semantics.
    Anthropic,
    /// OpenAI Chat/Responses cache telemetry semantics.
    OpenAI,
}

impl CacheTelemetryProvider {
    /// Returns the canonical provider string for serialized cache telemetry.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAI => "openai",
        }
    }
}

/// Per-call cache telemetry event.
///
/// Captures provider-agnostic cache metrics for a single LLM request.
/// The `agent_identity` field cross-references the Phase 3
/// [`AgentIdentity`] type for per-agent grouping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheTelemetryEvent {
    /// Request ID this telemetry pertains to.
    pub request_id: Uuid,
    /// Identity of the agent that issued the request.
    pub agent_identity: AgentIdentity,
    /// Number of tokens served from cache.
    pub cache_read_tokens: u64,
    /// Number of tokens written to cache.
    pub cache_creation_tokens: u64,
    /// Total prompt tokens (for hit rate calculation).
    pub total_prompt_tokens: u64,
    /// Computed cache hit rate `[0.0, 1.0]`.
    pub hit_rate: f64,
    /// Reason for cache miss, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub miss_reason: Option<CacheMissReason>,
    /// Structured miss diagnosis, when the miss can be justified safely.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub miss_diagnosis: Option<CacheMissDiagnosis>,
    /// Provider name (e.g., "anthropic", "openai").
    pub provider: String,
    /// When this telemetry was recorded.
    pub timestamp: DateTime<Utc>,
}

impl CacheTelemetryEvent {
    /// Computes hit rate from token counts. Returns `0.0` if
    /// `total_prompt_tokens` is zero to avoid division by zero.
    pub fn compute_hit_rate(cache_read_tokens: u64, total_prompt_tokens: u64) -> f64 {
        if total_prompt_tokens == 0 {
            0.0
        } else {
            cache_read_tokens as f64 / total_prompt_tokens as f64
        }
    }

    /// Builds a canonical cache telemetry event from normalized usage fields.
    ///
    /// Returns `None` when the normalized usage payload does not contain
    /// `prompt_tokens`, because Phase 10 does not invent missing totals.
    #[must_use]
    pub fn from_usage(
        request_id: Uuid,
        agent_identity: AgentIdentity,
        provider: CacheTelemetryProvider,
        usage: &Usage,
        timestamp: DateTime<Utc>,
        request_facts: Option<&CacheRequestFacts>,
    ) -> Option<Self> {
        let prompt_tokens = usage.prompt_tokens?;
        let cache_read_tokens = usage.cache_read_tokens.unwrap_or(0);

        let (cache_creation_tokens, total_prompt_tokens) = match provider {
            CacheTelemetryProvider::Anthropic => {
                let cache_creation_tokens = usage.cache_write_tokens.unwrap_or(0);
                let total_prompt_tokens = prompt_tokens + cache_read_tokens + cache_creation_tokens;
                (cache_creation_tokens, total_prompt_tokens)
            }
            CacheTelemetryProvider::OpenAI => (0, prompt_tokens),
        };

        let (miss_reason, miss_diagnosis) = if cache_read_tokens > 0 {
            (None, None)
        } else if matches!(provider, CacheTelemetryProvider::Anthropic) && cache_creation_tokens > 0
        {
            (Some(CacheMissReason::ColdStart), None)
        } else {
            classify_cache_miss(provider, request_facts)
        };

        Some(Self {
            request_id,
            agent_identity,
            cache_read_tokens,
            cache_creation_tokens,
            total_prompt_tokens,
            hit_rate: Self::compute_hit_rate(cache_read_tokens, total_prompt_tokens),
            miss_reason,
            miss_diagnosis,
            provider: provider.as_str().to_string(),
            timestamp,
        })
    }
}

fn classify_cache_miss(
    provider: CacheTelemetryProvider,
    request_facts: Option<&CacheRequestFacts>,
) -> (Option<CacheMissReason>, Option<CacheMissDiagnosis>) {
    if let Some(diagnosis) = prefix_mismatch_diagnosis(request_facts) {
        return (Some(CacheMissReason::PrefixMismatch), Some(diagnosis));
    }

    if let Some(diagnosis) = below_minimum_threshold_diagnosis(request_facts) {
        return (
            Some(CacheMissReason::BelowMinimumThreshold),
            Some(diagnosis),
        );
    }

    if let Some(diagnosis) = retention_expired_diagnosis(provider, request_facts) {
        return (Some(CacheMissReason::RetentionExpired), Some(diagnosis));
    }

    (
        Some(CacheMissReason::Unknown),
        Some(unknown_diagnosis(request_facts)),
    )
}

fn prefix_mismatch_diagnosis(
    request_facts: Option<&CacheRequestFacts>,
) -> Option<CacheMissDiagnosis> {
    let facts = request_facts?;
    let span_id = facts.first_mismatch_span_id.as_ref()?;
    let sequence_index = facts.first_mismatch_sequence_index?;
    let expected_hash_prefix = facts.expected_hash_prefix.as_ref()?;
    let actual_hash_prefix = facts.actual_hash_prefix.as_ref()?;

    Some(CacheMissDiagnosis {
        summary: format!(
            "Stable prefix diverged at span {} before cache reuse.",
            span_id
        ),
        recommendation: "Move or extract the mismatching block after the stable prefix."
            .to_string(),
        evidence: CacheMissEvidence::PrefixMismatch {
            first_mismatch_span_id: span_id.clone(),
            sequence_index,
            expected_hash_prefix: canonicalize_hash_prefix(expected_hash_prefix),
            actual_hash_prefix: canonicalize_hash_prefix(actual_hash_prefix),
        },
    })
}

fn below_minimum_threshold_diagnosis(
    request_facts: Option<&CacheRequestFacts>,
) -> Option<CacheMissDiagnosis> {
    let facts = request_facts?;
    let observed_prefix_tokens = facts.stable_prefix_tokens?;
    let required_min_tokens = facts.required_min_tokens?;
    if observed_prefix_tokens >= required_min_tokens {
        return None;
    }

    Some(CacheMissDiagnosis {
        summary: format!(
            "Stable prefix has {observed_prefix_tokens} tokens, below the {required_min_tokens}-token cache minimum."
        ),
        recommendation:
            "Increase the cacheable prefix above the provider minimum or stop expecting a hit."
                .to_string(),
        evidence: CacheMissEvidence::BelowMinimumThreshold {
            observed_prefix_tokens,
            required_min_tokens,
            estimation_source: "prompt_ir_token_metadata".to_string(),
        },
    })
}

fn retention_expired_diagnosis(
    provider: CacheTelemetryProvider,
    request_facts: Option<&CacheRequestFacts>,
) -> Option<CacheMissDiagnosis> {
    if !matches!(provider, CacheTelemetryProvider::Anthropic) {
        return None;
    }

    let facts = request_facts?;
    let observed_gap_secs = facts.observed_gap_secs?;
    let retention_window_secs = facts.retention_window_secs?;
    if observed_gap_secs <= retention_window_secs {
        return None;
    }

    Some(CacheMissDiagnosis {
        summary: format!(
            "Stable prefix reuse arrived {:.1}s after the {:.1}s retention window.",
            observed_gap_secs, retention_window_secs
        ),
        recommendation:
            "Reuse the stable prefix inside the active retention window or accept a cold rebuild."
                .to_string(),
        evidence: CacheMissEvidence::RetentionExpired {
            observed_gap_secs,
            retention_window_secs,
            provider_semantics:
                "anthropic prompt caching reuses prefixes inside the active retention window"
                    .to_string(),
        },
    })
}

fn unknown_diagnosis(request_facts: Option<&CacheRequestFacts>) -> CacheMissDiagnosis {
    let missing_facts = request_facts.map_or_else(
        || vec!["request_facts_unavailable".to_string()],
        |facts| facts.missing_facts.clone(),
    );

    CacheMissDiagnosis {
        summary: "Cache miss could not be classified from the available request facts.".to_string(),
        recommendation: "Capture request facts earlier or keep the miss classified as unknown."
            .to_string(),
        evidence: CacheMissEvidence::Unknown { missing_facts },
    }
}

fn canonicalize_hash_prefix(value: &str) -> String {
    const PREFIX: &str = "sha256:";
    const HEX_LEN: usize = 12;

    let suffix = value
        .strip_prefix(PREFIX)
        .unwrap_or(value)
        .chars()
        .take(HEX_LEN)
        .collect::<String>();

    format!("{PREFIX}{suffix}")
}

#[cfg(test)]
#[path = "../../tests/unit/acg/telemetry_tests.rs"]
mod tests;
