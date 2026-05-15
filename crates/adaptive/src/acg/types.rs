// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core data types for the Adaptive Cache Governor (ACG) crate.
//!
//! This module defines the vocabulary types used by the Adaptive Cache
//! Governor (ACG) system:
//! [`OptimizationIntent`] enum with 9 variants, per-variant payload structs,
//! [`OptimizationIntentBundle`], [`AgentIdentity`], and supporting enums
//! ([`SharingScope`], [`RetentionTier`], [`PlacementTarget`], [`ModelClass`],
//! [`IntentType`]).
//!
//! All types derive [`serde::Serialize`] and [`serde::Deserialize`] so they
//! can be round-tripped through JSON without loss.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ===================================================================
// Supporting enums
// ===================================================================

/// Sharing scope for cached content -- stability does not imply shareability.
/// Default is `Session` per security requirements.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharingScope {
    /// Request-scoped; content is not shared beyond the current request.
    Request,
    /// Session-scoped; content is shared within a single user session.
    #[default]
    Session,
    /// Tenant-scoped; content is shared across sessions within a tenant.
    Tenant,
    /// Globally shared; content is available across all tenants.
    Global,
}

/// Retention tier for cached state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionTier {
    /// Discarded after immediate use.
    Ephemeral,
    /// Retained for a short period (seconds to minutes).
    ShortLived,
    /// Retained for the duration of the session.
    SessionDuration,
    /// Retained beyond session boundaries.
    LongLived,
    /// Retained indefinitely.
    Permanent,
}

/// Target location for context placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementTarget {
    /// Stable content placed in the cacheable prefix zone.
    CacheablePrefix,
    /// Tool output deferred to a separate block.
    DeferredToolBlock,
    /// Large content replaced with a reference handle.
    ArtifactReference,
    /// Content fetched on demand rather than inlined.
    RetrievalOnDemand,
    /// Summarized session memory.
    SessionMemorySummary,
    /// Volatile content placed in the non-cacheable suffix.
    NonCacheableSuffix,
}

/// Model complexity/criticality class for routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelClass {
    /// Low-cost, high-throughput model for simple tasks.
    Economy,
    /// General-purpose model.
    Standard,
    /// High-capability model for complex reasoning.
    Premium,
    /// Most capable model, reserved for critical operations.
    Critical,
}

/// Discriminant enum for intent types (used in translation report outcomes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentType {
    /// Cache stability analysis intent.
    CacheStability,
    /// Content extraction and variable detection intent.
    ContentExtraction,
    /// Serialization and fanout optimization intent.
    Serialization,
    /// Latency and priority routing intent.
    Priority,
    /// Model routing and selection intent.
    ModelRouting,
    /// Context placement optimization intent.
    Placement,
    /// Cache retention policy intent.
    Retention,
    /// Tool scope and phase management intent.
    ToolScope,
    /// Content compression intent.
    Compression,
}

// ===================================================================
// Per-variant payload structs (9 total)
// ===================================================================

/// Cache stability analysis results for a prompt region.
///
/// Signals how stable a prefix is across requests and recommends
/// retention and sharing policies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheStabilityIntent {
    /// Stability score in the range `[0.0, 1.0]`.
    pub stability_score: f64,
    /// Byte offset marking the end of the stable prefix.
    pub stable_prefix_end: usize,
    /// Recommended retention tier based on stability analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub recommended_retention_tier: Option<RetentionTier>,
    /// Sharing scope label for this cached region.
    pub scope_label: SharingScope,
    /// Confidence in the stability assessment `[0.0, 1.0]`.
    pub confidence: f64,
    /// Number of observations backing this assessment.
    pub evidence_count: u32,
}

/// Content extraction intent for variable content detection.
///
/// Identifies dynamic regions within a prompt block that can be
/// extracted and templated for cache reuse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentExtractionIntent {
    /// Identifier of the prompt block containing the variable content.
    pub block_id: String,
    /// Pattern describing the variable content (e.g., regex or template syntax).
    pub variable_pattern: String,
    /// Strategy for extracting the variable content.
    pub extraction_strategy: String,
    /// Sharing scope for the extracted template.
    pub scope_label: SharingScope,
}

/// Serialization and fanout optimization intent.
///
/// Indicates that a prompt region is reused across multiple parallel
/// requests and can benefit from serialized (shared) caching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SerializationIntent {
    /// Number of parallel requests sharing this content.
    pub fanout_width: u32,
    /// Expected token savings from caching.
    pub expected_savings_tokens: u64,
    /// Probability that the cached content will be reused `[0.0, 1.0]`.
    pub reuse_probability: f64,
    /// Additional latency introduced by the serialization strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub added_latency_ms: Option<f64>,
    /// Sharing scope for the serialized content.
    pub scope_label: SharingScope,
}

/// Latency and priority routing intent.
///
/// Communicates the caller's latency sensitivity and workflow context
/// to influence scheduling and model selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriorityIntent {
    /// Latency sensitivity score `[0.0, 1.0]` where 1.0 is most sensitive.
    pub latency_sensitivity: f64,
    /// Current workflow phase label (e.g., "research", "synthesis").
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub workflow_phase: Option<String>,
    /// Caller tier label (e.g., "free", "premium", "enterprise").
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub caller_tier: Option<String>,
}

/// Model routing and selection intent.
///
/// Guides backend selection based on task complexity, criticality,
/// and fallback preferences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRoutingIntent {
    /// Target model class for this request.
    pub model_class: ModelClass,
    /// Estimated complexity of the task `[0.0, 1.0]`.
    pub complexity_score: f64,
    /// How critical correct output is `[0.0, 1.0]`.
    pub criticality: f64,
    /// Whether fallback to a lower model class is acceptable.
    pub fallback_allowed: bool,
}

/// Context placement optimization intent.
///
/// Recommends where a prompt block should be placed within the
/// prompt structure for optimal caching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementIntent {
    /// Identifier of the prompt block to place.
    pub block_id: String,
    /// Recommended placement target.
    pub target: PlacementTarget,
    /// Stability score of the block `[0.0, 1.0]`.
    pub stability_score: f64,
    /// Sharing scope for the placed content.
    pub scope_label: SharingScope,
}

/// Cache retention policy intent.
///
/// Recommends how long cached content should be retained based on
/// session patterns and inter-call timing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetentionIntent {
    /// Recommended retention tier.
    pub recommended_tier: RetentionTier,
    /// Expected session duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub expected_session_duration_secs: Option<f64>,
    /// Median inter-call gap in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub inter_call_gap_p50_ms: Option<f64>,
    /// Sharing scope for the retained content.
    pub scope_label: SharingScope,
}

/// Tool scope and phase management intent.
///
/// Communicates which tools are active in the current workflow phase
/// to enable tool schema optimization (e.g., deferred tool blocks).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolScopeIntent {
    /// Tools currently active in this workflow phase.
    pub active_tools: Vec<String>,
    /// Optional label for the current workflow phase.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub phase_label: Option<String>,
    /// Tools deferred to later phases.
    pub deferred_tools: Vec<String>,
}

/// Content compression intent.
///
/// Recommends compression of a prompt block, balancing token savings
/// against information loss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressionIntent {
    /// Identifier of the prompt block to compress.
    pub block_id: String,
    /// Achievable compression ratio `[0.0, 1.0]` where lower is more compressed.
    pub compression_ratio: f64,
    /// Whether the compression is reversible (lossless).
    pub reversible: bool,
    /// Contribution score of this block to output quality `[0.0, 1.0]`.
    pub contribution_score: f64,
}

// ===================================================================
// Main intent enum
// ===================================================================

/// A single optimization intent emitted by a behavioral model.
///
/// Each variant wraps a dedicated payload struct with fields specific
/// to that intent type. The enum uses internally-tagged JSON
/// representation with the `intent_type` field as the discriminant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "intent_type", rename_all = "snake_case")]
pub enum OptimizationIntent {
    /// Cache stability analysis results.
    CacheStability(CacheStabilityIntent),
    /// Content extraction and variable detection.
    ContentExtraction(ContentExtractionIntent),
    /// Serialization and fanout optimization.
    Serialization(SerializationIntent),
    /// Latency and priority routing.
    Priority(PriorityIntent),
    /// Model routing and selection.
    ModelRouting(ModelRoutingIntent),
    /// Context placement optimization.
    Placement(PlacementIntent),
    /// Cache retention policy.
    Retention(RetentionIntent),
    /// Tool scope and phase management.
    ToolScope(ToolScopeIntent),
    /// Content compression.
    Compression(CompressionIntent),
}

impl OptimizationIntent {
    /// Returns the intent type discriminant for this intent variant.
    pub fn discriminant(&self) -> IntentType {
        match self {
            Self::CacheStability(_) => IntentType::CacheStability,
            Self::ContentExtraction(_) => IntentType::ContentExtraction,
            Self::Serialization(_) => IntentType::Serialization,
            Self::Priority(_) => IntentType::Priority,
            Self::ModelRouting(_) => IntentType::ModelRouting,
            Self::Placement(_) => IntentType::Placement,
            Self::Retention(_) => IntentType::Retention,
            Self::ToolScope(_) => IntentType::ToolScope,
            Self::Compression(_) => IntentType::Compression,
        }
    }
}

// ===================================================================
// OptimizationIntentBundle
// ===================================================================

/// A bundle of optimization intents for a single request.
///
/// Bundles are the primary data contract between behavioral models
/// (which emit intents) and the translation layer (which converts
/// intents into provider-specific actions).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationIntentBundle {
    /// Unique identifier for this request.
    pub request_id: Uuid,
    /// Identity of the agent that generated this bundle.
    pub agent_identity: AgentIdentity,
    /// Version of the policy that produced these intents.
    pub policy_version: String,
    /// Ordered list of optimization intents.
    pub intents: Vec<OptimizationIntent>,
    /// When the bundle was created.
    pub created_at: DateTime<Utc>,
}

// ===================================================================
// AgentIdentity
// ===================================================================

/// Identity model for an agent type.
///
/// Used as a key for per-agent policy lookup, behavioral model selection,
/// and telemetry grouping. Two agents with identical identity fields are
/// considered the same agent type.
///
/// # Examples
///
/// ```
/// use nemo_flow_adaptive::acg::AgentIdentity;
/// use std::collections::HashMap;
///
/// let id = AgentIdentity {
///     agent_id: "research".to_string(),
///     template_version: "1.0.0".to_string(),
///     toolset_hash: "abc123".to_string(),
///     model_family: "claude".to_string(),
///     tenant_scope: "acme-corp".to_string(),
/// };
///
/// let mut policies = HashMap::new();
/// policies.insert(id.clone(), "aggressive-caching");
/// assert_eq!(policies.get(&id), Some(&"aggressive-caching"));
/// ```
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct AgentIdentity {
    /// Unique identifier for the agent (e.g., "research-agent").
    pub agent_id: String,
    /// Version of the prompt template in use.
    pub template_version: String,
    /// Hash of the active toolset configuration.
    pub toolset_hash: String,
    /// Model family name (e.g., "claude", "gpt").
    pub model_family: String,
    /// Tenant scope for isolation and access control.
    pub tenant_scope: String,
}

impl std::fmt::Display for AgentIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.agent_id, self.template_version)
    }
}

// ===================================================================
// Translation Report contract
// ===================================================================

/// Outcome status for a single intent translation.
///
/// Plugins return one of these for each intent in the bundle, describing
/// what happened when the plugin tried to express that intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranslationStatus {
    /// Intent was fully expressed in the native API call.
    Applied,
    /// Intent was partially expressed (e.g., reduced breakpoints due to model limits).
    Degraded,
    /// Intent was silently passed through with no action (e.g., not relevant to this backend).
    Ignored,
    /// Intent was actively rejected (e.g., unsafe for this request, feature disabled).
    Rejected,
}

/// Machine-readable reason for the translation outcome.
///
/// Each variant describes WHY an intent received its status. This allows
/// operators to distinguish between plugin limitations, backend limitations,
/// policy decisions, and safety constraints.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum ReasonCode {
    /// Intent was fully supported and applied.
    FullySupported,
    /// Backend does not support this intent type at all.
    UnsupportedByBackend,
    /// Backend supports the intent but the specific model lacks the feature.
    UnsupportedByModel,
    /// Intent was degraded due to backend-specific limits (e.g., max breakpoints).
    BackendLimitReached,
    /// Not enough historical evidence to apply the intent confidently.
    InsufficientEvidence,
    /// The feature is available but administratively disabled.
    FeatureDisabled,
    /// Applying this intent would be unsafe for the current request.
    UnsafeForRequest,
    /// The plugin implementation is incomplete for this intent type.
    PluginIncomplete,
    /// Intent was not relevant to the current request context.
    NotRelevant,
    /// Escape hatch for reason codes not yet in the enum.
    Custom {
        /// Human-readable reason string (for debugging, not machine consumption).
        reason: String,
    },
}

/// Records the outcome of translating a single optimization intent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntentOutcome {
    /// ID of the intent this outcome refers to.
    pub intent_id: Uuid,
    /// Type discriminant of the intent.
    pub intent_type: IntentType,
    /// What happened to this intent.
    pub status: TranslationStatus,
    /// Machine-readable reason for the outcome.
    pub reason: ReasonCode,
    /// Optional human-readable detail (for debugging, not machine consumption).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub detail: Option<String>,
}

/// A plugin's complete report on how it handled an intent bundle.
///
/// Every intent in the input bundle MUST have a corresponding outcome in the report.
/// This is the critical observability contract per the design doc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationReport {
    /// ID of the request this report pertains to.
    pub request_id: Uuid,
    /// Identity of the plugin that produced this report.
    pub plugin_id: String,
    /// Per-intent outcomes.
    pub outcomes: Vec<IntentOutcome>,
    /// When this report was generated.
    pub created_at: DateTime<Utc>,
}

impl TranslationReport {
    /// Returns `true` if every intent was fully applied.
    pub fn all_applied(&self) -> bool {
        self.outcomes
            .iter()
            .all(|o| o.status == TranslationStatus::Applied)
    }

    /// Filter outcomes by status.
    pub fn outcomes_by_status(&self, status: TranslationStatus) -> Vec<&IntentOutcome> {
        self.outcomes
            .iter()
            .filter(|o| o.status == status)
            .collect()
    }

    /// Count of outcomes with the given status.
    pub fn count_by_status(&self, status: TranslationStatus) -> usize {
        self.outcomes.iter().filter(|o| o.status == status).count()
    }

    /// Create a report where all intents are ignored (passthrough behavior).
    ///
    /// Generates one [`IntentOutcome`] per intent in the bundle, each with
    /// [`TranslationStatus::Ignored`] and the given reason code. This is the
    /// standard helper for passthrough and default plugin implementations.
    pub fn all_ignored(
        bundle: &OptimizationIntentBundle,
        plugin_id: &str,
        reason: ReasonCode,
        detail: Option<String>,
    ) -> Self {
        let outcomes = bundle
            .intents
            .iter()
            .map(|intent| IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: intent.discriminant(),
                status: TranslationStatus::Ignored,
                reason: reason.clone(),
                detail: detail.clone(),
            })
            .collect();
        Self {
            request_id: bundle.request_id,
            plugin_id: plugin_id.to_string(),
            outcomes,
            created_at: Utc::now(),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/acg/types_tests.rs"]
mod tests;
