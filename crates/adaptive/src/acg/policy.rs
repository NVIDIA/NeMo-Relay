// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy types for the Adaptive Cache Governor (ACG) system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::acg::profile::SessionArchetype;
use crate::acg::types::{AgentIdentity, ModelClass, RetentionTier, SharingScope};

/// Versioned wrapper for an ACG policy document.
///
/// This envelope binds a concrete policy payload to the agent identity it was
/// derived for and records the schema version and creation timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyEnvelope<T> {
    /// Agent identity the enclosed policy applies to.
    pub agent_identity: AgentIdentity,
    /// Version string for the policy schema or generation pipeline.
    pub policy_version: String,
    /// Timestamp when the policy document was created.
    pub created_at: DateTime<Utc>,
    /// Concrete policy payload.
    pub policy: T,
}

/// Rewrite category that an ACG policy can allow or deny.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformationClass {
    /// Normalize equivalent content into a canonical form.
    Canonicalization,
    /// Extract variable content into placeholders.
    VariableExtraction,
    /// Reorder prompt sections without changing their content.
    SectionReordering,
    /// Promote stable context earlier in the prompt.
    StableContextPromotion,
    /// Move context between placement regions.
    ContextPlacement,
    /// Compress content to reduce prompt size.
    Compression,
    /// Reduce tool-related scope or schema context.
    ToolScopeReduction,
}

/// Policy controlling when ACG outputs can be cached and reused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachePolicy {
    /// Minimum stability score required before caching is allowed.
    pub min_stability_score: f64,
    /// Minimum number of observations required before caching is allowed.
    pub min_evidence_count: u32,
    /// Default sharing scope used for cached artifacts.
    pub default_sharing_scope: SharingScope,
    /// Whether warm-first coordination is enabled for eligible fan-outs.
    pub warm_first_enabled: bool,
    /// Optional upper bound on fan-out width for warm-first coordination.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub max_fanout_for_warm_first: Option<u32>,
}

/// Policy controlling what prompt rewrites the runtime can apply automatically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewritePolicy {
    /// Transformation classes permitted by the policy.
    pub allowed_transformations: Vec<TransformationClass>,
    /// Whether every rewrite must pass a validation step before use.
    pub require_validation: bool,
    /// Highest automatically applicable risk tier, when bounded.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub max_auto_risk_tier: Option<u32>,
}

/// Retention-tier override for one sharing scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScopeRetentionOverride {
    /// Sharing scope the override applies to.
    pub scope: SharingScope,
    /// Retention tier to use for that scope.
    pub tier: RetentionTier,
}

/// Policy controlling how long learned artifacts should be retained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Default retention tier when no scope-specific override matches.
    pub default_tier: RetentionTier,
    /// Optional per-scope overrides for the default tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub scope_overrides: Option<Vec<ScopeRetentionOverride>>,
}

/// Routing override for one observed session archetype.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchetypeRoutingOverride {
    /// Session archetype the override applies to.
    pub archetype: SessionArchetype,
    /// Model class to prefer for that archetype.
    pub model_class: ModelClass,
}

/// Policy controlling model-class routing decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingPolicy {
    /// Default model class to select when no override matches.
    pub default_model_class: ModelClass,
    /// Whether fallback routing is allowed when the preferred class is unavailable.
    pub fallback_allowed: bool,
    /// Optional model-class overrides keyed by session archetype.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub archetype_overrides: Option<Vec<ArchetypeRoutingOverride>>,
    /// Optional session-level cost cap used during routing decisions.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub session_cost_cap: Option<f64>,
}
