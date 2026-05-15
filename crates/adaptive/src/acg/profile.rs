// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral profile types for the Adaptive Cache Governor (ACG) system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::acg::prompt_ir::SpanId;
use crate::acg::types::AgentIdentity;

/// Stability bucket assigned to a prompt block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StabilityClass {
    /// The block is highly consistent across observations.
    Stable,
    /// The block is somewhat consistent but still changes meaningfully.
    SemiStable,
    /// The block changes enough that it should be treated as variable.
    Variable,
}

/// Stability analysis result for one prompt block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockStabilityScore {
    /// Span identifier of the analyzed prompt block.
    pub span_id: SpanId,
    /// Stability bucket assigned to the block.
    pub classification: StabilityClass,
    /// Effective stability score in the range `[0.0, 1.0]`.
    pub score: f64,
    /// Confidence score in the range `[0.0, 1.0]`.
    pub confidence: f64,
    /// Number of observations that contained this block.
    pub observation_count: u32,
}

/// Percentile summary for a duration-like metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionSummary {
    /// 50th percentile value.
    pub p50: f64,
    /// 90th percentile value.
    pub p90: f64,
    /// 99th percentile value.
    pub p99: f64,
    /// Number of samples summarized by the distribution.
    pub sample_count: u32,
}

/// Coarse behavioral archetype inferred from observed runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionArchetype {
    /// Short sessions dominated by direct answers.
    FastAnswer,
    /// Sessions that repeatedly call tools in loops or fan-outs.
    ToolHeavyLoop,
    /// Longer workflows with extended execution lifetime.
    LongRunningWorkflow,
    /// Multi-turn diagnostic or debugging sessions.
    MultiTurnTroubleshooting,
}

/// Summary of observed parallelism behavior across runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParallelismPattern {
    /// Whether any tool fan-outs were observed.
    pub has_fanouts: bool,
    /// Typical observed fan-out width, when fan-outs were present.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub typical_fanout_width: Option<u32>,
    /// Whether the workflow was mostly serial despite any fan-outs.
    pub predominantly_serial: bool,
}

/// One phase of tool usage observed in an agent workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolUsagePhase {
    /// Human-readable label for the phase.
    pub phase_label: String,
    /// Tool names commonly observed in the phase.
    pub tools: Vec<String>,
    /// Fraction of runs that reached this phase.
    pub phase_reach_rate: f64,
}

/// Aggregated behavioral profile derived from observed runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehavioralProfile {
    /// Agent identity the profile applies to.
    pub agent_identity: AgentIdentity,
    /// Version string for the profile schema or derivation pipeline.
    pub profile_version: String,
    /// Stability summary for prompt blocks observed across runs.
    pub block_stability: Vec<BlockStabilityScore>,
    /// Observed session-duration distribution, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub session_duration: Option<DistributionSummary>,
    /// Observed inter-call-gap distribution, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub inter_call_gap: Option<DistributionSummary>,
    /// Observed parallelism behavior, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub parallelism: Option<ParallelismPattern>,
    /// Observed tool-usage phases.
    pub tool_usage_phases: Vec<ToolUsagePhase>,
    /// Dominant session archetype, when enough data exists to infer one.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub dominant_archetype: Option<SessionArchetype>,
    /// Number of observations used to derive the profile.
    pub observation_count: u32,
    /// Minimum number of observations required for the profile to be considered usable.
    pub minimum_observations: u32,
    /// Timestamp when the profile was last updated.
    pub updated_at: DateTime<Utc>,
}

impl BehavioralProfile {
    /// Report whether the profile has enough data to be trusted.
    ///
    /// # Returns
    /// `true` when [`Self::observation_count`] is at least
    /// [`Self::minimum_observations`] and `false` otherwise.
    pub fn has_sufficient_data(&self) -> bool {
        self.observation_count >= self.minimum_observations
    }
}
