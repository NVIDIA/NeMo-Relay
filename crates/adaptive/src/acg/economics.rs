// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal economics-aware breakpoint planning.

use serde_json::json;

use crate::acg::capability::{CacheEconomics, ModelFamilyCapabilities};
use crate::acg::debug as acg_debug;
use crate::acg::profile::StabilityClass;
use crate::acg::prompt_ir::{BlockContentType, PromptIR, PromptRole, ProvenanceLabel};
use crate::acg::stability::StabilityAnalysisResult;
use crate::acg::types::SharingScope;

impl CacheEconomics {
    fn breakeven_reads(&self, write_multiplier: f64) -> f64 {
        let cached_read_delta = 1.0 - self.read_multiplier;
        if cached_read_delta <= 0.0 {
            return f64::INFINITY;
        }
        (write_multiplier - 1.0) / cached_read_delta
    }

    fn marginal_net_savings(&self, tokens: u32, expected_reads: f64) -> f64 {
        let cached_read_delta = 1.0 - self.read_multiplier;
        let write_tax = self.write_short_multiplier - 1.0;
        f64::from(tokens) * ((cached_read_delta * expected_reads) - write_tax)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EconomicsPlan {
    pub planned_breakpoints: Vec<PlannedBreakpoint>,
    pub minimum_cacheable_tokens: u32,
    pub observed_reuse_horizon: f64,
    pub write_5m_breakeven_reads: f64,
    pub write_1h_breakeven_reads: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlannedBreakpoint {
    pub stable_prefix_end: usize,
    pub cumulative_tokens: u32,
    pub marginal_tokens: u32,
    pub expected_reads: f64,
    pub expected_net_savings: f64,
    pub scope: SharingScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateKind {
    User,
    ToolCluster,
    Retrieval,
    System,
    Structured,
    Generic,
}

impl CandidateKind {
    fn label(self) -> &'static str {
        match self {
            Self::User => "user_boundary",
            Self::ToolCluster => "tool_cluster_boundary",
            Self::Retrieval => "retrieval_boundary",
            Self::System => "system_boundary",
            Self::Structured => "structured_boundary",
            Self::Generic => "generic_boundary",
        }
    }

    fn priority(self) -> u8 {
        match self {
            Self::User => 5,
            Self::ToolCluster => 3,
            Self::Retrieval => 4,
            Self::System => 2,
            Self::Structured => 1,
            Self::Generic => 0,
        }
    }
}

#[derive(Debug, Clone)]
struct BreakpointCandidate {
    stable_prefix_end: usize,
    cumulative_tokens: u32,
    weakest_prefix_signal: f64,
    expected_reads: f64,
    kind: CandidateKind,
}

#[derive(Debug, Clone, Copy)]
struct PrefixStats {
    cumulative_tokens: u32,
    weakest_prefix_signal: f64,
}

#[derive(Debug, Clone, Copy)]
struct CandidateSelectionState {
    total_value: f64,
    previous_candidate_index: Option<usize>,
}

pub(crate) fn plan_breakpoints(
    prompt_ir: &PromptIR,
    stability: &StabilityAnalysisResult,
    observation_count: u32,
    capabilities: &ModelFamilyCapabilities,
) -> EconomicsPlan {
    let Some(pricing) = capabilities.cache_economics.as_ref() else {
        return plan_without_pricing(stability, observation_count, capabilities);
    };
    let observed_reuse_horizon = observation_count.saturating_sub(1) as f64;
    let minimum_cacheable_tokens = capabilities.min_cacheable_tokens.unwrap_or(u32::MAX);
    let max_breakpoints = capabilities.max_cache_breakpoints.unwrap_or(0) as usize;

    let mut plan = build_economics_plan(pricing, minimum_cacheable_tokens, observed_reuse_horizon);
    let skip_reasons = breakpoint_skip_reasons(
        observed_reuse_horizon,
        max_breakpoints,
        stability.stable_prefix_length,
        minimum_cacheable_tokens,
    );
    if !skip_reasons.is_empty() {
        acg_debug::emit(
            "economics_plan_skipped",
            json!({
                "reason": skip_reasons,
                "model_family": capabilities.model_family,
                "observation_count": observation_count,
                "observed_reuse_horizon": observed_reuse_horizon,
                "stable_prefix_length": stability.stable_prefix_length,
                "minimum_cacheable_tokens": minimum_cacheable_tokens,
                "max_breakpoints": max_breakpoints,
            }),
        );
        return plan;
    }

    let stable_prefix_end = stability.stable_prefix_length.min(prompt_ir.blocks.len());
    let prefix_stats = build_prefix_stats(
        prompt_ir,
        stability,
        stable_prefix_end,
        minimum_cacheable_tokens,
        observed_reuse_horizon,
        capabilities,
        observation_count,
    );
    let candidates = collect_breakpoint_candidates(
        prompt_ir,
        &prefix_stats,
        stable_prefix_end,
        observed_reuse_horizon,
        minimum_cacheable_tokens,
        capabilities,
    );

    if candidates.is_empty() {
        acg_debug::emit(
            "economics_plan_skipped",
            json!({
                "reason": "no_viable_candidates",
                "model_family": capabilities.model_family,
                "observation_count": observation_count,
                "observed_reuse_horizon": observed_reuse_horizon,
                "stable_prefix_length": stability.stable_prefix_length,
                "minimum_cacheable_tokens": minimum_cacheable_tokens,
            }),
        );
        return plan;
    }

    let Some(selected_candidate_indices) =
        select_breakpoint_candidates(&candidates, max_breakpoints, pricing)
    else {
        acg_debug::emit(
            "economics_plan_skipped",
            json!({
                "reason": "non_positive_expected_net_savings",
                "model_family": capabilities.model_family,
                "observation_count": observation_count,
                "observed_reuse_horizon": observed_reuse_horizon,
                "stable_prefix_length": stability.stable_prefix_length,
                "minimum_cacheable_tokens": minimum_cacheable_tokens,
            }),
        );
        return plan;
    };

    append_selected_breakpoints(
        &mut plan,
        &selected_candidate_indices,
        &candidates,
        pricing,
        capabilities,
        observed_reuse_horizon,
    );
    emit_plan_limit_reached_if_needed(&plan, &candidates, max_breakpoints, capabilities);

    acg_debug::emit(
        "economics_plan_result",
        json!({
            "model_family": capabilities.model_family,
            "observation_count": observation_count,
            "observed_reuse_horizon": observed_reuse_horizon,
            "minimum_cacheable_tokens": minimum_cacheable_tokens,
            "stable_prefix_length": stability.stable_prefix_length,
            "planned_breakpoints": plan
                .planned_breakpoints
                .iter()
                .map(|breakpoint| json!({
                    "stable_prefix_end": breakpoint.stable_prefix_end,
                    "cumulative_tokens": breakpoint.cumulative_tokens,
                    "marginal_tokens": breakpoint.marginal_tokens,
                    "expected_reads": breakpoint.expected_reads,
                    "expected_net_savings": breakpoint.expected_net_savings,
                }))
                .collect::<Vec<_>>(),
        }),
    );

    plan
}

fn plan_without_pricing(
    stability: &StabilityAnalysisResult,
    observation_count: u32,
    capabilities: &ModelFamilyCapabilities,
) -> EconomicsPlan {
    acg_debug::emit(
        "economics_plan_skipped",
        json!({
            "reason": "missing_cache_economics",
            "model_family": capabilities.model_family,
            "observation_count": observation_count,
            "stable_prefix_length": stability.stable_prefix_length,
        }),
    );

    EconomicsPlan {
        planned_breakpoints: Vec::new(),
        minimum_cacheable_tokens: capabilities.min_cacheable_tokens.unwrap_or(u32::MAX),
        observed_reuse_horizon: observation_count.saturating_sub(1) as f64,
        write_5m_breakeven_reads: f64::INFINITY,
        write_1h_breakeven_reads: f64::INFINITY,
    }
}

fn build_economics_plan(
    pricing: &CacheEconomics,
    minimum_cacheable_tokens: u32,
    observed_reuse_horizon: f64,
) -> EconomicsPlan {
    EconomicsPlan {
        planned_breakpoints: Vec::new(),
        minimum_cacheable_tokens,
        observed_reuse_horizon,
        write_5m_breakeven_reads: pricing.breakeven_reads(pricing.write_short_multiplier),
        write_1h_breakeven_reads: pricing
            .write_long_multiplier
            .map_or(f64::INFINITY, |multiplier| {
                pricing.breakeven_reads(multiplier)
            }),
    }
}

fn breakpoint_skip_reasons(
    observed_reuse_horizon: f64,
    max_breakpoints: usize,
    stable_prefix_length: usize,
    minimum_cacheable_tokens: u32,
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if observed_reuse_horizon <= 0.0 {
        reasons.push("reuse_horizon_non_positive");
    }
    if max_breakpoints == 0 {
        reasons.push("max_breakpoints_zero");
    }
    if stable_prefix_length == 0 {
        reasons.push("stable_prefix_empty");
    }
    if minimum_cacheable_tokens == u32::MAX {
        reasons.push("min_cacheable_tokens_unavailable");
    }
    reasons
}

fn append_selected_breakpoints(
    plan: &mut EconomicsPlan,
    selected_candidate_indices: &[usize],
    candidates: &[BreakpointCandidate],
    pricing: &CacheEconomics,
    capabilities: &ModelFamilyCapabilities,
    observed_reuse_horizon: f64,
) {
    let mut previous_cumulative_tokens = 0_u32;

    for &candidate_index in selected_candidate_indices {
        let candidate = &candidates[candidate_index];
        let marginal_tokens = candidate
            .cumulative_tokens
            .saturating_sub(previous_cumulative_tokens);
        let expected_net_savings =
            pricing.marginal_net_savings(marginal_tokens, candidate.expected_reads);

        if expected_net_savings <= 0.0 || marginal_tokens == 0 {
            acg_debug::emit(
                "economics_candidate_rejected",
                json!({
                    "reason": if marginal_tokens == 0 {
                        "zero_marginal_tokens"
                    } else {
                        "non_positive_expected_net_savings"
                    },
                    "model_family": capabilities.model_family,
                    "stable_prefix_end": candidate.stable_prefix_end,
                    "cumulative_tokens": candidate.cumulative_tokens,
                    "marginal_tokens": marginal_tokens,
                    "expected_reads": candidate.expected_reads,
                    "expected_net_savings": expected_net_savings,
                    "candidate_kind": candidate.kind.label(),
                }),
            );
            continue;
        }

        plan.planned_breakpoints.push(PlannedBreakpoint {
            stable_prefix_end: candidate.stable_prefix_end,
            cumulative_tokens: candidate.cumulative_tokens,
            marginal_tokens,
            expected_reads: candidate.expected_reads,
            expected_net_savings,
            scope: SharingScope::Session,
        });
        acg_debug::emit(
            "economics_candidate_selected",
            json!({
                "model_family": capabilities.model_family,
                "stable_prefix_end": candidate.stable_prefix_end,
                "cumulative_tokens": candidate.cumulative_tokens,
                "marginal_tokens": marginal_tokens,
                "observed_reuse_horizon": observed_reuse_horizon,
                "weakest_prefix_signal": candidate.weakest_prefix_signal,
                "expected_reads": candidate.expected_reads,
                "expected_net_savings": expected_net_savings,
                "candidate_kind": candidate.kind.label(),
            }),
        );
        previous_cumulative_tokens = candidate.cumulative_tokens;
    }
}

fn emit_plan_limit_reached_if_needed(
    plan: &EconomicsPlan,
    candidates: &[BreakpointCandidate],
    max_breakpoints: usize,
    capabilities: &ModelFamilyCapabilities,
) {
    if plan.planned_breakpoints.len() >= max_breakpoints && candidates.len() > max_breakpoints {
        acg_debug::emit(
            "economics_plan_limit_reached",
            json!({
                "model_family": capabilities.model_family,
                "max_breakpoints": max_breakpoints,
                "planned_breakpoints": plan.planned_breakpoints.len(),
            }),
        );
    }
}

fn build_prefix_stats(
    prompt_ir: &PromptIR,
    stability: &StabilityAnalysisResult,
    stable_prefix_end: usize,
    minimum_cacheable_tokens: u32,
    observed_reuse_horizon: f64,
    capabilities: &ModelFamilyCapabilities,
    observation_count: u32,
) -> Vec<PrefixStats> {
    let mut stats = Vec::with_capacity(stable_prefix_end);
    let mut cumulative_tokens = 0_u32;
    let mut weakest_prefix_signal = 1.0_f64;

    for (index, score) in stability.scores.iter().take(stable_prefix_end).enumerate() {
        if score.classification != StabilityClass::Stable {
            acg_debug::emit(
                "economics_candidate_rejected",
                json!({
                    "reason": "non_stable_block",
                    "model_family": capabilities.model_family,
                    "block_index": index,
                    "classification": format!("{:?}", score.classification),
                    "score": score.score,
                    "confidence": score.confidence,
                    "observation_count": observation_count,
                }),
            );
            break;
        }

        let Some((tokens, _)) = prompt_ir.blocks[index].estimated_tokens() else {
            acg_debug::emit(
                "economics_candidate_rejected",
                json!({
                    "reason": "token_count_unavailable", "block_index": index,
                    "model_family": capabilities.model_family,
                }),
            );
            break;
        };
        cumulative_tokens = cumulative_tokens.saturating_add(tokens);
        weakest_prefix_signal =
            weakest_prefix_signal.min(prefix_signal(score.score, score.confidence));

        if cumulative_tokens < minimum_cacheable_tokens {
            acg_debug::emit(
                "economics_candidate_rejected",
                json!({
                    "reason": "below_min_cacheable_tokens",
                    "model_family": capabilities.model_family,
                    "block_index": index,
                    "cumulative_tokens": cumulative_tokens,
                    "minimum_cacheable_tokens": minimum_cacheable_tokens,
                    "score": score.score,
                    "confidence": score.confidence,
                    "weakest_prefix_signal": weakest_prefix_signal,
                    "observed_reuse_horizon": observed_reuse_horizon,
                }),
            );
        }

        stats.push(PrefixStats {
            cumulative_tokens,
            weakest_prefix_signal,
        });
    }

    stats
}

fn collect_breakpoint_candidates(
    prompt_ir: &PromptIR,
    prefix_stats: &[PrefixStats],
    stable_prefix_end: usize,
    observed_reuse_horizon: f64,
    minimum_cacheable_tokens: u32,
    capabilities: &ModelFamilyCapabilities,
) -> Vec<BreakpointCandidate> {
    let mut candidates = Vec::new();
    let mut previous_cumulative_tokens = 0_u32;

    let scan_end = prefix_stats.len().min(stable_prefix_end);
    for (index, stats) in prefix_stats.iter().take(scan_end).enumerate() {
        let stats = *stats;
        if stats.cumulative_tokens < minimum_cacheable_tokens {
            previous_cumulative_tokens = stats.cumulative_tokens;
            continue;
        }

        let threshold_crossing = previous_cumulative_tokens < minimum_cacheable_tokens
            && stats.cumulative_tokens >= minimum_cacheable_tokens;
        let kind =
            classify_candidate_boundary(prompt_ir, index, stable_prefix_end, threshold_crossing);
        if let Some(kind) = kind {
            let candidate = BreakpointCandidate {
                stable_prefix_end: index + 1,
                cumulative_tokens: stats.cumulative_tokens,
                weakest_prefix_signal: stats.weakest_prefix_signal,
                expected_reads: observed_reuse_horizon * stats.weakest_prefix_signal,
                kind,
            };
            acg_debug::emit(
                "economics_candidate_generated",
                json!({
                    "model_family": capabilities.model_family,
                    "block_index": index,
                    "stable_prefix_end": candidate.stable_prefix_end,
                    "cumulative_tokens": candidate.cumulative_tokens,
                    "expected_reads": candidate.expected_reads,
                    "weakest_prefix_signal": candidate.weakest_prefix_signal,
                    "candidate_kind": candidate.kind.label(),
                }),
            );
            candidates.push(candidate);
        } else {
            acg_debug::emit(
                "economics_candidate_rejected",
                json!({
                    "reason": "non_semantic_internal_boundary",
                    "model_family": capabilities.model_family,
                    "block_index": index,
                    "cumulative_tokens": stats.cumulative_tokens,
                }),
            );
        }

        previous_cumulative_tokens = stats.cumulative_tokens;
    }

    candidates
}

fn classify_candidate_boundary(
    prompt_ir: &PromptIR,
    index: usize,
    stable_prefix_end: usize,
    threshold_crossing: bool,
) -> Option<CandidateKind> {
    let block = prompt_ir.blocks.get(index)?;
    let next = prompt_ir.blocks.get(index + 1);
    let next_is_tool_schema =
        next.is_some_and(|candidate| candidate.content_type == BlockContentType::ToolSchema);

    if block.content_type == BlockContentType::ToolSchema
        && next_is_tool_schema
        && !threshold_crossing
    {
        return None;
    }

    let kind = match (block.role, block.content_type) {
        (_, BlockContentType::ToolSchema) => CandidateKind::ToolCluster,
        _ if block.provenance == ProvenanceLabel::Retrieval => CandidateKind::Retrieval,
        (PromptRole::User, _) => CandidateKind::User,
        (
            _,
            BlockContentType::StructuredOutput
            | BlockContentType::StructuredInput
            | BlockContentType::Opaque
            | BlockContentType::ToolResult
            | BlockContentType::Image,
        ) => CandidateKind::Structured,
        (PromptRole::System, _) => CandidateKind::System,
        _ if index + 1 >= stable_prefix_end => CandidateKind::Generic,
        _ => CandidateKind::Generic,
    };

    Some(kind)
}

fn select_breakpoint_candidates(
    candidates: &[BreakpointCandidate],
    max_breakpoints: usize,
    pricing: &CacheEconomics,
) -> Option<Vec<usize>> {
    if candidates.is_empty() || max_breakpoints == 0 {
        return None;
    }

    let mut dp = vec![vec![None; max_breakpoints + 1]; candidates.len()];
    populate_candidate_selection_dp(&mut dp, candidates, max_breakpoints, pricing);
    let (candidate_index, breakpoint_count, _) = best_terminal_candidate(&dp, candidates)?;

    Some(reconstruct_selected_candidates(
        &dp,
        candidate_index,
        breakpoint_count,
    ))
}

fn populate_candidate_selection_dp(
    dp: &mut [Vec<Option<CandidateSelectionState>>],
    candidates: &[BreakpointCandidate],
    max_breakpoints: usize,
    pricing: &CacheEconomics,
) {
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        seed_candidate_selection_state(dp, candidate_index, candidate, pricing);
        extend_candidate_selection_state(dp, candidate_index, candidates, max_breakpoints, pricing);
    }
}

fn seed_candidate_selection_state(
    dp: &mut [Vec<Option<CandidateSelectionState>>],
    candidate_index: usize,
    candidate: &BreakpointCandidate,
    pricing: &CacheEconomics,
) {
    let base_value =
        pricing.marginal_net_savings(candidate.cumulative_tokens, candidate.expected_reads);
    if base_value > 0.0 {
        dp[candidate_index][1] = Some(CandidateSelectionState {
            total_value: base_value,
            previous_candidate_index: None,
        });
    }
}

fn extend_candidate_selection_state(
    dp: &mut [Vec<Option<CandidateSelectionState>>],
    candidate_index: usize,
    candidates: &[BreakpointCandidate],
    max_breakpoints: usize,
    pricing: &CacheEconomics,
) {
    for breakpoint_count in 2..=max_breakpoints.min(candidate_index + 1) {
        for previous_index in 0..candidate_index {
            let Some(proposal) = candidate_extension_proposal(
                dp,
                candidates,
                candidate_index,
                breakpoint_count,
                previous_index,
                pricing,
            ) else {
                continue;
            };

            if is_better_candidate_state(
                dp[candidate_index][breakpoint_count],
                proposal,
                candidates,
                candidate_index,
            ) {
                dp[candidate_index][breakpoint_count] = Some(proposal);
            }
        }
    }
}

fn candidate_extension_proposal(
    dp: &[Vec<Option<CandidateSelectionState>>],
    candidates: &[BreakpointCandidate],
    candidate_index: usize,
    breakpoint_count: usize,
    previous_index: usize,
    pricing: &CacheEconomics,
) -> Option<CandidateSelectionState> {
    let previous_state = dp[previous_index][breakpoint_count - 1]?;
    let marginal_tokens = candidates[candidate_index]
        .cumulative_tokens
        .saturating_sub(candidates[previous_index].cumulative_tokens);
    if marginal_tokens == 0 {
        return None;
    }

    let segment_value =
        pricing.marginal_net_savings(marginal_tokens, candidates[candidate_index].expected_reads);
    if segment_value <= 0.0 {
        return None;
    }

    Some(CandidateSelectionState {
        total_value: previous_state.total_value + segment_value,
        previous_candidate_index: Some(previous_index),
    })
}

fn best_terminal_candidate(
    dp: &[Vec<Option<CandidateSelectionState>>],
    candidates: &[BreakpointCandidate],
) -> Option<(usize, usize, CandidateSelectionState)> {
    let mut best_terminal: Option<(usize, usize, CandidateSelectionState)> = None;

    for (candidate_index, row) in dp.iter().enumerate() {
        for (breakpoint_count, state) in row.iter().enumerate().skip(1) {
            let Some(state) = state else {
                continue;
            };
            if is_better_terminal_state(
                best_terminal,
                (candidate_index, breakpoint_count, *state),
                candidates,
            ) {
                best_terminal = Some((candidate_index, breakpoint_count, *state));
            }
        }
    }

    best_terminal
}

fn reconstruct_selected_candidates(
    dp: &[Vec<Option<CandidateSelectionState>>],
    mut candidate_index: usize,
    mut breakpoint_count: usize,
) -> Vec<usize> {
    let mut selected = Vec::new();
    while breakpoint_count > 0 {
        selected.push(candidate_index);
        let state = dp[candidate_index][breakpoint_count]
            .expect("terminal DP state must be present during reconstruction");
        if let Some(previous_index) = state.previous_candidate_index {
            candidate_index = previous_index;
            breakpoint_count -= 1;
        } else {
            break;
        }
    }

    selected.reverse();
    selected
}

fn is_better_candidate_state(
    current: Option<CandidateSelectionState>,
    proposal: CandidateSelectionState,
    _candidates: &[BreakpointCandidate],
    _proposal_index: usize,
) -> bool {
    let Some(current) = current else {
        return true;
    };

    if proposal.total_value > current.total_value + 1e-9 {
        return true;
    }
    if current.total_value > proposal.total_value + 1e-9 {
        return false;
    }

    proposal.previous_candidate_index > current.previous_candidate_index
}

fn is_better_terminal_state(
    current: Option<(usize, usize, CandidateSelectionState)>,
    proposal: (usize, usize, CandidateSelectionState),
    candidates: &[BreakpointCandidate],
) -> bool {
    let Some((current_index, current_breakpoints, current_state)) = current else {
        return true;
    };
    let (proposal_index, proposal_breakpoints, proposal_state) = proposal;

    if proposal_state.total_value > current_state.total_value + 1e-9 {
        return true;
    }
    if current_state.total_value > proposal_state.total_value + 1e-9 {
        return false;
    }

    if proposal_breakpoints < current_breakpoints {
        return true;
    }
    if proposal_breakpoints > current_breakpoints {
        return false;
    }

    if candidates[proposal_index].kind.priority() > candidates[current_index].kind.priority() {
        return true;
    }
    if candidates[proposal_index].kind.priority() < candidates[current_index].kind.priority() {
        return false;
    }

    candidates[proposal_index].stable_prefix_end > candidates[current_index].stable_prefix_end
}

fn prefix_signal(score: f64, confidence: f64) -> f64 {
    score.clamp(0.0, 1.0) * confidence.clamp(0.0, 1.0)
}

#[cfg(test)]
#[path = "../../tests/unit/acg/economics_internal_tests.rs"]
mod tests;
