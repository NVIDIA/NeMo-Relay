// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Managed, bounded LLM optimization accounting.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde::Serialize;
use uuid::Uuid;

use crate::codec::optimization::{
    LlmOptimizationContribution, LlmOptimizationModel, LlmOptimizationSummary,
    LlmOptimizationSummaryStatus, LlmOptimizationTokens,
};
use crate::codec::response::{AnnotatedLlmResponse, CostSource, PricingResolver};

/// Maximum contributions retained for one LLM call.
pub const MAX_LLM_OPTIMIZATION_CONTRIBUTIONS: usize = 64;
/// Maximum serialized size of one complete contribution envelope.
pub const MAX_LLM_OPTIMIZATION_CONTRIBUTION_BYTES: usize = 16 * 1024;
/// Maximum aggregate serialized size of all contribution envelopes for one call.
pub const MAX_LLM_OPTIMIZATION_TOTAL_CONTRIBUTION_BYTES: usize = 256 * 1024;
/// Maximum contribution records inspected before the recorder seals itself.
///
/// Invalid records count toward this bound even though they do not consume an
/// accepted sequence number.
pub const MAX_LLM_OPTIMIZATION_CONTRIBUTION_ATTEMPTS: usize = 64;

#[derive(Debug, Default)]
struct AccumulatorState {
    contributions: Vec<LlmOptimizationContribution>,
    recorded_at: Vec<DateTime<Utc>>,
    total_contribution_bytes: usize,
    attempted_contributions: usize,
    emitted: usize,
    closed: bool,
    finished: bool,
    limitations: BTreeSet<String>,
    contribution_limit_exceeded: bool,
    invalid_payload_schema: bool,
}

/// Cloneable capability for adding evidence to the current managed LLM call.
///
/// A streaming execution intercept may capture this value before returning its
/// stream and use it when the route is committed by the first upstream item.
#[derive(Debug, Clone, Default)]
pub struct LlmOptimizationRecorder {
    state: Arc<Mutex<AccumulatorState>>,
}

impl LlmOptimizationRecorder {
    /// Record one contribution without blocking on I/O or exporter delivery.
    ///
    /// Returns `false` when the contribution is rejected by a payload/schema
    /// invariant, a per-call bound, or because accounting has already closed.
    /// Rejection never affects LLM execution and does not consume a sequence.
    #[must_use]
    pub fn record(&self, mut contribution: LlmOptimizationContribution) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if state.closed {
            return false;
        }
        if state.attempted_contributions >= MAX_LLM_OPTIMIZATION_CONTRIBUTION_ATTEMPTS {
            seal_for_contribution_limit(&mut state);
            return false;
        }
        state.attempted_contributions += 1;
        drop(state);

        match contribution.payload.as_ref() {
            Some(_payload) if contribution.payload_schema.is_none() => {
                if let Ok(mut state) = self.state.lock()
                    && !state.closed
                {
                    state.invalid_payload_schema = true;
                }
                return false;
            }
            _ => {}
        }

        // Relay always replaces producer-supplied identity. Serialization is
        // deliberately outside the accumulator lock; if another writer wins
        // the next sequence while we measure, retry with the new sequence.
        contribution.id = Some(Uuid::now_v7());
        loop {
            let sequence = {
                let Ok(state) = self.state.lock() else {
                    return false;
                };
                if state.closed {
                    return false;
                }
                if state.contributions.len() >= MAX_LLM_OPTIMIZATION_CONTRIBUTIONS {
                    drop(state);
                    self.note_contribution_limit_exceeded();
                    return false;
                }
                state.contributions.len() as u64
            };
            contribution.sequence = Some(sequence);

            let contribution_bytes =
                match bounded_json_size(&contribution, MAX_LLM_OPTIMIZATION_CONTRIBUTION_BYTES) {
                    Ok(size) => size,
                    Err(SerializedSizeError::LimitExceeded) => {
                        self.note_contribution_limit_exceeded();
                        return false;
                    }
                    Err(SerializedSizeError::Serialization) => {
                        if let Ok(mut state) = self.state.lock()
                            && !state.closed
                        {
                            state.invalid_payload_schema = true;
                        }
                        return false;
                    }
                };

            let Ok(mut state) = self.state.lock() else {
                return false;
            };
            if state.closed {
                return false;
            }
            if state.contributions.len() as u64 != sequence {
                continue;
            }
            let Some(total_contribution_bytes) = state
                .total_contribution_bytes
                .checked_add(contribution_bytes)
            else {
                seal_for_contribution_limit(&mut state);
                return false;
            };
            if total_contribution_bytes > MAX_LLM_OPTIMIZATION_TOTAL_CONTRIBUTION_BYTES {
                seal_for_contribution_limit(&mut state);
                return false;
            }

            state.total_contribution_bytes = total_contribution_bytes;
            state.contributions.push(contribution);
            state.recorded_at.push(Utc::now());
            return true;
        }
    }

    fn note_contribution_limit_exceeded(&self) {
        if let Ok(mut state) = self.state.lock()
            && !state.closed
        {
            seal_for_contribution_limit(&mut state);
        }
    }

    pub(crate) fn record_all(
        &self,
        contributions: impl IntoIterator<Item = LlmOptimizationContribution>,
    ) {
        for contribution in contributions {
            if !self.record(contribution) && self.is_closed() {
                break;
            }
        }
    }

    fn is_closed(&self) -> bool {
        self.state.lock().map(|state| state.closed).unwrap_or(true)
    }

    /// Snapshot contributions not yet accepted by mark delivery.
    ///
    /// This does not move the cursor. Call [`Self::mark_emitted`] only after
    /// the asynchronous dispatcher accepts an item.
    #[cfg(test)]
    pub(crate) fn unemitted(&self) -> Vec<LlmOptimizationContribution> {
        self.unemitted_with_timestamps()
            .into_iter()
            .map(|(contribution, _)| contribution)
            .collect()
    }

    /// Snapshot unacknowledged contributions with their acceptance time.
    ///
    /// The timestamp is captured only after the contribution wins its final
    /// sequence and size checks, so execution-time marks retain commit-time
    /// ordering even when they are emitted at the LLM close boundary.
    pub(crate) fn unemitted_with_timestamps(
        &self,
    ) -> Vec<(LlmOptimizationContribution, DateTime<Utc>)> {
        let Ok(state) = self.state.lock() else {
            return Vec::new();
        };
        let start = state.emitted.min(state.contributions.len());
        state.contributions[start..]
            .iter()
            .cloned()
            .zip(state.recorded_at[start..].iter().copied())
            .collect()
    }

    /// Advance the delivery cursor for a bounded number of accepted marks.
    pub(crate) fn mark_emitted(&self, count: usize) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state.emitted = state
            .emitted
            .saturating_add(count)
            .min(state.contributions.len());
    }

    /// Add a best-effort lifecycle limitation to the eventual summary.
    #[cfg(test)]
    pub(crate) fn note_limitation(&self, limitation: impl Into<String>) {
        if let Ok(mut state) = self.state.lock()
            && !state.closed
        {
            state.limitations.insert(limitation.into());
        }
    }

    /// Atomically seal contribution acceptance at an LLM close boundary.
    ///
    /// When `conditional_limitation` is supplied, it is added only if the call
    /// already has optimization evidence or accounting limitations. This keeps
    /// an interrupted but otherwise unoptimized stream from manufacturing an
    /// optimization summary.
    pub(crate) fn close_for_finalization(&self, conditional_limitation: Option<&str>) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if state.finished {
            return false;
        }
        let has_evidence = state.has_evidence();
        if has_evidence && let Some(limitation) = conditional_limitation {
            state.limitations.insert(limitation.to_string());
        }
        state.closed = true;
        has_evidence
    }

    fn finish(&self) -> FinishedContributions {
        let Ok(mut state) = self.state.lock() else {
            return FinishedContributions {
                contributions: Vec::new(),
                limitations: vec!["optimization_accumulator_unavailable".to_string()],
            };
        };
        if state.finished {
            return FinishedContributions {
                contributions: Vec::new(),
                limitations: Vec::new(),
            };
        }
        state.closed = true;
        state.finished = true;
        let mut limitations = std::mem::take(&mut state.limitations)
            .into_iter()
            .collect::<Vec<_>>();
        if state.contribution_limit_exceeded {
            limitations.push("contribution_limit_exceeded".to_string());
            state.contribution_limit_exceeded = false;
        }
        if state.invalid_payload_schema {
            limitations.push("invalid_contribution_payload_schema".to_string());
            state.invalid_payload_schema = false;
        }
        state.recorded_at.clear();
        FinishedContributions {
            contributions: std::mem::take(&mut state.contributions),
            limitations,
        }
    }
}

impl AccumulatorState {
    fn has_evidence(&self) -> bool {
        !self.contributions.is_empty()
            || !self.limitations.is_empty()
            || self.contribution_limit_exceeded
            || self.invalid_payload_schema
    }
}

fn seal_for_contribution_limit(state: &mut AccumulatorState) {
    state.contribution_limit_exceeded = true;
    state.closed = true;
}

#[derive(Debug)]
enum SerializedSizeError {
    LimitExceeded,
    Serialization,
}

fn bounded_json_size<T: Serialize>(value: &T, limit: usize) -> Result<usize, SerializedSizeError> {
    struct CountingWriter {
        size: usize,
        limit: usize,
        exceeded: bool,
    }

    impl std::io::Write for CountingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.size.saturating_add(bytes.len()) > self.limit {
                self.exceeded = true;
                return Err(std::io::Error::other(
                    "optimization contribution limit exceeded",
                ));
            }
            self.size += bytes.len();
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut writer = CountingWriter {
        size: 0,
        limit,
        exceeded: false,
    };
    if serde_json::to_writer(&mut writer, value).is_err() {
        return Err(if writer.exceeded {
            SerializedSizeError::LimitExceeded
        } else {
            SerializedSizeError::Serialization
        });
    }
    Ok(writer.size)
}

struct FinishedContributions {
    contributions: Vec<LlmOptimizationContribution>,
    limitations: Vec<String>,
}

tokio::task_local! {
    static CURRENT_LLM_OPTIMIZATION_RECORDER: LlmOptimizationRecorder;
}

/// Return a recorder for the current execution intercept, if it is managed by Relay.
#[must_use]
pub fn current_llm_optimization_recorder() -> Option<LlmOptimizationRecorder> {
    CURRENT_LLM_OPTIMIZATION_RECORDER
        .try_with(Clone::clone)
        .ok()
}

/// Best-effort shorthand for recording evidence on the current managed call.
#[must_use]
pub fn record_llm_optimization_contribution(contribution: LlmOptimizationContribution) -> bool {
    current_llm_optimization_recorder().is_some_and(|recorder| recorder.record(contribution))
}

pub(crate) async fn scope_llm_optimization_recorder<F: std::future::Future>(
    recorder: LlmOptimizationRecorder,
    future: F,
) -> F::Output {
    CURRENT_LLM_OPTIMIZATION_RECORDER
        .scope(recorder, future)
        .await
}

pub(crate) fn finalize_optimization_summary(
    recorder: &LlmOptimizationRecorder,
    mut response: Option<&mut AnnotatedLlmResponse>,
    requested_model: Option<&str>,
    pricing: &PricingResolver,
) -> Option<LlmOptimizationSummary> {
    let finished = recorder.finish();
    if finished.contributions.is_empty() && finished.limitations.is_empty() {
        return None;
    }

    let applied_routing = finished
        .contributions
        .iter()
        .filter(|contribution| contribution.applied)
        .filter(|contribution| {
            contribution.kind.as_str()
                == crate::codec::optimization::LlmOptimizationKind::MODEL_ROUTING
        })
        .collect::<Vec<_>>();
    let mut limitations = finished.limitations;
    let routing_ambiguous = applied_routing.len() > 1;
    if routing_ambiguous {
        limitations.push("multiple_routing_contributions".to_string());
    }

    let mut token_totals = CheckedTokenTotals::default();
    for contribution in finished
        .contributions
        .iter()
        .filter(|contribution| contribution.applied)
    {
        let is_routing = contribution.kind.as_str()
            == crate::codec::optimization::LlmOptimizationKind::MODEL_ROUTING;
        if routing_ambiguous && is_routing {
            continue;
        }
        if let Some(saved) = contribution
            .token_impact
            .as_ref()
            .and_then(|impact| impact.saved.as_ref())
        {
            token_totals.add_contribution(saved);
        }
    }
    let mut token_count_overflow = token_totals.overflow.any();
    if token_totals.missing_total {
        limitations.push("missing_token_savings_total".to_string());
    }
    if token_totals.inconsistent_total {
        limitations.push("inconsistent_token_savings_total".to_string());
    }
    let tokens_saved = token_totals.values.clone();

    let authoritative_transition = (applied_routing.len() == 1)
        .then(|| applied_routing[0].model_transition.as_ref())
        .flatten();
    let mut baseline_model = authoritative_transition.and_then(|route| route.baseline.clone());
    let contributed_effective_model =
        authoritative_transition.and_then(|route| route.effective.clone());

    // An applied routing contribution names the model Relay actually
    // dispatched. Prefer it over provider response aliases or deployment
    // names; fall back to response/request attribution when no router applies.
    let effective_model = contributed_effective_model
        .or_else(|| {
            response
                .as_ref()
                .and_then(|response| response.model.as_ref())
                .map(|model| LlmOptimizationModel::new(model.clone()))
        })
        .or_else(|| requested_model.map(LlmOptimizationModel::new));
    if (applied_routing.is_empty() || routing_ambiguous) && baseline_model.is_none() {
        baseline_model = effective_model.clone();
    }

    let mut effective_usage = response
        .as_ref()
        .and_then(|response| response.usage.clone());
    let mut baseline_derivation_incomplete =
        token_totals.missing_total || token_totals.inconsistent_total;
    if let Some(usage) = effective_usage.as_mut() {
        if usage.prompt_tokens.is_none() {
            limitations.push("missing_effective_prompt_tokens".to_string());
        }
        if usage.completion_tokens.is_none() {
            limitations.push("missing_effective_completion_tokens".to_string());
        }
        if usage.total_tokens.is_none() {
            match (usage.prompt_tokens, usage.completion_tokens) {
                (Some(prompt), Some(completion)) => match prompt.checked_add(completion) {
                    Some(total) => usage.total_tokens = Some(total),
                    None => token_count_overflow = true,
                },
                _ => limitations.push("missing_effective_total_tokens".to_string()),
            }
        }
    }
    if let (Some(inferred), Some(response_usage)) = (
        effective_usage
            .as_ref()
            .and_then(|usage| usage.total_tokens),
        response
            .as_mut()
            .and_then(|response| response.usage.as_mut()),
    ) && response_usage.total_tokens.is_none()
    {
        response_usage.total_tokens = Some(inferred);
    }
    let baseline_usage = effective_usage.as_ref().map(|usage| {
        let mut baseline = usage.clone();
        baseline.cost = None;
        token_count_overflow |= checked_add_observed_tokens(
            &mut baseline.prompt_tokens,
            tokens_saved.prompt_tokens,
            token_totals.overflow.prompt,
            "missing_effective_prompt_tokens",
            &mut limitations,
            &mut baseline_derivation_incomplete,
        );
        token_count_overflow |= checked_add_observed_tokens(
            &mut baseline.completion_tokens,
            tokens_saved.completion_tokens,
            token_totals.overflow.completion,
            "missing_effective_completion_tokens",
            &mut limitations,
            &mut baseline_derivation_incomplete,
        );
        token_count_overflow |= checked_add_observed_tokens(
            &mut baseline.cache_read_tokens,
            tokens_saved.cache_read_tokens,
            token_totals.overflow.cache_read,
            "missing_effective_cache_read_tokens",
            &mut limitations,
            &mut baseline_derivation_incomplete,
        );
        token_count_overflow |= checked_add_observed_tokens(
            &mut baseline.cache_write_tokens,
            tokens_saved.cache_write_tokens,
            token_totals.overflow.cache_write,
            "missing_effective_cache_write_tokens",
            &mut limitations,
            &mut baseline_derivation_incomplete,
        );
        token_count_overflow |= checked_add_observed_tokens(
            &mut baseline.total_tokens,
            tokens_saved.total_tokens,
            token_totals.overflow.total,
            "missing_effective_total_tokens",
            &mut limitations,
            &mut baseline_derivation_incomplete,
        );
        baseline
    });
    if token_count_overflow {
        limitations.push("token_count_overflow".to_string());
    }

    // A provider-reported amount remains authoritative. A model-pricing
    // estimate may have been calculated from a provider alias, so recompute
    // it against the route Relay actually dispatched.
    let provider_reported_cost = effective_usage
        .as_ref()
        .and_then(|usage| usage.cost.as_ref())
        .filter(|cost| cost.source == CostSource::ProviderReported)
        .cloned();
    let complete_core_usage = effective_usage
        .as_ref()
        .is_some_and(|usage| usage.prompt_tokens.is_some() && usage.completion_tokens.is_some());
    let actual_cost = provider_reported_cost.or_else(|| {
        if !complete_core_usage {
            return None;
        }
        let model = effective_model.as_ref()?;
        let usage = effective_usage.as_ref()?;
        pricing.estimate_cost_for_provider(model.provider.as_deref(), &model.model, usage)
    });
    if let Some(usage) = effective_usage.as_mut() {
        usage.cost.clone_from(&actual_cost);
    }
    if let Some(usage) = response
        .as_mut()
        .and_then(|response| response.usage.as_mut())
    {
        usage.cost.clone_from(&actual_cost);
    }

    let baseline_cost =
        (!token_count_overflow && !baseline_derivation_incomplete && complete_core_usage)
            .then_some(baseline_model.as_ref())
            .flatten()
            .and_then(|model| {
                pricing.estimate_cost_for_provider(
                    model.provider.as_deref(),
                    &model.model,
                    baseline_usage.as_ref()?,
                )
            });

    if effective_usage.is_none() {
        limitations.push("missing_effective_usage".to_string());
    }
    if effective_model.is_none() {
        limitations.push("missing_effective_model".to_string());
    }
    if baseline_model.is_none() {
        limitations.push("missing_baseline_model".to_string());
    }
    if baseline_cost.is_none()
        && baseline_model.is_some()
        && !token_count_overflow
        && !baseline_derivation_incomplete
        && complete_core_usage
    {
        limitations.push("missing_baseline_pricing".to_string());
    }
    if actual_cost.is_none() {
        limitations.push("missing_actual_cost".to_string());
    }

    let (estimated_cost_saved, currency) = calculate_estimated_cost_saved(
        baseline_cost.as_ref(),
        actual_cost.as_ref(),
        &mut limitations,
    );

    limitations.sort();
    limitations.dedup();
    let summary = LlmOptimizationSummary {
        schema_version: "1".to_string(),
        calculation_version: "1".to_string(),
        status: if limitations.is_empty() {
            LlmOptimizationSummaryStatus::Complete
        } else {
            LlmOptimizationSummaryStatus::Partial
        },
        limitations,
        baseline_model,
        effective_model,
        effective_usage,
        baseline_usage,
        tokens_saved,
        baseline_cost,
        actual_cost,
        estimated_cost_saved,
        currency,
        contributions: finished.contributions,
    };
    if let Some(response) = response {
        response.optimization_summary = Some(summary.clone());
    }
    Some(summary)
}

fn calculate_estimated_cost_saved(
    baseline_cost: Option<&crate::codec::response::CostEstimate>,
    actual_cost: Option<&crate::codec::response::CostEstimate>,
    limitations: &mut Vec<String>,
) -> (Option<f64>, Option<String>) {
    let baseline_total = baseline_cost.and_then(|cost| cost.total_or_component_sum());
    let actual_total = actual_cost.and_then(|cost| cost.total_or_component_sum());
    if baseline_cost.is_some() && baseline_total.is_none() {
        limitations.push("missing_baseline_cost_total".to_string());
    }
    if actual_cost.is_some() && actual_total.is_none() {
        limitations.push("missing_actual_cost_total".to_string());
    }

    match (baseline_cost, actual_cost) {
        (Some(baseline), Some(actual))
            if baseline.currency.eq_ignore_ascii_case(&actual.currency) =>
        {
            let saved = baseline_total
                .zip(actual_total)
                .map(|(baseline, actual)| baseline - actual);
            let currency = saved.is_some().then(|| baseline.currency.clone());
            (saved, currency)
        }
        (Some(_), Some(_)) => {
            limitations.push("cost_currency_mismatch".to_string());
            (None, None)
        }
        _ => (None, None),
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TokenOverflow {
    prompt: bool,
    completion: bool,
    cache_read: bool,
    cache_write: bool,
    total: bool,
}

impl TokenOverflow {
    fn any(self) -> bool {
        self.prompt || self.completion || self.cache_read || self.cache_write || self.total
    }
}

#[derive(Debug, Default)]
struct CheckedTokenTotals {
    values: LlmOptimizationTokens,
    overflow: TokenOverflow,
    missing_total: bool,
    inconsistent_total: bool,
}

impl CheckedTokenTotals {
    fn add_contribution(&mut self, other: &LlmOptimizationTokens) {
        checked_accumulate(
            &mut self.values.prompt_tokens,
            other.prompt_tokens,
            &mut self.overflow.prompt,
        );
        checked_accumulate(
            &mut self.values.completion_tokens,
            other.completion_tokens,
            &mut self.overflow.completion,
        );
        checked_accumulate(
            &mut self.values.cache_read_tokens,
            other.cache_read_tokens,
            &mut self.overflow.cache_read,
        );
        checked_accumulate(
            &mut self.values.cache_write_tokens,
            other.cache_write_tokens,
            &mut self.overflow.cache_write,
        );

        let (derived_total, derived_overflow) =
            checked_option_sum([other.prompt_tokens, other.completion_tokens]);
        self.overflow.total |= derived_overflow;
        if derived_overflow {
            self.values.total_tokens = None;
        }
        if let (Some(explicit), Some(prompt), Some(completion)) = (
            other.total_tokens,
            other.prompt_tokens,
            other.completion_tokens,
        ) && prompt
            .checked_add(completion)
            .is_some_and(|derived| derived != explicit)
        {
            self.inconsistent_total = true;
        }
        let contribution_total = other.total_tokens.or(derived_total);
        if contribution_total.is_none() {
            self.missing_total = true;
        }
        checked_accumulate(
            &mut self.values.total_tokens,
            contribution_total,
            &mut self.overflow.total,
        );
    }
}

fn checked_accumulate(target: &mut Option<u64>, value: Option<u64>, overflowed: &mut bool) {
    if *overflowed {
        return;
    }
    let Some(value) = value else {
        return;
    };
    match target.unwrap_or(0).checked_add(value) {
        Some(total) => *target = Some(total),
        None => {
            *target = None;
            *overflowed = true;
        }
    }
}

fn checked_add_observed_tokens(
    target: &mut Option<u64>,
    value: Option<u64>,
    value_overflowed: bool,
    missing_limitation: &'static str,
    limitations: &mut Vec<String>,
    baseline_derivation_incomplete: &mut bool,
) -> bool {
    if value_overflowed {
        *target = None;
        *baseline_derivation_incomplete = true;
        return true;
    }
    let Some(value) = value else {
        return false;
    };
    let Some(observed) = *target else {
        limitations.push(missing_limitation.to_string());
        *baseline_derivation_incomplete = true;
        return false;
    };
    match observed.checked_add(value) {
        Some(total) => {
            *target = Some(total);
            false
        }
        None => {
            *target = None;
            true
        }
    }
}

fn checked_option_sum(values: impl IntoIterator<Item = Option<u64>>) -> (Option<u64>, bool) {
    let mut present = false;
    let mut total = 0_u64;
    for value in values.into_iter().flatten() {
        present = true;
        let Some(next) = total.checked_add(value) else {
            return (None, true);
        };
        total = next;
    }
    (present.then_some(total), false)
}

#[cfg(test)]
#[path = "../../tests/unit/optimization_tests.rs"]
mod tests;
