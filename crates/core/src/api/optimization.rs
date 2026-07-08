// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Managed, bounded LLM optimization accounting.

use std::sync::{Arc, Mutex};

use uuid::Uuid;

use crate::codec::optimization::{
    LlmOptimizationContribution, LlmOptimizationModel, LlmOptimizationSummary,
    LlmOptimizationSummaryStatus, LlmOptimizationTokens,
};
use crate::codec::response::{AnnotatedLlmResponse, PricingResolver};

/// Maximum contributions retained for one LLM call.
pub const MAX_LLM_OPTIMIZATION_CONTRIBUTIONS: usize = 64;
/// Maximum serialized custom payload size for one contribution.
pub const MAX_LLM_OPTIMIZATION_PAYLOAD_BYTES: usize = 16 * 1024;
/// Maximum aggregate serialized custom payload size for one LLM call.
pub const MAX_LLM_OPTIMIZATION_TOTAL_PAYLOAD_BYTES: usize = 256 * 1024;

#[derive(Debug, Default)]
struct AccumulatorState {
    contributions: Vec<LlmOptimizationContribution>,
    total_payload_bytes: usize,
    emitted: usize,
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
    /// invariant or a per-call bound. Rejection never affects LLM execution.
    #[must_use]
    pub fn record(&self, mut contribution: LlmOptimizationContribution) -> bool {
        let payload_bytes = match contribution.payload.as_ref() {
            Some(_payload) if contribution.payload_schema.is_none() => {
                if let Ok(mut state) = self.state.lock() {
                    state.invalid_payload_schema = true;
                }
                return false;
            }
            Some(payload) => match bounded_json_size(payload, MAX_LLM_OPTIMIZATION_PAYLOAD_BYTES) {
                Ok(size) => size,
                Err(PayloadSizeError::LimitExceeded) => {
                    if let Ok(mut state) = self.state.lock() {
                        state.contribution_limit_exceeded = true;
                    }
                    return false;
                }
                Err(PayloadSizeError::Serialization) => {
                    if let Ok(mut state) = self.state.lock() {
                        state.invalid_payload_schema = true;
                    }
                    return false;
                }
            },
            None => 0,
        };

        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if state.contributions.len() >= MAX_LLM_OPTIMIZATION_CONTRIBUTIONS
            || payload_bytes > MAX_LLM_OPTIMIZATION_PAYLOAD_BYTES
            || state.total_payload_bytes.saturating_add(payload_bytes)
                > MAX_LLM_OPTIMIZATION_TOTAL_PAYLOAD_BYTES
        {
            state.contribution_limit_exceeded = true;
            return false;
        }

        contribution.id = Some(Uuid::now_v7());
        contribution.sequence = Some(state.contributions.len() as u64);
        state.total_payload_bytes += payload_bytes;
        state.contributions.push(contribution);
        true
    }

    pub(crate) fn record_all(
        &self,
        contributions: impl IntoIterator<Item = LlmOptimizationContribution>,
    ) {
        for contribution in contributions {
            let _ = self.record(contribution);
        }
    }

    pub(crate) fn take_unemitted(&self) -> Vec<LlmOptimizationContribution> {
        let Ok(mut state) = self.state.lock() else {
            return Vec::new();
        };
        let start = state.emitted.min(state.contributions.len());
        let contributions = state.contributions[start..].to_vec();
        state.emitted = state.contributions.len();
        contributions
    }

    fn finish(&self) -> FinishedContributions {
        let Ok(mut state) = self.state.lock() else {
            return FinishedContributions {
                contributions: Vec::new(),
                limitations: vec!["optimization_accumulator_unavailable".to_string()],
            };
        };
        let mut limitations = Vec::new();
        if state.contribution_limit_exceeded {
            limitations.push("contribution_limit_exceeded".to_string());
        }
        if state.invalid_payload_schema {
            limitations.push("invalid_contribution_payload_schema".to_string());
        }
        FinishedContributions {
            contributions: std::mem::take(&mut state.contributions),
            limitations,
        }
    }
}

enum PayloadSizeError {
    LimitExceeded,
    Serialization,
}

fn bounded_json_size(value: &serde_json::Value, limit: usize) -> Result<usize, PayloadSizeError> {
    struct CountingWriter {
        size: usize,
        limit: usize,
        exceeded: bool,
    }

    impl std::io::Write for CountingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.size.saturating_add(bytes.len()) > self.limit {
                self.exceeded = true;
                return Err(std::io::Error::other("optimization payload limit exceeded"));
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
            PayloadSizeError::LimitExceeded
        } else {
            PayloadSizeError::Serialization
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
    response: Option<&mut AnnotatedLlmResponse>,
    requested_model: Option<&str>,
    pricing: &PricingResolver,
) -> Option<LlmOptimizationSummary> {
    let finished = recorder.finish();
    if finished.contributions.is_empty() && finished.limitations.is_empty() {
        return None;
    }

    let mut tokens_saved = LlmOptimizationTokens::default();
    let mut baseline_model = None;
    let mut contributed_effective_model = None;
    for contribution in finished
        .contributions
        .iter()
        .filter(|contribution| contribution.applied)
    {
        if let Some(saved) = contribution
            .token_impact
            .as_ref()
            .and_then(|impact| impact.saved.as_ref())
        {
            tokens_saved.add_assign(saved);
        }
        if contribution.kind.as_str()
            == crate::codec::optimization::LlmOptimizationKind::MODEL_ROUTING
            && let Some(transition) = contribution.model_transition.as_ref()
        {
            if baseline_model.is_none() {
                baseline_model = transition.baseline.clone();
            }
            if transition.effective.is_some() {
                contributed_effective_model = transition.effective.clone();
            }
        }
    }

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
    if baseline_model.is_none() {
        baseline_model = effective_model.clone();
    }

    let effective_usage = response
        .as_ref()
        .and_then(|response| response.usage.clone());
    let baseline_usage = effective_usage.as_ref().map(|usage| {
        let mut baseline = usage.clone();
        baseline.cost = None;
        add_tokens(&mut baseline.prompt_tokens, tokens_saved.prompt_tokens);
        add_tokens(
            &mut baseline.completion_tokens,
            tokens_saved.completion_tokens,
        );
        add_tokens(
            &mut baseline.cache_read_tokens,
            tokens_saved.cache_read_tokens,
        );
        add_tokens(
            &mut baseline.cache_write_tokens,
            tokens_saved.cache_write_tokens,
        );
        let total_saved = tokens_saved
            .total_tokens
            .or_else(|| option_sum([tokens_saved.prompt_tokens, tokens_saved.completion_tokens]));
        add_tokens(&mut baseline.total_tokens, total_saved);
        baseline
    });

    let actual_cost = effective_usage
        .as_ref()
        .and_then(|usage| usage.cost.clone())
        .or_else(|| {
            let model = effective_model.as_ref()?;
            let usage = effective_usage.as_ref()?;
            pricing.estimate_cost_for_provider(model.provider.as_deref(), &model.model, usage)
        });
    let baseline_cost = baseline_model.as_ref().and_then(|model| {
        pricing.estimate_cost_for_provider(
            model.provider.as_deref(),
            &model.model,
            baseline_usage.as_ref()?,
        )
    });

    let mut limitations = finished.limitations;
    if effective_usage.is_none() {
        limitations.push("missing_effective_usage".to_string());
    }
    if baseline_model.is_none() {
        limitations.push("missing_baseline_model".to_string());
    }
    if baseline_cost.is_none() {
        limitations.push("missing_baseline_pricing".to_string());
    }
    if actual_cost.is_none() {
        limitations.push("missing_actual_cost".to_string());
    }

    let (estimated_cost_saved, currency) = match (&baseline_cost, &actual_cost) {
        (Some(baseline), Some(actual))
            if baseline.currency.eq_ignore_ascii_case(&actual.currency) =>
        {
            (
                baseline
                    .total_or_component_sum()
                    .zip(actual.total_or_component_sum())
                    .map(|(baseline, actual)| baseline - actual),
                Some(baseline.currency.clone()),
            )
        }
        (Some(_), Some(_)) => {
            limitations.push("cost_currency_mismatch".to_string());
            (None, None)
        }
        _ => (None, None),
    };

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

fn add_tokens(target: &mut Option<u64>, value: Option<u64>) {
    if let Some(value) = value {
        *target = Some(target.unwrap_or(0).saturating_add(value));
    }
}

fn option_sum(values: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    let mut present = false;
    let total = values.into_iter().flatten().fold(0_u64, |total, value| {
        present = true;
        total.saturating_add(value)
    });
    present.then_some(total)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::api::event::DataSchema;
    use crate::codec::optimization::{
        LlmOptimizationEvidenceQuality, LlmOptimizationModelTransition, LlmOptimizationTokenImpact,
    };
    use crate::codec::response::{PricingCatalog, Usage};
    use crate::json::Json;

    fn resolver() -> PricingResolver {
        resolver_with_rates(2.0, 1.0)
    }

    fn resolver_with_rates(baseline_input: f64, effective_input: f64) -> PricingResolver {
        let catalog = PricingCatalog::from_json_str(
            &json!({
                "version": 1,
                "entries": [
                    {"provider":"test","model_id":"baseline","pricing_as_of":"2026-07-08","pricing_source":"test-snapshot","rates":{"input_per_million":baseline_input,"output_per_million":4.0,"cache_read_per_million":0.5,"cache_write_per_million":3.0},"prompt_cache":{"read_accounting":"included_in_prompt_tokens"}},
                    {"provider":"test","model_id":"effective","pricing_as_of":"2026-07-08","pricing_source":"test-snapshot","rates":{"input_per_million":effective_input,"output_per_million":2.0,"cache_read_per_million":0.25,"cache_write_per_million":2.0},"prompt_cache":{"read_accounting":"included_in_prompt_tokens"}}
                ]
            })
            .to_string(),
        )
        .unwrap();
        PricingResolver::from_catalogs(vec![catalog])
    }

    fn contribution() -> LlmOptimizationContribution {
        let mut contribution = LlmOptimizationContribution::new(
            "test.optimizer",
            crate::codec::optimization::LlmOptimizationKind::model_routing(),
        );
        contribution.model_transition = Some(LlmOptimizationModelTransition {
            baseline: Some(LlmOptimizationModel::new("baseline").with_provider("test")),
            effective: Some(LlmOptimizationModel::new("effective").with_provider("test")),
        });
        contribution.token_impact = Some(LlmOptimizationTokenImpact {
            saved: Some(LlmOptimizationTokens::saved_prompt(200)),
            quality: Some(LlmOptimizationEvidenceQuality::Estimated),
            estimation_method: Some("test-tokenizer".to_string()),
            ..LlmOptimizationTokenImpact::default()
        });
        contribution
    }

    #[test]
    fn combined_summary_retains_token_evidence_and_snapshot_pricing() {
        let recorder = LlmOptimizationRecorder::default();
        assert!(recorder.record(contribution()));
        let mut response = AnnotatedLlmResponse {
            model: Some("effective".to_string()),
            usage: Some(Usage {
                prompt_tokens: Some(800),
                completion_tokens: Some(100),
                total_tokens: Some(900),
                ..Usage::default()
            }),
            ..AnnotatedLlmResponse::default()
        };
        let summary = finalize_optimization_summary(
            &recorder,
            Some(&mut response),
            Some("baseline"),
            &resolver(),
        )
        .unwrap();
        assert_eq!(summary.status, LlmOptimizationSummaryStatus::Complete);
        assert_eq!(summary.tokens_saved.prompt_tokens, Some(200));
        assert_eq!(
            summary.baseline_usage.as_ref().unwrap().prompt_tokens,
            Some(1000)
        );
        assert_eq!(summary.baseline_cost.as_ref().unwrap().total, Some(0.0024));
        assert_eq!(summary.actual_cost.as_ref().unwrap().total, Some(0.001));
        assert!((summary.estimated_cost_saved.unwrap() - 0.0014).abs() < 1e-12);
    }

    #[test]
    fn applied_route_is_the_authoritative_effective_model() {
        let recorder = LlmOptimizationRecorder::default();
        assert!(recorder.record(contribution()));
        let mut response = AnnotatedLlmResponse {
            // Providers may return an alias or deployment name rather than
            // the exact model Relay selected and sent upstream.
            model: Some("provider-response-alias".to_string()),
            usage: Some(Usage {
                prompt_tokens: Some(800),
                completion_tokens: Some(100),
                total_tokens: Some(900),
                ..Usage::default()
            }),
            ..AnnotatedLlmResponse::default()
        };
        let summary = finalize_optimization_summary(
            &recorder,
            Some(&mut response),
            Some("original-request-model"),
            &resolver(),
        )
        .unwrap();
        assert_eq!(summary.baseline_model.as_ref().unwrap().model, "baseline");
        assert_eq!(summary.effective_model.as_ref().unwrap().model, "effective");
    }

    #[test]
    fn unpriced_summary_is_partial_without_losing_tokens() {
        let recorder = LlmOptimizationRecorder::default();
        assert!(recorder.record(contribution()));
        let mut response = AnnotatedLlmResponse {
            model: Some("effective".to_string()),
            usage: Some(Usage {
                prompt_tokens: Some(8),
                ..Usage::default()
            }),
            ..AnnotatedLlmResponse::default()
        };
        let summary = finalize_optimization_summary(
            &recorder,
            Some(&mut response),
            None,
            &PricingResolver::default(),
        )
        .unwrap();
        assert_eq!(summary.status, LlmOptimizationSummaryStatus::Partial);
        assert_eq!(summary.tokens_saved.prompt_tokens, Some(200));
        assert!(summary.estimated_cost_saved.is_none());
    }

    #[test]
    fn zero_and_negative_savings_are_preserved() {
        for (baseline_rate, effective_rate, expected_sign) in [(0.0, 0.0, 0_i8), (0.5, 2.0, -1_i8)]
        {
            let recorder = LlmOptimizationRecorder::default();
            assert!(recorder.record(contribution()));
            let mut response = AnnotatedLlmResponse {
                model: Some("effective".to_string()),
                usage: Some(Usage {
                    prompt_tokens: Some(800),
                    total_tokens: Some(800),
                    ..Usage::default()
                }),
                ..AnnotatedLlmResponse::default()
            };
            let summary = finalize_optimization_summary(
                &recorder,
                Some(&mut response),
                None,
                &resolver_with_rates(baseline_rate, effective_rate),
            )
            .unwrap();
            let saved = summary.estimated_cost_saved.unwrap();
            match expected_sign {
                0 => assert_eq!(saved, 0.0),
                -1 => assert!(saved < 0.0),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn multiple_contributions_and_cache_savings_aggregate_explicitly() {
        let recorder = LlmOptimizationRecorder::default();
        for (producer, prompt, cache_read, cache_write) in
            [("test.one", 5, 7, 0), ("test.two", 11, 13, 17)]
        {
            let mut item = LlmOptimizationContribution::new(
                producer,
                crate::codec::optimization::LlmOptimizationKind::input_compression(),
            );
            item.token_impact = Some(LlmOptimizationTokenImpact {
                saved: Some(LlmOptimizationTokens {
                    prompt_tokens: Some(prompt),
                    cache_read_tokens: Some(cache_read),
                    cache_write_tokens: Some(cache_write),
                    total_tokens: Some(prompt),
                    ..LlmOptimizationTokens::default()
                }),
                ..LlmOptimizationTokenImpact::default()
            });
            assert!(recorder.record(item));
        }
        let mut response = AnnotatedLlmResponse {
            model: Some("effective".to_string()),
            usage: Some(Usage {
                prompt_tokens: Some(100),
                completion_tokens: Some(10),
                total_tokens: Some(110),
                cache_read_tokens: Some(20),
                cache_write_tokens: Some(3),
                ..Usage::default()
            }),
            ..AnnotatedLlmResponse::default()
        };
        let summary =
            finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver())
                .unwrap();
        assert_eq!(summary.tokens_saved.prompt_tokens, Some(16));
        assert_eq!(summary.tokens_saved.cache_read_tokens, Some(20));
        assert_eq!(summary.tokens_saved.cache_write_tokens, Some(17));
        assert_eq!(
            summary.baseline_usage.as_ref().unwrap().cache_read_tokens,
            Some(40)
        );
        assert_eq!(
            summary.baseline_usage.as_ref().unwrap().cache_write_tokens,
            Some(20)
        );
        assert_eq!(summary.contributions[0].sequence, Some(0));
        assert_eq!(summary.contributions[1].sequence, Some(1));
    }

    #[test]
    fn serialized_summary_can_be_repriced_with_a_new_catalog() {
        let recorder = LlmOptimizationRecorder::default();
        assert!(recorder.record(contribution()));
        let mut response = AnnotatedLlmResponse {
            model: Some("effective".to_string()),
            usage: Some(Usage {
                prompt_tokens: Some(800),
                completion_tokens: Some(100),
                total_tokens: Some(900),
                ..Usage::default()
            }),
            ..AnnotatedLlmResponse::default()
        };
        let original =
            finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver())
                .unwrap();
        let restored: LlmOptimizationSummary =
            serde_json::from_value(serde_json::to_value(&original).unwrap()).unwrap();
        let newer = resolver_with_rates(10.0, 5.0);
        let baseline = newer
            .estimate_cost_for_provider(
                Some("test"),
                "baseline",
                restored.baseline_usage.as_ref().unwrap(),
            )
            .unwrap()
            .total_or_component_sum()
            .unwrap();
        let actual = newer
            .estimate_cost_for_provider(
                Some("test"),
                "effective",
                restored.effective_usage.as_ref().unwrap(),
            )
            .unwrap()
            .total_or_component_sum()
            .unwrap();
        assert_ne!(baseline - actual, original.estimated_cost_saved.unwrap());
        assert_eq!(restored.tokens_saved.prompt_tokens, Some(200));
    }

    #[test]
    fn no_usage_is_an_explicit_partial_summary() {
        let recorder = LlmOptimizationRecorder::default();
        assert!(recorder.record(contribution()));
        let summary =
            finalize_optimization_summary(&recorder, None, Some("effective"), &resolver()).unwrap();
        assert_eq!(summary.status, LlmOptimizationSummaryStatus::Partial);
        assert!(
            summary
                .limitations
                .contains(&"missing_effective_usage".to_string())
        );
        assert_eq!(summary.tokens_saved.prompt_tokens, Some(200));
    }

    #[test]
    fn payload_byte_limits_are_enforced_without_unbounded_work() {
        let oversized = LlmOptimizationRecorder::default();
        let mut item = LlmOptimizationContribution::new("test", "custom");
        item.payload_schema = Some(DataSchema {
            name: "test.payload".to_string(),
            version: "1".to_string(),
        });
        item.payload = Some(Json::String("x".repeat(MAX_LLM_OPTIMIZATION_PAYLOAD_BYTES)));
        assert!(!oversized.record(item));

        let aggregate = LlmOptimizationRecorder::default();
        for index in 0..17 {
            let mut item = LlmOptimizationContribution::new(format!("test.{index}"), "custom");
            item.payload_schema = Some(DataSchema {
                name: "test.payload".to_string(),
                version: "1".to_string(),
            });
            item.payload = Some(Json::String("x".repeat(15_000)));
            assert!(aggregate.record(item));
        }
        let mut overflow = LlmOptimizationContribution::new("test.overflow", "custom");
        overflow.payload_schema = Some(DataSchema {
            name: "test.payload".to_string(),
            version: "1".to_string(),
        });
        overflow.payload = Some(Json::String("x".repeat(15_000)));
        assert!(!aggregate.record(overflow));
        assert!(
            aggregate
                .finish()
                .limitations
                .contains(&"contribution_limit_exceeded".to_string())
        );
    }

    #[test]
    fn bounds_and_invalid_payloads_are_best_effort_and_visible() {
        let recorder = LlmOptimizationRecorder::default();
        let mut invalid = LlmOptimizationContribution::new("test", "custom");
        invalid.payload = Some(json!({"evidence": true}));
        assert!(!recorder.record(invalid));
        for index in 0..MAX_LLM_OPTIMIZATION_CONTRIBUTIONS {
            assert!(recorder.record(LlmOptimizationContribution::new(
                format!("test.{index}"),
                "custom"
            )));
        }
        assert!(!recorder.record(LlmOptimizationContribution::new("overflow", "custom")));
        let summary =
            finalize_optimization_summary(&recorder, None, None, &PricingResolver::default())
                .unwrap();
        assert_eq!(
            summary.contributions.len(),
            MAX_LLM_OPTIMIZATION_CONTRIBUTIONS
        );
        assert!(
            summary
                .limitations
                .contains(&"contribution_limit_exceeded".to_string())
        );
        assert!(
            summary
                .limitations
                .contains(&"invalid_contribution_payload_schema".to_string())
        );
    }

    #[tokio::test]
    async fn recorder_can_be_captured_for_stream_commit() {
        let recorder = LlmOptimizationRecorder::default();
        let captured = scope_llm_optimization_recorder(recorder.clone(), async {
            current_llm_optimization_recorder().unwrap()
        })
        .await;
        assert!(captured.record(LlmOptimizationContribution::new("test.stream", "commit")));
        assert_eq!(recorder.finish().contributions.len(), 1);
    }
}
