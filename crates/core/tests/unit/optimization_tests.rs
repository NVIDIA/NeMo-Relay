// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for managed, bounded LLM optimization accounting.

use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::*;
use crate::api::event::DataSchema;
use crate::codec::optimization::{
    LlmOptimizationEvidenceQuality, LlmOptimizationModelTransition, LlmOptimizationTokenImpact,
};
use crate::codec::response::{
    CostEstimate, CostSource, PricingCatalog, PricingCatalogError, PricingSource, Usage,
};
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
            cost: Some(CostEstimate {
                total: Some(99.0),
                currency: "USD".to_string(),
                input: None,
                output: None,
                cache_read: None,
                cache_write: None,
                source: CostSource::ModelPricing,
                pricing_provider: Some("test".to_string()),
                pricing_model: Some("provider-response-alias".to_string()),
                pricing_as_of: Some("2020-01-01".to_string()),
                pricing_source: Some("stale-alias-pricing".to_string()),
            }),
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
    assert_eq!(summary.actual_cost.as_ref().unwrap().total, Some(0.001));
    assert_eq!(
        summary
            .actual_cost
            .as_ref()
            .unwrap()
            .pricing_model
            .as_deref(),
        Some("effective")
    );
    assert_eq!(
        response
            .usage
            .as_ref()
            .and_then(|usage| usage.cost.as_ref())
            .and_then(|cost| cost.pricing_model.as_deref()),
        Some("effective")
    );
}

#[test]
fn routed_model_preserves_provider_reported_cost_and_provenance() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(contribution()));
    let provider_cost = CostEstimate {
        total: Some(0.75),
        currency: "EUR".to_string(),
        input: Some(0.5),
        output: Some(0.25),
        cache_read: None,
        cache_write: None,
        source: CostSource::ProviderReported,
        pricing_provider: Some("provider-billing".to_string()),
        pricing_model: Some("provider-response-alias".to_string()),
        pricing_as_of: Some("2026-07-08".to_string()),
        pricing_source: Some("provider-invoice".to_string()),
    };
    let mut response = AnnotatedLlmResponse {
        model: Some("provider-response-alias".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(800),
            completion_tokens: Some(100),
            total_tokens: Some(900),
            cost: Some(provider_cost.clone()),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };

    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
    assert_eq!(summary.effective_model.as_ref().unwrap().model, "effective");
    assert_eq!(summary.actual_cost.as_ref(), Some(&provider_cost));
    assert_eq!(
        summary
            .effective_usage
            .as_ref()
            .and_then(|usage| usage.cost.as_ref()),
        Some(&provider_cost)
    );
    assert_eq!(
        response
            .usage
            .as_ref()
            .and_then(|usage| usage.cost.as_ref()),
        Some(&provider_cost)
    );
    assert!(
        summary
            .limitations
            .contains(&"cost_currency_mismatch".to_string())
    );
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
    for (baseline_rate, effective_rate, expected_sign) in [(0.0, 0.0, 0_i8), (0.5, 2.0, -1_i8)] {
        let recorder = LlmOptimizationRecorder::default();
        assert!(recorder.record(contribution()));
        let mut response = AnnotatedLlmResponse {
            model: Some("effective".to_string()),
            usage: Some(Usage {
                prompt_tokens: Some(800),
                completion_tokens: Some(0),
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
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
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
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
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
fn close_time_pricing_uses_the_loaded_resolver_without_reloading_sources() {
    struct CountingSource {
        loads: std::sync::Arc<AtomicUsize>,
        catalog: PricingCatalog,
    }

    impl PricingSource for CountingSource {
        fn source_name(&self) -> &str {
            "counting-test-source"
        }

        fn load_catalog(&self) -> Result<Option<PricingCatalog>, PricingCatalogError> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            Ok(Some(self.catalog.clone()))
        }
    }

    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [
                {"provider":"test","model_id":"baseline","pricing_as_of":"2026-07-08","pricing_source":"test","rates":{"input_per_million":2.0,"output_per_million":2.0},"prompt_cache":{"read_accounting":"included_in_prompt_tokens"}},
                {"provider":"test","model_id":"effective","pricing_as_of":"2026-07-08","pricing_source":"test","rates":{"input_per_million":1.0,"output_per_million":1.0},"prompt_cache":{"read_accounting":"included_in_prompt_tokens"}}
            ]
        })
        .to_string(),
    )
    .unwrap();
    let loads = std::sync::Arc::new(AtomicUsize::new(0));
    let pricing = PricingResolver::from_sources(vec![Box::new(CountingSource {
        loads: loads.clone(),
        catalog,
    })])
    .unwrap();
    assert_eq!(loads.load(Ordering::SeqCst), 1);

    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(contribution()));
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(0),
            total_tokens: Some(10),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };
    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &pricing).unwrap();
    assert!(summary.actual_cost.is_some());
    assert_eq!(loads.load(Ordering::SeqCst), 1);
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
fn full_contribution_byte_limits_are_enforced_without_unbounded_work() {
    let oversized = LlmOptimizationRecorder::default();
    let mut item = LlmOptimizationContribution::new("test", "custom");
    item.payload_schema = Some(DataSchema {
        name: "test.payload".to_string(),
        version: "1".to_string(),
    });
    item.payload = Some(Json::String(
        "x".repeat(MAX_LLM_OPTIMIZATION_CONTRIBUTION_BYTES),
    ));
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
    for index in 0..(MAX_LLM_OPTIMIZATION_CONTRIBUTION_ATTEMPTS - 1) {
        assert!(recorder.record(LlmOptimizationContribution::new(
            format!("test.{index}"),
            "custom"
        )));
    }
    assert!(!recorder.record(LlmOptimizationContribution::new("overflow", "custom")));
    let summary =
        finalize_optimization_summary(&recorder, None, None, &PricingResolver::default()).unwrap();
    assert_eq!(
        summary.contributions.len(),
        MAX_LLM_OPTIMIZATION_CONTRIBUTION_ATTEMPTS - 1
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

#[test]
fn full_envelope_fields_are_bounded_and_rejections_do_not_consume_sequences() {
    let oversized_producer_recorder = LlmOptimizationRecorder::default();
    let mut oversized_producer = LlmOptimizationContribution::new(
        "x".repeat(MAX_LLM_OPTIMIZATION_CONTRIBUTION_BYTES),
        "custom",
    );
    oversized_producer.id = Some(uuid::Uuid::nil());
    oversized_producer.sequence = Some(99);
    assert!(!oversized_producer_recorder.record(oversized_producer));
    assert!(
        !oversized_producer_recorder.record(LlmOptimizationContribution::new("sealed", "custom"))
    );
    assert!(
        oversized_producer_recorder
            .finish()
            .limitations
            .contains(&"contribution_limit_exceeded".to_string())
    );

    let oversized_extra_recorder = LlmOptimizationRecorder::default();
    let mut oversized_extra = LlmOptimizationContribution::new("test", "custom");
    oversized_extra.extra.insert(
        "future_evidence".to_string(),
        Json::String("x".repeat(MAX_LLM_OPTIMIZATION_CONTRIBUTION_BYTES)),
    );
    assert!(!oversized_extra_recorder.record(oversized_extra));
    assert!(!oversized_extra_recorder.record(LlmOptimizationContribution::new("sealed", "custom")));

    let oversized_kind = LlmOptimizationContribution::new(
        "test",
        "x".repeat(MAX_LLM_OPTIMIZATION_CONTRIBUTION_BYTES),
    );
    let mut oversized_model = LlmOptimizationContribution::new("test", "custom");
    oversized_model.model_transition = Some(LlmOptimizationModelTransition {
        baseline: Some(LlmOptimizationModel::new(
            "x".repeat(MAX_LLM_OPTIMIZATION_CONTRIBUTION_BYTES),
        )),
        effective: None,
    });
    let mut oversized_method = LlmOptimizationContribution::new("test", "custom");
    oversized_method.token_impact = Some(LlmOptimizationTokenImpact {
        estimation_method: Some("x".repeat(MAX_LLM_OPTIMIZATION_CONTRIBUTION_BYTES)),
        ..LlmOptimizationTokenImpact::default()
    });
    for oversized in [oversized_kind, oversized_model, oversized_method] {
        let recorder = LlmOptimizationRecorder::default();
        assert!(!recorder.record(oversized));
        assert!(!recorder.record(LlmOptimizationContribution::new("sealed", "custom")));
        assert!(
            recorder
                .finish()
                .limitations
                .contains(&"contribution_limit_exceeded".to_string())
        );
    }

    let recorder = LlmOptimizationRecorder::default();
    let mut malformed = LlmOptimizationContribution::new("malformed", "custom");
    malformed.payload = Some(json!({"missing": "schema"}));
    assert!(!recorder.record(malformed));
    let mut accepted = LlmOptimizationContribution::new("test", "custom");
    accepted.id = Some(uuid::Uuid::nil());
    accepted.sequence = Some(99);
    assert!(recorder.record(accepted));

    let finished = recorder.finish();
    assert_eq!(finished.contributions.len(), 1);
    assert_eq!(finished.contributions[0].sequence, Some(0));
    assert_ne!(finished.contributions[0].id, Some(uuid::Uuid::nil()));
    assert!(
        finished
            .limitations
            .contains(&"invalid_contribution_payload_schema".to_string())
    );
}

#[test]
fn record_all_bounds_malformed_attempts_and_keeps_accepted_sequences_dense() {
    struct CountingMalformed {
        next_calls: std::sync::Arc<AtomicUsize>,
    }

    impl Iterator for CountingMalformed {
        type Item = LlmOptimizationContribution;

        fn next(&mut self) -> Option<Self::Item> {
            self.next_calls.fetch_add(1, Ordering::SeqCst);
            let mut contribution = LlmOptimizationContribution::new("malformed", "custom");
            contribution.payload = Some(json!({"schema": "missing"}));
            Some(contribution)
        }
    }

    let next_calls = std::sync::Arc::new(AtomicUsize::new(0));
    let recorder = LlmOptimizationRecorder::default();
    recorder.record_all(CountingMalformed {
        next_calls: next_calls.clone(),
    });
    assert_eq!(
        next_calls.load(Ordering::SeqCst),
        MAX_LLM_OPTIMIZATION_CONTRIBUTION_ATTEMPTS + 1
    );
    assert!(!recorder.record(LlmOptimizationContribution::new("late", "custom")));
    let finished = recorder.finish();
    assert!(finished.contributions.is_empty());
    assert!(
        finished
            .limitations
            .contains(&"contribution_limit_exceeded".to_string())
    );
    assert!(
        finished
            .limitations
            .contains(&"invalid_contribution_payload_schema".to_string())
    );

    let dense = LlmOptimizationRecorder::default();
    let mut invalid = LlmOptimizationContribution::new("invalid", "custom");
    invalid.payload = Some(json!({"schema": "missing"}));
    dense.record_all([
        LlmOptimizationContribution::new("first", "custom"),
        invalid,
        LlmOptimizationContribution::new("second", "custom"),
    ]);
    let finished = dense.finish();
    assert_eq!(finished.contributions.len(), 2);
    assert_eq!(finished.contributions[0].sequence, Some(0));
    assert_eq!(finished.contributions[1].sequence, Some(1));
}

#[test]
fn delivery_cursor_advances_only_when_explicitly_acknowledged() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(LlmOptimizationContribution::new("one", "custom")));
    assert!(recorder.record(LlmOptimizationContribution::new("two", "custom")));

    assert_eq!(recorder.unemitted().len(), 2);
    assert_eq!(recorder.unemitted().len(), 2);
    recorder.mark_emitted(1);
    let remaining = recorder.unemitted();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].producer, "two");
    recorder.mark_emitted(usize::MAX);
    assert!(recorder.unemitted().is_empty());

    let finished = recorder.finish();
    assert_eq!(finished.contributions.len(), 2);
    recorder.mark_emitted(1);
    assert!(recorder.unemitted().is_empty());
}

#[test]
fn finish_closes_the_recorder_and_late_writes_are_rejected() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(LlmOptimizationContribution::new("before", "custom")));
    recorder.note_limitation("before_close");

    let finished = recorder.finish();
    assert_eq!(finished.contributions.len(), 1);
    assert_eq!(finished.limitations, vec!["before_close"]);
    assert!(!recorder.record(LlmOptimizationContribution::new("late", "custom")));
    recorder.note_limitation("after_close");

    let second_finish = recorder.finish();
    assert!(second_finish.contributions.is_empty());
    assert!(second_finish.limitations.is_empty());
}

#[test]
fn concurrent_recording_assigns_unique_dense_order() {
    let recorder = LlmOptimizationRecorder::default();
    let workers = (0..MAX_LLM_OPTIMIZATION_CONTRIBUTIONS)
        .map(|index| {
            let recorder = recorder.clone();
            std::thread::spawn(move || {
                recorder.record(LlmOptimizationContribution::new(
                    format!("worker.{index}"),
                    "custom",
                ))
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        assert!(worker.join().unwrap());
    }

    let finished = recorder.finish();
    let sequences = finished
        .contributions
        .iter()
        .map(|contribution| contribution.sequence.unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        sequences,
        (0..MAX_LLM_OPTIMIZATION_CONTRIBUTIONS as u64).collect::<Vec<_>>()
    );
    let ids = finished
        .contributions
        .iter()
        .map(|contribution| contribution.id.unwrap())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(ids.len(), MAX_LLM_OPTIMIZATION_CONTRIBUTIONS);
}

#[test]
fn concurrent_finish_is_an_atomic_acceptance_boundary() {
    let recorder = LlmOptimizationRecorder::default();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(
        MAX_LLM_OPTIMIZATION_CONTRIBUTIONS + 1,
    ));
    let workers = (0..MAX_LLM_OPTIMIZATION_CONTRIBUTIONS)
        .map(|index| {
            let recorder = recorder.clone();
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                recorder.record(LlmOptimizationContribution::new(
                    format!("worker.{index}"),
                    "custom",
                ))
            })
        })
        .collect::<Vec<_>>();

    barrier.wait();
    let finished = recorder.finish();
    let accepted = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .filter(|accepted| *accepted)
        .count();
    assert_eq!(accepted, finished.contributions.len());
    assert!(!recorder.record(LlmOptimizationContribution::new("late", "custom")));
}

#[test]
fn finalization_waits_for_reserved_record_attempts() {
    let recorder = LlmOptimizationRecorder::default();
    let attempt = recorder.reserve_record_attempt_for_test().unwrap();
    let finalizer = recorder.clone();
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        sender.send(finalizer.finish()).unwrap();
    });

    let deadline = Instant::now() + Duration::from_secs(1);
    while !recorder.is_closed_for_test() && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(recorder.is_closed_for_test());
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));

    drop(attempt);
    let finished = receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(finished.contributions.is_empty());
    assert!(finished.limitations.is_empty());
}

#[tokio::test]
async fn recorder_task_local_is_not_implicitly_inherited_by_spawned_tasks() {
    let recorder = LlmOptimizationRecorder::default();
    scope_llm_optimization_recorder(recorder, async {
        assert!(current_llm_optimization_recorder().is_some());
        assert!(
            tokio::spawn(async { current_llm_optimization_recorder().is_none() })
                .await
                .unwrap()
        );
    })
    .await;
}

fn compression_contribution(
    producer: &str,
    saved: LlmOptimizationTokens,
) -> LlmOptimizationContribution {
    let mut contribution = LlmOptimizationContribution::new(
        producer,
        crate::codec::optimization::LlmOptimizationKind::input_compression(),
    );
    contribution.token_impact = Some(LlmOptimizationTokenImpact {
        saved: Some(saved),
        ..LlmOptimizationTokenImpact::default()
    });
    contribution
}

#[test]
fn missing_observed_fields_are_not_fabricated_for_baseline_usage() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(compression_contribution(
        "compressor",
        LlmOptimizationTokens::saved_prompt(5),
    )));
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: None,
            completion_tokens: Some(2),
            total_tokens: Some(2),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };
    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();

    assert_eq!(summary.baseline_usage.as_ref().unwrap().prompt_tokens, None);
    assert!(summary.baseline_cost.is_none());
    assert!(summary.actual_cost.is_none());
    assert!(
        summary
            .limitations
            .contains(&"missing_effective_prompt_tokens".to_string())
    );
}

#[test]
fn empty_and_partial_usage_never_produce_complete_zero_cost_accounting() {
    for usage in [
        Usage::default(),
        Usage {
            prompt_tokens: Some(10),
            ..Usage::default()
        },
    ] {
        let recorder = LlmOptimizationRecorder::default();
        assert!(recorder.record(compression_contribution(
            "compressor",
            LlmOptimizationTokens::saved_prompt(2),
        )));
        let mut response = AnnotatedLlmResponse {
            model: Some("effective".to_string()),
            usage: Some(usage),
            ..AnnotatedLlmResponse::default()
        };
        let summary =
            finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver())
                .unwrap();
        assert_eq!(summary.status, LlmOptimizationSummaryStatus::Partial);
        assert!(summary.actual_cost.is_none());
        assert!(summary.baseline_cost.is_none());
        assert!(summary.estimated_cost_saved.is_none());
        assert!(
            summary
                .limitations
                .contains(&"missing_effective_completion_tokens".to_string())
        );
        assert!(
            summary
                .limitations
                .contains(&"missing_effective_total_tokens".to_string())
        );
    }
}

#[test]
fn effective_total_is_inferred_from_complete_core_usage() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(compression_contribution(
        "compressor",
        LlmOptimizationTokens::saved_prompt(3),
    )));
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(2),
            total_tokens: None,
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };
    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
    assert_eq!(
        summary.effective_usage.as_ref().unwrap().total_tokens,
        Some(12)
    );
    assert_eq!(
        summary.baseline_usage.as_ref().unwrap().total_tokens,
        Some(15)
    );
    assert!(
        !summary
            .limitations
            .contains(&"missing_effective_total_tokens".to_string())
    );
}

#[test]
fn mixed_and_inconsistent_saved_totals_have_explicit_semantics() {
    let mixed = LlmOptimizationRecorder::default();
    assert!(mixed.record(compression_contribution(
        "explicit",
        LlmOptimizationTokens {
            prompt_tokens: Some(5),
            total_tokens: Some(5),
            ..LlmOptimizationTokens::default()
        },
    )));
    assert!(mixed.record(compression_contribution(
        "derived",
        LlmOptimizationTokens {
            prompt_tokens: Some(7),
            total_tokens: None,
            ..LlmOptimizationTokens::default()
        },
    )));
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(20),
            completion_tokens: Some(0),
            total_tokens: Some(20),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };
    let summary =
        finalize_optimization_summary(&mixed, Some(&mut response), None, &resolver()).unwrap();
    assert_eq!(summary.tokens_saved.total_tokens, Some(12));
    assert!(
        !summary
            .limitations
            .contains(&"missing_token_savings_total".to_string())
    );

    let inconsistent = LlmOptimizationRecorder::default();
    assert!(inconsistent.record(compression_contribution(
        "inconsistent",
        LlmOptimizationTokens {
            prompt_tokens: Some(3),
            completion_tokens: Some(2),
            total_tokens: Some(9),
            ..LlmOptimizationTokens::default()
        },
    )));
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(20),
            completion_tokens: Some(5),
            total_tokens: Some(25),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };
    let summary =
        finalize_optimization_summary(&inconsistent, Some(&mut response), None, &resolver())
            .unwrap();
    assert_eq!(summary.tokens_saved.total_tokens, Some(9));
    assert!(summary.baseline_cost.is_none());
    assert!(
        summary
            .limitations
            .contains(&"inconsistent_token_savings_total".to_string())
    );

    let cache_only = LlmOptimizationRecorder::default();
    assert!(cache_only.record(compression_contribution(
        "cache-only",
        LlmOptimizationTokens {
            cache_read_tokens: Some(4),
            ..LlmOptimizationTokens::default()
        },
    )));
    let summary = finalize_optimization_summary(
        &cache_only,
        None,
        Some("effective"),
        &PricingResolver::default(),
    )
    .unwrap();
    assert!(summary.tokens_saved.total_tokens.is_none());
    assert!(
        summary
            .limitations
            .contains(&"missing_token_savings_total".to_string())
    );
}

#[test]
fn checked_token_aggregation_reports_overflow_without_clamping_evidence() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(compression_contribution(
        "large",
        LlmOptimizationTokens {
            prompt_tokens: Some(u64::MAX),
            total_tokens: Some(u64::MAX),
            ..LlmOptimizationTokens::default()
        },
    )));
    assert!(recorder.record(compression_contribution(
        "overflow",
        LlmOptimizationTokens::saved_prompt(1),
    )));
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(1),
            total_tokens: Some(1),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };

    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
    assert_eq!(summary.status, LlmOptimizationSummaryStatus::Partial);
    assert!(
        summary
            .limitations
            .contains(&"token_count_overflow".to_string())
    );
    assert_eq!(summary.tokens_saved.prompt_tokens, None);
    assert_eq!(summary.tokens_saved.total_tokens, None);
    assert_eq!(summary.baseline_usage.as_ref().unwrap().prompt_tokens, None);
    assert!(summary.baseline_cost.is_none());
    assert!(summary.estimated_cost_saved.is_none());
    assert_eq!(summary.contributions.len(), 2);
}

#[test]
fn checked_token_aggregation_accepts_the_exact_u64_boundary() {
    let recorder = LlmOptimizationRecorder::default();
    for (producer, saved) in [("almost-max", u64::MAX - 1), ("last-token", 1)] {
        assert!(recorder.record(compression_contribution(
            producer,
            LlmOptimizationTokens {
                prompt_tokens: Some(saved),
                ..LlmOptimizationTokens::default()
            },
        )));
    }
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(0),
            total_tokens: Some(0),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };

    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
    assert_eq!(summary.tokens_saved.prompt_tokens, Some(u64::MAX));
    assert_eq!(
        summary.baseline_usage.as_ref().unwrap().prompt_tokens,
        Some(u64::MAX)
    );
    assert_eq!(
        summary.baseline_usage.as_ref().unwrap().total_tokens,
        Some(u64::MAX)
    );
    assert!(
        !summary
            .limitations
            .contains(&"token_count_overflow".to_string())
    );
}

#[test]
fn checked_baseline_derivation_reports_effective_plus_saved_overflow() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(compression_contribution(
        "one-token",
        LlmOptimizationTokens::saved_prompt(1),
    )));
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(u64::MAX),
            completion_tokens: Some(0),
            total_tokens: Some(u64::MAX),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };

    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
    assert!(
        summary
            .limitations
            .contains(&"token_count_overflow".to_string())
    );
    assert_eq!(summary.tokens_saved.prompt_tokens, Some(1));
    assert_eq!(summary.baseline_usage.as_ref().unwrap().prompt_tokens, None);
    assert!(summary.baseline_cost.is_none());
    assert!(summary.actual_cost.is_some());
    assert!(summary.estimated_cost_saved.is_none());
}

#[test]
fn multiple_routing_authorities_are_ignored_but_compression_evidence_survives() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(contribution()));
    let mut second_route = contribution();
    second_route.model_transition = Some(LlmOptimizationModelTransition {
        baseline: Some(LlmOptimizationModel::new("other-baseline")),
        effective: Some(LlmOptimizationModel::new("other-effective")),
    });
    assert!(recorder.record(second_route));
    assert!(recorder.record(compression_contribution(
        "test.compressor",
        LlmOptimizationTokens::saved_prompt(7),
    )));
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(100),
            completion_tokens: Some(0),
            total_tokens: Some(100),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };

    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
    assert!(
        summary
            .limitations
            .contains(&"multiple_routing_contributions".to_string())
    );
    assert_eq!(summary.tokens_saved.prompt_tokens, Some(7));
    assert_eq!(summary.baseline_model.as_ref().unwrap().model, "effective");
    assert_eq!(summary.effective_model.as_ref().unwrap().model, "effective");
    assert_eq!(summary.contributions.len(), 3);
}

#[test]
fn nonapplied_routing_evidence_does_not_create_multiple_authorities() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(contribution()));
    for index in 0..5 {
        let mut hypothetical = contribution();
        hypothetical.producer = format!("hypothetical.{index}");
        hypothetical.applied = false;
        assert!(recorder.record(hypothetical));
    }
    let mut response = AnnotatedLlmResponse {
        model: Some("provider-alias".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(0),
            total_tokens: Some(10),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };
    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
    assert!(
        !summary
            .limitations
            .contains(&"multiple_routing_contributions".to_string())
    );
    assert_eq!(summary.baseline_model.as_ref().unwrap().model, "baseline");
    assert_eq!(summary.effective_model.as_ref().unwrap().model, "effective");
    assert_eq!(summary.contributions.len(), 6);
}

#[test]
fn incomplete_cost_object_makes_the_summary_partial() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(contribution()));
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(800),
            total_tokens: Some(800),
            cost: Some(CostEstimate {
                total: None,
                currency: "USD".to_string(),
                input: None,
                output: None,
                cache_read: None,
                cache_write: None,
                source: CostSource::ProviderReported,
                pricing_provider: Some("test".to_string()),
                pricing_model: Some("effective".to_string()),
                pricing_as_of: None,
                pricing_source: Some("provider".to_string()),
            }),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };

    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
    assert_eq!(summary.status, LlmOptimizationSummaryStatus::Partial);
    assert!(
        summary
            .limitations
            .contains(&"missing_actual_cost_total".to_string())
    );
    assert!(summary.estimated_cost_saved.is_none());
    assert!(summary.currency.is_none());
}

#[test]
fn exact_contribution_envelope_boundary_is_accepted() {
    let mut template = LlmOptimizationContribution::new("", "custom");
    template.id = Some(uuid::Uuid::nil());
    template.sequence = Some(0);
    let fixed_size = bounded_json_size(&template, usize::MAX).unwrap();
    let boundary_producer_len = MAX_LLM_OPTIMIZATION_CONTRIBUTION_BYTES - fixed_size;

    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(LlmOptimizationContribution::new(
        "x".repeat(boundary_producer_len),
        "custom",
    )));
    let accepted = recorder.finish();
    assert_eq!(accepted.contributions.len(), 1);
    assert_eq!(
        bounded_json_size(
            &accepted.contributions[0],
            MAX_LLM_OPTIMIZATION_CONTRIBUTION_BYTES,
        )
        .unwrap(),
        MAX_LLM_OPTIMIZATION_CONTRIBUTION_BYTES
    );

    let overflow = LlmOptimizationRecorder::default();
    assert!(!overflow.record(LlmOptimizationContribution::new(
        "x".repeat(boundary_producer_len + 1),
        "custom",
    )));
}

#[test]
fn prompt_completion_fallback_total_uses_checked_arithmetic() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(compression_contribution(
        "fallback-overflow",
        LlmOptimizationTokens {
            prompt_tokens: Some(u64::MAX),
            completion_tokens: Some(1),
            total_tokens: None,
            ..LlmOptimizationTokens::default()
        },
    )));
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            total_tokens: Some(2),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };

    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
    assert_eq!(summary.tokens_saved.prompt_tokens, Some(u64::MAX));
    assert_eq!(summary.tokens_saved.completion_tokens, Some(1));
    assert_eq!(summary.tokens_saved.total_tokens, None);
    assert_eq!(summary.baseline_usage.as_ref().unwrap().total_tokens, None);
    assert!(
        summary
            .limitations
            .contains(&"token_count_overflow".to_string())
    );
}

#[test]
fn one_overflowed_token_category_does_not_erase_unaffected_categories() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(compression_contribution(
        "first",
        LlmOptimizationTokens {
            prompt_tokens: Some(u64::MAX),
            completion_tokens: Some(3),
            ..LlmOptimizationTokens::default()
        },
    )));
    assert!(recorder.record(compression_contribution(
        "second",
        LlmOptimizationTokens {
            prompt_tokens: Some(1),
            completion_tokens: Some(4),
            ..LlmOptimizationTokens::default()
        },
    )));
    let mut response = AnnotatedLlmResponse {
        model: Some("effective".to_string()),
        usage: Some(Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            ..Usage::default()
        }),
        ..AnnotatedLlmResponse::default()
    };

    let summary =
        finalize_optimization_summary(&recorder, Some(&mut response), None, &resolver()).unwrap();
    assert_eq!(summary.tokens_saved.prompt_tokens, None);
    assert_eq!(summary.tokens_saved.completion_tokens, Some(7));
    assert_eq!(
        summary.baseline_usage.as_ref().unwrap().completion_tokens,
        Some(12)
    );
    assert!(
        summary
            .limitations
            .contains(&"token_count_overflow".to_string())
    );
}

fn synthetic_cost(total: Option<f64>, input: Option<f64>, currency: &str) -> CostEstimate {
    CostEstimate {
        total,
        currency: currency.to_string(),
        input,
        output: None,
        cache_read: None,
        cache_write: None,
        source: CostSource::ProviderReported,
        pricing_provider: None,
        pricing_model: None,
        pricing_as_of: None,
        pricing_source: None,
    }
}

#[test]
fn cost_savings_requires_complete_totals_and_matching_currency() {
    let empty_usd = synthetic_cost(None, None, "USD");
    let complete_usd = synthetic_cost(Some(2.0), None, "USD");

    let mut baseline_empty = Vec::new();
    assert_eq!(
        calculate_estimated_cost_saved(Some(&empty_usd), Some(&complete_usd), &mut baseline_empty,),
        (None, None)
    );
    assert_eq!(baseline_empty, vec!["missing_baseline_cost_total"]);

    let mut actual_empty = Vec::new();
    assert_eq!(
        calculate_estimated_cost_saved(Some(&complete_usd), Some(&empty_usd), &mut actual_empty,),
        (None, None)
    );
    assert_eq!(actual_empty, vec!["missing_actual_cost_total"]);

    let mut both_empty = Vec::new();
    assert_eq!(
        calculate_estimated_cost_saved(Some(&empty_usd), Some(&empty_usd), &mut both_empty),
        (None, None)
    );
    assert_eq!(
        both_empty,
        vec!["missing_baseline_cost_total", "missing_actual_cost_total"]
    );

    let baseline_components = synthetic_cost(None, Some(3.5), "USD");
    let actual_components = synthetic_cost(None, Some(1.25), "USD");
    let mut component_only = Vec::new();
    assert_eq!(
        calculate_estimated_cost_saved(
            Some(&baseline_components),
            Some(&actual_components),
            &mut component_only,
        ),
        (Some(2.25), Some("USD".to_string()))
    );
    assert!(component_only.is_empty());

    let eur = synthetic_cost(Some(1.0), None, "EUR");
    let mut mismatch = Vec::new();
    assert_eq!(
        calculate_estimated_cost_saved(Some(&complete_usd), Some(&eur), &mut mismatch),
        (None, None)
    );
    assert_eq!(mismatch, vec!["cost_currency_mismatch"]);
}

#[test]
fn missing_effective_model_is_an_explicit_limitation() {
    let recorder = LlmOptimizationRecorder::default();
    assert!(recorder.record(compression_contribution(
        "compressor",
        LlmOptimizationTokens::saved_prompt(3),
    )));
    let mut response = AnnotatedLlmResponse {
        model: None,
        usage: Some(Usage {
            prompt_tokens: Some(7),
            total_tokens: Some(7),
            cost: Some(synthetic_cost(Some(0.1), None, "USD")),
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
    assert!(summary.effective_model.is_none());
    assert!(summary.baseline_model.is_none());
    assert!(
        summary
            .limitations
            .contains(&"missing_effective_model".to_string())
    );
}
