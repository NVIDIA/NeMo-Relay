// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral coverage for independently owned, bounded diagnostic recording.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::thread;

use nemo_relay::diagnostics::{
    DiagnosticCorrelation, DiagnosticError, DiagnosticLimits, DiagnosticObservation,
    DiagnosticOutcome, DiagnosticQuery, DiagnosticSeverity, Diagnostics, SourceIdentity,
};
use uuid::Uuid;

fn diagnostics(limits: DiagnosticLimits) -> Diagnostics {
    Diagnostics::new(
        SourceIdentity {
            instance_id: Uuid::now_v7(),
            process_id: Some(std::process::id()),
            process_role: "test".to_owned(),
            relay_version: "test".to_owned(),
        },
        limits,
    )
}

#[test]
fn standalone_services_work_before_logging_and_do_not_share_state() {
    let first = diagnostics(DiagnosticLimits::default());
    let second = diagnostics(DiagnosticLimits::default());
    let component = first.register_component("exporter.logs").unwrap();
    let receipt = component
        .observe_failure(
            "otel.export_failed",
            "unavailable",
            5,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    assert_eq!(receipt.occurrence_count, 5);
    let view = component.snapshot().unwrap();
    assert_eq!(view.counters[0].occurrences.value.get(), 5);
    assert_eq!(view.counters[0].last_message, "unavailable");
    let observed = first.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(observed.events.len(), 1);
    assert_eq!(observed.events[0].outcome, DiagnosticOutcome::Failure);
    assert_eq!(observed.events[0].count.get(), 5);
    assert_ne!(first.source().instance_id, second.source().instance_id);
    assert!(
        second
            .snapshot(DiagnosticQuery::default())
            .unwrap()
            .components
            .is_empty()
    );
    let before = first.snapshot(DiagnosticQuery::default()).unwrap();
    let after = first.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(before.sequence, after.sequence);
    assert_eq!(before.components, after.components);
    assert_eq!(before.events, after.events);
    assert_eq!(before.coverage, after.coverage);
}

#[test]
fn success_and_failure_observations_keep_independent_counts_and_history() {
    let service = diagnostics(DiagnosticLimits::default());
    let component = service.register_component("exporter.logs").unwrap();
    component
        .observe_failure(
            "otel.export_failed",
            "unavailable",
            2,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    let success = component
        .observe_success(
            "otel.export_succeeded",
            "export completed",
            3,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    assert_eq!(success.occurrence_count, 3);
    component
        .observe_failure(
            "otel.export_failed",
            "failed again",
            1,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    component
        .record(DiagnosticObservation::success(
            "otel.export_succeeded",
            "another export completed",
            2,
        ))
        .unwrap();
    let snapshot = service.snapshot(DiagnosticQuery::default()).unwrap();
    let counters = &snapshot.components[0].counters;
    assert_eq!(counters.len(), 2);
    assert_eq!(counters[0].code, "otel.export_failed");
    assert_eq!(counters[0].occurrences.value.get(), 3);
    assert_eq!(counters[0].last_message, "failed again");
    assert_eq!(counters[1].code, "otel.export_succeeded");
    assert_eq!(counters[1].occurrences.value.get(), 5);
    assert_eq!(
        snapshot
            .events
            .iter()
            .map(|event| event.outcome)
            .collect::<Vec<_>>(),
        vec![
            DiagnosticOutcome::Failure,
            DiagnosticOutcome::Success,
            DiagnosticOutcome::Failure,
            DiagnosticOutcome::Success
        ]
    );
    assert_eq!(snapshot.events[1].severity, DiagnosticSeverity::Info);
    let json = serde_json::to_value(snapshot).unwrap();
    assert!(json["components"][0].get("conditions").is_none());
}

#[test]
fn last_recording_handle_closes_component_and_retains_observation_history() {
    let service = diagnostics(DiagnosticLimits::default());
    let component = service.register_component("exporter.logs").unwrap();
    let id = component.identity().instance_id;
    let reader = component.reader();
    let cloned = component.clone();
    component
        .observe_failure(
            "otel.export_failed",
            "unavailable",
            1,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    drop(component);
    assert!(reader.component_snapshot(id).unwrap().closed_at.is_none());
    drop(cloned);
    let retained = reader.component_snapshot(id).unwrap();
    assert!(retained.closed_at.is_some());
    assert_eq!(retained.counters[0].occurrences.value.get(), 1);
    let history = reader
        .snapshot(DiagnosticQuery {
            component_instance_id: Some(id),
            ..DiagnosticQuery::default()
        })
        .unwrap();
    assert_eq!(history.events.len(), 1);
    assert_eq!(history.events[0].outcome, DiagnosticOutcome::Failure);
}

#[test]
fn closed_components_are_evicted_before_live_registrations_and_generations_advance() {
    let service = diagnostics(DiagnosticLimits {
        max_components: 1,
        ..DiagnosticLimits::default()
    });
    let first = service.register_component("logging.file").unwrap();
    let first_identity = first.identity().clone();
    assert!(matches!(
        service.register_component("other"),
        Err(DiagnosticError::ComponentCapacity)
    ));
    first.close();
    assert!(!first.close());
    assert!(matches!(
        first.observe_failure(
            "logging.failed",
            "closed",
            1,
            DiagnosticCorrelation::default()
        ),
        Err(DiagnosticError::ComponentClosed)
    ));
    let second = service.register_component("logging.file").unwrap();
    assert_ne!(second.identity().instance_id, first_identity.instance_id);
    assert!(second.identity().generation > first_identity.generation);
    let view = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(view.components.len(), 1);
    assert_eq!(view.coverage.loss.components_evicted.value.get(), 1);
    assert_eq!(view.coverage.loss.components_rejected.value.get(), 1);
}

#[test]
fn concurrent_occurrences_are_counted_once_even_when_recent_events_are_evicted() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 7,
        ..DiagnosticLimits::default()
    });
    let component = service.register_component("exporter.logs").unwrap();
    let threads = (0..6)
        .map(|_| {
            let component = component.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    component
                        .observe_failure(
                            "otel.export_failed",
                            "unavailable",
                            1,
                            DiagnosticCorrelation::default(),
                        )
                        .unwrap();
                }
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }
    let view = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(view.components[0].counters[0].occurrences.value.get(), 600);
    assert_eq!(view.events.len(), 7);
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 593);
    assert!(
        view.events
            .windows(2)
            .all(|pair| pair[0].sequence < pair[1].sequence)
    );
}

#[test]
fn counter_overflow_is_explicit_and_other_codes_do_not_change_it() {
    let service = diagnostics(DiagnosticLimits::default());
    let component = service.register_component("exporter.logs").unwrap();
    let first = component
        .observe_failure(
            "otel.export_failed",
            "unavailable",
            u64::MAX,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    assert_eq!(first.occurrence_count, u64::MAX);
    assert!(!first.count_saturated);
    let second = component
        .observe_failure(
            "otel.export_failed",
            "unavailable",
            1,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    assert_eq!(second.occurrence_count, u64::MAX);
    assert!(second.count_saturated);
    component
        .observe_success(
            "otel.export_succeeded",
            "export completed",
            1,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    let view = component.snapshot().unwrap();
    assert_eq!(view.counters[0].occurrences.value.get(), u64::MAX);
    assert!(view.counters[0].occurrences.saturated);
    assert_eq!(view.counters[1].occurrences.value.get(), 1);
    assert!(!view.counters[1].occurrences.saturated);
    let json = serde_json::to_value(view).unwrap();
    assert_eq!(
        json["counters"][0]["occurrences"]["value"],
        u64::MAX.to_string()
    );
}

#[test]
fn keys_are_bounded_and_correlation_does_not_create_new_aggregate_keys() {
    let service = diagnostics(DiagnosticLimits {
        max_keys_per_component: 1,
        max_message_bytes: 7,
        max_correlation_bytes: 4,
        ..DiagnosticLimits::default()
    });
    let component = service.register_component("exporter.logs").unwrap();
    for index in 0..3 {
        component
            .observe_failure(
                "otel.export_failed",
                "é🙂unavailable",
                1,
                DiagnosticCorrelation {
                    session_id: Some(format!("session-{index}")),
                    request_id: Some(format!("request-{index}")),
                },
            )
            .unwrap();
    }
    assert!(matches!(
        component.observe_failure(
            "otel.other_failure",
            "other",
            1,
            DiagnosticCorrelation::default()
        ),
        Err(DiagnosticError::KeyCapacity)
    ));
    assert!(matches!(
        component.observe_failure(
            "otel.bad code",
            "other",
            1,
            DiagnosticCorrelation::default()
        ),
        Err(DiagnosticError::InvalidCode)
    ));
    let view = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(view.components[0].counters.len(), 1);
    assert_eq!(view.components[0].counters[0].occurrences.value.get(), 3);
    assert!(view.events.iter().all(|event| event.message.len() <= 7
        && event.correlation.session_id.as_ref().unwrap().len() <= 4
        && event.correlation.request_id.as_ref().unwrap().len() <= 4));
    assert_eq!(view.coverage.loss.messages_truncated.value.get(), 9);
    assert_eq!(view.coverage.loss.keys_rejected.value.get(), 2);
    assert_eq!(view.coverage.loss.observations_rejected.value.get(), 2);
}

#[test]
fn queries_filter_evidence_without_changing_counters_lifetimes_or_history() {
    let service = diagnostics(DiagnosticLimits::default());
    let first = service.register_component("exporter.logs").unwrap();
    let second = service.register_component("exporter.metrics").unwrap();
    let failure = first
        .observe_failure(
            "otel.export_failed",
            "unavailable",
            3,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    first
        .observe_success(
            "otel.export_succeeded",
            "export completed",
            2,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    second
        .observe_failure(
            "otel.export_failed",
            "unavailable",
            4,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    let before = service.snapshot(DiagnosticQuery::default()).unwrap();
    let filtered = service
        .snapshot(DiagnosticQuery {
            component_instance_id: Some(first.identity().instance_id),
            code: Some("otel.export_failed".to_owned()),
            after_sequence: Some(failure.sequence),
            ..DiagnosticQuery::default()
        })
        .unwrap();
    assert_eq!(filtered.components.len(), 1);
    assert_eq!(filtered.components[0].counters.len(), 1);
    assert_eq!(
        filtered.components[0].counters[0].occurrences.value.get(),
        3
    );
    assert!(
        filtered.events.is_empty(),
        "the event cursor does not reset cumulative counts"
    );
    let after = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(before.sequence, after.sequence);
    assert_eq!(before.components, after.components);
    assert_eq!(before.events, after.events);
    assert_eq!(before.coverage, after.coverage);
}

#[test]
fn event_retention_is_independent_of_projection_and_cumulative_state() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 0,
        ..DiagnosticLimits::default()
    });
    let component = service.register_component("exporter.logs").unwrap();
    let projected = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&projected);
    let guard = service
        .subscribe_projection(move |_| {
            count.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    component
        .observe_failure(
            "otel.export_failed",
            "unavailable",
            2,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    assert_eq!(projected.load(Ordering::Relaxed), 1);
    let view = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert!(view.events.is_empty());
    assert_eq!(view.components[0].counters[0].occurrences.value.get(), 2);
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 1);
    drop(guard);
    component
        .observe_failure(
            "otel.export_failed",
            "unavailable",
            1,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    assert_eq!(projected.load(Ordering::Relaxed), 1);
}

#[test]
fn projections_can_read_and_record_without_deadlocks_or_recursive_projection() {
    let service = diagnostics(DiagnosticLimits::default());
    let component = service.register_component("logging.file").unwrap();
    let reader = service.reader();
    let callback_component = component.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let _guard = service
        .subscribe_projection(move |_| {
            callback_calls.fetch_add(1, Ordering::Relaxed);
            assert!(
                !reader
                    .snapshot(DiagnosticQuery::default())
                    .unwrap()
                    .events
                    .is_empty()
            );
            callback_component
                .record_without_projection(DiagnosticObservation::failure(
                    "logging.sink_failed",
                    "sink unavailable",
                    1,
                ))
                .unwrap();
            callback_component
                .record(DiagnosticObservation::occurrence(
                    "logging.fallback_used",
                    DiagnosticSeverity::Warning,
                    "fallback",
                    1,
                ))
                .unwrap();
        })
        .unwrap();
    component
        .observe_failure(
            "logging.init_failed",
            "unavailable",
            1,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let view = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(view.events.len(), 3);
    assert_eq!(view.components[0].counters.len(), 3);
    assert_eq!(view.coverage.loss.projections_suppressed.value.get(), 1);
    assert_eq!(view.coverage.loss.projections_panicked.value.get(), 0);
}

#[test]
fn retained_byte_rejection_rolls_back_counter_and_event_together() {
    let service = diagnostics(DiagnosticLimits {
        max_retained_bytes: 600,
        ..DiagnosticLimits::default()
    });
    let component = service.register_component("exporter.logs").unwrap();
    let result = component.observe_failure(
        "otel.export_failed",
        "x".repeat(400),
        5,
        DiagnosticCorrelation::default(),
    );
    assert_eq!(result, Err(DiagnosticError::ByteCapacity));
    let view = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert!(view.components[0].counters.is_empty());
    assert!(view.events.is_empty());
    assert_eq!(view.coverage.loss.observations_rejected.value.get(), 1);
}

#[test]
fn snapshot_byte_truncation_is_explicit_and_does_not_evict_retained_state() {
    let service = diagnostics(DiagnosticLimits {
        max_snapshot_bytes: 2_000,
        ..DiagnosticLimits::default()
    });
    let component = service.register_component("exporter.logs").unwrap();
    component
        .observe_failure(
            "otel.export_failed",
            "x".repeat(2_000),
            7,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    let view = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert!(view.coverage.retention.query_truncated);
    assert!(serde_json::to_vec(&view).unwrap().len() <= 2_000);
    assert_eq!(
        view.coverage.retention.components_retained.get(),
        view.components.len() as u64
    );
    assert_eq!(
        view.coverage.retention.events_retained.get(),
        view.events.len() as u64
    );
    let retained = component.snapshot().unwrap();
    assert_eq!(retained.counters[0].occurrences.value.get(), 7);
    assert_eq!(retained.counters[0].last_message.len(), 2_000);
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 0);
}

#[test]
fn projection_panics_are_isolated_visible_and_do_not_undo_acceptance() {
    let service = diagnostics(DiagnosticLimits::default());
    let component = service.register_component("exporter.logs").unwrap();
    let _guard = service
        .subscribe_projection(|_| panic!("projection failed"))
        .unwrap();
    let delivered = Arc::new(AtomicUsize::new(0));
    let callback_delivered = Arc::clone(&delivered);
    let _healthy_guard = service
        .subscribe_projection(move |_| {
            callback_delivered.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    let receipt = component
        .observe_failure(
            "otel.export_failed",
            "unavailable",
            1,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    assert_eq!(receipt.occurrence_count, 1);
    let view = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(view.components[0].counters[0].occurrences.value.get(), 1);
    assert_eq!(view.coverage.loss.projections_panicked.value.get(), 1);
    assert_eq!(delivered.load(Ordering::Relaxed), 1);
}

#[test]
fn codes_accept_numeric_segments_and_reject_unbounded_or_empty_segments() {
    use nemo_relay::diagnostics::DiagnosticCode;
    assert!(DiagnosticCode::new("diagnostic.00").is_ok());
    for invalid in [
        "",
        "diagnostic..failed",
        "Diagnostic.failed",
        "diagnostic.failed/123",
        &"x".repeat(129),
    ] {
        assert_eq!(
            DiagnosticCode::new(invalid),
            Err(DiagnosticError::InvalidCode)
        );
    }
}

#[test]
fn zero_count_observations_are_rejected_without_mutating_evidence() {
    let service = diagnostics(DiagnosticLimits::default());
    let component = service.register_component("exporter.logs").unwrap();
    component
        .observe_failure(
            "otel.export_failed",
            "unavailable",
            2,
            DiagnosticCorrelation::default(),
        )
        .unwrap();
    let before = service.snapshot(DiagnosticQuery::default()).unwrap();
    for observation in [
        DiagnosticObservation::failure("otel.zero_failure", "no occurrences", 0),
        DiagnosticObservation::success("otel.zero_success", "no occurrences", 0),
        DiagnosticObservation::occurrence(
            "otel.zero_observation",
            DiagnosticSeverity::Info,
            "no occurrences",
            0,
        ),
    ] {
        assert_eq!(
            component.record(observation),
            Err(DiagnosticError::InvalidCount)
        );
    }
    let after = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(before.sequence, after.sequence);
    assert_eq!(before.components, after.components);
    assert_eq!(before.events, after.events);
    assert_eq!(after.coverage.loss.observations_rejected.value.get(), 3);
    assert_eq!(after.coverage.loss.keys_rejected.value.get(), 0);
}

#[test]
fn independent_service_projections_can_chain_but_cycles_are_suppressed() {
    let first = diagnostics(DiagnosticLimits::default());
    let second = diagnostics(DiagnosticLimits::default());
    let first_component = first.register_component("first").unwrap();
    let second_component = second.register_component("second").unwrap();
    let first_calls = Arc::new(AtomicUsize::new(0));
    let second_calls = Arc::new(AtomicUsize::new(0));
    let second_recorder = second_component.clone();
    let first_count = Arc::clone(&first_calls);
    let _first_guard = first
        .subscribe_projection(move |_| {
            first_count.fetch_add(1, Ordering::Relaxed);
            second_recorder
                .record(DiagnosticObservation::occurrence(
                    "source.forwarded",
                    DiagnosticSeverity::Info,
                    "forwarded",
                    1,
                ))
                .unwrap();
        })
        .unwrap();
    let first_recorder = first_component.clone();
    let second_count = Arc::clone(&second_calls);
    let _second_guard = second
        .subscribe_projection(move |_| {
            second_count.fetch_add(1, Ordering::Relaxed);
            first_recorder
                .record(DiagnosticObservation::occurrence(
                    "source.roundtrip",
                    DiagnosticSeverity::Info,
                    "roundtrip",
                    1,
                ))
                .unwrap();
        })
        .unwrap();
    first_component
        .record(DiagnosticObservation::occurrence(
            "source.origin",
            DiagnosticSeverity::Info,
            "origin",
            1,
        ))
        .unwrap();
    assert_eq!(first_calls.load(Ordering::Relaxed), 1);
    assert_eq!(second_calls.load(Ordering::Relaxed), 1);
    let first_view = first.snapshot(DiagnosticQuery::default()).unwrap();
    let second_view = second.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(first_view.events.len(), 2);
    assert_eq!(second_view.events.len(), 1);
    assert_eq!(
        first_view.coverage.loss.projections_suppressed.value.get(),
        1
    );
    assert_eq!(
        second_view.coverage.loss.projections_suppressed.value.get(),
        0
    );
}

#[test]
fn projection_chain_depth_is_bounded_and_later_independent_delivery_still_works() {
    let services = (0..64)
        .map(|_| diagnostics(DiagnosticLimits::default()))
        .collect::<Vec<_>>();
    let components = services
        .iter()
        .map(|service| service.register_component("projection").unwrap())
        .collect::<Vec<_>>();
    let delivered = Arc::new(AtomicUsize::new(0));
    let mut guards = Vec::new();
    for (index, service) in services.iter().enumerate() {
        let next = components.get(index + 1).cloned();
        let callback_delivered = Arc::clone(&delivered);
        guards.push(
            service
                .subscribe_projection(move |_| {
                    callback_delivered.fetch_add(1, Ordering::Relaxed);
                    if let Some(next) = &next {
                        next.record(DiagnosticObservation::occurrence(
                            "source.forwarded",
                            DiagnosticSeverity::Info,
                            "forwarded",
                            1,
                        ))
                        .unwrap();
                    }
                })
                .unwrap(),
        );
    }
    components[0]
        .record(DiagnosticObservation::occurrence(
            "source.origin",
            DiagnosticSeverity::Info,
            "origin",
            1,
        ))
        .unwrap();
    assert!(delivered.load(Ordering::Relaxed) < services.len());
    let suppressed = services
        .iter()
        .map(|service| {
            service
                .snapshot(DiagnosticQuery::default())
                .unwrap()
                .coverage
                .loss
                .projections_suppressed
                .value
                .get()
        })
        .sum::<u64>();
    assert_eq!(suppressed, 1);
    let before = delivered.load(Ordering::Relaxed);
    components
        .last()
        .unwrap()
        .record(DiagnosticObservation::occurrence(
            "source.later",
            DiagnosticSeverity::Info,
            "later",
            1,
        ))
        .unwrap();
    assert_eq!(delivered.load(Ordering::Relaxed), before + 1);
}

#[test]
fn retained_byte_budget_remains_exact_across_replacements_and_rollback() {
    const BUDGET: usize = 3_000;
    let service = diagnostics(DiagnosticLimits {
        max_components: 2,
        max_events: 4,
        max_retained_bytes: BUDGET,
        ..DiagnosticLimits::default()
    });
    let mut components = [
        service.register_component("first").unwrap(),
        service.register_component("second").unwrap(),
    ];
    let mut rejected = 0;
    for index in 0..100 {
        let component = &components[index % 2];
        let previous_count = component
            .snapshot()
            .unwrap()
            .counters
            .first()
            .map_or(0, |counter| counter.occurrences.value.get());
        // Escapes expand on the wire; raw UTF-8 length is not a byte-budget proxy.
        let message = "\u{1}\"é".repeat((index % 13) * 15);
        let result =
            component.observe_failure("test.failed", message, 1, DiagnosticCorrelation::default());
        let count = component
            .snapshot()
            .unwrap()
            .counters
            .first()
            .map_or(0, |counter| counter.occurrences.value.get());
        match result {
            Ok(receipt) => {
                assert_eq!(count, previous_count + 1);
                assert_eq!(receipt.occurrence_count, count);
            }
            Err(DiagnosticError::ByteCapacity) => {
                rejected += 1;
                assert_eq!(count, previous_count);
            }
            unexpected => panic!("unexpected recording outcome: {unexpected:?}"),
        }
        if index % 11 == 0 {
            let slot = index % 2;
            components[slot].close();
            components[slot] = service
                .register_component(if slot == 0 { "first" } else { "second" })
                .unwrap();
        }
        let snapshot = service.snapshot(DiagnosticQuery::default()).unwrap();
        let bytes = snapshot
            .components
            .iter()
            .map(|component| serde_json::to_vec(component).unwrap().len())
            .chain(
                snapshot
                    .events
                    .iter()
                    .map(|event| serde_json::to_vec(event).unwrap().len()),
            )
            .sum::<usize>();
        assert!(bytes <= BUDGET, "retained {bytes} encoded bytes");
    }
    assert!(rejected > 0, "exercise atomic byte-budget rollback");
    // A shrinking replacement must release its cached bytes rather than retaining
    // an accumulated size from previous rejected or superseded observations.
    components.sort_by_cached_key(|component| {
        std::cmp::Reverse(
            serde_json::to_vec(&component.snapshot().unwrap())
                .unwrap()
                .len(),
        )
    });
    for component in &components {
        component
            .observe_failure(
                "test.failed",
                "compact",
                1,
                DiagnosticCorrelation::default(),
            )
            .unwrap();
    }
}
