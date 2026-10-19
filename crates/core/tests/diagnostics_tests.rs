// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral contracts for the runtime-controlled bounded diagnostic foundation.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::thread;

use nemo_relay::diagnostics::*;
use uuid::Uuid;

thread_local! {
    // Other parallel tests and their runtime threads are deliberately excluded.
    static TRACKED_ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

struct CountingAllocator;

fn note_allocation() {
    let _ = TRACKED_ALLOCATIONS.try_with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value.saturating_add(1)));
        }
    });
}

// SAFETY: allocation/deallocation requests are forwarded unchanged to System.
// The optional thread-local counter does not allocate or expose allocator memory.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        note_allocation();
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations_during<T>(body: impl FnOnce() -> T) -> (T, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TRACKED_ALLOCATIONS.with(|count| count.set(None));
        }
    }
    TRACKED_ALLOCATIONS.with(|count| assert!(count.replace(Some(0)).is_none()));
    let reset = Reset;
    let value = body();
    let count = TRACKED_ALLOCATIONS.with(|count| count.replace(None).unwrap());
    drop(reset);
    (value, count)
}

const WRITE_FAILURE: FailureCode = FailureCode::Logging(LoggingFailureCode::SinkWriteFailed);
const WRITE: BuiltinDiagnosticCode = BuiltinDiagnosticCode::Failure(WRITE_FAILURE);
const SUCCESS: BuiltinDiagnosticCode = BuiltinDiagnosticCode::Success(SuccessCode::SinkInitialized);
const NOTICE: BuiltinDiagnosticCode = BuiltinDiagnosticCode::Notice(NoticeCode::GuardrailRejected);
const CODES: [BuiltinDiagnosticCode; 3] = [WRITE, SUCCESS, NOTICE];

fn limits() -> DiagnosticLimits {
    DiagnosticLimits {
        max_components: 4,
        max_counters: 64,
        max_events: 256,
        max_window_buckets: 32,
        max_extensions: 4,
        max_metadata_bytes: 4096,
        max_detail_bytes: 512,
        max_correlation_bytes: 32,
        detail_pool_bytes: 16_384,
        detail_block_bytes: 64,
        max_queries: 2,
        max_query_components: 8,
        max_query_events: 256,
        max_snapshot_bytes: 262_144,
        query_text_bytes: 65_536,
        max_projections: 4,
        max_projection_workspaces: 4,
        max_owned_bytes: 8 * 1024 * 1024,
        ..DiagnosticLimits::default()
    }
}

fn source() -> SourceIdentity {
    SourceIdentity {
        instance_id: Uuid::now_v7(),
        process_id: Some(std::process::id()),
        process_role: "test".into(),
        relay_version: "test".into(),
    }
}

fn diagnostics(limits: DiagnosticLimits) -> Diagnostics {
    Diagnostics::from_settings(
        source(),
        DiagnosticsSettings {
            enabled: true,
            limits,
        },
    )
    .unwrap()
}

fn register(service: &Diagnostics, key: &str) -> ComponentRecorder {
    service
        .register_component(ComponentRegistration::new(key, &CODES))
        .unwrap()
}

fn classification(class: FailureClass) -> Classification {
    Classification::failure(WRITE_FAILURE, class).unwrap()
}

fn record(recorder: &ComponentRecorder, count: u64) -> RecordReceipt {
    match recorder
        .record_without_projection_lazy(|| {
            Observation::new(classification(FailureClass::Unknown), "failure", count)
        })
        .unwrap()
    {
        RecordOutcome::Recorded(receipt) => receipt,
        RecordOutcome::Disabled => panic!("fixture must be enabled"),
    }
}

fn snapshot(service: &Diagnostics, query: DiagnosticQuery) -> DiagnosticSnapshot {
    service.snapshot(query).unwrap().to_snapshot().unwrap()
}

fn row<'a>(view: &'a DiagnosticSnapshot, key: &str) -> &'a ComponentSnapshot {
    view.components
        .iter()
        .find(|row| row.identity.key == key)
        .unwrap()
}

fn code_total(row: &ComponentSnapshot, code: BuiltinDiagnosticCode) -> u64 {
    row.counters
        .iter()
        .find(|counter| counter.code == code.canonical_name())
        .map_or(0, |counter| counter.occurrences.value.get())
}

#[test]
fn disabled_facade_has_no_identity_backend_or_lazy_work() {
    let service = Diagnostics::default();
    assert_eq!(
        service.reader().availability(),
        DiagnosticAvailability::Disabled
    );
    assert_eq!(service.memory_usage().reserved_bytes, 0);
    let (_, allocations) = allocations_during(|| {
        let (owner, recorder) = service
            .register_owned_component(ComponentRegistration::new("disabled", &CODES))
            .unwrap();
        assert!(recorder.identity().is_none());
        assert_eq!(
            recorder
                .record_lazy(|| panic!("disabled factory ran"))
                .unwrap(),
            RecordOutcome::Disabled
        );
        assert_eq!(
            recorder
                .capture_origin()
                .unwrap()
                .record_without_projection_lazy(|| panic!("disabled origin factory ran"))
                .unwrap(),
            RecordOutcome::Disabled
        );
        assert_eq!(
            service.snapshot(DiagnosticQuery::default()).unwrap_err(),
            DiagnosticError::Disabled
        );
        let _guard = service
            .subscribe_projection(|_| panic!("disabled projection ran"))
            .unwrap();
        assert!(!owner.close().unwrap());
        assert!(
            owner
                .refresh(ComponentOwnership::default())
                .unwrap()
                .is_none()
        );
    });
    assert_eq!(allocations, 0);
    assert_eq!(service.memory_usage().reserved_bytes, 0);
    assert_eq!(service.memory_usage().components, 0);
    assert_eq!(service.memory_usage().query_workspaces, 0);
}

#[test]
fn an_early_process_default_cannot_be_silently_upgraded() {
    let early = Diagnostics::process_default();
    assert_eq!(
        early.reader().availability(),
        DiagnosticAvailability::Disabled
    );
    let identity = early.source().instance_id;
    assert_eq!(
        Diagnostics::install_process_default(source(), DiagnosticsSettings::default()).unwrap_err(),
        DiagnosticError::AlreadyInitialized
    );
    assert_eq!(
        Diagnostics::process_default().source().instance_id,
        identity
    );
}

#[test]
fn independent_contexts_record_before_logging_and_reads_do_not_recount() {
    let first = diagnostics(limits());
    let second = diagnostics(limits());
    let component = register(&first, "producer");
    let accepted = record(&component, 5);
    assert_eq!(accepted.code_total.unwrap().value.get(), 5);
    let before = snapshot(&first, DiagnosticQuery::default());
    let after = snapshot(&first, DiagnosticQuery::default());
    assert_ne!(first.source().instance_id, second.source().instance_id);
    assert!(
        snapshot(&second, DiagnosticQuery::default())
            .components
            .is_empty()
    );
    assert_eq!(before.sequence, after.sequence);
    assert_eq!(before.events, after.events);
    assert_eq!(before.components[0].counters, after.components[0].counters);
    assert_eq!(before.coverage, after.coverage);
    assert_eq!(
        row(&after, "producer")
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        1
    );
    assert!(
        after
            .coverage
            .capabilities
            .iter()
            .any(|capability| capability.name == "otel"
                && capability.availability == DiagnosticAvailability::Unsupported
                && capability.reason.as_deref() == Some("not_integrated"))
    );
}

#[test]
fn two_and_three_input_deltas_produce_five_without_forwarding_totals() {
    let service = diagnostics(limits());
    let subscriber = Uuid::now_v7();
    let registration = |key: &'static str| ComponentRegistration {
        key,
        builtin_codes: &CODES,
        ownership: ComponentOwnership {
            subscriber_instance_id: Some(subscriber),
            ..ComponentOwnership::default()
        },
        window: None,
    };
    let first = service
        .register_component(registration("pipeline.a"))
        .unwrap();
    let second = service
        .register_component(registration("pipeline.b"))
        .unwrap();
    record(&first, 2);
    record(&second, 3);
    let view = snapshot(
        &service,
        DiagnosticQuery {
            ownership: ComponentOwnership {
                subscriber_instance_id: Some(subscriber),
                ..ComponentOwnership::default()
            },
            ..DiagnosticQuery::default()
        },
    );
    assert_eq!(view.components.len(), 2);
    assert_eq!(
        view.components
            .iter()
            .map(|row| code_total(row, WRITE))
            .sum::<u64>(),
        5
    );
    assert_eq!(
        view.components
            .iter()
            .map(|row| row.lifetime.totals.recorded_events_total.value.get())
            .sum::<u64>(),
        2
    );
    assert_eq!(
        view.events
            .iter()
            .map(|event| event.count.get())
            .sum::<u64>(),
        5
    );
}

#[test]
fn shared_ring_retains_latest_256_and_counts_all_257() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    for _ in 0..257 {
        record(&component, 1);
    }
    let view = snapshot(&service, DiagnosticQuery::default());
    let summary = &row(&view, "producer").lifetime.totals;
    assert_eq!(view.events.len(), 256);
    assert_eq!(summary.recorded_events_total.value.get(), 257);
    assert_eq!(summary.occurrences_total.value.get(), 257);
    assert_eq!(summary.history_evictions_total.value.get(), 1);
    assert_eq!(code_total(row(&view, "producer"), WRITE), 257);
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 1);
    assert!(
        view.events
            .windows(2)
            .all(|pair| pair[0].sequence < pair[1].sequence)
    );
}

#[test]
fn evictions_are_charged_to_the_displaced_owner() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 1,
        ..limits()
    });
    let first = register(&service, "first");
    let second = register(&service, "second");
    record(&first, 100);
    record(&second, 1);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(
        row(&view, "first")
            .lifetime
            .totals
            .history_evictions_total
            .value
            .get(),
        1
    );
    assert_eq!(
        row(&view, "second")
            .lifetime
            .totals
            .history_evictions_total
            .value
            .get(),
        0
    );
    assert_eq!(
        row(&view, "first")
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        1
    );
    assert_eq!(
        row(&view, "first")
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        100
    );
    assert_eq!(view.events[0].component.key, "second");
}

#[test]
fn detail_pressure_reclaims_blocks_and_impossible_detail_preserves_old_evidence() {
    let service = diagnostics(DiagnosticLimits {
        detail_pool_bytes: 8,
        detail_block_bytes: 4,
        ..limits()
    });
    let component = register(&service, "producer");
    for _ in 0..3 {
        component
            .record_without_projection_lazy(|| {
                Observation::new(classification(FailureClass::Unknown), "abcdefgh", 1)
            })
            .unwrap();
    }
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.events.len(), 1);
    assert_eq!(view.events[0].message(), Some("abcdefgh"));
    assert_eq!(
        row(&view, "producer")
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        3
    );
    assert_eq!(
        row(&view, "producer")
            .lifetime
            .totals
            .history_evictions_total
            .value
            .get(),
        2
    );
    let accepted = component
        .record_without_projection_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "detail cannot fit",
                2,
            )
        })
        .unwrap();
    assert!(matches!(
        accepted,
        RecordOutcome::Recorded(RecordReceipt {
            evidence: EvidenceDisposition::Truncated,
            ..
        })
    ));
    let after = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(
        after.events.len(),
        2,
        "an impossible incoming detail must not evict older evidence"
    );
    assert!(after.events[1].detail.is_none());
    assert_eq!(after.coverage.loss.details_omitted.value.get(), 1);
    assert_eq!(
        row(&after, "producer")
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        5
    );
    assert_eq!(service.memory_usage().detail_bytes, 8);
}

#[test]
fn zero_history_still_counts_and_projects_without_fabricating_evictions() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 0,
        detail_pool_bytes: 0,
        ..limits()
    });
    let component = register(&service, "producer");
    let observed = Arc::new(AtomicUsize::new(0));
    let callback = Arc::clone(&observed);
    let _guard = service
        .subscribe_projection(move |_| {
            callback.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    component
        .record_lazy(|| Observation::new(classification(FailureClass::Unknown), "failure", 2))
        .unwrap();
    let view = snapshot(&service, DiagnosticQuery::default());
    assert!(view.events.is_empty());
    assert_eq!(view.coverage.loss.events_omitted.value.get(), 1);
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 0);
    assert_eq!(code_total(row(&view, "producer"), WRITE), 2);
    assert_eq!(observed.load(Ordering::Relaxed), 1);
    assert_eq!(service.memory_usage().detail_bytes, 0);
}

#[test]
fn utf8_detail_and_correlation_are_bounded_without_creating_keys() {
    let service = diagnostics(DiagnosticLimits {
        max_detail_bytes: 7,
        max_correlation_bytes: 4,
        ..limits()
    });
    let component = register(&service, "producer");
    for _ in 0..3 {
        component
            .record_without_projection_lazy(|| {
                Observation::new(classification(FailureClass::Unknown), "é🙂abcdef", 1)
                    .with_correlation(DiagnosticCorrelationRef {
                        session_id: Some("session"),
                        request_id: Some("🙂request"),
                    })
            })
            .unwrap();
    }
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(row(&view, "producer").counters.len(), 1);
    assert_eq!(code_total(row(&view, "producer"), WRITE), 3);
    assert!(view.events.iter().all(|event| {
        event
            .detail
            .as_ref()
            .is_some_and(|detail| detail.truncated && detail.message.len() <= 7)
            && event.correlation.session_id.as_ref().unwrap().len() <= 4
            && event.correlation.request_id.as_ref().unwrap().len() <= 4
    }));
    assert_eq!(view.coverage.loss.messages_truncated.value.get(), 3);
    assert_eq!(view.coverage.loss.correlations_truncated.value.get(), 6);
}

#[test]
fn live_registration_reservations_are_never_displaced() {
    let service = diagnostics(DiagnosticLimits {
        max_components: 1,
        max_counters: 1,
        ..limits()
    });
    let component = service
        .register_component(ComponentRegistration::new("first", &[WRITE]))
        .unwrap();
    let identity = component.identity().unwrap().clone();
    assert_eq!(
        service
            .register_component(ComponentRegistration::new("second", &[WRITE]))
            .unwrap_err(),
        DiagnosticError::ComponentCapacity
    );
    record(&component, 2);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.components[0].identity, identity);
    assert_eq!(code_total(&view.components[0], WRITE), 2);
    assert_eq!(view.coverage.loss.components_rejected.value.get(), 1);
}

#[test]
fn closed_metadata_reclamation_expires_dependent_events_and_stale_handles() {
    let service = diagnostics(DiagnosticLimits {
        max_components: 1,
        ..limits()
    });
    let (owner, first) = service
        .register_owned_component(ComponentRegistration::new("producer", &CODES))
        .unwrap();
    let old = first.identity().unwrap().clone();
    record(&first, 1);
    assert!(owner.close().unwrap());
    assert!(!owner.close().unwrap());
    let second = register(&service, "producer");
    let view = snapshot(&service, DiagnosticQuery::default());
    assert!(view.events.is_empty());
    assert_ne!(old.instance_id, second.identity().unwrap().instance_id);
    assert!(old.generation < second.identity().unwrap().generation);
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 1);
    assert_eq!(view.coverage.loss.components_evicted.value.get(), 1);
    assert_eq!(
        first
            .record_without_projection_lazy(|| Observation::new(
                classification(FailureClass::Unknown),
                "stale",
                1
            ))
            .unwrap_err(),
        DiagnosticError::ComponentClosed
    );
    record(&second, 2);
    assert_eq!(
        code_total(
            row(&snapshot(&service, DiagnosticQuery::default()), "producer"),
            WRITE
        ),
        2
    );
}

#[test]
fn closed_component_reclamation_preserves_wrapped_history_order_and_reuses_detail() {
    let service = diagnostics(DiagnosticLimits {
        max_components: 2,
        max_events: 4,
        detail_pool_bytes: 512,
        detail_block_bytes: 64,
        ..limits()
    });
    let (owner, first) = service
        .register_owned_component(ComponentRegistration::new("first", &CODES))
        .unwrap();
    let second = register(&service, "second");
    let reserved = service.memory_usage().reserved_bytes;
    for count in 1..=6 {
        record(if count % 2 == 0 { &second } else { &first }, count);
    }
    assert_eq!(service.memory_usage().detail_bytes, 4 * 64);
    owner.close().unwrap();
    let third = register(&service, "third");
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(
        view.events
            .iter()
            .map(|event| event.count.get())
            .collect::<Vec<_>>(),
        [4, 6],
    );
    assert!(
        view.events
            .iter()
            .all(|event| event.component.key == "second")
    );
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 4);
    assert_eq!(service.memory_usage().detail_bytes, 2 * 64);

    record(&third, 7);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(
        view.events
            .iter()
            .map(|event| event.count.get())
            .collect::<Vec<_>>(),
        [4, 6, 7],
    );
    assert!(
        view.events
            .windows(2)
            .all(|pair| pair[0].sequence < pair[1].sequence)
    );
    assert!(
        view.events
            .iter()
            .all(|event| event.message() == Some("failure"))
    );
    assert_eq!(service.memory_usage().detail_bytes, 3 * 64);
    assert_eq!(service.memory_usage().reserved_bytes, reserved);
}

#[test]
fn final_recorder_drop_closes_without_a_reader_extending_registration_lifetime() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    let reader = component.reader();
    let cloned = component.clone();
    record(&component, 1);
    drop(component);
    assert!(
        reader
            .snapshot(DiagnosticQuery::default())
            .unwrap()
            .to_snapshot()
            .unwrap()
            .components[0]
            .closed_at
            .is_none()
    );
    drop(cloned);
    let retained = reader
        .snapshot(DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert!(retained.components[0].closed_at.is_some());
    assert_eq!(retained.events.len(), 1);
    assert_eq!(
        retained.components[0]
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        1
    );
}

#[test]
fn code_capacity_rejection_preserves_existing_component_accounting() {
    let service = diagnostics(DiagnosticLimits {
        max_counters: 1,
        ..limits()
    });
    let first = service
        .register_component(ComponentRegistration::new("first", &[WRITE]))
        .unwrap();
    assert_eq!(
        service
            .register_component(ComponentRegistration::new("second", &[WRITE]))
            .unwrap_err(),
        DiagnosticError::KeyCapacity
    );
    record(&first, 3);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.components.len(), 1);
    assert_eq!(code_total(&view.components[0], WRITE), 3);
    assert_eq!(view.coverage.loss.keys_rejected.value.get(), 1);
}

#[test]
fn refresh_preserves_lifetime_code_totals_window_and_history() {
    let service = diagnostics(DiagnosticLimits {
        max_components: 1,
        max_counters: CODES.len(),
        max_window_buckets: 3,
        ..limits()
    });
    let (owner, component) = service
        .register_owned_component(ComponentRegistration {
            key: "producer",
            builtin_codes: &CODES,
            ownership: ComponentOwnership {
                configuration_id: Some("old".into()),
                ..ComponentOwnership::default()
            },
            window: Some(WindowSettings {
                resolution_millis: 60_000,
                bucket_count: 3,
            }),
        })
        .unwrap();
    let registered = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(
        registered.coverage.availability,
        DiagnosticAvailability::NeverObserved
    );
    owner
        .refresh(ComponentOwnership {
            configuration_id: Some("old".into()),
            ..ComponentOwnership::default()
        })
        .unwrap()
        .unwrap();
    let empty_refresh = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(
        empty_refresh.coverage.availability,
        DiagnosticAvailability::NeverObserved
    );
    assert_eq!(empty_refresh.lifetime.totals, DiagnosticTotals::default());
    record(&component, 2);
    let before = snapshot(&service, DiagnosticQuery::default());
    let reserved = service.memory_usage().reserved_bytes;
    let generation = owner
        .refresh(ComponentOwnership {
            configuration_id: Some("new".into()),
            ..ComponentOwnership::default()
        })
        .unwrap()
        .unwrap();
    let refreshed = snapshot(&service, DiagnosticQuery::default());
    assert_refresh_preserves_lifetime_evidence(&before, &refreshed, generation);
    assert_eq!(service.memory_usage().reserved_bytes, reserved);
    assert_eq!(service.memory_usage().counters, CODES.len());
    assert_eq!(service.memory_usage().window_buckets, 3);

    record(&component, 3);
    let after = snapshot(&service, DiagnosticQuery::default());
    assert_new_refresh_counts_and_event_origins(&after);
}

fn assert_refresh_preserves_lifetime_evidence(
    before: &DiagnosticSnapshot,
    refreshed: &DiagnosticSnapshot,
    generation: DiagnosticNumber,
) {
    assert_eq!(refreshed.components.len(), 1);
    let current = &refreshed.components[0];
    assert_eq!(current.identity, before.components[0].identity);
    assert_eq!(
        current.lifetime.totals,
        before.components[0].lifetime.totals
    );
    assert_eq!(
        current.lifetime.last_failure_at,
        before.components[0].lifetime.last_failure_at
    );
    assert!(current.lifetime.elapsed_millis >= before.components[0].lifetime.elapsed_millis);
    assert_eq!(current.since_refresh.generation, generation);
    assert!(generation > before.components[0].since_refresh.generation);
    assert_eq!(
        current.since_refresh.ownership.configuration_id.as_deref(),
        Some("new")
    );
    assert_eq!(current.since_refresh.totals, DiagnosticTotals::default());
    assert_eq!(code_total(current, WRITE), 2);
    assert_eq!(refreshed.events, before.events);
    assert_eq!(refreshed.lifetime.totals, before.lifetime.totals);
    assert_eq!(
        current.windows[0]
            .buckets
            .iter()
            .map(|bucket| bucket.failures.value.get())
            .sum::<u64>(),
        2
    );
}

fn assert_new_refresh_counts_and_event_origins(after: &DiagnosticSnapshot) {
    let current = &after.components[0];
    assert_eq!(current.lifetime.totals.occurrences_total.value.get(), 5);
    assert_eq!(
        current.since_refresh.totals.occurrences_total.value.get(),
        3
    );
    assert_eq!(
        current
            .since_refresh
            .totals
            .recorded_events_total
            .value
            .get(),
        1
    );
    assert_eq!(code_total(current, WRITE), 5);
    assert_eq!(after.lifetime.totals.occurrences_total.value.get(), 5);
    assert_eq!(
        current.windows[0]
            .buckets
            .iter()
            .map(|bucket| bucket.failures.value.get())
            .sum::<u64>(),
        5
    );
    assert_eq!(
        after.events[0].origin.ownership.configuration_id.as_deref(),
        Some("old")
    );
    assert_eq!(
        after.events[1].origin.ownership.configuration_id.as_deref(),
        Some("new")
    );
}

#[test]
fn captured_origin_survives_many_refreshes_without_growing_backing_storage() {
    let service = diagnostics(DiagnosticLimits {
        max_components: 1,
        max_counters: CODES.len(),
        max_window_buckets: 3,
        ..limits()
    });
    let (owner, component) = service
        .register_owned_component(ComponentRegistration {
            key: "producer",
            builtin_codes: &CODES,
            ownership: ComponentOwnership {
                configuration_id: Some("old".into()),
                ..ComponentOwnership::default()
            },
            window: Some(WindowSettings {
                resolution_millis: 60_000,
                bucket_count: 3,
            }),
        })
        .unwrap();
    let operation = component.capture_origin().unwrap();
    record(&component, 7);
    let old_generation = snapshot(&service, DiagnosticQuery::default()).components[0]
        .since_refresh
        .generation;
    let projected = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&projected);
    let _projection = service
        .subscribe_projection(move |event| {
            captured.lock().unwrap().push(event.origin.clone());
        })
        .unwrap();
    // Construct caller-owned input strings before counting backend allocations.
    let replacements = (0..128)
        .map(|index| ComponentOwnership {
            configuration_id: Some(format!("reload.{index}")),
            ..ComponentOwnership::default()
        })
        .collect::<Vec<_>>();
    let reserved = service.memory_usage().reserved_bytes;
    let (_, allocations) = allocations_during(|| {
        for ownership in replacements {
            owner.refresh(ownership).unwrap().unwrap();
        }
    });
    assert_eq!(allocations, 0);
    assert_eq!(service.memory_usage().reserved_bytes, reserved);
    assert_eq!(service.memory_usage().components, 1);
    assert_eq!(service.memory_usage().counters, CODES.len());
    assert_eq!(service.memory_usage().window_buckets, 3);
    operation
        .record_lazy(|| Observation::new(classification(FailureClass::Unknown), "late", 3))
        .unwrap();
    record(&component, 1);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_latest_refresh_totals_include_late_observations(&view, old_generation);
    assert_captured_origin_evidence(&view, &projected, old_generation);

    assert_origin_filters_preserve_current_summaries(&service, &view, old_generation);
}

fn assert_latest_refresh_totals_include_late_observations(
    view: &DiagnosticSnapshot,
    old_generation: DiagnosticNumber,
) {
    assert_eq!(view.components.len(), 1);
    let current = &view.components[0];
    assert!(current.since_refresh.generation > old_generation);
    assert_eq!(
        current.since_refresh.ownership.configuration_id.as_deref(),
        Some("reload.127")
    );
    assert_eq!(current.lifetime.totals.occurrences_total.value.get(), 11);
    assert_eq!(
        current.since_refresh.totals.occurrences_total.value.get(),
        4
    );
    assert_eq!(
        current
            .since_refresh
            .totals
            .recorded_events_total
            .value
            .get(),
        2
    );
    assert_eq!(view.lifetime.totals.occurrences_total.value.get(), 11);
    assert_eq!(view.lifetime.totals.recorded_events_total.value.get(), 3);
    assert_eq!(code_total(current, WRITE), 11);
    assert_eq!(
        current.windows[0]
            .buckets
            .iter()
            .map(|bucket| bucket.failures.value.get())
            .sum::<u64>(),
        11
    );
}

fn assert_captured_origin_evidence(
    view: &DiagnosticSnapshot,
    projected: &Mutex<Vec<DiagnosticOrigin>>,
    old_generation: DiagnosticNumber,
) {
    assert_eq!(projected.lock().unwrap().len(), 1);
    assert_eq!(
        projected.lock().unwrap()[0].refresh_generation,
        old_generation
    );
    assert_eq!(
        projected.lock().unwrap()[0]
            .ownership
            .configuration_id
            .as_deref(),
        Some("old")
    );
    let late = view
        .events
        .iter()
        .find(|event| event.message() == Some("late"))
        .unwrap();
    assert_eq!(late.origin.refresh_generation, old_generation);
    assert_eq!(
        late.origin.ownership.configuration_id.as_deref(),
        Some("old")
    );
}

fn assert_origin_filters_preserve_current_summaries(
    service: &Diagnostics,
    view: &DiagnosticSnapshot,
    old_generation: DiagnosticNumber,
) {
    // An origin-generation filter selects evidence, never a historical summary.
    let origin_filtered = snapshot(
        service,
        DiagnosticQuery {
            origin_generation: Some(old_generation),
            ..DiagnosticQuery::default()
        },
    );
    assert_eq!(origin_filtered.components.len(), 1);
    assert_eq!(
        origin_filtered.components[0]
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        11
    );
    assert_eq!(
        origin_filtered.components[0]
            .since_refresh
            .totals
            .occurrences_total
            .value
            .get(),
        4
    );
    assert_eq!(origin_filtered.events.len(), 2);
    assert!(
        origin_filtered
            .events
            .iter()
            .all(|event| event.origin.refresh_generation == old_generation)
    );
    assert_eq!(origin_filtered.lifetime.totals, view.lifetime.totals);
    // Original-configuration evidence remains selectable after current ownership changes.
    let old_owner = snapshot(
        service,
        DiagnosticQuery {
            ownership: ComponentOwnership {
                configuration_id: Some("old".into()),
                ..ComponentOwnership::default()
            },
            ..DiagnosticQuery::default()
        },
    );
    assert!(old_owner.components.is_empty());
    assert_eq!(old_owner.events.len(), 2);
    assert_eq!(old_owner.lifetime.totals, view.lifetime.totals);
    let current_owner = snapshot(
        service,
        DiagnosticQuery {
            ownership: ComponentOwnership {
                configuration_id: Some("reload.127".into()),
                ..ComponentOwnership::default()
            },
            ..DiagnosticQuery::default()
        },
    );
    assert_eq!(current_owner.components.len(), 1);
    assert_eq!(
        current_owner.components[0]
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        11
    );
    assert_eq!(
        current_owner.components[0]
            .since_refresh
            .totals
            .occurrences_total
            .value
            .get(),
        4
    );
    assert_eq!(current_owner.events.len(), 1);
    assert_eq!(current_owner.events[0].count.get(), 1);
}

#[test]
fn old_event_eviction_after_refresh_updates_original_component_and_current_activity() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 1,
        ..limits()
    });
    let (owner, original) = service
        .register_owned_component(ComponentRegistration::new("original", &CODES))
        .unwrap();
    let incoming = register(&service, "incoming");
    record(&original, 100);
    owner
        .refresh(ComponentOwnership::default())
        .unwrap()
        .unwrap();
    record(&incoming, 3);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.components.len(), 2);
    let original = row(&view, "original");
    assert_eq!(original.lifetime.totals.occurrences_total.value.get(), 100);
    assert_eq!(
        original.lifetime.totals.history_evictions_total.value.get(),
        1
    );
    assert_eq!(
        original
            .since_refresh
            .totals
            .recorded_events_total
            .value
            .get(),
        0
    );
    assert_eq!(
        original.since_refresh.totals.occurrences_total.value.get(),
        0
    );
    assert_eq!(
        original
            .since_refresh
            .totals
            .history_evictions_total
            .value
            .get(),
        1
    );
    assert_eq!(
        row(&view, "incoming")
            .lifetime
            .totals
            .history_evictions_total
            .value
            .get(),
        0
    );
    assert_eq!(view.lifetime.totals.recorded_events_total.value.get(), 2);
    assert_eq!(view.lifetime.totals.occurrences_total.value.get(), 103);
    assert_eq!(view.lifetime.totals.history_evictions_total.value.get(), 1);
    assert_eq!(view.events.len(), 1);
    assert_eq!(view.events[0].component.key, "incoming");
}

#[test]
fn context_lifetime_totals_survive_closed_component_reclamation_and_filters() {
    let service = diagnostics(DiagnosticLimits {
        max_components: 1,
        max_counters: CODES.len(),
        ..limits()
    });
    let (owner, first) = service
        .register_owned_component(ComponentRegistration::new("producer", &CODES))
        .unwrap();
    let old_identity = first.identity().unwrap().clone();
    record(&first, 100);
    owner.close().unwrap();
    let replacement = register(&service, "producer");
    let replaced = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(replaced.components.len(), 1);
    assert_eq!(
        replaced.coverage.availability,
        DiagnosticAvailability::Available
    );
    assert_ne!(replaced.components[0].identity, old_identity);
    assert_eq!(
        replaced.components[0].lifetime.totals,
        DiagnosticTotals::default()
    );
    assert_eq!(replaced.lifetime.totals.occurrences_total.value.get(), 100);
    assert_eq!(
        replaced.lifetime.totals.recorded_events_total.value.get(),
        1
    );
    assert_eq!(
        replaced.lifetime.totals.history_evictions_total.value.get(),
        1
    );
    assert!(replaced.lifetime.last_failure_at.is_some());
    assert_eq!(replaced.coverage.loss.components_evicted.value.get(), 1);
    assert!(replaced.events.is_empty());
    record(&replacement, 3);
    let current = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(
        current.components[0]
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        3
    );
    assert_eq!(current.lifetime.totals.occurrences_total.value.get(), 103);
    assert_eq!(current.lifetime.totals.failures_total.value.get(), 103);
    let absent = snapshot(
        &service,
        DiagnosticQuery {
            component_key: Some("not.present".into()),
            max_components: Some(1),
            max_events: Some(1),
            ..DiagnosticQuery::default()
        },
    );
    assert!(absent.components.is_empty());
    assert!(absent.events.is_empty());
    assert_eq!(
        absent.coverage.availability,
        DiagnosticAvailability::Available
    );
    assert_eq!(absent.lifetime.totals, current.lifetime.totals);
}

#[test]
fn outcomes_windows_and_failure_classes_have_independent_scopes() {
    let service = diagnostics(limits());
    let component = service
        .register_component(ComponentRegistration {
            key: "producer",
            builtin_codes: &CODES,
            ownership: ComponentOwnership::default(),
            window: Some(WindowSettings {
                resolution_millis: 60_000,
                bucket_count: 3,
            }),
        })
        .unwrap();
    record(&component, 2);
    let after_first = snapshot(&service, DiagnosticQuery::default()).events[0].sequence;
    component
        .record_without_projection_lazy(|| {
            Observation::new(classification(FailureClass::Timeout), "timeout", 3)
        })
        .unwrap();
    component
        .record_without_projection_lazy(|| {
            Observation::new(
                Classification::success(SuccessCode::SinkInitialized),
                "success",
                4,
            )
        })
        .unwrap();
    component
        .record_without_projection_lazy(|| {
            Observation::new(
                Classification::notice(NoticeCode::GuardrailRejected),
                "notice",
                1,
            )
            .with_severity(DiagnosticSeverity::Error)
        })
        .unwrap();
    let filtered = snapshot(
        &service,
        DiagnosticQuery {
            code: Some(WRITE.canonical_name().into()),
            failure_class: Some("timeout".into()),
            after_sequence: Some(after_first),
            ..DiagnosticQuery::default()
        },
    );
    assert_eq!(filtered.events.len(), 1);
    assert_eq!(
        filtered.events[0].classification.failure_class.as_deref(),
        Some("timeout")
    );
    let component = &filtered.components[0];
    assert_eq!(code_total(component, WRITE), 5);
    assert_eq!(
        component.lifetime.totals.recorded_events_total.value.get(),
        4
    );
    assert_eq!(component.lifetime.totals.occurrences_total.value.get(), 10);
    assert_eq!(component.lifetime.totals.failures_total.value.get(), 5);
    assert_eq!(component.lifetime.totals.successes_total.value.get(), 4);
    assert_eq!(component.lifetime.totals.unspecified_total.value.get(), 1);
    assert_eq!(
        filtered.coverage.summary_scope,
        DiagnosticSummaryScope::WholeComponentLifetime
    );
    assert_mixed_outcome_window_totals(&component.windows[0]);
    assert!(component.lifetime.last_failure_at.is_some());
    assert!(component.lifetime.since_last_failure_millis.is_some());
    let before = snapshot(&service, DiagnosticQuery::default());
    let after = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(before.sequence, after.sequence);
    assert_eq!(
        before.components[0].lifetime.totals.occurrences_total,
        after.components[0].lifetime.totals.occurrences_total
    );
}

fn assert_mixed_outcome_window_totals(window: &DiagnosticWindow) {
    assert_eq!(window.resolution_millis.get(), 60_000);
    assert_eq!(window.bucket_capacity, 3);
    for bucket in &window.buckets {
        assert_eq!(
            bucket.end_elapsed_millis.get() - bucket.start_elapsed_millis.get(),
            window.resolution_millis.get()
        );
    }

    assert_eq!(
        window
            .buckets
            .iter()
            .map(|bucket| bucket.failures.value.get())
            .sum::<u64>(),
        5
    );
    assert_eq!(
        window
            .buckets
            .iter()
            .map(|bucket| bucket.successes.value.get())
            .sum::<u64>(),
        4
    );
    assert_eq!(
        window
            .buckets
            .iter()
            .map(|bucket| bucket.unspecified.value.get())
            .sum::<u64>(),
        1
    );
}

#[test]
fn counter_saturation_is_a_lower_bound_and_success_does_not_clear_failure() {
    let service = diagnostics(limits());
    let (owner, component) = service
        .register_owned_component(ComponentRegistration::new("producer", &CODES))
        .unwrap();
    let first = record(&component, u64::MAX).code_total.unwrap();
    let second = record(&component, 1).code_total.unwrap();
    assert!(!first.saturated);
    assert!(second.saturated);
    component
        .record_without_projection_lazy(|| {
            Observation::new(
                Classification::success(SuccessCode::SinkInitialized),
                "success",
                1,
            )
        })
        .unwrap();
    let view = snapshot(&service, DiagnosticQuery::default());
    let summary = &view.components[0];
    assert_eq!(code_total(summary, WRITE), u64::MAX);
    assert_eq!(code_total(summary, SUCCESS), 1);
    assert!(summary.lifetime.totals.failures_total.saturated);
    assert!(summary.lifetime.totals.occurrences_total.saturated);
    assert_eq!(summary.lifetime.totals.recorded_events_total.value.get(), 3);
    assert!(summary.lifetime.last_failure_at.is_some());
    assert_eq!(
        serde_json::to_value(summary).unwrap()["counters"][0]["occurrences"]["value"],
        u64::MAX.to_string()
    );
    assert!(view.lifetime.totals.failures_total.saturated);
    assert!(view.lifetime.totals.occurrences_total.saturated);
    owner
        .refresh(ComponentOwnership::default())
        .unwrap()
        .unwrap();
    record(&component, 2);
    let refreshed = snapshot(&service, DiagnosticQuery::default());
    assert!(refreshed.lifetime.totals.failures_total.saturated);
    assert!(
        refreshed.components[0]
            .lifetime
            .totals
            .failures_total
            .saturated
    );
    assert_eq!(
        refreshed.components[0]
            .since_refresh
            .totals
            .failures_total
            .value
            .get(),
        2
    );
    assert!(
        !refreshed.components[0]
            .since_refresh
            .totals
            .failures_total
            .saturated
    );
}

#[test]
fn invalid_observations_do_not_update_reserved_summaries() {
    let service = diagnostics(limits());
    let component = service
        .register_component(ComponentRegistration::new("producer", &[WRITE]))
        .unwrap();
    assert_eq!(
        component
            .record_lazy(|| Observation::new(classification(FailureClass::Unknown), "zero", 0))
            .unwrap_err(),
        DiagnosticError::InvalidCount
    );
    assert_eq!(
        component
            .record_lazy(|| Observation::new(
                Classification::success(SuccessCode::SinkInitialized),
                "undeclared",
                1
            ))
            .unwrap_err(),
        DiagnosticError::InvalidCode
    );
    let view = snapshot(&service, DiagnosticQuery::default());
    assert!(view.events.is_empty());
    assert_eq!(
        view.components[0]
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        0
    );
    assert_eq!(view.coverage.loss.observations_rejected.value.get(), 2);
    assert_eq!(view.lifetime.totals, DiagnosticTotals::default());
}

#[test]
fn query_leases_enforce_concurrency_and_return_their_permits() {
    let service = diagnostics(DiagnosticLimits {
        max_queries: 2,
        ..limits()
    });
    let component = register(&service, "producer");
    record(&component, 1);
    let first = service.snapshot(DiagnosticQuery::default()).unwrap();
    let second = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(service.memory_usage().query_workspaces, 2);
    assert_eq!(
        service.snapshot(DiagnosticQuery::default()).unwrap_err(),
        DiagnosticError::QueryCapacity
    );
    drop(first);
    let replacement = service.snapshot(DiagnosticQuery::default()).unwrap();
    assert_eq!(
        replacement
            .to_snapshot()
            .unwrap()
            .coverage
            .loss
            .queries_rejected
            .value
            .get(),
        1
    );
    drop(second);
    drop(replacement);
    assert_eq!(service.memory_usage().query_workspaces, 0);
    assert_eq!(
        snapshot(&service, DiagnosticQuery::default()).components[0]
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        1
    );
}

#[test]
fn bounded_pages_continue_without_recounting_or_widening_scope() {
    let service = diagnostics(limits());
    let components = [
        register(&service, "a"),
        register(&service, "b"),
        register(&service, "c"),
    ];
    for component in &components {
        record(component, 1);
    }
    let base = DiagnosticQuery {
        max_components: Some(1),
        max_events: Some(1),
        max_response_bytes: Some(4096),
        ..DiagnosticQuery::default()
    };
    let mut query = base.clone();
    let mut registrations = BTreeSet::new();
    let mut events = BTreeSet::new();
    let mut first_cursor = None;
    for _ in 0..8 {
        let lease = service.snapshot(query.clone()).unwrap();
        assert!(lease.as_bytes().len() <= 4096);
        let view = lease.to_snapshot().unwrap();
        assert_bounded_page_totals_and_unique_rows(&view, &mut registrations, &mut events);
        if first_cursor.is_none() {
            first_cursor = view.continuation.clone();
        }
        query.continuation = view.continuation;
        if query.continuation.is_none() {
            break;
        }
    }
    assert!(query.continuation.is_none());
    assert_eq!(registrations.len(), 3);
    assert_eq!(events.len(), 3);
    let cursor = first_cursor.unwrap();
    let changed_scope = DiagnosticQuery {
        component_key: Some("a".into()),
        continuation: Some(cursor.clone()),
        ..base.clone()
    };
    assert_eq!(
        service.snapshot(changed_scope).unwrap_err(),
        DiagnosticError::InvalidCursor
    );
    record(&components[0], 1);
    assert_eq!(
        service
            .snapshot(DiagnosticQuery {
                continuation: Some(cursor),
                ..base
            })
            .unwrap_err(),
        DiagnosticError::InvalidCursor
    );
    assert_eq!(
        snapshot(&service, DiagnosticQuery::default())
            .components
            .iter()
            .map(|row| row.lifetime.totals.occurrences_total.value.get())
            .sum::<u64>(),
        4
    );
}

fn assert_bounded_page_totals_and_unique_rows(
    view: &DiagnosticSnapshot,
    registrations: &mut BTreeSet<Uuid>,
    events: &mut BTreeSet<DiagnosticNumber>,
) {
    assert_eq!(view.lifetime.totals.occurrences_total.value.get(), 3);
    assert_eq!(view.lifetime.totals.recorded_events_total.value.get(), 3);
    assert!(view.components.len() <= 1 && view.events.len() <= 1);
    for row in &view.components {
        assert!(registrations.insert(row.identity.instance_id));
    }
    for event in &view.events {
        assert!(events.insert(event.sequence));
    }
}

#[test]
fn encoded_byte_pagination_keeps_history_and_makes_forward_progress() {
    let service = diagnostics(DiagnosticLimits {
        max_detail_bytes: 1024,
        ..limits()
    });
    let component = register(&service, "producer");
    let detail = "x".repeat(1000);
    for _ in 0..6 {
        component
            .record_without_projection_lazy(|| {
                Observation::new(classification(FailureClass::Unknown), detail.as_str(), 1)
            })
            .unwrap();
    }
    let mut query = DiagnosticQuery {
        max_components: Some(0),
        max_events: Some(6),
        max_response_bytes: Some(4096),
        ..DiagnosticQuery::default()
    };
    let mut events = BTreeSet::new();
    for _ in 0..10 {
        let lease = service.snapshot(query.clone()).unwrap();
        assert!(lease.as_bytes().len() <= 4096);
        let view = lease.to_snapshot().unwrap();
        assert!(
            !view.events.is_empty(),
            "a continuation must advance the selected history"
        );
        for event in &view.events {
            assert!(events.insert(event.sequence));
        }
        query.continuation = view.continuation;
        if query.continuation.is_none() {
            break;
        }
    }
    assert!(query.continuation.is_none());
    assert_eq!(events.len(), 6);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.events.len(), 6);
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 0);
    assert_eq!(code_total(&view.components[0], WRITE), 6);
    assert_eq!(
        service
            .snapshot(DiagnosticQuery {
                max_response_bytes: Some(64),
                ..DiagnosticQuery::default()
            })
            .unwrap_err(),
        DiagnosticError::SnapshotCapacity
    );
    assert_eq!(service.memory_usage().events, 6);
}

#[test]
fn projection_detachment_and_sink_feedback_do_not_change_retention_or_double_emit() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    let calls = Arc::new(AtomicUsize::new(0));
    let callback = Arc::clone(&calls);
    let guard = service
        .subscribe_projection(move |_| {
            callback.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    record(&component, 1);
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "sink feedback bypasses every projection"
    );
    component
        .record_lazy(|| Observation::new(classification(FailureClass::Unknown), "projected", 1))
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    drop(guard);
    component
        .record_lazy(|| Observation::new(classification(FailureClass::Unknown), "detached", 1))
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.events.len(), 3);
    assert_eq!(
        view.components[0]
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        3
    );
}

#[test]
fn projections_can_query_and_record_reentrantly_without_recursion() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    let reader = service.reader();
    let nested = component.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback = Arc::clone(&calls);
    let _guard = service
        .subscribe_projection(move |_| {
            callback.fetch_add(1, Ordering::Relaxed);
            assert!(
                !reader
                    .snapshot(DiagnosticQuery::default())
                    .unwrap()
                    .to_snapshot()
                    .unwrap()
                    .events
                    .is_empty()
            );
            record(&nested, 1);
            nested
                .record_lazy(|| {
                    Observation::new(classification(FailureClass::Unknown), "recursive", 1)
                })
                .unwrap();
        })
        .unwrap();
    component
        .record_lazy(|| Observation::new(classification(FailureClass::Unknown), "initial", 1))
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.events.len(), 3);
    assert_eq!(view.coverage.loss.projections_suppressed.value.get(), 1);
    assert_eq!(view.coverage.loss.projections_panicked.value.get(), 0);
}

#[test]
fn projection_panics_and_delivery_errors_are_visible_after_acceptance() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    let _panic = service
        .subscribe_projection(|_| panic!("expected projection failure"))
        .unwrap();
    let _error = service
        .subscribe_fallible_projection(|_| Err(ProjectionError::PayloadTooLarge))
        .unwrap();
    let delivered = Arc::new(AtomicUsize::new(0));
    let healthy = Arc::clone(&delivered);
    let _healthy = service
        .subscribe_projection(move |_| {
            healthy.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    component
        .record_lazy(|| Observation::new(classification(FailureClass::Unknown), "accepted", 1))
        .unwrap();
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(delivered.load(Ordering::Relaxed), 1);
    assert_eq!(view.coverage.loss.projections_panicked.value.get(), 1);
    assert_eq!(
        view.coverage.loss.projection_deliveries_omitted.value.get(),
        1
    );
    assert_eq!(
        view.components[0]
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        1
    );
}

#[test]
fn publication_workspace_pressure_omits_delivery_without_losing_totals() {
    let service = diagnostics(DiagnosticLimits {
        max_projection_workspaces: 1,
        ..limits()
    });
    let component = register(&service, "producer");
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Mutex::new(release_rx);
    let _guard = service
        .subscribe_projection(move |_| {
            started_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
        })
        .unwrap();
    let writer = component.clone();
    let background = thread::spawn(move || {
        writer
            .record_lazy(|| Observation::new(classification(FailureClass::Unknown), "first", 1))
            .unwrap()
    });
    started_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    component
        .record_lazy(|| Observation::new(classification(FailureClass::Unknown), "second", 1))
        .unwrap();
    release_tx.send(()).unwrap();
    background.join().unwrap();
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(
        view.components[0]
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        2
    );
    assert_eq!(view.events.len(), 2);
    assert_eq!(
        view.coverage.loss.projection_deliveries_omitted.value.get(),
        1
    );
    assert_eq!(service.memory_usage().projection_workspaces, 0);
}

#[test]
fn custom_extension_classes_are_admitted_once_and_overflow_is_summary_only() {
    let service = diagnostics(DiagnosticLimits {
        max_extensions: 1,
        ..limits()
    });
    let component = register(&service, "producer");
    let registration = ExtensionRegistration {
        namespace: "acme",
        name: "request_failed",
        description: "A producer-owned failure",
        allowed_classes: &[FailureClass::Unknown],
        outcome: DiagnosticOutcome::Failure,
    };
    let extension = component
        .register_extension(registration.clone())
        .unwrap()
        .unwrap();
    assert_eq!(
        component.register_extension(registration).unwrap(),
        Some(extension)
    );
    let class = component
        .register_extension_class(extension, "backend_overloaded")
        .unwrap()
        .unwrap();
    assert_eq!(
        component
            .register_extension_class(extension, "backend_overloaded")
            .unwrap(),
        Some(class)
    );
    for name in ["custom_two", "custom_three", "custom_four"] {
        component.register_extension_class(extension, name).unwrap();
    }
    assert_eq!(
        component
            .register_extension_class(extension, "custom_five")
            .unwrap_err(),
        DiagnosticError::MetadataCapacity
    );
    assert_eq!(
        component
            .register_extension_class(extension, "timeout")
            .unwrap_err(),
        DiagnosticError::InvalidClassification
    );
    component
        .record_without_projection_lazy(|| {
            Observation::extension_with_class(extension, class, "custom", 2)
        })
        .unwrap();
    assert_eq!(
        component
            .register_extension(ExtensionRegistration {
                namespace: "acme",
                name: "another_failure",
                description: "another",
                allowed_classes: &[FailureClass::Unknown],
                outcome: DiagnosticOutcome::Failure
            })
            .unwrap_err(),
        DiagnosticError::KeyCapacity
    );
    let overflow = component
        .record_without_projection_lazy(|| {
            Observation::classification_overflow(DiagnosticOutcome::Failure, 3)
        })
        .unwrap();
    assert!(matches!(
        overflow,
        RecordOutcome::Recorded(RecordReceipt {
            code_total: None,
            evidence: EvidenceDisposition::Dropped,
            ..
        })
    ));
    let view = snapshot(
        &service,
        DiagnosticQuery {
            failure_class: Some("backend_overloaded".into()),
            ..DiagnosticQuery::default()
        },
    );
    assert_eq!(view.events.len(), 1);
    assert_eq!(
        view.events[0].classification.canonical_code,
        "acme.request_failed"
    );
    assert_eq!(
        view.events[0].classification.failure_class.as_deref(),
        Some("backend_overloaded")
    );
    assert_eq!(view.components[0].counters[0].occurrences.value.get(), 2);
    assert_eq!(
        view.components[0]
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        2
    );
    assert_eq!(
        view.components[0]
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        5
    );
    assert_eq!(view.coverage.loss.classifications_overflowed.value.get(), 3);
    assert_eq!(view.coverage.loss.events_omitted.value.get(), 1);
    assert_eq!(service.memory_usage().extensions, 1);
}

#[test]
fn extension_handles_cannot_cross_registration_or_reserved_namespace_boundaries() {
    let service = diagnostics(limits());
    let first = register(&service, "first");
    let second = register(&service, "second");
    let extension = first
        .register_extension(ExtensionRegistration {
            namespace: "acme",
            name: "failed",
            description: "safe",
            allowed_classes: &[FailureClass::Unknown],
            outcome: DiagnosticOutcome::Failure,
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        second
            .record_without_projection_lazy(|| Observation::extension(
                extension,
                Some(FailureClass::Unknown),
                DiagnosticOutcome::Failure,
                "wrong owner",
                1
            ))
            .unwrap_err(),
        DiagnosticError::InvalidCode
    );
    for namespace in [
        "runtime",
        "plugin",
        "upstream",
        "observability",
        "logging",
        "binding",
        "otel",
    ] {
        assert_eq!(
            first
                .register_extension(ExtensionRegistration {
                    namespace,
                    name: "future_failure",
                    description: "reserved",
                    allowed_classes: &[FailureClass::Unknown],
                    outcome: DiagnosticOutcome::Failure
                })
                .unwrap_err(),
            DiagnosticError::InvalidCode
        );
    }
}

#[test]
fn concurrent_recording_counts_once_under_shared_history_pressure() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 7,
        ..limits()
    });
    let component = register(&service, "producer");
    let reserved = service.memory_usage().reserved_bytes;
    let threads = (0..6)
        .map(|_| {
            let writer = component.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    record(&writer, 1);
                }
            })
        })
        .collect::<Vec<_>>();
    for writer in threads {
        writer.join().unwrap();
    }
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(
        view.components[0]
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        600
    );
    assert_eq!(
        view.components[0]
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        600
    );
    assert_eq!(
        view.components[0]
            .lifetime
            .totals
            .history_evictions_total
            .value
            .get(),
        593
    );
    assert_eq!(view.events.len(), 7);
    assert_eq!(service.memory_usage().reserved_bytes, reserved);
}

#[test]
fn enabled_recording_and_leased_queries_do_not_allocate_new_backing_storage() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 3,
        detail_pool_bytes: 128,
        detail_block_bytes: 64,
        ..limits()
    });
    let component = service
        .register_component(ComponentRegistration {
            ownership: ComponentOwnership {
                configuration_id: Some("configuration".into()),
                responsibility: Some("responsibility".into()),
                ..ComponentOwnership::default()
            },
            ..ComponentRegistration::new("producer", &CODES)
        })
        .unwrap();
    record(&component, 1);
    drop(service.snapshot(DiagnosticQuery::default()).unwrap());
    let reserved = service.memory_usage().reserved_bytes;
    let (_, recording_allocations) = allocations_during(|| {
        for _ in 0..1000 {
            record(&component, 1);
        }
    });
    assert_eq!(recording_allocations, 0);
    let (page, query_allocations) =
        allocations_during(|| service.snapshot(DiagnosticQuery::default()).unwrap());
    assert_eq!(query_allocations, 0);
    let view = page.to_snapshot().unwrap(); // decoding explicitly allocates caller-owned DTOs
    assert_eq!(
        view.components[0]
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        1001
    );
    assert_eq!(view.events.len(), 2);
    assert_eq!(service.memory_usage().reserved_bytes, reserved);
}

#[test]
fn impossible_startup_profiles_fail_before_backend_allocation() {
    let profile = DiagnosticLimits {
        max_events: usize::MAX,
        ..limits()
    };
    assert_eq!(
        Diagnostics::from_settings(
            source(),
            DiagnosticsSettings {
                enabled: true,
                limits: profile
            }
        )
        .unwrap_err(),
        DiagnosticError::InvalidLimits
    );
    let profile = DiagnosticLimits {
        max_owned_bytes: 1,
        ..limits()
    };
    assert_eq!(
        Diagnostics::from_settings(
            source(),
            DiagnosticsSettings {
                enabled: true,
                limits: profile
            }
        )
        .unwrap_err(),
        DiagnosticError::InvalidLimits
    );
}

#[test]
fn default_profile_reports_bounded_compiled_backing_capacity() {
    let profile = DiagnosticLimits::default();
    let service = diagnostics(profile.clone());
    let usage = service.memory_usage();
    eprintln!(
        "default diagnostics backing bytes: {}",
        usage.reserved_bytes
    );
    assert!(usage.reserved_bytes > 0);
    assert!(usage.reserved_bytes <= profile.max_owned_bytes);
    assert_eq!(usage.components, 0);
    assert_eq!(usage.events, 0);
    assert_eq!(usage.query_workspaces, 0);
}

#[test]
fn warmed_projection_delivery_reuses_its_payload_and_callback_storage() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 3,
        detail_pool_bytes: 128,
        detail_block_bytes: 64,
        ..limits()
    });
    let attributed = service
        .register_component(ComponentRegistration {
            ownership: ComponentOwnership {
                subscriber_instance_id: Some(Uuid::nil()),
                plugin_instance_id: Some(Uuid::nil()),
                configuration_id: Some("configuration".into()),
                responsibility: Some("responsibility".into()),
            },
            ..ComponentRegistration::new("attributed", &CODES)
        })
        .unwrap();
    let plain = register(&service, "plain");
    let delivered = Arc::new(AtomicUsize::new(0));
    let callback = Arc::clone(&delivered);
    let _guard = service
        .subscribe_projection(move |event| {
            assert_eq!(event.code(), WRITE.canonical_name());
            let owner = &event.origin.ownership;
            if event.component.key == "attributed" {
                assert_eq!(owner.subscriber_instance_id, Some(Uuid::nil()));
                assert_eq!(owner.plugin_instance_id, Some(Uuid::nil()));
                assert_eq!(owner.configuration_id.as_deref(), Some("configuration"));
                assert_eq!(owner.responsibility.as_deref(), Some("responsibility"));
            } else {
                assert!(owner.subscriber_instance_id.is_none());
                assert!(owner.plugin_instance_id.is_none());
                assert!(owner.configuration_id.is_none());
                assert!(owner.responsibility.is_none());
            }
            callback.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    for component in [&attributed, &plain] {
        component
            .record_lazy(|| Observation::new(classification(FailureClass::Unknown), "projected", 1))
            .unwrap();
    }
    let (_, allocations) = allocations_during(|| {
        for _ in 0..100 {
            for component in [&attributed, &plain] {
                component
                    .record_lazy(|| {
                        Observation::new(classification(FailureClass::Unknown), "projected", 1)
                    })
                    .unwrap();
            }
        }
    });
    assert_eq!(allocations, 0);
    assert_eq!(delivered.load(Ordering::Relaxed), 202);
    assert_eq!(service.memory_usage().projection_workspaces, 0);
}

#[test]
fn extension_and_class_handles_cannot_resolve_in_an_independent_context() {
    let left = diagnostics(limits());
    let right = diagnostics(limits());
    let left_component = register(&left, "producer");
    let right_component = register(&right, "producer");
    let extension = |namespace: &'static str| ExtensionRegistration {
        namespace,
        name: "failed",
        description: "safe",
        allowed_classes: &[FailureClass::Unknown],
        outcome: DiagnosticOutcome::Failure,
    };
    let left_extension = left_component
        .register_extension(extension("left"))
        .unwrap()
        .unwrap();
    let right_extension = right_component
        .register_extension(extension("right"))
        .unwrap()
        .unwrap();
    let left_class = left_component
        .register_extension_class(left_extension, "left_cause")
        .unwrap()
        .unwrap();
    right_component
        .register_extension_class(right_extension, "right_cause")
        .unwrap();

    let foreign_code = right_component.record_without_projection_lazy(|| {
        Observation::extension(
            left_extension,
            Some(FailureClass::Unknown),
            DiagnosticOutcome::Failure,
            "wrong source",
            1,
        )
    });
    assert!(matches!(
        foreign_code,
        Err(DiagnosticError::InvalidCode | DiagnosticError::InvalidClassification)
    ));
    let foreign_extension_and_class = right_component.record_without_projection_lazy(|| {
        Observation::extension_with_class(left_extension, left_class, "wrong source pair", 1)
    });
    assert!(matches!(
        foreign_extension_and_class,
        Err(DiagnosticError::InvalidCode | DiagnosticError::InvalidClassification)
    ));
    let foreign_class = right_component.record_without_projection_lazy(|| {
        Observation::extension_with_class(right_extension, left_class, "wrong source class", 1)
    });
    assert!(matches!(
        foreign_class,
        Err(DiagnosticError::InvalidCode | DiagnosticError::InvalidClassification)
    ));
    assert!(matches!(
        right_component.register_extension_class(left_extension, "wrong_source"),
        Err(DiagnosticError::InvalidCode | DiagnosticError::InvalidClassification)
    ));
    let view = snapshot(&right, DiagnosticQuery::default());
    assert!(view.events.is_empty());
    assert_eq!(
        view.components[0]
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        0
    );
}

#[test]
fn typed_plugin_context_survives_projection_and_shared_history_without_extra_counters() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 1,
        ..limits()
    });
    let code = BuiltinDiagnosticCode::Failure(FailureCode::Plugin(PluginFailureCode::Common(
        CommonPluginFailureCode::InvocationFailed,
    )));
    let recorder = service
        .register_component(ComponentRegistration::new("test.plugin_context", &[code]))
        .unwrap();
    let contexts = [
        PluginObservationContext {
            execution: Some(PluginExecution::NativeAbi),
            phase: Some(PluginPhase::Activation),
        },
        PluginObservationContext {
            execution: Some(PluginExecution::WorkerProtocol),
            phase: Some(PluginPhase::Runtime),
        },
    ];
    let projected = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&projected);
    let _guard = service
        .subscribe_projection(move |event| captured.lock().unwrap().push(event.context))
        .unwrap();
    for (context, count) in contexts.into_iter().zip([2, 3]) {
        recorder
            .record_lazy(|| {
                Observation::new(
                    Classification::failure(
                        FailureCode::Plugin(PluginFailureCode::Common(
                            CommonPluginFailureCode::InvocationFailed,
                        )),
                        FailureClass::ExecutionFailed,
                    )
                    .unwrap(),
                    "operation failed",
                    count,
                )
                .with_plugin_context(context)
            })
            .unwrap();
    }
    assert_eq!(projected.lock().unwrap().as_slice(), &contexts[..]);
    let page = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(page.events.len(), 1);
    assert_eq!(page.events[0].context, contexts[1]);
    assert_eq!(page.components.len(), 1);
    assert_eq!(page.components[0].counters.len(), 1);
    assert_eq!(page.components[0].counters[0].occurrences.value.get(), 5);
    assert_eq!(
        page.components[0]
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        2
    );
    assert_eq!(
        page.components[0]
            .lifetime
            .totals
            .history_evictions_total
            .value
            .get(),
        1
    );
    let encoded = serde_json::to_value(page).unwrap();
    assert_eq!(
        encoded["events"][0]["context"]["execution"],
        "worker_protocol"
    );
    assert_eq!(encoded["events"][0]["context"]["phase"], "runtime");
}

/// Local diagnostic-path measurements only; this is not a workload qualification.
/// Run explicitly so routine correctness suites do not introduce timing gates.
#[test]
#[ignore = "informational local allocation/throughput/contention probe"]
fn diagnostic_recording_profile() {
    use std::hint::black_box;
    use std::sync::Barrier;
    use std::time::Instant;

    eprintln!(
        "informational local probe; debug_assertions={}",
        cfg!(debug_assertions)
    );
    const WRITERS: usize = 4;
    const OPERATIONS: usize = 25_000;
    const SAMPLE_EVERY: usize = 50;
    const DETAIL: &str = "bounded producer detail for local recording and contention measurement";

    let disabled = Diagnostics::default();
    let inert = register(&disabled, "disabled");
    let begin = Instant::now();
    for _ in 0..WRITERS * OPERATIONS {
        black_box(
            inert
                .record_lazy(|| panic!("disabled factory ran"))
                .unwrap(),
        );
    }
    let disabled_elapsed = begin.elapsed();
    eprintln!(
        "disabled: {} calls, {:.1} ns/call (includes black_box and loop)",
        WRITERS * OPERATIONS,
        disabled_elapsed.as_nanos() as f64 / (WRITERS * OPERATIONS) as f64,
    );

    for contended in [false, true] {
        let service = diagnostics(DiagnosticLimits::default());
        let recorders = (0..WRITERS)
            .map(|index| register(&service, &format!("profile.{index}")))
            .collect::<Vec<_>>();
        for recorder in &recorders {
            for _ in 0..100 {
                recorder
                    .record_without_projection_lazy(|| {
                        Observation::new(classification(FailureClass::Unknown), DETAIL, 1)
                    })
                    .unwrap();
            }
        }
        drop(service.snapshot(DiagnosticQuery::default()).unwrap());
        let reserved = service.memory_usage().reserved_bytes;
        let barrier = Barrier::new(if contended { WRITERS + 2 } else { 2 });
        let begin = Instant::now();
        let (mut samples, mut reads) = thread::scope(|scope| {
            let writers = recorders
                .iter()
                .take(if contended { WRITERS } else { 1 })
                .map(|recorder| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        let mut samples = Vec::with_capacity(OPERATIONS / SAMPLE_EVERY);
                        barrier.wait();
                        for index in 0..OPERATIONS {
                            let started = (index % SAMPLE_EVERY == 0).then(Instant::now);
                            let outcome = recorder
                                .record_without_projection_lazy(|| {
                                    Observation::new(
                                        classification(FailureClass::Unknown),
                                        DETAIL,
                                        1,
                                    )
                                })
                                .unwrap();
                            black_box(outcome);
                            if let Some(started) = started {
                                samples.push(started.elapsed().as_nanos());
                            }
                        }
                        samples
                    })
                })
                .collect::<Vec<_>>();
            let reader = contended.then(|| {
                let service = &service;
                let barrier = &barrier;
                scope.spawn(move || {
                    let mut reads = Vec::with_capacity(64);
                    barrier.wait();
                    for _ in 0..64 {
                        let started = Instant::now();
                        let page = service.snapshot(DiagnosticQuery::default()).unwrap();
                        black_box(page.as_bytes());
                        drop(page);
                        reads.push(started.elapsed().as_nanos());
                    }
                    reads
                })
            });
            barrier.wait();
            let samples = writers
                .into_iter()
                .flat_map(|writer| writer.join().unwrap())
                .collect::<Vec<_>>();
            (
                samples,
                reader.map_or_else(Vec::new, |reader| reader.join().unwrap()),
            )
        });
        let elapsed = begin.elapsed();
        samples.sort_unstable();
        reads.sort_unstable();
        let percentile = |samples: &[u128], percentile: usize| {
            samples[(samples.len() - 1) * percentile / 100] as f64 / 1000.0
        };
        let calls = OPERATIONS * if contended { WRITERS } else { 1 };
        eprintln!(
            "enabled {}: {} calls, {:.0} calls/s; sampled record us p50={:.2} p95={:.2} p99={:.2}; reserved={} bytes",
            if contended {
                "4 writers + reader"
            } else {
                "1 writer"
            },
            calls,
            calls as f64 / elapsed.as_secs_f64(),
            percentile(&samples, 50),
            percentile(&samples, 95),
            percentile(&samples, 99),
            reserved,
        );
        if !reads.is_empty() {
            eprintln!(
                "bounded snapshot us p50={:.2} p95={:.2} p99={:.2}",
                percentile(&reads, 50),
                percentile(&reads, 95),
                percentile(&reads, 99),
            );
        }
        let view = snapshot(&service, DiagnosticQuery::default());
        assert_eq!(
            view.components
                .iter()
                .map(|row| row.lifetime.totals.recorded_events_total.value.get())
                .sum::<u64>(),
            (calls + WRITERS * 100) as u64,
        );
        assert_eq!(service.memory_usage().reserved_bytes, reserved);
    }
}
