// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral contracts for the runtime-controlled bounded diagnostic foundation.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::num::NonZeroU64;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::thread;

use nemo_relay::diagnostics::*;
use uuid::Uuid;

#[path = "support/diagnostic_allocations.rs"]
mod diagnostic_allocations;
use diagnostic_allocations::{AllocationStats, allocation_stats_during, allocations_during};

const WRITE_FAILURE: FailureCode = FailureCode::Logging(LoggingFailureCode::SinkWriteFailed);
const WRITE: BuiltinDiagnosticCode = BuiltinDiagnosticCode::Failure(WRITE_FAILURE);
const SUCCESS: BuiltinDiagnosticCode = BuiltinDiagnosticCode::Success(SuccessCode::SinkInitialized);
const NOTICE: BuiltinDiagnosticCode =
    BuiltinDiagnosticCode::Notice(NoticeCode::ConfigurationInherited);
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
            Observation::new(
                classification(FailureClass::Unknown),
                "failure",
                NonZeroU64::new(count).unwrap(),
            )
        })
        .unwrap()
    {
        RecordOutcome::Recorded(receipt) => receipt,
        RecordOutcome::Disabled => panic!("fixture must be enabled"),
    }
}

fn snapshot(service: &Diagnostics, query: DiagnosticQuery) -> DiagnosticSnapshot {
    service.snapshot(&query).unwrap().to_snapshot().unwrap()
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
            service.snapshot(&DiagnosticQuery::default()).unwrap_err(),
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
    let usage = serde_json::to_value(service.reader().memory_usage()).unwrap();
    assert!(
        usage
            .as_object()
            .unwrap()
            .values()
            .all(|value| value.as_str() == Some("0"))
    );
}

// Process-default selection is permanent. Run only the selected public-contract
// scenario in a child rather than resetting globals or relying on test ordering.
fn isolated_process_default_test(body: impl FnOnce()) {
    const CHILD: &str = "NEMO_RELAY_TEST_DIAGNOSTICS_INSTALL_CHILD";
    let current = thread::current();
    let name = current.name().expect("test thread should be named");
    if std::env::var_os(CHILD).as_deref() == Some(std::ffi::OsStr::new(name)) {
        body();
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env(CHILD, name)
        .output()
        .expect("isolated process-default test should start");
    assert!(
        output.status.success(),
        "isolated test {name} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn assert_process_default_installation(enabled: bool) {
    let identity = source();
    let settings = DiagnosticsSettings {
        enabled,
        limits: DiagnosticLimits {
            max_events: 7,
            ..limits()
        },
    };
    let installed =
        Diagnostics::install_process_default(identity.clone(), settings.clone()).unwrap();
    let reader = installed.reader();
    let expected_availability = if enabled {
        DiagnosticAvailability::Available
    } else {
        DiagnosticAvailability::Disabled
    };
    assert_eq!(
        (
            installed.source(),
            installed.limits(),
            reader.availability()
        ),
        (&identity, &settings.limits, expected_availability),
    );
    let reserved = installed.memory_usage().reserved_bytes;
    if enabled {
        assert!(reserved > 0 && reserved <= settings.limits.max_owned_bytes);
    } else {
        assert_eq!(reserved, 0);
    }
    for _ in 0..3 {
        let selected = Diagnostics::process_default();
        assert!(std::ptr::eq(selected, installed));
        assert_eq!(
            (
                selected.source(),
                selected.limits(),
                selected.memory_usage().reserved_bytes,
            ),
            (&identity, &settings.limits, reserved),
        );
    }
    for replacement_enabled in [enabled, !enabled] {
        let replacement = DiagnosticsSettings {
            enabled: replacement_enabled,
            limits: DiagnosticLimits {
                max_events: 11,
                ..limits()
            },
        };
        assert_eq!(
            Diagnostics::install_process_default(source(), replacement).unwrap_err(),
            DiagnosticError::AlreadyInitialized,
        );
        let selected = Diagnostics::process_default();
        assert!(std::ptr::eq(selected, installed));
        assert_eq!(
            (
                selected.source(),
                selected.limits(),
                selected.memory_usage().reserved_bytes,
            ),
            (&identity, &settings.limits, reserved),
        );
        assert_eq!(reader.availability(), expected_availability);
        assert_eq!(selected.reader().availability(), expected_availability);
    }
}

#[test]
fn enabled_process_default_installation_preserves_configuration_and_rejects_replacement() {
    isolated_process_default_test(|| assert_process_default_installation(true));
}

#[test]
fn disabled_process_default_installation_preserves_configuration_and_rejects_replacement() {
    isolated_process_default_test(|| assert_process_default_installation(false));
}

#[test]
fn an_early_process_default_cannot_be_silently_upgraded() {
    isolated_process_default_test(|| {
        let early = Diagnostics::process_default();
        assert_eq!(
            early.reader().availability(),
            DiagnosticAvailability::Disabled
        );
        let identity = early.source().instance_id;
        assert_eq!(
            Diagnostics::install_process_default(
                source(),
                DiagnosticsSettings {
                    enabled: true,
                    ..Default::default()
                },
            )
            .unwrap_err(),
            DiagnosticError::AlreadyInitialized,
        );
        assert!(std::ptr::eq(Diagnostics::process_default(), early));
        assert_eq!(
            Diagnostics::process_default().source().instance_id,
            identity
        );
        assert_eq!(
            early.reader().availability(),
            DiagnosticAvailability::Disabled
        );
        assert_eq!(early.memory_usage().reserved_bytes, 0);
    });
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
                Observation::new(
                    classification(FailureClass::Unknown),
                    "abcdefgh",
                    NonZeroU64::MIN,
                )
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
                NonZeroU64::new(2).unwrap(),
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
fn impossible_correlation_preserves_history_with_pending_ascii_and_utf8_detail() {
    for (pool_bytes, detail, session) in [
        (12, "abcdefgh", "ijklmnop"),
        // Four three-byte scalars occupy four blocks, not three.
        (20, "€€€€", "abcdefgh"),
        // The incoming correlation also needs its exact UTF-8 block geometry.
        (20, "abcdefgh", "€€€€"),
    ] {
        let service = diagnostics(DiagnosticLimits {
            detail_pool_bytes: pool_bytes,
            detail_block_bytes: 4,
            ..limits()
        });
        let component = register(&service, "producer");
        component
            .record_without_projection_lazy(|| {
                Observation::new(
                    classification(FailureClass::Unknown),
                    "old!",
                    NonZeroU64::new(2).unwrap(),
                )
            })
            .unwrap();
        let receipt = component
            .record_without_projection_lazy(|| {
                Observation::new(
                    classification(FailureClass::Unknown),
                    detail,
                    NonZeroU64::new(3).unwrap(),
                )
                .with_correlation(DiagnosticCorrelation {
                    session_id: Some(session.into()),
                    request_id: None,
                })
            })
            .unwrap();
        assert!(matches!(
            receipt,
            RecordOutcome::Recorded(RecordReceipt {
                evidence: EvidenceDisposition::Truncated,
                ..
            })
        ));
        let view = snapshot(&service, DiagnosticQuery::default());
        assert_eq!(view.events.len(), 2);
        assert_eq!(view.events[0].message(), Some("old!"));
        assert_eq!(view.events[1].message(), Some(detail));
        assert!(view.events[1].correlation.session_id.is_none());
        assert_eq!(view.coverage.loss.events_evicted.value.get(), 0);
        assert_eq!(view.coverage.loss.correlations_omitted.value.get(), 1);
        assert_eq!(code_total(row(&view, "producer"), WRITE), 5);
    }
}

#[test]
fn impossible_request_preserves_history_and_pending_detail_and_session() {
    let service = diagnostics(DiagnosticLimits {
        detail_pool_bytes: 16,
        detail_block_bytes: 4,
        ..limits()
    });
    let component = register(&service, "producer");
    component
        .record_without_projection_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "old!",
                NonZeroU64::new(2).unwrap(),
            )
        })
        .unwrap();
    component
        .record_without_projection_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "abcd",
                NonZeroU64::new(3).unwrap(),
            )
            .with_correlation(DiagnosticCorrelation {
                session_id: Some("efghijkl".into()),
                request_id: Some("mnopqrst".into()),
            })
        })
        .unwrap();
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.events.len(), 2);
    assert_eq!(view.events[0].message(), Some("old!"));
    assert_eq!(view.events[1].message(), Some("abcd"));
    assert_eq!(
        view.events[1].correlation.session_id.as_deref(),
        Some("efghijkl")
    );
    assert!(view.events[1].correlation.request_id.is_none());
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 0);
    assert_eq!(view.coverage.loss.correlations_omitted.value.get(), 1);
    assert_eq!(code_total(row(&view, "producer"), WRITE), 5);
    assert_eq!(service.memory_usage().detail_bytes, 16);
}

#[test]
fn correlation_pressure_still_reclaims_history_when_pending_detail_leaves_capacity() {
    let service = diagnostics(DiagnosticLimits {
        detail_pool_bytes: 12,
        detail_block_bytes: 4,
        ..limits()
    });
    let component = register(&service, "producer");
    component
        .record_without_projection_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "old!",
                NonZeroU64::new(2).unwrap(),
            )
        })
        .unwrap();
    component
        .record_without_projection_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "abcd",
                NonZeroU64::new(3).unwrap(),
            )
            .with_correlation(DiagnosticCorrelation {
                session_id: Some("efghijkl".into()),
                request_id: None,
            })
        })
        .unwrap();
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.events.len(), 1);
    assert_eq!(view.events[0].message(), Some("abcd"));
    assert_eq!(
        view.events[0].correlation.session_id.as_deref(),
        Some("efghijkl")
    );
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 1);
    assert_eq!(view.coverage.loss.correlations_omitted.value.get(), 0);
    assert_eq!(code_total(row(&view, "producer"), WRITE), 5);
    assert_eq!(service.memory_usage().detail_bytes, 12);
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
        .record_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "failure",
                NonZeroU64::new(2).unwrap(),
            )
        })
        .unwrap();
    let view = snapshot(&service, DiagnosticQuery::default());
    assert!(view.events.is_empty());
    assert_eq!(view.coverage.loss.events_omitted.value.get(), 1);
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 0);
    assert_eq!(code_total(row(&view, "producer"), WRITE), 2);
    assert_eq!(observed.load(Ordering::Relaxed), 1);
    let usage = service.memory_usage();
    assert!(usage.event_slot_bytes > 0);
    assert_eq!(usage.event_capacity, 0);
    assert_eq!(usage.history_occupied_bytes, 0);
    assert_eq!(usage.detail_reserved_bytes, 0);
    assert_eq!(usage.detail_capacity_bytes, 0);
    assert_eq!(usage.detail_metadata_bytes, 0);
    assert_eq!(usage.detail_bytes, 0);
    assert_eq!(usage.detail_logical_bytes, 0);
    assert_eq!(usage.detail_slack_bytes, 0);
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
                Observation::new(
                    classification(FailureClass::Unknown),
                    "é🙂abcdef",
                    NonZeroU64::MIN,
                )
                .with_correlation(DiagnosticCorrelation {
                    session_id: Some("session".into()),
                    request_id: Some("🙂request".into()),
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
fn shortened_correlation_reports_truncated_without_detail_loss() {
    let service = diagnostics(DiagnosticLimits {
        max_correlation_bytes: 4,
        ..limits()
    });
    let component = register(&service, "producer");
    for (session_id, request_id, expected) in [
        (Some("session"), None, EvidenceDisposition::Truncated),
        (None, Some("request"), EvidenceDisposition::Truncated),
        (None, Some("🙂request"), EvidenceDisposition::Truncated),
        (Some("sess"), Some("🙂"), EvidenceDisposition::Retained),
    ] {
        let outcome = component
            .record_without_projection_lazy(|| {
                Observation::new(classification(FailureClass::Unknown), "ok", NonZeroU64::MIN)
                    .with_correlation(DiagnosticCorrelation {
                        session_id: session_id.map(Into::into),
                        request_id: request_id.map(Into::into),
                    })
            })
            .unwrap();
        let RecordOutcome::Recorded(receipt) = outcome else {
            panic!("fixture must be enabled");
        };
        assert_eq!(receipt.evidence, expected);
    }
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.events.len(), 4);
    assert_eq!(
        view.events[0].correlation.session_id.as_deref(),
        Some("sess")
    );
    assert_eq!(
        view.events[1].correlation.request_id.as_deref(),
        Some("requ")
    );
    assert_eq!(view.events[2].correlation.request_id.as_deref(), Some("🙂"));
    assert!(view.events.iter().all(|event| {
        event
            .detail
            .as_ref()
            .is_some_and(|detail| !detail.truncated && detail.message == "ok")
    }));
    assert_eq!(view.coverage.loss.messages_truncated.value.get(), 0);
    assert_eq!(view.coverage.loss.correlations_truncated.value.get(), 3);
    assert_eq!(view.coverage.loss.correlations_omitted.value.get(), 0);
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
                NonZeroU64::MIN
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
    assert_eq!(
        service.memory_usage().detail_logical_bytes,
        4 * "failure".len()
    );
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
    let reclaimed = service.memory_usage();
    assert_eq!(reclaimed.detail_bytes, 2 * 64);
    assert_eq!(reclaimed.detail_logical_bytes, 2 * "failure".len());
    assert_eq!(reclaimed.detail_slack_bytes, 2 * (64 - "failure".len()));
    assert_eq!(
        reclaimed.history_occupied_bytes,
        2 * reclaimed.event_slot_bytes
    );
    assert_eq!(reclaimed.reserved_bytes, reserved);

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
            .snapshot(&DiagnosticQuery::default())
            .unwrap()
            .to_snapshot()
            .unwrap()
            .components[0]
            .closed_at
            .is_none()
    );
    drop(cloned);
    let retained = reader
        .snapshot(&DiagnosticQuery::default())
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
fn rejected_refresh_metadata_is_counted_without_changing_component_state() {
    let service = diagnostics(DiagnosticLimits {
        max_identity_bytes: 128,
        max_metadata_bytes: 64,
        ..limits()
    });
    let (owner, component) = service
        .register_owned_component(ComponentRegistration {
            ownership: ComponentOwnership {
                configuration_id: Some("old".into()),
                ..Default::default()
            },
            ..ComponentRegistration::new("producer", &CODES)
        })
        .unwrap();
    record(&component, 3);
    let before = snapshot(&service, DiagnosticQuery::default());
    let memory_before = serde_json::to_value(service.memory_usage()).unwrap();
    let rejected_ownerships = [
        ComponentOwnership {
            configuration_id: Some("x".repeat(129)),
            ..Default::default()
        },
        ComponentOwnership {
            responsibility: Some("x".repeat(129)),
            ..Default::default()
        },
        // Fits the per-field limit, but exceeds the shared metadata budget.
        ComponentOwnership {
            configuration_id: Some("x".repeat(64)),
            ..Default::default()
        },
    ];
    for (index, ownership) in rejected_ownerships.into_iter().enumerate() {
        assert_eq!(
            owner.refresh(ownership),
            Err(DiagnosticError::MetadataCapacity)
        );
        let after = snapshot(&service, DiagnosticQuery::default());
        assert_eq!(
            after.coverage.loss.metadata_rejected.value.get(),
            index as u64 + 1
        );
        assert_eq!(after.sequence, before.sequence);
        assert_eq!(after.lifetime.totals, before.lifetime.totals);
        assert_eq!(after.events, before.events);
        assert_eq!(after.components.len(), 1);
        let component_before = &before.components[0];
        let component_after = &after.components[0];
        assert_eq!(component_after.identity, component_before.identity);
        assert_eq!(component_after.counters, component_before.counters);
        assert_eq!(
            component_after.lifetime.totals,
            component_before.lifetime.totals
        );
        let mut refresh_after = component_after.since_refresh.clone();
        // Capture time advances even when the refresh itself is rejected.
        refresh_after.elapsed_millis = component_before.since_refresh.elapsed_millis;
        assert_eq!(refresh_after, component_before.since_refresh);
        assert_eq!(
            serde_json::to_value(service.memory_usage()).unwrap(),
            memory_before
        );
    }
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
            captured.lock().unwrap().push(event.count);
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
        .record_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "late",
                NonZeroU64::new(3).unwrap(),
            )
        })
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
    projected: &Mutex<Vec<DiagnosticNumber>>,
    old_generation: DiagnosticNumber,
) {
    assert_eq!(projected.lock().unwrap().as_slice(), &[3.into()]);
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
            Observation::new(
                classification(FailureClass::Timeout),
                "timeout",
                NonZeroU64::new(3).unwrap(),
            )
        })
        .unwrap();
    component
        .record_without_projection_lazy(|| {
            Observation::new(
                Classification::success(SuccessCode::SinkInitialized),
                "success",
                NonZeroU64::new(4).unwrap(),
            )
        })
        .unwrap();
    component
        .record_without_projection_lazy(|| {
            Observation::new(
                Classification::notice(NoticeCode::ConfigurationInherited),
                "notice",
                NonZeroU64::MIN,
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
        filtered.events[0].classification.failure_class(),
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
                NonZeroU64::MIN,
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
            .record_lazy(|| Observation::new(
                Classification::success(SuccessCode::SinkInitialized),
                "undeclared",
                NonZeroU64::MIN
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
    assert_eq!(view.coverage.loss.observations_rejected.value.get(), 1);
    assert_eq!(view.lifetime.totals, DiagnosticTotals::default());
}

#[test]
fn decoded_payloads_are_validated_before_recording_changes_accounting() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    let delivered = Arc::new(AtomicUsize::new(0));
    let captured = Arc::clone(&delivered);
    let _guard = service
        .subscribe_projection(move |_| {
            captured.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    let baseline = snapshot(&service, DiagnosticQuery::default());

    for (code, class, expected) in [
        (
            "plugin.future_failure",
            Some("unknown"),
            DiagnosticError::InvalidCode,
        ),
        (
            "logging.sink.write_failed",
            Some("future_class"),
            DiagnosticError::InvalidClassification,
        ),
        (
            "logging.sink.write_failed",
            Some("authentication"),
            DiagnosticError::InvalidClassification,
        ),
        (
            "logging.sink.write_failed",
            None,
            DiagnosticError::InvalidClassification,
        ),
        (
            "logging.sink.initialized",
            Some("unknown"),
            DiagnosticError::InvalidClassification,
        ),
    ] {
        let classification: Classification = serde_json::from_value(serde_json::json!({
            "canonical_code": code,
            "failure_class": class,
        }))
        .unwrap();
        assert_eq!(
            component
                .record_lazy(|| Observation::new(
                    classification,
                    "invalid",
                    NonZeroU64::new(3).unwrap()
                ))
                .unwrap_err(),
            expected,
        );
    }
    for (execution, phase) in [("wasm_v2", "activation"), ("native_abi", "quiescing")] {
        let context: PluginObservationContext = serde_json::from_value(serde_json::json!({
            "execution": execution,
            "phase": phase,
        }))
        .unwrap();
        assert_eq!(
            component
                .record_lazy(|| {
                    Observation::new(
                        classification(FailureClass::Unknown),
                        "invalid",
                        NonZeroU64::new(3).unwrap(),
                    )
                    .with_plugin_context(context)
                })
                .unwrap_err(),
            DiagnosticError::InvalidContext,
        );
    }

    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.sequence, baseline.sequence);
    assert_eq!(view.lifetime.totals, baseline.lifetime.totals);
    assert_eq!(
        view.lifetime.last_failure_at,
        baseline.lifetime.last_failure_at
    );
    assert_eq!(view.components[0].counters, baseline.components[0].counters);
    assert_eq!(
        view.components[0].lifetime.totals,
        DiagnosticTotals::default()
    );
    assert!(view.events.is_empty());
    assert_eq!(view.coverage.loss.observations_rejected.value.get(), 7);
    assert_eq!(delivered.load(Ordering::Relaxed), 0);
}

#[test]
fn oversized_unrecognized_payload_names_are_rejected_without_allocating() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    let baseline = snapshot(&service, DiagnosticQuery::default());
    let reserved = service.memory_usage().reserved_bytes;
    let oversized = "x".repeat(65_536);
    // Caller-owned values are prepared before measuring admission. Rejection
    // must not format these names into newly allocated diagnostic errors.
    let inputs = [
        (
            Observation::new(
                Classification::from_opaque_names(
                    Cow::Owned(oversized.clone()),
                    Some("unknown".into()),
                ),
                "unknown code",
                NonZeroU64::MIN,
            ),
            DiagnosticError::InvalidCode,
        ),
        (
            Observation::new(
                Classification::from_opaque_names(
                    WRITE.canonical_name().into(),
                    Some(Cow::Owned(oversized.clone())),
                ),
                "unknown class",
                NonZeroU64::MIN,
            ),
            DiagnosticError::InvalidClassification,
        ),
        (
            Observation::new(
                classification(FailureClass::Unknown),
                "unknown execution",
                NonZeroU64::MIN,
            )
            .with_plugin_context(PluginObservationContext {
                execution: Some(PluginExecutionName::from_opaque_name(Cow::Owned(
                    oversized.clone(),
                ))),
                phase: Some(PluginPhase::Activation.into()),
            }),
            DiagnosticError::InvalidContext,
        ),
        (
            Observation::new(
                classification(FailureClass::Unknown),
                "unknown phase",
                NonZeroU64::MIN,
            )
            .with_plugin_context(PluginObservationContext {
                execution: Some(PluginExecution::NativeAbi.into()),
                phase: Some(PluginPhaseName::from_opaque_name(Cow::Owned(oversized))),
            }),
            DiagnosticError::InvalidContext,
        ),
    ];
    let mut inputs = inputs.into_iter();
    let (_, allocations) = allocations_during(|| {
        for (observation, expected) in &mut inputs {
            assert_eq!(component.record_lazy(|| observation).unwrap_err(), expected);
        }
    });
    assert_eq!(allocations, 0);
    assert_eq!(service.memory_usage().reserved_bytes, reserved);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.sequence, baseline.sequence);
    assert_eq!(view.lifetime.totals, baseline.lifetime.totals);
    assert_eq!(
        view.lifetime.last_failure_at,
        baseline.lifetime.last_failure_at
    );
    assert_eq!(view.components[0].counters, baseline.components[0].counters);
    assert_eq!(view.coverage.loss.observations_rejected.value.get(), 4);
    assert!(view.events.is_empty());
}

#[test]
fn decoded_extension_names_cannot_bypass_registered_handles() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    let handle = component
        .register_extension(ExtensionRegistration {
            namespace: "acme",
            name: "request_failed",
            description: "A producer-owned failure",
            allowed_classes: &[FailureClass::Unknown],
            outcome: DiagnosticOutcome::Failure,
        })
        .unwrap()
        .unwrap();
    let baseline = snapshot(&service, DiagnosticQuery::default());
    let classification: Classification = serde_json::from_value(serde_json::json!({
        "canonical_code": "acme.request_failed",
        "failure_class": "unknown",
    }))
    .unwrap();
    assert_eq!(
        component
            .record_lazy(|| Observation::new(
                classification,
                "unadmitted input",
                NonZeroU64::new(3).unwrap()
            ))
            .unwrap_err(),
        DiagnosticError::InvalidCode,
    );
    let rejected = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(rejected.sequence, baseline.sequence);
    assert_eq!(rejected.lifetime.totals, baseline.lifetime.totals);
    assert_eq!(
        rejected.lifetime.last_failure_at,
        baseline.lifetime.last_failure_at
    );
    assert_eq!(
        rejected.components[0].counters,
        baseline.components[0].counters
    );
    assert!(rejected.events.is_empty());
    component
        .record_lazy(|| {
            Observation::extension(
                handle,
                Some(FailureClass::Unknown),
                DiagnosticOutcome::Failure,
                "admitted",
                NonZeroU64::new(3).unwrap(),
            )
        })
        .unwrap();
    let accepted = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(accepted.events[0].code(), "acme.request_failed");
    assert_eq!(accepted.lifetime.totals.occurrences_total.value.get(), 3);
}

#[test]
fn query_leases_enforce_concurrency_and_return_their_permits() {
    let service = diagnostics(DiagnosticLimits {
        max_queries: 2,
        ..limits()
    });
    let component = register(&service, "producer");
    record(&component, 1);
    let first = service.snapshot(&DiagnosticQuery::default()).unwrap();
    let second = service.snapshot(&DiagnosticQuery::default()).unwrap();
    assert_eq!(service.memory_usage().query_workspaces, 2);
    assert_eq!(
        service.snapshot(&DiagnosticQuery::default()).unwrap_err(),
        DiagnosticError::QueryCapacity
    );
    drop(first);
    let replacement = service.snapshot(&DiagnosticQuery::default()).unwrap();
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
        let lease = service.snapshot(&query).unwrap();
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
        service.snapshot(&changed_scope).unwrap_err(),
        DiagnosticError::InvalidCursor
    );
    record(&components[0], 1);
    assert_eq!(
        service
            .snapshot(&DiagnosticQuery {
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
                Observation::new(
                    classification(FailureClass::Unknown),
                    detail.as_str(),
                    NonZeroU64::MIN,
                )
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
        let lease = service.snapshot(&query).unwrap();
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
            .snapshot(&DiagnosticQuery {
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
        .record_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "projected",
                NonZeroU64::MIN,
            )
        })
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    drop(guard);
    component
        .record_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "detached",
                NonZeroU64::MIN,
            )
        })
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
                    .snapshot(&DiagnosticQuery::default())
                    .unwrap()
                    .to_snapshot()
                    .unwrap()
                    .events
                    .is_empty()
            );
            record(&nested, 1);
            nested
                .record_lazy(|| {
                    Observation::new(
                        classification(FailureClass::Unknown),
                        "recursive",
                        NonZeroU64::MIN,
                    )
                })
                .unwrap();
        })
        .unwrap();
    component
        .record_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "initial",
                NonZeroU64::MIN,
            )
        })
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
        .record_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "accepted",
                NonZeroU64::MIN,
            )
        })
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
            .record_lazy(|| {
                Observation::new(
                    classification(FailureClass::Unknown),
                    "first",
                    NonZeroU64::MIN,
                )
            })
            .unwrap()
    });
    started_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    component
        .record_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "second",
                NonZeroU64::MIN,
            )
        })
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
fn extension_severity_defaults_follow_outcome_and_allow_explicit_override() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    for (name, outcome) in [
        ("failed", DiagnosticOutcome::Failure),
        ("succeeded", DiagnosticOutcome::Success),
        ("observed", DiagnosticOutcome::Unknown),
    ] {
        let class = (outcome == DiagnosticOutcome::Failure).then_some(FailureClass::Unknown);
        let extension = component
            .register_extension(ExtensionRegistration {
                namespace: "acme",
                name,
                description: "A producer observation",
                allowed_classes: if class.is_some() {
                    &[FailureClass::Unknown]
                } else {
                    &[]
                },
                outcome,
            })
            .unwrap()
            .unwrap();
        component
            .record_without_projection_lazy(|| {
                Observation::extension(
                    extension,
                    class,
                    outcome,
                    "default severity",
                    NonZeroU64::MIN,
                )
            })
            .unwrap();
        component
            .record_without_projection_lazy(|| {
                Observation::extension(
                    extension,
                    class,
                    outcome,
                    "explicit severity",
                    NonZeroU64::MIN,
                )
                .with_severity(DiagnosticSeverity::Warning)
            })
            .unwrap();
    }
    let view = snapshot(&service, DiagnosticQuery::default());
    let severities: Vec<_> = view.events.iter().map(|event| event.severity).collect();
    assert_eq!(
        severities,
        [
            DiagnosticSeverity::Error,
            DiagnosticSeverity::Warning,
            DiagnosticSeverity::Info,
            DiagnosticSeverity::Warning,
            DiagnosticSeverity::Info,
            DiagnosticSeverity::Warning,
        ]
    );
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
            Observation::extension_with_class(
                extension,
                class,
                "custom",
                NonZeroU64::new(2).unwrap(),
            )
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
            Observation::classification_overflow(
                DiagnosticOutcome::Failure,
                NonZeroU64::new(3).unwrap(),
            )
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
    assert_eq!(view.events[0].classification.code(), "acme.request_failed");
    assert_eq!(
        view.events[0].classification.failure_class(),
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
                NonZeroU64::MIN
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
    drop(service.snapshot(&DiagnosticQuery::default()).unwrap());
    let reserved = service.memory_usage().reserved_bytes;
    let (_, recording_allocations) = allocations_during(|| {
        for _ in 0..1000 {
            record(&component, 1);
        }
    });
    assert_eq!(recording_allocations, 0);
    let (page, query_allocations) =
        allocations_during(|| service.snapshot(&DiagnosticQuery::default()).unwrap());
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
fn impossible_query_capacities_are_invalid_limits() {
    for (case, profile) in [
        (
            "enabled reads require component page capacity",
            DiagnosticLimits {
                max_query_components: 0,
                ..limits()
            },
        ),
        (
            "enabled reads require response envelope capacity",
            DiagnosticLimits {
                max_snapshot_bytes: 0,
                ..limits()
            },
        ),
        (
            "component rows overflow",
            DiagnosticLimits {
                max_query_components: usize::MAX,
                ..limits()
            },
        ),
        (
            "event rows overflow",
            DiagnosticLimits {
                max_query_events: usize::MAX,
                ..limits()
            },
        ),
        (
            "query counter rows overflow",
            DiagnosticLimits {
                max_query_counters: usize::MAX,
                ..limits()
            },
        ),
        (
            "query window buckets overflow",
            DiagnosticLimits {
                max_query_window_buckets: usize::MAX,
                ..limits()
            },
        ),
        (
            "text capacity overflows page size",
            DiagnosticLimits {
                query_text_bytes: usize::MAX,
                ..limits()
            },
        ),
        (
            "output capacity overflows page size",
            DiagnosticLimits {
                max_snapshot_bytes: usize::MAX,
                ..limits()
            },
        ),
        (
            "query slots overflow",
            DiagnosticLimits {
                max_queries: usize::MAX,
                ..limits()
            },
        ),
        (
            "query pages overflow",
            DiagnosticLimits {
                max_queries: usize::MAX / limits().max_snapshot_bytes + 1,
                ..limits()
            },
        ),
    ] {
        assert_eq!(
            Diagnostics::from_settings(
                source(),
                DiagnosticsSettings {
                    enabled: true,
                    limits: profile,
                },
            )
            .unwrap_err(),
            DiagnosticError::InvalidLimits,
            "{case}",
        );
    }
}

#[test]
fn memory_usage_distinguishes_history_capacity_from_occupied_slots() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 2,
        detail_pool_bytes: 0,
        ..limits()
    });
    let component = register(&service, "producer");
    let initial = service.memory_usage();
    assert!(initial.event_slot_bytes > 0);
    assert_eq!(initial.event_capacity, 2);
    assert_eq!(initial.events, 0);
    assert_eq!(initial.history_occupied_bytes, 0);
    assert!(initial.history_reserved_bytes >= initial.event_capacity * initial.event_slot_bytes);
    assert_eq!(
        (
            initial.detail_reserved_bytes,
            initial.detail_capacity_bytes,
            initial.detail_metadata_bytes
        ),
        (0, 0, 0),
    );

    for count in 1..=3 {
        component
            .record_without_projection_lazy(|| {
                Observation::new(
                    classification(FailureClass::Unknown),
                    DiagnosticDetailRef::None,
                    NonZeroU64::MIN,
                )
            })
            .unwrap();
        let usage = service.memory_usage();
        assert_eq!(usage.events, count.min(initial.event_capacity));
        assert_eq!(
            usage.history_occupied_bytes,
            usage.events * usage.event_slot_bytes
        );
        assert_eq!(usage.history_reserved_bytes, initial.history_reserved_bytes);
        assert_eq!(usage.reserved_bytes, initial.reserved_bytes);
    }
    let reader = service.reader();
    let (usage, allocations) = allocations_during(|| reader.memory_usage());
    assert_eq!(allocations, 0);
    let wire = serde_json::to_value(usage).unwrap();
    assert_eq!(wire["event_slot_bytes"], usage.event_slot_bytes.to_string());
    assert_eq!(wire["event_capacity"], "2");
    assert_eq!(
        wire["history_occupied_bytes"],
        (2 * usage.event_slot_bytes).to_string()
    );
    assert_eq!(
        wire["history_reserved_bytes"],
        usage.history_reserved_bytes.to_string()
    );
}

#[test]
fn memory_usage_reports_retained_text_and_utf8_block_slack() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 1,
        detail_pool_bytes: 63,
        detail_block_bytes: 4,
        ..limits()
    });
    let component = register(&service, "producer");
    let initial = service.memory_usage();
    assert_eq!(initial.detail_capacity_bytes, 60);
    assert!(initial.detail_metadata_bytes > 0);
    assert_eq!(
        initial.detail_reserved_bytes,
        initial.detail_capacity_bytes + initial.detail_metadata_bytes
    );
    assert_eq!(
        (
            initial.detail_bytes,
            initial.detail_logical_bytes,
            initial.detail_slack_bytes
        ),
        (0, 0, 0),
    );

    let session = String::from("xy");
    let request = String::from("z");
    component
        .record_without_projection_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "€€€",
                NonZeroU64::MIN,
            )
            .with_correlation(DiagnosticCorrelation {
                session_id: Some(session.as_str().into()),
                request_id: Some(request.as_str().into()),
            })
        })
        .unwrap();
    let reader = service.reader();
    let (occupied, allocations) = allocations_during(|| reader.memory_usage());
    assert_eq!(allocations, 0);
    // Complete UTF-8 scalars share blocks across field boundaries:
    // [€, €, €x, yz]. All three fields use one chain.
    assert_eq!(occupied.detail_bytes, 4 * 4);
    assert_eq!(
        occupied.detail_logical_bytes,
        "€€€".len() + session.len() + request.len()
    );
    assert_eq!(occupied.detail_slack_bytes, 4);
    assert_eq!(
        (
            occupied.detail_reserved_bytes,
            occupied.detail_metadata_bytes
        ),
        (initial.detail_reserved_bytes, initial.detail_metadata_bytes),
    );

    component
        .record_without_projection_lazy(|| {
            Observation::new(classification(FailureClass::Unknown), "q", NonZeroU64::MIN)
        })
        .unwrap();
    let replaced = service.memory_usage();
    assert_eq!(replaced.events, 1);
    assert_eq!(replaced.detail_bytes, 4);
    assert_eq!(replaced.detail_logical_bytes, 1);
    assert_eq!(replaced.detail_slack_bytes, 3);
    assert_eq!(replaced.history_occupied_bytes, replaced.event_slot_bytes);
    assert_eq!(
        replaced.detail_reserved_bytes,
        initial.detail_reserved_bytes
    );
    assert_eq!(
        replaced.detail_metadata_bytes,
        initial.detail_metadata_bytes
    );
    assert_eq!(replaced.reserved_bytes, initial.reserved_bytes);
}

#[test]
fn default_profile_reports_bounded_compiled_backing_capacity() {
    let profile = DiagnosticLimits::default();
    let service = diagnostics(profile.clone());
    let usage = service.memory_usage();
    eprintln!(
        "default diagnostics backing bytes: {}; event_slot_bytes={}; history_reserved_bytes={}; detail_capacity_bytes={}; detail_metadata_bytes={}",
        usage.reserved_bytes,
        usage.event_slot_bytes,
        usage.history_reserved_bytes,
        usage.detail_capacity_bytes,
        usage.detail_metadata_bytes,
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
    let context = PluginObservationContext {
        execution: Some(PluginExecution::NativeAbi.into()),
        phase: Some(PluginPhase::Policy.into()),
    };
    let delivered = Arc::new(AtomicUsize::new(0));
    let callback = Arc::clone(&delivered);
    let _guard = service
        .subscribe_projection(move |event| {
            assert_eq!(event.code(), WRITE.canonical_name());
            assert_ne!(event.component_instance_id, Uuid::nil());
            serde_json::to_writer(std::io::sink(), event).unwrap();
            callback.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    let producers = [
        (&attributed, context),
        (&plain, PluginObservationContext::default()),
    ];
    for (component, context) in &producers {
        component
            .record_lazy(|| {
                Observation::new(
                    classification(FailureClass::Unknown),
                    "projected",
                    NonZeroU64::MIN,
                )
                .with_plugin_context(context.clone())
            })
            .unwrap();
    }
    let (_, allocations) = allocations_during(|| {
        for _ in 0..100 {
            for (component, context) in &producers {
                component
                    .record_lazy(|| {
                        Observation::new(
                            classification(FailureClass::Unknown),
                            "projected",
                            NonZeroU64::MIN,
                        )
                        .with_plugin_context(context.clone())
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
fn owned_shared_payloads_record_and_project_without_new_allocations() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    let session = format!("session-{}", 42);
    let request = format!("request-{}", 42);
    let delivered = Arc::new(AtomicUsize::new(0));
    let captured = Arc::clone(&delivered);
    let _guard = service
        .subscribe_projection(move |event| {
            assert_eq!(event.classification.code(), "logging.sink.write_failed");
            assert_eq!(event.classification.failure_class(), Some("unknown"));
            serde_json::to_writer(std::io::sink(), event).unwrap();
            captured.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    component
        .record_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "warm",
                NonZeroU64::MIN,
            )
            .with_plugin_context(PluginObservationContext {
                execution: Some(PluginExecution::NativeAbi.into()),
                phase: Some(PluginPhase::Policy.into()),
            })
            .with_correlation(DiagnosticCorrelation {
                session_id: Some(session.as_str().into()),
                request_id: Some(request.as_str().into()),
            })
        })
        .unwrap();

    // Build caller-owned inputs before measuring core recording and delivery.
    let observations: Vec<_> = (0..100)
        .map(|_| {
            Observation::new(
                Classification::from_opaque_names(
                    Cow::Owned("logging.sink.write_failed".to_owned()),
                    Some(Cow::Owned("unknown".to_owned())),
                ),
                "owned",
                NonZeroU64::MIN,
            )
            .with_plugin_context(PluginObservationContext {
                execution: Some(PluginExecutionName::from_opaque_name(Cow::Owned(
                    "native_abi".to_owned(),
                ))),
                phase: Some(PluginPhaseName::from_opaque_name(Cow::Owned(
                    "policy".to_owned(),
                ))),
            })
            .with_correlation(DiagnosticCorrelation {
                session_id: Some(session.as_str().into()),
                request_id: Some(request.as_str().into()),
            })
        })
        .collect();
    let mut observations = observations.into_iter();
    let reserved = service.memory_usage().reserved_bytes;
    let (_, allocations) = allocations_during(|| {
        for observation in &mut observations {
            component.record_lazy(|| observation).unwrap();
        }
    });
    assert_eq!(allocations, 0);
    assert_eq!(delivered.load(Ordering::Relaxed), 101);
    assert_eq!(service.memory_usage().reserved_bytes, reserved);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.lifetime.totals.occurrences_total.value.get(), 101);
    assert_eq!(
        view.events[0]
            .context
            .phase
            .as_ref()
            .map(PluginPhaseName::as_str),
        Some("policy")
    );
    assert_eq!(
        view.events[0].correlation.request_id.as_deref(),
        Some("request-42")
    );
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
            NonZeroU64::MIN,
        )
    });
    assert!(matches!(
        foreign_code,
        Err(DiagnosticError::InvalidCode | DiagnosticError::InvalidClassification)
    ));
    let foreign_extension_and_class = right_component.record_without_projection_lazy(|| {
        Observation::extension_with_class(
            left_extension,
            left_class,
            "wrong source pair",
            NonZeroU64::MIN,
        )
    });
    assert!(matches!(
        foreign_extension_and_class,
        Err(DiagnosticError::InvalidCode | DiagnosticError::InvalidClassification)
    ));
    let foreign_class = right_component.record_without_projection_lazy(|| {
        Observation::extension_with_class(
            right_extension,
            left_class,
            "wrong source class",
            NonZeroU64::MIN,
        )
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
fn typed_plugin_context_stays_in_history_while_counts_reach_projection() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 1,
        ..limits()
    });
    let code = BuiltinDiagnosticCode::Failure(FailureCode::Plugin(PluginFailureCode::Common(
        CommonPluginFailureCode::RegistrationFailed,
    )));
    let recorder = service
        .register_component(ComponentRegistration::new("test.plugin_context", &[code]))
        .unwrap();
    let contexts = [
        PluginObservationContext {
            execution: Some(PluginExecution::NativeAbi.into()),
            phase: Some(PluginPhase::Activation.into()),
        },
        PluginObservationContext {
            execution: Some(PluginExecution::WorkerProtocol.into()),
            phase: Some(PluginPhase::Runtime.into()),
        },
    ];
    let projected = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&projected);
    let _guard = service
        .subscribe_projection(move |event| captured.lock().unwrap().push(event.count))
        .unwrap();
    for (context, count) in contexts.iter().zip([2, 3]) {
        recorder
            .record_lazy(|| {
                Observation::new(
                    Classification::failure(
                        FailureCode::Plugin(PluginFailureCode::Common(
                            CommonPluginFailureCode::RegistrationFailed,
                        )),
                        FailureClass::Unknown,
                    )
                    .unwrap(),
                    "operation failed",
                    NonZeroU64::new(count).unwrap(),
                )
                .with_plugin_context(context.clone())
            })
            .unwrap();
    }
    let expected_contexts = contexts;
    assert_eq!(projected.lock().unwrap().as_slice(), &[2.into(), 3.into()]);
    let page = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(page.events.len(), 1);
    assert_eq!(page.events[0].context, expected_contexts[1]);
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

    profile_bounded_workloads();

    for contended in [false, true] {
        let service = diagnostics(DiagnosticLimits::default());
        let recorders = (0..WRITERS)
            .map(|index| register(&service, &format!("profile.{index}")))
            .collect::<Vec<_>>();
        for recorder in &recorders {
            for _ in 0..100 {
                recorder
                    .record_without_projection_lazy(|| {
                        Observation::new(
                            classification(FailureClass::Unknown),
                            DETAIL,
                            NonZeroU64::MIN,
                        )
                    })
                    .unwrap();
            }
        }
        drop(service.snapshot(&DiagnosticQuery::default()).unwrap());
        let reserved = service.memory_usage().reserved_bytes;
        let barrier = Barrier::new(if contended { WRITERS + 2 } else { 2 });
        let begin = Instant::now();
        let (mut samples, mut reads) = thread::scope(|scope| {
            let writers = recorders
                .iter()
                .take(if contended { WRITERS } else { 1 })
                .map(|recorder| {
                    let barrier = &barrier;
                    scope.spawn(move || profile_writer(recorder, barrier))
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
                        let page = service.snapshot(&DiagnosticQuery::default()).unwrap();
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

#[test]
fn builtin_query_selectors_are_independent_of_small_identity_limits() {
    let service = diagnostics(DiagnosticLimits {
        max_identity_bytes: 8,
        max_counters: BuiltinDiagnosticCode::all().len(),
        ..limits()
    });
    let component = service
        .register_component(ComponentRegistration::new(
            "sink",
            BuiltinDiagnosticCode::all(),
        ))
        .unwrap();
    for code in BuiltinDiagnosticCode::all() {
        let classes: Vec<_> = match code {
            BuiltinDiagnosticCode::Failure(_) => {
                code.allowed_classes().iter().copied().map(Some).collect()
            }
            _ => vec![None],
        };
        for class in classes {
            component
                .record_without_projection_lazy(|| {
                    Observation::new(
                        Classification::new(*code, class).unwrap(),
                        "observed",
                        NonZeroU64::MIN,
                    )
                })
                .unwrap();
            let query = DiagnosticQuery {
                component_key: Some("sink".into()),
                code: Some(code.canonical_name().into()),
                failure_class: class.map(|class| class.as_str().into()),
                ..Default::default()
            };
            let view = snapshot(&service, query);
            assert!(!view.events.is_empty(), "missing {code:?}/{class:?}");
            assert!(
                view.events
                    .iter()
                    .all(|event| event.code() == code.canonical_name())
            );
            assert!(view.events.iter().all(
                |event| event.classification.failure_class() == class.map(FailureClass::as_str)
            ));
        }
    }
    for query in [
        DiagnosticQuery {
            component_key: Some("x".repeat(9)),
            ..Default::default()
        },
        DiagnosticQuery {
            code: Some("x".repeat(9)),
            ..Default::default()
        },
        DiagnosticQuery {
            failure_class: Some("x".repeat(33)),
            ..Default::default()
        },
        DiagnosticQuery {
            ownership: ComponentOwnership {
                responsibility: Some("x".repeat(9)),
                ..Default::default()
            },
            ..Default::default()
        },
    ] {
        assert_eq!(
            service.snapshot(&query).unwrap_err(),
            DiagnosticError::InvalidQuery
        );
    }
}

#[test]
fn extension_causes_remain_queryable_above_the_identity_limit() {
    let service = diagnostics(DiagnosticLimits {
        max_identity_bytes: 8,
        ..limits()
    });
    let component = register(&service, "sink");
    let code = component
        .register_extension(ExtensionRegistration {
            namespace: "acme",
            name: "err",
            description: "custom",
            allowed_classes: &[],
            outcome: DiagnosticOutcome::Failure,
        })
        .unwrap()
        .unwrap();
    let name = "x".repeat(32);
    let class = component
        .register_extension_class(code, &name)
        .unwrap()
        .unwrap();
    component
        .record_without_projection_lazy(|| {
            Observation::extension_with_class(code, class, "custom", NonZeroU64::MIN)
        })
        .unwrap();
    let view = snapshot(
        &service,
        DiagnosticQuery {
            code: Some("acme.err".into()),
            failure_class: Some(name.clone()),
            ..Default::default()
        },
    );
    assert_eq!(view.events.len(), 1);
    assert_eq!(
        view.events[0].classification.failure_class(),
        Some(name.as_str())
    );
}

#[test]
fn warmed_small_response_budgets_paginate_full_history_without_allocating() {
    let service = diagnostics(DiagnosticLimits {
        max_detail_bytes: 1024,
        detail_pool_bytes: 262_144,
        query_text_bytes: 524_288,
        ..limits()
    });
    let component = register(&service, "producer");
    let text = "x".repeat(1000);
    for _ in 0..256 {
        component
            .record_without_projection_lazy(|| {
                Observation::new(
                    classification(FailureClass::Unknown),
                    text.as_str(),
                    NonZeroU64::MIN,
                )
            })
            .unwrap();
    }
    let mut query = DiagnosticQuery {
        max_components: Some(0),
        max_response_bytes: Some(4096),
        ..Default::default()
    };
    drop(service.snapshot(&query).unwrap());
    let mut sequences = BTreeSet::new();
    for _ in 0..256 {
        let (lease, allocations) = allocations_during(|| service.snapshot(&query).unwrap());
        assert_eq!(allocations, 0);
        assert!(lease.as_bytes().len() <= 4096);
        let view = lease.to_snapshot().unwrap();
        assert!(!view.events.is_empty());
        assert_eq!(
            view.coverage.retention.events_retained.get(),
            view.events.len() as u64
        );
        assert_eq!(
            view.coverage.retention.oldest_event_sequence,
            view.events.first().map(|e| e.sequence)
        );
        assert_eq!(
            view.coverage.retention.latest_event_sequence,
            view.events.last().map(|e| e.sequence)
        );
        for event in &view.events {
            assert!(sequences.insert(event.sequence));
        }
        query.continuation = view.continuation;
        if query.continuation.is_none() {
            break;
        }
    }
    assert!(query.continuation.is_none());
    assert_eq!(sequences.len(), 256);
    assert_eq!(service.memory_usage().events, 256);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.lifetime.totals.recorded_events_total.value.get(), 256);
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 0);
}

#[test]
fn large_interleaved_history_reclamation_preserves_order_counts_and_storage() {
    const CAPACITY: usize = 8192;
    const WRAP: usize = 32;
    let service = diagnostics(DiagnosticLimits {
        max_components: 2,
        max_events: CAPACITY,
        max_query_events: CAPACITY,
        max_queries: 1,
        max_snapshot_bytes: 8 * 1024 * 1024,
        query_text_bytes: 1024 * 1024,
        detail_pool_bytes: CAPACITY * 64,
        max_owned_bytes: 32 * 1024 * 1024,
        ..limits()
    });
    let (owner, first) = service
        .register_owned_component(ComponentRegistration::new("first", &CODES))
        .unwrap();
    let old_identity = first.identity().unwrap().clone();
    let second = register(&service, "second");
    let reserved = service.memory_usage().reserved_bytes;
    for pair in 1..=(CAPACITY + WRAP) / 2 {
        record(&first, (pair * 2 - 1) as u64);
        record(&second, (pair * 2) as u64);
    }
    owner.close().unwrap();
    let replacement = register(&service, "first");
    assert_ne!(
        replacement.identity().unwrap().instance_id,
        old_identity.instance_id
    );
    assert!(replacement.identity().unwrap().generation > old_identity.generation);
    let view = snapshot(&service, DiagnosticQuery::default());
    let expected: Vec<_> = (WRAP + 2..=CAPACITY + WRAP)
        .step_by(2)
        .map(|count| count as u64)
        .collect();
    assert_eq!(
        view.events
            .iter()
            .map(|event| event.count.get())
            .collect::<Vec<_>>(),
        expected
    );
    assert!(
        view.events
            .iter()
            .all(|event| event.component.key == "second" && event.message() == Some("failure"))
    );
    let evicted = (WRAP + CAPACITY / 2) as u64;
    assert_eq!(view.coverage.loss.events_evicted.value.get(), evicted);
    assert_eq!(
        view.lifetime.totals.history_evictions_total.value.get(),
        evicted
    );
    assert_eq!(
        row(&view, "second")
            .lifetime
            .totals
            .history_evictions_total
            .value
            .get(),
        (WRAP / 2) as u64
    );
    assert_eq!(
        view.lifetime.totals.recorded_events_total.value.get(),
        (CAPACITY + WRAP) as u64
    );
    let usage = service.memory_usage();
    assert_eq!(
        (usage.detail_bytes, usage.reserved_bytes),
        (CAPACITY / 2 * 64, reserved)
    );
    assert_eq!(
        first
            .record_without_projection_lazy(|| Observation::new(
                classification(FailureClass::Unknown),
                "stale",
                NonZeroU64::MIN,
            ))
            .unwrap_err(),
        DiagnosticError::ComponentClosed
    );
    record(&replacement, 1);
    let replaced = snapshot(
        &service,
        DiagnosticQuery {
            component_key: Some("first".into()),
            ..Default::default()
        },
    );
    assert_eq!(replaced.events.len(), 1);
    assert_eq!(
        replaced.events[0].component,
        *replacement.identity().unwrap()
    );
    assert_eq!(replaced.events[0].message(), Some("failure"));
    assert_eq!(replaced.coverage.loss.events_evicted.value.get(), evicted);
    let usage = service.memory_usage();
    assert_eq!(
        (usage.detail_bytes, usage.reserved_bytes),
        ((CAPACITY / 2 + 1) * 64, reserved)
    );
}

#[test]
fn rejected_record_returns_the_publication_permit_before_next_delivery() {
    let service = diagnostics(DiagnosticLimits {
        max_projection_workspaces: 1,
        ..limits()
    });
    let component = register(&service, "producer");
    let delivered = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&delivered);
    let _subscription = service
        .subscribe_projection(move |_| {
            count.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    assert!(
        component
            .record_lazy(|| Observation::new(
                Classification::success(SuccessCode::LoggingInitialized),
                "undeclared code",
                NonZeroU64::MIN,
            ))
            .is_err()
    );
    assert_eq!(service.memory_usage().projection_workspaces, 0);
    component
        .record_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "accepted",
                NonZeroU64::MIN,
            )
        })
        .unwrap();
    assert_eq!(delivered.load(Ordering::Relaxed), 1);
    let snapshot = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(
        snapshot.lifetime.totals.recorded_events_total.value.get(),
        1
    );
    assert_eq!(snapshot.coverage.loss.observations_rejected.value.get(), 1);
    assert_eq!(
        snapshot
            .coverage
            .loss
            .projection_deliveries_omitted
            .value
            .get(),
        0
    );
    assert_eq!(service.memory_usage().projection_workspaces, 0);
}

#[test]
fn extension_causes_reuse_the_reserved_builtin_name_catalogue() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    let extension = component
        .register_extension(ExtensionRegistration {
            namespace: "acme",
            name: "failure",
            description: "custom failure",
            allowed_classes: &[],
            outcome: DiagnosticOutcome::Failure,
        })
        .unwrap()
        .unwrap();
    let metadata = service.memory_usage().metadata_bytes;
    for name in [
        "invalid_input",
        "invalid_state",
        "not_found",
        "conflict",
        "authentication",
        "permission_denied",
        "unsupported",
        "incompatible",
        "timeout",
        "unavailable",
        "resource_exhausted",
        "io",
        "execution_failed",
        "policy_rejected",
        "internal",
        "unknown",
    ] {
        assert!(FailureClass::from_canonical_name(name).is_some());
        assert_eq!(
            component
                .register_extension_class(extension, name)
                .unwrap_err(),
            DiagnosticError::InvalidClassification
        );
    }
    assert_eq!(service.memory_usage().metadata_bytes, metadata);
    assert!(
        component
            .register_extension_class(extension, "backend_busy")
            .unwrap()
            .is_some()
    );
}

#[test]
fn query_counter_and_bucket_page_budgets_are_independent_of_store_capacity() {
    for (counter_cap, bucket_cap) in [(1, 3), (3, 1)] {
        let service = diagnostics(DiagnosticLimits {
            max_query_counters: counter_cap,
            max_query_window_buckets: bucket_cap,
            ..limits()
        });
        let mut recorders = Vec::new();
        for index in 1..=3 {
            let key = format!("component.{index}");
            let recorder = service
                .register_component(ComponentRegistration {
                    window: Some(WindowSettings {
                        resolution_millis: 60_000,
                        bucket_count: 1,
                    }),
                    ..ComponentRegistration::new(&key, &[WRITE])
                })
                .unwrap();
            record(&recorder, index);
            recorders.push(recorder);
        }
        let mut query = DiagnosticQuery {
            max_events: Some(0),
            ..Default::default()
        };
        let mut seen = BTreeSet::new();
        for _ in 0..3 {
            let view = snapshot(&service, query.clone());
            assert_eq!(view.components.len(), 1);
            assert_eq!(view.components[0].counters.len(), 1);
            assert_eq!(view.components[0].windows[0].buckets.len(), 1);
            assert!(seen.insert(view.components[0].identity.instance_id));
            assert_eq!(view.lifetime.totals.occurrences_total.value.get(), 6);
            query.continuation = view.continuation;
        }
        assert!(query.continuation.is_none());
        assert_eq!(seen.len(), 3);
        assert_eq!(service.memory_usage().counters, 3);
        assert_eq!(service.memory_usage().window_buckets, 3);
    }
}

#[test]
fn oversized_component_rows_report_query_capacity_without_losing_evidence() {
    for (counter_cap, bucket_cap, window) in [
        (1, 32, None),
        (
            64,
            0,
            Some(WindowSettings {
                resolution_millis: 60_000,
                bucket_count: 1,
            }),
        ),
    ] {
        let service = diagnostics(DiagnosticLimits {
            max_query_counters: counter_cap,
            max_query_window_buckets: bucket_cap,
            ..limits()
        });
        let component = service
            .register_component(ComponentRegistration {
                window,
                ..ComponentRegistration::new("producer", &[WRITE, SUCCESS])
            })
            .unwrap();
        record(&component, 2);
        component
            .record_without_projection_lazy(|| {
                Observation::new(
                    Classification::success(SuccessCode::SinkInitialized),
                    "success",
                    NonZeroU64::MIN,
                )
            })
            .unwrap();
        let reserved = service.memory_usage().reserved_bytes;
        assert_eq!(
            service.snapshot(&DiagnosticQuery::default()).unwrap_err(),
            DiagnosticError::SnapshotCapacity
        );
        let view = snapshot(
            &service,
            DiagnosticQuery {
                max_components: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(view.events.len(), 2);
        assert_eq!(view.lifetime.totals.occurrences_total.value.get(), 3);
        assert_eq!(view.coverage.loss.events_evicted.value.get(), 0);
        assert_eq!(service.memory_usage().reserved_bytes, reserved);
    }
}

#[test]
fn recording_and_projection_work_without_query_backing() {
    let with_queries = diagnostics(limits());
    let service = diagnostics(DiagnosticLimits {
        max_queries: 0,
        max_query_components: 0,
        max_query_events: 0,
        max_query_counters: 0,
        max_query_window_buckets: 0,
        max_snapshot_bytes: 0,
        query_text_bytes: 0,
        ..limits()
    });
    assert!(service.memory_usage().reserved_bytes < with_queries.memory_usage().reserved_bytes);
    assert_eq!(
        service.reader().availability(),
        DiagnosticAvailability::Available
    );
    let component = register(&service, "producer");
    let delivered = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&delivered);
    let _subscription = service
        .subscribe_projection(move |event| {
            count.fetch_add(event.count.get() as usize, Ordering::Relaxed);
        })
        .unwrap();
    component
        .record_lazy(|| {
            Observation::new(
                classification(FailureClass::Unknown),
                "failure",
                NonZeroU64::new(2).unwrap(),
            )
        })
        .unwrap();
    assert_eq!(delivered.load(Ordering::Relaxed), 2);
    assert_eq!(
        service.snapshot(&DiagnosticQuery::default()).unwrap_err(),
        DiagnosticError::QueryDisabled
    );
    assert_eq!(
        component
            .reader()
            .snapshot(&DiagnosticQuery::default())
            .unwrap_err(),
        DiagnosticError::QueryDisabled
    );
    assert_eq!(service.memory_usage().events, 1);
    assert_eq!(service.memory_usage().query_workspaces, 0);
    assert_eq!(service.memory_usage().projection_workspaces, 0);
}

#[test]
fn default_query_profile_is_smaller_than_store_sized_workspaces() {
    let profile = DiagnosticLimits::default();
    let previous_query_profile = DiagnosticLimits {
        max_query_counters: profile.max_counters,
        max_query_window_buckets: profile.max_window_buckets,
        max_snapshot_bytes: 524_288,
        query_text_bytes: 524_288,
        ..profile.clone()
    };
    let current = diagnostics(profile);
    let previous = diagnostics(previous_query_profile);
    let current_bytes = current.memory_usage().reserved_bytes;
    let previous_bytes = previous.memory_usage().reserved_bytes;
    eprintln!(
        "default context backing: {current_bytes} bytes; store-sized query profile: {previous_bytes} bytes"
    );
    assert!(current_bytes < previous_bytes / 2);
    assert_eq!(
        current.limits().max_counters,
        previous.limits().max_counters
    );
    assert_eq!(
        current.limits().max_window_buckets,
        previous.limits().max_window_buckets
    );
}

#[test]
fn component_counter_index_survives_refresh_and_reclaimed_lower_slots() {
    let service = diagnostics(DiagnosticLimits {
        max_components: 3,
        max_counters: 3,
        ..limits()
    });
    let (donor_owner, _donor) = service
        .register_owned_component(ComponentRegistration::new("donor", &[WRITE]))
        .unwrap();
    let (owner, target) = service
        .register_owned_component(ComponentRegistration::new("target", &[WRITE]))
        .unwrap();
    let last = service
        .register_component(ComponentRegistration::new("last", &[WRITE]))
        .unwrap();
    record(&target, 2);
    record(&last, 5);
    donor_owner.close().unwrap();
    let extension = target
        .register_extension(ExtensionRegistration {
            namespace: "acme",
            name: "failure",
            description: "custom failure",
            allowed_classes: &[FailureClass::Unknown],
            outcome: DiagnosticOutcome::Failure,
        })
        .unwrap()
        .unwrap();
    target
        .record_without_projection_lazy(|| {
            Observation::extension(
                extension,
                Some(FailureClass::Unknown),
                DiagnosticOutcome::Failure,
                "custom",
                NonZeroU64::new(3).unwrap(),
            )
        })
        .unwrap();
    owner
        .refresh(ComponentOwnership {
            configuration_id: Some("refreshed".into()),
            ..Default::default()
        })
        .unwrap();
    let view = snapshot(&service, DiagnosticQuery::default());
    let target_row = row(&view, "target");
    assert_eq!(
        target_row
            .counters
            .iter()
            .map(|counter| counter.code.as_str())
            .collect::<Vec<_>>(),
        ["acme.failure", WRITE.canonical_name()]
    );
    assert_eq!(
        target_row
            .counters
            .iter()
            .map(|counter| counter.occurrences.value.get())
            .collect::<Vec<_>>(),
        [3, 2]
    );
    owner.close().unwrap();
    let replacement = service
        .register_component(ComponentRegistration::new("target", &[WRITE]))
        .unwrap();
    assert_eq!(
        target
            .record_without_projection_lazy(|| Observation::new(
                classification(FailureClass::Unknown),
                "stale",
                NonZeroU64::MIN
            ))
            .unwrap_err(),
        DiagnosticError::ComponentClosed
    );
    assert_eq!(
        replacement
            .record_without_projection_lazy(|| Observation::extension(
                extension,
                Some(FailureClass::Unknown),
                DiagnosticOutcome::Failure,
                "stale extension",
                NonZeroU64::MIN
            ))
            .unwrap_err(),
        DiagnosticError::InvalidCode
    );
    record(&replacement, 1);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(code_total(row(&view, "target"), WRITE), 1);
    assert_eq!(code_total(row(&view, "last"), WRITE), 5);
}

#[test]
fn window_reservation_reuse_preserves_neighbors_and_rejects_stale_owner() {
    let service = diagnostics(DiagnosticLimits {
        max_components: 2,
        max_window_buckets: 3,
        ..limits()
    });
    let registration = |key, bucket_count| ComponentRegistration {
        key,
        builtin_codes: &CODES,
        ownership: ComponentOwnership::default(),
        window: Some(WindowSettings {
            resolution_millis: 60_000,
            bucket_count,
        }),
    };
    let (owner, donor) = service
        .register_owned_component(registration("donor", 2))
        .unwrap();
    let survivor = service
        .register_component(registration("survivor", 1))
        .unwrap();
    record(&donor, 2);
    record(&survivor, 5);
    owner.close().unwrap();
    let replacement = service
        .register_component(registration("replacement", 2))
        .unwrap();
    assert_eq!(
        donor
            .record_without_projection_lazy(|| Observation::new(
                classification(FailureClass::Unknown),
                "stale",
                NonZeroU64::MIN
            ))
            .unwrap_err(),
        DiagnosticError::ComponentClosed
    );
    record(&replacement, 3);
    let view = snapshot(&service, DiagnosticQuery::default());
    let failures = |key| {
        row(&view, key).windows[0]
            .buckets
            .iter()
            .map(|bucket| bucket.failures.value.get())
            .sum::<u64>()
    };
    assert_eq!(failures("survivor"), 5);
    assert_eq!(failures("replacement"), 3);
    assert_eq!(service.memory_usage().window_buckets, 3);
}

#[test]
fn packed_event_fields_share_blocks_without_recording_allocations() {
    let service = diagnostics(DiagnosticLimits {
        max_events: 256,
        detail_pool_bytes: 256 * 256,
        detail_block_bytes: 256,
        ..limits()
    });
    let component = register(&service, "producer");
    let (_, allocations) = allocations_during(|| {
        for _ in 0..256 {
            component
                .record_without_projection_lazy(|| {
                    Observation::new(
                        classification(FailureClass::Unknown),
                        "message🙂",
                        NonZeroU64::MIN,
                    )
                    .with_correlation(DiagnosticCorrelation {
                        session_id: Some("session€".into()),
                        request_id: Some("request".into()),
                    })
                })
                .unwrap();
        }
    });
    assert_eq!(allocations, 0);
    let occupied = service.memory_usage();
    assert_eq!(occupied.events, 256);
    assert_eq!(occupied.detail_bytes, 256 * 256);
    assert_eq!(
        occupied.detail_logical_bytes,
        256 * ("message🙂".len() + "session€".len() + "request".len())
    );
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.events.len(), 256);
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 0);
    for event in &view.events {
        assert_eq!(event.message(), Some("message🙂"));
        assert_eq!(event.correlation.session_id.as_deref(), Some("session€"));
        assert_eq!(event.correlation.request_id.as_deref(), Some("request"));
    }
    // Exactly one prior chain is reclaimed for another complete event.
    record(&component, 1);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.events.len(), 256);
    assert_eq!(view.coverage.loss.events_evicted.value.get(), 1);
    assert_eq!(code_total(&view.components[0], WRITE), 257);
}

#[test]
fn packed_text_preserves_empty_fields_without_pool_capacity() {
    let service = diagnostics(DiagnosticLimits {
        detail_pool_bytes: 0,
        ..limits()
    });
    let component = register(&service, "producer");
    component
        .record_without_projection_lazy(|| {
            Observation::new(classification(FailureClass::Unknown), "", NonZeroU64::MIN)
                .with_correlation(DiagnosticCorrelation {
                    session_id: Some("".into()),
                    request_id: None,
                })
        })
        .unwrap();
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.events[0].message(), Some(""));
    assert_eq!(view.events[0].correlation.session_id.as_deref(), Some(""));
    assert_eq!(view.events[0].correlation.request_id, None);
    assert_eq!(service.memory_usage().detail_bytes, 0);
    assert_eq!(view.coverage.loss.details_omitted.value.get(), 0);
    assert_eq!(view.coverage.loss.correlations_omitted.value.get(), 0);
}

#[test]
fn repeated_borrowed_queries_reuse_owned_selectors_without_allocations() {
    let service = diagnostics(limits());
    let component = service
        .register_component(ComponentRegistration {
            ownership: ComponentOwnership {
                configuration_id: Some("configuration".into()),
                responsibility: Some("owner".into()),
                ..Default::default()
            },
            ..ComponentRegistration::new("producer", &CODES)
        })
        .unwrap();
    record(&component, 2);
    let mut query = DiagnosticQuery {
        component_key: Some("producer".into()),
        code: Some(WRITE.canonical_name().into()),
        failure_class: Some(FailureClass::Unknown.as_str().into()),
        ownership: ComponentOwnership {
            configuration_id: Some("configuration".into()),
            responsibility: Some("owner".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let reader = service.reader();
    drop(reader.snapshot(&query).unwrap());
    for _ in 0..32 {
        let (lease, allocations) = allocations_during(|| reader.snapshot(&query).unwrap());
        assert_eq!(allocations, 0);
        let view = lease.to_snapshot().unwrap();
        assert_eq!(view.components.len(), 1);
        assert_eq!(view.events.len(), 1);
        // A live lease owns its captured page and does not borrow the query.
        query.continuation = view.continuation;
        assert!(!lease.as_bytes().is_empty());
    }
}

#[test]
fn extension_class_sets_cover_all_builtin_causes_and_reject_duplicates() {
    let service = diagnostics(limits());
    let component = register(&service, "producer");
    let classes: Vec<_> = enumset::EnumSet::<FailureClass>::all().iter().collect();
    let extension = component
        .register_extension(ExtensionRegistration {
            namespace: "acme",
            name: "failure",
            description: "all typed causes",
            allowed_classes: &classes,
            outcome: DiagnosticOutcome::Failure,
        })
        .unwrap()
        .unwrap();
    for class in &classes {
        component
            .record_without_projection_lazy(|| {
                Observation::extension(
                    extension,
                    Some(*class),
                    DiagnosticOutcome::Failure,
                    "observed",
                    NonZeroU64::MIN,
                )
            })
            .unwrap();
    }
    let occupied = service.memory_usage().counters;
    assert_eq!(
        component
            .register_extension(ExtensionRegistration {
                namespace: "acme",
                name: "duplicate",
                description: "invalid duplicate declaration",
                allowed_classes: &[FailureClass::Io, FailureClass::Io],
                outcome: DiagnosticOutcome::Failure,
            })
            .unwrap_err(),
        DiagnosticError::InvalidClassification
    );
    assert_eq!(service.memory_usage().counters, occupied);
    let view = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(view.events.len(), classes.len());
    assert_eq!(
        view.components[0]
            .counters
            .iter()
            .find(|counter| counter.code == "acme.failure")
            .unwrap()
            .occurrences
            .value
            .get(),
        classes.len() as u64
    );
    for (event, class) in view.events.iter().zip(classes) {
        assert_eq!(event.classification.failure_class(), Some(class.as_str()));
    }
}

#[test]
fn leased_json_matches_public_models_for_optional_borrowed_fields() {
    let service = diagnostics(limits());
    for (index, configuration) in [None, Some(""), Some("configuration")]
        .into_iter()
        .enumerate()
    {
        let component = service
            .register_component(ComponentRegistration {
                ownership: ComponentOwnership {
                    configuration_id: configuration.map(str::to_owned),
                    responsibility: configuration.map(str::to_owned),
                    ..Default::default()
                },
                ..ComponentRegistration::new(&format!("producer.{index}"), &CODES)
            })
            .unwrap();
        for classification in [
            classification(FailureClass::Unknown),
            Classification::success(SuccessCode::SinkInitialized),
            Classification::notice(NoticeCode::ConfigurationInherited),
        ] {
            component
                .record_without_projection_lazy(|| {
                    Observation::new(
                        classification,
                        configuration.map_or(DiagnosticDetailRef::None, DiagnosticDetailRef::Text),
                        NonZeroU64::MIN,
                    )
                    .with_correlation(DiagnosticCorrelation {
                        session_id: configuration.map(Into::into),
                        request_id: configuration.map(Into::into),
                    })
                })
                .unwrap();
        }
    }
    let lease = service.snapshot(&DiagnosticQuery::default()).unwrap();
    let actual: serde_json::Value = serde_json::from_slice(lease.as_bytes()).unwrap();
    let expected = serde_json::to_value(lease.to_snapshot().unwrap()).unwrap();
    assert_eq!(actual, expected);
}

fn measured_profile<T>(
    name: &str,
    operations: usize,
    body: impl FnOnce() -> T,
) -> (T, AllocationStats) {
    let begin = std::time::Instant::now();
    let (value, stats) = allocation_stats_during(body);
    eprintln!(
        "{name}: operations={operations} elapsed={:?} traffic={stats:?}",
        begin.elapsed()
    );
    (value, stats)
}

/// Exercise capacity pressure through public APIs; timings remain informational.
fn profile_bounded_workloads() {
    let service = profile_full_counter_pool();
    profile_small_response_pages(&service);
    drop(service);
    profile_interleaved_reclamation();
    profile_block_geometries();
}

fn profile_full_counter_pool() -> Diagnostics {
    const COMPONENTS: usize = 64;
    const OPERATIONS: usize = 4096;
    let codes = BuiltinDiagnosticCode::all();
    let service = diagnostics(DiagnosticLimits {
        max_components: COMPONENTS,
        max_counters: COMPONENTS * codes.len(),
        detail_pool_bytes: 256 * 256,
        detail_block_bytes: 256,
        ..limits()
    });
    let producers: Vec<_> = (0..COMPONENTS)
        .map(|index| {
            service
                .register_component(ComponentRegistration::new(
                    &format!("profile.{index}"),
                    codes,
                ))
                .unwrap()
        })
        .collect();
    assert_eq!(service.memory_usage().counters, COMPONENTS * codes.len());
    let reserved = service.memory_usage().reserved_bytes;
    let write = |producer: &ComponentRecorder| {
        producer
            .record_without_projection_lazy(|| {
                Observation::new(
                    classification(FailureClass::Unknown),
                    "message🙂",
                    NonZeroU64::MIN,
                )
                .with_correlation(DiagnosticCorrelation {
                    session_id: Some("session€".into()),
                    request_id: Some("request".into()),
                })
            })
            .unwrap()
    };
    write(&producers[0]);
    let (_, stats) = measured_profile(
        "full counter pool + packed message/session/request",
        OPERATIONS,
        || {
            for index in 0..OPERATIONS {
                write(&producers[index % COMPONENTS]);
            }
        },
    );
    assert_eq!(stats, AllocationStats::default());
    let usage = service.memory_usage();
    assert_eq!(usage.events, 256);
    assert_eq!(usage.detail_bytes, 256 * 256);
    assert_eq!(usage.reserved_bytes, reserved);

    service
}

fn profile_small_response_pages(service: &Diagnostics) {
    let mut query = DiagnosticQuery {
        max_components: Some(0),
        max_response_bytes: Some(4096),
        ..Default::default()
    };
    drop(service.snapshot(&query).unwrap());
    let mut events = 0;
    let mut pages = 0;
    let mut total_traffic = AllocationStats::default();
    loop {
        let (page, stats) = allocation_stats_during(|| service.snapshot(&query).unwrap());
        assert_eq!(stats, AllocationStats::default());
        assert!(page.as_bytes().len() <= 4096);
        // Explicit caller-owned decoding is measured separately from bounded capture.
        let (view, traffic) = allocation_stats_during(|| page.to_snapshot().unwrap());
        total_traffic.allocation_calls += traffic.allocation_calls;
        total_traffic.allocated_bytes += traffic.allocated_bytes;
        total_traffic.free_calls += traffic.free_calls;
        total_traffic.freed_bytes += traffic.freed_bytes;
        total_traffic.realloc_calls += traffic.realloc_calls;
        total_traffic.realloc_old_bytes += traffic.realloc_old_bytes;
        assert!(!view.events.is_empty());
        events += view.events.len();
        pages += 1;
        query.continuation = view.continuation;
        if query.continuation.is_none() {
            break;
        }
        assert!(pages <= 256);
    }
    assert_eq!(events, 256);
    assert!(pages > 1);
    eprintln!(
        "small responses: pages={pages} events={events}, capture traffic=zero; decode-only traffic={total_traffic:?} (owned result drops outside scope)"
    );
}

fn profile_interleaved_reclamation() {
    const HISTORY: usize = 1024;
    let service = diagnostics(DiagnosticLimits {
        max_components: 2,
        max_counters: 2,
        max_events: HISTORY,
        detail_pool_bytes: HISTORY * 64,
        ..limits()
    });
    let (owner, first) = service
        .register_owned_component(ComponentRegistration::new("first", &[WRITE]))
        .unwrap();
    let second = service
        .register_component(ComponentRegistration::new("second", &[WRITE]))
        .unwrap();
    for _ in 0..HISTORY / 2 {
        record(&first, 1);
        record(&second, 1);
    }
    owner.close().unwrap();
    let reserved = service.memory_usage().reserved_bytes;
    let (_, traffic) = measured_profile("interleaved component reclamation", HISTORY / 2, || {
        service
            .register_component(ComponentRegistration::new("replacement", &[WRITE]))
            .unwrap()
    });
    // Registration owns a new public identity/recorder; report that traffic rather
    // than asserting it shares the allocation-free recording contract.
    eprintln!("registration/reclamation traffic={traffic:?}");
    assert_eq!(service.memory_usage().events, HISTORY / 2);
    assert_eq!(service.memory_usage().detail_bytes, HISTORY / 2 * 64);
    assert_eq!(service.memory_usage().reserved_bytes, reserved);
    let page = snapshot(
        &service,
        DiagnosticQuery {
            max_events: Some(0),
            ..Default::default()
        },
    );
    assert_eq!(
        page.coverage.loss.events_evicted.value.get(),
        (HISTORY / 2) as u64
    );
    assert_eq!(
        page.lifetime.totals.recorded_events_total.value.get(),
        HISTORY as u64
    );
}

fn profile_block_geometries() {
    // Contrast supported block geometries under repeated whole-payload reclamation.
    let payload = "🙂".repeat(1024);
    for block in [4, 256] {
        let service = diagnostics(DiagnosticLimits {
            detail_block_bytes: block,
            detail_pool_bytes: 4096,
            max_detail_bytes: 4096,
            ..limits()
        });
        let producer = register(&service, "geometry");
        let write = || {
            producer
                .record_without_projection_lazy(|| {
                    Observation::new(
                        classification(FailureClass::Unknown),
                        payload.as_str(),
                        NonZeroU64::MIN,
                    )
                })
                .unwrap()
        };
        write();
        let (_, traffic) = measured_profile(
            &format!("4096-byte UTF-8 payload / {block}-byte blocks"),
            1000,
            || {
                for _ in 0..1000 {
                    write();
                }
            },
        );
        assert_eq!(traffic, AllocationStats::default());
        assert_eq!(service.memory_usage().events, 1);
        let view = snapshot(&service, DiagnosticQuery::default());
        assert_eq!(view.events[0].message(), Some(payload.as_str()));
        assert_eq!(view.coverage.loss.events_evicted.value.get(), 1000);
    }
}

#[test]
fn extension_names_cannot_authorize_operational_output() {
    let service = diagnostics(limits());
    let component = register(&service, "private.component");
    let extension = component
        .register_extension(ExtensionRegistration {
            namespace: "private",
            name: "secret_name",
            description: "private description",
            allowed_classes: &[FailureClass::Unknown],
            outcome: DiagnosticOutcome::Failure,
        })
        .unwrap()
        .unwrap();
    let delivered = Arc::new(AtomicUsize::new(0));
    let callback = delivered.clone();
    let _projection = service
        .subscribe_projection(move |_| {
            callback.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    component
        .record_lazy(|| {
            Observation::extension(
                extension,
                Some(FailureClass::Unknown),
                DiagnosticOutcome::Failure,
                "private evidence",
                NonZeroU64::MIN,
            )
        })
        .unwrap();
    assert_eq!(delivered.load(Ordering::Relaxed), 0);
    let page = snapshot(&service, DiagnosticQuery::default());
    assert_eq!(page.events[0].code(), "private.secret_name");
    assert_eq!(page.events[0].message(), Some("private evidence"));
    assert_eq!(page.coverage.loss.projections_suppressed.value.get(), 1);
}

fn profile_writer(recorder: &ComponentRecorder, barrier: &std::sync::Barrier) -> Vec<u128> {
    use std::hint::black_box;
    use std::time::Instant;
    const OPERATIONS: usize = 25_000;
    const SAMPLE_EVERY: usize = 50;
    const DETAIL: &str = "bounded producer detail for local recording and contention measurement";
    let mut samples = Vec::with_capacity(OPERATIONS / SAMPLE_EVERY);
    barrier.wait();
    for index in 0..OPERATIONS {
        let started = (index % SAMPLE_EVERY == 0).then(Instant::now);
        let outcome = recorder
            .record_without_projection_lazy(|| {
                Observation::new(
                    classification(FailureClass::Unknown),
                    DETAIL,
                    NonZeroU64::MIN,
                )
            })
            .unwrap();
        black_box(outcome);
        if let Some(started) = started {
            samples.push(started.elapsed().as_nanos());
        }
    }
    samples
}
