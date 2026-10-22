// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conservation, bounded allocation, mutation, and public read contracts.
use nemo_relay::diagnostics::*;
use std::num::NonZeroU64;
#[path = "support/diagnostic_allocations.rs"]
mod diagnostic_allocations;
use diagnostic_allocations::allocation_stats_during;
const CODE: BuiltinDiagnosticCode = BuiltinDiagnosticCode::Success(SuccessCode::SinkInitialized);
fn context(events: usize) -> Diagnostics {
    Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            limits: DiagnosticLimits {
                max_components: 4,
                max_events: events,
                max_queries: 2,
                ..Default::default()
            },
        },
    )
    .unwrap()
}
fn record(r: &ComponentRecorder, message: &str) {
    r.record_without_projection_lazy(|| {
        Observation::new(
            Classification::success(SuccessCode::SinkInitialized),
            message,
            NonZeroU64::MIN,
        )
        .with_correlation(DiagnosticCorrelation {
            session_id: Some("session".into()),
            request_id: Some("request".into()),
        })
    })
    .unwrap();
}
fn page(d: &Diagnostics, q: &DiagnosticQuery) -> DiagnosticMemoryPage {
    d.reader()
        .inspect_memory(q)
        .unwrap()
        .to_memory_page()
        .unwrap()
}
fn conserved(p: &DiagnosticMemoryPage) {
    assert_eq!(
        p.component_occupied_bytes.get() + p.shared_or_unassigned_reserved_bytes.get(),
        p.aggregate.reserved_bytes as u64
    );
    for e in &p.entries {
        assert_eq!(
            e.footprint.logical_text_bytes.get() + e.footprint.detail_slack_bytes.get(),
            e.footprint.detail_payload_bytes.get()
        );
        assert_eq!(
            e.footprint.header_bytes.get()
                + e.footprint.detail_payload_bytes.get()
                + e.footprint.detail_metadata_bytes.get(),
            e.footprint.occupied_bytes.get()
        );
    }
}
#[test]
fn entry_component_accounting_and_selection_conserve_reserved_backing() {
    let d = context(4);
    let (owner, a) = d
        .register_owned_component(ComponentRegistration {
            window: Some(WindowSettings {
                bucket_count: 3,
                resolution_millis: 1000,
            }),
            ..ComponentRegistration::new("memory.a", &[CODE])
        })
        .unwrap();
    let b = d
        .register_component(ComponentRegistration::new("memory.b", &[CODE]))
        .unwrap();
    a.register_extension(ExtensionRegistration {
        namespace: "acme",
        name: "memory_notice",
        description: "bounded extension",
        allowed_classes: &[],
        outcome: DiagnosticOutcome::Unknown,
    })
    .unwrap();
    record(&a, "🙂abcdef");
    record(&b, "");
    let all = page(&d, &DiagnosticQuery::default());
    conserved(&all);
    assert_eq!(all.components.len(), 2);
    assert_eq!(all.entries.len(), 2);
    assert_eq!(all.entries[0].footprint.logical_text_bytes.get(), 24);
    assert_eq!(all.entries[0].footprint.detail_payload_bytes.get(), 256);
    assert!(all.components[0].footprint.window_bytes.get() > 0);
    assert!(all.components[0].footprint.extension_bytes.get() > 0);
    assert!(all.components[1].footprint.counter_bytes.get() > 0);
    let filtered = page(
        &d,
        &DiagnosticQuery {
            component_key: Some("memory.a".into()),
            ..Default::default()
        },
    );
    conserved(&filtered);
    assert_eq!(
        filtered.component_occupied_bytes,
        all.component_occupied_bytes
    );
    assert_eq!(filtered.entries.len(), 1);
    assert_eq!(filtered.selection.events.get(), 1);
    for _ in 0..20 {
        owner.refresh(ComponentOwnership::default()).unwrap();
        record(&a, "longer");
    }
    let after = page(&d, &DiagnosticQuery::default());
    conserved(&after);
    assert_eq!(after.aggregate.reserved_bytes, all.aggregate.reserved_bytes);
    assert_eq!(after.entries.len(), 4);
    assert_eq!(
        after
            .components
            .iter()
            .map(|c| c.footprint.retained_event_bytes.get())
            .sum::<u64>(),
        after
            .entries
            .iter()
            .map(|e| e.footprint.occupied_bytes.get())
            .sum::<u64>()
    );
    owner.close().unwrap();
    drop(a);
    let closed = page(&d, &DiagnosticQuery::default());
    assert_eq!(closed.components.len(), 2);
    conserved(&closed);
}
#[test]
fn inspection_shares_permits_cursors_and_never_changes_evidence() {
    let d = context(8);
    let a = d
        .register_component(ComponentRegistration::new("a", &[CODE]))
        .unwrap();
    for _ in 0..4 {
        record(&a, "secret.marker");
    }
    let before = d
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    let q = DiagnosticQuery {
        max_events: Some(1),
        max_components: Some(0),
        ..Default::default()
    };
    let first = d.reader().inspect_memory(&q).unwrap();
    let data = first.to_memory_page().unwrap();
    assert!(
        !std::str::from_utf8(first.as_bytes())
            .unwrap()
            .contains("secret.marker")
    );
    assert!(
        !std::str::from_utf8(first.as_bytes())
            .unwrap()
            .contains("session")
    );
    let snapshot = d.snapshot(&DiagnosticQuery::default()).unwrap();
    assert_eq!(
        d.reader().inspect_memory(&q).unwrap_err(),
        DiagnosticError::QueryCapacity
    );
    drop(snapshot);
    let resume = DiagnosticQuery {
        continuation: data.continuation.clone(),
        ..q.clone()
    };
    assert_eq!(
        d.snapshot(&resume).unwrap_err(),
        DiagnosticError::InvalidCursor
    );
    let next = page(&d, &resume);
    assert!(next.entries[0].sequence > data.entries[0].sequence);
    drop(first);
    let after = d
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(before.sequence, after.sequence);
    assert_eq!(before.lifetime.totals, after.lifetime.totals);
    record(&a, "later");
    assert_eq!(
        d.reader().inspect_memory(&resume).unwrap_err(),
        DiagnosticError::InvalidCursor
    );
    assert_eq!(d.memory_usage().query_workspaces, 0);
    assert_eq!(
        d.reader()
            .inspect_memory(&DiagnosticQuery {
                max_response_bytes: Some(1),
                ..Default::default()
            })
            .unwrap_err(),
        DiagnosticError::SnapshotCapacity
    );
    assert_eq!(d.memory_usage().query_workspaces, 0);
}
#[test]
fn memory_cursor_rejects_foreign_sources_changed_selections_and_invalid_offsets() {
    let d = context(8);
    let recorder = d
        .register_component(ComponentRegistration::new("a", &[CODE]))
        .unwrap();
    record(&recorder, "one");
    record(&recorder, "two");
    let query = DiagnosticQuery {
        max_components: Some(0),
        max_events: Some(1),
        ..Default::default()
    };
    let first = page(&d, &query);
    let cursor = first.continuation.unwrap();
    let foreign = page(&context(0), &DiagnosticQuery::default())
        .source
        .instance_id;
    for mutation in 0..6 {
        let mut next = DiagnosticQuery {
            continuation: Some(cursor.clone()),
            ..query.clone()
        };
        let token = next.continuation.as_mut().unwrap();
        match mutation {
            0 => token.source_instance_id = foreign,
            1 => token.sequence = (token.sequence.get() + 1).into(),
            2 => token.scope = (token.scope.get() ^ 1).into(),
            3 => token.component_offset = u64::MAX.into(),
            4 => token.event_offset = u64::MAX.into(),
            5 => next.component_key = Some("other".into()),
            _ => unreachable!(),
        }
        assert_eq!(
            d.reader().inspect_memory(&next).unwrap_err(),
            DiagnosticError::InvalidCursor,
            "case {mutation}"
        );
        assert_eq!(d.memory_usage().query_workspaces, 0);
    }
    let resumed = page(
        &d,
        &DiagnosticQuery {
            continuation: Some(cursor),
            ..query
        },
    );
    assert_eq!(resumed.entries.len(), 1);
    assert!(resumed.entries[0].sequence > first.entries[0].sequence);
    assert!(resumed.continuation.is_none());
}

#[test]
fn memory_component_caps_advance_and_unfittable_identity_releases_the_permit() {
    let d = context(0);
    let _a = d
        .register_component(ComponentRegistration::new("a", &[CODE]))
        .unwrap();
    let _b = d
        .register_component(ComponentRegistration::new("b", &[CODE]))
        .unwrap();
    let query = DiagnosticQuery {
        max_components: Some(1),
        max_events: Some(0),
        ..Default::default()
    };
    let first = page(&d, &query);
    let second = page(
        &d,
        &DiagnosticQuery {
            continuation: first.continuation.clone(),
            ..query
        },
    );
    assert_eq!(first.components.len(), 1);
    assert_eq!(second.components.len(), 1);
    assert_ne!(
        first.components[0].component.instance_id,
        second.components[0].component.instance_id
    );
    assert!(second.continuation.is_none());
    assert_eq!(first.selection, second.selection);
    let d = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            limits: DiagnosticLimits {
                query_text_bytes: 1,
                max_queries: 1,
                ..Default::default()
            },
        },
    )
    .unwrap();
    let recorder = d
        .register_component(ComponentRegistration::new("long-key", &[CODE]))
        .unwrap();
    record(&recorder, "retained");
    for query in [
        DiagnosticQuery {
            max_events: Some(0),
            ..Default::default()
        },
        DiagnosticQuery {
            max_components: Some(0),
            ..Default::default()
        },
    ] {
        assert_eq!(
            d.reader().inspect_memory(&query).unwrap_err(),
            DiagnosticError::SnapshotCapacity
        );
        assert_eq!(d.memory_usage().query_workspaces, 0);
    }
    let empty = page(
        &d,
        &DiagnosticQuery {
            max_components: Some(0),
            max_events: Some(0),
            ..Default::default()
        },
    );
    assert!(empty.components.is_empty() && empty.entries.is_empty());
    assert!(empty.continuation.is_none());
    assert_eq!(empty.aggregate.events, 1);
    conserved(&empty);
}

#[test]
fn copied_typed_views_and_json_access_allocate_nothing_and_survive_eviction() {
    let d = context(2);
    let a = d
        .register_component(ComponentRegistration::new("a", &[CODE]))
        .unwrap();
    record(&a, "one");
    // Warm first-use runtime helpers before measuring steady-state capture.
    let _ = d
        .reader()
        .inspect_memory(&DiagnosticQuery::default())
        .unwrap();
    let (lease, stats) = allocation_stats_during(|| {
        d.reader()
            .inspect_memory(&DiagnosticQuery::default())
            .unwrap()
    });
    assert_eq!(stats.allocation_calls, 0, "{stats:?}");
    let (_, stats) = allocation_stats_during(|| {
        assert_eq!(lease.components().next().unwrap().key(), "a");
        assert_eq!(lease.entries().len(), 1);
        assert!(!lease.as_bytes().is_empty());
    });
    assert_eq!(stats.allocation_calls, 0);
    let snap = d.snapshot(&DiagnosticQuery::default()).unwrap();
    let (_, stats) = allocation_stats_during(|| {
        assert_eq!(snap.events().next().unwrap().detail(), Some("one"));
        assert_eq!(snap.components().next().unwrap().key(), "a");
    });
    assert_eq!(stats.allocation_calls, 0);
    for _ in 0..10 {
        record(&a, "new");
    }
    assert_eq!(snap.events().next().unwrap().detail(), Some("one"));
    assert_eq!(lease.entries().len(), 1);
    let (_, calls) = diagnostic_allocations::allocations_during(|| lease.as_bytes());
    assert_eq!(calls, 0);
    let (_, owned) = allocation_stats_during(|| lease.to_memory_page().unwrap());
    assert!(owned.allocation_calls > 0);
}
#[test]
fn empty_disabled_zero_history_and_reclaimed_registrations() {
    assert_eq!(
        Diagnostics::default()
            .reader()
            .inspect_memory(&DiagnosticQuery::default())
            .unwrap_err(),
        DiagnosticError::Disabled
    );
    let d = context(0);
    let a = d
        .register_component(ComponentRegistration::new("a", &[CODE]))
        .unwrap();
    record(&a, "omitted");
    let data = page(&d, &DiagnosticQuery::default());
    conserved(&data);
    assert!(data.entries.is_empty());
    assert_eq!(data.aggregate.detail_bytes, 0);
    let d = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            limits: DiagnosticLimits {
                max_components: 1,
                max_events: 4,
                ..Default::default()
            },
        },
    )
    .unwrap();
    let a = d
        .register_component(ComponentRegistration::new("a", &[CODE]))
        .unwrap();
    record(&a, "old");
    let id = a.identity().unwrap().instance_id;
    drop(a);
    let b = d
        .register_component(ComponentRegistration::new("b", &[CODE]))
        .unwrap();
    record(&b, "new");
    let data = page(&d, &DiagnosticQuery::default());
    conserved(&data);
    assert!(data.entries.iter().all(|e| e.component.instance_id != id));
}
#[test]
fn memory_page_prefix_traversal_matches_wire_under_byte_caps() {
    let d = context(20);
    let a = d
        .register_component(ComponentRegistration::new("a", &[CODE]))
        .unwrap();
    for _ in 0..20 {
        record(&a, "text");
    }
    let full = page(&d, &DiagnosticQuery::default());
    let size = d
        .reader()
        .inspect_memory(&DiagnosticQuery::default())
        .unwrap()
        .as_bytes()
        .len();
    let q = DiagnosticQuery {
        max_response_bytes: Some((size / 2) as u32),
        ..Default::default()
    };
    let lease = d.reader().inspect_memory(&q).unwrap();
    let partial = lease.to_memory_page().unwrap();
    assert_eq!(lease.entries().len(), partial.entries.len());
    assert!(partial.entries.len() < full.entries.len());
    assert!(partial.continuation.is_some());
    assert_eq!(partial.selection.events, full.selection.events);
    conserved(&partial);
}
#[test]
fn report_admitted_layout_and_scan_cost() {
    let d = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap();
    let recorders: Vec<_> = (0..d.limits().max_components)
        .map(|i| {
            let key = format!("benchmark.{i}");
            let recorder = d
                .register_component(ComponentRegistration {
                    window: Some(WindowSettings {
                        bucket_count: 60,
                        resolution_millis: 1000,
                    }),
                    ..ComponentRegistration::new(&key, &[CODE])
                })
                .unwrap();
            for name in ["first", "second"] {
                recorder
                    .register_extension(ExtensionRegistration {
                        namespace: "benchmark",
                        name,
                        description: "assigned extension storage",
                        allowed_classes: &[],
                        outcome: DiagnosticOutcome::Unknown,
                    })
                    .unwrap();
            }
            recorder
        })
        .collect();
    let message = "x".repeat(1000);
    for i in 0..d.limits().max_events {
        record(&recorders[i % recorders.len()], &message);
    }
    assert_eq!(d.memory_usage().events, d.limits().max_events);
    assert_eq!(d.memory_usage().detail_bytes, d.limits().detail_pool_bytes);
    let start = std::time::Instant::now();
    for _ in 0..100 {
        std::hint::black_box(
            d.reader()
                .inspect_memory(&DiagnosticQuery::default())
                .unwrap(),
        );
    }
    eprintln!(
        "memory_inspection: default_capacity={} reserved={} event_slot={} mean_capture_us={}",
        d.limits().max_events,
        d.memory_usage().reserved_bytes,
        d.memory_usage().event_slot_bytes,
        start.elapsed().as_micros() / 100
    );
}

#[test]
fn concurrent_recording_remains_bounded_during_memory_scans() {
    let d = context(64);
    let recorder = d
        .register_component(ComponentRegistration::new("concurrent", &[CODE]))
        .unwrap();
    let start = std::sync::Arc::new(std::sync::Barrier::new(2));
    let gate = start.clone();
    let writer = std::thread::spawn(move || {
        gate.wait();
        for _ in 0..2000 {
            recorder
                .record_without_projection_lazy(|| {
                    Observation::new(
                        Classification::success(SuccessCode::SinkInitialized),
                        "fixed",
                        NonZeroU64::MIN,
                    )
                })
                .unwrap();
        }
    });
    start.wait();
    let mut captures = 0;
    let begin = std::time::Instant::now();
    for _ in 0..100 {
        let lease = d
            .reader()
            .inspect_memory(&DiagnosticQuery::default())
            .unwrap();
        assert_eq!(
            lease.component_occupied_bytes().get()
                + lease.shared_or_unassigned_reserved_bytes().get(),
            lease.aggregate().reserved_bytes as u64
        );
        captures += 1;
    }
    writer.join().unwrap();
    let snapshot = d
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(
        snapshot.lifetime.totals.recorded_events_total.value.get(),
        2000
    );
    assert!(snapshot.events.len() <= 64);
    eprintln!(
        "concurrent_memory_scans: captures={captures} elapsed_us={}",
        begin.elapsed().as_micros()
    );
}

#[test]
fn shared_text_budget_can_paginate_entries_and_components_independently() {
    let d = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            limits: DiagnosticLimits {
                query_text_bytes: 3,
                ..Default::default()
            },
        },
    )
    .unwrap();
    let recorder = d
        .register_component(ComponentRegistration::new("abc", &[CODE]))
        .unwrap();
    record(&recorder, "first");
    record(&recorder, "second");
    let mut query = DiagnosticQuery::default();
    let (mut components, mut events) = (0, 0);
    for index in 0..3 {
        let data = page(&d, &query);
        conserved(&data);
        components += data.components.len();
        events += data.entries.len();
        query.continuation = data.continuation;
        assert_eq!(query.continuation.is_some(), index < 2);
    }
    assert_eq!((components, events), (1, 2));
    assert_eq!(d.memory_usage().query_workspaces, 0);
}
