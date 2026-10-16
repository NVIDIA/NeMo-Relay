// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resource routing boundaries shared by OTLP logs and metrics.

use super::*;
use crate::api::event::{BaseEvent, MarkEvent, ScopeEvent};
use crate::api::scope::ScopeType;

#[test]
fn signal_headers_reject_invalid_http_names_values_and_case_collisions() {
    for (key, value) in [("x invalid", "valid"), ("x-valid", "line\nbreak")] {
        let error =
            validate_signal_headers(&HashMap::from([(key.into(), value.into())])).unwrap_err();
        assert!(matches!(error, OpenTelemetryError::InvalidHeader { .. }));
        assert!(error.to_string().contains(key));
    }
    assert!(
        validate_signal_headers(&HashMap::from([
            ("X-Token".into(), "one".into()),
            ("x-token".into(), "two".into()),
        ]))
        .unwrap_err()
        .to_string()
        .contains("unique ignoring ASCII case")
    );
}

#[test]
fn signal_header_environment_rejects_collisions_and_invalid_variable_references() {
    let headers = HashMap::from([("Authorization".into(), "inline-token".into())]);
    let header_env = HashMap::from([("authorization".into(), "IGNORED_VARIABLE".into())]);
    assert!(
        resolve_header_env(&headers, &header_env)
            .unwrap_err()
            .to_string()
            .contains("unique across headers and header_env")
    );
    for variable in ["", " PADDED ", "WITH=EQUALS", "WITH\0NUL"] {
        let header_env = HashMap::from([("x-token".into(), variable.into())]);
        assert!(
            resolve_header_env(&HashMap::new(), &header_env)
                .unwrap_err()
                .to_string()
                .contains("header_env must name")
        );
    }
    let header_env = HashMap::from([("x invalid".into(), "IGNORED_VARIABLE".into())]);
    assert!(matches!(
        resolve_header_env(&HashMap::new(), &header_env).unwrap_err(),
        OpenTelemetryError::InvalidHeader { .. }
    ));
}

#[test]
fn grpc_metadata_rejects_invalid_names_and_control_characters() {
    for (key, value) in [("invalid key", "token"), ("authorization", "line\nbreak")] {
        assert!(matches!(
            build_grpc_metadata(&HashMap::from([(key.into(), value.into())])).unwrap_err(),
            OpenTelemetryError::InvalidGrpcHeader { .. }
        ));
    }
    let metadata = build_grpc_metadata(&HashMap::from([(
        "authorization".into(),
        "Bearer token".into(),
    )]))
    .unwrap();
    assert_eq!(metadata.get("authorization").unwrap(), "Bearer token");
}

#[test]
fn runtime_diagnostics_truncate_unicode_without_losing_occurrence_counts() {
    let diagnostics = SignalRuntimeDiagnostics::new(None);
    let message = "é".repeat(MAX_RUNTIME_DIAGNOSTIC_MESSAGE_CHARS + 1);
    assert_eq!(diagnostics.record("bounded", message.clone(), 2), 2);
    assert_eq!(diagnostics.record("bounded", message, 3), 5);
    let snapshot = diagnostics.snapshot();
    let entry = snapshot.get("bounded").unwrap();
    assert_eq!(entry.count, 5);
    assert_eq!(
        entry.message.chars().count(),
        MAX_RUNTIME_DIAGNOSTIC_MESSAGE_CHARS
    );
    assert!(entry.message.ends_with('…'));
}

#[test]
fn runtime_diagnostics_keep_success_separate_from_legacy_failure_reports() {
    let service = Diagnostics::default();
    let diagnostics = SignalRuntimeDiagnostics {
        state: Arc::new(Mutex::new(RuntimeDiagnosticState::default())),
        plugin_field: None,
        recorder: Some(service.register_component("observability.test").unwrap()),
    };
    diagnostics.record("otel.logs_export_failed", "export failed".into(), 2);
    diagnostics.record("otel.logs_export_failed", "export failed again".into(), 3);

    let recorder = diagnostics.recorder.as_ref().unwrap();
    let before = recorder.snapshot().unwrap();
    let failure = before
        .counters
        .iter()
        .find(|counter| counter.code == "otel.logs_export_failed")
        .unwrap();
    assert_eq!(failure.occurrences.value.get(), 5);
    let legacy_before = diagnostics.snapshot();
    for _ in 0..3 {
        assert_eq!(diagnostics.snapshot(), legacy_before);
        assert_eq!(recorder.snapshot().unwrap(), before);
    }

    diagnostics.record_success(
        "otel.logs_export_succeeded",
        "OpenTelemetry log export succeeded",
    );
    let after = recorder.snapshot().unwrap();
    assert_eq!(
        after
            .counters
            .iter()
            .find(|counter| counter.code == "otel.logs_export_failed")
            .unwrap(),
        failure
    );
    let success = after
        .counters
        .iter()
        .find(|counter| counter.code == "otel.logs_export_succeeded")
        .unwrap();
    assert_eq!(success.occurrences.value.get(), 1);
    assert_eq!(success.last_message, "OpenTelemetry log export succeeded");
    assert_eq!(diagnostics.snapshot(), legacy_before);
    assert!(
        diagnostics
            .snapshot()
            .get("otel.logs_export_succeeded")
            .is_none()
    );

    let evidence = service
        .snapshot(crate::diagnostics::DiagnosticQuery::default())
        .unwrap();
    assert_eq!(evidence.events.len(), 3);
    assert!(
        evidence.events[..2]
            .iter()
            .all(|event| { event.outcome == crate::diagnostics::DiagnosticOutcome::Failure })
    );
    let success_event = evidence.events.last().unwrap();
    assert_eq!(success_event.code, "otel.logs_export_succeeded");
    assert_eq!(
        success_event.outcome,
        crate::diagnostics::DiagnosticOutcome::Success
    );
    assert_eq!(success_event.count.get(), 1);
}

#[test]
fn legacy_metadata_codes_are_normalized_without_changing_subscriber_reports() {
    let diagnostics = SignalRuntimeDiagnostics::new(None);
    let code = "otel.resource_metadata_invalid.Application/Key";
    diagnostics.record(code, "A selected metadata value was rejected".into(), 1);
    diagnostics.record(code, "A selected metadata value was rejected".into(), 2);

    assert_eq!(diagnostics.snapshot().get(code).unwrap().count, 3);
    let snapshot = diagnostics.recorder.as_ref().unwrap().snapshot().unwrap();
    assert_eq!(snapshot.counters.len(), 1);
    let normalized = &snapshot.counters[0].code;
    assert!(normalized.starts_with("legacy.code."));
    assert!(!normalized.contains("Application"));
    assert!(crate::diagnostics::DiagnosticCode::new(normalized).is_ok());
    assert_eq!(snapshot.counters[0].occurrences.value.get(), 3);
}

#[test]
fn exporter_runtime_reports_a_panicking_constructor_without_hanging() {
    let error = build_in_owned_runtime::<(), _>("coverage-panicking-exporter", || {
        panic!("constructor failed");
    })
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("exporter runtime stopped unexpectedly")
    );
}

fn scope(
    id: Uuid,
    parent: Option<Uuid>,
    category: ScopeCategory,
    timestamp: DateTime<Utc>,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(id)
            .parent_uuid_opt(parent)
            .name("scope")
            .timestamp(timestamp)
            .build(),
        category,
        Vec::new(),
        ScopeType::Agent.into(),
        None,
    ))
}

fn mark(parent: Uuid, timestamp: DateTime<Utc>) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(parent)
            .name("late")
            .timestamp(timestamp)
            .build(),
        None,
        None,
    ))
}

#[test]
fn completed_resource_routes_expire_before_lookup_for_marks_and_child_starts() {
    let ttl = Duration::from_secs(60);
    let now = Utc::now();
    let diagnostics = SignalRuntimeDiagnostics::new(None);
    for child_start in [false, true] {
        let mut lineage = SignalResourceLineage::new();
        let root = Uuid::now_v7();
        lineage.process(
            &scope(root, None, ScopeCategory::Start, now),
            ttl,
            &diagnostics,
            || Some("root"),
        );
        assert_eq!(
            lineage.process(
                &scope(root, None, ScopeCategory::End, now),
                ttl,
                &diagnostics,
                || None
            ),
            Some("root")
        );
        assert_eq!(
            lineage.process(
                &mark(root, now + chrono::Duration::seconds(60)),
                ttl,
                &diagnostics,
                || None
            ),
            Some("root")
        );
        let after_ttl = now + chrono::Duration::seconds(61);
        let event = if child_start {
            scope(Uuid::now_v7(), Some(root), ScopeCategory::Start, after_ttl)
        } else {
            mark(root, after_ttl)
        };
        assert_eq!(lineage.process(&event, ttl, &diagnostics, || None), None);
        assert!(lineage.completed.is_empty());
        assert!(lineage.completed_expiry_index.is_empty());
    }
}

#[test]
fn active_resource_scope_capacity_preserves_existing_scopes_and_recovers_on_end() {
    let mut lineage = SignalResourceLineage::new();
    let diagnostics = SignalRuntimeDiagnostics::new(None);
    let ttl = Duration::from_secs(60);
    let now = Utc::now();
    let mut ids = Vec::new();
    for _ in 0..MAX_ACTIVE_RESOURCE_SCOPES {
        let id = Uuid::now_v7();
        assert_eq!(
            lineage.process(
                &scope(id, None, ScopeCategory::Start, now),
                ttl,
                &diagnostics,
                || Some("root")
            ),
            Some("root")
        );
        ids.push(id);
    }
    let much_later = now + chrono::Duration::days(1);
    // Repeated starts preserve the original identity even at capacity.
    assert_eq!(
        lineage.process(
            &scope(ids[0], None, ScopeCategory::Start, much_later),
            ttl,
            &diagnostics,
            || panic!("must reuse tracked route")
        ),
        Some("root")
    );
    let rejected = Uuid::now_v7();
    for id in [rejected, Uuid::now_v7(), Uuid::now_v7()] {
        assert_eq!(
            lineage.process(
                &scope(id, Some(ids[0]), ScopeCategory::Start, much_later),
                ttl,
                &diagnostics,
                || panic!("must not create a provider at capacity")
            ),
            None
        );
    }
    assert_eq!(lineage.active.len(), MAX_ACTIVE_RESOURCE_SCOPES);
    assert_eq!(
        diagnostics
            .snapshot()
            .get("otel.resource_metadata_active_scope_limit")
            .unwrap()
            .count,
        3
    );
    assert_eq!(
        lineage.process(&mark(ids[0], much_later), ttl, &diagnostics, || None),
        Some("root")
    );
    assert_eq!(
        lineage.process(
            &scope(ids[0], None, ScopeCategory::End, much_later),
            ttl,
            &diagnostics,
            || None
        ),
        Some("root")
    );
    assert_eq!(lineage.active.len(), MAX_ACTIVE_RESOURCE_SCOPES - 1);
    // An untracked scope stays on the base resource even after capacity returns.
    assert_eq!(
        lineage.process(&mark(rejected, much_later), ttl, &diagnostics, || None),
        None
    );
    assert_eq!(
        lineage.process(
            &scope(
                Uuid::now_v7(),
                Some(ids[1]),
                ScopeCategory::Start,
                much_later
            ),
            ttl,
            &diagnostics,
            || panic!("must inherit parent")
        ),
        Some("root")
    );
    assert_eq!(lineage.active.len(), MAX_ACTIVE_RESOURCE_SCOPES);
}
