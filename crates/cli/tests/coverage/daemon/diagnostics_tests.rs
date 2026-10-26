// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn daemon_producers_keep_counts_without_projecting_restricted_details() {
    let context = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            limits: DiagnosticLimits {
                max_events: 1,
                ..Default::default()
            },
        },
    )
    .unwrap();
    let owner = DaemonDiagnostics::new(&context, "daemon_broker").unwrap();
    let _projection = context
        .subscribe_projection(|_| panic!("restricted daemon detail projected"))
        .unwrap();
    for _ in 0..5 {
        owner.record(
            DaemonFailureCode::Forward,
            FailureClass::Timeout,
            "PRIVATE_ENDPOINT",
        );
    }
    let snapshot = context
        .snapshot(&DiagnosticQuery {
            component_key: Some("daemon.broker.forwarding".into()),
            ..Default::default()
        })
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(snapshot.components.len(), 1);
    assert_eq!(
        snapshot.components[0]
            .lifetime
            .totals
            .failures_total
            .value
            .get(),
        5
    );
    assert_eq!(snapshot.events.len(), 1);
    assert_eq!(owner.coverage().count(), 3);
    assert_eq!(snapshot.source, *context.source());
}
#[test]
fn disabled_daemon_recorders_do_not_allocate_retention_or_publish_coverage() {
    let context = Diagnostics::default();
    let owner = DaemonDiagnostics::new(&context, "daemon_worker").unwrap();
    owner.record(
        DaemonFailureCode::WorkerInitialize,
        FailureClass::Unknown,
        "ignored",
    );
    assert_eq!(owner.coverage().count(), 0);
    assert!(matches!(
        context.snapshot(&Default::default()),
        Err(DiagnosticError::Disabled)
    ));
}
#[test]
fn daemon_classification_never_guesses_from_error_text() {
    let classifier = DaemonErrorClassifier(DaemonFailureCode::WorkerInitialize);
    assert_eq!(
        classifier
            .classify(&CliError::Launch("timeout: permission denied".into()))
            .failure_class(),
        Some("unknown")
    );
    assert_eq!(
        classifier
            .classify(&CliError::Io(std::io::ErrorKind::TimedOut.into()))
            .failure_class(),
        Some("io")
    );
}
#[test]
fn collector_budget_is_separate_and_admitted_before_startup() {
    let mut config = CollectorConfig {
        collect_workers: true,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
    config.max_owned_bytes = super::super::broker::collection::owned_bytes() - 1;
    assert!(config.validate().is_err());
    config.collect_workers = false;
    assert!(config.validate().is_ok());
}

#[test]
fn diagnostic_grants_are_expiring_source_generation_and_policy_bound() {
    let generation = uuid::Uuid::now_v7().to_string();
    let grant = DiagnosticReadGrant::issue(uuid::Uuid::now_v7(), &generation, 8192).unwrap();
    let token = grant.token(1000);
    assert!(grant.verify(&token, 1000));
    assert!(!grant.verify(&token, 61_000));
    assert!(!grant.verify(&grant.token(121_001), 1000));
    assert!(!grant.verify("garbage", 1000));
    for changed in [0, 1, 2, 3] {
        let mut other = grant.clone();
        match changed {
            0 => other.source = uuid::Uuid::now_v7(),
            1 => other.generation = uuid::Uuid::now_v7().to_string(),
            2 => other.policy_revision = 2,
            _ => other.max_page_bytes = 16384,
        }
        assert!(!other.verify(&token, 1000));
    }
    assert!(!format!("{grant:?}").contains(grant.key.expose()));
    let authority = WorkerReadAuthority::default();
    let current = grant.token(now_unix_ms());
    authority.replace(Some(grant.clone()));
    assert!(authority.verify(&current));
    authority.replace(None);
    assert!(!authority.verify(&current));
    authority.replace(DiagnosticReadGrant::issue(grant.source, &generation, 8192));
    assert!(!authority.verify(&current));
}

#[test]
fn diagnostic_grants_reject_each_offer_mismatch_and_malformed_token() {
    let source = uuid::Uuid::now_v7();
    let generation = uuid::Uuid::now_v7().to_string();
    let grant = DiagnosticReadGrant::issue(source, &generation, 8192).unwrap();
    let offer = DiagnosticOffer {
        source,
        schema_version: DIAGNOSTIC_SCHEMA_VERSION,
        enabled: true,
        query_available: true,
        whole_context_read: true,
        max_page_bytes: 8192,
    };
    assert!(grant.matches(&offer, &generation));
    for field in 0..8 {
        let mut changed = offer.clone();
        let mut candidate = grant.clone();
        match field {
            0 => changed.source = uuid::Uuid::now_v7(),
            1 => candidate.generation.push('x'),
            2 => candidate.policy_revision += 1,
            3 => candidate.key = SensitiveString::new("short").unwrap(),
            4 => candidate.max_page_bytes = 0,
            5 => changed.max_page_bytes = 8191,
            6 => changed.whole_context_read = false,
            _ => changed.query_available = false,
        }
        assert!(!candidate.matches(&changed, &generation), "field {field}");
    }
    let mut disabled = offer;
    disabled.enabled = false;
    assert!(!grant.matches(&disabled, &generation));
    for token in ["garbage", "not-a-number.tag", "2000.!", "2000.YQ"] {
        assert!(!grant.verify(token, 1000));
    }
    assert!(DiagnosticReadGrant::issue(source, "", 8192).is_none());
    assert!(DiagnosticReadGrant::issue(source, &"x".repeat(257), 8192).is_none());
}
