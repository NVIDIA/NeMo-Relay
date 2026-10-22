// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
#[test]
fn flattened_errors_remain_unknown_and_typed_io_is_classified() {
    let classifier = CliErrorClassifier(CliFailureCode::Configuration);
    assert_eq!(
        classifier
            .classify(&CliError::Config("I/O: permission denied".into()))
            .failure_class(),
        Some("unknown")
    );
    assert_eq!(
        classifier
            .classify(&CliError::Io(std::io::ErrorKind::PermissionDenied.into()))
            .failure_class(),
        Some("io")
    );
}
#[test]
fn producers_keep_lifetime_deltas_after_eviction_and_do_not_project_restricted_errors() {
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
    let runtime = CliDiagnostics::new(context, false);
    let _projection = runtime
        .context
        .subscribe_projection(|_| panic!("state-only CLI evidence was projected"))
        .unwrap();
    for _ in 0..3 {
        runtime.record_error(
            Operation::Configuration,
            CliErrorClassifier(CliFailureCode::Configuration),
            &CliError::Config("PRIVATE_CONFIG_VALUE".into()),
        );
    }
    let snapshot = runtime
        .context
        .snapshot(&DiagnosticQuery {
            component_key: Some("cli.configuration".into()),
            ..Default::default()
        })
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(snapshot.events.len(), 1);
    assert!(
        snapshot.events[0]
            .message()
            .unwrap()
            .contains("PRIVATE_CONFIG_VALUE")
    );
    assert_eq!(
        snapshot.components[0]
            .lifetime
            .totals
            .failures_total
            .value
            .get(),
        3
    );
    assert_eq!(
        snapshot.components[0]
            .lifetime
            .totals
            .history_evictions_total
            .value
            .get(),
        2
    );
}
