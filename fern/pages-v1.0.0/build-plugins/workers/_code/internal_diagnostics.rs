// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_worker::PluginRuntime;
use nemo_relay_worker::diagnostics::{
    DiagnosticComponent, DiagnosticDeliveryError, DiagnosticExtensionDefinition,
    DiagnosticObservation, DiagnosticObservationOutcome as Outcome,
    DiagnosticObservationSeverity as Severity, WorkerDiagnostics, extension_code,
};

/// Register the lookup vocabulary once for this activation.
pub async fn register_lookup_diagnostics(
    runtime: &PluginRuntime,
) -> Result<(WorkerDiagnostics, DiagnosticComponent), DiagnosticDeliveryError> {
    let diagnostics = runtime.diagnostics().await.clone();
    let component = diagnostics
        .component(
            "example.lookup",
            vec![
                extension_code(DiagnosticExtensionDefinition {
                    name: "lookup_failed".into(),
                    description: "The lookup backend was unavailable".into(),
                    outcome: Outcome::Failure as i32,
                    allowed_failure_classes: vec!["unavailable".into()],
                }),
                extension_code(DiagnosticExtensionDefinition {
                    name: "lookup_completed".into(),
                    description: "A lookup completed".into(),
                    outcome: Outcome::Success as i32,
                    allowed_failure_classes: vec![],
                }),
            ],
        )
        .await?;
    Ok((diagnostics, component))
}

/// Record a lookup whose backend is known to be unavailable.
pub fn record_lookup_failure(
    component: &DiagnosticComponent,
    request_id: &str,
) -> Result<(), DiagnosticDeliveryError> {
    component.try_record_lazy(|| DiagnosticObservation {
        code: 1,
        count: 1,
        outcome: Outcome::Failure as i32,
        severity: Severity::Error as i32,
        failure_class: Some("unavailable".into()),
        detail: Some("The lookup backend was unavailable".into()),
        request_id: Some(request_id.into()),
        ..Default::default()
    })
}

/// Record an independently observed successful lookup.
pub fn record_lookup_success(
    component: &DiagnosticComponent,
    request_id: &str,
) -> Result<(), DiagnosticDeliveryError> {
    component.try_record_lazy(|| DiagnosticObservation {
        code: 2,
        count: 1,
        outcome: Outcome::Success as i32,
        severity: Severity::Info as i32,
        detail: Some("A lookup completed".into()),
        request_id: Some(request_id.into()),
        ..Default::default()
    })
}
