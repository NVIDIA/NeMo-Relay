// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU64;

use nemo_relay_plugin::PluginRuntime;
use nemo_relay_plugin::diagnostics::{
    ComponentDeclaration, DiagnosticCode, DiagnosticComponent, DiagnosticError, DiagnosticOutcome,
    Diagnostics, FailureClass, Observation, RecordOutcome,
};

/// Register the lookup vocabulary once for this activation.
pub fn register_lookup_diagnostics(
    runtime: &PluginRuntime,
) -> Result<(Diagnostics, DiagnosticComponent), DiagnosticError> {
    let diagnostics = runtime.diagnostics();
    let component = diagnostics.component_lazy(|| ComponentDeclaration {
        name: "example.lookup",
        codes: &[
            DiagnosticCode::Extension {
                name: "lookup_failed",
                description: "The lookup backend was unavailable",
                outcome: DiagnosticOutcome::Failure,
                allowed_classes: &[FailureClass::Unavailable],
            },
            DiagnosticCode::Extension {
                name: "lookup_completed",
                description: "A lookup completed",
                outcome: DiagnosticOutcome::Success,
                allowed_classes: &[],
            },
        ],
    })?;
    Ok((diagnostics, component))
}

/// Record a lookup whose backend is known to be unavailable.
pub fn record_lookup_failure(
    component: &DiagnosticComponent,
    request_id: &str,
) -> Result<RecordOutcome, DiagnosticError> {
    component.try_record_lazy(|| {
        let mut observation = Observation::failure(
            1,
            FailureClass::Unavailable,
            "The lookup backend was unavailable",
            NonZeroU64::MIN,
        );
        observation.correlation.request_id = Some(request_id.into());
        observation
    })
}

/// Record an independently observed successful lookup.
pub fn record_lookup_success(
    component: &DiagnosticComponent,
    request_id: &str,
) -> Result<RecordOutcome, DiagnosticError> {
    component.try_record_lazy(|| {
        let mut observation = Observation::success(2, "A lookup completed", NonZeroU64::MIN);
        observation.correlation.request_id = Some(request_id.into());
        observation
    })
}
