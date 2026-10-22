// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_types::diagnostics::{
    BuiltinDiagnosticCode, CommonPluginFailureCode, FailureCode, PluginFailureCode,
};
use nemo_relay_worker::diagnostics::{
    DiagnosticObservation, DiagnosticObservationOutcome, DiagnosticObservationSeverity,
    builtin_code,
};

let diagnostics = runtime.diagnostics().await;
if let Ok(component) = diagnostics
    .component(
        "example.internal",
        vec![builtin_code(BuiltinDiagnosticCode::Failure(
            FailureCode::Plugin(PluginFailureCode::Common(CommonPluginFailureCode::Internal)),
        ))],
    )
    .await
{
    // Evidence delivery is optional; preserve the operation's original result.
    let _ = component.try_record_lazy(|| DiagnosticObservation {
        code: 1,
        count: 1,
        outcome: DiagnosticObservationOutcome::Failure as i32,
        severity: DiagnosticObservationSeverity::Error as i32,
        failure_class: Some("internal".into()),
        detail: Some("The internal operation failed".into()),
        ..Default::default()
    });
}
