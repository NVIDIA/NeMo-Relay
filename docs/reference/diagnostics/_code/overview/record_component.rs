// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_runtime::diagnostics::{
    BuiltinDiagnosticCode, Classification, ComponentRegistration, DiagnosticQuery, Diagnostics,
    DiagnosticsSettings, FailureClass, FailureCode, LoggingFailureCode, Observation,
    SourceIdentity, SuccessCode,
};
use std::num::NonZeroU64;

let diagnostics = Diagnostics::from_settings(
    SourceIdentity {
        instance_id: uuid::Uuid::now_v7(),
        process_id: Some(std::process::id()),
        process_role: "embedded".into(),
        relay_version: "1.0.0".into(),
    },
    DiagnosticsSettings {
        enabled: true,
        ..Default::default()
    },
)?;
let failure_code = FailureCode::Logging(LoggingFailureCode::SinkWriteFailed);
let success_code = SuccessCode::SinkInitialized;
let codes = [
    BuiltinDiagnosticCode::Failure(failure_code),
    BuiltinDiagnosticCode::Success(success_code),
];
let component = diagnostics
    .register_component(ComponentRegistration::new("example.logging_sink", &codes))?;
component.record_lazy(|| {
    Observation::new(
        Classification::failure(failure_code, FailureClass::Io)
            .expect("supported sink failure class"),
        "The sink write failed",
        NonZeroU64::new(2).expect("positive diagnostic occurrence count"),
    )
})?;
component.record_lazy(|| {
    Observation::new(
        Classification::success(success_code),
        "The sink initialized",
        NonZeroU64::MIN,
    )
})?;

let page = diagnostics.snapshot(&DiagnosticQuery::default())?;
let encoded: &[u8] = page.as_bytes();
assert!(!encoded.is_empty());
let snapshot = page.to_snapshot()?;
assert_eq!(snapshot.components.len(), 1);
assert_eq!(snapshot.lifetime.totals.failures_total.value.get(), 2);
drop(page); // Return the leased query workspace.
