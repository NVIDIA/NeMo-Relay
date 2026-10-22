// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-owned CLI composition. Operations share bounded core recorders.
use crate::{configuration::RuntimeDiagnosticsConfig, error::CliError};
use nemo_relay::diagnostics::*;
use std::{num::NonZeroU64, sync::OnceLock};

static RUNTIME: OnceLock<CliDiagnostics> = OnceLock::new();

#[derive(Clone, Copy)]
pub(crate) enum Operation {
    Command,
    Configuration,
    Gateway,
    Acquire,
    Child,
    Hook,
    Upstream,
}
impl Operation {
    fn key(self) -> &'static str {
        match self {
            Self::Command => "cli.command",
            Self::Configuration => "cli.configuration",
            Self::Gateway => "cli.gateway",
            Self::Acquire => "cli.acquire",
            Self::Child => "cli.child",
            Self::Hook => "cli.hook",
            Self::Upstream => "cli.upstream",
        }
    }
    fn index(self) -> usize {
        self as usize
    }
}
const OPERATIONS: [Operation; 7] = [
    Operation::Command,
    Operation::Configuration,
    Operation::Gateway,
    Operation::Acquire,
    Operation::Child,
    Operation::Hook,
    Operation::Upstream,
];

pub(crate) struct CliDiagnostics {
    pub context: Diagnostics,
    pub http_read: bool,
    recorders: [Option<ComponentRecorder>; 7],
}
impl CliDiagnostics {
    pub(crate) fn new(context: Diagnostics, http_read: bool) -> Self {
        let recorders = OPERATIONS.map(|op| {
            let codes = BuiltinDiagnosticCode::all()
                .iter()
                .copied()
                .filter(|code| {
                    code.domain() == DiagnosticDomain::Cli
                        && match op {
                            Operation::Command => matches!(
                                code,
                                BuiltinDiagnosticCode::Failure(FailureCode::Cli(
                                    CliFailureCode::Command
                                )) | BuiltinDiagnosticCode::Success(
                                    SuccessCode::CliCommandCompleted
                                ) | BuiltinDiagnosticCode::Notice(
                                    NoticeCode::CliNonzeroExit | NoticeCode::CliPolicyRejected
                                )
                            ),
                            Operation::Configuration => matches!(
                                code,
                                BuiltinDiagnosticCode::Failure(FailureCode::Cli(
                                    CliFailureCode::Configuration
                                ))
                            ),
                            Operation::Gateway => code.canonical_name().starts_with("cli.gateway."),
                            Operation::Acquire => code.canonical_name().starts_with("cli.acquire"),
                            Operation::Child => code.canonical_name().starts_with("cli.child."),
                            Operation::Hook => code.canonical_name().starts_with("cli.hook."),
                            Operation::Upstream => {
                                code.canonical_name().starts_with("cli.upstream.")
                            }
                        }
                })
                .collect::<Vec<_>>();
            context
                .register_component(ComponentRegistration::new(op.key(), &codes))
                .ok()
        });
        Self {
            context,
            http_read,
            recorders,
        }
    }
    pub(crate) fn record(&self, operation: Operation, classification: Classification) {
        if let Some(recorder) = &self.recorders[operation.index()] {
            // Existing operational emitters own output; this branch only retains evidence.
            let _ = recorder.record_without_projection_lazy(|| {
                Observation::new(classification, "CLI operation observed", NonZeroU64::MIN)
            });
        }
    }
    pub(crate) fn record_error(
        &self,
        operation: Operation,
        classifier: CliErrorClassifier,
        error: &CliError,
    ) {
        if let Some(recorder) = &self.recorders[operation.index()] {
            let _ = recorder.record_without_projection_lazy(|| {
                Observation::new(
                    classifier.classify(error),
                    DiagnosticDetailRef::Display(error),
                    NonZeroU64::MIN,
                )
            });
        }
    }
    pub(crate) fn coverage(&self) -> Vec<&'static str> {
        OPERATIONS
            .into_iter()
            .filter(|op| {
                self.recorders[op.index()]
                    .as_ref()
                    .is_some_and(|r| r.identity().is_some())
            })
            .map(Operation::key)
            .collect()
    }
}

pub(crate) fn install(
    config: RuntimeDiagnosticsConfig,
) -> Result<&'static CliDiagnostics, CliError> {
    config.validate()?;
    let context = Diagnostics::install_process_default(
        SourceIdentity {
            instance_id: uuid::Uuid::now_v7(),
            process_id: Some(std::process::id()),
            process_role: "cli".into(),
            relay_version: env!("CARGO_PKG_VERSION").into(),
        },
        config.settings(),
    )
    .map_err(|e| CliError::Config(e.to_string()))?;
    RUNTIME
        .set(CliDiagnostics::new(context.clone(), config.http_read))
        .map_err(|_| CliError::Config("CLI diagnostics was already selected".into()))?;
    Ok(RUNTIME.get().expect("installed CLI diagnostics"))
}
pub(crate) fn runtime() -> Option<&'static CliDiagnostics> {
    RUNTIME.get()
}
pub(crate) fn record(operation: Operation, classification: Classification) {
    if let Some(runtime) = runtime() {
        runtime.record(operation, classification);
    }
}
pub(crate) fn success(operation: Operation, code: SuccessCode) {
    record(operation, Classification::success(code));
}
pub(crate) fn failure(operation: Operation, code: CliFailureCode, class: FailureClass) {
    if let Ok(classification) = Classification::failure(FailureCode::Cli(code), class) {
        record(operation, classification);
    }
}
/// Structured producer classification; flattened configuration/launch errors stay Unknown.
pub(crate) struct CliErrorClassifier(pub CliFailureCode);
impl DiagnosticClassifier<CliError> for CliErrorClassifier {
    fn classify(&self, error: &CliError) -> Classification {
        let class = match error {
            CliError::Io(_) => FailureClass::Io,
            CliError::InvalidPayload(_) => FailureClass::InvalidInput,
            CliError::Unauthorized(_) | CliError::RouteCredentialRejected(_) => {
                FailureClass::Authentication
            }
            CliError::GuardrailRejected(_) => FailureClass::PolicyRejected,
            CliError::Upstream(error) if error.is_timeout() => FailureClass::Timeout,
            CliError::Upstream(error) if error.is_connect() => FailureClass::Unavailable,
            CliError::McpFailure { source, .. } | CliError::HookDelivery { source } => {
                return self.classify(source);
            }
            _ => FailureClass::Unknown,
        };
        Classification::failure(FailureCode::Cli(self.0), class).expect("CLI cause is admitted")
    }
}

pub(crate) fn result<T>(
    operation: Operation,
    failure_code: CliFailureCode,
    success_code: Option<SuccessCode>,
    result: &Result<T, CliError>,
) {
    match result {
        Err(error) => {
            if let Some(runtime) = runtime() {
                runtime.record_error(operation, CliErrorClassifier(failure_code), error);
            }
        }
        Ok(_) => {
            if let Some(code) = success_code {
                success(operation, code);
            }
        }
    }
}
pub(crate) fn transport_error(error: &reqwest::Error) {
    let class = if error.is_timeout() {
        FailureClass::Timeout
    } else if error.is_connect() {
        FailureClass::Unavailable
    } else {
        FailureClass::Unknown
    };
    failure(
        Operation::Upstream,
        CliFailureCode::UpstreamTransport,
        class,
    );
}

#[cfg(test)]
#[path = "../tests/coverage/shared/runtime_diagnostics_tests.rs"]
mod tests;
