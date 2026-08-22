// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use nemo_relay_worker::{ConfigDiagnostic, DiagnosticLevel, Json, RuntimeRegistrationKind};
use serde::{Deserialize, Serialize};
use serde_json::Map;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub(crate) struct ExampleConfig {
    pub tag: String,
    pub observe: ObserveConfig,
    pub requests: RequestsConfig,
    pub execution: ExecutionConfig,
    pub runtime: RuntimeConfig,
    pub registration_control: RegistrationControlConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub(crate) struct ObserveConfig {
    pub enabled: bool,
    pub redact_keys: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub(crate) struct RequestsConfig {
    pub enabled: bool,
    pub mode: String,
    pub blocked_tools: Vec<String>,
    pub blocked_models: Vec<String>,
    pub header_name: String,
    pub header_value: String,
    pub priority: i32,
    pub break_chain: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub(crate) struct ExecutionConfig {
    pub enabled: bool,
    pub priority: i32,
    pub emit_pending_marks: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub(crate) struct RuntimeConfig {
    pub emit_marks: bool,
    pub emit_isolated_scope: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub(crate) struct RegistrationControlConfig {
    pub enabled: bool,
    pub kinds: Vec<RuntimeRegistrationKind>,
    pub registration_name: String,
    pub reason: String,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        Self {
            tag: "documentation".into(),
            observe: ObserveConfig::default(),
            requests: RequestsConfig::default(),
            execution: ExecutionConfig::default(),
            runtime: RuntimeConfig::default(),
            registration_control: RegistrationControlConfig::default(),
        }
    }
}

impl Default for ObserveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            redact_keys: vec!["secret".into()],
        }
    }
}

impl Default for RequestsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: "enforce".into(),
            blocked_tools: vec!["dangerous_tool".into()],
            blocked_models: vec!["restricted-model".into()],
            header_name: "x-nemo-relay-plugin".into(),
            header_value: "documentation".into(),
            priority: 20,
            break_chain: false,
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            priority: 30,
            emit_pending_marks: true,
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            emit_marks: true,
            emit_isolated_scope: true,
        }
    }
}

impl Default for RegistrationControlConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            kinds: vec![RuntimeRegistrationKind::Subscriber],
            registration_name: "documentation-controlled-subscriber".into(),
            reason: "disabled by documentation plugin".into(),
        }
    }
}

impl ExampleConfig {
    pub(crate) fn parse(config: &Json) -> Result<Self, String> {
        serde_json::from_value(config.clone()).map_err(|error| error.to_string())
    }
}

pub(crate) fn validate(config: &Json) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();
    validate_unknown_fields(config, &mut diagnostics);
    let parsed = match ExampleConfig::parse(config) {
        Ok(parsed) => parsed,
        Err(error) => {
            diagnostics.push(diagnostic(
                DiagnosticLevel::Error,
                "invalid_config",
                None,
                error,
            ));
            return diagnostics;
        }
    };
    if parsed.tag.is_empty() {
        diagnostics.push(diagnostic(
            DiagnosticLevel::Error,
            "empty_tag",
            Some("tag"),
            "tag must not be empty",
        ));
    }
    if parsed.requests.mode != "observe" && parsed.requests.mode != "enforce" {
        diagnostics.push(diagnostic(
            DiagnosticLevel::Error,
            "unsupported_mode",
            Some("requests.mode"),
            "requests.mode must be either observe or enforce",
        ));
    }
    if parsed.requests.header_name.is_empty() {
        diagnostics.push(diagnostic(
            DiagnosticLevel::Error,
            "invalid_header",
            Some("requests.header_name"),
            "requests.header_name must not be empty",
        ));
    }
    if parsed.requests.header_value.is_empty() {
        diagnostics.push(diagnostic(
            DiagnosticLevel::Error,
            "invalid_header",
            Some("requests.header_value"),
            "requests.header_value must not be empty",
        ));
    }
    if parsed.registration_control.kinds.is_empty() {
        diagnostics.push(diagnostic(
            DiagnosticLevel::Error,
            "invalid_registration_control",
            Some("registration_control.kinds"),
            "registration_control.kinds must not be empty",
        ));
    }
    for (field, value) in [
        (
            "registration_control.registration_name",
            &parsed.registration_control.registration_name,
        ),
        ("registration_control.reason", &parsed.registration_control.reason),
    ] {
        if value.is_empty() {
            diagnostics.push(diagnostic(
                DiagnosticLevel::Error,
                "invalid_registration_control",
                Some(field),
                format!("{field} must not be empty"),
            ));
        }
    }
    diagnostics
}

fn validate_unknown_fields(config: &Json, diagnostics: &mut Vec<ConfigDiagnostic>) {
    let Some(config) = config.as_object() else {
        return;
    };
    const TOP_LEVEL: &[&str] = &[
        "tag",
        "observe",
        "requests",
        "execution",
        "runtime",
        "registration_control",
    ];
    const OBSERVE: &[&str] = &["enabled", "redact_keys"];
    const REQUESTS: &[&str] = &[
        "enabled",
        "mode",
        "blocked_tools",
        "blocked_models",
        "header_name",
        "header_value",
        "priority",
        "break_chain",
    ];
    const EXECUTION: &[&str] = &["enabled", "priority", "emit_pending_marks"];
    const RUNTIME: &[&str] = &["emit_marks", "emit_isolated_scope"];
    const REGISTRATION_CONTROL: &[&str] = &["enabled", "kinds", "registration_name", "reason"];

    report_unknown(config, "", TOP_LEVEL, diagnostics);
    for (field, allowed) in [
        ("observe", OBSERVE),
        ("requests", REQUESTS),
        ("execution", EXECUTION),
        ("runtime", RUNTIME),
        ("registration_control", REGISTRATION_CONTROL),
    ] {
        if let Some(object) = config.get(field).and_then(Json::as_object) {
            report_unknown(object, field, allowed, diagnostics);
        }
    }
}

fn report_unknown(
    object: &Map<String, Json>,
    prefix: &str,
    allowed: &[&str],
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let allowed = allowed.iter().copied().collect::<HashSet<_>>();
    for key in object.keys().filter(|key| !allowed.contains(key.as_str())) {
        let field = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        diagnostics.push(diagnostic(
            DiagnosticLevel::Warning,
            "unknown_field",
            Some(&field),
            format!("unknown config field '{field}' is not supported"),
        ));
    }
}

fn diagnostic(
    level: DiagnosticLevel,
    suffix: &str,
    field: Option<&str>,
    message: impl Into<String>,
) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level,
        code: format!("examples.rust_grpc_worker.{suffix}"),
        component: Some("examples.rust_grpc_worker".into()),
        field: field.map(str::to_owned),
        message: message.into(),
    }
}
