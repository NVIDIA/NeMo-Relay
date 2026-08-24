// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use nemo_relay::api::registry::RuntimeRegistrationKind;

use nemo_relay::plugin::{ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, UnsupportedBehavior};
use serde::Deserialize;
use serde_json::{Map, Value as Json};

#[derive(Clone, Deserialize)]
#[serde(default)]
pub(crate) struct Settings {
    pub tag: String,
    pub observe: Observe,
    pub requests: Requests,
    pub execution: Execution,
    pub runtime: Runtime,
    pub registration_control: RegistrationControl,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub(crate) struct Observe {
    pub enabled: bool,
    pub redact_keys: Vec<String>,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub(crate) struct Requests {
    pub enabled: bool,
    pub mode: String,
    pub blocked_tools: Vec<String>,
    pub blocked_models: Vec<String>,
    pub header_name: String,
    pub header_value: String,
    pub priority: i32,
    pub break_chain: bool,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub(crate) struct Execution {
    pub enabled: bool,
    pub priority: i32,
    pub emit_pending_marks: bool,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub(crate) struct Runtime {
    pub emit_marks: bool,
    pub emit_isolated_scope: bool,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub(crate) struct RegistrationControl {
    pub enabled: bool,
    pub kinds: Vec<RuntimeRegistrationKind>,
    pub registration_name: String,
    pub reason: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            tag: "documentation".into(),
            observe: Observe::default(),
            requests: Requests::default(),
            execution: Execution::default(),
            runtime: Runtime::default(),
            registration_control: RegistrationControl::default(),
        }
    }
}

impl Default for Observe {
    fn default() -> Self {
        Self {
            enabled: true,
            redact_keys: vec!["secret".into()],
        }
    }
}

impl Default for Requests {
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

impl Default for Execution {
    fn default() -> Self {
        Self {
            enabled: true,
            priority: 30,
            emit_pending_marks: true,
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self {
            emit_marks: true,
            emit_isolated_scope: true,
        }
    }
}

impl Default for RegistrationControl {
    fn default() -> Self {
        Self {
            enabled: false,
            kinds: vec![RuntimeRegistrationKind::Subscriber],
            registration_name: "documentation-controlled-subscriber".into(),
            reason: "disabled by documentation plugin".into(),
        }
    }
}

pub(crate) fn parse(config: &Map<String, Json>) -> Result<Settings, String> {
    serde_json::from_value(Json::Object(config.clone())).map_err(|error| error.to_string())
}

pub(crate) fn validate(config: &Map<String, Json>, policy: &ConfigPolicy) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();
    report_unknown_fields(config, policy.unknown_field, &mut diagnostics);
    let settings = match parse(config) {
        Ok(settings) => settings,
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
    if settings.tag.is_empty() {
        diagnostics.push(diagnostic(
            DiagnosticLevel::Error,
            "invalid_tag",
            Some("tag"),
            "tag must be a non-empty string",
        ));
    }
    if settings.requests.mode != "observe" && settings.requests.mode != "enforce" {
        diagnostics.push(diagnostic(
            DiagnosticLevel::Error,
            "unsupported_mode",
            Some("requests.mode"),
            "requests.mode must be either observe or enforce",
        ));
    }
    if settings.requests.header_name.is_empty() {
        diagnostics.push(diagnostic(
            DiagnosticLevel::Error,
            "invalid_header",
            Some("requests.header_name"),
            "requests.header_name must be a non-empty string",
        ));
    }
    if settings.requests.header_value.is_empty() {
        diagnostics.push(diagnostic(
            DiagnosticLevel::Error,
            "invalid_header",
            Some("requests.header_value"),
            "requests.header_value must be a non-empty string",
        ));
    }
    if settings.registration_control.kinds.is_empty() {
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
            &settings.registration_control.registration_name,
        ),
        ("registration_control.reason", &settings.registration_control.reason),
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

fn report_unknown_fields(
    config: &Map<String, Json>,
    behavior: UnsupportedBehavior,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let level = match behavior {
        UnsupportedBehavior::Ignore => return,
        UnsupportedBehavior::Warn => DiagnosticLevel::Warning,
        UnsupportedBehavior::Error => DiagnosticLevel::Error,
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
    report_unknown(config, "", TOP_LEVEL, level, diagnostics);
    for (group, allowed) in [
        ("observe", OBSERVE),
        ("requests", REQUESTS),
        ("execution", EXECUTION),
        ("runtime", RUNTIME),
        ("registration_control", REGISTRATION_CONTROL),
    ] {
        if let Some(object) = config.get(group).and_then(Json::as_object) {
            report_unknown(object, group, allowed, level, diagnostics);
        }
    }
}

fn report_unknown(
    object: &Map<String, Json>,
    prefix: &str,
    allowed: &[&str],
    level: DiagnosticLevel,
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
            level,
            "unknown_field",
            Some(&field),
            format!("unknown field '{field}' is not supported"),
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
        code: format!("documentation-plugin.{suffix}"),
        component: Some("documentation-plugin".into()),
        field: field.map(str::to_owned),
        message: message.into(),
    }
}
