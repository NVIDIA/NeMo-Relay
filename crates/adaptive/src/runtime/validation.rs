// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_flow::plugin::{
    ConfigDiagnostic, ConfigPolicy, ConfigReport, DiagnosticLevel, UnsupportedBehavior,
};

use crate::config::{AdaptiveConfig, BackendSpec};

pub fn validate_config(config: &AdaptiveConfig) -> ConfigReport {
    let mut report = ConfigReport::default();

    if config.version != 1 {
        push_policy_diag(
            &mut report.diagnostics,
            config.policy.unsupported_value,
            "adaptive.unsupported_config_version",
            None,
            Some("version".to_string()),
            format!("adaptive config version {} is unsupported", config.version),
        );
    }

    if let Some(state) = &config.state {
        validate_backend(&mut report, &config.policy, &state.backend);
    }

    if config.telemetry.is_some() && config.state.is_none() {
        report.diagnostics.push(ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "adaptive.section_disabled_missing_state".to_string(),
            component: Some("telemetry".to_string()),
            field: None,
            message: "telemetry requires state backend and will be disabled".to_string(),
        });
    }

    if let Some(tool_parallelism) = &config.tool_parallelism
        && tool_parallelism.mode != "observe_only"
        && tool_parallelism.mode != "inject_hints"
        && tool_parallelism.mode != "schedule"
    {
        push_policy_diag(
            &mut report.diagnostics,
            config.policy.unsupported_value,
            "adaptive.unsupported_value",
            Some("tool_parallelism".to_string()),
            Some("mode".to_string()),
            format!(
                "tool_parallelism mode '{}' is unsupported; expected observe_only, inject_hints, or schedule",
                tool_parallelism.mode
            ),
        );
    }

    report
}

fn validate_backend(report: &mut ConfigReport, policy: &ConfigPolicy, backend: &BackendSpec) {
    let kind = backend.kind.as_str();
    match kind {
        "in_memory" => {}
        "redis" => {}
        _ => {
            push_policy_diag(
                &mut report.diagnostics,
                policy.unknown_component,
                "adaptive.unknown_backend",
                Some(kind.to_string()),
                None,
                format!("backend kind '{kind}' is unsupported"),
            );
        }
    }
}

fn push_policy_diag(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    behavior: UnsupportedBehavior,
    code: &str,
    component: Option<String>,
    field: Option<String>,
    message: String,
) {
    let level = match behavior {
        UnsupportedBehavior::Ignore => return,
        UnsupportedBehavior::Warn => DiagnosticLevel::Warning,
        UnsupportedBehavior::Error => DiagnosticLevel::Error,
    };

    diagnostics.push(ConfigDiagnostic {
        level,
        code: code.to_string(),
        component,
        field,
        message,
    });
}
