// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::plugin::{
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

    if config.acg.is_some() && config.state.is_none() {
        report.diagnostics.push(ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "adaptive.section_disabled_missing_state".to_string(),
            component: Some("acg".to_string()),
            field: None,
            message: "acg requires state backend and will be disabled".to_string(),
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
    if let Some(tool_parallelism) = &config.tool_parallelism
        && let Some(drift) = &tool_parallelism.drift
    {
        validate_positive_finite(
            &mut report,
            &config.policy,
            "tool_parallelism.drift",
            "threshold",
            drift.threshold,
        );
    }

    if let Some(adaptive_hints) = &config.adaptive_hints
        && let Some(governor) = &adaptive_hints.governor
    {
        validate_positive_finite(
            &mut report,
            &config.policy,
            "adaptive_hints.governor",
            "epsilon",
            governor.epsilon,
        );
    }

    if let Some(acg) = &config.acg
        && acg.provider != "anthropic"
        && acg.provider != "openai"
        && acg.provider != "passthrough"
    {
        push_policy_diag(
            &mut report.diagnostics,
            config.policy.unsupported_value,
            "adaptive.unsupported_value",
            Some("acg".to_string()),
            Some("provider".to_string()),
            format!(
                "acg provider '{}' is unsupported; expected anthropic, openai, or passthrough",
                acg.provider
            ),
        );
    }
    if let Some(acg) = &config.acg
        && let Some(convergence) = &acg.convergence
    {
        validate_convergence(&mut report, &config.policy, "acg.convergence", convergence);
    }
    if let Some(convergence) = &config.convergence {
        validate_convergence(&mut report, &config.policy, "convergence", convergence);
    }

    report
}

fn validate_convergence(
    report: &mut ConfigReport,
    policy: &ConfigPolicy,
    component: &str,
    convergence: &crate::config::ConvergenceConfig,
) {
    validate_positive_finite(report, policy, component, "epsilon", convergence.epsilon);
    if convergence.stability_window < 3 {
        push_policy_diag(
            &mut report.diagnostics,
            policy.unsupported_value,
            "adaptive.unsupported_value",
            Some(component.to_string()),
            Some("stability_window".to_string()),
            format!(
                "{component} stability_window must be at least 3, got {}",
                convergence.stability_window
            ),
        );
    }
}

fn validate_positive_finite(
    report: &mut ConfigReport,
    policy: &ConfigPolicy,
    component: &str,
    field: &str,
    value: f64,
) {
    if value.is_finite() && value > 0.0 {
        return;
    }
    push_policy_diag(
        &mut report.diagnostics,
        policy.unsupported_value,
        "adaptive.unsupported_value",
        Some(component.to_string()),
        Some(field.to_string()),
        format!("{component} {field} must be a positive finite number, got {value}"),
    );
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
