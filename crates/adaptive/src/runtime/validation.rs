// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::plugin::{
    ConfigDiagnostic, ConfigPolicy, ConfigReport, DiagnosticLevel, UnsupportedBehavior,
};
use serde_json::Value as Json;

use crate::config::{AdaptiveConfig, BackendSpec, ResponseCacheConfig};
use crate::response_cache::config::KEY_STRATEGY_EXACT_REQUEST;

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

    if let Some(response_cache) = &config.response_cache {
        validate_response_cache(&mut report, response_cache);
    }

    report
}

/// Validates the adaptive plugin's `response_cache` section.
///
/// These are hard errors (not policy-driven): an invalid response-cache config
/// fails adaptive runtime construction rather than silently disabling the cache,
/// so a misconfiguration is caught at startup instead of producing surprising
/// runtime behavior.
fn validate_response_cache(report: &mut ConfigReport, config: &ResponseCacheConfig) {
    if config.ttl_seconds == 0 {
        report.diagnostics.push(response_cache_error(
            "response_cache.invalid_ttl",
            Some("ttl_seconds"),
            "ttl_seconds must be greater than 0".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&config.bypass_rate) {
        report.diagnostics.push(response_cache_error(
            "response_cache.invalid_bypass_rate",
            Some("bypass_rate"),
            "bypass_rate must be in [0.0, 1.0]".to_string(),
        ));
    }
    if config.key_strategy != KEY_STRATEGY_EXACT_REQUEST {
        report.diagnostics.push(response_cache_error(
            "response_cache.unsupported_key_strategy",
            Some("key_strategy"),
            format!("unsupported key_strategy; only \"{KEY_STRATEGY_EXACT_REQUEST}\" is supported"),
        ));
    }
    // Dropping an answer-determining field merges requests that differ there.
    const RESERVED_SKIP_KEYS: &[&str] = &[
        "messages",
        "input",
        "prompt",
        "instructions",
        "system",
        "model",
        "tools",
        "tool_choice",
    ];
    for key in &config.skip_keys {
        if RESERVED_SKIP_KEYS.contains(&key.as_str()) {
            report.diagnostics.push(response_cache_error(
                "response_cache.reserved_skip_key",
                Some("skip_keys"),
                format!("skip_keys must not drop the answer-determining field '{key}'"),
            ));
        }
    }
    // Auth material must never enter the key or the stored entries.
    const AUTH_HEADERS: &[&str] = &[
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
    ];
    for name in &config.header_allowlist {
        if AUTH_HEADERS.contains(&name.to_ascii_lowercase().as_str()) {
            report.diagnostics.push(response_cache_error(
                "response_cache.auth_header_allowlisted",
                Some("header_allowlist"),
                format!("'{name}' is an auth header and must never be folded into cache keys"),
            ));
        }
    }
    match config.backend.kind.as_str() {
        "in_memory" => {
            // Without this check, a mistyped budget silently falls back to the
            // default and `max_bytes: 0` disables storage; reject both.
            if let Some(value) = config.backend.config.get("max_bytes")
                && value.as_u64().is_none_or(|value| value == 0)
            {
                report.diagnostics.push(response_cache_error(
                    "response_cache.invalid_backend_option",
                    Some("backend.config"),
                    "max_bytes must be a positive integer".to_string(),
                ));
            }
        }
        "redis" => {
            if !cfg!(feature = "redis-backend") {
                report.diagnostics.push(response_cache_error(
                    "response_cache.backend_unavailable",
                    Some("backend.kind"),
                    "redis backend requires building with the 'redis-backend' feature".to_string(),
                ));
            } else if config
                .backend
                .config
                .get("url")
                .and_then(Json::as_str)
                .is_none_or(|url| url.trim().is_empty())
            {
                report.diagnostics.push(response_cache_error(
                    "response_cache.missing_redis_url",
                    Some("backend.config.url"),
                    "redis backend requires backend.config.url".to_string(),
                ));
            }
            // A mistyped key_prefix would silently fall back to the shared
            // default prefix; reject it like a mistyped max_bytes.
            if let Some(value) = config.backend.config.get("key_prefix")
                && !value.is_string()
            {
                report.diagnostics.push(response_cache_error(
                    "response_cache.invalid_backend_option",
                    Some("backend.config"),
                    "key_prefix must be a string".to_string(),
                ));
            }
            // A shared redis cache with no namespace mixes every caller's
            // responses together — caution against cross-tenant reuse.
            if config.namespace.is_empty() {
                report.diagnostics.push(response_cache_warning(
                    "response_cache.shared_empty_namespace",
                    Some("namespace"),
                    "redis backend with an empty namespace shares cached responses across all \
                     callers; set a namespace per environment/tenant"
                        .to_string(),
                ));
            }
        }
        other => report.diagnostics.push(response_cache_error(
            "response_cache.unknown_backend",
            Some("backend.kind"),
            format!("unknown backend kind '{other}'"),
        )),
    }
}

fn response_cache_error(code: &str, field: Option<&str>, message: String) -> ConfigDiagnostic {
    response_cache_diag(DiagnosticLevel::Error, code, field, message)
}

fn response_cache_warning(code: &str, field: Option<&str>, message: String) -> ConfigDiagnostic {
    response_cache_diag(DiagnosticLevel::Warning, code, field, message)
}

fn response_cache_diag(
    level: DiagnosticLevel,
    code: &str,
    field: Option<&str>,
    message: String,
) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level,
        code: code.to_string(),
        component: Some("response_cache".to_string()),
        field: field.map(str::to_string),
        message,
    }
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
