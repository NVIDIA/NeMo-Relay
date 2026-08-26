// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use nemo_relay::plugin::{
    ConfigDiagnostic, ConfigPolicy, ConfigReport, DiagnosticLevel, UnsupportedBehavior,
};
use serde_json::Value as Json;

use crate::config::{AdaptiveConfig, BackendSpec, ResponseCacheConfig};
use crate::response_cache::config::{ResponseCacheKeyStrategy, ToolCacheConfig, ToolClass};
use crate::response_cache::tool::{is_supported_tool_pattern, wildcard_patterns_overlap};

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
    if config.namespace.trim().is_empty() {
        report.diagnostics.push(response_cache_error(
            "response_cache.missing_namespace",
            Some("namespace"),
            "namespace must identify one trusted cache-sharing domain; do not enable one \
             response cache across mutually untrusted tenants or upstreams"
                .to_string(),
        ));
    }
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
    if matches!(config.key_strategy, ResponseCacheKeyStrategy::Unknown(_)) {
        report.diagnostics.push(response_cache_error(
            "response_cache.unsupported_key_strategy",
            Some("key_strategy"),
            format!(
                "unsupported key_strategy '{}'; supported: \"exact_request\", \"logical\"",
                config.key_strategy.as_str()
            ),
        ));
    }
    // Auth material must never enter the key or the stored entries.
    const AUTH_HEADERS: &[&str] = &[
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "anthropic-api-key",
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
        }
        other => report.diagnostics.push(response_cache_error(
            "response_cache.unknown_backend",
            Some("backend.kind"),
            format!("unknown backend kind '{other}'"),
        )),
    }

    if let Some(tools) = &config.tools {
        validate_tool_cache(report, tools);
    }
}

fn validate_tool_cache(report: &mut ConfigReport, tools: &ToolCacheConfig) {
    validate_tool_policy(
        report,
        "default",
        tools.default.ttl_seconds,
        tools.default.bypass_rate,
    );
    if tools.default.cacheable {
        report.diagnostics.push(response_cache_error(
            "response_cache.tool_cacheable_default",
            Some("tools.default.cacheable"),
            "tools.default.cacheable must be false; use a named class or override to classify cacheable tools"
                .to_string(),
        ));
    }
    if !tools.default.members.is_empty() {
        report.diagnostics.push(response_cache_error(
            "response_cache.tool_default_members",
            Some("tools.default"),
            "tools.default.members is never matched (the default bucket applies to every \
             unclassified tool); move these names into a named class"
                .to_string(),
        ));
    }

    let mut owning_class: HashMap<&str, &str> = HashMap::new();
    for (class_name, class) in &tools.classes {
        validate_tool_cache_class(report, class_name, class, &mut owning_class);
    }

    validate_conflicting_tool_class_patterns(report, tools);

    for (tool_name, over) in &tools.overrides {
        if !is_supported_tool_pattern(tool_name) {
            report.diagnostics.push(response_cache_error(
                "response_cache.tool_invalid_pattern",
                Some("tools.overrides"),
                format!(
                    "tool override '{tool_name}' must be an exact name, 'prefix*', '*suffix', or \
                     '*contains*'"
                ),
            ));
        }
        validate_tool_policy(
            report,
            &format!("overrides.{tool_name}"),
            over.ttl_seconds,
            over.bypass_rate,
        );
        if over.cacheable == Some(true) && tool_name == "*" {
            report.diagnostics.push(response_cache_error(
                "response_cache.tool_catch_all_override",
                Some("tools.overrides"),
                format!(
                    "override '{tool_name}' sets cacheable = true for every tool; use a \
                     named override to classify cacheable tools"
                ),
            ));
        }
    }

    validate_conflicting_tool_override_patterns(report, tools);
}

fn validate_tool_cache_class<'a>(
    report: &mut ConfigReport,
    class_name: &'a str,
    class: &'a ToolClass,
    owning_class: &mut HashMap<&'a str, &'a str>,
) {
    validate_tool_policy(
        report,
        &format!("classes.{class_name}"),
        class.ttl_seconds,
        class.bypass_rate,
    );
    for member in &class.members {
        if !is_supported_tool_pattern(member) {
            report.diagnostics.push(response_cache_error(
                "response_cache.tool_invalid_pattern",
                Some("tools.classes"),
                format!(
                    "tool member '{member}' in class '{class_name}' must be an exact name, \
                     'prefix*', '*suffix', or '*contains*'"
                ),
            ));
        }
        if let Some(previous) = owning_class.get(member.as_str()) {
            if *previous != class_name {
                report.diagnostics.push(response_cache_error(
                    "response_cache.tool_multiple_classes",
                    Some("tools.classes"),
                    format!(
                        "tool member '{member}' appears in multiple classes ('{previous}' and \
                         '{class_name}'); a member — exact name or pattern — may appear in at \
                         most one class"
                    ),
                ));
            }
        } else {
            owning_class.insert(member.as_str(), class_name);
        }
        if class.cacheable && member == "*" {
            report.diagnostics.push(response_cache_error(
                "response_cache.tool_catch_all_member",
                Some("tools.classes"),
                format!(
                    "class '{class_name}' lists the catch-all member '{member}' with \
                     cacheable = true, which caches every tool; use a named class to \
                     classify cacheable tools"
                ),
            ));
        }
    }
}

fn validate_conflicting_tool_class_patterns(report: &mut ConfigReport, tools: &ToolCacheConfig) {
    for (index, (left_name, left_class)) in tools.classes.iter().enumerate() {
        for (right_name, right_class) in tools.classes.iter().skip(index + 1) {
            if left_class.cacheable == right_class.cacheable {
                continue;
            }
            let conflict = left_class.members.iter().any(|left_member| {
                left_member.contains('*')
                    && right_class.members.iter().any(|right_member| {
                        left_member != right_member
                            && right_member.contains('*')
                            && wildcard_patterns_overlap(left_member, right_member)
                    })
            });
            if conflict {
                report.diagnostics.push(response_cache_error(
                    "response_cache.tool_conflicting_classes",
                    Some("tools.classes"),
                    format!(
                        "classes '{left_name}' and '{right_name}' contain overlapping wildcard \
                         members with conflicting cacheable settings; split the patterns so one \
                         policy applies to every tool"
                    ),
                ));
            }
        }
    }
}

fn validate_conflicting_tool_override_patterns(report: &mut ConfigReport, tools: &ToolCacheConfig) {
    for (index, (left_name, left_override)) in tools.overrides.iter().enumerate() {
        for (right_name, right_override) in tools.overrides.iter().skip(index + 1) {
            // An omitted value inherits from whichever class wins for the
            // concrete tool name. Because overlapping patterns can select
            // different classes, only identical declarations are safe.
            let conflicting_cacheability = left_override.cacheable != right_override.cacheable;
            if !conflicting_cacheability
                || !left_name.contains('*')
                || !right_name.contains('*')
                || !wildcard_patterns_overlap(left_name, right_name)
            {
                continue;
            }
            report.diagnostics.push(response_cache_error(
                "response_cache.tool_conflicting_overrides",
                Some("tools.overrides"),
                format!(
                    "overrides '{left_name}' and '{right_name}' overlap with conflicting \
                     cacheable settings; split the patterns so one policy applies to every tool"
                ),
            ));
        }
    }
}

fn validate_tool_policy(
    report: &mut ConfigReport,
    location: &str,
    ttl_seconds: Option<u64>,
    bypass_rate: Option<f64>,
) {
    if ttl_seconds == Some(0) {
        report.diagnostics.push(response_cache_error(
            "response_cache.tool_invalid_ttl",
            Some("tools"),
            format!("tools.{location}.ttl_seconds must be greater than 0 when set"),
        ));
    }
    if let Some(rate) = bypass_rate
        && !(0.0..=1.0).contains(&rate)
    {
        report.diagnostics.push(response_cache_error(
            "response_cache.tool_invalid_bypass_rate",
            Some("tools"),
            format!("tools.{location}.bypass_rate must be in [0.0, 1.0] when set"),
        ));
    }
}

fn response_cache_error(code: &str, field: Option<&str>, message: String) -> ConfigDiagnostic {
    response_cache_diag(DiagnosticLevel::Error, code, field, message)
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
