// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Environment names shared by MCP generation and gateway compatibility checks.

use std::collections::BTreeSet;

use serde_json::Value;

const BASE_MCP_ENV_VARS: &[&str] = &[
    "ANTHROPIC_API_KEY",
    "APPDATA",
    "AWS_ACCESS_KEY_ID",
    "AWS_ALLOW_HTTP",
    "AWS_CONFIG_FILE",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SHARED_CREDENTIALS_FILE",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LOCALAPPDATA",
    "NEMO_RELAY_ANTHROPIC_BASE_URL",
    "NEMO_RELAY_MAX_HOOK_PAYLOAD_BYTES",
    "NEMO_RELAY_MAX_PASSTHROUGH_BODY_BYTES",
    "NEMO_RELAY_OPENAI_BASE_URL",
    "NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS",
    "NEMO_RELAY_PYTHON",
    "NO_PROXY",
    "OPENAI_API_KEY",
    "OTEL_EXPORTER_OTLP_COMPRESSION",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TIMEOUT",
    "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_SDK_DISABLED",
    "OTEL_SERVICE_NAME",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TMPDIR",
    "USERPROFILE",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "http_proxy",
    "https_proxy",
    "no_proxy",
];

const BLOCKED_MCP_ENV_VARS: &[&str] = &[
    "NEMO_RELAY_BINDING_KIND",
    "NEMO_RELAY_BOOTSTRAP_AGENT",
    "NEMO_RELAY_BOOTSTRAP_FINGERPRINT",
    "NEMO_RELAY_BOOTSTRAP_STATE_DIR",
    "NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN",
    "NEMO_RELAY_CONFIG_SCOPE",
    "NEMO_RELAY_FAIL_CLOSED",
    "NEMO_RELAY_GATEWAY_BIND",
    "NEMO_RELAY_GATEWAY_URL",
    "NEMO_RELAY_HOST_SOCKET",
    "NEMO_RELAY_MCP_GENERATION_FILE",
    "NEMO_RELAY_NATIVE_ABI_VERSION",
    "NEMO_RELAY_PLUGIN_BINARY",
    "NEMO_RELAY_PLUGIN_BIND",
    "NEMO_RELAY_SIDECAR_JOB_NAME",
    "NEMO_RELAY_PLUGIN_CONFIG_PATH",
    "NEMO_RELAY_PLUGIN_GATEWAY_URL",
    "NEMO_RELAY_PLUGIN_ID",
    "NEMO_RELAY_RUNTIME_OWNER",
    "NEMO_RELAY_WORKER_ENDPOINT_FILE",
    "NEMO_RELAY_WORKER_ID",
    "NEMO_RELAY_WORKER_SOCKET",
    "NEMO_RELAY_WORKER_TOKEN",
];

pub(crate) fn forwarded_names(
    environment: impl IntoIterator<Item = String>,
    config: Option<&Value>,
) -> Vec<String> {
    forwarded_names_for_platform(environment, config, cfg!(windows))
}

pub(crate) fn forwarded_names_for_platform(
    environment: impl IntoIterator<Item = String>,
    config: Option<&Value>,
    windows: bool,
) -> Vec<String> {
    let mut names = BASE_MCP_ENV_VARS
        .iter()
        .map(|name| (*name).to_string())
        .collect::<BTreeSet<_>>();
    for name in environment {
        if prefix_allowed(&name, windows) && !blocked(&name) {
            insert_name(&mut names, name, windows);
        }
    }
    if let Some(config) = config {
        collect_config_names(config, &mut names, windows);
    }
    names.into_iter().collect()
}

pub(crate) fn config_referenced_names(config: Option<&Value>) -> Vec<String> {
    config_referenced_names_for_platform(config, cfg!(windows))
}

pub(crate) fn config_referenced_names_for_platform(
    config: Option<&Value>,
    windows: bool,
) -> Vec<String> {
    let mut names = BTreeSet::new();
    if let Some(config) = config {
        collect_config_names(config, &mut names, windows);
    }
    names.into_iter().collect()
}

fn prefix_allowed(name: &str, windows: bool) -> bool {
    ["NEMO_RELAY_", "OTEL_", "AWS_"].iter().any(|prefix| {
        if windows {
            starts_with_ignore_ascii_case(name, prefix)
        } else {
            name.starts_with(prefix)
        }
    })
}

fn blocked(name: &str) -> bool {
    BLOCKED_MCP_ENV_VARS
        .iter()
        .any(|blocked| name.eq_ignore_ascii_case(blocked))
        || starts_with_ignore_ascii_case(name, "NEMO_RELAY_TEST_")
}

fn starts_with_ignore_ascii_case(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix))
}

fn insert_name(names: &mut BTreeSet<String>, name: String, windows: bool) {
    if !windows
        || !names
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(&name))
    {
        names.insert(name);
    }
}

fn collect_config_names(value: &Value, names: &mut BTreeSet<String>, windows: bool) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                match key.as_str() {
                    "header_env" => {
                        if let Some(headers) = value.as_object() {
                            for name in headers.values().filter_map(Value::as_str) {
                                if !name.is_empty() && !blocked(name) {
                                    insert_name(names, name.to_owned(), windows);
                                }
                            }
                        }
                    }
                    "secret_access_key_var" | "session_token_var" => {
                        if let Some(name) = value.as_str()
                            && !name.is_empty()
                            && !blocked(name)
                        {
                            insert_name(names, name.to_owned(), windows);
                        }
                    }
                    _ => collect_config_names(value, names, windows),
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_config_names(value, names, windows);
            }
        }
        _ => {}
    }
}
