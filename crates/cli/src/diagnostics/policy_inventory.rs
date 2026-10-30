// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Non-activating inventory of file exporters and configured privacy policies.

use super::{Check, Status, check_dir_writable, section_enabled};
use nemo_relay_pii_redaction::component::{PII_REDACTION_PLUGIN_KIND, PiiRedactionConfig};
use nemo_relay_runtime::plugin::PluginConfig;
use serde_json::Value;
use std::path::{Component, Path};

/// List file sinks and check enabled destinations without writing telemetry.
pub(super) fn observability_otlp_file_checks(
    config: &Value,
    component_enabled: bool,
) -> Vec<Check> {
    let enabled = component_enabled && section_enabled(config, "opentelemetry");
    config
        .get("opentelemetry")
        .and_then(|section| section.get("file_sinks"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, sink)| {
            let projection = sink.get("type").and_then(Value::as_str).unwrap_or("full");
            let directory = sink.get("output_directory").and_then(Value::as_str);
            let filename = sink.get("filename").and_then(Value::as_str);
            let destination = match (directory, filename) {
                (Some(directory), Some(filename)) => {
                    Path::new(directory).join(filename).display().to_string()
                }
                (Some(directory), None) => format!("{directory} (runtime-generated filename)"),
                _ => "output_directory missing".into(),
            };
            let (status, detail) = if !enabled {
                (Status::Info, "disabled; filesystem probe skipped".into())
            } else if let Some(directory) = directory {
                match file_destination_feasibility(Path::new(directory), filename) {
                    Ok(result) => result,
                    Err(error) => (
                        Status::Fail,
                        format!(
                            "destination unavailable: {error}; check path and write permissions"
                        ),
                    ),
                }
            } else {
                (Status::Fail, "configure output_directory".into())
            };
            Check {
                name: "OpenTelemetry file sink",
                status,
                details: format!(
                    "file_sinks[{index}] ({projection}); {}: {destination}; {detail}",
                    if enabled { "enabled" } else { "disabled" }
                ),
            }
        })
        .collect()
}

/// Check file access or create and remove a probe in the nearest existing folder.
/// Reject parent-folder traversal and output-file symlinks, matching the exporter.
fn file_destination_feasibility(
    directory: &Path,
    filename: Option<&str>,
) -> std::io::Result<(Status, String)> {
    if directory.as_os_str().is_empty()
        || directory
            .to_str()
            .is_some_and(|value| value.trim().is_empty())
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "output_directory must be a nonblank path",
        ));
    }
    if directory
        .components()
        .any(|component| component == Component::ParentDir)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "output_directory must not contain '..' path components; the OTLP exporter rejects parent-folder traversal",
        ));
    }
    if let Some(filename) = filename {
        let path = directory.join(filename);
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() {
                    return Err(std::io::Error::other(
                        "the OTLP exporter rejects output-file symlinks; configure a regular file",
                    ));
                }
                if !metadata.is_file() || metadata.permissions().readonly() {
                    return Err(std::io::Error::other(
                        "output must be a writable regular file",
                    ));
                }
                // Opening without create or truncate verifies access without changing telemetry.
                std::fs::OpenOptions::new().write(true).open(&path)?;
                return Ok((
                    Status::Pass,
                    "existing file appears writable; no content written; runtime export unverified"
                        .into(),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    let mut ancestor = directory.to_path_buf();
    loop {
        let candidate = if ancestor.as_os_str().is_empty() {
            Path::new(".")
        } else {
            &ancestor
        };
        match std::fs::metadata(candidate) {
            Ok(_) => {
                check_dir_writable(candidate)?;
                return if candidate == directory {
                    Ok((Status::Pass, "directory appears writable; temporary probe file created and removed; telemetry file not created; runtime export unverified".into()))
                } else {
                    Ok((
                        Status::Warn,
                        format!(
                            "directory not present; runtime will create it; ancestor {} appears writable; temporary probe file created and removed there; runtime export unverified",
                            candidate.display()
                        ),
                    ))
                };
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if !ancestor.pop() {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
    }
}

/// Summarize configured privacy coverage without exposing policy values or payloads.
pub(super) fn collect_pii_checks(checks: &mut Vec<Check>, plugins: &PluginConfig, invalid: bool) {
    let Some(component) = plugins
        .components
        .iter()
        .find(|component| component.kind == PII_REDACTION_PLUGIN_KIND)
    else {
        checks.push(Check {
            name: "PII redaction",
            status: Status::Info,
            details: "component not configured; no configured privacy policy".into(),
        });
        return;
    };
    if invalid {
        checks.push(Check {
            name: "PII redaction",
            status: Status::Fail,
            details: "config invalid (see plugin diagnostics); effective policy unavailable".into(),
        });
        return;
    }
    let Ok(config) =
        serde_json::from_value::<PiiRedactionConfig>(Value::Object(component.config.clone()))
    else {
        checks.push(Check {
            name: "PII redaction",
            status: Status::Fail,
            details: "invalid config; effective policy unavailable".into(),
        });
        return;
    };
    let state = if component.enabled {
        "enabled"
    } else {
        "disabled"
    };
    let surfaces = |flags: [bool; 5]| {
        ["input", "output", "mark", "tool_input", "tool_output"]
            .into_iter()
            .zip(flags)
            .map(|(name, enabled)| format!("{name}={}", if enabled { "on" } else { "off" }))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let profiles = if config.profiles.is_empty() {
        format!(
            "single policy; {}",
            surfaces(
                [
                    config.input,
                    config.output,
                    config.mark,
                    config.tool_input,
                    config.tool_output
                ]
                .map(|flag| flag && component.enabled)
            )
        )
    } else {
        config
            .profiles
            .iter()
            .enumerate()
            .map(|(index, profile)| {
                let enabled = component.enabled && profile.enabled;
                format!(
                    "profile_{index}={}; {}",
                    if enabled { "enabled" } else { "disabled" },
                    surfaces([enabled; 5])
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    checks.push(Check {
        name: "PII redaction", status: Status::Info,
        details: format!("component {state}; {profiles}; configuration only: runtime backend availability and content sanitization unverified"),
    });
}

/// Read trace endpoints using the current field name or its legacy alias.
pub(super) fn otlp_trace_endpoints(config: &Value) -> Option<&Vec<Value>> {
    let section = config.get("opentelemetry")?;
    section
        .get("traces")
        .or_else(|| section.get("endpoints"))?
        .as_array()
}

/// Warn about inactive destinations and name only the settings that are disabled.
pub(super) fn observability_otlp_enablement_checks(
    config: &Value,
    component_enabled: bool,
) -> Vec<Check> {
    let Some(section) = config.get("opentelemetry") else {
        return Vec::new();
    };
    let endpoints = otlp_trace_endpoints(config).map_or(0, Vec::len);
    let files = section
        .get("file_sinks")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let mut checks = Vec::new();
    if (!component_enabled || !section_enabled(config, "opentelemetry"))
        && (endpoints > 0 || files > 0)
    {
        let mut settings = Vec::new();
        if !component_enabled {
            settings.push("components[kind=observability].enabled = true");
        }
        if !section_enabled(config, "opentelemetry") {
            settings.push("components[kind=observability].config.opentelemetry.enabled = true");
        }
        let settings = settings.join(" and ");
        checks.push(Check {
            name: "OpenTelemetry enablement",
            status: Status::Warn,
            details: format!(
                "components[kind=observability].config.opentelemetry: {endpoints} trace endpoint(s) and {files} file sink(s) configured but disabled; set {settings} in plugins.toml to activate; no probes or exports performed"
            ),
        });
    }
    checks
}

/// Check the same header constraints used at OTLP activation, without retaining
/// resolved values in the report or sending them with the reachability probe.
pub(super) fn validate_otlp_credential_references(endpoint: &Value) -> Result<(), String> {
    if let Some(references) = endpoint.get("header_env").and_then(Value::as_object) {
        for (header, variable) in references {
            let Some(variable) = variable.as_str() else {
                return Err(format!("header_env.{header} must be a string"));
            };
            if variable.trim().is_empty()
                || variable.trim() != variable
                || variable.contains(['\0', '='])
            {
                return Err(format!(
                    "header_env.{header} must name a nonblank environment variable without surrounding whitespace, '=' or NUL"
                ));
            }
        }
    }
    for (header, value) in super::endpoint_headers(endpoint)? {
        if value.trim() != value {
            return Err(format!(
                "header {header:?} must contain a nonblank value without surrounding whitespace"
            ));
        }
    }
    Ok(())
}
