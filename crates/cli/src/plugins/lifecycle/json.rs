// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use nemo_relay::plugin::dynamic::DynamicPluginManifest;
use serde_json::{Value, json};

use crate::config::{DynamicPluginHostConfigStatus, ResolvedDynamicPluginConfig};
use crate::error::{CliError, PluginLifecycleFailureKind};

use super::render::{check_state_label, manifest_kind_label, runtime_state_label};
use super::state::ScopedDynamicPluginRecord;

pub(super) struct ValidateJsonContext<'a> {
    pub(super) command: &'static str,
    pub(super) target: Option<&'a str>,
    pub(super) target_kind: &'static str,
    pub(super) resolved_plugin_id: Option<&'a str>,
    pub(super) manifest: &'a DynamicPluginManifest,
    pub(super) manifest_ref: &'a str,
    pub(super) entry: Option<&'a ScopedDynamicPluginRecord>,
    pub(super) host_config: Option<&'a ResolvedDynamicPluginConfig>,
}

pub(super) fn print_json(value: &Value) -> Result<(), CliError> {
    let rendered = serde_json::to_string_pretty(value).map_err(|error| {
        CliError::Config(format!("could not serialize plugin JSON output: {error}"))
    })?;
    println!("{rendered}");
    Ok(())
}

pub(super) fn list_success_envelope(
    command: &'static str,
    target: Option<&str>,
    records: &[ScopedDynamicPluginRecord],
    host_config_by_id: &HashMap<String, ResolvedDynamicPluginConfig>,
) -> Value {
    success_envelope(
        command,
        target,
        Value::Array(
            records
                .iter()
                .map(|entry| {
                    let record = &entry.record;
                    json!({
                        "id": record.metadata.id,
                        "name": record.metadata.name,
                        "kind": manifest_kind_label(record.metadata.kind),
                        "enabled": record.spec.enabled,
                        "tombstoned": record.is_tombstoned(),
                        "validation_state": check_state_label(record.status.validation.manifest),
                        "runtime_state": if record.is_tombstoned() { "tombstoned" } else { runtime_state_label(record.status.runtime.state) },
                        "startup": record.status.startup_class.map(|value| format!("{value:?}").to_lowercase()),
                        "last_error": record.status.last_error.as_ref().map(|error| {
                            json!({
                                "phase": format!("{:?}", error.phase).to_lowercase(),
                                "code": error.code,
                                "message": error.message,
                            })
                        }),
                        "host_config": host_config_by_id
                            .get(&record.metadata.id)
                            .map(|plugin| host_config_status_label(plugin.host_config_status()))
                            .unwrap_or("missing"),
                    })
                })
                .collect(),
        ),
    )
}

pub(super) fn inspect_success_envelope(
    command: &'static str,
    target: &str,
    entry: &ScopedDynamicPluginRecord,
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    host_config: Option<&ResolvedDynamicPluginConfig>,
) -> Value {
    let record = &entry.record;
    success_envelope(
        command,
        Some(target),
        json!({
            "id": record.metadata.id,
            "name": record.metadata.name,
            "kind": manifest_kind_label(record.metadata.kind),
            "scope": entry.scope.label(),
            "manifest_ref": manifest_ref,
            "plugins_toml_path": entry.plugins_toml_path,
            "state_path": entry.state_path,
            "load": record.load,
            "compat": record.compatibility,
            "capabilities": manifest.capabilities.items.iter().map(|item| format!("{item:?}").to_lowercase()).collect::<Vec<_>>(),
            "metadata": record.metadata,
            "source": record.source,
            "spec": record.spec,
            "status": record.status,
            "host_config_status": host_config.map(|plugin| host_config_status_label(plugin.host_config_status())).unwrap_or("missing"),
            "host_config": host_config.map(|plugin| Value::Object(plugin.config.clone())).unwrap_or(Value::Null),
        }),
    )
}

pub(super) fn validate_success_envelope(context: ValidateJsonContext<'_>) -> Value {
    let notes = context
        .entry
        .and_then(|entry| entry.record.status.validation.message.clone())
        .into_iter()
        .collect::<Vec<_>>();

    success_envelope(
        context.command,
        context.target,
        json!({
            "target_kind": context.target_kind,
            "resolved_plugin_id": context.resolved_plugin_id.or(Some(context.manifest.plugin.id.as_str())),
            "valid": true,
            "errors": Vec::<String>::new(),
            "warnings": Vec::<String>::new(),
            "notes": notes,
            "manifest_ref": context.manifest_ref,
            "kind": manifest_kind_label(context.manifest.plugin.kind),
            "desired_enabled": context.entry.map(|entry| entry.record.spec.enabled),
            "host_config_status": context.host_config.map(|plugin| host_config_status_label(plugin.host_config_status())).unwrap_or("missing"),
        }),
    )
}

pub(super) fn failure_envelope(
    command: &'static str,
    target: Option<&str>,
    kind: PluginLifecycleFailureKind,
    message: &str,
) -> Value {
    let code = match kind {
        PluginLifecycleFailureKind::Failed => "operation_failed",
        PluginLifecycleFailureKind::NotFound => "not_found",
        PluginLifecycleFailureKind::Refused => "refused",
    };
    with_schema(json!({
        "ok": false,
        "command": command,
        "target": target,
        "error": {
            "code": code,
            "kind": kind,
            "message": message,
            "details": {}
        },
        "warnings": []
    }))
}

pub(super) fn generic_failure_envelope(
    command: &'static str,
    target: Option<&str>,
    message: &str,
) -> Value {
    failure_envelope(command, target, PluginLifecycleFailureKind::Failed, message)
}

fn success_envelope(command: &'static str, target: Option<&str>, data: Value) -> Value {
    with_schema(json!({
        "ok": true,
        "command": command,
        "target": target,
        "warnings": [],
        "data": data
    }))
}

fn with_schema(mut value: Value) -> Value {
    if let Some(object) = value.as_object_mut() {
        object.insert("schema_version".into(), json!(1));
    }
    value
}

fn host_config_status_label(status: DynamicPluginHostConfigStatus) -> &'static str {
    match status {
        DynamicPluginHostConfigStatus::Absent => "absent",
        DynamicPluginHostConfigStatus::Present => "present",
    }
}
