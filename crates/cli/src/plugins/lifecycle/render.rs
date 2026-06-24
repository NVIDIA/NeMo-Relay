// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use nemo_relay::plugin::dynamic::{
    DynamicPluginCompatibility, DynamicPluginKind, DynamicPluginLoadContract,
    DynamicPluginManifest, DynamicPluginRecord,
};
use serde_json::Value;

use crate::config::{DynamicPluginHostConfigStatus, ResolvedDynamicPluginConfig};

use super::state::ScopedDynamicPluginRecord;

pub(super) fn render_list(
    records: &[ScopedDynamicPluginRecord],
    host_config_by_id: &HashMap<String, ResolvedDynamicPluginConfig>,
) -> String {
    let mut lines = Vec::with_capacity(records.len() + 1);
    lines.push(format!(
        "{:<32} {:<8} {:<7} {:<10} {:<10} {}",
        "ID", "SCOPE", "ENABLED", "STATE", "VALIDATION", "HOST CONFIG"
    ));
    for entry in records {
        let host_config_status = host_config_by_id
            .get(&entry.record.metadata.id)
            .map(|plugin| host_config_status_label(plugin.host_config_status()))
            .unwrap_or("missing");
        lines.push(format!(
            "{:<32} {:<8} {:<7} {:<10} {:<10} {}",
            entry.record.metadata.id,
            entry.scope.label(),
            entry.record.spec.enabled,
            lifecycle_state_label(&entry.record),
            <&'static str>::from(entry.record.status.validation.manifest),
            host_config_status
        ));
    }
    lines.join("\n")
}

pub(super) fn render_inspect(
    entry: &ScopedDynamicPluginRecord,
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    host_config: Option<&ResolvedDynamicPluginConfig>,
) -> String {
    let record = &entry.record;
    let mut lines = vec![
        format!("id: {}", record.metadata.id),
        format!("scope: {}", entry.scope.label()),
        format!("kind: {}", manifest_kind_label(record.metadata.kind)),
        format!(
            "name: {}",
            record.metadata.name.as_deref().unwrap_or("<none>")
        ),
        format!(
            "version: {}",
            record.metadata.version.as_deref().unwrap_or("<none>")
        ),
        format!("manifest: {manifest_ref}"),
        format!("plugins_toml: {}", entry.plugins_toml_path.display()),
        format!("lifecycle_state_path: {}", entry.state_path.display()),
        format!(
            "source.manifest_ref: {}",
            record.source.manifest_ref.as_deref().unwrap_or("<none>")
        ),
        format!(
            "source.artifact_ref: {}",
            record.source.artifact_ref.as_deref().unwrap_or("<none>")
        ),
        format!(
            "source.environment_ref: {}",
            record.source.environment_ref.as_deref().unwrap_or("<none>")
        ),
        format!("desired.present: {}", record.spec.present),
        format!("desired.enabled: {}", record.spec.enabled),
        format!("generation: {}", record.metadata.generation),
        format!("reconciled: {}", record.is_reconciled()),
        format!(
            "host_config: {}",
            host_config
                .map(|plugin| host_config_status_label(plugin.host_config_status()))
                .unwrap_or("missing")
        ),
    ];

    match &record.compatibility {
        DynamicPluginCompatibility::Worker(compatibility) => {
            lines.push(format!("compat.relay: {}", compatibility.relay));
            lines.push(format!(
                "compat.worker_protocol: {}",
                compatibility.worker_protocol
            ));
        }
        DynamicPluginCompatibility::RustDynamic(compatibility) => {
            lines.push(format!("compat.relay: {}", compatibility.relay));
            lines.push(format!("compat.native_api: {}", compatibility.native_api));
        }
    }

    match &record.load {
        DynamicPluginLoadContract::Worker(load) => {
            lines.push(format!("load.runtime: {:?}", load.runtime).to_lowercase());
            lines.push(format!("load.entrypoint: {}", load.entrypoint));
        }
        DynamicPluginLoadContract::RustDynamic(load) => {
            lines.push(format!("load.library: {}", load.library));
            lines.push(format!("load.symbol: {}", load.symbol));
        }
    }

    lines.extend(render_status(record));
    lines.push(format!(
        "capabilities: {}",
        manifest
            .capabilities
            .items
            .iter()
            .map(|item| format!("{item:?}").to_lowercase())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    lines.push(format!(
        "host_config_json: {}",
        host_config
            .map(redacted_host_config_json)
            .filter(|config| !config.is_null())
            .map(|config| {
                serde_json::to_string_pretty(&config).expect("host config serializes")
            })
            .unwrap_or_else(|| "<none>".into())
    ));
    lines.join("\n")
}

pub(super) fn render_validation_summary(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    entry: Option<&ScopedDynamicPluginRecord>,
    host_config: Option<&ResolvedDynamicPluginConfig>,
) -> String {
    let mut lines = vec![
        format!("Dynamic plugin '{}' is valid.", manifest.plugin.id),
        format!("kind: {}", manifest_kind_label(manifest.plugin.kind)),
        format!("manifest: {manifest_ref}"),
    ];
    if let Some(entry) = entry {
        lines.push(format!("scope: {}", entry.scope.label()));
        lines.push(format!(
            "lifecycle_state_path: {}",
            entry.state_path.display()
        ));
        lines.push(format!("desired.enabled: {}", entry.record.spec.enabled));
        lines.push(format!(
            "host_config: {}",
            host_config
                .map(|plugin| host_config_status_label(plugin.host_config_status()))
                .unwrap_or("missing")
        ));
    }
    lines.join("\n")
}

fn render_status(record: &DynamicPluginRecord) -> Vec<String> {
    let mut lines = vec![
        format!(
            "status.validation.manifest: {}",
            <&'static str>::from(record.status.validation.manifest)
        ),
        format!(
            "status.validation.compatibility: {}",
            <&'static str>::from(record.status.validation.compatibility)
        ),
        format!(
            "status.validation.integrity: {}",
            <&'static str>::from(record.status.validation.integrity)
        ),
        format!(
            "status.validation.environment: {}",
            <&'static str>::from(record.status.validation.environment)
        ),
        format!(
            "status.validation.authenticity: {}",
            <&'static str>::from(record.status.validation.authenticity)
        ),
        format!(
            "status.validation.policy_satisfied: {}",
            <&'static str>::from(record.status.validation.policy_satisfied)
        ),
        format!(
            "status.runtime.state: {}",
            <&'static str>::from(record.status.runtime.state)
        ),
        format!(
            "status.runtime.observed_generation: {}",
            record.status.runtime.observed_generation
        ),
    ];
    if let Some(value) = record.status.validation.checked_at.as_deref() {
        lines.push(format!("status.validation.checked_at: {value}"));
    }
    if let Some(value) = record.status.validation.message.as_deref() {
        lines.push(format!("status.validation.message: {value}"));
    }
    if let Some(value) = record.status.runtime.started_at.as_deref() {
        lines.push(format!("status.runtime.started_at: {value}"));
    }
    if let Some(value) = record.status.runtime.updated_at.as_deref() {
        lines.push(format!("status.runtime.updated_at: {value}"));
    }
    if let Some(value) = record.status.runtime.message.as_deref() {
        lines.push(format!("status.runtime.message: {value}"));
    }
    if let Some(value) = record.status.startup_class {
        lines.push(format!("status.startup_class: {:?}", value).to_lowercase());
    }
    if let Some(value) = record.status.attestation_mode {
        lines.push(format!("status.attestation_mode: {:?}", value).to_lowercase());
    }
    if let Some(error) = record.status.last_error.as_ref() {
        lines.push(format!(
            "status.last_error: {}:{} {}",
            format!("{:?}", error.phase).to_lowercase(),
            error.code,
            error.message
        ));
    }
    lines
}

fn host_config_status_label(status: DynamicPluginHostConfigStatus) -> &'static str {
    match status {
        DynamicPluginHostConfigStatus::Absent => "absent",
        DynamicPluginHostConfigStatus::Present => "present",
    }
}

fn lifecycle_state_label(record: &DynamicPluginRecord) -> &'static str {
    if record.is_tombstoned() {
        "tombstoned"
    } else {
        record.status.runtime.state.into()
    }
}

pub(super) fn manifest_kind_label(kind: DynamicPluginKind) -> &'static str {
    match kind {
        DynamicPluginKind::RustDynamic => "rust_dynamic",
        DynamicPluginKind::Worker => "worker",
    }
}

pub(super) fn redacted_host_config_json(host_config: &ResolvedDynamicPluginConfig) -> Value {
    if host_config.config.is_empty() {
        return Value::Null;
    }

    Value::Object(
        host_config
            .config
            .keys()
            .cloned()
            .map(|key| (key, Value::String("<redacted>".into())))
            .collect(),
    )
}
