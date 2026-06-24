// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt::{self, Write as _};

use nemo_relay::plugin::dynamic::{
    DynamicPluginCompatibility, DynamicPluginLoadContract, DynamicPluginManifest,
    DynamicPluginRecord,
};
use serde_json::Value;

use crate::config::ResolvedDynamicPluginConfig;

use super::state::ScopedDynamicPluginRecord;

pub(super) fn render_list(
    records: &[ScopedDynamicPluginRecord],
    host_config_by_id: &HashMap<String, ResolvedDynamicPluginConfig>,
) -> String {
    let mut output = String::new();
    push_line(
        &mut output,
        format_args!(
            "{:<32} {:<8} {:<7} {:<10} {:<10} {}",
            "ID", "SCOPE", "ENABLED", "STATE", "VALIDATION", "HOST CONFIG"
        ),
    );
    for entry in records {
        let host_config_status = host_config_by_id
            .get(&entry.record.metadata.id)
            .map(|plugin| plugin.host_config_status().to_string())
            .unwrap_or_else(|| "missing".into());
        push_line(
            &mut output,
            format_args!(
                "{:<32} {:<8} {:<7} {:<10} {:<10} {}",
                entry.record.metadata.id,
                entry.scope,
                entry.record.spec.enabled,
                lifecycle_state_label(&entry.record),
                <&'static str>::from(entry.record.status.validation.manifest),
                host_config_status
            ),
        );
    }
    finish_output(output)
}

pub(super) fn render_inspect(
    entry: &ScopedDynamicPluginRecord,
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    host_config: Option<&ResolvedDynamicPluginConfig>,
) -> String {
    let record = &entry.record;
    let mut output = String::new();

    push_line(&mut output, format_args!("id: {}", record.metadata.id));
    push_line(&mut output, format_args!("scope: {}", entry.scope));
    push_line(&mut output, format_args!("kind: {}", record.metadata.kind));
    push_line(
        &mut output,
        format_args!(
            "name: {}",
            record.metadata.name.as_deref().unwrap_or("<none>")
        ),
    );
    push_line(
        &mut output,
        format_args!(
            "version: {}",
            record.metadata.version.as_deref().unwrap_or("<none>")
        ),
    );
    push_line(&mut output, format_args!("manifest: {manifest_ref}"));
    push_line(
        &mut output,
        format_args!("plugins_toml: {}", entry.plugins_toml_path.display()),
    );
    push_line(
        &mut output,
        format_args!("lifecycle_state_path: {}", entry.state_path.display()),
    );
    push_line(
        &mut output,
        format_args!(
            "source.manifest_ref: {}",
            record.source.manifest_ref.as_deref().unwrap_or("<none>")
        ),
    );
    push_line(
        &mut output,
        format_args!(
            "source.artifact_ref: {}",
            record.source.artifact_ref.as_deref().unwrap_or("<none>")
        ),
    );
    push_line(
        &mut output,
        format_args!(
            "source.environment_ref: {}",
            record.source.environment_ref.as_deref().unwrap_or("<none>")
        ),
    );
    push_line(
        &mut output,
        format_args!("desired.present: {}", record.spec.present),
    );
    push_line(
        &mut output,
        format_args!("desired.enabled: {}", record.spec.enabled),
    );
    push_line(
        &mut output,
        format_args!("generation: {}", record.metadata.generation),
    );
    push_line(
        &mut output,
        format_args!("reconciled: {}", record.is_reconciled()),
    );
    push_line(
        &mut output,
        format_args!(
            "host_config: {}",
            host_config
                .map(|plugin| plugin.host_config_status().to_string())
                .unwrap_or_else(|| "missing".into())
        ),
    );

    match &record.compatibility {
        DynamicPluginCompatibility::Worker(compatibility) => {
            push_line(
                &mut output,
                format_args!("compat.relay: {}", compatibility.relay),
            );
            push_line(
                &mut output,
                format_args!("compat.worker_protocol: {}", compatibility.worker_protocol),
            );
        }
        DynamicPluginCompatibility::RustDynamic(compatibility) => {
            push_line(
                &mut output,
                format_args!("compat.relay: {}", compatibility.relay),
            );
            push_line(
                &mut output,
                format_args!("compat.native_api: {}", compatibility.native_api),
            );
        }
    }

    match &record.load {
        DynamicPluginLoadContract::Worker(load) => {
            push_line(&mut output, format_args!("load.runtime: {}", load.runtime));
            push_line(
                &mut output,
                format_args!("load.entrypoint: {}", load.entrypoint),
            );
        }
        DynamicPluginLoadContract::RustDynamic(load) => {
            push_line(&mut output, format_args!("load.library: {}", load.library));
            push_line(&mut output, format_args!("load.symbol: {}", load.symbol));
        }
    }

    render_status(&mut output, record);
    push_line(
        &mut output,
        format_args!(
            "capabilities: {}",
            manifest
                .capabilities
                .items
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    );
    push_line(
        &mut output,
        format_args!(
            "host_config_json: {}",
            host_config
                .map(redacted_host_config_json)
                .filter(|config| !config.is_null())
                .map(|config| {
                    serde_json::to_string_pretty(&config).expect("host config serializes")
                })
                .unwrap_or_else(|| "<none>".into())
        ),
    );

    finish_output(output)
}

pub(super) fn render_validation_summary(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    entry: Option<&ScopedDynamicPluginRecord>,
    host_config: Option<&ResolvedDynamicPluginConfig>,
) -> String {
    let mut output = String::new();
    push_line(
        &mut output,
        format_args!("Dynamic plugin '{}' is valid.", manifest.plugin.id),
    );
    push_line(&mut output, format_args!("kind: {}", manifest.plugin.kind));
    push_line(&mut output, format_args!("manifest: {manifest_ref}"));
    if let Some(entry) = entry {
        push_line(&mut output, format_args!("scope: {}", entry.scope));
        push_line(
            &mut output,
            format_args!("lifecycle_state_path: {}", entry.state_path.display()),
        );
        push_line(
            &mut output,
            format_args!("desired.enabled: {}", entry.record.spec.enabled),
        );
        push_line(
            &mut output,
            format_args!(
                "host_config: {}",
                host_config
                    .map(|plugin| plugin.host_config_status().to_string())
                    .unwrap_or_else(|| "missing".into())
            ),
        );
    }
    finish_output(output)
}

fn render_status(output: &mut String, record: &DynamicPluginRecord) {
    push_line(
        output,
        format_args!(
            "status.validation.manifest: {}",
            <&'static str>::from(record.status.validation.manifest)
        ),
    );
    push_line(
        output,
        format_args!(
            "status.validation.compatibility: {}",
            <&'static str>::from(record.status.validation.compatibility)
        ),
    );
    push_line(
        output,
        format_args!(
            "status.validation.integrity: {}",
            <&'static str>::from(record.status.validation.integrity)
        ),
    );
    push_line(
        output,
        format_args!(
            "status.validation.environment: {}",
            <&'static str>::from(record.status.validation.environment)
        ),
    );
    push_line(
        output,
        format_args!(
            "status.validation.authenticity: {}",
            <&'static str>::from(record.status.validation.authenticity)
        ),
    );
    push_line(
        output,
        format_args!(
            "status.validation.policy_satisfied: {}",
            <&'static str>::from(record.status.validation.policy_satisfied)
        ),
    );
    push_line(
        output,
        format_args!(
            "status.runtime.state: {}",
            <&'static str>::from(record.status.runtime.state)
        ),
    );
    push_line(
        output,
        format_args!(
            "status.runtime.observed_generation: {}",
            record.status.runtime.observed_generation
        ),
    );
    if let Some(value) = record.status.validation.checked_at.as_deref() {
        push_line(
            output,
            format_args!("status.validation.checked_at: {value}"),
        );
    }
    if let Some(value) = record.status.validation.message.as_deref() {
        push_line(output, format_args!("status.validation.message: {value}"));
    }
    if let Some(value) = record.status.runtime.started_at.as_deref() {
        push_line(output, format_args!("status.runtime.started_at: {value}"));
    }
    if let Some(value) = record.status.runtime.updated_at.as_deref() {
        push_line(output, format_args!("status.runtime.updated_at: {value}"));
    }
    if let Some(value) = record.status.runtime.message.as_deref() {
        push_line(output, format_args!("status.runtime.message: {value}"));
    }
    if let Some(value) = record.status.startup_class {
        push_line(output, format_args!("status.startup_class: {}", value));
    }
    if let Some(value) = record.status.attestation_mode {
        push_line(output, format_args!("status.attestation_mode: {}", value));
    }
    if let Some(error) = record.status.last_error.as_ref() {
        push_line(
            output,
            format_args!(
                "status.last_error: {}:{} {}",
                error.phase, error.code, error.message
            ),
        );
    }
}

fn push_line(output: &mut String, args: fmt::Arguments<'_>) {
    output.write_fmt(args).expect("writing to string succeeds");
    output.push('\n');
}

fn finish_output(mut output: String) -> String {
    output.pop();
    output
}

fn lifecycle_state_label(record: &DynamicPluginRecord) -> &'static str {
    if record.is_tombstoned() {
        "tombstoned"
    } else {
        record.status.runtime.state.into()
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
