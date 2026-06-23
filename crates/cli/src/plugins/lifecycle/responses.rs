// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Machine-readable response layer for dynamic plugin lifecycle commands.
//!
//! This module owns the versioned response contract for
//! `plugins list`, `plugins inspect`, `plugins validate`, and structured
//! lifecycle errors. Command logic lives in `lifecycle.rs`; this file only
//! turns already-resolved state into stable responses serialized as JSON.

use std::collections::HashMap;

use nemo_relay::plugin::dynamic::{DynamicPluginFailurePhase, DynamicPluginManifest};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::config::{DynamicPluginHostConfigStatus, ResolvedDynamicPluginConfig};
use crate::error::{CliError, PluginLifecycleFailureKind};

use super::render::{check_state_label, manifest_kind_label, runtime_state_label};
use super::state::ScopedDynamicPluginRecord;

#[derive(Debug)]
pub(super) struct ValidateResponseInput<'a> {
    pub(super) command: &'static str,
    pub(super) target: Option<&'a str>,
    pub(super) target_kind: &'static str,
    pub(super) resolved_plugin_id: Option<&'a str>,
    pub(super) manifest: &'a DynamicPluginManifest,
    pub(super) manifest_ref: &'a str,
    pub(super) entry: Option<&'a ScopedDynamicPluginRecord>,
    pub(super) host_config: Option<&'a ResolvedDynamicPluginConfig>,
}

#[derive(Debug, Serialize)]
pub(super) struct ResponseEnvelope<T> {
    schema_version: u32,
    ok: bool,
    command: &'static str,
    target: Option<String>,
    warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ResponseError>,
}

#[derive(Debug, Serialize)]
pub(super) struct ResponseError {
    code: &'static str,
    kind: PluginLifecycleFailureKind,
    message: String,
    details: Map<String, Value>,
}

#[derive(Debug, Serialize)]
pub(super) struct ListEntryResponse {
    id: String,
    name: Option<String>,
    kind: &'static str,
    enabled: bool,
    tombstoned: bool,
    validation_state: &'static str,
    runtime_state: String,
    startup: Option<String>,
    last_error: Option<LastErrorResponse>,
    host_config: &'static str,
}

#[derive(Debug, Serialize)]
pub(super) struct LastErrorResponse {
    phase: String,
    code: String,
    message: String,
}

#[derive(Debug, Serialize)]
pub(super) struct InspectResponse {
    id: String,
    name: Option<String>,
    kind: &'static str,
    scope: &'static str,
    manifest_ref: String,
    plugins_toml_path: String,
    state_path: String,
    load: Value,
    compat: Value,
    capabilities: Vec<String>,
    metadata: Value,
    source: Value,
    spec: Value,
    status: Value,
    host_config_status: &'static str,
    host_config: Value,
}

#[derive(Debug, Serialize)]
pub(super) struct ValidateResponse {
    target_kind: &'static str,
    resolved_plugin_id: String,
    valid: bool,
    errors: Vec<String>,
    warnings: Vec<String>,
    notes: Vec<String>,
    manifest_ref: String,
    kind: &'static str,
    desired_enabled: Option<bool>,
    host_config_status: &'static str,
}

pub(super) fn print_response_json<T: Serialize>(value: &T) -> Result<(), CliError> {
    let rendered = serde_json::to_string_pretty(value).map_err(|error| {
        CliError::Config(format!("could not serialize plugin JSON output: {error}"))
    })?;
    println!("{rendered}");
    Ok(())
}

pub(super) fn list_success(
    command: &'static str,
    target: Option<&str>,
    records: &[ScopedDynamicPluginRecord],
    host_config_by_id: &HashMap<String, ResolvedDynamicPluginConfig>,
) -> ResponseEnvelope<Vec<ListEntryResponse>> {
    success(
        command,
        target,
        records
            .iter()
            .map(|entry| {
                let record = &entry.record;
                ListEntryResponse {
                    id: record.metadata.id.clone(),
                    name: record.metadata.name.clone(),
                    kind: manifest_kind_label(record.metadata.kind),
                    enabled: record.spec.enabled,
                    tombstoned: record.is_tombstoned(),
                    validation_state: check_state_label(record.status.validation.manifest),
                    runtime_state: if record.is_tombstoned() {
                        "tombstoned".into()
                    } else {
                        runtime_state_label(record.status.runtime.state).into()
                    },
                    startup: record
                        .status
                        .startup_class
                        .map(|value| format!("{value:?}").to_lowercase()),
                    last_error: record
                        .status
                        .last_error
                        .as_ref()
                        .map(|error| LastErrorResponse {
                            phase: failure_phase_label(error.phase).into(),
                            code: error.code.clone(),
                            message: error.message.clone(),
                        }),
                    host_config: host_config_by_id
                        .get(&record.metadata.id)
                        .map(|plugin| host_config_status_label(plugin.host_config_status()))
                        .unwrap_or("missing"),
                }
            })
            .collect(),
    )
}

pub(super) fn inspect_success(
    command: &'static str,
    target: &str,
    entry: &ScopedDynamicPluginRecord,
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    host_config: Option<&ResolvedDynamicPluginConfig>,
) -> ResponseEnvelope<InspectResponse> {
    let record = &entry.record;
    success(
        command,
        Some(target),
        InspectResponse {
            id: record.metadata.id.clone(),
            name: record.metadata.name.clone(),
            kind: manifest_kind_label(record.metadata.kind),
            scope: entry.scope.label(),
            manifest_ref: manifest_ref.into(),
            plugins_toml_path: entry.plugins_toml_path.display().to_string(),
            state_path: entry.state_path.display().to_string(),
            load: serde_json::to_value(&record.load)
                .expect("dynamic plugin load contract serializes to JSON"),
            compat: serde_json::to_value(&record.compatibility)
                .expect("dynamic plugin compatibility serializes to JSON"),
            capabilities: manifest
                .capabilities
                .items
                .iter()
                .map(|item| format!("{item:?}").to_lowercase())
                .collect(),
            metadata: serde_json::to_value(&record.metadata)
                .expect("dynamic plugin metadata serializes to JSON"),
            source: serde_json::to_value(&record.source)
                .expect("dynamic plugin source serializes to JSON"),
            spec: serde_json::to_value(&record.spec)
                .expect("dynamic plugin spec serializes to JSON"),
            status: serde_json::to_value(&record.status)
                .expect("dynamic plugin status serializes to JSON"),
            host_config_status: host_config
                .map(|plugin| host_config_status_label(plugin.host_config_status()))
                .unwrap_or("missing"),
            host_config: host_config
                .map(|plugin| Value::Object(plugin.config.clone()))
                .unwrap_or(Value::Null),
        },
    )
}

pub(super) fn validate_success(
    input: ValidateResponseInput<'_>,
) -> ResponseEnvelope<ValidateResponse> {
    let notes = input
        .entry
        .and_then(|entry| entry.record.status.validation.message.clone())
        .into_iter()
        .collect::<Vec<_>>();

    success(
        input.command,
        input.target,
        ValidateResponse {
            target_kind: input.target_kind,
            resolved_plugin_id: input
                .resolved_plugin_id
                .unwrap_or(input.manifest.plugin.id.as_str())
                .to_owned(),
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            notes,
            manifest_ref: input.manifest_ref.into(),
            kind: manifest_kind_label(input.manifest.plugin.kind),
            desired_enabled: input.entry.map(|entry| entry.record.spec.enabled),
            host_config_status: input
                .host_config
                .map(|plugin| host_config_status_label(plugin.host_config_status()))
                .unwrap_or("missing"),
        },
    )
}

pub(super) fn failure(
    command: &'static str,
    target: Option<&str>,
    kind: PluginLifecycleFailureKind,
    message: &str,
) -> ResponseEnvelope<Value> {
    ResponseEnvelope {
        schema_version: 1,
        ok: false,
        command,
        target: target.map(str::to_owned),
        warnings: Vec::new(),
        data: None,
        error: Some(ResponseError {
            code: failure_code(kind),
            kind,
            message: message.to_owned(),
            details: Map::new(),
        }),
    }
}

pub(super) fn generic_failure(
    command: &'static str,
    target: Option<&str>,
    message: &str,
) -> ResponseEnvelope<Value> {
    failure(command, target, PluginLifecycleFailureKind::Failed, message)
}

fn success<T: Serialize>(
    command: &'static str,
    target: Option<&str>,
    data: T,
) -> ResponseEnvelope<T> {
    ResponseEnvelope {
        schema_version: 1,
        ok: true,
        command,
        target: target.map(str::to_owned),
        warnings: Vec::new(),
        data: Some(data),
        error: None,
    }
}

fn host_config_status_label(status: DynamicPluginHostConfigStatus) -> &'static str {
    match status {
        DynamicPluginHostConfigStatus::Absent => "absent",
        DynamicPluginHostConfigStatus::Present => "present",
    }
}

fn failure_code(kind: PluginLifecycleFailureKind) -> &'static str {
    match kind {
        PluginLifecycleFailureKind::Failed => "operation_failed",
        PluginLifecycleFailureKind::NotFound => "not_found",
        PluginLifecycleFailureKind::Refused => "refused",
    }
}

fn failure_phase_label(phase: DynamicPluginFailurePhase) -> &'static str {
    match phase {
        DynamicPluginFailurePhase::Validation => "validation",
        DynamicPluginFailurePhase::Activation => "activation",
        DynamicPluginFailurePhase::Runtime => "runtime",
        DynamicPluginFailurePhase::Policy => "policy",
    }
}
