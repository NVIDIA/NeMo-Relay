// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Machine-readable response layer for dynamic plugin lifecycle commands.
//!
//! This module owns the versioned response contract for
//! `plugins list`, `plugins inspect`, `plugins validate`, and structured
//! lifecycle errors. Command logic lives in `lifecycle.rs`; this file only
//! turns already-resolved state into stable responses serialized as JSON.

use std::collections::HashMap;

use nemo_relay::plugin::dynamic::{
    DynamicPluginAttestationMode, DynamicPluginCheckState, DynamicPluginFailurePhase,
    DynamicPluginKind, DynamicPluginManifest, DynamicPluginStartupClass,
};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::config::{DynamicPluginHostConfigStatus, ResolvedDynamicPluginConfig};
use crate::error::{CliError, PluginLifecycleFailureKind};
use crate::plugins::policy::EvaluatedDynamicPluginHostPolicy;

use super::state::ScopedDynamicPluginRecord;
use super::{
    EvaluatedDynamicPluginTrust, inspect_compat_data, inspect_load_data, redacted_host_config_json,
};

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
    pub(super) policy: &'a EvaluatedDynamicPluginHostPolicy,
    pub(super) trust: &'a EvaluatedDynamicPluginTrust,
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
    kind: DynamicPluginKind,
    enabled: bool,
    tombstoned: bool,
    validation_state: DynamicPluginCheckState,
    policy_state: DynamicPluginCheckState,
    runtime_state: String,
    startup_class: Option<DynamicPluginStartupClass>,
    attestation_mode: Option<DynamicPluginAttestationMode>,
    last_error: Option<LastErrorResponse>,
    host_config: DynamicPluginHostConfigStatus,
}

#[derive(Debug, Serialize)]
pub(super) struct LastErrorResponse {
    phase: DynamicPluginFailurePhase,
    code: String,
    message: String,
}

#[derive(Debug, Serialize)]
pub(super) struct InspectResponse {
    id: String,
    name: Option<String>,
    kind: DynamicPluginKind,
    scope: super::state::RegistryScope,
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
    policy_state: DynamicPluginCheckState,
    startup_class: Option<DynamicPluginStartupClass>,
    attestation_mode: Option<DynamicPluginAttestationMode>,
    host_config_status: DynamicPluginHostConfigStatus,
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
    kind: DynamicPluginKind,
    policy_state: DynamicPluginCheckState,
    integrity_state: DynamicPluginCheckState,
    authenticity_state: DynamicPluginCheckState,
    startup_class: DynamicPluginStartupClass,
    attestation_mode: DynamicPluginAttestationMode,
    desired_enabled: Option<bool>,
    host_config_status: DynamicPluginHostConfigStatus,
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
                    kind: record.metadata.kind,
                    enabled: record.spec.enabled,
                    tombstoned: record.is_tombstoned(),
                    validation_state: record.status.validation.manifest,
                    policy_state: record.status.validation.policy_satisfied,
                    runtime_state: if record.is_tombstoned() {
                        "tombstoned".into()
                    } else {
                        <&'static str>::from(record.status.runtime.state).into()
                    },
                    startup_class: record.status.startup_class,
                    attestation_mode: record.status.attestation_mode,
                    last_error: record
                        .status
                        .last_error
                        .as_ref()
                        .map(|error| LastErrorResponse {
                            phase: error.phase,
                            code: error.code.clone(),
                            message: error.message.clone(),
                        }),
                    host_config: host_config_by_id
                        .get(&record.metadata.id)
                        .map(ResolvedDynamicPluginConfig::host_config_status)
                        .unwrap_or(DynamicPluginHostConfigStatus::Absent),
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
    success(
        command,
        Some(target),
        inspect_data(entry, manifest, manifest_ref, host_config),
    )
}

pub(super) fn inspect_data(
    entry: &ScopedDynamicPluginRecord,
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    host_config: Option<&ResolvedDynamicPluginConfig>,
) -> InspectResponse {
    let record = &entry.record;
    InspectResponse {
        id: record.metadata.id.clone(),
        name: record.metadata.name.clone(),
        kind: record.metadata.kind,
        scope: entry.scope,
        manifest_ref: manifest_ref.into(),
        plugins_toml_path: entry.plugins_toml_path.display().to_string(),
        state_path: entry.state_path.display().to_string(),
        load: inspect_load_data(record),
        compat: inspect_compat_data(record),
        capabilities: manifest
            .capabilities
            .items
            .iter()
            .map(ToString::to_string)
            .collect(),
        metadata: serde_json::to_value(&record.metadata)
            .expect("dynamic plugin metadata serializes to JSON"),
        source: serde_json::to_value(&record.source)
            .expect("dynamic plugin source serializes to JSON"),
        spec: serde_json::to_value(&record.spec).expect("dynamic plugin spec serializes to JSON"),
        status: serde_json::to_value(&record.status)
            .expect("dynamic plugin status serializes to JSON"),
        policy_state: record.status.validation.policy_satisfied,
        startup_class: record.status.startup_class,
        attestation_mode: record.status.attestation_mode,
        host_config_status: host_config
            .map(ResolvedDynamicPluginConfig::host_config_status)
            .unwrap_or(DynamicPluginHostConfigStatus::Absent),
        host_config: host_config
            .map(redacted_host_config_json)
            .unwrap_or(Value::Null),
    }
}

pub(super) fn validate_success(
    input: ValidateResponseInput<'_>,
) -> ResponseEnvelope<ValidateResponse> {
    let notes = input
        .entry
        .and_then(|entry| entry.record.status.validation.message.clone())
        .into_iter()
        .collect::<Vec<_>>();
    let valid = input.policy.policy_satisfied && input.trust.message.is_none();
    let errors = input
        .policy
        .message
        .iter()
        .chain(input.trust.message.iter())
        .cloned()
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
            valid,
            errors,
            warnings: Vec::new(),
            notes,
            manifest_ref: input.manifest_ref.into(),
            kind: input.manifest.plugin.kind,
            policy_state: input.policy.check_state(),
            integrity_state: input.trust.integrity,
            authenticity_state: input.trust.authenticity,
            startup_class: input.policy.startup_class,
            attestation_mode: input.policy.attestation_mode,
            desired_enabled: input.entry.map(|entry| entry.record.spec.enabled),
            host_config_status: input
                .host_config
                .map(ResolvedDynamicPluginConfig::host_config_status)
                .unwrap_or(DynamicPluginHostConfigStatus::Absent),
        },
    )
}

pub(super) fn failure(
    command: &'static str,
    target: Option<&str>,
    kind: PluginLifecycleFailureKind,
    code: Option<&'static str>,
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
            code: code.unwrap_or_else(|| failure_code(kind)),
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
    failure(
        command,
        target,
        PluginLifecycleFailureKind::Failed,
        None,
        message,
    )
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

fn failure_code(kind: PluginLifecycleFailureKind) -> &'static str {
    match kind {
        PluginLifecycleFailureKind::Failed => "operation_failed",
        PluginLifecycleFailureKind::NotFound => "not_found",
        PluginLifecycleFailureKind::Refused => "refused",
    }
}
