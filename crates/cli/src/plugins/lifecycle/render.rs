// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Human and machine-readable lifecycle result rendering.

use super::*;

pub(crate) fn render_plugin_error(
    error: &CliError,
    json: bool,
) -> Result<Option<ExitCode>, CliError> {
    let Some((command, target, kind, code, message)) = error.as_plugin_lifecycle_error_context()
    else {
        return Ok(None);
    };

    let exit_code = match kind {
        PluginLifecycleFailureKind::Failed => ExitCode::from(1),
        PluginLifecycleFailureKind::NotFound => ExitCode::from(2),
        PluginLifecycleFailureKind::Refused => ExitCode::from(3),
    };

    if json {
        print_response_json(&failure(command, target, kind, code, message))?;
    } else {
        eprintln!("{message}");
    }
    Ok(Some(exit_code))
}

pub(crate) fn render_generic_plugin_json_error(
    command: &'static str,
    target: Option<&str>,
    message: &str,
) -> Result<ExitCode, CliError> {
    print_response_json(&generic_failure(command, target, message))?;
    Ok(ExitCode::from(1))
}

pub(super) fn plugin_not_found(
    command: &'static str,
    target: Option<String>,
    message: impl Into<String>,
) -> CliError {
    CliError::PluginLifecycle {
        command,
        target,
        kind: PluginLifecycleFailureKind::NotFound,
        code: None,
        message: message.into(),
    }
}

pub(super) fn plugin_refused(
    command: &'static str,
    target: Option<String>,
    message: impl Into<String>,
) -> CliError {
    plugin_refused_with_code(command, target, "refused", message)
}

pub(super) fn plugin_refused_with_code(
    command: &'static str,
    target: Option<String>,
    code: &'static str,
    message: impl Into<String>,
) -> CliError {
    CliError::PluginLifecycle {
        command,
        target,
        kind: PluginLifecycleFailureKind::Refused,
        code: Some(code),
        message: message.into(),
    }
}

pub(super) fn plugin_failed_with_code(
    command: &'static str,
    target: Option<String>,
    code: &'static str,
    message: impl Into<String>,
) -> CliError {
    CliError::PluginLifecycle {
        command,
        target,
        kind: PluginLifecycleFailureKind::Failed,
        code: Some(code),
        message: message.into(),
    }
}

pub(super) fn trust_refusal_code(trust: &EvaluatedDynamicPluginTrust) -> &'static str {
    trust.refusal_code().unwrap_or("refused")
}

pub(super) fn list_validation_state(record: &DynamicPluginRecord) -> DynamicPluginCheckState {
    let validation = &record.status.validation;
    if validation.manifest == DynamicPluginCheckState::Invalid
        || validation.compatibility == DynamicPluginCheckState::Invalid
        || validation.integrity == DynamicPluginCheckState::Invalid
        || validation.environment == DynamicPluginCheckState::Invalid
        || validation.authenticity == DynamicPluginCheckState::Invalid
        || validation.policy_satisfied == DynamicPluginCheckState::Invalid
    {
        DynamicPluginCheckState::Invalid
    } else if validation.manifest == DynamicPluginCheckState::Unknown
        || validation.compatibility == DynamicPluginCheckState::Unknown
    {
        DynamicPluginCheckState::Unknown
    } else {
        DynamicPluginCheckState::Valid
    }
}

pub(super) struct PluginListView<'a> {
    pub(super) records: &'a [ScopedDynamicPluginRecord],
    pub(super) host_config_by_id: &'a HashMap<String, ResolvedDynamicPluginConfig>,
}

impl fmt::Display for PluginListView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let widths = PluginListWidths::from_records(self.records);

        write!(
            f,
            "{:<id_width$} {:<scope_width$} {:<enabled_width$} {:<state_width$} {:<validation_width$} {:<policy_width$} HOST CONFIG",
            "ID",
            "SCOPE",
            "ENABLED",
            "STATE",
            "VALIDATION",
            "POLICY",
            id_width = widths.id,
            scope_width = widths.scope,
            enabled_width = widths.enabled,
            state_width = widths.state,
            validation_width = widths.validation,
            policy_width = widths.policy,
        )?;
        for entry in self.records {
            let scope: &'static str = entry.scope.into();
            let validation: &'static str = list_validation_state(&entry.record).into();
            let policy: &'static str = entry.record.status.validation.policy_satisfied.into();
            write!(
                f,
                "\n{:<id_width$} {:<scope_width$} {:<enabled_width$} {:<state_width$} {:<validation_width$} {:<policy_width$} {}",
                entry.record.metadata.id,
                scope,
                entry.record.spec.enabled,
                lifecycle_state_label(&entry.record),
                validation,
                policy,
                host_config_label(self.host_config_by_id.get(&entry.record.metadata.id)),
                id_width = widths.id,
                scope_width = widths.scope,
                enabled_width = widths.enabled,
                state_width = widths.state,
                validation_width = widths.validation,
                policy_width = widths.policy,
            )?;
        }
        Ok(())
    }
}

struct PluginListWidths {
    id: usize,
    scope: usize,
    enabled: usize,
    state: usize,
    validation: usize,
    policy: usize,
}

impl PluginListWidths {
    fn from_records(records: &[ScopedDynamicPluginRecord]) -> Self {
        Self {
            id: column_width(
                "ID",
                records
                    .iter()
                    .map(|entry| entry.record.metadata.id.as_str()),
            ),
            scope: column_width(
                "SCOPE",
                records.iter().map(|entry| {
                    let scope: &'static str = entry.scope.into();
                    scope
                }),
            ),
            enabled: column_width(
                "ENABLED",
                records.iter().map(|entry| {
                    if entry.record.spec.enabled {
                        "true"
                    } else {
                        "false"
                    }
                }),
            ),
            state: column_width(
                "STATE",
                records
                    .iter()
                    .map(|entry| lifecycle_state_label(&entry.record)),
            ),
            validation: column_width(
                "VALIDATION",
                records.iter().map(|entry| {
                    let validation: &'static str = list_validation_state(&entry.record).into();
                    validation
                }),
            ),
            policy: column_width(
                "POLICY",
                records.iter().map(|entry| {
                    let policy: &'static str =
                        entry.record.status.validation.policy_satisfied.into();
                    policy
                }),
            ),
        }
    }
}

pub(super) fn column_width<'a>(
    header: &'static str,
    values: impl Iterator<Item = &'a str>,
) -> usize {
    values
        .map(str::len)
        .chain(std::iter::once(header.len()))
        .max()
        .unwrap_or(header.len())
}

pub(super) struct PluginInspectView<'a> {
    pub(super) entry: &'a ScopedDynamicPluginRecord,
    pub(super) manifest: &'a DynamicPluginManifest,
    pub(super) manifest_ref: &'a str,
    pub(super) host_config: Option<&'a ResolvedDynamicPluginConfig>,
}

impl fmt::Display for PluginInspectView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let view = inspect_data(
            self.entry,
            self.manifest,
            self.manifest_ref,
            self.host_config,
        );
        let yaml = serde_yaml::to_string(&view).map_err(|_| fmt::Error)?;
        write!(f, "{}", yaml.trim_end())
    }
}

pub(super) struct PluginValidationSummaryView<'a> {
    pub(super) manifest: &'a DynamicPluginManifest,
    pub(super) manifest_ref: &'a str,
    pub(super) entry: Option<&'a ScopedDynamicPluginRecord>,
    pub(super) host_config: Option<&'a ResolvedDynamicPluginConfig>,
    pub(super) policy: &'a EvaluatedDynamicPluginHostPolicy,
    pub(super) trust: &'a EvaluatedDynamicPluginTrust,
}

impl fmt::Display for PluginValidationSummaryView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let environment = self
            .entry
            .map(|entry| entry.record.status.validation.environment)
            .unwrap_or(DynamicPluginCheckState::Unknown);
        if self.policy.policy_satisfied
            && self.trust.is_satisfied()
            && environment != DynamicPluginCheckState::Invalid
        {
            writeln!(f, "Dynamic plugin '{}' is valid.", self.manifest.plugin.id)?;
        } else if self.policy.policy_satisfied
            && self.trust.is_satisfied()
            && environment == DynamicPluginCheckState::Invalid
        {
            writeln!(
                f,
                "Dynamic plugin '{}' manifest is valid, but its runtime environment is unavailable.",
                self.manifest.plugin.id
            )?;
        } else if self.policy.policy_satisfied {
            writeln!(
                f,
                "Dynamic plugin '{}' manifest is valid, but trust verification blocks it.",
                self.manifest.plugin.id
            )?;
        } else {
            writeln!(
                f,
                "Dynamic plugin '{}' manifest is valid, but host policy blocks it.",
                self.manifest.plugin.id
            )?;
        }
        writeln!(f, "kind: {}", self.manifest.plugin.kind)?;
        writeln!(
            f,
            "policy_state: {}",
            <&'static str>::from(self.policy.check_state())
        )?;
        writeln!(
            f,
            "integrity_state: {}",
            <&'static str>::from(self.trust.integrity)
        )?;
        writeln!(
            f,
            "environment_state: {}",
            <&'static str>::from(environment)
        )?;
        writeln!(
            f,
            "authenticity_state: {}",
            <&'static str>::from(self.trust.authenticity)
        )?;
        writeln!(f, "startup_class: {}", self.policy.startup_class)?;
        writeln!(f, "attestation_mode: {}", self.policy.attestation_mode)?;
        if let Some(failure) = self.policy.failure() {
            writeln!(
                f,
                "policy_error: {}",
                failure.display(&self.manifest.plugin.id)
            )?;
        }
        if let Some(failure) = self.trust.failure() {
            writeln!(
                f,
                "trust_error: {}",
                failure.display(&self.manifest.plugin.id)
            )?;
        }
        if let Some(entry) = self.entry {
            writeln!(f, "manifest: {}", self.manifest_ref)?;
            writeln!(f, "scope: {}", entry.scope)?;
            writeln!(f, "lifecycle_state_path: {}", entry.state_path.display())?;
            writeln!(f, "desired.enabled: {}", entry.record.spec.enabled)?;
            write!(f, "host_config: {}", host_config_label(self.host_config))?;
        } else {
            write!(f, "manifest: {}", self.manifest_ref)?;
        }
        Ok(())
    }
}

pub(super) fn lifecycle_state_label(record: &DynamicPluginRecord) -> &'static str {
    if record.is_tombstoned() {
        "tombstoned"
    } else {
        record.status.runtime.state.into()
    }
}

pub(super) fn host_config_label(host_config: Option<&ResolvedDynamicPluginConfig>) -> &'static str {
    host_config
        .map(|plugin| {
            let status: &'static str = plugin.host_config_status().into();
            status
        })
        .unwrap_or("absent")
}

pub(super) fn redacted_host_config_json(host_config: &ResolvedDynamicPluginConfig) -> Value {
    if host_config.config.is_empty() && !host_config.has_explicit_config {
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

pub(super) fn inspect_load_data(record: &DynamicPluginRecord) -> Value {
    match &record.load {
        DynamicPluginLoadContract::Worker(load) => serde_json::json!({
            "runtime": load.runtime,
            "entrypoint": load.entrypoint,
        }),
        DynamicPluginLoadContract::RustDynamic(load) => serde_json::json!({
            "library": load.library,
            "symbol": load.symbol,
        }),
    }
}

pub(super) fn inspect_compat_data(record: &DynamicPluginRecord) -> Value {
    match &record.compatibility {
        DynamicPluginCompatibility::Worker(compatibility) => serde_json::json!({
            "relay": compatibility.relay,
            "worker_protocol": compatibility.worker_protocol,
        }),
        DynamicPluginCompatibility::RustDynamic(compatibility) => serde_json::json!({
            "relay": compatibility.relay,
            "native_api": compatibility.native_api,
        }),
    }
}
