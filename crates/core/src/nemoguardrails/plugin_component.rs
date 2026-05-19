// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NeMo Guardrails plugin component contract.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

use crate::plugin::{
    ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, Plugin, PluginComponentSpec, PluginError,
    PluginRegistrationContext, Result as PluginResult, UnsupportedBehavior, deregister_plugin,
    lookup_plugin, register_plugin,
};

/// The plugin kind reserved for the planned first-party component.
pub const NEMO_GUARDRAILS_PLUGIN_KIND: &str = "nemoguardrails";

/// Top-level NeMo Guardrails component wrapper.
#[derive(Debug, Clone)]
pub struct ComponentSpec {
    /// Whether the component should be activated.
    pub enabled: bool,
    /// Component-local NeMo Guardrails config.
    pub config: NeMoGuardrailsConfig,
}

impl ComponentSpec {
    /// Creates an enabled NeMo Guardrails component spec.
    pub fn new(config: NeMoGuardrailsConfig) -> Self {
        Self {
            enabled: true,
            config,
        }
    }
}

impl From<ComponentSpec> for PluginComponentSpec {
    fn from(value: ComponentSpec) -> Self {
        let Json::Object(config) = serde_json::to_value(value.config)
            .expect("NeMo Guardrails config should serialize to an object")
        else {
            unreachable!("NeMo Guardrails config must serialize to an object");
        };

        PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: value.enabled,
            config,
        }
    }
}

/// Canonical config document for the planned NeMo Guardrails component.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct NeMoGuardrailsConfig {
    /// NeMo Guardrails config schema version.
    #[serde(default = "default_nemoguardrails_config_version")]
    pub version: u32,
    /// Backend mode: `remote` or `local`.
    #[serde(default = "default_mode")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "mode_schema"))]
    pub mode: String,
    /// Path to a native NeMo Guardrails config directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    /// Inline native NeMo Guardrails YAML config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_yaml: Option<String>,
    /// Optional inline Colang content. Valid only with `config_yaml`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colang_content: Option<String>,
    /// Provider request/response codec for LLM-managed surfaces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "codec_schema"))]
    pub codec: Option<String>,
    /// Whether to run input rails around managed LLM execution.
    #[serde(default = "default_true")]
    pub input: bool,
    /// Whether to run output rails around managed LLM execution.
    #[serde(default = "default_true")]
    pub output: bool,
    /// Whether to run tool-input rails around managed tool execution.
    #[serde(default)]
    pub tool_input: bool,
    /// Whether to run tool-output rails around managed tool execution.
    #[serde(default)]
    pub tool_output: bool,
    /// Intercept priority. Lower values run earlier.
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Remote-backend settings used when `mode = "remote"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote: Option<RemoteBackendConfig>,
    /// Local-backend settings used when `mode = "local"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local: Option<LocalBackendConfig>,
    /// Component-local unsupported-config policy.
    #[serde(default)]
    pub policy: ConfigPolicy,
}

impl Default for NeMoGuardrailsConfig {
    fn default() -> Self {
        Self {
            version: default_nemoguardrails_config_version(),
            mode: default_mode(),
            config_path: None,
            config_yaml: None,
            colang_content: None,
            codec: None,
            input: true,
            output: true,
            tool_input: false,
            tool_output: false,
            priority: default_priority(),
            remote: None,
            local: None,
            policy: ConfigPolicy::default(),
        }
    }
}

/// Remote-backend settings for a hosted NeMo Guardrails service.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RemoteBackendConfig {
    /// Base URL for the remote Guardrails service.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// One remote Guardrails config identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_id: Option<String>,
    /// Multiple remote Guardrails config identifiers to combine.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config_ids: Vec<String>,
    /// Static request headers sent to the remote service.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Request timeout in milliseconds.
    #[serde(default = "default_timeout_millis")]
    pub timeout_millis: u64,
}

impl Default for RemoteBackendConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            config_id: None,
            config_ids: vec![],
            headers: HashMap::new(),
            timeout_millis: default_timeout_millis(),
        }
    }
}

/// Local-backend settings for the Python `nemoguardrails` runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LocalBackendConfig {
    /// Optional import path for the Python runtime module.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub python_module: Option<String>,
}

crate::editor_config! {
    impl NeMoGuardrailsConfig {
        mode => {
            label: "mode",
            kind: Enum,
            values: ["remote", "local"],
        },
        config_path => { label: "config_path", kind: String, optional: true },
        config_yaml => { label: "config_yaml", kind: String, optional: true },
        colang_content => { label: "colang_content", kind: String, optional: true },
        codec => {
            label: "codec",
            kind: Enum,
            values: ["openai_chat", "openai_responses", "anthropic_messages"],
            optional: true,
        },
        input => { label: "input", kind: Boolean },
        output => { label: "output", kind: Boolean },
        tool_input => { label: "tool_input", kind: Boolean },
        tool_output => { label: "tool_output", kind: Boolean },
        priority => { label: "priority", kind: Integer },
        remote => {
            label: "remote",
            kind: Section,
            optional: true,
            nested: RemoteBackendConfig,
            default: RemoteBackendConfig,
        },
        local => {
            label: "local",
            kind: Section,
            optional: true,
            nested: LocalBackendConfig,
            default: LocalBackendConfig,
        },
        policy => {
            label: "policy",
            kind: Section,
            nested: ConfigPolicy,
            default: ConfigPolicy,
        },
    }
}

crate::editor_config! {
    impl RemoteBackendConfig {
        endpoint => { label: "endpoint", kind: String, optional: true },
        config_id => { label: "config_id", kind: String, optional: true },
        config_ids => { label: "config_ids", kind: Json },
        headers => { label: "headers", kind: StringMap },
        timeout_millis => { label: "timeout_millis", kind: Integer },
    }
}

crate::editor_config! {
    impl LocalBackendConfig {
        python_module => { label: "python_module", kind: String, optional: true },
    }
}

struct NeMoGuardrailsPlugin;

impl Plugin for NeMoGuardrailsPlugin {
    fn plugin_kind(&self) -> &str {
        NEMO_GUARDRAILS_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        validate_nemoguardrails_plugin_config(plugin_config)
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        _ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        Box::pin(async {
            Err(PluginError::RegistrationFailed(
                "built-in NeMo Guardrails plugin backend is not implemented yet".to_string(),
            ))
        })
    }
}

/// Registers the `nemoguardrails` component kind in the plugin registry.
pub fn register_nemoguardrails_component() -> PluginResult<()> {
    match register_plugin(Arc::new(NeMoGuardrailsPlugin)) {
        Ok(()) => Ok(()),
        Err(PluginError::RegistrationFailed(message))
            if message.contains("already registered")
                && lookup_plugin(NEMO_GUARDRAILS_PLUGIN_KIND).is_some() =>
        {
            Ok(())
        }
        Err(err) => Err(err),
    }
}

/// Deregisters the `nemoguardrails` component kind from the plugin registry.
pub fn deregister_nemoguardrails_component() -> bool {
    deregister_plugin(NEMO_GUARDRAILS_PLUGIN_KIND)
}

/// Returns the JSON Schema for the NeMo Guardrails component configuration.
#[cfg(feature = "schema")]
pub fn nemoguardrails_config_schema() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(NeMoGuardrailsConfig))
        .expect("NeMo Guardrails config schema should serialize")
}

#[cfg(feature = "schema")]
fn mode_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
    string_enum_schema(generator, &["remote", "local"], Some("remote"))
}

#[cfg(feature = "schema")]
fn codec_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
    string_enum_schema(
        generator,
        &["openai_chat", "openai_responses", "anthropic_messages"],
        None,
    )
}

#[cfg(feature = "schema")]
fn string_enum_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
    values: &[&str],
    default: Option<&str>,
) -> schemars::schema::Schema {
    let mut schema: schemars::schema::SchemaObject =
        <String as schemars::JsonSchema>::json_schema(generator).into();
    schema.enum_values = Some(
        values
            .iter()
            .map(|value| Json::String((*value).into()))
            .collect(),
    );
    if let Some(default) = default {
        schema.metadata().default = Some(Json::String(default.into()));
    }
    schema.into()
}

fn parse_nemoguardrails_config(
    plugin_config: &Map<String, Json>,
) -> PluginResult<NeMoGuardrailsConfig> {
    serde_json::from_value(Json::Object(plugin_config.clone())).map_err(|err| {
        PluginError::InvalidConfig(format!("invalid NeMo Guardrails plugin config: {err}"))
    })
}

fn validate_nemoguardrails_plugin_config(
    plugin_config: &Map<String, Json>,
) -> Vec<ConfigDiagnostic> {
    let config = match parse_nemoguardrails_config(plugin_config) {
        Ok(config) => config,
        Err(err) => {
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "nemoguardrails.invalid_plugin_config".to_string(),
                component: Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                field: None,
                message: err.to_string(),
            }];
        }
    };

    let mut diagnostics = vec![];

    validate_unknown_fields(
        &mut diagnostics,
        &config.policy,
        Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
        plugin_config,
        &[
            "version",
            "mode",
            "config_path",
            "config_yaml",
            "colang_content",
            "codec",
            "input",
            "output",
            "tool_input",
            "tool_output",
            "priority",
            "remote",
            "local",
            "policy",
        ],
    );

    validate_policy_fields(&mut diagnostics, &config.policy, plugin_config);
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "remote",
        &[
            "endpoint",
            "config_id",
            "config_ids",
            "headers",
            "timeout_millis",
        ],
    );
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "local",
        &["python_module"],
    );

    validate_version(&mut diagnostics, &config.policy, config.version);
    validate_mode(&mut diagnostics, &config.policy, &config.mode);
    validate_non_empty_strings(&mut diagnostics, &config.policy, &config);
    validate_config_shape(&mut diagnostics, &config.policy, &config);
    validate_codec_requirements(&mut diagnostics, &config.policy, &config);
    validate_surface_selection(&mut diagnostics, &config.policy, &config);

    diagnostics
}

fn validate_version(diagnostics: &mut Vec<ConfigDiagnostic>, policy: &ConfigPolicy, version: u32) {
    if version != 1 {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_config_version",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("version".to_string()),
            format!("NeMo Guardrails config version {version} is unsupported"),
        );
    }
}

fn validate_mode(diagnostics: &mut Vec<ConfigDiagnostic>, policy: &ConfigPolicy, mode: &str) {
    if !matches!(mode, "remote" | "local") {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("mode".to_string()),
            "mode must be 'remote' or 'local'".to_string(),
        );
    }
}

fn validate_non_empty_strings(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    if let Some(config_path) = &config.config_path
        && config_path.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("config_path".to_string()),
            "config_path must not be empty".to_string(),
        );
    }

    if let Some(config_yaml) = &config.config_yaml
        && config_yaml.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("config_yaml".to_string()),
            "config_yaml must not be empty".to_string(),
        );
    }

    if let Some(colang_content) = &config.colang_content
        && colang_content.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("colang_content".to_string()),
            "colang_content must not be empty".to_string(),
        );
    }

    if let Some(remote) = &config.remote {
        if let Some(endpoint) = &remote.endpoint
            && endpoint.trim().is_empty()
        {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemoguardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("remote.endpoint".to_string()),
                "remote.endpoint must not be empty".to_string(),
            );
        }
        if let Some(config_id) = &remote.config_id
            && config_id.trim().is_empty()
        {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemoguardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("remote.config_id".to_string()),
                "remote.config_id must not be empty".to_string(),
            );
        }
        for (index, config_id) in remote.config_ids.iter().enumerate() {
            if config_id.trim().is_empty() {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemoguardrails.unsupported_value",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    Some(format!("remote.config_ids[{index}]")),
                    "remote.config_ids entries must not be empty".to_string(),
                );
            }
        }
    }

    if let Some(local) = &config.local
        && let Some(python_module) = &local.python_module
        && python_module.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("local.python_module".to_string()),
            "local.python_module must not be empty".to_string(),
        );
    }
}

fn validate_config_shape(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    let has_config_path = config.config_path.is_some();
    let has_config_yaml = config.config_yaml.is_some();
    let has_colang_content = config.colang_content.is_some();
    let has_remote_config_id = config
        .remote
        .as_ref()
        .and_then(|remote| remote.config_id.as_ref())
        .is_some();
    let has_remote_config_ids = config
        .remote
        .as_ref()
        .map(|remote| !remote.config_ids.is_empty())
        .unwrap_or(false);

    match config.mode.as_str() {
        "local" => {
            if has_config_path == has_config_yaml {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemoguardrails.invalid_config_source",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    None,
                    "exactly one of config_path or config_yaml is required in local mode"
                        .to_string(),
                );
            }

            if has_colang_content && !has_config_yaml {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemoguardrails.unsupported_value",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    Some("colang_content".to_string()),
                    "colang_content can only be used with config_yaml".to_string(),
                );
            }

            if config.remote.is_some() {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemoguardrails.unsupported_value",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    Some("remote".to_string()),
                    "remote backend settings cannot be used when mode is 'local'".to_string(),
                );
            }
        }
        "remote" => {
            if has_config_path || has_config_yaml || has_colang_content {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemoguardrails.invalid_config_source",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    None,
                    "remote mode uses remote config identity and cannot include config_path, config_yaml, or colang_content".to_string(),
                );
            }

            if config.local.is_some() {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemoguardrails.unsupported_value",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    Some("local".to_string()),
                    "local backend settings cannot be used when mode is 'remote'".to_string(),
                );
            }

            match &config.remote {
                Some(remote)
                    if remote
                        .endpoint
                        .as_ref()
                        .is_some_and(|value| !value.trim().is_empty()) => {}
                _ => push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemoguardrails.unsupported_value",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    Some("remote.endpoint".to_string()),
                    "remote.endpoint is required when mode is 'remote'".to_string(),
                ),
            }

            if has_remote_config_id && has_remote_config_ids {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemoguardrails.unsupported_value",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    Some("remote".to_string()),
                    "remote.config_id and remote.config_ids cannot be used together".to_string(),
                );
            }

            if !(has_remote_config_id || has_remote_config_ids) {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemoguardrails.invalid_config_source",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    None,
                    "remote mode requires remote.config_id or remote.config_ids".to_string(),
                );
            }
        }
        _ => {}
    }
}

fn validate_codec_requirements(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    let llm_surface_enabled = config.input || config.output;
    if !llm_surface_enabled {
        return;
    }

    let Some(codec) = config.codec.as_deref() else {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("codec".to_string()),
            "codec is required when any LLM surface is enabled".to_string(),
        );
        return;
    };

    if !matches!(
        codec,
        "openai_chat" | "openai_responses" | "anthropic_messages"
    ) {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("codec".to_string()),
            "codec must be 'openai_chat', 'openai_responses', or 'anthropic_messages'".to_string(),
        );
    }
}

fn validate_surface_selection(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    if config.input || config.output || config.tool_input || config.tool_output {
        return;
    }

    push_policy_diag(
        diagnostics,
        policy.unsupported_value,
        "nemoguardrails.unsupported_value",
        Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
        None,
        "at least one Guardrails surface must be enabled".to_string(),
    );
}

fn validate_policy_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
) {
    if let Some(policy_json) = plugin_config.get("policy").and_then(Json::as_object) {
        validate_unknown_fields(
            diagnostics,
            policy,
            Some("policy".to_string()),
            policy_json,
            &["unknown_component", "unknown_field", "unsupported_value"],
        );
    }
}

fn validate_section_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
    section: &str,
    known_fields: &[&str],
) {
    if let Some(section_json) = plugin_config.get(section).and_then(Json::as_object) {
        validate_unknown_fields(
            diagnostics,
            policy,
            Some(section.to_string()),
            section_json,
            known_fields,
        );
    }
}

fn validate_unknown_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    component: Option<String>,
    config: &Map<String, Json>,
    known_fields: &[&str],
) {
    for field in config.keys() {
        if !known_fields.contains(&field.as_str()) {
            push_policy_diag(
                diagnostics,
                policy.unknown_field,
                "nemoguardrails.unknown_field",
                component.clone(),
                Some(field.clone()),
                format!(
                    "field '{}' is not recognized for '{}'",
                    field,
                    component.as_deref().unwrap_or("unknown")
                ),
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

fn default_nemoguardrails_config_version() -> u32 {
    1
}

fn default_mode() -> String {
    "remote".to_string()
}

fn default_true() -> bool {
    true
}

fn default_priority() -> i32 {
    100
}

fn default_timeout_millis() -> u64 {
    3_000
}

#[cfg(test)]
#[path = "../../tests/unit/nemoguardrails/plugin_component_tests.rs"]
mod tests;
