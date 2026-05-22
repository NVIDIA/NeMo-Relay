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

#[cfg(all(feature = "guardrails-remote", not(target_arch = "wasm32")))]
#[path = "remote.rs"]
mod remote;
#[cfg(all(feature = "guardrails-remote", not(target_arch = "wasm32")))]
use remote::register_remote_backend;

/// The plugin kind reserved for the planned first-party component.
pub const NEMO_GUARDRAILS_PLUGIN_KIND: &str = "nemo_guardrails";

#[cfg(any(target_arch = "wasm32", not(feature = "guardrails-remote")))]
fn register_remote_backend(
    _config: NeMoGuardrailsConfig,
    _ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    Err(PluginError::RegistrationFailed(
        "built-in NeMo Guardrails remote backend is unavailable in this build".to_string(),
    ))
}

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
    #[serde(default = "default_nemo_guardrails_config_version")]
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
    /// Default request semantics passed through to the selected Guardrails backend.
    ///
    /// This models request-time concepts such as rail selection and generation
    /// options without claiming backend parity for every Guardrails feature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_defaults: Option<RequestDefaultsConfig>,
    /// Component-local unsupported-config policy.
    #[serde(default)]
    pub policy: ConfigPolicy,
}

impl Default for NeMoGuardrailsConfig {
    fn default() -> Self {
        Self {
            version: default_nemo_guardrails_config_version(),
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
            request_defaults: None,
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

/// Default request semantics applied by the selected Guardrails backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RequestDefaultsConfig {
    /// Default context object passed into Guardrails requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Json>,
    /// Default remote thread identifier for continuation-aware requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    /// Default remote Guardrails state payload for continuation-aware requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<Json>,
    /// Default request-time rail selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rails: Option<RequestRailsConfig>,
    /// Default model parameters applied to Guardrails-backed LLM calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_params: Option<Json>,
    /// Whether to include raw LLM output in Guardrails responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_output: Option<bool>,
    /// Default output variables selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_vars: Option<Json>,
    /// Default generation-log selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log: Option<Json>,
}

/// Request-time rail selection for Guardrails generation.
///
/// These are backend request options, not top-level NeMo Relay interception
/// surfaces.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RequestRailsConfig {
    /// Input rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<RailSelector>,
    /// Output rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<RailSelector>,
    /// Retrieval rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval: Option<RailSelector>,
    /// Dialog rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialog: Option<bool>,
    /// Tool-output rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_output: Option<RailSelector>,
    /// Tool-input rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_input: Option<RailSelector>,
}

/// Rail-selection shape used by Guardrails generation options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum RailSelector {
    /// Enable or disable the whole rail family.
    Enabled(bool),
    /// Enable only named rails within a family.
    Named(Vec<String>),
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
        request_defaults => {
            label: "request_defaults",
            kind: Section,
            optional: true,
            nested: RequestDefaultsConfig,
            default: RequestDefaultsConfig,
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

crate::editor_config! {
    impl RequestDefaultsConfig {
        context => { label: "context", kind: Json, optional: true },
        thread_id => { label: "thread_id", kind: String, optional: true },
        state => { label: "state", kind: Json, optional: true },
        rails => {
            label: "rails",
            kind: Section,
            optional: true,
            nested: RequestRailsConfig,
            default: RequestRailsConfig,
        },
        llm_params => { label: "llm_params", kind: Json, optional: true },
        llm_output => { label: "llm_output", kind: Boolean, optional: true },
        output_vars => { label: "output_vars", kind: Json, optional: true },
        log => { label: "log", kind: Json, optional: true },
    }
}

crate::editor_config! {
    impl RequestRailsConfig {
        input => { label: "input", kind: Json, optional: true },
        output => { label: "output", kind: Json, optional: true },
        retrieval => { label: "retrieval", kind: Json, optional: true },
        dialog => { label: "dialog", kind: Boolean, optional: true },
        tool_output => { label: "tool_output", kind: Json, optional: true },
        tool_input => { label: "tool_input", kind: Json, optional: true },
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
        validate_nemo_guardrails_plugin_config(plugin_config)
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        let parsed = parse_nemo_guardrails_config(plugin_config);
        Box::pin(async move {
            let config = parsed?;
            register_nemo_guardrails_backend(config, ctx)
        })
    }
}

/// Registers the `nemo_guardrails` component kind in the plugin registry.
pub fn register_nemo_guardrails_component() -> PluginResult<()> {
    match register_plugin(Arc::new(NeMoGuardrailsPlugin)) {
        Ok(()) => Ok(()),
        Err(PluginError::RegistrationFailed(_))
            if lookup_plugin(NEMO_GUARDRAILS_PLUGIN_KIND).is_some() =>
        {
            Ok(())
        }
        Err(err) => Err(err),
    }
}

/// Deregisters the `nemo_guardrails` component kind from the plugin registry.
pub fn deregister_nemo_guardrails_component() -> bool {
    deregister_plugin(NEMO_GUARDRAILS_PLUGIN_KIND)
}

/// Returns the JSON Schema for the NeMo Guardrails component configuration.
#[cfg(feature = "schema")]
pub fn nemo_guardrails_config_schema() -> serde_json::Value {
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

fn register_nemo_guardrails_backend(
    config: NeMoGuardrailsConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    match config.mode.as_str() {
        "remote" => register_remote_backend(config, ctx),
        "local" => Err(PluginError::RegistrationFailed(
            "built-in NeMo Guardrails local backend is not implemented yet".to_string(),
        )),
        other => Err(PluginError::InvalidConfig(format!(
            "unsupported NeMo Guardrails mode '{other}'"
        ))),
    }
}

fn parse_nemo_guardrails_config(
    plugin_config: &Map<String, Json>,
) -> PluginResult<NeMoGuardrailsConfig> {
    serde_json::from_value(Json::Object(plugin_config.clone())).map_err(|err| {
        PluginError::InvalidConfig(format!("invalid NeMo Guardrails plugin config: {err}"))
    })
}

fn validate_nemo_guardrails_plugin_config(
    plugin_config: &Map<String, Json>,
) -> Vec<ConfigDiagnostic> {
    let config = match parse_nemo_guardrails_config(plugin_config) {
        Ok(config) => config,
        Err(err) => {
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "nemo_guardrails.invalid_plugin_config".to_string(),
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
            "request_defaults",
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
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "request_defaults",
        &[
            "context",
            "thread_id",
            "state",
            "rails",
            "llm_params",
            "llm_output",
            "output_vars",
            "log",
        ],
    );
    validate_nested_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "request_defaults",
        "rails",
        &[
            "input",
            "output",
            "retrieval",
            "dialog",
            "tool_output",
            "tool_input",
        ],
    );

    validate_version(&mut diagnostics, &config.policy, config.version);
    validate_mode(&mut diagnostics, &config.policy, &config.mode);
    validate_non_empty_strings(&mut diagnostics, &config.policy, &config);
    validate_config_shape(&mut diagnostics, &config.policy, &config);
    validate_codec_requirements(&mut diagnostics, &config.policy, &config);
    validate_surface_selection(&mut diagnostics, &config.policy, &config);
    validate_remote_backend_support(&mut diagnostics, &config.policy, &config);
    validate_request_defaults(&mut diagnostics, &config.policy, &config);

    diagnostics
}

fn validate_version(diagnostics: &mut Vec<ConfigDiagnostic>, policy: &ConfigPolicy, version: u32) {
    if version != 1 {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_config_version",
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
            "nemo_guardrails.unsupported_value",
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
            "nemo_guardrails.unsupported_value",
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
            "nemo_guardrails.unsupported_value",
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
            "nemo_guardrails.unsupported_value",
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
                "nemo_guardrails.unsupported_value",
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
                "nemo_guardrails.unsupported_value",
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
                    "nemo_guardrails.unsupported_value",
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
            "nemo_guardrails.unsupported_value",
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
    let flags = ConfigShapeFlags::from(config);

    match config.mode.as_str() {
        "local" => validate_local_config_shape(diagnostics, policy, config, &flags),
        "remote" => validate_remote_config_shape(diagnostics, policy, config, &flags),
        _ => {}
    }
}

struct ConfigShapeFlags {
    has_config_path: bool,
    has_config_yaml: bool,
    has_colang_content: bool,
    has_remote_config_id: bool,
    has_remote_config_ids: bool,
}

impl From<&NeMoGuardrailsConfig> for ConfigShapeFlags {
    fn from(config: &NeMoGuardrailsConfig) -> Self {
        Self {
            has_config_path: config.config_path.is_some(),
            has_config_yaml: config.config_yaml.is_some(),
            has_colang_content: config.colang_content.is_some(),
            has_remote_config_id: config
                .remote
                .as_ref()
                .and_then(|remote| remote.config_id.as_ref())
                .is_some(),
            has_remote_config_ids: config
                .remote
                .as_ref()
                .map(|remote| !remote.config_ids.is_empty())
                .unwrap_or(false),
        }
    }
}

fn validate_local_config_shape(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
    flags: &ConfigShapeFlags,
) {
    if flags.has_config_path == flags.has_config_yaml {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.invalid_config_source",
            None,
            "exactly one of config_path or config_yaml is required in local mode",
        );
    }

    if flags.has_colang_content && !flags.has_config_yaml {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some("colang_content"),
            "colang_content can only be used with config_yaml",
        );
    }

    if config.remote.is_some() {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some("remote"),
            "remote backend settings cannot be used when mode is 'local'",
        );
    }
}

fn validate_remote_config_shape(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
    flags: &ConfigShapeFlags,
) {
    if flags.has_config_path || flags.has_config_yaml || flags.has_colang_content {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.invalid_config_source",
            None,
            "remote mode uses remote config identity and cannot include config_path, config_yaml, or colang_content",
        );
    }

    if config.local.is_some() {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some("local"),
            "local backend settings cannot be used when mode is 'remote'",
        );
    }

    match &config.remote {
        Some(remote)
            if remote
                .endpoint
                .as_ref()
                .is_some_and(|value| !value.trim().is_empty()) => {}
        _ => push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some("remote.endpoint"),
            "remote.endpoint is required when mode is 'remote'",
        ),
    }

    if flags.has_remote_config_id && flags.has_remote_config_ids {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some("remote"),
            "remote.config_id and remote.config_ids cannot be used together",
        );
    }

    if !(flags.has_remote_config_id || flags.has_remote_config_ids) {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.invalid_config_source",
            None,
            "remote mode requires remote.config_id or remote.config_ids",
        );
    }
}

fn push_config_shape_diag(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    behavior: UnsupportedBehavior,
    code: &str,
    field: Option<&str>,
    message: &str,
) {
    push_policy_diag(
        diagnostics,
        behavior,
        code,
        Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
        field.map(str::to_string),
        message.to_string(),
    );
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
            "nemo_guardrails.unsupported_value",
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
            "nemo_guardrails.unsupported_value",
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
        "nemo_guardrails.unsupported_value",
        Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
        None,
        "at least one Guardrails surface must be enabled".to_string(),
    );
}

fn validate_remote_backend_support(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    if config.mode != "remote" {
        return;
    }

    if (config.input || config.output)
        && config
            .codec
            .as_deref()
            .is_some_and(|codec| codec != "openai_chat")
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("codec".to_string()),
            "remote mode currently supports only codec = 'openai_chat'".to_string(),
        );
    }
}

fn validate_request_defaults(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    let Some(request_defaults) = &config.request_defaults else {
        return;
    };

    validate_json_object_field(
        diagnostics,
        policy,
        request_defaults.context.as_ref(),
        "request_defaults.context",
        "request_defaults.context must be a JSON object",
    );
    if let Some(thread_id) = &request_defaults.thread_id {
        let trimmed_thread_id = thread_id.trim();
        if trimmed_thread_id.is_empty() {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemo_guardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("request_defaults.thread_id".to_string()),
                "request_defaults.thread_id must not be empty".to_string(),
            );
        } else if trimmed_thread_id.len() < 16 {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemo_guardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("request_defaults.thread_id".to_string()),
                "request_defaults.thread_id must be at least 16 characters long".to_string(),
            );
        }
    }
    validate_json_object_field(
        diagnostics,
        policy,
        request_defaults.state.as_ref(),
        "request_defaults.state",
        "request_defaults.state must be a JSON object",
    );
    if let Some(state) = request_defaults
        .state
        .as_ref()
        .and_then(|value| value.as_object())
    {
        let contains_supported_key = state.contains_key("events") || state.contains_key("state");
        let contains_unsupported_key = state.keys().any(|key| key != "events" && key != "state");
        if (!state.is_empty() && !contains_supported_key) || contains_unsupported_key {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemo_guardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("request_defaults.state".to_string()),
                "request_defaults.state must be empty or contain only 'events' or 'state'"
                    .to_string(),
            );
        }
    }
    validate_json_object_field(
        diagnostics,
        policy,
        request_defaults.llm_params.as_ref(),
        "request_defaults.llm_params",
        "request_defaults.llm_params must be a JSON object",
    );
    validate_json_object_field(
        diagnostics,
        policy,
        request_defaults.log.as_ref(),
        "request_defaults.log",
        "request_defaults.log must be a JSON object",
    );

    if let Some(output_vars) = &request_defaults.output_vars {
        match output_vars {
            Json::Bool(_) => {}
            Json::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    if !value.is_string()
                        || value.as_str().is_some_and(|entry| entry.trim().is_empty())
                    {
                        push_policy_diag(
                            diagnostics,
                            policy.unsupported_value,
                            "nemo_guardrails.unsupported_value",
                            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                            Some(format!("request_defaults.output_vars[{index}]")),
                            "request_defaults.output_vars array entries must be non-empty strings"
                                .to_string(),
                        );
                    }
                }
            }
            _ => push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemo_guardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("request_defaults.output_vars".to_string()),
                "request_defaults.output_vars must be a boolean or an array of strings".to_string(),
            ),
        }
    }

    if let Some(rails) = &request_defaults.rails {
        validate_rail_selector(
            diagnostics,
            policy,
            rails.input.as_ref(),
            "request_defaults.rails.input",
        );
        validate_rail_selector(
            diagnostics,
            policy,
            rails.output.as_ref(),
            "request_defaults.rails.output",
        );
        validate_rail_selector(
            diagnostics,
            policy,
            rails.retrieval.as_ref(),
            "request_defaults.rails.retrieval",
        );
        validate_rail_selector(
            diagnostics,
            policy,
            rails.tool_output.as_ref(),
            "request_defaults.rails.tool_output",
        );
        validate_rail_selector(
            diagnostics,
            policy,
            rails.tool_input.as_ref(),
            "request_defaults.rails.tool_input",
        );
    }
}

fn validate_json_object_field(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    value: Option<&Json>,
    field: &str,
    message: &str,
) {
    let Some(value) = value else {
        return;
    };

    if !value.is_object() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some(field.to_string()),
            message.to_string(),
        );
    }
}

fn validate_rail_selector(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    value: Option<&RailSelector>,
    field: &str,
) {
    let Some(value) = value else {
        return;
    };

    if let RailSelector::Named(names) = value {
        for (index, name) in names.iter().enumerate() {
            if name.trim().is_empty() {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemo_guardrails.unsupported_value",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    Some(format!("{field}[{index}]")),
                    "named rail selections must not contain empty strings".to_string(),
                );
            }
        }
    }
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

fn validate_nested_section_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
    section: &str,
    nested_section: &str,
    known_fields: &[&str],
) {
    if let Some(section_json) = plugin_config.get(section).and_then(Json::as_object)
        && let Some(nested_json) = section_json.get(nested_section).and_then(Json::as_object)
    {
        validate_unknown_fields(
            diagnostics,
            policy,
            Some(format!("{section}.{nested_section}")),
            nested_json,
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
                "nemo_guardrails.unknown_field",
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

fn default_nemo_guardrails_config_version() -> u32 {
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
#[path = "../../../tests/unit/plugins/nemo_guardrails/component_tests.rs"]
mod tests;
