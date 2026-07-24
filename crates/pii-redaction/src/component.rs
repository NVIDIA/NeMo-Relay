// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PII redaction plugin component contract.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use nemo_relay::codec::resolve::supported_codec_names;
use nemo_relay::plugin::{
    ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, Plugin, PluginComponentSpec, PluginError,
    PluginRegistrationContext, Result as PluginResult, UnsupportedBehavior,
    apply_global_config_policy, deregister_plugin, register_plugin,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

use super::builtin::{
    CompiledBuiltinBackend, llm_sanitize_request_callback, llm_sanitize_response_callback,
    tool_sanitize_callback,
};
#[cfg(test)]
pub(crate) use super::builtin::{hex_sha256, mask_text};
use super::detectors::{detector_regex_pattern, supported_detector_summary};
use super::local::register_local_backend;
pub use super::local::{clear_local_backend_provider, register_local_backend_provider};

/// The plugin kind reserved for the built-in privacy component.
pub const PII_REDACTION_PLUGIN_KIND: &str = "pii_redaction";

/// Top-level PII redaction component wrapper.
#[derive(Debug, Clone)]
pub struct ComponentSpec {
    /// Whether the component should be activated.
    pub enabled: bool,
    /// Component-local PII redaction config.
    pub config: PiiRedactionConfig,
}

impl ComponentSpec {
    /// Creates an enabled PII redaction component spec.
    pub fn new(config: PiiRedactionConfig) -> Self {
        Self {
            enabled: true,
            config,
        }
    }
}

impl From<ComponentSpec> for PluginComponentSpec {
    fn from(value: ComponentSpec) -> Self {
        let Json::Object(config) = serde_json::to_value(value.config)
            .expect("PII redaction config should serialize to an object")
        else {
            unreachable!("PII redaction config must serialize to an object");
        };

        PluginComponentSpec {
            kind: PII_REDACTION_PLUGIN_KIND.to_string(),
            enabled: value.enabled,
            config,
        }
    }
}

/// Canonical config document for the PII redaction component.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PiiRedactionConfig {
    /// PII redaction config schema version.
    #[serde(default = "default_pii_redaction_config_version")]
    pub version: u32,
    /// Backend mode: `builtin` or `local_model`.
    #[serde(default = "default_mode", skip_serializing_if = "is_default_mode")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "mode_schema"))]
    pub mode: String,
    /// Whether to sanitize managed LLM request payloads.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub input: bool,
    /// Whether to sanitize managed LLM response payloads.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub output: bool,
    /// Whether to sanitize mark event observability fields.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub mark: bool,
    /// Whether to sanitize managed tool request payloads.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub tool_input: bool,
    /// Whether to sanitize managed tool response payloads.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub tool_output: bool,
    /// Guardrail priority. Lower values run earlier.
    #[serde(
        default = "default_priority",
        skip_serializing_if = "is_default_priority"
    )]
    pub priority: i32,
    /// Compatibility fallback codec for LLM-managed surfaces without an active
    /// per-call codec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "codec_schema"))]
    pub codec: Option<String>,
    /// Ordered redaction profiles. When present, legacy backend and surface
    /// fields must be omitted and every enabled profile covers all supported
    /// sanitization surfaces.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub profiles: Vec<PiiRedactionProfile>,
    /// Built-in backend settings used when `mode = "builtin"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builtin: Option<BuiltinBackendConfig>,
    /// Local-backend settings used when `mode = "local_model"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local: Option<LocalBackendConfig>,
    /// Component-local unsupported-config policy.
    #[serde(default)]
    pub policy: ConfigPolicy,
}

impl Default for PiiRedactionConfig {
    fn default() -> Self {
        Self {
            version: default_pii_redaction_config_version(),
            mode: default_mode(),
            input: true,
            output: true,
            mark: true,
            tool_input: true,
            tool_output: true,
            priority: default_priority(),
            codec: None,
            profiles: Vec::new(),
            builtin: None,
            local: None,
            policy: ConfigPolicy::default(),
        }
    }
}

/// One ordered redaction policy within the singleton PII component.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PiiRedactionProfile {
    /// Whether this profile registers runtime sanitizers.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Backend mode: `builtin` or `local_model`.
    #[serde(default = "default_mode")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "mode_schema"))]
    pub mode: String,
    /// Guardrail priority. Lower values run earlier; array order breaks ties.
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Built-in backend settings used when `mode = "builtin"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builtin: Option<BuiltinBackendConfig>,
    /// Local-backend settings used when `mode = "local_model"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local: Option<LocalBackendConfig>,
}

impl Default for PiiRedactionProfile {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: default_mode(),
            priority: default_priority(),
            builtin: None,
            local: None,
        }
    }
}

/// Built-in redaction backend settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BuiltinBackendConfig {
    /// Optional semantic sanitization preset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "builtin_preset_schema"))]
    pub preset: Option<String>,
    /// Action applied to matching string leaves.
    #[serde(
        default = "default_builtin_action",
        skip_serializing_if = "is_default_builtin_action"
    )]
    #[cfg_attr(feature = "schema", schemars(schema_with = "builtin_action_schema"))]
    pub action: String,
    /// Exact JSON-pointer paths to sanitize. Empty means every string leaf.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_paths: Vec<String>,
    /// Regex pattern used when `action = "regex_replace"` or `action = "redact"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    /// Built-in detector preset used when you do not want to write a regex.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detector: Option<String>,
    /// Replacement text used when `action = "regex_replace"` or `action = "redact"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement: Option<String>,
    /// Masking token used when `action = "mask"`. Defaults to `*`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask_char: Option<String>,
    /// Number of leading characters to keep when `action = "mask"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unmasked_prefix: Option<usize>,
    /// Number of trailing characters to keep when `action = "mask"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unmasked_suffix: Option<usize>,
    /// How the trajectory preset handles opaque custom-mark payloads.
    #[serde(
        default = "default_custom_mark_payload_policy",
        skip_serializing_if = "is_default_custom_mark_payload_policy"
    )]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "custom_mark_payload_policy_schema")
    )]
    pub custom_mark_payload_policy: String,
}

impl Default for BuiltinBackendConfig {
    fn default() -> Self {
        Self {
            preset: None,
            action: default_builtin_action(),
            target_paths: Vec::new(),
            pattern: None,
            detector: None,
            replacement: None,
            mask_char: None,
            unmasked_prefix: None,
            unmasked_suffix: None,
            custom_mark_payload_policy: default_custom_mark_payload_policy(),
        }
    }
}

/// Local-backend settings for a future in-process local-model runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LocalBackendConfig {
    /// Optional local-model backend identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Optional model identifier reserved for future local-model runtimes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Optional detector profile reserved for future local-model runtimes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detector_profile: Option<String>,
    /// Whether a future local-model backend may use network calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_network: Option<bool>,
    /// Target latency budget hint for a future local-model backend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_latency_ms: Option<u64>,
}

nemo_relay::editor_config! {
    impl PiiRedactionConfig {
        mode => {
            label: "mode",
            kind: Enum,
            values: ["builtin", "local_model"],
        },
        input => { label: "input", kind: Boolean },
        output => { label: "output", kind: Boolean },
        mark => { label: "mark", kind: Boolean },
        tool_input => { label: "tool_input", kind: Boolean },
        tool_output => { label: "tool_output", kind: Boolean },
        priority => { label: "priority", kind: Integer },
        codec => {
            label: "codec",
            kind: Enum,
            values: ["openai_chat", "openai_responses", "anthropic_messages"],
            optional: true,
        },
        profiles => { label: "profiles", kind: List, list: &PII_REDACTION_PROFILE_LIST_ITEM },
        builtin => {
            label: "builtin",
            kind: Section,
            optional: true,
            nested: BuiltinBackendConfig,
            default: BuiltinBackendConfig,
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

nemo_relay::editor_config! {
    impl PiiRedactionProfile {
        enabled => { label: "enabled", kind: Boolean },
        mode => {
            label: "mode",
            kind: Enum,
            values: ["builtin", "local_model"],
        },
        priority => { label: "priority", kind: Integer },
        builtin => {
            label: "builtin",
            kind: Section,
            optional: true,
            nested: BuiltinBackendConfig,
            default: BuiltinBackendConfig,
        },
        local => {
            label: "local",
            kind: Section,
            optional: true,
            nested: LocalBackendConfig,
            default: LocalBackendConfig,
        },
    }
}

fn pii_redaction_profile_editor_schema() -> &'static nemo_relay::config_editor::EditorSchema {
    <PiiRedactionProfile as nemo_relay::config_editor::EditorConfig>::editor_schema()
}

fn default_pii_redaction_profile_editor_value() -> Json {
    serde_json::to_value(PiiRedactionProfile::default())
        .expect("PII redaction profile should serialize")
}

static PII_REDACTION_PROFILE_LIST_ITEM: nemo_relay::config_editor::EditorListItemSpec =
    nemo_relay::config_editor::EditorListItemSpec {
        kind: nemo_relay::config_editor::EditorFieldKind::Section,
        schema: Some(pii_redaction_profile_editor_schema),
        default: Some(default_pii_redaction_profile_editor_value),
        tagged_union: None,
        list_item: None,
    };

nemo_relay::editor_config! {
    impl BuiltinBackendConfig {
        preset => {
            label: "preset",
            kind: Enum,
            values: ["trajectory_context"],
            optional: true,
        },
        action => {
            label: "action",
            kind: Enum,
            values: ["remove", "redact", "regex_replace", "hash", "mask"],
        },
        target_paths => { label: "target_paths", kind: List, list: &nemo_relay::config_editor::STRING_LIST_ITEM },
        pattern => { label: "pattern", kind: String, optional: true },
        detector => {
            label: "detector",
            kind: Enum,
            values: [
                "email",
                "phone",
                "api_key",
                "ip_address",
                "ipv6",
                "url",
                "uuid",
                "bearer_token",
                "jwt",
                "credit_card",
                "aws_access_key_id",
                "aws_secret_access_key",
                "gcp_api_key",
                "azure_storage_account_key",
            ],
            optional: true,
        },
        replacement => { label: "replacement", kind: String, optional: true },
        mask_char => { label: "mask_char", kind: String, optional: true },
        unmasked_prefix => { label: "unmasked_prefix", kind: Integer, optional: true },
        unmasked_suffix => { label: "unmasked_suffix", kind: Integer, optional: true },
        custom_mark_payload_policy => {
            label: "custom_mark_payload_policy",
            kind: Enum,
            values: ["preserve", "redact_all_leaves"],
        },
    }
}

nemo_relay::editor_config! {
    impl LocalBackendConfig {
        backend => { label: "backend", kind: String, optional: true },
        model_id => { label: "model_id", kind: String, optional: true },
        detector_profile => { label: "detector_profile", kind: String, optional: true },
        allow_network => { label: "allow_network", kind: Boolean, optional: true },
        max_latency_ms => { label: "max_latency_ms", kind: Integer, optional: true },
    }
}

struct PiiRedactionPlugin;

impl Plugin for PiiRedactionPlugin {
    fn plugin_kind(&self) -> &str {
        PII_REDACTION_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        validate_pii_redaction_plugin_config(plugin_config)
    }

    fn validate_with_policy(
        &self,
        plugin_config: &Map<String, Json>,
        policy: &ConfigPolicy,
    ) -> Vec<ConfigDiagnostic> {
        validate_pii_redaction_plugin_config_with_policy(plugin_config, Some(policy))
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        let parsed = parse_pii_redaction_config(plugin_config);
        Box::pin(async move {
            let config = parsed?;
            register_pii_redaction_backend(config, ctx)
        })
    }
}

/// Registers the `pii_redaction` component kind in the plugin registry.
pub fn register_pii_redaction_component() -> PluginResult<()> {
    match register_plugin(Arc::new(PiiRedactionPlugin)) {
        Ok(()) => Ok(()),
        Err(PluginError::RegistrationFailed(message)) if message.contains("already registered") => {
            Ok(())
        }
        Err(err) => Err(err),
    }
}

/// Deregisters the `pii_redaction` component kind from the plugin registry.
pub fn deregister_pii_redaction_component() -> bool {
    deregister_plugin(PII_REDACTION_PLUGIN_KIND)
}

/// Returns the JSON Schema for the PII redaction component configuration.
#[cfg(feature = "schema")]
pub fn pii_redaction_config_schema() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(PiiRedactionConfig))
        .expect("PII redaction config schema should serialize")
}

#[cfg(feature = "schema")]
fn mode_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
    string_enum_schema(generator, &["builtin", "local_model"], Some("builtin"))
}

#[cfg(feature = "schema")]
fn builtin_action_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    string_enum_schema(
        generator,
        &["remove", "redact", "regex_replace", "hash", "mask"],
        Some("remove"),
    )
}

#[cfg(feature = "schema")]
fn builtin_preset_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    string_enum_schema(generator, &["trajectory_context"], None)
}

#[cfg(feature = "schema")]
fn custom_mark_payload_policy_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    string_enum_schema(
        generator,
        &["preserve", "redact_all_leaves"],
        Some("preserve"),
    )
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

fn register_pii_redaction_backend(
    config: PiiRedactionConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    if !config.profiles.is_empty() {
        let profiles = config.profiles.clone();
        for (index, profile) in profiles.into_iter().enumerate() {
            if !profile.enabled {
                continue;
            }
            let profile_name = format!("profile_{index}");
            let profile_config = profile_as_legacy_config(&config, profile);
            register_single_pii_redaction_backend(profile_config, ctx, Some(&profile_name))?;
        }
        return Ok(());
    }

    register_single_pii_redaction_backend(config, ctx, None)
}

fn register_single_pii_redaction_backend(
    config: PiiRedactionConfig,
    ctx: &mut PluginRegistrationContext,
    profile_name: Option<&str>,
) -> PluginResult<()> {
    match config.mode.as_str() {
        "builtin" => register_builtin_backend(config, ctx, profile_name),
        "local_model" => register_local_backend(config, ctx, profile_name),
        other => Err(PluginError::InvalidConfig(format!(
            "unsupported PII redaction mode '{other}'"
        ))),
    }
}

fn profile_as_legacy_config(
    component: &PiiRedactionConfig,
    profile: PiiRedactionProfile,
) -> PiiRedactionConfig {
    PiiRedactionConfig {
        version: component.version,
        mode: profile.mode,
        input: true,
        output: true,
        mark: true,
        tool_input: true,
        tool_output: true,
        priority: profile.priority,
        codec: component.codec.clone(),
        profiles: Vec::new(),
        builtin: profile.builtin,
        local: profile.local,
        policy: component.policy,
    }
}

fn parse_pii_redaction_config(
    plugin_config: &Map<String, Json>,
) -> PluginResult<PiiRedactionConfig> {
    serde_json::from_value(Json::Object(plugin_config.clone())).map_err(|err| {
        PluginError::InvalidConfig(format!("invalid PII redaction plugin config: {err}"))
    })
}

fn validate_pii_redaction_plugin_config(
    plugin_config: &Map<String, Json>,
) -> Vec<ConfigDiagnostic> {
    validate_pii_redaction_plugin_config_with_policy(plugin_config, None)
}

fn validate_pii_redaction_plugin_config_with_policy(
    plugin_config: &Map<String, Json>,
    policy: Option<&ConfigPolicy>,
) -> Vec<ConfigDiagnostic> {
    let mut config = match parse_pii_redaction_config(plugin_config) {
        Ok(config) => config,
        Err(err) => {
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "pii_redaction.invalid_plugin_config".to_string(),
                component: Some(PII_REDACTION_PLUGIN_KIND.to_string()),
                field: None,
                message: err.to_string(),
            }];
        }
    };
    if let Some(policy) = policy {
        config.policy = apply_global_config_policy(config.policy, policy);
    }

    let mut diagnostics = vec![];

    validate_unknown_fields(
        &mut diagnostics,
        &config.policy,
        Some(PII_REDACTION_PLUGIN_KIND.to_string()),
        plugin_config,
        &[
            "version",
            "mode",
            "input",
            "output",
            "mark",
            "tool_input",
            "tool_output",
            "priority",
            "codec",
            "profiles",
            "builtin",
            "local",
            "policy",
        ],
    );
    validate_policy_fields(&mut diagnostics, &config.policy, plugin_config);
    if plugin_config.contains_key("profiles") {
        validate_profile_configuration(&mut diagnostics, plugin_config, &config);
        validate_version(&mut diagnostics, &config.policy, config.version);
        validate_codec_requirements(&mut diagnostics, &config.policy, &config);
        return diagnostics;
    }

    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "builtin",
        &[
            "preset",
            "action",
            "target_paths",
            "pattern",
            "detector",
            "replacement",
            "mask_char",
            "unmasked_prefix",
            "unmasked_suffix",
            "custom_mark_payload_policy",
        ],
    );
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "local",
        &[
            "backend",
            "model_id",
            "detector_profile",
            "allow_network",
            "max_latency_ms",
        ],
    );
    validate_version(&mut diagnostics, &config.policy, config.version);
    validate_mode(&mut diagnostics, &config.policy, &config);
    validate_surface_selection(&mut diagnostics, &config.policy, &config);
    validate_codec_requirements(&mut diagnostics, &config.policy, &config);
    validate_builtin_mode_requirements(&mut diagnostics, &config.policy, plugin_config, &config);
    validate_builtin_action_requirements(&mut diagnostics, &config.policy, plugin_config, &config);
    validate_local_mode_requirements(&mut diagnostics, &config.policy, plugin_config, &config);

    diagnostics
}

fn validate_profile_configuration(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    plugin_config: &Map<String, Json>,
    config: &PiiRedactionConfig,
) {
    const LEGACY_FIELDS: &[&str] = &[
        "mode",
        "input",
        "output",
        "mark",
        "tool_input",
        "tool_output",
        "priority",
        "builtin",
        "local",
    ];
    for field in LEGACY_FIELDS {
        if plugin_config.contains_key(*field) {
            push_policy_diag(
                diagnostics,
                config.policy.unsupported_value,
                "pii_redaction.unsupported_value",
                Some(PII_REDACTION_PLUGIN_KIND.to_string()),
                Some((*field).to_string()),
                format!("legacy field '{field}' cannot be combined with profiles"),
            );
        }
    }

    if !config.profiles.iter().any(|profile| profile.enabled) {
        push_policy_diag(
            diagnostics,
            config.policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("profiles".to_string()),
            "profiles must contain at least one enabled redaction profile".to_string(),
        );
        return;
    }

    let raw_profiles = plugin_config
        .get("profiles")
        .and_then(Json::as_array)
        .expect("successfully parsed profiles must be an array");
    for (index, (profile, raw_profile)) in
        config.profiles.iter().zip(raw_profiles.iter()).enumerate()
    {
        let Json::Object(raw_profile) = raw_profile else {
            continue;
        };
        let profile_config = profile_as_legacy_config(config, profile.clone());
        let mut profile_diagnostics = Vec::new();
        validate_unknown_fields(
            &mut profile_diagnostics,
            &config.policy,
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            raw_profile,
            &["enabled", "mode", "priority", "builtin", "local"],
        );
        validate_section_fields(
            &mut profile_diagnostics,
            &config.policy,
            raw_profile,
            "builtin",
            &[
                "preset",
                "action",
                "target_paths",
                "pattern",
                "detector",
                "replacement",
                "mask_char",
                "unmasked_prefix",
                "unmasked_suffix",
                "custom_mark_payload_policy",
            ],
        );
        validate_section_fields(
            &mut profile_diagnostics,
            &config.policy,
            raw_profile,
            "local",
            &[
                "backend",
                "model_id",
                "detector_profile",
                "allow_network",
                "max_latency_ms",
            ],
        );
        validate_mode(&mut profile_diagnostics, &config.policy, &profile_config);
        validate_builtin_mode_requirements(
            &mut profile_diagnostics,
            &config.policy,
            raw_profile,
            &profile_config,
        );
        validate_builtin_action_requirements(
            &mut profile_diagnostics,
            &config.policy,
            raw_profile,
            &profile_config,
        );
        validate_local_mode_requirements(
            &mut profile_diagnostics,
            &config.policy,
            raw_profile,
            &profile_config,
        );
        if profile.mode == "local_model" && !raw_profile.contains_key("local") {
            push_policy_diag(
                &mut profile_diagnostics,
                config.policy.unsupported_value,
                "pii_redaction.unsupported_value",
                Some(PII_REDACTION_PLUGIN_KIND.to_string()),
                Some("local".to_string()),
                "`local` settings are required for a local-model profile".to_string(),
            );
        }
        prefix_profile_diagnostics(&mut profile_diagnostics, index);
        diagnostics.extend(profile_diagnostics);
    }
}

fn prefix_profile_diagnostics(diagnostics: &mut [ConfigDiagnostic], index: usize) {
    let prefix = format!("profiles[{index}]");
    for diagnostic in diagnostics {
        diagnostic.field = Some(match diagnostic.field.take() {
            Some(field) => format!("{prefix}.{field}"),
            None => prefix.clone(),
        });
    }
}

fn validate_mode(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &PiiRedactionConfig,
) {
    if matches!(config.mode.as_str(), "builtin" | "local_model") {
        return;
    }

    push_policy_diag(
        diagnostics,
        policy.unsupported_value,
        "pii_redaction.unsupported_value",
        Some(PII_REDACTION_PLUGIN_KIND.to_string()),
        Some("mode".to_string()),
        "mode must be 'builtin' or 'local_model'".to_string(),
    );
}

fn validate_surface_selection(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &PiiRedactionConfig,
) {
    if config.input || config.output || config.mark || config.tool_input || config.tool_output {
        return;
    }

    push_policy_diag(
        diagnostics,
        policy.unsupported_value,
        "pii_redaction.unsupported_value",
        Some(PII_REDACTION_PLUGIN_KIND.to_string()),
        None,
        "at least one redaction surface must be enabled".to_string(),
    );
}

fn validate_local_mode_requirements(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
    config: &PiiRedactionConfig,
) {
    if config.mode == "local_model" {
        return;
    }
    if !plugin_config.contains_key("local") {
        return;
    }

    push_policy_diag(
        diagnostics,
        policy.unsupported_value,
        "pii_redaction.unsupported_value",
        Some(PII_REDACTION_PLUGIN_KIND.to_string()),
        Some("local".to_string()),
        "`local` settings are valid only when mode = 'local_model'".to_string(),
    );
}

fn validate_builtin_mode_requirements(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
    config: &PiiRedactionConfig,
) {
    if config.mode == "builtin" {
        if plugin_config.contains_key("builtin") {
            return;
        }
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin".to_string()),
            "`builtin` settings are required when mode = 'builtin'".to_string(),
        );
        return;
    }
    if !plugin_config.contains_key("builtin") {
        return;
    }

    push_policy_diag(
        diagnostics,
        policy.unsupported_value,
        "pii_redaction.unsupported_value",
        Some(PII_REDACTION_PLUGIN_KIND.to_string()),
        Some("builtin".to_string()),
        "`builtin` settings are valid only when mode = 'builtin'".to_string(),
    );
}

fn validate_builtin_action_requirements(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
    config: &PiiRedactionConfig,
) {
    let Some(builtin) = config.builtin.as_ref() else {
        return;
    };

    if builtin.preset.is_some() {
        validate_builtin_preset_requirements(diagnostics, policy, plugin_config, builtin);
        return;
    }

    let custom_policy_configured = plugin_config
        .get("builtin")
        .and_then(Json::as_object)
        .is_some_and(|builtin| builtin.contains_key("custom_mark_payload_policy"));
    if custom_policy_configured {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin.custom_mark_payload_policy".to_string()),
            "builtin.custom_mark_payload_policy requires builtin.preset = 'trajectory_context'"
                .to_string(),
        );
    }

    if !matches!(
        builtin.action.as_str(),
        "remove" | "redact" | "regex_replace" | "hash" | "mask"
    ) {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin.action".to_string()),
            "builtin.action must be 'remove', 'redact', 'regex_replace', 'hash', or 'mask'"
                .to_string(),
        );
    }

    if matches!(builtin.action.as_str(), "regex_replace" | "redact")
        && builtin.pattern.is_none()
        && builtin.detector.is_none()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin.pattern".to_string()),
            "builtin.pattern or builtin.detector is required when builtin.action = 'regex_replace' or 'redact'"
                .to_string(),
        );
    }

    if builtin.pattern.is_some() && builtin.detector.is_some() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin.detector".to_string()),
            "builtin.pattern and builtin.detector cannot both be set".to_string(),
        );
    }

    if let Some(pattern) = builtin.pattern.as_deref()
        && let Err(err) = Regex::new(pattern)
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin.pattern".to_string()),
            format!("invalid builtin matcher regex '{pattern}': {err}"),
        );
    }

    if builtin
        .detector
        .as_deref()
        .is_some_and(|detector| detector_regex_pattern(detector).is_none())
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin.detector".to_string()),
            format!(
                "builtin.detector must be one of the supported built-in detector presets ({})",
                supported_detector_summary()
            ),
        );
    }

    if builtin.action == "mask"
        && builtin
            .mask_char
            .as_deref()
            .is_some_and(|mask_char| mask_char.is_empty())
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin.mask_char".to_string()),
            "builtin.mask_char must not be empty when builtin.action = 'mask'".to_string(),
        );
    }
}

fn validate_builtin_preset_requirements(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
    builtin: &BuiltinBackendConfig,
) {
    if builtin.preset.as_deref() != Some("trajectory_context") {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin.preset".to_string()),
            "builtin.preset must be 'trajectory_context'".to_string(),
        );
    }
    if !matches!(
        builtin.custom_mark_payload_policy.as_str(),
        "preserve" | "redact_all_leaves"
    ) {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin.custom_mark_payload_policy".to_string()),
            "builtin.custom_mark_payload_policy must be 'preserve' or 'redact_all_leaves'"
                .to_string(),
        );
    }
    let raw_builtin = plugin_config.get("builtin").and_then(Json::as_object);
    for field in [
        "action",
        "detector",
        "pattern",
        "target_paths",
        "mask_char",
        "unmasked_prefix",
        "unmasked_suffix",
    ] {
        if raw_builtin.is_some_and(|raw| raw.contains_key(field)) {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "pii_redaction.unsupported_value",
                Some(PII_REDACTION_PLUGIN_KIND.to_string()),
                Some(format!("builtin.{field}")),
                format!("builtin.{field} cannot be combined with builtin.preset"),
            );
        }
    }
}

fn validate_version(diagnostics: &mut Vec<ConfigDiagnostic>, policy: &ConfigPolicy, version: u32) {
    if version != default_pii_redaction_config_version() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_config_version",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("version".to_string()),
            format!("PII redaction config version {version} is unsupported"),
        );
    }
}

fn validate_codec_requirements(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &PiiRedactionConfig,
) {
    let llm_surface_enabled = config.input || config.output;
    if !llm_surface_enabled {
        return;
    }

    let Some(codec) = config.codec.as_deref() else {
        return;
    };

    if !supported_codec_names().contains(&codec) {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("codec".to_string()),
            "codec must be 'openai_chat', 'openai_responses', or 'anthropic_messages'".to_string(),
        );
    }
}

fn register_builtin_backend(
    config: PiiRedactionConfig,
    ctx: &mut PluginRegistrationContext,
    profile_name: Option<&str>,
) -> PluginResult<()> {
    let builtin = config.builtin.clone().ok_or_else(|| {
        PluginError::InvalidConfig("built-in PII redaction config is missing".to_string())
    })?;
    let compiled = CompiledBuiltinBackend::new(builtin, config.codec.clone())?;
    log::info!(
        target: "nemo_relay.plugin",
        event = "plugin_resource_validation_completed",
        plugin_kind = PII_REDACTION_PLUGIN_KIND,
        profile = profile_name.unwrap_or("legacy"),
        resource_count = 0;
        "Plugin resource validation completed"
    );

    if config.mark {
        ctx.register_mark_sanitize_guardrail(
            &registration_name(profile_name, "mark"),
            config.priority,
            super::builtin::event_sanitize_callback(compiled.clone()),
        )?;
    }

    if config.tool_input {
        let sanitizer = tool_sanitize_callback(compiled.clone());
        ctx.register_tool_sanitize_request_guardrail(
            &registration_name(profile_name, "tool_input"),
            config.priority,
            sanitizer,
        )?;
    }
    if config.tool_output {
        let sanitizer = tool_sanitize_callback(compiled.clone());
        ctx.register_tool_sanitize_response_guardrail(
            &registration_name(profile_name, "tool_output"),
            config.priority,
            sanitizer,
        )?;
    }
    if config.input {
        ctx.register_llm_sanitize_request_guardrail(
            &registration_name(profile_name, "input"),
            config.priority,
            llm_sanitize_request_callback(compiled.clone()),
        )?;
    }
    if config.input || config.tool_input {
        ctx.register_scope_sanitize_start_guardrail(
            &registration_name(
                profile_name,
                if profile_name.is_some() {
                    "scope_start"
                } else {
                    "input"
                },
            ),
            config.priority,
            super::builtin::scope_event_sanitize_callback(
                compiled.clone(),
                config.input,
                config.tool_input,
            ),
        )?;
    }
    if config.output {
        ctx.register_llm_sanitize_response_guardrail(
            &registration_name(profile_name, "output"),
            config.priority,
            llm_sanitize_response_callback(compiled.clone()),
        )?;
    }
    if config.output || config.tool_output {
        ctx.register_scope_sanitize_end_guardrail(
            &registration_name(
                profile_name,
                if profile_name.is_some() {
                    "scope_end"
                } else {
                    "output"
                },
            ),
            config.priority,
            super::builtin::scope_event_sanitize_callback(
                compiled,
                config.output,
                config.tool_output,
            ),
        )?;
    }

    Ok(())
}

fn registration_name(profile_name: Option<&str>, callback_name: &str) -> String {
    profile_name.map_or_else(
        || callback_name.to_string(),
        |profile_name| {
            format!(
                "{}/{callback_name}",
                profile_registration_prefix(profile_name)
            )
        },
    )
}

pub(super) fn profile_registration_prefix(profile_name: &str) -> String {
    profile_name
        .strip_prefix("profile_")
        .and_then(|index| index.parse::<usize>().ok())
        .map_or_else(
            || profile_name.to_string(),
            |index| format!("profile_{index:020}"),
        )
}

fn validate_unknown_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    component: Option<String>,
    plugin_config: &Map<String, Json>,
    supported: &[&str],
) {
    for field in plugin_config.keys() {
        if supported
            .iter()
            .any(|supported_field| supported_field == field)
        {
            continue;
        }
        push_policy_diag(
            diagnostics,
            policy.unknown_field,
            "pii_redaction.unknown_field",
            component.clone(),
            Some(field.clone()),
            format!("unknown field '{field}'"),
        );
    }
}

fn validate_policy_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
) {
    validate_section_fields(
        diagnostics,
        policy,
        plugin_config,
        "policy",
        &["unknown_component", "unknown_field", "unsupported_value"],
    );
}

fn validate_section_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
    section_name: &str,
    supported: &[&str],
) {
    let Some(value) = plugin_config.get(section_name) else {
        return;
    };

    let Json::Object(section) = value else {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some(section_name.to_string()),
            format!("'{section_name}' must be an object"),
        );
        return;
    };

    for field in section.keys() {
        if supported
            .iter()
            .any(|supported_field| supported_field == field)
        {
            continue;
        }
        push_policy_diag(
            diagnostics,
            policy.unknown_field,
            "pii_redaction.unknown_field",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some(format!("{section_name}.{field}")),
            format!("unknown field '{section_name}.{field}'"),
        );
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

fn default_pii_redaction_config_version() -> u32 {
    1
}

fn default_mode() -> String {
    "builtin".to_string()
}

fn default_builtin_action() -> String {
    "remove".to_string()
}

fn default_custom_mark_payload_policy() -> String {
    "preserve".to_string()
}

fn default_true() -> bool {
    true
}

fn default_priority() -> i32 {
    100
}

fn is_default_mode(mode: &str) -> bool {
    mode == "builtin"
}

fn is_default_builtin_action(action: &str) -> bool {
    action == "remove"
}

fn is_default_custom_mark_payload_policy(policy: &str) -> bool {
    policy == "preserve"
}

fn is_true(value: &bool) -> bool {
    *value
}

fn is_default_priority(priority: &i32) -> bool {
    *priority == default_priority()
}

#[cfg(test)]
#[path = "../tests/unit/component_tests.rs"]
mod tests;
