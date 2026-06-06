// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PII redaction plugin component contract.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use regex::Regex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};

use crate::api::llm::LlmRequest;
use crate::api::runtime::{LlmSanitizeRequestFn, LlmSanitizeResponseFn, ToolSanitizeFn};
use crate::codec::anthropic::AnthropicMessagesCodec;
use crate::codec::openai_chat::OpenAIChatCodec;
use crate::codec::openai_responses::OpenAIResponsesCodec;
use crate::codec::request::{ContentPart, MessageContent};
use crate::codec::response::{AnnotatedLlmResponse, FinishReason, ResponseToolCall};
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::plugin::{
    ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, Plugin, PluginComponentSpec, PluginError,
    PluginRegistrationContext, Result as PluginResult, UnsupportedBehavior, deregister_plugin,
    register_plugin,
};

#[path = "local.rs"]
mod local;
use local::register_local_backend;
pub use local::{clear_local_backend_provider, register_local_backend_provider};

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
    #[serde(default = "default_mode")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "mode_schema"))]
    pub mode: String,
    /// Whether to sanitize managed LLM request payloads.
    #[serde(default = "default_true")]
    pub input: bool,
    /// Whether to sanitize managed LLM response payloads.
    #[serde(default = "default_true")]
    pub output: bool,
    /// Whether to sanitize managed tool request payloads.
    #[serde(default = "default_true")]
    pub tool_input: bool,
    /// Whether to sanitize managed tool response payloads.
    #[serde(default = "default_true")]
    pub tool_output: bool,
    /// Guardrail priority. Lower values run earlier.
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Provider request/response codec for LLM-managed surfaces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "codec_schema"))]
    pub codec: Option<String>,
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
            tool_input: true,
            tool_output: true,
            priority: default_priority(),
            codec: None,
            builtin: None,
            local: None,
            policy: ConfigPolicy::default(),
        }
    }
}

/// Built-in redaction backend settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BuiltinBackendConfig {
    /// Action applied to matching string leaves.
    #[serde(default = "default_builtin_action")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "builtin_action_schema"))]
    pub action: String,
    /// Exact JSON-pointer paths to sanitize. Empty means every string leaf.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_paths: Vec<String>,
    /// Regex pattern used when `action = "regex_replace"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    /// Replacement text used when `action = "regex_replace"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement: Option<String>,
}

/// Local-backend settings for a future in-process local-model runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LocalBackendConfig {
    /// Optional local-model backend identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
}

crate::editor_config! {
    impl PiiRedactionConfig {
        mode => {
            label: "mode",
            kind: Enum,
            values: ["builtin", "local_model"],
        },
        input => { label: "input", kind: Boolean },
        output => { label: "output", kind: Boolean },
        tool_input => { label: "tool_input", kind: Boolean },
        tool_output => { label: "tool_output", kind: Boolean },
        priority => { label: "priority", kind: Integer },
        codec => {
            label: "codec",
            kind: Enum,
            values: ["openai_chat", "openai_responses", "anthropic_messages"],
            optional: true,
        },
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

crate::editor_config! {
    impl BuiltinBackendConfig {
        action => {
            label: "action",
            kind: Enum,
            values: ["remove", "regex_replace", "hash"],
        },
        target_paths => { label: "target_paths", kind: Json },
        pattern => { label: "pattern", kind: String, optional: true },
        replacement => { label: "replacement", kind: String, optional: true },
    }
}

crate::editor_config! {
    impl LocalBackendConfig {
        backend => { label: "backend", kind: String, optional: true },
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
        &["remove", "regex_replace", "hash"],
        Some("remove"),
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
    match config.mode.as_str() {
        "builtin" => register_builtin_backend(config, ctx),
        "local_model" => register_local_backend(config, ctx),
        other => Err(PluginError::InvalidConfig(format!(
            "unsupported PII redaction mode '{other}'"
        ))),
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
    let config = match parse_pii_redaction_config(plugin_config) {
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
            "tool_input",
            "tool_output",
            "priority",
            "codec",
            "builtin",
            "local",
            "policy",
        ],
    );
    validate_policy_fields(&mut diagnostics, &config.policy, plugin_config);
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "builtin",
        &["action", "target_paths", "pattern", "replacement"],
    );
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "local",
        &["backend"],
    );
    validate_mode(&mut diagnostics, &config.policy, &config);
    validate_surface_selection(&mut diagnostics, &config.policy, &config);
    validate_codec_requirements(&mut diagnostics, &config.policy, &config);
    validate_builtin_mode_requirements(&mut diagnostics, &config.policy, plugin_config, &config);
    validate_builtin_action_requirements(&mut diagnostics, &config.policy, &config);
    validate_local_mode_requirements(&mut diagnostics, &config.policy, plugin_config, &config);

    diagnostics
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
    if config.input || config.output || config.tool_input || config.tool_output {
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
    config: &PiiRedactionConfig,
) {
    let Some(builtin) = config.builtin.as_ref() else {
        return;
    };

    if !matches!(builtin.action.as_str(), "remove" | "regex_replace" | "hash") {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin.action".to_string()),
            "builtin.action must be 'remove', 'regex_replace', or 'hash'".to_string(),
        );
    }

    if builtin.action == "regex_replace" && builtin.pattern.is_none() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
            Some("builtin.pattern".to_string()),
            "builtin.pattern is required when builtin.action = 'regex_replace'".to_string(),
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
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "pii_redaction.unsupported_value",
            Some(PII_REDACTION_PLUGIN_KIND.to_string()),
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
) -> PluginResult<()> {
    let builtin = config.builtin.clone().ok_or_else(|| {
        PluginError::InvalidConfig("built-in PII redaction config is missing".to_string())
    })?;
    let compiled = CompiledBuiltinBackend::new(builtin, config.codec.clone())?;

    if config.tool_input {
        let sanitizer = tool_sanitize_callback(compiled.clone());
        ctx.register_tool_sanitize_request_guardrail("tool_input", config.priority, sanitizer)?;
    }
    if config.tool_output {
        let sanitizer = tool_sanitize_callback(compiled.clone());
        ctx.register_tool_sanitize_response_guardrail("tool_output", config.priority, sanitizer)?;
    }
    if config.input {
        let sanitizer = llm_sanitize_request_callback(compiled.clone());
        ctx.register_llm_sanitize_request_guardrail("input", config.priority, sanitizer)?;
    }
    if config.output {
        let sanitizer = llm_sanitize_response_callback(compiled);
        ctx.register_llm_sanitize_response_guardrail("output", config.priority, sanitizer)?;
    }

    Ok(())
}

#[derive(Clone)]
struct CompiledBuiltinBackend {
    action: BuiltinAction,
    target_paths: Arc<Vec<String>>,
    codec: Option<Arc<dyn BuiltinRequestResponseCodec>>,
    codec_name: Option<BuiltinCodecName>,
}

#[derive(Clone)]
enum BuiltinAction {
    Remove,
    Hash,
    RegexReplace {
        pattern: Arc<Regex>,
        replacement: Arc<String>,
    },
}

#[derive(Clone, Copy)]
enum BuiltinCodecName {
    OpenAIChat,
    OpenAIResponses,
    AnthropicMessages,
}

trait BuiltinRequestResponseCodec: LlmCodec + LlmResponseCodec + Send + Sync {}

impl<T> BuiltinRequestResponseCodec for T where T: LlmCodec + LlmResponseCodec + Send + Sync {}

impl CompiledBuiltinBackend {
    fn new(config: BuiltinBackendConfig, codec_name: Option<String>) -> PluginResult<Self> {
        let action = match config.action.as_str() {
            "remove" => BuiltinAction::Remove,
            "hash" => BuiltinAction::Hash,
            "regex_replace" => {
                let pattern_text = config.pattern.ok_or_else(|| {
                    PluginError::InvalidConfig(
                        "builtin.pattern is required when builtin.action = 'regex_replace'"
                            .to_string(),
                    )
                })?;
                let pattern = Regex::new(&pattern_text).map_err(|err| {
                    PluginError::InvalidConfig(format!(
                        "invalid builtin.pattern regex '{pattern_text}': {err}"
                    ))
                })?;
                BuiltinAction::RegexReplace {
                    pattern: Arc::new(pattern),
                    replacement: Arc::new(
                        config
                            .replacement
                            .unwrap_or_else(|| "[REDACTED]".to_string()),
                    ),
                }
            }
            other => {
                return Err(PluginError::InvalidConfig(format!(
                    "unsupported builtin.action '{other}'"
                )));
            }
        };

        Ok(Self {
            action,
            target_paths: Arc::new(config.target_paths),
            codec_name: codec_name.as_deref().and_then(BuiltinCodecName::parse),
            codec: codec_name
                .as_deref()
                .map(instantiate_builtin_codec)
                .transpose()?,
        })
    }

    fn sanitize_json_preorder_dfs(&self, value: Json) -> Json {
        self.sanitize_json_preorder_dfs_at_path(value, &mut Vec::new())
            .unwrap_or(Json::Null)
    }

    fn sanitize_json_preorder_dfs_at_path(
        &self,
        value: Json,
        path_segments: &mut Vec<String>,
    ) -> Option<Json> {
        match value {
            Json::String(text) => {
                if self.matches_current_preorder_path(path_segments) {
                    self.sanitize_string_value(text)
                } else {
                    Some(Json::String(text))
                }
            }
            Json::Array(items) => Some(Json::Array(
                items
                    .into_iter()
                    .enumerate()
                    .map(|(index, item)| {
                        path_segments.push(index.to_string());
                        let sanitized = self
                            .sanitize_json_preorder_dfs_at_path(item, path_segments)
                            .unwrap_or(Json::Null);
                        path_segments.pop();
                        sanitized
                    })
                    .collect(),
            )),
            Json::Object(map) => Some(Json::Object(
                map.into_iter()
                    .filter_map(|(key, value)| {
                        path_segments.push(escape_json_pointer_segment(&key));
                        let sanitized =
                            self.sanitize_json_preorder_dfs_at_path(value, path_segments);
                        path_segments.pop();
                        sanitized.map(|sanitized| (key, sanitized))
                    })
                    .collect(),
            )),
            other => {
                if self.matches_current_preorder_path(path_segments)
                    && matches!(self.action, BuiltinAction::Remove)
                {
                    None
                } else {
                    Some(other)
                }
            }
        }
    }

    fn matches_current_preorder_path(&self, path_segments: &[String]) -> bool {
        if self.target_paths.is_empty() {
            return true;
        }
        let current_path = render_json_pointer_path(path_segments);
        self.target_paths.iter().any(|path| path == &current_path)
    }

    fn sanitize_string_value(&self, text: String) -> Option<Json> {
        match &self.action {
            BuiltinAction::Remove => None,
            BuiltinAction::Hash => Some(Json::String(hex_sha256(&text))),
            BuiltinAction::RegexReplace {
                pattern,
                replacement,
            } => Some(Json::String(
                pattern
                    .replace_all(&text, replacement.as_str())
                    .into_owned(),
            )),
        }
    }

    fn sanitize_request_with_codec(&self, request: &LlmRequest) -> Option<LlmRequest> {
        let codec = self.codec.as_ref()?;
        let annotated = codec.decode(request).ok()?;
        let sanitized_annotated = sanitize_serializable_with_backend(self, annotated).ok()?;
        codec.encode(&sanitized_annotated, request).ok()
    }

    fn sanitize_response_with_codec(&self, payload: Json) -> Option<Json> {
        let codec = self.codec.as_ref()?;
        let codec_name = self.codec_name?;
        let annotated = codec.decode_response(&payload).ok()?;
        let sanitized_annotated = sanitize_serializable_with_backend(self, annotated).ok()?;
        Some(codec_name.overlay_response_payload(payload, &sanitized_annotated))
    }
}

fn tool_sanitize_callback(backend: CompiledBuiltinBackend) -> ToolSanitizeFn {
    Arc::new(move |_name: &str, payload: Json| backend.sanitize_json_preorder_dfs(payload))
}

fn llm_sanitize_request_callback(backend: CompiledBuiltinBackend) -> LlmSanitizeRequestFn {
    Arc::new(move |mut request: LlmRequest| {
        if let Some(encoded) = backend.sanitize_request_with_codec(&request) {
            return encoded;
        }
        request.content = backend.sanitize_json_preorder_dfs(request.content);
        request
    })
}

fn llm_sanitize_response_callback(backend: CompiledBuiltinBackend) -> LlmSanitizeResponseFn {
    Arc::new(move |payload: Json| {
        if backend.target_paths.is_empty() {
            return backend.sanitize_json_preorder_dfs(payload);
        }

        let payload = backend
            .sanitize_response_with_codec(payload.clone())
            .unwrap_or(payload);
        backend.sanitize_json_preorder_dfs(payload)
    })
}

fn render_json_pointer_path(path_segments: &[String]) -> String {
    if path_segments.is_empty() {
        return String::new();
    }
    let mut rendered = String::new();
    for segment in path_segments {
        rendered.push('/');
        rendered.push_str(segment);
    }
    rendered
}

fn escape_json_pointer_segment(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

fn hex_sha256(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn instantiate_builtin_codec(
    codec_name: &str,
) -> PluginResult<Arc<dyn BuiltinRequestResponseCodec>> {
    let codec: Arc<dyn BuiltinRequestResponseCodec> = match codec_name {
        "openai_chat" => Arc::new(OpenAIChatCodec),
        "openai_responses" => Arc::new(OpenAIResponsesCodec),
        "anthropic_messages" => Arc::new(AnthropicMessagesCodec),
        other => {
            return Err(PluginError::InvalidConfig(format!(
                "unsupported codec '{other}'"
            )));
        }
    };
    Ok(codec)
}

impl BuiltinCodecName {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "openai_chat" => Some(Self::OpenAIChat),
            "openai_responses" => Some(Self::OpenAIResponses),
            "anthropic_messages" => Some(Self::AnthropicMessages),
            _ => None,
        }
    }

    fn overlay_response_payload(self, payload: Json, annotated: &AnnotatedLlmResponse) -> Json {
        match self {
            Self::OpenAIChat => overlay_openai_chat_response(payload, annotated),
            Self::OpenAIResponses => overlay_openai_responses_response(payload, annotated),
            Self::AnthropicMessages => overlay_anthropic_response(payload, annotated),
        }
    }
}

fn overlay_openai_chat_response(mut payload: Json, annotated: &AnnotatedLlmResponse) -> Json {
    let Some(root) = payload.as_object_mut() else {
        return payload;
    };
    set_optional_string_field(root, "id", annotated.id.as_deref());
    set_optional_string_field(root, "model", annotated.model.as_deref());

    let Some(choice) = root
        .get_mut("choices")
        .and_then(Json::as_array_mut)
        .and_then(|choices| choices.first_mut())
        .and_then(Json::as_object_mut)
    else {
        return payload;
    };

    set_optional_string_field(
        choice,
        "finish_reason",
        annotated
            .finish_reason
            .as_ref()
            .map(openai_chat_finish_reason),
    );

    let Some(message) = choice.get_mut("message").and_then(Json::as_object_mut) else {
        return payload;
    };
    set_optional_string_field(
        message,
        "content",
        annotated_message_text(annotated.message.as_ref()).as_deref(),
    );
    overlay_openai_chat_tool_calls(message, annotated.tool_calls.as_deref());
    payload
}

fn overlay_openai_responses_response(mut payload: Json, annotated: &AnnotatedLlmResponse) -> Json {
    let Some(root) = payload.as_object_mut() else {
        return payload;
    };
    set_optional_string_field(root, "id", annotated.id.as_deref());
    set_optional_string_field(root, "model", annotated.model.as_deref());
    set_optional_string_field(
        root,
        "status",
        annotated
            .finish_reason
            .as_ref()
            .map(openai_responses_status),
    );

    if let Some(items) = root.get_mut("output").and_then(Json::as_array_mut) {
        overlay_output_text_blocks(items, annotated_message_text(annotated.message.as_ref()));
        overlay_openai_responses_tool_calls(items, annotated.tool_calls.as_deref());
    }
    payload
}

fn overlay_anthropic_response(mut payload: Json, annotated: &AnnotatedLlmResponse) -> Json {
    let Some(root) = payload.as_object_mut() else {
        return payload;
    };
    set_optional_string_field(root, "id", annotated.id.as_deref());
    set_optional_string_field(root, "model", annotated.model.as_deref());
    set_optional_string_field(
        root,
        "stop_reason",
        annotated.finish_reason.as_ref().map(anthropic_stop_reason),
    );

    if let Some(blocks) = root.get_mut("content").and_then(Json::as_array_mut) {
        overlay_anthropic_text_blocks(blocks, annotated_message_text(annotated.message.as_ref()));
        overlay_anthropic_tool_calls(blocks, annotated.tool_calls.as_deref());
    }
    payload
}

fn overlay_openai_chat_tool_calls(
    message: &mut Map<String, Json>,
    tool_calls: Option<&[ResponseToolCall]>,
) {
    let Some(raw_calls) = message.get_mut("tool_calls").and_then(Json::as_array_mut) else {
        return;
    };
    let Some(tool_calls) = tool_calls else {
        message.remove("tool_calls");
        return;
    };

    for (raw_call, sanitized_call) in raw_calls.iter_mut().zip(tool_calls.iter()) {
        let Some(raw_call) = raw_call.as_object_mut() else {
            continue;
        };
        set_optional_string_field(raw_call, "id", Some(sanitized_call.id.as_str()));
        let Some(function) = raw_call.get_mut("function").and_then(Json::as_object_mut) else {
            continue;
        };
        set_optional_string_field(function, "name", Some(sanitized_call.name.as_str()));
        set_optional_string_field(
            function,
            "arguments",
            Some(json_string(&sanitized_call.arguments).as_str()),
        );
    }
}

fn overlay_openai_responses_tool_calls(
    items: &mut [Json],
    tool_calls: Option<&[ResponseToolCall]>,
) {
    let Some(tool_calls) = tool_calls else {
        return;
    };

    let mut sanitized_calls = tool_calls.iter();
    for item in items {
        let Some(item_type) = item.get("type").and_then(Json::as_str) else {
            continue;
        };
        if item_type != "function_call" {
            continue;
        }
        let Some(raw_call) = item.as_object_mut() else {
            continue;
        };
        let Some(sanitized_call) = sanitized_calls.next() else {
            break;
        };
        set_optional_string_field(raw_call, "call_id", Some(sanitized_call.id.as_str()));
        set_optional_string_field(raw_call, "name", Some(sanitized_call.name.as_str()));
        set_optional_string_field(
            raw_call,
            "arguments",
            Some(json_string(&sanitized_call.arguments).as_str()),
        );
    }
}

fn overlay_anthropic_tool_calls(blocks: &mut [Json], tool_calls: Option<&[ResponseToolCall]>) {
    let Some(tool_calls) = tool_calls else {
        return;
    };

    let mut sanitized_calls = tool_calls.iter();
    for block in blocks {
        let Some(block_type) = block.get("type").and_then(Json::as_str) else {
            continue;
        };
        if block_type != "tool_use" {
            continue;
        }
        let Some(raw_call) = block.as_object_mut() else {
            continue;
        };
        let Some(sanitized_call) = sanitized_calls.next() else {
            break;
        };
        set_optional_string_field(raw_call, "id", Some(sanitized_call.id.as_str()));
        set_optional_string_field(raw_call, "name", Some(sanitized_call.name.as_str()));
        raw_call.insert("input".into(), sanitized_call.arguments.clone());
    }
}

fn overlay_output_text_blocks(items: &mut [Json], message_text: Option<String>) {
    let text_items = items.iter_mut().filter_map(|item| {
        (item.get("type").and_then(Json::as_str) == Some("message"))
            .then_some(item.get_mut("content"))
            .flatten()
            .and_then(Json::as_array_mut)
    });
    let Some(text) = message_text else {
        for content in text_items {
            for block in content.iter_mut() {
                if block.get("type").and_then(Json::as_str) == Some("output_text") {
                    if let Some(block) = block.as_object_mut() {
                        block.remove("text");
                    }
                }
            }
        }
        return;
    };

    let parts: Vec<&str> = text.split('\n').collect();
    for content in text_items {
        let mut text_blocks = content.iter_mut().filter_map(|block| {
            (block.get("type").and_then(Json::as_str) == Some("output_text"))
                .then_some(block.as_object_mut())
                .flatten()
        });
        for (index, block) in text_blocks.by_ref().enumerate() {
            let part = parts
                .get(index)
                .copied()
                .or_else(|| (index == 0).then_some(text.as_str()));
            set_optional_string_field(block, "text", part);
        }
    }
}

fn overlay_anthropic_text_blocks(blocks: &mut [Json], message_text: Option<String>) {
    let parts = message_text
        .as_deref()
        .map(|text| text.split('\n').collect::<Vec<_>>());
    let mut text_block_index = 0usize;

    for block in blocks {
        if block.get("type").and_then(Json::as_str) != Some("text") {
            continue;
        }
        let Some(block) = block.as_object_mut() else {
            continue;
        };
        let part = parts
            .as_ref()
            .and_then(|parts| parts.get(text_block_index).copied())
            .or_else(|| {
                (text_block_index == 0)
                    .then(|| message_text.as_deref())
                    .flatten()
            });
        set_optional_string_field(block, "text", part);
        text_block_index += 1;
    }
}

fn annotated_message_text(message: Option<&MessageContent>) -> Option<String> {
    match message? {
        MessageContent::Text(text) => Some(text.clone()),
        MessageContent::Parts(parts) => {
            let text_parts: Vec<&str> = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                })
                .collect();
            (!text_parts.is_empty()).then(|| text_parts.join("\n"))
        }
    }
}

fn set_optional_string_field(object: &mut Map<String, Json>, key: &str, value: Option<&str>) {
    match value {
        Some(value) => {
            object.insert(key.to_string(), Json::String(value.to_string()));
        }
        None => {
            object.remove(key);
        }
    }
}

fn json_string(value: &Json) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn openai_chat_finish_reason(reason: &FinishReason) -> &str {
    match reason {
        FinishReason::Complete => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolUse => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Unknown(other) => other.as_str(),
    }
}

fn openai_responses_status(reason: &FinishReason) -> &str {
    match reason {
        FinishReason::Complete => "completed",
        FinishReason::Length | FinishReason::ContentFilter => "incomplete",
        FinishReason::ToolUse => "completed",
        FinishReason::Unknown(other) => other.as_str(),
    }
}

fn anthropic_stop_reason(reason: &FinishReason) -> &str {
    match reason {
        FinishReason::Complete => "end_turn",
        FinishReason::Length => "max_tokens",
        FinishReason::ToolUse => "tool_use",
        FinishReason::ContentFilter => "refusal",
        FinishReason::Unknown(other) => other.as_str(),
    }
}

fn sanitize_serializable_with_backend<T>(
    backend: &CompiledBuiltinBackend,
    value: T,
) -> PluginResult<T>
where
    T: Serialize + DeserializeOwned,
{
    let value = serde_json::to_value(value).map_err(|err| {
        PluginError::Internal(format!(
            "failed to serialize value for PII redaction: {err}"
        ))
    })?;
    serde_json::from_value(backend.sanitize_json_preorder_dfs(value)).map_err(|err| {
        PluginError::Internal(format!(
            "failed to deserialize sanitized value for PII redaction: {err}"
        ))
    })
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
        &["unknown_field", "unsupported_value"],
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

fn default_true() -> bool {
    true
}

fn default_priority() -> i32 {
    100
}

#[cfg(test)]
#[path = "../../../tests/unit/plugins/pii_redaction/component_tests.rs"]
mod tests;
