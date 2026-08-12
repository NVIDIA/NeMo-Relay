// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rampart PII model and sanitizer support for the opt-in native plugin.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use nemo_relay::plugin::{
    ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, PluginError, Result as PluginResult,
    UnsupportedBehavior,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

mod model;
mod prefilter;
mod sanitizer;
mod tokenizer;

use model::RampartDetector;
use sanitizer::{
    RampartSanitizer, event_sanitize_callback, llm_sanitize_request_callback,
    llm_sanitize_response_callback, tool_sanitize_callback,
};

use nemo_relay::api::runtime::{
    EventSanitizeFn, LlmSanitizeRequestFn, LlmSanitizeResponseFn, ToolSanitizeFn,
};

/// Plugin kind for in-process Rampart PII redaction.
pub const RAMPART_PII_PLUGIN_KIND: &str = "pii_rampart";
/// Pinned Hugging Face model repository used by this plugin.
pub const RAMPART_MODEL_ID: &str = "nationaldesignstudio/rampart";
/// Pinned model revision whose files are accepted by this plugin.
pub const RAMPART_MODEL_REVISION: &str = "b1993e4e68b082835b80ffc65acc03325ea2e501";

const MAX_MODEL_PATH_BYTES: usize = 4096;
const MAX_TARGET_PATHS: usize = 256;
const MAX_TARGET_PATH_BYTES: usize = 1024;
const MAX_EXCLUDED_LABELS: usize = 128;
const MAX_LABEL_BYTES: usize = 128;
const MAX_REPLACEMENT_BYTES: usize = 1024;
const MAX_WINDOWS_PER_PAYLOAD: usize = 16;

/// Configuration for the opt-in Rampart PII native plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RampartPiiConfig {
    /// Configuration schema version.
    #[serde(default = "default_config_version")]
    pub version: u32,
    /// Local directory containing the pinned Rampart snapshot.
    pub model_path: String,
    /// Whether to sanitize managed LLM request payloads.
    #[serde(default = "default_true")]
    pub input: bool,
    /// Whether to sanitize managed LLM response payloads.
    #[serde(default = "default_true")]
    pub output: bool,
    /// Whether to sanitize mark event observability fields.
    #[serde(default = "default_true")]
    pub mark: bool,
    /// Whether to sanitize managed tool request payloads.
    #[serde(default = "default_true")]
    pub tool_input: bool,
    /// Whether to sanitize managed tool response payloads.
    #[serde(default = "default_true")]
    pub tool_output: bool,
    /// Guardrail priority. Lower values run earlier.
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Compatibility codec for calls without an active per-call codec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "codec_schema"))]
    pub codec: Option<String>,
    /// Optional semantic content-selection preset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "preset_schema"))]
    pub preset: Option<String>,
    /// Exact JSON-pointer paths selected for model inspection.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_paths: Vec<String>,
    /// JSON-pointer patterns selected for model inspection.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_path_patterns: Vec<String>,
    /// Minimum model confidence accepted for redaction.
    #[serde(default = "default_min_score")]
    pub min_score: f64,
    /// Exact, case-sensitive model labels that remain visible.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_labels: Vec<String>,
    /// Replacement applied to accepted model spans and failed batches.
    #[serde(default = "default_replacement")]
    pub replacement: String,
    /// How the trajectory preset handles opaque custom-mark payloads.
    #[serde(default = "default_custom_mark_payload_policy")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "custom_mark_payload_policy_schema")
    )]
    pub custom_mark_payload_policy: String,
    /// Maximum token windows accepted from one observability payload.
    #[serde(default = "default_max_windows_per_payload")]
    pub max_windows_per_payload: usize,
    /// Maximum short token windows requested per model invocation.
    ///
    /// The 512 padded-token budget can reduce the actual batch size.
    #[serde(default = "default_inference_batch_size")]
    pub inference_batch_size: usize,
    /// Component-local unsupported-config policy.
    #[serde(default)]
    pub policy: ConfigPolicy,
}

impl Default for RampartPiiConfig {
    fn default() -> Self {
        Self {
            version: default_config_version(),
            model_path: String::new(),
            input: true,
            output: true,
            mark: true,
            tool_input: true,
            tool_output: true,
            priority: default_priority(),
            codec: None,
            preset: None,
            target_paths: Vec::new(),
            target_path_patterns: Vec::new(),
            min_score: default_min_score(),
            excluded_labels: Vec::new(),
            replacement: default_replacement(),
            custom_mark_payload_policy: default_custom_mark_payload_policy(),
            max_windows_per_payload: default_max_windows_per_payload(),
            inference_batch_size: default_inference_batch_size(),
            policy: ConfigPolicy::default(),
        }
    }
}

/// Async sanitizer callbacks backed by one loaded Rampart model.
///
/// This bundle is used by the separately distributed native plugin. Keeping
/// the inference implementation here lets the static and dynamic adapters
/// share the same selection, codec, admission, and fail-closed behavior.
#[derive(Clone)]
pub struct RampartMiddlewareCallbacks {
    priority: i32,
    mark: Option<EventSanitizeFn>,
    tool_input: Option<ToolSanitizeFn>,
    tool_output: Option<ToolSanitizeFn>,
    input: Option<LlmSanitizeRequestFn>,
    output: Option<LlmSanitizeResponseFn>,
    scope_start: Option<EventSanitizeFn>,
    scope_end: Option<EventSanitizeFn>,
}

impl RampartMiddlewareCallbacks {
    /// Middleware priority configured for every Rampart sanitizer surface.
    #[must_use]
    pub fn priority(&self) -> i32 {
        self.priority
    }

    /// Mark-event sanitizer, when enabled.
    #[must_use]
    pub fn mark(&self) -> Option<EventSanitizeFn> {
        self.mark.clone()
    }

    /// Tool-request sanitizer, when enabled.
    #[must_use]
    pub fn tool_input(&self) -> Option<ToolSanitizeFn> {
        self.tool_input.clone()
    }

    /// Tool-response sanitizer, when enabled.
    #[must_use]
    pub fn tool_output(&self) -> Option<ToolSanitizeFn> {
        self.tool_output.clone()
    }

    /// LLM-request sanitizer, when enabled.
    #[must_use]
    pub fn input(&self) -> Option<LlmSanitizeRequestFn> {
        self.input.clone()
    }

    /// LLM-response sanitizer, when enabled.
    #[must_use]
    pub fn output(&self) -> Option<LlmSanitizeResponseFn> {
        self.output.clone()
    }

    /// Scope-start sanitizer, when an input surface is enabled.
    #[must_use]
    pub fn scope_start(&self) -> Option<EventSanitizeFn> {
        self.scope_start.clone()
    }

    /// Scope-end sanitizer, when an output surface is enabled.
    #[must_use]
    pub fn scope_end(&self) -> Option<EventSanitizeFn> {
        self.scope_end.clone()
    }
}

/// Validate one Rampart component config without loading model files.
#[must_use]
pub fn validate_config(plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
    validate_rampart_pii_config(plugin_config)
}

/// Load the configured Rampart model and construct its async middleware.
///
/// Native plugin activation is synchronous, so model loading happens during
/// explicit plugin activation rather than on the first managed call.
pub fn load_middleware(
    plugin_config: &Map<String, Json>,
) -> PluginResult<RampartMiddlewareCallbacks> {
    let config = parse_config(plugin_config)?;
    enforce_activation_invariants(&config)?;
    let detector = RampartDetector::load(
        PathBuf::from(&config.model_path),
        config.max_windows_per_payload,
        config.inference_batch_size,
    )?;
    let middleware = build_middleware(config, Arc::new(detector))?;
    log::info!(
        target: "nemo_relay.plugin",
        event = "plugin_resource_validation_completed",
        plugin_kind = RAMPART_PII_PLUGIN_KIND,
        model_id = RAMPART_MODEL_ID,
        model_revision = RAMPART_MODEL_REVISION,
        resource_count = 1;
        "Rampart PII model loaded in the Relay process"
    );
    Ok(middleware)
}

/// Returns the JSON Schema for Rampart PII configuration.
#[cfg(feature = "schema")]
pub fn rampart_pii_config_schema() -> Json {
    serde_json::to_value(schemars::schema_for!(RampartPiiConfig))
        .expect("Rampart PII config schema should serialize")
}

fn build_middleware(
    config: RampartPiiConfig,
    detector: Arc<RampartDetector>,
) -> PluginResult<RampartMiddlewareCallbacks> {
    let sanitizer = RampartSanitizer::new(config.clone(), detector)?;
    Ok(RampartMiddlewareCallbacks {
        priority: config.priority,
        mark: config
            .mark
            .then(|| event_sanitize_callback(sanitizer.clone(), None)),
        tool_input: config
            .tool_input
            .then(|| tool_sanitize_callback(sanitizer.clone())),
        tool_output: config
            .tool_output
            .then(|| tool_sanitize_callback(sanitizer.clone())),
        input: config
            .input
            .then(|| llm_sanitize_request_callback(sanitizer.clone())),
        output: config
            .output
            .then(|| llm_sanitize_response_callback(sanitizer.clone())),
        scope_start: (config.input || config.tool_input).then(|| {
            event_sanitize_callback(sanitizer.clone(), Some((config.input, config.tool_input)))
        }),
        scope_end: (config.output || config.tool_output)
            .then(|| event_sanitize_callback(sanitizer, Some((config.output, config.tool_output)))),
    })
}

fn parse_config(plugin_config: &Map<String, Json>) -> PluginResult<RampartPiiConfig> {
    serde_json::from_value(Json::Object(plugin_config.clone())).map_err(|error| {
        PluginError::InvalidConfig(format!("invalid Rampart PII plugin config: {error}"))
    })
}

fn validate_rampart_pii_config(plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
    let config = match parse_config(plugin_config) {
        Ok(config) => config,
        Err(error) => {
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "pii_rampart.invalid_plugin_config".into(),
                component: Some(RAMPART_PII_PLUGIN_KIND.into()),
                field: None,
                message: error.to_string(),
            }];
        }
    };
    let mut diagnostics = Vec::new();
    let supported = [
        "version",
        "model_path",
        "input",
        "output",
        "mark",
        "tool_input",
        "tool_output",
        "priority",
        "codec",
        "preset",
        "target_paths",
        "target_path_patterns",
        "min_score",
        "excluded_labels",
        "replacement",
        "custom_mark_payload_policy",
        "max_windows_per_payload",
        "inference_batch_size",
        "policy",
    ];
    for field in plugin_config.keys() {
        if !supported.contains(&field.as_str()) {
            push_diagnostic(
                &mut diagnostics,
                config.policy.unknown_field,
                "pii_rampart.unknown_field",
                Some(field.clone()),
                format!("unknown field '{field}'"),
            );
        }
    }
    if let Some(Json::Object(policy)) = plugin_config.get("policy") {
        for field in policy.keys() {
            if !["unknown_component", "unknown_field", "unsupported_value"]
                .contains(&field.as_str())
            {
                push_diagnostic(
                    &mut diagnostics,
                    config.policy.unknown_field,
                    "pii_rampart.unknown_field",
                    Some(format!("policy.{field}")),
                    format!("unknown field 'policy.{field}'"),
                );
            }
        }
    }

    for violation in config_value_violations(&config) {
        push_unsupported(
            &mut diagnostics,
            &config,
            violation.field,
            violation.message,
        );
    }
    diagnostics
}

struct ConfigViolation {
    field: &'static str,
    message: String,
}

impl ConfigViolation {
    fn new(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }
}

fn enforce_activation_invariants(config: &RampartPiiConfig) -> PluginResult<()> {
    let violations = config_value_violations(config);
    if violations.is_empty() {
        return Ok(());
    }
    let details = violations
        .into_iter()
        .map(|violation| format!("{}: {}", violation.field, violation.message))
        .collect::<Vec<_>>()
        .join("; ");
    Err(PluginError::InvalidConfig(format!(
        "invalid Rampart PII plugin config: {details}"
    )))
}

fn config_value_violations(config: &RampartPiiConfig) -> Vec<ConfigViolation> {
    let mut violations = Vec::new();
    if config.version != default_config_version() {
        violations.push(ConfigViolation::new(
            "version",
            format!(
                "Rampart PII config version {} is unsupported",
                config.version
            ),
        ));
    }
    if config.model_path.trim().is_empty() || config.model_path.len() > MAX_MODEL_PATH_BYTES {
        violations.push(ConfigViolation::new(
            "model_path",
            format!("model_path must be non-empty and at most {MAX_MODEL_PATH_BYTES} UTF-8 bytes"),
        ));
    } else if !PathBuf::from(&config.model_path).is_absolute() {
        violations.push(ConfigViolation::new(
            "model_path",
            "model_path must be absolute",
        ));
    }
    if !(config.input || config.output || config.mark || config.tool_input || config.tool_output) {
        violations.push(ConfigViolation::new(
            "input",
            "at least one sanitization surface must be enabled",
        ));
    }
    if let Some(codec) = config.codec.as_deref()
        && !supported_rampart_codec_names().contains(&codec)
    {
        let supported = supported_rampart_codec_names()
            .into_iter()
            .map(|name| format!("'{name}'"))
            .collect::<Vec<_>>()
            .join(", ");
        violations.push(ConfigViolation::new(
            "codec",
            format!("codec must be one of {supported}"),
        ));
    }
    validate_content_selection(config, &mut violations);
    if config.target_paths.len() + config.target_path_patterns.len() > MAX_TARGET_PATHS {
        violations.push(ConfigViolation::new(
            "target_paths",
            format!(
                "target_paths and target_path_patterns must contain at most {MAX_TARGET_PATHS} entries"
            ),
        ));
    }
    if config.target_paths.iter().any(|path| {
        path.is_empty() || path.len() > MAX_TARGET_PATH_BYTES || !is_valid_json_pointer(path)
    }) {
        violations.push(ConfigViolation::new(
            "target_paths",
            "target_paths entries must be bounded valid JSON pointers",
        ));
    }
    if config.target_path_patterns.iter().any(|path| {
        path.is_empty()
            || path.len() > MAX_TARGET_PATH_BYTES
            || !is_valid_json_pointer_pattern(path)
    }) {
        violations.push(ConfigViolation::new(
            "target_path_patterns",
            "target_path_patterns entries must be bounded JSON pointers with only complete '*' segments",
        ));
    }
    if !config.min_score.is_finite() || !(0.0..=1.0).contains(&config.min_score) {
        violations.push(ConfigViolation::new(
            "min_score",
            "min_score must be a finite number between 0 and 1",
        ));
    }
    if config.excluded_labels.len() > MAX_EXCLUDED_LABELS
        || config
            .excluded_labels
            .iter()
            .any(|label| label.trim().is_empty() || label.len() > MAX_LABEL_BYTES)
        || config.excluded_labels.iter().collect::<HashSet<_>>().len()
            != config.excluded_labels.len()
    {
        violations.push(ConfigViolation::new(
            "excluded_labels",
            format!(
                "excluded_labels must contain at most {MAX_EXCLUDED_LABELS} unique, bounded labels"
            ),
        ));
    }
    if config.replacement.len() > MAX_REPLACEMENT_BYTES {
        violations.push(ConfigViolation::new(
            "replacement",
            format!("replacement must not exceed {MAX_REPLACEMENT_BYTES} UTF-8 bytes"),
        ));
    }
    if !(1..=MAX_WINDOWS_PER_PAYLOAD).contains(&config.max_windows_per_payload) {
        violations.push(ConfigViolation::new(
            "max_windows_per_payload",
            format!("max_windows_per_payload must be between 1 and {MAX_WINDOWS_PER_PAYLOAD}"),
        ));
    }
    if !(1..=64).contains(&config.inference_batch_size) {
        violations.push(ConfigViolation::new(
            "inference_batch_size",
            "inference_batch_size must be between 1 and 64",
        ));
    }
    violations
}

fn supported_rampart_codec_names() -> Vec<&'static str> {
    vec![
        "openai_chat",
        "openai_responses",
        "anthropic_messages",
        "gemini_generate_content",
    ]
}

fn validate_content_selection(config: &RampartPiiConfig, violations: &mut Vec<ConfigViolation>) {
    match config.preset.as_deref() {
        Some("trajectory_context") => {
            if !config.target_paths.is_empty() || !config.target_path_patterns.is_empty() {
                violations.push(ConfigViolation::new(
                    "preset",
                    "preset cannot be combined with target_paths or target_path_patterns",
                ));
            }
        }
        Some(_) => violations.push(ConfigViolation::new(
            "preset",
            "preset must be 'trajectory_context'",
        )),
        None => {
            if config.target_paths.is_empty() && config.target_path_patterns.is_empty() {
                violations.push(ConfigViolation::new(
                    "target_paths",
                    "preset, target_paths, or target_path_patterns must select content fields",
                ));
            }
            if config.custom_mark_payload_policy != "preserve" {
                violations.push(ConfigViolation::new(
                    "custom_mark_payload_policy",
                    "custom_mark_payload_policy requires preset = 'trajectory_context'",
                ));
            }
        }
    }
    if !matches!(
        config.custom_mark_payload_policy.as_str(),
        "preserve" | "redact_all_leaves"
    ) {
        violations.push(ConfigViolation::new(
            "custom_mark_payload_policy",
            "custom_mark_payload_policy must be 'preserve' or 'redact_all_leaves'",
        ));
    }
}

fn push_unsupported(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    config: &RampartPiiConfig,
    field: &str,
    message: String,
) {
    push_diagnostic(
        diagnostics,
        config.policy.unsupported_value,
        "pii_rampart.unsupported_value",
        Some(field.into()),
        message,
    );
}

fn push_diagnostic(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    behavior: UnsupportedBehavior,
    code: &str,
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
        code: code.into(),
        component: Some(RAMPART_PII_PLUGIN_KIND.into()),
        field,
        message,
    });
}

fn is_valid_json_pointer(path: &str) -> bool {
    if path.is_empty() {
        return true;
    }
    if !path.starts_with('/') {
        return false;
    }
    let mut bytes = path.as_bytes().iter().copied();
    while let Some(byte) = bytes.next() {
        if byte == b'~' && !matches!(bytes.next(), Some(b'0' | b'1')) {
            return false;
        }
    }
    true
}

fn is_valid_json_pointer_pattern(path: &str) -> bool {
    is_valid_json_pointer(path)
        && path
            .strip_prefix('/')
            .unwrap_or_default()
            .split('/')
            .all(|segment| !segment.contains('*') || segment == "*")
}

fn default_config_version() -> u32 {
    1
}

fn default_true() -> bool {
    true
}

fn default_priority() -> i32 {
    100
}

fn default_min_score() -> f64 {
    0.4
}

fn default_replacement() -> String {
    "[REDACTED]".into()
}

fn default_custom_mark_payload_policy() -> String {
    "preserve".into()
}

fn default_max_windows_per_payload() -> usize {
    4
}

fn default_inference_batch_size() -> usize {
    16
}

#[cfg(feature = "schema")]
fn codec_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
    let mut schema: schemars::schema::SchemaObject =
        <String as schemars::JsonSchema>::json_schema(generator).into();
    schema.enum_values = Some(
        supported_rampart_codec_names()
            .into_iter()
            .map(|value| Json::String(value.into()))
            .collect(),
    );
    schema.into()
}

#[cfg(feature = "schema")]
fn preset_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
    let mut schema: schemars::schema::SchemaObject =
        <String as schemars::JsonSchema>::json_schema(generator).into();
    schema.enum_values = Some(vec![Json::String("trajectory_context".into())]);
    schema.into()
}

#[cfg(feature = "schema")]
fn custom_mark_payload_policy_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    let mut schema: schemars::schema::SchemaObject =
        <String as schemars::JsonSchema>::json_schema(generator).into();
    schema.enum_values = Some(
        ["preserve", "redact_all_leaves"]
            .into_iter()
            .map(|value| Json::String(value.into()))
            .collect(),
    );
    schema.into()
}

#[cfg(test)]
#[path = "../../tests/unit/rampart/mod_tests.rs"]
mod tests;
