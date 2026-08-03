// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process Rampart PII redaction plugin.

use std::collections::HashSet;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use nemo_relay::codec::resolve::supported_codec_names;
use nemo_relay::plugin::{
    ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, Plugin, PluginComponentSpec, PluginError,
    PluginRegistrationContext, Result as PluginResult, UnsupportedBehavior,
    apply_global_config_policy, deregister_plugin, register_plugin,
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

/// One configured Rampart PII component.
#[derive(Debug, Clone)]
pub struct ComponentSpec {
    /// Whether the component should be activated.
    pub enabled: bool,
    /// Component-local Rampart configuration.
    pub config: RampartPiiConfig,
}

impl ComponentSpec {
    /// Creates an enabled Rampart PII component spec.
    pub fn new(config: RampartPiiConfig) -> Self {
        Self {
            enabled: true,
            config,
        }
    }
}

impl From<ComponentSpec> for PluginComponentSpec {
    fn from(value: ComponentSpec) -> Self {
        let Json::Object(config) = serde_json::to_value(value.config)
            .expect("Rampart PII config should serialize to an object")
        else {
            unreachable!("Rampart PII config must serialize to an object");
        };
        PluginComponentSpec {
            kind: RAMPART_PII_PLUGIN_KIND.to_string(),
            enabled: value.enabled,
            config,
        }
    }
}

/// Configuration for the in-process Rampart PII component.
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
            target_paths: Vec::new(),
            target_path_patterns: Vec::new(),
            min_score: default_min_score(),
            excluded_labels: Vec::new(),
            replacement: default_replacement(),
            max_windows_per_payload: default_max_windows_per_payload(),
            inference_batch_size: default_inference_batch_size(),
            policy: ConfigPolicy::default(),
        }
    }
}

nemo_relay::editor_config! {
    impl RampartPiiConfig {
        model_path => { label: "model_path", kind: String },
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
        target_paths => {
            label: "target_paths",
            kind: List,
            list: &nemo_relay::config_editor::STRING_LIST_ITEM,
        },
        target_path_patterns => {
            label: "target_path_patterns",
            kind: List,
            list: &nemo_relay::config_editor::STRING_LIST_ITEM,
        },
        min_score => { label: "min_score", kind: Float },
        excluded_labels => {
            label: "excluded_labels",
            kind: List,
            list: &nemo_relay::config_editor::STRING_LIST_ITEM,
        },
        replacement => { label: "replacement", kind: String },
        max_windows_per_payload => { label: "max_windows_per_payload", kind: Integer },
        inference_batch_size => { label: "inference_batch_size", kind: Integer },
        policy => {
            label: "policy",
            kind: Section,
            nested: ConfigPolicy,
            default: ConfigPolicy,
        },
    }
}

struct RampartPiiPlugin;

impl Plugin for RampartPiiPlugin {
    fn plugin_kind(&self) -> &str {
        RAMPART_PII_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        validate_rampart_pii_config(plugin_config, None)
    }

    fn validate_with_policy(
        &self,
        plugin_config: &Map<String, Json>,
        policy: &ConfigPolicy,
    ) -> Vec<ConfigDiagnostic> {
        validate_rampart_pii_config(plugin_config, Some(policy))
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        let parsed = parse_config(plugin_config);
        Box::pin(async move {
            let config = parsed?;
            enforce_activation_invariants(&config)?;
            let model_path = PathBuf::from(&config.model_path);
            let max_windows = config.max_windows_per_payload;
            let batch_size = config.inference_batch_size;
            let detector = tokio::task::spawn_blocking(move || {
                RampartDetector::load(model_path, max_windows, batch_size)
            })
            .await
            .map_err(|error| {
                PluginError::Internal(format!("Rampart model initialization task failed: {error}"))
            })??;
            let sanitizer = RampartSanitizer::new(config.clone(), Arc::new(detector))?;
            register_sanitizers(&config, sanitizer, ctx)?;
            log::info!(
                target: "nemo_relay.plugin",
                event = "plugin_resource_validation_completed",
                plugin_kind = RAMPART_PII_PLUGIN_KIND,
                model_id = RAMPART_MODEL_ID,
                model_revision = RAMPART_MODEL_REVISION,
                resource_count = 1;
                "Rampart PII model loaded in the Relay process"
            );
            Ok(())
        })
    }
}

/// Registers the `pii_rampart` component kind.
pub fn register_rampart_pii_component() -> PluginResult<()> {
    match register_plugin(Arc::new(RampartPiiPlugin)) {
        Ok(()) => Ok(()),
        Err(PluginError::RegistrationFailed(message)) if message.contains("already registered") => {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

/// Deregisters the `pii_rampart` component kind.
pub fn deregister_rampart_pii_component() -> bool {
    deregister_plugin(RAMPART_PII_PLUGIN_KIND)
}

/// Returns the JSON Schema for Rampart PII configuration.
#[cfg(feature = "schema")]
pub fn rampart_pii_config_schema() -> Json {
    serde_json::to_value(schemars::schema_for!(RampartPiiConfig))
        .expect("Rampart PII config schema should serialize")
}

fn register_sanitizers(
    config: &RampartPiiConfig,
    sanitizer: RampartSanitizer,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    if config.mark {
        ctx.register_mark_sanitize_guardrail(
            "mark",
            config.priority,
            event_sanitize_callback(sanitizer.clone(), None),
        )?;
    }
    if config.tool_input {
        ctx.register_tool_sanitize_request_guardrail(
            "tool_input",
            config.priority,
            tool_sanitize_callback(sanitizer.clone()),
        )?;
    }
    if config.tool_output {
        ctx.register_tool_sanitize_response_guardrail(
            "tool_output",
            config.priority,
            tool_sanitize_callback(sanitizer.clone()),
        )?;
    }
    if config.input {
        ctx.register_llm_sanitize_request_guardrail(
            "input",
            config.priority,
            llm_sanitize_request_callback(sanitizer.clone()),
        )?;
    }
    if config.input || config.tool_input {
        ctx.register_scope_sanitize_start_guardrail(
            "scope_start",
            config.priority,
            event_sanitize_callback(sanitizer.clone(), Some((config.input, config.tool_input))),
        )?;
    }
    if config.output {
        ctx.register_llm_sanitize_response_guardrail(
            "output",
            config.priority,
            llm_sanitize_response_callback(sanitizer.clone()),
        )?;
    }
    if config.output || config.tool_output {
        ctx.register_scope_sanitize_end_guardrail(
            "scope_end",
            config.priority,
            event_sanitize_callback(sanitizer, Some((config.output, config.tool_output))),
        )?;
    }
    Ok(())
}

fn parse_config(plugin_config: &Map<String, Json>) -> PluginResult<RampartPiiConfig> {
    serde_json::from_value(Json::Object(plugin_config.clone())).map_err(|error| {
        PluginError::InvalidConfig(format!("invalid Rampart PII plugin config: {error}"))
    })
}

fn validate_rampart_pii_config(
    plugin_config: &Map<String, Json>,
    global_policy: Option<&ConfigPolicy>,
) -> Vec<ConfigDiagnostic> {
    let mut config = match parse_config(plugin_config) {
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
    if let Some(global_policy) = global_policy {
        config.policy = apply_global_config_policy(config.policy, global_policy);
    }

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
        "target_paths",
        "target_path_patterns",
        "min_score",
        "excluded_labels",
        "replacement",
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
        && !supported_codec_names().contains(&codec)
    {
        violations.push(ConfigViolation::new(
            "codec",
            "codec must be 'openai_chat', 'openai_responses', or 'anthropic_messages'",
        ));
    }
    if config.target_paths.is_empty() && config.target_path_patterns.is_empty() {
        violations.push(ConfigViolation::new(
            "target_paths",
            "target_paths or target_path_patterns must select explicit content fields",
        ));
    }
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
        ["openai_chat", "openai_responses", "anthropic_messages"]
            .into_iter()
            .map(|value| Json::String(value.into()))
            .collect(),
    );
    schema.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> Map<String, Json> {
        let Json::Object(config) = serde_json::to_value(RampartPiiConfig {
            model_path: "/tmp/rampart".into(),
            target_path_patterns: vec!["/messages/*/content".into()],
            ..RampartPiiConfig::default()
        })
        .unwrap() else {
            unreachable!()
        };
        config
    }

    #[test]
    fn validates_explicit_model_and_content_paths() {
        assert!(validate_rampart_pii_config(&valid_config(), None).is_empty());

        let mut config = valid_config();
        config.insert("model_path".into(), Json::String("relative/model".into()));
        config.insert(
            "target_path_patterns".into(),
            serde_json::json!(["/messages/pre*fix/content"]),
        );
        let diagnostics = validate_rampart_pii_config(&config, None);
        assert!(
            diagnostics
                .iter()
                .any(|item| item.field.as_deref() == Some("model_path"))
        );
        assert!(
            diagnostics
                .iter()
                .any(|item| item.field.as_deref() == Some("target_path_patterns"))
        );
    }

    #[test]
    fn bounds_the_configurable_window_budget() {
        assert_eq!(RampartPiiConfig::default().max_windows_per_payload, 4);

        let mut config = valid_config();
        config.insert(
            "max_windows_per_payload".into(),
            Json::from(MAX_WINDOWS_PER_PAYLOAD),
        );
        assert!(validate_rampart_pii_config(&config, None).is_empty());

        config.insert(
            "max_windows_per_payload".into(),
            Json::from(MAX_WINDOWS_PER_PAYLOAD + 1),
        );
        assert!(
            validate_rampart_pii_config(&config, None)
                .iter()
                .any(|item| item.field.as_deref() == Some("max_windows_per_payload"))
        );

        config.insert("max_windows_per_payload".into(), Json::from(0_usize));
        assert!(
            validate_rampart_pii_config(&config, None)
                .iter()
                .any(|item| item.field.as_deref() == Some("max_windows_per_payload"))
        );
    }

    #[tokio::test]
    async fn registration_enforces_safety_invariants_when_policy_warns() {
        let cases = [
            (
                "target_paths",
                serde_json::json!(["messages/0/content"]),
                "target_paths entries",
            ),
            (
                "target_paths",
                serde_json::json!([""]),
                "target_paths entries",
            ),
            (
                "target_path_patterns",
                serde_json::json!([""]),
                "target_path_patterns entries",
            ),
            ("min_score", serde_json::json!(1.1), "min_score must"),
        ];

        for (field, value, expected) in cases {
            let mut config = valid_config();
            config.insert(
                "policy".into(),
                serde_json::json!({"unsupported_value": "warn"}),
            );
            config.insert(field.into(), value);
            let diagnostics = validate_rampart_pii_config(&config, None);
            assert!(
                diagnostics.iter().any(|diagnostic| {
                    diagnostic.level == DiagnosticLevel::Warning
                        && diagnostic.field.as_deref() == Some(field)
                }),
                "expected a warning for {field}: {diagnostics:?}"
            );

            let plugin = RampartPiiPlugin;
            let mut context = PluginRegistrationContext::with_namespace("rampart-test::");
            let error = plugin
                .register(&config, &mut context)
                .await
                .expect_err("unsafe configuration must fail registration");
            assert!(
                error.to_string().contains(expected),
                "unexpected registration error for {field}: {error}"
            );
        }
    }

    #[test]
    fn component_spec_uses_independent_plugin_kind() {
        let spec: PluginComponentSpec = ComponentSpec::new(RampartPiiConfig {
            model_path: "/tmp/rampart".into(),
            target_paths: vec!["/message".into()],
            ..RampartPiiConfig::default()
        })
        .into();
        assert_eq!(spec.kind, RAMPART_PII_PLUGIN_KIND);
        assert_ne!(spec.kind, crate::component::PII_REDACTION_PLUGIN_KIND);
    }
}
