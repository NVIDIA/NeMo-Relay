// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional observability integrations for NeMo Relay Core.

use crate::api::event::{EventNormalizationExt, is_valid_event_metadata_attribute_key};
use crate::codec::response::{AnnotatedLlmResponse, ApiSpecificResponse, Usage};
use serde::{Deserialize, Serialize};

/// Copies a projected OTLP attribute to a second attribute name.
///
/// `key` names the fully-qualified projected attribute and `alias` names the
/// additional attribute to emit with the same typed value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OtlpAttributeMapping {
    /// Fully-qualified projected attribute to copy.
    pub key: String,
    /// Additional attribute name receiving the copied value.
    pub alias: String,
}

impl OtlpAttributeMapping {
    /// Creates an attribute mapping.
    pub fn new(key: impl Into<String>, alias: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            alias: alias.into(),
        }
    }
}

#[cfg(test)]
use std::sync::Mutex;

#[cfg(test)]
pub(crate) fn test_mutex() -> &'static Mutex<()> {
    crate::shared_runtime::runtime_owner_test_mutex()
}

pub mod atif;
pub mod atof;
mod confined_fs;
pub(crate) mod manual;
pub(crate) mod openinference;
pub mod otel;
mod otel_genai;
pub mod otel_logs;
pub mod otel_metrics;
mod otel_signal;
pub mod plugin_component;
mod private_file;

pub use otel_signal::{OpenTelemetryRuntimeDiagnostic, OpenTelemetryRuntimeDiagnostics};

/// Return the provider-independent input total used by semantic observability
/// projections. Anthropic reports uncached, cache-read, and cache-creation
/// input tokens separately, while providers such as OpenAI include cache reads
/// in their prompt count.
pub(crate) fn input_tokens_including_cache(
    provider: Option<&str>,
    response: Option<&AnnotatedLlmResponse>,
    usage: &Usage,
) -> Option<u64> {
    let prompt_tokens = usage.prompt_tokens?;
    if !uses_separate_anthropic_cache_tokens(provider, response) {
        return Some(prompt_tokens);
    }
    [usage.cache_read_tokens, usage.cache_write_tokens]
        .into_iter()
        .flatten()
        .try_fold(prompt_tokens, u64::checked_add)
}

/// Return the provider-independent prompt-plus-completion total used by
/// semantic observability projections.
pub(crate) fn total_tokens_including_cache(
    provider: Option<&str>,
    response: Option<&AnnotatedLlmResponse>,
    usage: &Usage,
) -> Option<u64> {
    if !uses_separate_anthropic_cache_tokens(provider, response) {
        return usage.total_tokens;
    }
    match (
        input_tokens_including_cache(provider, response, usage),
        usage.completion_tokens,
    ) {
        (Some(input), Some(output)) => input.checked_add(output),
        _ => usage.total_tokens,
    }
}

fn uses_separate_anthropic_cache_tokens(
    provider: Option<&str>,
    response: Option<&AnnotatedLlmResponse>,
) -> bool {
    response.is_some_and(|response| {
        matches!(
            response.api_specific.as_ref(),
            Some(ApiSpecificResponse::AnthropicMessages { .. })
        )
    }) || provider.is_some_and(|provider| {
        let provider = provider.to_ascii_lowercase();
        provider.starts_with("anthropic") || provider.starts_with("claude")
    })
}

/// Export representation for point-in-time mark events.
///
/// Marks remain canonical ATOF events regardless of this setting. Exporters
/// apply the selected projection only when translating those events into a
/// downstream trace format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum MarkProjection {
    /// Use each exporter’s native handling for marks.
    #[default]
    Inherit,
    /// Force marks into exporter-native trace span events.
    Event,
    /// Render non-excluded marks as zero-duration trace child spans so
    /// trace-tree consumers can display them directly. High-volume
    /// `llm.chunk` marks remain exporter-native events.
    Tool,
}

/// Semantic projection emitted by an OpenTelemetry exporter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum OpenTelemetryType {
    /// Relay's complete lifecycle projection, including `nemo_relay.*` attributes.
    #[default]
    Full,
    /// OpenTelemetry GenAI semantic conventions.
    GenAi,
    /// OpenInference semantic conventions.
    #[serde(rename = "openinference")]
    OpenInference,
}

/// Default mark names excluded from tool projection because they are emitted
/// at high volume and are better represented as exporter-native events.
/// Return the default mark names excluded from OpenTelemetry projections.
pub fn default_mark_exclude_names() -> Vec<String> {
    vec!["llm.chunk".to_string()]
}

pub(crate) fn relay_trace_id(uuid: uuid::Uuid) -> opentelemetry::trace::TraceId {
    opentelemetry::trace::TraceId::from_bytes(*uuid.as_bytes())
}

pub(crate) fn relay_span_id(uuid: uuid::Uuid) -> opentelemetry::trace::SpanId {
    let mut bytes = [0; 8];
    bytes.copy_from_slice(&uuid.as_bytes()[8..]);
    opentelemetry::trace::SpanId::from_bytes(bytes)
}

/// Format a W3C traceparent from Relay's deterministic trace and span IDs.
pub(crate) fn format_traceparent(trace_uuid: uuid::Uuid, span_uuid: uuid::Uuid) -> String {
    let trace_id = trace_uuid
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let span_id = &span_uuid.as_bytes()[8..];
    let span_id = span_id
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("00-{trace_id}-{span_id}-01")
}

pub(crate) fn push_common_optimization_attributes(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    summary: &crate::codec::optimization::LlmOptimizationSummary,
) {
    push_optimization_models_and_tokens(attributes, summary);
    push_optimization_cost(attributes, "baseline", summary.baseline_cost.as_ref());
    push_optimization_cost(attributes, "actual", summary.actual_cost.as_ref());
    push_optimization_savings_and_status(attributes, summary);
    push_optimization_pricing_provenance(attributes, summary);
}

fn push_optimization_models_and_tokens(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    summary: &crate::codec::optimization::LlmOptimizationSummary,
) {
    if let Some(model) = summary.baseline_model.as_ref() {
        attributes.push(opentelemetry::KeyValue::new(
            "nemo_relay.llm.optimization.baseline_model",
            model.model.clone(),
        ));
    }
    if let Some(model) = summary.effective_model.as_ref() {
        attributes.push(opentelemetry::KeyValue::new(
            "nemo_relay.llm.optimization.effective_model",
            model.model.clone(),
        ));
    }
    if let Some(tokens) = summary.tokens_saved.prompt_tokens {
        attributes.push(opentelemetry::KeyValue::new(
            "nemo_relay.llm.optimization.prompt_tokens_saved",
            i64::try_from(tokens).unwrap_or(i64::MAX),
        ));
    }
    if let Some(tokens) = summary.tokens_saved.total_tokens {
        attributes.push(opentelemetry::KeyValue::new(
            "nemo_relay.llm.optimization.total_tokens_saved",
            i64::try_from(tokens).unwrap_or(i64::MAX),
        ));
    }
}

fn push_optimization_cost(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    label: &str,
    cost: Option<&crate::codec::response::CostEstimate>,
) {
    let Some(cost) = cost else {
        return;
    };
    if let Some(total) = cost.total_or_component_sum() {
        attributes.push(opentelemetry::KeyValue::new(
            format!("nemo_relay.llm.optimization.{label}_cost"),
            total,
        ));
    }
    attributes.push(opentelemetry::KeyValue::new(
        format!("nemo_relay.llm.optimization.{label}_cost_currency"),
        cost.currency.clone(),
    ));
    if let Some(source) = cost.pricing_source.as_ref() {
        attributes.push(opentelemetry::KeyValue::new(
            format!("nemo_relay.llm.optimization.{label}_pricing_source"),
            source.clone(),
        ));
    }
    if let Some(as_of) = cost.pricing_as_of.as_ref() {
        attributes.push(opentelemetry::KeyValue::new(
            format!("nemo_relay.llm.optimization.{label}_pricing_as_of"),
            as_of.clone(),
        ));
    }
}

fn push_optimization_savings_and_status(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    summary: &crate::codec::optimization::LlmOptimizationSummary,
) {
    if let Some(saved) = summary.estimated_cost_saved {
        attributes.push(opentelemetry::KeyValue::new(
            "nemo_relay.llm.optimization.estimated_cost_saved",
            saved,
        ));
        if let Some(currency) = summary.currency.as_ref() {
            attributes.push(opentelemetry::KeyValue::new(
                "nemo_relay.llm.optimization.estimated_cost_saved_currency",
                currency.clone(),
            ));
        }
    }
    if let Some(currency) = summary.currency.as_ref() {
        attributes.push(opentelemetry::KeyValue::new(
            "nemo_relay.llm.optimization.currency",
            currency.clone(),
        ));
    }
    let status = match summary.status {
        crate::codec::optimization::LlmOptimizationSummaryStatus::Complete => "complete",
        crate::codec::optimization::LlmOptimizationSummaryStatus::Partial => "partial",
    };
    attributes.push(opentelemetry::KeyValue::new(
        "nemo_relay.llm.optimization.status",
        status,
    ));
}

fn push_optimization_pricing_provenance(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    summary: &crate::codec::optimization::LlmOptimizationSummary,
) {
    let source = summary
        .baseline_cost
        .as_ref()
        .and_then(|cost| cost.pricing_source.as_ref())
        .or_else(|| {
            summary
                .actual_cost
                .as_ref()
                .and_then(|cost| cost.pricing_source.as_ref())
        });
    if let Some(source) = source {
        attributes.push(opentelemetry::KeyValue::new(
            "nemo_relay.llm.optimization.pricing_source",
            source.clone(),
        ));
    }
    let as_of = summary
        .baseline_cost
        .as_ref()
        .and_then(|cost| cost.pricing_as_of.as_ref())
        .or_else(|| {
            summary
                .actual_cost
                .as_ref()
                .and_then(|cost| cost.pricing_as_of.as_ref())
        });
    if let Some(as_of) = as_of {
        attributes.push(opentelemetry::KeyValue::new(
            "nemo_relay.llm.optimization.pricing_as_of",
            as_of.clone(),
        ));
    }
}

/// Validates OTLP attribute mappings shared by exporter configuration surfaces.
pub fn validate_attribute_mappings(
    mappings: &[OtlpAttributeMapping],
) -> std::result::Result<(), String> {
    let mut aliases = std::collections::HashSet::new();
    for mapping in mappings {
        if is_blank_attribute_mapping_name(&mapping.key) {
            return Err("attribute mapping key must not be blank".to_string());
        }
        if is_blank_attribute_mapping_name(&mapping.alias) {
            return Err("attribute mapping alias must not be blank".to_string());
        }
        if !aliases.insert(mapping.alias.trim()) {
            return Err(format!(
                "attribute mapping alias {:?} is duplicated",
                mapping.alias
            ));
        }
    }
    Ok(())
}

/// Validates literal Event metadata prefixes promoted to OTLP attributes.
pub fn validate_metadata_promotion_prefixes(
    prefixes: &[String],
) -> std::result::Result<(), String> {
    let mut unique = std::collections::HashSet::new();
    for prefix in prefixes {
        if is_blank_attribute_mapping_name(prefix) {
            return Err("metadata promotion prefix must not be blank".to_string());
        }
        if prefix.trim() != prefix {
            return Err(format!(
                "metadata promotion prefix {prefix:?} must not have surrounding whitespace"
            ));
        }
        if prefix.contains(['*', '?', '[', ']']) {
            return Err(format!(
                "metadata promotion prefix {prefix:?} must be a literal prefix, not a glob"
            ));
        }
        if !is_valid_metadata_promotion_prefix(prefix) {
            return Err(format!(
                "metadata promotion prefix {prefix:?} must contain letter, number, underscore, or hyphen segments separated by single dots, with an optional trailing dot"
            ));
        }
        if !unique.insert(prefix.as_str()) {
            return Err(format!(
                "metadata promotion prefix {prefix:?} is duplicated"
            ));
        }
    }
    Ok(())
}

fn is_valid_metadata_promotion_prefix(prefix: &str) -> bool {
    let key = prefix.strip_suffix('.').unwrap_or(prefix);
    is_valid_event_metadata_attribute_key(key)
}

fn is_blank_attribute_mapping_name(value: &str) -> bool {
    value.chars().all(|character| {
        character.is_whitespace()
            || matches!(
                unicode_general_category::get_general_category(character),
                unicode_general_category::GeneralCategory::Control
                    | unicode_general_category::GeneralCategory::Format
            )
    })
}

/// Projects only top-level JSON fields as OTLP attributes.
///
/// Nested objects and arrays remain JSON strings so arbitrary payloads do not
/// create ambiguous dotted attribute paths or unbounded attribute sets.
pub(crate) fn push_top_level_json_attributes(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    prefix: &str,
    value: Option<&crate::json::Json>,
) {
    let Some(value) = value else {
        return;
    };
    match value {
        crate::json::Json::Object(values) => {
            for (field, value) in values {
                push_top_level_json_value(attributes, &format!("{prefix}.{field}"), value);
            }
        }
        value => push_top_level_json_value(attributes, prefix, value),
    }
}

/// Project an opaque tool-result annotation as one JSON-valued attribute.
///
/// The annotation is deliberately not flattened because its schema belongs to
/// the application rather than Relay.
pub(crate) fn push_tool_result_annotation_attribute(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    event: &crate::api::event::Event,
) {
    if event.scope_category() != Some(crate::api::event::ScopeCategory::End)
        || event.category() != Some(&crate::api::event::EventCategory::tool())
    {
        return;
    }
    let Some(annotation) = tool_result_annotation(event) else {
        return;
    };
    if let Ok(value) = serde_json::to_string(&annotation) {
        attributes.push(opentelemetry::KeyValue::new(
            "nemo_relay.tool.result.annotation",
            value,
        ));
    }
}

pub(crate) fn tool_result_annotation(
    event: &crate::api::event::Event,
) -> Option<crate::json::Json> {
    event.tool_result_annotation()
}

/// Adds canonical session-correlation attributes from event metadata and the
/// active scope-stack instance.
pub(crate) fn push_session_identity_attributes(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    event: &crate::api::event::Event,
) {
    use opentelemetry::KeyValue;

    let metadata = event.metadata();
    if let Some(session_id) = metadata
        .and_then(|value| value.get("session_id"))
        .and_then(crate::json::Json::as_str)
    {
        attributes.push(KeyValue::new("session.id", session_id.to_string()));
    }
    if let Some(user_id) = metadata
        .and_then(|value| value.get("user_id"))
        .and_then(crate::json::Json::as_str)
    {
        attributes.push(KeyValue::new("user.id", user_id.to_string()));
    }
    if let Some(agent_kind) = metadata
        .and_then(|value| value.get("agent_kind"))
        .and_then(crate::json::Json::as_str)
    {
        attributes.push(KeyValue::new(
            "nemo_relay.agent.kind",
            agent_kind.to_string(),
        ));
    }
    if let Some(root_uuid) = event.propagation_root_uuid() {
        attributes.push(KeyValue::new(
            "nemo_relay.session.instance_id",
            root_uuid.to_string(),
        ));
    }
}

/// Serializes a value and projects its top-level JSON fields as OTLP attributes.
pub(crate) fn push_serialized_top_level_attributes<T: Serialize + ?Sized>(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    prefix: &str,
    value: Option<&T>,
) {
    let Some(value) = value else {
        return;
    };
    if let Ok(value) = serde_json::to_value(value) {
        push_top_level_json_attributes(attributes, prefix, Some(&value));
    }
}

fn push_top_level_json_value(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    key: &str,
    value: &crate::json::Json,
) {
    use opentelemetry::KeyValue;

    match value {
        crate::json::Json::Null => {}
        crate::json::Json::Bool(value) => attributes.push(KeyValue::new(key.to_string(), *value)),
        crate::json::Json::String(value) => {
            attributes.push(KeyValue::new(key.to_string(), value.clone()))
        }
        crate::json::Json::Number(value) => {
            if let Some(value) = value.as_i64() {
                attributes.push(KeyValue::new(key.to_string(), value));
            } else if let Some(value) = value.as_u64() {
                if let Ok(value) = i64::try_from(value) {
                    attributes.push(KeyValue::new(key.to_string(), value));
                } else {
                    attributes.push(KeyValue::new(key.to_string(), value.to_string()));
                }
            } else if let Some(value) = value.as_f64() {
                attributes.push(KeyValue::new(key.to_string(), value));
            }
        }
        crate::json::Json::Array(_) | crate::json::Json::Object(_) => {
            if let Ok(value) = serde_json::to_string(value) {
                attributes.push(KeyValue::new(key.to_string(), value));
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetadataPromotionIssue {
    pub(crate) key: String,
    pub(crate) reason: &'static str,
}

/// The result of promoting Event metadata into typed OTLP attributes.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct MetadataPromotionResult {
    pub(crate) issues: Vec<MetadataPromotionIssue>,
    pub(crate) promoted_keys: std::collections::HashSet<String>,
}

const RESERVED_OTEL_ATTRIBUTE_NAMESPACES: &[&str] = &[
    "error.",
    "exception.",
    "gen_ai.",
    "input.",
    "llm.",
    "nemo_relay.",
    "openinference.",
    "output.",
    "server.",
    "service.",
    "session.",
    "tool.",
    "tool_call.",
    "user.",
];

fn is_reserved_otel_attribute_key(key: &str) -> bool {
    key == "metadata"
        || RESERVED_OTEL_ATTRIBUTE_NAMESPACES
            .iter()
            .any(|namespace| key.starts_with(namespace))
}

/// Copies selected top-level Event metadata entries to typed OTLP attributes.
///
/// Existing projection-owned attributes always win. Metadata is read without
/// modification, and unsupported values are returned to the caller for
/// bounded runtime diagnostics.
pub(crate) fn promote_event_metadata_attributes(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    event: &crate::api::event::Event,
    prefixes: &[String],
    protected_keys: &std::collections::HashSet<String>,
) -> MetadataPromotionResult {
    if prefixes.is_empty() {
        return MetadataPromotionResult::default();
    }
    let Some(metadata) = event.metadata().and_then(crate::json::Json::as_object) else {
        return MetadataPromotionResult::default();
    };
    let mut existing_keys = attributes
        .iter()
        .map(|attribute| attribute.key.as_str().to_string())
        .chain(protected_keys.iter().cloned())
        .collect::<std::collections::HashSet<_>>();
    let mut result = MetadataPromotionResult::default();
    for (key, value) in metadata {
        if !prefixes.iter().any(|prefix| key.starts_with(prefix)) {
            continue;
        }
        if is_reserved_otel_attribute_key(key) {
            result.issues.push(MetadataPromotionIssue {
                key: key.clone(),
                reason: "attribute key is reserved by Relay or an OpenTelemetry projection",
            });
            continue;
        }
        if existing_keys.contains(key) {
            continue;
        }
        match metadata_value_to_otel(value) {
            Ok(value) => {
                attributes.push(opentelemetry::KeyValue::new(key.clone(), value));
                existing_keys.insert(key.clone());
                result.promoted_keys.insert(key.clone());
            }
            Err(reason) => result.issues.push(MetadataPromotionIssue {
                key: key.clone(),
                reason,
            }),
        }
    }
    result
}

fn metadata_value_to_otel(
    value: &crate::json::Json,
) -> std::result::Result<opentelemetry::Value, &'static str> {
    use opentelemetry::Value;

    match value {
        crate::json::Json::String(value) => Ok(Value::String(value.clone().into())),
        crate::json::Json::Bool(value) => Ok(Value::Bool(*value)),
        crate::json::Json::Number(value) => metadata_number_to_otel(value),
        crate::json::Json::Array(values) => metadata_array_to_otel(values),
        crate::json::Json::Null => Err("null values are not OTLP attributes"),
        crate::json::Json::Object(_) => Err("object values are not supported"),
    }
}

fn metadata_number_to_otel(
    value: &serde_json::Number,
) -> std::result::Result<opentelemetry::Value, &'static str> {
    use opentelemetry::Value;

    if let Some(value) = value.as_i64() {
        return Ok(Value::I64(value));
    }
    if let Some(value) = value.as_u64() {
        return i64::try_from(value)
            .map(Value::I64)
            .map_err(|_| "unsigned integer is larger than OTLP i64");
    }
    value
        .as_f64()
        .map(Value::F64)
        .ok_or("number is not representable as an OTLP attribute")
}

fn metadata_array_to_otel(
    values: &[crate::json::Json],
) -> std::result::Result<opentelemetry::Value, &'static str> {
    use opentelemetry::{Array, Value};

    let Some(first) = values.first() else {
        return Ok(Value::Array(Array::String(Vec::new())));
    };
    match first {
        crate::json::Json::String(_) => values
            .iter()
            .map(|value| value.as_str().map(|value| value.to_string().into()))
            .collect::<Option<Vec<_>>>()
            .map(|values| Value::Array(Array::String(values)))
            .ok_or("array values must have one primitive type"),
        crate::json::Json::Bool(_) => values
            .iter()
            .map(crate::json::Json::as_bool)
            .collect::<Option<Vec<_>>>()
            .map(|values| Value::Array(Array::Bool(values)))
            .ok_or("array values must have one primitive type"),
        crate::json::Json::Number(_) => metadata_number_array_to_otel(values),
        crate::json::Json::Null => Err("arrays of null are not OTLP attributes"),
        crate::json::Json::Array(_) | crate::json::Json::Object(_) => {
            Err("nested arrays and objects are not supported")
        }
    }
}

fn metadata_number_array_to_otel(
    values: &[crate::json::Json],
) -> std::result::Result<opentelemetry::Value, &'static str> {
    use opentelemetry::{Array, Value};

    let numbers = values
        .iter()
        .map(crate::json::Json::as_number)
        .collect::<Option<Vec<_>>>()
        .ok_or("array values must have one primitive type")?;
    let integers = numbers
        .iter()
        .map(|value| {
            value
                .as_i64()
                .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
        })
        .collect::<Option<Vec<_>>>();
    if let Some(values) = integers {
        return Ok(Value::Array(Array::I64(values)));
    }
    if numbers
        .iter()
        .any(|value| value.as_u64().is_some_and(|value| value > i64::MAX as u64))
    {
        return Err("array contains an unsigned integer larger than OTLP i64");
    }
    numbers
        .iter()
        .map(|value| value.as_f64())
        .collect::<Option<Vec<_>>>()
        .map(|values| Value::Array(Array::F64(values)))
        .ok_or("number array is not representable as an OTLP attribute")
}
pub(crate) fn apply_attribute_mappings(
    attributes: &mut Vec<opentelemetry::KeyValue>,
    mappings: &[OtlpAttributeMapping],
) {
    attributes.extend(attribute_mapping_aliases(attributes, mappings));
}

/// Keeps the start attributes needed to resolve mappings at the end of a span.
///
/// The final span attributes must still take precedence over mapped aliases, so
/// retain both mapped source keys and aliases that were already present at
/// start. The span itself owns all other start attributes and does not need a
/// second copy in the active-span state.
pub(crate) fn attribute_mapping_inputs(
    attributes: &[opentelemetry::KeyValue],
    mappings: &[OtlpAttributeMapping],
) -> Vec<opentelemetry::KeyValue> {
    attributes
        .iter()
        .filter(|attribute| {
            mappings.iter().any(|mapping| {
                attribute.key.as_str() == mapping.key || attribute.key.as_str() == mapping.alias
            })
        })
        .cloned()
        .collect()
}

/// Resolves typed aliases from a complete set of projected attributes.
///
/// Callers that project a span across multiple lifecycle events must pass every
/// real span attribute so projected fields always take precedence over aliases.
pub(crate) fn attribute_mapping_aliases(
    projected_attributes: &[opentelemetry::KeyValue],
    mappings: &[OtlpAttributeMapping],
) -> Vec<opentelemetry::KeyValue> {
    if mappings.is_empty() {
        return Vec::new();
    }
    let existing = projected_attributes
        .iter()
        .map(|attribute| attribute.key.as_str().to_string())
        .collect::<std::collections::HashSet<_>>();
    mappings
        .iter()
        .filter(|mapping| !existing.contains(mapping.alias.as_str()))
        .filter_map(|mapping| {
            projected_attributes
                .iter()
                .rev()
                .find(|attribute| attribute.key.as_str() == mapping.key)
                .map(|attribute| {
                    opentelemetry::KeyValue::new(mapping.alias.clone(), attribute.value.clone())
                })
        })
        .collect()
}

/// Returns whether a mark matches a configured projection exclusion.
///
/// Agent hook adapters may preserve the canonical event name in metadata while
/// using a generic mark name, so both representations are matched.
pub(crate) fn mark_name_is_excluded(
    event: &crate::api::event::Event,
    excluded_names: &[String],
) -> bool {
    excluded_names.iter().any(|name| {
        event.name() == name
            || event
                .metadata()
                .and_then(crate::json::Json::as_object)
                .and_then(|metadata| metadata.get("hook_event_name"))
                .and_then(crate::json::Json::as_str)
                == Some(name.as_str())
    })
}

/// Resolves a configured mark projection for one event.
///
/// Exclusions only affect tool projection; all other modes retain their
/// configured exporter-native behavior.
pub(crate) fn effective_mark_projection(
    event: &crate::api::event::Event,
    projection: MarkProjection,
    excluded_names: &[String],
) -> MarkProjection {
    if projection == MarkProjection::Tool && mark_name_is_excluded(event, excluded_names) {
        MarkProjection::Inherit
    } else {
        projection
    }
}

#[cfg(test)]
#[path = "../../tests/unit/observability/exporter_parity_tests.rs"]
mod exporter_parity_tests;

pub(crate) fn estimate_cost_for_response_or_requested_model(
    event: &crate::api::event::Event,
    response_model: Option<&str>,
    usage: &crate::codec::response::Usage,
) -> Option<crate::codec::response::CostEstimate> {
    estimate_cost_for_response_or_model(
        Some(event.name()),
        event.model_name(),
        response_model,
        usage,
    )
}

pub(crate) fn estimate_cost_for_response_or_model(
    provider: Option<&str>,
    requested_model: Option<&str>,
    response_model: Option<&str>,
    usage: &crate::codec::response::Usage,
) -> Option<crate::codec::response::CostEstimate> {
    // Prefer the provider-echoed model, but fall back to the requested model
    // when pricing does not recognize the echoed model alias.
    if let Some(model_name) = response_model
        && let Some(cost) =
            crate::codec::response::estimate_cost_for_provider(provider, model_name, usage)
    {
        return Some(cost);
    }

    let requested_model = requested_model?;
    if response_model == Some(requested_model) {
        return None;
    }
    crate::codec::response::estimate_cost_for_provider(provider, requested_model, usage)
}

pub(crate) fn merge_usage(
    primary: Option<&crate::codec::response::Usage>,
    secondary: Option<&crate::codec::response::Usage>,
) -> Option<crate::codec::response::Usage> {
    match (primary, secondary) {
        (None, None) => None,
        (None, Some(usage)) | (Some(usage), None) => Some(usage.clone()),
        (Some(primary), Some(secondary)) => Some(crate::codec::response::Usage {
            prompt_tokens: primary.prompt_tokens.or(secondary.prompt_tokens),
            completion_tokens: primary.completion_tokens.or(secondary.completion_tokens),
            total_tokens: primary.total_tokens.or(secondary.total_tokens),
            cache_read_tokens: primary.cache_read_tokens.or(secondary.cache_read_tokens),
            cache_write_tokens: primary.cache_write_tokens.or(secondary.cache_write_tokens),
            cost: primary.cost.clone().or_else(|| secondary.cost.clone()),
        }),
    }
}

pub(crate) fn model_name_for_llm_event(event: &crate::api::event::Event) -> Option<String> {
    if event.category().map(|category| category.as_str()) != Some("llm") {
        return None;
    }
    let manual_response_model =
        manual::model_name_from_manual_llm_output(event.output()).map(ToOwned::to_owned);
    let manual_request_model =
        manual::model_name_from_manual_llm_output(event.input()).map(ToOwned::to_owned);
    event
        .normalized_llm_response()
        .and_then(|response| response.as_ref().model.clone())
        .or(manual_response_model)
        .or_else(|| event.model_name().map(ToOwned::to_owned))
        .or_else(|| {
            event
                .normalized_llm_request()
                .and_then(|request| request.as_ref().model.clone())
        })
        .or(manual_request_model)
}

pub(crate) fn set_span_status_from_event_metadata<S>(span: &mut S, event: &crate::api::event::Event)
where
    S: opentelemetry::trace::Span,
{
    let Some(metadata) = event.metadata() else {
        return;
    };
    let Some(status_code) = metadata
        .get("otel.status_code")
        .and_then(crate::json::Json::as_str)
    else {
        return;
    };

    let status = match status_code {
        "OK" => opentelemetry::trace::Status::Ok,
        "ERROR" => opentelemetry::trace::Status::error(
            metadata
                .get("otel.status_description")
                .and_then(crate::json::Json::as_str)
                .unwrap_or_default()
                .to_string(),
        ),
        "UNSET" => opentelemetry::trace::Status::Unset,
        _ => {
            log::warn!(
                target: "nemo_relay.observability",
                event = "invalid_status_code",
                status_code = "invalid";
                "Unrecognized OpenTelemetry status code; using unset status"
            );
            opentelemetry::trace::Status::Unset
        }
    };
    span.set_status(status);
}

#[cfg(test)]
#[path = "../../tests/unit/observability/attribute_projection_tests.rs"]
mod attribute_projection_tests;

#[cfg(test)]
#[path = "../../tests/unit/observability/mod_tests.rs"]
mod tests;
