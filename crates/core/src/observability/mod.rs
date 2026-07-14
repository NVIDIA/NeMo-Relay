// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional observability integrations for NeMo Relay Core.

use crate::api::event::EventNormalizationExt;
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
pub(crate) mod manual;
#[cfg(feature = "openinference")]
pub mod openinference;
#[cfg(feature = "otel")]
pub mod otel;
pub mod plugin_component;

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

/// Default mark names excluded from tool projection because they are emitted
/// at high volume and are better represented as exporter-native events.
pub(crate) fn default_mark_exclude_names() -> Vec<String> {
    vec!["llm.chunk".to_string()]
}

/// Validates OTLP attribute mappings shared by exporter configuration surfaces.
pub fn validate_attribute_mappings(
    mappings: &[OtlpAttributeMapping],
) -> std::result::Result<(), String> {
    let mut aliases = std::collections::HashSet::new();
    for mapping in mappings {
        if mapping.key.trim().is_empty() {
            return Err("attribute mapping key must not be blank".to_string());
        }
        if mapping.alias.trim().is_empty() {
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

#[cfg(any(feature = "otel", feature = "openinference"))]
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

#[cfg(any(feature = "otel", feature = "openinference"))]
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

#[cfg(any(feature = "otel", feature = "openinference"))]
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

#[cfg(any(feature = "otel", feature = "openinference"))]
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
#[cfg(any(feature = "otel", feature = "openinference"))]
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
#[cfg(any(feature = "otel", feature = "openinference"))]
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
#[cfg(any(feature = "otel", feature = "openinference"))]
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
#[cfg(any(feature = "otel", feature = "openinference"))]
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

#[cfg(all(test, feature = "otel", feature = "openinference"))]
#[path = "../../tests/unit/observability/exporter_parity_tests.rs"]
mod exporter_parity_tests;

#[cfg(any(feature = "otel", feature = "openinference"))]
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
    if let Some(model_name) = event.model_name() {
        return Some(model_name.to_string());
    }
    if event.category().map(|category| category.as_str()) != Some("llm") {
        return None;
    }
    event
        .normalized_llm_response()
        .and_then(|response| response.as_ref().model.clone())
        .or_else(|| {
            event
                .normalized_llm_request()
                .and_then(|request| request.as_ref().model.clone())
        })
        .or_else(|| {
            event
                .output()
                .or_else(|| event.input())
                .and_then(|payload| manual::model_name_from_manual_llm_output(Some(payload)))
                .map(ToOwned::to_owned)
        })
}

#[cfg(any(feature = "otel", feature = "openinference"))]
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
        other => {
            eprintln!("Unrecognized OTEL status code in event metadata: {other}");
            opentelemetry::trace::Status::Unset
        }
    };
    span.set_status(status);
}

#[cfg(all(test, any(feature = "otel", feature = "openinference")))]
mod attribute_projection_tests {
    use super::{
        OtlpAttributeMapping, apply_attribute_mappings, attribute_mapping_inputs,
        push_top_level_json_attributes,
    };

    #[test]
    fn retains_only_mapping_sources_and_existing_aliases_between_span_events() {
        let attributes = vec![
            opentelemetry::KeyValue::new("source", "value"),
            opentelemetry::KeyValue::new("alias", "existing"),
            opentelemetry::KeyValue::new("large.request", "payload"),
        ];

        let retained =
            attribute_mapping_inputs(&attributes, &[OtlpAttributeMapping::new("source", "alias")]);

        assert_eq!(retained.len(), 2);
        assert!(
            retained
                .iter()
                .any(|attribute| attribute.key.as_str() == "source")
        );
        assert!(
            retained
                .iter()
                .any(|attribute| attribute.key.as_str() == "alias")
        );
    }

    #[test]
    fn projects_typed_json_and_copies_configured_aliases() {
        let mut attributes = Vec::new();
        push_top_level_json_attributes(
            &mut attributes,
            "nemo_relay.start.metadata",
            Some(&serde_json::json!({
                "tenant": "acme",
                "attempt": 2,
                "tags": ["a", "b"],
                "context": {"region": "us-east-1"},
                "request": {"id": "nested-id"},
                "request.id": "flat-id",
                "event_id": 18446744073709551615u64
            })),
        );
        apply_attribute_mappings(
            &mut attributes,
            &[OtlpAttributeMapping::new(
                "nemo_relay.start.metadata.tenant",
                "tenant.id",
            )],
        );

        let values = attributes
            .iter()
            .map(|attribute| (attribute.key.as_str(), attribute.value.to_string()))
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(
            values.get("nemo_relay.start.metadata.tenant"),
            Some(&"acme".to_string())
        );
        assert_eq!(
            values.get("nemo_relay.start.metadata.attempt"),
            Some(&"2".to_string())
        );
        assert_eq!(
            values.get("nemo_relay.start.metadata.tags"),
            Some(&"[\"a\",\"b\"]".to_string())
        );
        assert_eq!(
            values.get("nemo_relay.start.metadata.context"),
            Some(&"{\"region\":\"us-east-1\"}".to_string())
        );
        assert_eq!(
            values.get("nemo_relay.start.metadata.request"),
            Some(&"{\"id\":\"nested-id\"}".to_string())
        );
        assert_eq!(
            values.get("nemo_relay.start.metadata.request.id"),
            Some(&"flat-id".to_string())
        );
        assert_eq!(
            values.get("nemo_relay.start.metadata.event_id"),
            Some(&"18446744073709551615".to_string())
        );
        assert_eq!(values.get("tenant.id"), Some(&"acme".to_string()));
        assert!(!values.contains_key("nemo_relay.start.metadata_json"));
    }

    #[test]
    fn rejects_invalid_attribute_mappings() {
        assert!(
            super::validate_attribute_mappings(&[OtlpAttributeMapping::new("", "alias")]).is_err()
        );
        assert!(
            super::validate_attribute_mappings(&[
                OtlpAttributeMapping::new("one", "duplicate"),
                OtlpAttributeMapping::new("two", "duplicate"),
            ])
            .is_err()
        );
        assert!(
            super::validate_attribute_mappings(&[
                OtlpAttributeMapping::new("one", "duplicate"),
                OtlpAttributeMapping::new("two", " duplicate "),
            ])
            .is_err()
        );
    }
}
