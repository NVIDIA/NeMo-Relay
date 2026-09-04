// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structure-preserving removal of conversational trajectory content.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::Value as Json;

use nemo_relay::api::event::{
    CategoryProfile, Event, LOG_SEVERITY_METADATA_KEY, LogSeverity, METRIC_DATA_SCHEMA_NAME,
    METRIC_DATA_SCHEMA_VERSION, MetricEnvelope,
};
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::response::AnnotatedLlmResponse;

const TRUSTED_STRING_SCOPE_METADATA_FIELDS: &[&str] = &[
    "nemo_relay_scope_role",
    "agent_kind",
    "hook_event_name",
    "gateway_config_profile",
    "gateway_mode",
    "turn_source",
    "harness",
    "source",
    "identity_quality",
    "gateway_path",
    "llm_correlation_status",
    "llm_correlation_source",
    "tool_correlation_status",
    "tool_correlation_source",
    "otel.status_code",
    "fidelity_source",
];

const TRUSTED_BOOLEAN_SCOPE_METADATA_FIELDS: &[&str] = &["provider_payload_exact"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CustomMarkPayloadPolicy {
    Preserve,
    RedactAllLeaves,
}

impl CustomMarkPayloadPolicy {
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value {
            "preserve" => Some(Self::Preserve),
            "redact_all_leaves" => Some(Self::RedactAllLeaves),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub(super) struct TrajectorySanitizer {
    replacement: Arc<String>,
    custom_mark_payload_policy: CustomMarkPayloadPolicy,
    metric_string_attribute_allowlist: Arc<BTreeMap<String, BTreeSet<String>>>,
}

impl TrajectorySanitizer {
    pub(super) fn new(
        replacement: String,
        policy: CustomMarkPayloadPolicy,
        metric_string_attribute_allowlist: BTreeMap<String, Vec<String>>,
    ) -> Self {
        Self {
            replacement: Arc::new(replacement),
            custom_mark_payload_policy: policy,
            metric_string_attribute_allowlist: Arc::new(
                metric_string_attribute_allowlist
                    .into_iter()
                    .map(|(attribute, values)| (attribute, values.into_iter().collect()))
                    .collect(),
            ),
        }
    }

    pub(super) fn sanitize_tool_payload(&self, value: Json) -> Json {
        redact_all_leaves(value, &self.replacement)
    }

    pub(super) fn sanitize_provider_payload(&self, value: Json) -> Json {
        redact_semantic_content(value, &self.replacement, None)
    }

    pub(super) fn sanitize_annotated_request(
        &self,
        request: AnnotatedLlmRequest,
    ) -> Option<AnnotatedLlmRequest> {
        let mut value = match serde_json::to_value(request) {
            Ok(value) => value,
            Err(_) => return log_annotated_llm_payload_omitted("request", "serialization failure"),
        };
        let preserved = take_root_fields(
            &value,
            &[
                "model",
                "tool_choice",
                "store",
                "previous_response_id",
                "truncation",
                "include",
                "service_tier",
                "parallel_tool_calls",
                "max_output_tokens",
                "max_tool_calls",
                "top_logprobs",
                "stream",
            ],
        );
        value = redact_semantic_content(value, &self.replacement, None);
        restore_root_fields(&mut value, preserved);
        match serde_json::from_value(value) {
            Ok(request) => Some(request),
            Err(_) => log_annotated_llm_payload_omitted("request", "deserialization failure"),
        }
    }

    pub(super) fn sanitize_annotated_response(
        &self,
        response: AnnotatedLlmResponse,
    ) -> Option<AnnotatedLlmResponse> {
        let mut value = match serde_json::to_value(response) {
            Ok(value) => value,
            Err(_) => {
                return log_annotated_llm_payload_omitted("response", "serialization failure");
            }
        };
        let mut preserved = take_root_fields(
            &value,
            &[
                "id",
                "model",
                "finish_reason",
                "usage",
                "optimization_summary",
            ],
        );
        if let Some((_, summary)) = preserved
            .iter_mut()
            .find(|(field, _)| field == "optimization_summary")
        {
            sanitize_optimization_payloads(summary, &self.replacement);
        }
        value = redact_semantic_content(value, &self.replacement, None);
        restore_root_fields(&mut value, preserved);
        match serde_json::from_value(value) {
            Ok(response) => Some(response),
            Err(_) => log_annotated_llm_payload_omitted("response", "deserialization failure"),
        }
    }

    pub(super) fn sanitize_event_fields(
        &self,
        event: &Event,
        mut fields: nemo_relay::api::event::EventSanitizeFields,
    ) -> nemo_relay::api::event::EventSanitizeFields {
        let log_severity = valid_mark_log_severity(event, fields.metadata.as_ref());
        if is_relay_metric_mark(event) {
            fields.data = fields
                .data
                .and_then(|data| self.sanitize_metric_envelope(data));
            fields.metadata = fields
                .metadata
                .map(|value| redact_semantic_content(value, &self.replacement, None));
            fields.category_profile = fields
                .category_profile
                .and_then(|profile| sanitize_category_profile(profile, &self.replacement));
            return restore_log_severity(fields, log_severity);
        }

        let category = event.category().map(|category| category.as_str());
        let specialized_scope =
            matches!(event, Event::Scope(_)) && matches!(category, Some("llm" | "tool"));
        let unknown_custom_mark = matches!(event, Event::Mark(_))
            && category == Some("custom")
            && !is_known_content_bearing_mark(event.name());

        if unknown_custom_mark {
            if self.custom_mark_payload_policy == CustomMarkPayloadPolicy::RedactAllLeaves {
                fields.data = fields
                    .data
                    .map(|value| redact_all_leaves(value, &self.replacement));
                fields.metadata = fields
                    .metadata
                    .map(|value| redact_all_leaves(value, &self.replacement));
                fields.category_profile = fields
                    .category_profile
                    .and_then(|profile| redact_custom_category_profile(profile, self));
            }
            return restore_log_severity(fields, log_severity);
        }

        if !specialized_scope {
            fields.data = fields
                .data
                .map(|value| redact_semantic_content(value, &self.replacement, None));
        }
        fields.metadata = fields.metadata.map(|value| {
            if matches!(event, Event::Scope(_)) {
                sanitize_scope_metadata(value, &self.replacement)
            } else {
                redact_semantic_content(value, &self.replacement, None)
            }
        });
        fields.category_profile = fields
            .category_profile
            .and_then(|profile| sanitize_category_profile(profile, &self.replacement));
        restore_log_severity(fields, log_severity)
    }

    /// Redact optional metric text without modifying required export fields.
    fn sanitize_metric_envelope(&self, data: Json) -> Option<Json> {
        let mut envelope = match serde_json::from_value::<MetricEnvelope>(data) {
            Ok(envelope) => envelope,
            Err(_) => {
                return log_metric_envelope_omitted("metric envelope deserialization failure");
            }
        };
        if envelope.validate().is_err() {
            return log_metric_envelope_omitted(
                "metric envelope validation failure before redaction",
            );
        }
        for measurement in &mut envelope.measurements {
            measurement.description = measurement
                .description
                .take()
                .map(|_| (*self.replacement).clone());
            measurement.attributes = measurement.attributes.take().map(|attributes| {
                redact_metric_string_attributes(
                    attributes,
                    &self.replacement,
                    &self.metric_string_attribute_allowlist,
                )
            });
        }
        if envelope.validate().is_err() {
            return log_metric_envelope_omitted(
                "metric envelope validation failure after redaction",
            );
        }
        match serde_json::to_value(envelope) {
            Ok(data) => Some(data),
            Err(_) => log_metric_envelope_omitted("metric envelope serialization failure"),
        }
    }
}

fn log_annotated_llm_payload_omitted<T>(direction: &str, reason: &str) -> Option<T> {
    log::warn!(
        target: "nemo_relay.plugin",
        event = "pii_llm_payload_omitted",
        codec_kind = "annotated",
        reason;
        "PII redaction omitted an annotated LLM {direction} payload"
    );
    None
}

fn log_metric_envelope_omitted(reason: &str) -> Option<Json> {
    log::warn!(
        target: "nemo_relay.plugin",
        event = "pii_metric_envelope_omitted",
        reason;
        "PII redaction omitted a metric envelope"
    );
    None
}

fn valid_mark_log_severity(event: &Event, metadata: Option<&Json>) -> Option<LogSeverity> {
    if !matches!(event, Event::Mark(_)) {
        return None;
    }
    metadata
        .and_then(Json::as_object)
        .and_then(|metadata| metadata.get(LOG_SEVERITY_METADATA_KEY))
        .and_then(Json::as_str)
        .and_then(|value| value.parse::<LogSeverity>().ok())
}

fn restore_log_severity(
    mut fields: nemo_relay::api::event::EventSanitizeFields,
    severity: Option<LogSeverity>,
) -> nemo_relay::api::event::EventSanitizeFields {
    if let (Some(severity), Some(Json::Object(metadata))) = (severity, fields.metadata.as_mut()) {
        metadata.insert(
            LOG_SEVERITY_METADATA_KEY.to_string(),
            Json::String(severity.as_str().to_string()),
        );
    }
    fields
}

/// Return whether an event carries Relay's typed metric schema.
pub(crate) fn is_relay_metric_mark(event: &Event) -> bool {
    matches!(event, Event::Mark(_))
        && event.data_schema().is_some_and(|schema| {
            schema.name == METRIC_DATA_SCHEMA_NAME && schema.version == METRIC_DATA_SCHEMA_VERSION
        })
}

/// Redact strings in a typed metric attribute object.
fn redact_metric_string_attributes(
    value: Json,
    replacement: &str,
    allowlist: &BTreeMap<String, BTreeSet<String>>,
) -> Json {
    match value {
        Json::Object(values) => Json::Object(
            values
                .into_iter()
                .map(|(key, value)| {
                    let value = redact_metric_string_attribute(&key, value, replacement, allowlist);
                    (key, value)
                })
                .collect(),
        ),
        value => value,
    }
}

/// Redact an individual typed metric attribute when it contains text.
fn redact_metric_string_attribute(
    attribute: &str,
    value: Json,
    replacement: &str,
    allowlist: &BTreeMap<String, BTreeSet<String>>,
) -> Json {
    match value {
        Json::String(value) => Json::String(
            if metric_string_value_is_allowed(allowlist, attribute, &value) {
                value
            } else {
                replacement.to_string()
            },
        ),
        Json::Array(values) if values.iter().all(Json::is_string) => Json::Array(
            values
                .into_iter()
                .map(|value| {
                    let Json::String(value) = value else {
                        unreachable!("checked that every metric attribute value is a string");
                    };
                    Json::String(
                        if metric_string_value_is_allowed(allowlist, attribute, &value) {
                            value
                        } else {
                            replacement.to_string()
                        },
                    )
                })
                .collect(),
        ),
        value => value,
    }
}

fn metric_string_value_is_allowed(
    allowlist: &BTreeMap<String, BTreeSet<String>>,
    attribute: &str,
    value: &str,
) -> bool {
    allowlist
        .get(attribute)
        .is_some_and(|values| values.contains(value))
}

fn sanitize_scope_metadata(value: Json, replacement: &str) -> Json {
    let Json::Object(values) = value else {
        return redact_semantic_content(value, replacement, None);
    };
    Json::Object(
        values
            .into_iter()
            .map(|(key, value)| {
                let value = if is_trusted_scope_metadata_value(&key, &value) {
                    value
                } else {
                    redact_semantic_content(value, replacement, Some(&key))
                };
                (key, value)
            })
            .collect(),
    )
}

fn is_trusted_scope_metadata_value(key: &str, value: &Json) -> bool {
    match value {
        Json::String(_) => TRUSTED_STRING_SCOPE_METADATA_FIELDS.contains(&key),
        Json::Bool(_) => TRUSTED_BOOLEAN_SCOPE_METADATA_FIELDS.contains(&key),
        _ => false,
    }
}

fn is_known_content_bearing_mark(name: &str) -> bool {
    matches!(
        name,
        "llm.chunk" | "nemo_relay.llm.optimization" | "skill.load"
    )
}

fn sanitize_category_profile(
    mut profile: CategoryProfile,
    replacement: &str,
) -> Option<CategoryProfile> {
    profile.annotated_request = profile.annotated_request.as_ref().and_then(|request| {
        TrajectorySanitizer::new(
            replacement.to_string(),
            CustomMarkPayloadPolicy::Preserve,
            BTreeMap::new(),
        )
        .sanitize_annotated_request((**request).clone())
        .map(Arc::new)
    });
    profile.annotated_response = profile.annotated_response.as_ref().and_then(|response| {
        TrajectorySanitizer::new(
            replacement.to_string(),
            CustomMarkPayloadPolicy::Preserve,
            BTreeMap::new(),
        )
        .sanitize_annotated_response((**response).clone())
        .map(Arc::new)
    });
    profile.extra = profile
        .extra
        .into_iter()
        .map(|(key, value)| {
            let value = redact_semantic_content(value, replacement, Some(&key));
            (key, value)
        })
        .collect();
    Some(profile)
}

fn redact_custom_category_profile(
    mut profile: CategoryProfile,
    sanitizer: &TrajectorySanitizer,
) -> Option<CategoryProfile> {
    profile.annotated_request = profile.annotated_request.as_ref().and_then(|request| {
        sanitizer
            .sanitize_annotated_request((**request).clone())
            .map(Arc::new)
    });
    profile.annotated_response = profile.annotated_response.as_ref().and_then(|response| {
        sanitizer
            .sanitize_annotated_response((**response).clone())
            .map(Arc::new)
    });
    profile.extra = profile
        .extra
        .into_iter()
        .map(|(key, value)| {
            let value = redact_all_leaves(value, &sanitizer.replacement);
            (key, value)
        })
        .collect();
    Some(profile)
}

fn take_root_fields(value: &Json, fields: &[&str]) -> Vec<(String, Json)> {
    let Some(object) = value.as_object() else {
        return Vec::new();
    };
    fields
        .iter()
        .filter_map(|field| {
            object
                .get(*field)
                .cloned()
                .map(|value| ((*field).to_string(), value))
        })
        .collect()
}

fn restore_root_fields(value: &mut Json, fields: Vec<(String, Json)>) {
    let Some(object) = value.as_object_mut() else {
        return;
    };
    object.extend(fields);
}

fn sanitize_optimization_payloads(value: &mut Json, replacement: &str) {
    let Some(contributions) = value.get_mut("contributions").and_then(Json::as_array_mut) else {
        return;
    };
    for contribution in contributions {
        let Some(contribution) = contribution.as_object_mut() else {
            continue;
        };
        if let Some(payload) = contribution.get_mut("payload") {
            *payload = redact_all_leaves(payload.take(), replacement);
        }
        let known = [
            "id",
            "sequence",
            "producer",
            "kind",
            "applied",
            "model_transition",
            "token_impact",
            "payload_schema",
            "payload",
        ];
        for (key, value) in contribution.iter_mut() {
            if !known.contains(&key.as_str()) {
                *value = redact_all_leaves(value.take(), replacement);
            }
        }
    }
}

pub(super) fn redact_all_leaves(value: Json, replacement: &str) -> Json {
    match value {
        Json::Null => Json::Null,
        Json::Bool(_) => Json::Bool(false),
        Json::Number(_) => Json::from(0),
        Json::String(_) => Json::String(replacement.to_string()),
        Json::Array(values) => Json::Array(
            values
                .into_iter()
                .map(|value| redact_all_leaves(value, replacement))
                .collect(),
        ),
        Json::Object(values) => Json::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, redact_all_leaves(value, replacement)))
                .collect(),
        ),
    }
}

fn redact_semantic_content(value: Json, replacement: &str, field: Option<&str>) -> Json {
    match value {
        Json::Null => Json::Null,
        value @ (Json::Bool(_) | Json::Number(_) | Json::String(_)) => {
            redact_semantic_scalar(value, replacement, field)
        }
        Json::Array(values) => Json::Array(
            values
                .into_iter()
                .map(|value| redact_semantic_content(value, replacement, field))
                .collect(),
        ),
        Json::Object(values) => redact_semantic_object(values, replacement, field),
    }
}

fn redact_semantic_scalar(value: Json, replacement: &str, field: Option<&str>) -> Json {
    match value {
        Json::Bool(value) if field.is_some_and(preserve_analytical_bool) => Json::Bool(value),
        Json::Bool(_) => Json::Bool(false),
        Json::Number(value) if field.is_some_and(preserve_analytical_number) => Json::Number(value),
        Json::Number(_) => Json::from(0),
        Json::String(value) if field.is_some_and(preserve_analytical_string) => Json::String(value),
        Json::String(value) if field == Some("arguments") => {
            redact_stringified_json(value, replacement)
        }
        Json::String(_) => Json::String(replacement.to_string()),
        _ => unreachable!("only scalar JSON values are passed to redact_semantic_scalar"),
    }
}

fn redact_semantic_object(
    values: serde_json::Map<String, Json>,
    replacement: &str,
    field: Option<&str>,
) -> Json {
    let preserve_tool_name = preserves_tool_or_function_name(field, &values);
    Json::Object(
        values
            .into_iter()
            .map(|(key, value)| {
                let value = if key == "name" && preserve_tool_name && value.is_string() {
                    value
                } else {
                    redact_semantic_content(value, replacement, Some(&key))
                };
                (key, value)
            })
            .collect(),
    )
}

fn redact_stringified_json(value: String, replacement: &str) -> Json {
    let Ok(parsed) = serde_json::from_str::<Json>(&value) else {
        return Json::String(replacement.to_string());
    };
    let scrubbed = redact_all_leaves(parsed, replacement);
    Json::String(serde_json::to_string(&scrubbed).unwrap_or_else(|_| replacement.to_string()))
}

fn preserve_analytical_bool(key: &str) -> bool {
    matches!(
        key,
        "applied"
            | "enabled"
            | "store"
            | "stream"
            | "parallel_tool_calls"
            | "required"
            | "additionalProperties"
    )
}

fn preserve_analytical_number(key: &str) -> bool {
    matches!(
        key,
        "chunk_index"
            | "index"
            | "attempt"
            | "sequence"
            | "priority"
            | "version"
            | "temperature"
            | "top_p"
            | "top_logprobs"
            | "max_tokens"
            | "max_output_tokens"
            | "max_tool_calls"
            | "total"
            | "input"
            | "output"
            | "cache_read"
            | "cache_write"
            | "confidence"
            | "logprob"
    ) || key.ends_with("_tokens")
        || key.ends_with("_count")
        || key.ends_with("_index")
        || key.ends_with("_indices")
        || key.ends_with("_cost")
        || key.ends_with("_latency")
        || key.ends_with("_millis")
        || key.ends_with("_ms")
        || key.ends_with("_seconds")
        || key.ends_with("_timestamp")
}

fn preserve_analytical_string(key: &str) -> bool {
    if matches!(key, "token" | "token_id") {
        return false;
    }
    matches!(
        key,
        "role"
            | "type"
            | "api"
            | "kind"
            | "producer"
            | "subtype"
            | "model"
            | "model_name"
            | "provider"
            | "protocol"
            | "backend"
            | "tier"
            | "status"
            | "finish_reason"
            | "stop_reason"
            | "service_tier"
            | "mode"
            | "quality"
            | "estimation_method"
            | "currency"
            | "pricing_source"
            | "pricing_provider"
            | "pricing_model"
            | "pricing_as_of"
            | "required"
            | "version"
            | "event_type"
            | "object"
            | "object_type"
            | "system_fingerprint"
            | "truncation"
            | "include"
            | "detail"
            | "media_type"
            | "selected_model"
            | "selected_backend"
            | "selected_tier"
            | "selected_protocol"
            | "selected_route"
            | "selected_endpoint"
            | "baseline_model"
            | "baseline_backend"
            | "baseline_tier"
            | "baseline_protocol"
            | "baseline_route"
            | "effective_model"
            | "effective_backend"
            | "effective_tier"
            | "effective_protocol"
            | "effective_route"
    ) || key == "id"
        || key.ends_with("_id")
        || key.ends_with("_uuid")
}

fn preserves_tool_or_function_name(
    container: Option<&str>,
    object: &serde_json::Map<String, Json>,
) -> bool {
    matches!(container, Some("function" | "tools"))
        || object
            .get("type")
            .and_then(Json::as_str)
            .is_some_and(|kind| matches!(kind, "function" | "function_call" | "tool_use"))
}
