// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structure-preserving removal of conversational trajectory content.

use std::sync::Arc;

use serde_json::Value as Json;

use nemo_relay::api::event::{CategoryProfile, Event};
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::response::AnnotatedLlmResponse;

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
}

impl TrajectorySanitizer {
    pub(super) fn new(replacement: String, policy: CustomMarkPayloadPolicy) -> Self {
        Self {
            replacement: Arc::new(replacement),
            custom_mark_payload_policy: policy,
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
        let mut value = serde_json::to_value(request).ok()?;
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
        serde_json::from_value(value).ok()
    }

    pub(super) fn sanitize_annotated_response(
        &self,
        response: AnnotatedLlmResponse,
    ) -> Option<AnnotatedLlmResponse> {
        let mut value = serde_json::to_value(response).ok()?;
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
        serde_json::from_value(value).ok()
    }

    pub(super) fn sanitize_event_fields(
        &self,
        event: &Event,
        mut fields: nemo_relay::api::event::EventSanitizeFields,
    ) -> nemo_relay::api::event::EventSanitizeFields {
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
            return fields;
        }

        if !specialized_scope {
            fields.data = fields
                .data
                .map(|value| redact_semantic_content(value, &self.replacement, None));
        }
        fields.metadata = fields
            .metadata
            .map(|value| redact_semantic_content(value, &self.replacement, None));
        fields.category_profile = fields
            .category_profile
            .and_then(|profile| sanitize_category_profile(profile, &self.replacement));
        fields
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
        TrajectorySanitizer::new(replacement.to_string(), CustomMarkPayloadPolicy::Preserve)
            .sanitize_annotated_request((**request).clone())
            .map(Arc::new)
    });
    profile.annotated_response = profile.annotated_response.as_ref().and_then(|response| {
        TrajectorySanitizer::new(replacement.to_string(), CustomMarkPayloadPolicy::Preserve)
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
        Json::Bool(value) => {
            if field.is_some_and(preserve_analytical_bool) {
                Json::Bool(value)
            } else {
                Json::Bool(false)
            }
        }
        Json::Number(value) => {
            if field.is_some_and(preserve_analytical_number) {
                Json::Number(value)
            } else {
                Json::from(0)
            }
        }
        Json::String(value) => {
            if field.is_some_and(preserve_analytical_string) {
                Json::String(value)
            } else if field == Some("arguments") {
                redact_stringified_json(value, replacement)
            } else {
                Json::String(replacement.to_string())
            }
        }
        Json::Array(values) => Json::Array(
            values
                .into_iter()
                .map(|value| redact_semantic_content(value, replacement, field))
                .collect(),
        ),
        Json::Object(values) => {
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
    }
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
