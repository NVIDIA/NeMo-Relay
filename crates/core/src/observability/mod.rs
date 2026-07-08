// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional observability integrations for NeMo Relay Core.

use crate::api::event::EventNormalizationExt;
use serde::{Deserialize, Serialize};

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
