// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional observability integrations for NeMo Relay Core.

#[cfg(test)]
use std::sync::Mutex;

#[cfg(test)]
pub(crate) fn test_mutex() -> &'static Mutex<()> {
    crate::shared_runtime::runtime_owner_test_mutex()
}

pub mod atif;
pub mod atof;
#[cfg(any(feature = "otel", feature = "openinference"))]
pub(crate) mod manual;
#[cfg(feature = "openinference")]
pub mod openinference;
#[cfg(feature = "otel")]
pub mod otel;
pub mod plugin_component;

#[cfg(any(feature = "otel", feature = "openinference"))]
pub(crate) fn estimate_cost_for_response_or_requested_model(
    event: &crate::api::event::Event,
    response_model: Option<&str>,
    usage: &crate::codec::response::Usage,
) -> Option<crate::codec::response::CostEstimate> {
    // Prefer the provider-echoed model, but fall back to the requested model
    // when pricing does not recognize the echoed model alias.
    if let Some(model_name) = response_model
        && let Some(cost) = crate::codec::response::estimate_cost_for_provider(
            Some(event.name()),
            model_name,
            usage,
        )
    {
        return Some(cost);
    }

    let event_model = event.model_name()?;
    if response_model == Some(event_model) {
        return None;
    }
    crate::codec::response::estimate_cost_for_provider(Some(event.name()), event_model, usage)
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
