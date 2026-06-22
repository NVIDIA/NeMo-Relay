// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use uuid::Uuid;

use crate::api::llm::LlmRequest;
use crate::api::runtime::EventSubscriberFn;
use crate::api::runtime::global_context;
use crate::api::runtime::{current_scope_stack, task_scope_top};
use crate::api::scope::ScopeHandle;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::traits::LlmCodec;
use crate::error::{FlowError, Result};
use crate::json::{Json, merge_json};
use crate::shared_runtime::ensure_process_runtime_owner;

pub(crate) fn resolve_parent_uuid(parent: Option<&ScopeHandle>) -> Option<Uuid> {
    Some(
        parent
            .map(|handle| handle.uuid)
            .unwrap_or_else(|| task_scope_top().uuid),
    )
}

pub(crate) fn snapshot_event_subscribers(
    scope_local_subscribers: Vec<EventSubscriberFn>,
) -> Result<Vec<EventSubscriberFn>> {
    let context = global_context();
    let state = context
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    Ok(state.collect_event_subscribers(&scope_local_subscribers))
}

pub(crate) fn ensure_runtime_owner() -> Result<()> {
    ensure_process_runtime_owner()
}

pub(crate) fn metadata_with_otel_status(
    metadata: Option<Json>,
    status_code: &'static str,
    status_message: Option<String>,
) -> Option<Json> {
    let mut status = serde_json::Map::new();
    status.insert(
        "otel.status_code".to_string(),
        Json::String(status_code.to_string()),
    );

    // In the OTel spec, the status description should only be set if the status code is ERROR.
    // https://opentelemetry.io/docs/specs/otel/trace/api/#set-status
    if status_code == "ERROR"
        && let Some(status_message) = status_message
    {
        status.insert(
            "otel.status_description".to_string(),
            Json::String(status_message),
        );
    }
    let mut metadata = merge_json(metadata, Some(Json::Object(status)));

    // Explicitly remove any existing otel.status_description if the status code is not ERROR.
    if status_code != "ERROR"
        && let Some(Json::Object(metadata)) = metadata.as_mut()
    {
        metadata.remove("otel.status_description");
    }
    metadata
}

pub(crate) fn run_request_intercepts_with_codec(
    name: &str,
    request: LlmRequest,
    codec: Option<Arc<dyn LlmCodec>>,
) -> Result<(LlmRequest, Option<Arc<AnnotatedLlmRequest>>)> {
    let original = request.clone();
    let annotated = match &codec {
        Some(codec) => Some(codec.decode(&request)?),
        None => None,
    };

    let entries = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard
            .collect_scope_local_registries(|registries| &registries.llm_request_intercepts);

        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.llm_request_intercept_entries(&scope_locals)
    };

    let (intercepted_request, intercepted_annotated) =
        crate::api::runtime::NemoRelayContextState::llm_request_intercepts_snapshot_chain(
            name, request, annotated, &entries,
        )?;

    match (codec, intercepted_annotated) {
        (Some(codec), Some(annotated)) => {
            let mut encoded = codec.encode(&annotated, &original)?;
            encoded.headers = intercepted_request.headers;
            Ok((encoded, Some(Arc::new(annotated))))
        }
        _ => Ok((intercepted_request, None)),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/shared_tests.rs"]
mod tests;
