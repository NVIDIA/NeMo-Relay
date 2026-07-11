// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use uuid::Uuid;

use crate::api::event::{Event, ScopeCategory};
use crate::api::llm::LlmRequest;
use crate::api::runtime::global_context;
use crate::api::runtime::{EventSubscriberFn, NemoRelayContextState, ScopeStackHandle};
use crate::api::runtime::{current_scope_stack, task_scope_top};
use crate::api::scope::ScopeHandle;
use crate::api::scope::ScopeType;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::traits::LlmCodec;
use crate::error::{FlowError, Result};
use crate::json::{Json, merge_json};
use crate::shared_runtime::ensure_process_runtime_owner;

/// Header carrying the current Dynamo agent session ID.
pub const DYNAMO_SESSION_ID_HEADER_KEY: &str = "x-dynamo-session-id";
/// Header carrying the parent Dynamo agent session ID.
pub const DYNAMO_PARENT_SESSION_ID_HEADER_KEY: &str = "x-dynamo-parent-session-id";

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

/// Apply the event sanitizer chain visible on the current scope stack.
pub(crate) fn sanitize_event(event: Event) -> Option<Event> {
    sanitize_event_with_scope_stack(event, &current_scope_stack())
}

/// Apply the event sanitizer chain visible on a captured scope stack.
pub(crate) fn sanitize_event_with_scope_stack(
    event: Event,
    scope_stack: &ScopeStackHandle,
) -> Option<Event> {
    let entries = {
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let context = global_context();
        let state = match context.read() {
            Ok(state) => state,
            Err(_) => return None,
        };
        match &event {
            Event::Mark(_) => {
                let locals = scope_guard.collect_scope_local_registries(|registries| {
                    &registries.mark_sanitize_guardrails
                });
                NemoRelayContextState::event_sanitize_entries(
                    &state.mark_sanitize_guardrails,
                    &locals,
                )
            }
            Event::Scope(scope) if scope.scope_category == ScopeCategory::Start => {
                let locals = scope_guard.collect_scope_local_registries(|registries| {
                    &registries.scope_sanitize_start_guardrails
                });
                NemoRelayContextState::event_sanitize_entries(
                    &state.scope_sanitize_start_guardrails,
                    &locals,
                )
            }
            Event::Scope(_) => {
                let locals = scope_guard.collect_scope_local_registries(|registries| {
                    &registries.scope_sanitize_end_guardrails
                });
                NemoRelayContextState::event_sanitize_entries(
                    &state.scope_sanitize_end_guardrails,
                    &locals,
                )
            }
        }
    };
    Some(NemoRelayContextState::event_sanitize_snapshot_chain(
        event, &entries,
    ))
}

pub(crate) fn ensure_runtime_owner() -> Result<()> {
    ensure_process_runtime_owner()
}

/// Resolve the current and parent agent session IDs from the active scope stack.
///
/// The most recent two explicit Agent scopes are used.
/// Harness-specific session metadata takes precedence over the scope name, while
/// names keep application-created scopes useful when no metadata is attached.
pub(crate) fn resolve_agent_session_ids() -> Option<(String, Option<String>)> {
    let stack = current_scope_stack();
    let stack = stack.read().ok()?;
    let mut agent_scopes = stack
        .scopes()
        .iter()
        .skip(1)
        .filter(|scope| matches!(scope.scope_type, ScopeType::Agent))
        .rev();
    let current = agent_scopes.next().map(agent_scope_id)?;
    let parent = agent_scopes.next().map(agent_scope_id);
    Some((current, parent))
}

fn agent_scope_id(scope: &ScopeHandle) -> String {
    scope
        .metadata
        .as_ref()
        .and_then(|metadata| {
            [
                "codex_subagent_session_id",
                "subagent_session_id",
                "subagent_id",
                "session_id",
            ]
            .into_iter()
            .find_map(|key| metadata.get(key).and_then(|value| value.as_str()))
        })
        .unwrap_or(&scope.name)
        .to_string()
}

pub(crate) fn inject_dynamo_session_ids(request: &mut LlmRequest) {
    let Some((current, parent)) = resolve_agent_session_ids() else {
        return;
    };

    request.headers.insert(
        DYNAMO_SESSION_ID_HEADER_KEY.to_string(),
        Json::String(current),
    );
    match parent {
        Some(parent) => {
            request.headers.insert(
                DYNAMO_PARENT_SESSION_ID_HEADER_KEY.to_string(),
                Json::String(parent),
            );
        }
        None => {
            request.headers.remove(DYNAMO_PARENT_SESSION_ID_HEADER_KEY);
        }
    }
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

pub(crate) type InterceptedLlmRequest = (
    LlmRequest,
    Option<Arc<AnnotatedLlmRequest>>,
    Vec<crate::api::event::PendingMarkSpec>,
    Vec<crate::codec::optimization::LlmOptimizationContribution>,
);

#[cfg(test)]
pub(crate) fn run_request_intercepts_with_codec(
    name: &str,
    request: LlmRequest,
    codec: Option<Arc<dyn LlmCodec>>,
) -> Result<InterceptedLlmRequest> {
    run_request_intercepts_with_codec_inner(name, request, codec, None)
}

/// Run request intercepts and record optimization contributions directly into
/// the managed call's bounded accumulator as each intercept completes.
pub(crate) fn run_request_intercepts_with_codec_and_recorder(
    name: &str,
    request: LlmRequest,
    codec: Option<Arc<dyn LlmCodec>>,
    recorder: &crate::api::optimization::LlmOptimizationRecorder,
) -> Result<InterceptedLlmRequest> {
    run_request_intercepts_with_codec_inner(name, request, codec, Some(recorder))
}

fn run_request_intercepts_with_codec_inner(
    name: &str,
    request: LlmRequest,
    codec: Option<Arc<dyn LlmCodec>>,
    recorder: Option<&crate::api::optimization::LlmOptimizationRecorder>,
) -> Result<InterceptedLlmRequest> {
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

    let outcome = crate::api::runtime::NemoRelayContextState::llm_request_intercepts_snapshot_chain_with_recorder(
            name,
            request,
            annotated,
            &entries,
            codec.is_some(),
            recorder,
        )?;
    let mut request = outcome.request;
    inject_dynamo_session_ids(&mut request);
    let pending_marks = outcome.pending_marks;
    let optimization_contributions = outcome.optimization_contributions;

    match (codec, outcome.annotated_request) {
        (Some(codec), Some(annotated)) => {
            let mut encoded = codec.encode(&annotated, &request)?;
            encoded.headers.extend(request.headers);
            Ok((
                encoded,
                Some(Arc::new(annotated)),
                pending_marks,
                optimization_contributions,
            ))
        }
        (_, annotated) => Ok((
            request,
            annotated.map(Arc::new),
            pending_marks,
            optimization_contributions,
        )),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/shared_tests.rs"]
mod tests;
