// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use uuid::Uuid;

use crate::api::event::{Event, EventSanitizeFields, ScopeCategory};
use crate::api::llm::LlmRequest;
use crate::api::registry::{EventMetadataInjector, Guardrail, RuntimeRegistrationKind};
use crate::api::runtime::global_context;
use crate::api::runtime::scope_stack::traceparent_for_llm;
use crate::api::runtime::{
    EventSanitizeFn, EventSubscriberFn, NemoRelayContextState, ScopeStackHandle,
};
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
/// Header carrying the W3C trace context for an outbound provider request.
pub const TRACEPARENT_HEADER_KEY: &str = "traceparent";

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
        .map_err(|error| FlowError::Internal(error.to_string()))?
        .registry_snapshot(&[RuntimeRegistrationKind::Subscriber]);
    Ok(state.collect_event_subscribers(&scope_local_subscribers))
}

/// Snapshot the event sanitizers visible to an event without invoking them.
///
/// Scope and mark emission use this to capture middleware ownership while the
/// scope is still active, then let the serial dispatcher sanitize and publish
/// the immutable event snapshot later. This keeps public scope APIs
/// synchronous while ensuring scope removal cannot affect queued work.
pub(crate) fn snapshot_event_sanitizers(
    event: &Event,
    scope_stack: &ScopeStackHandle,
) -> Option<Vec<Guardrail<EventSanitizeFn>>> {
    let locals = {
        let scope_guard = match scope_stack.read() {
            Ok(guard) => guard,
            Err(error) => {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "event_sanitizer_snapshot_failed";
                    "Event sanitizer snapshot failed; clearing observability fields: {error}"
                );
                return Some(vec![Guardrail::new(
                    "event-sanitizer-snapshot-failure",
                    i32::MIN,
                    Arc::new(|_, _| Box::pin(async { Ok(EventSanitizeFields::default()) })),
                )]);
            }
        };
        match &event {
            Event::Mark(_) => scope_guard
                .snapshot_scope_local_registries(|registries| &registries.mark_sanitize_guardrails),
            Event::Scope(scope) if scope.scope_category == ScopeCategory::Start => scope_guard
                .snapshot_scope_local_registries(|registries| {
                    &registries.scope_sanitize_start_guardrails
                }),
            Event::Scope(_) => scope_guard.snapshot_scope_local_registries(|registries| {
                &registries.scope_sanitize_end_guardrails
            }),
        }
    };
    let local_refs = locals.iter().collect::<Vec<_>>();
    let context = global_context();
    let registration_kind = match event {
        Event::Mark(_) => RuntimeRegistrationKind::MarkSanitizeGuardrail,
        Event::Scope(scope) if scope.scope_category == ScopeCategory::Start => {
            RuntimeRegistrationKind::ScopeSanitizeStartGuardrail
        }
        Event::Scope(_) => RuntimeRegistrationKind::ScopeSanitizeEndGuardrail,
    };
    let state = match context.read() {
        Ok(state) => state.registry_snapshot(&[registration_kind]),
        Err(error) => {
            log::error!(
                target: "nemo_relay.runtime",
                event = "event_sanitizer_snapshot_failed";
                "Event sanitizer snapshot failed; clearing observability fields: {error}"
            );
            return Some(vec![Guardrail::new(
                "event-sanitizer-snapshot-failure",
                i32::MIN,
                Arc::new(|_, _| Box::pin(async { Ok(EventSanitizeFields::default()) })),
            )]);
        }
    };
    let entries = match &event {
        Event::Mark(_) => NemoRelayContextState::event_sanitize_entries(
            &state.mark_sanitize_guardrails,
            &local_refs,
            RuntimeRegistrationKind::MarkSanitizeGuardrail,
        ),
        Event::Scope(scope) if scope.scope_category == ScopeCategory::Start => {
            NemoRelayContextState::event_sanitize_entries(
                &state.scope_sanitize_start_guardrails,
                &local_refs,
                RuntimeRegistrationKind::ScopeSanitizeStartGuardrail,
            )
        }
        Event::Scope(_) => NemoRelayContextState::event_sanitize_entries(
            &state.scope_sanitize_end_guardrails,
            &local_refs,
            RuntimeRegistrationKind::ScopeSanitizeEndGuardrail,
        ),
    };
    Some(entries)
}

/// Snapshot the Event metadata injectors visible to a queued Event.
pub(crate) fn snapshot_event_metadata_injectors(
    scope_stack: &ScopeStackHandle,
) -> Vec<EventMetadataInjector> {
    let locals = match scope_stack.read() {
        Ok(guard) => guard,
        Err(error) => {
            log::error!(
                target: "nemo_relay.runtime",
                event = "event_metadata_injector_snapshot_failed";
                "Event metadata injector snapshot failed; continuing without injection: {error}"
            );
            return Vec::new();
        }
    }
    .snapshot_scope_local_registries(|registries| &registries.event_metadata_injectors);
    let context = global_context();
    let state = match context.read() {
        Ok(state) => state.registry_snapshot(&[RuntimeRegistrationKind::EventMetadataInjector]),
        Err(error) => {
            log::error!(
                target: "nemo_relay.runtime",
                event = "event_metadata_injector_snapshot_failed";
                "Event metadata injector snapshot failed; continuing without injection: {error}"
            );
            return Vec::new();
        }
    };
    let local_refs = locals.iter().collect::<Vec<_>>();
    NemoRelayContextState::event_metadata_injector_entries(
        &state.event_metadata_injectors,
        &local_refs,
    )
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

pub(crate) fn inject_traceparent_value(request: &mut LlmRequest, value: String) {
    request
        .headers
        .retain(|key, _| !key.eq_ignore_ascii_case(TRACEPARENT_HEADER_KEY));
    request
        .headers
        .insert(TRACEPARENT_HEADER_KEY.to_string(), Json::String(value));
}

pub(crate) fn inject_traceparent(request: &mut LlmRequest, parent_uuid: Uuid) -> Result<()> {
    let value = traceparent_for_llm(parent_uuid)?;
    inject_traceparent_value(request, value);
    Ok(())
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

pub(crate) fn metadata_with_otel_error(metadata: Option<Json>, error: &FlowError) -> Option<Json> {
    let mut metadata = metadata_with_otel_status(metadata, "ERROR", Some(error.to_string()));
    if let Some(Json::Object(metadata)) = metadata.as_mut() {
        metadata
            .entry("error.type".to_string())
            .or_insert_with(|| Json::String(error.otel_error_type().to_string()));
        if let Some(exception_type) = error.exception_type() {
            metadata
                .entry("exception.type".to_string())
                .or_insert_with(|| Json::String(exception_type.to_string()));
        }
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
pub(crate) async fn run_request_intercepts_with_codec(
    name: &str,
    request: LlmRequest,
    codec: Option<Arc<dyn LlmCodec>>,
) -> Result<InterceptedLlmRequest> {
    run_request_intercepts_with_codec_inner(name, request, codec, None).await
}

/// Run request intercepts and record optimization contributions directly into
/// the managed call's bounded accumulator as each intercept completes.
pub(crate) async fn run_request_intercepts_with_codec_and_recorder(
    name: &str,
    request: LlmRequest,
    codec: Option<Arc<dyn LlmCodec>>,
    recorder: &crate::api::optimization::LlmOptimizationRecorder,
) -> Result<InterceptedLlmRequest> {
    run_request_intercepts_with_codec_inner(name, request, codec, Some(recorder)).await
}

async fn run_request_intercepts_with_codec_inner(
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
        let scope_locals = scope_stack
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?
            .snapshot_scope_local_registries(|registries| &registries.llm_request_intercepts);
        let scope_local_refs = scope_locals.iter().collect::<Vec<_>>();

        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?
            .registry_snapshot(&[RuntimeRegistrationKind::LlmRequestIntercept]);
        state.llm_request_intercept_entries(&scope_local_refs)
    };

    let outcome = crate::api::runtime::NemoRelayContextState::llm_request_intercepts_snapshot_chain_with_recorder(
            name,
            request,
            annotated,
            &entries,
            codec.is_some(),
            recorder,
        )
        .await?;
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
