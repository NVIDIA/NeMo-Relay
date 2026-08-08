// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::sync::Arc;

use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use typed_builder::TypedBuilder;
use uuid::Uuid;

use crate::api::event::{
    BaseEvent, CategoryProfile, DataSchema, Event, EventCategory, MarkEvent, PendingMarkSpec,
};
use crate::api::optimization::{
    LlmOptimizationRecorder, finalize_optimization_summary, scope_llm_optimization_recorder,
};
#[cfg(test)]
use crate::api::runtime::LlmCodecIdentity;
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::global_context;
use crate::api::runtime::state::contextualize_stream;
use crate::api::runtime::subscriber_dispatcher::{
    PendingPublication, dispatch_reserved_sanitized_event, dispatch_sanitized_event,
    dispatch_transformed_event, register_pending_publication,
};
use crate::api::runtime::{
    EventSubscriberFn, LlmCollectorFn, LlmExecutionNextFn, LlmFinalizerFn, LlmJsonStream,
    LlmSanitizeRequestContext, LlmSanitizeResponseContext, LlmStreamExecutionNextFn,
    MiddlewareContinuationContext, with_active_event_uuid,
};
use crate::api::runtime::{ScopeStackHandle, capture_traceparent, current_scope_stack};
use crate::api::scope::event;
use crate::api::scope::{EmitMarkEventParams, ScopeHandle};
use crate::api::shared::{
    ensure_runtime_owner, inject_dynamo_session_ids, inject_traceparent, inject_traceparent_value,
    metadata_with_otel_error, metadata_with_otel_status, resolve_parent_uuid,
    run_request_intercepts_with_codec_and_recorder, snapshot_event_sanitizers,
    snapshot_event_subscribers,
};
use crate::codec::request::{AnnotatedLlmRequest, Message};
use crate::codec::response::{AnnotatedLlmResponse, attach_estimated_cost_for_provider};
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::error::{FlowError, Result};
use crate::json::Json;
use crate::stream::LlmStreamWrapper;

pub use nemo_relay_types::api::llm::{
    LLM_REQUEST_INTERCEPT_OUTCOME_SCHEMA, LlmAttributes, LlmRequest, LlmRequestInterceptOutcome,
};

const OBSERVABILITY_CREDENTIAL_HEADERS: [&str; 7] = [
    "authorization",
    "proxy-authorization",
    "cookie",
    "x-api-key",
    "api-key",
    "anthropic-api-key",
    "x-goog-api-key",
];

fn queue_sanitized_event_with_scope_stack(
    event: Event,
    subscribers: &[EventSubscriberFn],
    scope_stack: &ScopeStackHandle,
) -> bool {
    let sanitizers = snapshot_event_sanitizers(&event, scope_stack).unwrap_or_default();
    dispatch_sanitized_event(event, sanitizers, subscribers, scope_stack.clone())
}

#[derive(Clone)]
struct CapturedLlmScopeStack(ScopeStackHandle);

impl Default for CapturedLlmScopeStack {
    fn default() -> Self {
        Self(current_scope_stack())
    }
}

impl std::fmt::Debug for CapturedLlmScopeStack {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("CapturedLlmScopeStack(..)")
    }
}

/// Runtime-owned handle identifying an active or completed LLM call.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct LlmHandle {
    /// Unique LLM-call identifier.
    #[builder(default = Uuid::now_v7())]
    pub uuid: Uuid,
    /// Timestamp captured when the LLM handle was created.
    #[builder(default = Utc::now())]
    pub started_at: DateTime<Utc>,
    /// Provider or logical call name recorded on lifecycle events.
    ///
    /// Gateway-managed provider calls use provider route names such as
    /// `anthropic.messages`; event normalization may reuse those route names as
    /// codec hints when raw request shapes overlap across providers.
    #[builder(setter(into))]
    pub name: String,
    /// Optional application payload stored on the handle.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional metadata attached to the LLM span.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// LLM behavior flags.
    #[builder(default = LlmAttributes::empty())]
    pub attributes: LlmAttributes,
    /// UUID of the parent scope, if any.
    #[builder(default)]
    pub parent_uuid: Option<Uuid>,
    /// Optional normalized model name for observability.
    #[builder(default, setter(into))]
    pub model_name: Option<String>,
    /// Bounded, in-memory optimization evidence recorder for this call.
    #[serde(skip, default)]
    #[builder(default)]
    pub optimization_recorder: LlmOptimizationRecorder,
    /// Scope stack captured when the LLM lifecycle starts.
    ///
    /// Close-time work can run from a different task, especially for streams,
    /// so optimization marks must not consult the poller's ambient scope.
    #[serde(skip, default)]
    #[builder(setter(skip), default)]
    captured_scope_stack: CapturedLlmScopeStack,
}

impl LlmHandle {
    pub(crate) fn captured_scope_stack(&self) -> &ScopeStackHandle {
        &self.captured_scope_stack.0
    }
}

/// Builder parameters for [`NemoRelayContextState::create_llm_handle`].
#[derive(Debug, Clone, TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct CreateLlmHandleParams<'a> {
    /// Logical provider or model family name. Gateway-managed provider calls
    /// should pass the provider route name, for example `anthropic.messages`.
    pub name: &'a str,
    /// Optional UUID reserved before request interception so outbound
    /// propagation can identify the emitted LLM span.
    #[builder(default)]
    pub uuid: Option<Uuid>,
    /// Optional parent scope UUID.
    #[builder(default)]
    pub parent_uuid: Option<uuid::Uuid>,
    /// LLM attribute bitflags.
    #[builder(default = LlmAttributes::empty())]
    pub attributes: LlmAttributes,
    /// Optional application payload stored on the handle.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional metadata stored on the handle.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional normalized model name stored on the handle.
    #[builder(default, setter(into))]
    pub model_name: Option<String>,
    /// Optional timestamp captured as the handle start time and reused by the
    /// emitted start event. When omitted, the current UTC time is used.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Builder parameters for [`NemoRelayContextState::build_llm_end_event`].
#[derive(Clone, TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct EndLlmHandleParams<'a> {
    /// LLM handle to serialize into the emitted end event.
    pub handle: &'a LlmHandle,
    /// Optional data payload merged over the handle data.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional metadata payload merged over the handle metadata.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional normalized response annotation produced by a response codec.
    #[builder(default)]
    pub annotated_response: Option<Arc<AnnotatedLlmResponse>>,
    /// Optional timestamp recorded on the emitted end event. When omitted, the
    /// runtime records the current UTC time, or one microsecond after the
    /// handle start time if the current time is not later.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Builder parameters for [`llm_call`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct LlmCallParams<'a> {
    /// Logical provider or model family name recorded on the span.
    pub name: &'a str,
    /// Raw request associated with the span.
    pub request: &'a LlmRequest,
    /// Optional explicit parent scope.
    #[builder(default)]
    pub parent: Option<&'a ScopeHandle>,
    /// LLM attribute bitflags applied to the span.
    #[builder(default = LlmAttributes::empty())]
    pub attributes: LlmAttributes,
    /// Optional application payload stored on the handle but not emitted as
    /// Agent Trajectory Observability Format (ATOF) data.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional JSON metadata recorded on the start event.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional normalized model name recorded separately from the request payload.
    #[builder(default, setter(into))]
    pub model_name: Option<String>,
    /// Optional normalized request annotation produced by a codec.
    #[builder(default)]
    pub annotated_request: Option<Arc<AnnotatedLlmRequest>>,
    /// Optional timestamp captured as the handle start time and reused by the
    /// emitted start event. When omitted, the current UTC time is used.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Builder parameters for [`llm_call_execute`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct LlmCallExecuteParams {
    /// Logical provider or model family name recorded on emitted events.
    #[builder(setter(into))]
    pub name: String,
    /// Raw request passed into the managed pipeline.
    pub request: LlmRequest,
    /// Provider callback or execution continuation.
    pub func: LlmExecutionNextFn,
    /// Optional explicit parent scope for the emitted LLM span.
    #[builder(default)]
    pub parent: Option<ScopeHandle>,
    /// LLM attribute bitflags applied to the managed span.
    #[builder(default = LlmAttributes::empty())]
    pub attributes: LlmAttributes,
    /// Optional application payload stored on the handle but not emitted as
    /// Agent Trajectory Observability Format (ATOF) data.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional JSON metadata recorded on emitted events.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional normalized model name for observability output.
    #[builder(default, setter(into))]
    pub model_name: Option<String>,
    /// Optional request codec used to produce annotated request data.
    #[builder(default)]
    pub codec: Option<Arc<dyn LlmCodec>>,
    /// Optional response codec used to attach annotated response data.
    #[builder(default)]
    pub response_codec: Option<Arc<dyn LlmResponseCodec>>,
}

/// Builder parameters for [`llm_stream_call_execute`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct LlmStreamCallExecuteParams {
    /// Logical provider or model family name recorded on emitted events.
    #[builder(setter(into))]
    pub name: String,
    /// Raw request passed into the managed pipeline.
    pub request: LlmRequest,
    /// Streaming provider callback or execution continuation.
    pub func: LlmStreamExecutionNextFn,
    /// Per-chunk collector callback used to accumulate stream state.
    pub collector: LlmCollectorFn,
    /// Finalizer callback used to construct the completed response.
    pub finalizer: LlmFinalizerFn,
    /// Optional explicit parent scope for the emitted LLM span.
    #[builder(default)]
    pub parent: Option<ScopeHandle>,
    /// LLM attribute bitflags applied to the managed span.
    #[builder(default = LlmAttributes::empty())]
    pub attributes: LlmAttributes,
    /// Optional application payload stored on the handle but not emitted as
    /// Agent Trajectory Observability Format (ATOF) data.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional JSON metadata recorded on emitted events.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional normalized model name for observability output.
    #[builder(default, setter(into))]
    pub model_name: Option<String>,
    /// Optional request codec used to produce annotated request data.
    #[builder(default)]
    pub codec: Option<Arc<dyn LlmCodec>>,
    /// Optional response codec used to attach annotated response data.
    #[builder(default)]
    pub response_codec: Option<Arc<dyn LlmResponseCodec>>,
}

/// Builder parameters for [`llm_call_end`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct LlmCallEndParams<'a> {
    /// LLM handle to close.
    pub handle: &'a LlmHandle,
    /// Raw provider response associated with the end event.
    pub response: Json,
    /// Optional application payload retained for compatibility; Agent
    /// Trajectory Observability Format (ATOF) data is the response.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional JSON metadata recorded on the end event.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional normalized response annotation produced by a response codec.
    #[builder(default)]
    pub annotated_response: Option<Arc<AnnotatedLlmResponse>>,
    /// Optional response codec used to produce an annotation from sanitized event data.
    #[builder(default)]
    pub response_codec: Option<Arc<dyn LlmResponseCodec>>,
    /// Optional timestamp recorded on the emitted end event. When omitted, the
    /// runtime records the current UTC time, or one microsecond after the
    /// handle start time if the current time is not later.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

fn create_llm_handle(params: CreateLlmHandleParams<'_>) -> Result<LlmHandle> {
    ensure_runtime_owner()?;
    let context = global_context();
    let state = context
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    Ok(state.create_llm_handle(params))
}

fn request_turn_projection_needed<T>(
    items: &[T],
    is_user: &impl Fn(&T) -> bool,
    is_instruction: &impl Fn(&T) -> bool,
) -> bool {
    let Some(last_index) = items.len().checked_sub(1) else {
        return false;
    };
    match items.iter().rposition(is_user) {
        Some(start) => items[..start].iter().any(|item| !is_instruction(item)),
        None => items
            .iter()
            .enumerate()
            .any(|(index, item)| index != last_index && !is_instruction(item)),
    }
}

fn retain_current_request_turn<T>(
    items: &mut Vec<T>,
    is_user: impl Fn(&T) -> bool,
    is_instruction: impl Fn(&T) -> bool,
) -> bool {
    if !request_turn_projection_needed(items, &is_user, &is_instruction) {
        return false;
    }
    let last_index = items.len() - 1;
    let Some(start) = items.iter().rposition(is_user) else {
        let mut index = 0;
        items.retain(|item| {
            let retain = index == last_index || is_instruction(item);
            index += 1;
            retain
        });
        return true;
    };
    let mut current_turn = items.split_off(start);
    items.retain(is_instruction);
    items.append(&mut current_turn);
    true
}

fn project_llm_request_to_current_user_turn(
    request: &mut LlmRequest,
    annotated_request: &mut Option<Arc<AnnotatedLlmRequest>>,
    request_codec: Option<&dyn LlmCodec>,
) {
    let Some(annotation) = annotated_request.as_mut() else {
        return;
    };
    if !request_turn_projection_needed(
        &annotation.messages,
        &|message| matches!(message, Message::User { .. }),
        &|message| matches!(message, Message::System { .. }),
    ) {
        return;
    }
    let original_annotation = request_codec.map(|_| Arc::clone(annotation));
    let projected = limit_annotated_request_history_to_current_user_turn(Arc::make_mut(annotation));
    debug_assert!(projected);
    if let Some(codec) = request_codec {
        match codec.encode(annotation, request) {
            Ok(encoded) => *request = encoded,
            Err(_) => {
                log::warn!(
                    target: "nemo_relay.observability",
                    event = "projection_failed",
                    projection = "llm_current_turn",
                    recovery = "preserve_full_history";
                    "LLM request projection failed; preserving full event history"
                );
                *annotation = original_annotation
                    .expect("codec-backed projection should preserve the original annotation")
            }
        }
    }
}

fn limit_annotated_request_history_to_current_user_turn(
    annotated_request: &mut AnnotatedLlmRequest,
) -> bool {
    retain_current_request_turn(
        &mut annotated_request.messages,
        |message| matches!(message, Message::User { .. }),
        |message| matches!(message, Message::System { .. }),
    )
}

async fn emit_llm_start_with_subscribers(
    handle: &LlmHandle,
    request: &LlmRequest,
    annotated_request: Option<Arc<AnnotatedLlmRequest>>,
    request_codec: Option<Arc<dyn LlmCodec>>,
    subscribers: &[EventSubscriberFn],
) -> Result<()> {
    ensure_runtime_owner()?;
    let (entries, full_payloads_enabled) = {
        let scope_stack = handle.captured_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.llm_sanitize_request_guardrails
        });
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        (
            state.llm_sanitize_request_entries(&scope_locals),
            state.observability_full_payloads_enabled,
        )
    };
    let observable_request = remove_observability_credential_headers(request.clone());
    let mut sanitized_request = NemoRelayContextState::llm_sanitize_request_snapshot_chain(
        observable_request.clone(),
        LlmSanitizeRequestContext::for_request_codec(request_codec.clone()),
        &entries,
    )
    .await;
    let request_changed = sanitized_request
        .as_ref()
        .is_some_and(|sanitized_request| sanitized_request != &observable_request);
    let mut annotated_request = match (sanitized_request.as_ref(), request_codec.as_deref()) {
        (Some(sanitized_request), Some(codec)) if request_changed => {
            codec.decode(sanitized_request).ok().map(Arc::new)
        }
        (Some(_), _) if !request_changed => annotated_request,
        (None, _) => None,
        (Some(_), _) => None,
    };
    let scope_stack = handle.captured_scope_stack();
    let agent_is_fresh = {
        let mut scope_guard = scope_stack.write().expect("scope stack lock poisoned");
        scope_guard.take_agent_freshness(handle.parent_uuid)
    };
    if !full_payloads_enabled
        && !agent_is_fresh
        && let Some(sanitized_request) = sanitized_request.as_mut()
    {
        project_llm_request_to_current_user_turn(
            sanitized_request,
            &mut annotated_request,
            request_codec.as_deref(),
        );
    }
    let input = sanitized_request
        .as_ref()
        .and_then(|sanitized_request| serde_json::to_value(sanitized_request).ok());
    let event = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.build_llm_start_event(handle, input, annotated_request)
    };
    queue_sanitized_event_with_scope_stack(event, subscribers, scope_stack);
    Ok(())
}

fn remove_observability_credential_headers(mut request: LlmRequest) -> LlmRequest {
    request.headers.retain(|name, _| {
        !OBSERVABILITY_CREDENTIAL_HEADERS
            .iter()
            .any(|credential_header| name.eq_ignore_ascii_case(credential_header))
    });
    request
}

/// Synchronous test seam retained for lifecycle unit tests. Public manual
/// lifecycle emission is synchronous too, but its work is queued; this helper
/// exercises the managed start-event transformation directly.
#[cfg(test)]
fn emit_llm_start(
    handle: &LlmHandle,
    request: &LlmRequest,
    annotated_request: Option<Arc<AnnotatedLlmRequest>>,
    request_codec: Option<Arc<dyn LlmCodec>>,
) -> Result<()> {
    let subscribers = {
        let scope_stack = handle.captured_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        snapshot_event_subscribers(scope_guard.collect_scope_local_subscribers())?
    };
    crate::api::runtime::subscriber_dispatcher::block_on_sanitizer_future(
        emit_llm_start_with_subscribers(
            handle,
            request,
            annotated_request,
            request_codec,
            &subscribers,
        ),
    )
    .map_err(FlowError::Internal)?
}

async fn emit_pending_request_marks(
    handle: &LlmHandle,
    marks: Vec<PendingMarkSpec>,
    subscribers: &[EventSubscriberFn],
) -> Result<()> {
    if marks.is_empty() {
        return Ok(());
    }
    ensure_runtime_owner()?;
    let timestamp = handle.started_at + TimeDelta::microseconds(1);
    for mark in marks {
        let event = Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .name(mark.name)
                .parent_uuid(handle.uuid)
                .timestamp(timestamp)
                .data_opt(mark.data)
                .metadata_opt(mark.metadata)
                .build(),
            mark.category,
            mark.category_profile,
        ));
        queue_sanitized_event_with_scope_stack(event, subscribers, handle.captured_scope_stack());
    }
    Ok(())
}

pub(crate) async fn emit_optimization_marks(handle: &LlmHandle, subscribers: &[EventSubscriberFn]) {
    emit_optimization_marks_with_async(
        handle,
        subscribers,
        |event| async { Some(event) },
        |event, subscribers| {
            queue_sanitized_event_with_scope_stack(
                event.clone(),
                subscribers,
                handle.captured_scope_stack(),
            )
        },
    )
    .await;
}

pub(crate) async fn emit_reserved_optimization_marks(
    handle: &LlmHandle,
    subscribers: &[EventSubscriberFn],
) {
    emit_optimization_marks_with_async(
        handle,
        subscribers,
        |event| async { Some(event) },
        |event, subscribers| {
            let sanitizers =
                snapshot_event_sanitizers(event, handle.captured_scope_stack()).unwrap_or_default();
            dispatch_reserved_sanitized_event(
                event.clone(),
                sanitizers,
                subscribers,
                handle.captured_scope_stack().clone(),
            )
        },
    )
    .await;
}

/// Queue optimization marks from a synchronous lifecycle API.
///
/// The public manual lifecycle APIs must not await middleware. Capture each
/// event's sanitizer chain now and enqueue the immutable snapshots ahead of
/// the corresponding end event, preserving publication order.
fn enqueue_optimization_marks(handle: &LlmHandle, subscribers: &[EventSubscriberFn]) {
    let contributions = handle.optimization_recorder.unemitted_with_timestamps();
    if contributions.is_empty() || ensure_runtime_owner().is_err() {
        return;
    }
    let scope_stack = handle.captured_scope_stack().clone();
    for (contribution, recorded_at) in contributions {
        let event = optimization_mark_event(handle, &contribution, recorded_at);
        let sanitizers = snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
        if dispatch_sanitized_event(event, sanitizers, subscribers, scope_stack.clone()) {
            handle.optimization_recorder.mark_emitted(1);
        } else {
            break;
        }
    }
}

async fn emit_optimization_marks_with_async<F, Fut>(
    handle: &LlmHandle,
    subscribers: &[EventSubscriberFn],
    mut sanitize: F,
    mut enqueue: impl FnMut(&Event, &[EventSubscriberFn]) -> bool,
) where
    F: FnMut(Event) -> Fut,
    Fut: Future<Output = Option<Event>>,
{
    let contributions = handle.optimization_recorder.unemitted_with_timestamps();
    if contributions.is_empty() {
        return;
    }
    if ensure_runtime_owner().is_err() {
        log::warn!(
            target: "nemo_relay.observability",
            event = "optimization_marks_skipped",
            reason = "runtime_owner_unavailable",
            contribution_count = contributions.len();
            "LLM optimization marks were skipped"
        );
        return;
    }
    for (contribution, recorded_at) in contributions {
        let event = optimization_mark_event(handle, &contribution, recorded_at);
        let Some(event) = sanitize(event).await else {
            // Sanitizers currently rewrite fields rather than intentionally
            // dropping events. `None` means the sanitizer context was
            // unavailable, so preserve this ordered suffix for a later retry.
            break;
        };
        if enqueue(&event, subscribers) {
            handle.optimization_recorder.mark_emitted(1);
        } else {
            // Preserve this item and the remaining ordered suffix for a later
            // lifecycle boundary. Accounting remains best effort and must not
            // alter the provider result.
            break;
        }
    }
}

fn optimization_mark_event(
    handle: &LlmHandle,
    contribution: &crate::codec::optimization::LlmOptimizationContribution,
    recorded_at: DateTime<Utc>,
) -> Event {
    let offset = contribution.sequence.unwrap_or(0).saturating_add(2);
    let offset = i64::try_from(offset).unwrap_or(i64::MAX);
    let request_ordered_timestamp = handle.started_at + TimeDelta::microseconds(offset);
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("nemo_relay.llm.optimization")
            .parent_uuid(handle.uuid)
            .timestamp(recorded_at.max(request_ordered_timestamp))
            .data(serde_json::to_value(contribution).unwrap_or(Json::Null))
            .data_schema(DataSchema {
                name: "nemo.relay.llm_optimization_contribution".to_string(),
                version: "1".to_string(),
            })
            .build(),
        Some(EventCategory::custom()),
        Some(
            CategoryProfile::builder()
                .subtype("nemo_relay.llm.optimization")
                .build(),
        ),
    ))
}

/// Synchronous test seam for optimization-mark accounting. Production paths
/// always use [`emit_optimization_marks_with_async`]; unit tests use this seam
/// to isolate cursor behavior from asynchronous event publication.
#[cfg(test)]
fn emit_optimization_marks_with<F>(
    handle: &LlmHandle,
    subscribers: &[EventSubscriberFn],
    mut sanitize: F,
    mut enqueue: impl FnMut(&Event, &[EventSubscriberFn]) -> bool,
) where
    F: FnMut(Event) -> Option<Event>,
{
    let contributions = handle.optimization_recorder.unemitted_with_timestamps();
    if contributions.is_empty() || ensure_runtime_owner().is_err() {
        return;
    }
    for (contribution, recorded_at) in contributions {
        let event = optimization_mark_event(handle, &contribution, recorded_at);
        let Some(event) = sanitize(event) else {
            break;
        };
        if enqueue(&event, subscribers) {
            handle.optimization_recorder.mark_emitted(1);
        } else {
            break;
        }
    }
}

/// Start a manual LLM lifecycle span.
///
/// This emits an LLM-start event after applying sanitize-request guardrails to
/// the payload recorded for observability.
///
/// If a sanitizer errors or panics, Relay omits the payload and request
/// annotation and does not run remaining sanitizers.
///
/// # Parameters
/// - `name`: Logical provider or model family name recorded on the span.
/// - `request`: Raw [`LlmRequest`] associated with the span.
/// - `parent`: Optional explicit parent scope.
/// - `attributes`: LLM attribute bitflags applied to the span.
/// - `data`: Optional application payload stored on the returned handle. The
///   emitted start event data is the sanitized `request` payload.
/// - `metadata`: Optional JSON metadata recorded on the start event.
/// - `model_name`: Optional normalized model name recorded separately from the
///   request payload.
/// - `annotated_request`: Optional normalized request annotation produced by a
///   codec.
/// - `timestamp`: Optional timestamp recorded as the handle start time and on
///   the emitted start event. When `None`, the current UTC time is used.
///
/// # Returns
/// A [`Result`] containing the created [`LlmHandle`] after its start-event
/// snapshot has been submitted for queued publication.
///
/// # Errors
/// Returns an error when the runtime owner check fails or when internal state
/// cannot be read safely. Dispatcher submission failures are logged because
/// observability publication is best effort.
///
/// # Notes
/// The runtime removes standard credential headers (`authorization`,
/// `proxy-authorization`, `cookie`, `x-api-key`, `api-key`,
/// `anthropic-api-key`, and `x-goog-api-key`) from the event-only request copy
/// before sanitize-request guardrails run. This does not change the
/// caller-owned [`LlmRequest`]. By default, when the owning agent is not fresh,
/// the emitted request annotation is limited to the current user turn. Managed
/// calls with a request codec also apply that projection to the event input,
/// without changing the request used for provider execution. The observability
/// plugin's `enable_full_payloads` option disables this projection.
pub fn llm_call(params: LlmCallParams<'_>) -> Result<LlmHandle> {
    let handle_params = CreateLlmHandleParams::builder()
        .name(params.name)
        .parent_uuid_opt(resolve_parent_uuid(params.parent))
        .attributes(params.attributes)
        .data_opt(params.data)
        .metadata_opt(params.metadata)
        .model_name_opt(params.model_name)
        .timestamp_opt(params.timestamp)
        .build();
    let handle = create_llm_handle(handle_params)?;
    let scope_stack = handle.captured_scope_stack().clone();
    let (entries, subscribers, agent_is_fresh, full_payloads_enabled) = {
        let mut scope_guard = scope_stack
            .write()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.llm_sanitize_request_guardrails
        });
        let subscribers =
            snapshot_event_subscribers(scope_guard.collect_scope_local_subscribers())?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let entries = state.llm_sanitize_request_entries(&scope_locals);
        let full_payloads_enabled = state.observability_full_payloads_enabled;
        drop(state);
        let agent_is_fresh = scope_guard.take_agent_freshness(handle.parent_uuid);
        (entries, subscribers, agent_is_fresh, full_payloads_enabled)
    };
    // Middleware and event publication only observe a credential-free copy.
    // Keep `params.request` untouched: it remains the caller/provider request.
    let request = remove_observability_credential_headers(params.request.clone());
    let annotated_request = params.annotated_request;
    let event = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.build_llm_start_event(&handle, None, None)
    };
    let queued_handle = handle.clone();
    let event_sanitizers = snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
    dispatch_transformed_event(
        event,
        Box::new(move |event| {
            Box::pin(async move {
                let mut sanitized_request =
                    NemoRelayContextState::llm_sanitize_request_snapshot_chain(
                        request.clone(),
                        LlmSanitizeRequestContext::default(),
                        &entries,
                    )
                    .await;
                let request_changed = sanitized_request
                    .as_ref()
                    .is_some_and(|sanitized| sanitized != &request);
                let mut annotation = if sanitized_request.is_none() || request_changed {
                    None
                } else {
                    annotated_request
                };
                if !full_payloads_enabled
                    && !agent_is_fresh
                    && let Some(sanitized_request) = sanitized_request.as_mut()
                {
                    project_llm_request_to_current_user_turn(
                        sanitized_request,
                        &mut annotation,
                        None,
                    );
                }
                let input = sanitized_request
                    .as_ref()
                    .and_then(|request| serde_json::to_value(request).ok());
                let context = global_context();
                match context.read() {
                    Ok(state) => state.build_llm_start_event(&queued_handle, input, annotation),
                    Err(_) => event,
                }
            })
        }),
        event_sanitizers,
        &subscribers,
        scope_stack,
    );
    Ok(handle)
}

#[derive(Clone, Copy)]
struct LlmCallEndBehavior {
    response_codec_errors_fatal: bool,
    attach_estimated_cost: bool,
}

struct LlmEndPayload {
    data: Option<Json>,
    annotated_response: Option<Arc<AnnotatedLlmResponse>>,
    decode_error: Option<FlowError>,
}

async fn build_llm_end_payload(
    handle: &LlmHandle,
    response: Json,
    fallback_data: Option<Json>,
    annotated_response: Option<Arc<AnnotatedLlmResponse>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
    entries: &[crate::api::registry::Guardrail<crate::api::runtime::LlmSanitizeResponseFn>],
    behavior: LlmCallEndBehavior,
) -> LlmEndPayload {
    let response_was_null_without_fallback = response.is_null() && fallback_data.is_none();
    let response = if response.is_null() {
        fallback_data.unwrap_or(response)
    } else {
        response
    };
    let sanitized_response = NemoRelayContextState::llm_sanitize_response_snapshot_chain(
        response.clone(),
        LlmSanitizeResponseContext::for_response_codec(response_codec.clone()),
        entries,
    )
    .await;
    let response_changed = sanitized_response
        .as_ref()
        .is_some_and(|sanitized_response| sanitized_response != &response);
    let data = match sanitized_response {
        Some(response) if response_was_null_without_fallback && response.is_null() => None,
        response => response,
    };
    let annotation_omitted = data.as_ref().is_none_or(Json::is_null);
    let (mut annotated_response, decode_error) = if annotation_omitted {
        (None, None)
    } else {
        resolve_llm_end_annotation(
            (!response_changed).then_some(annotated_response).flatten(),
            response_codec,
            data.as_ref(),
            &behavior,
            &handle.name,
        )
    };
    let pricing = crate::codec::response::active_pricing_resolver();
    let summary = finalize_optimization_summary(
        &handle.optimization_recorder,
        annotated_response.as_mut(),
        handle.model_name.as_deref(),
        &pricing,
    );
    if !annotation_omitted
        && annotated_response.is_none()
        && let Some(summary) = summary
    {
        annotated_response = Some(AnnotatedLlmResponse {
            optimization_summary: Some(summary),
            ..AnnotatedLlmResponse::default()
        });
    }
    LlmEndPayload {
        data,
        annotated_response: annotated_response.map(Arc::new),
        decode_error,
    }
}

/// Finish a manual LLM lifecycle span.
///
/// This emits an LLM-end event for a handle previously returned by
/// [`llm_call`].
///
/// # Parameters
/// - `handle`: LLM handle to close.
/// - `response`: Raw provider response associated with the end event.
/// - `data`: Optional application payload retained for compatibility. When the
///   raw `response` is JSON null, this payload is sanitized in its place.
/// - `metadata`: Optional JSON metadata recorded on the end event.
/// - `annotated_response`: Optional normalized response annotation produced by
///   a response codec. When omitted and `response_codec` is supplied, the
///   annotation is decoded from the sanitized end-event payload.
/// - `response_codec`: Optional response codec used to produce a normalized
///   response annotation from the sanitized end-event payload.
/// - `timestamp`: Optional timestamp recorded on the emitted end event. When
///   `None`, the runtime uses the current UTC time, or one microsecond after
///   the handle start time if the current time is not later.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when the end event has been queued for
/// sanitization and publication.
///
/// # Errors
/// Returns an error when the runtime owner check fails or internal state cannot
/// be read safely. Dispatcher submission failures are logged because
/// observability publication is best effort. Sanitizer errors discovered during
/// queued publication are logged and fail closed by omitting the governed payload.
/// Response-codec errors retain their documented fallback behavior.
///
/// # Notes
/// Sanitize-response guardrails affect only the emitted end-event payload, not
/// the caller-owned `response` value. If a sanitizer errors or panics, Relay
/// omits the payload and response annotation and does not run remaining sanitizers.
pub fn llm_call_end(params: LlmCallEndParams<'_>) -> Result<()> {
    ensure_runtime_owner()?;
    let scope_stack = params.handle.captured_scope_stack().clone();
    let (entries, subscribers) = {
        let scope_guard = scope_stack
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.llm_sanitize_response_guardrails
        });
        let subscribers =
            snapshot_event_subscribers(scope_guard.collect_scope_local_subscribers())?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        (
            state.llm_sanitize_response_entries(&scope_locals),
            subscribers,
        )
    };
    let response = params.response;
    let fallback_data = params.data;
    let handle = params.handle.clone();
    let metadata = params.metadata;
    let timestamp = params.timestamp;
    let annotated_response = params.annotated_response;
    let response_codec = params.response_codec;
    handle.optimization_recorder.close_for_finalization(None);
    enqueue_optimization_marks(&handle, &subscribers);
    let event = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.build_llm_end_event(
            EndLlmHandleParams::builder()
                .handle(&handle)
                .data(Json::Null)
                .metadata_opt(metadata.clone())
                .annotated_response_opt(annotated_response.clone())
                .timestamp_opt(timestamp)
                .build(),
        )
    };
    let event_sanitizers = snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
    dispatch_transformed_event(
        event,
        Box::new(move |event| {
            Box::pin(async move {
                let payload = build_llm_end_payload(
                    &handle,
                    response,
                    fallback_data,
                    annotated_response,
                    response_codec,
                    &entries,
                    LlmCallEndBehavior {
                        response_codec_errors_fatal: false,
                        attach_estimated_cost: false,
                    },
                )
                .await;
                if let Some(error) = payload.decode_error {
                    log::error!(
                        target: "nemo_relay.runtime",
                        event = "manual_llm_response_codec_failed";
                        "Manual LLM response annotation failed during queued publication: {error}"
                    );
                }
                let context = global_context();
                let Ok(state) = context.read() else {
                    return event;
                };
                let end_metadata = metadata_with_otel_status(metadata, "OK", None);
                state.build_llm_end_event(
                    EndLlmHandleParams::builder()
                        .handle(&handle)
                        .data_opt(payload.data)
                        .metadata_opt(end_metadata)
                        .annotated_response_opt(payload.annotated_response)
                        .timestamp_opt(timestamp)
                        .build(),
                )
            })
        }),
        event_sanitizers,
        &subscribers,
        scope_stack,
    );
    Ok(())
}

async fn llm_call_end_with_behavior(
    params: LlmCallEndParams<'_>,
    behavior: LlmCallEndBehavior,
    lifecycle_subscribers: Option<&[EventSubscriberFn]>,
) -> Result<()> {
    let LlmCallEndParams {
        handle,
        response,
        data,
        metadata,
        annotated_response,
        response_codec,
        timestamp,
    } = params;
    ensure_runtime_owner()?;
    let (entries, subscribers) = {
        let scope_stack = handle.captured_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.llm_sanitize_response_guardrails
        });
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = match lifecycle_subscribers {
            Some(subscribers) => subscribers.to_vec(),
            None => snapshot_event_subscribers(scope_subscribers)?,
        };
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let entries = state.llm_sanitize_response_entries(&scope_locals);
        (entries, subscribers)
    };
    handle.optimization_recorder.close_for_finalization(None);
    emit_optimization_marks(handle, &subscribers).await;
    let payload = build_llm_end_payload(
        handle,
        response,
        data,
        annotated_response,
        response_codec,
        &entries,
        behavior,
    )
    .await;
    let event = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let end_metadata = metadata_with_otel_status(metadata, "OK", None);
        state.build_llm_end_event(
            EndLlmHandleParams::builder()
                .handle(handle)
                .data_opt(payload.data)
                .metadata_opt(end_metadata)
                .annotated_response_opt(payload.annotated_response)
                .timestamp_opt(timestamp)
                .build(),
        )
    };
    queue_sanitized_event_with_scope_stack(event, &subscribers, handle.captured_scope_stack());
    if let Some(error) = payload.decode_error
        && behavior.response_codec_errors_fatal
    {
        Err(error)
    } else {
        Ok(())
    }
}

#[cfg(test)]
fn sanitize_context_for_request_codec(codec: Option<&dyn LlmCodec>) -> LlmSanitizeRequestContext {
    LlmSanitizeRequestContext::with_identity(
        codec.map_or(LlmCodecIdentity::None, LlmCodec::codec_identity),
    )
}

#[cfg(test)]
pub(crate) fn sanitize_context_for_response_codec(
    codec: Option<&dyn LlmResponseCodec>,
) -> LlmSanitizeResponseContext {
    LlmSanitizeResponseContext::with_identity(
        codec.map_or(LlmCodecIdentity::None, LlmResponseCodec::codec_identity),
    )
}

fn resolve_llm_end_annotation(
    annotated_response: Option<Arc<AnnotatedLlmResponse>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
    data: Option<&Json>,
    behavior: &LlmCallEndBehavior,
    provider_name: &str,
) -> (Option<AnnotatedLlmResponse>, Option<FlowError>) {
    if let Some(annotated_response) = annotated_response {
        return (Some((*annotated_response).clone()), None);
    }
    let (Some(codec), Some(response)) = (response_codec, data) else {
        return (None, None);
    };
    match codec.decode_response(response) {
        Ok(mut decoded) => {
            if behavior.attach_estimated_cost {
                attach_estimated_cost_for_provider(&mut decoded, Some(provider_name));
            }
            (Some(decoded), None)
        }
        Err(error) => (None, Some(error)),
    }
}

async fn emit_llm_end_without_output(
    handle: &LlmHandle,
    metadata: Option<Json>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
    lifecycle_subscribers: Option<&[EventSubscriberFn]>,
) -> Result<()> {
    ensure_runtime_owner()?;
    let (entries, subscribers) = {
        let scope_stack = handle.captured_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.llm_sanitize_response_guardrails
        });
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = match lifecycle_subscribers {
            Some(subscribers) => subscribers.to_vec(),
            None => snapshot_event_subscribers(scope_subscribers)?,
        };
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let entries = state.llm_sanitize_response_entries(&scope_locals);
        (entries, subscribers)
    };
    let had_fallback_data = handle.data.is_some();
    let data = if let Some(data) = handle.data.clone() {
        NemoRelayContextState::llm_sanitize_response_snapshot_chain(
            data,
            LlmSanitizeResponseContext::for_response_codec(response_codec),
            &entries,
        )
        .await
    } else {
        None
    };
    let annotation_omitted =
        (had_fallback_data && data.is_none()) || data.as_ref().is_some_and(Json::is_null);
    handle.optimization_recorder.close_for_finalization(None);
    emit_optimization_marks(handle, &subscribers).await;
    let pricing = crate::codec::response::active_pricing_resolver();
    let annotated_response = (!annotation_omitted)
        .then(|| {
            finalize_optimization_summary(
                &handle.optimization_recorder,
                None,
                handle.model_name.as_deref(),
                &pricing,
            )
        })
        .flatten()
        .map(|summary| {
            Arc::new(AnnotatedLlmResponse {
                optimization_summary: Some(summary),
                ..AnnotatedLlmResponse::default()
            })
        });
    let event = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.end_llm_handle(handle, data, metadata, annotated_response)
    };
    queue_sanitized_event_with_scope_stack(event, &subscribers, handle.captured_scope_stack());
    Ok(())
}

struct ManagedLlmCompletion {
    handle: Option<LlmHandle>,
    metadata: Option<Json>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
    subscribers: Vec<EventSubscriberFn>,
    pending_publication: Option<PendingPublication>,
}

impl ManagedLlmCompletion {
    fn new(
        handle: &LlmHandle,
        metadata: Option<Json>,
        response_codec: Option<Arc<dyn LlmResponseCodec>>,
        subscribers: &[EventSubscriberFn],
    ) -> Self {
        Self {
            handle: Some(handle.clone()),
            metadata,
            response_codec,
            subscribers: subscribers.to_vec(),
            pending_publication: (!subscribers.is_empty())
                .then(register_pending_publication)
                .flatten(),
        }
    }

    fn disarm(&mut self) {
        self.handle = None;
        drop(self.pending_publication.take());
    }
}

impl Drop for ManagedLlmCompletion {
    fn drop(&mut self) {
        let pending_publication = self.pending_publication.take();
        let Some(handle) = self.handle.take() else {
            return;
        };
        let metadata = metadata_with_otel_status(
            self.metadata.take(),
            "ERROR",
            Some("LLM execution cancelled".into()),
        );
        let scope_stack = handle.captured_scope_stack().clone();
        let entries = match scope_stack.read() {
            Ok(scope_guard) => {
                let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
                    &registries.llm_sanitize_response_guardrails
                });
                global_context()
                    .read()
                    .map(|state| state.llm_sanitize_response_entries(&scope_locals))
                    .unwrap_or_default()
            }
            Err(_) => Vec::new(),
        };
        handle
            .optimization_recorder
            .close_for_finalization(Some("execution_cancelled"));
        enqueue_optimization_marks(&handle, &self.subscribers);
        let event = global_context()
            .read()
            .ok()
            .map(|state| state.end_llm_handle(&handle, None, metadata.clone(), None));
        let Some(event) = event else {
            return;
        };
        let event_sanitizers = snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
        let response_codec = self.response_codec.take();
        let subscribers = std::mem::take(&mut self.subscribers);
        let fallback_data = handle.data.clone();
        dispatch_transformed_event(
            event,
            Box::new(move |event| {
                Box::pin(async move {
                    let Some(data) = fallback_data else {
                        return event;
                    };
                    let data = NemoRelayContextState::llm_sanitize_response_snapshot_chain(
                        data,
                        LlmSanitizeResponseContext::for_response_codec(response_codec),
                        &entries,
                    )
                    .await;
                    let annotation_omitted = data.as_ref().is_none_or(Json::is_null);
                    let annotated_response = (!annotation_omitted)
                        .then(|| {
                            let pricing = crate::codec::response::active_pricing_resolver();
                            finalize_optimization_summary(
                                &handle.optimization_recorder,
                                None,
                                handle.model_name.as_deref(),
                                &pricing,
                            )
                        })
                        .flatten()
                        .map(|summary| {
                            Arc::new(AnnotatedLlmResponse {
                                optimization_summary: Some(summary),
                                ..AnnotatedLlmResponse::default()
                            })
                        });
                    global_context()
                        .read()
                        .map(|state| {
                            state.end_llm_handle(&handle, data, metadata, annotated_response)
                        })
                        .unwrap_or(event)
                })
            }),
            event_sanitizers,
            &subscribers,
            scope_stack,
        );
        drop(pending_publication);
    }
}

/// Execute an LLM call through the managed middleware pipeline.
///
/// This runs conditional-execution guardrails, request intercepts, and
/// sanitize-request guardrails, emits the LLM-start event, then runs execution
/// intercepts, the provider callback when it is not replaced, and
/// sanitize-response guardrails in the runtime-defined order.
///
/// # Parameters
/// - `name`: Logical provider or model family name recorded on emitted events.
/// - `request`: Raw [`LlmRequest`] passed into the managed pipeline.
/// - `func`: Provider callback or execution continuation.
/// - `parent`: Optional explicit parent scope for the emitted LLM span.
/// - `attributes`: LLM attribute bitflags applied to the managed span.
/// - `data`: Optional application payload stored on the managed LLM handle. It
///   may be used on failure end events that have no output payload.
/// - `metadata`: Optional JSON metadata recorded on emitted events.
/// - `model_name`: Optional normalized model name for observability output.
/// - `codec`: Optional request codec used to produce annotated request data for
///   intercepts and events.
/// - `response_codec`: Optional response codec used to attach annotated
///   response data to the end event.
///
/// # Returns
/// A [`Result`] containing the raw JSON response returned by the callback or
/// an execution intercept.
///
/// # Errors
/// Returns [`FlowError::GuardrailRejected`] when conditional-execution
/// guardrails block the call, or any error raised by request intercepts,
/// execution intercepts, codecs, or the callback itself.
///
/// # Notes
/// The LLM-start event is emitted before execution intercepts run. Before
/// sanitize-request guardrails run, the runtime removes standard credential
/// headers from the event-only request copy; the request passed to execution is
/// unchanged. When execution fails after that point, the runtime still emits an
/// LLM-end event without an output payload.
///
/// Response codecs enrich observability output only and do not change the
/// value returned to the caller.
pub async fn llm_call_execute(params: LlmCallExecuteParams) -> Result<Json> {
    let LlmCallExecuteParams {
        name,
        request,
        func,
        parent,
        attributes,
        data,
        metadata,
        model_name,
        codec,
        response_codec,
    } = params;
    ensure_runtime_owner()?;
    {
        let (entries, subscribers, parent_uuid, guardrail_metadata) = {
            let scope_stack = current_scope_stack();
            let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
            let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
                &registries.llm_conditional_execution_guardrails
            });
            let scope_subscribers = scope_guard.collect_scope_local_subscribers();
            let context = global_context();
            let state = context
                .read()
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            let entries = state.llm_conditional_execution_entries(&scope_locals);
            let subscribers = state.collect_event_subscribers(&scope_subscribers);
            (
                entries,
                subscribers,
                resolve_parent_uuid(parent.as_ref()),
                metadata.clone(),
            )
        };
        if let Some(error) = NemoRelayContextState::llm_conditional_execution_snapshot_chain(
            &request,
            &entries,
            &subscribers,
            parent_uuid,
            guardrail_metadata,
        )
        .await?
        {
            let mut rejection_data = json!({});
            if let Some(object) = rejection_data.as_object_mut() {
                object.insert("rejected".into(), json!(true));
                object.insert("rejection_reason".into(), json!(&error));
            }
            let _ = event(
                EmitMarkEventParams::builder()
                    .name(&name)
                    .parent_opt(parent.as_ref())
                    .data(rejection_data)
                    .metadata_opt(metadata.clone())
                    .build(),
            );
            return Err(FlowError::GuardrailRejected(error));
        }
    }

    let request_codec = codec.clone();
    let llm_uuid = Uuid::now_v7();
    let optimization_recorder = LlmOptimizationRecorder::default();
    let (mut intercepted_request, annotated_request, pending_marks, optimization_contributions) =
        scope_llm_optimization_recorder(optimization_recorder.clone(), async {
            run_request_intercepts_with_codec_and_recorder(
                &name,
                request,
                codec,
                &optimization_recorder,
            )
            .await
        })
        .await?;

    let mut handle = create_llm_handle(
        CreateLlmHandleParams::builder()
            .name(name.as_str())
            .uuid(llm_uuid)
            .parent_uuid_opt(resolve_parent_uuid(parent.as_ref()))
            .attributes(attributes)
            .data_opt(data.clone())
            .metadata_opt(metadata.clone())
            .model_name_opt(model_name)
            .build(),
    )?;
    handle.optimization_recorder = optimization_recorder;
    let lifecycle_subscribers = {
        let scope_stack = handle.captured_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        snapshot_event_subscribers(scope_guard.collect_scope_local_subscribers())?
    };
    emit_llm_start_with_subscribers(
        &handle,
        &intercepted_request,
        annotated_request.clone(),
        request_codec.clone(),
        &lifecycle_subscribers,
    )
    .await?;
    inject_traceparent(&mut intercepted_request, handle.uuid)?;
    emit_pending_request_marks(&handle, pending_marks, &lifecycle_subscribers).await?;
    handle
        .optimization_recorder
        .record_all(optimization_contributions);
    emit_optimization_marks(&handle, &lifecycle_subscribers).await;

    let mut completion = ManagedLlmCompletion::new(
        &handle,
        metadata.clone(),
        response_codec.clone(),
        &lifecycle_subscribers,
    );
    let execution_name = name.clone();
    let event_uuid = handle.uuid;
    let execution = with_active_event_uuid(
        event_uuid,
        scope_llm_optimization_recorder(handle.optimization_recorder.clone(), async move {
            let execution = {
                let scope_stack = current_scope_stack();
                let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
                let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
                    &registries.llm_execution_intercepts
                });
                let context = global_context();
                let state = context
                    .read()
                    .map_err(|error| FlowError::Internal(error.to_string()))?;
                state.llm_build_execution_chain(&execution_name, func, &scope_locals)
            };
            execution(intercepted_request).await
        }),
    )
    .await;

    match execution {
        Ok(response) => {
            llm_call_end_with_behavior(
                LlmCallEndParams::builder()
                    .handle(&handle)
                    .response(response.clone())
                    .data_opt(data)
                    .metadata_opt(metadata)
                    .response_codec_opt(response_codec)
                    .build(),
                LlmCallEndBehavior {
                    response_codec_errors_fatal: false,
                    attach_estimated_cost: true,
                },
                Some(&lifecycle_subscribers),
            )
            .await?;
            completion.disarm();
            Ok(response)
        }
        Err(error) => {
            let end_metadata = metadata_with_otel_error(metadata, &error);
            let _ = emit_llm_end_without_output(
                &handle,
                end_metadata,
                response_codec,
                Some(&lifecycle_subscribers),
            )
            .await;
            completion.disarm();
            Err(error)
        }
    }
}

/// Execute a streaming LLM call through the managed middleware pipeline.
///
/// This runs the same pre-execution middleware as [`llm_call_execute`], emits
/// the LLM-start event, and then wraps the provider stream so chunk callbacks
/// and finalization can emit a single LLM-end event when streaming completes.
///
/// # Parameters
/// - `name`: Logical provider or model family name recorded on emitted events.
/// - `request`: Raw [`LlmRequest`] passed into the managed pipeline.
/// - `func`: Streaming provider callback or execution continuation.
/// - `collector`: Per-chunk collector callback used to accumulate stream state.
/// - `finalizer`: Finalizer callback used to construct the completed response.
/// - `parent`: Optional explicit parent scope for the emitted LLM span.
/// - `attributes`: LLM attribute bitflags applied to the managed span.
/// - `data`: Optional application payload stored on the managed LLM handle. It
///   may be used on failure end events that have no output payload.
/// - `metadata`: Optional JSON metadata recorded on emitted events.
/// - `model_name`: Optional normalized model name for observability output.
/// - `codec`: Optional request codec used to produce annotated request data for
///   intercepts and events.
/// - `response_codec`: Optional response codec used to attach annotated
///   response data to the end event.
///
/// # Returns
/// A [`Result`] containing a boxed stream of JSON chunks.
///
/// # Errors
/// Returns [`FlowError::GuardrailRejected`] when conditional-execution
/// guardrails block the call, or any error raised by request intercepts,
/// execution intercepts, stream callbacks, codecs, or the provider callback.
///
/// # Notes
/// The LLM-start event is emitted before stream execution intercepts run.
/// Before sanitize-request guardrails run, the runtime removes standard
/// credential headers from the event-only request copy; the request passed to
/// stream execution is unchanged.
///
/// The returned stream emits chunk-level results while the runtime defers the
/// LLM-end event until the collector and finalizer complete.
pub async fn llm_stream_call_execute(params: LlmStreamCallExecuteParams) -> Result<LlmJsonStream> {
    let LlmStreamCallExecuteParams {
        name,
        request,
        func,
        collector,
        finalizer,
        parent,
        attributes,
        data,
        metadata,
        model_name,
        codec,
        response_codec,
    } = params;
    ensure_runtime_owner()?;
    {
        let (entries, subscribers, parent_uuid, guardrail_metadata) = {
            let scope_stack = current_scope_stack();
            let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
            let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
                &registries.llm_conditional_execution_guardrails
            });
            let scope_subscribers = scope_guard.collect_scope_local_subscribers();
            let context = global_context();
            let state = context
                .read()
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            let entries = state.llm_conditional_execution_entries(&scope_locals);
            let subscribers = state.collect_event_subscribers(&scope_subscribers);
            (
                entries,
                subscribers,
                resolve_parent_uuid(parent.as_ref()),
                metadata.clone(),
            )
        };
        if let Some(error) = NemoRelayContextState::llm_conditional_execution_snapshot_chain(
            &request,
            &entries,
            &subscribers,
            parent_uuid,
            guardrail_metadata,
        )
        .await?
        {
            let mut rejection_data = json!({});
            if let Some(object) = rejection_data.as_object_mut() {
                object.insert("rejected".into(), json!(true));
                object.insert("rejection_reason".into(), json!(&error));
            }
            let _ = event(
                EmitMarkEventParams::builder()
                    .name(&name)
                    .parent_opt(parent.as_ref())
                    .data(rejection_data)
                    .metadata_opt(metadata.clone())
                    .build(),
            );
            return Err(FlowError::GuardrailRejected(error));
        }
    }

    let request_codec = codec.clone();
    let llm_uuid = Uuid::now_v7();
    let optimization_recorder = LlmOptimizationRecorder::default();
    let (mut intercepted_request, annotated_request, pending_marks, optimization_contributions) =
        scope_llm_optimization_recorder(optimization_recorder.clone(), async {
            run_request_intercepts_with_codec_and_recorder(
                &name,
                request,
                codec,
                &optimization_recorder,
            )
            .await
        })
        .await?;

    let mut handle = create_llm_handle(
        CreateLlmHandleParams::builder()
            .name(name.as_str())
            .uuid(llm_uuid)
            .parent_uuid_opt(resolve_parent_uuid(parent.as_ref()))
            .attributes(attributes)
            .data_opt(data.clone())
            .metadata_opt(metadata.clone())
            .model_name_opt(model_name)
            .build(),
    )?;
    handle.optimization_recorder = optimization_recorder;
    let lifecycle_subscribers = {
        let scope_stack = handle.captured_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        snapshot_event_subscribers(scope_guard.collect_scope_local_subscribers())?
    };
    emit_llm_start_with_subscribers(
        &handle,
        &intercepted_request,
        annotated_request,
        request_codec.clone(),
        &lifecycle_subscribers,
    )
    .await?;
    inject_traceparent(&mut intercepted_request, handle.uuid)?;
    emit_pending_request_marks(&handle, pending_marks, &lifecycle_subscribers).await?;
    handle
        .optimization_recorder
        .record_all(optimization_contributions);
    emit_optimization_marks(&handle, &lifecycle_subscribers).await;

    let mut completion = ManagedLlmCompletion::new(
        &handle,
        metadata.clone(),
        response_codec.clone(),
        &lifecycle_subscribers,
    );
    let execution_name = name.clone();
    let event_uuid = handle.uuid;
    let execution = with_active_event_uuid(
        event_uuid,
        scope_llm_optimization_recorder(handle.optimization_recorder.clone(), async move {
            let execution = {
                let scope_stack = current_scope_stack();
                let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
                let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
                    &registries.llm_stream_execution_intercepts
                });
                let context = global_context();
                let state = context
                    .read()
                    .map_err(|error| FlowError::Internal(error.to_string()))?;
                state.llm_stream_build_execution_chain(&execution_name, func, &scope_locals)
            };
            let execution_context = MiddlewareContinuationContext::capture();
            execution(intercepted_request)
                .await
                .map(|stream| contextualize_stream(stream, execution_context))
        }),
    )
    .await;

    match execution {
        Ok(raw_stream) => {
            let wrapper = LlmStreamWrapper::new_managed(
                raw_stream,
                handle,
                collector,
                finalizer,
                metadata,
                response_codec,
                lifecycle_subscribers,
            );
            completion.disarm();
            Ok(LlmJsonStream::from_closeable(wrapper))
        }
        Err(error) => {
            let end_metadata = metadata_with_otel_error(metadata, &error);
            let _ = emit_llm_end_without_output(
                &handle,
                end_metadata,
                response_codec,
                Some(&lifecycle_subscribers),
            )
            .await;
            completion.disarm();
            Err(error)
        }
    }
}

/// Run only the LLM request-intercept chain.
///
/// This applies the currently active global and scope-local request intercepts
/// without emitting lifecycle events or invoking provider execution.
///
/// # Parameters
/// - `name`: Logical provider or model family name used when resolving the
///   intercept chain.
/// - `request`: Raw [`LlmRequest`] to transform.
///
/// # Returns
/// A [`Result`] containing the transformed [`LlmRequest`].
///
/// # Errors
/// Returns any error raised by the request-intercept chain.
///
/// # Notes
/// Conditional guardrails, codecs, and execution intercepts are not run by
/// this helper.
/// Run the LLM request-intercept chain and return its complete outcome.
///
/// This helper does not emit the returned marks because it does not own an LLM
/// lifecycle. Callers must attach them to the lifecycle they own.
pub async fn llm_request_intercepts(
    name: &str,
    request: LlmRequest,
) -> Result<LlmRequestInterceptOutcome> {
    ensure_runtime_owner()?;
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
    let mut outcome = NemoRelayContextState::llm_request_intercepts_snapshot_chain(
        name, request, None, &entries, false,
    )
    .await?;
    inject_dynamo_session_ids(&mut outcome.request);
    if let Ok(traceparent) = capture_traceparent() {
        inject_traceparent_value(&mut outcome.request, traceparent);
    }
    Ok(outcome)
}

/// Run only the LLM conditional-execution guardrail chain.
///
/// This evaluates whether an LLM call should be allowed to proceed without
/// invoking request intercepts or execution. Each evaluated guardrail emits an
/// automatic guardrail scope start/end pair for observability.
///
/// # Parameters
/// - `request`: Raw [`LlmRequest`] to validate.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when all guardrails allow execution.
///
/// # Errors
/// Returns [`FlowError::GuardrailRejected`] when a guardrail blocks execution,
/// or any error raised by the guardrail chain itself.
///
/// # Notes
/// This helper is useful for preflight checks when the caller needs the
/// rejection result without starting an LLM span. Guardrail scopes are still
/// emitted for the conditional checks themselves.
pub async fn llm_conditional_execution(request: &LlmRequest) -> Result<()> {
    ensure_runtime_owner()?;
    let (entries, subscribers, parent_uuid) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.llm_conditional_execution_guardrails
        });
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let entries = state.llm_conditional_execution_entries(&scope_locals);
        let subscribers = state.collect_event_subscribers(&scope_subscribers);
        (entries, subscribers, resolve_parent_uuid(None))
    };
    if let Some(error) = NemoRelayContextState::llm_conditional_execution_snapshot_chain(
        request,
        &entries,
        &subscribers,
        parent_uuid,
        None,
    )
    .await?
    {
        return Err(FlowError::GuardrailRejected(error));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/llm_api_tests.rs"]
mod tests;
