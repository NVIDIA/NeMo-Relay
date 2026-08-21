// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;

use crate::api::event::{BaseEvent, Event, MarkEvent, PendingMarkSpec};
use crate::api::registry::RuntimeRegistrationKind;
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::current_scope_stack;
use crate::api::runtime::global_context;
use crate::api::runtime::subscriber_dispatcher::{
    PendingPublication, dispatch_sanitized_event, dispatch_transformed_event,
    register_pending_publication,
};
use crate::api::runtime::{
    EventSubscriberFn, ScopeStackHandle, ToolExecutionNextFn, with_active_event_uuid,
};
use crate::api::scope::event;
use crate::api::scope::{EmitMarkEventParams, ScopeHandle, metadata_with_log_severity};
use crate::api::shared::{
    ensure_runtime_owner, metadata_with_otel_error, metadata_with_otel_status, resolve_parent_uuid,
    snapshot_event_sanitizers, snapshot_event_subscribers,
};
use crate::api::skill_load;
use crate::error::{FlowError, Result};
use crate::json::Json;
use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;
use uuid::Uuid;

pub use nemo_relay_types::api::tool::{
    TOOL_EXECUTION_INTERCEPT_OUTCOME_SCHEMA, TOOL_EXECUTION_RESULT_SCHEMA, ToolAttributes,
    ToolExecutionInterceptOutcome, ToolExecutionResult,
};

fn queue_sanitized_event_with_scope_stack(
    event: Event,
    subscribers: &[EventSubscriberFn],
    scope_stack: ScopeStackHandle,
) -> bool {
    let sanitizers = snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
    dispatch_sanitized_event(event, sanitizers, subscribers, scope_stack)
}

/// Runtime-owned handle identifying an active or completed tool call.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct ToolHandle {
    /// Unique tool-call identifier.
    #[builder(default = Uuid::now_v7())]
    pub uuid: Uuid,
    /// Timestamp captured when the tool handle was created.
    #[builder(default = Utc::now())]
    pub started_at: DateTime<Utc>,
    /// Tool name recorded on lifecycle events.
    #[builder(setter(into))]
    pub name: String,
    /// Optional application payload stored on the handle.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional metadata attached to the tool span.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Tool behavior flags.
    #[builder(default = ToolAttributes::empty())]
    pub attributes: ToolAttributes,
    /// UUID of the parent scope, if any.
    #[builder(default)]
    pub parent_uuid: Option<Uuid>,
    /// Optional provider-specific tool-call correlation identifier.
    #[builder(default, setter(into))]
    pub tool_call_id: Option<String>,
}

/// Builder parameters for [`NemoRelayContextState::create_tool_handle`].
#[derive(Debug, Clone, TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct CreateToolHandleParams<'a> {
    /// Tool name recorded on emitted events.
    pub name: &'a str,
    /// Optional parent scope UUID.
    #[builder(default)]
    pub parent_uuid: Option<uuid::Uuid>,
    /// Tool attribute bitflags.
    #[builder(default = ToolAttributes::empty())]
    pub attributes: ToolAttributes,
    /// Optional application payload stored on the handle.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional metadata stored on the handle.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional provider-specific correlation identifier.
    #[builder(default, setter(into))]
    pub tool_call_id: Option<String>,
    /// Optional timestamp captured as the handle start time and reused by the
    /// emitted start event. When omitted, the current UTC time is used.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

fn resolve_skill_loads(
    name: &str,
    args: &Json,
    metadata: Option<&Json>,
) -> Vec<skill_load::SkillLoad> {
    let already_handled = metadata
        .and_then(Json::as_object)
        .and_then(|metadata| metadata.get(skill_load::HANDLED_METADATA_KEY))
        .and_then(Json::as_bool)
        .unwrap_or(false);
    if already_handled {
        Vec::new()
    } else if let Some(skill_loads) = skill_load::precomputed(metadata) {
        skill_loads
    } else {
        skill_load::detect(name, args)
    }
}

/// Builder parameters for [`NemoRelayContextState::build_tool_end_event`].
#[derive(Debug, Clone, TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct EndToolHandleParams<'a> {
    /// Tool handle to serialize into the emitted end event.
    pub handle: &'a ToolHandle,
    /// Optional data payload merged over the handle data.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional metadata payload merged over the handle metadata.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional timestamp recorded on the emitted end event. When omitted, the
    /// runtime records the current UTC time, or one microsecond after the
    /// handle start time if the current time is not later.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Builder parameters for [`tool_call`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct ToolCallParams<'a> {
    /// Tool name recorded on the emitted lifecycle event.
    pub name: &'a str,
    /// Raw tool arguments associated with the span.
    pub args: Json,
    /// Optional explicit parent scope.
    #[builder(default)]
    pub parent: Option<&'a ScopeHandle>,
    /// Tool attribute bitflags applied to the span.
    #[builder(default = ToolAttributes::empty())]
    pub attributes: ToolAttributes,
    /// Optional application payload stored on the handle but not emitted as
    /// Agent Trajectory Observability Format (ATOF) data.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional JSON metadata recorded on the start event.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional provider-specific correlation identifier.
    #[builder(default, setter(into))]
    pub tool_call_id: Option<String>,
    /// Optional timestamp captured as the handle start time and reused by the
    /// emitted start event. When omitted, the current UTC time is used.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Builder parameters for [`tool_call_execute`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct ToolCallExecuteParams {
    /// Tool name recorded on emitted lifecycle events.
    #[builder(setter(into))]
    pub name: String,
    /// Raw tool arguments passed into the managed pipeline.
    pub args: Json,
    /// Tool callback or execution continuation.
    pub func: ToolExecutionNextFn,
    /// Optional explicit parent scope for the emitted tool span.
    #[builder(default)]
    pub parent: Option<ScopeHandle>,
    /// Tool attribute bitflags applied to the managed span.
    #[builder(default = ToolAttributes::empty())]
    pub attributes: ToolAttributes,
    /// Optional application payload stored on the handle but not emitted as
    /// Agent Trajectory Observability Format (ATOF) data.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional JSON metadata recorded on emitted events.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional provider-specific correlation identifier.
    #[builder(default, setter(into))]
    pub tool_call_id: Option<String>,
}

/// Builder parameters for [`tool_call_end`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct ToolCallEndParams<'a> {
    /// Tool handle to close.
    pub handle: &'a ToolHandle,
    /// Application result and optional opaque annotation associated with the
    /// end event.
    pub execution_result: ToolExecutionResult,
    /// Optional application payload retained for compatibility; Agent
    /// Trajectory Observability Format (ATOF) data is the result.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional JSON metadata recorded on the end event.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional timestamp recorded on the emitted end event. When omitted, the
    /// runtime records the current UTC time, or one microsecond after the
    /// handle start time if the current time is not later.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Start a manual tool lifecycle span.
///
/// This submits a tool-start event for queued sanitize-request guardrails and
/// publication without waiting for that work.
///
/// # Parameters
/// - `name`: Tool name recorded on the emitted lifecycle event.
/// - `args`: Raw tool arguments associated with the span.
/// - `parent`: Optional explicit parent scope.
/// - `attributes`: Tool attribute bitflags applied to the span.
/// - `data`: Optional application payload stored on the returned handle. The
///   emitted start event data is the sanitized `args` payload.
/// - `metadata`: Optional JSON metadata recorded on the start event.
/// - `tool_call_id`: Optional provider-specific correlation identifier.
/// - `timestamp`: Optional timestamp recorded as the handle start time and on
///   the emitted start event. When `None`, the current UTC time is used.
///
/// # Returns
/// A [`Result`] containing the created [`ToolHandle`] after its start-event
/// snapshot has been submitted for queued publication.
///
/// # Errors
/// Returns an error when the runtime owner check fails or when internal state
/// cannot be read safely. Dispatcher submission failures are logged because
/// observability publication is best effort.
///
/// # Notes
/// Sanitize-request guardrails affect only the emitted start-event payload, not
/// the caller-owned `args` value. If a sanitizer errors or panics, Relay omits
/// that observability payload and does not run remaining sanitizers.
pub fn tool_call(params: ToolCallParams<'_>) -> Result<ToolHandle> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let (scope_locals, scope_subscribers) = {
        let scope_guard = scope_stack
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        (
            scope_guard.snapshot_scope_local_registries(|registries| {
                &registries.tool_sanitize_request_guardrails
            }),
            scope_guard.collect_scope_local_subscribers(),
        )
    };
    let (entries, subscribers) = {
        let scope_local_refs = scope_locals.iter().collect::<Vec<_>>();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?
            .registry_snapshot(&[RuntimeRegistrationKind::ToolSanitizeRequestGuardrail]);
        (
            state.tool_sanitize_request_entries(&scope_local_refs),
            subscribers,
        )
    };
    let skill_loads = resolve_skill_loads(params.name, &params.args, params.metadata.as_ref());
    let raw_args = params.args;
    let (handle, event, marks) = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let handle = state.create_tool_handle(
            CreateToolHandleParams::builder()
                .name(params.name)
                .parent_uuid_opt(resolve_parent_uuid(params.parent))
                .attributes(params.attributes)
                .data_opt(params.data)
                .metadata_opt(params.metadata)
                .tool_call_id_opt(params.tool_call_id)
                .timestamp_opt(params.timestamp)
                .build(),
        );
        let event = state.build_tool_start_event(&handle, None);
        let marks = skill_loads
            .into_iter()
            .map(|skill_load| {
                state.create_event(MarkEvent::new(
                    BaseEvent::builder()
                        .name("skill.load")
                        .parent_uuid(handle.uuid)
                        .timestamp(handle.started_at)
                        .data(json!({"skill_name": skill_load.name}))
                        .metadata(json!({
                            "skill_load_source": <&str>::from(skill_load.source),
                            "tool_name": handle.name,
                        }))
                        .build(),
                    None,
                    None,
                ))
            })
            .collect::<Vec<_>>();
        (handle, event, marks)
    };
    let tool_name = handle.name.clone();
    let event_sanitizers = snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
    dispatch_transformed_event(
        event,
        Box::new(move |mut event| {
            Box::pin(async move {
                let sanitized = NemoRelayContextState::tool_sanitize_request_snapshot_chain(
                    &tool_name, raw_args, &entries,
                )
                .await;
                let mut fields = event.sanitize_fields();
                fields.data = sanitized;
                event.apply_sanitize_fields(fields);
                event
            })
        }),
        event_sanitizers,
        &subscribers,
        scope_stack.clone(),
    );
    for mark in marks {
        let sanitizers = snapshot_event_sanitizers(&mark, &scope_stack).unwrap_or_default();
        dispatch_sanitized_event(mark, sanitizers, &subscribers, scope_stack.clone());
    }
    Ok(handle)
}

async fn tool_call_with_subscriber_snapshot(
    params: ToolCallParams<'_>,
) -> Result<(ToolHandle, Vec<EventSubscriberFn>)> {
    ensure_runtime_owner()?;
    let parent_uuid = resolve_parent_uuid(params.parent);
    let scope_stack = current_scope_stack();
    let (scope_locals, scope_subscribers) = {
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        (
            scope_guard.snapshot_scope_local_registries(|registries| {
                &registries.tool_sanitize_request_guardrails
            }),
            scope_guard.collect_scope_local_subscribers(),
        )
    };
    let (entries, subscribers) = {
        let scope_local_refs = scope_locals.iter().collect::<Vec<_>>();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?
            .registry_snapshot(&[RuntimeRegistrationKind::ToolSanitizeRequestGuardrail]);
        let entries = state.tool_sanitize_request_entries(&scope_local_refs);
        (entries, subscribers)
    };
    let skill_loads = resolve_skill_loads(params.name, &params.args, params.metadata.as_ref());
    let raw_args = params.args;
    let (handle, event, marks) = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let handle_params = CreateToolHandleParams::builder()
            .name(params.name)
            .parent_uuid_opt(parent_uuid)
            .attributes(params.attributes)
            .data_opt(params.data)
            .metadata_opt(params.metadata)
            .tool_call_id_opt(params.tool_call_id)
            .timestamp_opt(params.timestamp)
            .build();
        let handle = state.create_tool_handle(handle_params);
        let event = state.build_tool_start_event(&handle, None);
        let marks = skill_loads
            .into_iter()
            .map(|skill_load| {
                state.create_event(MarkEvent::new(
                    BaseEvent::builder()
                        .name("skill.load")
                        .parent_uuid(handle.uuid)
                        .timestamp(handle.started_at)
                        .data(json!({"skill_name": skill_load.name}))
                        .metadata(json!({
                            "skill_load_source": <&str>::from(skill_load.source),
                            "tool_name": handle.name,
                        }))
                        .build(),
                    None,
                    None,
                ))
            })
            .collect::<Vec<_>>();
        (handle, event, marks)
    };
    let event_sanitizers = snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
    let tool_name = handle.name.clone();
    dispatch_transformed_event(
        event,
        Box::new(move |mut event| {
            Box::pin(async move {
                let sanitized = NemoRelayContextState::tool_sanitize_request_snapshot_chain(
                    &tool_name, raw_args, &entries,
                )
                .await;
                let mut fields = event.sanitize_fields();
                fields.data = sanitized;
                event.apply_sanitize_fields(fields);
                event
            })
        }),
        event_sanitizers,
        &subscribers,
        scope_stack.clone(),
    );
    for mark in marks {
        let sanitizers = snapshot_event_sanitizers(&mark, &scope_stack).unwrap_or_default();
        dispatch_sanitized_event(mark, sanitizers, &subscribers, scope_stack.clone());
    }
    Ok((handle, subscribers))
}

/// Finish a manual tool lifecycle span.
///
/// This submits a tool-end event for queued sanitization and publication for a
/// handle previously returned by [`tool_call`].
///
/// # Parameters
/// - `handle`: Tool handle to close.
/// - `execution_result`: Application result and optional opaque annotation
///   associated with the end event.
/// - `data`: Optional application payload retained for compatibility. The
///   emitted end event data is the sanitized application result unless it
///   sanitizes to JSON null, in which case this payload is used.
/// - `metadata`: Optional JSON metadata recorded on the end event.
/// - `timestamp`: Optional timestamp recorded on the emitted end event. When
///   `None`, the runtime uses the current UTC time, or one microsecond after
///   the handle start time if the current time is not later.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when the end-event snapshot has been submitted
/// for queued publication.
///
/// # Errors
/// Returns an error when the runtime owner check fails or when internal state
/// cannot be read safely. Dispatcher submission failures are logged because
/// observability publication is best effort.
///
/// # Notes
/// Sanitize-response guardrails affect only the emitted end-event payload, not
/// the caller-owned execution result. General event sanitizers can rewrite or
/// remove the emitted annotation. If a sanitizer errors or panics, Relay omits
/// the governed observability fields and does not run remaining sanitizers.
pub fn tool_call_end(params: ToolCallEndParams<'_>) -> Result<()> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let (scope_locals, scope_subscribers) = {
        let scope_guard = scope_stack
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        (
            scope_guard.snapshot_scope_local_registries(|registries| {
                &registries.tool_sanitize_response_guardrails
            }),
            scope_guard.collect_scope_local_subscribers(),
        )
    };
    let (entries, subscribers) = {
        let scope_local_refs = scope_locals.iter().collect::<Vec<_>>();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?
            .registry_snapshot(&[RuntimeRegistrationKind::ToolSanitizeResponseGuardrail]);
        (
            state.tool_sanitize_response_entries(&scope_local_refs),
            subscribers,
        )
    };
    let ToolExecutionResult { result, annotation } = params.execution_result;
    let fallback = params.data;
    let event = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.build_tool_end_event(
            EndToolHandleParams::builder()
                .handle(params.handle)
                .data(Json::Null)
                .metadata_opt(params.metadata)
                .timestamp_opt(params.timestamp)
                .build(),
        )
    };
    let tool_name = params.handle.name.clone();
    let event_sanitizers = snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
    dispatch_transformed_event(
        event,
        Box::new(move |mut event| {
            Box::pin(async move {
                let sanitized = NemoRelayContextState::tool_sanitize_response_snapshot_chain(
                    &tool_name, result, &entries,
                )
                .await;
                let mut fields = event.sanitize_fields();
                fields.data = sanitized.and_then(|value| {
                    if value.is_null() {
                        fallback
                    } else {
                        Some(value)
                    }
                });
                event.apply_sanitize_fields(fields);
                attach_tool_result_annotation(&mut event, annotation);
                event
            })
        }),
        event_sanitizers,
        &subscribers,
        scope_stack,
    );
    Ok(())
}

async fn tool_call_end_with_pending_marks(
    params: ToolCallEndParams<'_>,
    pending_marks: Vec<PendingMarkSpec>,
    lifecycle_subscribers: Option<&[EventSubscriberFn]>,
) -> Result<()> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let (scope_locals, scope_subscribers) = {
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        (
            scope_guard.snapshot_scope_local_registries(|registries| {
                &registries.tool_sanitize_response_guardrails
            }),
            scope_guard.collect_scope_local_subscribers(),
        )
    };
    let (entries, subscribers) = {
        let scope_local_refs = scope_locals.iter().collect::<Vec<_>>();
        let subscribers = if lifecycle_subscribers.is_some() {
            Vec::new()
        } else {
            snapshot_event_subscribers(scope_subscribers)?
        };
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?
            .registry_snapshot(&[RuntimeRegistrationKind::ToolSanitizeResponseGuardrail]);
        let entries = state.tool_sanitize_response_entries(&scope_local_refs);
        (entries, subscribers)
    };
    let subscribers = lifecycle_subscribers.unwrap_or(&subscribers);
    let event = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.build_tool_end_event(
            EndToolHandleParams::builder()
                .handle(params.handle)
                .data(Json::Null)
                .metadata_opt(params.metadata)
                .timestamp_opt(params.timestamp)
                .build(),
        )
    };
    let mut marks = Vec::with_capacity(pending_marks.len());
    for (index, mark) in pending_marks.into_iter().enumerate() {
        let timestamp = *event.timestamp()
            + TimeDelta::microseconds(i64::try_from(index).unwrap_or_default() + 1);
        let metadata = match metadata_with_log_severity(mark.metadata, mark.severity) {
            Ok(metadata) => metadata,
            Err(error) => {
                let tool_uuid = params.handle.uuid.to_string();
                log::warn!(
                    target: "nemo_relay.observability",
                    event = "tool_pending_mark_dropped",
                    tool_name = params.handle.name.as_str(),
                    tool_uuid = tool_uuid.as_str(),
                    pending_mark_index = index,
                    pending_mark_name = mark.name.as_str();
                    "Tool pending mark was dropped because its severity metadata is invalid: {error}"
                );
                continue;
            }
        };
        marks.push(Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .name(mark.name)
                .parent_uuid(params.handle.uuid)
                .timestamp(timestamp)
                .data_opt(mark.data)
                .data_schema_opt(mark.data_schema)
                .metadata_opt(metadata)
                .build(),
            mark.category,
            mark.category_profile,
        )));
    }
    let event_sanitizers = snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
    let tool_name = params.handle.name.clone();
    let ToolExecutionResult { result, annotation } = params.execution_result;
    let fallback = params.data;
    dispatch_transformed_event(
        event,
        Box::new(move |mut event| {
            Box::pin(async move {
                let sanitized = NemoRelayContextState::tool_sanitize_response_snapshot_chain(
                    &tool_name, result, &entries,
                )
                .await;
                let mut fields = event.sanitize_fields();
                fields.data = sanitized.and_then(|value| {
                    if value.is_null() {
                        fallback
                    } else {
                        Some(value)
                    }
                });
                event.apply_sanitize_fields(fields);
                attach_tool_result_annotation(&mut event, annotation);
                event
            })
        }),
        event_sanitizers,
        subscribers,
        scope_stack.clone(),
    );
    for mark in marks {
        let sanitizers = snapshot_event_sanitizers(&mark, &scope_stack).unwrap_or_default();
        dispatch_sanitized_event(mark, sanitizers, subscribers, scope_stack.clone());
    }
    Ok(())
}

fn attach_tool_result_annotation(event: &mut Event, annotation: Option<Json>) {
    let Some(annotation) = annotation.filter(|value| !value.is_null()) else {
        return;
    };
    if let Some(profile) = event.category_profile_mut() {
        profile.tool_result_annotation = Some(annotation);
    }
}

fn emit_tool_end_without_output(
    handle: &ToolHandle,
    metadata: Option<Json>,
    lifecycle_subscribers: &[EventSubscriberFn],
    scope_stack: ScopeStackHandle,
) -> Result<()> {
    ensure_runtime_owner()?;
    let event = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.end_tool_handle(handle, handle.data.clone(), metadata)
    };
    queue_sanitized_event_with_scope_stack(event, lifecycle_subscribers, scope_stack);
    Ok(())
}

struct ManagedToolCompletion {
    handle: Option<ToolHandle>,
    metadata: Option<Json>,
    subscribers: Vec<EventSubscriberFn>,
    scope_stack: ScopeStackHandle,
    pending_publication: Option<PendingPublication>,
}

impl ManagedToolCompletion {
    fn new(
        handle: &ToolHandle,
        metadata: Option<Json>,
        subscribers: &[EventSubscriberFn],
        scope_stack: ScopeStackHandle,
    ) -> Self {
        Self {
            handle: Some(handle.clone()),
            metadata,
            subscribers: subscribers.to_vec(),
            scope_stack,
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

impl Drop for ManagedToolCompletion {
    fn drop(&mut self) {
        let pending_publication = self.pending_publication.take();
        let Some(handle) = self.handle.take() else {
            return;
        };
        let metadata = metadata_with_otel_status(
            self.metadata.take(),
            "ERROR",
            Some("tool execution cancelled".into()),
        );
        let _ = emit_tool_end_without_output(
            &handle,
            metadata,
            &self.subscribers,
            self.scope_stack.clone(),
        );
        drop(pending_publication);
    }
}

/// Execute a tool call through the managed middleware pipeline.
///
/// This runs conditional-execution guardrails, request intercepts,
/// sanitize-request guardrails, execution intercepts, the tool callback, and
/// sanitize-response guardrails in the runtime-defined order.
///
/// # Parameters
/// - `name`: Tool name recorded on emitted lifecycle events.
/// - `args`: Raw tool arguments passed into the managed pipeline.
/// - `func`: Tool callback or execution continuation.
/// - `parent`: Optional explicit parent scope for the emitted tool span.
/// - `attributes`: Tool attribute bitflags applied to the managed span.
/// - `data`: Optional application payload stored on the managed tool handle.
///   It may be used on failure end events that have no output payload.
/// - `metadata`: Optional JSON metadata recorded on emitted events.
/// - `tool_call_id`: Optional provider-specific correlation identifier recorded
///   on both managed lifecycle events.
///
/// # Returns
/// A [`Result`] containing the application result and optional opaque
/// annotation returned by the callback or an execution intercept.
///
/// # Errors
/// Returns [`FlowError::GuardrailRejected`] when conditional-execution
/// guardrails block the call, or any error raised by request intercepts,
/// execution intercepts, or the callback itself.
///
/// # Notes
/// When execution fails after the start event has been emitted, the runtime
/// still emits a tool-end event without an output payload.
pub async fn tool_call_execute(params: ToolCallExecuteParams) -> Result<ToolExecutionResult> {
    let ToolCallExecuteParams {
        name,
        args,
        func,
        parent,
        attributes,
        data,
        metadata,
        tool_call_id,
    } = params;
    ensure_runtime_owner()?;
    {
        let (entries, subscribers, parent_uuid, guardrail_metadata) = {
            let scope_stack = current_scope_stack();
            let (scope_locals, scope_subscribers) = {
                let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
                (
                    scope_guard.snapshot_scope_local_registries(|registries| {
                        &registries.tool_conditional_execution_guardrails
                    }),
                    scope_guard.collect_scope_local_subscribers(),
                )
            };
            let scope_local_refs = scope_locals.iter().collect::<Vec<_>>();
            let context = global_context();
            let state = context
                .read()
                .map_err(|error| FlowError::Internal(error.to_string()))?
                .registry_snapshot(&[
                    RuntimeRegistrationKind::ToolConditionalExecutionGuardrail,
                    RuntimeRegistrationKind::Subscriber,
                ]);
            let entries = state.tool_conditional_execution_entries(&scope_local_refs);
            let subscribers = state.collect_event_subscribers(&scope_subscribers);
            (
                entries,
                subscribers,
                resolve_parent_uuid(parent.as_ref()),
                metadata.clone(),
            )
        };
        if let Some(error) = NemoRelayContextState::tool_conditional_execution_snapshot_chain(
            &name,
            &args,
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

    let intercept_entries = {
        let scope_stack = current_scope_stack();
        let scope_locals = scope_stack
            .read()
            .expect("scope stack lock poisoned")
            .snapshot_scope_local_registries(|registries| &registries.tool_request_intercepts);
        let scope_local_refs = scope_locals.iter().collect::<Vec<_>>();
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?
            .registry_snapshot(&[RuntimeRegistrationKind::ToolRequestIntercept]);
        state.tool_request_intercept_entries(&scope_local_refs)
    };
    let intercepted_args = NemoRelayContextState::tool_request_intercepts_snapshot_chain(
        &name,
        args,
        &intercept_entries,
    )
    .await?;

    let (handle, lifecycle_subscribers) = tool_call_with_subscriber_snapshot(
        ToolCallParams::builder()
            .name(name.as_str())
            .args(intercepted_args.clone())
            .parent_opt(parent.as_ref())
            .attributes(attributes)
            .data_opt(data.clone())
            .metadata_opt(metadata.clone())
            .tool_call_id_opt(tool_call_id)
            .build(),
    )
    .await?;

    let lifecycle_scope_stack = current_scope_stack();
    let mut completion = ManagedToolCompletion::new(
        &handle,
        metadata.clone(),
        &lifecycle_subscribers,
        lifecycle_scope_stack.clone(),
    );
    let execution_name = name.clone();
    let execution = with_active_event_uuid(handle.uuid, async move {
        let execution = {
            let scope_stack = current_scope_stack();
            let scope_locals = scope_stack
                .read()
                .expect("scope stack lock poisoned")
                .snapshot_scope_local_registries(|registries| {
                    &registries.tool_execution_intercepts
                });
            let scope_local_refs = scope_locals.iter().collect::<Vec<_>>();
            let context = global_context();
            let state = context
                .read()
                .map_err(|error| FlowError::Internal(error.to_string()))?
                .registry_snapshot(&[RuntimeRegistrationKind::ToolExecutionIntercept]);
            state.tool_build_execution_chain(&execution_name, func, &scope_local_refs)
        };
        execution(intercepted_args).await
    })
    .await;
    match execution {
        Ok(mut outcome) => {
            let pending_marks = std::mem::take(&mut outcome.pending_marks);
            let execution_result = outcome.into_execution_result();
            let end_metadata = metadata_with_otel_status(metadata, "OK", None);
            tool_call_end_with_pending_marks(
                ToolCallEndParams::builder()
                    .handle(&handle)
                    .execution_result(execution_result.clone())
                    .data_opt(data)
                    .metadata_opt(end_metadata)
                    .build(),
                pending_marks,
                Some(&lifecycle_subscribers),
            )
            .await?;
            completion.disarm();
            Ok(execution_result)
        }
        Err(error) => {
            let end_metadata = metadata_with_otel_error(metadata, &error);
            let _ = emit_tool_end_without_output(
                &handle,
                end_metadata,
                &lifecycle_subscribers,
                lifecycle_scope_stack,
            );
            completion.disarm();
            Err(error)
        }
    }
}

/// Run only the tool request-intercept chain.
///
/// This applies the currently active global and scope-local request intercepts
/// without emitting lifecycle events or invoking tool execution.
///
/// # Parameters
/// - `name`: Tool name used when resolving the intercept chain.
/// - `args`: Raw tool arguments to transform.
///
/// # Returns
/// A [`Result`] containing the transformed JSON arguments.
///
/// # Errors
/// Returns any error raised by the request-intercept chain.
///
/// # Notes
/// Conditional guardrails and execution intercepts are not run by this helper.
pub async fn tool_request_intercepts(name: &str, args: Json) -> Result<Json> {
    ensure_runtime_owner()?;
    let entries = {
        let scope_stack = current_scope_stack();
        let scope_locals = scope_stack
            .read()
            .expect("scope stack lock poisoned")
            .snapshot_scope_local_registries(|registries| &registries.tool_request_intercepts);
        let scope_local_refs = scope_locals.iter().collect::<Vec<_>>();
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?
            .registry_snapshot(&[RuntimeRegistrationKind::ToolRequestIntercept]);
        state.tool_request_intercept_entries(&scope_local_refs)
    };
    NemoRelayContextState::tool_request_intercepts_snapshot_chain(name, args, &entries).await
}

/// Run only the tool conditional-execution guardrail chain.
///
/// This evaluates whether a tool call should be allowed to proceed without
/// invoking request intercepts or execution. Each evaluated guardrail emits an
/// automatic guardrail scope start/end pair for observability.
///
/// # Parameters
/// - `name`: Tool name used when resolving the guardrail chain.
/// - `args`: Raw tool arguments to validate.
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
/// rejection result without starting a tool span. Guardrail scopes are still
/// emitted for the conditional checks themselves.
pub async fn tool_conditional_execution(name: &str, args: &Json) -> Result<()> {
    ensure_runtime_owner()?;
    let (entries, subscribers, parent_uuid) = {
        let scope_stack = current_scope_stack();
        let (scope_locals, scope_subscribers) = {
            let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
            (
                scope_guard.snapshot_scope_local_registries(|registries| {
                    &registries.tool_conditional_execution_guardrails
                }),
                scope_guard.collect_scope_local_subscribers(),
            )
        };
        let scope_local_refs = scope_locals.iter().collect::<Vec<_>>();
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?
            .registry_snapshot(&[
                RuntimeRegistrationKind::ToolConditionalExecutionGuardrail,
                RuntimeRegistrationKind::Subscriber,
            ]);
        let entries = state.tool_conditional_execution_entries(&scope_local_refs);
        let subscribers = state.collect_event_subscribers(&scope_subscribers);
        (entries, subscribers, resolve_parent_uuid(None))
    };
    if let Some(error) = NemoRelayContextState::tool_conditional_execution_snapshot_chain(
        name,
        args,
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
#[path = "../../tests/unit/tool_api_tests.rs"]
mod tests;
