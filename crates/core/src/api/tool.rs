// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;

use crate::api::event::{BaseEvent, Event, MarkEvent, PendingMarkSpec};
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::current_scope_stack;
use crate::api::runtime::global_context;
use crate::api::runtime::{EventSubscriberFn, ToolExecutionNextFn};
use crate::api::scope::event;
use crate::api::scope::{EmitMarkEventParams, ScopeHandle};
use crate::api::shared::{
    ensure_runtime_owner, metadata_with_otel_status, resolve_parent_uuid, sanitize_event,
    snapshot_event_subscribers,
};
use crate::api::skill_load;
use crate::error::{FlowError, Result};
use crate::json::Json;
use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;
use uuid::Uuid;

pub use nemo_relay_types::api::tool::{ToolAttributes, ToolExecutionInterceptOutcome};

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
}

/// Builder parameters for [`tool_call_end`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct ToolCallEndParams<'a> {
    /// Tool handle to close.
    pub handle: &'a ToolHandle,
    /// Raw tool result associated with the end event.
    pub result: Json,
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
/// This emits a tool-start event after applying sanitize-request guardrails to
/// the payload recorded for observability.
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
/// A [`Result`] containing the created [`ToolHandle`].
///
/// # Errors
/// Returns an error when the runtime owner check fails or when internal state
/// cannot be read safely.
///
/// # Notes
/// Sanitize-request guardrails affect only the emitted start-event payload, not
/// the caller-owned `args` value.
pub fn tool_call(params: ToolCallParams<'_>) -> Result<ToolHandle> {
    let (handle, _) = tool_call_with_subscriber_snapshot(params)?;
    Ok(handle)
}

fn tool_call_with_subscriber_snapshot(
    params: ToolCallParams<'_>,
) -> Result<(ToolHandle, Vec<EventSubscriberFn>)> {
    ensure_runtime_owner()?;
    let parent_uuid = resolve_parent_uuid(params.parent);
    let (entries, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.tool_sanitize_request_guardrails
        });
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let entries = state.tool_sanitize_request_entries(&scope_locals);
        (entries, subscribers)
    };
    let handled_skill_loads = params
        .metadata
        .as_ref()
        .and_then(Json::as_object)
        .and_then(|metadata| metadata.get(skill_load::HANDLED_METADATA_KEY))
        .and_then(Json::as_bool)
        .is_some_and(|handled| handled);
    let skill_loads = if handled_skill_loads {
        Vec::new()
    } else if let Some(skill_loads) = skill_load::precomputed(params.metadata.as_ref()) {
        skill_loads
    } else {
        skill_load::detect(params.name, &params.args)
    };
    let sanitized_args = NemoRelayContextState::tool_sanitize_request_snapshot_chain(
        params.name,
        params.args,
        &entries,
    );
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
        let event = state.build_tool_start_event(&handle, Some(sanitized_args));
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
                            "skill_load_source": skill_load.source.as_str(),
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
    let marks = marks
        .into_iter()
        .filter_map(sanitize_event)
        .collect::<Vec<_>>();
    if let Some(event) = sanitize_event(event) {
        NemoRelayContextState::emit_event(&event, &subscribers);
    }
    for mark in marks {
        NemoRelayContextState::emit_event(&mark, &subscribers);
    }
    Ok((handle, subscribers))
}

/// Finish a manual tool lifecycle span.
///
/// This emits a tool-end event for a handle previously returned by
/// [`tool_call`].
///
/// # Parameters
/// - `handle`: Tool handle to close.
/// - `result`: Raw tool result associated with the end event.
/// - `data`: Optional application payload retained for compatibility. The
///   emitted end event data is the sanitized `result` unless it sanitizes to
///   JSON null, in which case this payload is used.
/// - `metadata`: Optional JSON metadata recorded on the end event.
/// - `timestamp`: Optional timestamp recorded on the emitted end event. When
///   `None`, the runtime uses the current UTC time, or one microsecond after
///   the handle start time if the current time is not later.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when the end event has been emitted.
///
/// # Errors
/// Returns an error when the runtime owner check fails or when internal state
/// cannot be read safely.
///
/// # Notes
/// Sanitize-response guardrails affect only the emitted end-event payload, not
/// the caller-owned `result` value.
pub fn tool_call_end(params: ToolCallEndParams<'_>) -> Result<()> {
    tool_call_end_with_pending_marks(params, Vec::new(), None)
}

fn tool_call_end_with_pending_marks(
    params: ToolCallEndParams<'_>,
    pending_marks: Vec<PendingMarkSpec>,
    lifecycle_subscribers: Option<&[EventSubscriberFn]>,
) -> Result<()> {
    ensure_runtime_owner()?;
    let (entries, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.tool_sanitize_response_guardrails
        });
        let subscribers = if lifecycle_subscribers.is_some() {
            Vec::new()
        } else {
            snapshot_event_subscribers(scope_guard.collect_scope_local_subscribers())?
        };
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let entries = state.tool_sanitize_response_entries(&scope_locals);
        (entries, subscribers)
    };
    let subscribers = lifecycle_subscribers.unwrap_or(&subscribers);
    let sanitized_result = NemoRelayContextState::tool_sanitize_response_snapshot_chain(
        &params.handle.name,
        params.result,
        &entries,
    );
    let data = if sanitized_result.is_null() {
        params.data
    } else {
        Some(sanitized_result)
    };
    let event = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.build_tool_end_event(
            EndToolHandleParams::builder()
                .handle(params.handle)
                .data_opt(data)
                .metadata_opt(params.metadata)
                .timestamp_opt(params.timestamp)
                .build(),
        )
    };
    let marks = pending_marks
        .into_iter()
        .enumerate()
        .map(|(index, mark)| {
            let timestamp = *event.timestamp()
                + TimeDelta::microseconds(i64::try_from(index).unwrap_or_default() + 1);
            Event::Mark(MarkEvent::new(
                BaseEvent::builder()
                    .name(mark.name)
                    .parent_uuid(params.handle.uuid)
                    .timestamp(timestamp)
                    .data_opt(mark.data)
                    .metadata_opt(mark.metadata)
                    .build(),
                mark.category,
                mark.category_profile,
            ))
        })
        .filter_map(sanitize_event)
        .collect::<Vec<_>>();
    if let Some(event) = sanitize_event(event) {
        NemoRelayContextState::emit_event(&event, subscribers);
    }
    for mark in marks {
        NemoRelayContextState::emit_event(&mark, subscribers);
    }
    Ok(())
}

fn emit_tool_end_without_output(
    handle: &ToolHandle,
    metadata: Option<Json>,
    lifecycle_subscribers: &[EventSubscriberFn],
) -> Result<()> {
    ensure_runtime_owner()?;
    let event = {
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.end_tool_handle(handle, handle.data.clone(), metadata)
    };
    if let Some(event) = sanitize_event(event) {
        NemoRelayContextState::emit_event(&event, lifecycle_subscribers);
    }
    Ok(())
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
///
/// # Returns
/// A [`Result`] containing the raw tool result returned by the callback or an
/// execution intercept.
///
/// # Errors
/// Returns [`FlowError::GuardrailRejected`] when conditional-execution
/// guardrails block the call, or any error raised by request intercepts,
/// execution intercepts, or the callback itself.
///
/// # Notes
/// When execution fails after the start event has been emitted, the runtime
/// still emits a tool-end event without an output payload.
pub async fn tool_call_execute(params: ToolCallExecuteParams) -> Result<Json> {
    let ToolCallExecuteParams {
        name,
        args,
        func,
        parent,
        attributes,
        data,
        metadata,
    } = params;
    ensure_runtime_owner()?;
    {
        let (entries, subscribers, parent_uuid, guardrail_metadata) = {
            let scope_stack = current_scope_stack();
            let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
            let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
                &registries.tool_conditional_execution_guardrails
            });
            let scope_subscribers = scope_guard.collect_scope_local_subscribers();
            let context = global_context();
            let state = context
                .read()
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            let entries = state.tool_conditional_execution_entries(&scope_locals);
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
        )? {
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
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard
            .collect_scope_local_registries(|registries| &registries.tool_request_intercepts);
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.tool_request_intercept_entries(&scope_locals)
    };
    let intercepted_args = NemoRelayContextState::tool_request_intercepts_snapshot_chain(
        &name,
        args,
        &intercept_entries,
    )?;

    let (handle, lifecycle_subscribers) = tool_call_with_subscriber_snapshot(
        ToolCallParams::builder()
            .name(name.as_str())
            .args(intercepted_args.clone())
            .parent_opt(parent.as_ref())
            .attributes(attributes)
            .data_opt(data.clone())
            .metadata_opt(metadata.clone())
            .build(),
    )?;

    let execution = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard
            .collect_scope_local_registries(|registries| &registries.tool_execution_intercepts);
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.tool_build_execution_chain(&name, func, &scope_locals)
    };

    match execution(intercepted_args).await {
        Ok(outcome) => {
            let ToolExecutionInterceptOutcome {
                result,
                pending_marks,
            } = outcome;
            let end_metadata = metadata_with_otel_status(metadata, "OK", None);
            tool_call_end_with_pending_marks(
                ToolCallEndParams::builder()
                    .handle(&handle)
                    .result(result.clone())
                    .data_opt(data)
                    .metadata_opt(end_metadata)
                    .build(),
                pending_marks,
                Some(&lifecycle_subscribers),
            )?;
            Ok(result)
        }
        Err(error) => {
            let end_metadata =
                metadata_with_otel_status(metadata, "ERROR", Some(error.to_string()));
            let _ = emit_tool_end_without_output(&handle, end_metadata, &lifecycle_subscribers);
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
pub fn tool_request_intercepts(name: &str, args: Json) -> Result<Json> {
    ensure_runtime_owner()?;
    let entries = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard
            .collect_scope_local_registries(|registries| &registries.tool_request_intercepts);
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.tool_request_intercept_entries(&scope_locals)
    };
    NemoRelayContextState::tool_request_intercepts_snapshot_chain(name, args, &entries)
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
pub fn tool_conditional_execution(name: &str, args: &Json) -> Result<()> {
    ensure_runtime_owner()?;
    let (entries, subscribers, parent_uuid) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.tool_conditional_execution_guardrails
        });
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let entries = state.tool_conditional_execution_entries(&scope_locals);
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
    )? {
        return Err(FlowError::GuardrailRejected(error));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/tool_api_tests.rs"]
mod tests;
