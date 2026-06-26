// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;

use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::ToolExecutionNextFn;
use crate::api::runtime::current_scope_stack;
use crate::api::runtime::global_context;
use crate::api::scope::event;
use crate::api::scope::{EmitMarkEventParams, ScopeHandle};
use crate::api::shared::{
    ensure_runtime_owner, metadata_with_otel_status, resolve_parent_uuid,
    snapshot_event_subscribers,
};
use crate::error::{FlowError, Result};
use crate::json::Json;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;
use uuid::Uuid;

pub use nemo_relay_types::api::tool::ToolAttributes;

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
    let sanitized_args = NemoRelayContextState::tool_sanitize_request_snapshot_chain(
        params.name,
        params.args,
        &entries,
    );
    let (handle, event) = {
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
        (handle, event)
    };
    NemoRelayContextState::emit_event(&event, &subscribers);
    Ok(handle)
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
    ensure_runtime_owner()?;
    let (entries, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.tool_sanitize_response_guardrails
        });
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let entries = state.tool_sanitize_response_entries(&scope_locals);
        (entries, subscribers)
    };
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
    NemoRelayContextState::emit_event(&event, &subscribers);
    Ok(())
}

fn emit_tool_end_without_output(handle: &ToolHandle, metadata: Option<Json>) -> Result<()> {
    ensure_runtime_owner()?;
    let (event, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let event = state.end_tool_handle(handle, handle.data.clone(), metadata);
        (event, subscribers)
    };
    NemoRelayContextState::emit_event(&event, &subscribers);
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

    let handle = tool_call(
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
        Ok(result) => {
            let end_metadata = metadata_with_otel_status(metadata, "OK", None);
            tool_call_end(
                ToolCallEndParams::builder()
                    .handle(&handle)
                    .result(result.clone())
                    .data_opt(data)
                    .metadata_opt(end_metadata)
                    .build(),
            )?;
            Ok(result)
        }
        Err(error) => {
            let end_metadata =
                metadata_with_otel_status(metadata, "ERROR", Some(error.to_string()));
            let _ = emit_tool_end_without_output(&handle, end_metadata);
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
