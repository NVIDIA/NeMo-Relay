// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use bitflags::bitflags;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use typed_builder::TypedBuilder;
use uuid::Uuid;

use crate::api::runtime::NemoFlowContextState;
use crate::api::runtime::current_scope_stack;
use crate::api::runtime::global_context;
use crate::api::runtime::{
    LlmCollectorFn, LlmExecutionNextFn, LlmFinalizerFn, LlmJsonStream, LlmStreamExecutionNextFn,
};
use crate::api::scope::event;
use crate::api::scope::{EmitMarkEventParams, ScopeHandle};
use crate::api::shared::{
    ensure_runtime_owner, resolve_parent_uuid, run_request_intercepts_with_codec,
    snapshot_event_subscribers,
};
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::response::AnnotatedLlmResponse;
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::error::{FlowError, Result};
use crate::json::Json;
use crate::stream::LlmStreamWrapper;

bitflags! {
    /// Bitflags that modify LLM-call behavior and observability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct LlmAttributes: u32 {
        /// Marks the request as stateful from the runtime's perspective.
        const STATEFUL = 0b01;
        /// Marks the request as streaming.
        const STREAMING = 0b10;
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
}

/// JSON-shaped LLM request payload passed through the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    /// Provider-specific request headers.
    pub headers: serde_json::Map<String, Json>,
    /// Provider-specific request body.
    pub content: Json,
}

/// Builder parameters for [`NemoFlowContextState::create_llm_handle`].
#[derive(Debug, Clone, TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct CreateLlmHandleParams<'a> {
    /// Logical provider or model family name.
    pub name: &'a str,
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

/// Builder parameters for [`NemoFlowContextState::build_llm_end_event`].
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

fn emit_llm_start(
    handle: &LlmHandle,
    request: &LlmRequest,
    annotated_request: Option<Arc<AnnotatedLlmRequest>>,
) -> Result<()> {
    ensure_runtime_owner()?;
    let (event, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.llm_sanitize_request_guardrails
        });
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;

        let sanitized_request = state.llm_sanitize_request_chain(request.clone(), &scope_locals);
        let input = serde_json::to_value(&sanitized_request).unwrap_or(Json::Null);
        let event = state.build_llm_start_event(handle, Some(input), annotated_request);
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

/// Start a manual LLM lifecycle span.
///
/// This emits an LLM-start event after applying sanitize-request guardrails to
/// the payload recorded for observability.
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
/// A [`Result`] containing the created [`LlmHandle`].
///
/// # Errors
/// Returns an error when the runtime owner check fails or when internal state
/// cannot be read safely.
///
/// # Notes
/// Sanitize-request guardrails affect only the emitted start-event payload, not
/// the caller-owned [`LlmRequest`].
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
    emit_llm_start(&handle, params.request, params.annotated_request)?;
    Ok(handle)
}

/// Finish a manual LLM lifecycle span.
///
/// This emits an LLM-end event for a handle previously returned by
/// [`llm_call`].
///
/// # Parameters
/// - `handle`: LLM handle to close.
/// - `response`: Raw provider response associated with the end event.
/// - `data`: Optional application payload retained for compatibility. The
///   emitted end event data is the sanitized `response` unless it sanitizes to
///   JSON null, in which case this payload is used.
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
/// A [`Result`] that is `Ok(())` when the end event has been emitted.
///
/// # Errors
/// Returns an error when the runtime owner check fails, internal state cannot be
/// read safely, or response codec decoding fails.
///
/// # Notes
/// Sanitize-response guardrails affect only the emitted end-event payload, not
/// the caller-owned `response` value.
pub fn llm_call_end(params: LlmCallEndParams<'_>) -> Result<()> {
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
    let mut decode_error = None;
    let (event, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.llm_sanitize_response_guardrails
        });
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;

        let sanitized_response = state.llm_sanitize_response_chain(response, &scope_locals);
        let data = if sanitized_response.is_null() {
            data
        } else {
            Some(sanitized_response)
        };
        let annotated_response = match annotated_response {
            Some(annotated_response) => Some(annotated_response),
            None => match (response_codec.as_ref(), data.as_ref()) {
                (Some(codec), Some(response)) => match codec.decode_response(response) {
                    Ok(decoded) => Some(Arc::new(decoded)),
                    Err(error) => {
                        decode_error = Some(error);
                        None
                    }
                },
                _ => None,
            },
        };
        let event = state.build_llm_end_event(
            EndLlmHandleParams::builder()
                .handle(handle)
                .data_opt(data)
                .metadata_opt(metadata)
                .annotated_response_opt(annotated_response)
                .timestamp_opt(timestamp)
                .build(),
        );
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    if let Some(error) = decode_error {
        Err(error)
    } else {
        Ok(())
    }
}

fn emit_llm_end_without_output(handle: &LlmHandle, metadata: Option<Json>) -> Result<()> {
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
        let event = state.end_llm_handle(handle, handle.data.clone(), metadata, None);
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

/// Execute an LLM call through the managed middleware pipeline.
///
/// This runs conditional-execution guardrails, request intercepts,
/// sanitize-request guardrails, execution intercepts, the provider callback,
/// and sanitize-response guardrails in the runtime-defined order.
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
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.llm_conditional_execution_guardrails
        });
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        if let Some(error) = state.llm_conditional_execution_chain(&request, &scope_locals)? {
            drop(state);
            drop(scope_guard);
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

    let (intercepted_request, annotated_request) =
        run_request_intercepts_with_codec(&name, request, codec)?;

    let handle = create_llm_handle(
        CreateLlmHandleParams::builder()
            .name(name.as_str())
            .parent_uuid_opt(resolve_parent_uuid(parent.as_ref()))
            .attributes(attributes)
            .data_opt(data.clone())
            .metadata_opt(metadata.clone())
            .model_name_opt(model_name)
            .build(),
    )?;
    emit_llm_start(&handle, &intercepted_request, annotated_request.clone())?;

    let execution = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard
            .collect_scope_local_registries(|registries| &registries.llm_execution_intercepts);
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.llm_build_execution_chain(&name, func, &scope_locals)
    };

    match execution(intercepted_request).await {
        Ok(response) => {
            let annotated_response = response_codec
                .as_ref()
                .and_then(|codec| codec.decode_response(&response).ok())
                .map(Arc::new);
            llm_call_end(
                LlmCallEndParams::builder()
                    .handle(&handle)
                    .response(response.clone())
                    .data_opt(data)
                    .metadata_opt(metadata)
                    .annotated_response_opt(annotated_response)
                    .build(),
            )?;
            Ok(response)
        }
        Err(error) => {
            let _ = emit_llm_end_without_output(&handle, metadata);
            Err(error)
        }
    }
}

/// Execute a streaming LLM call through the managed middleware pipeline.
///
/// This runs the same pre-execution middleware as [`llm_call_execute`] and
/// then wraps the provider stream so chunk callbacks and finalization can emit
/// a single LLM-end event when streaming completes.
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
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.llm_conditional_execution_guardrails
        });
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        if let Some(error) = state.llm_conditional_execution_chain(&request, &scope_locals)? {
            drop(state);
            drop(scope_guard);
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

    let (intercepted_request, annotated_request) =
        run_request_intercepts_with_codec(&name, request, codec)?;

    let handle = create_llm_handle(
        CreateLlmHandleParams::builder()
            .name(name.as_str())
            .parent_uuid_opt(resolve_parent_uuid(parent.as_ref()))
            .attributes(attributes)
            .data_opt(data.clone())
            .metadata_opt(metadata.clone())
            .model_name_opt(model_name)
            .build(),
    )?;
    emit_llm_start(&handle, &intercepted_request, annotated_request)?;

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
        state.llm_stream_build_execution_chain(&name, func, &scope_locals)
    };

    match execution(intercepted_request).await {
        Ok(raw_stream) => {
            let wrapper = LlmStreamWrapper::new(
                raw_stream,
                handle,
                collector,
                finalizer,
                data,
                metadata,
                response_codec,
            );
            Ok(Box::pin(wrapper) as LlmJsonStream)
        }
        Err(error) => {
            let _ = emit_llm_end_without_output(&handle, metadata);
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
pub fn llm_request_intercepts(name: &str, request: LlmRequest) -> Result<LlmRequest> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
    let scope_locals =
        scope_guard.collect_scope_local_registries(|registries| &registries.llm_request_intercepts);
    let context = global_context();
    let state = context
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let (request, _) = state.llm_request_intercepts_chain(name, request, None, &scope_locals)?;
    Ok(request)
}

/// Run only the LLM conditional-execution guardrail chain.
///
/// This evaluates whether an LLM call should be allowed to proceed without
/// emitting lifecycle events or invoking request intercepts or execution.
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
/// rejection result without starting an LLM span.
pub fn llm_conditional_execution(request: &LlmRequest) -> Result<()> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
    let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
        &registries.llm_conditional_execution_guardrails
    });
    let context = global_context();
    let state = context
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    if let Some(error) = state.llm_conditional_execution_chain(request, &scope_locals)? {
        return Err(FlowError::GuardrailRejected(error));
    }
    Ok(())
}
