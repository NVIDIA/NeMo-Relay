// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;
use tokio_stream::Stream;

use crate::api::scope::event;
use crate::api::shared::{
    ensure_runtime_owner, resolve_parent_uuid, run_request_intercepts_with_codec,
    snapshot_event_subscribers,
};
use crate::codec::request::AnnotatedLLMRequest;
use crate::codec::response::AnnotatedLLMResponse;
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::context::callbacks::{LlmExecutionNextFn, LlmStreamExecutionNextFn};
use crate::context::global::global_context;
use crate::context::scope_stack::current_scope_stack;
use crate::context::state::NemoFlowContextState;
use crate::error::{FlowError, Result};
use crate::json::Json;
use crate::stream::LlmStreamWrapper;
use crate::types::llm::{LLMAttributes, LLMHandle, LLMRequest};
use crate::types::scope::ScopeHandle;

#[allow(clippy::too_many_arguments)]
pub fn llm_call(
    name: &str,
    request: &LLMRequest,
    parent: Option<&ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    annotated_request: Option<Arc<AnnotatedLLMRequest>>,
) -> Result<LLMHandle> {
    ensure_runtime_owner()?;
    let parent_uuid = resolve_parent_uuid(parent);
    let (handle, event, subscribers) = {
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
        let handle =
            state.create_llm_handle(name, parent_uuid, attributes, data, metadata, model_name);
        let event = state.build_llm_start_event(&handle, Some(input), annotated_request);
        (handle, event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(handle)
}

pub fn llm_call_end(
    handle: &LLMHandle,
    response: Json,
    data: Option<Json>,
    metadata: Option<Json>,
    annotated_response: Option<Arc<AnnotatedLLMResponse>>,
) -> Result<()> {
    ensure_runtime_owner()?;
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
        let event = state.end_llm_handle(
            handle,
            data,
            metadata,
            Some(sanitized_response),
            annotated_response,
        );
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

fn emit_llm_end_without_output(
    handle: &LLMHandle,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
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
        let event = state.end_llm_handle(handle, data, metadata, None, None);
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn llm_call_execute(
    name: &str,
    request: LLMRequest,
    func: LlmExecutionNextFn,
    parent: Option<ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    codec: Option<Arc<dyn LlmCodec>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
) -> Result<Json> {
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
            let mut rejection_data = data.clone().unwrap_or_else(|| json!({}));
            if let Some(object) = rejection_data.as_object_mut() {
                object.insert("rejected".into(), json!(true));
                object.insert("rejection_reason".into(), json!(&error));
            }
            let _ = event(
                name,
                parent.as_ref(),
                Some(rejection_data),
                metadata.clone(),
            );
            return Err(FlowError::GuardrailRejected(error));
        }
    }

    let (intercepted_request, annotated_request) =
        run_request_intercepts_with_codec(name, request, codec)?;

    let handle = llm_call(
        name,
        &intercepted_request,
        parent.as_ref(),
        attributes,
        data.clone(),
        metadata.clone(),
        model_name,
        annotated_request,
    )?;

    let execution = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard
            .collect_scope_local_registries(|registries| &registries.llm_execution_intercepts);
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.llm_build_execution_chain(name, func, &scope_locals)
    };

    match execution(intercepted_request).await {
        Ok(response) => {
            let annotated_response = response_codec
                .as_ref()
                .and_then(|codec| codec.decode_response(&response).ok())
                .map(Arc::new);
            llm_call_end(
                &handle,
                response.clone(),
                data,
                metadata,
                annotated_response,
            )?;
            Ok(response)
        }
        Err(error) => {
            let _ = emit_llm_end_without_output(&handle, data, metadata);
            Err(error)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn llm_stream_call_execute(
    name: &str,
    request: LLMRequest,
    func: LlmStreamExecutionNextFn,
    collector: Box<dyn FnMut(Json) -> Result<()> + Send>,
    finalizer: Box<dyn FnOnce() -> Json + Send>,
    parent: Option<ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    codec: Option<Arc<dyn LlmCodec>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
) -> Result<Pin<Box<dyn Stream<Item = Result<Json>> + Send>>> {
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
            let mut rejection_data = data.clone().unwrap_or_else(|| json!({}));
            if let Some(object) = rejection_data.as_object_mut() {
                object.insert("rejected".into(), json!(true));
                object.insert("rejection_reason".into(), json!(&error));
            }
            let _ = event(
                name,
                parent.as_ref(),
                Some(rejection_data),
                metadata.clone(),
            );
            return Err(FlowError::GuardrailRejected(error));
        }
    }

    let (intercepted_request, annotated_request) =
        run_request_intercepts_with_codec(name, request, codec)?;

    let handle = llm_call(
        name,
        &intercepted_request,
        parent.as_ref(),
        attributes,
        data.clone(),
        metadata.clone(),
        model_name,
        annotated_request,
    )?;

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
        state.llm_stream_build_execution_chain(name, func, &scope_locals)
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
            Ok(Box::pin(wrapper) as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
        }
        Err(error) => {
            let _ = emit_llm_end_without_output(&handle, data, metadata);
            Err(error)
        }
    }
}

pub fn llm_request_intercepts(name: &str, request: LLMRequest) -> Result<LLMRequest> {
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

pub fn llm_conditional_execution(request: &LLMRequest) -> Result<()> {
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
