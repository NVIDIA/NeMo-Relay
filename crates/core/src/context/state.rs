// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use uuid::Uuid;

use crate::codec::request::AnnotatedLLMRequest;
use crate::codec::response::AnnotatedLLMResponse;
use crate::context::callbacks::{
    EventSubscriberFn, LlmConditionalFn, LlmExecutionFn, LlmExecutionNextFn, LlmRequestInterceptFn,
    LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionFn, LlmStreamExecutionNextFn,
    ToolConditionalFn, ToolExecutionFn, ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
};
use crate::context::registries::{
    merge_execution_intercept_callables, merge_guardrail_entries, merge_intercept_entries,
};
use crate::json::{Json, merge_json};
use crate::registry::SortedRegistry;
use crate::types::event::Event;
use crate::types::llm::{LLMAttributes, LLMHandle, LLMRequest};
use crate::types::middleware::{ExecutionIntercept, GuardrailEntry, Intercept};
use crate::types::scope::{ScopeAttributes, ScopeHandle, ScopeType};
use crate::types::tool::{ToolAttributes, ToolHandle};

pub struct NemoFlowContextState {
    pub tool_sanitize_request_guardrails: SortedRegistry<GuardrailEntry<ToolSanitizeFn>>,
    pub tool_sanitize_response_guardrails: SortedRegistry<GuardrailEntry<ToolSanitizeFn>>,
    pub tool_conditional_execution_guardrails: SortedRegistry<GuardrailEntry<ToolConditionalFn>>,
    pub tool_request_intercepts: SortedRegistry<Intercept<ToolInterceptFn>>,
    pub tool_execution_intercepts: SortedRegistry<ExecutionIntercept<ToolExecutionFn>>,
    pub llm_sanitize_request_guardrails: SortedRegistry<GuardrailEntry<LlmSanitizeRequestFn>>,
    pub llm_sanitize_response_guardrails: SortedRegistry<GuardrailEntry<LlmSanitizeResponseFn>>,
    pub llm_conditional_execution_guardrails: SortedRegistry<GuardrailEntry<LlmConditionalFn>>,
    pub llm_request_intercepts: SortedRegistry<Intercept<LlmRequestInterceptFn>>,
    pub llm_execution_intercepts: SortedRegistry<ExecutionIntercept<LlmExecutionFn>>,
    pub llm_stream_execution_intercepts: SortedRegistry<ExecutionIntercept<LlmStreamExecutionFn>>,
    pub event_subscribers: HashMap<String, EventSubscriberFn>,
    pub extensions: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl NemoFlowContextState {
    pub fn new() -> Self {
        Self {
            tool_sanitize_request_guardrails: SortedRegistry::new(|entry| entry.priority),
            tool_sanitize_response_guardrails: SortedRegistry::new(|entry| entry.priority),
            tool_conditional_execution_guardrails: SortedRegistry::new(|entry| entry.priority),
            tool_request_intercepts: SortedRegistry::new(|entry| entry.priority),
            tool_execution_intercepts: SortedRegistry::new(|entry| entry.priority),
            llm_sanitize_request_guardrails: SortedRegistry::new(|entry| entry.priority),
            llm_sanitize_response_guardrails: SortedRegistry::new(|entry| entry.priority),
            llm_conditional_execution_guardrails: SortedRegistry::new(|entry| entry.priority),
            llm_request_intercepts: SortedRegistry::new(|entry| entry.priority),
            llm_execution_intercepts: SortedRegistry::new(|entry| entry.priority),
            llm_stream_execution_intercepts: SortedRegistry::new(|entry| entry.priority),
            event_subscribers: HashMap::new(),
            extensions: HashMap::new(),
        }
    }

    pub fn set_extension<T: Any + Send + Sync>(&mut self, key: impl Into<String>, value: T) {
        self.extensions.insert(key.into(), Box::new(value));
    }

    pub fn get_extension<T: Any + Send + Sync>(&self, key: &str) -> Option<&T> {
        self.extensions
            .get(key)
            .and_then(|value| value.downcast_ref::<T>())
    }

    pub fn get_extension_mut<T: Any + Send + Sync>(&mut self, key: &str) -> Option<&mut T> {
        self.extensions
            .get_mut(key)
            .and_then(|value| value.downcast_mut::<T>())
    }

    pub fn remove_extension(&mut self, key: &str) -> bool {
        self.extensions.remove(key).is_some()
    }

    pub fn collect_event_subscribers(
        &self,
        scope_local_subscribers: &[EventSubscriberFn],
    ) -> Vec<EventSubscriberFn> {
        let mut subscribers =
            Vec::with_capacity(self.event_subscribers.len() + scope_local_subscribers.len());
        subscribers.extend(self.event_subscribers.values().cloned());
        subscribers.extend(scope_local_subscribers.iter().cloned());
        subscribers
    }

    pub fn emit_event(event: &Event, subscribers: &[EventSubscriberFn]) {
        for subscriber in subscribers {
            subscriber(event);
        }
    }

    pub fn create_event(
        &self,
        name: &str,
        parent_uuid: Option<Uuid>,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> Event {
        Event::mark(parent_uuid, Uuid::now_v7(), name, data, metadata)
    }

    pub fn create_scope_handle(
        &self,
        name: &str,
        parent_uuid: Option<Uuid>,
        scope_type: ScopeType,
        attributes: ScopeAttributes,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> ScopeHandle {
        ScopeHandle::new(
            name.to_string(),
            scope_type,
            attributes,
            parent_uuid,
            data,
            metadata,
        )
    }

    pub fn build_scope_start_event(&self, handle: &ScopeHandle) -> Event {
        Event::scope_start(
            handle.parent_uuid,
            handle.uuid,
            handle.name.clone(),
            handle.data.clone(),
            handle.metadata.clone(),
            handle.attributes,
            handle.scope_type,
        )
    }

    pub fn end_scope_handle(&self, handle: &ScopeHandle) -> Event {
        Event::scope_end(
            handle.parent_uuid,
            handle.uuid,
            handle.name.clone(),
            handle.data.clone(),
            handle.metadata.clone(),
            handle.attributes,
            handle.scope_type,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_tool_handle(
        &self,
        name: &str,
        parent_uuid: Option<Uuid>,
        attributes: ToolAttributes,
        data: Option<Json>,
        metadata: Option<Json>,
        tool_call_id: Option<String>,
    ) -> ToolHandle {
        let mut handle = ToolHandle::new(name.to_string(), attributes, parent_uuid, data, metadata);
        handle.tool_call_id = tool_call_id;
        handle
    }

    pub fn build_tool_start_event(&self, handle: &ToolHandle, input: Option<Json>) -> Event {
        Event::tool_start(
            handle.parent_uuid,
            handle.uuid,
            handle.name.clone(),
            handle.data.clone(),
            handle.metadata.clone(),
            handle.attributes,
            input,
            handle.tool_call_id.clone(),
        )
    }

    pub fn end_tool_handle(
        &self,
        handle: &ToolHandle,
        data: Option<Json>,
        metadata: Option<Json>,
        output: Option<Json>,
    ) -> Event {
        Event::tool_end(
            handle.parent_uuid,
            handle.uuid,
            handle.name.clone(),
            merge_json(handle.data.clone(), data),
            merge_json(handle.metadata.clone(), metadata),
            handle.attributes,
            output,
            handle.tool_call_id.clone(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_llm_handle(
        &self,
        name: &str,
        parent_uuid: Option<Uuid>,
        attributes: LLMAttributes,
        data: Option<Json>,
        metadata: Option<Json>,
        model_name: Option<String>,
    ) -> LLMHandle {
        let mut handle = LLMHandle::new(name.to_string(), attributes, parent_uuid, data, metadata);
        handle.model_name = model_name;
        handle
    }

    pub fn build_llm_start_event(
        &self,
        handle: &LLMHandle,
        input: Option<Json>,
        annotated_request: Option<Arc<AnnotatedLLMRequest>>,
    ) -> Event {
        Event::llm_start(
            handle.parent_uuid,
            handle.uuid,
            handle.name.clone(),
            handle.data.clone(),
            handle.metadata.clone(),
            handle.attributes,
            input,
            handle.model_name.clone(),
            annotated_request,
        )
    }

    pub fn end_llm_handle(
        &self,
        handle: &LLMHandle,
        data: Option<Json>,
        metadata: Option<Json>,
        output: Option<Json>,
        annotated_response: Option<Arc<AnnotatedLLMResponse>>,
    ) -> Event {
        Event::llm_end(
            handle.parent_uuid,
            handle.uuid,
            handle.name.clone(),
            merge_json(handle.data.clone(), data),
            merge_json(handle.metadata.clone(), metadata),
            handle.attributes,
            output,
            handle.model_name.clone(),
            annotated_response,
        )
    }

    pub fn tool_sanitize_request_chain(
        &self,
        name: &str,
        args: Json,
        scope_locals: &[&SortedRegistry<GuardrailEntry<ToolSanitizeFn>>],
    ) -> Json {
        let entries = merge_guardrail_entries(&self.tool_sanitize_request_guardrails, scope_locals);
        let mut value = args;
        for entry in entries {
            value = (entry.guardrail)(name, value);
        }
        value
    }

    pub fn tool_sanitize_response_chain(
        &self,
        name: &str,
        result: Json,
        scope_locals: &[&SortedRegistry<GuardrailEntry<ToolSanitizeFn>>],
    ) -> Json {
        let entries =
            merge_guardrail_entries(&self.tool_sanitize_response_guardrails, scope_locals);
        let mut value = result;
        for entry in entries {
            value = (entry.guardrail)(name, value);
        }
        value
    }

    pub fn tool_conditional_execution_chain(
        &self,
        name: &str,
        args: &Json,
        scope_locals: &[&SortedRegistry<GuardrailEntry<ToolConditionalFn>>],
    ) -> crate::error::Result<Option<String>> {
        let entries =
            merge_guardrail_entries(&self.tool_conditional_execution_guardrails, scope_locals);
        for entry in entries {
            if let Some(error) = (entry.guardrail)(name, args)? {
                return Ok(Some(error));
            }
        }
        Ok(None)
    }

    pub fn tool_request_intercepts_chain(
        &self,
        name: &str,
        args: Json,
        scope_locals: &[&SortedRegistry<Intercept<ToolInterceptFn>>],
    ) -> crate::error::Result<Json> {
        let entries = merge_intercept_entries(&self.tool_request_intercepts, scope_locals);
        let mut value = args;
        for entry in entries {
            value = (entry.callable)(name, value)?;
            if entry.break_chain {
                break;
            }
        }
        Ok(value)
    }

    pub fn tool_build_execution_chain(
        &self,
        name: &str,
        default_fn: ToolExecutionNextFn,
        scope_locals: &[&SortedRegistry<ExecutionIntercept<ToolExecutionFn>>],
    ) -> ToolExecutionNextFn {
        let matching =
            merge_execution_intercept_callables(&self.tool_execution_intercepts, scope_locals);
        let mut next = default_fn;
        let name = name.to_string();
        for (callable, _) in matching.into_iter().rev() {
            let current_next = next.clone();
            let current_name = name.clone();
            next = Arc::new(move |args| callable(&current_name, args, current_next.clone()));
        }
        next
    }

    pub fn llm_sanitize_request_chain(
        &self,
        request: LLMRequest,
        scope_locals: &[&SortedRegistry<GuardrailEntry<LlmSanitizeRequestFn>>],
    ) -> LLMRequest {
        let entries = merge_guardrail_entries(&self.llm_sanitize_request_guardrails, scope_locals);
        let mut value = request;
        for entry in entries {
            value = (entry.guardrail)(value);
        }
        value
    }

    pub fn llm_sanitize_response_chain(
        &self,
        response: Json,
        scope_locals: &[&SortedRegistry<GuardrailEntry<LlmSanitizeResponseFn>>],
    ) -> Json {
        let entries = merge_guardrail_entries(&self.llm_sanitize_response_guardrails, scope_locals);
        let mut value = response;
        for entry in entries {
            value = (entry.guardrail)(value);
        }
        value
    }

    pub fn llm_conditional_execution_chain(
        &self,
        request: &LLMRequest,
        scope_locals: &[&SortedRegistry<GuardrailEntry<LlmConditionalFn>>],
    ) -> crate::error::Result<Option<String>> {
        let entries =
            merge_guardrail_entries(&self.llm_conditional_execution_guardrails, scope_locals);
        for entry in entries {
            if let Some(error) = (entry.guardrail)(request)? {
                return Ok(Some(error));
            }
        }
        Ok(None)
    }

    pub fn llm_request_intercepts_chain(
        &self,
        name: &str,
        request: LLMRequest,
        annotated: Option<AnnotatedLLMRequest>,
        scope_locals: &[&SortedRegistry<Intercept<LlmRequestInterceptFn>>],
    ) -> crate::error::Result<(LLMRequest, Option<AnnotatedLLMRequest>)> {
        let entries = merge_intercept_entries(&self.llm_request_intercepts, scope_locals);
        let mut request_value = request;
        let mut annotated_value = annotated;
        for entry in entries {
            let (new_request, new_annotated) =
                (entry.callable)(name, request_value, annotated_value)?;
            request_value = new_request;
            annotated_value = new_annotated;
            if entry.break_chain {
                break;
            }
        }
        Ok((request_value, annotated_value))
    }

    pub fn llm_build_execution_chain(
        &self,
        name: &str,
        default_fn: LlmExecutionNextFn,
        scope_locals: &[&SortedRegistry<ExecutionIntercept<LlmExecutionFn>>],
    ) -> LlmExecutionNextFn {
        let matching =
            merge_execution_intercept_callables(&self.llm_execution_intercepts, scope_locals);
        let mut next = default_fn;
        let name = name.to_string();
        for (callable, _) in matching.into_iter().rev() {
            let current_next = next.clone();
            let current_name = name.clone();
            next = Arc::new(move |request| callable(&current_name, request, current_next.clone()));
        }
        next
    }

    #[allow(clippy::type_complexity)]
    pub fn llm_stream_build_execution_chain(
        &self,
        name: &str,
        default_fn: LlmStreamExecutionNextFn,
        scope_locals: &[&SortedRegistry<ExecutionIntercept<LlmStreamExecutionFn>>],
    ) -> LlmStreamExecutionNextFn {
        let matching = merge_execution_intercept_callables(
            &self.llm_stream_execution_intercepts,
            scope_locals,
        );
        let mut next = default_fn;
        let name = name.to_string();
        for (callable, _) in matching.into_iter().rev() {
            let current_next = next.clone();
            let current_name = name.clone();
            next = Arc::new(move |request| callable(&current_name, request, current_next.clone()));
        }
        next
    }
}

impl Default for NemoFlowContextState {
    fn default() -> Self {
        Self::new()
    }
}
