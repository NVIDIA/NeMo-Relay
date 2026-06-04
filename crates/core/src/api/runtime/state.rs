// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-global runtime state and middleware-chain builders.
//!
//! [`NemoRelayContextState`] owns the registries and helper methods that power
//! the public scope, tool, and LLM APIs. Advanced integrations can use this
//! type directly to register middleware, attach runtime extensions, and build
//! the resolved callback chains that the higher-level API layer executes.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
    llm_attributes_to_strings, scope_attributes_to_strings, tool_attributes_to_strings,
};
use crate::api::llm::{CreateLlmHandleParams, EndLlmHandleParams};
use crate::api::llm::{LlmHandle, LlmRequest};
use crate::api::registry::{ExecutionIntercept, Guardrail, Intercept};
use crate::api::runtime::callbacks::{
    EventSubscriberFn, LlmConditionalFn, LlmExecutionFn, LlmExecutionNextFn, LlmRequestInterceptFn,
    LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionFn, LlmStreamExecutionNextFn,
    LlmStreamExecutionRegistryRefs, ToolConditionalFn, ToolExecutionFn, ToolExecutionNextFn,
    ToolInterceptFn, ToolSanitizeFn,
};
use crate::api::runtime::subscriber_dispatcher;
use crate::api::scope::{CreateScopeHandleParams, EndScopeHandleParams, ScopeHandle, ScopeType};
use crate::api::tool::ToolHandle;
use crate::api::tool::{CreateToolHandleParams, EndToolHandleParams};
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::response::AnnotatedLlmResponse;
use crate::context::registries::{
    merge_execution_intercept_callables, merge_guardrail_entries, merge_intercept_entries,
};
use crate::json::{Json, merge_json};
use crate::registry::SortedRegistry;
use chrono::{Duration, Utc};
use serde_json::json;
use uuid::Uuid;

/// Process-global runtime state backing middleware and event emission.
///
/// The public API layer stores one shared instance of this type for the
/// process. It contains global middleware registries, lifecycle subscribers,
/// and arbitrary extension slots used by bindings or integrations.
pub struct NemoRelayContextState {
    /// Global tool request sanitizers applied to emitted tool-start payloads.
    pub(crate) tool_sanitize_request_guardrails: SortedRegistry<Guardrail<ToolSanitizeFn>>,
    /// Global tool response sanitizers applied to emitted tool-end payloads.
    pub(crate) tool_sanitize_response_guardrails: SortedRegistry<Guardrail<ToolSanitizeFn>>,
    /// Global tool guardrails that can reject execution before the callback runs.
    pub(crate) tool_conditional_execution_guardrails: SortedRegistry<Guardrail<ToolConditionalFn>>,
    /// Global tool request intercepts that can rewrite arguments before execution.
    pub(crate) tool_request_intercepts: SortedRegistry<Intercept<ToolInterceptFn>>,
    /// Global tool execution intercepts that wrap or replace callback execution.
    pub(crate) tool_execution_intercepts: SortedRegistry<ExecutionIntercept<ToolExecutionFn>>,
    /// Global LLM request sanitizers applied to emitted LLM-start payloads.
    pub(crate) llm_sanitize_request_guardrails: SortedRegistry<Guardrail<LlmSanitizeRequestFn>>,
    /// Global LLM response sanitizers applied to emitted LLM-end payloads.
    pub(crate) llm_sanitize_response_guardrails: SortedRegistry<Guardrail<LlmSanitizeResponseFn>>,
    /// Global LLM guardrails that can reject execution before the provider callback runs.
    pub(crate) llm_conditional_execution_guardrails: SortedRegistry<Guardrail<LlmConditionalFn>>,
    /// Global LLM request intercepts that can rewrite or annotate requests.
    pub(crate) llm_request_intercepts: SortedRegistry<Intercept<LlmRequestInterceptFn>>,
    /// Global non-streaming LLM execution intercepts that wrap callback execution.
    pub(crate) llm_execution_intercepts: SortedRegistry<ExecutionIntercept<LlmExecutionFn>>,
    /// Global streaming LLM execution intercepts that wrap stream-producing callbacks.
    pub(crate) llm_stream_execution_intercepts:
        SortedRegistry<ExecutionIntercept<LlmStreamExecutionFn>>,
    /// Global lifecycle subscribers notified after runtime events are emitted.
    pub(crate) event_subscribers: HashMap<String, EventSubscriberFn>,
    /// Arbitrary binding- or integration-specific runtime extensions.
    pub(crate) extensions: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl NemoRelayContextState {
    /// Create an empty runtime state with no registered middleware.
    ///
    /// # Returns
    /// A [`NemoRelayContextState`] with empty registries, no subscribers, and no
    /// extensions.
    pub fn new() -> Self {
        Self {
            tool_sanitize_request_guardrails: SortedRegistry::new(),
            tool_sanitize_response_guardrails: SortedRegistry::new(),
            tool_conditional_execution_guardrails: SortedRegistry::new(),
            tool_request_intercepts: SortedRegistry::new(),
            tool_execution_intercepts: SortedRegistry::new(),
            llm_sanitize_request_guardrails: SortedRegistry::new(),
            llm_sanitize_response_guardrails: SortedRegistry::new(),
            llm_conditional_execution_guardrails: SortedRegistry::new(),
            llm_request_intercepts: SortedRegistry::new(),
            llm_execution_intercepts: SortedRegistry::new(),
            llm_stream_execution_intercepts: SortedRegistry::new(),
            event_subscribers: HashMap::new(),
            extensions: HashMap::new(),
        }
    }

    /// Store an arbitrary runtime extension under `key`.
    ///
    /// Extensions let bindings or integrations attach shared state to the
    /// process-global runtime without adding new first-class fields.
    ///
    /// # Parameters
    /// - `key`: Stable identifier for the extension slot.
    /// - `value`: Typed extension value to store.
    pub fn set_extension<T: Any + Send + Sync>(&mut self, key: impl Into<String>, value: T) {
        self.extensions.insert(key.into(), Box::new(value));
    }

    /// Borrow a typed runtime extension by key.
    ///
    /// # Parameters
    /// - `key`: Extension slot name.
    ///
    /// # Returns
    /// `Some(&T)` when an extension exists under `key` with the requested type
    /// and `None` otherwise.
    pub fn get_extension<T: Any + Send + Sync>(&self, key: &str) -> Option<&T> {
        self.extensions
            .get(key)
            .and_then(|value| value.downcast_ref::<T>())
    }

    /// Mutably borrow a typed runtime extension by key.
    ///
    /// # Parameters
    /// - `key`: Extension slot name.
    ///
    /// # Returns
    /// `Some(&mut T)` when an extension exists under `key` with the requested
    /// type and `None` otherwise.
    pub fn get_extension_mut<T: Any + Send + Sync>(&mut self, key: &str) -> Option<&mut T> {
        self.extensions
            .get_mut(key)
            .and_then(|value| value.downcast_mut::<T>())
    }

    /// Remove a runtime extension by key.
    ///
    /// # Parameters
    /// - `key`: Extension slot name.
    ///
    /// # Returns
    /// `true` when an extension was removed and `false` when no extension was
    /// stored under `key`.
    pub fn remove_extension(&mut self, key: &str) -> bool {
        self.extensions.remove(key).is_some()
    }

    /// Combine global and scope-local subscribers into one delivery list.
    ///
    /// # Parameters
    /// - `scope_local_subscribers`: Subscribers collected from the active scope
    ///   stack.
    ///
    /// # Returns
    /// A vector containing all global subscribers followed by the provided
    /// scope-local subscribers.
    pub(crate) fn collect_event_subscribers(
        &self,
        scope_local_subscribers: &[EventSubscriberFn],
    ) -> Vec<EventSubscriberFn> {
        let mut subscribers =
            Vec::with_capacity(self.event_subscribers.len() + scope_local_subscribers.len());
        subscribers.extend(self.event_subscribers.values().cloned());
        subscribers.extend(scope_local_subscribers.iter().cloned());
        subscribers
    }

    /// Deliver an event to every subscriber in order.
    ///
    /// # Parameters
    /// - `event`: Fully constructed lifecycle event to deliver.
    /// - `subscribers`: Subscribers that should observe the event.
    pub(crate) fn emit_event(event: &Event, subscribers: &[EventSubscriberFn]) {
        subscriber_dispatcher::dispatch_event(event, subscribers);
    }

    /// Build a standalone mark event.
    ///
    /// # Parameters
    /// - `params`: A pre-built [`MarkEvent`] to wrap in an [`Event`].
    ///
    /// # Returns
    /// A mark [`Event`] containing the provided [`MarkEvent`].
    pub fn create_event(&self, params: MarkEvent) -> Event {
        Event::Mark(params)
    }

    /// Create a new scope handle.
    ///
    /// # Parameters
    /// - `name`: Human-readable scope name.
    /// - `parent_uuid`: Optional parent scope UUID.
    /// - `scope_type`: Semantic category of the scope.
    /// - `attributes`: Scope attribute bitflags.
    /// - `data`: Optional application payload stored on the handle.
    /// - `metadata`: Optional metadata stored on the handle.
    /// - `timestamp`: Optional handle start time. When omitted, the current
    ///   UTC time is used.
    ///
    /// # Returns
    /// A new [`ScopeHandle`] with a fresh UUID.
    pub fn create_scope_handle(&self, params: CreateScopeHandleParams<'_>) -> ScopeHandle {
        ScopeHandle::builder()
            .name(params.name)
            .scope_type(params.scope_type)
            .started_at(params.timestamp.unwrap_or_else(Utc::now))
            .attributes(params.attributes)
            .parent_uuid_opt(params.parent_uuid)
            .data_opt(params.data)
            .metadata_opt(params.metadata)
            .build()
    }

    /// Build a scope-start event from a handle.
    ///
    /// # Parameters
    /// - `handle`: Scope handle to serialize into an event.
    /// - `data`: Optional semantic input payload exported on the start event.
    ///
    /// # Returns
    /// A scope-start [`Event`] derived from the provided handle.
    pub fn build_scope_start_event(&self, handle: &ScopeHandle, data: Option<Json>) -> Event {
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(handle.parent_uuid)
                .uuid(handle.uuid)
                .timestamp(handle.started_at)
                .name(handle.name.as_str())
                .data_opt(data)
                .metadata_opt(handle.metadata.clone())
                .build(),
            ScopeCategory::Start,
            scope_attributes_to_strings(handle.attributes),
            EventCategory::from(handle.scope_type),
            None,
        ))
    }

    /// Build a scope-end event from a handle.
    ///
    /// # Parameters
    /// - `handle`: Scope handle to serialize into an event.
    /// - `data`: Optional data payload returned from the scope.
    /// - `metadata`: Optional metadata payload merged over `handle.metadata`.
    ///
    /// # Returns
    /// A scope-end [`Event`] derived from the provided handle.
    pub fn end_scope_handle(
        &self,
        handle: &ScopeHandle,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> Event {
        self.build_scope_end_event(
            EndScopeHandleParams::builder()
                .handle(handle)
                .data_opt(data)
                .metadata_opt(metadata)
                .build(),
        )
    }

    /// Build a scope-end event from builder parameters.
    ///
    /// The `metadata` payload is merged over the metadata already stored on
    /// the handle.
    ///
    /// # Parameters
    /// - `params`: Scope end-event builder parameters.
    ///
    /// # Returns
    /// A scope-end [`Event`] derived from the provided parameters.
    pub fn build_scope_end_event(&self, params: EndScopeHandleParams<'_>) -> Event {
        let handle = params.handle;
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(handle.parent_uuid)
                .uuid(handle.uuid)
                .timestamp(
                    params
                        .timestamp
                        .unwrap_or_else(|| end_timestamp_after(handle.started_at)),
                )
                .name(handle.name.as_str())
                .data_opt(params.data)
                .metadata_opt(merge_json(handle.metadata.clone(), params.metadata))
                .build(),
            ScopeCategory::End,
            scope_attributes_to_strings(handle.attributes),
            EventCategory::from(handle.scope_type),
            None,
        ))
    }

    /// Create a new tool handle.
    ///
    /// # Parameters
    /// - `name`: Tool name recorded on emitted events.
    /// - `parent_uuid`: Optional parent scope UUID.
    /// - `attributes`: Tool attribute bitflags.
    /// - `data`: Optional application payload stored on the handle.
    /// - `metadata`: Optional metadata stored on the handle.
    /// - `tool_call_id`: Optional provider-specific correlation identifier.
    /// - `timestamp`: Optional handle start time. When omitted, the current
    ///   UTC time is used.
    ///
    /// # Returns
    /// A new [`ToolHandle`] with a fresh UUID.
    pub fn create_tool_handle(&self, params: CreateToolHandleParams<'_>) -> ToolHandle {
        ToolHandle::builder()
            .name(params.name)
            .started_at(params.timestamp.unwrap_or_else(Utc::now))
            .attributes(params.attributes)
            .parent_uuid_opt(params.parent_uuid)
            .data_opt(params.data)
            .metadata_opt(params.metadata)
            .tool_call_id_opt(params.tool_call_id)
            .build()
    }

    /// Build a tool-start event from a handle.
    ///
    /// # Parameters
    /// - `handle`: Tool handle to serialize into an event.
    /// - `data`: Optional tool input payload.
    ///
    /// # Returns
    /// A tool-start [`Event`] derived from the provided handle.
    pub fn build_tool_start_event(&self, handle: &ToolHandle, data: Option<Json>) -> Event {
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(handle.parent_uuid)
                .uuid(handle.uuid)
                .timestamp(handle.started_at)
                .name(handle.name.as_str())
                .data_opt(data)
                .metadata_opt(handle.metadata.clone())
                .build(),
            ScopeCategory::Start,
            tool_attributes_to_strings(handle.attributes),
            EventCategory::tool(),
            Some(
                CategoryProfile::builder()
                    .tool_call_id_opt(handle.tool_call_id.clone())
                    .build(),
            ),
        ))
    }

    /// Build a tool-end event from a handle and optional overrides.
    ///
    /// # Parameters
    /// - `handle`: Tool handle to serialize into an event.
    /// - `data`: Optional end-event data payload.
    /// - `metadata`: Optional metadata payload merged over `handle.metadata`.
    ///
    /// # Returns
    /// A tool-end [`Event`] derived from the provided handle.
    pub fn end_tool_handle(
        &self,
        handle: &ToolHandle,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> Event {
        self.build_tool_end_event(
            EndToolHandleParams::builder()
                .handle(handle)
                .data_opt(data)
                .metadata_opt(metadata)
                .build(),
        )
    }

    /// Build a tool-end event from builder parameters.
    ///
    /// The `metadata` payload is merged over the metadata already stored on
    /// the handle.
    ///
    /// # Parameters
    /// - `params`: Tool end-event builder parameters.
    ///
    /// # Returns
    /// A tool-end [`Event`] derived from the provided parameters.
    pub fn build_tool_end_event(&self, params: EndToolHandleParams<'_>) -> Event {
        let handle = params.handle;
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(handle.parent_uuid)
                .uuid(handle.uuid)
                .timestamp(
                    params
                        .timestamp
                        .unwrap_or_else(|| end_timestamp_after(handle.started_at)),
                )
                .name(handle.name.as_str())
                .data_opt(params.data)
                .metadata_opt(merge_json(handle.metadata.clone(), params.metadata))
                .build(),
            ScopeCategory::End,
            tool_attributes_to_strings(handle.attributes),
            EventCategory::tool(),
            Some(
                CategoryProfile::builder()
                    .tool_call_id_opt(handle.tool_call_id.clone())
                    .build(),
            ),
        ))
    }

    /// Create a new LLM handle.
    ///
    /// # Parameters
    /// - `name`: Logical provider or model family name.
    /// - `parent_uuid`: Optional parent scope UUID.
    /// - `attributes`: LLM attribute bitflags.
    /// - `data`: Optional application payload stored on the handle.
    /// - `metadata`: Optional metadata stored on the handle.
    /// - `model_name`: Optional normalized model name stored on the handle.
    /// - `timestamp`: Optional handle start time. When omitted, the current
    ///   UTC time is used.
    ///
    /// # Returns
    /// A new [`LlmHandle`] with a fresh UUID.
    pub fn create_llm_handle(&self, params: CreateLlmHandleParams<'_>) -> LlmHandle {
        LlmHandle::builder()
            .name(params.name)
            .started_at(params.timestamp.unwrap_or_else(Utc::now))
            .attributes(params.attributes)
            .parent_uuid_opt(params.parent_uuid)
            .data_opt(params.data)
            .metadata_opt(params.metadata)
            .model_name_opt(params.model_name)
            .build()
    }

    /// Build an LLM-start event from a handle.
    ///
    /// # Parameters
    /// - `handle`: LLM handle to serialize into an event.
    /// - `data`: Sanitized LLM request payload.
    /// - `annotated_request`: Optional normalized request annotation.
    ///
    /// # Returns
    /// An LLM-start [`Event`] derived from the provided handle.
    pub fn build_llm_start_event(
        &self,
        handle: &LlmHandle,
        data: Option<Json>,
        annotated_request: Option<Arc<AnnotatedLlmRequest>>,
    ) -> Event {
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(handle.parent_uuid)
                .uuid(handle.uuid)
                .timestamp(handle.started_at)
                .name(handle.name.as_str())
                .data_opt(data)
                .metadata_opt(handle.metadata.clone())
                .build(),
            ScopeCategory::Start,
            llm_attributes_to_strings(handle.attributes),
            EventCategory::llm(),
            Some(
                CategoryProfile::builder()
                    .model_name_opt(handle.model_name.clone())
                    .annotated_request_opt(annotated_request)
                    .build(),
            ),
        ))
    }

    /// Build an LLM-end event from a handle and optional overrides.
    ///
    /// # Parameters
    /// - `handle`: LLM handle to serialize into an event.
    /// - `data`: Sanitized LLM response payload.
    /// - `metadata`: Optional metadata payload merged over `handle.metadata`.
    /// - `annotated_response`: Optional normalized response annotation.
    ///
    /// # Returns
    /// An LLM-end [`Event`] derived from the provided handle.
    pub fn end_llm_handle(
        &self,
        handle: &LlmHandle,
        data: Option<Json>,
        metadata: Option<Json>,
        annotated_response: Option<Arc<AnnotatedLlmResponse>>,
    ) -> Event {
        self.build_llm_end_event(
            EndLlmHandleParams::builder()
                .handle(handle)
                .data_opt(data)
                .metadata_opt(metadata)
                .annotated_response_opt(annotated_response)
                .build(),
        )
    }

    /// Build an LLM-end event from builder parameters.
    ///
    /// The `metadata` payload is merged over the metadata already stored on
    /// the handle.
    ///
    /// # Parameters
    /// - `params`: LLM end-event builder parameters.
    ///
    /// # Returns
    /// An LLM-end [`Event`] derived from the provided parameters.
    pub fn build_llm_end_event(&self, params: EndLlmHandleParams<'_>) -> Event {
        let handle = params.handle;
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(handle.parent_uuid)
                .uuid(handle.uuid)
                .timestamp(
                    params
                        .timestamp
                        .unwrap_or_else(|| end_timestamp_after(handle.started_at)),
                )
                .name(handle.name.as_str())
                .data_opt(params.data)
                .metadata_opt(merge_json(handle.metadata.clone(), params.metadata))
                .build(),
            ScopeCategory::End,
            llm_attributes_to_strings(handle.attributes),
            EventCategory::llm(),
            Some(
                CategoryProfile::builder()
                    .model_name_opt(handle.model_name.clone())
                    .annotated_response_opt(params.annotated_response)
                    .build(),
            ),
        ))
    }

    fn emit_guardrail_scope_start(
        name: &str,
        parent_uuid: Option<Uuid>,
        metadata: Option<Json>,
        input: Json,
        subscribers: &[EventSubscriberFn],
    ) -> ScopeHandle {
        let handle = ScopeHandle::builder()
            .name(name)
            .scope_type(ScopeType::Guardrail)
            .parent_uuid_opt(parent_uuid)
            .metadata_opt(metadata)
            .build();
        let event = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(handle.parent_uuid)
                .uuid(handle.uuid)
                .timestamp(handle.started_at)
                .name(handle.name.as_str())
                .data(input)
                .metadata_opt(handle.metadata.clone())
                .build(),
            ScopeCategory::Start,
            scope_attributes_to_strings(handle.attributes),
            EventCategory::from(handle.scope_type),
            None,
        ));
        Self::emit_event(&event, subscribers);
        handle
    }

    fn emit_guardrail_scope_end(
        handle: &ScopeHandle,
        output: Json,
        subscribers: &[EventSubscriberFn],
    ) {
        let event = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid_opt(handle.parent_uuid)
                .uuid(handle.uuid)
                .timestamp(end_timestamp_after(handle.started_at))
                .name(handle.name.as_str())
                .data(output)
                .metadata_opt(handle.metadata.clone())
                .build(),
            ScopeCategory::End,
            scope_attributes_to_strings(handle.attributes),
            EventCategory::from(handle.scope_type),
            None,
        ));
        Self::emit_event(&event, subscribers);
    }

    /// Run tool request sanitizers across global and scope-local registries.
    ///
    /// # Parameters
    /// - `name`: Tool name associated with the request.
    /// - `args`: Raw tool arguments to sanitize for observability.
    /// - `scope_locals`: Scope-local sanitizer registries collected from the
    ///   active scope stack.
    ///
    /// # Returns
    /// The sanitized JSON payload after every matching guardrail has run.
    pub(crate) fn tool_sanitize_request_chain(
        &self,
        name: &str,
        args: Json,
        scope_locals: &[&SortedRegistry<Guardrail<ToolSanitizeFn>>],
    ) -> Json {
        let entries = merge_guardrail_entries(&self.tool_sanitize_request_guardrails, scope_locals);
        let mut value = args;
        for entry in entries {
            value = (entry.payload)(name, value);
        }
        value
    }

    /// Run tool response sanitizers across global and scope-local registries.
    ///
    /// # Parameters
    /// - `name`: Tool name associated with the response.
    /// - `result`: Raw tool result to sanitize for observability.
    /// - `scope_locals`: Scope-local sanitizer registries collected from the
    ///   active scope stack.
    ///
    /// # Returns
    /// The sanitized JSON payload after every matching guardrail has run.
    pub(crate) fn tool_sanitize_response_chain(
        &self,
        name: &str,
        result: Json,
        scope_locals: &[&SortedRegistry<Guardrail<ToolSanitizeFn>>],
    ) -> Json {
        let entries =
            merge_guardrail_entries(&self.tool_sanitize_response_guardrails, scope_locals);
        let mut value = result;
        for entry in entries {
            value = (entry.payload)(name, value);
        }
        value
    }

    /// Snapshot tool conditional-execution guardrails in priority order.
    ///
    /// # Parameters
    /// - `scope_locals`: Scope-local conditional guardrail registries collected
    ///   from the active scope stack.
    ///
    /// # Returns
    /// Named guardrail snapshots that can be evaluated after registry locks
    /// are released.
    pub(crate) fn tool_conditional_execution_entries(
        &self,
        scope_locals: &[&SortedRegistry<Guardrail<ToolConditionalFn>>],
    ) -> Vec<Guardrail<ToolConditionalFn>> {
        merge_guardrail_entries(&self.tool_conditional_execution_guardrails, scope_locals)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Evaluate a snapshot of tool conditional-execution guardrails in priority order.
    ///
    /// This function emits guardrail scope start/end events while evaluating
    /// the provided entries. Callers should pass entries snapped from the
    /// global and scope-local registries so subscriber callbacks run without
    /// registry locks held. If `entries` is empty, no guardrail scopes are
    /// emitted. Guardrail start events identify the guardrail and target but
    /// intentionally omit raw tool arguments from their event data.
    ///
    /// # Parameters
    /// - `name`: Tool name associated with the request.
    /// - `args`: Tool arguments to validate.
    /// - `entries`: Borrowed conditional guardrail snapshots to evaluate.
    /// - `subscribers`: Event subscribers that should observe guardrail scope
    ///   start/end events.
    /// - `parent_uuid`: Optional parent scope UUID for emitted guardrail
    ///   scopes.
    /// - `metadata`: Optional metadata attached to emitted guardrail scopes.
    ///
    /// # Returns
    /// A [`Result`](crate::error::Result) containing `Ok(None)` when execution
    /// is allowed or `Ok(Some(reason))` when a guardrail rejects the call.
    ///
    /// # Errors
    /// Propagates any error returned by a guardrail callback after emitting the
    /// corresponding guardrail scope end event.
    pub(crate) fn tool_conditional_execution_snapshot_chain(
        name: &str,
        args: &Json,
        entries: &[Guardrail<ToolConditionalFn>],
        subscribers: &[EventSubscriberFn],
        parent_uuid: Option<Uuid>,
        metadata: Option<Json>,
    ) -> crate::error::Result<Option<String>> {
        for entry in entries {
            let handle = Self::emit_guardrail_scope_start(
                &entry.name,
                parent_uuid,
                metadata.clone(),
                json!({
                    "kind": "tool_conditional_execution",
                    "target_name": name,
                }),
                subscribers,
            );
            let result = (entry.payload)(name, args);
            let output = match &result {
                Ok(Some(reason)) => json!({
                    "allowed": false,
                    "rejected": true,
                    "rejection_reason": reason,
                }),
                Ok(None) => json!({
                    "allowed": true,
                    "rejected": false,
                }),
                Err(error) => json!({
                    "allowed": false,
                    "error": error.to_string(),
                }),
            };
            Self::emit_guardrail_scope_end(&handle, output, subscribers);
            if let Some(error) = result? {
                return Ok(Some(error));
            }
        }
        Ok(None)
    }

    /// Run tool request intercepts in priority order.
    ///
    /// # Parameters
    /// - `name`: Tool name associated with the request.
    /// - `args`: Tool arguments to pass through the intercept chain.
    /// - `scope_locals`: Scope-local request intercept registries collected
    ///   from the active scope stack.
    ///
    /// # Returns
    /// A [`Result`] containing the final JSON argument payload.
    ///
    /// # Errors
    /// Propagates any error returned by an intercept callback.
    ///
    /// # Notes
    /// If an intercept entry has `break_chain` enabled, later intercepts are
    /// skipped after that entry runs.
    pub(crate) fn tool_request_intercepts_chain(
        &self,
        name: &str,
        args: Json,
        scope_locals: &[&SortedRegistry<Intercept<ToolInterceptFn>>],
    ) -> crate::error::Result<Json> {
        let entries = merge_intercept_entries(&self.tool_request_intercepts, scope_locals);
        let mut value = args;
        for entry in entries {
            value = (entry.payload.callable)(name, value)?;
            if entry.payload.break_chain {
                break;
            }
        }
        Ok(value)
    }

    /// Build the composed tool execution continuation chain.
    ///
    /// # Parameters
    /// - `name`: Tool name passed into each execution intercept.
    /// - `default_fn`: Base tool callback that should run after all intercepts.
    /// - `scope_locals`: Scope-local execution intercept registries collected
    ///   from the active scope stack.
    ///
    /// # Returns
    /// A composed [`ToolExecutionNextFn`] that wraps `default_fn` in every
    /// matching execution intercept.
    pub(crate) fn tool_build_execution_chain(
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

    /// Run LLM request sanitizers across global and scope-local registries.
    ///
    /// # Parameters
    /// - `request`: Raw LLM request to sanitize for observability.
    /// - `scope_locals`: Scope-local sanitizer registries collected from the
    ///   active scope stack.
    ///
    /// # Returns
    /// The sanitized [`LlmRequest`] after every matching guardrail has run.
    pub(crate) fn llm_sanitize_request_chain(
        &self,
        request: LlmRequest,
        scope_locals: &[&SortedRegistry<Guardrail<LlmSanitizeRequestFn>>],
    ) -> LlmRequest {
        let entries = merge_guardrail_entries(&self.llm_sanitize_request_guardrails, scope_locals);
        let mut value = request;
        for entry in entries {
            value = (entry.payload)(value);
        }
        value
    }

    /// Run LLM response sanitizers across global and scope-local registries.
    ///
    /// # Parameters
    /// - `response`: Raw response payload to sanitize for observability.
    /// - `scope_locals`: Scope-local sanitizer registries collected from the
    ///   active scope stack.
    ///
    /// # Returns
    /// The sanitized response payload after every matching guardrail has run.
    pub(crate) fn llm_sanitize_response_chain(
        &self,
        response: Json,
        scope_locals: &[&SortedRegistry<Guardrail<LlmSanitizeResponseFn>>],
    ) -> Json {
        let entries = merge_guardrail_entries(&self.llm_sanitize_response_guardrails, scope_locals);
        let mut value = response;
        for entry in entries {
            value = (entry.payload)(value);
        }
        value
    }

    /// Snapshot LLM conditional-execution guardrails in priority order.
    ///
    /// # Parameters
    /// - `scope_locals`: Scope-local conditional guardrail registries collected
    ///   from the active scope stack.
    ///
    /// # Returns
    /// Named guardrail snapshots that can be evaluated after registry locks
    /// are released.
    pub(crate) fn llm_conditional_execution_entries(
        &self,
        scope_locals: &[&SortedRegistry<Guardrail<LlmConditionalFn>>],
    ) -> Vec<Guardrail<LlmConditionalFn>> {
        merge_guardrail_entries(&self.llm_conditional_execution_guardrails, scope_locals)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Evaluate a snapshot of LLM conditional-execution guardrails in priority order.
    ///
    /// This function emits guardrail scope start/end events while evaluating
    /// the provided entries. Callers should pass entries snapped from the
    /// global and scope-local registries so subscriber callbacks run without
    /// registry locks held. If `entries` is empty, no guardrail scopes are
    /// emitted. Guardrail start events identify the guardrail but intentionally
    /// omit raw LLM requests from their event data.
    ///
    /// # Parameters
    /// - `request`: LLM request to validate.
    /// - `entries`: Borrowed conditional guardrail snapshots to evaluate.
    /// - `subscribers`: Event subscribers that should observe guardrail scope
    ///   start/end events.
    /// - `parent_uuid`: Optional parent scope UUID for emitted guardrail
    ///   scopes.
    /// - `metadata`: Optional metadata attached to emitted guardrail scopes.
    ///
    /// # Returns
    /// A [`Result`](crate::error::Result) containing `Ok(None)` when execution
    /// is allowed or `Ok(Some(reason))` when a guardrail rejects the call.
    ///
    /// # Errors
    /// Propagates any error returned by a guardrail callback after emitting the
    /// corresponding guardrail scope end event.
    pub(crate) fn llm_conditional_execution_snapshot_chain(
        request: &LlmRequest,
        entries: &[Guardrail<LlmConditionalFn>],
        subscribers: &[EventSubscriberFn],
        parent_uuid: Option<Uuid>,
        metadata: Option<Json>,
    ) -> crate::error::Result<Option<String>> {
        for entry in entries {
            let handle = Self::emit_guardrail_scope_start(
                &entry.name,
                parent_uuid,
                metadata.clone(),
                json!({
                    "kind": "llm_conditional_execution",
                }),
                subscribers,
            );
            let result = (entry.payload)(request);
            let output = match &result {
                Ok(Some(reason)) => json!({
                    "allowed": false,
                    "rejected": true,
                    "rejection_reason": reason,
                }),
                Ok(None) => json!({
                    "allowed": true,
                    "rejected": false,
                }),
                Err(error) => json!({
                    "allowed": false,
                    "error": error.to_string(),
                }),
            };
            Self::emit_guardrail_scope_end(&handle, output, subscribers);
            if let Some(error) = result? {
                return Ok(Some(error));
            }
        }
        Ok(None)
    }

    /// Run LLM request intercepts in priority order.
    ///
    /// # Parameters
    /// - `name`: Logical provider or model family name.
    /// - `request`: LLM request to pass through the intercept chain.
    /// - `annotated`: Optional normalized request annotation to carry through
    ///   the chain.
    /// - `scope_locals`: Scope-local request intercept registries collected
    ///   from the active scope stack.
    ///
    /// # Returns
    /// A [`Result`] containing the final request and annotation pair.
    ///
    /// # Errors
    /// Propagates any error returned by an intercept callback.
    ///
    /// # Notes
    /// If an intercept entry has `break_chain` enabled, later intercepts are
    /// skipped after that entry runs.
    pub(crate) fn llm_request_intercepts_chain(
        &self,
        name: &str,
        request: LlmRequest,
        annotated: Option<AnnotatedLlmRequest>,
        scope_locals: &[&SortedRegistry<Intercept<LlmRequestInterceptFn>>],
    ) -> crate::error::Result<(LlmRequest, Option<AnnotatedLlmRequest>)> {
        let entries = merge_intercept_entries(&self.llm_request_intercepts, scope_locals);
        let mut request_value = request;
        let mut annotated_value = annotated;
        for entry in entries {
            let (new_request, new_annotated) =
                (entry.payload.callable)(name, request_value, annotated_value)?;
            request_value = new_request;
            annotated_value = new_annotated;
            if entry.payload.break_chain {
                break;
            }
        }
        Ok((request_value, annotated_value))
    }

    /// Build the composed non-streaming LLM execution continuation chain.
    ///
    /// # Parameters
    /// - `name`: Logical provider or model family name passed into each
    ///   execution intercept.
    /// - `default_fn`: Base provider callback that should run after all
    ///   intercepts.
    /// - `scope_locals`: Scope-local execution intercept registries collected
    ///   from the active scope stack.
    ///
    /// # Returns
    /// A composed [`LlmExecutionNextFn`] that wraps `default_fn` in every
    /// matching execution intercept.
    pub(crate) fn llm_build_execution_chain(
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

    /// Build the composed streaming LLM execution continuation chain.
    ///
    /// # Parameters
    /// - `name`: Logical provider or model family name passed into each
    ///   execution intercept.
    /// - `default_fn`: Base stream-producing callback that should run after all
    ///   intercepts.
    /// - `scope_locals`: Scope-local execution intercept registries collected
    ///   from the active scope stack.
    ///
    /// # Returns
    /// A composed [`LlmStreamExecutionNextFn`] that wraps `default_fn` in every
    /// matching execution intercept.
    pub(crate) fn llm_stream_build_execution_chain(
        &self,
        name: &str,
        default_fn: LlmStreamExecutionNextFn,
        scope_locals: LlmStreamExecutionRegistryRefs<'_>,
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

fn end_timestamp_after(started_at: chrono::DateTime<Utc>) -> chrono::DateTime<Utc> {
    let now = Utc::now();
    if now > started_at {
        now
    } else {
        started_at + Duration::microseconds(1)
    }
}

impl Default for NemoRelayContextState {
    fn default() -> Self {
        Self::new()
    }
}
