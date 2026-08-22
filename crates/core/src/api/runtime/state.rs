// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-global runtime state and middleware-chain builders.
//!
//! [`NemoRelayContextState`] owns the registries and helper methods that power
//! the public scope, tool, and LLM APIs. Advanced integrations can use this
//! type directly to register middleware, attach runtime extensions, and build
//! the resolved callback chains that the higher-level API layer executes.

use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures_util::{FutureExt, Stream};

use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, EventSanitizeFields, MarkEvent,
    ScopeCategory, ScopeEvent, is_valid_event_metadata_attribute_key, llm_attributes_to_strings,
    scope_attributes_to_strings, tool_attributes_to_strings,
};
use crate::api::llm::{CreateLlmHandleParams, EndLlmHandleParams};
use crate::api::llm::{LlmHandle, LlmRequest};
use crate::api::registry::{
    EventMetadataInjector, ExecutionIntercept, Guardrail, Intercept, RuntimeRegistrationKind,
    runtime_registration_is_enabled,
};
use crate::api::runtime::ScopeStackHandle;
use crate::api::runtime::callbacks::{
    EventSanitizeFn, EventSubscriberFn, LlmConditionalFn, LlmExecutionFn, LlmExecutionNextFn,
    LlmJsonStream, LlmRequestInterceptFn, LlmSanitizeRequestContext, LlmSanitizeRequestFn,
    LlmSanitizeResponseContext, LlmSanitizeResponseFn, LlmStreamExecutionFn,
    LlmStreamExecutionNextFn, LlmStreamExecutionRegistryRefs, LlmStreamInner, ToolConditionalFn,
    ToolExecutionFn, ToolExecutionNextFn, ToolExecutionOutcomeNextFn, ToolInterceptFn,
    ToolSanitizeFn,
};
use crate::api::runtime::continuation_context::{
    MiddlewareContinuationContext, MiddlewareContinuationGuard, MiddlewareContinuationLease,
};
use crate::api::runtime::subscriber_dispatcher;
use crate::api::scope::{CreateScopeHandleParams, EndScopeHandleParams, ScopeHandle, ScopeType};
use crate::api::shared::snapshot_event_sanitizers;
use crate::api::tool::ToolHandle;
use crate::api::tool::{
    CreateToolHandleParams, EndToolHandleParams, ToolExecutionInterceptOutcome,
};
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::response::AnnotatedLlmResponse;
use crate::context::registries::{
    merge_event_metadata_injector_entries, merge_execution_intercept_callables,
    merge_guardrail_entries, merge_intercept_entries,
};
use crate::error::FlowError;
use crate::json::{Json, merge_json};
use crate::registry::SortedRegistry;
use chrono::{Duration, Utc};
use serde_json::json;
use uuid::Uuid;

struct ContinuationGuardedLlmStream {
    inner: LlmJsonStream,
    guard: Option<MiddlewareContinuationGuard>,
}

struct ContextualizedLlmStream {
    inner: LlmJsonStream,
    context: MiddlewareContinuationContext,
}

impl Stream for ContextualizedLlmStream {
    type Item = crate::error::Result<Json>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let context = this.context.clone();
        let inner = &mut this.inner;
        let future = context.run(futures_util::future::poll_fn(|inner_cx| {
            Pin::new(&mut *inner).poll_next(inner_cx)
        }));
        tokio::pin!(future);
        future.poll(cx)
    }
}

impl LlmStreamInner for ContextualizedLlmStream {
    fn terminalize(self: Pin<&mut Self>) {
        self.get_mut().inner.terminalize();
    }

    fn close(
        self: Pin<&mut Self>,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + '_>> {
        Box::pin(async move {
            let this = self.get_mut();
            let context = this.context.clone();
            context.run(this.inner.close()).await
        })
    }
}

pub(crate) fn contextualize_stream(
    stream: LlmJsonStream,
    context: MiddlewareContinuationContext,
) -> LlmJsonStream {
    LlmJsonStream::from_closeable(ContextualizedLlmStream {
        inner: stream,
        context,
    })
}

impl Stream for ContinuationGuardedLlmStream {
    type Item = crate::error::Result<Json>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_next(cx);
        if matches!(&result, Poll::Ready(None)) {
            this.guard.take();
        }
        result
    }
}

impl LlmStreamInner for ContinuationGuardedLlmStream {
    fn terminalize(self: Pin<&mut Self>) {
        let this = self.get_mut();
        this.guard.take();
        this.inner.terminalize();
    }

    fn close(
        self: Pin<&mut Self>,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + '_>> {
        Box::pin(async move {
            let this = self.get_mut();
            let guard = this.guard.take();
            let result = this.inner.close().await;
            drop(guard);
            result
        })
    }
}

fn guard_stream_continuation(
    stream: LlmJsonStream,
    guard: MiddlewareContinuationGuard,
) -> LlmJsonStream {
    LlmJsonStream::from_closeable(ContinuationGuardedLlmStream {
        inner: stream,
        guard: Some(guard),
    })
}

struct GuardrailScopeCompletion<'a> {
    handle: Option<ScopeHandle>,
    subscribers: &'a [EventSubscriberFn],
    scope_stack: ScopeStackHandle,
    pending_publication: Option<subscriber_dispatcher::PendingPublication>,
}

impl GuardrailScopeCompletion<'_> {
    fn new(
        handle: ScopeHandle,
        subscribers: &[EventSubscriberFn],
        scope_stack: ScopeStackHandle,
    ) -> GuardrailScopeCompletion<'_> {
        GuardrailScopeCompletion {
            handle: Some(handle),
            subscribers,
            scope_stack,
            pending_publication: (!subscribers.is_empty())
                .then(subscriber_dispatcher::register_pending_publication)
                .flatten(),
        }
    }

    fn finish(mut self, output: Json) {
        let handle = self.handle.take().expect("guardrail scope handle");
        NemoRelayContextState::emit_guardrail_scope_end(
            &handle,
            output,
            self.subscribers,
            self.scope_stack.clone(),
        );
        drop(self.pending_publication.take());
    }
}

impl Drop for GuardrailScopeCompletion<'_> {
    fn drop(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };
        NemoRelayContextState::emit_guardrail_scope_end(
            &handle,
            json!({
                "allowed": false,
                "cancelled": true,
                "error": "guardrail evaluation cancelled",
            }),
            self.subscribers,
            self.scope_stack.clone(),
        );
        drop(self.pending_publication.take());
    }
}

/// Process-global runtime state backing middleware and event emission.
///
/// The public API layer stores one shared instance of this type for the
/// process. It contains global middleware registries, lifecycle subscribers,
/// and arbitrary extension slots used by bindings or integrations.
pub struct NemoRelayContextState {
    /// Global Event metadata injectors applied before Event sanitizers.
    pub(crate) event_metadata_injectors: SortedRegistry<EventMetadataInjector>,
    /// Global mark event field sanitizers.
    pub(crate) mark_sanitize_guardrails: SortedRegistry<Guardrail<EventSanitizeFn>>,
    /// Global scope-start event field sanitizers.
    pub(crate) scope_sanitize_start_guardrails: SortedRegistry<Guardrail<EventSanitizeFn>>,
    /// Global scope-end event field sanitizers.
    pub(crate) scope_sanitize_end_guardrails: SortedRegistry<Guardrail<EventSanitizeFn>>,
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
    /// Whether LLM start events retain complete sanitized request payloads.
    pub(crate) observability_full_payloads_enabled: bool,
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
            event_metadata_injectors: SortedRegistry::new(),
            mark_sanitize_guardrails: SortedRegistry::new(),
            scope_sanitize_start_guardrails: SortedRegistry::new(),
            scope_sanitize_end_guardrails: SortedRegistry::new(),
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
            observability_full_payloads_enabled: false,
            extensions: HashMap::new(),
        }
    }

    /// Clone the requested registration-bearing portions of runtime state.
    ///
    /// Eligibility callbacks must run without holding the process-global
    /// runtime-state lock. Callers use this snapshot to release that lock
    /// before resolving conditional middleware guardrails. Unrequested
    /// registries and runtime extensions are intentionally excluded.
    pub(crate) fn registry_snapshot(&self, kinds: &[RuntimeRegistrationKind]) -> Self {
        let mut snapshot = Self::new();
        snapshot.observability_full_payloads_enabled = self.observability_full_payloads_enabled;
        for kind in kinds {
            match kind {
                RuntimeRegistrationKind::Subscriber => {
                    snapshot.event_subscribers = self.event_subscribers.clone();
                }
                RuntimeRegistrationKind::EventMetadataInjector => {
                    snapshot.event_metadata_injectors = self.event_metadata_injectors.clone();
                }
                RuntimeRegistrationKind::MarkSanitizeGuardrail => {
                    snapshot.mark_sanitize_guardrails = self.mark_sanitize_guardrails.clone();
                }
                RuntimeRegistrationKind::ScopeSanitizeStartGuardrail => {
                    snapshot.scope_sanitize_start_guardrails =
                        self.scope_sanitize_start_guardrails.clone();
                }
                RuntimeRegistrationKind::ScopeSanitizeEndGuardrail => {
                    snapshot.scope_sanitize_end_guardrails =
                        self.scope_sanitize_end_guardrails.clone();
                }
                RuntimeRegistrationKind::ToolSanitizeRequestGuardrail => {
                    snapshot.tool_sanitize_request_guardrails =
                        self.tool_sanitize_request_guardrails.clone();
                }
                RuntimeRegistrationKind::ToolSanitizeResponseGuardrail => {
                    snapshot.tool_sanitize_response_guardrails =
                        self.tool_sanitize_response_guardrails.clone();
                }
                RuntimeRegistrationKind::ToolConditionalExecutionGuardrail => {
                    snapshot.tool_conditional_execution_guardrails =
                        self.tool_conditional_execution_guardrails.clone();
                }
                RuntimeRegistrationKind::ToolRequestIntercept => {
                    snapshot.tool_request_intercepts = self.tool_request_intercepts.clone();
                }
                RuntimeRegistrationKind::ToolExecutionIntercept => {
                    snapshot.tool_execution_intercepts = self.tool_execution_intercepts.clone();
                }
                RuntimeRegistrationKind::LlmSanitizeRequestGuardrail => {
                    snapshot.llm_sanitize_request_guardrails =
                        self.llm_sanitize_request_guardrails.clone();
                }
                RuntimeRegistrationKind::LlmSanitizeResponseGuardrail => {
                    snapshot.llm_sanitize_response_guardrails =
                        self.llm_sanitize_response_guardrails.clone();
                }
                RuntimeRegistrationKind::LlmConditionalExecutionGuardrail => {
                    snapshot.llm_conditional_execution_guardrails =
                        self.llm_conditional_execution_guardrails.clone();
                }
                RuntimeRegistrationKind::LlmRequestIntercept => {
                    snapshot.llm_request_intercepts = self.llm_request_intercepts.clone();
                }
                RuntimeRegistrationKind::LlmExecutionIntercept => {
                    snapshot.llm_execution_intercepts = self.llm_execution_intercepts.clone();
                }
                RuntimeRegistrationKind::LlmStreamExecutionIntercept => {
                    snapshot.llm_stream_execution_intercepts =
                        self.llm_stream_execution_intercepts.clone();
                }
            }
        }
        snapshot
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
        subscribers.extend(
            self.event_subscribers
                .iter()
                .filter(|(name, _)| {
                    runtime_registration_is_enabled(RuntimeRegistrationKind::Subscriber, name)
                })
                .map(|(_, callback)| callback.clone()),
        );
        subscribers.extend(scope_local_subscribers.iter().cloned());
        subscribers
    }

    /// Deliver an event to every subscriber in order.
    ///
    /// # Parameters
    /// - `event`: Fully constructed lifecycle event to deliver.
    /// - `subscribers`: Subscribers that should observe the event.
    #[cfg(test)]
    pub(crate) fn emit_event(event: &Event, subscribers: &[EventSubscriberFn]) {
        let _ = subscriber_dispatcher::dispatch_event(event, subscribers);
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
    /// - `name`: Logical provider or model family name. Gateway-managed LLM
    ///   calls use provider route names such as `anthropic.messages`, which
    ///   become the emitted event name.
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
            .uuid(params.uuid.unwrap_or_else(Uuid::now_v7))
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
        scope_stack: ScopeStackHandle,
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
        let sanitizers = snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
        subscriber_dispatcher::dispatch_sanitized_event(
            event,
            sanitizers,
            subscribers,
            scope_stack,
        );
        handle
    }

    fn emit_guardrail_scope_end(
        handle: &ScopeHandle,
        output: Json,
        subscribers: &[EventSubscriberFn],
        scope_stack: ScopeStackHandle,
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
        let sanitizers = snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
        subscriber_dispatcher::dispatch_sanitized_event(
            event,
            sanitizers,
            subscribers,
            scope_stack,
        );
    }

    /// Snapshot event sanitizer entries in priority order.
    pub(crate) fn event_sanitize_entries(
        global: &SortedRegistry<Guardrail<EventSanitizeFn>>,
        scope_locals: &[&SortedRegistry<Guardrail<EventSanitizeFn>>],
        kind: RuntimeRegistrationKind,
    ) -> Vec<Guardrail<EventSanitizeFn>> {
        merge_guardrail_entries(global, scope_locals, kind)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Snapshot Event metadata injector entries in deterministic priority order.
    pub(crate) fn event_metadata_injector_entries(
        global: &SortedRegistry<EventMetadataInjector>,
        scope_locals: &[&SortedRegistry<EventMetadataInjector>],
    ) -> Vec<EventMetadataInjector> {
        merge_event_metadata_injector_entries(global, scope_locals)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Apply insert-only metadata additions from an injector snapshot.
    pub(crate) async fn event_metadata_injection_snapshot_chain(
        mut event: Event,
        entries: &[EventMetadataInjector],
    ) -> Event {
        for entry in entries {
            let callback = Arc::clone(&entry.payload);
            let context = Arc::new(event);
            let callback_context = Arc::clone(&context);
            let outcome = AssertUnwindSafe(async move { callback(callback_context).await })
                .catch_unwind()
                .await;
            event = Arc::try_unwrap(context).unwrap_or_else(|context| (*context).clone());

            let attributes = match outcome {
                Ok(Ok(attributes)) => attributes,
                Ok(Err(error)) => {
                    log::error!(
                        target: "nemo_relay.runtime",
                        event = "event_metadata_injector_failed",
                        injector = entry.name.as_str(),
                        event_name = event.name();
                        "Event metadata injector failed; omitting its attributes: {error}"
                    );
                    continue;
                }
                Err(_) => {
                    log::error!(
                        target: "nemo_relay.runtime",
                        event = "event_metadata_injector_panicked",
                        injector = entry.name.as_str(),
                        event_name = event.name();
                        "Event metadata injector panicked; omitting its attributes"
                    );
                    continue;
                }
            };

            if let Err(error) = validate_event_metadata_attributes(&attributes) {
                log::error!(
                    target: "nemo_relay.runtime",
                    event = "event_metadata_injector_invalid_output",
                    injector = entry.name.as_str(),
                    event_name = event.name();
                    "Event metadata injector returned invalid attributes; omitting them: {error}"
                );
                continue;
            }

            if attributes.is_empty() {
                continue;
            }

            let mut fields = event.sanitize_fields();
            let mut metadata = match fields.metadata.take() {
                None => serde_json::Map::new(),
                Some(Json::Object(metadata)) => metadata,
                Some(metadata) => {
                    fields.metadata = Some(metadata);
                    log::error!(
                        target: "nemo_relay.runtime",
                        event = "event_metadata_injector_non_object_metadata",
                        injector = entry.name.as_str(),
                        event_name = event.name();
                        "Event metadata is not an object; omitting injected attributes"
                    );
                    continue;
                }
            };
            for (key, value) in attributes {
                metadata.entry(key).or_insert(value);
            }
            fields.metadata = Some(Json::Object(metadata));
            event.apply_sanitize_fields(fields);
        }
        event
    }
    /// Apply an event sanitizer snapshot to the mutable observability fields.
    pub(crate) async fn event_sanitize_snapshot_chain(
        mut event: Event,
        entries: &[Guardrail<EventSanitizeFn>],
    ) -> Event {
        for entry in entries {
            let fields = event.sanitize_fields();
            let callback = Arc::clone(&entry.payload);
            let context = Arc::new(event);
            let callback_context = Arc::clone(&context);
            let outcome = AssertUnwindSafe(async move { callback(callback_context, fields).await })
                .catch_unwind()
                .await;
            event = Arc::try_unwrap(context).unwrap_or_else(|context| (*context).clone());
            match outcome {
                Ok(Ok(fields)) => event.apply_sanitize_fields(fields),
                Ok(Err(_error)) => {
                    log::error!(
                        target: "nemo_relay.runtime",
                        event = "event_sanitizer_failed",
                        sanitizer = entry.name.as_str(),
                        event_name = event.name();
                        "Event sanitizer failed; clearing observability fields"
                    );
                    event.apply_sanitize_fields(EventSanitizeFields::default());
                    break;
                }
                Err(_) => {
                    log::error!(
                        target: "nemo_relay.runtime",
                        event = "event_sanitizer_panicked",
                        sanitizer = entry.name.as_str(),
                        event_name = event.name();
                        "Event sanitizer panicked; clearing observability fields"
                    );
                    event.apply_sanitize_fields(EventSanitizeFields::default());
                    break;
                }
            }
        }
        event
    }

    /// Snapshot tool request sanitizers in priority order.
    ///
    /// # Parameters
    /// - `scope_locals`: Scope-local sanitizer registries collected from the
    ///   active scope stack.
    ///
    /// # Returns
    /// Named sanitizer snapshots that can be evaluated after registry locks
    /// are released.
    pub(crate) fn tool_sanitize_request_entries(
        &self,
        scope_locals: &[&SortedRegistry<Guardrail<ToolSanitizeFn>>],
    ) -> Vec<Guardrail<ToolSanitizeFn>> {
        merge_guardrail_entries(
            &self.tool_sanitize_request_guardrails,
            scope_locals,
            RuntimeRegistrationKind::ToolSanitizeRequestGuardrail,
        )
        .into_iter()
        .cloned()
        .collect()
    }

    /// Run a snapshot of tool request sanitizers in priority order.
    ///
    /// # Parameters
    /// - `name`: Tool name associated with the request.
    /// - `args`: Raw tool arguments to sanitize for observability.
    /// - `entries`: Sanitizer snapshots to evaluate.
    ///
    /// # Returns
    /// The sanitized JSON payload after every provided guardrail has run, or
    /// `None` when a sanitizer failure omits the payload.
    pub(crate) async fn tool_sanitize_request_snapshot_chain(
        name: &str,
        args: Json,
        entries: &[Guardrail<ToolSanitizeFn>],
    ) -> Option<Json> {
        let mut value = Some(args);
        for entry in entries {
            if let Some(current) = value.take() {
                let callback = Arc::clone(&entry.payload);
                let callback_name = name.to_string();
                match AssertUnwindSafe(async move { callback(callback_name, current).await })
                    .catch_unwind()
                    .await
                {
                    Ok(Ok(next)) => value = Some(next),
                    Ok(Err(_error)) => log::error!(
                        target: "nemo_relay.runtime",
                        event = "tool_request_sanitizer_failed",
                        sanitizer = entry.name.as_str(),
                        tool_name = name;
                        "Tool request sanitizer failed; omitting the observability payload"
                    ),
                    Err(_) => log::error!(
                        target: "nemo_relay.runtime",
                        event = "tool_request_sanitizer_panicked",
                        sanitizer = entry.name.as_str(),
                        tool_name = name;
                        "Tool request sanitizer panicked; omitting the observability payload"
                    ),
                }
            }
        }
        value
    }

    /// Snapshot tool response sanitizers in priority order.
    ///
    /// # Parameters
    /// - `scope_locals`: Scope-local sanitizer registries collected from the
    ///   active scope stack.
    ///
    /// # Returns
    /// Named sanitizer snapshots that can be evaluated after registry locks
    /// are released.
    pub(crate) fn tool_sanitize_response_entries(
        &self,
        scope_locals: &[&SortedRegistry<Guardrail<ToolSanitizeFn>>],
    ) -> Vec<Guardrail<ToolSanitizeFn>> {
        merge_guardrail_entries(
            &self.tool_sanitize_response_guardrails,
            scope_locals,
            RuntimeRegistrationKind::ToolSanitizeResponseGuardrail,
        )
        .into_iter()
        .cloned()
        .collect()
    }

    /// Run a snapshot of tool response sanitizers in priority order.
    ///
    /// # Parameters
    /// - `name`: Tool name associated with the response.
    /// - `result`: Application-owned tool result JSON to sanitize for
    ///   observability.
    /// - `entries`: Sanitizer snapshots to evaluate.
    ///
    /// # Returns
    /// The sanitized JSON payload after every provided guardrail has run, or
    /// `None` when a sanitizer failure omits the payload.
    pub(crate) async fn tool_sanitize_response_snapshot_chain(
        name: &str,
        result: Json,
        entries: &[Guardrail<ToolSanitizeFn>],
    ) -> Option<Json> {
        let mut value = Some(result);
        for entry in entries {
            if let Some(current) = value.take() {
                let callback = Arc::clone(&entry.payload);
                let callback_name = name.to_string();
                match AssertUnwindSafe(async move { callback(callback_name, current).await })
                    .catch_unwind()
                    .await
                {
                    Ok(Ok(next)) => value = Some(next),
                    Ok(Err(_error)) => log::error!(
                        target: "nemo_relay.runtime",
                        event = "tool_response_sanitizer_failed",
                        sanitizer = entry.name.as_str(),
                        tool_name = name;
                        "Tool response sanitizer failed; omitting the observability payload"
                    ),
                    Err(_) => log::error!(
                        target: "nemo_relay.runtime",
                        event = "tool_response_sanitizer_panicked",
                        sanitizer = entry.name.as_str(),
                        tool_name = name;
                        "Tool response sanitizer panicked; omitting the observability payload"
                    ),
                }
            }
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
        merge_guardrail_entries(
            &self.tool_conditional_execution_guardrails,
            scope_locals,
            RuntimeRegistrationKind::ToolConditionalExecutionGuardrail,
        )
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
    pub(crate) async fn tool_conditional_execution_snapshot_chain(
        name: &str,
        args: &Json,
        entries: &[Guardrail<ToolConditionalFn>],
        subscribers: &[EventSubscriberFn],
        parent_uuid: Option<Uuid>,
        metadata: Option<Json>,
    ) -> crate::error::Result<Option<String>> {
        for entry in entries {
            let scope_stack = super::current_scope_stack();
            let handle = Self::emit_guardrail_scope_start(
                &entry.name,
                parent_uuid,
                metadata.clone(),
                json!({
                    "kind": "tool_conditional_execution",
                    "target_name": name,
                }),
                subscribers,
                scope_stack.clone(),
            );
            let completion = GuardrailScopeCompletion::new(handle, subscribers, scope_stack);
            let callback = Arc::clone(&entry.payload);
            let callback_name = name.to_string();
            let callback_args = args.clone();
            let result =
                match AssertUnwindSafe(async move { callback(callback_name, callback_args).await })
                    .catch_unwind()
                    .await
                {
                    Ok(result) => result,
                    Err(_) => Err(FlowError::Internal(format!(
                        "tool conditional guardrail '{}' panicked",
                        entry.name
                    ))),
                };
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
            completion.finish(output);
            if let Some(error) = result? {
                return Ok(Some(error));
            }
        }
        Ok(None)
    }

    /// Snapshot tool request intercepts in priority order.
    ///
    /// # Parameters
    /// - `scope_locals`: Scope-local request intercept registries collected
    ///   from the active scope stack.
    ///
    /// # Returns
    /// Named intercept snapshots that can be evaluated after registry locks
    /// are released.
    pub(crate) fn tool_request_intercept_entries(
        &self,
        scope_locals: &[&SortedRegistry<Intercept<ToolInterceptFn>>],
    ) -> Vec<Intercept<ToolInterceptFn>> {
        merge_intercept_entries(
            &self.tool_request_intercepts,
            scope_locals,
            RuntimeRegistrationKind::ToolRequestIntercept,
        )
        .into_iter()
        .cloned()
        .collect()
    }

    /// Run a snapshot of tool request intercepts in priority order.
    ///
    /// # Parameters
    /// - `name`: Tool name associated with the request.
    /// - `args`: Tool arguments to pass through the intercept chain.
    /// - `entries`: Intercept snapshots to evaluate.
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
    pub(crate) async fn tool_request_intercepts_snapshot_chain(
        name: &str,
        args: Json,
        entries: &[Intercept<ToolInterceptFn>],
    ) -> crate::error::Result<Json> {
        let mut value = args;
        for entry in entries {
            let callback = Arc::clone(&entry.payload.callable);
            let callback_name = name.to_string();
            value = match AssertUnwindSafe(async move { callback(callback_name, value).await })
                .catch_unwind()
                .await
            {
                Ok(result) => result?,
                Err(_) => {
                    return Err(FlowError::Internal(format!(
                        "tool request intercept '{}' panicked",
                        entry.name
                    )));
                }
            };
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
    /// A composed [`ToolExecutionOutcomeNextFn`] that wraps `default_fn` in
    /// every matching execution intercept.
    pub(crate) fn tool_build_execution_chain(
        &self,
        name: &str,
        default_fn: ToolExecutionNextFn,
        scope_locals: &[&SortedRegistry<ExecutionIntercept<ToolExecutionFn>>],
    ) -> ToolExecutionOutcomeNextFn {
        let matching = merge_execution_intercept_callables(
            &self.tool_execution_intercepts,
            scope_locals,
            RuntimeRegistrationKind::ToolExecutionIntercept,
        );
        let mut next: ToolExecutionOutcomeNextFn = Arc::new(move |args| {
            let default_fn = default_fn.clone();
            Box::pin(async move {
                default_fn(args)
                    .await
                    .map(ToolExecutionInterceptOutcome::from)
            })
        });
        let name = name.to_string();
        for (callable, _) in matching.into_iter().rev() {
            let current_next = next.clone();
            let current_name = name.clone();
            next = Arc::new(move |args| {
                let callable = callable.clone();
                let current_name = current_name.clone();
                let (continuation, continuation_guard) = MiddlewareContinuationLease::capture();
                let next_sequence = Arc::new(AtomicUsize::new(0));
                let downstream_marks = Arc::new(Mutex::new(Vec::new()));
                let raw_next: ToolExecutionNextFn = {
                    let current_next = current_next.clone();
                    let continuation = continuation.clone();
                    let next_sequence = next_sequence.clone();
                    let downstream_marks = downstream_marks.clone();
                    Arc::new(move |args| {
                        let sequence = next_sequence.fetch_add(1, Ordering::Relaxed);
                        let current_next = current_next.clone();
                        let invocation = continuation.begin();
                        let downstream_marks = downstream_marks.clone();
                        Box::pin(async move {
                            let mut outcome =
                                invocation?.invoke(move || current_next(args)).await?;
                            let pending_marks = std::mem::take(&mut outcome.pending_marks);
                            downstream_marks
                                .lock()
                                .expect("tool pending mark accumulator lock poisoned")
                                .push((sequence, pending_marks));
                            Ok(outcome.into_execution_result())
                        })
                    })
                };
                Box::pin(async move {
                    let outcome = callable(&current_name, args, raw_next).await;
                    drop(continuation_guard);
                    let mut outcome = outcome?;
                    let mut downstream_batches = std::mem::take(
                        &mut *downstream_marks
                            .lock()
                            .expect("tool pending mark accumulator lock poisoned"),
                    );
                    downstream_batches.sort_by_key(|(sequence, _)| *sequence);
                    let mut marks = downstream_batches
                        .into_iter()
                        .flat_map(|(_, marks)| marks)
                        .collect::<Vec<_>>();
                    marks.append(&mut outcome.pending_marks);
                    outcome.pending_marks = marks;
                    Ok(outcome)
                })
            });
        }
        next
    }

    /// Snapshot LLM request sanitizers in priority order.
    ///
    /// # Parameters
    /// - `scope_locals`: Scope-local sanitizer registries collected from the
    ///   active scope stack.
    ///
    /// # Returns
    /// Named sanitizer snapshots that can be evaluated after registry locks
    /// are released.
    pub(crate) fn llm_sanitize_request_entries(
        &self,
        scope_locals: &[&SortedRegistry<Guardrail<LlmSanitizeRequestFn>>],
    ) -> Vec<Guardrail<LlmSanitizeRequestFn>> {
        merge_guardrail_entries(
            &self.llm_sanitize_request_guardrails,
            scope_locals,
            RuntimeRegistrationKind::LlmSanitizeRequestGuardrail,
        )
        .into_iter()
        .cloned()
        .collect()
    }

    /// Run a snapshot of LLM request sanitizers in priority order.
    ///
    /// # Parameters
    /// - `request`: Raw LLM request to sanitize for observability.
    /// - `entries`: Sanitizer snapshots to evaluate.
    ///
    /// # Returns
    /// The sanitized [`LlmRequest`] after every provided guardrail has run, or
    /// `None` when a sanitizer errors or panics.
    pub(crate) async fn llm_sanitize_request_snapshot_chain(
        request: LlmRequest,
        context: LlmSanitizeRequestContext,
        entries: &[Guardrail<LlmSanitizeRequestFn>],
    ) -> Option<LlmRequest> {
        let mut value = Some(request);
        for entry in entries {
            if let Some(current) = value.take() {
                let callback = Arc::clone(&entry.payload);
                let callback_value = current.clone();
                let callback_context = context.clone();
                match AssertUnwindSafe(
                    async move { callback(callback_value, callback_context).await },
                )
                .catch_unwind()
                .await
                {
                    Ok(Ok(next)) => value = next,
                    Ok(Err(_error)) => {
                        log::error!(
                            target: "nemo_relay.runtime",
                            event = "llm_request_sanitizer_failed",
                            sanitizer = entry.name.as_str();
                            "LLM request sanitizer failed; omitting the observability payload"
                        );
                    }
                    Err(_) => {
                        log::error!(
                            target: "nemo_relay.runtime",
                            event = "llm_request_sanitizer_panicked",
                            sanitizer = entry.name.as_str();
                            "LLM request sanitizer panicked; omitting the observability payload"
                        );
                    }
                }
            }
        }
        value
    }

    /// Snapshot LLM response sanitizers in priority order.
    ///
    /// # Parameters
    /// - `scope_locals`: Scope-local sanitizer registries collected from the
    ///   active scope stack.
    ///
    /// # Returns
    /// Named sanitizer snapshots that can be evaluated after registry locks
    /// are released.
    pub(crate) fn llm_sanitize_response_entries(
        &self,
        scope_locals: &[&SortedRegistry<Guardrail<LlmSanitizeResponseFn>>],
    ) -> Vec<Guardrail<LlmSanitizeResponseFn>> {
        merge_guardrail_entries(
            &self.llm_sanitize_response_guardrails,
            scope_locals,
            RuntimeRegistrationKind::LlmSanitizeResponseGuardrail,
        )
        .into_iter()
        .cloned()
        .collect()
    }

    /// Run a snapshot of LLM response sanitizers in priority order.
    ///
    /// # Parameters
    /// - `response`: Raw response payload to sanitize for observability.
    /// - `entries`: Sanitizer snapshots to evaluate.
    ///
    /// # Returns
    /// The sanitized response payload after every provided guardrail has run,
    /// or `None` when a sanitizer errors or panics.
    pub(crate) async fn llm_sanitize_response_snapshot_chain(
        response: Json,
        context: LlmSanitizeResponseContext,
        entries: &[Guardrail<LlmSanitizeResponseFn>],
    ) -> Option<Json> {
        let mut value = Some(response);
        for entry in entries {
            if let Some(current) = value.take() {
                let callback = Arc::clone(&entry.payload);
                let callback_value = current.clone();
                let callback_context = context.clone();
                match AssertUnwindSafe(
                    async move { callback(callback_value, callback_context).await },
                )
                .catch_unwind()
                .await
                {
                    Ok(Ok(next)) => value = next,
                    Ok(Err(_error)) => {
                        log::error!(
                            target: "nemo_relay.runtime",
                            event = "llm_response_sanitizer_failed",
                            sanitizer = entry.name.as_str();
                            "LLM response sanitizer failed; omitting the observability payload"
                        );
                    }
                    Err(_) => {
                        log::error!(
                            target: "nemo_relay.runtime",
                            event = "llm_response_sanitizer_panicked",
                            sanitizer = entry.name.as_str();
                            "LLM response sanitizer panicked; omitting the observability payload"
                        );
                    }
                }
            }
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
        merge_guardrail_entries(
            &self.llm_conditional_execution_guardrails,
            scope_locals,
            RuntimeRegistrationKind::LlmConditionalExecutionGuardrail,
        )
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
    pub(crate) async fn llm_conditional_execution_snapshot_chain(
        request: &LlmRequest,
        entries: &[Guardrail<LlmConditionalFn>],
        subscribers: &[EventSubscriberFn],
        parent_uuid: Option<Uuid>,
        metadata: Option<Json>,
    ) -> crate::error::Result<Option<String>> {
        for entry in entries {
            let scope_stack = super::current_scope_stack();
            let handle = Self::emit_guardrail_scope_start(
                &entry.name,
                parent_uuid,
                metadata.clone(),
                json!({
                    "kind": "llm_conditional_execution",
                }),
                subscribers,
                scope_stack.clone(),
            );
            let completion = GuardrailScopeCompletion::new(handle, subscribers, scope_stack);
            let callback = Arc::clone(&entry.payload);
            let callback_request = request.clone();
            let result = match AssertUnwindSafe(async move { callback(callback_request).await })
                .catch_unwind()
                .await
            {
                Ok(result) => result,
                Err(_) => Err(FlowError::Internal(format!(
                    "LLM conditional guardrail '{}' panicked",
                    entry.name
                ))),
            };
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
            completion.finish(output);
            if let Some(error) = result? {
                return Ok(Some(error));
            }
        }
        Ok(None)
    }

    /// Snapshot LLM request intercepts in priority order.
    ///
    /// # Parameters
    /// - `scope_locals`: Scope-local request intercept registries collected
    ///   from the active scope stack.
    ///
    /// # Returns
    /// Named intercept snapshots that can be evaluated after registry locks
    /// are released.
    pub(crate) fn llm_request_intercept_entries(
        &self,
        scope_locals: &[&SortedRegistry<Intercept<LlmRequestInterceptFn>>],
    ) -> Vec<Intercept<LlmRequestInterceptFn>> {
        merge_intercept_entries(
            &self.llm_request_intercepts,
            scope_locals,
            RuntimeRegistrationKind::LlmRequestIntercept,
        )
        .into_iter()
        .cloned()
        .collect()
    }

    /// Run a snapshot of LLM request intercepts in priority order.
    ///
    /// # Parameters
    /// - `name`: Logical provider or model family name.
    /// - `request`: LLM request to pass through the intercept chain.
    /// - `annotated`: Optional normalized request annotation to carry through
    ///   the chain.
    /// - `entries`: Intercept snapshots to evaluate.
    /// - `codec_active`: Whether request content is owned by the normalized
    ///   annotation and must remain unchanged by callbacks.
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
    pub(crate) async fn llm_request_intercepts_snapshot_chain(
        name: &str,
        request: LlmRequest,
        annotated: Option<AnnotatedLlmRequest>,
        entries: &[Intercept<LlmRequestInterceptFn>],
        codec_active: bool,
    ) -> crate::error::Result<crate::api::llm::LlmRequestInterceptOutcome> {
        Self::llm_request_intercepts_snapshot_chain_with_recorder(
            name,
            request,
            annotated,
            entries,
            codec_active,
            None,
        )
        .await
    }

    /// Run a request-intercept snapshot while ingesting optimization evidence
    /// directly into the managed call's bounded accumulator.
    pub(crate) async fn llm_request_intercepts_snapshot_chain_with_recorder(
        name: &str,
        request: LlmRequest,
        annotated: Option<AnnotatedLlmRequest>,
        entries: &[Intercept<LlmRequestInterceptFn>],
        codec_active: bool,
        optimization_recorder: Option<&crate::api::optimization::LlmOptimizationRecorder>,
    ) -> crate::error::Result<crate::api::llm::LlmRequestInterceptOutcome> {
        let mut request_value = request;
        let mut annotated_value = annotated;
        let mut pending_marks = Vec::new();
        let mut optimization_contributions = Vec::new();
        for entry in entries {
            let input_content = request_value.content.clone();
            let callback = Arc::clone(&entry.payload.callable);
            let callback_name = name.to_string();
            let outcome = match AssertUnwindSafe(async move {
                callback(callback_name, request_value, annotated_value).await
            })
            .catch_unwind()
            .await
            {
                Ok(result) => result?,
                Err(_) => {
                    return Err(FlowError::Internal(format!(
                        "LLM request intercept '{}' panicked",
                        entry.name
                    )));
                }
            };
            if codec_active && outcome.request.content != input_content {
                return Err(crate::error::FlowError::InvalidArgument(format!(
                    "LLM request intercept '{}' changed request.content while a request codec is active; modify annotated_request instead",
                    entry.name
                )));
            }
            if codec_active && outcome.annotated_request.is_none() {
                return Err(crate::error::FlowError::InvalidArgument(format!(
                    "LLM request intercept '{}' omitted annotated_request while a request codec is active",
                    entry.name
                )));
            }
            request_value = outcome.request;
            annotated_value = outcome.annotated_request;
            pending_marks.extend(outcome.pending_marks);
            if let Some(recorder) = optimization_recorder {
                recorder.record_all(outcome.optimization_contributions);
            } else {
                optimization_contributions.extend(outcome.optimization_contributions);
            }
            if entry.payload.break_chain {
                break;
            }
        }
        Ok(crate::api::llm::LlmRequestInterceptOutcome {
            request: request_value,
            annotated_request: annotated_value,
            pending_marks,
            optimization_contributions,
        })
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
        let matching = merge_execution_intercept_callables(
            &self.llm_execution_intercepts,
            scope_locals,
            RuntimeRegistrationKind::LlmExecutionIntercept,
        );
        let mut next = default_fn;
        let name = name.to_string();
        for (callable, _) in matching.into_iter().rev() {
            let current_next = next.clone();
            let current_name = name.clone();
            next = Arc::new(move |request| {
                let callable = callable.clone();
                let current_next = current_next.clone();
                let current_name = current_name.clone();
                Box::pin(async move {
                    let (continuation, continuation_guard) = MiddlewareContinuationLease::capture();
                    let raw_next: LlmExecutionNextFn = Arc::new(move |request| {
                        let invocation = continuation.begin();
                        let current_next = current_next.clone();
                        Box::pin(
                            async move { invocation?.invoke(move || current_next(request)).await },
                        )
                    });
                    let result = callable(&current_name, request, raw_next).await;
                    drop(continuation_guard);
                    result
                })
            });
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
            RuntimeRegistrationKind::LlmStreamExecutionIntercept,
        );
        let mut next = default_fn;
        let name = name.to_string();
        for (callable, _) in matching.into_iter().rev() {
            let current_next = next.clone();
            let current_name = name.clone();
            next = Arc::new(move |request| {
                let callable = callable.clone();
                let current_next = current_next.clone();
                let current_name = current_name.clone();
                Box::pin(async move {
                    let (continuation, continuation_guard) = MiddlewareContinuationLease::capture();
                    let raw_next: LlmStreamExecutionNextFn = Arc::new(move |request| {
                        let invocation = continuation.begin();
                        let current_next = current_next.clone();
                        Box::pin(async move {
                            let invocation = invocation?;
                            let context = invocation.context().clone();
                            let stream = invocation.invoke(move || current_next(request)).await?;
                            Ok(contextualize_stream(stream, context))
                        })
                    });
                    let result = callable(&current_name, request, raw_next).await;
                    result.map(|stream| guard_stream_continuation(stream, continuation_guard))
                })
            });
        }
        next
    }
}

fn validate_event_metadata_attributes(attributes: &BTreeMap<String, Json>) -> Result<(), String> {
    for (key, value) in attributes {
        if !is_valid_event_metadata_attribute_key(key) {
            return Err(format!(
                "attribute key `{key}` must contain letter, number, underscore, or hyphen segments separated by single dots"
            ));
        }
        if !is_otel_compatible_attribute_value(value) {
            return Err(format!(
                "attribute `{key}` must be a primitive or homogeneous primitive list"
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OtelAttributePrimitiveKind {
    Boolean,
    Number,
    String,
}

fn otel_compatible_attribute_number_kind(
    value: &serde_json::Number,
) -> Option<OtelAttributePrimitiveKind> {
    if let Some(value) = value.as_u64() {
        return i64::try_from(value)
            .is_ok()
            .then_some(OtelAttributePrimitiveKind::Number);
    }
    value.as_f64().map(|_| OtelAttributePrimitiveKind::Number)
}

fn is_otel_compatible_attribute_number(value: &serde_json::Number) -> bool {
    otel_compatible_attribute_number_kind(value).is_some()
}

fn is_otel_compatible_attribute_value(value: &Json) -> bool {
    fn primitive_kind(value: &Json) -> Option<OtelAttributePrimitiveKind> {
        match value {
            Json::Bool(_) => Some(OtelAttributePrimitiveKind::Boolean),
            Json::Number(value) => otel_compatible_attribute_number_kind(value),
            Json::String(_) => Some(OtelAttributePrimitiveKind::String),
            _ => None,
        }
    }

    match value {
        Json::Bool(_) | Json::String(_) => true,
        Json::Number(value) => is_otel_compatible_attribute_number(value),
        Json::Array(values) => match values.first().and_then(primitive_kind) {
            None => values.is_empty(),
            Some(kind) => values
                .iter()
                .all(|value| primitive_kind(value) == Some(kind)),
        },
        Json::Null | Json::Object(_) => false,
    }
}

fn end_timestamp_after(started_at: chrono::DateTime<Utc>) -> chrono::DateTime<Utc> {
    let now = Utc::now();
    std::cmp::max(now, started_at + Duration::microseconds(1))
}

impl Default for NemoRelayContextState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/runtime_state_tests.rs"]
mod tests;
