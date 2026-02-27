// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global context, scope stack, and middleware chain execution.
//!
//! This module contains:
//!
//! - **Callable type aliases** — function signature types for all guardrails, intercepts,
//!   execution functions, and event subscribers.
//! - **[`ScopeStack`]** — a stack of [`ScopeHandle`]s with
//!   an immovable root scope.
//! - **Task-local and thread-local scope storage** — [`TASK_SCOPE_STACK`] for async
//!   contexts, with a thread-local fallback for synchronous code.
//! - **[`NVAgentRTContextState`]** — the central state object holding all registered
//!   middleware and subscribers, plus methods for chain execution and handle lifecycle.
//! - **[`global_context`]** — returns the process-wide singleton context, lazily
//!   initialized on first access.

use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use tokio_stream::Stream;
use uuid::Uuid;

use crate::error::{AgentRtError, Result};
use crate::json::{merge_json, Json};
use crate::registry::SortedRegistry;
use crate::types::*;

// ---------------------------------------------------------------------------
// Callable type aliases
// ---------------------------------------------------------------------------

/// Tool request/response sanitizer: `(tool_name, args_or_result) -> transformed`.
pub type ToolSanitizeFn = Box<dyn Fn(&str, Json) -> Json + Send + Sync>;
/// Tool conditional execution guardrail: `(tool_name, args) -> Option<rejection_reason>`.
pub type ToolConditionalFn = Box<dyn Fn(&str, &Json) -> Option<String> + Send + Sync>;
/// Tool request/response intercept: `(tool_name, value) -> transformed`.
pub type ToolInterceptFn = Box<dyn Fn(&str, Json) -> Json + Send + Sync>;
/// Tool execution function: `(args) -> Future<Result<Json>>`.
pub type ToolExecutionFn =
    Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync>;
/// Tool execution conditional: `(tool_name, args) -> should_intercept`.
pub type ToolExecutionConditionalFn = Box<dyn Fn(&str, &Json) -> bool + Send + Sync>;

/// LLM request sanitizer: `(request) -> sanitized_request`.
pub type LlmSanitizeRequestFn = Box<dyn Fn(LLMRequest) -> LLMRequest + Send + Sync>;
/// LLM response sanitizer: `(response) -> sanitized_response`.
pub type LlmSanitizeResponseFn = Box<dyn Fn(Json) -> Json + Send + Sync>;
/// LLM conditional execution guardrail: `(request) -> Option<rejection_reason>`.
pub type LlmConditionalFn = Box<dyn Fn(&LLMRequest) -> Option<String> + Send + Sync>;
/// LLM request intercept: `(request) -> transformed_request`.
pub type LlmRequestInterceptFn = Box<dyn Fn(LLMRequest) -> LLMRequest + Send + Sync>;
/// LLM response intercept: `(response) -> transformed_response`.
pub type LlmResponseInterceptFn = Box<dyn Fn(Json) -> Json + Send + Sync>;
/// LLM streaming response intercept: `(chunk) -> transformed_chunk`.
pub type LlmStreamResponseInterceptFn = Box<dyn Fn(String) -> String + Send + Sync>;
/// LLM execution function: `(request) -> Future<Result<Json>>`.
pub type LlmExecutionFn =
    Box<dyn Fn(LLMRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync>;
/// LLM execution conditional: `(request) -> should_intercept`.
pub type LlmExecutionConditionalFn = Box<dyn Fn(&LLMRequest) -> bool + Send + Sync>;
/// LLM streaming execution function: `(request) -> Future<Result<Stream<Item = Result<String>>>>`.
pub type LlmStreamExecutionFn = Box<
    dyn Fn(
            LLMRequest,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>>
                    + Send,
            >,
        > + Send
        + Sync,
>;
/// LLM streaming execution conditional: `(request) -> should_intercept`.
pub type LlmStreamExecutionConditionalFn = Box<dyn Fn(&LLMRequest) -> bool + Send + Sync>;

/// Event subscriber callback: `(event) -> ()`. Called for every lifecycle event.
pub type EventSubscriberFn = Box<dyn Fn(&Event) + Send + Sync>;

// ---------------------------------------------------------------------------
// Scope stack
// ---------------------------------------------------------------------------

/// A stack of [`ScopeHandle`]s representing the current execution hierarchy.
///
/// The stack is initialized with an immovable root scope (name `"root"`,
/// type [`ScopeType::Agent`]). Scopes are pushed when entering a new
/// execution context and removed by UUID when exiting. The root scope
/// at index 0 can never be removed.
#[derive(Debug, Clone)]
pub struct ScopeStack {
    stack: Vec<ScopeHandle>,
}

impl ScopeStack {
    /// Creates a new `ScopeStack` with a root scope that has an auto-generated UUID.
    ///
    /// The root scope is always present and cannot be removed.
    pub fn new() -> Self {
        let root = ScopeHandle::new(
            "root".to_string(),
            ScopeType::Agent,
            ScopeAttributes::empty(),
            None,
        );
        Self { stack: vec![root] }
    }

    /// Pushes a new scope handle onto the top of the stack.
    pub fn push(&mut self, handle: ScopeHandle) {
        self.stack.push(handle);
    }

    /// Returns a reference to the top scope. Always returns Some because the root is always present.
    pub fn top(&self) -> &ScopeHandle {
        self.stack
            .last()
            .expect("scope stack should never be empty")
    }

    /// Removes a scope by UUID and returns it, or `None` if not found or if
    /// the UUID belongs to the root scope (which cannot be removed).
    pub fn remove(&mut self, uuid: &Uuid) -> Option<ScopeHandle> {
        // Never remove the root (index 0)
        if let Some(pos) = self.stack.iter().position(|h| h.uuid == *uuid) {
            if pos == 0 {
                return None; // cannot remove root
            }
            Some(self.stack.remove(pos))
        } else {
            None
        }
    }
}

impl Default for ScopeStack {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Scope stack handle (shareable, isolated reference)
// ---------------------------------------------------------------------------

/// Opaque, shareable reference to an isolated scope stack.
///
/// Each `ScopeStackHandle` wraps an independent [`ScopeStack`] behind an
/// `Arc<RwLock<...>>`, allowing it to be shared across async boundaries and
/// threads while maintaining per-request/per-task isolation.
pub type ScopeStackHandle = Arc<std::sync::RwLock<ScopeStack>>;

/// Creates a new isolated scope stack (with its own root scope) and returns
/// a shareable handle to it.
pub fn create_scope_stack() -> ScopeStackHandle {
    Arc::new(std::sync::RwLock::new(ScopeStack::new()))
}

// ---------------------------------------------------------------------------
// Task-local + thread-local scope storage
// ---------------------------------------------------------------------------

tokio::task_local! {
    /// Task-local scope stack handle, set via `TASK_SCOPE_STACK.scope(...)`.
    /// Preferred over the thread-local fallback in async contexts.
    pub static TASK_SCOPE_STACK: ScopeStackHandle;
}

thread_local! {
    /// Thread-local scope stack handle, used as a fallback when no task-local
    /// scope is set. Each thread gets its own independent scope stack by default.
    static THREAD_SCOPE_STACK: RefCell<ScopeStackHandle> = RefCell::new(create_scope_stack());
}

/// Returns the [`ScopeStackHandle`] for the current execution context.
///
/// Checks the task-local first; falls back to the thread-local if no task-local
/// scope is set. The returned handle can be stored and later bound to other
/// tasks or threads via [`TASK_SCOPE_STACK`] or [`set_thread_scope_stack`].
pub fn current_scope_stack() -> ScopeStackHandle {
    TASK_SCOPE_STACK
        .try_with(|s| s.clone())
        .unwrap_or_else(|_| THREAD_SCOPE_STACK.with(|s| s.borrow().clone()))
}

/// Binds a specific [`ScopeStackHandle`] to the current thread's thread-local storage.
///
/// This is primarily used by FFI consumers (e.g. Go goroutines) that need to
/// pin a particular scope stack to an OS thread before making API calls.
pub fn set_thread_scope_stack(handle: ScopeStackHandle) {
    THREAD_SCOPE_STACK.with(|s| *s.borrow_mut() = handle);
}

/// Returns a clone of the top scope handle from the current execution context.
///
/// Checks the task-local stack first; falls back to the thread-local stack if
/// no task-local scope is set. Always succeeds because the root scope is always present.
pub fn task_scope_top() -> ScopeHandle {
    let stack = current_scope_stack();
    let guard = stack.read().expect("scope stack lock poisoned");
    guard.top().clone()
}

/// Pushes a scope handle onto the current execution context's scope stack.
///
/// Uses the task-local stack if available, otherwise falls back to the thread-local stack.
pub fn task_scope_push(handle: ScopeHandle) {
    let stack = current_scope_stack();
    let mut guard = stack.write().expect("scope stack lock poisoned");
    guard.push(handle);
}

/// Removes a scope handle by UUID from the current execution context's scope stack.
///
/// Returns the removed handle on success, or [`AgentRtError::NotFound`] if the
/// UUID is not in the stack (or refers to the immovable root scope).
pub fn task_scope_remove(uuid: &Uuid) -> Result<ScopeHandle> {
    let stack = current_scope_stack();
    let mut guard = stack.write().expect("scope stack lock poisoned");
    guard
        .remove(uuid)
        .ok_or_else(|| AgentRtError::NotFound("scope handle not found".into()))
}

// ---------------------------------------------------------------------------
// Context state
// ---------------------------------------------------------------------------

/// The central state object holding all registered middleware and event subscribers.
///
/// This struct contains sorted registries for every category of guardrail and
/// intercept (tool and LLM, request and response), as well as event subscribers.
/// It also provides methods for running middleware chains and managing handle
/// lifecycle events.
///
/// In production use, a single instance is held behind the [`global_context`]
/// singleton (`Arc<RwLock<NVAgentRTContextState>>`).
pub struct NVAgentRTContextState {
    /// Registry of tool request sanitize guardrails.
    pub tool_sanitize_request_guardrails: SortedRegistry<GuardrailEntry<ToolSanitizeFn>>,
    /// Registry of tool response sanitize guardrails.
    pub tool_sanitize_response_guardrails: SortedRegistry<GuardrailEntry<ToolSanitizeFn>>,
    /// Registry of tool conditional execution guardrails.
    pub tool_conditional_execution_guardrails: SortedRegistry<GuardrailEntry<ToolConditionalFn>>,

    /// Registry of tool request intercepts.
    pub tool_request_intercepts: SortedRegistry<Intercept<ToolInterceptFn>>,
    /// Registry of tool response intercepts.
    pub tool_response_intercepts: SortedRegistry<Intercept<ToolInterceptFn>>,
    /// Registry of tool execution intercepts (conditionally replace execution).
    pub tool_execution_intercepts:
        SortedRegistry<ExecutionIntercept<ToolExecutionConditionalFn, ToolExecutionFn>>,

    /// Registry of LLM request sanitize guardrails.
    pub llm_sanitize_request_guardrails: SortedRegistry<GuardrailEntry<LlmSanitizeRequestFn>>,
    /// Registry of LLM response sanitize guardrails.
    pub llm_sanitize_response_guardrails: SortedRegistry<GuardrailEntry<LlmSanitizeResponseFn>>,
    /// Registry of LLM conditional execution guardrails.
    pub llm_conditional_execution_guardrails: SortedRegistry<GuardrailEntry<LlmConditionalFn>>,

    /// Registry of LLM request intercepts.
    pub llm_request_intercepts: SortedRegistry<Intercept<LlmRequestInterceptFn>>,
    /// Registry of LLM response intercepts.
    pub llm_response_intercepts: SortedRegistry<Intercept<LlmResponseInterceptFn>>,
    /// Registry of LLM streaming response intercepts (per-chunk).
    pub llm_stream_response_intercepts: SortedRegistry<Intercept<LlmStreamResponseInterceptFn>>,
    /// Registry of LLM execution intercepts (conditionally replace execution).
    pub llm_execution_intercepts:
        SortedRegistry<ExecutionIntercept<LlmExecutionConditionalFn, LlmExecutionFn>>,
    /// Registry of LLM streaming execution intercepts.
    pub llm_stream_execution_intercepts:
        SortedRegistry<ExecutionIntercept<LlmStreamExecutionConditionalFn, LlmStreamExecutionFn>>,

    /// Named event subscribers, keyed by subscriber name.
    pub event_subscribers: HashMap<String, EventSubscriberFn>,
}

impl NVAgentRTContextState {
    /// Creates a new context state with empty registries and no subscribers.
    pub fn new() -> Self {
        Self {
            tool_sanitize_request_guardrails: SortedRegistry::new(|e| e.priority),
            tool_sanitize_response_guardrails: SortedRegistry::new(|e| e.priority),
            tool_conditional_execution_guardrails: SortedRegistry::new(|e| e.priority),
            tool_request_intercepts: SortedRegistry::new(|e| e.priority),
            tool_response_intercepts: SortedRegistry::new(|e| e.priority),
            tool_execution_intercepts: SortedRegistry::new(|e| e.priority),
            llm_sanitize_request_guardrails: SortedRegistry::new(|e| e.priority),
            llm_sanitize_response_guardrails: SortedRegistry::new(|e| e.priority),
            llm_conditional_execution_guardrails: SortedRegistry::new(|e| e.priority),
            llm_request_intercepts: SortedRegistry::new(|e| e.priority),
            llm_response_intercepts: SortedRegistry::new(|e| e.priority),
            llm_stream_response_intercepts: SortedRegistry::new(|e| e.priority),
            llm_execution_intercepts: SortedRegistry::new(|e| e.priority),
            llm_stream_execution_intercepts: SortedRegistry::new(|e| e.priority),
            event_subscribers: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Event emission
    // -----------------------------------------------------------------------

    /// Dispatches an event to all registered subscribers.
    pub fn emit_event(&self, event: &Event) {
        for sub in self.event_subscribers.values() {
            sub(event);
        }
    }

    // -----------------------------------------------------------------------
    // Handle creation / destruction
    // -----------------------------------------------------------------------

    /// Creates and emits a standalone marker event (EventType::Mark).
    pub fn create_event(
        &self,
        name: &str,
        parent_uuid: Option<Uuid>,
        data: Option<Json>,
        metadata: Option<Json>,
    ) {
        let event = Event::new(
            parent_uuid,
            Uuid::new_v4(),
            Some(name.to_string()),
            data,
            metadata,
            None,
            EventType::Mark,
            None,
        );
        self.emit_event(&event);
    }

    /// Creates a new scope handle and emits a Start event.
    pub fn create_scope_handle(
        &self,
        name: &str,
        parent_uuid: Option<Uuid>,
        scope_type: ScopeType,
        attributes: ScopeAttributes,
    ) -> ScopeHandle {
        let handle = ScopeHandle::new(name.to_string(), scope_type, attributes, parent_uuid);
        let event = Event::new(
            handle.parent_uuid,
            handle.uuid,
            Some(handle.name.clone()),
            handle.data.clone(),
            handle.metadata.clone(),
            Some(HandleAttributes::Scope(handle.attributes)),
            EventType::Start,
            Some(handle.scope_type),
        );
        self.emit_event(&event);
        handle
    }

    /// Emits an End event for the given scope handle.
    pub fn end_scope_handle(&self, scope: &ScopeHandle) {
        let event = Event::new(
            scope.parent_uuid,
            scope.uuid,
            Some(scope.name.clone()),
            scope.data.clone(),
            scope.metadata.clone(),
            Some(HandleAttributes::Scope(scope.attributes)),
            EventType::End,
            Some(scope.scope_type),
        );
        self.emit_event(&event);
    }

    /// Creates a new tool handle and emits a Start event.
    pub fn create_tool_handle(
        &self,
        name: &str,
        parent_uuid: Option<Uuid>,
        attributes: ToolAttributes,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> ToolHandle {
        let handle = ToolHandle::new(name.to_string(), attributes, parent_uuid, data, metadata);
        let event = Event::new(
            handle.parent_uuid,
            handle.uuid,
            Some(handle.name.clone()),
            handle.data.clone(),
            handle.metadata.clone(),
            Some(HandleAttributes::Tool(handle.attributes)),
            EventType::Start,
            Some(ScopeType::Tool),
        );
        self.emit_event(&event);
        handle
    }

    /// Emits an End event for the given tool handle, merging any additional data/metadata.
    pub fn end_tool_handle(&self, handle: &ToolHandle, data: Option<Json>, metadata: Option<Json>) {
        let event = Event::new(
            handle.parent_uuid,
            handle.uuid,
            Some(handle.name.clone()),
            merge_json(handle.data.clone(), data),
            merge_json(handle.metadata.clone(), metadata),
            Some(HandleAttributes::Tool(handle.attributes)),
            EventType::End,
            Some(ScopeType::Tool),
        );
        self.emit_event(&event);
    }

    /// Creates a new LLM handle and emits a Start event.
    pub fn create_llm_handle(
        &self,
        name: &str,
        parent_uuid: Option<Uuid>,
        attributes: LLMAttributes,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> LLMHandle {
        let handle = LLMHandle::new(name.to_string(), attributes, parent_uuid, data, metadata);
        let event = Event::new(
            handle.parent_uuid,
            handle.uuid,
            Some(handle.name.clone()),
            handle.data.clone(),
            handle.metadata.clone(),
            Some(HandleAttributes::Llm(handle.attributes)),
            EventType::Start,
            Some(ScopeType::Llm),
        );
        self.emit_event(&event);
        handle
    }

    /// Emits an End event for the given LLM handle, merging any additional data/metadata.
    pub fn end_llm_handle(&self, handle: &LLMHandle, data: Option<Json>, metadata: Option<Json>) {
        let event = Event::new(
            handle.parent_uuid,
            handle.uuid,
            Some(handle.name.clone()),
            merge_json(handle.data.clone(), data),
            merge_json(handle.metadata.clone(), metadata),
            Some(HandleAttributes::Llm(handle.attributes)),
            EventType::End,
            Some(ScopeType::Llm),
        );
        self.emit_event(&event);
    }

    // -----------------------------------------------------------------------
    // Chain runners
    // -----------------------------------------------------------------------

    /// Sanitize chain: run each guardrail in order, piping the value through.
    pub fn run_sanitize_chain<F, V>(registry: &mut SortedRegistry<GuardrailEntry<F>>, value: V) -> V
    where
        F: Fn(V) -> V,
    {
        let mut v = value;
        for entry in registry.sorted_values() {
            v = (entry.guardrail)(v);
        }
        v
    }

    /// Conditional chain: return the first error, or None.
    pub fn run_conditional_chain<F, V>(
        registry: &mut SortedRegistry<GuardrailEntry<F>>,
        value: &V,
    ) -> Option<String>
    where
        F: Fn(&V) -> Option<String>,
    {
        for entry in registry.sorted_values() {
            if let Some(err) = (entry.guardrail)(value) {
                return Some(err);
            }
        }
        None
    }

    /// Intercept chain: run each intercept, break if break_chain is set.
    pub fn run_intercept_chain<F, V>(registry: &mut SortedRegistry<Intercept<F>>, value: V) -> V
    where
        F: Fn(V) -> V,
    {
        let mut v = value;
        for entry in registry.sorted_values() {
            v = (entry.callable)(v);
            if entry.break_chain {
                break;
            }
        }
        v
    }

    // -----------------------------------------------------------------------
    // Tool chain methods
    // -----------------------------------------------------------------------

    /// Runs the tool request sanitize guardrail chain, piping args through each guardrail.
    pub fn tool_sanitize_request_chain(&mut self, name: &str, args: Json) -> Json {
        let mut v = args;
        for entry in self.tool_sanitize_request_guardrails.sorted_values() {
            v = (entry.guardrail)(name, v);
        }
        v
    }

    /// Runs the tool response sanitize guardrail chain, piping the result through each guardrail.
    pub fn tool_sanitize_response_chain(&mut self, name: &str, result: Json) -> Json {
        let mut v = result;
        for entry in self.tool_sanitize_response_guardrails.sorted_values() {
            v = (entry.guardrail)(name, v);
        }
        v
    }

    /// Runs the tool conditional execution guardrail chain. Returns the first rejection reason, or `None` if all pass.
    pub fn tool_conditional_execution_chain(&mut self, name: &str, args: &Json) -> Option<String> {
        for entry in self.tool_conditional_execution_guardrails.sorted_values() {
            if let Some(err) = (entry.guardrail)(name, args) {
                return Some(err);
            }
        }
        None
    }

    /// Runs the tool request intercept chain, piping args through each intercept (with optional break).
    pub fn tool_request_intercepts_chain(&mut self, name: &str, args: Json) -> Json {
        let mut v = args;
        for entry in self.tool_request_intercepts.sorted_values() {
            v = (entry.callable)(name, v);
            if entry.break_chain {
                break;
            }
        }
        v
    }

    /// Runs the tool response intercept chain, piping the result through each intercept (with optional break).
    pub fn tool_response_intercepts_chain(&mut self, name: &str, result: Json) -> Json {
        let mut v = result;
        for entry in self.tool_response_intercepts.sorted_values() {
            v = (entry.callable)(name, v);
            if entry.break_chain {
                break;
            }
        }
        v
    }

    /// Find the matching tool execution intercept, or return None to use default.
    /// Returns true if an intercept was found and its index.
    pub fn tool_find_execution_intercept(&mut self, name: &str, args: &Json) -> bool {
        for entry in self.tool_execution_intercepts.sorted_values() {
            if (entry.conditional)(name, args) {
                return true;
            }
        }
        false
    }

    /// Call the matching tool execution intercept. Must be called after tool_find_execution_intercept returned true.
    pub fn tool_call_execution_intercept(
        &mut self,
        name: &str,
        args: Json,
    ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> {
        for entry in self.tool_execution_intercepts.sorted_values() {
            if (entry.conditional)(name, &args) {
                return (entry.callable)(args);
            }
        }
        unreachable!()
    }

    // -----------------------------------------------------------------------
    // LLM chain methods
    // -----------------------------------------------------------------------

    /// Runs the LLM request sanitize guardrail chain, piping the request through each guardrail.
    pub fn llm_sanitize_request_chain(&mut self, request: LLMRequest) -> LLMRequest {
        let mut v = request;
        for entry in self.llm_sanitize_request_guardrails.sorted_values() {
            v = (entry.guardrail)(v);
        }
        v
    }

    /// Runs the LLM response sanitize guardrail chain, piping the response through each guardrail.
    pub fn llm_sanitize_response_chain(&mut self, response: Json) -> Json {
        let mut v = response;
        for entry in self.llm_sanitize_response_guardrails.sorted_values() {
            v = (entry.guardrail)(v);
        }
        v
    }

    /// Runs the LLM conditional execution guardrail chain. Returns the first rejection reason, or `None` if all pass.
    pub fn llm_conditional_execution_chain(&mut self, request: &LLMRequest) -> Option<String> {
        for entry in self.llm_conditional_execution_guardrails.sorted_values() {
            if let Some(err) = (entry.guardrail)(request) {
                return Some(err);
            }
        }
        None
    }

    /// Runs the LLM request intercept chain, piping the request through each intercept (with optional break).
    pub fn llm_request_intercepts_chain(&mut self, request: LLMRequest) -> LLMRequest {
        let mut v = request;
        for entry in self.llm_request_intercepts.sorted_values() {
            v = (entry.callable)(v);
            if entry.break_chain {
                break;
            }
        }
        v
    }

    /// Runs the LLM response intercept chain, piping the response through each intercept (with optional break).
    pub fn llm_response_intercepts_chain(&mut self, response: Json) -> Json {
        let mut v = response;
        for entry in self.llm_response_intercepts.sorted_values() {
            v = (entry.callable)(v);
            if entry.break_chain {
                break;
            }
        }
        v
    }

    /// Runs the LLM stream response intercept chain on a single chunk string.
    pub fn llm_stream_response_intercepts_chain(&mut self, chunk: String) -> String {
        let mut v = chunk;
        for entry in self.llm_stream_response_intercepts.sorted_values() {
            v = (entry.callable)(v);
            if entry.break_chain {
                break;
            }
        }
        v
    }

    /// Checks if any LLM execution intercept matches the given request.
    pub fn llm_find_execution_intercept(&mut self, request: &LLMRequest) -> bool {
        for entry in self.llm_execution_intercepts.sorted_values() {
            if (entry.conditional)(request) {
                return true;
            }
        }
        false
    }

    /// Invokes the first matching LLM execution intercept. Must be called after `llm_find_execution_intercept` returned `true`.
    pub fn llm_call_execution_intercept(
        &mut self,
        request: LLMRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> {
        for entry in self.llm_execution_intercepts.sorted_values() {
            if (entry.conditional)(&request) {
                return (entry.callable)(request);
            }
        }
        unreachable!()
    }

    /// Checks if any LLM streaming execution intercept matches the given request.
    pub fn llm_stream_find_execution_intercept(&mut self, request: &LLMRequest) -> bool {
        for entry in self.llm_stream_execution_intercepts.sorted_values() {
            if (entry.conditional)(request) {
                return true;
            }
        }
        false
    }

    /// Invokes the first matching LLM streaming execution intercept. Must be called after `llm_stream_find_execution_intercept` returned `true`.
    #[allow(clippy::type_complexity)]
    pub fn llm_stream_call_execution_intercept(
        &mut self,
        request: LLMRequest,
    ) -> Pin<
        Box<dyn Future<Output = Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>> + Send>,
    > {
        for entry in self.llm_stream_execution_intercepts.sorted_values() {
            if (entry.conditional)(&request) {
                return (entry.callable)(request);
            }
        }
        unreachable!()
    }
}

impl Default for NVAgentRTContextState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

static GLOBAL_CONTEXT: std::sync::OnceLock<Arc<RwLock<NVAgentRTContextState>>> =
    std::sync::OnceLock::new();

/// Returns the process-wide singleton [`NVAgentRTContextState`], lazily initialized.
///
/// The returned `Arc<RwLock<...>>` can be cloned cheaply and shared across threads.
/// All public API functions in [`crate::api`] use this internally.
pub fn global_context() -> Arc<RwLock<NVAgentRTContextState>> {
    GLOBAL_CONTEXT
        .get_or_init(|| Arc::new(RwLock::new(NVAgentRTContextState::new())))
        .clone()
}

#[cfg(test)]
#[allow(clippy::type_complexity)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_push_pop() {
        // Uses thread-local fallback (no tokio task context in #[test])
        // Root scope is always present
        assert_eq!(task_scope_top().name, "root");

        let ctx = NVAgentRTContextState::new();
        let handle =
            ctx.create_scope_handle("test", None, ScopeType::Agent, ScopeAttributes::empty());
        task_scope_push(handle.clone());
        let top = task_scope_top();
        assert_eq!(top.name, "test");

        let removed = task_scope_remove(&handle.uuid).unwrap();
        ctx.end_scope_handle(&removed);

        // After pop, root scope is on top again
        assert_eq!(task_scope_top().name, "root");
    }

    #[test]
    fn test_event_subscriber() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let mut ctx = NVAgentRTContextState::new();
        let count = Arc::new(AtomicU32::new(0));
        let count_clone = count.clone();

        ctx.event_subscribers.insert(
            "test_sub".into(),
            Box::new(move |_event: &Event| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );

        // create_scope_handle emits a START event
        let handle = ctx.create_scope_handle(
            "scope1",
            None,
            ScopeType::Function,
            ScopeAttributes::empty(),
        );
        assert_eq!(count.load(Ordering::SeqCst), 1);

        // end_scope_handle emits an END event
        ctx.end_scope_handle(&handle);
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    // -- ScopeStack tests --

    #[test]
    fn test_scope_stack_new_has_root() {
        let stack = ScopeStack::new();
        assert_eq!(stack.top().name, "root");
        assert_eq!(stack.top().scope_type, ScopeType::Agent);
        assert!(stack.top().parent_uuid.is_none());
    }

    #[test]
    fn test_scope_stack_default() {
        let stack = ScopeStack::default();
        assert_eq!(stack.top().name, "root");
    }

    #[test]
    fn test_scope_stack_push_changes_top() {
        let mut stack = ScopeStack::new();
        let handle = ScopeHandle::new(
            "child".into(),
            ScopeType::Function,
            ScopeAttributes::empty(),
            None,
        );
        let uuid = handle.uuid;
        stack.push(handle);
        assert_eq!(stack.top().name, "child");
        assert_eq!(stack.top().uuid, uuid);
    }

    #[test]
    fn test_scope_stack_remove_restores_top() {
        let mut stack = ScopeStack::new();
        let handle = ScopeHandle::new(
            "child".into(),
            ScopeType::Function,
            ScopeAttributes::empty(),
            None,
        );
        let uuid = handle.uuid;
        stack.push(handle);
        let removed = stack.remove(&uuid).unwrap();
        assert_eq!(removed.name, "child");
        assert_eq!(stack.top().name, "root");
    }

    #[test]
    fn test_scope_stack_cannot_remove_root() {
        let mut stack = ScopeStack::new();
        let root_uuid = stack.top().uuid;
        assert!(stack.remove(&root_uuid).is_none());
        assert_eq!(stack.top().name, "root");
    }

    #[test]
    fn test_scope_stack_remove_nonexistent() {
        let mut stack = ScopeStack::new();
        let random_uuid = Uuid::new_v4();
        assert!(stack.remove(&random_uuid).is_none());
    }

    #[test]
    fn test_scope_stack_multiple_push_pop() {
        let mut stack = ScopeStack::new();
        let h1 = ScopeHandle::new("a".into(), ScopeType::Agent, ScopeAttributes::empty(), None);
        let h2 = ScopeHandle::new(
            "b".into(),
            ScopeType::Function,
            ScopeAttributes::empty(),
            None,
        );
        let h3 = ScopeHandle::new("c".into(), ScopeType::Tool, ScopeAttributes::empty(), None);
        let u1 = h1.uuid;
        let u2 = h2.uuid;
        let u3 = h3.uuid;

        stack.push(h1);
        stack.push(h2);
        stack.push(h3);
        assert_eq!(stack.top().name, "c");

        stack.remove(&u3);
        assert_eq!(stack.top().name, "b");

        stack.remove(&u2);
        assert_eq!(stack.top().name, "a");

        stack.remove(&u1);
        assert_eq!(stack.top().name, "root");
    }

    #[test]
    fn test_scope_stack_remove_middle() {
        let mut stack = ScopeStack::new();
        let h1 = ScopeHandle::new("a".into(), ScopeType::Agent, ScopeAttributes::empty(), None);
        let h2 = ScopeHandle::new(
            "b".into(),
            ScopeType::Function,
            ScopeAttributes::empty(),
            None,
        );
        let u1 = h1.uuid;
        stack.push(h1);
        stack.push(h2);
        // Remove middle element
        stack.remove(&u1);
        assert_eq!(stack.top().name, "b");
    }

    // -- task_scope_remove error --

    #[test]
    fn test_task_scope_remove_nonexistent() {
        let result = task_scope_remove(&Uuid::new_v4());
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentRtError::NotFound(msg) => assert!(msg.contains("not found")),
            e => panic!("expected NotFound, got {e:?}"),
        }
    }

    // -- NVAgentRTContextState tests --

    #[test]
    fn test_context_state_new_empty() {
        let ctx = NVAgentRTContextState::new();
        assert!(ctx.event_subscribers.is_empty());
    }

    #[test]
    fn test_context_state_default() {
        let ctx = NVAgentRTContextState::default();
        assert!(ctx.event_subscribers.is_empty());
    }

    // -- Event emission tests --

    #[test]
    fn test_emit_event_multiple_subscribers() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let mut ctx = NVAgentRTContextState::new();
        let c1 = Arc::new(AtomicU32::new(0));
        let c2 = Arc::new(AtomicU32::new(0));
        let c1c = c1.clone();
        let c2c = c2.clone();

        ctx.event_subscribers.insert(
            "s1".into(),
            Box::new(move |_| {
                c1c.fetch_add(1, Ordering::SeqCst);
            }),
        );
        ctx.event_subscribers.insert(
            "s2".into(),
            Box::new(move |_| {
                c2c.fetch_add(1, Ordering::SeqCst);
            }),
        );

        let event = Event::new(
            None,
            Uuid::new_v4(),
            None,
            None,
            None,
            None,
            EventType::Mark,
            None,
        );
        ctx.emit_event(&event);

        assert_eq!(c1.load(Ordering::SeqCst), 1);
        assert_eq!(c2.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_emit_event_no_subscribers() {
        let ctx = NVAgentRTContextState::new();
        // Should not panic with no subscribers
        let event = Event::new(
            None,
            Uuid::new_v4(),
            None,
            None,
            None,
            None,
            EventType::Mark,
            None,
        );
        ctx.emit_event(&event);
    }

    // -- create_event tests --

    #[test]
    fn test_create_event_emits_mark() {
        use std::sync::Mutex;

        let mut ctx = NVAgentRTContextState::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();

        ctx.event_subscribers.insert(
            "capture".into(),
            Box::new(move |e: &Event| {
                events_clone.lock().unwrap().push(e.clone());
            }),
        );

        ctx.create_event(
            "my_mark",
            Some(Uuid::new_v4()),
            Some(serde_json::json!({"x": 1})),
            None,
        );

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].event_type, EventType::Mark);
        assert_eq!(captured[0].name, Some("my_mark".into()));
    }

    // -- Handle creation/destruction event tests --

    #[test]
    fn test_create_scope_handle_emits_start() {
        use std::sync::Mutex;

        let mut ctx = NVAgentRTContextState::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        ctx.event_subscribers.insert(
            "cap".into(),
            Box::new(move |e: &Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        );

        let handle = ctx.create_scope_handle(
            "sc",
            Some(Uuid::new_v4()),
            ScopeType::Retriever,
            ScopeAttributes::PARALLEL,
        );

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].event_type, EventType::Start);
        assert_eq!(captured[0].uuid, handle.uuid);
        assert_eq!(captured[0].scope_type, Some(ScopeType::Retriever));
        assert_eq!(
            captured[0].attributes,
            Some(HandleAttributes::Scope(ScopeAttributes::PARALLEL))
        );
    }

    #[test]
    fn test_end_scope_handle_emits_end() {
        use std::sync::Mutex;

        let mut ctx = NVAgentRTContextState::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        ctx.event_subscribers.insert(
            "cap".into(),
            Box::new(move |e: &Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        );

        let handle =
            ctx.create_scope_handle("sc", None, ScopeType::Agent, ScopeAttributes::empty());
        ctx.end_scope_handle(&handle);

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].event_type, EventType::Start);
        assert_eq!(captured[1].event_type, EventType::End);
        assert_eq!(captured[1].uuid, handle.uuid);
    }

    #[test]
    fn test_create_tool_handle_emits_start() {
        use std::sync::Mutex;

        let mut ctx = NVAgentRTContextState::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        ctx.event_subscribers.insert(
            "cap".into(),
            Box::new(move |e: &Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        );

        let _handle = ctx.create_tool_handle(
            "my_tool",
            None,
            ToolAttributes::LOCAL,
            Some(serde_json::json!({"k": "v"})),
            None,
        );

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].event_type, EventType::Start);
        assert_eq!(captured[0].scope_type, Some(ScopeType::Tool));
        assert_eq!(
            captured[0].attributes,
            Some(HandleAttributes::Tool(ToolAttributes::LOCAL))
        );
        assert_eq!(captured[0].name, Some("my_tool".into()));
    }

    #[test]
    fn test_end_tool_handle_merges_data() {
        use std::sync::Mutex;

        let mut ctx = NVAgentRTContextState::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        ctx.event_subscribers.insert(
            "cap".into(),
            Box::new(move |e: &Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        );

        let handle = ctx.create_tool_handle(
            "t",
            None,
            ToolAttributes::empty(),
            Some(serde_json::json!({"a": 1})),
            None,
        );
        ctx.end_tool_handle(&handle, Some(serde_json::json!({"b": 2})), None);

        let captured = events.lock().unwrap();
        let end_event = &captured[1];
        assert_eq!(end_event.event_type, EventType::End);
        // Data should be merged: {"a": 1, "b": 2}
        let data = end_event.data.as_ref().unwrap();
        assert_eq!(data["a"], 1);
        assert_eq!(data["b"], 2);
    }

    #[test]
    fn test_create_llm_handle_emits_start() {
        use std::sync::Mutex;

        let mut ctx = NVAgentRTContextState::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        ctx.event_subscribers.insert(
            "cap".into(),
            Box::new(move |e: &Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        );

        let _handle = ctx.create_llm_handle("llm", None, LLMAttributes::STREAMING, None, None);

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].scope_type, Some(ScopeType::Llm));
        assert_eq!(
            captured[0].attributes,
            Some(HandleAttributes::Llm(LLMAttributes::STREAMING))
        );
    }

    #[test]
    fn test_end_llm_handle_merges_metadata() {
        use std::sync::Mutex;

        let mut ctx = NVAgentRTContextState::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        ctx.event_subscribers.insert(
            "cap".into(),
            Box::new(move |e: &Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        );

        let handle = ctx.create_llm_handle(
            "llm",
            None,
            LLMAttributes::empty(),
            None,
            Some(serde_json::json!({"m1": true})),
        );
        ctx.end_llm_handle(&handle, None, Some(serde_json::json!({"m2": false})));

        let captured = events.lock().unwrap();
        let end_event = &captured[1];
        let meta = end_event.metadata.as_ref().unwrap();
        assert_eq!(meta["m1"], true);
        assert_eq!(meta["m2"], false);
    }

    // -- Chain runner tests --

    #[test]
    fn test_tool_sanitize_request_chain() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.tool_sanitize_request_guardrails
            .register(
                "g1".into(),
                GuardrailEntry {
                    priority: 10,
                    guardrail: Box::new(|_name: &str, mut args: Json| {
                        args.as_object_mut()
                            .unwrap()
                            .insert("sanitized".into(), serde_json::json!(true));
                        args
                    }),
                },
            )
            .unwrap();

        let result = ctx.tool_sanitize_request_chain("tool", serde_json::json!({"input": "data"}));
        assert_eq!(result["input"], "data");
        assert_eq!(result["sanitized"], true);
    }

    #[test]
    fn test_tool_sanitize_request_chain_multiple_priority_order() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.tool_sanitize_request_guardrails
            .register(
                "g2".into(),
                GuardrailEntry {
                    priority: 20,
                    guardrail: Box::new(|_name: &str, mut args: Json| {
                        args.as_object_mut()
                            .unwrap()
                            .insert("step".into(), serde_json::json!("second"));
                        args
                    }),
                },
            )
            .unwrap();
        ctx.tool_sanitize_request_guardrails
            .register(
                "g1".into(),
                GuardrailEntry {
                    priority: 10,
                    guardrail: Box::new(|_name: &str, mut args: Json| {
                        args.as_object_mut()
                            .unwrap()
                            .insert("step".into(), serde_json::json!("first"));
                        args
                    }),
                },
            )
            .unwrap();

        let result = ctx.tool_sanitize_request_chain("tool", serde_json::json!({}));
        // g1 runs first (priority 10), then g2 (priority 20) overwrites
        assert_eq!(result["step"], "second");
    }

    #[test]
    fn test_tool_sanitize_response_chain() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.tool_sanitize_response_guardrails
            .register(
                "g1".into(),
                GuardrailEntry {
                    priority: 1,
                    guardrail: Box::new(|_name: &str, mut result: Json| {
                        result
                            .as_object_mut()
                            .unwrap()
                            .insert("clean".into(), serde_json::json!(true));
                        result
                    }),
                },
            )
            .unwrap();

        let result = ctx.tool_sanitize_response_chain("tool", serde_json::json!({"output": "ok"}));
        assert_eq!(result["clean"], true);
        assert_eq!(result["output"], "ok");
    }

    #[test]
    fn test_tool_conditional_execution_chain_passes() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.tool_conditional_execution_guardrails
            .register(
                "g1".into(),
                GuardrailEntry {
                    priority: 1,
                    guardrail: Box::new(|_name: &str, _args: &Json| None),
                },
            )
            .unwrap();

        let result = ctx.tool_conditional_execution_chain("tool", &serde_json::json!({}));
        assert!(result.is_none());
    }

    #[test]
    fn test_tool_conditional_execution_chain_rejects() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.tool_conditional_execution_guardrails
            .register(
                "blocker".into(),
                GuardrailEntry {
                    priority: 1,
                    guardrail: Box::new(|_name: &str, _args: &Json| Some("not allowed".into())),
                },
            )
            .unwrap();

        let result = ctx.tool_conditional_execution_chain("tool", &serde_json::json!({}));
        assert_eq!(result, Some("not allowed".into()));
    }

    #[test]
    fn test_tool_conditional_first_rejection_wins() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.tool_conditional_execution_guardrails
            .register(
                "g1".into(),
                GuardrailEntry {
                    priority: 1,
                    guardrail: Box::new(|_name: &str, _args: &Json| Some("first".into())),
                },
            )
            .unwrap();
        ctx.tool_conditional_execution_guardrails
            .register(
                "g2".into(),
                GuardrailEntry {
                    priority: 2,
                    guardrail: Box::new(|_name: &str, _args: &Json| Some("second".into())),
                },
            )
            .unwrap();

        let result = ctx.tool_conditional_execution_chain("tool", &serde_json::json!({}));
        assert_eq!(result, Some("first".into()));
    }

    #[test]
    fn test_tool_request_intercepts_chain() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.tool_request_intercepts
            .register(
                "i1".into(),
                Intercept {
                    priority: 1,
                    break_chain: false,
                    callable: Box::new(|_name: &str, mut args: Json| {
                        args.as_object_mut()
                            .unwrap()
                            .insert("intercepted".into(), serde_json::json!(true));
                        args
                    }),
                },
            )
            .unwrap();

        let result =
            ctx.tool_request_intercepts_chain("tool", serde_json::json!({"original": true}));
        assert_eq!(result["original"], true);
        assert_eq!(result["intercepted"], true);
    }

    #[test]
    fn test_tool_request_intercepts_chain_break() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.tool_request_intercepts
            .register(
                "i1".into(),
                Intercept {
                    priority: 1,
                    break_chain: true,
                    callable: Box::new(|_name: &str, mut args: Json| {
                        args.as_object_mut()
                            .unwrap()
                            .insert("from_i1".into(), serde_json::json!(true));
                        args
                    }),
                },
            )
            .unwrap();
        ctx.tool_request_intercepts
            .register(
                "i2".into(),
                Intercept {
                    priority: 2,
                    break_chain: false,
                    callable: Box::new(|_name: &str, mut args: Json| {
                        args.as_object_mut()
                            .unwrap()
                            .insert("from_i2".into(), serde_json::json!(true));
                        args
                    }),
                },
            )
            .unwrap();

        let result = ctx.tool_request_intercepts_chain("tool", serde_json::json!({}));
        assert_eq!(result["from_i1"], true);
        // i2 should NOT have run due to break_chain
        assert!(result.get("from_i2").is_none());
    }

    #[test]
    fn test_tool_response_intercepts_chain() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.tool_response_intercepts
            .register(
                "i1".into(),
                Intercept {
                    priority: 1,
                    break_chain: false,
                    callable: Box::new(
                        |_name: &str, result: Json| serde_json::json!({"wrapped": result}),
                    ),
                },
            )
            .unwrap();

        let result = ctx.tool_response_intercepts_chain("tool", serde_json::json!("original"));
        assert_eq!(result["wrapped"], "original");
    }

    #[test]
    fn test_tool_find_execution_intercept_none() {
        let mut ctx = NVAgentRTContextState::new();
        assert!(!ctx.tool_find_execution_intercept("tool", &serde_json::json!({})));
    }

    #[test]
    fn test_tool_find_execution_intercept_conditional_false() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.tool_execution_intercepts
            .register(
                "ei1".into(),
                ExecutionIntercept {
                    priority: 1,
                    conditional: Box::new(|_name: &str, _args: &Json| false)
                        as ToolExecutionConditionalFn,
                    callable: Box::new(|_args: Json| {
                        Box::pin(async { Ok(serde_json::json!({})) })
                            as std::pin::Pin<
                                Box<dyn std::future::Future<Output = Result<Json>> + Send>,
                            >
                    }) as ToolExecutionFn,
                },
            )
            .unwrap();
        assert!(!ctx.tool_find_execution_intercept("tool", &serde_json::json!({})));
    }

    #[test]
    fn test_tool_find_execution_intercept_conditional_true() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.tool_execution_intercepts
            .register(
                "ei1".into(),
                ExecutionIntercept {
                    priority: 1,
                    conditional: Box::new(|_name: &str, _args: &Json| true)
                        as ToolExecutionConditionalFn,
                    callable: Box::new(|_args: Json| {
                        Box::pin(async { Ok(serde_json::json!({"intercepted": true})) })
                            as std::pin::Pin<
                                Box<dyn std::future::Future<Output = Result<Json>> + Send>,
                            >
                    }) as ToolExecutionFn,
                },
            )
            .unwrap();
        assert!(ctx.tool_find_execution_intercept("tool", &serde_json::json!({})));
    }

    // -- LLM chain methods --

    #[test]
    fn test_llm_sanitize_request_chain() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.llm_sanitize_request_guardrails
            .register(
                "g1".into(),
                GuardrailEntry {
                    priority: 1,
                    guardrail: Box::new(|mut req: LLMRequest| {
                        req.url = "https://sanitized.example.com".into();
                        req
                    }),
                },
            )
            .unwrap();

        let req = LLMRequest {
            method: "POST".into(),
            url: "https://original.example.com".into(),
            headers: serde_json::Map::new(),
            body: serde_json::json!({}),
        };
        let result = ctx.llm_sanitize_request_chain(req);
        assert_eq!(result.url, "https://sanitized.example.com");
        assert_eq!(result.method, "POST");
    }

    #[test]
    fn test_llm_sanitize_response_chain() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.llm_sanitize_response_guardrails
            .register(
                "g1".into(),
                GuardrailEntry {
                    priority: 1,
                    guardrail: Box::new(|mut resp: Json| {
                        resp.as_object_mut()
                            .unwrap()
                            .insert("sanitized".into(), serde_json::json!(true));
                        resp
                    }),
                },
            )
            .unwrap();

        let result = ctx.llm_sanitize_response_chain(serde_json::json!({"data": "test"}));
        assert_eq!(result["sanitized"], true);
    }

    #[test]
    fn test_llm_conditional_execution_chain_passes() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.llm_conditional_execution_guardrails
            .register(
                "g1".into(),
                GuardrailEntry {
                    priority: 1,
                    guardrail: Box::new(|_req: &LLMRequest| None),
                },
            )
            .unwrap();

        let req = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: serde_json::json!({}),
        };
        assert!(ctx.llm_conditional_execution_chain(&req).is_none());
    }

    #[test]
    fn test_llm_conditional_execution_chain_rejects() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.llm_conditional_execution_guardrails
            .register(
                "blocker".into(),
                GuardrailEntry {
                    priority: 1,
                    guardrail: Box::new(|_req: &LLMRequest| Some("blocked".into())),
                },
            )
            .unwrap();

        let req = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: serde_json::json!({}),
        };
        assert_eq!(
            ctx.llm_conditional_execution_chain(&req),
            Some("blocked".into())
        );
    }

    #[test]
    fn test_llm_request_intercepts_chain() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.llm_request_intercepts
            .register(
                "i1".into(),
                Intercept {
                    priority: 1,
                    break_chain: false,
                    callable: Box::new(|mut req: LLMRequest| {
                        req.headers
                            .insert("X-Intercepted".into(), serde_json::json!("true"));
                        req
                    }),
                },
            )
            .unwrap();

        let req = LLMRequest {
            method: "GET".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: serde_json::json!(null),
        };
        let result = ctx.llm_request_intercepts_chain(req);
        assert_eq!(result.headers["X-Intercepted"], "true");
    }

    #[test]
    fn test_llm_request_intercepts_break_chain() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.llm_request_intercepts
            .register(
                "i1".into(),
                Intercept {
                    priority: 1,
                    break_chain: true,
                    callable: Box::new(|mut req: LLMRequest| {
                        req.url = "https://intercepted1.com".into();
                        req
                    }),
                },
            )
            .unwrap();
        ctx.llm_request_intercepts
            .register(
                "i2".into(),
                Intercept {
                    priority: 2,
                    break_chain: false,
                    callable: Box::new(|mut req: LLMRequest| {
                        req.url = "https://intercepted2.com".into();
                        req
                    }),
                },
            )
            .unwrap();

        let req = LLMRequest {
            method: "POST".into(),
            url: "https://original.com".into(),
            headers: serde_json::Map::new(),
            body: serde_json::json!({}),
        };
        let result = ctx.llm_request_intercepts_chain(req);
        assert_eq!(result.url, "https://intercepted1.com");
    }

    #[test]
    fn test_llm_response_intercepts_chain() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.llm_response_intercepts
            .register(
                "i1".into(),
                Intercept {
                    priority: 1,
                    break_chain: false,
                    callable: Box::new(|mut resp: Json| {
                        resp.as_object_mut()
                            .unwrap()
                            .insert("modified".into(), serde_json::json!(true));
                        resp
                    }),
                },
            )
            .unwrap();

        let result = ctx.llm_response_intercepts_chain(serde_json::json!({"original": true}));
        assert_eq!(result["modified"], true);
        assert_eq!(result["original"], true);
    }

    #[test]
    fn test_llm_stream_response_intercepts_chain() {
        let mut ctx = NVAgentRTContextState::new();
        ctx.llm_stream_response_intercepts
            .register(
                "i1".into(),
                Intercept {
                    priority: 1,
                    break_chain: false,
                    callable: Box::new(|chunk: String| format!("modified: {}", chunk)),
                },
            )
            .unwrap();

        let result = ctx.llm_stream_response_intercepts_chain("original".to_string());
        assert_eq!(result, "modified: original");
    }

    #[test]
    fn test_llm_find_execution_intercept_none() {
        let mut ctx = NVAgentRTContextState::new();
        let req = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: serde_json::json!({}),
        };
        assert!(!ctx.llm_find_execution_intercept(&req));
    }

    #[test]
    fn test_llm_stream_find_execution_intercept_none() {
        let mut ctx = NVAgentRTContextState::new();
        let req = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: serde_json::json!({}),
        };
        assert!(!ctx.llm_stream_find_execution_intercept(&req));
    }

    // -- Generic chain runners --

    #[test]
    fn test_run_sanitize_chain_empty() {
        let mut reg: SortedRegistry<GuardrailEntry<Box<dyn Fn(i32) -> i32>>> =
            SortedRegistry::new(|e| e.priority);
        let result = NVAgentRTContextState::run_sanitize_chain(&mut reg, 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_run_sanitize_chain_multiple() {
        let mut reg: SortedRegistry<GuardrailEntry<Box<dyn Fn(i32) -> i32>>> =
            SortedRegistry::new(|e| e.priority);
        reg.register(
            "a".into(),
            GuardrailEntry {
                priority: 1,
                guardrail: Box::new(|x| x + 1),
            },
        )
        .unwrap();
        reg.register(
            "b".into(),
            GuardrailEntry {
                priority: 2,
                guardrail: Box::new(|x| x * 2),
            },
        )
        .unwrap();
        let result = NVAgentRTContextState::run_sanitize_chain(&mut reg, 5);
        // (5 + 1) * 2 = 12
        assert_eq!(result, 12);
    }

    #[test]
    fn test_run_conditional_chain_all_pass() {
        let mut reg: SortedRegistry<GuardrailEntry<Box<dyn Fn(&i32) -> Option<String>>>> =
            SortedRegistry::new(|e| e.priority);
        reg.register(
            "a".into(),
            GuardrailEntry {
                priority: 1,
                guardrail: Box::new(|_| None),
            },
        )
        .unwrap();
        reg.register(
            "b".into(),
            GuardrailEntry {
                priority: 2,
                guardrail: Box::new(|_| None),
            },
        )
        .unwrap();
        assert!(NVAgentRTContextState::run_conditional_chain(&mut reg, &42).is_none());
    }

    #[test]
    fn test_run_conditional_chain_first_fails() {
        let mut reg: SortedRegistry<GuardrailEntry<Box<dyn Fn(&i32) -> Option<String>>>> =
            SortedRegistry::new(|e| e.priority);
        reg.register(
            "a".into(),
            GuardrailEntry {
                priority: 1,
                guardrail: Box::new(|_| Some("err".into())),
            },
        )
        .unwrap();
        reg.register(
            "b".into(),
            GuardrailEntry {
                priority: 2,
                guardrail: Box::new(|_| None),
            },
        )
        .unwrap();
        assert_eq!(
            NVAgentRTContextState::run_conditional_chain(&mut reg, &42),
            Some("err".into())
        );
    }

    #[test]
    fn test_run_intercept_chain_no_break() {
        let mut reg: SortedRegistry<Intercept<Box<dyn Fn(i32) -> i32>>> =
            SortedRegistry::new(|e| e.priority);
        reg.register(
            "a".into(),
            Intercept {
                priority: 1,
                break_chain: false,
                callable: Box::new(|x| x + 10),
            },
        )
        .unwrap();
        reg.register(
            "b".into(),
            Intercept {
                priority: 2,
                break_chain: false,
                callable: Box::new(|x| x + 100),
            },
        )
        .unwrap();
        let result = NVAgentRTContextState::run_intercept_chain(&mut reg, 0);
        assert_eq!(result, 110);
    }

    #[test]
    fn test_run_intercept_chain_with_break() {
        let mut reg: SortedRegistry<Intercept<Box<dyn Fn(i32) -> i32>>> =
            SortedRegistry::new(|e| e.priority);
        reg.register(
            "a".into(),
            Intercept {
                priority: 1,
                break_chain: true,
                callable: Box::new(|x| x + 10),
            },
        )
        .unwrap();
        reg.register(
            "b".into(),
            Intercept {
                priority: 2,
                break_chain: false,
                callable: Box::new(|x| x + 100),
            },
        )
        .unwrap();
        let result = NVAgentRTContextState::run_intercept_chain(&mut reg, 0);
        // Only 'a' runs, 'b' is skipped
        assert_eq!(result, 10);
    }

    #[test]
    fn test_run_intercept_chain_empty() {
        let mut reg: SortedRegistry<Intercept<Box<dyn Fn(i32) -> i32>>> =
            SortedRegistry::new(|e| e.priority);
        let result = NVAgentRTContextState::run_intercept_chain(&mut reg, 42);
        assert_eq!(result, 42);
    }
}
