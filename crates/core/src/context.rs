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
//! - **[`NatNexusContextState`]** — the central state object holding all registered
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

use crate::error::{NexusError, Result};
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
/// Tool request intercept: `(tool_name, value) -> transformed`.
pub type ToolInterceptFn = Box<dyn Fn(&str, Json) -> Json + Send + Sync>;
/// Tool execution "next" function (FnOnce — each chain link is single-use):
/// `(args) -> Future<Result<Json>>`.
pub type ToolExecutionNextFn =
    Box<dyn FnOnce(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send>;
/// Tool execution intercept function: `(name, args, next) -> Future<Result<Json>>`.
/// Uses `Arc` because chain-building needs to clone it.
pub type ToolExecutionFn = Arc<
    dyn Fn(&str, Json, ToolExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
>;

/// LLM request sanitizer: `(request) -> sanitized_request`.
pub type LlmSanitizeRequestFn = Box<dyn Fn(LLMRequest) -> LLMRequest + Send + Sync>;
/// LLM response sanitizer: `(response) -> sanitized_response`.
pub type LlmSanitizeResponseFn = Box<dyn Fn(Json) -> Json + Send + Sync>;
/// LLM conditional execution guardrail: `(request) -> Option<rejection_reason>`.
pub type LlmConditionalFn = Box<dyn Fn(&LLMRequest) -> Option<String> + Send + Sync>;
/// LLM request intercept: `(name, request) -> transformed_request`.
pub type LlmRequestInterceptFn = Box<dyn Fn(&str, LLMRequest) -> LLMRequest + Send + Sync>;
/// LLM execution "next" function (FnOnce — each chain link is single-use):
/// `(request) -> Future<Result<Json>>`.
pub type LlmExecutionNextFn =
    Box<dyn FnOnce(LLMRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send>;
/// LLM execution intercept function: `(name, request, next) -> Future<Result<Json>>`.
/// Uses `Arc` because chain-building needs to clone it.
pub type LlmExecutionFn = Arc<
    dyn Fn(
            &str,
            LLMRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
>;
/// LLM streaming execution "next" function (FnOnce):
/// `(request) -> Future<Result<Stream<Item = Result<Json>>>>`.
pub type LlmStreamExecutionNextFn = Box<
    dyn FnOnce(
            LLMRequest,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Pin<Box<dyn Stream<Item = Result<Json>> + Send>>>>
                    + Send,
            >,
        > + Send,
>;
/// LLM streaming execution intercept function: `(name, request, next) -> Future<Result<Stream>>`.
/// Uses `Arc` because chain-building needs to clone it.
pub type LlmStreamExecutionFn = Arc<
    dyn Fn(
            &str,
            LLMRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Pin<Box<dyn Stream<Item = Result<Json>> + Send>>>>
                    + Send,
            >,
        > + Send
        + Sync,
>;

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
pub struct ScopeStack {
    stack: Vec<ScopeHandle>,
    /// Per-scope middleware registries, keyed by scope UUID. Lazily populated
    /// on first scope-local registration and automatically cleaned up on scope pop.
    scope_registries: HashMap<Uuid, ScopeLocalRegistries>,
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
            None,
            None,
        );
        Self {
            stack: vec![root],
            scope_registries: HashMap::new(),
        }
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

    /// Returns the UUID of the root (bottom-most) scope.
    ///
    /// This is O(1) — just reads the first element of the scope stack vec.
    /// Used for concurrent agent isolation (each scope stack has its own root).
    pub fn root_uuid(&self) -> Uuid {
        self.stack
            .first()
            .expect("scope stack should never be empty")
            .uuid
    }

    /// Finds a scope by UUID and returns a reference to it, or `None` if not found.
    pub fn find(&self, uuid: &Uuid) -> Option<&ScopeHandle> {
        self.stack.iter().find(|h| h.uuid == *uuid)
    }

    /// Removes a scope by UUID and returns it, or `None` if not found or if
    /// the UUID belongs to the root scope (which cannot be removed).
    ///
    /// Also removes any scope-local registries associated with the scope.
    pub fn remove(&mut self, uuid: &Uuid) -> Option<ScopeHandle> {
        // Never remove the root (index 0)
        if let Some(pos) = self.stack.iter().position(|h| h.uuid == *uuid) {
            if pos == 0 {
                return None; // cannot remove root
            }
            self.scope_registries.remove(uuid);
            Some(self.stack.remove(pos))
        } else {
            None
        }
    }

    /// Returns a mutable reference to the scope-local registries for the given
    /// scope UUID, creating a new empty set if none exists yet.
    ///
    /// Returns `None` if the UUID is not in the current scope stack.
    pub fn local_registries_mut(&mut self, uuid: &Uuid) -> Option<&mut ScopeLocalRegistries> {
        // Verify the scope UUID exists in the stack
        if !self.stack.iter().any(|h| h.uuid == *uuid) {
            return None;
        }
        Some(self.scope_registries.entry(*uuid).or_default())
    }

    /// Collects references to scope-local registries for a specific field,
    /// in stack order (root to top). Used by chain methods to merge with global.
    ///
    /// The closure `field` extracts the desired registry from each `ScopeLocalRegistries`.
    pub fn collect_scope_local_registries<'a, T>(
        &'a self,
        field: impl Fn(&'a ScopeLocalRegistries) -> &'a SortedRegistry<T>,
    ) -> Vec<&'a SortedRegistry<T>> {
        self.stack
            .iter()
            .filter_map(|h| self.scope_registries.get(&h.uuid))
            .map(field)
            .collect()
    }

    /// Collects scope-local event subscribers in stack order (root to top).
    pub fn collect_scope_local_subscribers(&self) -> Vec<&HashMap<String, EventSubscriberFn>> {
        self.stack
            .iter()
            .filter_map(|h| self.scope_registries.get(&h.uuid))
            .map(|r| &r.event_subscribers)
            .collect()
    }
}

impl std::fmt::Debug for ScopeStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopeStack")
            .field("stack", &self.stack)
            .field("scope_registries_count", &self.scope_registries.len())
            .finish()
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

    /// Tracks whether [`set_thread_scope_stack`] has been explicitly called on
    /// this thread. Used by [`scope_stack_active`] to distinguish an
    /// explicitly-bound scope stack from the auto-created default.
    static THREAD_SCOPE_STACK_EXPLICIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
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
///
/// After this call, [`scope_stack_active`] will return `true` for this thread.
pub fn set_thread_scope_stack(handle: ScopeStackHandle) {
    THREAD_SCOPE_STACK.with(|s| *s.borrow_mut() = handle);
    THREAD_SCOPE_STACK_EXPLICIT.with(|f| f.set(true));
}

/// Syncs a [`ScopeStackHandle`] to the current thread's thread-local storage
/// **without** marking it as explicitly set.
///
/// This is used internally by binding layers (e.g. Python's `get_scope_stack()`)
/// that need to keep the Rust thread-local in sync with a higher-level context
/// mechanism (e.g. `contextvars.ContextVar`) without affecting the
/// [`scope_stack_active`] flag.
pub fn sync_thread_scope_stack(handle: ScopeStackHandle) {
    THREAD_SCOPE_STACK.with(|s| *s.borrow_mut() = handle);
}

/// Returns whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns `true` if the caller is inside a tokio task with [`TASK_SCOPE_STACK`]
/// set, or if [`set_thread_scope_stack`] has been called on the current OS
/// thread. Returns `false` when only the auto-created default is present.
pub fn scope_stack_active() -> bool {
    TASK_SCOPE_STACK
        .try_with(|_| true)
        .unwrap_or_else(|_| THREAD_SCOPE_STACK_EXPLICIT.with(|f| f.get()))
}

/// Captures the current scope stack for propagation to a worker thread.
///
/// Returns a clone of the current [`ScopeStackHandle`] that can be passed to
/// [`set_thread_scope_stack`] on a worker thread. Fails if no scope stack has
/// been explicitly initialized in the current context (i.e.,
/// [`scope_stack_active`] returns `false`).
pub fn propagate_scope_to_thread() -> Result<ScopeStackHandle> {
    if !scope_stack_active() {
        return Err(NexusError::Internal(
            "no active scope stack in current context; \
             call create_scope_stack() and set_thread_scope_stack() first"
                .into(),
        ));
    }
    Ok(current_scope_stack())
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
/// Returns the removed handle on success, or [`NexusError::NotFound`] if the
/// UUID is not in the stack (or refers to the immovable root scope).
pub fn task_scope_remove(uuid: &Uuid) -> Result<ScopeHandle> {
    let stack = current_scope_stack();
    let mut guard = stack.write().expect("scope stack lock poisoned");
    guard
        .remove(uuid)
        .ok_or_else(|| NexusError::NotFound("scope handle not found".into()))
}

// ---------------------------------------------------------------------------
// Scope-local registries
// ---------------------------------------------------------------------------

/// Per-scope middleware registries. Mirrors the 11 sorted registries plus
/// event subscribers from [`NatNexusContextState`], but scoped to a single
/// scope in the stack. Created lazily on first scope-local registration and
/// dropped automatically when the scope is popped.
pub struct ScopeLocalRegistries {
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
}

impl ScopeLocalRegistries {
    /// Creates a new set of empty scope-local registries.
    pub fn new() -> Self {
        Self {
            tool_sanitize_request_guardrails: SortedRegistry::new(|e| e.priority),
            tool_sanitize_response_guardrails: SortedRegistry::new(|e| e.priority),
            tool_conditional_execution_guardrails: SortedRegistry::new(|e| e.priority),
            tool_request_intercepts: SortedRegistry::new(|e| e.priority),
            tool_execution_intercepts: SortedRegistry::new(|e| e.priority),
            llm_sanitize_request_guardrails: SortedRegistry::new(|e| e.priority),
            llm_sanitize_response_guardrails: SortedRegistry::new(|e| e.priority),
            llm_conditional_execution_guardrails: SortedRegistry::new(|e| e.priority),
            llm_request_intercepts: SortedRegistry::new(|e| e.priority),
            llm_execution_intercepts: SortedRegistry::new(|e| e.priority),
            llm_stream_execution_intercepts: SortedRegistry::new(|e| e.priority),
            event_subscribers: HashMap::new(),
        }
    }
}

impl Default for ScopeLocalRegistries {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Merge helpers — combine global + scope-local registries by priority
// ---------------------------------------------------------------------------

/// Collects sorted entries from multiple registries and merges them by priority.
///
/// Entries from the global registry and each scope-local registry are flattened
/// into a single list, then stable-sorted by priority (ascending). At equal
/// priority, global entries appear before scope-local entries (since global
/// is collected first), and ancestor-scope entries appear before descendant ones.
pub fn merge_guardrail_entries<'a, F>(
    global: &'a SortedRegistry<GuardrailEntry<F>>,
    scope_locals: &'a [&SortedRegistry<GuardrailEntry<F>>],
) -> Vec<&'a GuardrailEntry<F>> {
    let mut all: Vec<(&GuardrailEntry<F>, i32)> = Vec::new();
    for entry in global.sorted_values() {
        all.push((entry, entry.priority));
    }
    for reg in scope_locals {
        for entry in reg.sorted_values() {
            all.push((entry, entry.priority));
        }
    }
    all.sort_by_key(|&(_, p)| p);
    all.into_iter().map(|(e, _)| e).collect()
}

/// Collects sorted entries from multiple intercept registries and merges by priority.
pub fn merge_intercept_entries<'a, F>(
    global: &'a SortedRegistry<Intercept<F>>,
    scope_locals: &'a [&SortedRegistry<Intercept<F>>],
) -> Vec<&'a Intercept<F>> {
    let mut all: Vec<(&Intercept<F>, i32)> = Vec::new();
    for entry in global.sorted_values() {
        all.push((entry, entry.priority));
    }
    for reg in scope_locals {
        for entry in reg.sorted_values() {
            all.push((entry, entry.priority));
        }
    }
    all.sort_by_key(|&(_, p)| p);
    all.into_iter().map(|(e, _)| e).collect()
}

/// Collects sorted entries from multiple execution intercept registries, merged by priority.
/// Returns cloned `Arc<callable>`s ready for chain building.
pub fn merge_execution_intercept_callables<F: Clone>(
    global: &SortedRegistry<ExecutionIntercept<F>>,
    scope_locals: &[&SortedRegistry<ExecutionIntercept<F>>],
) -> Vec<(F, i32)> {
    let mut all: Vec<(F, i32)> = Vec::new();
    for entry in global.sorted_values() {
        all.push((entry.callable.clone(), entry.priority));
    }
    for reg in scope_locals {
        for entry in reg.sorted_values() {
            all.push((entry.callable.clone(), entry.priority));
        }
    }
    all.sort_by_key(|&(_, p)| p);
    all
}

// ---------------------------------------------------------------------------
// Context state
// ---------------------------------------------------------------------------

/// The central state object holding all registered middleware and event subscribers.
///
/// This struct contains sorted registries for every category of guardrail and
/// intercept (tool and LLM; request and execution intercepts), as well as event subscribers.
/// It also provides methods for running middleware chains and managing handle
/// lifecycle events.
///
/// In production use, a single instance is held behind the [`global_context`]
/// singleton (`Arc<RwLock<NatNexusContextState>>`).
pub struct NatNexusContextState {
    /// Registry of tool request sanitize guardrails.
    pub tool_sanitize_request_guardrails: SortedRegistry<GuardrailEntry<ToolSanitizeFn>>,
    /// Registry of tool response sanitize guardrails.
    pub tool_sanitize_response_guardrails: SortedRegistry<GuardrailEntry<ToolSanitizeFn>>,
    /// Registry of tool conditional execution guardrails.
    pub tool_conditional_execution_guardrails: SortedRegistry<GuardrailEntry<ToolConditionalFn>>,

    /// Registry of tool request intercepts.
    pub tool_request_intercepts: SortedRegistry<Intercept<ToolInterceptFn>>,
    /// Registry of tool execution intercepts (middleware chain pattern).
    pub tool_execution_intercepts: SortedRegistry<ExecutionIntercept<ToolExecutionFn>>,

    /// Registry of LLM request sanitize guardrails.
    pub llm_sanitize_request_guardrails: SortedRegistry<GuardrailEntry<LlmSanitizeRequestFn>>,
    /// Registry of LLM response sanitize guardrails.
    pub llm_sanitize_response_guardrails: SortedRegistry<GuardrailEntry<LlmSanitizeResponseFn>>,
    /// Registry of LLM conditional execution guardrails.
    pub llm_conditional_execution_guardrails: SortedRegistry<GuardrailEntry<LlmConditionalFn>>,

    /// Registry of LLM request intercepts.
    pub llm_request_intercepts: SortedRegistry<Intercept<LlmRequestInterceptFn>>,
    /// Registry of LLM execution intercepts (middleware chain pattern).
    pub llm_execution_intercepts: SortedRegistry<ExecutionIntercept<LlmExecutionFn>>,
    /// Registry of LLM streaming execution intercepts.
    pub llm_stream_execution_intercepts: SortedRegistry<ExecutionIntercept<LlmStreamExecutionFn>>,

    /// Named event subscribers, keyed by subscriber name.
    pub event_subscribers: HashMap<String, EventSubscriberFn>,
}

impl NatNexusContextState {
    /// Creates a new context state with empty registries and no subscribers.
    pub fn new() -> Self {
        Self {
            tool_sanitize_request_guardrails: SortedRegistry::new(|e| e.priority),
            tool_sanitize_response_guardrails: SortedRegistry::new(|e| e.priority),
            tool_conditional_execution_guardrails: SortedRegistry::new(|e| e.priority),
            tool_request_intercepts: SortedRegistry::new(|e| e.priority),
            tool_execution_intercepts: SortedRegistry::new(|e| e.priority),
            llm_sanitize_request_guardrails: SortedRegistry::new(|e| e.priority),
            llm_sanitize_response_guardrails: SortedRegistry::new(|e| e.priority),
            llm_conditional_execution_guardrails: SortedRegistry::new(|e| e.priority),
            llm_request_intercepts: SortedRegistry::new(|e| e.priority),
            llm_execution_intercepts: SortedRegistry::new(|e| e.priority),
            llm_stream_execution_intercepts: SortedRegistry::new(|e| e.priority),
            event_subscribers: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Event emission
    // -----------------------------------------------------------------------

    /// Dispatches an event to all registered subscribers (global + scope-local).
    pub fn emit_event(
        &self,
        event: &Event,
        scope_local_subscribers: &[&HashMap<String, EventSubscriberFn>],
    ) {
        for sub in self.event_subscribers.values() {
            sub(event);
        }
        for subs in scope_local_subscribers {
            for sub in subs.values() {
                sub(event);
            }
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
        root_uuid: Option<Uuid>,
        scope_local_subscribers: &[&HashMap<String, EventSubscriberFn>],
    ) {
        let event = Event::builder(Uuid::new_v4(), EventType::Mark)
            .parent_uuid(parent_uuid)
            .name(name)
            .data(data)
            .metadata(metadata)
            .root_uuid(root_uuid)
            .build();
        self.emit_event(&event, scope_local_subscribers);
    }

    /// Creates a new scope handle and emits a Start event.
    #[allow(clippy::too_many_arguments)]
    pub fn create_scope_handle(
        &self,
        name: &str,
        parent_uuid: Option<Uuid>,
        scope_type: ScopeType,
        attributes: ScopeAttributes,
        root_uuid: Option<Uuid>,
        data: Option<Json>,
        metadata: Option<Json>,
        scope_local_subscribers: &[&HashMap<String, EventSubscriberFn>],
    ) -> ScopeHandle {
        let handle = ScopeHandle::new(
            name.to_string(),
            scope_type,
            attributes,
            parent_uuid,
            data,
            metadata,
        );
        let event = Event::builder(handle.uuid, EventType::Start)
            .parent_uuid(handle.parent_uuid)
            .name(handle.name.clone())
            .data(handle.data.clone())
            .metadata(handle.metadata.clone())
            .attributes(HandleAttributes::Scope(handle.attributes))
            .scope_type(handle.scope_type)
            .root_uuid(root_uuid)
            .build();
        self.emit_event(&event, scope_local_subscribers);
        handle
    }

    /// Emits an End event for the given scope handle.
    pub fn end_scope_handle(
        &self,
        scope: &ScopeHandle,
        root_uuid: Option<Uuid>,
        scope_local_subscribers: &[&HashMap<String, EventSubscriberFn>],
    ) {
        let event = Event::builder(scope.uuid, EventType::End)
            .parent_uuid(scope.parent_uuid)
            .name(scope.name.clone())
            .data(scope.data.clone())
            .metadata(scope.metadata.clone())
            .attributes(HandleAttributes::Scope(scope.attributes))
            .scope_type(scope.scope_type)
            .root_uuid(root_uuid)
            .build();
        self.emit_event(&event, scope_local_subscribers);
    }

    /// Creates a new tool handle and emits a Start event.
    ///
    /// The `input` field on the Start event is populated with the sanitized args.
    /// The `tool_call_id` is propagated from the handle to the event.
    /// The `root_uuid` is set from the current scope stack.
    #[allow(clippy::too_many_arguments)]
    pub fn create_tool_handle(
        &self,
        name: &str,
        parent_uuid: Option<Uuid>,
        attributes: ToolAttributes,
        data: Option<Json>,
        metadata: Option<Json>,
        tool_call_id: Option<String>,
        input: Option<Json>,
        root_uuid: Option<Uuid>,
        scope_local_subscribers: &[&HashMap<String, EventSubscriberFn>],
    ) -> ToolHandle {
        let mut handle = ToolHandle::new(name.to_string(), attributes, parent_uuid, data, metadata);
        handle.tool_call_id = tool_call_id;
        let event = Event::builder(handle.uuid, EventType::Start)
            .parent_uuid(handle.parent_uuid)
            .name(handle.name.clone())
            .data(handle.data.clone())
            .metadata(handle.metadata.clone())
            .attributes(HandleAttributes::Tool(handle.attributes))
            .scope_type(ScopeType::Tool)
            .input(input)
            .tool_call_id(handle.tool_call_id.clone())
            .root_uuid(root_uuid)
            .build();
        self.emit_event(&event, scope_local_subscribers);
        handle
    }

    /// Emits an End event for the given tool handle, merging any additional data/metadata.
    ///
    /// The `output` field on the End event is populated with the sanitized result.
    pub fn end_tool_handle(
        &self,
        handle: &ToolHandle,
        data: Option<Json>,
        metadata: Option<Json>,
        output: Option<Json>,
        root_uuid: Option<Uuid>,
        scope_local_subscribers: &[&HashMap<String, EventSubscriberFn>],
    ) {
        let event = Event::builder(handle.uuid, EventType::End)
            .parent_uuid(handle.parent_uuid)
            .name(handle.name.clone())
            .data(merge_json(handle.data.clone(), data))
            .metadata(merge_json(handle.metadata.clone(), metadata))
            .attributes(HandleAttributes::Tool(handle.attributes))
            .scope_type(ScopeType::Tool)
            .output(output)
            .tool_call_id(handle.tool_call_id.clone())
            .root_uuid(root_uuid)
            .build();
        self.emit_event(&event, scope_local_subscribers);
    }

    /// Creates a new LLM handle and emits a Start event.
    ///
    /// The `input` field on the Start event is populated with the sanitized request.
    /// The `model_name` is propagated from the handle to the event.
    /// The `root_uuid` is set from the current scope stack.
    #[allow(clippy::too_many_arguments)]
    pub fn create_llm_handle(
        &self,
        name: &str,
        parent_uuid: Option<Uuid>,
        attributes: LLMAttributes,
        data: Option<Json>,
        metadata: Option<Json>,
        model_name: Option<String>,
        input: Option<Json>,
        root_uuid: Option<Uuid>,
        scope_local_subscribers: &[&HashMap<String, EventSubscriberFn>],
    ) -> LLMHandle {
        let mut handle = LLMHandle::new(name.to_string(), attributes, parent_uuid, data, metadata);
        handle.model_name = model_name;
        let event = Event::builder(handle.uuid, EventType::Start)
            .parent_uuid(handle.parent_uuid)
            .name(handle.name.clone())
            .data(handle.data.clone())
            .metadata(handle.metadata.clone())
            .attributes(HandleAttributes::Llm(handle.attributes))
            .scope_type(ScopeType::Llm)
            .input(input)
            .model_name(handle.model_name.clone())
            .root_uuid(root_uuid)
            .build();
        self.emit_event(&event, scope_local_subscribers);
        handle
    }

    /// Emits an End event for the given LLM handle, merging any additional data/metadata.
    ///
    /// The `output` field on the End event is populated with the sanitized response.
    pub fn end_llm_handle(
        &self,
        handle: &LLMHandle,
        data: Option<Json>,
        metadata: Option<Json>,
        output: Option<Json>,
        root_uuid: Option<Uuid>,
        scope_local_subscribers: &[&HashMap<String, EventSubscriberFn>],
    ) {
        let event = Event::builder(handle.uuid, EventType::End)
            .parent_uuid(handle.parent_uuid)
            .name(handle.name.clone())
            .data(merge_json(handle.data.clone(), data))
            .metadata(merge_json(handle.metadata.clone(), metadata))
            .attributes(HandleAttributes::Llm(handle.attributes))
            .scope_type(ScopeType::Llm)
            .output(output)
            .model_name(handle.model_name.clone())
            .root_uuid(root_uuid)
            .build();
        self.emit_event(&event, scope_local_subscribers);
    }

    // -----------------------------------------------------------------------
    // Chain runners
    // -----------------------------------------------------------------------

    /// Sanitize chain: run each guardrail in order, piping the value through.
    pub fn run_sanitize_chain<F, V>(registry: &SortedRegistry<GuardrailEntry<F>>, value: V) -> V
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
        registry: &SortedRegistry<GuardrailEntry<F>>,
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
    pub fn run_intercept_chain<F, V>(registry: &SortedRegistry<Intercept<F>>, value: V) -> V
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
    /// Merges global + scope-local entries by priority.
    pub fn tool_sanitize_request_chain(
        &self,
        name: &str,
        args: Json,
        scope_locals: &[&SortedRegistry<GuardrailEntry<ToolSanitizeFn>>],
    ) -> Json {
        let entries = merge_guardrail_entries(&self.tool_sanitize_request_guardrails, scope_locals);
        let mut v = args;
        for entry in entries {
            v = (entry.guardrail)(name, v);
        }
        v
    }

    /// Runs the tool response sanitize guardrail chain, piping the result through each guardrail.
    /// Merges global + scope-local entries by priority.
    pub fn tool_sanitize_response_chain(
        &self,
        name: &str,
        result: Json,
        scope_locals: &[&SortedRegistry<GuardrailEntry<ToolSanitizeFn>>],
    ) -> Json {
        let entries =
            merge_guardrail_entries(&self.tool_sanitize_response_guardrails, scope_locals);
        let mut v = result;
        for entry in entries {
            v = (entry.guardrail)(name, v);
        }
        v
    }

    /// Runs the tool conditional execution guardrail chain. Returns the first rejection reason, or `None` if all pass.
    /// Merges global + scope-local entries by priority.
    pub fn tool_conditional_execution_chain(
        &self,
        name: &str,
        args: &Json,
        scope_locals: &[&SortedRegistry<GuardrailEntry<ToolConditionalFn>>],
    ) -> Option<String> {
        let entries =
            merge_guardrail_entries(&self.tool_conditional_execution_guardrails, scope_locals);
        for entry in entries {
            if let Some(err) = (entry.guardrail)(name, args) {
                return Some(err);
            }
        }
        None
    }

    /// Runs the tool request intercept chain, piping args through each intercept (with optional break).
    /// Merges global + scope-local entries by priority.
    pub fn tool_request_intercepts_chain(
        &self,
        name: &str,
        args: Json,
        scope_locals: &[&SortedRegistry<Intercept<ToolInterceptFn>>],
    ) -> Json {
        let entries = merge_intercept_entries(&self.tool_request_intercepts, scope_locals);
        let mut v = args;
        for entry in entries {
            v = (entry.callable)(name, v);
            if entry.break_chain {
                break;
            }
        }
        v
    }

    /// Build a middleware chain of all matching tool execution intercepts.
    /// Returns a single `FnOnce` that, when called, runs through the chain
    /// and ultimately calls `default_fn` if no intercept short-circuits.
    ///
    /// Intercepts are sorted by priority ascending. The lowest-priority
    /// (first) matching intercept becomes the outermost wrapper.
    /// Merges global + scope-local entries by priority.
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
            let current_next = next;
            let n = name.clone();
            next = Box::new(move |args| callable(&n, args, current_next));
        }
        next
    }

    // -----------------------------------------------------------------------
    // LLM chain methods
    // -----------------------------------------------------------------------

    /// Runs the LLM request sanitize guardrail chain, piping the request through each guardrail.
    /// Merges global + scope-local entries by priority.
    pub fn llm_sanitize_request_chain(
        &self,
        request: LLMRequest,
        scope_locals: &[&SortedRegistry<GuardrailEntry<LlmSanitizeRequestFn>>],
    ) -> LLMRequest {
        let entries = merge_guardrail_entries(&self.llm_sanitize_request_guardrails, scope_locals);
        let mut v = request;
        for entry in entries {
            v = (entry.guardrail)(v);
        }
        v
    }

    /// Runs the LLM response sanitize guardrail chain, piping the response through each guardrail.
    /// Merges global + scope-local entries by priority.
    pub fn llm_sanitize_response_chain(
        &self,
        response: Json,
        scope_locals: &[&SortedRegistry<GuardrailEntry<LlmSanitizeResponseFn>>],
    ) -> Json {
        let entries = merge_guardrail_entries(&self.llm_sanitize_response_guardrails, scope_locals);
        let mut v = response;
        for entry in entries {
            v = (entry.guardrail)(v);
        }
        v
    }

    /// Runs the LLM conditional execution guardrail chain. Returns the first rejection reason, or `None` if all pass.
    /// Merges global + scope-local entries by priority.
    pub fn llm_conditional_execution_chain(
        &self,
        request: &LLMRequest,
        scope_locals: &[&SortedRegistry<GuardrailEntry<LlmConditionalFn>>],
    ) -> Option<String> {
        let entries =
            merge_guardrail_entries(&self.llm_conditional_execution_guardrails, scope_locals);
        for entry in entries {
            if let Some(err) = (entry.guardrail)(request) {
                return Some(err);
            }
        }
        None
    }

    /// Runs the LLM request intercept chain on `LLMRequest`, piping through each intercept (with optional break).
    /// Merges global + scope-local entries by priority.
    pub fn llm_request_intercepts_chain(
        &self,
        name: &str,
        request: LLMRequest,
        scope_locals: &[&SortedRegistry<Intercept<LlmRequestInterceptFn>>],
    ) -> LLMRequest {
        let entries = merge_intercept_entries(&self.llm_request_intercepts, scope_locals);
        let mut v = request;
        for entry in entries {
            v = (entry.callable)(name, v);
            if entry.break_chain {
                break;
            }
        }
        v
    }

    /// Build a middleware chain of all matching LLM execution intercepts.
    /// Returns a single `FnOnce` that, when called, runs through the chain
    /// and ultimately calls `default_fn` if no intercept short-circuits.
    /// Merges global + scope-local entries by priority.
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
            let current_next = next;
            let n = name.clone();
            next = Box::new(move |request| callable(&n, request, current_next));
        }
        next
    }

    /// Build a middleware chain of all matching LLM streaming execution intercepts.
    /// Returns a single `FnOnce` that, when called, runs through the chain
    /// and ultimately calls `default_fn` if no intercept short-circuits.
    /// Merges global + scope-local entries by priority.
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
            let current_next = next;
            let n = name.clone();
            next = Box::new(move |request| callable(&n, request, current_next));
        }
        next
    }
}

impl Default for NatNexusContextState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

static GLOBAL_CONTEXT: std::sync::OnceLock<Arc<RwLock<NatNexusContextState>>> =
    std::sync::OnceLock::new();

/// Returns the process-wide singleton [`NatNexusContextState`], lazily initialized.
///
/// The returned `Arc<RwLock<...>>` can be cloned cheaply and shared across threads.
/// All public API functions in [`crate::api`] use this internally.
pub fn global_context() -> Arc<RwLock<NatNexusContextState>> {
    GLOBAL_CONTEXT
        .get_or_init(|| Arc::new(RwLock::new(NatNexusContextState::new())))
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

        let ctx = NatNexusContextState::new();
        let handle = ctx.create_scope_handle(
            "test",
            None,
            ScopeType::Agent,
            ScopeAttributes::empty(),
            None,
            None,
            None,
            &[],
        );
        task_scope_push(handle.clone());
        let top = task_scope_top();
        assert_eq!(top.name, "test");

        let removed = task_scope_remove(&handle.uuid).unwrap();
        ctx.end_scope_handle(&removed, None, &[]);

        // After pop, root scope is on top again
        assert_eq!(task_scope_top().name, "root");
    }

    #[test]
    fn test_event_subscriber() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let mut ctx = NatNexusContextState::new();
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
            None,
            None,
            None,
            &[],
        );
        assert_eq!(count.load(Ordering::SeqCst), 1);

        // end_scope_handle emits an END event
        ctx.end_scope_handle(&handle, None, &[]);
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
            None,
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
            None,
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
        let h1 = ScopeHandle::new(
            "a".into(),
            ScopeType::Agent,
            ScopeAttributes::empty(),
            None,
            None,
            None,
        );
        let h2 = ScopeHandle::new(
            "b".into(),
            ScopeType::Function,
            ScopeAttributes::empty(),
            None,
            None,
            None,
        );
        let h3 = ScopeHandle::new(
            "c".into(),
            ScopeType::Tool,
            ScopeAttributes::empty(),
            None,
            None,
            None,
        );
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
        let h1 = ScopeHandle::new(
            "a".into(),
            ScopeType::Agent,
            ScopeAttributes::empty(),
            None,
            None,
            None,
        );
        let h2 = ScopeHandle::new(
            "b".into(),
            ScopeType::Function,
            ScopeAttributes::empty(),
            None,
            None,
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
            NexusError::NotFound(msg) => assert!(msg.contains("not found")),
            e => panic!("expected NotFound, got {e:?}"),
        }
    }

    // -- NatNexusContextState tests --

    #[test]
    fn test_context_state_new_empty() {
        let ctx = NatNexusContextState::new();
        assert!(ctx.event_subscribers.is_empty());
    }

    #[test]
    fn test_context_state_default() {
        let ctx = NatNexusContextState::default();
        assert!(ctx.event_subscribers.is_empty());
    }

    // -- Event emission tests --

    #[test]
    fn test_emit_event_multiple_subscribers() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let mut ctx = NatNexusContextState::new();
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
        ctx.emit_event(&event, &[]);

        assert_eq!(c1.load(Ordering::SeqCst), 1);
        assert_eq!(c2.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_emit_event_no_subscribers() {
        let ctx = NatNexusContextState::new();
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
        ctx.emit_event(&event, &[]);
    }

    // -- create_event tests --

    #[test]
    fn test_create_event_emits_mark() {
        use std::sync::Mutex;

        let mut ctx = NatNexusContextState::new();
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
            None,
            &[],
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

        let mut ctx = NatNexusContextState::new();
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
            None,
            None,
            None,
            &[],
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

        let mut ctx = NatNexusContextState::new();
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
            None,
            ScopeType::Agent,
            ScopeAttributes::empty(),
            None,
            None,
            None,
            &[],
        );
        ctx.end_scope_handle(&handle, None, &[]);

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].event_type, EventType::Start);
        assert_eq!(captured[1].event_type, EventType::End);
        assert_eq!(captured[1].uuid, handle.uuid);
    }

    #[test]
    fn test_create_tool_handle_emits_start() {
        use std::sync::Mutex;

        let mut ctx = NatNexusContextState::new();
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
            None,
            None,
            None,
            &[],
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

        let mut ctx = NatNexusContextState::new();
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
            None,
            None,
            None,
            &[],
        );
        ctx.end_tool_handle(
            &handle,
            Some(serde_json::json!({"b": 2})),
            None,
            None,
            None,
            &[],
        );

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

        let mut ctx = NatNexusContextState::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        ctx.event_subscribers.insert(
            "cap".into(),
            Box::new(move |e: &Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        );

        let _handle = ctx.create_llm_handle(
            "llm",
            None,
            LLMAttributes::STREAMING,
            None,
            None,
            None,
            None,
            None,
            &[],
        );

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

        let mut ctx = NatNexusContextState::new();
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
            None,
            None,
            None,
            &[],
        );
        ctx.end_llm_handle(
            &handle,
            None,
            Some(serde_json::json!({"m2": false})),
            None,
            None,
            &[],
        );

        let captured = events.lock().unwrap();
        let end_event = &captured[1];
        let meta = end_event.metadata.as_ref().unwrap();
        assert_eq!(meta["m1"], true);
        assert_eq!(meta["m2"], false);
    }

    // -- Chain runner tests --

    #[test]
    fn test_tool_sanitize_request_chain() {
        let mut ctx = NatNexusContextState::new();
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

        let result =
            ctx.tool_sanitize_request_chain("tool", serde_json::json!({"input": "data"}), &[]);
        assert_eq!(result["input"], "data");
        assert_eq!(result["sanitized"], true);
    }

    #[test]
    fn test_tool_sanitize_request_chain_multiple_priority_order() {
        let mut ctx = NatNexusContextState::new();
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

        let result = ctx.tool_sanitize_request_chain("tool", serde_json::json!({}), &[]);
        // g1 runs first (priority 10), then g2 (priority 20) overwrites
        assert_eq!(result["step"], "second");
    }

    #[test]
    fn test_tool_sanitize_response_chain() {
        let mut ctx = NatNexusContextState::new();
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

        let result =
            ctx.tool_sanitize_response_chain("tool", serde_json::json!({"output": "ok"}), &[]);
        assert_eq!(result["clean"], true);
        assert_eq!(result["output"], "ok");
    }

    #[test]
    fn test_tool_conditional_execution_chain_passes() {
        let mut ctx = NatNexusContextState::new();
        ctx.tool_conditional_execution_guardrails
            .register(
                "g1".into(),
                GuardrailEntry {
                    priority: 1,
                    guardrail: Box::new(|_name: &str, _args: &Json| None),
                },
            )
            .unwrap();

        let result = ctx.tool_conditional_execution_chain("tool", &serde_json::json!({}), &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_tool_conditional_execution_chain_rejects() {
        let mut ctx = NatNexusContextState::new();
        ctx.tool_conditional_execution_guardrails
            .register(
                "blocker".into(),
                GuardrailEntry {
                    priority: 1,
                    guardrail: Box::new(|_name: &str, _args: &Json| Some("not allowed".into())),
                },
            )
            .unwrap();

        let result = ctx.tool_conditional_execution_chain("tool", &serde_json::json!({}), &[]);
        assert_eq!(result, Some("not allowed".into()));
    }

    #[test]
    fn test_tool_conditional_first_rejection_wins() {
        let mut ctx = NatNexusContextState::new();
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

        let result = ctx.tool_conditional_execution_chain("tool", &serde_json::json!({}), &[]);
        assert_eq!(result, Some("first".into()));
    }

    #[test]
    fn test_tool_request_intercepts_chain() {
        let mut ctx = NatNexusContextState::new();
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
            ctx.tool_request_intercepts_chain("tool", serde_json::json!({"original": true}), &[]);
        assert_eq!(result["original"], true);
        assert_eq!(result["intercepted"], true);
    }

    #[test]
    fn test_tool_request_intercepts_chain_break() {
        let mut ctx = NatNexusContextState::new();
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

        let result = ctx.tool_request_intercepts_chain("tool", serde_json::json!({}), &[]);
        assert_eq!(result["from_i1"], true);
        // i2 should NOT have run due to break_chain
        assert!(result.get("from_i2").is_none());
    }

    #[tokio::test]
    async fn test_tool_build_execution_chain_no_intercepts() {
        let ctx = NatNexusContextState::new();
        let default_fn: ToolExecutionNextFn =
            Box::new(|_args| Box::pin(async { Ok(serde_json::json!({"default": true})) }));
        let chain = ctx.tool_build_execution_chain("test_tool", default_fn, &[]);
        let result = chain(serde_json::json!({})).await.unwrap();
        assert_eq!(result["default"], true);
    }

    #[tokio::test]
    async fn test_tool_build_execution_chain_passthrough() {
        let mut ctx = NatNexusContextState::new();
        // An intercept that simply passes through to next (equivalent to old conditional=false)
        ctx.tool_execution_intercepts
            .register(
                "ei1".into(),
                ExecutionIntercept {
                    priority: 1,
                    callable: Arc::new(|_name: &str, args: Json, next: ToolExecutionNextFn| {
                        Box::pin(async move { next(args).await })
                            as std::pin::Pin<
                                Box<dyn std::future::Future<Output = Result<Json>> + Send>,
                            >
                    }) as ToolExecutionFn,
                },
            )
            .unwrap();
        let default_fn: ToolExecutionNextFn =
            Box::new(|_args| Box::pin(async { Ok(serde_json::json!({"default": true})) }));
        let chain = ctx.tool_build_execution_chain("test_tool", default_fn, &[]);
        let result = chain(serde_json::json!({})).await.unwrap();
        // Intercept passes through, so default should run
        assert_eq!(result["default"], true);
    }

    #[tokio::test]
    async fn test_tool_build_execution_chain_short_circuit() {
        let mut ctx = NatNexusContextState::new();
        ctx.tool_execution_intercepts
            .register(
                "ei1".into(),
                ExecutionIntercept {
                    priority: 1,
                    callable: Arc::new(|_name: &str, _args: Json, _next: ToolExecutionNextFn| {
                        Box::pin(async { Ok(serde_json::json!({"intercepted": true})) })
                            as std::pin::Pin<
                                Box<dyn std::future::Future<Output = Result<Json>> + Send>,
                            >
                    }) as ToolExecutionFn,
                },
            )
            .unwrap();
        let default_fn: ToolExecutionNextFn =
            Box::new(|_args| Box::pin(async { Ok(serde_json::json!({"default": true})) }));
        let chain = ctx.tool_build_execution_chain("test_tool", default_fn, &[]);
        let result = chain(serde_json::json!({})).await.unwrap();
        // Intercept short-circuits (skips next), so intercept result should be returned
        assert_eq!(result["intercepted"], true);
    }

    #[tokio::test]
    async fn test_tool_build_execution_chain_calls_next() {
        let mut ctx = NatNexusContextState::new();
        ctx.tool_execution_intercepts
            .register(
                "ei1".into(),
                ExecutionIntercept {
                    priority: 1,
                    callable: Arc::new(|_name: &str, args: Json, next: ToolExecutionNextFn| {
                        Box::pin(async move {
                            let mut result = next(args).await?;
                            result
                                .as_object_mut()
                                .unwrap()
                                .insert("wrapped".into(), serde_json::json!(true));
                            Ok(result)
                        })
                            as std::pin::Pin<
                                Box<dyn std::future::Future<Output = Result<Json>> + Send>,
                            >
                    }) as ToolExecutionFn,
                },
            )
            .unwrap();
        let default_fn: ToolExecutionNextFn =
            Box::new(|_args| Box::pin(async { Ok(serde_json::json!({"default": true})) }));
        let chain = ctx.tool_build_execution_chain("test_tool", default_fn, &[]);
        let result = chain(serde_json::json!({})).await.unwrap();
        // Intercept calls next, then wraps result
        assert_eq!(result["default"], true);
        assert_eq!(result["wrapped"], true);
    }

    #[tokio::test]
    async fn test_tool_build_execution_chain_multiple_intercepts() {
        let mut ctx = NatNexusContextState::new();
        // Lower priority = outermost wrapper
        ctx.tool_execution_intercepts
            .register(
                "outer".into(),
                ExecutionIntercept {
                    priority: 1,
                    callable: Arc::new(|_name: &str, args: Json, next: ToolExecutionNextFn| {
                        Box::pin(async move {
                            let mut result = next(args).await?;
                            result
                                .as_object_mut()
                                .unwrap()
                                .insert("outer".into(), serde_json::json!(true));
                            Ok(result)
                        })
                            as std::pin::Pin<
                                Box<dyn std::future::Future<Output = Result<Json>> + Send>,
                            >
                    }) as ToolExecutionFn,
                },
            )
            .unwrap();
        ctx.tool_execution_intercepts
            .register(
                "inner".into(),
                ExecutionIntercept {
                    priority: 2,
                    callable: Arc::new(|_name: &str, args: Json, next: ToolExecutionNextFn| {
                        Box::pin(async move {
                            let mut result = next(args).await?;
                            result
                                .as_object_mut()
                                .unwrap()
                                .insert("inner".into(), serde_json::json!(true));
                            Ok(result)
                        })
                            as std::pin::Pin<
                                Box<dyn std::future::Future<Output = Result<Json>> + Send>,
                            >
                    }) as ToolExecutionFn,
                },
            )
            .unwrap();
        let default_fn: ToolExecutionNextFn =
            Box::new(|_args| Box::pin(async { Ok(serde_json::json!({"default": true})) }));
        let chain = ctx.tool_build_execution_chain("test_tool", default_fn, &[]);
        let result = chain(serde_json::json!({})).await.unwrap();
        // Both intercepts call next, so all three flags should be set
        assert_eq!(result["default"], true);
        assert_eq!(result["inner"], true);
        assert_eq!(result["outer"], true);
    }

    // -- LLM chain methods --

    #[test]
    fn test_llm_sanitize_request_chain() {
        let mut ctx = NatNexusContextState::new();
        ctx.llm_sanitize_request_guardrails
            .register(
                "g1".into(),
                GuardrailEntry {
                    priority: 1,
                    guardrail: Box::new(|mut req: LLMRequest| {
                        req.headers
                            .insert("sanitized".into(), serde_json::json!(true));
                        req
                    }),
                },
            )
            .unwrap();

        let req = LLMRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({"messages": []}),
        };
        let result = ctx.llm_sanitize_request_chain(req, &[]);
        assert_eq!(result.headers["sanitized"], true);
        assert_eq!(result.content, serde_json::json!({"messages": []}));
    }

    #[test]
    fn test_llm_sanitize_response_chain() {
        let mut ctx = NatNexusContextState::new();
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

        let result = ctx.llm_sanitize_response_chain(serde_json::json!({"data": "test"}), &[]);
        assert_eq!(result["sanitized"], true);
    }

    #[test]
    fn test_llm_conditional_execution_chain_passes() {
        let mut ctx = NatNexusContextState::new();
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
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        };
        assert!(ctx.llm_conditional_execution_chain(&req, &[]).is_none());
    }

    #[test]
    fn test_llm_conditional_execution_chain_rejects() {
        let mut ctx = NatNexusContextState::new();
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
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        };
        assert_eq!(
            ctx.llm_conditional_execution_chain(&req, &[]),
            Some("blocked".into())
        );
    }

    #[test]
    fn test_llm_request_intercepts_chain() {
        let mut ctx = NatNexusContextState::new();
        ctx.llm_request_intercepts
            .register(
                "i1".into(),
                Intercept {
                    priority: 1,
                    break_chain: false,
                    callable: Box::new(|_name: &str, mut req: LLMRequest| {
                        req.headers
                            .insert("intercepted".into(), serde_json::json!(true));
                        req
                    }),
                },
            )
            .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({"messages": []}),
        };
        let result = ctx.llm_request_intercepts_chain("test_llm", request, &[]);
        assert_eq!(result.headers["intercepted"], true);
        assert_eq!(result.content["messages"], serde_json::json!([]));
    }

    #[test]
    fn test_llm_request_intercepts_break_chain() {
        let mut ctx = NatNexusContextState::new();
        ctx.llm_request_intercepts
            .register(
                "i1".into(),
                Intercept {
                    priority: 1,
                    break_chain: true,
                    callable: Box::new(|_name: &str, mut req: LLMRequest| {
                        req.headers
                            .insert("from_i1".into(), serde_json::json!(true));
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
                    callable: Box::new(|_name: &str, mut req: LLMRequest| {
                        req.headers
                            .insert("from_i2".into(), serde_json::json!(true));
                        req
                    }),
                },
            )
            .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        };
        let result = ctx.llm_request_intercepts_chain("test_llm", request, &[]);
        assert_eq!(result.headers["from_i1"], true);
        assert!(result.headers.get("from_i2").is_none());
    }

    #[tokio::test]
    async fn test_llm_build_execution_chain_no_intercepts() {
        let ctx = NatNexusContextState::new();
        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({"messages": []}),
        };
        let default_fn: LlmExecutionNextFn =
            Box::new(|_req| Box::pin(async { Ok(serde_json::json!({"default": true})) }));
        let chain = ctx.llm_build_execution_chain("test_llm", default_fn, &[]);
        let result = chain(request).await.unwrap();
        assert_eq!(result["default"], true);
    }

    #[tokio::test]
    async fn test_llm_stream_build_execution_chain_no_intercepts() {
        use futures::StreamExt;
        let ctx = NatNexusContextState::new();
        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({"messages": []}),
        };
        let default_fn: LlmStreamExecutionNextFn = Box::new(|_req| {
            Box::pin(async {
                let stream: Pin<Box<dyn Stream<Item = Result<Json>> + Send>> =
                    Box::pin(futures::stream::once(async {
                        Ok(serde_json::json!({"token": "chunk"}))
                    }));
                Ok(stream)
            })
        });
        let chain = ctx.llm_stream_build_execution_chain("test_llm", default_fn, &[]);
        let stream = chain(request).await.unwrap();
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].as_ref().unwrap(),
            &serde_json::json!({"token": "chunk"})
        );
    }

    // -- Generic chain runners --

    #[test]
    fn test_run_sanitize_chain_empty() {
        let reg: SortedRegistry<GuardrailEntry<Box<dyn Fn(i32) -> i32>>> =
            SortedRegistry::new(|e| e.priority);
        let result = NatNexusContextState::run_sanitize_chain(&reg, 42);
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
        let result = NatNexusContextState::run_sanitize_chain(&reg, 5);
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
        assert!(NatNexusContextState::run_conditional_chain(&reg, &42).is_none());
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
            NatNexusContextState::run_conditional_chain(&reg, &42),
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
        let result = NatNexusContextState::run_intercept_chain(&reg, 0);
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
        let result = NatNexusContextState::run_intercept_chain(&reg, 0);
        // Only 'a' runs, 'b' is skipped
        assert_eq!(result, 10);
    }

    #[test]
    fn test_run_intercept_chain_empty() {
        let reg: SortedRegistry<Intercept<Box<dyn Fn(i32) -> i32>>> =
            SortedRegistry::new(|e| e.priority);
        let result = NatNexusContextState::run_intercept_chain(&reg, 42);
        assert_eq!(result, 42);
    }
}
