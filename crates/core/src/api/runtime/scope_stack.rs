// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scope stack storage and propagation helpers.
//!
//! The runtime tracks the current scope hierarchy through a shared
//! [`ScopeStack`] stored in task-local or thread-local state. Advanced callers
//! can use this module to inspect the active scope chain, attach scope-local
//! middleware, or propagate scope context into worker threads.

use std::cell::RefCell;
use std::sync::{Arc, RwLock};

use uuid::Uuid;

use crate::api::runtime::callbacks::EventSubscriberFn;
use crate::api::scope::{ScopeHandle, ScopeType};
use crate::context::registries::ScopeLocalRegistries;
use crate::error::{FlowError, Result};
use crate::registry::SortedRegistry;

/// Mutable stack of active scopes plus their scope-local registries.
///
/// The stack always contains an implicit root scope. Additional scopes are
/// pushed as the public API opens lifecycle spans and removed when those spans
/// close.
pub struct ScopeStack {
    stack: Vec<ScopeHandle>,
    scope_registries: std::collections::HashMap<Uuid, ScopeLocalRegistries>,
}

impl ScopeStack {
    /// Create a new scope stack containing only the implicit root scope.
    ///
    /// # Returns
    /// A [`ScopeStack`] initialized with a single root scope and no
    /// scope-local registries.
    pub fn new() -> Self {
        let root = ScopeHandle::builder()
            .name("root")
            .scope_type(ScopeType::Agent)
            .build();
        Self {
            stack: vec![root],
            scope_registries: std::collections::HashMap::new(),
        }
    }

    /// Push a scope handle onto the top of the stack.
    ///
    /// # Parameters
    /// - `handle`: Scope handle to make the new top-most active scope.
    pub fn push(&mut self, handle: ScopeHandle) {
        self.stack.push(handle);
    }

    /// Return the current top-most scope handle.
    ///
    /// # Returns
    /// A shared reference to the active scope at the top of the stack.
    ///
    /// # Notes
    /// This function never returns `None` because the implicit root scope is
    /// always present.
    pub fn top(&self) -> &ScopeHandle {
        self.stack
            .last()
            .expect("scope stack should never be empty")
    }

    /// Return the current top-most scope handle mutably.
    ///
    /// # Returns
    /// A mutable reference to the active scope at the top of the stack.
    pub fn top_mut(&mut self) -> &mut ScopeHandle {
        self.stack
            .last_mut()
            .expect("scope stack should never be empty")
    }

    /// Return the UUID of the implicit root scope.
    ///
    /// # Returns
    /// The stable UUID of the root scope stored at the bottom of the stack.
    pub fn root_uuid(&self) -> Uuid {
        self.stack
            .first()
            .expect("scope stack should never be empty")
            .uuid
    }

    /// Return the full ordered stack of scope handles.
    ///
    /// # Returns
    /// A slice of scopes ordered from root to the current top-most scope.
    pub fn scopes(&self) -> &[ScopeHandle] {
        &self.stack
    }

    /// Find a scope handle by UUID.
    ///
    /// # Parameters
    /// - `uuid`: UUID of the scope to search for.
    ///
    /// # Returns
    /// `Some(&ScopeHandle)` when the scope is active on this stack and `None`
    /// otherwise.
    pub fn find(&self, uuid: &Uuid) -> Option<&ScopeHandle> {
        self.stack.iter().find(|handle| handle.uuid == *uuid)
    }

    /// Remove the current top scope if it matches `uuid`.
    ///
    /// # Parameters
    /// - `uuid`: UUID of the scope expected to be at the top of the stack.
    ///
    /// # Returns
    /// A [`Result`] containing the removed [`ScopeHandle`].
    ///
    /// # Errors
    /// Returns [`FlowError::InvalidArgument`] when the scope exists but is not
    /// the current top of the stack or when the caller attempts to remove the
    /// implicit root scope. Returns [`FlowError::NotFound`] when the UUID is
    /// not present on the stack.
    pub fn remove(&mut self, uuid: &Uuid) -> Result<ScopeHandle> {
        let top = self
            .stack
            .last()
            .expect("scope stack should never be empty");
        if top.uuid == *uuid {
            if self.stack.len() == 1 {
                return Err(FlowError::InvalidArgument(
                    "root scope cannot be removed".into(),
                ));
            }
            self.scope_registries.remove(uuid);
            return Ok(self
                .stack
                .pop()
                .expect("scope stack should contain a removable top scope"));
        }

        if self.stack.iter().any(|handle| handle.uuid == *uuid) {
            return Err(FlowError::InvalidArgument(
                "scope handle is not at the top of the stack".into(),
            ));
        }

        Err(FlowError::NotFound("scope handle not found".into()))
    }

    /// Get or create the scope-local registries for an active scope.
    ///
    /// # Parameters
    /// - `uuid`: UUID of an active scope on this stack.
    ///
    /// # Returns
    /// `Some(&mut ScopeLocalRegistries)` when the scope is active and `None`
    /// otherwise.
    ///
    /// # Notes
    /// When the scope is active but has no registries yet, this function
    /// creates an empty scope-local registry set first.
    pub fn local_registries_mut(&mut self, uuid: &Uuid) -> Option<&mut ScopeLocalRegistries> {
        if !self.stack.iter().any(|handle| handle.uuid == *uuid) {
            return None;
        }
        Some(self.scope_registries.entry(*uuid).or_default())
    }

    /// Collect one registry field from every active scope that owns it.
    ///
    /// # Parameters
    /// - `field`: Projection function selecting the registry field to collect
    ///   from each scope-local registry.
    ///
    /// # Returns
    /// A vector of registry references ordered from root toward the current
    /// top-most scope.
    pub fn collect_scope_local_registries<'a, T>(
        &'a self,
        field: impl Fn(&'a ScopeLocalRegistries) -> &'a SortedRegistry<T>,
    ) -> Vec<&'a SortedRegistry<T>> {
        self.stack
            .iter()
            .filter_map(|handle| self.scope_registries.get(&handle.uuid))
            .map(field)
            .collect()
    }

    /// Collect all scope-local subscribers visible from the active stack.
    ///
    /// # Returns
    /// A vector of subscribers collected from each active scope that owns
    /// scope-local registries.
    pub fn collect_scope_local_subscribers(&self) -> Vec<EventSubscriberFn> {
        self.stack
            .iter()
            .filter_map(|handle| self.scope_registries.get(&handle.uuid))
            .flat_map(|registries| registries.event_subscribers.values().cloned())
            .collect()
    }

    /// Return the scope-local registries for `uuid` without creating them.
    ///
    /// # Parameters
    /// - `uuid`: UUID of the scope whose registries should be borrowed.
    ///
    /// # Returns
    /// `Some(&ScopeLocalRegistries)` when registries already exist for that
    /// scope and `None` otherwise.
    pub fn scope_registries_get(&self, uuid: &Uuid) -> Option<&ScopeLocalRegistries> {
        self.scope_registries.get(uuid)
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

/// Shared handle type for the runtime scope stack.
///
/// The runtime stores the active [`ScopeStack`] behind an [`Arc`] and [`RwLock`]
/// so bindings can propagate it across execution contexts while still allowing
/// concurrent readers.
pub type ScopeStackHandle = Arc<RwLock<ScopeStack>>;

/// Captured thread-local scope stack binding.
///
/// This preserves both the visible scope stack handle and whether it was
/// explicitly installed on the current thread.
#[derive(Clone)]
pub struct ThreadScopeStackBinding {
    stack: ScopeStackHandle,
    explicit: bool,
}

/// Create a new scope stack handle with an implicit root scope.
///
/// The returned handle wraps a freshly initialized [`ScopeStack`] inside an
/// [`Arc`] and [`RwLock`] so it can be shared across async tasks or threads.
///
/// # Returns
/// A new [`ScopeStackHandle`] containing exactly one implicit root scope.
///
/// # Notes
/// The root scope is always present and cannot be removed.
pub fn create_scope_stack() -> ScopeStackHandle {
    Arc::new(RwLock::new(ScopeStack::new()))
}

tokio::task_local! {
    /// Task-local scope stack handle used by async execution contexts.
    pub static TASK_SCOPE_STACK: ScopeStackHandle;
}

thread_local! {
    /// Thread-local fallback scope stack for non-task contexts.
    static THREAD_SCOPE_STACK: RefCell<ScopeStackHandle> = RefCell::new(create_scope_stack());
    /// Whether the current thread explicitly owns a scope stack.
    static THREAD_SCOPE_STACK_EXPLICIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Return the scope stack visible to the current execution context.
///
/// This resolves task-local scope state first and otherwise falls back to the
/// current thread-local scope stack handle.
///
/// # Returns
/// The active [`ScopeStackHandle`] for the current async task or thread.
///
/// # Notes
/// When no explicit thread-local stack has been installed yet, the default
/// per-thread root-only stack is returned.
pub fn current_scope_stack() -> ScopeStackHandle {
    TASK_SCOPE_STACK
        .try_with(|stack| stack.clone())
        .unwrap_or_else(|_| THREAD_SCOPE_STACK.with(|stack| stack.borrow().clone()))
}

/// Install an explicit scope stack for the current thread.
///
/// This replaces the thread-local scope stack handle and marks the current
/// thread as explicitly scope-aware for later propagation checks.
///
/// # Parameters
/// - `handle`: Scope stack handle to install for the current thread.
///
/// # Returns
/// `()`.
///
/// # Notes
/// Use this when propagating an existing scope stack into worker threads.
pub fn set_thread_scope_stack(handle: ScopeStackHandle) {
    THREAD_SCOPE_STACK.with(|stack| *stack.borrow_mut() = handle);
    THREAD_SCOPE_STACK_EXPLICIT.with(|flag| flag.set(true));
}

/// Capture the current thread-local scope stack binding.
///
/// This is intended for foreign runtimes that temporarily bind a scope stack to
/// an OS thread and need to restore the exact previous state before releasing
/// that thread back to their scheduler.
///
/// # Returns
/// A [`ThreadScopeStackBinding`] containing the current thread-local stack and
/// explicit-binding flag.
pub fn capture_thread_scope_stack() -> ThreadScopeStackBinding {
    let stack = THREAD_SCOPE_STACK.with(|stack| stack.borrow().clone());
    let explicit = THREAD_SCOPE_STACK_EXPLICIT.with(|flag| flag.get());
    ThreadScopeStackBinding { stack, explicit }
}

/// Restore a previously captured thread-local scope stack binding.
///
/// # Parameters
/// - `binding`: Captured binding to restore on the current thread.
///
/// # Returns
/// `()`.
pub fn restore_thread_scope_stack(binding: ThreadScopeStackBinding) {
    THREAD_SCOPE_STACK.with(|stack| *stack.borrow_mut() = binding.stack);
    THREAD_SCOPE_STACK_EXPLICIT.with(|flag| flag.set(binding.explicit));
}

/// Synchronize the thread-local scope stack without marking it explicit.
///
/// This updates the thread-local slot used by native runtime code while
/// preserving whether the thread was explicitly marked as owning a scope stack.
///
/// # Parameters
/// - `handle`: Scope stack handle to synchronize into thread-local storage.
///
/// # Returns
/// `()`.
///
/// # Notes
/// Python bindings use this to mirror `ContextVar` state into Rust without
/// forcing `scope_stack_active()` to become `true` for the thread.
pub fn sync_thread_scope_stack(handle: ScopeStackHandle) {
    THREAD_SCOPE_STACK.with(|stack| *stack.borrow_mut() = handle);
}

/// Report whether the current context has an explicitly active scope stack.
///
/// This checks task-local state first and otherwise falls back to the
/// thread-local explicit flag.
///
/// # Returns
/// `true` when the current async task or thread already owns an active scope
/// stack and `false` otherwise.
///
/// # Notes
/// A synchronized thread-local stack does not count as explicit unless it was
/// installed through [`set_thread_scope_stack`].
pub fn scope_stack_active() -> bool {
    TASK_SCOPE_STACK
        .try_with(|_| true)
        .unwrap_or_else(|_| THREAD_SCOPE_STACK_EXPLICIT.with(|flag| flag.get()))
}

/// Capture the current scope stack handle for use in another thread.
///
/// This returns the handle currently visible to the caller so it can be passed
/// into [`set_thread_scope_stack`] elsewhere.
///
/// # Returns
/// A [`Result`] containing the active [`ScopeStackHandle`].
///
/// # Errors
/// Returns an error when the current context does not yet own an active scope
/// stack.
///
/// # Notes
/// The returned handle is shared; it does not clone the underlying stack.
pub fn propagate_scope_to_thread() -> Result<ScopeStackHandle> {
    if !scope_stack_active() {
        return Err(FlowError::Internal(
            "no active scope stack in current context; call create_scope_stack() and set_thread_scope_stack() first"
                .into(),
        ));
    }
    Ok(current_scope_stack())
}

/// Clone the current top-most scope handle from the active stack.
///
/// # Returns
/// A cloned [`ScopeHandle`] representing the current active scope.
pub fn task_scope_top() -> ScopeHandle {
    let stack = current_scope_stack();
    let guard = stack.read().expect("scope stack lock poisoned");
    guard.top().clone()
}

/// Push a scope handle onto the active stack.
///
/// # Parameters
/// - `handle`: Scope handle to push onto the current execution context's stack.
pub fn task_scope_push(handle: ScopeHandle) {
    let stack = current_scope_stack();
    let mut guard = stack.write().expect("scope stack lock poisoned");
    guard.push(handle);
}

/// Remove a scope handle from the active stack.
///
/// # Parameters
/// - `uuid`: UUID of the scope expected to be at the top of the active stack.
///
/// # Returns
/// A [`Result`] containing the removed [`ScopeHandle`].
///
/// # Errors
/// Propagates the same errors returned by [`ScopeStack::remove`].
pub fn task_scope_remove(uuid: &Uuid) -> Result<ScopeHandle> {
    let stack = current_scope_stack();
    let mut guard = stack.write().expect("scope stack lock poisoned");
    guard.remove(uuid)
}
