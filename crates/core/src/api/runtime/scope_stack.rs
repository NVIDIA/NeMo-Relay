// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scope stack storage and propagation helpers.
//!
//! The runtime tracks the current scope hierarchy through a shared
//! [`ScopeStack`] stored in task-local or thread-local state. Advanced callers
//! can use this module to inspect the active scope chain or propagate scope
//! context into worker threads.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::runtime::callbacks::EventSubscriberFn;
use crate::api::scope::{ScopeHandle, ScopeType};
use crate::context::registries::ScopeLocalRegistries;
use crate::error::{FlowError, Result};
use crate::registry::{RegistryEntry, SortedRegistry};

/// Mutable stack of active scopes plus their scope-local registries.
///
/// The stack always contains an implicit root agent scope. It owns freshness
/// for work that is not nested under an explicit agent; non-agent scopes inherit
/// their nearest agent's freshness instead of creating a separate budget.
/// Additional scopes are pushed as the public API opens lifecycle spans and
/// removed when those spans close.
pub struct ScopeStack {
    stack: Vec<ScopeHandle>,
    scope_registries: HashMap<Uuid, ScopeLocalRegistries>,
    fresh_agents: HashSet<Uuid>,
    propagated_parent_uuid: Option<Uuid>,
    propagated_root_uuid: Option<Uuid>,
    is_rootless_propagation: bool,
}

/// Versioned, transport-neutral causal context for crossing a Relay boundary.
///
/// Applications are responsible for serializing, transporting, authenticating,
/// and trusting this value. It intentionally contains only Relay identifiers;
/// OpenTelemetry `traceparent` and `tracestate` remain transport sidecars. A
/// context without a `root_uuid` preserves Relay event parentage when imported.
/// The first local OpenTelemetry span created after import starts a new trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropagationContext {
    /// Wire-format version. Version 1 is the only currently supported value.
    pub version: u16,
    /// Stable session root when the sending application knows one. When this
    /// root is omitted, the first local OpenTelemetry span after import starts
    /// a new trace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_uuid: Option<Uuid>,
    /// Immediate Relay event or scope that caused the boundary crossing.
    pub parent_uuid: Uuid,
}

impl PropagationContext {
    /// The current wire-format version.
    pub const VERSION: u16 = 1;

    /// Serialize this validated context for application-managed transport.
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        Ok(serde_json::to_string(self).expect("PropagationContext is always JSON serializable"))
    }

    /// Convert this rooted context to a W3C `traceparent` header value.
    pub fn to_traceparent(&self) -> Result<String> {
        self.validate()?;
        let Some(root_uuid) = self.root_uuid else {
            return Err(FlowError::InvalidArgument(
                "rootless propagation context cannot be converted to traceparent".into(),
            ));
        };
        Ok(crate::observability::format_traceparent(
            root_uuid,
            self.parent_uuid,
        ))
    }

    /// Deserialize and validate a context received from application-managed transport.
    pub fn from_json(value: &str) -> Result<Self> {
        let context: Self = serde_json::from_str(value).map_err(|error| {
            FlowError::InvalidArgument(format!("invalid propagation context JSON: {error}"))
        })?;
        context.validate()?;
        Ok(context)
    }

    /// Validate a context received from an untrusted transport.
    pub fn validate(&self) -> Result<()> {
        if self.version != Self::VERSION {
            return Err(FlowError::InvalidArgument(format!(
                "unsupported propagation context version {}; expected {}",
                self.version,
                Self::VERSION
            )));
        }
        for (name, uuid) in [("parent_uuid", self.parent_uuid)]
            .into_iter()
            .chain(self.root_uuid.map(|uuid| ("root_uuid", uuid)))
        {
            let bytes = uuid.as_bytes();
            if bytes.iter().all(|byte| *byte == 0) || bytes[8..].iter().all(|byte| *byte == 0) {
                return Err(FlowError::InvalidArgument(format!(
                    "propagation context {name} is not a usable Relay identifier"
                )));
            }
        }
        Ok(())
    }
}

impl ScopeStack {
    fn snapshot(&self) -> Self {
        Self {
            stack: self.stack.clone(),
            scope_registries: self.scope_registries.clone(),
            fresh_agents: self.fresh_agents.clone(),
            propagated_parent_uuid: self.propagated_parent_uuid,
            propagated_root_uuid: self.propagated_root_uuid,
            is_rootless_propagation: self.is_rootless_propagation,
        }
    }

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
        let root_uuid = root.uuid;
        Self {
            stack: vec![root],
            scope_registries: HashMap::new(),
            fresh_agents: HashSet::from([root_uuid]),
            propagated_parent_uuid: None,
            propagated_root_uuid: None,
            is_rootless_propagation: false,
        }
    }

    fn from_propagation(context: &PropagationContext) -> Result<Self> {
        context.validate()?;
        let (root, parent) = match context.root_uuid {
            Some(root_uuid) => {
                let root = ScopeHandle::builder()
                    .uuid(root_uuid)
                    .name("propagated-root")
                    .scope_type(ScopeType::Agent)
                    .build();
                let parent = (root_uuid != context.parent_uuid).then(|| {
                    ScopeHandle::builder()
                        .uuid(context.parent_uuid)
                        .parent_uuid(root_uuid)
                        .name("propagated-parent")
                        .scope_type(ScopeType::Unknown)
                        .build()
                });
                (root, parent)
            }
            None => (
                ScopeHandle::builder()
                    .uuid(context.parent_uuid)
                    .name("propagated-root")
                    .scope_type(ScopeType::Agent)
                    .build(),
                None,
            ),
        };
        let root_uuid = root.uuid;
        let mut stack = vec![root];
        if let Some(parent) = parent {
            stack.push(parent);
        }
        Ok(Self {
            stack,
            scope_registries: HashMap::new(),
            fresh_agents: HashSet::from([root_uuid]),
            propagated_parent_uuid: context.root_uuid.map(|_| context.parent_uuid),
            propagated_root_uuid: context.root_uuid,
            is_rootless_propagation: context.root_uuid.is_none(),
        })
    }

    /// Push a scope handle onto the top of the stack.
    ///
    /// # Parameters
    /// - `handle`: Scope handle to make the new top-most active scope.
    pub fn push(&mut self, handle: ScopeHandle) {
        if matches!(handle.scope_type, ScopeType::Agent) {
            self.fresh_agents.insert(handle.uuid);
        }
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

    /// Return the causal root that should be attached to emitted events.
    pub(crate) fn event_propagation_root_uuid(&self) -> Option<Uuid> {
        self.propagated_root_uuid
            .or_else(|| {
                self.stack
                    .iter()
                    .skip(1)
                    .find(|scope| scope.scope_type == ScopeType::Agent)
                    .map(|scope| scope.uuid)
            })
            .or_else(|| (!self.is_rootless_propagation).then(|| self.root_uuid()))
    }

    /// Whether `uuid` is the synthetic parent imported from propagation.
    pub fn is_propagated_parent(&self, uuid: Uuid) -> bool {
        self.propagated_parent_uuid == Some(uuid)
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
            self.fresh_agents.remove(uuid);
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

    fn owning_agent_uuid(&self, parent_uuid: Option<Uuid>) -> Uuid {
        let search_end = parent_uuid
            .and_then(|parent_uuid| {
                self.stack
                    .iter()
                    .position(|scope| scope.uuid == parent_uuid)
            })
            .map_or(self.stack.len(), |index| index + 1);
        self.stack[..search_end]
            .iter()
            .rev()
            .find(|scope| matches!(scope.scope_type, ScopeType::Agent))
            .map(|scope| scope.uuid)
            .expect("scope stack should always contain an owning agent")
    }

    /// Return whether the owning agent is fresh, then mark it stale.
    pub(crate) fn take_agent_freshness(&mut self, parent_uuid: Option<Uuid>) -> bool {
        let uuid = self.owning_agent_uuid(parent_uuid);
        self.fresh_agents.remove(&uuid)
    }

    /// Mark the agent that owns a compaction event as fresh.
    pub(crate) fn mark_agent_fresh(&mut self, parent_uuid: Option<Uuid>) {
        let uuid = self.owning_agent_uuid(parent_uuid);
        self.fresh_agents.insert(uuid);
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
    pub(crate) fn local_registries_mut(
        &mut self,
        uuid: &Uuid,
    ) -> Option<&mut ScopeLocalRegistries> {
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
    pub(crate) fn collect_scope_local_registries<'a, T: RegistryEntry>(
        &'a self,
        field: impl Fn(&'a ScopeLocalRegistries) -> &'a SortedRegistry<T>,
    ) -> Vec<&'a SortedRegistry<T>> {
        self.stack
            .iter()
            .filter_map(|handle| self.scope_registries.get(&handle.uuid))
            .map(field)
            .collect()
    }

    /// Clone one registry field from every active scope that owns it.
    ///
    /// Eligibility callbacks must not run while the scope-stack lock is held.
    /// Resolution paths use these owned snapshots before consulting global
    /// conditional middleware guardrails.
    pub(crate) fn snapshot_scope_local_registries<T: RegistryEntry + Clone>(
        &self,
        field: impl Fn(&ScopeLocalRegistries) -> &SortedRegistry<T>,
    ) -> Vec<SortedRegistry<T>> {
        self.collect_scope_local_registries(field)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Collect all scope-local subscribers visible from the active stack.
    ///
    /// # Returns
    /// A vector of subscribers collected from each active scope that owns
    /// scope-local registries.
    pub(crate) fn collect_scope_local_subscribers(&self) -> Vec<EventSubscriberFn> {
        self.stack
            .iter()
            .filter_map(|handle| self.scope_registries.get(&handle.uuid))
            .flat_map(|registries| registries.event_subscribers.values().cloned())
            .collect()
    }
}

impl std::fmt::Debug for ScopeStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopeStack")
            .field("stack", &self.stack)
            .field("scope_registries_count", &self.scope_registries.len())
            .field("fresh_agent_count", &self.fresh_agents.len())
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

impl ThreadScopeStackBinding {
    /// Return the captured thread-local scope stack handle.
    pub fn stack(&self) -> ScopeStackHandle {
        self.stack.clone()
    }
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

/// Clone a scope stack into an isolated emission-time snapshot.
#[doc(hidden)]
pub(crate) fn snapshot_scope_stack(handle: &ScopeStackHandle) -> Result<ScopeStackHandle> {
    let stack = handle
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .snapshot();
    Ok(Arc::new(RwLock::new(stack)))
}

/// Create an isolated scope stack rooted below a supplied propagation context.
///
/// The imported handles are synthetic bookkeeping only; Relay never emits their
/// lifecycle events or transfers scope-local registrations across the boundary.
pub fn create_scope_stack_from_propagation(
    context: &PropagationContext,
) -> Result<ScopeStackHandle> {
    Ok(Arc::new(RwLock::new(ScopeStack::from_propagation(
        context,
    )?)))
}

/// Create an isolated scope stack below the current causal parent.
///
/// Capture the parent before spawning concurrent work, then install the
/// returned stack with `TASK_SCOPE_STACK.scope(...)`. The fork preserves event
/// parentage and its Relay root but does not transfer scope-local registrations.
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> nemo_relay::error::Result<()> {
/// use nemo_relay::api::runtime::{TASK_SCOPE_STACK, fork_scope_stack};
///
/// let stack = fork_scope_stack()?;
/// tokio::spawn(TASK_SCOPE_STACK.scope(stack, async {
///     // Relay work in an isolated child task.
/// }));
/// # Ok(())
/// # }
/// ```
pub fn fork_scope_stack() -> Result<ScopeStackHandle> {
    let context = capture_propagation_context()?;
    create_scope_stack_from_propagation(&context)
}

/// Capture the current causal parent and its Relay root when available.
///
/// Importing the returned context preserves Relay event parentage. A rootless
/// imported stack remains rootless until a local Agent scope establishes a new
/// root; otherwise the context continues the originating Relay-derived
/// observability trace. Use
/// [`capture_rootless_propagation_context`] when the receiver must start a new
/// trace instead.
pub fn capture_propagation_context() -> Result<PropagationContext> {
    let active_uuid = active_event_uuid();
    let parent_uuid = active_uuid.unwrap_or_else(|| task_scope_top().uuid);
    let stack = current_scope_stack();
    let stack_guard = stack
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let root_uuid = stack_guard.event_propagation_root_uuid();
    Ok(PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid,
        parent_uuid,
    })
}

/// Capture the current causal parent without a root UUID.
///
/// Importing the returned context preserves Relay event parentage but starts a
/// new local OpenTelemetry trace.
pub fn capture_rootless_propagation_context() -> Result<PropagationContext> {
    capture_propagation_context_with_root(None)
}

/// Capture the current causal parent and an application-supplied session root.
pub fn capture_propagation_context_with_root(
    root_uuid: Option<Uuid>,
) -> Result<PropagationContext> {
    let context = PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid,
        parent_uuid: ACTIVE_EVENT_UUID
            .try_with(|uuid| *uuid)
            .unwrap_or_else(|_| task_scope_top().uuid),
    };
    context.validate()?;
    Ok(context)
}

/// Capture the current rooted Relay context as a W3C `traceparent` value.
pub fn capture_traceparent() -> Result<String> {
    let active_uuid = active_event_uuid();
    let parent_uuid = active_uuid.unwrap_or_else(|| task_scope_top().uuid);
    let stack = current_scope_stack();
    let stack_guard = stack
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let root_uuid = stack_guard
        .propagated_root_uuid
        .or_else(|| stack_guard.scopes().get(1).map(|scope| scope.uuid))
        .or(active_uuid)
        .ok_or_else(|| {
            FlowError::InvalidArgument(
                "no emitted Relay scope is available for traceparent capture".into(),
            )
        })?;
    Ok(crate::observability::format_traceparent(
        root_uuid,
        parent_uuid,
    ))
}

pub(crate) fn traceparent_for_llm(parent_uuid: Uuid) -> Result<String> {
    let stack = current_scope_stack();
    let stack_guard = stack
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let root_uuid = stack_guard
        .propagated_root_uuid
        .or_else(|| stack_guard.scopes().get(1).map(|scope| scope.uuid))
        .unwrap_or(parent_uuid);
    Ok(crate::observability::format_traceparent(
        root_uuid,
        parent_uuid,
    ))
}

tokio::task_local! {
    /// Task-local scope stack handle used by async execution contexts.
    pub static TASK_SCOPE_STACK: ScopeStackHandle;
    /// Managed tool or LLM event currently executing in this task.
    static ACTIVE_EVENT_UUID: Uuid;
}

/// Run a future with `uuid` as the causally active managed event.
pub async fn with_active_event_uuid<T>(uuid: Uuid, future: impl Future<Output = T>) -> T {
    ACTIVE_EVENT_UUID.scope(uuid, future).await
}

pub(crate) fn active_event_uuid() -> Option<Uuid> {
    ACTIVE_EVENT_UUID.try_with(|uuid| *uuid).ok()
}

thread_local! {
    /// Synchronous override used by native plugin callbacks that need to run a
    /// bounded block with an isolated stack even inside a task-local context.
    static SCOPE_STACK_OVERRIDE: RefCell<Option<ScopeStackHandle>> = const { RefCell::new(None) };
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
    if let Some(stack) = SCOPE_STACK_OVERRIDE.with(|stack| stack.borrow().clone()) {
        return stack;
    }
    TASK_SCOPE_STACK
        .try_with(|stack| stack.clone())
        .unwrap_or_else(|_| THREAD_SCOPE_STACK.with(|stack| stack.borrow().clone()))
}

/// Return a scope stack explicitly bound to the current task or override.
///
/// Unlike [`current_scope_stack`], this does not fall back to ambient
/// thread-local state. Continuation adapters use it to distinguish an
/// intentional per-call scope selection from an unrelated runtime-worker
/// thread binding.
pub(crate) fn current_context_scope_stack() -> Option<ScopeStackHandle> {
    SCOPE_STACK_OVERRIDE
        .with(|stack| stack.borrow().clone())
        .or_else(|| TASK_SCOPE_STACK.try_with(Clone::clone).ok())
}

/// Run a synchronous callback with `handle` as the visible scope stack.
///
/// This override takes precedence over task-local and thread-local stacks for
/// the duration of the callback and is restored even when the callback panics.
pub fn with_scope_stack<T>(handle: ScopeStackHandle, f: impl FnOnce() -> T) -> T {
    struct OverrideGuard {
        previous: Option<ScopeStackHandle>,
    }

    impl Drop for OverrideGuard {
        fn drop(&mut self) {
            let previous = self.previous.take();
            SCOPE_STACK_OVERRIDE.with(|stack| *stack.borrow_mut() = previous);
        }
    }

    let previous = SCOPE_STACK_OVERRIDE.with(|stack| stack.replace(Some(handle)));
    let _guard = OverrideGuard { previous };
    f()
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
    if SCOPE_STACK_OVERRIDE.with(|stack| stack.borrow().is_some()) {
        return true;
    }
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
