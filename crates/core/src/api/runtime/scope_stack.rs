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
use std::sync::{Arc, RwLock, Weak};

use opentelemetry::propagation::TextMapPropagator;
use opentelemetry::trace::{SpanContext, TraceContextExt, TraceFlags, TraceState};
use opentelemetry_sdk::propagation::TraceContextPropagator;
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
    propagated_traceparent: Option<String>,
    propagated_tracestate: Option<String>,
    is_rootless_propagation: bool,
}

/// Versioned, transport-neutral causal context for crossing a Relay boundary.
///
/// Applications are responsible for serializing, transporting, authenticating,
/// and trusting this value. It intentionally contains only Relay identifiers;
/// OpenTelemetry `traceparent` and `tracestate` are optional transport fields.
/// A valid pair continues an upstream OpenTelemetry trace while Relay UUIDs
/// continue to own Relay event parentage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropagationContext {
    /// Wire-format version. Version 1 is the only currently supported value.
    pub version: u16,
    /// Stable session root when the sending application knows one. When this
    /// root and a valid `traceparent` are both absent, the first local
    /// OpenTelemetry span after import starts a new trace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_uuid: Option<Uuid>,
    /// Immediate Relay event or scope that caused the boundary crossing.
    pub parent_uuid: Uuid,
    /// Optional W3C traceparent header for the remote OpenTelemetry parent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub traceparent: Option<String>,
    /// Optional W3C tracestate header associated with `traceparent`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracestate: Option<String>,
}

fn is_usable_relay_identifier(uuid: Uuid) -> bool {
    let bytes = uuid.as_bytes();
    bytes[8..].iter().any(|byte| *byte != 0)
}

impl PropagationContext {
    /// The current wire-format version.
    pub const VERSION: u16 = 1;

    /// Serialize this validated context for application-managed transport.
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        Ok(serde_json::to_string(&self.clone().normalized())
            .expect("PropagationContext is always JSON serializable"))
    }

    /// Convert this context to a W3C `traceparent` header value.
    ///
    /// A valid imported W3C context is forwarded even without a Relay root;
    /// otherwise a Relay root is required to derive a trace ID.
    pub fn to_traceparent(&self) -> Result<String> {
        self.validate()?;
        let context = self.clone().normalized();
        if let Some(traceparent) = context.traceparent {
            return Ok(traceparent);
        }
        let Some(root_uuid) = context.root_uuid else {
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
        Ok(context.normalized())
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
            if !is_usable_relay_identifier(uuid) {
                return Err(FlowError::InvalidArgument(format!(
                    "propagation context {name} is not a usable Relay identifier"
                )));
            }
        }
        Ok(())
    }

    /// Discard invalid W3C headers while retaining valid Relay identifiers.
    pub fn normalized(mut self) -> Self {
        let (traceparent, tracestate, _) =
            normalize_w3c_headers(self.traceparent.as_deref(), self.tracestate.as_deref());
        self.traceparent = traceparent;
        self.tracestate = tracestate;
        self
    }
}

/// Enforce the W3C list grammar before handing vendor state to OpenTelemetry.
/// The SDK parser alone permits control characters, duplicate keys, and long lists.
fn parse_w3c_tracestate(header: &str) -> Option<TraceState> {
    let mut members = Vec::new();
    let mut keys = HashSet::new();
    for (index, member) in header.split(',').enumerate() {
        if index >= 32 {
            return None;
        }
        let member = member.trim_matches([' ', '\t']);
        if member.is_empty() {
            continue;
        }
        let (key, value) = member.split_once('=')?;
        let name_chars = |name: &str| {
            name.bytes().all(|b| {
                b.is_ascii_lowercase()
                    || b.is_ascii_digit()
                    || matches!(b, b'_' | b'-' | b'*' | b'/')
            })
        };
        let valid_key = if let Some((tenant, system)) = key.split_once('@') {
            (1..=241).contains(&tenant.len())
                && tenant.as_bytes()[0].is_ascii_alphanumeric()
                && name_chars(tenant)
                && (1..=14).contains(&system.len())
                && system.as_bytes()[0].is_ascii_lowercase()
                && name_chars(system)
        } else {
            (1..=256).contains(&key.len())
                && key.as_bytes()[0].is_ascii_lowercase()
                && name_chars(key)
        };
        if !valid_key
            || !keys.insert(key)
            || !(1..=256).contains(&value.len())
            || !value
                .bytes()
                .all(|b| (0x20..=0x7e).contains(&b) && !matches!(b, b',' | b'='))
        {
            return None;
        }
        members.push((key, value));
    }
    TraceState::from_key_value(members).ok()
}

/// Validate and canonicalize a W3C header pair without ever logging header values.
pub(crate) fn normalize_w3c_headers(
    traceparent: Option<&str>,
    tracestate: Option<&str>,
) -> (Option<String>, Option<String>, Option<SpanContext>) {
    let Some(traceparent) = traceparent else {
        if tracestate.is_some() {
            log::warn!(target: "nemo_relay.runtime", event = "invalid_w3c_trace_context"; "Ignoring tracestate without traceparent");
        }
        return (None, None, None);
    };
    let mut carrier = HashMap::from([("traceparent".to_string(), traceparent.to_string())]);
    if let Some(tracestate) = tracestate {
        if let Some(state) = parse_w3c_tracestate(tracestate) {
            carrier.insert("tracestate".to_string(), state.header());
        } else {
            log::warn!(target: "nemo_relay.runtime", event = "invalid_w3c_trace_context"; "Ignoring invalid W3C tracestate");
        }
    }
    let context = TraceContextPropagator::new().extract(&carrier);
    if !context.span().span_context().is_valid() {
        log::warn!(target: "nemo_relay.runtime", event = "invalid_w3c_trace_context"; "Ignoring invalid W3C trace context");
        return (None, None, None);
    }
    let mut canonical = HashMap::new();
    TraceContextPropagator::new().inject_context(&context, &mut canonical);
    let tracestate = canonical
        .remove("tracestate")
        .filter(|value| !value.is_empty());
    (
        canonical.remove("traceparent"),
        tracestate,
        Some(context.span().span_context().clone()),
    )
}

pub(crate) fn w3c_span_context(
    traceparent: Option<&str>,
    tracestate: Option<&str>,
) -> Option<SpanContext> {
    normalize_w3c_headers(traceparent, tracestate).2
}

/// Validated W3C Trace Context retained internally across local stack forks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct W3cTraceContext {
    traceparent: String,
    tracestate: Option<String>,
}

impl W3cTraceContext {
    pub(crate) fn new(traceparent: impl Into<String>, tracestate: Option<String>) -> Result<Self> {
        let traceparent = traceparent.into();
        let (traceparent, tracestate, _) =
            normalize_w3c_headers(Some(&traceparent), tracestate.as_deref());
        let Some(traceparent) = traceparent else {
            return Err(FlowError::InvalidArgument(
                "invalid W3C trace context".into(),
            ));
        };
        Ok(Self {
            traceparent,
            tracestate,
        })
    }

    pub(crate) fn traceparent(&self) -> &str {
        &self.traceparent
    }

    pub(crate) fn tracestate(&self) -> Option<&str> {
        self.tracestate.as_deref()
    }

    pub(crate) fn span_context(&self) -> Option<SpanContext> {
        w3c_span_context(Some(&self.traceparent), self.tracestate.as_deref())
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
            propagated_traceparent: self.propagated_traceparent.clone(),
            propagated_tracestate: self.propagated_tracestate.clone(),
            is_rootless_propagation: self.is_rootless_propagation,
        }
    }

    pub(crate) fn root_only_snapshot(&self) -> Self {
        let root = self
            .stack
            .first()
            .expect("scope stack should never be empty")
            .clone();
        let mut scope_registries = HashMap::new();
        if let Some(registries) = self.scope_registries.get(&root.uuid) {
            scope_registries.insert(root.uuid, registries.clone());
        }
        Self {
            stack: vec![root.clone()],
            scope_registries,
            fresh_agents: self
                .fresh_agents
                .contains(&root.uuid)
                .then_some(root.uuid)
                .into_iter()
                .collect(),
            propagated_parent_uuid: self.propagated_parent_uuid,
            propagated_root_uuid: self.propagated_root_uuid,
            propagated_traceparent: self.propagated_traceparent.clone(),
            propagated_tracestate: self.propagated_tracestate.clone(),
            is_rootless_propagation: self.is_rootless_propagation,
        }
    }

    pub(crate) fn snapshot_through_scope(&self, scope_uuid: &Uuid) -> Option<Self> {
        let end = self
            .stack
            .iter()
            .position(|scope| scope.uuid == *scope_uuid)?;
        let stack = self.stack[..=end].to_vec();
        let visible = stack.iter().map(|scope| scope.uuid).collect::<HashSet<_>>();
        Some(Self {
            stack,
            scope_registries: self
                .scope_registries
                .iter()
                .filter(|(uuid, _)| visible.contains(uuid))
                .map(|(uuid, registries)| (*uuid, registries.clone()))
                .collect(),
            fresh_agents: self.fresh_agents.intersection(&visible).copied().collect(),
            propagated_parent_uuid: self.propagated_parent_uuid,
            propagated_root_uuid: self.propagated_root_uuid,
            propagated_traceparent: self.propagated_traceparent.clone(),
            propagated_tracestate: self.propagated_tracestate.clone(),
            is_rootless_propagation: self.is_rootless_propagation,
        })
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
            propagated_traceparent: None,
            propagated_tracestate: None,
            is_rootless_propagation: false,
        }
    }

    fn from_propagation(context: &PropagationContext) -> Result<Self> {
        context.validate()?;
        let context = context.clone().normalized();
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
            // The imported parent identifies the boundary crossing even when
            // Relay intentionally has no propagated root. OTel projections use
            // this marker to recognize a valid W3C remote parent.
            propagated_parent_uuid: Some(context.parent_uuid),
            propagated_root_uuid: context.root_uuid,
            propagated_traceparent: context.traceparent,
            propagated_tracestate: context.tracestate,
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
                    .find(|scope| {
                        scope.scope_type == ScopeType::Agent
                            && is_usable_relay_identifier(scope.uuid)
                    })
                    .map(|scope| scope.uuid)
            })
            .or_else(|| (!self.is_rootless_propagation).then(|| self.root_uuid()))
    }

    /// Return the synthetic parent imported from a rooted propagation context.
    pub(crate) fn event_propagation_parent_uuid(&self) -> Option<Uuid> {
        self.propagated_parent_uuid
    }

    /// Return the W3C parent carried by this stack for an emitted event.
    ///
    /// Local parents also receive a derived context so a subscriber registered
    /// after the parent opened can still join the canonical Relay trace.
    pub(crate) fn event_w3c_headers(
        &self,
        parent_uuid: Option<Uuid>,
    ) -> (Option<String>, Option<String>) {
        let Some(parent_uuid) = parent_uuid else {
            return (None, None);
        };
        let parent_is_propagated = self.propagated_parent_uuid == Some(parent_uuid);
        if parent_is_propagated {
            if self.propagated_traceparent.is_some() {
                return (
                    self.propagated_traceparent.clone(),
                    self.propagated_tracestate.clone(),
                );
            }
            if self.propagated_root_uuid.is_none() {
                return (None, None);
            }
        }
        self.w3c_headers_for_span(parent_uuid, parent_uuid)
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

#[derive(Clone)]
pub(crate) struct AnchoredActiveEvent {
    event_uuid: Uuid,
    // Propagated stacks may share a root UUID, so retain the captured Arc allocation identity.
    scope_stack: Weak<RwLock<ScopeStack>>,
    anchor_scope_uuid: Uuid,
}

/// Captured thread-local scope stack binding.
///
/// This preserves the visible scope stack handle, whether it was explicitly
/// installed on the current thread, and any managed event context bound at
/// capture time.
#[derive(Clone)]
pub struct ThreadScopeStackBinding {
    stack: ScopeStackHandle,
    explicit: bool,
    active_event: Option<AnchoredActiveEvent>,
    active_event_trace_context: Option<W3cTraceContext>,
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

pub(crate) fn root_scope_stack_snapshot(stack: &ScopeStackHandle) -> Result<ScopeStackHandle> {
    let root_only = stack
        .read()
        .map_err(|error| FlowError::Internal(format!("scope stack lock poisoned: {error}")))?
        .root_only_snapshot();
    Ok(Arc::new(RwLock::new(root_only)))
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
/// parentage, its Relay root, and the current W3C Trace Context, but does not
/// transfer scope-local registrations.
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

fn captured_w3c_headers(
    stack: &ScopeStack,
    active_uuid: Option<Uuid>,
    parent_uuid: Uuid,
) -> (Option<String>, Option<String>) {
    if let Some(active_context) = active_event_trace_context() {
        (
            Some(active_context.traceparent.clone()),
            active_context.tracestate.clone(),
        )
    } else if active_uuid.is_some() {
        // A managed callback is an emitted local span even though its UUID is
        // not stored on the lexical scope stack. Reparent the active trace to
        // that callback before propagating it.
        stack.w3c_headers_for_span(parent_uuid, stack.top().uuid)
    } else {
        // A synthetic imported parent may have an unrelated W3C span ID.
        // Preserve that exact remote parent until Relay emits a local span.
        stack.event_w3c_headers(Some(parent_uuid))
    }
}

fn captured_imported_w3c_headers(
    stack: &ScopeStack,
    parent_uuid: Uuid,
) -> (Option<String>, Option<String>) {
    if stack.propagated_parent_uuid == Some(parent_uuid) {
        return (
            stack.propagated_traceparent.clone(),
            stack.propagated_tracestate.clone(),
        );
    }
    w3c_span_context(
        stack.propagated_traceparent.as_deref(),
        stack.propagated_tracestate.as_deref(),
    )
    .map(|parent| w3c_headers_for_parent(parent, parent_uuid))
    .unwrap_or_default()
}

/// Capture the current causal parent and its Relay root when available.
///
/// Importing the returned context preserves Relay event parentage. A rootless
/// imported stack remains rootless until a local Agent scope establishes a new
/// root; otherwise the context continues the originating Relay-derived
/// observability trace. Use [`capture_rootless_propagation_context`] when the
/// receiver must omit the Relay root; a previously imported W3C parent is
/// still retained.
pub fn capture_propagation_context() -> Result<PropagationContext> {
    let active_uuid = active_event_uuid();
    let parent_uuid = active_uuid.unwrap_or_else(|| task_scope_top().uuid);
    let stack = current_scope_stack();
    let stack_guard = stack
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let root_uuid = stack_guard.event_propagation_root_uuid();
    let (traceparent, tracestate) = captured_w3c_headers(&stack_guard, active_uuid, parent_uuid);
    let traceparent = traceparent
        .or_else(|| active_uuid.map(|uuid| crate::observability::format_traceparent(uuid, uuid)));
    let context = PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid,
        parent_uuid,
        traceparent,
        tracestate,
    };
    context.validate()?;
    Ok(context)
}

/// Capture the current causal parent without a root UUID.
///
/// Importing the returned context preserves Relay event parentage and any
/// previously imported W3C parent while omitting the Relay root. A locally
/// derived W3C parent is omitted, so the receiver starts a separate trace.
pub fn capture_rootless_propagation_context() -> Result<PropagationContext> {
    let parent_uuid = active_event_uuid().unwrap_or_else(|| task_scope_top().uuid);
    let stack = current_scope_stack();
    let stack_guard = stack
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let (traceparent, tracestate) = captured_imported_w3c_headers(&stack_guard, parent_uuid);
    let context = PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: None,
        parent_uuid,
        traceparent,
        tracestate,
    };
    context.validate()?;
    Ok(context)
}

/// Capture the current causal parent and an application-supplied session root.
/// Without an imported W3C parent, the supplied root determines the receiver's
/// trace. An imported W3C parent retains precedence over the Relay root.
/// Passing `None` has the same behavior as [`capture_rootless_propagation_context`].
pub fn capture_propagation_context_with_root(
    root_uuid: Option<Uuid>,
) -> Result<PropagationContext> {
    let Some(root_uuid) = root_uuid else {
        return capture_rootless_propagation_context();
    };
    let mut context = capture_rootless_propagation_context()?;
    context.root_uuid = Some(root_uuid);
    context.validate()?;
    Ok(context)
}

pub(crate) fn capture_w3c_trace_context() -> Result<W3cTraceContext> {
    let context = capture_propagation_context()?;
    let Some(traceparent) = context.traceparent else {
        return Err(FlowError::InvalidArgument(
            "no emitted Relay scope is available for W3C trace context capture".into(),
        ));
    };
    W3cTraceContext::new(traceparent, context.tracestate).map_err(|_| {
        FlowError::InvalidArgument(
            "no emitted Relay scope is available for W3C trace context capture".into(),
        )
    })
}

/// Capture the current canonical W3C `traceparent` value.
pub fn capture_traceparent() -> Result<String> {
    Ok(capture_w3c_trace_context()?.traceparent)
}

impl ScopeStack {
    fn propagated_span_context(&self, uuid: Uuid) -> Option<SpanContext> {
        if self.propagated_parent_uuid != Some(uuid) {
            return None;
        }
        if let Some(context) = w3c_span_context(
            self.propagated_traceparent.as_deref(),
            self.propagated_tracestate.as_deref(),
        ) {
            return Some(context);
        }
        let root_uuid = self.propagated_root_uuid?;
        if !is_usable_relay_identifier(root_uuid) || !is_usable_relay_identifier(uuid) {
            return None;
        }
        Some(SpanContext::new(
            crate::observability::relay_trace_id(root_uuid),
            crate::observability::relay_span_id(uuid),
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        ))
    }

    fn local_span_context(&self, uuid: Uuid) -> Option<SpanContext> {
        self.local_span_context_inner(uuid, self.stack.len(), &mut HashSet::new())
    }

    fn local_span_context_inner(
        &self,
        uuid: Uuid,
        before_index: usize,
        visiting: &mut HashSet<Uuid>,
    ) -> Option<SpanContext> {
        if !is_usable_relay_identifier(uuid) || !visiting.insert(uuid) {
            return None;
        }
        if let Some(context) = self.propagated_span_context(uuid) {
            visiting.remove(&uuid);
            return Some(context);
        }
        let (index, scope) = self
            .stack
            .iter()
            .enumerate()
            .take(before_index)
            .skip(1)
            .rev()
            .find(|(_, scope)| scope.uuid == uuid)?;
        let stack_parent_context = |visiting: &mut HashSet<Uuid>| {
            index
                .checked_sub(1)
                .and_then(|parent_index| self.stack.get(parent_index))
                .and_then(|parent| self.parent_span_context_inner(parent.uuid, index, visiting))
        };
        let parent_context = match scope.parent_uuid {
            Some(parent_uuid) => self
                .parent_span_context_inner(parent_uuid, index, visiting)
                .or_else(|| {
                    // A stacked scope can also name a completed parent. Use
                    // the scope below it, not itself, to recover the trace.
                    if is_usable_relay_identifier(parent_uuid) && self.find(&parent_uuid).is_none()
                    {
                        stack_parent_context(visiting)
                    } else {
                        None
                    }
                }),
            None => stack_parent_context(visiting),
        };
        let context = match parent_context {
            Some(parent) => SpanContext::new(
                parent.trace_id(),
                crate::observability::relay_span_id(uuid),
                parent.trace_flags(),
                false,
                parent.trace_state().clone(),
            ),
            None => SpanContext::new(
                crate::observability::relay_trace_id(uuid),
                crate::observability::relay_span_id(uuid),
                TraceFlags::SAMPLED,
                false,
                TraceState::default(),
            ),
        };
        visiting.remove(&uuid);
        context.is_valid().then_some(context)
    }

    fn parent_span_context_inner(
        &self,
        uuid: Uuid,
        before_index: usize,
        visiting: &mut HashSet<Uuid>,
    ) -> Option<SpanContext> {
        self.propagated_span_context(uuid)
            .or_else(|| self.local_span_context_inner(uuid, before_index, visiting))
    }

    fn w3c_headers_for_span(
        &self,
        span_uuid: Uuid,
        causal_parent_uuid: Uuid,
    ) -> (Option<String>, Option<String>) {
        if span_uuid == causal_parent_uuid
            && self.propagated_parent_uuid == Some(causal_parent_uuid)
            && self.propagated_traceparent.is_some()
        {
            return (
                self.propagated_traceparent.clone(),
                self.propagated_tracestate.clone(),
            );
        }
        self.local_span_context(causal_parent_uuid)
            .or_else(|| {
                // Explicit handles may refer to completed scopes. Preserve the
                // stack's trace fallback when the parent is no longer present;
                // the handle's UUID alone cannot recover a different trace.
                if is_usable_relay_identifier(causal_parent_uuid)
                    && self.find(&causal_parent_uuid).is_none()
                {
                    self.local_span_context(self.top().uuid)
                } else {
                    None
                }
            })
            .map(|parent| w3c_headers_for_parent(parent, span_uuid))
            .unwrap_or_default()
    }
}

fn w3c_headers_for_parent(
    parent: SpanContext,
    parent_uuid: Uuid,
) -> (Option<String>, Option<String>) {
    let child = SpanContext::new(
        parent.trace_id(),
        crate::observability::relay_span_id(parent_uuid),
        parent.trace_flags(),
        false,
        parent.trace_state().clone(),
    );
    let context = opentelemetry::Context::new().with_remote_span_context(child);
    let mut carrier = HashMap::new();
    TraceContextPropagator::new().inject_context(&context, &mut carrier);
    (
        carrier.remove("traceparent"),
        carrier
            .remove("tracestate")
            .filter(|value| !value.is_empty()),
    )
}

pub(crate) fn trace_context_for_managed_span(
    span_uuid: Uuid,
    causal_parent_uuid: Option<Uuid>,
) -> Result<W3cTraceContext> {
    let stack = current_scope_stack();
    let stack_guard = stack
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let (traceparent, tracestate) = if causal_parent_uuid == active_event_uuid() {
        active_event_trace_context()
            .and_then(|context| context.span_context())
            .map(|context| w3c_headers_for_parent(context, span_uuid))
            .unwrap_or_else(|| stack_guard.w3c_headers_for_span(span_uuid, stack_guard.top().uuid))
    } else {
        causal_parent_uuid
            .map(|parent_uuid| stack_guard.w3c_headers_for_span(span_uuid, parent_uuid))
            .unwrap_or_default()
    };
    W3cTraceContext::new(
        traceparent
            .unwrap_or_else(|| crate::observability::format_traceparent(span_uuid, span_uuid)),
        tracestate,
    )
}

pub(crate) fn capture_trace_context() -> Result<(String, Option<String>)> {
    let context = capture_w3c_trace_context()?;
    Ok((context.traceparent, context.tracestate))
}

tokio::task_local! {
    /// Task-local scope stack handle used by async execution contexts.
    pub static TASK_SCOPE_STACK: ScopeStackHandle;
    /// Managed tool or LLM event currently executing in this task.
    static ACTIVE_EVENT: AnchoredActiveEvent;
    /// Exact W3C context of the managed event when one was captured at start.
    static ACTIVE_EVENT_TRACE_CONTEXT: Option<W3cTraceContext>;
}

/// Run a future with `uuid` as the causally active managed event.
pub async fn with_active_event_uuid<T>(uuid: Uuid, future: impl Future<Output = T>) -> T {
    with_active_event_trace_context(uuid, None, future).await
}

pub(crate) async fn with_active_event_trace_context<T>(
    uuid: Uuid,
    trace_context: Option<W3cTraceContext>,
    future: impl Future<Output = T>,
) -> T {
    let (scope_stack, anchor_scope_uuid) = scope_stack_identity_and_anchor();
    let active_event = AnchoredActiveEvent {
        event_uuid: uuid,
        scope_stack,
        anchor_scope_uuid,
    };
    with_anchored_active_event(active_event, trace_context, future).await
}

/// Bind a synchronous tool lifecycle subscriber to the observed tool scope.
/// The event may already be closed, so anchor it to the delivery snapshot rather
/// than changing the live scope stack or reopening the tool.
pub(crate) fn with_thread_active_event_trace_context<T>(
    uuid: Uuid,
    trace_context: Option<W3cTraceContext>,
    callback: impl FnOnce() -> T,
) -> T {
    struct RestoreActiveEvent {
        event: Option<AnchoredActiveEvent>,
        trace_context: Option<W3cTraceContext>,
    }

    impl Drop for RestoreActiveEvent {
        fn drop(&mut self) {
            THREAD_ACTIVE_EVENT.with(|event| *event.borrow_mut() = self.event.take());
            THREAD_ACTIVE_EVENT_TRACE_CONTEXT
                .with(|context| *context.borrow_mut() = self.trace_context.take());
        }
    }

    let (scope_stack, anchor_scope_uuid) = scope_stack_identity_and_anchor();
    let event = AnchoredActiveEvent {
        event_uuid: uuid,
        scope_stack,
        anchor_scope_uuid,
    };
    let _restore = RestoreActiveEvent {
        event: THREAD_ACTIVE_EVENT.with(|current| current.replace(Some(event))),
        trace_context: THREAD_ACTIVE_EVENT_TRACE_CONTEXT
            .with(|current| current.replace(trace_context)),
    };
    callback()
}

pub(crate) async fn with_anchored_active_event<T>(
    active_event: AnchoredActiveEvent,
    trace_context: Option<W3cTraceContext>,
    future: impl Future<Output = T>,
) -> T {
    ACTIVE_EVENT
        .scope(
            active_event,
            ACTIVE_EVENT_TRACE_CONTEXT.scope(trace_context, future),
        )
        .await
}

pub(crate) fn capture_anchored_active_event() -> Option<AnchoredActiveEvent> {
    ACTIVE_EVENT
        .try_with(Clone::clone)
        .ok()
        .or_else(thread_active_event)
}

pub(crate) fn rebind_active_event_to_stack(
    active_event: AnchoredActiveEvent,
    scope_stack: &ScopeStackHandle,
) -> AnchoredActiveEvent {
    let guard = scope_stack
        .read()
        .unwrap_or_else(|error| error.into_inner());
    AnchoredActiveEvent {
        event_uuid: active_event.event_uuid,
        scope_stack: Arc::downgrade(scope_stack),
        anchor_scope_uuid: if guard.find(&active_event.anchor_scope_uuid).is_some() {
            active_event.anchor_scope_uuid
        } else {
            guard.top().uuid
        },
    }
}

pub(crate) fn active_event_uuid() -> Option<Uuid> {
    ACTIVE_EVENT
        .try_with(|event| event.event_uuid)
        .ok()
        .or_else(thread_active_event_uuid)
}

pub(crate) fn active_event_trace_context() -> Option<W3cTraceContext> {
    match ACTIVE_EVENT_TRACE_CONTEXT.try_with(Clone::clone) {
        Ok(context) => context,
        Err(_) => {
            thread_active_event()?;
            THREAD_ACTIVE_EVENT_TRACE_CONTEXT.with(|context| context.borrow().clone())
        }
    }
}

fn thread_active_event() -> Option<AnchoredActiveEvent> {
    let mut event = THREAD_ACTIVE_EVENT.with(|active| active.borrow().clone())?;
    let stack = current_scope_stack();
    let event_stack = event.scope_stack.upgrade()?;
    if !Arc::ptr_eq(&event_stack, &stack) {
        return None;
    }
    let guard = stack.read().unwrap_or_else(|error| error.into_inner());
    if event.anchor_scope_uuid != guard.top().uuid {
        if guard.find(&event.anchor_scope_uuid).is_some() {
            return None;
        }
        event.anchor_scope_uuid = guard.top().uuid;
        THREAD_ACTIVE_EVENT.with(|active| *active.borrow_mut() = Some(event.clone()));
    }
    Some(event)
}

pub(crate) fn thread_active_event_uuid() -> Option<Uuid> {
    thread_active_event().map(|event| event.event_uuid)
}

thread_local! {
    /// Synchronous override used by native plugin callbacks that need to run a
    /// bounded block with an isolated stack even inside a task-local context.
    static SCOPE_STACK_OVERRIDE: RefCell<Option<ScopeStackHandle>> = const { RefCell::new(None) };
    /// Thread-local fallback scope stack for non-task contexts.
    static THREAD_SCOPE_STACK: RefCell<ScopeStackHandle> = RefCell::new(create_scope_stack());
    /// Whether the current thread explicitly owns a scope stack.
    static THREAD_SCOPE_STACK_EXPLICIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Managed event propagated into a foreign executor with the thread scope binding.
    static THREAD_ACTIVE_EVENT: RefCell<Option<AnchoredActiveEvent>> = const { RefCell::new(None) };
    /// Exact W3C context associated with the propagated managed event.
    static THREAD_ACTIVE_EVENT_TRACE_CONTEXT: RefCell<Option<W3cTraceContext>> = const { RefCell::new(None) };
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
    clear_thread_active_event_for_stack_change(&handle);
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
/// A [`ThreadScopeStackBinding`] containing the current thread-local stack,
/// explicit-binding flag, and active managed event context.
pub fn capture_thread_scope_stack() -> ThreadScopeStackBinding {
    let stack = THREAD_SCOPE_STACK.with(|stack| stack.borrow().clone());
    let explicit = THREAD_SCOPE_STACK_EXPLICIT.with(|flag| flag.get());
    ThreadScopeStackBinding {
        stack,
        explicit,
        active_event: THREAD_ACTIVE_EVENT.with(|event| event.borrow().clone()),
        active_event_trace_context: THREAD_ACTIVE_EVENT_TRACE_CONTEXT
            .with(|context| context.borrow().clone()),
    }
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
    THREAD_ACTIVE_EVENT.with(|event| *event.borrow_mut() = binding.active_event);
    THREAD_ACTIVE_EVENT_TRACE_CONTEXT
        .with(|context| *context.borrow_mut() = binding.active_event_trace_context);
}

#[cfg(feature = "worker-grpc")]
pub(crate) fn install_thread_continuation_context(
    scope_stack: &ScopeStackHandle,
    active_event: Option<AnchoredActiveEvent>,
    active_event_trace_context: Option<W3cTraceContext>,
) -> ThreadScopeStackBinding {
    let previous = capture_thread_scope_stack();
    sync_thread_scope_stack(scope_stack.clone());
    THREAD_ACTIVE_EVENT.with(|event| {
        *event.borrow_mut() =
            active_event.map(|event| rebind_active_event_to_stack(event, scope_stack));
    });
    THREAD_ACTIVE_EVENT_TRACE_CONTEXT
        .with(|context| *context.borrow_mut() = active_event_trace_context);
    previous
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
    clear_thread_active_event_for_stack_change(&handle);
    THREAD_SCOPE_STACK.with(|stack| *stack.borrow_mut() = handle);
}

fn clear_thread_active_event_for_stack_change(handle: &ScopeStackHandle) {
    let stack_changed = THREAD_SCOPE_STACK.with(|current| !Arc::ptr_eq(&current.borrow(), handle));
    if stack_changed {
        THREAD_ACTIVE_EVENT.with(|event| *event.borrow_mut() = None);
        THREAD_ACTIVE_EVENT_TRACE_CONTEXT.with(|context| *context.borrow_mut() = None);
    }
}

/// Synchronize the task-local managed event onto an isolated thread stack.
///
/// Native async callbacks run on a plugin-owned executor. The host snapshots
/// the callback's visible stack before crossing that boundary, so the managed
/// event must be rebound to the snapshot's allocation while retaining its
/// original scope anchor and W3C context.
pub(crate) fn sync_thread_active_event_for_stack(scope_stack: &ScopeStackHandle) {
    let active_event = capture_anchored_active_event()
        .map(|active_event| rebind_active_event_to_stack(active_event, scope_stack));
    let trace_context = active_event
        .as_ref()
        .and_then(|_| active_event_trace_context());
    THREAD_ACTIVE_EVENT.with(|event| *event.borrow_mut() = active_event);
    THREAD_ACTIVE_EVENT_TRACE_CONTEXT.with(|context| *context.borrow_mut() = trace_context);
}

fn scope_stack_identity_and_anchor() -> (Weak<RwLock<ScopeStack>>, Uuid) {
    let stack = current_scope_stack();
    let guard = stack.read().unwrap_or_else(|error| error.into_inner());
    (Arc::downgrade(&stack), guard.top().uuid)
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
