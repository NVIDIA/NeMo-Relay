// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::api::event::{BaseEvent, MarkEvent};
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::global_context;
use crate::api::runtime::{
    current_scope_stack, task_scope_push, task_scope_remove, task_scope_top,
};
use crate::api::shared::{ensure_runtime_owner, resolve_parent_uuid, snapshot_event_subscribers};
use crate::error::{FlowError, Result};
use crate::json::Json;
use bitflags::bitflags;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;
use uuid::Uuid;

use crate::api::llm::LlmAttributes;
use crate::api::tool::ToolAttributes;

bitflags! {
    /// Bitflags that modify scope behavior and observability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ScopeAttributes: u32 {
        /// Marks the scope as running in parallel with sibling work.
        const PARALLEL    = 0b01;
        /// Marks the scope as safe to move across execution contexts.
        const RELOCATABLE = 0b10;
    }
}

/// Semantic category attached to a scope lifecycle span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScopeType {
    /// A top-level agent or workflow scope.
    Agent,
    /// A generic function or application step.
    Function,
    /// A tool lifecycle scope.
    Tool,
    /// An LLM lifecycle scope.
    Llm,
    /// A retrieval step such as document search.
    Retriever,
    /// An embedding generation step.
    Embedder,
    /// A reranking step.
    Reranker,
    /// A guardrail or validation step.
    Guardrail,
    /// An evaluation or scoring step.
    Evaluator,
    /// A caller-defined custom scope category.
    Custom,
    /// A fallback for unknown or unsupported scope categories.
    Unknown,
}

impl ScopeType {
    /// Return the stable lowercase string form used for encoded scope types.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Function => "function",
            Self::Tool => "tool",
            Self::Llm => "llm",
            Self::Retriever => "retriever",
            Self::Embedder => "embedder",
            Self::Reranker => "reranker",
            Self::Guardrail => "guardrail",
            Self::Evaluator => "evaluator",
            Self::Custom => "custom",
            Self::Unknown => "unknown",
        }
    }
}

/// Attribute bitflags attached to a concrete handle kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandleAttributes {
    /// Scope-specific attributes.
    Scope(ScopeAttributes),
    /// Tool-specific attributes.
    Tool(ToolAttributes),
    /// LLM-specific attributes.
    Llm(LlmAttributes),
}

/// Runtime-owned handle identifying an active or completed scope.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct ScopeHandle {
    /// Unique scope identifier.
    #[builder(default = Uuid::now_v7())]
    pub uuid: Uuid,
    /// Timestamp captured when the scope handle was created.
    #[builder(default = Utc::now())]
    pub started_at: DateTime<Utc>,
    /// Semantic category of the scope.
    pub scope_type: ScopeType,
    /// Human-readable scope name.
    #[builder(setter(into))]
    pub name: String,
    /// Optional application payload stored on the handle.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional metadata attached to the scope.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Scope behavior flags.
    #[builder(default = ScopeAttributes::empty())]
    pub attributes: ScopeAttributes,
    /// UUID of the parent scope, if any.
    #[builder(default)]
    pub parent_uuid: Option<Uuid>,
}

/// Builder parameters for [`push_scope`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct PushScopeParams<'a> {
    /// Human-readable scope name recorded on emitted lifecycle events.
    pub name: &'a str,
    /// Semantic category for the new scope.
    pub scope_type: ScopeType,
    /// Optional explicit parent scope.
    #[builder(default)]
    pub parent: Option<&'a ScopeHandle>,
    /// Scope attribute bitflags applied to the new scope.
    #[builder(default = ScopeAttributes::empty())]
    pub attributes: ScopeAttributes,
    /// Optional application payload stored on the scope handle.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional JSON metadata recorded on the emitted start event.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional JSON payload exported as the scope start event data.
    #[builder(default)]
    pub input: Option<Json>,
    /// Optional timestamp recorded on the emitted start event.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Builder parameters for [`NemoRelayContextState::create_scope_handle`].
#[derive(Debug, Clone, TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct CreateScopeHandleParams<'a> {
    /// Human-readable scope name.
    pub name: &'a str,
    /// Optional parent scope UUID.
    #[builder(default)]
    pub parent_uuid: Option<Uuid>,
    /// Semantic category of the scope.
    pub scope_type: ScopeType,
    /// Scope attribute bitflags.
    #[builder(default = ScopeAttributes::empty())]
    pub attributes: ScopeAttributes,
    /// Optional application payload stored on the handle.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional metadata stored on the handle.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional timestamp captured as the handle start time and reused by the
    /// emitted start event. When omitted, the current UTC time is used.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Builder parameters for [`NemoRelayContextState::build_scope_end_event`].
#[derive(Debug, Clone, TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct EndScopeHandleParams<'a> {
    /// Scope handle to serialize into the emitted end event.
    pub handle: &'a ScopeHandle,
    /// Optional JSON payload exported as the semantic scope output.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional metadata to be appended to the metadata set when the scope was created.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional timestamp recorded on the emitted end event. When omitted, the
    /// runtime records the current UTC time, or one microsecond after the
    /// handle start time if the current time is not later.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Builder parameters for [`pop_scope`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct PopScopeParams<'a> {
    /// UUID of the scope that should be popped.
    pub handle_uuid: &'a Uuid,
    /// Optional JSON payload exported as the semantic scope output.
    #[builder(default)]
    pub output: Option<Json>,
    /// Optional JSON payload metadata to be appended to the metadata set when the scope was created.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional timestamp recorded on the emitted end event. When omitted, the
    /// runtime records the current UTC time, or one microsecond after the
    /// handle start time if the current time is not later.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Builder parameters for [`event`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct EmitMarkEventParams<'a> {
    /// Event name to emit.
    pub name: &'a str,
    /// Optional explicit parent scope.
    #[builder(default)]
    pub parent: Option<&'a ScopeHandle>,
    /// Optional JSON payload recorded as the mark data.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional JSON metadata recorded on the emitted event.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional timestamp recorded on the emitted mark event. When omitted, the
    /// current UTC time is used.
    #[builder(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Return the current scope at the top of the active stack.
///
/// This reads the task-local or thread-local scope stack without mutating it
/// and returns a clone of the current top-most [`ScopeHandle`].
///
/// # Returns
/// A [`Result`] containing the current [`ScopeHandle`] when the runtime owner
/// check succeeds.
///
/// # Errors
/// Returns an error when the current binding has not initialized the shared
/// runtime ownership correctly.
pub fn get_handle() -> Result<ScopeHandle> {
    ensure_runtime_owner()?;
    Ok(task_scope_top())
}

/// Push a new scope onto the active scope stack.
///
/// This creates a new [`ScopeHandle`], emits a scope-start event to global and
/// scope-local subscribers, and makes the new scope the current top of stack.
///
/// # Parameters
/// - `name`: Human-readable scope name recorded on emitted lifecycle events.
/// - `scope_type`: Semantic category for the new scope.
/// - `parent`: Optional explicit parent scope. When `None`, the current top of
///   stack is used as the parent.
/// - `attributes`: Bitflags that modify scope behavior and observability.
/// - `data`: Optional application payload stored on the returned handle.
/// - `metadata`: Optional JSON metadata recorded on the emitted start event.
/// - `input`: Optional JSON payload exported as the Agent Trajectory
///   Observability Format (ATOF) data payload.
/// - `timestamp`: Optional timestamp recorded as the handle start time and on
///   the emitted start event. When `None`, the current UTC time is used.
///
/// # Returns
/// A [`Result`] containing the newly created [`ScopeHandle`].
///
/// # Errors
/// Returns an error when the runtime owner check fails or when internal state
/// cannot be read safely.
///
/// # Notes
/// Scope-local subscribers attached to ancestor scopes observe the emitted
/// start event before the function returns.
pub fn push_scope(params: PushScopeParams<'_>) -> Result<ScopeHandle> {
    ensure_runtime_owner()?;
    let parent_uuid = resolve_parent_uuid(params.parent);
    let (handle, event, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let handle_params = CreateScopeHandleParams::builder()
            .name(params.name)
            .parent_uuid_opt(parent_uuid)
            .scope_type(params.scope_type)
            .attributes(params.attributes)
            .data_opt(params.data)
            .metadata_opt(params.metadata)
            .timestamp_opt(params.timestamp)
            .build();
        let handle = state.create_scope_handle(handle_params);
        let event = state.build_scope_start_event(&handle, params.input);
        (handle, event, subscribers)
    };
    task_scope_push(handle.clone());
    NemoRelayContextState::emit_event(&event, &subscribers);
    Ok(handle)
}

/// Pop the current scope from the active scope stack.
///
/// This emits a scope-end event for the target scope and removes any
/// scope-local registrations owned by that scope.
///
/// # Parameters
/// - `handle_uuid`: UUID of the scope that should be popped.
/// - `output`: Optional JSON payload exported as the semantic scope output.
/// - `timestamp`: Optional timestamp recorded on the emitted end event. When
///   `None`, the runtime uses the current UTC time, or one microsecond after
///   the handle start time if the current time is not later.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when the scope was popped successfully.
///
/// # Errors
/// Returns [`FlowError::InvalidArgument`] when the target scope exists but is
/// not the current top of stack, and [`FlowError::NotFound`] when the UUID is
/// unknown to the active stack.
///
/// # Notes
/// The implicit root scope cannot be removed.
pub fn pop_scope(params: PopScopeParams<'_>) -> Result<()> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let (scope, event, subscribers) = {
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let top = scope_guard.top();
        if top.uuid != *params.handle_uuid {
            if scope_guard.find(params.handle_uuid).is_some() {
                return Err(FlowError::InvalidArgument(
                    "scope handle is not at the top of the stack".into(),
                ));
            }
            return Err(FlowError::NotFound("scope handle not found".into()));
        }
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let scope = top.clone();
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let event = state.build_scope_end_event(
            EndScopeHandleParams::builder()
                .handle(&scope)
                .data_opt(params.output)
                .timestamp_opt(params.timestamp)
                .metadata_opt(params.metadata)
                .build(),
        );
        (scope, event, subscribers)
    };
    let removed = task_scope_remove(params.handle_uuid)?;
    debug_assert_eq!(removed.uuid, scope.uuid);
    NemoRelayContextState::emit_event(&event, &subscribers);
    Ok(())
}

/// Emit a standalone mark event under the current or provided scope.
///
/// This creates a point-in-time lifecycle event without pushing or popping a
/// new scope.
///
/// # Parameters
/// - `name`: Event name to emit.
/// - `parent`: Optional explicit parent scope. When `None`, the current top of
///   stack is used.
/// - `data`: Optional JSON payload recorded on the emitted event.
/// - `metadata`: Optional JSON metadata recorded on the emitted event.
/// - `timestamp`: Optional timestamp recorded on the emitted mark event. When
///   `None`, the current UTC time is used.
///
/// # Returns
/// A [`Result`] that is `Ok(())` after the event has been emitted.
///
/// # Errors
/// Returns an error when the runtime owner check fails or when internal state
/// cannot be read safely.
///
/// # Notes
/// Scope-local subscribers attached to ancestor scopes observe the emitted
/// mark event just like scope, tool, and LLM lifecycle events.
pub fn event(params: EmitMarkEventParams<'_>) -> Result<()> {
    ensure_runtime_owner()?;
    let parent_uuid = resolve_parent_uuid(params.parent);
    let (event, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let event = state.create_event(MarkEvent::new(
            BaseEvent::builder()
                .name(params.name)
                .parent_uuid_opt(parent_uuid)
                .timestamp(params.timestamp.unwrap_or_else(Utc::now))
                .data_opt(params.data)
                .metadata_opt(params.metadata)
                .build(),
            None,
            None,
        ));
        (event, subscribers)
    };
    NemoRelayContextState::emit_event(&event, &subscribers);
    Ok(())
}
