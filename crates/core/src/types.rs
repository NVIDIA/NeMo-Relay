// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core data types for the Nexus runtime.
//!
//! This module defines the fundamental types used throughout the framework:
//!
//! - **Attribute bitflags** — [`ScopeAttributes`], [`ToolAttributes`], [`LLMAttributes`]
//! - **Enums** — [`ScopeType`]
//! - **Handle types** — [`ScopeHandle`], [`ToolHandle`], [`LLMHandle`], [`HandleAttributes`]
//! - **Request/response types** — [`LLMRequest`]
//! - **Event types** — [`Event`]
//! - **Middleware containers** — [`Intercept`], [`ExecutionIntercept`], [`GuardrailEntry`]

use bitflags::bitflags;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use std::sync::Arc;

use crate::codec::{AnnotatedLLMRequest, AnnotatedLLMResponse};
use crate::json::Json;

// ---------------------------------------------------------------------------
// Attribute flags
// ---------------------------------------------------------------------------

bitflags! {
    /// Attribute flags for execution scopes.
    ///
    /// These flags describe behavioral properties of a scope and can be combined
    /// using bitwise OR.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ScopeAttributes: u32 {
        /// The scope supports parallel execution of child operations.
        const PARALLEL    = 0b01;
        /// The scope can be relocated (moved between execution contexts).
        const RELOCATABLE = 0b10;
    }
}

bitflags! {
    /// Attribute flags for tool handles.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ToolAttributes: u32 {
        /// The tool executes locally (as opposed to a remote/API tool).
        const LOCAL = 0b01;
    }
}

bitflags! {
    /// Attribute flags for LLM handles.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct LLMAttributes: u32 {
        /// The LLM call is stateless (no conversation history maintained).
        const STATELESS = 0b01;
        /// The LLM call uses streaming (SSE) responses.
        const STREAMING = 0b10;
    }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// The type of an execution scope, indicating what kind of component owns it.
///
/// Serializes to/from lowercase strings (e.g., `"agent"`, `"tool"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScopeType {
    /// An autonomous agent scope.
    Agent,
    /// A function/subroutine scope.
    Function,
    /// A tool invocation scope.
    Tool,
    /// An LLM call scope.
    Llm,
    /// A retriever (e.g., vector search) scope.
    Retriever,
    /// An embedding model scope.
    Embedder,
    /// A reranker model scope.
    Reranker,
    /// A guardrail evaluation scope.
    Guardrail,
    /// An evaluator/judge scope.
    Evaluator,
    /// A user-defined custom scope type.
    Custom,
    /// An unknown or unspecified scope type.
    Unknown,
}

// ---------------------------------------------------------------------------
// Handle types
// ---------------------------------------------------------------------------

/// Unified attributes enum so an [`Event`] can carry the attribute set of
/// whichever handle type produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandleAttributes {
    /// Attributes from a [`ScopeHandle`].
    Scope(ScopeAttributes),
    /// Attributes from a [`ToolHandle`].
    Tool(ToolAttributes),
    /// Attributes from an [`LLMHandle`].
    Llm(LLMAttributes),
}

/// A handle representing an active execution scope in the scope stack.
///
/// Scope handles form a hierarchical tree via `parent_uuid`. Every scope stack
/// starts with a root scope (name `"root"`, type [`ScopeType::Agent`]) that
/// cannot be removed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeHandle {
    /// Unique identifier for this scope, generated as a v4 UUID on creation.
    pub uuid: Uuid,
    /// The kind of component that owns this scope.
    pub scope_type: ScopeType,
    /// Human-readable name for this scope.
    pub name: String,
    /// Optional application-specific data attached to this scope.
    pub data: Option<Json>,
    /// Optional metadata (e.g., tracing info) attached to this scope.
    pub metadata: Option<Json>,
    /// Behavioral attribute flags for this scope.
    pub attributes: ScopeAttributes,
    /// UUID of the parent scope, or `None` for the root scope.
    pub parent_uuid: Option<Uuid>,
}

impl ScopeHandle {
    /// Creates a new scope handle with a fresh v4 UUID.
    pub fn new(
        name: String,
        scope_type: ScopeType,
        attributes: ScopeAttributes,
        parent_uuid: Option<Uuid>,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            scope_type,
            name,
            data,
            metadata,
            attributes,
            parent_uuid,
        }
    }
}

/// A handle representing an active tool invocation.
///
/// Created by [`nat_nexus_tool_call`](crate::api::nat_nexus_tool_call) and
/// ended by [`nat_nexus_tool_call_end`](crate::api::nat_nexus_tool_call_end).
/// Each handle gets a unique v4 UUID and emits `Start`/`End` lifecycle events
/// to all registered subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolHandle {
    /// Unique identifier for this tool invocation.
    pub uuid: Uuid,
    /// The tool name (e.g., `"web_search"`, `"calculator"`).
    pub name: String,
    /// Optional application-specific data (e.g., sanitized arguments).
    pub data: Option<Json>,
    /// Optional metadata (e.g., tracing info).
    pub metadata: Option<Json>,
    /// Behavioral attribute flags for this tool call.
    pub attributes: ToolAttributes,
    /// UUID of the parent scope or handle.
    pub parent_uuid: Option<Uuid>,
    /// External correlation ID for tool calls (e.g., from LLM tool_call responses).
    pub tool_call_id: Option<String>,
}

impl ToolHandle {
    /// Creates a new tool handle with a fresh v4 UUID.
    pub fn new(
        name: String,
        attributes: ToolAttributes,
        parent_uuid: Option<Uuid>,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            name,
            data,
            metadata,
            attributes,
            parent_uuid,
            tool_call_id: None,
        }
    }
}

/// A handle representing an active LLM call.
///
/// Created by [`nat_nexus_llm_call`](crate::api::nat_nexus_llm_call) and
/// ended by [`nat_nexus_llm_call_end`](crate::api::nat_nexus_llm_call_end).
/// For streaming calls, the [`LlmStreamWrapper`](crate::stream::LlmStreamWrapper)
/// automatically emits the `End` event when the stream is exhausted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMHandle {
    /// Unique identifier for this LLM call.
    pub uuid: Uuid,
    /// The LLM provider or model name.
    pub name: String,
    /// Optional application-specific data (e.g., sanitized request).
    pub data: Option<Json>,
    /// Optional metadata (e.g., tracing info).
    pub metadata: Option<Json>,
    /// Behavioral attribute flags for this LLM call.
    pub attributes: LLMAttributes,
    /// UUID of the parent scope or handle.
    pub parent_uuid: Option<Uuid>,
    /// LLM model identifier (e.g., `"gpt-4"`, `"claude-3-opus"`).
    pub model_name: Option<String>,
}

impl LLMHandle {
    /// Creates a new LLM handle with a fresh v4 UUID.
    pub fn new(
        name: String,
        attributes: LLMAttributes,
        parent_uuid: Option<Uuid>,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            name,
            data,
            metadata,
            attributes,
            parent_uuid,
            model_name: None,
        }
    }
}

// ---------------------------------------------------------------------------
// LLMRequest
// ---------------------------------------------------------------------------

/// The canonical request representation that flows through the entire LLM
/// middleware pipeline: intercepts, guardrails, and execution functions.
///
/// The `headers` field carries generic metadata (e.g., HTTP headers, SDK
/// options) and `content` holds the payload in whatever format the LLM SDK
/// uses. Intercepts and guardrails can read and modify both fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRequest {
    /// Metadata key-value pairs (e.g., HTTP headers, SDK options).
    pub headers: serde_json::Map<String, Json>,
    /// The request payload (e.g., messages, parameters).
    pub content: Json,
}

// ---------------------------------------------------------------------------
// Container types for registries
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- ScopeAttributes bitflags --

    #[test]
    fn test_scope_attributes_empty() {
        let attrs = ScopeAttributes::empty();
        assert!(!attrs.contains(ScopeAttributes::PARALLEL));
        assert!(!attrs.contains(ScopeAttributes::RELOCATABLE));
        assert!(attrs.bits() == 0);
    }

    #[test]
    fn test_scope_attributes_individual() {
        assert_eq!(ScopeAttributes::PARALLEL.bits(), 0b01);
        assert_eq!(ScopeAttributes::RELOCATABLE.bits(), 0b10);
    }

    #[test]
    fn test_scope_attributes_combined() {
        let attrs = ScopeAttributes::PARALLEL | ScopeAttributes::RELOCATABLE;
        assert!(attrs.contains(ScopeAttributes::PARALLEL));
        assert!(attrs.contains(ScopeAttributes::RELOCATABLE));
        assert_eq!(attrs.bits(), 0b11);
    }

    #[test]
    fn test_scope_attributes_serde_roundtrip() {
        let attrs = ScopeAttributes::PARALLEL | ScopeAttributes::RELOCATABLE;
        let json = serde_json::to_string(&attrs).unwrap();
        let deserialized: ScopeAttributes = serde_json::from_str(&json).unwrap();
        assert_eq!(attrs, deserialized);
    }

    // -- ToolAttributes bitflags --

    #[test]
    fn test_tool_attributes() {
        assert_eq!(ToolAttributes::LOCAL.bits(), 0b01);
        let empty = ToolAttributes::empty();
        assert!(!empty.contains(ToolAttributes::LOCAL));
    }

    #[test]
    fn test_tool_attributes_serde_roundtrip() {
        let attrs = ToolAttributes::LOCAL;
        let json = serde_json::to_string(&attrs).unwrap();
        let deserialized: ToolAttributes = serde_json::from_str(&json).unwrap();
        assert_eq!(attrs, deserialized);
    }

    // -- LLMAttributes bitflags --

    #[test]
    fn test_llm_attributes_individual() {
        assert_eq!(LLMAttributes::STATELESS.bits(), 0b01);
        assert_eq!(LLMAttributes::STREAMING.bits(), 0b10);
    }

    #[test]
    fn test_llm_attributes_combined() {
        let attrs = LLMAttributes::STATELESS | LLMAttributes::STREAMING;
        assert!(attrs.contains(LLMAttributes::STATELESS));
        assert!(attrs.contains(LLMAttributes::STREAMING));
    }

    #[test]
    fn test_llm_attributes_serde_roundtrip() {
        let attrs = LLMAttributes::STATELESS | LLMAttributes::STREAMING;
        let json = serde_json::to_string(&attrs).unwrap();
        let deserialized: LLMAttributes = serde_json::from_str(&json).unwrap();
        assert_eq!(attrs, deserialized);
    }

    // -- ScopeType --

    #[test]
    fn test_scope_type_all_variants() {
        let variants = vec![
            ScopeType::Agent,
            ScopeType::Function,
            ScopeType::Tool,
            ScopeType::Llm,
            ScopeType::Retriever,
            ScopeType::Embedder,
            ScopeType::Reranker,
            ScopeType::Guardrail,
            ScopeType::Evaluator,
            ScopeType::Custom,
            ScopeType::Unknown,
        ];
        assert_eq!(variants.len(), 11);
    }

    #[test]
    fn test_scope_type_serde_roundtrip() {
        let variants = vec![
            (ScopeType::Agent, "\"agent\""),
            (ScopeType::Function, "\"function\""),
            (ScopeType::Tool, "\"tool\""),
            (ScopeType::Llm, "\"llm\""),
            (ScopeType::Retriever, "\"retriever\""),
            (ScopeType::Embedder, "\"embedder\""),
            (ScopeType::Reranker, "\"reranker\""),
            (ScopeType::Guardrail, "\"guardrail\""),
            (ScopeType::Evaluator, "\"evaluator\""),
            (ScopeType::Custom, "\"custom\""),
            (ScopeType::Unknown, "\"unknown\""),
        ];
        for (variant, expected_json) in variants {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(json, expected_json);
            let deserialized: ScopeType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    // -- HandleAttributes --

    #[test]
    fn test_handle_attributes_variants() {
        let scope = HandleAttributes::Scope(ScopeAttributes::PARALLEL);
        let tool = HandleAttributes::Tool(ToolAttributes::LOCAL);
        let _llm = HandleAttributes::Llm(LLMAttributes::STREAMING);

        // Equality
        assert_eq!(scope, HandleAttributes::Scope(ScopeAttributes::PARALLEL));
        assert_ne!(scope, HandleAttributes::Scope(ScopeAttributes::RELOCATABLE));
        assert_ne!(scope, tool);
    }

    #[test]
    fn test_handle_attributes_serde_roundtrip() {
        let attrs = HandleAttributes::Tool(ToolAttributes::LOCAL);
        let json = serde_json::to_string(&attrs).unwrap();
        let deserialized: HandleAttributes = serde_json::from_str(&json).unwrap();
        assert_eq!(attrs, deserialized);
    }

    // -- ScopeHandle --

    #[test]
    fn test_scope_handle_new() {
        let handle = ScopeHandle::new(
            "my_scope".to_string(),
            ScopeType::Agent,
            ScopeAttributes::PARALLEL,
            None,
            None,
            None,
        );
        assert_eq!(handle.name, "my_scope");
        assert_eq!(handle.scope_type, ScopeType::Agent);
        assert_eq!(handle.attributes, ScopeAttributes::PARALLEL);
        assert!(handle.parent_uuid.is_none());
        assert!(handle.data.is_none());
        assert!(handle.metadata.is_none());
        // UUID should be valid v4
        assert!(!handle.uuid.is_nil());
    }

    #[test]
    fn test_scope_handle_with_parent() {
        let parent_uuid = Uuid::new_v4();
        let handle = ScopeHandle::new(
            "child".to_string(),
            ScopeType::Function,
            ScopeAttributes::empty(),
            Some(parent_uuid),
            None,
            None,
        );
        assert_eq!(handle.parent_uuid, Some(parent_uuid));
    }

    #[test]
    fn test_scope_handle_unique_uuids() {
        let h1 = ScopeHandle::new(
            "a".into(),
            ScopeType::Agent,
            ScopeAttributes::empty(),
            None,
            None,
            None,
        );
        let h2 = ScopeHandle::new(
            "a".into(),
            ScopeType::Agent,
            ScopeAttributes::empty(),
            None,
            None,
            None,
        );
        assert_ne!(h1.uuid, h2.uuid);
    }

    #[test]
    fn test_scope_handle_serde_roundtrip() {
        let handle = ScopeHandle::new(
            "test".to_string(),
            ScopeType::Tool,
            ScopeAttributes::RELOCATABLE,
            Some(Uuid::new_v4()),
            None,
            None,
        );
        let json = serde_json::to_string(&handle).unwrap();
        let deserialized: ScopeHandle = serde_json::from_str(&json).unwrap();
        assert_eq!(handle.uuid, deserialized.uuid);
        assert_eq!(handle.name, deserialized.name);
        assert_eq!(handle.scope_type, deserialized.scope_type);
        assert_eq!(handle.attributes, deserialized.attributes);
        assert_eq!(handle.parent_uuid, deserialized.parent_uuid);
    }

    // -- ToolHandle --

    #[test]
    fn test_tool_handle_new() {
        let parent_uuid = Uuid::new_v4();
        let data = Some(json!({"key": "value"}));
        let metadata = Some(json!({"version": 1}));
        let handle = ToolHandle::new(
            "my_tool".to_string(),
            ToolAttributes::LOCAL,
            Some(parent_uuid),
            data.clone(),
            metadata.clone(),
        );
        assert_eq!(handle.name, "my_tool");
        assert_eq!(handle.attributes, ToolAttributes::LOCAL);
        assert_eq!(handle.parent_uuid, Some(parent_uuid));
        assert_eq!(handle.data, data);
        assert_eq!(handle.metadata, metadata);
        assert!(!handle.uuid.is_nil());
        assert!(handle.tool_call_id.is_none());
    }

    #[test]
    fn test_tool_handle_tool_call_id() {
        let mut handle = ToolHandle::new(
            "my_tool".to_string(),
            ToolAttributes::LOCAL,
            None,
            None,
            None,
        );
        handle.tool_call_id = Some("call_abc123".to_string());
        assert_eq!(handle.tool_call_id, Some("call_abc123".to_string()));
    }

    #[test]
    fn test_tool_handle_serde_roundtrip() {
        let handle = ToolHandle::new(
            "tool".into(),
            ToolAttributes::empty(),
            None,
            Some(json!({"x": 1})),
            None,
        );
        let json_str = serde_json::to_string(&handle).unwrap();
        let deserialized: ToolHandle = serde_json::from_str(&json_str).unwrap();
        assert_eq!(handle.uuid, deserialized.uuid);
        assert_eq!(handle.name, deserialized.name);
    }

    // -- LLMHandle --

    #[test]
    fn test_llm_handle_new() {
        let handle = LLMHandle::new(
            "gpt".to_string(),
            LLMAttributes::STATELESS | LLMAttributes::STREAMING,
            None,
            None,
            Some(json!({"model": "gpt-4"})),
        );
        assert_eq!(handle.name, "gpt");
        assert!(handle.attributes.contains(LLMAttributes::STATELESS));
        assert!(handle.attributes.contains(LLMAttributes::STREAMING));
        assert!(handle.data.is_none());
        assert!(handle.metadata.is_some());
        assert!(handle.model_name.is_none());
    }

    #[test]
    fn test_llm_handle_model_name() {
        let mut handle =
            LLMHandle::new("gpt".to_string(), LLMAttributes::empty(), None, None, None);
        handle.model_name = Some("gpt-4".to_string());
        assert_eq!(handle.model_name, Some("gpt-4".to_string()));
    }

    // -- LLMRequest --

    #[test]
    fn test_llm_request_serde() {
        let mut headers = serde_json::Map::new();
        headers.insert("Authorization".into(), json!("Bearer token"));
        let req = LLMRequest {
            headers,
            content: json!({"messages": []}),
        };
        let json_str = serde_json::to_string(&req).unwrap();
        let deserialized: LLMRequest = serde_json::from_str(&json_str).unwrap();
        assert_eq!(req.headers, deserialized.headers);
        assert_eq!(req.content, deserialized.content);
    }

    // -- Event --

    #[test]
    fn test_event_new() {
        let parent = Uuid::new_v4();
        let uuid = Uuid::new_v4();
        let event = Event::scope_start(
            Some(parent),
            uuid,
            "test_event",
            Some(json!({"key": "val"})),
            None,
            ScopeAttributes::empty(),
            ScopeType::Agent,
        );
        assert_eq!(event.parent_uuid(), Some(parent));
        assert_eq!(event.uuid(), uuid);
        assert_eq!(event.name(), "test_event");
        assert_eq!(event.kind(), "ScopeStart");
        assert_eq!(event.scope_type(), Some(ScopeType::Agent));
        assert!(event.data().is_some());
        assert!(event.metadata().is_none());
    }

    #[test]
    fn test_event_serde_roundtrip() {
        let event = Event::mark(None, Uuid::new_v4(), "evt", None, None);
        let json_str = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&json_str).unwrap();
        assert_eq!(event.uuid(), deserialized.uuid());
        assert_eq!(event.kind(), deserialized.kind());
    }

    #[test]
    fn test_event_timestamp_is_recent() {
        let before = chrono::Utc::now();
        let event = Event::mark(None, Uuid::new_v4(), "", None, None);
        let after = chrono::Utc::now();
        assert!(*event.timestamp() >= before);
        assert!(*event.timestamp() <= after);
    }

    // -- Event new fields --

    #[test]
    fn test_event_new_fields_default_none() {
        let event = Event::mark(None, Uuid::new_v4(), "", None, None);
        assert!(event.input().is_none());
        assert!(event.output().is_none());
        assert!(event.model_name().is_none());
        assert!(event.tool_call_id().is_none());
    }

    #[test]
    fn test_event_serde_roundtrip_with_new_fields() {
        let event = Event::tool_end(
            None,
            Uuid::new_v4(),
            "test",
            None,
            None,
            ToolAttributes::empty(),
            Some(json!({"result": "world"})),
            Some("call_abc".to_string()),
        );

        let json_str = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.output(), Some(&json!({"result": "world"})));
        assert_eq!(deserialized.tool_call_id(), Some("call_abc"));
    }

    #[test]
    fn test_mark_event_defaults() {
        let uuid = Uuid::new_v4();
        let event = Event::mark(None, uuid, "", None, None);
        assert_eq!(event.uuid(), uuid);
        assert_eq!(event.kind(), "Mark");
        assert!(event.parent_uuid().is_none());
        assert_eq!(event.name(), "");
        assert!(event.data().is_none());
        assert!(event.metadata().is_none());
        assert!(event.attributes().is_none());
        assert!(event.scope_type().is_none());
        assert!(event.input().is_none());
        assert!(event.output().is_none());
        assert!(event.model_name().is_none());
        assert!(event.tool_call_id().is_none());
    }

    #[test]
    fn test_tool_start_event_all_fields() {
        let uuid = Uuid::new_v4();
        let parent = Uuid::new_v4();
        let event = Event::tool_start(
            Some(parent),
            uuid,
            "my_tool",
            Some(json!({"custom": true})),
            Some(json!({"trace": "abc"})),
            ToolAttributes::LOCAL,
            Some(json!({"args": [1, 2]})),
            Some("call_xyz".to_string()),
        );

        assert_eq!(event.uuid(), uuid);
        assert_eq!(event.kind(), "ToolStart");
        assert_eq!(event.parent_uuid(), Some(parent));
        assert_eq!(event.name(), "my_tool");
        assert_eq!(event.data(), Some(&json!({"custom": true})));
        assert_eq!(event.metadata(), Some(&json!({"trace": "abc"})));
        assert_eq!(
            event.attributes(),
            Some(HandleAttributes::Tool(ToolAttributes::LOCAL))
        );
        assert!(event.scope_type().is_none());
        assert_eq!(event.input(), Some(&json!({"args": [1, 2]})));
        assert!(event.output().is_none());
        assert!(event.model_name().is_none());
        assert_eq!(event.tool_call_id(), Some("call_xyz"));
    }

    #[test]
    fn test_event_constructor_timestamp_is_recent() {
        let before = chrono::Utc::now();
        let event = Event::mark(None, Uuid::new_v4(), "", None, None);
        let after = chrono::Utc::now();
        assert!(*event.timestamp() >= before);
        assert!(*event.timestamp() <= after);
    }
}

/// A priority-ordered intercept that transforms data flowing through a chain.
///
/// Intercepts are executed in ascending priority order. Each intercept receives
/// the current value, transforms it, and passes it to the next intercept in the
/// chain. If `break_chain` is `true`, no further intercepts in the chain execute.
pub struct Intercept<F> {
    /// Sort priority (lower = earlier). Determines execution order within the chain.
    pub priority: i32,
    /// If `true`, stop the chain after this intercept runs (short-circuit).
    pub break_chain: bool,
    /// The transformation function.
    pub callable: F,
}

/// An execution intercept that participates in a middleware chain.
///
/// Each callable receives a `next` function to invoke the next intercept or the
/// original execution path. If the intercept does not want to apply, it should
/// simply call `next(args)` directly to pass through. Multiple intercepts
/// compose in priority order.
pub struct ExecutionIntercept<F> {
    /// Sort priority (lower = checked first).
    pub priority: i32,
    /// The execution function. Call `next` to continue the chain or skip it to short-circuit.
    pub callable: F,
}

/// A guardrail entry in a priority-ordered registry.
///
/// Guardrails are executed in ascending priority order. They can sanitize data
/// (transform it) or conditionally gate execution (return an error to reject).
pub struct GuardrailEntry<F> {
    /// Sort priority (lower = earlier).
    pub priority: i32,
    /// The guardrail function (sanitizer or conditional check).
    pub guardrail: F,
}

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

/// A scope start event emitted when a scope is pushed onto the scope stack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScopeStartEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: ScopeAttributes,
    pub scope_type: ScopeType,
}

/// A scope end event emitted when a scope is popped from the scope stack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScopeEndEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: ScopeAttributes,
    pub scope_type: ScopeType,
}

/// A tool start event emitted when a tool handle is created.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolStartEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: ToolAttributes,
    pub input: Option<Json>,
    pub tool_call_id: Option<String>,
}

/// A tool end event emitted when a tool handle is completed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolEndEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: ToolAttributes,
    pub output: Option<Json>,
    pub tool_call_id: Option<String>,
}

/// An LLM start event emitted when an LLM handle is created.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LLMStartEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: LLMAttributes,
    pub input: Option<Json>,
    pub model_name: Option<String>,
    /// Structured view of the request, populated when a request codec is active.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub annotated_request: Option<Arc<AnnotatedLLMRequest>>,
}

/// An LLM end event emitted when an LLM handle is completed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LLMEndEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: LLMAttributes,
    pub output: Option<Json>,
    pub model_name: Option<String>,
    /// Structured view of the response, populated when a response codec is active.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub annotated_response: Option<Arc<AnnotatedLLMResponse>>,
}

/// A standalone mark event emitted via [`nat_nexus_event`](crate::api::nat_nexus_event).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
}

/// A lifecycle event emitted to all registered subscribers.
///
/// Events are produced when scopes, tool handles, or LLM handles are created
/// or destroyed, and when explicit marker events are fired via
/// [`nat_nexus_event`](crate::api::nat_nexus_event). Subscribers receive
/// a reference to each event and can use them for logging, tracing, metrics,
/// or other observability tasks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Event {
    ScopeStart(ScopeStartEvent),
    ScopeEnd(ScopeEndEvent),
    ToolStart(ToolStartEvent),
    ToolEnd(ToolEndEvent),
    LLMStart(LLMStartEvent),
    LLMEnd(LLMEndEvent),
    Mark(MarkEvent),
}

impl Event {
    fn now() -> DateTime<Utc> {
        Utc::now()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scope_start(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: ScopeAttributes,
        scope_type: ScopeType,
    ) -> Self {
        Self::ScopeStart(ScopeStartEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            scope_type,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scope_end(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: ScopeAttributes,
        scope_type: ScopeType,
    ) -> Self {
        Self::ScopeEnd(ScopeEndEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            scope_type,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tool_start(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: ToolAttributes,
        input: Option<Json>,
        tool_call_id: Option<String>,
    ) -> Self {
        Self::ToolStart(ToolStartEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            input,
            tool_call_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tool_end(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: ToolAttributes,
        output: Option<Json>,
        tool_call_id: Option<String>,
    ) -> Self {
        Self::ToolEnd(ToolEndEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            output,
            tool_call_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn llm_start(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: LLMAttributes,
        input: Option<Json>,
        model_name: Option<String>,
        annotated_request: Option<Arc<AnnotatedLLMRequest>>,
    ) -> Self {
        Self::LLMStart(LLMStartEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            input,
            model_name,
            annotated_request,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn llm_end(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: LLMAttributes,
        output: Option<Json>,
        model_name: Option<String>,
        annotated_response: Option<Arc<AnnotatedLLMResponse>>,
    ) -> Self {
        Self::LLMEnd(LLMEndEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            output,
            model_name,
            annotated_response,
        })
    }

    pub fn mark(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> Self {
        Self::Mark(MarkEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
        })
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Event::ScopeStart(_) => "ScopeStart",
            Event::ScopeEnd(_) => "ScopeEnd",
            Event::ToolStart(_) => "ToolStart",
            Event::ToolEnd(_) => "ToolEnd",
            Event::LLMStart(_) => "LLMStart",
            Event::LLMEnd(_) => "LLMEnd",
            Event::Mark(_) => "Mark",
        }
    }

    pub fn parent_uuid(&self) -> Option<Uuid> {
        match self {
            Event::ScopeStart(event) => event.parent_uuid,
            Event::ScopeEnd(event) => event.parent_uuid,
            Event::ToolStart(event) => event.parent_uuid,
            Event::ToolEnd(event) => event.parent_uuid,
            Event::LLMStart(event) => event.parent_uuid,
            Event::LLMEnd(event) => event.parent_uuid,
            Event::Mark(event) => event.parent_uuid,
        }
    }

    pub fn uuid(&self) -> Uuid {
        match self {
            Event::ScopeStart(event) => event.uuid,
            Event::ScopeEnd(event) => event.uuid,
            Event::ToolStart(event) => event.uuid,
            Event::ToolEnd(event) => event.uuid,
            Event::LLMStart(event) => event.uuid,
            Event::LLMEnd(event) => event.uuid,
            Event::Mark(event) => event.uuid,
        }
    }

    pub fn timestamp(&self) -> &DateTime<Utc> {
        match self {
            Event::ScopeStart(event) => &event.timestamp,
            Event::ScopeEnd(event) => &event.timestamp,
            Event::ToolStart(event) => &event.timestamp,
            Event::ToolEnd(event) => &event.timestamp,
            Event::LLMStart(event) => &event.timestamp,
            Event::LLMEnd(event) => &event.timestamp,
            Event::Mark(event) => &event.timestamp,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Event::ScopeStart(event) => &event.name,
            Event::ScopeEnd(event) => &event.name,
            Event::ToolStart(event) => &event.name,
            Event::ToolEnd(event) => &event.name,
            Event::LLMStart(event) => &event.name,
            Event::LLMEnd(event) => &event.name,
            Event::Mark(event) => &event.name,
        }
    }

    pub fn data(&self) -> Option<&Json> {
        match self {
            Event::ScopeStart(event) => event.data.as_ref(),
            Event::ScopeEnd(event) => event.data.as_ref(),
            Event::ToolStart(event) => event.data.as_ref(),
            Event::ToolEnd(event) => event.data.as_ref(),
            Event::LLMStart(event) => event.data.as_ref(),
            Event::LLMEnd(event) => event.data.as_ref(),
            Event::Mark(event) => event.data.as_ref(),
        }
    }

    pub fn metadata(&self) -> Option<&Json> {
        match self {
            Event::ScopeStart(event) => event.metadata.as_ref(),
            Event::ScopeEnd(event) => event.metadata.as_ref(),
            Event::ToolStart(event) => event.metadata.as_ref(),
            Event::ToolEnd(event) => event.metadata.as_ref(),
            Event::LLMStart(event) => event.metadata.as_ref(),
            Event::LLMEnd(event) => event.metadata.as_ref(),
            Event::Mark(event) => event.metadata.as_ref(),
        }
    }

    pub fn attributes(&self) -> Option<HandleAttributes> {
        match self {
            Event::ScopeStart(event) => Some(HandleAttributes::Scope(event.attributes)),
            Event::ScopeEnd(event) => Some(HandleAttributes::Scope(event.attributes)),
            Event::ToolStart(event) => Some(HandleAttributes::Tool(event.attributes)),
            Event::ToolEnd(event) => Some(HandleAttributes::Tool(event.attributes)),
            Event::LLMStart(event) => Some(HandleAttributes::Llm(event.attributes)),
            Event::LLMEnd(event) => Some(HandleAttributes::Llm(event.attributes)),
            Event::Mark(_) => None,
        }
    }

    pub fn scope_type(&self) -> Option<ScopeType> {
        match self {
            Event::ScopeStart(event) => Some(event.scope_type),
            Event::ScopeEnd(event) => Some(event.scope_type),
            Event::ToolStart(_)
            | Event::ToolEnd(_)
            | Event::LLMStart(_)
            | Event::LLMEnd(_)
            | Event::Mark(_) => None,
        }
    }

    pub fn input(&self) -> Option<&Json> {
        match self {
            Event::ToolStart(event) => event.input.as_ref(),
            Event::LLMStart(event) => event.input.as_ref(),
            _ => None,
        }
    }

    pub fn output(&self) -> Option<&Json> {
        match self {
            Event::ToolEnd(event) => event.output.as_ref(),
            Event::LLMEnd(event) => event.output.as_ref(),
            _ => None,
        }
    }

    pub fn model_name(&self) -> Option<&str> {
        match self {
            Event::LLMStart(event) => event.model_name.as_deref(),
            Event::LLMEnd(event) => event.model_name.as_deref(),
            _ => None,
        }
    }

    pub fn tool_call_id(&self) -> Option<&str> {
        match self {
            Event::ToolStart(event) => event.tool_call_id.as_deref(),
            Event::ToolEnd(event) => event.tool_call_id.as_deref(),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — annotated event serde
// ---------------------------------------------------------------------------

#[cfg(test)]
mod annotated_event_tests {
    use super::*;
    use std::sync::Arc;

    use crate::codec::{
        AnnotatedLLMRequest, AnnotatedLLMResponse, FinishReason, MessageContent, Usage,
    };

    /// Helper: build a minimal LLMStartEvent with deterministic timestamp.
    fn make_llm_start_event(annotated_request: Option<Arc<AnnotatedLLMRequest>>) -> LLMStartEvent {
        LLMStartEvent {
            parent_uuid: None,
            uuid: Uuid::nil(),
            timestamp: chrono::DateTime::UNIX_EPOCH,
            name: "test-llm".into(),
            data: None,
            metadata: None,
            attributes: LLMAttributes::empty(),
            input: Some(serde_json::json!({"messages": []})),
            model_name: Some("gpt-4".into()),
            annotated_request,
        }
    }

    /// Helper: build a minimal LLMEndEvent with deterministic timestamp.
    fn make_llm_end_event(annotated_response: Option<Arc<AnnotatedLLMResponse>>) -> LLMEndEvent {
        LLMEndEvent {
            parent_uuid: None,
            uuid: Uuid::nil(),
            timestamp: chrono::DateTime::UNIX_EPOCH,
            name: "test-llm".into(),
            data: None,
            metadata: None,
            attributes: LLMAttributes::empty(),
            output: Some(serde_json::json!({"choices": []})),
            model_name: Some("gpt-4".into()),
            annotated_response,
        }
    }

    // -------------------------------------------------------------------
    // LLMStartEvent serde round-trip tests
    // -------------------------------------------------------------------

    #[test]
    fn test_llm_start_event_none_annotated_request_serde_round_trip() {
        let event = make_llm_start_event(None);
        let json_val = serde_json::to_value(&event).unwrap();

        // With skip_serializing_if, None should NOT produce an annotated_request key
        assert!(
            json_val.get("annotated_request").is_none(),
            "annotated_request key should be absent when None"
        );

        let deserialized: LLMStartEvent = serde_json::from_value(json_val).unwrap();
        assert_eq!(event, deserialized);
    }

    #[test]
    fn test_llm_start_event_with_annotated_request_serde_round_trip() {
        let annotated = AnnotatedLLMRequest {
            messages: vec![],
            model: Some("gpt-4".into()),
            params: None,
            tools: None,
            tool_choice: None,
            extra: serde_json::Map::new(),
        };
        let event = make_llm_start_event(Some(Arc::new(annotated)));
        let json_val = serde_json::to_value(&event).unwrap();

        // The annotated_request key should be present with model "gpt-4"
        let ar = json_val
            .get("annotated_request")
            .expect("annotated_request key should be present");
        assert_eq!(ar.get("model").and_then(|v| v.as_str()), Some("gpt-4"));

        let deserialized: LLMStartEvent = serde_json::from_value(json_val).unwrap();
        assert_eq!(event, deserialized);
    }

    // -------------------------------------------------------------------
    // LLMEndEvent serde round-trip tests
    // -------------------------------------------------------------------

    #[test]
    fn test_llm_end_event_none_annotated_response_serde_round_trip() {
        let event = make_llm_end_event(None);
        let json_val = serde_json::to_value(&event).unwrap();

        // With skip_serializing_if, None should NOT produce an annotated_response key
        assert!(
            json_val.get("annotated_response").is_none(),
            "annotated_response key should be absent when None"
        );

        let deserialized: LLMEndEvent = serde_json::from_value(json_val).unwrap();
        assert_eq!(event, deserialized);
    }

    #[test]
    fn test_llm_end_event_with_annotated_response_serde_round_trip() {
        let annotated = AnnotatedLLMResponse {
            id: Some("chatcmpl-123".into()),
            model: Some("gpt-4".into()),
            message: Some(MessageContent::Text("Hello".into())),
            finish_reason: Some(FinishReason::Complete),
            usage: Some(Usage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                total_tokens: Some(30),
                cache_read_tokens: None,
                cache_write_tokens: None,
            }),
            tool_calls: None,
            api_specific: None,
            extra: serde_json::Map::new(),
        };
        let event = make_llm_end_event(Some(Arc::new(annotated)));
        let json_val = serde_json::to_value(&event).unwrap();

        // The annotated_response key should be present
        let ar = json_val
            .get("annotated_response")
            .expect("annotated_response key should be present");
        assert_eq!(ar.get("id").and_then(|v| v.as_str()), Some("chatcmpl-123"));
        assert_eq!(ar.get("model").and_then(|v| v.as_str()), Some("gpt-4"));

        let deserialized: LLMEndEvent = serde_json::from_value(json_val).unwrap();
        assert_eq!(event, deserialized);
    }

    // -------------------------------------------------------------------
    // Arc clone semantics
    // -------------------------------------------------------------------

    #[test]
    fn test_arc_wrapped_event_clone_preserves_equality() {
        let annotated = Arc::new(AnnotatedLLMResponse {
            id: Some("chatcmpl-456".into()),
            model: Some("gpt-4".into()),
            message: Some(MessageContent::Text("World".into())),
            finish_reason: Some(FinishReason::Complete),
            usage: None,
            tool_calls: None,
            api_specific: None,
            extra: serde_json::Map::new(),
        });

        let event = make_llm_end_event(Some(Arc::clone(&annotated)));
        let cloned = event.clone();

        // Value equality
        assert_eq!(event, cloned);

        // Arc sharing: original annotated + event + cloned event = 3 strong refs
        // (annotated local var + event.annotated_response + cloned.annotated_response)
        assert_eq!(Arc::strong_count(&annotated), 3);
    }
}
