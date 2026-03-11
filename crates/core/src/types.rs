// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core data types for the NVMagic runtime.
//!
//! This module defines the fundamental types used throughout the framework:
//!
//! - **Attribute bitflags** — [`ScopeAttributes`], [`ToolAttributes`], [`LLMAttributes`]
//! - **Enums** — [`ScopeType`], [`EventType`]
//! - **Handle types** — [`ScopeHandle`], [`ToolHandle`], [`LLMHandle`], [`HandleAttributes`]
//! - **Request/response types** — [`LLMRequest`]
//! - **Event types** — [`Event`]
//! - **Middleware containers** — [`Intercept`], [`ExecutionIntercept`], [`GuardrailEntry`]

use bitflags::bitflags;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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

/// The type of a lifecycle event.
///
/// Serializes to/from lowercase strings (e.g., `"start"`, `"end"`, `"mark"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventType {
    /// A scope or handle has been created / entered.
    Start,
    /// A scope or handle has been destroyed / exited.
    End,
    /// A standalone marker event (not tied to scope lifecycle).
    Mark,
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
    ///
    /// The `data` and `metadata` fields are initialized to `None`.
    pub fn new(
        name: String,
        scope_type: ScopeType,
        attributes: ScopeAttributes,
        parent_uuid: Option<Uuid>,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            scope_type,
            name,
            data: None,
            metadata: None,
            attributes,
            parent_uuid,
        }
    }
}

/// A handle representing an active tool invocation.
///
/// Created by [`nvmagic_tool_call`](crate::api::nvmagic_tool_call) and
/// ended by [`nvmagic_tool_call_end`](crate::api::nvmagic_tool_call_end).
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
/// Created by [`nvmagic_llm_call`](crate::api::nvmagic_llm_call) and
/// ended by [`nvmagic_llm_call_end`](crate::api::nvmagic_llm_call_end).
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

/// An opaque request structure representing an outgoing LLM API call.
///
/// This is the canonical request representation passed through the LLM
/// guardrail pipeline. The `headers` field carries generic metadata
/// (not necessarily HTTP headers), and `content` holds the payload in
/// whatever format the LLM SDK uses.
///
/// Intercepts operate on the native `Json` representation directly;
/// `LLMRequest` is only used by guardrails for structured access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRequest {
    /// Metadata key-value pairs (e.g., HTTP headers, SDK options).
    pub headers: serde_json::Map<String, Json>,
    /// The request payload (e.g., messages, parameters).
    pub content: Json,
}

/// An opaque response structure representing an LLM API response.
///
/// This is the canonical response representation passed through the LLM
/// response guardrail and intercept pipelines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    /// The response payload.
    pub data: Json,
}

/// Trait for converting between native (opaque `Json`) and formal types.
///
/// Users who need guardrails to access structured fields can implement this
/// trait to derive [`LLMRequest`] and [`LLMResponse`] from their SDK-specific
/// native format. The default [`IdentityConverter`] simply wraps the native
/// value as `content` / `data`.
pub trait LLMConverter: Send + Sync {
    /// Derives a formal [`LLMRequest`] from the native request payload.
    fn to_request(&self, native: &Json) -> LLMRequest;
    /// Derives a formal [`LLMResponse`] from the native response payload.
    fn to_response(&self, native: &Json) -> LLMResponse;
}

/// Default converter that passes native `Json` through as-is.
///
/// - `to_request` → `LLMRequest { headers: {}, content: native.clone() }`
/// - `to_response` → `LLMResponse { data: native.clone() }`
pub struct IdentityConverter;

impl LLMConverter for IdentityConverter {
    fn to_request(&self, native: &Json) -> LLMRequest {
        LLMRequest {
            headers: serde_json::Map::new(),
            content: native.clone(),
        }
    }

    fn to_response(&self, native: &Json) -> LLMResponse {
        LLMResponse {
            data: native.clone(),
        }
    }
}

/// A boxed closure that converts native Json to a formal [`LLMRequest`].
pub type ToRequestFn = Box<dyn Fn(&Json) -> LLMRequest + Send + Sync>;

/// A boxed closure that converts native Json to a formal [`LLMResponse`].
pub type ToResponseFn = Box<dyn Fn(&Json) -> LLMResponse + Send + Sync>;

/// Returns a [`ToRequestFn`] that uses the [`IdentityConverter`].
pub fn identity_to_request() -> ToRequestFn {
    Box::new(|native| IdentityConverter.to_request(native))
}

/// Returns a [`ToResponseFn`] that uses the [`IdentityConverter`].
pub fn identity_to_response() -> ToResponseFn {
    Box::new(|native| IdentityConverter.to_response(native))
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

    // -- EventType --

    #[test]
    fn test_event_type_serde() {
        let variants = vec![
            (EventType::Start, "\"start\""),
            (EventType::End, "\"end\""),
            (EventType::Mark, "\"mark\""),
        ];
        for (variant, expected_json) in variants {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(json, expected_json);
            let deserialized: EventType = serde_json::from_str(&json).unwrap();
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
        );
        assert_eq!(handle.parent_uuid, Some(parent_uuid));
    }

    #[test]
    fn test_scope_handle_unique_uuids() {
        let h1 = ScopeHandle::new("a".into(), ScopeType::Agent, ScopeAttributes::empty(), None);
        let h2 = ScopeHandle::new("a".into(), ScopeType::Agent, ScopeAttributes::empty(), None);
        assert_ne!(h1.uuid, h2.uuid);
    }

    #[test]
    fn test_scope_handle_serde_roundtrip() {
        let handle = ScopeHandle::new(
            "test".to_string(),
            ScopeType::Tool,
            ScopeAttributes::RELOCATABLE,
            Some(Uuid::new_v4()),
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

    #[test]
    fn test_llm_response_serde() {
        let resp = LLMResponse {
            data: json!({"choices": [{"text": "hello"}]}),
        };
        let json_str = serde_json::to_string(&resp).unwrap();
        let deserialized: LLMResponse = serde_json::from_str(&json_str).unwrap();
        assert_eq!(resp.data, deserialized.data);
    }

    #[test]
    fn test_identity_converter() {
        let converter = IdentityConverter;
        let native = json!({"messages": [{"role": "user", "content": "hi"}]});
        let req = converter.to_request(&native);
        assert!(req.headers.is_empty());
        assert_eq!(req.content, native);

        let resp = converter.to_response(&native);
        assert_eq!(resp.data, native);
    }

    // -- Event --

    #[test]
    fn test_event_new() {
        let parent = Uuid::new_v4();
        let uuid = Uuid::new_v4();
        let event = Event::new(
            Some(parent),
            uuid,
            Some("test_event".into()),
            Some(json!({"key": "val"})),
            None,
            Some(HandleAttributes::Scope(ScopeAttributes::empty())),
            EventType::Start,
            Some(ScopeType::Agent),
        );
        assert_eq!(event.parent_uuid, Some(parent));
        assert_eq!(event.uuid, uuid);
        assert_eq!(event.name, Some("test_event".into()));
        assert_eq!(event.event_type, EventType::Start);
        assert_eq!(event.scope_type, Some(ScopeType::Agent));
        assert!(event.data.is_some());
        assert!(event.metadata.is_none());
    }

    #[test]
    fn test_event_serde_roundtrip() {
        let event = Event::new(
            None,
            Uuid::new_v4(),
            Some("evt".into()),
            None,
            None,
            None,
            EventType::Mark,
            None,
        );
        let json_str = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&json_str).unwrap();
        assert_eq!(event.uuid, deserialized.uuid);
        assert_eq!(event.event_type, deserialized.event_type);
    }

    #[test]
    fn test_event_timestamp_is_recent() {
        let before = chrono::Utc::now();
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
        let after = chrono::Utc::now();
        assert!(event.timestamp >= before);
        assert!(event.timestamp <= after);
    }

    // -- Event new fields --

    #[test]
    fn test_event_new_fields_default_none() {
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
        assert!(event.input.is_none());
        assert!(event.output.is_none());
        assert!(event.model_name.is_none());
        assert!(event.tool_call_id.is_none());
        assert!(event.root_uuid.is_none());
    }

    #[test]
    fn test_event_serde_roundtrip_with_new_fields() {
        let root_uuid = Uuid::new_v4();
        let mut event = Event::new(
            None,
            Uuid::new_v4(),
            Some("test".into()),
            None,
            None,
            None,
            EventType::Start,
            Some(ScopeType::Tool),
        );
        event.input = Some(json!({"args": "hello"}));
        event.output = Some(json!({"result": "world"}));
        event.model_name = Some("gpt-4".to_string());
        event.tool_call_id = Some("call_abc".to_string());
        event.root_uuid = Some(root_uuid);

        let json_str = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.input, Some(json!({"args": "hello"})));
        assert_eq!(deserialized.output, Some(json!({"result": "world"})));
        assert_eq!(deserialized.model_name, Some("gpt-4".to_string()));
        assert_eq!(deserialized.tool_call_id, Some("call_abc".to_string()));
        assert_eq!(deserialized.root_uuid, Some(root_uuid));
    }

    // -- EventBuilder --

    #[test]
    fn test_event_builder_defaults() {
        let uuid = Uuid::new_v4();
        let event = Event::builder(uuid, EventType::Mark).build();
        assert_eq!(event.uuid, uuid);
        assert_eq!(event.event_type, EventType::Mark);
        assert!(event.parent_uuid.is_none());
        assert!(event.name.is_none());
        assert!(event.data.is_none());
        assert!(event.metadata.is_none());
        assert!(event.attributes.is_none());
        assert!(event.scope_type.is_none());
        assert!(event.input.is_none());
        assert!(event.output.is_none());
        assert!(event.model_name.is_none());
        assert!(event.tool_call_id.is_none());
        assert!(event.root_uuid.is_none());
    }

    #[test]
    fn test_event_builder_all_setters() {
        let uuid = Uuid::new_v4();
        let parent = Uuid::new_v4();
        let root = Uuid::new_v4();
        let event = Event::builder(uuid, EventType::Start)
            .parent_uuid(Some(parent))
            .name("my_tool")
            .data(Some(json!({"custom": true})))
            .metadata(Some(json!({"trace": "abc"})))
            .attributes(HandleAttributes::Tool(ToolAttributes::LOCAL))
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"args": [1, 2]})))
            .output(Some(json!({"result": 3})))
            .model_name(Some("gpt-4".to_string()))
            .tool_call_id(Some("call_xyz".to_string()))
            .root_uuid(Some(root))
            .build();

        assert_eq!(event.uuid, uuid);
        assert_eq!(event.event_type, EventType::Start);
        assert_eq!(event.parent_uuid, Some(parent));
        assert_eq!(event.name, Some("my_tool".into()));
        assert_eq!(event.data, Some(json!({"custom": true})));
        assert_eq!(event.metadata, Some(json!({"trace": "abc"})));
        assert_eq!(
            event.attributes,
            Some(HandleAttributes::Tool(ToolAttributes::LOCAL))
        );
        assert_eq!(event.scope_type, Some(ScopeType::Tool));
        assert_eq!(event.input, Some(json!({"args": [1, 2]})));
        assert_eq!(event.output, Some(json!({"result": 3})));
        assert_eq!(event.model_name, Some("gpt-4".to_string()));
        assert_eq!(event.tool_call_id, Some("call_xyz".to_string()));
        assert_eq!(event.root_uuid, Some(root));
    }

    #[test]
    fn test_event_builder_timestamp_is_recent() {
        let before = chrono::Utc::now();
        let event = Event::builder(Uuid::new_v4(), EventType::End).build();
        let after = chrono::Utc::now();
        assert!(event.timestamp >= before);
        assert!(event.timestamp <= after);
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
/// The `conditional` function is checked first; if it returns `true`, the
/// `callable` is included in the middleware chain. Each callable receives a
/// `next` function to invoke the next matching intercept or the original
/// execution path. Multiple intercepts compose in priority order.
pub struct ExecutionIntercept<C, F> {
    /// Sort priority (lower = checked first).
    pub priority: i32,
    /// A predicate that determines whether this intercept should handle the call.
    pub conditional: C,
    /// The replacement execution function, invoked if `conditional` returns `true`.
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

/// A lifecycle event emitted to all registered subscribers.
///
/// Events are produced when scopes, tool handles, or LLM handles are created
/// or destroyed, and when explicit marker events are fired via
/// [`nvmagic_event`](crate::api::nvmagic_event). Subscribers receive
/// a reference to each event and can use them for logging, tracing, metrics,
/// or other observability tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// UUID of the parent scope or handle, if any.
    pub parent_uuid: Option<Uuid>,
    /// UUID of the entity that produced this event.
    pub uuid: Uuid,
    /// UTC timestamp of when this event was created.
    pub timestamp: DateTime<Utc>,
    /// Human-readable name of the source entity.
    pub name: Option<String>,
    /// Optional application-specific data snapshot.
    pub data: Option<Json>,
    /// Optional metadata snapshot.
    pub metadata: Option<Json>,
    /// Attribute flags of the source handle, if applicable.
    pub attributes: Option<HandleAttributes>,
    /// Whether this is a start, end, or marker event.
    pub event_type: EventType,
    /// The scope type of the source entity, if applicable.
    pub scope_type: Option<ScopeType>,
    /// Post-guardrail input (tool args, LLM request).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Json>,
    /// Post-guardrail output (tool result, LLM response).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Json>,
    /// LLM model identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// External correlation ID for tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Root scope UUID for concurrent agent isolation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_uuid: Option<Uuid>,
}

impl Event {
    /// Creates a new event with the current UTC timestamp.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: Option<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: Option<HandleAttributes>,
        event_type: EventType,
        scope_type: Option<ScopeType>,
    ) -> Self {
        Self {
            parent_uuid,
            uuid,
            timestamp: Utc::now(),
            name,
            data,
            metadata,
            attributes,
            event_type,
            scope_type,
            input: None,
            output: None,
            model_name: None,
            tool_call_id: None,
            root_uuid: None,
        }
    }

    /// Returns a builder initialized with the required fields.
    pub fn builder(uuid: Uuid, event_type: EventType) -> EventBuilder {
        EventBuilder {
            uuid,
            event_type,
            parent_uuid: None,
            name: None,
            data: None,
            metadata: None,
            attributes: None,
            scope_type: None,
            input: None,
            output: None,
            model_name: None,
            tool_call_id: None,
            root_uuid: None,
        }
    }
}

/// Builder for constructing [`Event`] instances.
///
/// Created via [`Event::builder`]. All fields except `uuid`, `event_type`,
/// and `timestamp` (auto-set to `Utc::now()`) default to `None`.
pub struct EventBuilder {
    uuid: Uuid,
    event_type: EventType,
    parent_uuid: Option<Uuid>,
    name: Option<String>,
    data: Option<Json>,
    metadata: Option<Json>,
    attributes: Option<HandleAttributes>,
    scope_type: Option<ScopeType>,
    input: Option<Json>,
    output: Option<Json>,
    model_name: Option<String>,
    tool_call_id: Option<String>,
    root_uuid: Option<Uuid>,
}

impl EventBuilder {
    /// Sets the parent UUID.
    pub fn parent_uuid(mut self, parent_uuid: Option<Uuid>) -> Self {
        self.parent_uuid = parent_uuid;
        self
    }

    /// Sets the event name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the application-specific data.
    pub fn data(mut self, data: Option<Json>) -> Self {
        self.data = data;
        self
    }

    /// Sets the metadata.
    pub fn metadata(mut self, metadata: Option<Json>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets the handle attributes.
    pub fn attributes(mut self, attributes: HandleAttributes) -> Self {
        self.attributes = Some(attributes);
        self
    }

    /// Sets the scope type.
    pub fn scope_type(mut self, scope_type: ScopeType) -> Self {
        self.scope_type = Some(scope_type);
        self
    }

    /// Sets the post-guardrail input.
    pub fn input(mut self, input: Option<Json>) -> Self {
        self.input = input;
        self
    }

    /// Sets the post-guardrail output.
    pub fn output(mut self, output: Option<Json>) -> Self {
        self.output = output;
        self
    }

    /// Sets the LLM model name.
    pub fn model_name(mut self, model_name: Option<String>) -> Self {
        self.model_name = model_name;
        self
    }

    /// Sets the tool call ID.
    pub fn tool_call_id(mut self, tool_call_id: Option<String>) -> Self {
        self.tool_call_id = tool_call_id;
        self
    }

    /// Sets the root scope UUID.
    pub fn root_uuid(mut self, root_uuid: Option<Uuid>) -> Self {
        self.root_uuid = root_uuid;
        self
    }

    /// Builds the [`Event`] with the current UTC timestamp.
    pub fn build(self) -> Event {
        Event {
            parent_uuid: self.parent_uuid,
            uuid: self.uuid,
            timestamp: Utc::now(),
            name: self.name,
            data: self.data,
            metadata: self.metadata,
            attributes: self.attributes,
            event_type: self.event_type,
            scope_type: self.scope_type,
            input: self.input,
            output: self.output,
            model_name: self.model_name,
            tool_call_id: self.tool_call_id,
            root_uuid: self.root_uuid,
        }
    }
}
