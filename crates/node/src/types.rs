//! Type definitions for the NVAgentRT Node.js NAPI bindings.
//!
//! Contains enums, handle wrappers, request/response structures, event types,
//! and attribute constants that are exposed to JavaScript/TypeScript consumers.
//! Doc comments on `#[napi]` items are emitted into the generated `index.d.ts`.

use napi_derive::napi;
use serde::Serialize;
use serde_json::Value as Json;

use nvagentrt_core::types as core_types;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// The type of an execution scope in the agent runtime hierarchy.
#[napi]
pub enum ScopeType {
    /// An autonomous agent scope.
    Agent,
    /// A generic function invocation scope.
    Function,
    /// A tool execution scope.
    Tool,
    /// A large language model call scope.
    Llm,
    /// A retriever (vector search / RAG) scope.
    Retriever,
    /// An embedding model scope.
    Embedder,
    /// A reranker model scope.
    Reranker,
    /// A guardrail evaluation scope.
    Guardrail,
    /// An evaluator / scoring scope.
    Evaluator,
    /// A user-defined custom scope type.
    Custom,
    /// An unknown or unclassified scope type.
    Unknown,
}

impl From<ScopeType> for core_types::ScopeType {
    fn from(v: ScopeType) -> Self {
        match v {
            ScopeType::Agent => core_types::ScopeType::Agent,
            ScopeType::Function => core_types::ScopeType::Function,
            ScopeType::Tool => core_types::ScopeType::Tool,
            ScopeType::Llm => core_types::ScopeType::Llm,
            ScopeType::Retriever => core_types::ScopeType::Retriever,
            ScopeType::Embedder => core_types::ScopeType::Embedder,
            ScopeType::Reranker => core_types::ScopeType::Reranker,
            ScopeType::Guardrail => core_types::ScopeType::Guardrail,
            ScopeType::Evaluator => core_types::ScopeType::Evaluator,
            ScopeType::Custom => core_types::ScopeType::Custom,
            ScopeType::Unknown => core_types::ScopeType::Unknown,
        }
    }
}

impl From<core_types::ScopeType> for ScopeType {
    fn from(v: core_types::ScopeType) -> Self {
        match v {
            core_types::ScopeType::Agent => ScopeType::Agent,
            core_types::ScopeType::Function => ScopeType::Function,
            core_types::ScopeType::Tool => ScopeType::Tool,
            core_types::ScopeType::Llm => ScopeType::Llm,
            core_types::ScopeType::Retriever => ScopeType::Retriever,
            core_types::ScopeType::Embedder => ScopeType::Embedder,
            core_types::ScopeType::Reranker => ScopeType::Reranker,
            core_types::ScopeType::Guardrail => ScopeType::Guardrail,
            core_types::ScopeType::Evaluator => ScopeType::Evaluator,
            core_types::ScopeType::Custom => ScopeType::Custom,
            core_types::ScopeType::Unknown => ScopeType::Unknown,
        }
    }
}

/// The type of a lifecycle event emitted by the runtime.
#[napi]
#[allow(dead_code)]
pub enum EventType {
    /// A scope or operation has started.
    Start,
    /// A scope or operation has ended.
    End,
    /// A user-defined mark event within a scope.
    Mark,
}

impl From<core_types::EventType> for EventType {
    fn from(v: core_types::EventType) -> Self {
        match v {
            core_types::EventType::Start => EventType::Start,
            core_types::EventType::End => EventType::End,
            core_types::EventType::Mark => EventType::Mark,
        }
    }
}

// ---------------------------------------------------------------------------
// Handle wrappers
// ---------------------------------------------------------------------------

/// Handle to an isolated scope stack for per-request/per-task isolation.
#[napi]
pub struct JsScopeStack {
    pub(crate) inner: nvagentrt_core::ScopeStackHandle,
}

#[napi]
impl JsScopeStack {
    /// Creates a new isolated scope stack with its own root scope.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: nvagentrt_core::create_scope_stack(),
        }
    }
}

impl From<nvagentrt_core::ScopeStackHandle> for JsScopeStack {
    fn from(h: nvagentrt_core::ScopeStackHandle) -> Self {
        Self { inner: h }
    }
}

/// A handle to an execution scope in the agent runtime.
///
/// Scopes form a hierarchical stack representing the current execution context
/// (e.g., agent -> function -> tool). Use this handle to reference a specific scope
/// when pushing child scopes, emitting events, or making tool/LLM calls.
#[napi]
pub struct JsScopeHandle {
    pub(crate) inner: core_types::ScopeHandle,
}

#[napi]
impl JsScopeHandle {
    /// The unique identifier for this scope.
    #[napi(getter)]
    pub fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    /// The human-readable name of this scope.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// The type of this scope (Agent, Tool, Llm, etc.).
    #[napi(getter)]
    pub fn scope_type(&self) -> ScopeType {
        self.inner.scope_type.into()
    }

    /// Bitfield of scope attributes (e.g., PARALLEL, RELOCATABLE).
    #[napi(getter)]
    pub fn attributes(&self) -> u32 {
        self.inner.attributes.bits()
    }

    /// The UUID of this scope's parent, or `null` if this is the root scope.
    #[napi(getter)]
    pub fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }

    /// Optional user-defined data associated with this scope.
    #[napi(getter)]
    pub fn data(&self) -> Option<serde_json::Value> {
        self.inner.data.clone()
    }

    /// Optional metadata associated with this scope.
    #[napi(getter)]
    pub fn metadata(&self) -> Option<serde_json::Value> {
        self.inner.metadata.clone()
    }
}

impl From<core_types::ScopeHandle> for JsScopeHandle {
    fn from(h: core_types::ScopeHandle) -> Self {
        Self { inner: h }
    }
}

/// A handle representing an in-progress tool call.
///
/// Returned by `toolCall()` and used to signal completion via `toolCallEnd()`.
#[napi]
pub struct JsToolHandle {
    pub(crate) inner: core_types::ToolHandle,
}

#[napi]
impl JsToolHandle {
    /// The unique identifier for this tool call.
    #[napi(getter)]
    pub fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    /// The name of the tool being called.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Bitfield of tool attributes (e.g., LOCAL).
    #[napi(getter)]
    pub fn attributes(&self) -> u32 {
        self.inner.attributes.bits()
    }

    /// The UUID of the parent scope that initiated this tool call, or `null`.
    #[napi(getter)]
    pub fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
}

impl From<core_types::ToolHandle> for JsToolHandle {
    fn from(h: core_types::ToolHandle) -> Self {
        Self { inner: h }
    }
}

/// A handle representing an in-progress LLM call.
///
/// Returned by `llmCall()` and used to signal completion via `llmCallEnd()`.
#[napi]
pub struct JsLLMHandle {
    pub(crate) inner: core_types::LLMHandle,
}

#[napi]
impl JsLLMHandle {
    /// The unique identifier for this LLM call.
    #[napi(getter)]
    pub fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    /// The name of the LLM provider being called.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Bitfield of LLM attributes (e.g., STATELESS, STREAMING).
    #[napi(getter)]
    pub fn attributes(&self) -> u32 {
        self.inner.attributes.bits()
    }

    /// The UUID of the parent scope that initiated this LLM call, or `null`.
    #[napi(getter)]
    pub fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
}

impl From<core_types::LLMHandle> for JsLLMHandle {
    fn from(h: core_types::LLMHandle) -> Self {
        Self { inner: h }
    }
}

// ---------------------------------------------------------------------------
// LLMRequest
// ---------------------------------------------------------------------------

/// Initialization object for constructing a `JsLLMRequest`.
#[napi(object)]
pub struct JsLLMRequestInit {
    /// The HTTP method (e.g., "POST").
    pub method: String,
    /// The endpoint URL for the LLM API.
    pub url: String,
    /// HTTP headers as a JSON object of key-value string pairs.
    pub headers: serde_json::Value,
    /// The request body as a JSON value.
    pub body: serde_json::Value,
}

/// An LLM HTTP request, encapsulating method, URL, headers, and body.
///
/// Construct via `new JsLLMRequest({ method, url, headers, body })`.
#[napi]
pub struct JsLLMRequest {
    pub(crate) inner: core_types::LLMRequest,
}

#[napi]
impl JsLLMRequest {
    /// Create a new LLM request from the provided initialization fields.
    #[napi(constructor)]
    pub fn new(init: JsLLMRequestInit) -> Self {
        let headers = match init.headers {
            Json::Object(m) => m,
            _ => serde_json::Map::new(),
        };
        Self {
            inner: core_types::LLMRequest {
                method: init.method,
                url: init.url,
                headers,
                body: init.body,
            },
        }
    }

    /// The HTTP method for this request.
    #[napi(getter)]
    pub fn method(&self) -> String {
        self.inner.method.clone()
    }

    /// The endpoint URL for this request.
    #[napi(getter)]
    pub fn url(&self) -> String {
        self.inner.url.clone()
    }

    /// The HTTP headers as a JSON object.
    #[napi(getter)]
    pub fn headers(&self) -> serde_json::Value {
        Json::Object(self.inner.headers.clone())
    }

    /// The request body as a JSON value.
    #[napi(getter)]
    pub fn body(&self) -> serde_json::Value {
        self.inner.body.clone()
    }
}

// ---------------------------------------------------------------------------
// SseEvent
// ---------------------------------------------------------------------------

/// A Server-Sent Event (SSE) received during an LLM streaming response.
#[napi(object)]
pub struct JsSseEvent {
    /// The event type field (e.g., "message"), or `null` if absent.
    pub event: Option<String>,
    /// The event data payload.
    pub data: String,
    /// The event ID, or `null` if absent.
    pub id: Option<String>,
    /// The reconnection retry interval in milliseconds, or `null` if absent.
    pub retry: Option<f64>,
}

impl From<core_types::SseEvent> for JsSseEvent {
    fn from(e: core_types::SseEvent) -> Self {
        Self {
            event: e.event,
            data: e.data,
            id: e.id,
            retry: e.retry.map(|r| r as f64),
        }
    }
}

impl From<JsSseEvent> for core_types::SseEvent {
    fn from(e: JsSseEvent) -> Self {
        Self {
            event: e.event,
            data: e.data,
            id: e.id,
            retry: e.retry.map(|r| r as u64),
        }
    }
}

// ---------------------------------------------------------------------------
// Event (read-only, for subscribers)
// ---------------------------------------------------------------------------

/// A read-only lifecycle event delivered to subscribers.
///
/// Represents a point-in-time occurrence in the agent runtime such as scope start/end
/// or a custom mark event.
#[napi(object)]
#[derive(Serialize)]
pub struct JsEvent {
    /// The UUID of the parent scope, or `null` for root-level events.
    pub parent_uuid: Option<String>,
    /// The unique identifier for this event.
    pub uuid: String,
    /// ISO 8601 timestamp of when the event occurred.
    pub timestamp: String,
    /// The name associated with this event, if any.
    pub name: Option<String>,
    /// Optional user-defined data attached to the event.
    pub data: Option<serde_json::Value>,
    /// Optional metadata attached to the event.
    pub metadata: Option<serde_json::Value>,
    /// The event type as an integer: 0 = Start, 1 = End, 2 = Mark.
    pub event_type: i32,
    /// The scope type as an integer (0=Agent, 1=Function, ..., 10=Unknown), or `null` if absent.
    pub scope_type: Option<i32>,
}

impl From<&core_types::Event> for JsEvent {
    fn from(e: &core_types::Event) -> Self {
        Self {
            parent_uuid: e.parent_uuid.map(|u| u.to_string()),
            uuid: e.uuid.to_string(),
            timestamp: e.timestamp.to_rfc3339(),
            name: e.name.clone(),
            data: e.data.clone(),
            metadata: e.metadata.clone(),
            event_type: match e.event_type {
                core_types::EventType::Start => 0,
                core_types::EventType::End => 1,
                core_types::EventType::Mark => 2,
            },
            scope_type: e.scope_type.map(|st| match st {
                core_types::ScopeType::Agent => 0,
                core_types::ScopeType::Function => 1,
                core_types::ScopeType::Tool => 2,
                core_types::ScopeType::Llm => 3,
                core_types::ScopeType::Retriever => 4,
                core_types::ScopeType::Embedder => 5,
                core_types::ScopeType::Reranker => 6,
                core_types::ScopeType::Guardrail => 7,
                core_types::ScopeType::Evaluator => 8,
                core_types::ScopeType::Custom => 9,
                core_types::ScopeType::Unknown => 10,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Attribute constants
// ---------------------------------------------------------------------------

/// Scope attribute flag: the scope supports parallel execution of children.
#[napi]
pub const SCOPE_ATTR_PARALLEL: u32 = core_types::ScopeAttributes::PARALLEL.bits();
/// Scope attribute flag: the scope can be relocated to a different parent.
#[napi]
pub const SCOPE_ATTR_RELOCATABLE: u32 = core_types::ScopeAttributes::RELOCATABLE.bits();
/// Tool attribute flag: the tool executes locally (not via remote API).
#[napi]
pub const TOOL_ATTR_LOCAL: u32 = core_types::ToolAttributes::LOCAL.bits();
/// LLM attribute flag: the LLM call is stateless (no conversation context).
#[napi]
pub const LLM_ATTR_STATELESS: u32 = core_types::LLMAttributes::STATELESS.bits();
/// LLM attribute flag: the LLM call uses streaming responses.
#[napi]
pub const LLM_ATTR_STREAMING: u32 = core_types::LLMAttributes::STREAMING.bits();
