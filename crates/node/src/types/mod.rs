// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Type definitions for the NeMo Flow Node.js NAPI bindings.
//!
//! Contains enums, handle wrappers, request/response structures, event types,
//! and attribute constants that are exposed to JavaScript/TypeScript consumers.
//! Doc comments on `#[napi]` items are emitted into the generated `index.d.ts`.

use napi_derive::napi;
use nemo_flow::api::runtime::{ScopeStackHandle, create_scope_stack};
use serde::Serialize;
use serde_json::Value as Json;

use nemo_flow::api::event::Event;
use nemo_flow::api::llm::{LlmHandle as CoreLlmHandle, LlmRequest as CoreLlmRequest};
use nemo_flow::api::scope::{ScopeHandle as CoreScopeHandle, ScopeType as CoreScopeType};
use nemo_flow::api::tool::ToolHandle as CoreToolHandle;
use nemo_flow::codec::request::AnnotatedLlmRequest;
use nemo_flow::codec::traits::{LlmCodec, LlmResponseCodec};

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

impl From<ScopeType> for CoreScopeType {
    fn from(v: ScopeType) -> Self {
        match v {
            ScopeType::Agent => CoreScopeType::Agent,
            ScopeType::Function => CoreScopeType::Function,
            ScopeType::Tool => CoreScopeType::Tool,
            ScopeType::Llm => CoreScopeType::Llm,
            ScopeType::Retriever => CoreScopeType::Retriever,
            ScopeType::Embedder => CoreScopeType::Embedder,
            ScopeType::Reranker => CoreScopeType::Reranker,
            ScopeType::Guardrail => CoreScopeType::Guardrail,
            ScopeType::Evaluator => CoreScopeType::Evaluator,
            ScopeType::Custom => CoreScopeType::Custom,
            ScopeType::Unknown => CoreScopeType::Unknown,
        }
    }
}

impl From<CoreScopeType> for ScopeType {
    fn from(v: CoreScopeType) -> Self {
        match v {
            CoreScopeType::Agent => ScopeType::Agent,
            CoreScopeType::Function => ScopeType::Function,
            CoreScopeType::Tool => ScopeType::Tool,
            CoreScopeType::Llm => ScopeType::Llm,
            CoreScopeType::Retriever => ScopeType::Retriever,
            CoreScopeType::Embedder => ScopeType::Embedder,
            CoreScopeType::Reranker => ScopeType::Reranker,
            CoreScopeType::Guardrail => ScopeType::Guardrail,
            CoreScopeType::Evaluator => ScopeType::Evaluator,
            CoreScopeType::Custom => ScopeType::Custom,
            CoreScopeType::Unknown => ScopeType::Unknown,
        }
    }
}

// ---------------------------------------------------------------------------
// Handle wrappers
// ---------------------------------------------------------------------------

/// Handle to an isolated scope stack for per-request/per-task isolation.
#[napi]
pub struct ScopeStack {
    pub(crate) inner: ScopeStackHandle,
}

#[napi]
impl ScopeStack {
    /// Creates a new isolated scope stack with its own root scope.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: create_scope_stack(),
        }
    }
}

impl From<ScopeStackHandle> for ScopeStack {
    fn from(h: ScopeStackHandle) -> Self {
        Self { inner: h }
    }
}

/// A handle to an execution scope in the agent runtime.
///
/// Scopes form a hierarchical stack representing the current execution context
/// (e.g., agent -> function -> tool). Use this handle to reference a specific scope
/// when pushing child scopes, emitting events, or making tool/LLM calls.
#[napi]
pub struct ScopeHandle {
    pub(crate) inner: CoreScopeHandle,
}

#[napi]
impl ScopeHandle {
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

impl From<CoreScopeHandle> for ScopeHandle {
    fn from(h: CoreScopeHandle) -> Self {
        Self { inner: h }
    }
}

/// A handle representing an in-progress tool call.
///
/// Returned by `toolCall()` and used to signal completion via `toolCallEnd()`.
#[napi]
pub struct ToolHandle {
    pub(crate) inner: CoreToolHandle,
}

#[napi]
impl ToolHandle {
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

impl From<CoreToolHandle> for ToolHandle {
    fn from(h: CoreToolHandle) -> Self {
        Self { inner: h }
    }
}

/// A handle representing an in-progress LLM call.
///
/// Returned by `llmCall()` and used to signal completion via `llmCallEnd()`.
#[napi]
pub struct LlmHandle {
    pub(crate) inner: CoreLlmHandle,
}

#[napi]
impl LlmHandle {
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

impl From<CoreLlmHandle> for LlmHandle {
    fn from(h: CoreLlmHandle) -> Self {
        Self { inner: h }
    }
}

// ---------------------------------------------------------------------------
// LlmRequest
// ---------------------------------------------------------------------------

/// An LLM request, encapsulating headers and content.
///
/// Construct via `new LlmRequest(headers, content)`.
#[napi]
pub struct LlmRequest {
    pub(crate) inner: CoreLlmRequest,
}

#[napi]
impl LlmRequest {
    /// Create a new LLM request from headers and content.
    #[napi(constructor)]
    pub fn new(headers: serde_json::Value, content: serde_json::Value) -> Self {
        let headers = match headers {
            Json::Object(m) => m,
            _ => serde_json::Map::new(),
        };
        Self {
            inner: CoreLlmRequest { headers, content },
        }
    }

    /// The metadata headers as a JSON object.
    #[napi(getter)]
    pub fn headers(&self) -> serde_json::Value {
        Json::Object(self.inner.headers.clone())
    }

    /// The request payload as a JSON value.
    #[napi(getter)]
    pub fn content(&self) -> serde_json::Value {
        self.inner.content.clone()
    }
}

// ---------------------------------------------------------------------------
// Event (read-only, for subscribers)
// ---------------------------------------------------------------------------

/// A read-only ATOF lifecycle event delivered to subscribers.
#[derive(Serialize)]
#[serde(transparent)]
pub struct JsEvent(serde_json::Value);

impl From<&Event> for JsEvent {
    fn from(e: &Event) -> Self {
        Self(e.to_json_value())
    }
}
// ---------------------------------------------------------------------------
// Built-in codec classes
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Chat Completions API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decodeResponse). Construct with `new OpenAIChatCodec()`.
#[napi(js_name = "OpenAIChatCodec")]
pub struct OpenAIChatCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn LlmResponseCodec>,
}

#[napi]
impl OpenAIChatCodec {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner_codec: std::sync::Arc::new(nemo_flow::codec::openai_chat::OpenAIChatCodec),
            inner_response_codec: std::sync::Arc::new(
                nemo_flow::codec::openai_chat::OpenAIChatCodec,
            ),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[napi]
    pub fn decode(&self, request: Json) -> napi::Result<Json> {
        let llm_req: CoreLlmRequest = serde_json::from_value(request)
            .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&annotated).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Encode structured changes back into an opaque LLM request.
    #[napi]
    pub fn encode(&self, annotated: Json, original: Json) -> napi::Result<Json> {
        let ann: AnnotatedLlmRequest = serde_json::from_value(annotated)
            .map_err(|e| napi::Error::from_reason(format!("invalid AnnotatedLlmRequest: {e}")))?;
        let orig: CoreLlmRequest = serde_json::from_value(original)
            .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
        let result = self
            .inner_codec
            .encode(&ann, &orig)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Decode a raw LLM response into structured form.
    #[napi(js_name = "decodeResponse")]
    pub fn decode_response(&self, response: Json) -> napi::Result<Json> {
        let annotated = self
            .inner_response_codec
            .decode_response(&response)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&annotated).map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// Built-in codec for the OpenAI Responses API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decodeResponse). Construct with `new OpenAIResponsesCodec()`.
#[napi(js_name = "OpenAIResponsesCodec")]
pub struct OpenAIResponsesCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn LlmResponseCodec>,
}

#[napi]
impl OpenAIResponsesCodec {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner_codec: std::sync::Arc::new(
                nemo_flow::codec::openai_responses::OpenAIResponsesCodec,
            ),
            inner_response_codec: std::sync::Arc::new(
                nemo_flow::codec::openai_responses::OpenAIResponsesCodec,
            ),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[napi]
    pub fn decode(&self, request: Json) -> napi::Result<Json> {
        let llm_req: CoreLlmRequest = serde_json::from_value(request)
            .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&annotated).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Encode structured changes back into an opaque LLM request.
    #[napi]
    pub fn encode(&self, annotated: Json, original: Json) -> napi::Result<Json> {
        let ann: AnnotatedLlmRequest = serde_json::from_value(annotated)
            .map_err(|e| napi::Error::from_reason(format!("invalid AnnotatedLlmRequest: {e}")))?;
        let orig: CoreLlmRequest = serde_json::from_value(original)
            .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
        let result = self
            .inner_codec
            .encode(&ann, &orig)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Decode a raw LLM response into structured form.
    #[napi(js_name = "decodeResponse")]
    pub fn decode_response(&self, response: Json) -> napi::Result<Json> {
        let annotated = self
            .inner_response_codec
            .decode_response(&response)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&annotated).map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// Built-in codec for the Anthropic Messages API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decodeResponse). Construct with `new AnthropicMessagesCodec()`.
#[napi]
pub struct AnthropicMessagesCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn LlmResponseCodec>,
}

#[napi]
impl AnthropicMessagesCodec {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner_codec: std::sync::Arc::new(nemo_flow::codec::anthropic::AnthropicMessagesCodec),
            inner_response_codec: std::sync::Arc::new(
                nemo_flow::codec::anthropic::AnthropicMessagesCodec,
            ),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[napi]
    pub fn decode(&self, request: Json) -> napi::Result<Json> {
        let llm_req: CoreLlmRequest = serde_json::from_value(request)
            .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&annotated).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Encode structured changes back into an opaque LLM request.
    #[napi]
    pub fn encode(&self, annotated: Json, original: Json) -> napi::Result<Json> {
        let ann: AnnotatedLlmRequest = serde_json::from_value(annotated)
            .map_err(|e| napi::Error::from_reason(format!("invalid AnnotatedLlmRequest: {e}")))?;
        let orig: CoreLlmRequest = serde_json::from_value(original)
            .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
        let result = self
            .inner_codec
            .encode(&ann, &orig)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Decode a raw LLM response into structured form.
    #[napi(js_name = "decodeResponse")]
    pub fn decode_response(&self, response: Json) -> napi::Result<Json> {
        let annotated = self
            .inner_response_codec
            .decode_response(&response)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&annotated).map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}
