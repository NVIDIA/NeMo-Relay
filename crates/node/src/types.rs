// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Type definitions for the NeMo Flow Node.js NAPI bindings.
//!
//! Contains enums, handle wrappers, request/response structures, event types,
//! and attribute constants that are exposed to JavaScript/TypeScript consumers.
//! Doc comments on `#[napi]` items are emitted into the generated `index.d.ts`.

use napi_derive::napi;
use serde::Serialize;
use serde_json::Value as Json;

use nemo_flow_core::types as core_types;

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

// ---------------------------------------------------------------------------
// Handle wrappers
// ---------------------------------------------------------------------------

/// Handle to an isolated scope stack for per-request/per-task isolation.
#[napi]
pub struct JsScopeStack {
    pub(crate) inner: nemo_flow_core::ScopeStackHandle,
}

#[napi]
impl JsScopeStack {
    /// Creates a new isolated scope stack with its own root scope.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: nemo_flow_core::create_scope_stack(),
        }
    }
}

impl From<nemo_flow_core::ScopeStackHandle> for JsScopeStack {
    fn from(h: nemo_flow_core::ScopeStackHandle) -> Self {
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
    /// Metadata key-value pairs (e.g., HTTP headers, SDK options).
    pub headers: serde_json::Value,
    /// The request payload (e.g., messages, parameters).
    pub content: serde_json::Value,
}

/// An LLM request, encapsulating headers and content.
///
/// Construct via `new JsLLMRequest({ headers, content })`.
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
                headers,
                content: init.content,
            },
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

/// A read-only lifecycle event delivered to subscribers as a discriminated union.
#[derive(Serialize)]
#[serde(tag = "kind")]
pub enum JsEvent {
    ScopeStart {
        parent_uuid: Option<String>,
        uuid: String,
        timestamp: String,
        name: String,
        data: Option<serde_json::Value>,
        metadata: Option<serde_json::Value>,
        attributes: u32,
        scope_type: i32,
    },
    ScopeEnd {
        parent_uuid: Option<String>,
        uuid: String,
        timestamp: String,
        name: String,
        data: Option<serde_json::Value>,
        metadata: Option<serde_json::Value>,
        attributes: u32,
        scope_type: i32,
    },
    ToolStart {
        parent_uuid: Option<String>,
        uuid: String,
        timestamp: String,
        name: String,
        data: Option<serde_json::Value>,
        metadata: Option<serde_json::Value>,
        attributes: u32,
        input: Option<serde_json::Value>,
        tool_call_id: Option<String>,
    },
    ToolEnd {
        parent_uuid: Option<String>,
        uuid: String,
        timestamp: String,
        name: String,
        data: Option<serde_json::Value>,
        metadata: Option<serde_json::Value>,
        attributes: u32,
        output: Option<serde_json::Value>,
        tool_call_id: Option<String>,
    },
    LLMStart {
        parent_uuid: Option<String>,
        uuid: String,
        timestamp: String,
        name: String,
        data: Option<serde_json::Value>,
        metadata: Option<serde_json::Value>,
        attributes: u32,
        input: Option<serde_json::Value>,
        model_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        annotated_request: Option<serde_json::Value>,
    },
    LLMEnd {
        parent_uuid: Option<String>,
        uuid: String,
        timestamp: String,
        name: String,
        data: Option<serde_json::Value>,
        metadata: Option<serde_json::Value>,
        attributes: u32,
        output: Option<serde_json::Value>,
        model_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        annotated_response: Option<serde_json::Value>,
    },
    Mark {
        parent_uuid: Option<String>,
        uuid: String,
        timestamp: String,
        name: String,
        data: Option<serde_json::Value>,
        metadata: Option<serde_json::Value>,
    },
}

impl From<&core_types::Event> for JsEvent {
    fn from(e: &core_types::Event) -> Self {
        match e {
            core_types::Event::ScopeStart(event) => Self::ScopeStart {
                parent_uuid: event.parent_uuid.map(|u| u.to_string()),
                uuid: event.uuid.to_string(),
                timestamp: event.timestamp.to_rfc3339(),
                name: event.name.clone(),
                data: event.data.clone(),
                metadata: event.metadata.clone(),
                attributes: event.attributes.bits(),
                scope_type: ScopeType::from(event.scope_type) as i32,
            },
            core_types::Event::ScopeEnd(event) => Self::ScopeEnd {
                parent_uuid: event.parent_uuid.map(|u| u.to_string()),
                uuid: event.uuid.to_string(),
                timestamp: event.timestamp.to_rfc3339(),
                name: event.name.clone(),
                data: event.data.clone(),
                metadata: event.metadata.clone(),
                attributes: event.attributes.bits(),
                scope_type: ScopeType::from(event.scope_type) as i32,
            },
            core_types::Event::ToolStart(event) => Self::ToolStart {
                parent_uuid: event.parent_uuid.map(|u| u.to_string()),
                uuid: event.uuid.to_string(),
                timestamp: event.timestamp.to_rfc3339(),
                name: event.name.clone(),
                data: event.data.clone(),
                metadata: event.metadata.clone(),
                attributes: event.attributes.bits(),
                input: event.input.clone(),
                tool_call_id: event.tool_call_id.clone(),
            },
            core_types::Event::ToolEnd(event) => Self::ToolEnd {
                parent_uuid: event.parent_uuid.map(|u| u.to_string()),
                uuid: event.uuid.to_string(),
                timestamp: event.timestamp.to_rfc3339(),
                name: event.name.clone(),
                data: event.data.clone(),
                metadata: event.metadata.clone(),
                attributes: event.attributes.bits(),
                output: event.output.clone(),
                tool_call_id: event.tool_call_id.clone(),
            },
            core_types::Event::LLMStart(event) => Self::LLMStart {
                parent_uuid: event.parent_uuid.map(|u| u.to_string()),
                uuid: event.uuid.to_string(),
                timestamp: event.timestamp.to_rfc3339(),
                name: event.name.clone(),
                data: event.data.clone(),
                metadata: event.metadata.clone(),
                attributes: event.attributes.bits(),
                input: event.input.clone(),
                model_name: event.model_name.clone(),
                annotated_request: event
                    .annotated_request
                    .as_ref()
                    .and_then(|a| serde_json::to_value(a.as_ref()).ok()),
            },
            core_types::Event::LLMEnd(event) => Self::LLMEnd {
                parent_uuid: event.parent_uuid.map(|u| u.to_string()),
                uuid: event.uuid.to_string(),
                timestamp: event.timestamp.to_rfc3339(),
                name: event.name.clone(),
                data: event.data.clone(),
                metadata: event.metadata.clone(),
                attributes: event.attributes.bits(),
                output: event.output.clone(),
                model_name: event.model_name.clone(),
                annotated_response: event
                    .annotated_response
                    .as_ref()
                    .and_then(|a| serde_json::to_value(a.as_ref()).ok()),
            },
            core_types::Event::Mark(event) => Self::Mark {
                parent_uuid: event.parent_uuid.map(|u| u.to_string()),
                uuid: event.uuid.to_string(),
                timestamp: event.timestamp.to_rfc3339(),
                name: event.name.clone(),
                data: event.data.clone(),
                metadata: event.metadata.clone(),
            },
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

// ---------------------------------------------------------------------------
// Built-in codec classes
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Chat Completions API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decodeResponse). Construct with `new OpenAIChatCodec()`.
#[napi]
pub struct JsOpenAIChatCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn nemo_flow_core::codec::LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn nemo_flow_core::codec::LlmResponseCodec>,
}

#[napi]
impl JsOpenAIChatCodec {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner_codec: std::sync::Arc::new(nemo_flow_core::codec::OpenAIChatCodec),
            inner_response_codec: std::sync::Arc::new(nemo_flow_core::codec::OpenAIChatCodec),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[napi]
    pub fn decode(&self, request: Json) -> napi::Result<Json> {
        let llm_req: nemo_flow_core::types::LLMRequest = serde_json::from_value(request)
            .map_err(|e| napi::Error::from_reason(format!("invalid LLMRequest: {e}")))?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&annotated).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Encode structured changes back into an opaque LLM request.
    #[napi]
    pub fn encode(&self, annotated: Json, original: Json) -> napi::Result<Json> {
        let ann: nemo_flow_core::codec::AnnotatedLLMRequest = serde_json::from_value(annotated)
            .map_err(|e| napi::Error::from_reason(format!("invalid AnnotatedLLMRequest: {e}")))?;
        let orig: nemo_flow_core::types::LLMRequest = serde_json::from_value(original)
            .map_err(|e| napi::Error::from_reason(format!("invalid LLMRequest: {e}")))?;
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
#[napi]
pub struct JsOpenAIResponsesCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn nemo_flow_core::codec::LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn nemo_flow_core::codec::LlmResponseCodec>,
}

#[napi]
impl JsOpenAIResponsesCodec {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner_codec: std::sync::Arc::new(nemo_flow_core::codec::OpenAIResponsesCodec),
            inner_response_codec: std::sync::Arc::new(nemo_flow_core::codec::OpenAIResponsesCodec),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[napi]
    pub fn decode(&self, request: Json) -> napi::Result<Json> {
        let llm_req: nemo_flow_core::types::LLMRequest = serde_json::from_value(request)
            .map_err(|e| napi::Error::from_reason(format!("invalid LLMRequest: {e}")))?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&annotated).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Encode structured changes back into an opaque LLM request.
    #[napi]
    pub fn encode(&self, annotated: Json, original: Json) -> napi::Result<Json> {
        let ann: nemo_flow_core::codec::AnnotatedLLMRequest = serde_json::from_value(annotated)
            .map_err(|e| napi::Error::from_reason(format!("invalid AnnotatedLLMRequest: {e}")))?;
        let orig: nemo_flow_core::types::LLMRequest = serde_json::from_value(original)
            .map_err(|e| napi::Error::from_reason(format!("invalid LLMRequest: {e}")))?;
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
pub struct JsAnthropicMessagesCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn nemo_flow_core::codec::LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn nemo_flow_core::codec::LlmResponseCodec>,
}

#[napi]
impl JsAnthropicMessagesCodec {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner_codec: std::sync::Arc::new(nemo_flow_core::codec::AnthropicMessagesCodec),
            inner_response_codec: std::sync::Arc::new(
                nemo_flow_core::codec::AnthropicMessagesCodec,
            ),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[napi]
    pub fn decode(&self, request: Json) -> napi::Result<Json> {
        let llm_req: nemo_flow_core::types::LLMRequest = serde_json::from_value(request)
            .map_err(|e| napi::Error::from_reason(format!("invalid LLMRequest: {e}")))?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_value(&annotated).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Encode structured changes back into an opaque LLM request.
    #[napi]
    pub fn encode(&self, annotated: Json, original: Json) -> napi::Result<Json> {
        let ann: nemo_flow_core::codec::AnnotatedLLMRequest = serde_json::from_value(annotated)
            .map_err(|e| napi::Error::from_reason(format!("invalid AnnotatedLLMRequest: {e}")))?;
        let orig: nemo_flow_core::types::LLMRequest = serde_json::from_value(original)
            .map_err(|e| napi::Error::from_reason(format!("invalid LLMRequest: {e}")))?;
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

#[cfg(test)]
#[path = "types_coverage_tests.rs"]
mod coverage_tests;
