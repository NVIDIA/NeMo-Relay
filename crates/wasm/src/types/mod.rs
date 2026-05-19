// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WebAssembly-friendly wrapper types for the NeMo Flow runtime.
//!
//! This module mirrors the Node binding pattern: exported Rust wrapper types
//! use the canonical JS-facing names, while imported core runtime types are
//! aliased as `Core*` internally.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use nemo_flow::api::event::Event;
#[cfg(test)]
use nemo_flow::api::llm::LlmAttributes;
use nemo_flow::api::llm::{LlmHandle as CoreLlmHandle, LlmRequest as CoreLlmRequest};
use nemo_flow::api::runtime::{ScopeStackHandle, create_scope_stack};
#[cfg(test)]
use nemo_flow::api::scope::ScopeAttributes;
use nemo_flow::api::scope::{ScopeHandle as CoreScopeHandle, ScopeType as CoreScopeType};
#[cfg(test)]
use nemo_flow::api::tool::ToolAttributes;
use nemo_flow::api::tool::ToolHandle as CoreToolHandle;
use nemo_flow::codec::request::AnnotatedLlmRequest;
use nemo_flow::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_flow::error::FlowError;
use nemo_flow::json::Json;

// ---------------------------------------------------------------------------
// Enums and constants used by the WebAssembly bindings.
// ---------------------------------------------------------------------------

fn string_to_js(value: &str) -> JsValue {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

/// The type of an execution scope in the agent runtime hierarchy.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScopeType {
    /// Top-level agent or workflow scope.
    Agent,
    /// Generic function or application-step scope.
    Function,
    /// Tool lifecycle scope.
    Tool,
    /// LLM lifecycle scope.
    Llm,
    /// Retrieval scope such as search or lookup.
    Retriever,
    /// Embedding-generation scope.
    Embedder,
    /// Reranking scope.
    Reranker,
    /// Guardrail or validation scope.
    Guardrail,
    /// Evaluation or scoring scope.
    Evaluator,
    /// Caller-defined custom scope.
    Custom,
    /// Fallback value for unknown scope categories.
    Unknown,
}

// Attribute constants

/// Scope attribute flag indicating parallel execution.
pub const SCOPE_PARALLEL: u32 = 0b01;
/// Scope attribute flag indicating the scope may be relocated.
pub const SCOPE_RELOCATABLE: u32 = 0b10;
/// Tool attribute flag indicating remote execution.
pub const TOOL_REMOTE: u32 = 0b01;
/// LLM attribute flag indicating a stateful call.
pub const LLM_STATEFUL: u32 = 0b01;
/// LLM attribute flag indicating a streaming call.
pub const LLM_STREAMING: u32 = 0b10;

impl From<ScopeType> for CoreScopeType {
    fn from(value: ScopeType) -> Self {
        match value {
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
    fn from(value: CoreScopeType) -> Self {
        match value {
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
// Handle wrappers — exposed as wasm_bindgen classes
// ---------------------------------------------------------------------------

/// Handle representing an active scope in the scope stack.
///
/// Provides read-only access to the scope's UUID, name, type, attributes,
/// parent UUID, data, and metadata.
#[wasm_bindgen(js_name = ScopeHandle)]
pub struct ScopeHandle {
    /// The underlying core `ScopeHandle` containing UUID, name, type, and attributes.
    pub(crate) inner: CoreScopeHandle,
}

#[wasm_bindgen(js_class = ScopeHandle)]
impl ScopeHandle {
    /// Returns the unique identifier of this scope as a string.
    #[wasm_bindgen(getter)]
    pub fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    /// Returns the human-readable name of this scope.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Returns the scope type.
    #[wasm_bindgen(getter, js_name = "scopeType")]
    pub fn scope_type(&self) -> ScopeType {
        self.inner.scope_type.into()
    }

    /// Returns the scope attribute bitfield.
    #[wasm_bindgen(getter)]
    pub fn attributes(&self) -> u32 {
        self.inner.attributes.bits()
    }

    /// Returns the UUID of this scope's parent, or `null` if it has no parent.
    #[wasm_bindgen(
        getter,
        js_name = "parentUuid",
        unchecked_return_type = "string | null"
    )]
    pub fn parent_uuid(&self) -> JsValue {
        match self.inner.parent_uuid {
            Some(uuid) => string_to_js(&uuid.to_string()),
            None => JsValue::NULL,
        }
    }

    /// Returns the optional JSON data payload attached to this scope, or `null`.
    #[wasm_bindgen(getter, unchecked_return_type = "Json | null")]
    pub fn data(&self) -> JsValue {
        match &self.inner.data {
            Some(v) => v
                .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
                .unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }

    /// Returns the optional JSON metadata payload attached to this scope, or `null`.
    #[wasm_bindgen(getter, unchecked_return_type = "Json | null")]
    pub fn metadata(&self) -> JsValue {
        match &self.inner.metadata {
            Some(v) => v
                .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
                .unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }
}

impl From<CoreScopeHandle> for ScopeHandle {
    fn from(h: CoreScopeHandle) -> Self {
        Self { inner: h }
    }
}

/// Handle representing an active tool invocation.
///
/// Provides read-only access to the tool's UUID, name, attributes, and parent UUID.
#[wasm_bindgen(js_name = ToolHandle)]
pub struct ToolHandle {
    /// The underlying core `ToolHandle` containing UUID, name, and attributes.
    pub(crate) inner: CoreToolHandle,
}

#[wasm_bindgen(js_class = ToolHandle)]
impl ToolHandle {
    /// Returns the unique identifier of this tool invocation as a string.
    #[wasm_bindgen(getter)]
    pub fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    /// Returns the tool name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Returns the tool attribute bitfield.
    #[wasm_bindgen(getter)]
    pub fn attributes(&self) -> u32 {
        self.inner.attributes.bits()
    }

    /// Returns the UUID of the parent scope, or `null` if there is no parent.
    #[wasm_bindgen(
        getter,
        js_name = "parentUuid",
        unchecked_return_type = "string | null"
    )]
    pub fn parent_uuid(&self) -> JsValue {
        match self.inner.parent_uuid {
            Some(uuid) => string_to_js(&uuid.to_string()),
            None => JsValue::NULL,
        }
    }
}

impl From<CoreToolHandle> for ToolHandle {
    fn from(h: CoreToolHandle) -> Self {
        Self { inner: h }
    }
}

/// Handle representing an active LLM invocation.
///
/// Provides read-only access to the LLM call's UUID, name, attributes, and parent UUID.
#[wasm_bindgen(js_name = LlmHandle)]
pub struct LlmHandle {
    /// The underlying core `LlmHandle` containing UUID, name, and attributes.
    pub(crate) inner: CoreLlmHandle,
}

#[wasm_bindgen(js_class = LlmHandle)]
impl LlmHandle {
    /// Returns the unique identifier of this LLM invocation as a string.
    #[wasm_bindgen(getter)]
    pub fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    /// Returns the LLM provider/model name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Returns the LLM attribute bitfield.
    #[wasm_bindgen(getter)]
    pub fn attributes(&self) -> u32 {
        self.inner.attributes.bits()
    }

    /// Returns the UUID of the parent scope, or `null` if there is no parent.
    #[wasm_bindgen(
        getter,
        js_name = "parentUuid",
        unchecked_return_type = "string | null"
    )]
    pub fn parent_uuid(&self) -> JsValue {
        match self.inner.parent_uuid {
            Some(uuid) => string_to_js(&uuid.to_string()),
            None => JsValue::NULL,
        }
    }
}

impl From<CoreLlmHandle> for LlmHandle {
    fn from(h: CoreLlmHandle) -> Self {
        Self { inner: h }
    }
}

// ---------------------------------------------------------------------------
// Scope stack handle
// ---------------------------------------------------------------------------

/// Handle to an isolated scope stack for per-request/per-task isolation.
///
/// In a WebAssembly environment (browser/Node.js), there is no native async-local
/// storage, so scope stacks are passed explicitly. Create one per logical
/// request and pass it to scope-stack-aware API variants.
#[wasm_bindgen(js_name = ScopeStack)]
pub struct ScopeStack {
    pub(crate) inner: ScopeStackHandle,
}

#[wasm_bindgen(js_class = ScopeStack)]
impl ScopeStack {
    /// Creates a new isolated scope stack with its own root scope.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: create_scope_stack(),
        }
    }
}

impl Default for ScopeStack {
    fn default() -> Self {
        Self::new()
    }
}

impl From<ScopeStackHandle> for ScopeStack {
    fn from(h: ScopeStackHandle) -> Self {
        Self { inner: h }
    }
}

// ---------------------------------------------------------------------------
// LlmRequest
// ---------------------------------------------------------------------------

/// Represents an outbound LLM request with headers and content.
///
/// Construct via `new LlmRequest(headers, content)` from JavaScript.
#[wasm_bindgen(js_name = LlmRequest)]
pub struct LlmRequest {
    /// The underlying core `LlmRequest` containing headers and content.
    pub(crate) inner: CoreLlmRequest,
}

#[wasm_bindgen(js_class = LlmRequest)]
impl LlmRequest {
    /// Creates a new LLM request.
    ///
    /// - `headers` - JSON object of metadata key-value pairs.
    /// - `content` - JSON request payload.
    #[wasm_bindgen(constructor)]
    pub fn new(
        #[wasm_bindgen(unchecked_param_type = "JsonObject | null")] headers: JsValue,
        #[wasm_bindgen(unchecked_param_type = "Json")] content: JsValue,
    ) -> Result<LlmRequest, JsValue> {
        let headers_json: Json = serde_wasm_bindgen::from_value(headers)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let content_json: Json = serde_wasm_bindgen::from_value(content)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let headers_map = match headers_json {
            Json::Object(m) => m,
            _ => serde_json::Map::new(),
        };

        Ok(Self {
            inner: CoreLlmRequest {
                headers: headers_map,
                content: content_json,
            },
        })
    }

    /// Returns the headers as a JSON object.
    #[wasm_bindgen(getter, unchecked_return_type = "JsonObject")]
    pub fn headers(&self) -> JsValue {
        self.inner
            .headers
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .unwrap_or(JsValue::NULL)
    }

    /// Sets the headers from a JSON object.
    #[wasm_bindgen(setter)]
    pub fn set_headers(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "JsonObject")] headers: JsValue,
    ) {
        if let Ok(Json::Object(m)) = serde_wasm_bindgen::from_value::<Json>(headers) {
            self.inner.headers = m;
        }
    }

    /// Returns the request content as a JSON value.
    #[wasm_bindgen(getter, unchecked_return_type = "Json")]
    pub fn content(&self) -> JsValue {
        self.inner
            .content
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .unwrap_or(JsValue::NULL)
    }

    /// Sets the request content from a JSON value.
    #[wasm_bindgen(setter)]
    pub fn set_content(&mut self, #[wasm_bindgen(unchecked_param_type = "Json")] content: JsValue) {
        if let Ok(val) = serde_wasm_bindgen::from_value::<Json>(content) {
            self.inner.content = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Event (serialized to JS object for subscribers)
// ---------------------------------------------------------------------------

/// Serializable ATOF event delivered to subscribers.
#[derive(Serialize)]
#[serde(transparent)]
pub struct WasmEvent(Json);

impl From<&Event> for WasmEvent {
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
/// (decode_response). Construct with `new OpenAIChatCodec()`.
#[wasm_bindgen(js_name = OpenAIChatCodec)]
pub struct OpenAIChatCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn LlmResponseCodec>,
}

impl Default for OpenAIChatCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen(js_class = OpenAIChatCodec)]
impl OpenAIChatCodec {
    /// Create a codec for the OpenAI Chat Completions wire format.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner_codec: std::sync::Arc::new(nemo_flow::codec::openai_chat::OpenAIChatCodec),
            inner_response_codec: std::sync::Arc::new(
                nemo_flow::codec::openai_chat::OpenAIChatCodec,
            ),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[wasm_bindgen(unchecked_return_type = "Json")]
    pub fn decode(
        &self,
        #[wasm_bindgen(unchecked_param_type = "Json")] request: JsValue,
    ) -> Result<JsValue, JsValue> {
        let req_json = crate::convert::js_to_json(&request)?;
        let llm_req: CoreLlmRequest = serde_json::from_value(req_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Encode structured changes back into an opaque LLM request.
    #[wasm_bindgen(unchecked_return_type = "Json")]
    pub fn encode(
        &self,
        #[wasm_bindgen(unchecked_param_type = "Json")] annotated: JsValue,
        #[wasm_bindgen(unchecked_param_type = "Json")] original: JsValue,
    ) -> Result<JsValue, JsValue> {
        let ann_json = crate::convert::js_to_json(&annotated)?;
        let orig_json = crate::convert::js_to_json(&original)?;
        let ann: AnnotatedLlmRequest = serde_json::from_value(ann_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let orig: CoreLlmRequest = serde_json::from_value(orig_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let result = self
            .inner_codec
            .encode(&ann, &orig)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&result)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Decode a raw LLM response into structured form.
    #[wasm_bindgen(js_name = "decodeResponse", unchecked_return_type = "Json")]
    pub fn decode_response(
        &self,
        #[wasm_bindgen(unchecked_param_type = "Json")] response: JsValue,
    ) -> Result<JsValue, JsValue> {
        let resp_json = crate::convert::js_to_json(&response)?;
        let annotated = self
            .inner_response_codec
            .decode_response(&resp_json)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        Ok(crate::convert::json_to_js(&json))
    }
}

/// Built-in codec for the OpenAI Responses API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decode_response). Construct with `new OpenAIResponsesCodec()`.
#[wasm_bindgen(js_name = OpenAIResponsesCodec)]
pub struct OpenAIResponsesCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn LlmResponseCodec>,
}

impl Default for OpenAIResponsesCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen(js_class = OpenAIResponsesCodec)]
impl OpenAIResponsesCodec {
    /// Create a codec for the OpenAI Responses API wire format.
    #[wasm_bindgen(constructor)]
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
    #[wasm_bindgen(unchecked_return_type = "Json")]
    pub fn decode(
        &self,
        #[wasm_bindgen(unchecked_param_type = "Json")] request: JsValue,
    ) -> Result<JsValue, JsValue> {
        let req_json = crate::convert::js_to_json(&request)?;
        let llm_req: CoreLlmRequest = serde_json::from_value(req_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Encode structured changes back into an opaque LLM request.
    #[wasm_bindgen(unchecked_return_type = "Json")]
    pub fn encode(
        &self,
        #[wasm_bindgen(unchecked_param_type = "Json")] annotated: JsValue,
        #[wasm_bindgen(unchecked_param_type = "Json")] original: JsValue,
    ) -> Result<JsValue, JsValue> {
        let ann_json = crate::convert::js_to_json(&annotated)?;
        let orig_json = crate::convert::js_to_json(&original)?;
        let ann: AnnotatedLlmRequest = serde_json::from_value(ann_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let orig: CoreLlmRequest = serde_json::from_value(orig_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let result = self
            .inner_codec
            .encode(&ann, &orig)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&result)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Decode a raw LLM response into structured form.
    #[wasm_bindgen(js_name = "decodeResponse", unchecked_return_type = "Json")]
    pub fn decode_response(
        &self,
        #[wasm_bindgen(unchecked_param_type = "Json")] response: JsValue,
    ) -> Result<JsValue, JsValue> {
        let resp_json = crate::convert::js_to_json(&response)?;
        let annotated = self
            .inner_response_codec
            .decode_response(&resp_json)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        Ok(crate::convert::json_to_js(&json))
    }
}

/// Built-in codec for the Anthropic Messages API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decode_response). Construct with `new AnthropicMessagesCodec()`.
#[wasm_bindgen(js_name = AnthropicMessagesCodec)]
pub struct AnthropicMessagesCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn LlmResponseCodec>,
}

impl Default for AnthropicMessagesCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen(js_class = AnthropicMessagesCodec)]
impl AnthropicMessagesCodec {
    /// Create a codec for the Anthropic Messages API wire format.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner_codec: std::sync::Arc::new(nemo_flow::codec::anthropic::AnthropicMessagesCodec),
            inner_response_codec: std::sync::Arc::new(
                nemo_flow::codec::anthropic::AnthropicMessagesCodec,
            ),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[wasm_bindgen(unchecked_return_type = "Json")]
    pub fn decode(
        &self,
        #[wasm_bindgen(unchecked_param_type = "Json")] request: JsValue,
    ) -> Result<JsValue, JsValue> {
        let req_json = crate::convert::js_to_json(&request)?;
        let llm_req: CoreLlmRequest = serde_json::from_value(req_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Encode structured changes back into an opaque LLM request.
    #[wasm_bindgen(unchecked_return_type = "Json")]
    pub fn encode(
        &self,
        #[wasm_bindgen(unchecked_param_type = "Json")] annotated: JsValue,
        #[wasm_bindgen(unchecked_param_type = "Json")] original: JsValue,
    ) -> Result<JsValue, JsValue> {
        let ann_json = crate::convert::js_to_json(&annotated)?;
        let orig_json = crate::convert::js_to_json(&original)?;
        let ann: AnnotatedLlmRequest = serde_json::from_value(ann_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let orig: CoreLlmRequest = serde_json::from_value(orig_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let result = self
            .inner_codec
            .encode(&ann, &orig)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&result)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Decode a raw LLM response into structured form.
    #[wasm_bindgen(js_name = "decodeResponse", unchecked_return_type = "Json")]
    pub fn decode_response(
        &self,
        #[wasm_bindgen(unchecked_param_type = "Json")] response: JsValue,
    ) -> Result<JsValue, JsValue> {
        let resp_json = crate::convert::js_to_json(&response)?;
        let annotated = self
            .inner_response_codec
            .decode_response(&resp_json)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        Ok(crate::convert::json_to_js(&json))
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/types_tests.rs"]
mod tests;
