// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WASM-friendly wrapper types and integer constants for the NeMo Flow runtime.
//!
//! Because `wasm_bindgen` does not support Rust enums with data or bitflags
//! natively, this module re-exports scope types and attribute flags as plain
//! integer constants, and wraps core handle types (`ScopeHandle`, `ToolHandle`,
//! `LLMHandle`) in lightweight `#[wasm_bindgen]` structs with getter accessors.
//!
//! # Scope Type Constants
//!
//! | Constant               | Value | Description       |
//! |------------------------|-------|-------------------|
//! | `SCOPE_TYPE_AGENT`     | 0     | Agent scope       |
//! | `SCOPE_TYPE_FUNCTION`  | 1     | Function scope    |
//! | `SCOPE_TYPE_TOOL`      | 2     | Tool scope        |
//! | `SCOPE_TYPE_LLM`       | 3     | LLM scope         |
//! | `SCOPE_TYPE_RETRIEVER` | 4     | Retriever scope   |
//! | `SCOPE_TYPE_EMBEDDER`  | 5     | Embedder scope    |
//! | `SCOPE_TYPE_RERANKER`  | 6     | Reranker scope    |
//! | `SCOPE_TYPE_GUARDRAIL` | 7     | Guardrail scope   |
//! | `SCOPE_TYPE_EVALUATOR` | 8     | Evaluator scope   |
//! | `SCOPE_TYPE_CUSTOM`    | 9     | Custom scope      |
//! | `SCOPE_TYPE_UNKNOWN`   | 10    | Unknown scope     |
//!
//! # Attribute Flag Constants
//!
//! - `SCOPE_PARALLEL` (0b01) -- scope executes in parallel.
//! - `SCOPE_RELOCATABLE` (0b10) -- scope may be relocated.
//! - `TOOL_LOCAL` (0b01) -- tool executes locally only.
//! - `LLM_STATELESS` (0b01) -- LLM call is stateless.
//! - `LLM_STREAMING` (0b10) -- LLM call uses streaming.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use nemo_flow::codec::request::AnnotatedLLMRequest;
use nemo_flow::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_flow::context::scope_stack::{ScopeStackHandle, create_scope_stack};
use nemo_flow::error::FlowError;
use nemo_flow::types::event::Event;
#[cfg(test)]
use nemo_flow::types::llm::LLMAttributes;
use nemo_flow::types::llm::{LLMHandle, LLMRequest};
#[cfg(test)]
use nemo_flow::types::scope::ScopeAttributes;
use nemo_flow::types::scope::{ScopeHandle, ScopeType};
#[cfg(test)]
use nemo_flow::types::tool::ToolAttributes;
use nemo_flow::types::tool::ToolHandle;

// ---------------------------------------------------------------------------
// Enums (exposed as plain constants -- wasm_bindgen doesn't support const defs)
// JS consumers use the integer values directly (e.g. ScopeType.AGENT = 0).
// ---------------------------------------------------------------------------

/// Scope type constant for an agent scope.
pub const SCOPE_TYPE_AGENT: i32 = 0;
/// Scope type constant for a function scope.
pub const SCOPE_TYPE_FUNCTION: i32 = 1;
/// Scope type constant for a tool scope.
pub const SCOPE_TYPE_TOOL: i32 = 2;
/// Scope type constant for an LLM scope.
pub const SCOPE_TYPE_LLM: i32 = 3;
/// Scope type constant for a retriever scope.
pub const SCOPE_TYPE_RETRIEVER: i32 = 4;
/// Scope type constant for an embedder scope.
pub const SCOPE_TYPE_EMBEDDER: i32 = 5;
/// Scope type constant for a reranker scope.
pub const SCOPE_TYPE_RERANKER: i32 = 6;
/// Scope type constant for a guardrail scope.
pub const SCOPE_TYPE_GUARDRAIL: i32 = 7;
/// Scope type constant for an evaluator scope.
pub const SCOPE_TYPE_EVALUATOR: i32 = 8;
/// Scope type constant for a custom scope.
pub const SCOPE_TYPE_CUSTOM: i32 = 9;
/// Scope type constant for an unknown scope type.
pub const SCOPE_TYPE_UNKNOWN: i32 = 10;

// Attribute constants

/// Scope attribute flag indicating parallel execution.
pub const SCOPE_PARALLEL: u32 = 0b01;
/// Scope attribute flag indicating the scope may be relocated.
pub const SCOPE_RELOCATABLE: u32 = 0b10;
/// Tool attribute flag indicating local-only execution.
pub const TOOL_LOCAL: u32 = 0b01;
/// LLM attribute flag indicating a stateless call.
pub const LLM_STATELESS: u32 = 0b01;
/// LLM attribute flag indicating a streaming call.
pub const LLM_STREAMING: u32 = 0b10;

/// Converts an integer constant to the corresponding core `ScopeType` enum variant.
pub fn i32_to_scope_type(v: i32) -> ScopeType {
    match v {
        0 => ScopeType::Agent,
        1 => ScopeType::Function,
        2 => ScopeType::Tool,
        3 => ScopeType::Llm,
        4 => ScopeType::Retriever,
        5 => ScopeType::Embedder,
        6 => ScopeType::Reranker,
        7 => ScopeType::Guardrail,
        8 => ScopeType::Evaluator,
        9 => ScopeType::Custom,
        _ => ScopeType::Unknown,
    }
}

/// Converts a core `ScopeType` enum variant to its integer constant representation.
pub fn scope_type_to_i32(v: ScopeType) -> i32 {
    match v {
        ScopeType::Agent => 0,
        ScopeType::Function => 1,
        ScopeType::Tool => 2,
        ScopeType::Llm => 3,
        ScopeType::Retriever => 4,
        ScopeType::Embedder => 5,
        ScopeType::Reranker => 6,
        ScopeType::Guardrail => 7,
        ScopeType::Evaluator => 8,
        ScopeType::Custom => 9,
        ScopeType::Unknown => 10,
    }
}

// ---------------------------------------------------------------------------
// Handle wrappers — exposed as wasm_bindgen classes
// ---------------------------------------------------------------------------

/// Handle representing an active scope in the scope stack.
///
/// Provides read-only access to the scope's UUID, name, type, attributes,
/// parent UUID, data, and metadata.
#[wasm_bindgen]
pub struct WasmScopeHandle {
    /// The underlying core `ScopeHandle` containing UUID, name, type, and attributes.
    pub(crate) inner: ScopeHandle,
}

#[wasm_bindgen]
impl WasmScopeHandle {
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

    /// Returns the scope type as an integer constant (see `SCOPE_TYPE_*`).
    #[wasm_bindgen(getter, js_name = "scopeType")]
    pub fn scope_type(&self) -> i32 {
        scope_type_to_i32(self.inner.scope_type)
    }

    /// Returns the scope attribute bitfield.
    #[wasm_bindgen(getter)]
    pub fn attributes(&self) -> u32 {
        self.inner.attributes.bits()
    }

    /// Returns the UUID of this scope's parent, or `undefined` if it has no parent.
    #[wasm_bindgen(getter, js_name = "parentUuid")]
    pub fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }

    /// Returns the optional JSON data payload attached to this scope, or `null`.
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> JsValue {
        match &self.inner.data {
            Some(v) => v
                .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
                .unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }

    /// Returns the optional JSON metadata payload attached to this scope, or `null`.
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> JsValue {
        match &self.inner.metadata {
            Some(v) => v
                .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
                .unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }
}

impl From<ScopeHandle> for WasmScopeHandle {
    fn from(h: ScopeHandle) -> Self {
        Self { inner: h }
    }
}

/// Handle representing an active tool invocation.
///
/// Provides read-only access to the tool's UUID, name, attributes, and parent UUID.
#[wasm_bindgen]
pub struct WasmToolHandle {
    /// The underlying core `ToolHandle` containing UUID, name, and attributes.
    pub(crate) inner: ToolHandle,
}

#[wasm_bindgen]
impl WasmToolHandle {
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

    /// Returns the UUID of the parent scope, or `undefined` if there is no parent.
    #[wasm_bindgen(getter, js_name = "parentUuid")]
    pub fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
}

impl From<ToolHandle> for WasmToolHandle {
    fn from(h: ToolHandle) -> Self {
        Self { inner: h }
    }
}

/// Handle representing an active LLM invocation.
///
/// Provides read-only access to the LLM call's UUID, name, attributes, and parent UUID.
#[wasm_bindgen]
pub struct WasmLLMHandle {
    /// The underlying core `LLMHandle` containing UUID, name, and attributes.
    pub(crate) inner: LLMHandle,
}

#[wasm_bindgen]
impl WasmLLMHandle {
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

    /// Returns the UUID of the parent scope, or `undefined` if there is no parent.
    #[wasm_bindgen(getter, js_name = "parentUuid")]
    pub fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
}

impl From<LLMHandle> for WasmLLMHandle {
    fn from(h: LLMHandle) -> Self {
        Self { inner: h }
    }
}

// ---------------------------------------------------------------------------
// Scope stack handle
// ---------------------------------------------------------------------------

/// Handle to an isolated scope stack for per-request/per-task isolation.
///
/// In a WASM environment (browser/Node.js), there is no native async-local
/// storage, so scope stacks are passed explicitly. Create one per logical
/// request and pass it to scope-stack-aware API variants.
#[wasm_bindgen]
pub struct WasmScopeStack {
    pub(crate) inner: ScopeStackHandle,
}

#[wasm_bindgen]
impl WasmScopeStack {
    /// Creates a new isolated scope stack with its own root scope.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: create_scope_stack(),
        }
    }
}

impl Default for WasmScopeStack {
    fn default() -> Self {
        Self::new()
    }
}

impl From<ScopeStackHandle> for WasmScopeStack {
    fn from(h: ScopeStackHandle) -> Self {
        Self { inner: h }
    }
}

// ---------------------------------------------------------------------------
// LLMRequest
// ---------------------------------------------------------------------------

/// Represents an outbound LLM request with headers and content.
///
/// Construct via `new WasmLLMRequest(headers, content)` from JavaScript.
#[wasm_bindgen]
pub struct WasmLLMRequest {
    /// The underlying core `LLMRequest` containing headers and content.
    pub(crate) inner: LLMRequest,
}

#[wasm_bindgen]
impl WasmLLMRequest {
    /// Creates a new LLM request.
    ///
    /// - `headers` - JSON object of metadata key-value pairs.
    /// - `content` - JSON request payload.
    #[wasm_bindgen(constructor)]
    pub fn new(headers: JsValue, content: JsValue) -> Result<WasmLLMRequest, JsValue> {
        let headers_json: serde_json::Value = serde_wasm_bindgen::from_value(headers)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let content_json: serde_json::Value = serde_wasm_bindgen::from_value(content)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let headers_map = match headers_json {
            serde_json::Value::Object(m) => m,
            _ => serde_json::Map::new(),
        };

        Ok(Self {
            inner: LLMRequest {
                headers: headers_map,
                content: content_json,
            },
        })
    }

    /// Returns the headers as a JSON object.
    #[wasm_bindgen(getter)]
    pub fn headers(&self) -> JsValue {
        self.inner
            .headers
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .unwrap_or(JsValue::NULL)
    }

    /// Sets the headers from a JSON object.
    #[wasm_bindgen(setter)]
    pub fn set_headers(&mut self, headers: JsValue) {
        if let Ok(serde_json::Value::Object(m)) =
            serde_wasm_bindgen::from_value::<serde_json::Value>(headers)
        {
            self.inner.headers = m;
        }
    }

    /// Returns the request content as a JSON value.
    #[wasm_bindgen(getter)]
    pub fn content(&self) -> JsValue {
        self.inner
            .content
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .unwrap_or(JsValue::NULL)
    }

    /// Sets the request content from a JSON value.
    #[wasm_bindgen(setter)]
    pub fn set_content(&mut self, content: JsValue) {
        if let Ok(val) = serde_wasm_bindgen::from_value::<serde_json::Value>(content) {
            self.inner.content = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Event (serialized to JS object for subscribers)
// ---------------------------------------------------------------------------

/// Serializable representation of a lifecycle event delivered to subscribers.
///
/// Converted from core `Event` and serialized to a plain JS object via `serde_wasm_bindgen`.
#[derive(Serialize)]
#[serde(tag = "kind")]
pub enum WasmEvent {
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

impl From<&Event> for WasmEvent {
    fn from(e: &Event) -> Self {
        match e {
            Event::ScopeStart(event) => Self::ScopeStart {
                parent_uuid: event.parent_uuid.map(|u| u.to_string()),
                uuid: event.uuid.to_string(),
                timestamp: event.timestamp.to_rfc3339(),
                name: event.name.clone(),
                data: event.data.clone(),
                metadata: event.metadata.clone(),
                attributes: event.attributes.bits(),
                scope_type: scope_type_to_i32(event.scope_type),
            },
            Event::ScopeEnd(event) => Self::ScopeEnd {
                parent_uuid: event.parent_uuid.map(|u| u.to_string()),
                uuid: event.uuid.to_string(),
                timestamp: event.timestamp.to_rfc3339(),
                name: event.name.clone(),
                data: event.data.clone(),
                metadata: event.metadata.clone(),
                attributes: event.attributes.bits(),
                scope_type: scope_type_to_i32(event.scope_type),
            },
            Event::ToolStart(event) => Self::ToolStart {
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
            Event::ToolEnd(event) => Self::ToolEnd {
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
            Event::LLMStart(event) => Self::LLMStart {
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
            Event::LLMEnd(event) => Self::LLMEnd {
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
            Event::Mark(event) => Self::Mark {
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
// Built-in codec classes
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Chat Completions API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decode_response). Construct with `new WasmOpenAIChatCodec()`.
#[wasm_bindgen]
pub struct WasmOpenAIChatCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn LlmResponseCodec>,
}

impl Default for WasmOpenAIChatCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmOpenAIChatCodec {
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
    pub fn decode(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req_json = crate::convert::js_to_json(&request)?;
        let llm_req: LLMRequest = serde_json::from_value(req_json)
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
    pub fn encode(&self, annotated: JsValue, original: JsValue) -> Result<JsValue, JsValue> {
        let ann_json = crate::convert::js_to_json(&annotated)?;
        let orig_json = crate::convert::js_to_json(&original)?;
        let ann: AnnotatedLLMRequest = serde_json::from_value(ann_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let orig: LLMRequest = serde_json::from_value(orig_json)
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
    pub fn decode_response(&self, response: JsValue) -> Result<JsValue, JsValue> {
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
/// (decode_response). Construct with `new WasmOpenAIResponsesCodec()`.
#[wasm_bindgen]
pub struct WasmOpenAIResponsesCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn LlmResponseCodec>,
}

impl Default for WasmOpenAIResponsesCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmOpenAIResponsesCodec {
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
    pub fn decode(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req_json = crate::convert::js_to_json(&request)?;
        let llm_req: LLMRequest = serde_json::from_value(req_json)
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
    pub fn encode(&self, annotated: JsValue, original: JsValue) -> Result<JsValue, JsValue> {
        let ann_json = crate::convert::js_to_json(&annotated)?;
        let orig_json = crate::convert::js_to_json(&original)?;
        let ann: AnnotatedLLMRequest = serde_json::from_value(ann_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let orig: LLMRequest = serde_json::from_value(orig_json)
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
    pub fn decode_response(&self, response: JsValue) -> Result<JsValue, JsValue> {
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
/// (decode_response). Construct with `new WasmAnthropicMessagesCodec()`.
#[wasm_bindgen]
pub struct WasmAnthropicMessagesCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn LlmResponseCodec>,
}

impl Default for WasmAnthropicMessagesCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmAnthropicMessagesCodec {
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
    pub fn decode(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req_json = crate::convert::js_to_json(&request)?;
        let llm_req: LLMRequest = serde_json::from_value(req_json)
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
    pub fn encode(&self, annotated: JsValue, original: JsValue) -> Result<JsValue, JsValue> {
        let ann_json = crate::convert::js_to_json(&annotated)?;
        let orig_json = crate::convert::js_to_json(&original)?;
        let ann: AnnotatedLLMRequest = serde_json::from_value(ann_json)
            .map_err(|e| crate::convert::to_js_err(FlowError::Internal(e.to_string())))?;
        let orig: LLMRequest = serde_json::from_value(orig_json)
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
    pub fn decode_response(&self, response: JsValue) -> Result<JsValue, JsValue> {
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
