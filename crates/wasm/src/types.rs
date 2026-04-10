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

use nemo_flow::types as core_types;

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
pub fn i32_to_scope_type(v: i32) -> core_types::ScopeType {
    match v {
        0 => core_types::ScopeType::Agent,
        1 => core_types::ScopeType::Function,
        2 => core_types::ScopeType::Tool,
        3 => core_types::ScopeType::Llm,
        4 => core_types::ScopeType::Retriever,
        5 => core_types::ScopeType::Embedder,
        6 => core_types::ScopeType::Reranker,
        7 => core_types::ScopeType::Guardrail,
        8 => core_types::ScopeType::Evaluator,
        9 => core_types::ScopeType::Custom,
        _ => core_types::ScopeType::Unknown,
    }
}

/// Converts a core `ScopeType` enum variant to its integer constant representation.
pub fn scope_type_to_i32(v: core_types::ScopeType) -> i32 {
    match v {
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
    pub(crate) inner: core_types::ScopeHandle,
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

impl From<core_types::ScopeHandle> for WasmScopeHandle {
    fn from(h: core_types::ScopeHandle) -> Self {
        Self { inner: h }
    }
}

/// Handle representing an active tool invocation.
///
/// Provides read-only access to the tool's UUID, name, attributes, and parent UUID.
#[wasm_bindgen]
pub struct WasmToolHandle {
    /// The underlying core `ToolHandle` containing UUID, name, and attributes.
    pub(crate) inner: core_types::ToolHandle,
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

impl From<core_types::ToolHandle> for WasmToolHandle {
    fn from(h: core_types::ToolHandle) -> Self {
        Self { inner: h }
    }
}

/// Handle representing an active LLM invocation.
///
/// Provides read-only access to the LLM call's UUID, name, attributes, and parent UUID.
#[wasm_bindgen]
pub struct WasmLLMHandle {
    /// The underlying core `LLMHandle` containing UUID, name, and attributes.
    pub(crate) inner: core_types::LLMHandle,
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

impl From<core_types::LLMHandle> for WasmLLMHandle {
    fn from(h: core_types::LLMHandle) -> Self {
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
    pub(crate) inner: nemo_flow::ScopeStackHandle,
}

#[wasm_bindgen]
impl WasmScopeStack {
    /// Creates a new isolated scope stack with its own root scope.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: nemo_flow::create_scope_stack(),
        }
    }
}

impl Default for WasmScopeStack {
    fn default() -> Self {
        Self::new()
    }
}

impl From<nemo_flow::ScopeStackHandle> for WasmScopeStack {
    fn from(h: nemo_flow::ScopeStackHandle) -> Self {
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
    pub(crate) inner: core_types::LLMRequest,
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
            inner: core_types::LLMRequest {
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

impl From<&core_types::Event> for WasmEvent {
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
                scope_type: scope_type_to_i32(event.scope_type),
            },
            core_types::Event::ScopeEnd(event) => Self::ScopeEnd {
                parent_uuid: event.parent_uuid.map(|u| u.to_string()),
                uuid: event.uuid.to_string(),
                timestamp: event.timestamp.to_rfc3339(),
                name: event.name.clone(),
                data: event.data.clone(),
                metadata: event.metadata.clone(),
                attributes: event.attributes.bits(),
                scope_type: scope_type_to_i32(event.scope_type),
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
// Built-in codec classes
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Chat Completions API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decode_response). Construct with `new WasmOpenAIChatCodec()`.
#[wasm_bindgen]
pub struct WasmOpenAIChatCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn nemo_flow::codec::LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn nemo_flow::codec::LlmResponseCodec>,
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
            inner_codec: std::sync::Arc::new(nemo_flow::codec::OpenAIChatCodec),
            inner_response_codec: std::sync::Arc::new(nemo_flow::codec::OpenAIChatCodec),
        }
    }

    /// Decode an opaque LLM request into structured form.
    pub fn decode(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req_json = crate::convert::js_to_json(&request)?;
        let llm_req: nemo_flow::types::LLMRequest =
            serde_json::from_value(req_json).map_err(|e| {
                crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
            })?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated).map_err(|e| {
            crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
        })?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Encode structured changes back into an opaque LLM request.
    pub fn encode(&self, annotated: JsValue, original: JsValue) -> Result<JsValue, JsValue> {
        let ann_json = crate::convert::js_to_json(&annotated)?;
        let orig_json = crate::convert::js_to_json(&original)?;
        let ann: nemo_flow::codec::AnnotatedLLMRequest =
            serde_json::from_value(ann_json).map_err(|e| {
                crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
            })?;
        let orig: nemo_flow::types::LLMRequest =
            serde_json::from_value(orig_json).map_err(|e| {
                crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
            })?;
        let result = self
            .inner_codec
            .encode(&ann, &orig)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&result).map_err(|e| {
            crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
        })?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Decode a raw LLM response into structured form.
    pub fn decode_response(&self, response: JsValue) -> Result<JsValue, JsValue> {
        let resp_json = crate::convert::js_to_json(&response)?;
        let annotated = self
            .inner_response_codec
            .decode_response(&resp_json)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated).map_err(|e| {
            crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
        })?;
        Ok(crate::convert::json_to_js(&json))
    }
}

/// Built-in codec for the OpenAI Responses API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decode_response). Construct with `new WasmOpenAIResponsesCodec()`.
#[wasm_bindgen]
pub struct WasmOpenAIResponsesCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn nemo_flow::codec::LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn nemo_flow::codec::LlmResponseCodec>,
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
            inner_codec: std::sync::Arc::new(nemo_flow::codec::OpenAIResponsesCodec),
            inner_response_codec: std::sync::Arc::new(nemo_flow::codec::OpenAIResponsesCodec),
        }
    }

    /// Decode an opaque LLM request into structured form.
    pub fn decode(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req_json = crate::convert::js_to_json(&request)?;
        let llm_req: nemo_flow::types::LLMRequest =
            serde_json::from_value(req_json).map_err(|e| {
                crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
            })?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated).map_err(|e| {
            crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
        })?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Encode structured changes back into an opaque LLM request.
    pub fn encode(&self, annotated: JsValue, original: JsValue) -> Result<JsValue, JsValue> {
        let ann_json = crate::convert::js_to_json(&annotated)?;
        let orig_json = crate::convert::js_to_json(&original)?;
        let ann: nemo_flow::codec::AnnotatedLLMRequest =
            serde_json::from_value(ann_json).map_err(|e| {
                crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
            })?;
        let orig: nemo_flow::types::LLMRequest =
            serde_json::from_value(orig_json).map_err(|e| {
                crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
            })?;
        let result = self
            .inner_codec
            .encode(&ann, &orig)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&result).map_err(|e| {
            crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
        })?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Decode a raw LLM response into structured form.
    pub fn decode_response(&self, response: JsValue) -> Result<JsValue, JsValue> {
        let resp_json = crate::convert::js_to_json(&response)?;
        let annotated = self
            .inner_response_codec
            .decode_response(&resp_json)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated).map_err(|e| {
            crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
        })?;
        Ok(crate::convert::json_to_js(&json))
    }
}

/// Built-in codec for the Anthropic Messages API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decode_response). Construct with `new WasmAnthropicMessagesCodec()`.
#[wasm_bindgen]
pub struct WasmAnthropicMessagesCodec {
    pub(crate) inner_codec: std::sync::Arc<dyn nemo_flow::codec::LlmCodec>,
    pub(crate) inner_response_codec: std::sync::Arc<dyn nemo_flow::codec::LlmResponseCodec>,
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
            inner_codec: std::sync::Arc::new(nemo_flow::codec::AnthropicMessagesCodec),
            inner_response_codec: std::sync::Arc::new(nemo_flow::codec::AnthropicMessagesCodec),
        }
    }

    /// Decode an opaque LLM request into structured form.
    pub fn decode(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req_json = crate::convert::js_to_json(&request)?;
        let llm_req: nemo_flow::types::LLMRequest =
            serde_json::from_value(req_json).map_err(|e| {
                crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
            })?;
        let annotated = self
            .inner_codec
            .decode(&llm_req)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated).map_err(|e| {
            crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
        })?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Encode structured changes back into an opaque LLM request.
    pub fn encode(&self, annotated: JsValue, original: JsValue) -> Result<JsValue, JsValue> {
        let ann_json = crate::convert::js_to_json(&annotated)?;
        let orig_json = crate::convert::js_to_json(&original)?;
        let ann: nemo_flow::codec::AnnotatedLLMRequest =
            serde_json::from_value(ann_json).map_err(|e| {
                crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
            })?;
        let orig: nemo_flow::types::LLMRequest =
            serde_json::from_value(orig_json).map_err(|e| {
                crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
            })?;
        let result = self
            .inner_codec
            .encode(&ann, &orig)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&result).map_err(|e| {
            crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
        })?;
        Ok(crate::convert::json_to_js(&json))
    }

    /// Decode a raw LLM response into structured form.
    pub fn decode_response(&self, response: JsValue) -> Result<JsValue, JsValue> {
        let resp_json = crate::convert::js_to_json(&response)?;
        let annotated = self
            .inner_response_codec
            .decode_response(&resp_json)
            .map_err(crate::convert::to_js_err)?;
        let json = serde_json::to_value(&annotated).map_err(|e| {
            crate::convert::to_js_err(nemo_flow::FlowError::Internal(e.to_string()))
        })?;
        Ok(crate::convert::json_to_js(&json))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use uuid::Uuid;

    #[test]
    fn test_scope_type_conversion_round_trip() {
        let pairs = [
            (SCOPE_TYPE_AGENT, core_types::ScopeType::Agent),
            (SCOPE_TYPE_FUNCTION, core_types::ScopeType::Function),
            (SCOPE_TYPE_TOOL, core_types::ScopeType::Tool),
            (SCOPE_TYPE_LLM, core_types::ScopeType::Llm),
            (SCOPE_TYPE_RETRIEVER, core_types::ScopeType::Retriever),
            (SCOPE_TYPE_EMBEDDER, core_types::ScopeType::Embedder),
            (SCOPE_TYPE_RERANKER, core_types::ScopeType::Reranker),
            (SCOPE_TYPE_GUARDRAIL, core_types::ScopeType::Guardrail),
            (SCOPE_TYPE_EVALUATOR, core_types::ScopeType::Evaluator),
            (SCOPE_TYPE_CUSTOM, core_types::ScopeType::Custom),
            (SCOPE_TYPE_UNKNOWN, core_types::ScopeType::Unknown),
        ];

        for (raw, scope_type) in pairs {
            assert_eq!(i32_to_scope_type(raw), scope_type);
            assert_eq!(scope_type_to_i32(scope_type), raw);
        }
        assert_eq!(i32_to_scope_type(999), core_types::ScopeType::Unknown);
    }

    #[test]
    fn test_handle_wrappers_and_scope_stack_default() {
        let parent_uuid = Uuid::new_v4();

        let scope = WasmScopeHandle::from(core_types::ScopeHandle::new(
            "scope".into(),
            core_types::ScopeType::Guardrail,
            core_types::ScopeAttributes::PARALLEL,
            Some(parent_uuid),
            Some(json!({"data": true})),
            Some(json!({"meta": true})),
        ));
        assert_eq!(scope.name(), "scope");
        assert_eq!(scope.scope_type(), SCOPE_TYPE_GUARDRAIL);
        assert_eq!(scope.attributes(), SCOPE_PARALLEL);
        assert_eq!(scope.parent_uuid(), Some(parent_uuid.to_string()));
        assert!(!scope.uuid().is_empty());

        let tool = WasmToolHandle::from(core_types::ToolHandle::new(
            "tool".into(),
            core_types::ToolAttributes::LOCAL,
            Some(parent_uuid),
            None,
            None,
        ));
        assert_eq!(tool.name(), "tool");
        assert_eq!(tool.attributes(), TOOL_LOCAL);
        assert_eq!(tool.parent_uuid(), Some(parent_uuid.to_string()));
        assert!(!tool.uuid().is_empty());

        let llm = WasmLLMHandle::from(core_types::LLMHandle::new(
            "llm".into(),
            core_types::LLMAttributes::STATELESS | core_types::LLMAttributes::STREAMING,
            Some(parent_uuid),
            None,
            None,
        ));
        assert_eq!(llm.name(), "llm");
        assert_eq!(llm.attributes(), LLM_STATELESS | LLM_STREAMING);
        assert_eq!(llm.parent_uuid(), Some(parent_uuid.to_string()));
        assert!(!llm.uuid().is_empty());

        let scope_stack = WasmScopeStack::default();
        assert!(std::sync::Arc::strong_count(&scope_stack.inner) >= 1);
    }

    #[test]
    fn test_wasm_event_conversion_maps_fields() {
        let parent_uuid = Some(Uuid::new_v4());
        let uuid = Uuid::new_v4();
        let event = core_types::Event::mark(
            parent_uuid,
            uuid,
            "wasm-event",
            Some(json!({"data": 1})),
            Some(json!({"meta": 2})),
        );

        let wasm_event = WasmEvent::from(&event);
        match wasm_event {
            WasmEvent::Mark {
                parent_uuid: wasm_parent_uuid,
                uuid: wasm_uuid,
                timestamp,
                name,
                data,
                metadata,
            } => {
                assert_eq!(wasm_parent_uuid, parent_uuid.map(|value| value.to_string()));
                assert_eq!(wasm_uuid, uuid.to_string());
                assert_eq!(name, "wasm-event");
                assert_eq!(data, Some(json!({"data": 1})));
                assert_eq!(metadata, Some(json!({"meta": 2})));
                assert!(!timestamp.is_empty());
            }
            _ => panic!("expected Mark event"),
        }
    }

    #[test]
    fn test_wasm_scope_type_is_only_present_on_scope_events() {
        let scope_event = core_types::Event::scope_end(
            None,
            Uuid::new_v4(),
            "scope-event",
            None,
            None,
            core_types::ScopeAttributes::empty(),
            core_types::ScopeType::Function,
        );
        match WasmEvent::from(&scope_event) {
            WasmEvent::ScopeEnd { scope_type, .. } => assert_eq!(scope_type, SCOPE_TYPE_FUNCTION),
            _ => panic!("expected ScopeEnd event"),
        }

        let tool_event = core_types::Event::tool_start(
            None,
            Uuid::new_v4(),
            "tool-event",
            None,
            None,
            core_types::ToolAttributes::empty(),
            None,
            None,
        );
        match WasmEvent::from(&tool_event) {
            WasmEvent::ToolStart { .. } => {}
            _ => panic!("expected ToolStart event"),
        }

        let llm_event = core_types::Event::llm_start(
            None,
            Uuid::new_v4(),
            "llm-event",
            None,
            None,
            core_types::LLMAttributes::empty(),
            None,
            None,
            None,
        );
        match WasmEvent::from(&llm_event) {
            WasmEvent::LLMStart { .. } => {}
            _ => panic!("expected LLMStart event"),
        }
    }
}
