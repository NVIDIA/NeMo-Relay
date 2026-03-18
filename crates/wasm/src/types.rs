// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WASM-friendly wrapper types and integer constants for the Nexus runtime.
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

use nvidia_nat_nexus_core::types as core_types;

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
    pub(crate) inner: nvidia_nat_nexus_core::ScopeStackHandle,
}

#[wasm_bindgen]
impl WasmScopeStack {
    /// Creates a new isolated scope stack with its own root scope.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: nvidia_nat_nexus_core::create_scope_stack(),
        }
    }
}

impl Default for WasmScopeStack {
    fn default() -> Self {
        Self::new()
    }
}

impl From<nvidia_nat_nexus_core::ScopeStackHandle> for WasmScopeStack {
    fn from(h: nvidia_nat_nexus_core::ScopeStackHandle) -> Self {
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
pub struct WasmEvent {
    /// UUID of the parent scope, if any.
    pub parent_uuid: Option<String>,
    /// Unique identifier for this event.
    pub uuid: String,
    /// ISO 8601 timestamp of when the event occurred.
    pub timestamp: String,
    /// Optional event name.
    pub name: Option<String>,
    /// Optional JSON data payload.
    pub data: Option<serde_json::Value>,
    /// Optional JSON metadata payload.
    pub metadata: Option<serde_json::Value>,
    /// Event type: 0 = Start, 1 = End, 2 = Mark.
    pub event_type: i32,
    /// Scope type as an integer constant, if associated with a scope.
    pub scope_type: Option<i32>,
    /// Post-guardrail input (tool args, LLM request body) as serialized JSON string.
    pub input: Option<String>,
    /// Post-guardrail output (tool result, LLM response) as serialized JSON string.
    pub output: Option<String>,
    /// LLM model identifier.
    pub model_name: Option<String>,
    /// External correlation ID for tool calls.
    pub tool_call_id: Option<String>,
    /// UUID of the root scope for concurrent agent isolation.
    pub root_uuid: Option<String>,
}

impl From<&core_types::Event> for WasmEvent {
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
            scope_type: e.scope_type.map(scope_type_to_i32),
            input: e
                .input
                .as_ref()
                .map(|v| serde_json::to_string(v).unwrap_or_default()),
            output: e
                .output
                .as_ref()
                .map(|v| serde_json::to_string(v).unwrap_or_default()),
            model_name: e.model_name.clone(),
            tool_call_id: e.tool_call_id.clone(),
            root_uuid: e.root_uuid.map(|u| u.to_string()),
        }
    }
}
