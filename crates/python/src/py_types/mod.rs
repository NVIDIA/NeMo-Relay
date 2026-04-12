// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing type wrappers for NeMo Flow core types.
//!
//! Each type wraps its corresponding `nemo_flow::types` struct and exposes
//! properties via `#[getter]`. Doc comments on `#[pyclass]` and `#[pymethods]`
//! become Python `help()` output.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use nemo_flow::context::scope_stack::ScopeStackHandle;
use nemo_flow::error::Result as FlowResult;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use nemo_flow::codec::request::{
    AnnotatedLLMRequest, GenerationParams, Message, ToolChoice, ToolDefinition,
};
use nemo_flow::codec::response::AnnotatedLLMResponse;
use nemo_flow::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_flow::types::event::{
    LLMEndEvent, LLMStartEvent, MarkEvent, ScopeEndEvent, ScopeStartEvent, ToolEndEvent,
    ToolStartEvent,
};
use nemo_flow::types::llm::{LLMAttributes, LLMHandle, LLMRequest};
use nemo_flow::types::scope::{ScopeAttributes, ScopeHandle, ScopeType as CoreScopeType};
use nemo_flow::types::tool::{ToolAttributes, ToolHandle};

use crate::convert::{json_to_py, opt_json_to_py, py_to_json};

fn py_string_map(obj: &Bound<'_, PyAny>, field_name: &str) -> PyResult<HashMap<String, String>> {
    let json = py_to_json(obj)?;
    let serde_json::Value::Object(map) = json else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{field_name} must be a dict[str, str]"
        )));
    };

    let mut out = HashMap::with_capacity(map.len());
    for (key, value) in map {
        let serde_json::Value::String(value) = value else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{field_name} must be a dict[str, str]"
            )));
        };
        out.insert(key, value);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// LlmStream (async iterator)
// ---------------------------------------------------------------------------

/// An async iterator that yields parsed JSON chunks from a streaming LLM response.
///
/// Use ``async for chunk in stream:`` to consume chunks. Each chunk is a
/// Python object (converted from JSON). The stream automatically emits an
/// End lifecycle event when exhausted.
#[pyclass(name = "LlmStream")]
pub struct PyLlmStream {
    pub receiver: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<FlowResult<serde_json::Value>>>,
}

#[pymethods]
impl PyLlmStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // We need to get a reference to the receiver inside the tokio Mutex.
        // Since PyLlmStream is behind a PyRef (shared), we use tokio::sync::Mutex.
        let receiver_ptr = &self.receiver
            as *const tokio::sync::Mutex<
                tokio::sync::mpsc::Receiver<FlowResult<serde_json::Value>>,
            >;
        // SAFETY: The PyLlmStream outlives this future because Python holds a reference to it.
        // The tokio Mutex ensures exclusive access to the receiver.
        let receiver_ref = unsafe { &*receiver_ptr };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = receiver_ref.lock().await;
            let next_item = guard.recv().await;
            match next_item {
                None => Err(PyErr::new::<pyo3::exceptions::PyStopAsyncIteration, _>(
                    "stream exhausted",
                )),
                Some(Ok(value)) => Python::attach(|py| json_to_py(py, &value)),
                Some(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    e.to_string(),
                )),
            }
        })
    }
}

// ---------------------------------------------------------------------------
// ScopeStack (per-request isolation handle)
// ---------------------------------------------------------------------------

/// An isolated scope stack for per-request/per-task isolation.
///
/// Each ``ScopeStack`` wraps an independent scope stack with its own root
/// scope. Use ``create_scope_stack()`` to obtain one.
#[pyclass(name = "ScopeStack")]
pub struct PyScopeStack(pub ScopeStackHandle);

#[pymethods]
impl PyScopeStack {
    fn __repr__(&self) -> String {
        "<ScopeStack>".to_string()
    }
}

// ---------------------------------------------------------------------------
// ScopeAttributes (bitflag wrapper)
// ---------------------------------------------------------------------------

/// Bitflag attributes for execution scopes.
///
/// Flags can be combined with ``|`` (e.g., ``ScopeAttributes(ScopeAttributes.PARALLEL | ScopeAttributes.RELOCATABLE)``).
///
/// Class attributes:
///     PARALLEL (int): The scope supports parallel child operations.
///     RELOCATABLE (int): The scope can be moved between execution contexts.
///
/// Properties:
///     is_parallel (bool): Whether PARALLEL is set.
///     is_relocatable (bool): Whether RELOCATABLE is set.
///     value (int): Raw bitflag value.
#[pyclass(name = "ScopeAttributes", from_py_object)]
#[derive(Clone)]
pub struct PyScopeAttributes {
    pub inner: ScopeAttributes,
}

#[pymethods]
impl PyScopeAttributes {
    #[new]
    #[pyo3(signature = (value: "int"=0), text_signature = "(value: int = 0)")]
    fn new(value: u32) -> Self {
        Self {
            inner: ScopeAttributes::from_bits_truncate(value),
        }
    }

    #[classattr]
    const PARALLEL: u32 = ScopeAttributes::PARALLEL.bits();

    #[classattr]
    const RELOCATABLE: u32 = ScopeAttributes::RELOCATABLE.bits();

    #[getter]
    fn is_parallel(&self) -> bool {
        self.inner.contains(ScopeAttributes::PARALLEL)
    }

    #[getter]
    fn is_relocatable(&self) -> bool {
        self.inner.contains(ScopeAttributes::RELOCATABLE)
    }

    fn __or__(&self, other: &PyScopeAttributes) -> PyScopeAttributes {
        PyScopeAttributes {
            inner: self.inner | other.inner,
        }
    }

    fn __and__(&self, other: &PyScopeAttributes) -> PyScopeAttributes {
        PyScopeAttributes {
            inner: self.inner & other.inner,
        }
    }

    fn __repr__(&self) -> String {
        format!("ScopeAttributes({:?})", self.inner)
    }

    #[getter]
    fn value(&self) -> u32 {
        self.inner.bits()
    }
}

// ---------------------------------------------------------------------------
// ToolAttributes (bitflag wrapper)
// ---------------------------------------------------------------------------

/// Bitflag attributes for tool handles.
///
/// Class attributes:
///     LOCAL (int): The tool executes locally.
///
/// Properties:
///     is_local (bool): Whether LOCAL is set.
///     value (int): Raw bitflag value.
#[pyclass(name = "ToolAttributes", from_py_object)]
#[derive(Clone)]
pub struct PyToolAttributes {
    pub inner: ToolAttributes,
}

#[pymethods]
impl PyToolAttributes {
    #[new]
    #[pyo3(signature = (value: "int"=0), text_signature = "(value: int = 0)")]
    fn new(value: u32) -> Self {
        Self {
            inner: ToolAttributes::from_bits_truncate(value),
        }
    }

    #[classattr]
    const LOCAL: u32 = ToolAttributes::LOCAL.bits();

    #[getter]
    fn is_local(&self) -> bool {
        self.inner.contains(ToolAttributes::LOCAL)
    }

    fn __or__(&self, other: &PyToolAttributes) -> PyToolAttributes {
        PyToolAttributes {
            inner: self.inner | other.inner,
        }
    }

    fn __and__(&self, other: &PyToolAttributes) -> PyToolAttributes {
        PyToolAttributes {
            inner: self.inner & other.inner,
        }
    }

    fn __repr__(&self) -> String {
        format!("ToolAttributes({:?})", self.inner)
    }

    #[getter]
    fn value(&self) -> u32 {
        self.inner.bits()
    }
}

// ---------------------------------------------------------------------------
// LLMAttributes (bitflag wrapper)
// ---------------------------------------------------------------------------

/// Bitflag attributes for LLM handles.
///
/// Class attributes:
///     STATELESS (int): The LLM call is stateless.
///     STREAMING (int): The LLM call uses streaming responses.
///
/// Properties:
///     is_stateless (bool): Whether STATELESS is set.
///     is_streaming (bool): Whether STREAMING is set.
///     value (int): Raw bitflag value.
#[pyclass(name = "LLMAttributes", from_py_object)]
#[derive(Clone)]
pub struct PyLLMAttributes {
    pub inner: LLMAttributes,
}

#[pymethods]
impl PyLLMAttributes {
    #[new]
    #[pyo3(signature = (value: "int"=0), text_signature = "(value: int = 0)")]
    fn new(value: u32) -> Self {
        Self {
            inner: LLMAttributes::from_bits_truncate(value),
        }
    }

    #[classattr]
    const STATELESS: u32 = LLMAttributes::STATELESS.bits();

    #[classattr]
    const STREAMING: u32 = LLMAttributes::STREAMING.bits();

    #[getter]
    fn is_stateless(&self) -> bool {
        self.inner.contains(LLMAttributes::STATELESS)
    }

    #[getter]
    fn is_streaming(&self) -> bool {
        self.inner.contains(LLMAttributes::STREAMING)
    }

    fn __or__(&self, other: &PyLLMAttributes) -> PyLLMAttributes {
        PyLLMAttributes {
            inner: self.inner | other.inner,
        }
    }

    fn __and__(&self, other: &PyLLMAttributes) -> PyLLMAttributes {
        PyLLMAttributes {
            inner: self.inner & other.inner,
        }
    }

    fn __repr__(&self) -> String {
        format!("LLMAttributes({:?})", self.inner)
    }

    #[getter]
    fn value(&self) -> u32 {
        self.inner.bits()
    }
}

// ---------------------------------------------------------------------------
// ScopeType enum
// ---------------------------------------------------------------------------

/// The type of an execution scope, indicating what component owns it.
///
/// Variants: Agent, Function, Tool, Llm, Retriever, Embedder, Reranker,
/// Guardrail, Evaluator, Custom, Unknown.
#[pyclass(name = "ScopeType", eq, eq_int, from_py_object)]
#[derive(Clone, PartialEq)]
pub enum PyScopeType {
    Agent = 0,
    Function = 1,
    Tool = 2,
    Llm = 3,
    Retriever = 4,
    Embedder = 5,
    Reranker = 6,
    Guardrail = 7,
    Evaluator = 8,
    Custom = 9,
    Unknown = 10,
}

impl From<PyScopeType> for CoreScopeType {
    fn from(py: PyScopeType) -> Self {
        match py {
            PyScopeType::Agent => CoreScopeType::Agent,
            PyScopeType::Function => CoreScopeType::Function,
            PyScopeType::Tool => CoreScopeType::Tool,
            PyScopeType::Llm => CoreScopeType::Llm,
            PyScopeType::Retriever => CoreScopeType::Retriever,
            PyScopeType::Embedder => CoreScopeType::Embedder,
            PyScopeType::Reranker => CoreScopeType::Reranker,
            PyScopeType::Guardrail => CoreScopeType::Guardrail,
            PyScopeType::Evaluator => CoreScopeType::Evaluator,
            PyScopeType::Custom => CoreScopeType::Custom,
            PyScopeType::Unknown => CoreScopeType::Unknown,
        }
    }
}

impl From<CoreScopeType> for PyScopeType {
    fn from(st: CoreScopeType) -> Self {
        match st {
            CoreScopeType::Agent => PyScopeType::Agent,
            CoreScopeType::Function => PyScopeType::Function,
            CoreScopeType::Tool => PyScopeType::Tool,
            CoreScopeType::Llm => PyScopeType::Llm,
            CoreScopeType::Retriever => PyScopeType::Retriever,
            CoreScopeType::Embedder => PyScopeType::Embedder,
            CoreScopeType::Reranker => PyScopeType::Reranker,
            CoreScopeType::Guardrail => PyScopeType::Guardrail,
            CoreScopeType::Evaluator => PyScopeType::Evaluator,
            CoreScopeType::Custom => PyScopeType::Custom,
            CoreScopeType::Unknown => PyScopeType::Unknown,
        }
    }
}

// ---------------------------------------------------------------------------
// ScopeHandle
// ---------------------------------------------------------------------------

/// A handle representing an active execution scope in the scope stack.
///
/// Properties:
///     uuid (str): Unique identifier.
///     name (str): Human-readable scope name.
///     scope_type (ScopeType): The kind of component owning this scope.
///     attributes (ScopeAttributes): Behavioral flags.
///     parent_uuid (str | None): Parent scope UUID.
///     data (Any | None): Application-specific data.
///     metadata (Any | None): Metadata (e.g., tracing info).
#[pyclass(name = "ScopeHandle", from_py_object)]
#[derive(Clone)]
pub struct PyScopeHandle {
    pub inner: ScopeHandle,
}

#[pymethods]
impl PyScopeHandle {
    #[getter]
    fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn scope_type(&self) -> PyScopeType {
        self.inner.scope_type.into()
    }

    #[getter]
    fn attributes(&self) -> PyScopeAttributes {
        PyScopeAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }

    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "ScopeHandle(name='{}', uuid='{}')",
            self.inner.name, self.inner.uuid
        )
    }
}

impl From<ScopeHandle> for PyScopeHandle {
    fn from(h: ScopeHandle) -> Self {
        Self { inner: h }
    }
}

// ---------------------------------------------------------------------------
// ToolHandle
// ---------------------------------------------------------------------------

/// A handle representing an active tool invocation.
///
/// Properties:
///     uuid (str): Unique identifier.
///     name (str): Tool name.
///     attributes (ToolAttributes): Behavioral flags.
///     parent_uuid (str | None): Parent scope UUID.
///     data (Any | None): Application-specific data.
///     metadata (Any | None): Metadata.
#[pyclass(name = "ToolHandle", from_py_object)]
#[derive(Clone)]
pub struct PyToolHandle {
    pub inner: ToolHandle,
}

#[pymethods]
impl PyToolHandle {
    #[getter]
    fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn attributes(&self) -> PyToolAttributes {
        PyToolAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }

    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolHandle(name='{}', uuid='{}')",
            self.inner.name, self.inner.uuid
        )
    }
}

impl From<ToolHandle> for PyToolHandle {
    fn from(h: ToolHandle) -> Self {
        Self { inner: h }
    }
}

// ---------------------------------------------------------------------------
// LLMHandle
// ---------------------------------------------------------------------------

/// A handle representing an active LLM call.
///
/// Properties:
///     uuid (str): Unique identifier.
///     name (str): LLM provider/model name.
///     attributes (LLMAttributes): Behavioral flags.
///     parent_uuid (str | None): Parent scope UUID.
///     data (Any | None): Application-specific data.
///     metadata (Any | None): Metadata.
#[pyclass(name = "LLMHandle", from_py_object)]
#[derive(Clone)]
pub struct PyLLMHandle {
    pub inner: LLMHandle,
}

#[pymethods]
impl PyLLMHandle {
    #[getter]
    fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn attributes(&self) -> PyLLMAttributes {
        PyLLMAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }

    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "LLMHandle(name='{}', uuid='{}')",
            self.inner.name, self.inner.uuid
        )
    }
}

impl From<LLMHandle> for PyLLMHandle {
    fn from(h: LLMHandle) -> Self {
        Self { inner: h }
    }
}

// ---------------------------------------------------------------------------
// LLMRequest
// ---------------------------------------------------------------------------

/// An opaque request structure representing an outgoing LLM API call.
///
/// Properties:
///     headers (dict): Metadata key-value pairs.
///     content (Any): The request payload.
#[pyclass(name = "LLMRequest", from_py_object)]
#[derive(Clone)]
pub struct PyLLMRequest {
    pub inner: LLMRequest,
}

#[pymethods]
impl PyLLMRequest {
    /// Create a new LLMRequest.
    ///
    /// Args:
    ///     headers: A dict of metadata key-value pairs.
    ///     content: The request payload (any JSON-serializable object).
    #[new]
    #[pyo3(
        signature = (headers: "dict[str, str]", content: "object"),
        text_signature = "(headers: dict[str, str], content: object)"
    )]
    fn new(headers: &Bound<'_, PyDict>, content: &Bound<'_, PyAny>) -> PyResult<Self> {
        let headers_json = py_to_json(headers.as_any())?;
        let headers_map = match headers_json {
            serde_json::Value::Object(m) => m,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "headers must be a dict",
                ));
            }
        };
        let content_json = py_to_json(content)?;
        Ok(Self {
            inner: LLMRequest {
                headers: headers_map,
                content: content_json,
            },
        })
    }

    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &serde_json::Value::Object(self.inner.headers.clone()))
    }

    #[getter]
    fn content(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.content)
    }

    fn __repr__(&self) -> String {
        "LLMRequest(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

#[pyclass(name = "ScopeStartEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyScopeStartEvent {
    pub inner: ScopeStartEvent,
}

#[pymethods]
impl PyScopeStartEvent {
    #[getter]
    fn kind(&self) -> &'static str {
        "ScopeStart"
    }
    #[getter]
    fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    fn attributes(&self) -> PyScopeAttributes {
        PyScopeAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    fn scope_type(&self) -> PyScopeType {
        self.inner.scope_type.into()
    }
}

#[pyclass(name = "ScopeEndEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyScopeEndEvent {
    pub inner: ScopeEndEvent,
}

#[pymethods]
impl PyScopeEndEvent {
    #[getter]
    fn kind(&self) -> &'static str {
        "ScopeEnd"
    }
    #[getter]
    fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    fn attributes(&self) -> PyScopeAttributes {
        PyScopeAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    fn scope_type(&self) -> PyScopeType {
        self.inner.scope_type.into()
    }
}

#[pyclass(name = "ToolStartEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyToolStartEvent {
    pub inner: ToolStartEvent,
}

#[pymethods]
impl PyToolStartEvent {
    #[getter]
    fn kind(&self) -> &'static str {
        "ToolStart"
    }
    #[getter]
    fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    fn attributes(&self) -> PyToolAttributes {
        PyToolAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    fn input(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.input)
    }

    #[getter]
    fn tool_call_id(&self) -> Option<String> {
        self.inner.tool_call_id.clone()
    }
}

#[pyclass(name = "ToolEndEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyToolEndEvent {
    pub inner: ToolEndEvent,
}

#[pymethods]
impl PyToolEndEvent {
    #[getter]
    fn kind(&self) -> &'static str {
        "ToolEnd"
    }
    #[getter]
    fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    fn attributes(&self) -> PyToolAttributes {
        PyToolAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    fn output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.output)
    }

    #[getter]
    fn tool_call_id(&self) -> Option<String> {
        self.inner.tool_call_id.clone()
    }
}

#[pyclass(name = "LLMStartEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyLLMStartEvent {
    pub inner: LLMStartEvent,
}

#[pymethods]
impl PyLLMStartEvent {
    #[getter]
    fn kind(&self) -> &'static str {
        "LLMStart"
    }
    #[getter]
    fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    fn attributes(&self) -> PyLLMAttributes {
        PyLLMAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    fn input(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.input)
    }

    #[getter]
    fn model_name(&self) -> Option<String> {
        self.inner.model_name.clone()
    }

    #[getter]
    fn annotated_request(&self) -> Option<PyAnnotatedLLMRequest> {
        self.inner
            .annotated_request
            .as_ref()
            .map(|a| PyAnnotatedLLMRequest {
                inner: a.as_ref().clone(),
            })
    }
}

#[pyclass(name = "LLMEndEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyLLMEndEvent {
    pub inner: LLMEndEvent,
}

#[pymethods]
impl PyLLMEndEvent {
    #[getter]
    fn kind(&self) -> &'static str {
        "LLMEnd"
    }
    #[getter]
    fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    fn attributes(&self) -> PyLLMAttributes {
        PyLLMAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    fn output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.output)
    }

    #[getter]
    fn model_name(&self) -> Option<String> {
        self.inner.model_name.clone()
    }

    #[getter]
    fn annotated_response(&self) -> Option<PyAnnotatedLLMResponse> {
        self.inner
            .annotated_response
            .as_ref()
            .map(|a| PyAnnotatedLLMResponse {
                inner: a.as_ref().clone(),
            })
    }
}

#[pyclass(name = "MarkEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMarkEvent {
    pub inner: MarkEvent,
}

#[pymethods]
impl PyMarkEvent {
    #[getter]
    fn kind(&self) -> &'static str {
        "Mark"
    }
    #[getter]
    fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
}

// ---------------------------------------------------------------------------
// AtifExporter
// ---------------------------------------------------------------------------

/// ATIF trajectory exporter that collects events and exports ATIF trajectories.
///
/// Create an exporter, register it as an event subscriber, then call
/// ``export()`` or ``export_json()`` to produce an ATIF trajectory.
///
/// Example:
/// ```python
/// exporter = AtifExporter("session-1", "my-agent", "1.0.0", model_name="gpt-4")
/// exporter.register("atif")
/// # ... run agent ...
/// trajectory = exporter.export()
/// exporter.deregister("atif")
/// ```
#[pyclass(name = "AtifExporter")]
pub struct PyAtifExporter {
    inner: nemo_flow::atif::AtifExporter,
}

#[pymethods]
impl PyAtifExporter {
    #[new]
    #[pyo3(signature = (session_id, agent_name, agent_version, *, model_name=None, tool_definitions=None, extra=None))]
    fn new(
        session_id: String,
        agent_name: String,
        agent_version: String,
        model_name: Option<String>,
        tool_definitions: Option<&Bound<'_, pyo3::types::PyList>>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let tool_defs = match tool_definitions {
            Some(list) => {
                let mut defs = Vec::new();
                for item in list.iter() {
                    defs.push(py_to_json(&item)?);
                }
                Some(defs)
            }
            None => None,
        };
        let extra_json = match extra {
            Some(obj) if !obj.is_none() => Some(py_to_json(obj)?),
            _ => None,
        };
        let agent_info = nemo_flow::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: tool_defs,
            extra: extra_json,
        };
        Ok(Self {
            inner: nemo_flow::atif::AtifExporter::new(session_id, agent_info),
        })
    }

    /// Register this exporter as an event subscriber with the given name.
    fn register(&self, name: String) -> PyResult<()> {
        let subscriber = self.inner.subscriber();
        nemo_flow::api::subscriber::register_subscriber(&name, subscriber)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Deregister the event subscriber with the given name.
    ///
    /// Returns ``True`` if a subscriber with that name was found and removed.
    fn deregister(&self, name: String) -> PyResult<bool> {
        nemo_flow::api::subscriber::deregister_subscriber(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Export the collected events as an ATIF trajectory dict.
    ///
    /// Returns:
    ///     A dict representing the ATIF trajectory.
    fn export(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let trajectory = self.inner.export();
        let value = serde_json::to_value(&trajectory).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {e}"))
        })?;
        json_to_py(py, &value)
    }

    /// Export the collected events as a JSON string.
    ///
    /// Returns:
    ///     A JSON string representing the ATIF trajectory.
    fn export_json(&self) -> PyResult<String> {
        let trajectory = self.inner.export();
        serde_json::to_string(&trajectory).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {e}"))
        })
    }

    /// Clear all collected events.
    fn clear(&self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        "<AtifExporter>".to_string()
    }
}

// ---------------------------------------------------------------------------
// OpenTelemetry subscriber
// ---------------------------------------------------------------------------

/// Mutable configuration object for the OpenTelemetry subscriber.
///
/// Create the config, update fields as needed, then pass it to
/// ``OpenTelemetrySubscriber(config)``.
///
/// Example:
/// ```python
/// config = OpenTelemetryConfig()
/// config.endpoint = "http://localhost:4318/v1/traces"
/// config.service_name = "demo-agent"
/// config.headers = {"authorization": "Bearer token"}
/// ```
#[pyclass(name = "OpenTelemetryConfig")]
pub struct PyOpenTelemetryConfig {
    #[pyo3(get, set)]
    transport: String,
    #[pyo3(get, set)]
    endpoint: Option<String>,
    #[pyo3(get, set)]
    service_name: String,
    #[pyo3(get, set)]
    service_namespace: Option<String>,
    #[pyo3(get, set)]
    service_version: Option<String>,
    #[pyo3(get, set)]
    instrumentation_scope: String,
    #[pyo3(get, set)]
    timeout_millis: u64,
    headers: HashMap<String, String>,
    resource_attributes: HashMap<String, String>,
}

impl PyOpenTelemetryConfig {
    fn to_rust_config(&self) -> PyResult<nemo_flow::observability::otel::OpenTelemetryConfig> {
        let mut config = match self.transport.as_str() {
            "http_binary" => nemo_flow::observability::otel::OpenTelemetryConfig::http_binary(
                self.service_name.clone(),
            ),
            "grpc" => {
                nemo_flow::observability::otel::OpenTelemetryConfig::grpc(self.service_name.clone())
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "transport must be 'http_binary' or 'grpc', got {other:?}"
                )));
            }
        }
        .with_instrumentation_scope(self.instrumentation_scope.clone())
        .with_timeout(Duration::from_millis(self.timeout_millis));

        if let Some(endpoint) = &self.endpoint {
            config = config.with_endpoint(endpoint.clone());
        }
        if let Some(namespace) = &self.service_namespace {
            config = config.with_service_namespace(namespace.clone());
        }
        if let Some(version) = &self.service_version {
            config = config.with_service_version(version.clone());
        }
        for (key, value) in &self.headers {
            config = config.with_header(key.clone(), value.clone());
        }
        for (key, value) in &self.resource_attributes {
            config = config.with_resource_attribute(key.clone(), value.clone());
        }
        Ok(config)
    }
}

#[pymethods]
impl PyOpenTelemetryConfig {
    #[new]
    fn new() -> Self {
        Self {
            transport: "http_binary".to_string(),
            endpoint: None,
            service_name: "nemo-flow".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "nemo-flow-otel".to_string(),
            timeout_millis: 3_000,
            headers: HashMap::new(),
            resource_attributes: HashMap::new(),
        }
    }

    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &serde_json::to_value(&self.headers).unwrap_or_default())
    }

    #[setter]
    fn set_headers(&mut self, headers: &Bound<'_, PyAny>) -> PyResult<()> {
        self.headers = py_string_map(headers, "headers")?;
        Ok(())
    }

    #[getter]
    fn resource_attributes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &serde_json::to_value(&self.resource_attributes).unwrap_or_default(),
        )
    }

    #[setter]
    fn set_resource_attributes(&mut self, resource_attributes: &Bound<'_, PyAny>) -> PyResult<()> {
        self.resource_attributes = py_string_map(resource_attributes, "resource_attributes")?;
        Ok(())
    }

    fn set_header(&mut self, key: String, value: String) {
        self.headers.insert(key, value);
    }

    fn set_resource_attribute(&mut self, key: String, value: String) {
        self.resource_attributes.insert(key, value);
    }

    fn __repr__(&self) -> String {
        format!(
            "<OpenTelemetryConfig transport={:?} endpoint={:?}>",
            self.transport, self.endpoint
        )
    }
}

/// OpenTelemetry-backed event subscriber.
///
/// Construct it from an ``OpenTelemetryConfig``, register it with a subscriber
/// name, then call ``force_flush()`` or ``shutdown()`` when appropriate.
#[pyclass(name = "OpenTelemetrySubscriber")]
pub struct PyOpenTelemetrySubscriber {
    inner: nemo_flow::observability::otel::OpenTelemetrySubscriber,
}

#[pymethods]
impl PyOpenTelemetrySubscriber {
    #[new]
    fn new(config: PyRef<'_, PyOpenTelemetryConfig>) -> PyResult<Self> {
        let inner =
            nemo_flow::observability::otel::OpenTelemetrySubscriber::new(config.to_rust_config()?)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Register this subscriber globally with the given name.
    fn register(&self, name: String) -> PyResult<()> {
        self.inner
            .register(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Deregister a subscriber by name. Returns ``True`` if found.
    fn deregister(&self, name: String) -> PyResult<bool> {
        self.inner
            .deregister(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Force a flush of finished spans through the exporter.
    fn force_flush(&self) -> PyResult<()> {
        self.inner
            .force_flush()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Shut down the underlying tracer provider.
    fn shutdown(&self) -> PyResult<()> {
        self.inner
            .shutdown()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "<OpenTelemetrySubscriber>".to_string()
    }
}

/// Mutable config object for ``OpenInferenceSubscriber``.
///
/// Example:
/// ```python
/// config = OpenInferenceConfig()
/// config.endpoint = "http://localhost:4318/v1/traces"
/// config.service_name = "demo-agent"
/// config.headers = {"authorization": "Bearer token"}
/// ```
#[pyclass(name = "OpenInferenceConfig")]
pub struct PyOpenInferenceConfig {
    #[pyo3(get, set)]
    transport: String,
    #[pyo3(get, set)]
    endpoint: Option<String>,
    #[pyo3(get, set)]
    service_name: String,
    #[pyo3(get, set)]
    service_namespace: Option<String>,
    #[pyo3(get, set)]
    service_version: Option<String>,
    #[pyo3(get, set)]
    instrumentation_scope: String,
    #[pyo3(get, set)]
    timeout_millis: u64,
    headers: HashMap<String, String>,
    resource_attributes: HashMap<String, String>,
}

impl PyOpenInferenceConfig {
    fn to_rust_config(
        &self,
    ) -> PyResult<nemo_flow::observability::openinference::OpenInferenceConfig> {
        let transport = match self.transport.as_str() {
            "http_binary" => nemo_flow::observability::openinference::OtlpTransport::HttpBinary,
            "grpc" => nemo_flow::observability::openinference::OtlpTransport::Grpc,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "transport must be 'http_binary' or 'grpc', got {other:?}"
                )));
            }
        };

        let mut config = nemo_flow::observability::openinference::OpenInferenceConfig::new()
            .with_transport(transport)
            .with_service_name(self.service_name.clone())
            .with_instrumentation_scope(self.instrumentation_scope.clone())
            .with_timeout(Duration::from_millis(self.timeout_millis));

        if let Some(endpoint) = &self.endpoint {
            config = config.with_endpoint(endpoint.clone());
        }
        if let Some(namespace) = &self.service_namespace {
            config = config.with_service_namespace(namespace.clone());
        }
        if let Some(version) = &self.service_version {
            config = config.with_service_version(version.clone());
        }
        for (key, value) in &self.headers {
            config = config.with_header(key.clone(), value.clone());
        }
        for (key, value) in &self.resource_attributes {
            config = config.with_resource_attribute(key.clone(), value.clone());
        }
        Ok(config)
    }
}

#[pymethods]
impl PyOpenInferenceConfig {
    #[new]
    fn new() -> Self {
        Self {
            transport: "http_binary".to_string(),
            endpoint: None,
            service_name: "nemo-flow".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "nemo-flow-openinference".to_string(),
            timeout_millis: 3_000,
            headers: HashMap::new(),
            resource_attributes: HashMap::new(),
        }
    }

    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &serde_json::to_value(&self.headers).unwrap_or_default())
    }

    #[setter]
    fn set_headers(&mut self, headers: &Bound<'_, PyAny>) -> PyResult<()> {
        self.headers = py_string_map(headers, "headers")?;
        Ok(())
    }

    #[getter]
    fn resource_attributes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &serde_json::to_value(&self.resource_attributes).unwrap_or_default(),
        )
    }

    #[setter]
    fn set_resource_attributes(&mut self, resource_attributes: &Bound<'_, PyAny>) -> PyResult<()> {
        self.resource_attributes = py_string_map(resource_attributes, "resource_attributes")?;
        Ok(())
    }

    fn set_header(&mut self, key: String, value: String) {
        self.headers.insert(key, value);
    }

    fn set_resource_attribute(&mut self, key: String, value: String) {
        self.resource_attributes.insert(key, value);
    }

    fn __repr__(&self) -> String {
        format!(
            "<OpenInferenceConfig transport={:?} endpoint={:?}>",
            self.transport, self.endpoint
        )
    }
}

/// OpenInference-backed event subscriber.
#[pyclass(name = "OpenInferenceSubscriber")]
pub struct PyOpenInferenceSubscriber {
    inner: nemo_flow::observability::openinference::OpenInferenceSubscriber,
}

#[pymethods]
impl PyOpenInferenceSubscriber {
    #[new]
    fn new(config: PyRef<'_, PyOpenInferenceConfig>) -> PyResult<Self> {
        let inner = nemo_flow::observability::openinference::OpenInferenceSubscriber::new(
            config.to_rust_config()?,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn register(&self, name: String) -> PyResult<()> {
        self.inner
            .register(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn deregister(&self, name: String) -> PyResult<bool> {
        self.inner
            .deregister(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn force_flush(&self) -> PyResult<()> {
        self.inner
            .force_flush()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn shutdown(&self) -> PyResult<()> {
        self.inner
            .shutdown()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "<OpenInferenceSubscriber>".to_string()
    }
}

// ---------------------------------------------------------------------------
// AnnotatedLLMRequest
// ---------------------------------------------------------------------------

/// A structured view of an LLM request produced by a Codec.
///
/// Provides typed access to conversation messages, model name, generation
/// parameters, tool definitions, tool choice, and extensible extra fields.
///
/// Properties:
///     messages (list): Parsed conversation messages (list of dicts with a ``role`` key).
///     model (str | None): Model identifier (e.g., ``"gpt-4"``).
///     params (dict | None): Normalized generation parameters.
///     tools (list | None): Tool definitions (function schemas).
///     tool_choice (Any | None): Tool choice control.
///     extra (dict): Provider-specific extra fields.
///
/// Helper methods:
///     system_prompt() -> str | None: Text of the first system message.
///     last_user_message() -> str | None: Text of the last user message.
///     has_tool_calls() -> bool: Whether any assistant message has tool calls.
#[pyclass(name = "AnnotatedLLMRequest", from_py_object)]
#[derive(Clone)]
pub struct PyAnnotatedLLMRequest {
    pub inner: AnnotatedLLMRequest,
}

#[pymethods]
impl PyAnnotatedLLMRequest {
    /// Create a new AnnotatedLLMRequest.
    ///
    /// Args:
    ///     messages: A list of message dicts, each with a ``role`` key.
    ///     model: Optional model identifier.
    ///     params: Optional generation parameters dict.
    ///     tools: Optional list of tool definition dicts.
    ///     tool_choice: Optional tool choice control.
    ///     extra: Optional dict of provider-specific extra fields.
    #[new]
    #[pyo3(signature = (messages, *, model=None, params=None, tools=None, tool_choice=None, extra=None))]
    fn new(
        messages: &Bound<'_, PyAny>,
        model: Option<String>,
        params: Option<&Bound<'_, PyAny>>,
        tools: Option<&Bound<'_, PyAny>>,
        tool_choice: Option<&Bound<'_, PyAny>>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let msgs: Vec<Message> = pythonize::depythonize(messages).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid messages: each dict must include a 'role' key (user/system/assistant/tool): {e}"
            ))
        })?;
        let gen_params: Option<GenerationParams> = match params {
            Some(p) if !p.is_none() => Some(pythonize::depythonize(p).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid params: {e}"))
            })?),
            _ => None,
        };
        let tool_defs: Option<Vec<ToolDefinition>> = match tools {
            Some(t) if !t.is_none() => Some(pythonize::depythonize(t).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid tools: {e}"))
            })?),
            _ => None,
        };
        let tc: Option<ToolChoice> = match tool_choice {
            Some(tc_val) if !tc_val.is_none() => {
                Some(pythonize::depythonize(tc_val).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("invalid tool_choice: {e}"))
                })?)
            }
            _ => None,
        };
        let extra_map: serde_json::Map<String, serde_json::Value> = match extra {
            Some(e) if !e.is_none() => pythonize::depythonize(e).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid extra: {e}"))
            })?,
            _ => serde_json::Map::new(),
        };
        Ok(Self {
            inner: AnnotatedLLMRequest {
                messages: msgs,
                model,
                params: gen_params,
                tools: tool_defs,
                tool_choice: tc,
                extra: extra_map,
            },
        })
    }

    #[getter]
    fn messages(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::to_value(&self.inner.messages).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("serialization error: {e}"))
        })?;
        json_to_py(py, &value)
    }

    #[setter]
    fn set_messages(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.messages = pythonize::depythonize(value).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid messages: each dict must include a 'role' key (user/system/assistant/tool): {e}"
            ))
        })?;
        Ok(())
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    #[setter]
    fn set_model(&mut self, value: Option<String>) {
        self.inner.model = value;
    }

    #[getter]
    fn params(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.params {
            Some(p) => {
                let value = serde_json::to_value(p).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("serialization error: {e}"))
                })?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[setter]
    fn set_params(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            self.inner.params = None;
        } else {
            self.inner.params = Some(pythonize::depythonize(value).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid params: {e}"))
            })?);
        }
        Ok(())
    }

    #[getter]
    fn tools(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.tools {
            Some(t) => {
                let value = serde_json::to_value(t).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("serialization error: {e}"))
                })?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[setter]
    fn set_tools(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            self.inner.tools = None;
        } else {
            self.inner.tools = Some(pythonize::depythonize(value).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid tools: {e}"))
            })?);
        }
        Ok(())
    }

    #[getter]
    fn tool_choice(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.tool_choice {
            Some(tc) => {
                let value = serde_json::to_value(tc).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("serialization error: {e}"))
                })?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[setter]
    fn set_tool_choice(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            self.inner.tool_choice = None;
        } else {
            self.inner.tool_choice = Some(pythonize::depythonize(value).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid tool_choice: {e}"))
            })?);
        }
        Ok(())
    }

    #[getter]
    fn extra(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::Value::Object(self.inner.extra.clone());
        json_to_py(py, &value)
    }

    #[setter]
    fn set_extra(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.extra = pythonize::depythonize(value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid extra: {e}")))?;
        Ok(())
    }

    /// Extract the text content of the first system message, if any.
    fn system_prompt(&self) -> Option<String> {
        self.inner.system_prompt().map(|s| s.to_string())
    }

    /// Get the text content of the last user message, if any.
    fn last_user_message(&self) -> Option<String> {
        self.inner.last_user_message().map(|s| s.to_string())
    }

    /// Check if any assistant message contains tool calls.
    fn has_tool_calls(&self) -> bool {
        self.inner.has_tool_calls()
    }

    fn __repr__(&self) -> String {
        format!(
            "<AnnotatedLLMRequest messages={} model={:?}>",
            self.inner.messages.len(),
            self.inner.model
        )
    }
}

// ---------------------------------------------------------------------------
// AnnotatedLLMResponse (read-only wrapper)
// ---------------------------------------------------------------------------

/// Structured view of an LLM response produced by a response codec.
///
/// Read-only: fields are accessed via properties. Complex fields
/// (message, tool_calls, usage, api_specific) return Python dicts/lists.
///
/// Properties:
///     id -> str | None: Response ID from the API.
///     model -> str | None: The model that served the request.
///     message -> Any | None: The assistant's response content.
///     tool_calls -> list | None: Tool calls requested by the model.
///     finish_reason -> str | None: Why generation stopped.
///     usage -> dict | None: Token usage statistics.
///     api_specific -> dict | None: API-specific response data.
///     extra -> dict: Unmodeled top-level fields (catch-all).
///
/// Helper methods:
///     response_text() -> str | None: Text content of the response message.
///     has_tool_calls() -> bool: Whether the response contains tool calls.
#[pyclass(name = "AnnotatedLLMResponse", skip_from_py_object)]
#[derive(Clone)]
pub struct PyAnnotatedLLMResponse {
    pub inner: AnnotatedLLMResponse,
}

#[pymethods]
impl PyAnnotatedLLMResponse {
    #[getter]
    fn id(&self) -> Option<String> {
        self.inner.id.clone()
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    #[getter]
    fn message(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.message {
            Some(m) => {
                let value = serde_json::to_value(m).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("serialization error: {e}"))
                })?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[getter]
    fn tool_calls(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.tool_calls {
            Some(tc) => {
                let value = serde_json::to_value(tc).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("serialization error: {e}"))
                })?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[getter]
    fn finish_reason(&self) -> Option<String> {
        self.inner
            .finish_reason
            .as_ref()
            .and_then(|fr| serde_json::to_value(fr).ok())
            .and_then(|v| v.as_str().map(|s| s.to_string()))
    }

    #[getter]
    fn usage(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.usage {
            Some(u) => {
                let value = serde_json::to_value(u).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("serialization error: {e}"))
                })?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[getter]
    fn api_specific(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.api_specific {
            Some(a) => {
                let value = serde_json::to_value(a).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("serialization error: {e}"))
                })?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[getter]
    fn extra(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &serde_json::Value::Object(self.inner.extra.clone()))
    }

    /// Extract the text content of the response message.
    fn response_text(&self) -> Option<String> {
        self.inner.response_text().map(|s| s.to_string())
    }

    /// Check if the response contains any tool calls.
    fn has_tool_calls(&self) -> bool {
        self.inner.has_tool_calls()
    }

    fn __repr__(&self) -> String {
        format!(
            "<AnnotatedLLMResponse id={:?} model={:?}>",
            self.inner.id, self.inner.model
        )
    }
}

// ---------------------------------------------------------------------------
// Built-in LLM Codec pyclasses
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Chat Completions API.
///
/// Implements both ``LlmCodec`` (decode/encode for requests) and
/// ``LlmResponseCodec`` (decode_response for responses).
///
/// Example:
/// ```python
/// from nemo_flow.codecs import OpenAIChatCodec
/// codec = OpenAIChatCodec()
/// annotated_req = codec.decode(request)
/// annotated_resp = codec.decode_response(response)
/// ```
#[pyclass(name = "OpenAIChatCodec")]
pub struct PyOpenAIChatCodec {
    pub(crate) inner_codec: Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: Arc<dyn LlmResponseCodec>,
}

#[pymethods]
impl PyOpenAIChatCodec {
    #[new]
    fn new() -> Self {
        Self {
            inner_codec: Arc::new(nemo_flow::codec::openai_chat::OpenAIChatCodec),
            inner_response_codec: Arc::new(nemo_flow::codec::openai_chat::OpenAIChatCodec),
        }
    }

    /// Parse an opaque ``LLMRequest`` into a structured ``AnnotatedLLMRequest``.
    fn decode(&self, request: &PyLLMRequest) -> PyResult<PyAnnotatedLLMRequest> {
        self.inner_codec
            .decode(&request.inner)
            .map(|r| PyAnnotatedLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Merge structured changes back into the opaque request.
    fn encode(
        &self,
        annotated: &PyAnnotatedLLMRequest,
        original: &PyLLMRequest,
    ) -> PyResult<PyLLMRequest> {
        self.inner_codec
            .encode(&annotated.inner, &original.inner)
            .map(|r| PyLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Parse a raw JSON response into a structured ``AnnotatedLLMResponse``.
    fn decode_response(&self, response: &Bound<'_, PyAny>) -> PyResult<PyAnnotatedLLMResponse> {
        let json = py_to_json(response)?;
        self.inner_response_codec
            .decode_response(&json)
            .map(|r| PyAnnotatedLLMResponse { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> &'static str {
        "<OpenAIChatCodec>"
    }
}

/// Built-in codec for the OpenAI Responses API.
///
/// Implements both ``LlmCodec`` (decode/encode for requests) and
/// ``LlmResponseCodec`` (decode_response for responses).
///
/// Example:
/// ```python
/// from nemo_flow.codecs import OpenAIResponsesCodec
/// codec = OpenAIResponsesCodec()
/// annotated_req = codec.decode(request)
/// annotated_resp = codec.decode_response(response)
/// ```
#[pyclass(name = "OpenAIResponsesCodec")]
pub struct PyOpenAIResponsesCodec {
    pub(crate) inner_codec: Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: Arc<dyn LlmResponseCodec>,
}

#[pymethods]
impl PyOpenAIResponsesCodec {
    #[new]
    fn new() -> Self {
        Self {
            inner_codec: Arc::new(nemo_flow::codec::openai_responses::OpenAIResponsesCodec),
            inner_response_codec: Arc::new(
                nemo_flow::codec::openai_responses::OpenAIResponsesCodec,
            ),
        }
    }

    /// Parse an opaque ``LLMRequest`` into a structured ``AnnotatedLLMRequest``.
    fn decode(&self, request: &PyLLMRequest) -> PyResult<PyAnnotatedLLMRequest> {
        self.inner_codec
            .decode(&request.inner)
            .map(|r| PyAnnotatedLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Merge structured changes back into the opaque request.
    fn encode(
        &self,
        annotated: &PyAnnotatedLLMRequest,
        original: &PyLLMRequest,
    ) -> PyResult<PyLLMRequest> {
        self.inner_codec
            .encode(&annotated.inner, &original.inner)
            .map(|r| PyLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Parse a raw JSON response into a structured ``AnnotatedLLMResponse``.
    fn decode_response(&self, response: &Bound<'_, PyAny>) -> PyResult<PyAnnotatedLLMResponse> {
        let json = py_to_json(response)?;
        self.inner_response_codec
            .decode_response(&json)
            .map(|r| PyAnnotatedLLMResponse { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> &'static str {
        "<OpenAIResponsesCodec>"
    }
}

/// Built-in codec for the Anthropic Messages API.
///
/// Implements both ``LlmCodec`` (decode/encode for requests) and
/// ``LlmResponseCodec`` (decode_response for responses).
///
/// Example:
/// ```python
/// from nemo_flow.codecs import AnthropicMessagesCodec
/// codec = AnthropicMessagesCodec()
/// annotated_req = codec.decode(request)
/// annotated_resp = codec.decode_response(response)
/// ```
#[pyclass(name = "AnthropicMessagesCodec")]
pub struct PyAnthropicMessagesCodec {
    pub(crate) inner_codec: Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: Arc<dyn LlmResponseCodec>,
}

#[pymethods]
impl PyAnthropicMessagesCodec {
    #[new]
    fn new() -> Self {
        Self {
            inner_codec: Arc::new(nemo_flow::codec::anthropic::AnthropicMessagesCodec),
            inner_response_codec: Arc::new(nemo_flow::codec::anthropic::AnthropicMessagesCodec),
        }
    }

    /// Parse an opaque ``LLMRequest`` into a structured ``AnnotatedLLMRequest``.
    fn decode(&self, request: &PyLLMRequest) -> PyResult<PyAnnotatedLLMRequest> {
        self.inner_codec
            .decode(&request.inner)
            .map(|r| PyAnnotatedLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Merge structured changes back into the opaque request.
    fn encode(
        &self,
        annotated: &PyAnnotatedLLMRequest,
        original: &PyLLMRequest,
    ) -> PyResult<PyLLMRequest> {
        self.inner_codec
            .encode(&annotated.inner, &original.inner)
            .map(|r| PyLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Parse a raw JSON response into a structured ``AnnotatedLLMResponse``.
    fn decode_response(&self, response: &Bound<'_, PyAny>) -> PyResult<PyAnnotatedLLMResponse> {
        let json = py_to_json(response)?;
        self.inner_response_codec
            .decode_response(&json)
            .map(|r| PyAnnotatedLLMResponse { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> &'static str {
        "<AnthropicMessagesCodec>"
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScopeStack>()?;
    m.add_class::<PyLlmStream>()?;
    m.add_class::<PyScopeAttributes>()?;
    m.add_class::<PyToolAttributes>()?;
    m.add_class::<PyLLMAttributes>()?;
    m.add_class::<PyScopeType>()?;
    m.add_class::<PyScopeHandle>()?;
    m.add_class::<PyToolHandle>()?;
    m.add_class::<PyLLMHandle>()?;
    m.add_class::<PyLLMRequest>()?;
    m.add_class::<PyAnnotatedLLMRequest>()?;
    m.add_class::<PyAnnotatedLLMResponse>()?;
    m.add_class::<PyScopeStartEvent>()?;
    m.add_class::<PyScopeEndEvent>()?;
    m.add_class::<PyToolStartEvent>()?;
    m.add_class::<PyToolEndEvent>()?;
    m.add_class::<PyLLMStartEvent>()?;
    m.add_class::<PyLLMEndEvent>()?;
    m.add_class::<PyMarkEvent>()?;
    m.add_class::<PyAtifExporter>()?;
    m.add_class::<PyOpenTelemetryConfig>()?;
    m.add_class::<PyOpenTelemetrySubscriber>()?;
    m.add_class::<PyOpenInferenceConfig>()?;
    m.add_class::<PyOpenInferenceSubscriber>()?;
    m.add_class::<PyOpenAIChatCodec>()?;
    m.add_class::<PyOpenAIResponsesCodec>()?;
    m.add_class::<PyAnthropicMessagesCodec>()?;
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/coverage/py_types_coverage_tests.rs"]
mod coverage_tests;
