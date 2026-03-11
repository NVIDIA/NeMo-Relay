// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing type wrappers for NVMagic core types.
//!
//! Each type wraps its corresponding `nvmagic_core::types` struct and exposes
//! properties via `#[getter]`. Doc comments on `#[pyclass]` and `#[pymethods]`
//! become Python `help()` output.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use nvmagic_core::types as core_types;

use crate::convert::{json_to_py, opt_json_to_py, py_to_json};

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
    pub receiver:
        tokio::sync::Mutex<tokio::sync::mpsc::Receiver<nvmagic_core::Result<serde_json::Value>>>,
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
                tokio::sync::mpsc::Receiver<nvmagic_core::Result<serde_json::Value>>,
            >;
        // SAFETY: The PyLlmStream outlives this future because Python holds a reference to it.
        // The tokio Mutex ensures exclusive access to the receiver.
        let receiver_ref = unsafe { &*receiver_ptr };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = receiver_ref.lock().await;
            match guard.recv().await {
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
pub struct PyScopeStack(pub nvmagic_core::ScopeStackHandle);

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
    pub inner: core_types::ScopeAttributes,
}

#[pymethods]
impl PyScopeAttributes {
    #[new]
    #[pyo3(signature = (value=0))]
    fn new(value: u32) -> Self {
        Self {
            inner: core_types::ScopeAttributes::from_bits_truncate(value),
        }
    }

    #[classattr]
    const PARALLEL: u32 = core_types::ScopeAttributes::PARALLEL.bits();

    #[classattr]
    const RELOCATABLE: u32 = core_types::ScopeAttributes::RELOCATABLE.bits();

    #[getter]
    fn is_parallel(&self) -> bool {
        self.inner.contains(core_types::ScopeAttributes::PARALLEL)
    }

    #[getter]
    fn is_relocatable(&self) -> bool {
        self.inner
            .contains(core_types::ScopeAttributes::RELOCATABLE)
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
    pub inner: core_types::ToolAttributes,
}

#[pymethods]
impl PyToolAttributes {
    #[new]
    #[pyo3(signature = (value=0))]
    fn new(value: u32) -> Self {
        Self {
            inner: core_types::ToolAttributes::from_bits_truncate(value),
        }
    }

    #[classattr]
    const LOCAL: u32 = core_types::ToolAttributes::LOCAL.bits();

    #[getter]
    fn is_local(&self) -> bool {
        self.inner.contains(core_types::ToolAttributes::LOCAL)
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
    pub inner: core_types::LLMAttributes,
}

#[pymethods]
impl PyLLMAttributes {
    #[new]
    #[pyo3(signature = (value=0))]
    fn new(value: u32) -> Self {
        Self {
            inner: core_types::LLMAttributes::from_bits_truncate(value),
        }
    }

    #[classattr]
    const STATELESS: u32 = core_types::LLMAttributes::STATELESS.bits();

    #[classattr]
    const STREAMING: u32 = core_types::LLMAttributes::STREAMING.bits();

    #[getter]
    fn is_stateless(&self) -> bool {
        self.inner.contains(core_types::LLMAttributes::STATELESS)
    }

    #[getter]
    fn is_streaming(&self) -> bool {
        self.inner.contains(core_types::LLMAttributes::STREAMING)
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

impl From<PyScopeType> for core_types::ScopeType {
    fn from(py: PyScopeType) -> Self {
        match py {
            PyScopeType::Agent => core_types::ScopeType::Agent,
            PyScopeType::Function => core_types::ScopeType::Function,
            PyScopeType::Tool => core_types::ScopeType::Tool,
            PyScopeType::Llm => core_types::ScopeType::Llm,
            PyScopeType::Retriever => core_types::ScopeType::Retriever,
            PyScopeType::Embedder => core_types::ScopeType::Embedder,
            PyScopeType::Reranker => core_types::ScopeType::Reranker,
            PyScopeType::Guardrail => core_types::ScopeType::Guardrail,
            PyScopeType::Evaluator => core_types::ScopeType::Evaluator,
            PyScopeType::Custom => core_types::ScopeType::Custom,
            PyScopeType::Unknown => core_types::ScopeType::Unknown,
        }
    }
}

impl From<core_types::ScopeType> for PyScopeType {
    fn from(st: core_types::ScopeType) -> Self {
        match st {
            core_types::ScopeType::Agent => PyScopeType::Agent,
            core_types::ScopeType::Function => PyScopeType::Function,
            core_types::ScopeType::Tool => PyScopeType::Tool,
            core_types::ScopeType::Llm => PyScopeType::Llm,
            core_types::ScopeType::Retriever => PyScopeType::Retriever,
            core_types::ScopeType::Embedder => PyScopeType::Embedder,
            core_types::ScopeType::Reranker => PyScopeType::Reranker,
            core_types::ScopeType::Guardrail => PyScopeType::Guardrail,
            core_types::ScopeType::Evaluator => PyScopeType::Evaluator,
            core_types::ScopeType::Custom => PyScopeType::Custom,
            core_types::ScopeType::Unknown => PyScopeType::Unknown,
        }
    }
}

// ---------------------------------------------------------------------------
// EventType enum
// ---------------------------------------------------------------------------

/// The type of a lifecycle event.
///
/// Variants: Start (scope/handle created), End (scope/handle destroyed),
/// Mark (standalone marker event).
#[pyclass(name = "EventType", eq, eq_int, from_py_object)]
#[derive(Clone, PartialEq)]
pub enum PyEventType {
    Start = 0,
    End = 1,
    Mark = 2,
}

impl From<PyEventType> for core_types::EventType {
    fn from(py: PyEventType) -> Self {
        match py {
            PyEventType::Start => core_types::EventType::Start,
            PyEventType::End => core_types::EventType::End,
            PyEventType::Mark => core_types::EventType::Mark,
        }
    }
}

impl From<core_types::EventType> for PyEventType {
    fn from(et: core_types::EventType) -> Self {
        match et {
            core_types::EventType::Start => PyEventType::Start,
            core_types::EventType::End => PyEventType::End,
            core_types::EventType::Mark => PyEventType::Mark,
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
    pub inner: core_types::ScopeHandle,
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

impl From<core_types::ScopeHandle> for PyScopeHandle {
    fn from(h: core_types::ScopeHandle) -> Self {
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
    pub inner: core_types::ToolHandle,
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

impl From<core_types::ToolHandle> for PyToolHandle {
    fn from(h: core_types::ToolHandle) -> Self {
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
    pub inner: core_types::LLMHandle,
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

impl From<core_types::LLMHandle> for PyLLMHandle {
    fn from(h: core_types::LLMHandle) -> Self {
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
    pub inner: core_types::LLMRequest,
}

#[pymethods]
impl PyLLMRequest {
    /// Create a new LLMRequest.
    ///
    /// Args:
    ///     headers: A dict of metadata key-value pairs.
    ///     content: The request payload (any JSON-serializable object).
    #[new]
    fn new(headers: &Bound<'_, PyDict>, content: &Bound<'_, PyAny>) -> PyResult<Self> {
        let headers_json = py_to_json(headers.as_any())?;
        let headers_map = match headers_json {
            serde_json::Value::Object(m) => m,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "headers must be a dict",
                ))
            }
        };
        let content_json = py_to_json(content)?;
        Ok(Self {
            inner: core_types::LLMRequest {
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
// LLMResponse
// ---------------------------------------------------------------------------

/// An opaque response structure representing an LLM API response.
///
/// Properties:
///     data (Any): The response payload.
#[pyclass(name = "LLMResponse", from_py_object)]
#[derive(Clone)]
pub struct PyLLMResponse {
    pub inner: core_types::LLMResponse,
}

#[pymethods]
impl PyLLMResponse {
    /// Create a new LLMResponse.
    ///
    /// Args:
    ///     data: The response payload (any JSON-serializable object).
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let data_json = py_to_json(data)?;
        Ok(Self {
            inner: core_types::LLMResponse { data: data_json },
        })
    }

    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.data)
    }

    fn __repr__(&self) -> String {
        "LLMResponse(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

/// A lifecycle event emitted to registered subscribers.
///
/// Properties:
///     parent_uuid (str | None): Parent scope/handle UUID.
///     uuid (str): UUID of the entity that produced this event.
///     timestamp (str): ISO 8601 UTC timestamp.
///     name (str | None): Name of the source entity.
///     data (Any | None): Application-specific data snapshot.
///     metadata (Any | None): Metadata snapshot.
///     event_type (EventType): Start, End, or Mark.
///     scope_type (ScopeType | None): Scope type of the source entity.
#[pyclass(name = "Event", skip_from_py_object)]
#[derive(Clone)]
pub struct PyEvent {
    pub inner: core_types::Event,
}

#[pymethods]
impl PyEvent {
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
    fn name(&self) -> Option<String> {
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
    fn event_type(&self) -> PyEventType {
        self.inner.event_type.into()
    }

    #[getter]
    fn scope_type(&self) -> Option<PyScopeType> {
        self.inner.scope_type.map(|st| st.into())
    }

    #[getter]
    fn input(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.input)
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
    fn tool_call_id(&self) -> Option<String> {
        self.inner.tool_call_id.clone()
    }

    #[getter]
    fn root_uuid(&self) -> Option<String> {
        self.inner.root_uuid.map(|u| u.to_string())
    }

    fn __repr__(&self) -> String {
        format!(
            "Event(name={:?}, event_type={:?}, uuid='{}')",
            self.inner.name, self.inner.event_type, self.inner.uuid
        )
    }
}

impl From<core_types::Event> for PyEvent {
    fn from(e: core_types::Event) -> Self {
        Self { inner: e }
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
/// Example::
///
///     exporter = AtifExporter("session-1", "my-agent", "1.0.0", model_name="gpt-4")
///     exporter.register("atif")
///     # ... run agent ...
///     trajectory = exporter.export()
///     exporter.deregister("atif")
#[pyclass(name = "AtifExporter")]
pub struct PyAtifExporter {
    inner: nvmagic_core::atif::AtifExporter,
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
        let agent_info = nvmagic_core::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: tool_defs,
            extra: extra_json,
        };
        Ok(Self {
            inner: nvmagic_core::atif::AtifExporter::new(session_id, agent_info),
        })
    }

    /// Register this exporter as an event subscriber with the given name.
    fn register(&self, name: String) -> PyResult<()> {
        let subscriber = self.inner.subscriber();
        nvmagic_core::nvmagic_register_subscriber(&name, subscriber)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Deregister the event subscriber with the given name.
    ///
    /// Returns ``True`` if a subscriber with that name was found and removed.
    fn deregister(&self, name: String) -> PyResult<bool> {
        nvmagic_core::nvmagic_deregister_subscriber(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Export the collected events as an ATIF trajectory dict.
    ///
    /// Args:
    ///     root_uuid: If provided, only events matching this root UUID are included.
    ///
    /// Returns:
    ///     A dict representing the ATIF trajectory.
    #[pyo3(signature = (root_uuid=None))]
    fn export(&self, py: Python<'_>, root_uuid: Option<String>) -> PyResult<Py<PyAny>> {
        let uuid = match root_uuid {
            Some(s) => Some(uuid::Uuid::parse_str(&s).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid UUID: {e}"))
            })?),
            None => None,
        };
        let trajectory = self.inner.export(uuid);
        let value = serde_json::to_value(&trajectory).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {e}"))
        })?;
        json_to_py(py, &value)
    }

    /// Export the collected events as a JSON string.
    ///
    /// Args:
    ///     root_uuid: If provided, only events matching this root UUID are included.
    ///
    /// Returns:
    ///     A JSON string representing the ATIF trajectory.
    #[pyo3(signature = (root_uuid=None))]
    fn export_json(&self, root_uuid: Option<String>) -> PyResult<String> {
        let uuid = match root_uuid {
            Some(s) => Some(uuid::Uuid::parse_str(&s).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid UUID: {e}"))
            })?),
            None => None,
        };
        let trajectory = self.inner.export(uuid);
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
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScopeStack>()?;
    m.add_class::<PyLlmStream>()?;
    m.add_class::<PyScopeAttributes>()?;
    m.add_class::<PyToolAttributes>()?;
    m.add_class::<PyLLMAttributes>()?;
    m.add_class::<PyScopeType>()?;
    m.add_class::<PyEventType>()?;
    m.add_class::<PyScopeHandle>()?;
    m.add_class::<PyToolHandle>()?;
    m.add_class::<PyLLMHandle>()?;
    m.add_class::<PyLLMRequest>()?;
    m.add_class::<PyLLMResponse>()?;
    m.add_class::<PyEvent>()?;
    m.add_class::<PyAtifExporter>()?;
    Ok(())
}
