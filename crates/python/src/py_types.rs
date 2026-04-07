// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing type wrappers for Nexus core types.
//!
//! Each type wraps its corresponding `nvidia_nat_nexus_core::types` struct and exposes
//! properties via `#[getter]`. Doc comments on `#[pyclass]` and `#[pymethods]`
//! become Python `help()` output.

use std::collections::HashMap;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use nvidia_nat_nexus_core::types as core_types;

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
    pub receiver: tokio::sync::Mutex<
        tokio::sync::mpsc::Receiver<nvidia_nat_nexus_core::Result<serde_json::Value>>,
    >,
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
                tokio::sync::mpsc::Receiver<nvidia_nat_nexus_core::Result<serde_json::Value>>,
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
pub struct PyScopeStack(pub nvidia_nat_nexus_core::ScopeStackHandle);

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
    inner: nvidia_nat_nexus_core::atif::AtifExporter,
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
        let agent_info = nvidia_nat_nexus_core::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: tool_defs,
            extra: extra_json,
        };
        Ok(Self {
            inner: nvidia_nat_nexus_core::atif::AtifExporter::new(session_id, agent_info),
        })
    }

    /// Register this exporter as an event subscriber with the given name.
    fn register(&self, name: String) -> PyResult<()> {
        let subscriber = self.inner.subscriber();
        nvidia_nat_nexus_core::nat_nexus_register_subscriber(&name, subscriber)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Deregister the event subscriber with the given name.
    ///
    /// Returns ``True`` if a subscriber with that name was found and removed.
    fn deregister(&self, name: String) -> PyResult<bool> {
        nvidia_nat_nexus_core::nat_nexus_deregister_subscriber(&name)
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
// OpenTelemetry subscriber
// ---------------------------------------------------------------------------

/// Mutable configuration object for the OpenTelemetry subscriber.
///
/// Create the config, update fields as needed, then pass it to
/// ``OpenTelemetrySubscriber(config)``.
///
/// Example::
///
///     config = OpenTelemetryConfig()
///     config.endpoint = "http://localhost:4318/v1/traces"
///     config.service_name = "demo-agent"
///     config.headers = {"authorization": "Bearer token"}
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
    fn to_rust_config(&self) -> PyResult<nvidia_nat_nexus_otel::OpenTelemetryConfig> {
        let mut config = match self.transport.as_str() {
            "http_binary" => {
                nvidia_nat_nexus_otel::OpenTelemetryConfig::http_binary(self.service_name.clone())
            }
            "grpc" => nvidia_nat_nexus_otel::OpenTelemetryConfig::grpc(self.service_name.clone()),
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
            service_name: "nat-nexus".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "nvidia-nat-nexus-otel".to_string(),
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
    inner: nvidia_nat_nexus_otel::OpenTelemetrySubscriber,
}

#[pymethods]
impl PyOpenTelemetrySubscriber {
    #[new]
    fn new(config: PyRef<'_, PyOpenTelemetryConfig>) -> PyResult<Self> {
        let inner = nvidia_nat_nexus_otel::OpenTelemetrySubscriber::new(config.to_rust_config()?)
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
/// Example::
///
///     config = OpenInferenceConfig()
///     config.endpoint = "http://localhost:4318/v1/traces"
///     config.service_name = "demo-agent"
///     config.headers = {"authorization": "Bearer token"}
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
    fn to_rust_config(&self) -> PyResult<nvidia_nat_nexus_openinference::OpenInferenceConfig> {
        let transport = match self.transport.as_str() {
            "http_binary" => nvidia_nat_nexus_openinference::OtlpTransport::HttpBinary,
            "grpc" => nvidia_nat_nexus_openinference::OtlpTransport::Grpc,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "transport must be 'http_binary' or 'grpc', got {other:?}"
                )));
            }
        };

        let mut config = nvidia_nat_nexus_openinference::OpenInferenceConfig::new()
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
            service_name: "nat-nexus".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "nvidia-nat-nexus-openinference".to_string(),
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
    inner: nvidia_nat_nexus_openinference::OpenInferenceSubscriber,
}

#[pymethods]
impl PyOpenInferenceSubscriber {
    #[new]
    fn new(config: PyRef<'_, PyOpenInferenceConfig>) -> PyResult<Self> {
        let inner =
            nvidia_nat_nexus_openinference::OpenInferenceSubscriber::new(config.to_rust_config()?)
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
    m.add_class::<PyEvent>()?;
    m.add_class::<PyAtifExporter>()?;
    m.add_class::<PyOpenTelemetryConfig>()?;
    m.add_class::<PyOpenTelemetrySubscriber>()?;
    m.add_class::<PyOpenInferenceConfig>()?;
    m.add_class::<PyOpenInferenceSubscriber>()?;
    Ok(())
}

#[cfg(test)]
#[path = "py_types_coverage_tests.rs"]
mod coverage_tests;
