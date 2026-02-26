//! Python-facing type wrappers for NVAgentRT core types.
//!
//! Each type wraps its corresponding `nvagentrt_core::types` struct and exposes
//! properties via `#[getter]`. Doc comments on `#[pyclass]` and `#[pymethods]`
//! become Python `help()` output.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use nvagentrt_core::types as core_types;

use crate::convert::{json_to_py, opt_json_to_py, py_to_json};

// ---------------------------------------------------------------------------
// SseEvent
// ---------------------------------------------------------------------------

/// A Server-Sent Events (SSE) event used in streaming LLM responses.
///
/// Attributes:
///     data (str): The event payload.
///     event (str | None): The event type.
///     id (str | None): The event ID.
///     retry (int | None): Reconnection time in milliseconds.
#[pyclass(name = "SseEvent", from_py_object)]
#[derive(Clone)]
pub struct PySseEvent {
    pub inner: core_types::SseEvent,
}

#[pymethods]
impl PySseEvent {
    /// Create a new SseEvent.
    ///
    /// Args:
    ///     data: The event payload string.
    ///     event: Optional event type name.
    ///     id: Optional event ID.
    ///     retry: Optional reconnection time in milliseconds.
    #[new]
    #[pyo3(signature = (data, event=None, id=None, retry=None))]
    fn new(data: String, event: Option<String>, id: Option<String>, retry: Option<u64>) -> Self {
        Self {
            inner: core_types::SseEvent {
                event,
                data,
                id,
                retry,
            },
        }
    }

    #[getter]
    fn event(&self) -> Option<String> {
        self.inner.event.clone()
    }

    #[getter]
    fn data(&self) -> String {
        self.inner.data.clone()
    }

    #[getter]
    fn id(&self) -> Option<String> {
        self.inner.id.clone()
    }

    #[getter]
    fn retry(&self) -> Option<u64> {
        self.inner.retry
    }

    fn __repr__(&self) -> String {
        format!(
            "SseEvent(data={:?}, event={:?}, id={:?}, retry={:?})",
            self.inner.data, self.inner.event, self.inner.id, self.inner.retry
        )
    }
}

impl From<core_types::SseEvent> for PySseEvent {
    fn from(e: core_types::SseEvent) -> Self {
        Self { inner: e }
    }
}

// ---------------------------------------------------------------------------
// LlmStream (async iterator)
// ---------------------------------------------------------------------------

/// An async iterator that yields SSE text chunks from a streaming LLM response.
///
/// Use ``async for chunk in stream:`` to consume chunks. Each chunk is a raw
/// SSE text string. The stream automatically emits an End lifecycle event
/// when exhausted.
#[pyclass(name = "LlmStream")]
pub struct PyLlmStream {
    pub receiver: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<nvagentrt_core::Result<String>>>,
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
                tokio::sync::mpsc::Receiver<nvagentrt_core::Result<String>>,
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
                Some(Ok(text)) => Ok(text),
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
pub struct PyScopeStack(pub nvagentrt_core::ScopeStackHandle);

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

/// An HTTP-like request representing an outgoing LLM API call.
///
/// Properties:
///     method (str): HTTP method (typically "POST").
///     url (str): The LLM API endpoint URL.
///     headers (dict): HTTP headers.
///     body (Any): The request body (typically a dict).
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
    ///     method: HTTP method (e.g., "POST").
    ///     url: The LLM API endpoint URL.
    ///     headers: A dict of HTTP headers.
    ///     body: The request body (any JSON-serializable object).
    #[new]
    fn new(
        method: String,
        url: String,
        headers: &Bound<'_, PyDict>,
        body: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let headers_json = py_to_json(headers.as_any())?;
        let headers_map = match headers_json {
            serde_json::Value::Object(m) => m,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "headers must be a dict",
                ))
            }
        };
        let body_json = py_to_json(body)?;
        Ok(Self {
            inner: core_types::LLMRequest {
                method,
                url,
                headers: headers_map,
                body: body_json,
            },
        })
    }

    #[getter]
    fn method(&self) -> String {
        self.inner.method.clone()
    }

    #[getter]
    fn url(&self) -> String {
        self.inner.url.clone()
    }

    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &serde_json::Value::Object(self.inner.headers.clone()))
    }

    #[getter]
    fn body(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.body)
    }

    fn __repr__(&self) -> String {
        format!(
            "LLMRequest(method='{}', url='{}')",
            self.inner.method, self.inner.url
        )
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
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScopeStack>()?;
    m.add_class::<PySseEvent>()?;
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
    Ok(())
}
