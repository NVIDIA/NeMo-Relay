// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
    pub(crate) fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub(crate) fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
    pub(crate) fn __repr__(&self) -> String {
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
    pub(crate) fn new(value: u32) -> Self {
        Self {
            inner: ScopeAttributes::from_bits_truncate(value),
        }
    }

    #[classattr]
    pub(crate) const PARALLEL: u32 = ScopeAttributes::PARALLEL.bits();

    #[classattr]
    pub(crate) const RELOCATABLE: u32 = ScopeAttributes::RELOCATABLE.bits();

    #[getter]
    pub(crate) fn is_parallel(&self) -> bool {
        self.inner.contains(ScopeAttributes::PARALLEL)
    }

    #[getter]
    pub(crate) fn is_relocatable(&self) -> bool {
        self.inner.contains(ScopeAttributes::RELOCATABLE)
    }

    pub(crate) fn __or__(&self, other: &PyScopeAttributes) -> PyScopeAttributes {
        PyScopeAttributes {
            inner: self.inner | other.inner,
        }
    }

    pub(crate) fn __and__(&self, other: &PyScopeAttributes) -> PyScopeAttributes {
        PyScopeAttributes {
            inner: self.inner & other.inner,
        }
    }

    pub(crate) fn __repr__(&self) -> String {
        format!("ScopeAttributes({:?})", self.inner)
    }

    #[getter]
    pub(crate) fn value(&self) -> u32 {
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
    pub(crate) fn new(value: u32) -> Self {
        Self {
            inner: ToolAttributes::from_bits_truncate(value),
        }
    }

    #[classattr]
    pub(crate) const LOCAL: u32 = ToolAttributes::LOCAL.bits();

    #[getter]
    pub(crate) fn is_local(&self) -> bool {
        self.inner.contains(ToolAttributes::LOCAL)
    }

    pub(crate) fn __or__(&self, other: &PyToolAttributes) -> PyToolAttributes {
        PyToolAttributes {
            inner: self.inner | other.inner,
        }
    }

    pub(crate) fn __and__(&self, other: &PyToolAttributes) -> PyToolAttributes {
        PyToolAttributes {
            inner: self.inner & other.inner,
        }
    }

    pub(crate) fn __repr__(&self) -> String {
        format!("ToolAttributes({:?})", self.inner)
    }

    #[getter]
    pub(crate) fn value(&self) -> u32 {
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
    pub(crate) fn new(value: u32) -> Self {
        Self {
            inner: LLMAttributes::from_bits_truncate(value),
        }
    }

    #[classattr]
    pub(crate) const STATELESS: u32 = LLMAttributes::STATELESS.bits();

    #[classattr]
    pub(crate) const STREAMING: u32 = LLMAttributes::STREAMING.bits();

    #[getter]
    pub(crate) fn is_stateless(&self) -> bool {
        self.inner.contains(LLMAttributes::STATELESS)
    }

    #[getter]
    pub(crate) fn is_streaming(&self) -> bool {
        self.inner.contains(LLMAttributes::STREAMING)
    }

    pub(crate) fn __or__(&self, other: &PyLLMAttributes) -> PyLLMAttributes {
        PyLLMAttributes {
            inner: self.inner | other.inner,
        }
    }

    pub(crate) fn __and__(&self, other: &PyLLMAttributes) -> PyLLMAttributes {
        PyLLMAttributes {
            inner: self.inner & other.inner,
        }
    }

    pub(crate) fn __repr__(&self) -> String {
        format!("LLMAttributes({:?})", self.inner)
    }

    #[getter]
    pub(crate) fn value(&self) -> u32 {
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
    pub(crate) fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub(crate) fn scope_type(&self) -> PyScopeType {
        self.inner.scope_type.into()
    }

    #[getter]
    pub(crate) fn attributes(&self) -> PyScopeAttributes {
        PyScopeAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }

    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }

    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }

    pub(crate) fn __repr__(&self) -> String {
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
    pub(crate) fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub(crate) fn attributes(&self) -> PyToolAttributes {
        PyToolAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }

    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }

    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }

    pub(crate) fn __repr__(&self) -> String {
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
    pub(crate) fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub(crate) fn attributes(&self) -> PyLLMAttributes {
        PyLLMAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }

    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }

    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }

    pub(crate) fn __repr__(&self) -> String {
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
    pub(crate) fn new(headers: &Bound<'_, PyAny>, content: &Bound<'_, PyAny>) -> PyResult<Self> {
        let headers_json = py_to_json(headers)?;
        let headers_map = match headers_json {
            serde_json::Value::Object(m) => m,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "not an instance of 'dict'",
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
    pub(crate) fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &serde_json::Value::Object(self.inner.headers.clone()))
    }

    #[getter]
    pub(crate) fn content(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.content)
    }

    pub(crate) fn __repr__(&self) -> String {
        "LLMRequest(...)".to_string()
    }
}
