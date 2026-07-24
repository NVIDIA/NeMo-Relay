// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;

use super::{
    AnnotatedLLMRequest, Bound, CoreScopeType, FlowResult, LlmAttributes, LlmCodecIdentity,
    LlmHandle, LlmRequest, PyAnnotatedLLMRequest, PyAny, PyErr, PyRef, PyResult, Python,
    ScopeAttributes, ScopeHandle, ScopeStackHandle, ToolAttributes, ToolHandle, json_to_py,
    opt_json_to_py, py_to_json,
};
use nemo_relay::api::event::{CategoryProfile, EventCategory, PendingMarkSpec};
use nemo_relay::api::llm::LlmRequestInterceptOutcome;
use nemo_relay::api::tool::ToolExecutionInterceptOutcome;

/// Structured identity of the codec active during LLM sanitization.
#[pyclass(name = "LlmCodecIdentity", frozen)]
pub struct PyLlmCodecIdentity {
    pub(crate) inner: LlmCodecIdentity,
}

#[pymethods]
impl PyLlmCodecIdentity {
    /// Identity variant: ``none``, ``builtin``, ``runtime``, or ``opaque``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            LlmCodecIdentity::None => "none",
            LlmCodecIdentity::BuiltIn(_) => "builtin",
            LlmCodecIdentity::Runtime(_) => "runtime",
            LlmCodecIdentity::Opaque => "opaque",
        }
    }

    /// Stable built-in or runtime codec ID, when this identity has one.
    #[getter]
    fn id(&self) -> Option<String> {
        match &self.inner {
            LlmCodecIdentity::BuiltIn(codec) => Some(codec.id().to_owned()),
            LlmCodecIdentity::Runtime(id) => Some(id.clone()),
            LlmCodecIdentity::None | LlmCodecIdentity::Opaque => None,
        }
    }
}

/// Structured per-call context delivered to LLM sanitizer callbacks.
#[pyclass(name = "LlmSanitizeContext", frozen)]
pub struct PyLlmSanitizeContext {
    pub(crate) codec: LlmCodecIdentity,
}

#[pymethods]
impl PyLlmSanitizeContext {
    /// The active codec identity for this request or response payload.
    #[getter]
    fn codec(&self) -> PyLlmCodecIdentity {
        PyLlmCodecIdentity {
            inner: self.codec.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// LlmStream (async iterator)
// ---------------------------------------------------------------------------

/// An async iterator that yields parsed JSON chunks from a streaming LLM response.
///
/// Use ``async for chunk in stream:`` to consume chunks. Each chunk is a
/// Python object (converted from JSON). The stream automatically emits an
/// End lifecycle event when exhausted. When stopping early, await
/// ``stream.aclose()`` to wait for producer cleanup and emit an interrupted
/// End event.
#[pyclass(name = "LlmStream")]
pub struct PyLlmStream {
    pub receiver:
        Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<FlowResult<serde_json::Value>>>>,
    pub cancel: tokio::sync::watch::Sender<bool>,
    pub closed: tokio::sync::watch::Receiver<Option<Result<(), String>>>,
}

#[pymethods]
impl PyLlmStream {
    pub(crate) fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub(crate) fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let receiver = Arc::clone(&self.receiver);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = receiver.lock().await;
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

    /// Stop forwarding chunks, wait for producer cleanup, and release the native stream.
    ///
    /// This operation is idempotent and raises when producer cleanup fails.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let cancel = self.cancel.clone();
        let mut closed = self.closed.clone();
        let receiver = Arc::clone(&self.receiver);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            cancel.send_replace(true);
            while closed.borrow().is_none() {
                closed.changed().await.map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "stream close task ended before releasing the native stream",
                    )
                })?;
            }
            let result = closed.borrow().clone().expect("close state checked above");
            let mut receiver = receiver.lock().await;
            receiver.close();
            while receiver.try_recv().is_ok() {}
            result.map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
        })
    }
}

impl Drop for PyLlmStream {
    fn drop(&mut self) {
        self.cancel.send_replace(true);
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
///     REMOTE (int): The tool executes remotely.
///
/// Properties:
///     is_remote (bool): Whether REMOTE is set.
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
    pub(crate) const REMOTE: u32 = ToolAttributes::REMOTE.bits();

    #[getter]
    pub(crate) fn is_remote(&self) -> bool {
        self.inner.contains(ToolAttributes::REMOTE)
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
// LlmAttributes (bitflag wrapper)
// ---------------------------------------------------------------------------

/// Bitflag attributes for LLM handles.
///
/// Class attributes:
///     STATEFUL (int): The LLM call is stateful.
///     STREAMING (int): The LLM call uses streaming responses.
///
/// Properties:
///     is_stateful (bool): Whether STATEFUL is set.
///     is_streaming (bool): Whether STREAMING is set.
///     value (int): Raw bitflag value.
#[pyclass(name = "LLMAttributes", from_py_object)]
#[derive(Clone)]
pub struct PyLLMAttributes {
    pub inner: LlmAttributes,
}

#[pymethods]
impl PyLLMAttributes {
    #[new]
    #[pyo3(signature = (value: "int"=0), text_signature = "(value: int = 0)")]
    pub(crate) fn new(value: u32) -> Self {
        Self {
            inner: LlmAttributes::from_bits_truncate(value),
        }
    }

    #[classattr]
    pub(crate) const STATEFUL: u32 = LlmAttributes::STATEFUL.bits();

    #[classattr]
    pub(crate) const STREAMING: u32 = LlmAttributes::STREAMING.bits();

    #[getter]
    pub(crate) fn is_stateful(&self) -> bool {
        self.inner.contains(LlmAttributes::STATEFUL)
    }

    #[getter]
    pub(crate) fn is_streaming(&self) -> bool {
        self.inner.contains(LlmAttributes::STREAMING)
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
    pub inner: LlmHandle,
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

impl From<LlmHandle> for PyLLMHandle {
    fn from(h: LlmHandle) -> Self {
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
    pub inner: LlmRequest,
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
            inner: LlmRequest {
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

/// A mark to emit immediately after the managed LLM start event.
#[pyclass(name = "PendingMarkSpec", from_py_object)]
#[derive(Clone)]
pub struct PyPendingMarkSpec {
    pub inner: PendingMarkSpec,
}

#[pymethods]
impl PyPendingMarkSpec {
    #[new]
    #[pyo3(signature = (name, category=None, category_profile=None, data=None, metadata=None))]
    fn new(
        name: String,
        category: Option<String>,
        category_profile: Option<&Bound<'_, PyAny>>,
        data: Option<&Bound<'_, PyAny>>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let category = category
            .map(|value| serde_json::from_value::<EventCategory>(serde_json::Value::String(value)))
            .transpose()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let category_profile = category_profile
            .map(py_to_json)
            .transpose()?
            .map(serde_json::from_value::<CategoryProfile>)
            .transpose()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner: PendingMarkSpec {
                name,
                category,
                category_profile,
                data: data.map(py_to_json).transpose()?,
                metadata: metadata.map(py_to_json).transpose()?,
            },
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn category(&self) -> Option<String> {
        self.inner
            .category
            .as_ref()
            .and_then(|value| serde_json::to_value(value).ok())
            .and_then(|value| value.as_str().map(str::to_owned))
    }

    #[getter]
    fn category_profile(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(
            py,
            &self
                .inner
                .category_profile
                .as_ref()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?,
        )
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

/// Canonical result returned by Python LLM request intercepts.
#[pyclass(name = "LLMRequestInterceptOutcome", from_py_object)]
#[derive(Clone)]
pub struct PyLLMRequestInterceptOutcome {
    pub inner: LlmRequestInterceptOutcome,
}

#[pymethods]
impl PyLLMRequestInterceptOutcome {
    #[new]
    #[pyo3(signature = (request, annotated_request=None, pending_marks=Vec::new(), optimization_contributions=None))]
    fn new(
        request: PyLLMRequest,
        annotated_request: Option<PyAnnotatedLLMRequest>,
        pending_marks: Vec<PyPendingMarkSpec>,
        optimization_contributions: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let optimization_contributions = optimization_contributions
            .map(py_to_json)
            .transpose()?
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid optimization_contributions: {error}"
                ))
            })?
            .unwrap_or_default();
        Ok(Self {
            inner: LlmRequestInterceptOutcome {
                request: request.inner,
                annotated_request: annotated_request.map(|value| value.inner),
                pending_marks: pending_marks.into_iter().map(|value| value.inner).collect(),
                optimization_contributions,
            },
        })
    }

    #[getter]
    fn request(&self) -> PyLLMRequest {
        PyLLMRequest {
            inner: self.inner.request.clone(),
        }
    }

    #[getter]
    fn annotated_request(&self) -> Option<PyAnnotatedLLMRequest> {
        self.inner
            .annotated_request
            .clone()
            .map(|inner: AnnotatedLLMRequest| PyAnnotatedLLMRequest { inner })
    }

    #[getter]
    fn pending_marks(&self) -> Vec<PyPendingMarkSpec> {
        self.inner
            .pending_marks
            .iter()
            .cloned()
            .map(|inner| PyPendingMarkSpec { inner })
            .collect()
    }

    #[getter]
    fn optimization_contributions(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &serde_json::to_value(&self.inner.optimization_contributions)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
        )
    }
}

/// Canonical result returned by Python tool execution intercepts.
#[pyclass(name = "ToolExecutionInterceptOutcome", from_py_object)]
#[derive(Clone)]
pub struct PyToolExecutionInterceptOutcome {
    pub inner: ToolExecutionInterceptOutcome,
}

#[pymethods]
impl PyToolExecutionInterceptOutcome {
    #[new]
    #[pyo3(signature = (result, pending_marks=Vec::new()))]
    fn new(result: &Bound<'_, PyAny>, pending_marks: Vec<PyPendingMarkSpec>) -> PyResult<Self> {
        Ok(Self {
            inner: ToolExecutionInterceptOutcome {
                result: py_to_json(result)?,
                pending_marks: pending_marks.into_iter().map(|value| value.inner).collect(),
            },
        })
    }

    #[getter]
    fn result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.result)
    }

    #[getter]
    fn pending_marks(&self) -> Vec<PyPendingMarkSpec> {
        self.inner
            .pending_marks
            .iter()
            .cloned()
            .map(|inner| PyPendingMarkSpec { inner })
            .collect()
    }
}
