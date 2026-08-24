// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;

use super::codecs::{PyLlmSanitizeRequestCodec, PyLlmSanitizeResponseCodec};
use super::{
    AnnotatedLLMRequest, Bound, CoreScopeType, FlowResult, LlmAttributes, LlmCodecIdentity,
    LlmHandle, LlmRequest, PyAnnotatedLLMRequest, PyAny, PyErr, PyRef, PyResult, Python,
    ScopeAttributes, ScopeHandle, ScopeStackHandle, ToolAttributes, ToolHandle, json_to_py,
    opt_json_to_py, py_to_json,
};
use nemo_relay::api::event::{
    CategoryProfile, DataSchema, EventCategory, LogSeverity, MetricKind, MetricMeasurement,
    MetricValueType, PendingMarkSpec,
};
use nemo_relay::api::llm::LlmRequestInterceptOutcome;
use nemo_relay::api::runtime::subscriber_dispatcher::PublicationBuffer;
use nemo_relay::api::runtime::{
    LlmSanitizeRequestContext, LlmSanitizeResponseContext, PropagationContext,
    ThreadScopeStackBinding,
};
use nemo_relay::api::tool::{ToolExecutionInterceptOutcome, ToolExecutionResult};

pub(crate) const TOOL_EXECUTION_INTERCEPT_RESULT_ERROR: &str =
    "ToolExecutionInterceptOutcome result must be the downstream payload, not ToolExecutionResult";

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

/// Structured per-call context delivered to LLM request sanitizer callbacks.
#[pyclass(name = "LlmSanitizeRequestContext", frozen)]
pub struct PyLlmSanitizeRequestContext {
    pub(crate) inner: LlmSanitizeRequestContext,
}

#[pymethods]
impl PyLlmSanitizeRequestContext {
    /// The active codec identity for this request or response payload.
    #[getter]
    fn codec(&self) -> PyLlmCodecIdentity {
        PyLlmCodecIdentity {
            inner: self.inner.codec().clone(),
        }
    }

    /// Resolve the active request codec.
    fn resolve_codec(&self) -> Option<PyLlmSanitizeRequestCodec> {
        self.inner
            .resolve_codec()
            .map(|inner| PyLlmSanitizeRequestCodec { inner })
    }
}

/// Structured per-call context delivered to LLM response sanitizer callbacks.
#[pyclass(name = "LlmSanitizeResponseContext", frozen)]
pub struct PyLlmSanitizeResponseContext {
    pub(crate) inner: LlmSanitizeResponseContext,
}

#[pymethods]
impl PyLlmSanitizeResponseContext {
    /// The active codec identity for this request or response payload.
    #[getter]
    fn codec(&self) -> PyLlmCodecIdentity {
        PyLlmCodecIdentity {
            inner: self.inner.codec().clone(),
        }
    }

    /// Resolve the active response codec.
    fn resolve_codec(&self) -> Option<PyLlmSanitizeResponseCodec> {
        self.inner
            .resolve_codec()
            .map(|inner| PyLlmSanitizeResponseCodec { inner })
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
pub struct PyScopeStack {
    pub(crate) inner: ScopeStackHandle,
    pub(crate) publication_buffer: Option<PublicationBuffer>,
}

#[pymethods]
impl PyScopeStack {
    pub(crate) fn __repr__(&self) -> String {
        "<ScopeStack>".to_string()
    }
}

/// Opaque captured native thread binding used to restore a Python scope context.
#[pyclass(name = "_ThreadScopeStackBinding")]
pub struct PyThreadScopeStackBinding {
    pub(crate) inner: ThreadScopeStackBinding,
    pub(crate) publication_buffer: Option<PublicationBuffer>,
}

#[pymethods]
impl PyThreadScopeStackBinding {
    pub(crate) fn __repr__(&self) -> String {
        "<_ThreadScopeStackBinding>".to_string()
    }
}

/// Transport-neutral causal context used to continue Relay work remotely.
#[pyclass(name = "PropagationContext", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPropagationContext {
    pub(crate) inner: PropagationContext,
}

#[pymethods]
impl PyPropagationContext {
    #[new]
    #[pyo3(signature = (parent_uuid, root_uuid=None, version=1))]
    fn new(parent_uuid: &str, root_uuid: Option<&str>, version: u16) -> PyResult<Self> {
        let context = PropagationContext {
            version,
            root_uuid: root_uuid
                .map(uuid::Uuid::parse_str)
                .transpose()
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
            parent_uuid: uuid::Uuid::parse_str(parent_uuid)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
        };
        context
            .validate()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner: context })
    }

    #[getter]
    fn version(&self) -> u16 {
        self.inner.version
    }

    #[getter]
    fn root_uuid(&self) -> Option<String> {
        self.inner.root_uuid.map(|uuid| uuid.to_string())
    }

    #[getter]
    fn parent_uuid(&self) -> String {
        self.inner.parent_uuid.to_string()
    }

    /// Serialize this context to the Relay JSON wire format.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    /// Convert this rooted context to a W3C ``traceparent`` value.
    fn to_traceparent(&self) -> PyResult<String> {
        self.inner
            .to_traceparent()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and validate a context from the Relay JSON wire format.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        PropagationContext::from_json(value)
            .map(|inner| Self { inner })
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
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

/// Severity attached to a mark projected as an OpenTelemetry log.
#[pyclass(name = "LogSeverity", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyLogSeverity {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

impl From<PyLogSeverity> for LogSeverity {
    fn from(value: PyLogSeverity) -> Self {
        match value {
            PyLogSeverity::Trace => Self::Trace,
            PyLogSeverity::Debug => Self::Debug,
            PyLogSeverity::Info => Self::Info,
            PyLogSeverity::Warn => Self::Warn,
            PyLogSeverity::Error => Self::Error,
        }
    }
}

impl From<LogSeverity> for PyLogSeverity {
    fn from(value: LogSeverity) -> Self {
        match value {
            LogSeverity::Trace => Self::Trace,
            LogSeverity::Debug => Self::Debug,
            LogSeverity::Info => Self::Info,
            LogSeverity::Warn => Self::Warn,
            LogSeverity::Error => Self::Error,
        }
    }
}

/// OpenTelemetry instrument kind recorded by a metric mark.
#[pyclass(name = "MetricKind", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyMetricKind {
    Counter = 0,
    UpDownCounter = 1,
    Gauge = 2,
    Histogram = 3,
}

impl From<PyMetricKind> for MetricKind {
    fn from(value: PyMetricKind) -> Self {
        match value {
            PyMetricKind::Counter => Self::Counter,
            PyMetricKind::UpDownCounter => Self::UpDownCounter,
            PyMetricKind::Gauge => Self::Gauge,
            PyMetricKind::Histogram => Self::Histogram,
        }
    }
}

impl From<MetricKind> for PyMetricKind {
    fn from(value: MetricKind) -> Self {
        match value {
            MetricKind::Counter => Self::Counter,
            MetricKind::UpDownCounter => Self::UpDownCounter,
            MetricKind::Gauge => Self::Gauge,
            MetricKind::Histogram => Self::Histogram,
        }
    }
}

/// Numeric representation used by a metric measurement.
#[pyclass(name = "MetricValueType", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyMetricValueType {
    U64 = 0,
    I64 = 1,
    F64 = 2,
}

impl From<PyMetricValueType> for MetricValueType {
    fn from(value: PyMetricValueType) -> Self {
        match value {
            PyMetricValueType::U64 => Self::U64,
            PyMetricValueType::I64 => Self::I64,
            PyMetricValueType::F64 => Self::F64,
        }
    }
}

impl From<MetricValueType> for PyMetricValueType {
    fn from(value: MetricValueType) -> Self {
        match value {
            MetricValueType::U64 => Self::U64,
            MetricValueType::I64 => Self::I64,
            MetricValueType::F64 => Self::F64,
        }
    }
}

/// One typed recording operation emitted by ``scope.metric``.
#[pyclass(name = "MetricMeasurement", from_py_object)]
#[derive(Clone)]
pub struct PyMetricMeasurement {
    pub(crate) inner: MetricMeasurement,
}

#[pymethods]
impl PyMetricMeasurement {
    #[new]
    #[pyo3(signature = (name, kind, value_type, value, unit=None, description=None, attributes=None, boundaries=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        kind: PyMetricKind,
        value_type: PyMetricValueType,
        value: &Bound<'_, PyAny>,
        unit: Option<String>,
        description: Option<String>,
        attributes: Option<&Bound<'_, PyAny>>,
        boundaries: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MetricMeasurement {
                name,
                kind: kind.into(),
                value_type: value_type.into(),
                value: py_to_json(value)?,
                unit,
                description,
                attributes: attributes.map(py_to_json).transpose()?,
                boundaries,
            },
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn kind(&self) -> PyMetricKind {
        self.inner.kind.into()
    }

    #[getter]
    fn value_type(&self) -> PyMetricValueType {
        self.inner.value_type.into()
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.value)
    }

    #[getter]
    fn unit(&self) -> Option<&str> {
        self.inner.unit.as_deref()
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.inner.description.as_deref()
    }

    #[getter]
    fn attributes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.attributes)
    }

    #[getter]
    fn boundaries(&self) -> Option<Vec<f64>> {
        self.inner.boundaries.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "<MetricMeasurement name={:?} kind={} value_type={}>",
            self.inner.name, self.inner.kind, self.inner.value_type
        )
    }
}

fn parse_data_schema(value: &Bound<'_, PyAny>) -> PyResult<DataSchema> {
    serde_json::from_value(py_to_json(value)?).map_err(|error| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "data_schema must contain string name and version fields: {error}"
        ))
    })
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
    #[pyo3(signature = (name, category=None, category_profile=None, data=None, metadata=None, data_schema=None, severity=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        category: Option<String>,
        category_profile: Option<&Bound<'_, PyAny>>,
        data: Option<&Bound<'_, PyAny>>,
        metadata: Option<&Bound<'_, PyAny>>,
        data_schema: Option<&Bound<'_, PyAny>>,
        severity: Option<PyLogSeverity>,
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
                data_schema: data_schema.map(parse_data_schema).transpose()?,
                metadata: metadata.map(py_to_json).transpose()?,
                severity: severity.map(Into::into),
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
    fn data_schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = self
            .inner
            .data_schema
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        opt_json_to_py(py, &value)
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }

    #[getter]
    fn severity(&self) -> Option<PyLogSeverity> {
        self.inner.severity.map(Into::into)
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

/// Canonical application-visible result returned by Python tool producers.
#[pyclass(name = "ToolExecutionResult", from_py_object, generic)]
#[derive(Clone)]
pub struct PyToolExecutionResult {
    result: Arc<Py<PyAny>>,
    annotation: Option<Arc<Py<PyAny>>>,
}

#[pymethods]
impl PyToolExecutionResult {
    #[new]
    #[pyo3(signature = (result, annotation=None))]
    fn new(result: &Bound<'_, PyAny>, annotation: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            result: Arc::new(result.clone().unbind()),
            annotation: annotation
                .filter(|value| !value.is_none())
                .map(|value| Arc::new(value.clone().unbind())),
        })
    }

    #[getter]
    fn result(&self, py: Python<'_>) -> Py<PyAny> {
        self.result.as_ref().clone_ref(py)
    }

    #[getter]
    fn annotation(&self, py: Python<'_>) -> Py<PyAny> {
        self.annotation
            .as_ref()
            .map_or_else(|| py.None(), |value| value.as_ref().clone_ref(py))
    }
}

impl PyToolExecutionResult {
    pub(crate) fn from_inner(py: Python<'_>, inner: ToolExecutionResult) -> PyResult<Self> {
        Ok(Self {
            result: Arc::new(json_to_py(py, &inner.result)?),
            annotation: inner
                .annotation
                .as_ref()
                .map(|value| json_to_py(py, value))
                .transpose()?
                .map(Arc::new),
        })
    }

    pub(crate) fn to_inner(&self, py: Python<'_>) -> PyResult<ToolExecutionResult> {
        Ok(ToolExecutionResult {
            result: py_to_json(self.result.as_ref().bind(py))?,
            annotation: self
                .annotation
                .as_ref()
                .map(|value| py_to_json(value.as_ref().bind(py)))
                .transpose()?,
        })
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
    #[pyo3(signature = (result, pending_marks=Vec::new(), *, annotation=None))]
    fn new(
        result: &Bound<'_, PyAny>,
        pending_marks: Vec<PyPendingMarkSpec>,
        annotation: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if result.is_instance_of::<PyToolExecutionResult>() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                TOOL_EXECUTION_INTERCEPT_RESULT_ERROR,
            ));
        }
        Ok(Self {
            inner: ToolExecutionInterceptOutcome {
                result: py_to_json(result)?,
                annotation: crate::convert::opt_py_to_json(annotation)?,
                pending_marks: pending_marks.into_iter().map(|value| value.inner).collect(),
            },
        })
    }

    #[getter]
    fn result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.result)
    }

    #[getter]
    fn annotation(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.annotation)
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
