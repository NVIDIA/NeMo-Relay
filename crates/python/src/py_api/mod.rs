// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing API functions for the NeMo Relay runtime.
//!
//! Each `#[pyfunction]` here is registered into the `_native` module and
//! delegates to the corresponding function in [`nemo_relay::api`].
//! The Python wrapper modules (`nemo_relay.scope`, `nemo_relay.tools`, etc.)
//! re-export these under shorter, idiomatic names.

use std::sync::Arc;

use nemo_relay::api::llm as core_llm_api;
use nemo_relay::api::llm::LlmAttributes;
use nemo_relay::api::registry as core_registry_api;
use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn};
use nemo_relay::api::runtime::{
    TASK_SCOPE_STACK, create_scope_stack as create_scope_stack_handle,
    current_scope_stack as current_scope_stack_handle, scope_stack_active as scope_stack_is_active,
    set_thread_scope_stack as bind_thread_scope_stack,
    sync_thread_scope_stack as sync_bound_thread_scope_stack, task_scope_top,
};
use nemo_relay::api::scope as core_scope_api;
use nemo_relay::api::scope::ScopeAttributes;
use nemo_relay::api::subscriber as core_subscriber_api;
use nemo_relay::api::tool as core_tool_api;
use nemo_relay::api::tool::ToolAttributes;
use nemo_relay::codec::response::AnnotatedLlmResponse;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::{FlowError, Result as FlowResult};
use pyo3::prelude::*;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::convert::{json_to_py, opt_py_to_json, opt_py_to_timestamp, py_to_json};
use crate::py_callable;
use crate::py_types::{
    PyAnnotatedLLMResponse, PyAnthropicMessagesCodec, PyLLMAttributes, PyLLMHandle, PyLLMRequest,
    PyLlmStream, PyOpenAIChatCodec, PyOpenAIResponsesCodec, PyScopeAttributes, PyScopeHandle,
    PyScopeStack, PyScopeType, PyToolAttributes, PyToolHandle,
};

pub(crate) type RustJsonStream =
    std::pin::Pin<Box<dyn tokio_stream::Stream<Item = FlowResult<serde_json::Value>> + Send>>;

/// Convert an [`FlowError`] into a Python `RuntimeError`.
fn to_py_err(e: FlowError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
}

fn py_llm_response_codec(
    response_codec: Option<&Bound<'_, PyAny>>,
) -> Option<Arc<dyn LlmResponseCodec>> {
    response_codec.and_then(|c| -> Option<Arc<dyn LlmResponseCodec>> {
        if c.is_none() {
            return None;
        }
        // Try to extract as a built-in codec first (avoids Python method dispatch overhead)
        if let Ok(builtin) = c.extract::<pyo3::PyRef<'_, PyOpenAIChatCodec>>() {
            return Some(builtin.inner_response_codec.clone());
        }
        if let Ok(builtin) = c.extract::<pyo3::PyRef<'_, PyOpenAIResponsesCodec>>() {
            return Some(builtin.inner_response_codec.clone());
        }
        if let Ok(builtin) = c.extract::<pyo3::PyRef<'_, PyAnthropicMessagesCodec>>() {
            return Some(builtin.inner_response_codec.clone());
        }
        // Fall back to wrapping the Python object as a custom response codec
        Some(Arc::new(py_callable::PyLlmResponseCodecWrapper {
            py_codec: c.clone().unbind(),
        }))
    })
}

fn py_annotated_llm_response(
    annotated_response: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<AnnotatedLlmResponse>>> {
    let Some(annotated_response) = annotated_response else {
        return Ok(None);
    };
    if annotated_response.is_none() {
        return Ok(None);
    }

    if let Ok(response) = annotated_response.cast::<PyAnnotatedLLMResponse>() {
        let response = response.borrow();
        return Ok(Some(Arc::new(response.inner.clone())));
    }

    let value = py_to_json(annotated_response)?;
    serde_json::from_value::<AnnotatedLlmResponse>(value)
        .map(|response| Some(Arc::new(response)))
        .map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid annotated_response: {error}"
            ))
        })
}

pub(crate) async fn forward_stream_to_channel(
    mut stream: RustJsonStream,
    tx: tokio::sync::mpsc::Sender<FlowResult<serde_json::Value>>,
) {
    while let Some(item) = stream.next().await {
        if tx.send(item).await.is_err() {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Scope stack creation
// ---------------------------------------------------------------------------

/// Create a new isolated scope stack with its own root scope.
///
/// Returns:
///     A ``ScopeStack`` that can be used for per-request or per-task isolation.
#[pyfunction]
pub fn create_scope_stack() -> PyScopeStack {
    PyScopeStack(create_scope_stack_handle())
}

/// Bind a ``ScopeStack`` to the current thread's thread-local storage.
///
/// This ensures that subsequent NeMo Relay API calls on this thread use the given
/// scope stack rather than a default one. Primarily useful when propagating
/// scope context into worker threads (e.g. ``ThreadPoolExecutor``).
///
/// Args:
///     stack: The ``ScopeStack`` to bind to the current thread.
#[pyfunction]
pub fn set_thread_scope_stack(stack: &PyScopeStack) {
    bind_thread_scope_stack(stack.0.clone());
}

/// Sync a ``ScopeStack`` to the current thread's Rust thread-local storage
/// **without** marking it as explicitly set.
///
/// This is used internally by ``nemo_relay.get_scope_stack()`` to keep the Rust
/// thread-local in sync with the Python ``contextvars.ContextVar`` without
/// affecting ``scope_stack_active()``.
#[pyfunction]
pub fn sync_thread_scope_stack(stack: &PyScopeStack) {
    sync_bound_thread_scope_stack(stack.0.clone());
}

/// Return whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns ``True`` if ``set_thread_scope_stack`` has been called on the
/// current thread, or the caller is inside a tokio task with a task-local
/// scope stack. Returns ``False`` when only the auto-created default is
/// present.
///
/// .. note::
///     The Python-level ``nemo_relay.scope_stack_active()`` wrapper also
///     checks the ``contextvars.ContextVar`` and should be preferred in
///     Python code. This native function is useful for non-async contexts
///     where ``contextvars`` are not involved.
#[pyfunction]
#[pyo3(name = "scope_stack_active")]
pub fn py_scope_stack_active() -> bool {
    scope_stack_is_active()
}

enum PySubscriptionInner {
    Global(core_subscriber_api::SubscriberHandle),
    Scope(core_subscriber_api::ScopeSubscriberHandle),
}

/// Closeable event subscriber registration returned by ``subscribe`` helpers.
#[pyclass(name = "Subscription")]
pub struct PySubscription {
    inner: Option<PySubscriptionInner>,
}

#[pymethods]
impl PySubscription {
    /// Close this subscription.
    ///
    /// Returns ``True`` when a live subscriber was removed and ``False`` when
    /// the subscription had already been closed or scope cleanup already
    /// removed it.
    fn close(&mut self) -> PyResult<bool> {
        let Some(inner) = self.inner.as_ref() else {
            return Ok(false);
        };
        let removed = match inner {
            PySubscriptionInner::Global(handle) => handle.close(),
            PySubscriptionInner::Scope(handle) => handle.close(),
        }
        .map_err(to_py_err)?;
        self.inner = None;
        Ok(removed)
    }

    /// Enter a Python context manager.
    fn __enter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    /// Exit a Python context manager, closing the subscription.
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let _ = self.close()?;
        Ok(false)
    }
}

impl Drop for PySubscription {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let _ = match inner {
                PySubscriptionInner::Global(handle) => handle.close(),
                PySubscriptionInner::Scope(handle) => handle.close(),
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Return the current scope handle from the task-local scope stack.
///
/// Returns the topmost `ScopeHandle` or raises `RuntimeError` if the
/// scope stack is empty.
#[pyfunction]
#[pyo3(signature = () -> "ScopeHandle", text_signature = "() -> ScopeHandle")]
fn get_handle() -> PyResult<PyScopeHandle> {
    core_scope_api::get_handle()
        .map(PyScopeHandle::from)
        .map_err(to_py_err)
}

/// Push a new child scope onto the scope stack.
///
/// Args:
///     name: Human-readable scope name (e.g. ``"my-agent"``).
///     scope_type: The kind of scope (``ScopeType.Agent``, etc.).
///     handle: Optional parent scope. Defaults to the current top of stack.
///     attributes: Optional bitflags (e.g. ``ScopeAttributes.PARALLEL``).
///     data: Optional JSON-serializable application payload stored on the scope handle.
///     metadata: Optional JSON-serializable metadata recorded on the start event.
///     input: Optional JSON-serializable semantic input payload for the scope start event.
///     timestamp: Optional timezone-aware ``datetime.datetime`` recorded as the handle
///         start time and on the emitted start event. When omitted, the current
///         runtime time is used.
///
/// Returns:
///     The newly created ``ScopeHandle``.
///
/// Raises:
///     RuntimeError: If the scope stack is empty and no parent handle is given.
///     TypeError: If ``timestamp`` is not a ``datetime.datetime``.
///     ValueError: If ``timestamp`` is a naive datetime.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    name: "str",
    scope_type: "ScopeType",
    *,
    handle: "ScopeHandle | None"=None,
    attributes: "ScopeAttributes | None"=None,
	    data: "object | None"=None,
	    metadata: "object | None"=None,
	    input: "object | None"=None,
	    timestamp: "datetime.datetime | None"=None
) -> "ScopeHandle", text_signature = "(name: str, scope_type: ScopeType, *, handle: ScopeHandle | None = None, attributes: ScopeAttributes | None = None, data: object | None = None, metadata: object | None = None, input: object | None = None, timestamp: datetime.datetime | None = None) -> ScopeHandle")]
fn push_scope(
    name: &str,
    scope_type: PyScopeType,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyScopeAttributes>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
    input: Option<&Bound<'_, PyAny>>,
    timestamp: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyScopeHandle> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(ScopeAttributes::empty());
    let d = opt_py_to_json(data)?;
    let meta = opt_py_to_json(metadata)?;
    let input = opt_py_to_json(input)?;
    let timestamp = opt_py_to_timestamp(timestamp)?;
    core_scope_api::push_scope(
        core_scope_api::PushScopeParams::builder()
            .name(name)
            .scope_type(scope_type.into())
            .parent_opt(handle.as_ref().map(|h| &h.inner))
            .attributes(attrs)
            .data_opt(d)
            .metadata_opt(meta)
            .input_opt(input)
            .timestamp_opt(timestamp)
            .build(),
    )
    .map(PyScopeHandle::from)
    .map_err(to_py_err)
}

/// Remove a scope from the stack and emit an ``End`` event.
///
/// Args:
///     handle: The current top-of-stack scope handle returned by ``push``.
///     output: Optional JSON-serializable semantic output payload for the scope end event.
///     timestamp: Optional timezone-aware ``datetime.datetime`` for the emitted end event.
///         When omitted, the runtime default end timestamp is used.
///
/// Raises:
///     RuntimeError: If the scope is not the current top scope or is not found
///         on the stack.
///     TypeError: If ``timestamp`` is not a ``datetime.datetime``.
///     ValueError: If ``timestamp`` is a naive datetime.
#[pyfunction]
#[pyo3(signature = (handle: "ScopeHandle", output: "object | None"=None, timestamp: "datetime.datetime | None"=None) -> "None", text_signature = "(handle: ScopeHandle, output: object | None = None, timestamp: datetime.datetime | None = None) -> None")]
fn pop_scope(
    handle: &PyScopeHandle,
    output: Option<&Bound<'_, PyAny>>,
    timestamp: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let output = opt_py_to_json(output)?;
    let timestamp = opt_py_to_timestamp(timestamp)?;
    core_scope_api::pop_scope(
        core_scope_api::PopScopeParams::builder()
            .handle_uuid(&handle.inner.uuid)
            .output_opt(output)
            .timestamp_opt(timestamp)
            .build(),
    )
    .map_err(to_py_err)
}

/// Emit a ``Mark`` event under the current or specified scope.
///
/// Args:
///     name: Event name.
///     handle: Optional parent scope handle. Defaults to current top of stack.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
///     timestamp: Optional timezone-aware ``datetime.datetime`` for the emitted mark event.
///         When omitted, the current runtime time is used.
///
/// Raises:
///     TypeError: If ``timestamp`` is not a ``datetime.datetime``.
///     ValueError: If ``timestamp`` is a naive datetime.
#[pyfunction]
#[pyo3(signature = (
    name: "str",
    *,
	    handle: "ScopeHandle | None"=None,
	    data: "object | None"=None,
	    metadata: "object | None"=None,
	    timestamp: "datetime.datetime | None"=None
) -> "None", text_signature = "(name: str, *, handle: ScopeHandle | None = None, data: object | None = None, metadata: object | None = None, timestamp: datetime.datetime | None = None) -> None")]
fn event(
    name: &str,
    handle: Option<PyScopeHandle>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
    timestamp: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    let timestamp = opt_py_to_timestamp(timestamp)?;
    core_scope_api::event(
        core_scope_api::EmitMarkEventParams::builder()
            .name(name)
            .parent_opt(handle.as_ref().map(|h| &h.inner))
            .data_opt(data)
            .metadata_opt(metadata)
            .timestamp_opt(timestamp)
            .build(),
    )
    .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begin a tool call — creates a ``ToolHandle`` and emits a ``Start`` event.
///
/// This is the manual (non-execute) entry point: callers are responsible
/// for invoking the tool themselves and later calling ``tool_call_end``.
/// Sanitize-request guardrails affect the emitted start-event payload; request
/// and execution intercepts run only through ``tool_call_execute``.
///
/// Args:
///     name: Tool name.
///     args: JSON-serializable tool arguments recorded on the start event after
///         sanitize-request guardrails.
///     handle: Optional parent scope handle.
///     attributes: Optional ``ToolAttributes`` bitflags.
///     data: Optional JSON-serializable application payload stored on the tool handle.
///     metadata: Optional JSON-serializable metadata recorded on the start event.
///     tool_call_id: Optional provider-specific tool-call correlation ID.
///     timestamp: Optional timezone-aware ``datetime.datetime`` recorded as the handle
///         start time and on the emitted start event. When omitted, the current
///         runtime time is used.
///
/// Returns:
///     A ``ToolHandle`` that must be passed to ``tool_call_end``.
///
/// Raises:
///     TypeError: If ``timestamp`` is not a ``datetime.datetime``.
///     ValueError: If ``timestamp`` is a naive datetime.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    name: "str",
    args: "object",
    *,
    handle: "ScopeHandle | None"=None,
    attributes: "ToolAttributes | None"=None,
	    data: "object | None"=None,
	    metadata: "object | None"=None,
	    tool_call_id: "str | None"=None,
	    timestamp: "datetime.datetime | None"=None
) -> "ToolHandle", text_signature = "(name: str, args: object, *, handle: ScopeHandle | None = None, attributes: ToolAttributes | None = None, data: object | None = None, metadata: object | None = None, tool_call_id: str | None = None, timestamp: datetime.datetime | None = None) -> ToolHandle")]
fn tool_call(
    name: &str,
    args: &Bound<'_, PyAny>,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyToolAttributes>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
    tool_call_id: Option<String>,
    timestamp: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyToolHandle> {
    let args_json = py_to_json(args)?;
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(ToolAttributes::empty());
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    let timestamp = opt_py_to_timestamp(timestamp)?;
    core_tool_api::tool_call(
        core_tool_api::ToolCallParams::builder()
            .name(name)
            .args(args_json)
            .parent_opt(handle.as_ref().map(|h| &h.inner))
            .attributes(attrs)
            .data_opt(data)
            .metadata_opt(metadata)
            .tool_call_id_opt(tool_call_id)
            .timestamp_opt(timestamp)
            .build(),
    )
    .map(PyToolHandle::from)
    .map_err(to_py_err)
}

/// End a tool call — records the result and emits an ``End`` event.
///
/// Sanitize-response guardrails affect the emitted end-event payload; response
/// intercepts run only through ``tool_call_execute``.
///
/// Args:
///     handle: The ``ToolHandle`` returned by ``tool_call``.
///     result: JSON-serializable tool result recorded on the end event after
///         sanitize-response guardrails unless it sanitizes to JSON null.
///     data: Optional JSON-serializable payload used when the sanitized result is JSON null.
///     metadata: Optional JSON-serializable metadata recorded on the end event.
///     timestamp: Optional timezone-aware ``datetime.datetime`` for the emitted end event.
///         When omitted, the runtime default end timestamp is used.
///
/// Raises:
///     TypeError: If ``timestamp`` is not a ``datetime.datetime``.
///     ValueError: If ``timestamp`` is a naive datetime.
#[pyfunction]
#[pyo3(signature = (
    handle: "ToolHandle",
    result: "object",
	    *,
	    data: "object | None"=None,
	    metadata: "object | None"=None,
	    timestamp: "datetime.datetime | None"=None
) -> "None", text_signature = "(handle: ToolHandle, result: object, *, data: object | None = None, metadata: object | None = None, timestamp: datetime.datetime | None = None) -> None")]
fn tool_call_end(
    handle: &PyToolHandle,
    result: &Bound<'_, PyAny>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
    timestamp: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let result_json = py_to_json(result)?;
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    let timestamp = opt_py_to_timestamp(timestamp)?;
    core_tool_api::tool_call_end(
        core_tool_api::ToolCallEndParams::builder()
            .handle(&handle.inner)
            .result(result_json)
            .data_opt(data)
            .metadata_opt(metadata)
            .timestamp_opt(timestamp)
            .build(),
    )
    .map_err(to_py_err)
}

/// Execute a tool call through the full middleware pipeline.
///
/// Runs conditional-execution guardrails (on raw args) → request intercepts →
/// sanitize-request guardrails (for the emitted ``Start`` event payload) →
/// execution intercepts → the supplied function → sanitize-response
/// guardrails (for the emitted ``End`` event payload), then returns the final
/// result. On rejection, only a standalone ``Mark`` event is emitted (no
/// ``Start``/``End`` pair) and ``GuardrailRejected`` is raised.
///
/// Args:
///     name: Tool name.
///     args: JSON-serializable tool arguments.
///     func: An async callable ``(args) -> result`` that performs the tool work.
///     handle: Optional parent scope handle.
///     attributes: Optional ``ToolAttributes`` bitflags.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
/// Returns:
///     An awaitable that resolves to the tool result after execution
///     intercepts. Sanitize guardrails do not rewrite the value returned to
///     the caller.
#[pyfunction]
#[pyo3(signature = (
    name: "str",
    args: "object",
    func: "object",
    *,
    handle: "ScopeHandle | None"=None,
    attributes: "ToolAttributes | None"=None,
    data: "object | None"=None,
    metadata: "object | None"=None
) -> "object", text_signature = "(name: str, args: object, func: object, *, handle: ScopeHandle | None = None, attributes: ToolAttributes | None = None, data: object | None = None, metadata: object | None = None) -> object")]
#[allow(clippy::too_many_arguments)]
fn tool_call_execute<'py>(
    py: Python<'py>,
    name: String,
    args: &Bound<'py, PyAny>,
    func: Py<PyAny>,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyToolAttributes>,
    data: Option<&Bound<'py, PyAny>>,
    metadata: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let args_json = py_to_json(args)?;
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(ToolAttributes::empty());
    let data_json = opt_py_to_json(data)?;
    let metadata_json = opt_py_to_json(metadata)?;
    let exec_fn = py_callable::wrap_py_tool_exec_fn(func);
    let default_fn: ToolExecutionNextFn = Arc::new(move |args| exec_fn(args));
    let parent_handle = handle.map(|h| h.inner).unwrap_or_else(task_scope_top);

    let scope_stack = current_scope_stack_handle();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        TASK_SCOPE_STACK
            .scope(scope_stack, async move {
                let result = core_tool_api::tool_call_execute(
                    core_tool_api::ToolCallExecuteParams::builder()
                        .name(name)
                        .args(args_json)
                        .func(default_fn)
                        .parent(parent_handle)
                        .attributes(attrs)
                        .data_opt(data_json)
                        .metadata_opt(metadata_json)
                        .build(),
                )
                .await
                .map_err(to_py_err)?;
                Python::attach(|py| json_to_py(py, &result))
            })
            .await
    })
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begin an LLM call — creates an ``LlmHandle`` and emits a ``Start`` event.
///
/// This is the manual (non-execute) entry point: callers are responsible
/// for performing the LLM request themselves and later calling ``llm_call_end``.
/// Sanitize-request guardrails affect the emitted start-event payload; request
/// and execution intercepts run only through ``llm_call_execute``.
///
/// Args:
///     name: Model/provider name.
///     request: An ``LlmRequest`` with headers and content.
///     handle: Optional parent scope handle.
///     attributes: Optional ``LlmAttributes`` bitflags.
///     data: Optional JSON-serializable application payload stored on the LLM handle.
///     metadata: Optional JSON-serializable metadata recorded on the start event.
///     model_name: Optional normalized model name recorded in the LLM event category profile.
///     timestamp: Optional timezone-aware ``datetime.datetime`` recorded as the handle
///         start time and on the emitted start event. When omitted, the current
///         runtime time is used.
///
/// Returns:
///     An ``LlmHandle`` that must be passed to ``llm_call_end``.
///
/// Raises:
///     TypeError: If ``timestamp`` is not a ``datetime.datetime``.
///     ValueError: If ``timestamp`` is a naive datetime.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    name: "str",
    request: "LlmRequest",
    *,
    handle: "ScopeHandle | None"=None,
    attributes: "LlmAttributes | None"=None,
	    data: "object | None"=None,
	    metadata: "object | None"=None,
	    model_name: "str | None"=None,
	    timestamp: "datetime.datetime | None"=None
) -> "LlmHandle", text_signature = "(name: str, request: LlmRequest, *, handle: ScopeHandle | None = None, attributes: LlmAttributes | None = None, data: object | None = None, metadata: object | None = None, model_name: str | None = None, timestamp: datetime.datetime | None = None) -> LlmHandle")]
fn llm_call(
    name: &str,
    request: PyLLMRequest,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyLLMAttributes>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
    model_name: Option<String>,
    timestamp: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyLLMHandle> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(LlmAttributes::empty());
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    let timestamp = opt_py_to_timestamp(timestamp)?;
    let params = core_llm_api::LlmCallParams::builder()
        .name(name)
        .request(&request.inner)
        .parent_opt(handle.as_ref().map(|h| &h.inner))
        .attributes(attrs)
        .data_opt(data)
        .metadata_opt(metadata)
        .model_name_opt(model_name)
        .timestamp_opt(timestamp)
        .build();
    core_llm_api::llm_call(params)
        .map(PyLLMHandle::from)
        .map_err(to_py_err)
}

/// End an LLM call — records the response and emits an ``End`` event.
///
/// Sanitize-response guardrails affect the emitted end-event payload; response
/// intercepts run only through ``llm_call_execute``.
///
/// Args:
///     handle: The ``LlmHandle`` returned by ``llm_call``.
///     response: JSON-serializable LLM response recorded on the end event after
///         sanitize-response guardrails unless it sanitizes to JSON null.
///     data: Optional JSON-serializable payload used when the sanitized response is JSON null.
///     metadata: Optional JSON-serializable metadata recorded on the end event.
///     annotated_response: Optional normalized response annotation, either as an
///         ``AnnotatedLLMResponse`` instance or a JSON object matching that schema.
///     response_codec: Optional response codec used to decode ``response`` into
///         an annotated response for observability when ``annotated_response`` is omitted.
///     timestamp: Optional timezone-aware ``datetime.datetime`` for the emitted end event.
///         When omitted, the runtime default end timestamp is used.
///
/// Raises:
///     TypeError: If ``timestamp`` is not a ``datetime.datetime``.
///     ValueError: If ``timestamp`` is a naive datetime.
#[pyfunction]
#[pyo3(signature = (
    handle: "LlmHandle",
    response: "object",
	    *,
	    data: "object | None"=None,
	    metadata: "object | None"=None,
	    annotated_response: "AnnotatedLLMResponse | object | None"=None,
	    response_codec: "object | None"=None,
	    timestamp: "datetime.datetime | None"=None
) -> "None", text_signature = "(handle: LlmHandle, response: object, *, data: object | None = None, metadata: object | None = None, annotated_response: AnnotatedLLMResponse | object | None = None, response_codec: object | None = None, timestamp: datetime.datetime | None = None) -> None")]
fn llm_call_end(
    handle: &PyLLMHandle,
    response: &Bound<'_, PyAny>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
    annotated_response: Option<&Bound<'_, PyAny>>,
    response_codec: Option<&Bound<'_, PyAny>>,
    timestamp: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let response_json = py_to_json(response)?;
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    let response_codec = py_llm_response_codec(response_codec);
    let annotated_response = py_annotated_llm_response(annotated_response)?;
    let timestamp = opt_py_to_timestamp(timestamp)?;
    core_llm_api::llm_call_end(
        core_llm_api::LlmCallEndParams::builder()
            .handle(&handle.inner)
            .response(response_json)
            .data_opt(data)
            .metadata_opt(metadata)
            .annotated_response_opt(annotated_response)
            .response_codec_opt(response_codec)
            .timestamp_opt(timestamp)
            .build(),
    )
    .map_err(to_py_err)
}

/// Execute an LLM call through the full middleware pipeline.
///
/// Runs conditional-execution guardrails (on raw request) → request intercepts →
/// sanitize-request guardrails (for the emitted ``Start`` event payload) →
/// execution intercepts → the supplied function → sanitize-response
/// guardrails (for the emitted ``End`` event payload), then returns the final
/// response. On rejection, only a standalone ``Mark`` event is emitted (no
/// ``Start``/``End`` pair) and ``GuardrailRejected`` is raised.
///
/// Args:
///     name: Model/provider name.
///     request: An ``LlmRequest`` with headers and content.
///     func: An async callable ``(LlmRequest) -> dict`` that performs the LLM call.
///     handle: Optional parent scope handle.
///     attributes: Optional ``LlmAttributes`` bitflags.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
///     model_name: Optional normalized model name recorded in emitted LLM events.
///     codec: Optional request codec used for annotated-aware request intercepts.
///     response_codec: Optional response codec used to attach annotated response data
///         to emitted end events.
///
/// Returns:
///     An awaitable that resolves to the LLM response after execution
///     intercepts. Sanitize guardrails do not rewrite the value returned to
///     the caller.
#[pyfunction]
#[pyo3(signature = (
    name: "str",
    request: "LlmRequest",
    func: "object",
    *,
    handle: "ScopeHandle | None"=None,
    attributes: "LlmAttributes | None"=None,
    data: "object | None"=None,
    metadata: "object | None"=None,
    model_name: "str | None"=None,
    codec: "object | None"=None,
    response_codec: "object | None"=None
) -> "object", text_signature = "(name: str, request: LlmRequest, func: object, *, handle: ScopeHandle | None = None, attributes: LlmAttributes | None = None, data: object | None = None, metadata: object | None = None, model_name: str | None = None, codec: object | None = None, response_codec: object | None = None) -> object")]
#[allow(clippy::too_many_arguments)]
fn llm_call_execute<'py>(
    py: Python<'py>,
    name: String,
    request: PyLLMRequest,
    func: Py<PyAny>,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyLLMAttributes>,
    data: Option<&Bound<'py, PyAny>>,
    metadata: Option<&Bound<'py, PyAny>>,
    model_name: Option<String>,
    codec: Option<&Bound<'py, PyAny>>,
    response_codec: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(LlmAttributes::empty());
    let data_json = opt_py_to_json(data)?;
    let metadata_json = opt_py_to_json(metadata)?;
    let exec_fn = py_callable::wrap_py_llm_exec_fn(func);
    let default_fn: LlmExecutionNextFn = Arc::new(move |req| exec_fn(req));
    let parent_handle = handle.map(|h| h.inner).unwrap_or_else(task_scope_top);
    let codec_arc: Option<Arc<dyn LlmCodec>> = codec.map(|c| {
        Arc::new(py_callable::PyLlmCodecWrapper {
            py_codec: c.clone().unbind(),
        }) as Arc<dyn LlmCodec>
    });
    let response_codec_arc = py_llm_response_codec(response_codec);

    let scope_stack = current_scope_stack_handle();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        TASK_SCOPE_STACK
            .scope(scope_stack, async move {
                let params = core_llm_api::LlmCallExecuteParams::builder()
                    .name(name)
                    .request(request.inner)
                    .func(default_fn)
                    .parent(parent_handle)
                    .attributes(attrs)
                    .data_opt(data_json)
                    .metadata_opt(metadata_json)
                    .model_name_opt(model_name)
                    .codec_opt(codec_arc)
                    .response_codec_opt(response_codec_arc)
                    .build();
                let result = core_llm_api::llm_call_execute(params)
                    .await
                    .map_err(to_py_err)?;
                Python::attach(|py| json_to_py(py, &result))
            })
            .await
    })
}

/// Execute a streaming LLM call through the full middleware pipeline.
///
/// Like ``llm_call_execute``, conditional-execution guardrails run first on
/// the raw request. If accepted, the execution function returns an async
/// iterator of JSON chunks. The runtime wraps the stream with
/// ``LlmStreamWrapper`` so that stream execution intercepts can inspect or
/// transform each chunk in flight.
///
/// Args:
///     name: Model/provider name.
///     request: An ``LlmRequest`` with headers and content.
///     func: An async callable ``(LlmRequest) -> AsyncIterator[Any]`` that returns JSON chunks.
///     collector: A callable ``(Any) -> None`` invoked with each intercepted chunk
///         (after stream execution intercepts have been applied).
///     finalizer: A callable ``() -> Any`` invoked once when the stream is exhausted.
///         Its return value is the aggregated response (converted to JSON).
///     handle: Optional parent scope handle.
///     attributes: Optional ``LlmAttributes`` bitflags.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
///     model_name: Optional normalized model name recorded in emitted LLM events.
///     codec: Optional request codec used for annotated-aware request intercepts.
///     response_codec: Optional response codec used to attach annotated response data
///         to emitted end events.
///
/// Returns:
///     An awaitable that resolves to an ``LlmStream`` async iterator of JSON chunks.
#[pyfunction]
#[pyo3(signature = (
    name: "str",
    request: "LlmRequest",
    func: "object",
    collector: "object",
    finalizer: "object",
    *,
    handle: "ScopeHandle | None"=None,
    attributes: "LlmAttributes | None"=None,
    data: "object | None"=None,
    metadata: "object | None"=None,
    model_name: "str | None"=None,
    codec: "object | None"=None,
    response_codec: "object | None"=None
) -> "object", text_signature = "(name: str, request: LlmRequest, func: object, collector: object, finalizer: object, *, handle: ScopeHandle | None = None, attributes: LlmAttributes | None = None, data: object | None = None, metadata: object | None = None, model_name: str | None = None, codec: object | None = None, response_codec: object | None = None) -> object")]
#[allow(clippy::too_many_arguments)]
fn llm_stream_call_execute<'py>(
    py: Python<'py>,
    name: String,
    request: PyLLMRequest,
    func: Py<PyAny>,
    collector: Py<PyAny>,
    finalizer: Py<PyAny>,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyLLMAttributes>,
    data: Option<&Bound<'py, PyAny>>,
    metadata: Option<&Bound<'py, PyAny>>,
    model_name: Option<String>,
    codec: Option<&Bound<'py, PyAny>>,
    response_codec: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(LlmAttributes::empty());
    let data_json = opt_py_to_json(data)?;
    let metadata_json = opt_py_to_json(metadata)?;
    let exec_fn = py_callable::wrap_py_llm_stream_exec_fn(func);
    let default_fn: LlmStreamExecutionNextFn = Arc::new(move |req| exec_fn(req));
    let collector_fn = py_callable::wrap_py_collector_fn(collector);
    let finalizer_fn = py_callable::wrap_py_finalizer_fn(finalizer);
    let parent_handle = handle.map(|h| h.inner).unwrap_or_else(task_scope_top);
    let codec_arc: Option<Arc<dyn LlmCodec>> = codec.map(|c| {
        Arc::new(py_callable::PyLlmCodecWrapper {
            py_codec: c.clone().unbind(),
        }) as Arc<dyn LlmCodec>
    });
    let response_codec_arc = py_llm_response_codec(response_codec);

    let scope_stack = current_scope_stack_handle();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        TASK_SCOPE_STACK
            .scope(scope_stack, async move {
                let params = core_llm_api::LlmStreamCallExecuteParams::builder()
                    .name(name)
                    .request(request.inner)
                    .func(default_fn)
                    .collector(collector_fn)
                    .finalizer(finalizer_fn)
                    .parent(parent_handle)
                    .attributes(attrs)
                    .data_opt(data_json)
                    .metadata_opt(metadata_json)
                    .model_name_opt(model_name)
                    .codec_opt(codec_arc)
                    .response_codec_opt(response_codec_arc)
                    .build();
                let rust_stream = core_llm_api::llm_stream_call_execute(params)
                    .await
                    .map_err(to_py_err)?;

                // Spawn a tokio task that drains the Rust stream into an mpsc channel
                let (tx, rx) = tokio::sync::mpsc::channel::<FlowResult<serde_json::Value>>(32);
                tokio::spawn(forward_stream_to_channel(rust_stream, tx));

                Ok(PyLlmStream {
                    receiver: tokio::sync::Mutex::new(rx),
                })
            })
            .await
    })
}

// ---------------------------------------------------------------------------
// Guardrail registrations (macro-generated)
// ---------------------------------------------------------------------------

/// Macro that generates a register/deregister pair for tool guardrails
/// whose callback signature is `(tool_name: str, json: Any) -> Any`.
macro_rules! py_guardrail_tool_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $core_register:path, $core_deregister:path, $wrapper:path) => {
        $(#[$reg_meta])*
        #[pyfunction]
        fn $register_name(name: &str, priority: i32, guardrail: Py<PyAny>) -> PyResult<()> {
            $core_register(name, priority, $wrapper(guardrail)).map_err(to_py_err)
        }

        /// Remove the previously registered guardrail by name.
        #[pyfunction]
        fn $deregister_name(name: &str) -> PyResult<bool> {
            $core_deregister(name).map_err(to_py_err)
        }
    };
}

py_guardrail_tool_api!(
    /// Register a tool sanitize-request guardrail.
    ///
    /// Callback: ``(tool_name: str, args: Any) -> Any`` — returns sanitized args.
    register_tool_sanitize_request_guardrail,
    deregister_tool_sanitize_request_guardrail,
    core_registry_api::register_tool_sanitize_request_guardrail,
    core_registry_api::deregister_tool_sanitize_request_guardrail,
    py_callable::wrap_py_tool_fn
);

py_guardrail_tool_api!(
    /// Register a tool sanitize-response guardrail.
    ///
    /// Callback: ``(tool_name: str, result: Any) -> Any`` — returns sanitized result.
    register_tool_sanitize_response_guardrail,
    deregister_tool_sanitize_response_guardrail,
    core_registry_api::register_tool_sanitize_response_guardrail,
    core_registry_api::deregister_tool_sanitize_response_guardrail,
    py_callable::wrap_py_tool_fn
);

/// Register a tool conditional-execution guardrail.
///
/// Callback: ``(tool_name: str, args: Any) -> Optional[str]``.
/// Return ``None`` to allow execution, or a rejection reason string to block it.
#[pyfunction]
fn register_tool_conditional_execution_guardrail(
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    core_registry_api::register_tool_conditional_execution_guardrail(
        name,
        priority,
        py_callable::wrap_py_tool_conditional_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered tool conditional-execution guardrail.
#[pyfunction]
fn deregister_tool_conditional_execution_guardrail(name: &str) -> PyResult<bool> {
    core_registry_api::deregister_tool_conditional_execution_guardrail(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Tool intercept registrations
// ---------------------------------------------------------------------------

/// Macro that generates a register/deregister pair for tool intercepts
/// whose callback signature is `(tool_name: str, json: Any) -> Any`.
macro_rules! py_intercept_tool_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $core_register:path, $core_deregister:path, $wrapper:path) => {
        $(#[$reg_meta])*
        #[pyfunction]
        fn $register_name(
            name: &str,
            priority: i32,
            break_chain: bool,
            callable: Py<PyAny>,
        ) -> PyResult<()> {
            $core_register(name, priority, break_chain, $wrapper(callable)).map_err(to_py_err)
        }

        /// Remove the previously registered intercept by name.
        #[pyfunction]
        fn $deregister_name(name: &str) -> PyResult<bool> {
            $core_deregister(name).map_err(to_py_err)
        }
    };
}

py_intercept_tool_api!(
    /// Register a tool request intercept.
    ///
    /// Callback: ``(tool_name: str, args: Any) -> Any`` — transforms tool arguments.
    /// If ``break_chain`` is ``True``, no lower-priority intercepts run after this one.
    register_tool_request_intercept,
    deregister_tool_request_intercept,
    core_registry_api::register_tool_request_intercept,
    core_registry_api::deregister_tool_request_intercept,
    py_callable::wrap_py_tool_request_intercept_fn
);

/// Register a tool execution intercept that can replace the tool function.
///
/// ``callable``: ``async (args: Any, next) -> Any`` — middleware intercept function.
/// Call ``await next(args)`` to invoke the next intercept or original
/// implementation; skip calling ``next`` to short-circuit.
#[pyfunction]
fn register_tool_execution_intercept(
    name: &str,
    priority: i32,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core_registry_api::register_tool_execution_intercept(
        name,
        priority,
        py_callable::wrap_py_tool_exec_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered tool execution intercept.
#[pyfunction]
fn deregister_tool_execution_intercept(name: &str) -> PyResult<bool> {
    core_registry_api::deregister_tool_execution_intercept(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register an LLM sanitize-request guardrail.
///
/// Callback: ``(request: LlmRequest) -> LlmRequest`` — returns a sanitized request.
#[pyfunction]
fn register_llm_sanitize_request_guardrail(
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    core_registry_api::register_llm_sanitize_request_guardrail(
        name,
        priority,
        py_callable::wrap_py_llm_sanitize_request_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM sanitize-request guardrail.
#[pyfunction]
fn deregister_llm_sanitize_request_guardrail(name: &str) -> PyResult<bool> {
    core_registry_api::deregister_llm_sanitize_request_guardrail(name).map_err(to_py_err)
}

/// Register an LLM sanitize-response guardrail.
///
/// Callback: ``(response: dict) -> dict`` — returns a sanitized response.
#[pyfunction]
fn register_llm_sanitize_response_guardrail(
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    core_registry_api::register_llm_sanitize_response_guardrail(
        name,
        priority,
        py_callable::wrap_py_llm_sanitize_response_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM sanitize-response guardrail.
#[pyfunction]
fn deregister_llm_sanitize_response_guardrail(name: &str) -> PyResult<bool> {
    core_registry_api::deregister_llm_sanitize_response_guardrail(name).map_err(to_py_err)
}

/// Register an LLM conditional-execution guardrail.
///
/// Callback: ``(request: LlmRequest) -> Optional[str]``.
/// Return ``None`` to allow execution, or a rejection reason string to block it.
#[pyfunction]
fn register_llm_conditional_execution_guardrail(
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    core_registry_api::register_llm_conditional_execution_guardrail(
        name,
        priority,
        py_callable::wrap_py_llm_conditional_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM conditional-execution guardrail.
#[pyfunction]
fn deregister_llm_conditional_execution_guardrail(name: &str) -> PyResult<bool> {
    core_registry_api::deregister_llm_conditional_execution_guardrail(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register an LLM request intercept.
///
/// Callback: ``(name: str, request: LlmRequest, annotated: AnnotatedLLMRequest | None) -> (LlmRequest, AnnotatedLLMRequest | None)``
/// — transforms the LLM request and optional annotated request.
/// If ``break_chain`` is ``True``, no lower-priority intercepts run after this one.
#[pyfunction]
fn register_llm_request_intercept(
    name: &str,
    priority: i32,
    break_chain: bool,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core_registry_api::register_llm_request_intercept(
        name,
        priority,
        break_chain,
        py_callable::wrap_py_llm_request_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM request intercept.
#[pyfunction]
fn deregister_llm_request_intercept(name: &str) -> PyResult<bool> {
    core_registry_api::deregister_llm_request_intercept(name).map_err(to_py_err)
}

/// Register an LLM execution intercept that can replace the LLM call.
///
/// ``callable``: ``async (native: Any, next) -> Any`` — middleware intercept function.
/// Call ``await next(native)`` to invoke the next intercept or original
/// implementation; skip calling ``next`` to short-circuit.
#[pyfunction]
fn register_llm_execution_intercept(
    name: &str,
    priority: i32,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core_registry_api::register_llm_execution_intercept(
        name,
        priority,
        py_callable::wrap_py_llm_exec_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM execution intercept.
#[pyfunction]
fn deregister_llm_execution_intercept(name: &str) -> PyResult<bool> {
    core_registry_api::deregister_llm_execution_intercept(name).map_err(to_py_err)
}

/// Register an LLM stream-execution intercept that can replace the streaming LLM call.
///
/// ``callable``: ``async (native: Any, next) -> AsyncIterator[Any]`` —
/// middleware streaming intercept function.
/// Call ``await next(native)`` to invoke the next intercept or original
/// streaming implementation; skip calling ``next`` to short-circuit.
#[pyfunction]
fn register_llm_stream_execution_intercept(
    name: &str,
    priority: i32,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core_registry_api::register_llm_stream_execution_intercept(
        name,
        priority,
        py_callable::wrap_py_llm_stream_exec_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM stream-execution intercept.
#[pyfunction]
fn deregister_llm_stream_execution_intercept(name: &str) -> PyResult<bool> {
    core_registry_api::deregister_llm_stream_execution_intercept(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Standalone middleware chains
// ---------------------------------------------------------------------------

/// Run the registered tool request intercept chain on the given arguments.
///
/// Returns the transformed arguments after all intercepts have been applied.
///
/// Args:
///     name: Tool name.
///     args: Tool arguments (any JSON-serializable object).
///
/// Returns:
///     The (possibly transformed) arguments.
#[pyfunction]
fn tool_request_intercepts<'py>(
    py: Python<'py>,
    name: &str,
    args: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let args_json = py_to_json(args)?;
    let result = core_tool_api::tool_request_intercepts(name, args_json).map_err(to_py_err)?;
    json_to_py(py, &result)
}

/// Run the registered tool conditional execution guardrail chain.
///
/// Raises ``RuntimeError`` with the rejection reason if any guardrail rejects.
///
/// Args:
///     name: Tool name.
///     args: Tool arguments (any JSON-serializable object).
#[pyfunction]
fn tool_conditional_execution(name: &str, args: &Bound<'_, PyAny>) -> PyResult<()> {
    let args_json = py_to_json(args)?;
    core_tool_api::tool_conditional_execution(name, &args_json).map_err(to_py_err)
}

/// Run the registered LLM request intercept chain on the given request.
///
/// Returns the transformed request after all intercepts have been applied.
///
/// Args:
///     request: An ``LlmRequest`` object.
///
/// Returns:
///     The (possibly transformed) ``LlmRequest``.
#[pyfunction]
fn llm_request_intercepts(name: &str, request: PyLLMRequest) -> PyResult<PyLLMRequest> {
    let result = core_llm_api::llm_request_intercepts(name, request.inner).map_err(to_py_err)?;
    Ok(PyLLMRequest { inner: result })
}

/// Run the registered LLM conditional execution guardrail chain.
///
/// Raises ``RuntimeError`` with the rejection reason if any guardrail rejects.
///
/// Args:
///     request: An ``LlmRequest`` object.
#[pyfunction]
fn llm_conditional_execution(request: PyLLMRequest) -> PyResult<()> {
    core_llm_api::llm_conditional_execution(&request.inner).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

/// Register an event subscriber.
///
/// Callback: ``(event: Event) -> None`` — called for every lifecycle event
/// (scope start/end, tool start/end, LLM start/end, marks).
///
/// Args:
///     name: Unique subscriber name (used for deregistration).
///     callback: The subscriber callable.
///
/// Raises:
///     RuntimeError: If a subscriber with this name already exists.
#[pyfunction]
fn register_subscriber(name: &str, callback: Py<PyAny>) -> PyResult<()> {
    core_subscriber_api::register_subscriber(name, py_callable::wrap_py_event_subscriber(callback))
        .map_err(to_py_err)
}

/// Register an anonymous global event subscriber and return a closeable handle.
#[pyfunction]
fn subscribe(callback: Py<PyAny>) -> PyResult<PySubscription> {
    core_subscriber_api::subscribe(py_callable::wrap_py_event_subscriber(callback))
        .map(|handle| PySubscription {
            inner: Some(PySubscriptionInner::Global(handle)),
        })
        .map_err(to_py_err)
}

/// Remove a previously registered event subscriber.
///
/// Returns ``True`` if a subscriber with that name was found and removed.
#[pyfunction]
fn deregister_subscriber(name: &str) -> PyResult<bool> {
    core_subscriber_api::deregister_subscriber(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Scope-local guardrail registrations (macro-generated)
// ---------------------------------------------------------------------------

/// Parse a UUID string, returning a PyErr on failure.
fn parse_uuid(scope_uuid: &str) -> PyResult<Uuid> {
    Uuid::parse_str(scope_uuid)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid UUID: {e}")))
}

/// Macro that generates a scope-local register/deregister pair for guardrails
/// whose callback signature is `(tool_name: str, json: Any) -> Any`.
macro_rules! py_scope_local_guardrail_tool_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $core_register:path, $core_deregister:path, $wrapper:path) => {
        $(#[$reg_meta])*
        #[pyfunction]
        fn $register_name(scope_uuid: &str, name: &str, priority: i32, guardrail: Py<PyAny>) -> PyResult<()> {
            let uuid = parse_uuid(scope_uuid)?;
            $core_register(&uuid, name, priority, $wrapper(guardrail)).map_err(to_py_err)
        }

        /// Remove the previously registered scope-local guardrail by name.
        #[pyfunction]
        fn $deregister_name(scope_uuid: &str, name: &str) -> PyResult<bool> {
            let uuid = parse_uuid(scope_uuid)?;
            $core_deregister(&uuid, name).map_err(to_py_err)
        }
    };
}

py_scope_local_guardrail_tool_api!(
    /// Register a scope-local tool sanitize-request guardrail.
    scope_register_tool_sanitize_request_guardrail,
    scope_deregister_tool_sanitize_request_guardrail,
    core_registry_api::scope_register_tool_sanitize_request_guardrail,
    core_registry_api::scope_deregister_tool_sanitize_request_guardrail,
    py_callable::wrap_py_tool_fn
);

py_scope_local_guardrail_tool_api!(
    /// Register a scope-local tool sanitize-response guardrail.
    scope_register_tool_sanitize_response_guardrail,
    scope_deregister_tool_sanitize_response_guardrail,
    core_registry_api::scope_register_tool_sanitize_response_guardrail,
    core_registry_api::scope_deregister_tool_sanitize_response_guardrail,
    py_callable::wrap_py_tool_fn
);

/// Register a scope-local tool conditional-execution guardrail.
#[pyfunction]
fn scope_register_tool_conditional_execution_guardrail(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_register_tool_conditional_execution_guardrail(
        &uuid,
        name,
        priority,
        py_callable::wrap_py_tool_conditional_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered scope-local tool conditional-execution guardrail.
#[pyfunction]
fn scope_deregister_tool_conditional_execution_guardrail(
    scope_uuid: &str,
    name: &str,
) -> PyResult<bool> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_deregister_tool_conditional_execution_guardrail(&uuid, name)
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Scope-local tool intercept registrations
// ---------------------------------------------------------------------------

/// Macro that generates a scope-local register/deregister pair for tool intercepts
/// whose callback signature is `(tool_name: str, json: Any) -> Any`.
macro_rules! py_scope_local_intercept_tool_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $core_register:path, $core_deregister:path, $wrapper:path) => {
        $(#[$reg_meta])*
        #[pyfunction]
        fn $register_name(
            scope_uuid: &str,
            name: &str,
            priority: i32,
            break_chain: bool,
            callable: Py<PyAny>,
        ) -> PyResult<()> {
            let uuid = parse_uuid(scope_uuid)?;
            $core_register(&uuid, name, priority, break_chain, $wrapper(callable)).map_err(to_py_err)
        }

        /// Remove the previously registered scope-local intercept by name.
        #[pyfunction]
        fn $deregister_name(scope_uuid: &str, name: &str) -> PyResult<bool> {
            let uuid = parse_uuid(scope_uuid)?;
            $core_deregister(&uuid, name).map_err(to_py_err)
        }
    };
}

py_scope_local_intercept_tool_api!(
    /// Register a scope-local tool request intercept.
    scope_register_tool_request_intercept,
    scope_deregister_tool_request_intercept,
    core_registry_api::scope_register_tool_request_intercept,
    core_registry_api::scope_deregister_tool_request_intercept,
    py_callable::wrap_py_tool_request_intercept_fn
);

/// Register a scope-local tool execution intercept.
#[pyfunction]
fn scope_register_tool_execution_intercept(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    callable: Py<PyAny>,
) -> PyResult<()> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_register_tool_execution_intercept(
        &uuid,
        name,
        priority,
        py_callable::wrap_py_tool_exec_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered scope-local tool execution intercept.
#[pyfunction]
fn scope_deregister_tool_execution_intercept(scope_uuid: &str, name: &str) -> PyResult<bool> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_deregister_tool_execution_intercept(&uuid, name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Scope-local LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register a scope-local LLM sanitize-request guardrail.
#[pyfunction]
fn scope_register_llm_sanitize_request_guardrail(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_register_llm_sanitize_request_guardrail(
        &uuid,
        name,
        priority,
        py_callable::wrap_py_llm_sanitize_request_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered scope-local LLM sanitize-request guardrail.
#[pyfunction]
fn scope_deregister_llm_sanitize_request_guardrail(scope_uuid: &str, name: &str) -> PyResult<bool> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_deregister_llm_sanitize_request_guardrail(&uuid, name)
        .map_err(to_py_err)
}

/// Register a scope-local LLM sanitize-response guardrail.
#[pyfunction]
fn scope_register_llm_sanitize_response_guardrail(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_register_llm_sanitize_response_guardrail(
        &uuid,
        name,
        priority,
        py_callable::wrap_py_llm_sanitize_response_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered scope-local LLM sanitize-response guardrail.
#[pyfunction]
fn scope_deregister_llm_sanitize_response_guardrail(
    scope_uuid: &str,
    name: &str,
) -> PyResult<bool> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_deregister_llm_sanitize_response_guardrail(&uuid, name)
        .map_err(to_py_err)
}

/// Register a scope-local LLM conditional-execution guardrail.
#[pyfunction]
fn scope_register_llm_conditional_execution_guardrail(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_register_llm_conditional_execution_guardrail(
        &uuid,
        name,
        priority,
        py_callable::wrap_py_llm_conditional_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered scope-local LLM conditional-execution guardrail.
#[pyfunction]
fn scope_deregister_llm_conditional_execution_guardrail(
    scope_uuid: &str,
    name: &str,
) -> PyResult<bool> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_deregister_llm_conditional_execution_guardrail(&uuid, name)
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Scope-local LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register a scope-local LLM request intercept.
#[pyfunction]
fn scope_register_llm_request_intercept(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    break_chain: bool,
    callable: Py<PyAny>,
) -> PyResult<()> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_register_llm_request_intercept(
        &uuid,
        name,
        priority,
        break_chain,
        py_callable::wrap_py_llm_request_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered scope-local LLM request intercept.
#[pyfunction]
fn scope_deregister_llm_request_intercept(scope_uuid: &str, name: &str) -> PyResult<bool> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_deregister_llm_request_intercept(&uuid, name).map_err(to_py_err)
}

/// Register a scope-local LLM execution intercept.
#[pyfunction]
fn scope_register_llm_execution_intercept(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    callable: Py<PyAny>,
) -> PyResult<()> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_register_llm_execution_intercept(
        &uuid,
        name,
        priority,
        py_callable::wrap_py_llm_exec_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered scope-local LLM execution intercept.
#[pyfunction]
fn scope_deregister_llm_execution_intercept(scope_uuid: &str, name: &str) -> PyResult<bool> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_deregister_llm_execution_intercept(&uuid, name).map_err(to_py_err)
}

/// Register a scope-local LLM stream-execution intercept.
#[pyfunction]
fn scope_register_llm_stream_execution_intercept(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    callable: Py<PyAny>,
) -> PyResult<()> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_register_llm_stream_execution_intercept(
        &uuid,
        name,
        priority,
        py_callable::wrap_py_llm_stream_exec_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered scope-local LLM stream-execution intercept.
#[pyfunction]
fn scope_deregister_llm_stream_execution_intercept(scope_uuid: &str, name: &str) -> PyResult<bool> {
    let uuid = parse_uuid(scope_uuid)?;
    core_registry_api::scope_deregister_llm_stream_execution_intercept(&uuid, name)
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Scope-local subscriber registrations
// ---------------------------------------------------------------------------

/// Register a scope-local event subscriber.
#[pyfunction]
fn scope_register_subscriber(scope_uuid: &str, name: &str, callback: Py<PyAny>) -> PyResult<()> {
    let uuid = parse_uuid(scope_uuid)?;
    core_subscriber_api::scope_register_subscriber(
        &uuid,
        name,
        py_callable::wrap_py_event_subscriber(callback),
    )
    .map_err(to_py_err)
}

/// Register an anonymous scope-local event subscriber and return a closeable handle.
#[pyfunction]
fn scope_subscribe(scope_uuid: &str, callback: Py<PyAny>) -> PyResult<PySubscription> {
    let uuid = parse_uuid(scope_uuid)?;
    core_subscriber_api::scope_subscribe(&uuid, py_callable::wrap_py_event_subscriber(callback))
        .map(|handle| PySubscription {
            inner: Some(PySubscriptionInner::Scope(handle)),
        })
        .map_err(to_py_err)
}

/// Remove a previously registered scope-local event subscriber.
#[pyfunction]
fn scope_deregister_subscriber(scope_uuid: &str, name: &str) -> PyResult<bool> {
    let uuid = parse_uuid(scope_uuid)?;
    core_subscriber_api::scope_deregister_subscriber(&uuid, name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register all API functions into the given `PyModule`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySubscription>()?;

    // Scope stack creation / binding / query
    m.add_function(wrap_pyfunction!(create_scope_stack, m)?)?;
    m.add_function(wrap_pyfunction!(set_thread_scope_stack, m)?)?;
    m.add_function(wrap_pyfunction!(sync_thread_scope_stack, m)?)?;
    m.add_function(wrap_pyfunction!(py_scope_stack_active, m)?)?;

    // Scope/handle ops
    m.add_function(wrap_pyfunction!(get_handle, m)?)?;
    m.add_function(wrap_pyfunction!(push_scope, m)?)?;
    m.add_function(wrap_pyfunction!(pop_scope, m)?)?;
    m.add_function(wrap_pyfunction!(event, m)?)?;

    // Tool lifecycle
    m.add_function(wrap_pyfunction!(tool_call, m)?)?;
    m.add_function(wrap_pyfunction!(tool_call_end, m)?)?;
    m.add_function(wrap_pyfunction!(tool_call_execute, m)?)?;

    // LLM lifecycle
    m.add_function(wrap_pyfunction!(llm_call, m)?)?;
    m.add_function(wrap_pyfunction!(llm_call_end, m)?)?;
    m.add_function(wrap_pyfunction!(llm_call_execute, m)?)?;
    m.add_function(wrap_pyfunction!(llm_stream_call_execute, m)?)?;

    // Tool guardrails
    m.add_function(wrap_pyfunction!(
        register_tool_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        deregister_tool_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        register_tool_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        deregister_tool_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        register_tool_conditional_execution_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        deregister_tool_conditional_execution_guardrail,
        m
    )?)?;

    // Tool intercepts
    m.add_function(wrap_pyfunction!(register_tool_request_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(deregister_tool_request_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(register_tool_execution_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(deregister_tool_execution_intercept, m)?)?;

    // LLM guardrails
    m.add_function(wrap_pyfunction!(
        register_llm_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        deregister_llm_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        register_llm_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        deregister_llm_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        register_llm_conditional_execution_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        deregister_llm_conditional_execution_guardrail,
        m
    )?)?;

    // LLM intercepts
    m.add_function(wrap_pyfunction!(register_llm_request_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(deregister_llm_request_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(register_llm_execution_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(deregister_llm_execution_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(
        register_llm_stream_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        deregister_llm_stream_execution_intercept,
        m
    )?)?;

    // Subscribers
    m.add_function(wrap_pyfunction!(register_subscriber, m)?)?;
    m.add_function(wrap_pyfunction!(subscribe, m)?)?;
    m.add_function(wrap_pyfunction!(deregister_subscriber, m)?)?;

    // Scope-local tool guardrails
    m.add_function(wrap_pyfunction!(
        scope_register_tool_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_deregister_tool_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_register_tool_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_deregister_tool_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_register_tool_conditional_execution_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_deregister_tool_conditional_execution_guardrail,
        m
    )?)?;

    // Scope-local tool intercepts
    m.add_function(wrap_pyfunction!(scope_register_tool_request_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(
        scope_deregister_tool_request_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_register_tool_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_deregister_tool_execution_intercept,
        m
    )?)?;

    // Scope-local LLM guardrails
    m.add_function(wrap_pyfunction!(
        scope_register_llm_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_deregister_llm_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_register_llm_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_deregister_llm_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_register_llm_conditional_execution_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_deregister_llm_conditional_execution_guardrail,
        m
    )?)?;

    // Scope-local LLM intercepts
    m.add_function(wrap_pyfunction!(scope_register_llm_request_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(scope_deregister_llm_request_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(scope_register_llm_execution_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(
        scope_deregister_llm_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_register_llm_stream_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        scope_deregister_llm_stream_execution_intercept,
        m
    )?)?;

    // Scope-local subscribers
    m.add_function(wrap_pyfunction!(scope_register_subscriber, m)?)?;
    m.add_function(wrap_pyfunction!(scope_subscribe, m)?)?;
    m.add_function(wrap_pyfunction!(scope_deregister_subscriber, m)?)?;

    // Standalone middleware chains
    m.add_function(wrap_pyfunction!(tool_request_intercepts, m)?)?;
    m.add_function(wrap_pyfunction!(tool_conditional_execution, m)?)?;
    m.add_function(wrap_pyfunction!(llm_request_intercepts, m)?)?;
    m.add_function(wrap_pyfunction!(llm_conditional_execution, m)?)?;

    Ok(())
}

#[cfg(test)]
#[path = "../../tests/coverage/py_api_coverage_tests.rs"]
mod coverage_tests;
