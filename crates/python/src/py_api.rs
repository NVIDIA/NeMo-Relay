// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing API functions for the Nexus runtime.
//!
//! Each `#[pyfunction]` here is registered into the `_native` module and
//! delegates to the corresponding function in [`nvidia_nat_nexus_core::api`].
//! The Python wrapper modules (`nat_nexus.scope`, `nat_nexus.tools`, etc.)
//! re-export these under shorter, idiomatic names.

use nvidia_nat_nexus_core as core;
use nvidia_nat_nexus_core::types as core_types;
use pyo3::prelude::*;
use tokio_stream::StreamExt;

use crate::convert::{json_to_py, opt_py_to_json, py_to_json};
use crate::py_callable;
use crate::py_types::*;

/// Convert an [`MagicError`] into a Python `RuntimeError`.
fn to_py_err(e: core::MagicError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
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
    PyScopeStack(nvidia_nat_nexus_core::create_scope_stack())
}

/// Bind a ``ScopeStack`` to the current thread's thread-local storage.
///
/// This ensures that subsequent Nexus API calls on this thread use the given
/// scope stack rather than a default one. Primarily useful when propagating
/// scope context into worker threads (e.g. ``ThreadPoolExecutor``).
///
/// Args:
///     stack: The ``ScopeStack`` to bind to the current thread.
#[pyfunction]
pub fn set_thread_scope_stack(stack: &PyScopeStack) {
    nvidia_nat_nexus_core::set_thread_scope_stack(stack.0.clone());
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Return the current scope handle from the task-local scope stack.
///
/// Returns the topmost `ScopeHandle` or raises `RuntimeError` if the
/// scope stack is empty.
#[pyfunction]
#[pyo3(signature = ())]
fn nat_nexus_get_handle() -> PyResult<PyScopeHandle> {
    core::nat_nexus_get_handle()
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
///     metadata: Optional JSON-serializable metadata to attach to the scope.
///
/// Returns:
///     The newly created ``ScopeHandle``.
///
/// Raises:
///     RuntimeError: If the scope stack is empty and no parent handle is given.
#[pyfunction]
#[pyo3(signature = (name, scope_type, *, handle=None, attributes=None, data=None, metadata=None))]
fn nat_nexus_push_scope(
    name: &str,
    scope_type: PyScopeType,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyScopeAttributes>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyScopeHandle> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(core_types::ScopeAttributes::empty());
    let d = opt_py_to_json(data)?;
    let meta = opt_py_to_json(metadata)?;
    core::nat_nexus_push_scope(
        name,
        scope_type.into(),
        handle.as_ref().map(|h| &h.inner),
        attrs,
        d,
        meta,
    )
    .map(PyScopeHandle::from)
    .map_err(to_py_err)
}

/// Remove a scope from the stack and emit an ``End`` event.
///
/// Args:
///     handle: The scope handle returned by ``push``.
///
/// Raises:
///     RuntimeError: If the scope is not found on the stack.
#[pyfunction]
fn nat_nexus_pop_scope(handle: &PyScopeHandle) -> PyResult<()> {
    core::nat_nexus_pop_scope(&handle.inner.uuid).map_err(to_py_err)
}

/// Emit a ``Mark`` event under the current or specified scope.
///
/// Args:
///     name: Event name.
///     handle: Optional parent scope handle. Defaults to current top of stack.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
#[pyfunction]
#[pyo3(signature = (name, *, handle=None, data=None, metadata=None))]
fn nat_nexus_event(
    name: &str,
    handle: Option<PyScopeHandle>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    core::nat_nexus_event(name, handle.as_ref().map(|h| &h.inner), data, metadata)
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begin a tool call — creates a ``ToolHandle`` and emits a ``Start`` event.
///
/// This is the manual (non-execute) entry point: callers are responsible
/// for invoking the tool themselves and later calling ``tool_call_end``.
///
/// Args:
///     name: Tool name.
///     args: JSON-serializable tool arguments.
///     handle: Optional parent scope handle.
///     attributes: Optional ``ToolAttributes`` bitflags.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
///
/// Returns:
///     A ``ToolHandle`` that must be passed to ``tool_call_end``.
#[pyfunction]
#[pyo3(signature = (name, args, *, handle=None, attributes=None, data=None, metadata=None, tool_call_id=None))]
fn nat_nexus_tool_call(
    name: &str,
    args: &Bound<'_, PyAny>,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyToolAttributes>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
    tool_call_id: Option<String>,
) -> PyResult<PyToolHandle> {
    let args_json = py_to_json(args)?;
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(core_types::ToolAttributes::empty());
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    core::nat_nexus_tool_call(
        name,
        args_json,
        handle.as_ref().map(|h| &h.inner),
        attrs,
        data,
        metadata,
        tool_call_id,
    )
    .map(PyToolHandle::from)
    .map_err(to_py_err)
}

/// End a tool call — records the result and emits an ``End`` event.
///
/// Args:
///     handle: The ``ToolHandle`` returned by ``tool_call``.
///     result: JSON-serializable tool result.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
#[pyfunction]
#[pyo3(signature = (handle, result, *, data=None, metadata=None))]
fn nat_nexus_tool_call_end(
    handle: &PyToolHandle,
    result: &Bound<'_, PyAny>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let result_json = py_to_json(result)?;
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    core::nat_nexus_tool_call_end(&handle.inner, result_json, data, metadata).map_err(to_py_err)
}

/// Execute a tool call through the full middleware pipeline.
///
/// Runs conditional-execution guardrails (on raw args) → request intercepts →
/// sanitize-request guardrails → execution intercepts → the supplied
/// function → response intercepts → sanitize-response guardrails, then
/// returns the final result. On rejection, only a standalone ``Mark`` event
/// is emitted (no ``Start``/``End`` pair) and ``GuardrailRejected`` is raised.
///
/// Args:
///     name: Tool name.
///     args: JSON-serializable tool arguments.
///     func: An async callable ``(args) -> result`` that performs the tool work.
///     handle: Optional parent scope handle.
///     attributes: Optional ``ToolAttributes`` bitflags.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
///
/// Returns:
///     An awaitable that resolves to the (possibly transformed) tool result.
#[pyfunction]
#[pyo3(signature = (name, args, func, *, handle=None, attributes=None, data=None, metadata=None))]
#[allow(clippy::too_many_arguments)]
fn nat_nexus_tool_call_execute<'py>(
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
        .unwrap_or(core_types::ToolAttributes::empty());
    let data_json = opt_py_to_json(data)?;
    let metadata_json = opt_py_to_json(metadata)?;
    let exec_fn = py_callable::wrap_py_tool_exec_fn(func);
    // Wrap the Fn-returning default callable into a FnOnce (ToolExecutionNextFn)
    let default_fn: nvidia_nat_nexus_core::ToolExecutionNextFn =
        Box::new(move |args| exec_fn(args));
    let parent_handle = handle.map(|h| h.inner).unwrap_or_else(core::task_scope_top);

    let scope_stack = nvidia_nat_nexus_core::current_scope_stack();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        nvidia_nat_nexus_core::TASK_SCOPE_STACK
            .scope(scope_stack, async move {
                let result = core::nat_nexus_tool_call_execute(
                    &name,
                    args_json,
                    default_fn,
                    Some(parent_handle),
                    attrs,
                    data_json,
                    metadata_json,
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

/// Begin an LLM call — creates an ``LLMHandle`` and emits a ``Start`` event.
///
/// This is the manual (non-execute) entry point: callers are responsible
/// for performing the LLM request themselves and later calling ``llm_call_end``.
///
/// Args:
///     name: Model/provider name.
///     request: An ``LLMRequest`` with headers and content.
///     handle: Optional parent scope handle.
///     attributes: Optional ``LLMAttributes`` bitflags.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
///
/// Returns:
///     An ``LLMHandle`` that must be passed to ``llm_call_end``.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (name, request, *, handle=None, attributes=None, data=None, metadata=None, model_name=None))]
fn nat_nexus_llm_call(
    name: &str,
    request: PyLLMRequest,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyLLMAttributes>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
    model_name: Option<String>,
) -> PyResult<PyLLMHandle> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(core_types::LLMAttributes::empty());
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    core::nat_nexus_llm_call(
        name,
        &request.inner,
        handle.as_ref().map(|h| &h.inner),
        attrs,
        data,
        metadata,
        model_name,
    )
    .map(PyLLMHandle::from)
    .map_err(to_py_err)
}

/// End an LLM call — records the response and emits an ``End`` event.
///
/// Args:
///     handle: The ``LLMHandle`` returned by ``llm_call``.
///     response: JSON-serializable LLM response.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
#[pyfunction]
#[pyo3(signature = (handle, response, *, data=None, metadata=None))]
fn nat_nexus_llm_call_end(
    handle: &PyLLMHandle,
    response: &Bound<'_, PyAny>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let response_json = py_to_json(response)?;
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    core::nat_nexus_llm_call_end(&handle.inner, response_json, data, metadata).map_err(to_py_err)
}

/// Execute an LLM call through the full middleware pipeline.
///
/// Runs conditional-execution guardrails (on raw request) → request intercepts →
/// sanitize-request guardrails → execution intercepts → the supplied
/// function → response intercepts → sanitize-response guardrails, then
/// returns the final response. On rejection, only a standalone ``Mark`` event
/// is emitted (no ``Start``/``End`` pair) and ``GuardrailRejected`` is raised.
///
/// Args:
///     name: Model/provider name.
///     request: An ``LLMRequest`` with headers and content.
///     func: An async callable ``(LLMRequest) -> dict`` that performs the LLM call.
///     handle: Optional parent scope handle.
///     attributes: Optional ``LLMAttributes`` bitflags.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
///
/// Returns:
///     An awaitable that resolves to the (possibly transformed) LLM response.
#[pyfunction]
#[pyo3(signature = (name, request, func, *, handle=None, attributes=None, data=None, metadata=None, model_name=None))]
#[allow(clippy::too_many_arguments)]
fn nat_nexus_llm_call_execute<'py>(
    py: Python<'py>,
    name: String,
    request: PyLLMRequest,
    func: Py<PyAny>,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyLLMAttributes>,
    data: Option<&Bound<'py, PyAny>>,
    metadata: Option<&Bound<'py, PyAny>>,
    model_name: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(core_types::LLMAttributes::empty());
    let data_json = opt_py_to_json(data)?;
    let metadata_json = opt_py_to_json(metadata)?;
    let exec_fn = py_callable::wrap_py_llm_exec_fn(func);
    let default_fn: nvidia_nat_nexus_core::LlmExecutionNextFn = Box::new(move |req| exec_fn(req));
    let parent_handle = handle.map(|h| h.inner).unwrap_or_else(core::task_scope_top);

    let scope_stack = nvidia_nat_nexus_core::current_scope_stack();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        nvidia_nat_nexus_core::TASK_SCOPE_STACK
            .scope(scope_stack, async move {
                let result = core::nat_nexus_llm_call_execute(
                    &name,
                    request.inner,
                    default_fn,
                    Some(parent_handle),
                    attrs,
                    data_json,
                    metadata_json,
                    model_name,
                )
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
/// ``LlmStreamWrapper`` so that stream-response intercepts can inspect or
/// transform each chunk in flight.
///
/// Args:
///     name: Model/provider name.
///     request: An ``LLMRequest`` with headers and content.
///     func: An async callable ``(LLMRequest) -> AsyncIterator[Any]`` that returns JSON chunks.
///     collector: A callable ``(Any) -> None`` invoked with each intercepted chunk
///         (after stream response intercepts have been applied).
///     finalizer: A callable ``() -> Any`` invoked once when the stream is exhausted.
///         Its return value is the aggregated response (converted to JSON).
///     handle: Optional parent scope handle.
///     attributes: Optional ``LLMAttributes`` bitflags.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
///
/// Returns:
///     An awaitable that resolves to an ``LlmStream`` async iterator of JSON chunks.
#[pyfunction]
#[pyo3(signature = (name, request, func, collector, finalizer, *, handle=None, attributes=None, data=None, metadata=None, model_name=None))]
#[allow(clippy::too_many_arguments)]
fn nat_nexus_llm_stream_call_execute<'py>(
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
) -> PyResult<Bound<'py, PyAny>> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(core_types::LLMAttributes::empty());
    let data_json = opt_py_to_json(data)?;
    let metadata_json = opt_py_to_json(metadata)?;
    let exec_fn = py_callable::wrap_py_llm_stream_exec_fn(func);
    let default_fn: nvidia_nat_nexus_core::LlmStreamExecutionNextFn =
        Box::new(move |req| exec_fn(req));
    let collector_fn = py_callable::wrap_py_collector_fn(collector);
    let finalizer_fn = py_callable::wrap_py_finalizer_fn(finalizer);
    let parent_handle = handle.map(|h| h.inner).unwrap_or_else(core::task_scope_top);

    let scope_stack = nvidia_nat_nexus_core::current_scope_stack();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        nvidia_nat_nexus_core::TASK_SCOPE_STACK
            .scope(scope_stack, async move {
                let rust_stream = core::nat_nexus_llm_stream_call_execute(
                    &name,
                    request.inner,
                    default_fn,
                    collector_fn,
                    finalizer_fn,
                    Some(parent_handle),
                    attrs,
                    data_json,
                    metadata_json,
                    model_name,
                )
                .await
                .map_err(to_py_err)?;

                // Spawn a tokio task that drains the Rust stream into an mpsc channel
                let (tx, rx) = tokio::sync::mpsc::channel::<
                    nvidia_nat_nexus_core::Result<serde_json::Value>,
                >(32);
                tokio::spawn(async move {
                    let mut stream = rust_stream;
                    while let Some(item) = stream.next().await {
                        if tx.send(item).await.is_err() {
                            break; // receiver dropped
                        }
                    }
                });

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
    nat_nexus_register_tool_sanitize_request_guardrail,
    nat_nexus_deregister_tool_sanitize_request_guardrail,
    core::nat_nexus_register_tool_sanitize_request_guardrail,
    core::nat_nexus_deregister_tool_sanitize_request_guardrail,
    py_callable::wrap_py_tool_fn
);

py_guardrail_tool_api!(
    /// Register a tool sanitize-response guardrail.
    ///
    /// Callback: ``(tool_name: str, result: Any) -> Any`` — returns sanitized result.
    nat_nexus_register_tool_sanitize_response_guardrail,
    nat_nexus_deregister_tool_sanitize_response_guardrail,
    core::nat_nexus_register_tool_sanitize_response_guardrail,
    core::nat_nexus_deregister_tool_sanitize_response_guardrail,
    py_callable::wrap_py_tool_fn
);

/// Register a tool conditional-execution guardrail.
///
/// Callback: ``(tool_name: str, args: Any) -> Optional[str]``.
/// Return ``None`` to allow execution, or a rejection reason string to block it.
#[pyfunction]
fn nat_nexus_register_tool_conditional_execution_guardrail(
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    core::nat_nexus_register_tool_conditional_execution_guardrail(
        name,
        priority,
        py_callable::wrap_py_tool_conditional_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered tool conditional-execution guardrail.
#[pyfunction]
fn nat_nexus_deregister_tool_conditional_execution_guardrail(name: &str) -> PyResult<bool> {
    core::nat_nexus_deregister_tool_conditional_execution_guardrail(name).map_err(to_py_err)
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
    nat_nexus_register_tool_request_intercept,
    nat_nexus_deregister_tool_request_intercept,
    core::nat_nexus_register_tool_request_intercept,
    core::nat_nexus_deregister_tool_request_intercept,
    py_callable::wrap_py_tool_fn
);

py_intercept_tool_api!(
    /// Register a tool response intercept.
    ///
    /// Callback: ``(tool_name: str, result: Any) -> Any`` — transforms tool result.
    /// If ``break_chain`` is ``True``, no lower-priority intercepts run after this one.
    nat_nexus_register_tool_response_intercept,
    nat_nexus_deregister_tool_response_intercept,
    core::nat_nexus_register_tool_response_intercept,
    core::nat_nexus_deregister_tool_response_intercept,
    py_callable::wrap_py_tool_fn
);

/// Register a tool execution intercept that can replace the tool function.
///
/// ``callable``: ``async (args: Any, next) -> Any`` — middleware intercept function.
/// Call ``await next(args)`` to invoke the next intercept or original
/// implementation; skip calling ``next`` to short-circuit.
#[pyfunction]
fn nat_nexus_register_tool_execution_intercept(
    name: &str,
    priority: i32,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core::nat_nexus_register_tool_execution_intercept(
        name,
        priority,
        py_callable::wrap_py_tool_exec_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered tool execution intercept.
#[pyfunction]
fn nat_nexus_deregister_tool_execution_intercept(name: &str) -> PyResult<bool> {
    core::nat_nexus_deregister_tool_execution_intercept(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register an LLM sanitize-request guardrail.
///
/// Callback: ``(request: LLMRequest) -> LLMRequest`` — returns a sanitized request.
#[pyfunction]
fn nat_nexus_register_llm_sanitize_request_guardrail(
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    core::nat_nexus_register_llm_sanitize_request_guardrail(
        name,
        priority,
        py_callable::wrap_py_llm_sanitize_request_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM sanitize-request guardrail.
#[pyfunction]
fn nat_nexus_deregister_llm_sanitize_request_guardrail(name: &str) -> PyResult<bool> {
    core::nat_nexus_deregister_llm_sanitize_request_guardrail(name).map_err(to_py_err)
}

/// Register an LLM sanitize-response guardrail.
///
/// Callback: ``(response: dict) -> dict`` — returns a sanitized response.
#[pyfunction]
fn nat_nexus_register_llm_sanitize_response_guardrail(
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    core::nat_nexus_register_llm_sanitize_response_guardrail(
        name,
        priority,
        py_callable::wrap_py_llm_sanitize_response_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM sanitize-response guardrail.
#[pyfunction]
fn nat_nexus_deregister_llm_sanitize_response_guardrail(name: &str) -> PyResult<bool> {
    core::nat_nexus_deregister_llm_sanitize_response_guardrail(name).map_err(to_py_err)
}

/// Register an LLM conditional-execution guardrail.
///
/// Callback: ``(request: LLMRequest) -> Optional[str]``.
/// Return ``None`` to allow execution, or a rejection reason string to block it.
#[pyfunction]
fn nat_nexus_register_llm_conditional_execution_guardrail(
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    core::nat_nexus_register_llm_conditional_execution_guardrail(
        name,
        priority,
        py_callable::wrap_py_llm_conditional_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM conditional-execution guardrail.
#[pyfunction]
fn nat_nexus_deregister_llm_conditional_execution_guardrail(name: &str) -> PyResult<bool> {
    core::nat_nexus_deregister_llm_conditional_execution_guardrail(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register an LLM request intercept.
///
/// Callback: ``(request: LLMRequest) -> LLMRequest`` — transforms the LLM request.
/// If ``break_chain`` is ``True``, no lower-priority intercepts run after this one.
#[pyfunction]
fn nat_nexus_register_llm_request_intercept(
    name: &str,
    priority: i32,
    break_chain: bool,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core::nat_nexus_register_llm_request_intercept(
        name,
        priority,
        break_chain,
        py_callable::wrap_py_llm_request_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM request intercept.
#[pyfunction]
fn nat_nexus_deregister_llm_request_intercept(name: &str) -> PyResult<bool> {
    core::nat_nexus_deregister_llm_request_intercept(name).map_err(to_py_err)
}

/// Register an LLM execution intercept that can replace the LLM call.
///
/// ``callable``: ``async (native: Any, next) -> Any`` — middleware intercept function.
/// Call ``await next(native)`` to invoke the next intercept or original
/// implementation; skip calling ``next`` to short-circuit.
#[pyfunction]
fn nat_nexus_register_llm_execution_intercept(
    name: &str,
    priority: i32,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core::nat_nexus_register_llm_execution_intercept(
        name,
        priority,
        py_callable::wrap_py_llm_exec_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM execution intercept.
#[pyfunction]
fn nat_nexus_deregister_llm_execution_intercept(name: &str) -> PyResult<bool> {
    core::nat_nexus_deregister_llm_execution_intercept(name).map_err(to_py_err)
}

/// Register an LLM stream-execution intercept that can replace the streaming LLM call.
///
/// ``callable``: ``async (native: Any, next) -> AsyncIterator[Any]`` —
/// middleware streaming intercept function.
/// Call ``await next(native)`` to invoke the next intercept or original
/// streaming implementation; skip calling ``next`` to short-circuit.
#[pyfunction]
fn nat_nexus_register_llm_stream_execution_intercept(
    name: &str,
    priority: i32,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core::nat_nexus_register_llm_stream_execution_intercept(
        name,
        priority,
        py_callable::wrap_py_llm_stream_exec_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM stream-execution intercept.
#[pyfunction]
fn nat_nexus_deregister_llm_stream_execution_intercept(name: &str) -> PyResult<bool> {
    core::nat_nexus_deregister_llm_stream_execution_intercept(name).map_err(to_py_err)
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
fn nat_nexus_tool_request_intercepts<'py>(
    py: Python<'py>,
    name: &str,
    args: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let args_json = py_to_json(args)?;
    let result = core::nat_nexus_tool_request_intercepts(name, args_json).map_err(to_py_err)?;
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
fn nat_nexus_tool_conditional_execution(name: &str, args: &Bound<'_, PyAny>) -> PyResult<()> {
    let args_json = py_to_json(args)?;
    core::nat_nexus_tool_conditional_execution(name, &args_json).map_err(to_py_err)
}

/// Run the registered tool response intercept chain on the given result.
///
/// Returns the transformed result after all intercepts have been applied.
///
/// Args:
///     name: Tool name.
///     result: Tool result (any JSON-serializable object).
///
/// Returns:
///     The (possibly transformed) result.
#[pyfunction]
fn nat_nexus_tool_response_intercepts<'py>(
    py: Python<'py>,
    name: &str,
    result: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let result_json = py_to_json(result)?;
    let transformed =
        core::nat_nexus_tool_response_intercepts(name, result_json).map_err(to_py_err)?;
    json_to_py(py, &transformed)
}

/// Run the registered LLM request intercept chain on the given request.
///
/// Returns the transformed request after all intercepts have been applied.
///
/// Args:
///     request: An ``LLMRequest`` object.
///
/// Returns:
///     The (possibly transformed) ``LLMRequest``.
#[pyfunction]
fn nat_nexus_llm_request_intercepts(request: PyLLMRequest) -> PyResult<PyLLMRequest> {
    let result = core::nat_nexus_llm_request_intercepts(request.inner).map_err(to_py_err)?;
    Ok(PyLLMRequest { inner: result })
}

/// Run the registered LLM conditional execution guardrail chain.
///
/// Raises ``RuntimeError`` with the rejection reason if any guardrail rejects.
///
/// Args:
///     request: An ``LLMRequest`` object.
#[pyfunction]
fn nat_nexus_llm_conditional_execution(request: PyLLMRequest) -> PyResult<()> {
    core::nat_nexus_llm_conditional_execution(&request.inner).map_err(to_py_err)
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
fn nat_nexus_register_subscriber(name: &str, callback: Py<PyAny>) -> PyResult<()> {
    core::nat_nexus_register_subscriber(name, py_callable::wrap_py_event_subscriber(callback))
        .map_err(to_py_err)
}

/// Remove a previously registered event subscriber.
///
/// Returns ``True`` if a subscriber with that name was found and removed.
#[pyfunction]
fn nat_nexus_deregister_subscriber(name: &str) -> PyResult<bool> {
    core::nat_nexus_deregister_subscriber(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register all API functions into the given `PyModule`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Scope stack creation / binding
    m.add_function(wrap_pyfunction!(create_scope_stack, m)?)?;
    m.add_function(wrap_pyfunction!(set_thread_scope_stack, m)?)?;

    // Scope/handle ops
    m.add_function(wrap_pyfunction!(nat_nexus_get_handle, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_push_scope, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_pop_scope, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_event, m)?)?;

    // Tool lifecycle
    m.add_function(wrap_pyfunction!(nat_nexus_tool_call, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_tool_call_end, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_tool_call_execute, m)?)?;

    // LLM lifecycle
    m.add_function(wrap_pyfunction!(nat_nexus_llm_call, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_llm_call_end, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_llm_call_execute, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_llm_stream_call_execute, m)?)?;

    // Tool guardrails
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_tool_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_tool_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_tool_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_tool_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_tool_conditional_execution_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_tool_conditional_execution_guardrail,
        m
    )?)?;

    // Tool intercepts
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_tool_request_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_tool_request_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_tool_response_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_tool_response_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_tool_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_tool_execution_intercept,
        m
    )?)?;

    // LLM guardrails
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_llm_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_llm_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_llm_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_llm_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_llm_conditional_execution_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_llm_conditional_execution_guardrail,
        m
    )?)?;

    // LLM intercepts
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_llm_request_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_llm_request_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_llm_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_llm_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_register_llm_stream_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nat_nexus_deregister_llm_stream_execution_intercept,
        m
    )?)?;

    // Subscribers
    m.add_function(wrap_pyfunction!(nat_nexus_register_subscriber, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_deregister_subscriber, m)?)?;

    // Standalone middleware chains
    m.add_function(wrap_pyfunction!(nat_nexus_tool_request_intercepts, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_tool_conditional_execution, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_tool_response_intercepts, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_llm_request_intercepts, m)?)?;
    m.add_function(wrap_pyfunction!(nat_nexus_llm_conditional_execution, m)?)?;

    Ok(())
}
