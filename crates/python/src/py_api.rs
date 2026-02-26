// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing API functions for the NVAgentRT runtime.
//!
//! Each `#[pyfunction]` here is registered into the `_native` module and
//! delegates to the corresponding function in [`nvagentrt_core::api`].
//! The Python wrapper modules (`nvagentrt.scope`, `nvagentrt.tools`, etc.)
//! re-export these under shorter, idiomatic names.

use nvagentrt_core as core;
use nvagentrt_core::types as core_types;
use pyo3::prelude::*;
use tokio_stream::StreamExt;

use crate::convert::{json_to_py, opt_py_to_json, py_to_json};
use crate::py_callable;
use crate::py_types::*;

/// Convert an [`AgentRtError`] into a Python `RuntimeError`.
fn to_py_err(e: core::AgentRtError) -> PyErr {
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
    PyScopeStack(nvagentrt_core::create_scope_stack())
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
fn nvagentrt_get_handle() -> PyResult<PyScopeHandle> {
    core::nvagentrt_get_handle()
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
///
/// Returns:
///     The newly created ``ScopeHandle``.
///
/// Raises:
///     RuntimeError: If the scope stack is empty and no parent handle is given.
#[pyfunction]
#[pyo3(signature = (name, scope_type, *, handle=None, attributes=None))]
fn nvagentrt_push_scope(
    name: &str,
    scope_type: PyScopeType,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyScopeAttributes>,
) -> PyResult<PyScopeHandle> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(core_types::ScopeAttributes::empty());
    core::nvagentrt_push_scope(
        name,
        scope_type.into(),
        handle.as_ref().map(|h| &h.inner),
        attrs,
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
fn nvagentrt_pop_scope(handle: &PyScopeHandle) -> PyResult<()> {
    core::nvagentrt_pop_scope(&handle.inner.uuid).map_err(to_py_err)
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
fn nvagentrt_event(
    name: &str,
    handle: Option<PyScopeHandle>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    core::nvagentrt_event(name, handle.as_ref().map(|h| &h.inner), data, metadata)
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
#[pyo3(signature = (name, args, *, handle=None, attributes=None, data=None, metadata=None))]
fn nvagentrt_tool_call(
    name: &str,
    args: &Bound<'_, PyAny>,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyToolAttributes>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyToolHandle> {
    let args_json = py_to_json(args)?;
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(core_types::ToolAttributes::empty());
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    core::nvagentrt_tool_call(
        name,
        args_json,
        handle.as_ref().map(|h| &h.inner),
        attrs,
        data,
        metadata,
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
fn nvagentrt_tool_call_end(
    handle: &PyToolHandle,
    result: &Bound<'_, PyAny>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let result_json = py_to_json(result)?;
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    core::nvagentrt_tool_call_end(&handle.inner, result_json, data, metadata).map_err(to_py_err)
}

/// Execute a tool call through the full middleware pipeline.
///
/// Runs request intercepts → sanitize-request guardrails →
/// conditional-execution guardrails → execution intercepts → the supplied
/// function → response intercepts → sanitize-response guardrails, then
/// returns the final result.
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
fn nvagentrt_tool_call_execute<'py>(
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
    let parent_handle = handle.map(|h| h.inner).unwrap_or_else(core::task_scope_top);

    let scope_stack = nvagentrt_core::current_scope_stack();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        nvagentrt_core::TASK_SCOPE_STACK
            .scope(scope_stack, async move {
                let result = core::nvagentrt_tool_call_execute(
                    &name,
                    args_json,
                    exec_fn,
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
///     request: An ``LLMRequest`` describing the HTTP call.
///     handle: Optional parent scope handle.
///     attributes: Optional ``LLMAttributes`` bitflags.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
///
/// Returns:
///     An ``LLMHandle`` that must be passed to ``llm_call_end``.
#[pyfunction]
#[pyo3(signature = (name, request, *, handle=None, attributes=None, data=None, metadata=None))]
fn nvagentrt_llm_call(
    name: &str,
    request: &PyLLMRequest,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyLLMAttributes>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyLLMHandle> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(core_types::LLMAttributes::empty());
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    core::nvagentrt_llm_call(
        name,
        &request.inner,
        handle.as_ref().map(|h| &h.inner),
        attrs,
        data,
        metadata,
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
fn nvagentrt_llm_call_end(
    handle: &PyLLMHandle,
    response: &Bound<'_, PyAny>,
    data: Option<&Bound<'_, PyAny>>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let response_json = py_to_json(response)?;
    let data = opt_py_to_json(data)?;
    let metadata = opt_py_to_json(metadata)?;
    core::nvagentrt_llm_call_end(&handle.inner, response_json, data, metadata).map_err(to_py_err)
}

/// Execute an LLM call through the full middleware pipeline.
///
/// Runs request intercepts → sanitize-request guardrails →
/// conditional-execution guardrails → execution intercepts → the supplied
/// function → response intercepts → sanitize-response guardrails, then
/// returns the final response.
///
/// Args:
///     name: Model/provider name.
///     request: An ``LLMRequest`` describing the HTTP call.
///     func: An async callable ``(LLMRequest) -> response`` that performs the LLM call.
///     handle: Optional parent scope handle.
///     attributes: Optional ``LLMAttributes`` bitflags.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
///
/// Returns:
///     An awaitable that resolves to the (possibly transformed) LLM response.
#[pyfunction]
#[pyo3(signature = (name, request, func, *, handle=None, attributes=None, data=None, metadata=None))]
#[allow(clippy::too_many_arguments)]
fn nvagentrt_llm_call_execute<'py>(
    py: Python<'py>,
    name: String,
    request: PyLLMRequest,
    func: Py<PyAny>,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyLLMAttributes>,
    data: Option<&Bound<'py, PyAny>>,
    metadata: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(core_types::LLMAttributes::empty());
    let data_json = opt_py_to_json(data)?;
    let metadata_json = opt_py_to_json(metadata)?;
    let exec_fn = py_callable::wrap_py_llm_exec_fn(func);
    let parent_handle = handle.map(|h| h.inner).unwrap_or_else(core::task_scope_top);

    let scope_stack = nvagentrt_core::current_scope_stack();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        nvagentrt_core::TASK_SCOPE_STACK
            .scope(scope_stack, async move {
                let result = core::nvagentrt_llm_call_execute(
                    &name,
                    request.inner,
                    exec_fn,
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

/// Execute a streaming LLM call through the full middleware pipeline.
///
/// Like ``llm_call_execute`` but the execution function returns an async
/// iterator of SSE chunks. The runtime wraps the stream with
/// ``LlmStreamWrapper`` so that stream-response intercepts can inspect or
/// transform each SSE event in flight.
///
/// Args:
///     name: Model/provider name.
///     request: An ``LLMRequest`` describing the HTTP call.
///     func: An async callable ``(LLMRequest) -> AsyncIterator[str]`` that returns raw SSE text chunks.
///     handle: Optional parent scope handle.
///     attributes: Optional ``LLMAttributes`` bitflags.
///     data: Optional JSON-serializable application data.
///     metadata: Optional JSON-serializable metadata.
///
/// Returns:
///     An awaitable that resolves to an ``LlmStream`` async iterator of SSE text chunks.
#[pyfunction]
#[pyo3(signature = (name, request, func, *, handle=None, attributes=None, data=None, metadata=None))]
#[allow(clippy::too_many_arguments)]
fn nvagentrt_llm_stream_call_execute<'py>(
    py: Python<'py>,
    name: String,
    request: PyLLMRequest,
    func: Py<PyAny>,
    handle: Option<PyScopeHandle>,
    attributes: Option<PyLLMAttributes>,
    data: Option<&Bound<'py, PyAny>>,
    metadata: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let attrs = attributes
        .map(|a| a.inner)
        .unwrap_or(core_types::LLMAttributes::empty());
    let data_json = opt_py_to_json(data)?;
    let metadata_json = opt_py_to_json(metadata)?;
    let exec_fn = py_callable::wrap_py_llm_stream_exec_fn(func);
    let parent_handle = handle.map(|h| h.inner).unwrap_or_else(core::task_scope_top);

    let scope_stack = nvagentrt_core::current_scope_stack();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        nvagentrt_core::TASK_SCOPE_STACK
            .scope(scope_stack, async move {
                let rust_stream = core::nvagentrt_llm_stream_call_execute(
                    &name,
                    request.inner,
                    exec_fn,
                    Some(parent_handle),
                    attrs,
                    data_json,
                    metadata_json,
                )
                .await
                .map_err(to_py_err)?;

                // Spawn a tokio task that drains the Rust stream into an mpsc channel
                let (tx, rx) = tokio::sync::mpsc::channel::<nvagentrt_core::Result<String>>(32);
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
    ($register_name:ident, $deregister_name:ident, $core_register:path, $core_deregister:path, $wrapper:path) => {
        #[pyfunction]
        fn $register_name(name: &str, priority: i32, guardrail: Py<PyAny>) -> PyResult<()> {
            $core_register(name, priority, $wrapper(guardrail)).map_err(to_py_err)
        }

        #[pyfunction]
        fn $deregister_name(name: &str) -> PyResult<bool> {
            $core_deregister(name).map_err(to_py_err)
        }
    };
}

// Register a tool sanitize-request guardrail.
//
// Callback: ``(tool_name: str, args: Any) -> Any`` — returns sanitized args.
//
// Deregister with ``deregister_tool_sanitize_request_guardrail``.
py_guardrail_tool_api!(
    nvagentrt_register_tool_sanitize_request_guardrail,
    nvagentrt_deregister_tool_sanitize_request_guardrail,
    core::nvagentrt_register_tool_sanitize_request_guardrail,
    core::nvagentrt_deregister_tool_sanitize_request_guardrail,
    py_callable::wrap_py_tool_fn
);

// Register a tool sanitize-response guardrail.
//
// Callback: ``(tool_name: str, result: Any) -> Any`` — returns sanitized result.
//
// Deregister with ``deregister_tool_sanitize_response_guardrail``.
py_guardrail_tool_api!(
    nvagentrt_register_tool_sanitize_response_guardrail,
    nvagentrt_deregister_tool_sanitize_response_guardrail,
    core::nvagentrt_register_tool_sanitize_response_guardrail,
    core::nvagentrt_deregister_tool_sanitize_response_guardrail,
    py_callable::wrap_py_tool_fn
);

/// Register a tool conditional-execution guardrail.
///
/// Callback: ``(tool_name: str, args: Any) -> Optional[str]``.
/// Return ``None`` to allow execution, or a rejection reason string to block it.
#[pyfunction]
fn nvagentrt_register_tool_conditional_execution_guardrail(
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    core::nvagentrt_register_tool_conditional_execution_guardrail(
        name,
        priority,
        py_callable::wrap_py_tool_conditional_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered tool conditional-execution guardrail.
#[pyfunction]
fn nvagentrt_deregister_tool_conditional_execution_guardrail(name: &str) -> PyResult<bool> {
    core::nvagentrt_deregister_tool_conditional_execution_guardrail(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Tool intercept registrations
// ---------------------------------------------------------------------------

/// Macro that generates a register/deregister pair for tool intercepts
/// whose callback signature is `(tool_name: str, json: Any) -> Any`.
macro_rules! py_intercept_tool_api {
    ($register_name:ident, $deregister_name:ident, $core_register:path, $core_deregister:path, $wrapper:path) => {
        #[pyfunction]
        fn $register_name(
            name: &str,
            priority: i32,
            break_chain: bool,
            callable: Py<PyAny>,
        ) -> PyResult<()> {
            $core_register(name, priority, break_chain, $wrapper(callable)).map_err(to_py_err)
        }

        #[pyfunction]
        fn $deregister_name(name: &str) -> PyResult<bool> {
            $core_deregister(name).map_err(to_py_err)
        }
    };
}

// Register a tool request intercept.
//
// Callback: ``(tool_name: str, args: Any) -> Any`` — transforms tool arguments.
// If ``break_chain`` is ``True``, no lower-priority intercepts run after this one.
py_intercept_tool_api!(
    nvagentrt_register_tool_request_intercept,
    nvagentrt_deregister_tool_request_intercept,
    core::nvagentrt_register_tool_request_intercept,
    core::nvagentrt_deregister_tool_request_intercept,
    py_callable::wrap_py_tool_fn
);

// Register a tool response intercept.
//
// Callback: ``(tool_name: str, result: Any) -> Any`` — transforms tool result.
// If ``break_chain`` is ``True``, no lower-priority intercepts run after this one.
py_intercept_tool_api!(
    nvagentrt_register_tool_response_intercept,
    nvagentrt_deregister_tool_response_intercept,
    core::nvagentrt_register_tool_response_intercept,
    core::nvagentrt_deregister_tool_response_intercept,
    py_callable::wrap_py_tool_fn
);

/// Register a tool execution intercept that can replace the tool function.
///
/// ``conditional``: ``(tool_name: str, args: Any) -> bool`` — return ``True``
/// to activate this intercept for the given call.
///
/// ``callable``: ``async (args: Any) -> Any`` — replacement execution function.
#[pyfunction]
fn nvagentrt_register_tool_execution_intercept(
    name: &str,
    priority: i32,
    conditional: Py<PyAny>,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core::nvagentrt_register_tool_execution_intercept(
        name,
        priority,
        py_callable::wrap_py_tool_exec_conditional_fn(conditional),
        py_callable::wrap_py_tool_exec_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered tool execution intercept.
#[pyfunction]
fn nvagentrt_deregister_tool_execution_intercept(name: &str) -> PyResult<bool> {
    core::nvagentrt_deregister_tool_execution_intercept(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register an LLM sanitize-request guardrail.
///
/// Callback: ``(request: LLMRequest) -> LLMRequest`` — returns a sanitized request.
#[pyfunction]
fn nvagentrt_register_llm_sanitize_request_guardrail(
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    core::nvagentrt_register_llm_sanitize_request_guardrail(
        name,
        priority,
        py_callable::wrap_py_llm_sanitize_request_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM sanitize-request guardrail.
#[pyfunction]
fn nvagentrt_deregister_llm_sanitize_request_guardrail(name: &str) -> PyResult<bool> {
    core::nvagentrt_deregister_llm_sanitize_request_guardrail(name).map_err(to_py_err)
}

/// Macro that generates a register/deregister pair for guardrails
/// whose callback signature is `(json: Any) -> Any`.
macro_rules! py_guardrail_json_api {
    ($register_name:ident, $deregister_name:ident, $core_register:path, $core_deregister:path) => {
        #[pyfunction]
        fn $register_name(name: &str, priority: i32, guardrail: Py<PyAny>) -> PyResult<()> {
            $core_register(name, priority, py_callable::wrap_py_json_fn(guardrail))
                .map_err(to_py_err)
        }

        #[pyfunction]
        fn $deregister_name(name: &str) -> PyResult<bool> {
            $core_deregister(name).map_err(to_py_err)
        }
    };
}

// Register an LLM sanitize-response guardrail.
//
// Callback: ``(response: Any) -> Any`` — returns a sanitized response.
py_guardrail_json_api!(
    nvagentrt_register_llm_sanitize_response_guardrail,
    nvagentrt_deregister_llm_sanitize_response_guardrail,
    core::nvagentrt_register_llm_sanitize_response_guardrail,
    core::nvagentrt_deregister_llm_sanitize_response_guardrail
);

/// Register an LLM conditional-execution guardrail.
///
/// Callback: ``(request: LLMRequest) -> Optional[str]``.
/// Return ``None`` to allow execution, or a rejection reason string to block it.
#[pyfunction]
fn nvagentrt_register_llm_conditional_execution_guardrail(
    name: &str,
    priority: i32,
    guardrail: Py<PyAny>,
) -> PyResult<()> {
    core::nvagentrt_register_llm_conditional_execution_guardrail(
        name,
        priority,
        py_callable::wrap_py_llm_conditional_fn(guardrail),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM conditional-execution guardrail.
#[pyfunction]
fn nvagentrt_deregister_llm_conditional_execution_guardrail(name: &str) -> PyResult<bool> {
    core::nvagentrt_deregister_llm_conditional_execution_guardrail(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register an LLM request intercept.
///
/// Callback: ``(request: LLMRequest) -> LLMRequest`` — transforms the request.
/// If ``break_chain`` is ``True``, no lower-priority intercepts run after this one.
#[pyfunction]
fn nvagentrt_register_llm_request_intercept(
    name: &str,
    priority: i32,
    break_chain: bool,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core::nvagentrt_register_llm_request_intercept(
        name,
        priority,
        break_chain,
        py_callable::wrap_py_llm_request_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM request intercept.
#[pyfunction]
fn nvagentrt_deregister_llm_request_intercept(name: &str) -> PyResult<bool> {
    core::nvagentrt_deregister_llm_request_intercept(name).map_err(to_py_err)
}

/// Register an LLM response intercept.
///
/// Callback: ``(response: Any) -> Any`` — transforms the LLM response.
/// If ``break_chain`` is ``True``, no lower-priority intercepts run after this one.
#[pyfunction]
fn nvagentrt_register_llm_response_intercept(
    name: &str,
    priority: i32,
    break_chain: bool,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core::nvagentrt_register_llm_response_intercept(
        name,
        priority,
        break_chain,
        py_callable::wrap_py_json_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM response intercept.
#[pyfunction]
fn nvagentrt_deregister_llm_response_intercept(name: &str) -> PyResult<bool> {
    core::nvagentrt_deregister_llm_response_intercept(name).map_err(to_py_err)
}

/// Register an LLM stream-response intercept.
///
/// Callback: ``(event: SseEvent) -> SseEvent`` — transforms each SSE event in a stream.
/// If ``break_chain`` is ``True``, no lower-priority intercepts run after this one.
#[pyfunction]
fn nvagentrt_register_llm_stream_response_intercept(
    name: &str,
    priority: i32,
    break_chain: bool,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core::nvagentrt_register_llm_stream_response_intercept(
        name,
        priority,
        break_chain,
        py_callable::wrap_py_sse_intercept_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM stream-response intercept.
#[pyfunction]
fn nvagentrt_deregister_llm_stream_response_intercept(name: &str) -> PyResult<bool> {
    core::nvagentrt_deregister_llm_stream_response_intercept(name).map_err(to_py_err)
}

/// Register an LLM execution intercept that can replace the LLM call.
///
/// ``conditional``: ``(request: LLMRequest) -> bool`` — return ``True``
/// to activate this intercept for the given call.
///
/// ``callable``: ``async (request: LLMRequest) -> Any`` — replacement execution function.
#[pyfunction]
fn nvagentrt_register_llm_execution_intercept(
    name: &str,
    priority: i32,
    conditional: Py<PyAny>,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core::nvagentrt_register_llm_execution_intercept(
        name,
        priority,
        py_callable::wrap_py_llm_exec_conditional_fn(conditional),
        py_callable::wrap_py_llm_exec_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM execution intercept.
#[pyfunction]
fn nvagentrt_deregister_llm_execution_intercept(name: &str) -> PyResult<bool> {
    core::nvagentrt_deregister_llm_execution_intercept(name).map_err(to_py_err)
}

/// Register an LLM stream-execution intercept that can replace the streaming LLM call.
///
/// ``conditional``: ``(request: LLMRequest) -> bool`` — return ``True``
/// to activate this intercept for the given call.
///
/// ``callable``: ``async (request: LLMRequest) -> AsyncIterator[str]`` —
/// replacement streaming execution function.
#[pyfunction]
fn nvagentrt_register_llm_stream_execution_intercept(
    name: &str,
    priority: i32,
    conditional: Py<PyAny>,
    callable: Py<PyAny>,
) -> PyResult<()> {
    core::nvagentrt_register_llm_stream_execution_intercept(
        name,
        priority,
        py_callable::wrap_py_llm_exec_conditional_fn(conditional),
        py_callable::wrap_py_llm_stream_exec_fn(callable),
    )
    .map_err(to_py_err)
}

/// Remove a previously registered LLM stream-execution intercept.
#[pyfunction]
fn nvagentrt_deregister_llm_stream_execution_intercept(name: &str) -> PyResult<bool> {
    core::nvagentrt_deregister_llm_stream_execution_intercept(name).map_err(to_py_err)
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
fn nvagentrt_register_subscriber(name: &str, callback: Py<PyAny>) -> PyResult<()> {
    core::nvagentrt_register_subscriber(name, py_callable::wrap_py_event_subscriber(callback))
        .map_err(to_py_err)
}

/// Remove a previously registered event subscriber.
///
/// Returns ``True`` if a subscriber with that name was found and removed.
#[pyfunction]
fn nvagentrt_deregister_subscriber(name: &str) -> PyResult<bool> {
    core::nvagentrt_deregister_subscriber(name).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register all API functions into the given `PyModule`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Scope stack creation
    m.add_function(wrap_pyfunction!(create_scope_stack, m)?)?;

    // Scope/handle ops
    m.add_function(wrap_pyfunction!(nvagentrt_get_handle, m)?)?;
    m.add_function(wrap_pyfunction!(nvagentrt_push_scope, m)?)?;
    m.add_function(wrap_pyfunction!(nvagentrt_pop_scope, m)?)?;
    m.add_function(wrap_pyfunction!(nvagentrt_event, m)?)?;

    // Tool lifecycle
    m.add_function(wrap_pyfunction!(nvagentrt_tool_call, m)?)?;
    m.add_function(wrap_pyfunction!(nvagentrt_tool_call_end, m)?)?;
    m.add_function(wrap_pyfunction!(nvagentrt_tool_call_execute, m)?)?;

    // LLM lifecycle
    m.add_function(wrap_pyfunction!(nvagentrt_llm_call, m)?)?;
    m.add_function(wrap_pyfunction!(nvagentrt_llm_call_end, m)?)?;
    m.add_function(wrap_pyfunction!(nvagentrt_llm_call_execute, m)?)?;
    m.add_function(wrap_pyfunction!(nvagentrt_llm_stream_call_execute, m)?)?;

    // Tool guardrails
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_tool_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_tool_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_tool_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_tool_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_tool_conditional_execution_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_tool_conditional_execution_guardrail,
        m
    )?)?;

    // Tool intercepts
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_tool_request_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_tool_request_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_tool_response_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_tool_response_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_tool_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_tool_execution_intercept,
        m
    )?)?;

    // LLM guardrails
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_llm_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_llm_sanitize_request_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_llm_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_llm_sanitize_response_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_llm_conditional_execution_guardrail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_llm_conditional_execution_guardrail,
        m
    )?)?;

    // LLM intercepts
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_llm_request_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_llm_request_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_llm_response_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_llm_response_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_llm_stream_response_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_llm_stream_response_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_llm_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_llm_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_register_llm_stream_execution_intercept,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nvagentrt_deregister_llm_stream_execution_intercept,
        m
    )?)?;

    // Subscribers
    m.add_function(wrap_pyfunction!(nvagentrt_register_subscriber, m)?)?;
    m.add_function(wrap_pyfunction!(nvagentrt_deregister_subscriber, m)?)?;

    Ok(())
}
