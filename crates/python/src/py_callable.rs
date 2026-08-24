// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-to-Rust callback wrappers.
//!
//! Each `wrap_py_*` function takes a Python callable (`Py<PyAny>`) and returns
//! a Rust closure that the core library can store and invoke.  The wrappers
//! handle:
//!
//! - **GIL acquisition** — every call back into Python goes through
//!   `Python::attach`.
//! - **Type conversion** — Python objects are converted to/from
//!   `serde_json::Value` via the helpers in [`crate::convert`].
//! - **Async bridging** — for functions that may return a Python coroutine,
//!   the wrapper detects `__await__` and uses `pyo3_async_runtimes` to drive
//!   the coroutine on the tokio runtime.
//! - **Middleware `next` functions** — execution intercepts receive a
//!   `PyToolNextFn`, `PyLlmNextFn`, or `PyLlmStreamNextFn` wrapper that
//!   Python code can `await` to invoke the next layer in the chain.

#![allow(clippy::type_complexity)]

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use nemo_relay::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, PublicationContext, capture_nested_publication_buffer, publication_context,
};
use nemo_relay::api::runtime::{
    EventMetadataInjectorFn, EventSanitizeFn, EventSubscriberFn, LlmConditionalFn,
    LlmExecutionNextFn, LlmJsonStream, LlmRequestInterceptFn, LlmSanitizeRequestContext,
    LlmSanitizeRequestFn, LlmSanitizeResponseContext, LlmSanitizeResponseFn,
    LlmStreamExecutionNextFn, LlmStreamInner, MiddlewareContinuationContext, ScopeStackHandle,
    ToolConditionalFn, ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
    capture_propagation_context, capture_traceparent, current_scope_stack,
};
use nemo_relay::error::{FlowError, Result as FlowResult};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_async_runtimes::TaskLocals;
use serde_json::Value as Json;
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;

use nemo_relay::api::event::{Event, EventSanitizeFields};
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::tool::ToolExecutionResult;
use nemo_relay::codec::request::AnnotatedLlmRequest as AnnotatedLLMRequest;
use nemo_relay::codec::response::AnnotatedLlmResponse as AnnotatedLLMResponse;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};

use crate::convert::{json_to_py, py_to_json};
use crate::py_types::{
    PyAnnotatedLLMRequest, PyAnnotatedLLMResponse, PyLLMRequest, PyLLMRequestInterceptOutcome,
    PyLlmSanitizeRequestContext, PyLlmSanitizeResponseContext, PyScopeStack,
    PyToolExecutionInterceptOutcome, PyToolExecutionResult, TOOL_EXECUTION_INTERCEPT_RESULT_ERROR,
};

type PyValueFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;

fn python_callback_error(error: PyErr) -> FlowError {
    let exception_type = Python::attach(|py| {
        error
            .get_type(py)
            .getattr("__name__")
            .and_then(|name| name.extract::<String>())
            .unwrap_or_else(|_| "Exception".to_string())
    });
    FlowError::CallbackException {
        message: error.to_string(),
        exception_type,
    }
}

fn tool_execution_callback_result_error(error: impl std::fmt::Display) -> FlowError {
    FlowError::InvalidArgument(format!(
        "tool execution callback must return ToolExecutionResult; return ToolExecutionResult(payload), not a raw dict: {error}"
    ))
}

fn tool_execution_intercept_outcome_error(error: impl std::fmt::Display) -> FlowError {
    FlowError::InvalidArgument(format!(
        "tool execution intercept must return ToolExecutionInterceptOutcome; after await next(args), return ToolExecutionInterceptOutcome(result=downstream.result, annotation=downstream.annotation): {error}"
    ))
}

fn tool_execution_intercept_callback_error(error: PyErr) -> FlowError {
    if error
        .to_string()
        .contains(TOOL_EXECUTION_INTERCEPT_RESULT_ERROR)
    {
        tool_execution_intercept_outcome_error(error)
    } else {
        python_callback_error(error)
    }
}

struct CancellablePyFuture {
    inner: PyValueFuture,
    scheduled: Arc<Mutex<ScheduledAwaitable>>,
}

impl Future for CancellablePyFuture {
    type Output = PyResult<Py<PyAny>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.inner.as_mut().poll(cx) {
            Poll::Ready(result) => {
                this.scheduled.lock().expect("scheduled awaitable").task = None;
                Poll::Ready(result)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for CancellablePyFuture {
    fn drop(&mut self) {
        let task = {
            let mut scheduled = self.scheduled.lock().expect("scheduled awaitable");
            scheduled.cancelled = true;
            scheduled.task.take()
        };
        let Some(task) = task else { return };
        Python::attach(|py| {
            let _ = cancel_python_task(py, &task, &self.scheduled);
        });
    }
}

struct ScheduledAwaitable {
    task: Option<Py<PyAny>>,
    task_locals: TaskLocals,
    cancelled: bool,
}

#[pyclass]
struct SchedulePythonAwaitable {
    awaitable: Option<Py<PyAny>>,
    sender: Option<tokio::sync::oneshot::Sender<PyResult<Py<PyAny>>>>,
    scheduled: Arc<Mutex<ScheduledAwaitable>>,
}

#[pymethods]
impl SchedulePythonAwaitable {
    fn __call__(&mut self, py: Python<'_>) {
        let result = match self.awaitable.take() {
            Some(awaitable) => {
                let result = py
                    .import("asyncio")
                    .and_then(|asyncio| asyncio.getattr("ensure_future"))
                    .and_then(|ensure_future| ensure_future.call1((awaitable.bind(py),)))
                    .map(Bound::unbind);
                if result.is_err() {
                    let _ = awaitable.bind(py).call_method0("close");
                }
                result
            }
            None => Err(PyRuntimeError::new_err(
                "Python awaitable was already scheduled",
            )),
        };

        if let Ok(task) = &result {
            let cancelled = {
                let mut scheduled = self.scheduled.lock().expect("scheduled awaitable");
                scheduled.task = Some(task.clone_ref(py));
                scheduled.cancelled
            };
            if cancelled {
                let _ = task.bind(py).call_method0("cancel");
            }
        }

        if let Some(sender) = self.sender.take() {
            let _ = sender.send(result);
        }
    }
}

impl Drop for SchedulePythonAwaitable {
    fn drop(&mut self) {
        let Some(awaitable) = self.awaitable.take() else {
            return;
        };
        Python::attach(|py| {
            let _ = awaitable.bind(py).call_method0("close");
        });
    }
}

tokio::task_local! {
    pub(crate) static PY_AWAITABLES_ALLOWED: bool;
}

fn schedule_python_awaitable(
    py: Python<'_>,
    awaitable: Py<PyAny>,
    task_locals: &TaskLocals,
) -> PyResult<(
    tokio::sync::oneshot::Receiver<PyResult<Py<PyAny>>>,
    Arc<Mutex<ScheduledAwaitable>>,
)> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let scheduled = Arc::new(Mutex::new(ScheduledAwaitable {
        task: None,
        task_locals: task_locals.clone(),
        cancelled: false,
    }));
    let kwargs = PyDict::new(py);
    kwargs.set_item("context", task_locals.context(py))?;
    task_locals.event_loop(py).call_method(
        "call_soon_threadsafe",
        (SchedulePythonAwaitable {
            awaitable: Some(awaitable),
            sender: Some(sender),
            scheduled: scheduled.clone(),
        },),
        Some(&kwargs),
    )?;
    Ok((receiver, scheduled))
}

fn cancel_python_task(
    py: Python<'_>,
    task: &Py<PyAny>,
    scheduled: &Arc<Mutex<ScheduledAwaitable>>,
) -> PyResult<()> {
    let event_loop = {
        let scheduled = scheduled.lock().expect("scheduled awaitable");
        scheduled.task_locals.event_loop(py)
    };
    if event_loop.call_method0("is_closed")?.is_truthy()? {
        return Ok(());
    }
    let cancel = task.bind(py).getattr("cancel")?;
    let on_event_loop = py
        .import("asyncio")?
        .getattr("get_running_loop")?
        .call0()
        .is_ok_and(|running_loop| running_loop.is(&event_loop));
    if on_event_loop {
        cancel.call0()?;
    } else {
        event_loop.call_method1("call_soon_threadsafe", (cancel,))?;
    }
    Ok(())
}

fn cancellable_future_with_locals(
    py: Python<'_>,
    result: Py<PyAny>,
    task_locals: &TaskLocals,
) -> FlowResult<PyValueFuture> {
    let (receiver, scheduled) = schedule_python_awaitable(py, result, task_locals)
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let task_locals = task_locals.clone();
    let inner = async move {
        let task = receiver
            .await
            .map_err(|_| PyRuntimeError::new_err("Python awaitable scheduling was cancelled"))??;
        Python::attach(|py| {
            pyo3_async_runtimes::into_future_with_locals(&task_locals, task.into_bound(py))
        })?
        .await
    };
    Ok(Box::pin(CancellablePyFuture {
        inner: Box::pin(inner),
        scheduled,
    }))
}

fn reject_awaitable_from_sync_caller(result: &Bound<'_, PyAny>) -> FlowResult<()> {
    if PY_AWAITABLES_ALLOWED
        .try_with(|allowed| *allowed)
        .unwrap_or(true)
    {
        return Ok(());
    }
    let _ = result.call_method0("close");
    Err(FlowError::Internal(
        "awaitable Python middleware requires an async caller".into(),
    ))
}

fn validate_python_llm_sanitizer_signature(py_fn: &Py<PyAny>) -> PyResult<()> {
    Python::attach(|py| {
        let inspect = py.import("inspect")?;
        let signature = inspect.call_method1("signature", (py_fn.bind(py),)).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "LLM sanitizer callback signature cannot be inspected; use a callable that accepts `(payload, context)`",
            )
        })?;
        if signature
            .call_method1("bind", (py.None(), py.None()))
            .is_ok()
        {
            return Ok(());
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "LLM sanitizer callback must accept `(payload, context)`",
        ))
    })
}

fn split_json_or_future_with_locals(
    py: Python<'_>,
    result: Py<PyAny>,
    task_locals: Option<&TaskLocals>,
) -> FlowResult<Result<Json, PyValueFuture>> {
    let bound = result.bind(py);
    if bound.getattr("__await__").is_ok() {
        reject_awaitable_from_sync_caller(bound)?;
        let future: PyValueFuture = match task_locals {
            Some(locals) => cancellable_future_with_locals(py, result, locals)?,
            None => Box::pin(
                pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                    .map_err(|error| FlowError::Internal(error.to_string()))?,
            ),
        };
        Ok(Err(future))
    } else {
        let json = py_to_json(bound).map_err(|error| FlowError::Internal(error.to_string()))?;
        Ok(Ok(json))
    }
}

async fn resolve_json_or_future(
    outcome: FlowResult<Result<Json, PyValueFuture>>,
) -> FlowResult<Json> {
    match outcome? {
        Ok(json) => Ok(json),
        Err(future) => {
            let py_result = future.await.map_err(python_callback_error)?;
            Python::attach(|py| {
                py_to_json(py_result.bind(py))
                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))
            })
        }
    }
}

fn split_py_object_or_future_with_locals(
    py: Python<'_>,
    result: Py<PyAny>,
    task_locals: Option<&TaskLocals>,
    invocation_context: Option<&Bound<'_, PyAny>>,
) -> FlowResult<Result<Py<PyAny>, PyValueFuture>> {
    let bound = result.bind(py);
    if bound.getattr("__await__").is_ok() {
        reject_awaitable_from_sync_caller(bound)?;
        let future: PyValueFuture = match task_locals {
            Some(locals) => cancellable_future_with_locals(py, result, locals)?,
            None => {
                let invocation_context = invocation_context.map(|context| context.clone().unbind());
                Box::pin(async move {
                    tokio::task::spawn_blocking(move || {
                        Python::attach(|py| {
                            let coroutine = py
                                .import("nemo_relay._event_sanitizer_context")
                                .and_then(|module| module.getattr("await_result"))
                                .and_then(|await_result| await_result.call1((result.bind(py),)))?;
                            let asyncio_run = py
                                .import("asyncio")
                                .and_then(|asyncio| asyncio.getattr("run"))?;
                            match invocation_context {
                                Some(context) => context
                                    .bind(py)
                                    .call_method1("run", (asyncio_run, coroutine))
                                    .map(Bound::unbind),
                                None => asyncio_run.call1((coroutine,)).map(Bound::unbind),
                            }
                        })
                    })
                    .await
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
                })
            }
        };
        Ok(Err(future))
    } else {
        Ok(Ok(result))
    }
}

fn capture_python_task_locals() -> Option<TaskLocals> {
    Python::attach(|py| pyo3_async_runtimes::tokio::get_current_locals(py).ok())
}

struct PythonPublicationContext {
    task_locals: Option<TaskLocals>,
    context: Py<PyAny>,
}

pub(crate) fn capture_python_publication_context() -> Option<PublicationContext> {
    Python::attach(|py| {
        let context = py
            .import("contextvars")
            .and_then(|module| module.call_method0("copy_context"))
            .ok()?
            .unbind();
        Some(Arc::new(PythonPublicationContext {
            task_locals: pyo3_async_runtimes::tokio::get_current_locals(py).ok(),
            context,
        }) as PublicationContext)
    })
}

fn running_task_locals(locals: TaskLocals) -> Option<TaskLocals> {
    let live = Python::attach(|py| {
        let event_loop = locals.event_loop(py);
        let running = event_loop
            .call_method0("is_running")
            .and_then(|value| value.extract::<bool>())
            .unwrap_or(false);
        let closed = event_loop
            .call_method0("is_closed")
            .and_then(|value| value.extract::<bool>())
            .unwrap_or(true);
        running && !closed
    });
    live.then_some(locals)
}

fn fresh_running_task_locals(locals: TaskLocals) -> Option<TaskLocals> {
    let locals = running_task_locals(locals)?;
    Python::attach(|py| {
        let context = locals.context(py).call_method0("copy").ok()?;
        Some(TaskLocals::new(locals.event_loop(py)).with_context(context))
    })
}

fn task_locals_with_running_loop(registered: Option<&TaskLocals>) -> Option<TaskLocals> {
    publication_context::<PythonPublicationContext>()
        .and_then(|context| context.task_locals.clone())
        .and_then(fresh_running_task_locals)
        .or_else(|| capture_python_task_locals().and_then(fresh_running_task_locals))
        .or_else(|| registered.cloned().and_then(fresh_running_task_locals))
}

fn copy_publication_invocation<'py>(
    py: Python<'py>,
    context: &PythonPublicationContext,
    fallback_task_locals: Option<TaskLocals>,
) -> PyResult<(Bound<'py, PyAny>, Option<TaskLocals>)> {
    copy_publication_invocation_with_buffer(
        py,
        context,
        fallback_task_locals,
        capture_nested_publication_buffer(),
    )
}

fn copy_publication_invocation_with_buffer<'py>(
    py: Python<'py>,
    context: &PythonPublicationContext,
    fallback_task_locals: Option<TaskLocals>,
    publication_buffer: Option<PublicationBuffer>,
) -> PyResult<(Bound<'py, PyAny>, Option<TaskLocals>)> {
    let invocation_context = context.context.bind(py).call_method0("copy")?;
    // The dispatcher already installs an isolated emission-time snapshot.
    // Retain that handle so callback-local scope mutations stay visible to
    // nested events without exposing stack cloning as a public runtime API.
    let scope_stack = current_scope_stack();
    let scope_stack = Py::new(
        py,
        PyScopeStack {
            inner: scope_stack,
            publication_buffer,
        },
    )?;
    let nemo_relay = py.import("nemo_relay")?;
    if let Ok(scope_stack_var) = nemo_relay.getattr("_scope_stack_var") {
        invocation_context.call_method1("run", (scope_stack_var.getattr("set")?, scope_stack))?;
    }
    let task_locals = context
        .task_locals
        .clone()
        .and_then(running_task_locals)
        .or(fallback_task_locals)
        .map(|locals| {
            TaskLocals::new(locals.event_loop(py)).with_context(invocation_context.clone())
        });
    Ok((invocation_context, task_locals))
}

fn copy_middleware_invocation<'py>(
    py: Python<'py>,
    fallback_task_locals: Option<TaskLocals>,
) -> PyResult<(Option<Bound<'py, PyAny>>, Option<TaskLocals>)> {
    let (invocation_context, task_locals) =
        if let Some(context) = publication_context::<PythonPublicationContext>() {
            let (context, task_locals) =
                copy_publication_invocation(py, &context, fallback_task_locals)?;
            (Some(context), task_locals)
        } else if let Some(locals) = fallback_task_locals {
            let invocation_context = locals.context(py).call_method0("copy")?;
            let task_locals =
                TaskLocals::new(locals.event_loop(py)).with_context(invocation_context.clone());
            (Some(invocation_context), Some(task_locals))
        } else {
            (None, None)
        };
    if let Some(context) = invocation_context.as_ref() {
        let parent_var = py
            .import("nemo_relay")
            .and_then(|module| module.getattr("_propagation_parent_var"));
        if let Ok(parent_var) = parent_var {
            let propagation_context = capture_propagation_context()
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            let propagation_parent_uuid = propagation_context.parent_uuid.to_string();
            context.call_method1("run", (parent_var.getattr("set")?, propagation_parent_uuid))?;
            let root_var = py
                .import("nemo_relay")
                .and_then(|module| module.getattr("_propagation_root_var"))?;
            let root_uuid = context.call_method1("run", (root_var.getattr("get")?,))?;
            if root_uuid.is_none()
                && let Ok(traceparent) = capture_traceparent()
            {
                let propagation_root_uuid = traceparent
                    .get(3..35)
                    .and_then(|value| uuid::Uuid::parse_str(value).ok())
                    .ok_or_else(|| PyRuntimeError::new_err("invalid Relay traceparent"))?
                    .to_string();
                context.call_method1("run", (root_var.getattr("set")?, propagation_root_uuid))?;
            }
        }
    }
    Ok((invocation_context, task_locals))
}

fn loop_affine_callback(
    py: Python<'_>,
    callback: &Bound<'_, PyAny>,
    task_locals: Option<&TaskLocals>,
    sanitizer: bool,
) -> PyResult<Py<PyAny>> {
    if task_locals.is_none() {
        return Ok(callback.clone().unbind());
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("sanitizer", sanitizer)?;
    py.import("nemo_relay._event_sanitizer_context")?
        .getattr("loop_affine")?
        .call((callback,), Some(&kwargs))
        .map(Bound::unbind)
}

async fn resolve_py_object_or_future(
    outcome: FlowResult<Result<Py<PyAny>, PyValueFuture>>,
) -> FlowResult<Py<PyAny>> {
    match outcome? {
        Ok(value) => Ok(value),
        Err(future) => future.await.map_err(python_callback_error),
    }
}

async fn resolve_py_tool_execution_intercept_outcome(
    outcome: FlowResult<Result<Py<PyAny>, PyValueFuture>>,
) -> FlowResult<Py<PyAny>> {
    match outcome? {
        Ok(value) => Ok(value),
        Err(future) => future
            .await
            .map_err(tool_execution_intercept_callback_error),
    }
}

fn next_async_iter_coro(async_iter: &Arc<Py<PyAny>>) -> FlowResult<Option<Py<PyAny>>> {
    Python::attach(|py| {
        py.import("nemo_relay._event_sanitizer_context")
            .and_then(|module| module.getattr("async_iter_next"))
            .and_then(|next| next.call1((async_iter.bind(py),)))
            .map(|coro| Some(coro.unbind()))
            .map_err(|error| FlowError::Internal(error.to_string()))
    })
}

async fn schedule_async_iter_task(coro: Py<PyAny>) -> FlowResult<Py<PyAny>> {
    let receiver = Python::attach(|py| {
        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
        schedule_python_awaitable(py, coro, &locals).map(|(receiver, _)| receiver)
    })
    .map_err(|error| FlowError::Internal(error.to_string()))?;
    receiver
        .await
        .map_err(|_| FlowError::Internal("Python awaitable scheduling was cancelled".into()))?
        .map_err(|error| FlowError::Internal(error.to_string()))
}

fn cancel_async_iter_task(task: &Py<PyAny>) -> FlowResult<()> {
    Python::attach(|py| {
        pyo3_async_runtimes::tokio::get_current_locals(py)
            .and_then(|locals| {
                let cancel = task.bind(py).getattr("cancel")?;
                locals
                    .event_loop(py)
                    .call_method1("call_soon_threadsafe", (cancel,))
            })
            .map(|_| ())
            .map_err(|error| FlowError::Internal(error.to_string()))
    })
}

enum AsyncIterTaskResult {
    Item(Json),
    End,
    Cancelled,
}

async fn await_async_iter_task_result(task: Py<PyAny>) -> FlowResult<AsyncIterTaskResult> {
    let future = Python::attach(|py| {
        pyo3_async_runtimes::tokio::into_future(task.into_bound(py))
            .map_err(|e| FlowError::Internal(e.to_string()))
    })?;

    match future.await {
        Ok(result) => Python::attach(|py| {
            py_to_json(result.bind(py))
                .map(AsyncIterTaskResult::Item)
                .map_err(|e| FlowError::Internal(e.to_string()))
        }),
        Err(error) => Python::attach(|py| {
            let cancelled_error = py
                .import("asyncio")
                .and_then(|asyncio| asyncio.getattr("CancelledError"))
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            if error.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                Ok(AsyncIterTaskResult::End)
            } else if error.is_instance(py, &cancelled_error) {
                Ok(AsyncIterTaskResult::Cancelled)
            } else {
                Err(python_callback_error(error))
            }
        }),
    }
}

#[cfg(test)]
async fn await_async_iter_task(task: Py<PyAny>) -> FlowResult<Option<Json>> {
    match await_async_iter_task_result(task).await? {
        AsyncIterTaskResult::Item(value) => Ok(Some(value)),
        AsyncIterTaskResult::End => Ok(None),
        AsyncIterTaskResult::Cancelled => Err(FlowError::Internal(
            "async iterator task was cancelled".into(),
        )),
    }
}

#[cfg(test)]
async fn await_async_iter_value(coro: Py<PyAny>) -> FlowResult<Option<Json>> {
    await_async_iter_task(schedule_async_iter_task(coro).await?).await
}

async fn close_async_iter(async_iter: &Arc<Py<PyAny>>) -> FlowResult<()> {
    let close = Python::attach(|py| {
        py.import("nemo_relay._event_sanitizer_context")
            .and_then(|module| module.getattr("async_iter_close"))
            .and_then(|close| close.call1((async_iter.bind(py),)))
            .map(Bound::unbind)
            .map_err(|error| FlowError::Internal(error.to_string()))
    });
    let close = close?;
    let task = schedule_async_iter_task(close).await?;
    let future = Python::attach(|py| {
        pyo3_async_runtimes::tokio::into_future(task.into_bound(py))
            .map_err(|error| FlowError::Internal(error.to_string()))
    })?;
    future
        .await
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    Ok(())
}

async fn send_async_iter_value(
    tx: &tokio::sync::mpsc::Sender<FlowResult<Json>>,
    value: FlowResult<Json>,
    cancel: &mut tokio::sync::watch::Receiver<bool>,
    async_iter: &Arc<Py<PyAny>>,
) -> FlowResult<bool> {
    let sent = tokio::select! {
        _ = cancel.changed() => {
            close_async_iter(async_iter).await?;
            return Ok(false);
        }
        sent = tx.send(value) => sent,
    };
    if sent.is_err() {
        close_async_iter(async_iter).await?;
        return Ok(false);
    }
    Ok(true)
}

async fn forward_async_iter_result(
    next_value: FlowResult<AsyncIterTaskResult>,
    tx: &tokio::sync::mpsc::Sender<FlowResult<Json>>,
    cancel: &mut tokio::sync::watch::Receiver<bool>,
    async_iter: &Arc<Py<PyAny>>,
) -> Option<FlowResult<()>> {
    match next_value {
        Ok(AsyncIterTaskResult::Item(value)) => {
            match send_async_iter_value(tx, Ok(value), cancel, async_iter).await {
                Ok(true) => None,
                Ok(false) => Some(Ok(())),
                Err(error) => Some(Err(error)),
            }
        }
        Ok(AsyncIterTaskResult::End) => Some(Ok(())),
        Ok(AsyncIterTaskResult::Cancelled) => Some(close_async_iter(async_iter).await.and(Err(
            FlowError::Internal("async iterator task was cancelled".into()),
        ))),
        Err(error) => Some(
            match send_async_iter_value(tx, Err(error), cancel, async_iter).await {
                Ok(true) => close_async_iter(async_iter).await,
                Ok(false) => Ok(()),
                Err(error) => Err(error),
            },
        ),
    }
}

async fn forward_async_iter(
    async_iter: Arc<Py<PyAny>>,
    tx: tokio::sync::mpsc::Sender<FlowResult<Json>>,
    mut cancel: tokio::sync::watch::Receiver<bool>,
    closed: tokio::sync::watch::Sender<Option<FlowResult<()>>>,
) {
    let result = loop {
        if *cancel.borrow() {
            break close_async_iter(&async_iter).await;
        }
        let next_value = match next_async_iter_coro(&async_iter) {
            Ok(None) => break Ok(()),
            Ok(Some(coro)) => match schedule_async_iter_task(coro).await {
                Ok(task) => {
                    let task_for_future = Python::attach(|py| task.clone_ref(py));
                    let mut next_value = Box::pin(await_async_iter_task_result(task_for_future));
                    tokio::select! {
                        _ = cancel.changed() => {
                            let _ = cancel_async_iter_task(&task);
                            let next_result = next_value.await;
                            let close_result = close_async_iter(&async_iter).await;
                            break match next_result {
                                Err(error) => Err(error),
                                Ok(_) => close_result,
                            };
                        }
                        value = &mut next_value => value,
                    }
                }
                Err(error) => Err(error),
            },
            Err(error) => Err(error),
        };

        if let Some(result) =
            forward_async_iter_result(next_value, &tx, &mut cancel, &async_iter).await
        {
            break result;
        }
    };
    closed.send_replace(Some(result));
}

#[derive(Clone)]
struct PythonAsyncIteratorClose {
    cancel: tokio::sync::watch::Sender<bool>,
    closed: tokio::sync::watch::Receiver<Option<FlowResult<()>>>,
}

struct PythonAsyncIteratorStream {
    receiver: ReceiverStream<FlowResult<Json>>,
    close: PythonAsyncIteratorClose,
}

impl Stream for PythonAsyncIteratorStream {
    type Item = FlowResult<Json>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.receiver).poll_next(cx)
    }
}

impl Drop for PythonAsyncIteratorStream {
    fn drop(&mut self) {
        self.close.cancel.send_replace(true);
    }
}

impl LlmStreamInner for PythonAsyncIteratorStream {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = FlowResult<()>> + Send + '_>> {
        let this = self.get_mut();
        this.close.cancel.send_replace(true);
        this.receiver.close();
        while this.receiver.as_mut().try_recv().is_ok() {}
        let close = this.close.clone();
        Box::pin(async move {
            let mut closed = close.closed;
            while closed.borrow().is_none() {
                closed.changed().await.map_err(|_| {
                    FlowError::Internal("Python stream cleanup task ended early".into())
                })?;
            }
            closed.borrow().clone().expect("close state checked above")
        })
    }
}

fn stream_from_async_iter(
    async_iter: Py<PyAny>,
    task_locals: Option<TaskLocals>,
) -> FlowResult<LlmJsonStream> {
    let (tx, rx) = tokio::sync::mpsc::channel::<FlowResult<Json>>(32);
    let task_locals = match task_locals {
        Some(locals) => locals,
        None => Python::attach(|py| {
            pyo3_async_runtimes::tokio::get_current_locals(py)
                .map_err(|e: pyo3::PyErr| FlowError::Internal(e.to_string()))
        })?,
    };

    let async_iter = Arc::new(async_iter);
    let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
    let (closed, closed_rx) = tokio::sync::watch::channel(None);
    tokio::spawn(pyo3_async_runtimes::tokio::scope(task_locals, async move {
        forward_async_iter(async_iter, tx, cancel_rx, closed).await;
    }));

    let stream = PythonAsyncIteratorStream {
        receiver: ReceiverStream::new(rx),
        close: PythonAsyncIteratorClose {
            cancel,
            closed: closed_rx,
        },
    };
    Ok(LlmJsonStream::from_closeable(stream))
}

/// Wrap a Python callable `(str, Json) -> Json` for tool sanitize/intercept fns.
pub fn wrap_py_tool_fn(py_fn: Py<PyAny>) -> ToolSanitizeFn {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(move |name: String, args: Json| {
        let py_fn = py_fn.clone();
        let task_locals = task_locals_with_running_loop(task_locals.as_ref());
        let publication_context = publication_context::<PythonPublicationContext>();
        let publication_buffer = capture_nested_publication_buffer();
        let publication = nemo_relay::api::runtime::subscriber_dispatcher::in_dispatcher_callback();
        Box::pin(async move {
            let result = resolve_py_object_or_future(Python::attach(|py| {
                let (invocation_context, task_locals) = match publication_context.as_ref() {
                    Some(context) => {
                        let (context, publication_task_locals) =
                            copy_publication_invocation_with_buffer(
                                py,
                                context,
                                task_locals,
                                publication_buffer.clone(),
                            )
                            .map_err(|error| FlowError::Internal(error.to_string()))?;
                        (Some(context), publication_task_locals)
                    }
                    None => { copy_middleware_invocation(py, task_locals) }
                        .map_err(|error| FlowError::Internal(error.to_string()))?,
                };
                let py_args = json_to_py(py, &args)
                    .map_err(|e| FlowError::Internal(format!("tool json_to_py failed: {e}")))?;
                let loop_affine = task_locals.is_some();
                let callback =
                    loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), publication)
                        .map_err(|error| FlowError::Internal(error.to_string()))?;
                let result = match (invocation_context.as_ref(), publication && !loop_affine) {
                    (Some(context), true) => py
                        .import("nemo_relay._event_sanitizer_context")
                        .and_then(|module| module.getattr("invoke"))
                        .and_then(|invoke| {
                            context.call_method1("run", (invoke, callback.bind(py), name, py_args))
                        }),
                    (Some(context), false) => {
                        context.call_method1("run", (callback.bind(py), name, py_args))
                    }
                    (None, true) => py
                        .import("nemo_relay._event_sanitizer_context")
                        .and_then(|module| module.getattr("invoke"))
                        .and_then(|invoke| invoke.call1((callback.bind(py), name, py_args))),
                    (None, false) => callback.bind(py).call1((name, py_args)),
                }
                .map_err(|e| FlowError::Internal(format!("Python tool callback failed: {e}")))?;
                split_py_object_or_future_with_locals(
                    py,
                    result.unbind(),
                    task_locals.as_ref(),
                    invocation_context.as_ref(),
                )
            }))
            .await
            .and_then(|result| {
                Python::attach(|py| {
                    py_to_json(result.bind(py))
                        .map_err(|e| FlowError::Internal(format!("tool py_to_json failed: {e}")))
                })
            });
            if let Err(error) = &result {
                eprintln!("nemo_relay: Python tool sanitizer callable failed: {error}");
            }
            result
        })
    })
}

/// Wrap a Python callable `(str, Json) -> Optional[str]` for tool conditional guardrails.
pub fn wrap_py_tool_conditional_fn(py_fn: Py<PyAny>) -> ToolConditionalFn {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(move |name: String, args: Json| {
        let py_fn = py_fn.clone();
        let task_locals = task_locals_with_running_loop(task_locals.as_ref());
        Box::pin(async move {
            let result = resolve_py_object_or_future(Python::attach(|py| {
                let (invocation_context, task_locals) = copy_middleware_invocation(py, task_locals)
                    .map_err(|error| FlowError::Internal(error.to_string()))?;
                let py_args =
                    json_to_py(py, &args).map_err(|e| FlowError::Internal(e.to_string()))?;
                let callback =
                    loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), false)
                        .map_err(|error| FlowError::Internal(error.to_string()))?;
                let result = match invocation_context.as_ref() {
                    Some(context) => {
                        context.call_method1("run", (callback.bind(py), name, py_args))
                    }
                    None => callback.bind(py).call1((name, py_args)),
                }
                .map_err(|e| FlowError::Internal(e.to_string()))?;
                split_py_object_or_future_with_locals(
                    py,
                    result.unbind(),
                    task_locals.as_ref(),
                    invocation_context.as_ref(),
                )
            }))
            .await?;
            Python::attach(|py| {
                let bound = result.bind(py);
                if bound.is_none() {
                    Ok(None)
                } else {
                    bound.extract::<String>().map(Some).map_err(|e| {
                        FlowError::Internal(format!(
                            "tool conditional guardrail returned unexpected type: {e}"
                        ))
                    })
                }
            })
        })
    })
}

/// Wrap a Python callable `(str, Json) -> Json` for tool request intercepts.
pub fn wrap_py_tool_request_intercept_fn(py_fn: Py<PyAny>) -> ToolInterceptFn {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(move |name: String, args: Json| {
        let py_fn = py_fn.clone();
        let task_locals = task_locals_with_running_loop(task_locals.as_ref());
        Box::pin(async move {
            resolve_json_or_future(Python::attach(|py| {
                let (invocation_context, task_locals) = copy_middleware_invocation(py, task_locals)
                    .map_err(|error| FlowError::Internal(error.to_string()))?;
                let py_args =
                    json_to_py(py, &args).map_err(|e| FlowError::Internal(e.to_string()))?;
                let callback =
                    loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), false)
                        .map_err(|error| FlowError::Internal(error.to_string()))?;
                let result = match invocation_context.as_ref() {
                    Some(context) => {
                        context.call_method1("run", (callback.bind(py), name, py_args))
                    }
                    None => callback.bind(py).call1((name, py_args)),
                }
                .map_err(|e| FlowError::Internal(e.to_string()))?;
                split_json_or_future_with_locals(py, result.unbind(), task_locals.as_ref())
            }))
            .await
        })
    })
}

/// Wrap a Python callable `(Json) -> ToolExecutionResult` for tool execution.
/// Supports both sync and async Python callables. If the callable returns a
/// coroutine, it is awaited via the pyo3-async-runtimes bridge.
pub fn wrap_py_tool_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<
    dyn Fn(Json) -> Pin<Box<dyn Future<Output = FlowResult<ToolExecutionResult>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = std::sync::Arc::new(py_fn);
    let registered_task_locals = capture_python_task_locals();
    Box::new(move |args: Json| {
        let py_fn = py_fn.clone();
        let task_locals = task_locals_with_running_loop(registered_task_locals.as_ref());
        Box::pin(async move {
            let result = resolve_py_object_or_future(Python::attach(|py| {
                let (invocation_context, task_locals) = copy_middleware_invocation(py, task_locals)
                    .map_err(|error| FlowError::Internal(error.to_string()))?;
                let py_args =
                    json_to_py(py, &args).map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                let callback =
                    loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), false)
                        .map_err(|error| FlowError::Internal(error.to_string()))?;
                let result = match invocation_context.as_ref() {
                    Some(context) => context.call_method1("run", (callback.bind(py), py_args)),
                    None => callback.bind(py).call1((py_args,)),
                }
                .map_err(python_callback_error)?;
                split_py_object_or_future_with_locals(
                    py,
                    result.unbind(),
                    task_locals.as_ref(),
                    invocation_context.as_ref(),
                )
            }))
            .await?;
            Python::attach(|py| {
                result
                    .extract::<PyToolExecutionResult>(py)
                    .map_err(tool_execution_callback_result_error)?
                    .to_inner(py)
                    .map_err(|error| FlowError::Internal(error.to_string()))
            })
        })
    })
}

fn python_invocation_scope_stack(py: Python<'_>) -> PyResult<Option<ScopeStackHandle>> {
    python_invocation_scope_stack_from_module(py, "nemo_relay")
}

fn python_invocation_scope_stack_from_module(
    py: Python<'_>,
    module_name: &str,
) -> PyResult<Option<ScopeStackHandle>> {
    let Ok(nemo_relay) = py.import(module_name) else {
        return Ok(None);
    };
    let Ok(scope_stack_var) = nemo_relay.getattr("_scope_stack_var") else {
        // Embedded users may load only the native module, without the Python
        // wrapper that owns the task-local scope stack.
        return Ok(None);
    };
    let scope_stack = scope_stack_var.call_method1("get", (py.None(),))?;
    if scope_stack.is_none() {
        return Ok(None);
    }
    let scope_stack: PyRef<'_, PyScopeStack> = scope_stack.extract()?;
    Ok(Some(scope_stack.inner.clone()))
}

fn isolated_python_continuation_context(
    py: Python<'_>,
    context: &MiddlewareContinuationContext,
) -> PyResult<MiddlewareContinuationContext> {
    let context = match python_invocation_scope_stack(py)? {
        Some(scope_stack) => context.isolated_with_scope_stack(&scope_stack),
        None => context.isolated(),
    };
    context.map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// Python-callable wrapper for the Rust `ToolExecutionNextFn`.
///
/// The Python intercept calls `await next(args)` to invoke the next layer
/// in the middleware chain (or the original default function).  The wrapper
/// is reusable — calling `next` multiple times is supported (retry patterns).
#[pyclass]
struct PyToolNextFn {
    inner: ToolExecutionNextFn,
    context: MiddlewareContinuationContext,
}

#[pymethods]
impl PyToolNextFn {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let next = self.inner.clone();
        let context = isolated_python_continuation_context(py, &self.context)?;
        let json_args = py_to_json(args)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = context
                .invoke(move || next(json_args))
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Python::attach(|py| PyToolExecutionResult::from_inner(py, result))
        })
    }
}

/// Python-callable wrapper for the Rust `LlmExecutionNextFn`.
/// Reusable — calling `next` multiple times is supported (retry patterns).
#[pyclass]
struct PyLlmNextFn {
    inner: LlmExecutionNextFn,
    context: MiddlewareContinuationContext,
}

#[pymethods]
impl PyLlmNextFn {
    fn __call__<'py>(&self, py: Python<'py>, request: PyLLMRequest) -> PyResult<Bound<'py, PyAny>> {
        let next = self.inner.clone();
        let context = isolated_python_continuation_context(py, &self.context)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = context
                .invoke(move || next(request.inner))
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Python::attach(|py| json_to_py(py, &result))
        })
    }
}

/// Python-callable wrapper for the Rust `LlmStreamExecutionNextFn`.
/// Reusable — calling `next` multiple times is supported (retry patterns).
#[pyclass]
struct PyLlmStreamNextFn {
    inner: LlmStreamExecutionNextFn,
    context: MiddlewareContinuationContext,
}

#[pymethods]
impl PyLlmStreamNextFn {
    fn __call__<'py>(&self, py: Python<'py>, request: PyLLMRequest) -> PyResult<Bound<'py, PyAny>> {
        let next = self.inner.clone();
        let context = isolated_python_continuation_context(py, &self.context)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let rust_stream = context
                .invoke(move || next(request.inner))
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // Drain into mpsc channel and return PyLlmStream
            let (tx, rx) = tokio::sync::mpsc::channel::<FlowResult<Json>>(32);
            let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
            let (closed, closed_rx) = tokio::sync::watch::channel(None);
            tokio::spawn(async move {
                context
                    .run(crate::py_api::forward_stream_to_channel(
                        rust_stream,
                        tx,
                        cancel_rx,
                        closed,
                    ))
                    .await;
            });

            Ok(crate::py_types::PyLlmStream {
                receiver: Arc::new(tokio::sync::Mutex::new(rx)),
                cancel,
                closed: closed_rx,
            })
        })
    }
}

/// Wrap a Python callable `(Json, next) -> ToolExecutionInterceptOutcome` for tool execution intercepts.
/// The `next` parameter is a `PyToolNextFn` that the Python code can `await`.
pub fn wrap_py_tool_exec_intercept_fn(
    py_fn: Py<PyAny>,
) -> nemo_relay::api::runtime::ToolExecutionFn {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(move |name: &str, args: Json, next: ToolExecutionNextFn| {
        let py_fn = py_fn.clone();
        let name = name.to_string();
        let task_locals = task_locals_with_running_loop(task_locals.as_ref());
        Box::pin(async move {
            let result = resolve_py_tool_execution_intercept_outcome(Python::attach(|py| {
                let (invocation_context, task_locals) = copy_middleware_invocation(py, task_locals)
                    .map_err(|error| FlowError::Internal(error.to_string()))?;
                let py_args =
                    json_to_py(py, &args).map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                let py_next = PyToolNextFn {
                    inner: next,
                    context: MiddlewareContinuationContext::capture(),
                };
                let py_next = py_next
                    .into_pyobject(py)
                    .map_err(|e| FlowError::Internal(e.to_string()))?
                    .into_any();
                let callback =
                    loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), false)
                        .map_err(|error| FlowError::Internal(error.to_string()))?;
                let result = match invocation_context.as_ref() {
                    Some(context) => {
                        context.call_method1("run", (callback.bind(py), &name, py_args, py_next))
                    }
                    None => callback.bind(py).call1((&name, py_args, py_next)),
                }
                .map_err(tool_execution_intercept_callback_error)?;
                split_py_object_or_future_with_locals(
                    py,
                    result.unbind(),
                    task_locals.as_ref(),
                    invocation_context.as_ref(),
                )
            }))
            .await?;
            Python::attach(|py| {
                result
                    .extract::<PyToolExecutionInterceptOutcome>(py)
                    .map(|value| value.inner)
                    .map_err(tool_execution_intercept_outcome_error)
            })
        })
    })
}

/// Wrap a Python callable `(name, LlmRequest, next) -> dict` for LLM execution intercepts.
pub fn wrap_py_llm_exec_intercept_fn(
    py_fn: Py<PyAny>,
) -> Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(
        move |name: &str, request: LlmRequest, next: LlmExecutionNextFn| {
            let py_fn = py_fn.clone();
            let name = name.to_string();
            let task_locals = task_locals_with_running_loop(task_locals.as_ref());
            Box::pin(async move {
                let result = resolve_py_object_or_future(Python::attach(|py| {
                    let (invocation_context, task_locals) =
                        copy_middleware_invocation(py, task_locals)
                            .map_err(|error| FlowError::Internal(error.to_string()))?;
                    let py_req = PyLLMRequest { inner: request };
                    let py_next = PyLlmNextFn {
                        inner: next,
                        context: MiddlewareContinuationContext::capture(),
                    };
                    let py_req = py_req
                        .into_pyobject(py)
                        .map_err(|e| FlowError::Internal(e.to_string()))?
                        .into_any();
                    let py_next = py_next
                        .into_pyobject(py)
                        .map_err(|e| FlowError::Internal(e.to_string()))?
                        .into_any();
                    let callback =
                        loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), false)
                            .map_err(|error| FlowError::Internal(error.to_string()))?;
                    let result = match invocation_context.as_ref() {
                        Some(context) => {
                            context.call_method1("run", (callback.bind(py), &name, py_req, py_next))
                        }
                        None => callback.bind(py).call1((&name, py_req, py_next)),
                    }
                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                    split_py_object_or_future_with_locals(
                        py,
                        result.unbind(),
                        task_locals.as_ref(),
                        invocation_context.as_ref(),
                    )
                }))
                .await?;
                Python::attach(|py| {
                    py_to_json(result.bind(py))
                        .map_err(|e: PyErr| FlowError::Internal(e.to_string()))
                })
            })
        },
    )
}

/// Wrap a Python callable `(LlmRequest, next) -> AsyncIterator[Any]` for LLM
/// stream execution intercepts.
///
/// The Python callable may return the async iterator directly or return an
/// awaitable that resolves to one. The resulting iterator is drained on the
/// Tokio runtime and forwarded into a Rust `Stream<Item = Result<Json>>`.
pub fn wrap_py_llm_stream_exec_intercept_fn(
    py_fn: Py<PyAny>,
) -> Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = FlowResult<LlmJsonStream>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(
        move |_name: &str, request: LlmRequest, next: LlmStreamExecutionNextFn| {
            let py_fn = py_fn.clone();
            let task_locals = task_locals_with_running_loop(task_locals.as_ref());
            Box::pin(async move {
                let (outcome, invocation_task_locals) = Python::attach(|py| {
                    let (invocation_context, task_locals) =
                        copy_middleware_invocation(py, task_locals)
                            .map_err(|error| FlowError::Internal(error.to_string()))?;
                    let py_req = PyLLMRequest { inner: request };
                    let py_next = PyLlmStreamNextFn {
                        inner: next,
                        context: MiddlewareContinuationContext::capture(),
                    };
                    let py_req = py_req
                        .into_pyobject(py)
                        .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?
                        .into_any();
                    let py_next = py_next
                        .into_pyobject(py)
                        .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?
                        .into_any();
                    let callback =
                        loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), false)
                            .map_err(|error| FlowError::Internal(error.to_string()))?;
                    let result = match invocation_context.as_ref() {
                        Some(context) => {
                            context.call_method1("run", (callback.bind(py), py_req, py_next))
                        }
                        None => callback.bind(py).call1((py_req, py_next)),
                    }
                    .map_err(python_callback_error)?;
                    let outcome = split_py_object_or_future_with_locals(
                        py,
                        result.unbind(),
                        task_locals.as_ref(),
                        invocation_context.as_ref(),
                    )?;
                    Ok::<_, FlowError>((outcome, task_locals))
                })?;
                let async_iter = resolve_py_object_or_future(Ok(outcome)).await?;

                stream_from_async_iter(async_iter, invocation_task_locals)
            })
        },
    )
}

/// Wrap a Python callable `(LlmRequest, LlmSanitizeRequestContext) -> Optional<LlmRequest>`.
fn wrap_py_llm_sanitize_request_callback(py_fn: Py<PyAny>) -> LlmSanitizeRequestFn {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(
        move |request: LlmRequest, context: LlmSanitizeRequestContext| {
            let py_fn = py_fn.clone();
            let task_locals = task_locals_with_running_loop(task_locals.as_ref());
            let publication_context = publication_context::<PythonPublicationContext>();
            let publication_buffer = capture_nested_publication_buffer();
            let publication =
                nemo_relay::api::runtime::subscriber_dispatcher::in_dispatcher_callback();
            Box::pin(async move {
                let result = resolve_py_object_or_future(Python::attach(|py| {
                    let (invocation_context, task_locals) = match publication_context.as_ref() {
                        Some(context) => {
                            let (context, publication_task_locals) =
                                copy_publication_invocation_with_buffer(
                                    py,
                                    context,
                                    task_locals,
                                    publication_buffer.clone(),
                                )
                                .map_err(|error| FlowError::Internal(error.to_string()))?;
                            (Some(context), publication_task_locals)
                        }
                        None => { copy_middleware_invocation(py, task_locals) }
                            .map_err(|error| FlowError::Internal(error.to_string()))?,
                    };
                    let args = (
                        PyLLMRequest { inner: request },
                        PyLlmSanitizeRequestContext { inner: context },
                    );
                    let loop_affine = task_locals.is_some();
                    let callback =
                        loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), publication)
                            .map_err(|error| FlowError::Internal(error.to_string()))?;
                    let result = match (invocation_context.as_ref(), publication && !loop_affine) {
                        (Some(context), true) => py
                            .import("nemo_relay._event_sanitizer_context")
                            .and_then(|module| module.getattr("invoke"))
                            .and_then(|invoke| {
                                context.call_method1(
                                    "run",
                                    (invoke, callback.bind(py), args.0, args.1),
                                )
                            }),
                        (Some(context), false) => {
                            context.call_method1("run", (callback.bind(py), args.0, args.1))
                        }
                        (None, true) => py
                            .import("nemo_relay._event_sanitizer_context")
                            .and_then(|module| module.getattr("invoke"))
                            .and_then(|invoke| invoke.call1((callback.bind(py), args.0, args.1))),
                        (None, false) => callback.bind(py).call1(args),
                    }
                    .map_err(|e| FlowError::Internal(e.to_string()))?;
                    split_py_object_or_future_with_locals(
                        py,
                        result.unbind(),
                        task_locals.as_ref(),
                        invocation_context.as_ref(),
                    )
                }))
                .await
                .and_then(|result| {
                    Python::attach(|py| {
                        if result.is_none(py) {
                            Ok(None)
                        } else {
                            result
                                .extract::<PyLLMRequest>(py)
                                .map(|request| Some(request.inner))
                                .map_err(|error| {
                                    FlowError::Internal(format!(
                                        "LLM sanitize request returned unexpected type: {error}"
                                    ))
                                })
                        }
                    })
                });
                if let Err(error) = &result {
                    eprintln!("nemo_relay: Python LLM sanitize request callable failed: {error}");
                }
                result
            })
        },
    )
}

/// Wrap a Python callable `(LlmRequest) -> Optional[str]` for LLM conditional guardrails.
pub fn wrap_py_llm_conditional_fn(py_fn: Py<PyAny>) -> LlmConditionalFn {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(move |request: LlmRequest| {
        let py_fn = py_fn.clone();
        let task_locals = task_locals_with_running_loop(task_locals.as_ref());
        Box::pin(async move {
            let result = resolve_py_object_or_future(Python::attach(|py| {
                let (invocation_context, task_locals) = copy_middleware_invocation(py, task_locals)
                    .map_err(|error| FlowError::Internal(error.to_string()))?;
                let request = PyLLMRequest { inner: request };
                let callback =
                    loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), false)
                        .map_err(|error| FlowError::Internal(error.to_string()))?;
                let result = match invocation_context.as_ref() {
                    Some(context) => context.call_method1("run", (callback.bind(py), request)),
                    None => callback.bind(py).call1((request,)),
                }
                .map_err(|e| FlowError::Internal(e.to_string()))?;
                split_py_object_or_future_with_locals(
                    py,
                    result.unbind(),
                    task_locals.as_ref(),
                    invocation_context.as_ref(),
                )
            }))
            .await?;
            Python::attach(|py| {
                let bound = result.bind(py);
                if bound.is_none() {
                    Ok(None)
                } else {
                    bound.extract::<String>().map(Some).map_err(|e| {
                        FlowError::Internal(format!(
                            "LLM conditional guardrail returned unexpected type: {e}"
                        ))
                    })
                }
            })
        })
    })
}

/// Wrap a Python callable for unified LLM request intercepts.
///
/// The Python function receives ``(name: str, request: LlmRequest, annotated: AnnotatedLLMRequest | None)``
/// and must return ``LLMRequestInterceptOutcome``.
/// When ``annotated`` is present, request content is read-only and provider-body
/// edits must be made through the returned annotation; headers remain writable.
pub fn wrap_py_llm_request_intercept_fn(py_fn: Py<PyAny>) -> LlmRequestInterceptFn {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(
        move |name: String, request: LlmRequest, annotated: Option<AnnotatedLLMRequest>| {
            let py_fn = py_fn.clone();
            let task_locals = task_locals_with_running_loop(task_locals.as_ref());
            Box::pin(async move {
                let result = resolve_py_object_or_future(Python::attach(|py| {
                    let (invocation_context, task_locals) =
                        copy_middleware_invocation(py, task_locals)
                            .map_err(|error| FlowError::Internal(error.to_string()))?;
                    let py_req = PyLLMRequest { inner: request };
                    let py_ann: Py<PyAny> = match annotated {
                        Some(ann) => {
                            let wrapper = PyAnnotatedLLMRequest { inner: ann };
                            wrapper
                                .into_pyobject(py)
                                .map_err(|e| {
                                    FlowError::Internal(format!(
                                        "Failed to convert AnnotatedLLMRequest to Python: {e}"
                                    ))
                                })?
                                .into_any()
                                .unbind()
                        }
                        None => py.None(),
                    };
                    let callback =
                        loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), false)
                            .map_err(|error| FlowError::Internal(error.to_string()))?;
                    let result = match invocation_context.as_ref() {
                        Some(context) => {
                            context.call_method1("run", (callback.bind(py), name, py_req, py_ann))
                        }
                        None => callback.bind(py).call1((name, py_req, py_ann)),
                    }
                    .map_err(|e| {
                        FlowError::Internal(format!("LLM request intercept callable failed: {e}"))
                    })?;

                    split_py_object_or_future_with_locals(
                        py,
                        result.unbind(),
                        task_locals.as_ref(),
                        invocation_context.as_ref(),
                    )
                }))
                .await?;
                Python::attach(|py| {
                    result
                        .extract::<PyLLMRequestInterceptOutcome>(py)
                        .map(|value| value.inner)
                        .map_err(|e| {
                            FlowError::Internal(format!(
                                "LLM request intercept must return LLMRequestInterceptOutcome: {e}"
                            ))
                        })
                })
            })
        },
    )
}

/// Wrap a Python callable `(LlmRequest) -> dict` for LLM execution.
/// Supports both sync and async Python callables.
pub fn wrap_py_llm_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>> + Send + Sync>
{
    let py_fn = std::sync::Arc::new(py_fn);
    let registered_task_locals = capture_python_task_locals();
    Box::new(move |request: LlmRequest| {
        let py_fn = py_fn.clone();
        let task_locals = task_locals_with_running_loop(registered_task_locals.as_ref());
        Box::pin(async move {
            resolve_json_or_future(Python::attach(|py| {
                let (invocation_context, task_locals) = copy_middleware_invocation(py, task_locals)
                    .map_err(|error| FlowError::Internal(error.to_string()))?;
                let py_req = PyLLMRequest { inner: request };
                let callback =
                    loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), false)
                        .map_err(|error| FlowError::Internal(error.to_string()))?;
                let result = match invocation_context.as_ref() {
                    Some(context) => context.call_method1("run", (callback.bind(py), py_req)),
                    None => callback.bind(py).call1((py_req,)),
                }
                .map_err(python_callback_error)?;
                split_json_or_future_with_locals(py, result.unbind(), task_locals.as_ref())
            }))
            .await
        })
    })
}

/// Wrap a Python async generator `(LlmRequest) -> AsyncIterator[Any]` for LLM
/// stream execution.
///
/// The returned future resolves to a Rust stream backed by a Tokio task that
/// repeatedly awaits `__anext__()` and forwards JSON-converted chunks through a
/// channel.
pub fn wrap_py_llm_stream_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<
    dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = FlowResult<LlmJsonStream>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = std::sync::Arc::new(py_fn);
    let registered_task_locals = capture_python_task_locals();
    Box::new(move |request: LlmRequest| {
        let py_fn = py_fn.clone();
        let task_locals = task_locals_with_running_loop(registered_task_locals.as_ref());
        Box::pin(async move {
            let (outcome, invocation_task_locals) = Python::attach(|py| {
                let (invocation_context, task_locals) = copy_middleware_invocation(py, task_locals)
                    .map_err(|error| FlowError::Internal(error.to_string()))?;
                let py_req = PyLLMRequest { inner: request };
                let callback =
                    loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), false)
                        .map_err(|error| FlowError::Internal(error.to_string()))?;
                let result = match invocation_context.as_ref() {
                    Some(context) => context.call_method1("run", (callback.bind(py), py_req)),
                    None => callback.bind(py).call1((py_req,)),
                }
                .map_err(python_callback_error)?;
                let outcome = split_py_object_or_future_with_locals(
                    py,
                    result.unbind(),
                    task_locals.as_ref(),
                    invocation_context.as_ref(),
                )?;
                Ok::<_, FlowError>((outcome, task_locals))
            })?;
            let async_iter = resolve_py_object_or_future(Ok(outcome)).await?;
            stream_from_async_iter(async_iter, invocation_task_locals)
        })
    })
}

/// Wrap a Python callable `(Any) -> None` as a collector for streaming LLM calls.
///
/// The collector is invoked with each intercepted chunk (after stream response
/// intercepts have been applied). It receives a single JSON-converted Python
/// object argument. If the Python callable raises an exception, it is converted
/// to a `FlowError::CallbackException` and returned as `Err`, which terminates the
/// stream. If the callable returns normally (including `None`), the collector
/// returns `Ok(())`.
pub fn wrap_py_collector_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn FnMut(Json) -> std::result::Result<(), FlowError> + Send> {
    Box::new(move |chunk: Json| {
        Python::attach(|py| {
            let py_chunk = json_to_py(py, &chunk)
                .map_err(|e| FlowError::Internal(format!("collector json_to_py failed: {e}")))?;
            py_fn
                .call1(py, (py_chunk,))
                .map_err(python_callback_error)?;
            Ok(())
        })
    })
}

/// Wrap a Python callable `() -> Any` as a finalizer for streaming LLM calls.
///
/// The finalizer is called once when the stream is fully consumed or explicitly
/// closed. Its return value is converted from a Python object to
/// `serde_json::Value` (Json) and used as the aggregated response.
pub fn wrap_py_finalizer_fn(py_fn: Py<PyAny>) -> Box<dyn FnOnce() -> Json + Send> {
    Box::new(move || {
        Python::attach(|py| {
            let result = match py_fn.call0(py) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("nemo_relay: Python finalizer callable failed: {e}");
                    return Json::Null;
                }
            };
            py_to_json(result.bind(py)).unwrap_or_else(|e| {
                eprintln!("nemo_relay: py_to_json failed in finalizer: {e}");
                Json::Null
            })
        })
    })
}

/// Wrap a Python callable `(Json, LlmSanitizeResponseContext) -> Optional[Json]`.
fn wrap_py_llm_sanitize_response_callback(py_fn: Py<PyAny>) -> LlmSanitizeResponseFn {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(move |response: Json, context: LlmSanitizeResponseContext| {
        let py_fn = py_fn.clone();
        let task_locals = task_locals_with_running_loop(task_locals.as_ref());
        let publication_context = publication_context::<PythonPublicationContext>();
        let publication_buffer = capture_nested_publication_buffer();
        let publication = nemo_relay::api::runtime::subscriber_dispatcher::in_dispatcher_callback();
        Box::pin(async move {
            let result = resolve_py_object_or_future(Python::attach(|py| {
                let (invocation_context, task_locals) = match publication_context.as_ref() {
                    Some(context) => {
                        let (context, publication_task_locals) =
                            copy_publication_invocation_with_buffer(
                                py,
                                context,
                                task_locals,
                                publication_buffer.clone(),
                            )
                            .map_err(|error| FlowError::Internal(error.to_string()))?;
                        (Some(context), publication_task_locals)
                    }
                    None => { copy_middleware_invocation(py, task_locals) }
                        .map_err(|error| FlowError::Internal(error.to_string()))?,
                };
                let py_context = PyLlmSanitizeResponseContext { inner: context };
                let py_response = json_to_py(py, &response)
                    .map_err(|error| FlowError::Internal(error.to_string()))?;
                let loop_affine = task_locals.is_some();
                let callback =
                    loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), publication)
                        .map_err(|error| FlowError::Internal(error.to_string()))?;
                let result = match (invocation_context.as_ref(), publication && !loop_affine) {
                    (Some(context), true) => py
                        .import("nemo_relay._event_sanitizer_context")
                        .and_then(|module| module.getattr("invoke"))
                        .and_then(|invoke| {
                            context.call_method1(
                                "run",
                                (invoke, callback.bind(py), py_response, py_context),
                            )
                        }),
                    (Some(context), false) => {
                        context.call_method1("run", (callback.bind(py), py_response, py_context))
                    }
                    (None, true) => py
                        .import("nemo_relay._event_sanitizer_context")
                        .and_then(|module| module.getattr("invoke"))
                        .and_then(|invoke| {
                            invoke.call1((callback.bind(py), py_response, py_context))
                        }),
                    (None, false) => callback.bind(py).call1((py_response, py_context)),
                }
                .map_err(|error| FlowError::Internal(error.to_string()))?;
                split_py_object_or_future_with_locals(
                    py,
                    result.unbind(),
                    task_locals.as_ref(),
                    invocation_context.as_ref(),
                )
            }))
            .await
            .and_then(|result| {
                Python::attach(|py| {
                    if result.is_none(py) {
                        Ok(None)
                    } else {
                        py_to_json(result.bind(py))
                            .map(Some)
                            .map_err(|error| FlowError::Internal(error.to_string()))
                    }
                })
            });
            if let Err(error) = &result {
                eprintln!("nemo_relay: Python LLM sanitize response callable failed: {error}");
            }
            result
        })
    })
}

/// Wrap a Python LLM sanitize-request callback.
pub fn wrap_py_llm_sanitize_request_fn(py_fn: Py<PyAny>) -> PyResult<LlmSanitizeRequestFn> {
    validate_python_llm_sanitizer_signature(&py_fn)?;
    Ok(wrap_py_llm_sanitize_request_callback(py_fn))
}

/// Wrap a Python LLM sanitize-response callback.
pub fn wrap_py_llm_sanitize_response_fn(py_fn: Py<PyAny>) -> PyResult<LlmSanitizeResponseFn> {
    validate_python_llm_sanitizer_signature(&py_fn)?;
    Ok(wrap_py_llm_sanitize_response_callback(py_fn))
}

/// Wrap a Python callable `(Event) -> None` for event subscribers.
pub fn wrap_py_event_subscriber(py_fn: Py<PyAny>) -> EventSubscriberFn {
    Arc::new(move |event: &Event| {
        Python::attach(|py| {
            let result = match event {
                Event::Scope(inner) => py_fn.call1(
                    py,
                    (crate::py_types::PyScopeEvent {
                        inner: inner.clone(),
                    },),
                ),
                Event::Mark(inner) => py_fn.call1(
                    py,
                    (crate::py_types::PyMarkEvent {
                        inner: inner.clone(),
                    },),
                ),
            };
            if let Err(e) = result {
                eprintln!("Event subscriber error: {e}");
            }
        })
    })
}

fn prepare_event_sanitizer_invocation<'py>(
    py: Python<'py>,
    publication_context: Option<&PythonPublicationContext>,
    task_locals: Option<TaskLocals>,
    publication_buffer: Option<PublicationBuffer>,
) -> FlowResult<(Option<Bound<'py, PyAny>>, Option<TaskLocals>)> {
    match publication_context {
        Some(context) => {
            let (context, task_locals) = copy_publication_invocation_with_buffer(
                py,
                context,
                task_locals,
                publication_buffer,
            )
            .map_err(|error| FlowError::Internal(error.to_string()))?;
            Ok((Some(context), task_locals))
        }
        None => copy_middleware_invocation(py, task_locals)
            .map_err(|error| FlowError::Internal(error.to_string())),
    }
}

fn py_event_object(py: Python<'_>, event: &Event) -> PyResult<Py<PyAny>> {
    match event {
        Event::Scope(inner) => Py::new(
            py,
            crate::py_types::PyScopeEvent {
                inner: inner.clone(),
            },
        )
        .map(|value| value.into_any()),
        Event::Mark(inner) => Py::new(
            py,
            crate::py_types::PyMarkEvent {
                inner: inner.clone(),
            },
        )
        .map(|value| value.into_any()),
    }
}

fn call_event_sanitizer(
    py: Python<'_>,
    invoke: &Bound<'_, PyAny>,
    callback: &Py<PyAny>,
    invocation_context: Option<&Bound<'_, PyAny>>,
    loop_affine: bool,
    py_event: Py<PyAny>,
    py_fields: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let result = match (invocation_context, loop_affine) {
        (Some(context), false) => {
            context.call_method1("run", (invoke, callback.bind(py), py_event, py_fields))
        }
        (None, false) => invoke.call1((callback.bind(py), py_event, py_fields)),
        (Some(context), true) => {
            context.call_method1("run", (callback.bind(py), py_event, py_fields))
        }
        (None, true) => callback.bind(py).call1((py_event, py_fields)),
    }?;
    Ok(result.unbind())
}

fn start_py_event_sanitizer(
    py: Python<'_>,
    py_fn: &Py<PyAny>,
    event: &Event,
    fields: &EventSanitizeFields,
    publication_context: Option<&PythonPublicationContext>,
    task_locals: Option<TaskLocals>,
    publication_buffer: Option<PublicationBuffer>,
) -> FlowResult<std::result::Result<Py<PyAny>, PyValueFuture>> {
    let (invocation_context, task_locals) = prepare_event_sanitizer_invocation(
        py,
        publication_context,
        task_locals,
        publication_buffer,
    )?;
    let py_event =
        py_event_object(py, event).map_err(|error| FlowError::Internal(error.to_string()))?;
    let fields_json =
        serde_json::to_value(fields).map_err(|error| FlowError::Internal(error.to_string()))?;
    let py_fields =
        json_to_py(py, &fields_json).map_err(|error| FlowError::Internal(error.to_string()))?;
    let invoke = py
        .import("nemo_relay._event_sanitizer_context")
        .and_then(|module| module.getattr("invoke"))
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let loop_affine = task_locals.is_some();
    let callback = loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), true)
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let result = call_event_sanitizer(
        py,
        &invoke,
        &callback,
        invocation_context.as_ref(),
        loop_affine,
        py_event,
        py_fields,
    )
    .map_err(|error| FlowError::Internal(error.to_string()))?;
    split_py_object_or_future_with_locals(
        py,
        result,
        task_locals.as_ref(),
        invocation_context.as_ref(),
    )
}

/// Wrap a Python callable ``(Event, EventSanitizeFields) -> EventSanitizeFields``.
pub fn wrap_py_event_sanitize_fn(py_fn: Py<PyAny>) -> EventSanitizeFn {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(move |event: Arc<Event>, fields: EventSanitizeFields| {
        let py_fn = py_fn.clone();
        let task_locals = task_locals_with_running_loop(task_locals.as_ref());
        let publication_context = publication_context::<PythonPublicationContext>();
        let publication_buffer = capture_nested_publication_buffer();
        Box::pin(async move {
            let result = Python::attach(|py| {
                start_py_event_sanitizer(
                    py,
                    py_fn.as_ref(),
                    event.as_ref(),
                    &fields,
                    publication_context.as_deref(),
                    task_locals,
                    publication_buffer,
                )
            });
            let result = resolve_py_object_or_future(result)
                .await
                .and_then(|result| {
                    Python::attach(|py| {
                        py_to_json(result.bind(py))
                            .map_err(|error| FlowError::Internal(error.to_string()))
                            .and_then(|value| {
                                serde_json::from_value(value).map_err(|error| {
                                    FlowError::Internal(format!(
                                        "invalid event sanitizer result: {error}"
                                    ))
                                })
                            })
                    })
                });
            if let Err(error) = &result {
                eprintln!("nemo_relay: Python event sanitizer callable failed: {error}");
            }
            result
        })
    })
}

fn call_event_metadata_injector(
    py: Python<'_>,
    invoke: &Bound<'_, PyAny>,
    callback: &Py<PyAny>,
    invocation_context: Option<&Bound<'_, PyAny>>,
    loop_affine: bool,
    py_event: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let result = match (invocation_context, loop_affine) {
        (Some(context), false) => {
            context.call_method1("run", (invoke, callback.bind(py), py_event))
        }
        (None, false) => invoke.call1((callback.bind(py), py_event)),
        (Some(context), true) => context.call_method1("run", (callback.bind(py), py_event)),
        (None, true) => callback.bind(py).call1((py_event,)),
    }?;
    Ok(result.unbind())
}

fn start_py_event_metadata_injector(
    py: Python<'_>,
    py_fn: &Py<PyAny>,
    event: &Event,
    publication_context: Option<&PythonPublicationContext>,
    task_locals: Option<TaskLocals>,
    publication_buffer: Option<PublicationBuffer>,
) -> FlowResult<std::result::Result<Py<PyAny>, PyValueFuture>> {
    let (invocation_context, task_locals) = prepare_event_sanitizer_invocation(
        py,
        publication_context,
        task_locals,
        publication_buffer,
    )?;
    let py_event =
        py_event_object(py, event).map_err(|error| FlowError::Internal(error.to_string()))?;
    let invoke = py
        .import("nemo_relay._event_sanitizer_context")
        .and_then(|module| module.getattr("invoke"))
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let loop_affine = task_locals.is_some();
    let callback = loop_affine_callback(py, py_fn.bind(py), task_locals.as_ref(), true)
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let result = call_event_metadata_injector(
        py,
        &invoke,
        &callback,
        invocation_context.as_ref(),
        loop_affine,
        py_event,
    )
    .map_err(python_callback_error)?;
    split_py_object_or_future_with_locals(
        py,
        result,
        task_locals.as_ref(),
        invocation_context.as_ref(),
    )
}

/// Wrap a Python callable ``(Event) -> dict[str, Json]``.
pub fn wrap_py_event_metadata_injector_fn(py_fn: Py<PyAny>) -> EventMetadataInjectorFn {
    let py_fn = Arc::new(py_fn);
    let task_locals = capture_python_task_locals();
    Arc::new(move |event: Arc<Event>| {
        let py_fn = py_fn.clone();
        let task_locals = task_locals_with_running_loop(task_locals.as_ref());
        let publication_context = publication_context::<PythonPublicationContext>();
        let publication_buffer = capture_nested_publication_buffer();
        Box::pin(async move {
            let result = Python::attach(|py| {
                start_py_event_metadata_injector(
                    py,
                    py_fn.as_ref(),
                    event.as_ref(),
                    publication_context.as_deref(),
                    task_locals,
                    publication_buffer,
                )
            });
            let result = resolve_py_object_or_future(result)
                .await
                .and_then(|result| {
                    Python::attach(|py| {
                        py_to_json(result.bind(py))
                            .map_err(|error| FlowError::Internal(error.to_string()))
                            .and_then(|value| {
                                serde_json::from_value::<BTreeMap<String, Json>>(value).map_err(
                                    |error| {
                                        FlowError::Internal(format!(
                                            "invalid event metadata injector result: {error}"
                                        ))
                                    },
                                )
                            })
                    })
                });
            if let Err(error) = &result {
                eprintln!("nemo_relay: Python event metadata injector failed: {error}");
            }
            result
        })
    })
}
// ---------------------------------------------------------------------------
// LLM Codec wrapper
// ---------------------------------------------------------------------------

/// Wraps a Python object with ``decode``/``encode`` methods into the Rust
/// [`LlmCodec`] trait so it can be stored in the global codec registry.
///
/// The Python codec object must implement:
/// - ``decode(request: LlmRequest) -> AnnotatedLLMRequest``
/// - ``encode(annotated: AnnotatedLLMRequest, original: LlmRequest) -> LlmRequest``
pub(crate) struct PyLlmCodecWrapper {
    pub py_codec: Py<PyAny>,
}

// SAFETY: The Py<PyAny> handle is GIL-independent (ref-counted via Python's
// allocator). All access goes through `Python::attach` which acquires the GIL.
unsafe impl Send for PyLlmCodecWrapper {}
unsafe impl Sync for PyLlmCodecWrapper {}

impl LlmCodec for PyLlmCodecWrapper {
    fn decode(&self, request: &LlmRequest) -> FlowResult<AnnotatedLLMRequest> {
        Python::attach(|py| {
            let py_req = PyLLMRequest {
                inner: request.clone(),
            };
            let result = self
                .py_codec
                .call_method1(py, "decode", (py_req,))
                .map_err(|e| FlowError::Internal(format!("Codec decode() failed: {e}")))?;
            result
                .extract::<PyAnnotatedLLMRequest>(py)
                .map(|r| r.inner)
                .map_err(|e| {
                    FlowError::Internal(format!(
                        "Codec decode() returned unexpected type (expected AnnotatedLLMRequest): {e}"
                    ))
                })
        })
    }

    fn encode(
        &self,
        annotated: &AnnotatedLLMRequest,
        original: &LlmRequest,
    ) -> FlowResult<LlmRequest> {
        Python::attach(|py| {
            let py_ann = PyAnnotatedLLMRequest {
                inner: annotated.clone(),
            };
            let py_orig = PyLLMRequest {
                inner: original.clone(),
            };
            let result = self
                .py_codec
                .call_method1(py, "encode", (py_ann, py_orig))
                .map_err(|e| FlowError::Internal(format!("Codec encode() failed: {e}")))?;
            result
                .extract::<PyLLMRequest>(py)
                .map(|r| r.inner)
                .map_err(|e| {
                    FlowError::Internal(format!(
                        "Codec encode() returned unexpected type (expected LlmRequest): {e}"
                    ))
                })
        })
    }
}

// ---------------------------------------------------------------------------
// LLM Response Codec wrapper
// ---------------------------------------------------------------------------

/// Wraps a Python object implementing the ``LlmResponseCodec`` protocol (``decode_response``).
///
/// The Python response codec object must implement:
/// - ``decode_response(response: Any) -> AnnotatedLLMResponse``
pub(crate) struct PyLlmResponseCodecWrapper {
    pub py_codec: Py<PyAny>,
}

// SAFETY: The Py<PyAny> handle is GIL-independent (ref-counted via Python's
// allocator). All access goes through `Python::attach` which acquires the GIL.
unsafe impl Send for PyLlmResponseCodecWrapper {}
unsafe impl Sync for PyLlmResponseCodecWrapper {}

impl LlmResponseCodec for PyLlmResponseCodecWrapper {
    fn decode_response(&self, response: &Json) -> FlowResult<AnnotatedLLMResponse> {
        Python::attach(|py| {
            let py_resp = json_to_py(py, response).map_err(|e| {
                FlowError::Internal(format!(
                    "Response codec: failed to convert JSON to Python: {e}"
                ))
            })?;
            let result = self
                .py_codec
                .call_method1(py, "decode_response", (py_resp,))
                .map_err(|e| {
                    FlowError::Internal(format!("Response codec decode_response() failed: {e}"))
                })?;
            // PyAnnotatedLLMResponse has skip_from_py_object, so use downcast
            // on the bound reference instead of extract.
            let bound = result.bind(py);
            let py_ref: pyo3::PyRef<'_, PyAnnotatedLLMResponse> = bound
                .cast::<PyAnnotatedLLMResponse>()
                .map_err(|e| {
                    FlowError::Internal(format!(
                        "Response codec decode_response() returned unexpected type (expected AnnotatedLLMResponse): {e}"
                    ))
                })?
                .borrow();
            Ok(py_ref.inner.clone())
        })
    }
}

#[cfg(test)]
#[path = "../tests/coverage/py_callable_coverage_tests.rs"]
mod coverage_tests;
