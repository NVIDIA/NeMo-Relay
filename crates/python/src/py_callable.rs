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

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use nemo_relay::api::runtime::{
    EventSubscriberFn, LlmConditionalFn, LlmExecutionNextFn, LlmRequestInterceptFn,
    LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionNextFn, ToolConditionalFn,
    ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
};
use nemo_relay::error::{FlowError, Result as FlowResult};
use pyo3::prelude::*;
use serde_json::Value as Json;
use tokio_stream::Stream;

use nemo_relay::api::event::Event;
use nemo_relay::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay::codec::request::AnnotatedLlmRequest as AnnotatedLLMRequest;
use nemo_relay::codec::response::AnnotatedLlmResponse as AnnotatedLLMResponse;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};

use crate::convert::{json_to_py, py_to_json};
use crate::py_types::{
    PyAnnotatedLLMRequest, PyAnnotatedLLMResponse, PyLLMRequest, PyLLMRequestInterceptOutcome,
};

type PyValueFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;

fn split_json_or_future(
    py: Python<'_>,
    result: Py<PyAny>,
) -> FlowResult<Result<Json, PyValueFuture>> {
    let bound = result.bind(py);
    if bound.getattr("__await__").is_ok() {
        let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
            .map_err(|e| FlowError::Internal(e.to_string()))?;
        Ok(Err(Box::pin(future) as PyValueFuture))
    } else {
        let json = py_to_json(bound).map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
        Ok(Ok(json))
    }
}

async fn resolve_json_or_future(
    outcome: FlowResult<Result<Json, PyValueFuture>>,
) -> FlowResult<Json> {
    match outcome? {
        Ok(json) => Ok(json),
        Err(future) => {
            let py_result = future
                .await
                .map_err(|e| FlowError::Internal(e.to_string()))?;
            Python::attach(|py| {
                py_to_json(py_result.bind(py))
                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))
            })
        }
    }
}

fn split_py_object_or_future(
    py: Python<'_>,
    result: Py<PyAny>,
) -> FlowResult<Result<Py<PyAny>, PyValueFuture>> {
    let bound = result.bind(py);
    if bound.getattr("__await__").is_ok() {
        let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
            .map_err(|e| FlowError::Internal(e.to_string()))?;
        Ok(Err(Box::pin(future) as PyValueFuture))
    } else {
        Ok(Ok(result))
    }
}

async fn resolve_py_object_or_future(
    outcome: FlowResult<Result<Py<PyAny>, PyValueFuture>>,
) -> FlowResult<Py<PyAny>> {
    match outcome? {
        Ok(value) => Ok(value),
        Err(future) => future.await.map_err(|e| FlowError::Internal(e.to_string())),
    }
}

fn next_async_iter_coro(async_iter: &Arc<Py<PyAny>>) -> FlowResult<Option<Py<PyAny>>> {
    Python::attach(|py| {
        let iter = async_iter.bind(py);
        match iter.call_method0("__anext__") {
            Ok(coro) => Ok(Some(coro.unbind())),
            Err(error) => {
                if error.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                    Ok(None)
                } else {
                    Err(FlowError::Internal(error.to_string()))
                }
            }
        }
    })
}

async fn await_async_iter_value(coro: Py<PyAny>) -> FlowResult<Option<Json>> {
    let future = Python::attach(|py| {
        pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
            .map_err(|e| FlowError::Internal(e.to_string()))
    })?;

    match future.await {
        Ok(result) => Python::attach(|py| {
            py_to_json(result.bind(py))
                .map(Some)
                .map_err(|e| FlowError::Internal(e.to_string()))
        }),
        Err(error) => Python::attach(|py| {
            if error.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                Ok(None)
            } else {
                Err(FlowError::Internal(error.to_string()))
            }
        }),
    }
}

async fn forward_async_iter(
    async_iter: Arc<Py<PyAny>>,
    tx: tokio::sync::mpsc::Sender<FlowResult<Json>>,
) {
    loop {
        let next_value = match next_async_iter_coro(&async_iter) {
            Ok(None) => break,
            Ok(Some(coro)) => await_async_iter_value(coro).await,
            Err(error) => Err(error),
        };

        match next_value {
            Ok(Some(value)) => {
                if tx.send(Ok(value)).await.is_err() {
                    break;
                }
            }
            Ok(None) => break,
            Err(error) => {
                let _ = tx.send(Err(error)).await;
                break;
            }
        }
    }
}

fn stream_from_async_iter(
    async_iter: Py<PyAny>,
) -> FlowResult<Pin<Box<dyn Stream<Item = FlowResult<Json>> + Send>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<FlowResult<Json>>(32);
    let task_locals = Python::attach(|py| {
        pyo3_async_runtimes::tokio::get_current_locals(py)
            .map_err(|e: pyo3::PyErr| FlowError::Internal(e.to_string()))
    })?;

    let async_iter = Arc::new(async_iter);
    tokio::spawn(pyo3_async_runtimes::tokio::scope(task_locals, async move {
        forward_async_iter(async_iter, tx).await;
    }));

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Box::pin(stream) as Pin<Box<dyn Stream<Item = FlowResult<Json>> + Send>>)
}

/// Wrap a Python callable `(str, Json) -> Json` for tool sanitize/intercept fns.
pub fn wrap_py_tool_fn(py_fn: Py<PyAny>) -> ToolSanitizeFn {
    Arc::new(move |name: &str, args: Json| {
        Python::attach(|py| {
            let py_args = match json_to_py(py, &args) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("nemo_relay: json_to_py failed in tool fn for '{name}': {e}");
                    return args.clone();
                }
            };
            let result = match py_fn.call1(py, (name, py_args)) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("nemo_relay: Python tool callable failed for '{name}': {e}");
                    return args.clone();
                }
            };
            py_to_json(result.bind(py)).unwrap_or_else(|e| {
                eprintln!("nemo_relay: py_to_json failed in tool fn for '{name}': {e}");
                args.clone()
            })
        })
    })
}

/// Wrap a Python callable `(str, Json) -> Optional[str]` for tool conditional guardrails.
pub fn wrap_py_tool_conditional_fn(py_fn: Py<PyAny>) -> ToolConditionalFn {
    Arc::new(move |name: &str, args: &Json| {
        Python::attach(|py| {
            let py_args = json_to_py(py, args).map_err(|e| {
                FlowError::Internal(format!(
                    "tool conditional json_to_py failed for '{name}': {e}"
                ))
            })?;
            let result = py_fn.call1(py, (name, py_args)).map_err(|e| {
                FlowError::Internal(format!(
                    "Python tool conditional callable failed for '{name}': {e}"
                ))
            })?;
            let bound = result.bind(py);
            if bound.is_none() {
                Ok(None)
            } else {
                bound.extract::<String>().map(Some).map_err(|e| {
                    FlowError::Internal(format!(
                        "tool conditional guardrail for '{name}' returned unexpected type (expected str or None): {e}"
                    ))
                })
            }
        })
    })
}

/// Wrap a Python callable `(str, Json) -> Json` for tool request intercepts.
pub fn wrap_py_tool_request_intercept_fn(py_fn: Py<PyAny>) -> ToolInterceptFn {
    Arc::new(move |name: &str, args: Json| {
        Python::attach(|py| {
            let py_args = json_to_py(py, &args).map_err(|e| {
                FlowError::Internal(format!("tool callback json_to_py failed for '{name}': {e}"))
            })?;
            let result = py_fn.call1(py, (name, py_args)).map_err(|e| {
                FlowError::Internal(format!("Python tool callable failed for '{name}': {e}"))
            })?;
            py_to_json(result.bind(py)).map_err(|e| {
                FlowError::Internal(format!("tool callback py_to_json failed for '{name}': {e}"))
            })
        })
    })
}

/// Wrap a Python callable `(Json) -> Json` for tool execution intercepts.
/// Supports both sync and async Python callables. If the callable returns a
/// coroutine, it is awaited via the pyo3-async-runtimes bridge.
pub fn wrap_py_tool_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>> + Send + Sync> {
    let py_fn = std::sync::Arc::new(py_fn);
    Box::new(move |args: Json| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            // Call the Python function and check if it returns a coroutine
            let outcome: FlowResult<
                Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
            > = Python::attach(|py| {
                let py_args =
                    json_to_py(py, &args).map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                let result = py_fn
                    .call1(py, (py_args,))
                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;

                // Detect coroutine by checking for __await__
                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| FlowError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future)
                        as Pin<
                            Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                        >))
                } else {
                    let json =
                        py_to_json(bound).map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| FlowError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| FlowError::Internal(e.to_string()))
                    })
                }
            }
        })
    })
}

/// Python-callable wrapper for the Rust `ToolExecutionNextFn`.
///
/// The Python intercept calls `await next(args)` to invoke the next layer
/// in the middleware chain (or the original default function).  The wrapper
/// is reusable — calling `next` multiple times is supported (retry patterns).
#[pyclass]
struct PyToolNextFn {
    inner: ToolExecutionNextFn,
}

#[pymethods]
impl PyToolNextFn {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let next = self.inner.clone();
        let json_args = py_to_json(args)?;
        let future = next(json_args);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = future
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Python::attach(|py| json_to_py(py, &result))
        })
    }
}

/// Python-callable wrapper for the Rust `LlmExecutionNextFn`.
/// Reusable — calling `next` multiple times is supported (retry patterns).
#[pyclass]
struct PyLlmNextFn {
    inner: LlmExecutionNextFn,
}

#[pymethods]
impl PyLlmNextFn {
    fn __call__<'py>(&self, py: Python<'py>, request: PyLLMRequest) -> PyResult<Bound<'py, PyAny>> {
        let next = self.inner.clone();
        let future = next(request.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = future
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
}

#[pymethods]
impl PyLlmStreamNextFn {
    fn __call__<'py>(&self, py: Python<'py>, request: PyLLMRequest) -> PyResult<Bound<'py, PyAny>> {
        let next = self.inner.clone();
        let future = next(request.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let rust_stream = future
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // Drain into mpsc channel and return PyLlmStream
            let (tx, rx) = tokio::sync::mpsc::channel::<FlowResult<Json>>(32);
            tokio::spawn(async move {
                use tokio_stream::StreamExt;
                let mut stream = rust_stream;
                while let Some(item) = stream.next().await {
                    if tx.send(item).await.is_err() {
                        break;
                    }
                }
            });

            Ok(crate::py_types::PyLlmStream {
                receiver: tokio::sync::Mutex::new(rx),
            })
        })
    }
}

/// Wrap a Python callable `(Json, next) -> Json` for tool execution intercepts.
/// The `next` parameter is a `PyToolNextFn` that the Python code can `await`.
pub fn wrap_py_tool_exec_intercept_fn(
    py_fn: Py<PyAny>,
) -> Arc<
    dyn Fn(
            &str,
            Json,
            ToolExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = Arc::new(py_fn);
    Arc::new(move |name: &str, args: Json, next: ToolExecutionNextFn| {
        let py_fn = py_fn.clone();
        let name = name.to_string();
        Box::pin(async move {
            let outcome: FlowResult<
                Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
            > = Python::attach(|py| {
                let py_args =
                    json_to_py(py, &args).map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                let py_next = PyToolNextFn { inner: next };
                let result = py_fn
                    .call1(
                        py,
                        (
                            &name,
                            py_args,
                            py_next
                                .into_pyobject(py)
                                .map_err(|e| FlowError::Internal(e.to_string()))?
                                .into_any(),
                        ),
                    )
                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;

                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| FlowError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future)
                        as Pin<
                            Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                        >))
                } else {
                    let json =
                        py_to_json(bound).map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| FlowError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| FlowError::Internal(e.to_string()))
                    })
                }
            }
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
    Arc::new(
        move |name: &str, request: LlmRequest, next: LlmExecutionNextFn| {
            let py_fn = py_fn.clone();
            let name = name.to_string();
            Box::pin(async move {
                let outcome: FlowResult<
                    Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
                > = Python::attach(|py| {
                    let py_req = PyLLMRequest { inner: request };
                    let py_next = PyLlmNextFn { inner: next };
                    let result = py_fn
                        .call1(
                            py,
                            (
                                &name,
                                py_req
                                    .into_pyobject(py)
                                    .map_err(|e| FlowError::Internal(e.to_string()))?
                                    .into_any(),
                                py_next
                                    .into_pyobject(py)
                                    .map_err(|e| FlowError::Internal(e.to_string()))?
                                    .into_any(),
                            ),
                        )
                        .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;

                    let bound = result.bind(py);
                    if bound.getattr("__await__").is_ok() {
                        let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                            .map_err(|e| FlowError::Internal(e.to_string()))?;
                        Ok(Err(Box::pin(future)
                            as Pin<
                                Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                            >))
                    } else {
                        let json = py_to_json(bound)
                            .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                        Ok(Ok(json))
                    }
                });

                match outcome? {
                    Ok(json) => Ok(json),
                    Err(future) => {
                        let py_result = future
                            .await
                            .map_err(|e| FlowError::Internal(e.to_string()))?;
                        Python::attach(|py| {
                            py_to_json(py_result.bind(py))
                                .map_err(|e: PyErr| FlowError::Internal(e.to_string()))
                        })
                    }
                }
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
        ) -> Pin<
            Box<
                dyn Future<
                        Output = FlowResult<Pin<Box<dyn Stream<Item = FlowResult<Json>> + Send>>>,
                    > + Send,
            >,
        > + Send
        + Sync,
> {
    let py_fn = Arc::new(py_fn);
    Arc::new(
        move |_name: &str, request: LlmRequest, next: LlmStreamExecutionNextFn| {
            let py_fn = py_fn.clone();
            Box::pin(async move {
                let async_iter = resolve_py_object_or_future(Python::attach(|py| {
                    let py_req = PyLLMRequest { inner: request };
                    let py_next = PyLlmStreamNextFn { inner: next };
                    let result = py_fn
                        .call1(
                            py,
                            (
                                py_req
                                    .into_pyobject(py)
                                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?
                                    .into_any(),
                                py_next
                                    .into_pyobject(py)
                                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?
                                    .into_any(),
                            ),
                        )
                        .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                    split_py_object_or_future(py, result)
                }))
                .await?;

                stream_from_async_iter(async_iter)
            })
        },
    )
}

/// Wrap a Python callable `(LlmRequest) -> LlmRequest` for LLM sanitize request guardrails.
pub fn wrap_py_llm_sanitize_request_fn(py_fn: Py<PyAny>) -> LlmSanitizeRequestFn {
    Arc::new(move |request: LlmRequest| {
        Python::attach(|py| {
            let py_req = PyLLMRequest {
                inner: request.clone(),
            };
            let result = match py_fn.call1(py, (py_req,)) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("nemo_relay: LLM sanitize request guardrail callable failed: {e}");
                    return request;
                }
            };
            let extracted = result.extract::<PyLLMRequest>(py);
            match extracted {
                Ok(r) => r.inner,
                Err(e) => {
                    eprintln!(
                        "nemo_relay: LLM sanitize request guardrail returned unexpected type \
                         (expected LlmRequest): {e}"
                    );
                    request
                }
            }
        })
    })
}

/// Wrap a Python callable `(LlmRequest) -> Optional[str]` for LLM conditional guardrails.
pub fn wrap_py_llm_conditional_fn(py_fn: Py<PyAny>) -> LlmConditionalFn {
    Arc::new(move |request: &LlmRequest| {
        Python::attach(|py| {
            let py_req = PyLLMRequest {
                inner: request.clone(),
            };
            let result = py_fn.call1(py, (py_req,)).map_err(|e| {
                FlowError::Internal(format!("LLM conditional guardrail callable failed: {e}"))
            })?;
            let bound = result.bind(py);
            if bound.is_none() {
                Ok(None)
            } else {
                bound.extract::<String>().map(Some).map_err(|e| {
                    FlowError::Internal(format!(
                        "LLM conditional guardrail returned unexpected type (expected str or None): {e}"
                    ))
                })
            }
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
    Arc::new(
        move |name: &str,
              request: LlmRequest,
              annotated: Option<AnnotatedLLMRequest>|
              -> FlowResult<LlmRequestInterceptOutcome> {
            Python::attach(|py| {
                let py_req = PyLLMRequest {
                    inner: request.clone(),
                };
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
                let result = py_fn.call1(py, (name, py_req, py_ann)).map_err(|e| {
                    FlowError::Internal(format!("LLM request intercept callable failed: {e}"))
                })?;

                result
                    .extract::<PyLLMRequestInterceptOutcome>(py)
                    .map(|value| value.inner)
                    .map_err(|e| {
                        FlowError::Internal(format!(
                            "LLM request intercept must return LLMRequestInterceptOutcome: {e}"
                        ))
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
    Box::new(move |request: LlmRequest| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            resolve_json_or_future(Python::attach(|py| {
                let py_req = PyLLMRequest { inner: request };
                let result = py_fn
                    .call1(py, (py_req,))
                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                split_json_or_future(py, result)
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
    dyn Fn(
            LlmRequest,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = FlowResult<Pin<Box<dyn Stream<Item = FlowResult<Json>> + Send>>>,
                    > + Send,
            >,
        > + Send
        + Sync,
> {
    let py_fn = std::sync::Arc::new(py_fn);
    Box::new(move |request: LlmRequest| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            let async_iter: Py<PyAny> = Python::attach(|py| {
                let py_req = PyLLMRequest { inner: request };
                py_fn
                    .call1(py, (py_req,))
                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))
            })?;
            stream_from_async_iter(async_iter)
        })
    })
}

/// Wrap a Python callable `(Any) -> None` as a collector for streaming LLM calls.
///
/// The collector is invoked with each intercepted chunk (after stream response
/// intercepts have been applied). It receives a single JSON-converted Python
/// object argument. If the Python callable raises an exception, it is converted
/// to a `FlowError::Internal` and returned as `Err`, which terminates the
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
                .map_err(|e| FlowError::Internal(format!("Python collector error: {e}")))?;
            Ok(())
        })
    })
}

/// Wrap a Python callable `() -> Any` as a finalizer for streaming LLM calls.
///
/// The finalizer is called once when the stream is fully consumed. Its return
/// value is converted from a Python object to `serde_json::Value` (Json) and
/// used as the aggregated response.
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

/// Wrap a Python callable `(dict) -> dict` for LLM sanitize response guardrails.
pub fn wrap_py_llm_sanitize_response_fn(py_fn: Py<PyAny>) -> LlmSanitizeResponseFn {
    Arc::new(move |response: Json| {
        Python::attach(|py| {
            let py_resp = match json_to_py(py, &response) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "nemo_relay: json_to_py failed in LLM sanitize response guardrail: {e}"
                    );
                    return response.clone();
                }
            };
            let result = match py_fn.call1(py, (py_resp,)) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("nemo_relay: LLM sanitize response guardrail callable failed: {e}");
                    return response.clone();
                }
            };
            py_to_json(result.bind(py)).unwrap_or_else(|e| {
                eprintln!("nemo_relay: py_to_json failed in LLM sanitize response guardrail: {e}");
                response.clone()
            })
        })
    })
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
