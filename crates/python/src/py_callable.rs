// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::type_complexity)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::Value as Json;
use tokio_stream::Stream;

use nvmagic_core::types::{LLMRequest, LLMResponse};
use nvmagic_core::{LlmExecutionNextFn, LlmStreamExecutionNextFn, MagicError, ToolExecutionNextFn};

use crate::convert::{json_to_py, py_to_json};
use crate::py_types::{PyLLMRequest, PyLLMResponse};

/// Wrap a Python callable `(str, Json) -> Json` for tool sanitize/intercept fns.
pub fn wrap_py_tool_fn(py_fn: Py<PyAny>) -> Box<dyn Fn(&str, Json) -> Json + Send + Sync> {
    Box::new(move |name: &str, args: Json| {
        Python::attach(|py| {
            let py_args = json_to_py(py, &args).expect("json_to_py failed");
            let result = py_fn
                .call1(py, (name, py_args))
                .expect("Python callable failed");
            py_to_json(result.bind(py)).expect("py_to_json failed")
        })
    })
}

/// Wrap a Python callable `(str, Json) -> Optional[str]` for tool conditional guardrails.
pub fn wrap_py_tool_conditional_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(&str, &Json) -> Option<String> + Send + Sync> {
    Box::new(move |name: &str, args: &Json| {
        Python::attach(|py| {
            let py_args = json_to_py(py, args).expect("json_to_py failed");
            let result = py_fn
                .call1(py, (name, py_args))
                .expect("Python callable failed");
            let bound = result.bind(py);
            if bound.is_none() {
                None
            } else {
                Some(bound.extract::<String>().expect("Expected str or None"))
            }
        })
    })
}

/// Wrap a Python callable `(str, Json) -> bool` for tool execution conditional.
pub fn wrap_py_tool_exec_conditional_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(&str, &Json) -> bool + Send + Sync> {
    Box::new(move |name: &str, args: &Json| {
        Python::attach(|py| {
            let py_args = json_to_py(py, args).expect("json_to_py failed");
            let result = py_fn
                .call1(py, (name, py_args))
                .expect("Python callable failed");
            result.extract::<bool>(py).expect("Expected bool")
        })
    })
}

/// Wrap a Python callable `(Json) -> Json` for tool execution intercepts.
/// Supports both sync and async Python callables. If the callable returns a
/// coroutine, it is awaited via the pyo3-async-runtimes bridge.
pub fn wrap_py_tool_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<
    dyn Fn(Json) -> Pin<Box<dyn Future<Output = nvmagic_core::Result<Json>> + Send>> + Send + Sync,
> {
    let py_fn = std::sync::Arc::new(py_fn);
    Box::new(move |args: Json| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            // Call the Python function and check if it returns a coroutine
            let outcome: nvmagic_core::Result<
                Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
            > = Python::attach(|py| {
                let py_args = json_to_py(py, &args)
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;
                let result = py_fn
                    .call1(py, (py_args,))
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;

                // Detect coroutine by checking for __await__
                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| MagicError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future)
                        as Pin<
                            Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                        >))
                } else {
                    let json = py_to_json(bound)
                        .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| MagicError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| MagicError::Internal(e.to_string()))
                    })
                }
            }
        })
    })
}

/// Python-callable wrapper for the Rust `ToolExecutionNextFn`.
///
/// The Python intercept calls `await next(args)` to invoke the next layer
/// in the middleware chain (or the original default function).
#[pyclass]
struct PyToolNextFn {
    inner: std::sync::Mutex<Option<ToolExecutionNextFn>>,
}

#[pymethods]
impl PyToolNextFn {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let next = self.inner.lock().unwrap().take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("next() called more than once")
        })?;
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
#[pyclass]
struct PyLlmNextFn {
    inner: std::sync::Mutex<Option<LlmExecutionNextFn>>,
}

#[pymethods]
impl PyLlmNextFn {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        native: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let next = self.inner.lock().unwrap().take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("next() called more than once")
        })?;
        let json_native = py_to_json(native)?;
        let future = next(json_native);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = future
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Python::attach(|py| json_to_py(py, &result))
        })
    }
}

/// Python-callable wrapper for the Rust `LlmStreamExecutionNextFn`.
#[pyclass]
struct PyLlmStreamNextFn {
    inner: std::sync::Mutex<Option<LlmStreamExecutionNextFn>>,
}

#[pymethods]
impl PyLlmStreamNextFn {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        native: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let next = self.inner.lock().unwrap().take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("next() called more than once")
        })?;
        let json_native = py_to_json(native)?;
        let future = next(json_native);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let rust_stream = future
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // Drain into mpsc channel and return PyLlmStream
            let (tx, rx) = tokio::sync::mpsc::channel::<nvmagic_core::Result<Json>>(32);
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
            Json,
            ToolExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = nvmagic_core::Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = Arc::new(py_fn);
    Arc::new(move |args: Json, next: ToolExecutionNextFn| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            let outcome: nvmagic_core::Result<
                Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
            > = Python::attach(|py| {
                let py_args = json_to_py(py, &args)
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;
                let py_next = PyToolNextFn {
                    inner: std::sync::Mutex::new(Some(next)),
                };
                let result = py_fn
                    .call1(
                        py,
                        (
                            py_args,
                            py_next
                                .into_pyobject(py)
                                .map_err(|e| MagicError::Internal(e.to_string()))?
                                .into_any(),
                        ),
                    )
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;

                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| MagicError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future)
                        as Pin<
                            Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                        >))
                } else {
                    let json = py_to_json(bound)
                        .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| MagicError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| MagicError::Internal(e.to_string()))
                    })
                }
            }
        })
    })
}

/// Wrap a Python callable `(dict, next) -> dict` for LLM execution intercepts.
pub fn wrap_py_llm_exec_intercept_fn(
    py_fn: Py<PyAny>,
) -> Arc<
    dyn Fn(
            Json,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = nvmagic_core::Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = Arc::new(py_fn);
    Arc::new(move |native: Json, next: LlmExecutionNextFn| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            let outcome: nvmagic_core::Result<
                Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
            > = Python::attach(|py| {
                let py_native = json_to_py(py, &native)
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;
                let py_next = PyLlmNextFn {
                    inner: std::sync::Mutex::new(Some(next)),
                };
                let result = py_fn
                    .call1(
                        py,
                        (
                            py_native,
                            py_next
                                .into_pyobject(py)
                                .map_err(|e| MagicError::Internal(e.to_string()))?
                                .into_any(),
                        ),
                    )
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;

                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| MagicError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future)
                        as Pin<
                            Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                        >))
                } else {
                    let json = py_to_json(bound)
                        .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| MagicError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| MagicError::Internal(e.to_string()))
                    })
                }
            }
        })
    })
}

/// Wrap a Python callable `(dict, next) -> AsyncIterator[Any]` for LLM stream execution intercepts.
pub fn wrap_py_llm_stream_exec_intercept_fn(
    py_fn: Py<PyAny>,
) -> Arc<
    dyn Fn(
            Json,
            LlmStreamExecutionNextFn,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = nvmagic_core::Result<
                            Pin<Box<dyn Stream<Item = nvmagic_core::Result<Json>> + Send>>,
                        >,
                    > + Send,
            >,
        > + Send
        + Sync,
> {
    let py_fn = Arc::new(py_fn);
    Arc::new(move |native: Json, next: LlmStreamExecutionNextFn| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            // Call the Python function to get the async iterator object
            let async_iter: Py<PyAny> = Python::attach(|py| {
                let py_native = json_to_py(py, &native)
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;
                let py_next = PyLlmStreamNextFn {
                    inner: std::sync::Mutex::new(Some(next)),
                };
                py_fn
                    .call1(
                        py,
                        (
                            py_native,
                            py_next
                                .into_pyobject(py)
                                .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?
                                .into_any(),
                        ),
                    )
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))
            })?;

            let (tx, rx) = tokio::sync::mpsc::channel::<nvmagic_core::Result<Json>>(32);

            let task_locals = Python::attach(|py| {
                pyo3_async_runtimes::tokio::get_current_locals(py)
                    .map_err(|e: pyo3::PyErr| MagicError::Internal(e.to_string()))
            })?;

            let async_iter = Arc::new(async_iter);
            tokio::spawn(pyo3_async_runtimes::tokio::scope(task_locals, async move {
                loop {
                    let async_iter_clone = async_iter.clone();
                    let coro_result: Result<Option<Py<PyAny>>, _> = Python::attach(|py| {
                        let iter = async_iter_clone.bind(py);
                        match iter.call_method0("__anext__") {
                            Ok(coro) => Ok(Some(coro.unbind())),
                            Err(e) => {
                                if e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                                    Ok(None)
                                } else {
                                    Err(MagicError::Internal(e.to_string()))
                                }
                            }
                        }
                    });

                    match coro_result {
                        Ok(None) => break,
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            break;
                        }
                        Ok(Some(coro)) => {
                            let future_result = Python::attach(|py| {
                                pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
                                    .map_err(|e| MagicError::Internal(e.to_string()))
                            });
                            let awaited: Result<Json, _> = match future_result {
                                Ok(future) => match future.await {
                                    Ok(result) => Python::attach(|py| {
                                        py_to_json(result.bind(py))
                                            .map_err(|e| MagicError::Internal(e.to_string()))
                                    }),
                                    Err(e) => Python::attach(|py| {
                                        if e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                                            return Err(MagicError::Internal("__stop__".into()));
                                        }
                                        Err(MagicError::Internal(e.to_string()))
                                    }),
                                },
                                Err(e) => Err(e),
                            };

                            match awaited {
                                Ok(value) => {
                                    if tx.send(Ok(value)).await.is_err() {
                                        break;
                                    }
                                }
                                Err(MagicError::Internal(ref msg)) if msg == "__stop__" => break,
                                Err(e) => {
                                    let _ = tx.send(Err(e)).await;
                                    break;
                                }
                            }
                        }
                    }
                }
            }));

            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            Ok(Box::pin(stream)
                as Pin<
                    Box<dyn Stream<Item = nvmagic_core::Result<Json>> + Send>,
                >)
        })
    })
}

/// Wrap a Python callable `(LLMRequest) -> LLMRequest` for LLM sanitize request guardrails.
pub fn wrap_py_llm_sanitize_request_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(LLMRequest) -> LLMRequest + Send + Sync> {
    Box::new(move |request: LLMRequest| {
        Python::attach(|py| {
            let py_req = PyLLMRequest { inner: request };
            let result = py_fn.call1(py, (py_req,)).expect("Python callable failed");
            result
                .extract::<PyLLMRequest>(py)
                .expect("Expected LLMRequest")
                .inner
        })
    })
}

/// Wrap a Python callable `(Json) -> Json` for LLM sanitize response / response intercepts.
pub fn wrap_py_json_fn(py_fn: Py<PyAny>) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    Box::new(move |value: Json| {
        Python::attach(|py| {
            let py_val = json_to_py(py, &value).expect("json_to_py failed");
            let result = py_fn.call1(py, (py_val,)).expect("Python callable failed");
            py_to_json(result.bind(py)).expect("py_to_json failed")
        })
    })
}

/// Wrap a Python callable `(LLMRequest) -> Optional[str]` for LLM conditional guardrails.
pub fn wrap_py_llm_conditional_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(&LLMRequest) -> Option<String> + Send + Sync> {
    Box::new(move |request: &LLMRequest| {
        Python::attach(|py| {
            let py_req = PyLLMRequest {
                inner: request.clone(),
            };
            let result = py_fn.call1(py, (py_req,)).expect("Python callable failed");
            let bound = result.bind(py);
            if bound.is_none() {
                None
            } else {
                Some(bound.extract::<String>().expect("Expected str or None"))
            }
        })
    })
}

/// Wrap a Python callable `(dict) -> dict` for LLM request intercepts.
/// Request intercepts now operate on opaque Json, not LLMRequest.
pub fn wrap_py_llm_request_intercept_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    wrap_py_json_fn(py_fn)
}

/// Wrap a Python callable `(dict) -> bool` for LLM execution conditional.
pub fn wrap_py_llm_exec_conditional_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(&Json) -> bool + Send + Sync> {
    Box::new(move |native: &Json| {
        Python::attach(|py| {
            let py_native = json_to_py(py, native).expect("json_to_py failed");
            let result = py_fn
                .call1(py, (py_native,))
                .expect("Python callable failed");
            result.extract::<bool>(py).expect("Expected bool")
        })
    })
}

/// Wrap a Python callable `(dict) -> dict` for LLM execution.
/// Supports both sync and async Python callables.
pub fn wrap_py_llm_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<
    dyn Fn(Json) -> Pin<Box<dyn Future<Output = nvmagic_core::Result<Json>> + Send>> + Send + Sync,
> {
    let py_fn = std::sync::Arc::new(py_fn);
    Box::new(move |native: Json| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            let outcome: nvmagic_core::Result<
                Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
            > = Python::attach(|py| {
                let py_native = json_to_py(py, &native)
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;
                let result = py_fn
                    .call1(py, (py_native,))
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;

                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| MagicError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future)
                        as Pin<
                            Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                        >))
                } else {
                    let json = py_to_json(bound)
                        .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| MagicError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| MagicError::Internal(e.to_string()))
                    })
                }
            }
        })
    })
}

/// Wrap a Python async generator `(dict) -> AsyncIterator[Any]` for LLM stream execution.
/// Returns a future that resolves to a `Stream<Item = Result<Json>>`.
pub fn wrap_py_llm_stream_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<
    dyn Fn(
            Json,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = nvmagic_core::Result<
                            Pin<Box<dyn Stream<Item = nvmagic_core::Result<Json>> + Send>>,
                        >,
                    > + Send,
            >,
        > + Send
        + Sync,
> {
    let py_fn = std::sync::Arc::new(py_fn);
    Box::new(move |native: Json| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            // Call the Python function to get the async iterator object
            let async_iter: Py<PyAny> = Python::attach(|py| {
                let py_native = json_to_py(py, &native)
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))?;
                py_fn
                    .call1(py, (py_native,))
                    .map_err(|e: PyErr| MagicError::Internal(e.to_string()))
            })?;

            let (tx, rx) = tokio::sync::mpsc::channel::<nvmagic_core::Result<Json>>(32);

            // Capture the Python event loop context so the spawned task can use
            // pyo3_async_runtimes::tokio::into_future (which needs TaskLocals).
            let task_locals = Python::attach(|py| {
                pyo3_async_runtimes::tokio::get_current_locals(py)
                    .map_err(|e: pyo3::PyErr| MagicError::Internal(e.to_string()))
            })?;

            // Spawn a task that drains the Python async iterator into the channel.
            // Wrap with scope() to propagate the event loop context.
            let async_iter = std::sync::Arc::new(async_iter);
            tokio::spawn(pyo3_async_runtimes::tokio::scope(task_locals, async move {
                loop {
                    let async_iter_clone = async_iter.clone();
                    // Call __anext__ to get the coroutine
                    let coro_result: Result<Option<Py<PyAny>>, _> = Python::attach(|py| {
                        let iter = async_iter_clone.bind(py);
                        match iter.call_method0("__anext__") {
                            Ok(coro) => Ok(Some(coro.unbind())),
                            Err(e) => {
                                if e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                                    Ok(None)
                                } else {
                                    Err(MagicError::Internal(e.to_string()))
                                }
                            }
                        }
                    });

                    match coro_result {
                        Ok(None) => break, // StopAsyncIteration — stream done
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            break;
                        }
                        Ok(Some(coro)) => {
                            // Await the coroutine using pyo3_async_runtimes
                            let future_result = Python::attach(|py| {
                                pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
                                    .map_err(|e| MagicError::Internal(e.to_string()))
                            });
                            let awaited: Result<Json, _> = match future_result {
                                Ok(future) => match future.await {
                                    Ok(result) => Python::attach(|py| {
                                        py_to_json(result.bind(py))
                                            .map_err(|e| MagicError::Internal(e.to_string()))
                                    }),
                                    Err(e) => Python::attach(|py| {
                                        if e.is_instance_of::<
                                            pyo3::exceptions::PyStopAsyncIteration,
                                        >(py)
                                        {
                                            return Err(MagicError::Internal("__stop__".into()));
                                        }
                                        Err(MagicError::Internal(e.to_string()))
                                    }),
                                },
                                Err(e) => Err(e),
                            };

                            match awaited {
                                Ok(value) => {
                                    if tx.send(Ok(value)).await.is_err() {
                                        break; // receiver dropped
                                    }
                                }
                                Err(MagicError::Internal(ref msg)) if msg == "__stop__" => break,
                                Err(e) => {
                                    let _ = tx.send(Err(e)).await;
                                    break;
                                }
                            }
                        }
                    }
                }
            }));

            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            Ok(Box::pin(stream)
                as Pin<
                    Box<dyn Stream<Item = nvmagic_core::Result<Json>> + Send>,
                >)
        })
    })
}

/// Wrap a Python callable `(Any) -> None` as a collector for streaming LLM calls.
///
/// The collector is invoked with each intercepted chunk (after stream response
/// intercepts have been applied). It receives a single JSON-converted Python
/// object argument and returns nothing.
pub fn wrap_py_collector_fn(py_fn: Py<PyAny>) -> Box<dyn FnMut(Json) + Send> {
    Box::new(move |chunk: Json| {
        Python::attach(|py| {
            let py_chunk = json_to_py(py, &chunk).expect("json_to_py failed in collector");
            py_fn
                .call1(py, (py_chunk,))
                .expect("Python collector callable failed");
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
            let result = py_fn.call0(py).expect("Python finalizer callable failed");
            py_to_json(result.bind(py)).expect("py_to_json failed in finalizer")
        })
    })
}

/// Wrap a Python callable `(LLMResponse) -> LLMResponse` for LLM response intercepts.
pub fn wrap_py_llm_response_intercept_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(LLMResponse) -> LLMResponse + Send + Sync> {
    Box::new(move |response: LLMResponse| {
        Python::attach(|py| {
            let py_resp = PyLLMResponse { inner: response };
            let result = py_fn.call1(py, (py_resp,)).expect("Python callable failed");
            result
                .extract::<PyLLMResponse>(py)
                .expect("Expected LLMResponse")
                .inner
        })
    })
}

/// Wrap a Python callable `(LLMResponse) -> LLMResponse` for LLM sanitize response guardrails.
pub fn wrap_py_llm_sanitize_response_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(LLMResponse) -> LLMResponse + Send + Sync> {
    wrap_py_llm_response_intercept_fn(py_fn)
}

/// Wrap a Python callable `(Event) -> None` for event subscribers.
pub fn wrap_py_event_subscriber(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(&nvmagic_core::Event) + Send + Sync> {
    Box::new(move |event: &nvmagic_core::Event| {
        Python::attach(|py| {
            let py_event = crate::py_types::PyEvent {
                inner: event.clone(),
            };
            if let Err(e) = py_fn.call1(py, (py_event,)) {
                eprintln!("Event subscriber error: {e}");
            }
        })
    })
}
