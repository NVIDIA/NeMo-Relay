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

use pyo3::prelude::*;
use serde_json::Value as Json;
use tokio_stream::Stream;

use nvidia_nat_nexus_core::codec::{AnnotatedLLMRequest, LlmCodec};
use nvidia_nat_nexus_core::types::LLMRequest;
use nvidia_nat_nexus_core::{
    LlmConditionalFn, LlmExecutionNextFn, LlmRequestInterceptFn, LlmStreamExecutionNextFn,
    NexusError, ToolConditionalFn, ToolExecutionNextFn, ToolInterceptFn,
};

use crate::convert::{json_to_py, py_to_json};
use crate::py_types::{PyAnnotatedLLMRequest, PyLLMRequest};

/// Wrap a Python callable `(str, Json) -> Json` for tool sanitize/intercept fns.
pub fn wrap_py_tool_fn(py_fn: Py<PyAny>) -> Box<dyn Fn(&str, Json) -> Json + Send + Sync> {
    Box::new(move |name: &str, args: Json| {
        Python::attach(|py| {
            let py_args = match json_to_py(py, &args) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("nat_nexus: json_to_py failed in tool fn for '{name}': {e}");
                    return args.clone();
                }
            };
            let result = match py_fn.call1(py, (name, py_args)) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("nat_nexus: Python tool callable failed for '{name}': {e}");
                    return args.clone();
                }
            };
            py_to_json(result.bind(py)).unwrap_or_else(|e| {
                eprintln!("nat_nexus: py_to_json failed in tool fn for '{name}': {e}");
                args.clone()
            })
        })
    })
}

/// Wrap a Python callable `(str, Json) -> Optional[str]` for tool conditional guardrails.
pub fn wrap_py_tool_conditional_fn(py_fn: Py<PyAny>) -> ToolConditionalFn {
    Box::new(move |name: &str, args: &Json| {
        Python::attach(|py| {
            let py_args = json_to_py(py, args).map_err(|e| {
                NexusError::Internal(format!(
                    "tool conditional json_to_py failed for '{name}': {e}"
                ))
            })?;
            let result = py_fn.call1(py, (name, py_args)).map_err(|e| {
                NexusError::Internal(format!(
                    "Python tool conditional callable failed for '{name}': {e}"
                ))
            })?;
            let bound = result.bind(py);
            if bound.is_none() {
                Ok(None)
            } else {
                bound.extract::<String>().map(Some).map_err(|e| {
                    NexusError::Internal(format!(
                        "tool conditional guardrail for '{name}' returned unexpected type (expected str or None): {e}"
                    ))
                })
            }
        })
    })
}

/// Wrap a Python callable `(str, Json) -> Json` for tool request intercepts.
pub fn wrap_py_tool_request_intercept_fn(py_fn: Py<PyAny>) -> ToolInterceptFn {
    Box::new(move |name: &str, args: Json| {
        Python::attach(|py| {
            let py_args = json_to_py(py, &args).map_err(|e| {
                NexusError::Internal(format!("tool callback json_to_py failed for '{name}': {e}"))
            })?;
            let result = py_fn.call1(py, (name, py_args)).map_err(|e| {
                NexusError::Internal(format!("Python tool callable failed for '{name}': {e}"))
            })?;
            py_to_json(result.bind(py)).map_err(|e| {
                NexusError::Internal(format!("tool callback py_to_json failed for '{name}': {e}"))
            })
        })
    })
}

/// Wrap a Python callable `(Json) -> Json` for tool execution intercepts.
/// Supports both sync and async Python callables. If the callable returns a
/// coroutine, it is awaited via the pyo3-async-runtimes bridge.
pub fn wrap_py_tool_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<
    dyn Fn(Json) -> Pin<Box<dyn Future<Output = nvidia_nat_nexus_core::Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = std::sync::Arc::new(py_fn);
    Box::new(move |args: Json| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            // Call the Python function and check if it returns a coroutine
            let outcome: nvidia_nat_nexus_core::Result<
                Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
            > = Python::attach(|py| {
                let py_args = json_to_py(py, &args)
                    .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?;
                let result = py_fn
                    .call1(py, (py_args,))
                    .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?;

                // Detect coroutine by checking for __await__
                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| NexusError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future)
                        as Pin<
                            Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                        >))
                } else {
                    let json = py_to_json(bound)
                        .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| NexusError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| NexusError::Internal(e.to_string()))
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
            let (tx, rx) = tokio::sync::mpsc::channel::<nvidia_nat_nexus_core::Result<Json>>(32);
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
        ) -> Pin<Box<dyn Future<Output = nvidia_nat_nexus_core::Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = Arc::new(py_fn);
    Arc::new(move |name: &str, args: Json, next: ToolExecutionNextFn| {
        let py_fn = py_fn.clone();
        let name = name.to_string();
        Box::pin(async move {
            let outcome: nvidia_nat_nexus_core::Result<
                Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
            > = Python::attach(|py| {
                let py_args = json_to_py(py, &args)
                    .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?;
                let py_next = PyToolNextFn { inner: next };
                let result = py_fn
                    .call1(
                        py,
                        (
                            &name,
                            py_args,
                            py_next
                                .into_pyobject(py)
                                .map_err(|e| NexusError::Internal(e.to_string()))?
                                .into_any(),
                        ),
                    )
                    .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?;

                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| NexusError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future)
                        as Pin<
                            Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                        >))
                } else {
                    let json = py_to_json(bound)
                        .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| NexusError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| NexusError::Internal(e.to_string()))
                    })
                }
            }
        })
    })
}

/// Wrap a Python callable `(name, LLMRequest, next) -> dict` for LLM execution intercepts.
pub fn wrap_py_llm_exec_intercept_fn(
    py_fn: Py<PyAny>,
) -> Arc<
    dyn Fn(
            &str,
            LLMRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = nvidia_nat_nexus_core::Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = Arc::new(py_fn);
    Arc::new(
        move |name: &str, request: LLMRequest, next: LlmExecutionNextFn| {
            let py_fn = py_fn.clone();
            let name = name.to_string();
            Box::pin(async move {
                let outcome: nvidia_nat_nexus_core::Result<
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
                                    .map_err(|e| NexusError::Internal(e.to_string()))?
                                    .into_any(),
                                py_next
                                    .into_pyobject(py)
                                    .map_err(|e| NexusError::Internal(e.to_string()))?
                                    .into_any(),
                            ),
                        )
                        .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?;

                    let bound = result.bind(py);
                    if bound.getattr("__await__").is_ok() {
                        let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                            .map_err(|e| NexusError::Internal(e.to_string()))?;
                        Ok(Err(Box::pin(future)
                            as Pin<
                                Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                            >))
                    } else {
                        let json = py_to_json(bound)
                            .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?;
                        Ok(Ok(json))
                    }
                });

                match outcome? {
                    Ok(json) => Ok(json),
                    Err(future) => {
                        let py_result = future
                            .await
                            .map_err(|e| NexusError::Internal(e.to_string()))?;
                        Python::attach(|py| {
                            py_to_json(py_result.bind(py))
                                .map_err(|e: PyErr| NexusError::Internal(e.to_string()))
                        })
                    }
                }
            })
        },
    )
}

/// Wrap a Python callable `(LLMRequest, next) -> AsyncIterator[Any]` for LLM stream execution intercepts.
pub fn wrap_py_llm_stream_exec_intercept_fn(
    py_fn: Py<PyAny>,
) -> Arc<
    dyn Fn(
            &str,
            LLMRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = nvidia_nat_nexus_core::Result<
                            Pin<Box<dyn Stream<Item = nvidia_nat_nexus_core::Result<Json>> + Send>>,
                        >,
                    > + Send,
            >,
        > + Send
        + Sync,
> {
    let py_fn = Arc::new(py_fn);
    Arc::new(
        move |_name: &str, request: LLMRequest, next: LlmStreamExecutionNextFn| {
            let py_fn = py_fn.clone();
            Box::pin(async move {
                // Call the Python function. It may return the async iterator directly,
                // or a coroutine that resolves to one.
                let async_iter: Py<PyAny> = match Python::attach(|py| {
                    let py_req = PyLLMRequest { inner: request };
                    let py_next = PyLlmStreamNextFn { inner: next };
                    let result = py_fn
                        .call1(
                            py,
                            (
                                py_req
                                    .into_pyobject(py)
                                    .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?
                                    .into_any(),
                                py_next
                                    .into_pyobject(py)
                                    .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?
                                    .into_any(),
                            ),
                        )
                        .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?;

                    let bound = result.bind(py);
                    if bound.getattr("__await__").is_ok() {
                        let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                            .map_err(|e| NexusError::Internal(e.to_string()))?;
                        Ok::<
                            Result<
                                Py<PyAny>,
                                Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>,
                            >,
                            NexusError,
                        >(Err(Box::pin(future)
                            as Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>))
                    } else {
                        Ok(Ok(result))
                    }
                })? {
                    Ok(iter) => iter,
                    Err(future) => future
                        .await
                        .map_err(|e| NexusError::Internal(e.to_string()))?,
                };

                let (tx, rx) =
                    tokio::sync::mpsc::channel::<nvidia_nat_nexus_core::Result<Json>>(32);

                let task_locals = Python::attach(|py| {
                    pyo3_async_runtimes::tokio::get_current_locals(py)
                        .map_err(|e: pyo3::PyErr| NexusError::Internal(e.to_string()))
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
                                    if e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(
                                        py,
                                    ) {
                                        Ok(None)
                                    } else {
                                        Err(NexusError::Internal(e.to_string()))
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
                                        .map_err(|e| NexusError::Internal(e.to_string()))
                                });
                                let awaited: Result<Json, _> = match future_result {
                                    Ok(future) => match future.await {
                                        Ok(result) => Python::attach(|py| {
                                            py_to_json(result.bind(py))
                                                .map_err(|e| NexusError::Internal(e.to_string()))
                                        }),
                                        Err(e) => Python::attach(|py| {
                                            if e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                                            return Err(NexusError::Internal("__stop__".into()));
                                        }
                                            Err(NexusError::Internal(e.to_string()))
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
                                    Err(NexusError::Internal(ref msg)) if msg == "__stop__" => {
                                        break
                                    }
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
                        Box<dyn Stream<Item = nvidia_nat_nexus_core::Result<Json>> + Send>,
                    >)
            })
        },
    )
}

/// Wrap a Python callable `(LLMRequest) -> LLMRequest` for LLM sanitize request guardrails.
pub fn wrap_py_llm_sanitize_request_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(LLMRequest) -> LLMRequest + Send + Sync> {
    Box::new(move |request: LLMRequest| {
        Python::attach(|py| {
            let py_req = PyLLMRequest {
                inner: request.clone(),
            };
            let result = match py_fn.call1(py, (py_req,)) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("nat_nexus: LLM sanitize request guardrail callable failed: {e}");
                    return request;
                }
            };
            match result.extract::<PyLLMRequest>(py) {
                Ok(r) => r.inner,
                Err(e) => {
                    eprintln!(
                        "nat_nexus: LLM sanitize request guardrail returned unexpected type \
                         (expected LLMRequest): {e}"
                    );
                    request
                }
            }
        })
    })
}

/// Wrap a Python callable `(LLMRequest) -> Optional[str]` for LLM conditional guardrails.
pub fn wrap_py_llm_conditional_fn(py_fn: Py<PyAny>) -> LlmConditionalFn {
    Box::new(move |request: &LLMRequest| {
        Python::attach(|py| {
            let py_req = PyLLMRequest {
                inner: request.clone(),
            };
            let result = py_fn.call1(py, (py_req,)).map_err(|e| {
                NexusError::Internal(format!("LLM conditional guardrail callable failed: {e}"))
            })?;
            let bound = result.bind(py);
            if bound.is_none() {
                Ok(None)
            } else {
                bound.extract::<String>().map(Some).map_err(|e| {
                    NexusError::Internal(format!(
                        "LLM conditional guardrail returned unexpected type (expected str or None): {e}"
                    ))
                })
            }
        })
    })
}

/// Wrap a Python callable for unified LLM request intercepts.
///
/// The Python function receives ``(name: str, request: LLMRequest, annotated: AnnotatedLLMRequest | None)``
/// and must return ``(LLMRequest, AnnotatedLLMRequest | None)``.
pub fn wrap_py_llm_request_intercept_fn(py_fn: Py<PyAny>) -> LlmRequestInterceptFn {
    Box::new(
        move |name: &str,
              request: LLMRequest,
              annotated: Option<AnnotatedLLMRequest>|
              -> nvidia_nat_nexus_core::Result<(LLMRequest, Option<AnnotatedLLMRequest>)> {
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
                                NexusError::Internal(format!(
                                    "Failed to convert AnnotatedLLMRequest to Python: {e}"
                                ))
                            })?
                            .into_any()
                            .unbind()
                    }
                    None => py.None(),
                };
                let result = py_fn.call1(py, (name, py_req, py_ann)).map_err(|e| {
                    NexusError::Internal(format!("LLM request intercept callable failed: {e}"))
                })?;

                // Extract the tuple (LLMRequest, AnnotatedLLMRequest | None)
                let tuple = result.bind(py);
                let new_req: PyLLMRequest = tuple
                    .get_item(0)
                    .map_err(|e| {
                        NexusError::Internal(format!(
                            "LLM request intercept result[0] extraction failed: {e}"
                        ))
                    })?
                    .extract()
                    .map_err(|e| {
                        NexusError::Internal(format!(
                            "LLM request intercept result[0] is not LLMRequest: {e}"
                        ))
                    })?;
                let ann_item = tuple.get_item(1).map_err(|e| {
                    NexusError::Internal(format!(
                        "LLM request intercept result[1] extraction failed: {e}"
                    ))
                })?;
                let new_ann = if ann_item.is_none() {
                    None
                } else {
                    Some(
                        ann_item
                            .extract::<PyAnnotatedLLMRequest>()
                            .map_err(|e| {
                                NexusError::Internal(format!(
                                    "LLM request intercept result[1] is not AnnotatedLLMRequest: {e}"
                                ))
                            })?
                            .inner,
                    )
                };

                Ok((new_req.inner, new_ann))
            })
        },
    )
}

/// Wrap a Python callable `(LLMRequest) -> dict` for LLM execution.
/// Supports both sync and async Python callables.
pub fn wrap_py_llm_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<
    dyn Fn(LLMRequest) -> Pin<Box<dyn Future<Output = nvidia_nat_nexus_core::Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = std::sync::Arc::new(py_fn);
    Box::new(move |request: LLMRequest| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            let outcome: nvidia_nat_nexus_core::Result<
                Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
            > = Python::attach(|py| {
                let py_req = PyLLMRequest { inner: request };
                let result = py_fn
                    .call1(py, (py_req,))
                    .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?;

                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| NexusError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future)
                        as Pin<
                            Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                        >))
                } else {
                    let json = py_to_json(bound)
                        .map_err(|e: PyErr| NexusError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| NexusError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| NexusError::Internal(e.to_string()))
                    })
                }
            }
        })
    })
}

/// Wrap a Python async generator `(LLMRequest) -> AsyncIterator[Any]` for LLM stream execution.
/// Returns a future that resolves to a `Stream<Item = Result<Json>>`.
pub fn wrap_py_llm_stream_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<
    dyn Fn(
            LLMRequest,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = nvidia_nat_nexus_core::Result<
                            Pin<Box<dyn Stream<Item = nvidia_nat_nexus_core::Result<Json>> + Send>>,
                        >,
                    > + Send,
            >,
        > + Send
        + Sync,
> {
    let py_fn = std::sync::Arc::new(py_fn);
    Box::new(move |request: LLMRequest| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            // Call the Python function to get the async iterator object
            let async_iter: Py<PyAny> = Python::attach(|py| {
                let py_req = PyLLMRequest { inner: request };
                py_fn
                    .call1(py, (py_req,))
                    .map_err(|e: PyErr| NexusError::Internal(e.to_string()))
            })?;

            let (tx, rx) = tokio::sync::mpsc::channel::<nvidia_nat_nexus_core::Result<Json>>(32);

            // Capture the Python event loop context so the spawned task can use
            // pyo3_async_runtimes::tokio::into_future (which needs TaskLocals).
            let task_locals = Python::attach(|py| {
                pyo3_async_runtimes::tokio::get_current_locals(py)
                    .map_err(|e: pyo3::PyErr| NexusError::Internal(e.to_string()))
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
                                    Err(NexusError::Internal(e.to_string()))
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
                                    .map_err(|e| NexusError::Internal(e.to_string()))
                            });
                            let awaited: Result<Json, _> = match future_result {
                                Ok(future) => match future.await {
                                    Ok(result) => Python::attach(|py| {
                                        py_to_json(result.bind(py))
                                            .map_err(|e| NexusError::Internal(e.to_string()))
                                    }),
                                    Err(e) => Python::attach(|py| {
                                        if e.is_instance_of::<
                                            pyo3::exceptions::PyStopAsyncIteration,
                                        >(py)
                                        {
                                            return Err(NexusError::Internal("__stop__".into()));
                                        }
                                        Err(NexusError::Internal(e.to_string()))
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
                                Err(NexusError::Internal(ref msg)) if msg == "__stop__" => break,
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
                    Box<dyn Stream<Item = nvidia_nat_nexus_core::Result<Json>> + Send>,
                >)
        })
    })
}

/// Wrap a Python callable `(Any) -> None` as a collector for streaming LLM calls.
///
/// The collector is invoked with each intercepted chunk (after stream response
/// intercepts have been applied). It receives a single JSON-converted Python
/// object argument. If the Python callable raises an exception, it is converted
/// to a `NexusError::Internal` and returned as `Err`, which terminates the
/// stream. If the callable returns normally (including `None`), the collector
/// returns `Ok(())`.
pub fn wrap_py_collector_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn FnMut(Json) -> std::result::Result<(), NexusError> + Send> {
    Box::new(move |chunk: Json| {
        Python::attach(|py| {
            let py_chunk = json_to_py(py, &chunk)
                .map_err(|e| NexusError::Internal(format!("collector json_to_py failed: {e}")))?;
            py_fn
                .call1(py, (py_chunk,))
                .map_err(|e| NexusError::Internal(format!("Python collector error: {e}")))?;
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
                    eprintln!("nat_nexus: Python finalizer callable failed: {e}");
                    return Json::Null;
                }
            };
            py_to_json(result.bind(py)).unwrap_or_else(|e| {
                eprintln!("nat_nexus: py_to_json failed in finalizer: {e}");
                Json::Null
            })
        })
    })
}

/// Wrap a Python callable `(dict) -> dict` for LLM sanitize response guardrails.
pub fn wrap_py_llm_sanitize_response_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    Box::new(move |response: Json| {
        Python::attach(|py| {
            let py_resp = match json_to_py(py, &response) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "nat_nexus: json_to_py failed in LLM sanitize response guardrail: {e}"
                    );
                    return response.clone();
                }
            };
            let result = match py_fn.call1(py, (py_resp,)) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("nat_nexus: LLM sanitize response guardrail callable failed: {e}");
                    return response.clone();
                }
            };
            py_to_json(result.bind(py)).unwrap_or_else(|e| {
                eprintln!("nat_nexus: py_to_json failed in LLM sanitize response guardrail: {e}");
                response.clone()
            })
        })
    })
}

/// Wrap a Python callable `(Event) -> None` for event subscribers.
pub fn wrap_py_event_subscriber(py_fn: Py<PyAny>) -> nvidia_nat_nexus_core::EventSubscriberFn {
    Arc::new(move |event: &nvidia_nat_nexus_core::Event| {
        Python::attach(|py| {
            let result = match event {
                nvidia_nat_nexus_core::Event::ScopeStart(inner) => py_fn.call1(
                    py,
                    (crate::py_types::PyScopeStartEvent {
                        inner: inner.clone(),
                    },),
                ),
                nvidia_nat_nexus_core::Event::ScopeEnd(inner) => py_fn.call1(
                    py,
                    (crate::py_types::PyScopeEndEvent {
                        inner: inner.clone(),
                    },),
                ),
                nvidia_nat_nexus_core::Event::ToolStart(inner) => py_fn.call1(
                    py,
                    (crate::py_types::PyToolStartEvent {
                        inner: inner.clone(),
                    },),
                ),
                nvidia_nat_nexus_core::Event::ToolEnd(inner) => py_fn.call1(
                    py,
                    (crate::py_types::PyToolEndEvent {
                        inner: inner.clone(),
                    },),
                ),
                nvidia_nat_nexus_core::Event::LLMStart(inner) => py_fn.call1(
                    py,
                    (crate::py_types::PyLLMStartEvent {
                        inner: inner.clone(),
                    },),
                ),
                nvidia_nat_nexus_core::Event::LLMEnd(inner) => py_fn.call1(
                    py,
                    (crate::py_types::PyLLMEndEvent {
                        inner: inner.clone(),
                    },),
                ),
                nvidia_nat_nexus_core::Event::Mark(inner) => py_fn.call1(
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
/// - ``decode(request: LLMRequest) -> AnnotatedLLMRequest``
/// - ``encode(annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest``
pub(crate) struct PyLlmCodecWrapper {
    pub py_codec: Py<PyAny>,
}

// SAFETY: The Py<PyAny> handle is GIL-independent (ref-counted via Python's
// allocator). All access goes through `Python::attach` which acquires the GIL.
unsafe impl Send for PyLlmCodecWrapper {}
unsafe impl Sync for PyLlmCodecWrapper {}

impl LlmCodec for PyLlmCodecWrapper {
    fn decode(&self, request: &LLMRequest) -> nvidia_nat_nexus_core::Result<AnnotatedLLMRequest> {
        Python::attach(|py| {
            let py_req = PyLLMRequest {
                inner: request.clone(),
            };
            let result = self
                .py_codec
                .call_method1(py, "decode", (py_req,))
                .map_err(|e| NexusError::Internal(format!("Codec decode() failed: {e}")))?;
            result
                .extract::<PyAnnotatedLLMRequest>(py)
                .map(|r| r.inner)
                .map_err(|e| {
                    NexusError::Internal(format!(
                        "Codec decode() returned unexpected type (expected AnnotatedLLMRequest): {e}"
                    ))
                })
        })
    }

    fn encode(
        &self,
        annotated: &AnnotatedLLMRequest,
        original: &LLMRequest,
    ) -> nvidia_nat_nexus_core::Result<LLMRequest> {
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
                .map_err(|e| NexusError::Internal(format!("Codec encode() failed: {e}")))?;
            result
                .extract::<PyLLMRequest>(py)
                .map(|r| r.inner)
                .map_err(|e| {
                    NexusError::Internal(format!(
                        "Codec encode() returned unexpected type (expected LLMRequest): {e}"
                    ))
                })
        })
    }
}
