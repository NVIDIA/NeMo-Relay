#![allow(clippy::type_complexity)]

use std::future::Future;
use std::pin::Pin;

use pyo3::prelude::*;
use serde_json::Value as Json;
use tokio_stream::Stream;

use nvagentrt_core::types::{LLMRequest, SseEvent};
use nvagentrt_core::AgentRtError;

use crate::convert::{json_to_py, py_to_json};
use crate::py_types::{PyLLMRequest, PySseEvent};

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
    dyn Fn(Json) -> Pin<Box<dyn Future<Output = nvagentrt_core::Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = std::sync::Arc::new(py_fn);
    Box::new(move |args: Json| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            // Call the Python function and check if it returns a coroutine
            let outcome: nvagentrt_core::Result<
                Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
            > = Python::attach(|py| {
                let py_args = json_to_py(py, &args)
                    .map_err(|e: PyErr| AgentRtError::Internal(e.to_string()))?;
                let result = py_fn
                    .call1(py, (py_args,))
                    .map_err(|e: PyErr| AgentRtError::Internal(e.to_string()))?;

                // Detect coroutine by checking for __await__
                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| AgentRtError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future)
                        as Pin<
                            Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                        >))
                } else {
                    let json = py_to_json(bound)
                        .map_err(|e: PyErr| AgentRtError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| AgentRtError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| AgentRtError::Internal(e.to_string()))
                    })
                }
            }
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

/// Wrap a Python callable `(LLMRequest) -> LLMRequest` for LLM request intercepts.
pub fn wrap_py_llm_request_intercept_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(LLMRequest) -> LLMRequest + Send + Sync> {
    wrap_py_llm_sanitize_request_fn(py_fn)
}

/// Wrap a Python callable `(LLMRequest) -> bool` for LLM execution conditional.
pub fn wrap_py_llm_exec_conditional_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(&LLMRequest) -> bool + Send + Sync> {
    Box::new(move |request: &LLMRequest| {
        Python::attach(|py| {
            let py_req = PyLLMRequest {
                inner: request.clone(),
            };
            let result = py_fn.call1(py, (py_req,)).expect("Python callable failed");
            result.extract::<bool>(py).expect("Expected bool")
        })
    })
}

/// Wrap a Python callable `(LLMRequest) -> Json` for LLM execution intercepts.
/// Supports both sync and async Python callables.
pub fn wrap_py_llm_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<
    dyn Fn(LLMRequest) -> Pin<Box<dyn Future<Output = nvagentrt_core::Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let py_fn = std::sync::Arc::new(py_fn);
    Box::new(move |request: LLMRequest| {
        let py_fn = py_fn.clone();
        Box::pin(async move {
            let outcome: nvagentrt_core::Result<
                Result<Json, Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>>,
            > = Python::attach(|py| {
                let py_req = PyLLMRequest { inner: request };
                let result = py_fn
                    .call1(py, (py_req,))
                    .map_err(|e: PyErr| AgentRtError::Internal(e.to_string()))?;

                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| AgentRtError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future)
                        as Pin<
                            Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>,
                        >))
                } else {
                    let json = py_to_json(bound)
                        .map_err(|e: PyErr| AgentRtError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| AgentRtError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| AgentRtError::Internal(e.to_string()))
                    })
                }
            }
        })
    })
}

/// Wrap a Python async generator `(LLMRequest) -> AsyncIterator[str]` for LLM stream execution.
/// Returns a future that resolves to a `Stream<Item = Result<String>>`.
pub fn wrap_py_llm_stream_exec_fn(
    py_fn: Py<PyAny>,
) -> Box<
    dyn Fn(
            LLMRequest,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = nvagentrt_core::Result<
                            Pin<Box<dyn Stream<Item = nvagentrt_core::Result<String>> + Send>>,
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
                    .map_err(|e: PyErr| AgentRtError::Internal(e.to_string()))
            })?;

            let (tx, rx) = tokio::sync::mpsc::channel::<nvagentrt_core::Result<String>>(32);

            // Capture the Python event loop context so the spawned task can use
            // pyo3_async_runtimes::tokio::into_future (which needs TaskLocals).
            let task_locals = Python::attach(|py| {
                pyo3_async_runtimes::tokio::get_current_locals(py)
                    .map_err(|e: pyo3::PyErr| AgentRtError::Internal(e.to_string()))
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
                                    Err(AgentRtError::Internal(e.to_string()))
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
                                    .map_err(|e| AgentRtError::Internal(e.to_string()))
                            });
                            let awaited: Result<String, _> = match future_result {
                                Ok(future) => match future.await {
                                    Ok(result) => Python::attach(|py| {
                                        result
                                            .extract::<String>(py)
                                            .map_err(|e| AgentRtError::Internal(e.to_string()))
                                    }),
                                    Err(e) => Python::attach(|py| {
                                        if e.is_instance_of::<
                                            pyo3::exceptions::PyStopAsyncIteration,
                                        >(py)
                                        {
                                            return Err(AgentRtError::Internal("__stop__".into()));
                                        }
                                        Err(AgentRtError::Internal(e.to_string()))
                                    }),
                                },
                                Err(e) => Err(e),
                            };

                            match awaited {
                                Ok(text) => {
                                    if tx.send(Ok(text)).await.is_err() {
                                        break; // receiver dropped
                                    }
                                }
                                Err(AgentRtError::Internal(ref msg)) if msg == "__stop__" => break,
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
                    Box<dyn Stream<Item = nvagentrt_core::Result<String>> + Send>,
                >)
        })
    })
}

/// Wrap a Python callable `(SseEvent) -> SseEvent` for LLM stream response intercepts.
pub fn wrap_py_sse_intercept_fn(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(SseEvent) -> SseEvent + Send + Sync> {
    Box::new(move |event: SseEvent| {
        Python::attach(|py| {
            let py_event = PySseEvent { inner: event };
            let result = py_fn
                .call1(py, (py_event,))
                .expect("Python SSE intercept callable failed");
            result
                .extract::<PySseEvent>(py)
                .expect("Expected SseEvent return")
                .inner
        })
    })
}

/// Wrap a Python callable `(Event) -> None` for event subscribers.
pub fn wrap_py_event_subscriber(
    py_fn: Py<PyAny>,
) -> Box<dyn Fn(&nvagentrt_core::Event) + Send + Sync> {
    Box::new(move |event: &nvagentrt_core::Event| {
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
