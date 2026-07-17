// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for py callable coverage in the NeMo Relay Python crate.

use super::*;

use std::ffi::CString;
use std::sync::Arc;

use pyo3::types::PyModule;
use serde_json::json;

fn load_module<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyModule> {
    let code = CString::new(code).unwrap();
    let file_name = CString::new("py_callable_coverage_tests.py").unwrap();
    let module_name = CString::new("py_callable_coverage_tests").unwrap();
    PyModule::from_code(py, &code, &file_name, &module_name).unwrap()
}

fn make_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}),
    }
}

fn with_event_loop<T>(py: Python<'_>, f: impl FnOnce(Bound<'_, PyAny>) -> T) -> T {
    let asyncio = py.import("asyncio").unwrap();
    let event_loop = asyncio.call_method0("new_event_loop").unwrap();
    asyncio
        .call_method1("set_event_loop", (&event_loop,))
        .unwrap();
    let result = f(event_loop.clone().into_any());
    asyncio
        .call_method1("set_event_loop", (py.None(),))
        .unwrap();
    event_loop.call_method0("close").unwrap();
    result
}

#[test]
fn sync_wrappers_and_codec_errors_cover_remaining_branches() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let module = load_module(
            py,
            r#"
def sync_tool_exec(args):
    return {"sync_tool": args["x"] + 1}

def sync_tool_intercept(name, args, next):
    return ToolOutcome({"name": name, "value": args["x"] + 2})

def sync_llm_exec(request):
    return {"model": request.content["model"], "mode": "sync"}

def sync_llm_intercept(name, request, next):
    return {"name": name, "model": request.content["model"], "mode": "sync"}

def request_echo(name, request, annotated):
    return Outcome(request, annotated)

def request_bad_annotated(name, request, annotated):
    return (request, {"bad": True})

def request_short_tuple(name, request, annotated):
    return (request,)

def collector_ok(chunk):
    return None

def finalizer_bad_json():
    return object()

def llm_resp_bad_json(response):
    return object()

class BadCodec:
    def decode(self, request):
        return {"bad": True}

    def encode(self, annotated, original):
        return {"bad": True}

class BadResponseCodec:
    def decode_response(self, response):
        return {"bad": True}

class RaisingResponseCodec:
    def decode_response(self, response):
        raise RuntimeError("decode boom")
"#,
        );
        module
            .setattr(
                "Outcome",
                py.get_type::<crate::py_types::PyLLMRequestInterceptOutcome>(),
            )
            .unwrap();
        module
            .setattr(
                "ToolOutcome",
                py.get_type::<crate::py_types::PyToolExecutionInterceptOutcome>(),
            )
            .unwrap();

        let tool_exec_py: Py<PyAny> = module.getattr("sync_tool_exec").unwrap().unbind();
        let tool_intercept_py: Py<PyAny> = module.getattr("sync_tool_intercept").unwrap().unbind();
        let llm_exec_py: Py<PyAny> = module.getattr("sync_llm_exec").unwrap().unbind();
        let llm_intercept_py: Py<PyAny> = module.getattr("sync_llm_intercept").unwrap().unbind();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            let tool_exec = wrap_py_tool_exec_fn(tool_exec_py);
            assert_eq!(
                tool_exec(json!({"x": 2})).await.unwrap(),
                json!({"sync_tool": 3})
            );

            let tool_intercept = wrap_py_tool_exec_intercept_fn(tool_intercept_py);
            let tool_next: ToolExecutionNextFn =
                Arc::new(|args| Box::pin(async move { Ok(json!({"next": args["x"]})) }));
            assert_eq!(
                tool_intercept("tool", json!({"x": 3}), tool_next)
                    .await
                    .unwrap(),
                json!({"name": "tool", "value": 5}).into()
            );

            let llm_exec = wrap_py_llm_exec_fn(llm_exec_py);
            assert_eq!(
                llm_exec(make_request()).await.unwrap(),
                json!({"model": "test-model", "mode": "sync"})
            );

            let llm_intercept = wrap_py_llm_exec_intercept_fn(llm_intercept_py);
            let llm_next: LlmExecutionNextFn = Arc::new(|request| {
                Box::pin(async move { Ok(json!({"model": request.content["model"]})) })
            });
            assert_eq!(
                llm_intercept("llm", make_request(), llm_next)
                    .await
                    .unwrap(),
                json!({"name": "llm", "model": "test-model", "mode": "sync"})
            );
        });

        let request_intercept =
            wrap_py_llm_request_intercept_fn(module.getattr("request_echo").unwrap().unbind());
        let annotated: AnnotatedLLMRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "annotated"}],
            "model": "codec-model"
        }))
        .unwrap();
        let outcome = request_intercept("llm", make_request(), Some(annotated.clone())).unwrap();
        assert_eq!(
            outcome.annotated_request.unwrap().last_user_message(),
            Some("annotated")
        );

        let bad_request_intercept = wrap_py_llm_request_intercept_fn(
            module.getattr("request_bad_annotated").unwrap().unbind(),
        );
        assert!(
            bad_request_intercept("llm", make_request(), Some(annotated))
                .unwrap_err()
                .to_string()
                .contains("must return LLMRequestInterceptOutcome")
        );

        let short_request_intercept = wrap_py_llm_request_intercept_fn(
            module.getattr("request_short_tuple").unwrap().unbind(),
        );
        assert!(
            short_request_intercept("llm", make_request(), None)
                .unwrap_err()
                .to_string()
                .contains("must return LLMRequestInterceptOutcome")
        );

        let mut collector = wrap_py_collector_fn(module.getattr("collector_ok").unwrap().unbind());
        collector(json!({"chunk": 1})).unwrap();

        let finalizer =
            wrap_py_finalizer_fn(module.getattr("finalizer_bad_json").unwrap().unbind());
        assert_eq!(finalizer(), serde_json::Value::Null);

        let llm_response =
            wrap_py_llm_sanitize_response_fn(module.getattr("llm_resp_bad_json").unwrap().unbind());
        assert_eq!(llm_response(json!({"ok": true})), json!({"ok": true}));

        let bad_codec = PyLlmCodecWrapper {
            py_codec: module
                .getattr("BadCodec")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        };
        assert!(
            bad_codec
                .decode(&make_request())
                .unwrap_err()
                .to_string()
                .contains("expected AnnotatedLLMRequest")
        );

        let encode_request: AnnotatedLLMRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hello"}],
            "model": "codec-model"
        }))
        .unwrap();
        assert!(
            bad_codec
                .encode(&encode_request, &make_request())
                .unwrap_err()
                .to_string()
                .contains("expected LlmRequest")
        );

        let bad_response_codec = PyLlmResponseCodecWrapper {
            py_codec: module
                .getattr("BadResponseCodec")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        };
        assert!(
            bad_response_codec
                .decode_response(&json!({"id": "bad"}))
                .unwrap_err()
                .to_string()
                .contains("expected AnnotatedLLMResponse")
        );

        let raising_response_codec = PyLlmResponseCodecWrapper {
            py_codec: module
                .getattr("RaisingResponseCodec")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        };
        assert!(
            raising_response_codec
                .decode_response(&json!({"id": "bad"}))
                .unwrap_err()
                .to_string()
                .contains("decode_response() failed")
        );
    });
}

#[test]
fn async_iter_helpers_cover_stop_error_and_dropped_receiver_paths() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let module = load_module(
            py,
            r#"
import asyncio

class StopIter:
    def __anext__(self):
        raise StopAsyncIteration

class ErrorIter:
    def __anext__(self):
        raise RuntimeError("next boom")

class ValueIter:
    def __init__(self, value):
        self.value = value
        self.done = False

    def __anext__(self):
        async def inner():
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return self.value
        return inner()

class DroppedReceiverIter:
    def __init__(self, value):
        self.value = value
        self.closed = False

    def __anext__(self):
        async def inner():
            return self.value
        return inner()

    async def aclose(self):
        self.closed = True

async def coro_value():
    return {"value": 1}

async def coro_stop():
    raise StopAsyncIteration

async def coro_error():
    raise RuntimeError("await boom")

async def coro_cancel():
    raise asyncio.CancelledError("user cancellation")

async def coro_non_json():
    return object()
"#,
        );

        let stop_iter_cls: Py<PyAny> = module.getattr("StopIter").unwrap().unbind();
        let error_iter_cls: Py<PyAny> = module.getattr("ErrorIter").unwrap().unbind();
        let value_iter_cls: Py<PyAny> = module.getattr("ValueIter").unwrap().unbind();
        let dropped_receiver_iter_cls: Py<PyAny> =
            module.getattr("DroppedReceiverIter").unwrap().unbind();
        let coro_value_fn: Py<PyAny> = module.getattr("coro_value").unwrap().unbind();
        let coro_stop_fn: Py<PyAny> = module.getattr("coro_stop").unwrap().unbind();
        let coro_error_fn: Py<PyAny> = module.getattr("coro_error").unwrap().unbind();
        let coro_cancel_fn: Py<PyAny> = module.getattr("coro_cancel").unwrap().unbind();
        let coro_non_json_fn: Py<PyAny> = module.getattr("coro_non_json").unwrap().unbind();

        assert!(
            next_async_iter_coro(&Arc::new(stop_iter_cls.call0(py).unwrap()))
                .unwrap()
                .is_none()
        );
        assert!(
            next_async_iter_coro(&Arc::new(error_iter_cls.call0(py).unwrap()))
                .unwrap_err()
                .to_string()
                .contains("next boom")
        );

        let value_payload = crate::convert::json_to_py(py, &json!({"x": 1})).unwrap();
        let dropped_payload = crate::convert::json_to_py(py, &json!({"x": 2})).unwrap();
        let no_loop_payload = crate::convert::json_to_py(py, &json!({"x": 3})).unwrap();
        let no_loop_iter = value_iter_cls.call1(py, (no_loop_payload,)).unwrap();

        with_event_loop(py, |event_loop| {
            let _runtime = tokio::runtime::Runtime::new().unwrap();
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                let value =
                    await_async_iter_value(Python::attach(|py| coro_value_fn.call0(py).unwrap()))
                        .await
                        .unwrap();
                assert_eq!(value.unwrap(), json!({"value": 1}));

                assert!(
                    await_async_iter_value(Python::attach(|py| coro_stop_fn.call0(py).unwrap()))
                        .await
                        .unwrap()
                        .is_none()
                );

                assert!(
                    await_async_iter_value(Python::attach(|py| coro_error_fn.call0(py).unwrap()))
                        .await
                        .unwrap_err()
                        .to_string()
                        .contains("await boom")
                );
                assert!(
                    await_async_iter_value(Python::attach(|py| coro_cancel_fn.call0(py).unwrap()))
                        .await
                        .unwrap_err()
                        .to_string()
                        .contains("cancelled")
                );
                assert!(
                    await_async_iter_value(Python::attach(|py| coro_non_json_fn
                        .call0(py)
                        .unwrap()))
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("Failed to convert to JSON")
                );

                let (tx, mut rx) = tokio::sync::mpsc::channel(2);
                let (_cancel, cancel_rx) = tokio::sync::watch::channel(false);
                let (closed, _closed_rx) = tokio::sync::watch::channel(None);
                forward_async_iter(
                    Arc::new(Python::attach(|py| {
                        value_iter_cls.call1(py, (value_payload.bind(py),)).unwrap()
                    })),
                    tx,
                    cancel_rx,
                    closed,
                )
                .await;
                assert_eq!(rx.recv().await.unwrap().unwrap(), json!({"x": 1}));
                assert!(rx.recv().await.is_none());

                let (tx, mut rx) = tokio::sync::mpsc::channel(1);
                let (_cancel, cancel_rx) = tokio::sync::watch::channel(false);
                let (closed, _closed_rx) = tokio::sync::watch::channel(None);
                forward_async_iter(
                    Arc::new(Python::attach(|py| error_iter_cls.call0(py).unwrap())),
                    tx,
                    cancel_rx,
                    closed,
                )
                .await;
                assert!(
                    rx.recv()
                        .await
                        .unwrap()
                        .unwrap_err()
                        .to_string()
                        .contains("next boom")
                );

                let (tx, rx) = tokio::sync::mpsc::channel(1);
                drop(rx);
                let (_cancel, cancel_rx) = tokio::sync::watch::channel(false);
                let (closed, _closed_rx) = tokio::sync::watch::channel(None);
                let dropped_iter = Python::attach(|py| {
                    dropped_receiver_iter_cls
                        .call1(py, (dropped_payload.bind(py),))
                        .unwrap()
                });
                forward_async_iter(
                    Arc::new(Python::attach(|py| dropped_iter.clone_ref(py))),
                    tx,
                    cancel_rx,
                    closed,
                )
                .await;
                assert!(Python::attach(|py| {
                    dropped_iter
                        .bind(py)
                        .getattr("closed")
                        .unwrap()
                        .is_truthy()
                        .unwrap()
                }));
                Ok(())
            })
            .unwrap();
        });

        let no_loop_err = match stream_from_async_iter(no_loop_iter) {
            Ok(_) => panic!("expected missing event loop error"),
            Err(err) => err,
        };
        assert!(no_loop_err.to_string().contains("no running event loop"));
    });
}

#[test]
fn next_wrappers_cover_success_and_error_paths() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let tool_args = crate::convert::json_to_py(py, &json!({"x": 7})).unwrap();
        let helpers = load_module(
            py,
            r#"
async def await_value(awaitable):
    return await awaitable

async def collect_stream(awaitable):
    stream = await awaitable
    items = []
    async for chunk in stream:
        items.append(chunk)
    return items
"#,
        );
        let await_value_fn: Py<PyAny> = helpers.getattr("await_value").unwrap().unbind();
        let collect_stream_fn: Py<PyAny> = helpers.getattr("collect_stream").unwrap().unbind();

        with_event_loop(py, |event_loop| {
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                let tool_next = PyToolNextFn {
                    inner: Arc::new(|args| Box::pin(async move { Ok(json!({"echo": args["x"]})) })),
                };
                let tool_awaitable = Python::attach(|py| {
                    tool_next.__call__(py, tool_args.bind(py)).unwrap().unbind()
                });
                let tool_future = Python::attach(|py| {
                    let helper_call = await_value_fn
                        .call1(py, (tool_awaitable.bind(py),))
                        .unwrap();
                    pyo3_async_runtimes::tokio::into_future(helper_call.into_bound(py)).unwrap()
                });
                let tool_result = tool_future.await.unwrap();
                assert_eq!(
                    Python::attach(|py| crate::convert::py_to_json(tool_result.bind(py)).unwrap()),
                    json!({"echo": 7})
                );

                let tool_next_err = PyToolNextFn {
                    inner: Arc::new(|_| {
                        Box::pin(async { Err(FlowError::Internal("tool next boom".into())) })
                    }),
                };
                let tool_err_awaitable = Python::attach(|py| {
                    tool_next_err
                        .__call__(py, tool_args.bind(py))
                        .unwrap()
                        .unbind()
                });
                let tool_err_future = Python::attach(|py| {
                    let helper_call = await_value_fn
                        .call1(py, (tool_err_awaitable.bind(py),))
                        .unwrap();
                    pyo3_async_runtimes::tokio::into_future(helper_call.into_bound(py)).unwrap()
                });
                assert!(
                    tool_err_future
                        .await
                        .unwrap_err()
                        .to_string()
                        .contains("tool next boom")
                );

                let llm_next = PyLlmNextFn {
                    inner: Arc::new(|request| {
                        Box::pin(async move { Ok(json!({"model": request.content["model"]})) })
                    }),
                };
                let llm_awaitable = Python::attach(|py| {
                    llm_next
                        .__call__(
                            py,
                            PyLLMRequest {
                                inner: make_request(),
                            },
                        )
                        .unwrap()
                        .unbind()
                });
                let llm_future = Python::attach(|py| {
                    let helper_call = await_value_fn.call1(py, (llm_awaitable.bind(py),)).unwrap();
                    pyo3_async_runtimes::tokio::into_future(helper_call.into_bound(py)).unwrap()
                });
                let llm_result = llm_future.await.unwrap();
                assert_eq!(
                    Python::attach(|py| crate::convert::py_to_json(llm_result.bind(py)).unwrap()),
                    json!({"model": "test-model"})
                );

                let llm_next_err = PyLlmNextFn {
                    inner: Arc::new(|_| {
                        Box::pin(async { Err(FlowError::Internal("llm next boom".into())) })
                    }),
                };
                let llm_err_awaitable = Python::attach(|py| {
                    llm_next_err
                        .__call__(
                            py,
                            PyLLMRequest {
                                inner: make_request(),
                            },
                        )
                        .unwrap()
                        .unbind()
                });
                let llm_err_future = Python::attach(|py| {
                    let helper_call = await_value_fn
                        .call1(py, (llm_err_awaitable.bind(py),))
                        .unwrap();
                    pyo3_async_runtimes::tokio::into_future(helper_call.into_bound(py)).unwrap()
                });
                assert!(
                    llm_err_future
                        .await
                        .unwrap_err()
                        .to_string()
                        .contains("llm next boom")
                );

                let stream_next = PyLlmStreamNextFn {
                    inner: Arc::new(|_| {
                        Box::pin(async move {
                            Ok(nemo_relay::api::runtime::LlmJsonStream::new(
                                tokio_stream::iter(vec![Ok(json!({"chunk": 1}))]),
                            ))
                        })
                    }),
                };
                let stream_awaitable = Python::attach(|py| {
                    stream_next
                        .__call__(
                            py,
                            PyLLMRequest {
                                inner: make_request(),
                            },
                        )
                        .unwrap()
                        .unbind()
                });
                let stream_future = Python::attach(|py| {
                    let helper_call = collect_stream_fn
                        .call1(py, (stream_awaitable.bind(py),))
                        .unwrap();
                    pyo3_async_runtimes::tokio::into_future(helper_call.into_bound(py)).unwrap()
                });
                let stream_items = stream_future.await.unwrap();
                assert_eq!(
                    Python::attach(|py| crate::convert::py_to_json(stream_items.bind(py)).unwrap()),
                    json!([{"chunk": 1}])
                );

                let stream_next_err = PyLlmStreamNextFn {
                    inner: Arc::new(|_| {
                        Box::pin(async { Err(FlowError::Internal("stream next boom".into())) })
                    }),
                };
                let stream_err_awaitable = Python::attach(|py| {
                    stream_next_err
                        .__call__(
                            py,
                            PyLLMRequest {
                                inner: make_request(),
                            },
                        )
                        .unwrap()
                        .unbind()
                });
                let stream_err_future = Python::attach(|py| {
                    let helper_call = collect_stream_fn
                        .call1(py, (stream_err_awaitable.bind(py),))
                        .unwrap();
                    pyo3_async_runtimes::tokio::into_future(helper_call.into_bound(py)).unwrap()
                });
                assert!(
                    stream_err_future
                        .await
                        .unwrap_err()
                        .to_string()
                        .contains("stream next boom")
                );
                Ok(())
            })
            .unwrap();
        });
    });
}

#[test]
fn event_sanitize_wrapper_covers_conversion_success_and_fail_open_paths() {
    use nemo_relay::api::event::{BaseEvent, MarkEvent};

    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let module = load_module(
            py,
            r#"
def sanitize(event, fields):
    assert event.kind == "mark"
    fields["data"] = {"safe": event.name}
    fields["metadata"] = None
    return fields

def raises(event, fields):
    raise RuntimeError("sanitize boom")

def invalid(event, fields):
    return "not fields"
"#,
        );
        let event = Event::Mark(MarkEvent::new(
            BaseEvent::builder().name("checkpoint").build(),
            None,
            None,
        ));
        let fields = EventSanitizeFields {
            data: Some(json!({"secret": true})),
            category_profile: None,
            metadata: Some(json!({"secret": true})),
        };

        let sanitized = wrap_py_event_sanitize_fn(module.getattr("sanitize").unwrap().unbind())(
            &event,
            fields.clone(),
        );
        assert_eq!(sanitized.data, Some(json!({"safe": "checkpoint"})));
        assert_eq!(sanitized.metadata, None);

        let raised = wrap_py_event_sanitize_fn(module.getattr("raises").unwrap().unbind())(
            &event,
            fields.clone(),
        );
        assert_eq!(raised, fields);

        let invalid = wrap_py_event_sanitize_fn(module.getattr("invalid").unwrap().unbind())(
            &event,
            fields.clone(),
        );
        assert_eq!(invalid, fields);
    });
}
