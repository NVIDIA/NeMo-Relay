// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::CString;
use std::pin::Pin;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde_json::{json, Value as Json};
use tokio_stream::Stream;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::py_callable::{
    wrap_py_collector_fn, wrap_py_event_subscriber, wrap_py_finalizer_fn,
    wrap_py_llm_conditional_fn, wrap_py_llm_exec_fn, wrap_py_llm_exec_intercept_fn,
    wrap_py_llm_request_intercept_fn, wrap_py_llm_sanitize_request_fn,
    wrap_py_llm_sanitize_response_fn, wrap_py_llm_stream_exec_fn,
    wrap_py_llm_stream_exec_intercept_fn, wrap_py_tool_conditional_fn, wrap_py_tool_exec_fn,
    wrap_py_tool_exec_intercept_fn, wrap_py_tool_fn, wrap_py_tool_request_intercept_fn,
};
use nvidia_nat_nexus_core::types::{Event, LLMRequest, ToolAttributes};
use nvidia_nat_nexus_core::{LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn};

fn load_module<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyModule> {
    let code = CString::new(code).unwrap();
    let file_name = CString::new("coverage_tests.py").unwrap();
    let module_name = CString::new("coverage_tests").unwrap();
    PyModule::from_code(py, &code, &file_name, &module_name).unwrap()
}

fn make_request() -> LLMRequest {
    LLMRequest {
        headers: serde_json::Map::from_iter([("x-trace".into(), json!("1"))]),
        content: json!({"model": "test-model", "messages": []}),
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
fn test_native_module_registers_types_and_api_functions() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_native_test").unwrap();
        crate::_native(&module).unwrap();

        assert!(module.getattr("ScopeStack").is_ok());
        assert!(module.getattr("AtifExporter").is_ok());
        assert!(module.getattr("create_scope_stack").is_ok());
        assert!(module.getattr("nat_nexus_llm_stream_call_execute").is_ok());
    });
}

#[test]
fn test_convert_helpers_error_on_non_json_python_objects() {
    Python::initialize();
    Python::attach(|py| {
        let builtins = PyModule::import(py, "builtins").unwrap();
        let object = builtins.getattr("object").unwrap().call0().unwrap();

        let err = crate::convert::py_to_json(&object).unwrap_err();
        assert!(err.to_string().contains("Failed to convert to JSON"));

        let err = crate::convert::opt_py_to_json(Some(&object)).unwrap_err();
        assert!(err.to_string().contains("Failed to convert to JSON"));
    });
}

#[test]
fn test_convert_helpers_roundtrip_optional_and_none_paths() {
    Python::initialize();
    Python::attach(|py| {
        let module = load_module(
            py,
            r#"
payload = {"nested": {"value": 7}, "items": [1, 2, 3]}
"#,
        );
        let payload = module.getattr("payload").unwrap();

        let json_value = crate::convert::py_to_json(&payload).unwrap();
        assert_eq!(
            json_value,
            json!({"nested": {"value": 7}, "items": [1, 2, 3]})
        );

        let py_value = crate::convert::json_to_py(py, &json_value).unwrap();
        let roundtrip = crate::convert::py_to_json(py_value.bind(py)).unwrap();
        assert_eq!(roundtrip, json_value);

        assert_eq!(crate::convert::opt_py_to_json(None).unwrap(), None);
        assert_eq!(
            crate::convert::opt_py_to_json(Some(py.None().bind(py))).unwrap(),
            None
        );

        let none_obj = crate::convert::opt_json_to_py(py, &None).unwrap();
        assert!(none_obj.bind(py).is_none());

        let some_obj =
            crate::convert::opt_json_to_py(py, &Some(json!({"status": "ok", "count": 2}))).unwrap();
        let roundtrip_some = crate::convert::py_to_json(some_obj.bind(py)).unwrap();
        assert_eq!(roundtrip_some, json!({"status": "ok", "count": 2}));
    });
}

#[test]
fn test_py_api_forward_stream_to_channel_exits_when_receiver_is_dropped() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let stream: crate::py_api::RustJsonStream = Box::pin(tokio_stream::iter(vec![
            Ok(json!({"chunk": 1})),
            Ok(json!({"chunk": 2})),
        ]));
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);

        crate::py_api::forward_stream_to_channel(stream, tx).await;
    });
}

#[test]
fn test_register_exposes_all_native_api_functions() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_api_test").unwrap();
        crate::py_api::register(&module).unwrap();

        let expected = [
            "create_scope_stack",
            "set_thread_scope_stack",
            "sync_thread_scope_stack",
            "scope_stack_active",
            "nat_nexus_get_handle",
            "nat_nexus_push_scope",
            "nat_nexus_pop_scope",
            "nat_nexus_event",
            "nat_nexus_tool_call",
            "nat_nexus_tool_call_end",
            "nat_nexus_tool_call_execute",
            "nat_nexus_llm_call",
            "nat_nexus_llm_call_end",
            "nat_nexus_llm_call_execute",
            "nat_nexus_llm_stream_call_execute",
            "nat_nexus_register_tool_sanitize_request_guardrail",
            "nat_nexus_deregister_tool_sanitize_request_guardrail",
            "nat_nexus_register_tool_sanitize_response_guardrail",
            "nat_nexus_deregister_tool_sanitize_response_guardrail",
            "nat_nexus_register_tool_conditional_execution_guardrail",
            "nat_nexus_deregister_tool_conditional_execution_guardrail",
            "nat_nexus_register_tool_request_intercept",
            "nat_nexus_deregister_tool_request_intercept",
            "nat_nexus_register_tool_execution_intercept",
            "nat_nexus_deregister_tool_execution_intercept",
            "nat_nexus_register_llm_sanitize_request_guardrail",
            "nat_nexus_deregister_llm_sanitize_request_guardrail",
            "nat_nexus_register_llm_sanitize_response_guardrail",
            "nat_nexus_deregister_llm_sanitize_response_guardrail",
            "nat_nexus_register_llm_conditional_execution_guardrail",
            "nat_nexus_deregister_llm_conditional_execution_guardrail",
            "nat_nexus_register_llm_request_intercept",
            "nat_nexus_deregister_llm_request_intercept",
            "nat_nexus_register_llm_execution_intercept",
            "nat_nexus_deregister_llm_execution_intercept",
            "nat_nexus_register_llm_stream_execution_intercept",
            "nat_nexus_deregister_llm_stream_execution_intercept",
            "nat_nexus_register_subscriber",
            "nat_nexus_deregister_subscriber",
            "nat_nexus_scope_register_tool_sanitize_request_guardrail",
            "nat_nexus_scope_deregister_tool_sanitize_request_guardrail",
            "nat_nexus_scope_register_tool_sanitize_response_guardrail",
            "nat_nexus_scope_deregister_tool_sanitize_response_guardrail",
            "nat_nexus_scope_register_tool_conditional_execution_guardrail",
            "nat_nexus_scope_deregister_tool_conditional_execution_guardrail",
            "nat_nexus_scope_register_tool_request_intercept",
            "nat_nexus_scope_deregister_tool_request_intercept",
            "nat_nexus_scope_register_tool_execution_intercept",
            "nat_nexus_scope_deregister_tool_execution_intercept",
            "nat_nexus_scope_register_llm_sanitize_request_guardrail",
            "nat_nexus_scope_deregister_llm_sanitize_request_guardrail",
            "nat_nexus_scope_register_llm_sanitize_response_guardrail",
            "nat_nexus_scope_deregister_llm_sanitize_response_guardrail",
            "nat_nexus_scope_register_llm_conditional_execution_guardrail",
            "nat_nexus_scope_deregister_llm_conditional_execution_guardrail",
            "nat_nexus_scope_register_llm_request_intercept",
            "nat_nexus_scope_deregister_llm_request_intercept",
            "nat_nexus_scope_register_llm_execution_intercept",
            "nat_nexus_scope_deregister_llm_execution_intercept",
            "nat_nexus_scope_register_llm_stream_execution_intercept",
            "nat_nexus_scope_deregister_llm_stream_execution_intercept",
            "nat_nexus_scope_register_subscriber",
            "nat_nexus_scope_deregister_subscriber",
            "nat_nexus_tool_request_intercepts",
            "nat_nexus_tool_conditional_execution",
            "nat_nexus_llm_request_intercepts",
            "nat_nexus_llm_conditional_execution",
        ];

        for name in expected {
            assert!(module.getattr(name).is_ok(), "missing binding: {name}");
        }
    });
}

#[test]
fn test_sync_wrapper_fallbacks_and_helpers() {
    Python::initialize();
    Python::attach(|py| {
        let module = load_module(
            py,
            r#"
def tool_ok(name, args):
    return {"seen": args["x"], "name": name}

def tool_fail(name, args):
    raise RuntimeError("tool boom")

def tool_cond_bad(name, args):
    return 123

def llm_sanitize_bad(request):
    return {"bad": True}

def llm_cond_bad(request):
    return 123

def llm_cond_none(request):
    return None

def llm_req_bad(name, request):
    return {"bad": True}

def llm_resp_fail(response):
    raise RuntimeError("resp boom")

def collector_fail(chunk):
    raise RuntimeError("collector boom")

def finalizer_fail():
    raise RuntimeError("finalizer boom")

def event_fail(event):
    raise RuntimeError("subscriber boom")
"#,
        );

        let tool_ok = wrap_py_tool_fn(module.getattr("tool_ok").unwrap().unbind());
        assert_eq!(
            tool_ok("demo", json!({"x": 1})),
            json!({"seen": 1, "name": "demo"})
        );

        let tool_fail = wrap_py_tool_fn(module.getattr("tool_fail").unwrap().unbind());
        assert_eq!(tool_fail("demo", json!({"x": 1})), json!({"x": 1}));

        let tool_cond =
            wrap_py_tool_conditional_fn(module.getattr("tool_cond_bad").unwrap().unbind());
        assert!(tool_cond("demo", &json!({"x": 1}))
            .unwrap_err()
            .to_string()
            .contains("expected str or None"));

        let request = make_request();
        let llm_sanitize =
            wrap_py_llm_sanitize_request_fn(module.getattr("llm_sanitize_bad").unwrap().unbind());
        assert_eq!(llm_sanitize(request.clone()).content, request.content);

        let llm_cond = wrap_py_llm_conditional_fn(module.getattr("llm_cond_bad").unwrap().unbind());
        assert!(llm_cond(&request)
            .unwrap_err()
            .to_string()
            .contains("expected str or None"));
        let llm_cond_none =
            wrap_py_llm_conditional_fn(module.getattr("llm_cond_none").unwrap().unbind());
        assert_eq!(llm_cond_none(&request).unwrap(), None);

        let llm_req =
            wrap_py_llm_request_intercept_fn(module.getattr("llm_req_bad").unwrap().unbind());
        assert!(llm_req("demo", request.clone(), None)
            .unwrap_err()
            .to_string()
            .contains("intercept callable failed"));

        let tool_req =
            wrap_py_tool_request_intercept_fn(module.getattr("tool_fail").unwrap().unbind());
        assert!(tool_req("demo", json!({"x": 1}))
            .unwrap_err()
            .to_string()
            .contains("Python tool callable failed"));

        let llm_resp =
            wrap_py_llm_sanitize_response_fn(module.getattr("llm_resp_fail").unwrap().unbind());
        assert_eq!(llm_resp(json!({"ok": true})), json!({"ok": true}));

        let mut collector =
            wrap_py_collector_fn(module.getattr("collector_fail").unwrap().unbind());
        assert!(collector(json!({"chunk": 1}))
            .unwrap_err()
            .to_string()
            .contains("collector"));

        let finalizer = wrap_py_finalizer_fn(module.getattr("finalizer_fail").unwrap().unbind());
        assert_eq!(finalizer(), Json::Null);

        let subscriber = wrap_py_event_subscriber(module.getattr("event_fail").unwrap().unbind());
        let event = Event::tool_start(
            Some(Uuid::new_v4()),
            Uuid::new_v4(),
            "evt",
            None,
            None,
            ToolAttributes::empty(),
            None,
            None,
        );
        subscriber(&event);
    });
}

#[test]
fn test_async_exec_and_intercept_wrappers() {
    Python::initialize();
    Python::attach(|py| {
        let module = load_module(
            py,
            r#"
async def tool_exec(args):
    return {"tool": args["x"] + 1}

async def tool_intercept(name, args, next):
    result = await next({"x": args["x"] + 1})
    result["wrapped"] = True
    return result

async def llm_exec(request):
    return {"model": request.content["model"]}

async def llm_intercept(name, request, next):
    result = await next(request)
    result["wrapped"] = True
    return result
"#,
        );

        let tool_exec_py: Py<PyAny> = module.getattr("tool_exec").unwrap().unbind();
        let tool_intercept_py: Py<PyAny> = module.getattr("tool_intercept").unwrap().unbind();
        let llm_exec_py: Py<PyAny> = module.getattr("llm_exec").unwrap().unbind();
        let llm_intercept_py: Py<PyAny> = module.getattr("llm_intercept").unwrap().unbind();

        with_event_loop(py, |event_loop| {
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                let tool_exec = wrap_py_tool_exec_fn(tool_exec_py);
                assert_eq!(
                    tool_exec(json!({"x": 2})).await.unwrap(),
                    json!({"tool": 3})
                );

                let tool_intercept = wrap_py_tool_exec_intercept_fn(tool_intercept_py);
                let tool_next: ToolExecutionNextFn =
                    Arc::new(|args| Box::pin(async move { Ok(json!({"next": args["x"]})) }));
                assert_eq!(
                    tool_intercept("tool", json!({"x": 2}), tool_next)
                        .await
                        .unwrap(),
                    json!({"next": 3, "wrapped": true})
                );

                let llm_exec = wrap_py_llm_exec_fn(llm_exec_py);
                assert_eq!(
                    llm_exec(make_request()).await.unwrap(),
                    json!({"model": "test-model"})
                );

                let llm_intercept = wrap_py_llm_exec_intercept_fn(llm_intercept_py);
                let llm_next: LlmExecutionNextFn = Arc::new(|request| {
                    Box::pin(async move { Ok(json!({"model": request.content["model"]})) })
                });
                assert_eq!(
                    llm_intercept("llm", make_request(), llm_next)
                        .await
                        .unwrap(),
                    json!({"model": "test-model", "wrapped": true})
                );
                Ok(())
            })
            .unwrap();
        });
    });
}

#[test]
fn test_stream_wrappers_cover_async_iterator_paths() {
    Python::initialize();
    Python::attach(|py| {
        let module = load_module(
            py,
            r#"
async def llm_stream(request):
    yield {"chunk": 1}
    yield {"chunk": 2}

async def llm_stream_intercept(request, next):
    return await next(request)
"#,
        );

        let stream_exec_py: Py<PyAny> = module.getattr("llm_stream").unwrap().unbind();
        let stream_intercept_py: Py<PyAny> =
            module.getattr("llm_stream_intercept").unwrap().unbind();

        with_event_loop(py, |event_loop| {
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                let stream_exec = wrap_py_llm_stream_exec_fn(stream_exec_py);
                let mut stream = stream_exec(make_request()).await.unwrap();
                let mut seen = Vec::new();
                while let Some(chunk) = stream.next().await {
                    seen.push(chunk.unwrap());
                }
                assert_eq!(seen, vec![json!({"chunk": 1}), json!({"chunk": 2})]);

                let stream_intercept = wrap_py_llm_stream_exec_intercept_fn(stream_intercept_py);
                let stream_next: LlmStreamExecutionNextFn = Arc::new(|_request| {
                    Box::pin(async move {
                        let chunks = vec![Ok(json!({"chunk": "a"})), Ok(json!({"chunk": "b"}))];
                        Ok(Box::pin(tokio_stream::iter(chunks))
                            as Pin<
                                Box<dyn Stream<Item = nvidia_nat_nexus_core::Result<Json>> + Send>,
                            >)
                    })
                });
                let mut stream = stream_intercept("llm", make_request(), stream_next)
                    .await
                    .unwrap();
                let mut seen = Vec::new();
                while let Some(chunk) = stream.next().await {
                    seen.push(chunk.unwrap());
                }
                assert_eq!(seen, vec![json!({"chunk": "a"}), json!({"chunk": "b"})]);
                Ok(())
            })
            .unwrap();
        });
    });
}

#[test]
fn test_async_wrapper_error_paths_and_sync_stream_intercept() {
    Python::initialize();
    Python::attach(|py| {
        let module = load_module(
            py,
            r#"
async def tool_exec_fail(args):
    raise RuntimeError("tool exec boom")

async def tool_intercept_fail(name, args, next):
    raise RuntimeError("tool intercept boom")

async def llm_exec_fail(request):
    raise RuntimeError("llm exec boom")

async def llm_intercept_fail(name, request, next):
    raise RuntimeError("llm intercept boom")

async def llm_stream_fail(request):
    raise RuntimeError("stream fail")
    yield {"never": True}

def llm_stream_intercept_sync(request, next):
    class _Iter:
        def __init__(self):
            self.done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return {"chunk": "sync"}
    return _Iter()

async def llm_stream_intercept_fail(request, next):
    raise RuntimeError("stream intercept boom")
"#,
        );

        let tool_exec_fail_py: Py<PyAny> = module.getattr("tool_exec_fail").unwrap().unbind();
        let tool_intercept_fail_py: Py<PyAny> =
            module.getattr("tool_intercept_fail").unwrap().unbind();
        let llm_exec_fail_py: Py<PyAny> = module.getattr("llm_exec_fail").unwrap().unbind();
        let llm_intercept_fail_py: Py<PyAny> =
            module.getattr("llm_intercept_fail").unwrap().unbind();
        let llm_stream_fail_py: Py<PyAny> = module.getattr("llm_stream_fail").unwrap().unbind();
        let llm_stream_intercept_sync_py: Py<PyAny> = module
            .getattr("llm_stream_intercept_sync")
            .unwrap()
            .unbind();
        let llm_stream_intercept_fail_py: Py<PyAny> = module
            .getattr("llm_stream_intercept_fail")
            .unwrap()
            .unbind();

        with_event_loop(py, |event_loop| {
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                let tool_exec = wrap_py_tool_exec_fn(tool_exec_fail_py);
                assert!(tool_exec(json!({"x": 1}))
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("tool exec boom"));

                let tool_intercept = wrap_py_tool_exec_intercept_fn(tool_intercept_fail_py);
                let tool_next: ToolExecutionNextFn =
                    Arc::new(|args| Box::pin(async move { Ok(args) }));
                assert!(tool_intercept("tool", json!({"x": 1}), tool_next)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("tool intercept boom"));

                let llm_exec = wrap_py_llm_exec_fn(llm_exec_fail_py);
                assert!(llm_exec(make_request())
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("llm exec boom"));

                let llm_intercept = wrap_py_llm_exec_intercept_fn(llm_intercept_fail_py);
                let llm_next: LlmExecutionNextFn = Arc::new(|request| {
                    Box::pin(async move { Ok(json!({"model": request.content["model"]})) })
                });
                assert!(llm_intercept("llm", make_request(), llm_next)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("llm intercept boom"));

                let stream_exec = wrap_py_llm_stream_exec_fn(llm_stream_fail_py);
                let mut stream = stream_exec(make_request()).await.unwrap();
                assert!(stream
                    .next()
                    .await
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("stream fail"));

                let stream_intercept =
                    wrap_py_llm_stream_exec_intercept_fn(llm_stream_intercept_sync_py);
                let stream_next: LlmStreamExecutionNextFn = Arc::new(|_request| {
                    Box::pin(async move {
                        let chunks = vec![Ok(json!({"chunk": "downstream"}))];
                        Ok(Box::pin(tokio_stream::iter(chunks))
                            as Pin<
                                Box<dyn Stream<Item = nvidia_nat_nexus_core::Result<Json>> + Send>,
                            >)
                    })
                });
                let mut stream = stream_intercept("llm", make_request(), stream_next)
                    .await
                    .unwrap();
                assert_eq!(
                    stream.next().await.unwrap().unwrap(),
                    json!({"chunk": "sync"})
                );
                assert!(stream.next().await.is_none());

                let failing_stream_intercept =
                    wrap_py_llm_stream_exec_intercept_fn(llm_stream_intercept_fail_py);
                let stream_next: LlmStreamExecutionNextFn = Arc::new(|_request| {
                    Box::pin(async move {
                        Ok(Box::pin(tokio_stream::iter(vec![Ok(json!({"chunk": 1}))]))
                            as Pin<
                                Box<dyn Stream<Item = nvidia_nat_nexus_core::Result<Json>> + Send>,
                            >)
                    })
                });
                let err = match failing_stream_intercept("llm", make_request(), stream_next).await {
                    Ok(_) => panic!("expected stream intercept failure"),
                    Err(err) => err,
                };
                assert!(err.to_string().contains("stream intercept boom"));

                Ok(())
            })
            .unwrap();
        });
    });
}
