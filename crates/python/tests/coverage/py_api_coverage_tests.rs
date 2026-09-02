// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for py api coverage in the NeMo Relay Python crate.

use super::*;

use std::ffi::CString;
use std::sync::mpsc;
use std::time::Duration;

use pyo3::types::PyModule;
use serde_json::json;
use uuid::Uuid;

fn load_module<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyModule> {
    let code = CString::new(code).unwrap();
    let file_name = CString::new("py_api_coverage_tests.py").unwrap();
    let module_name = CString::new("py_api_coverage_tests").unwrap();
    PyModule::from_code(py, &code, &file_name, &module_name).unwrap()
}

fn py_dict<'py>(py: Python<'py>, value: serde_json::Value) -> Bound<'py, pyo3::PyAny> {
    crate::convert::json_to_py(py, &value)
        .unwrap()
        .into_bound(py)
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

fn test_loop(py: Python<'_>, closed: bool) -> Bound<'_, PyAny> {
    let module = load_module(
        py,
        r#"
import threading

class Future:
    def __init__(self):
        self._callbacks = []
        self._cancelled = False

    def add_done_callback(self, callback):
        self._callbacks.append(callback)

    def cancelled(self):
        return self._cancelled

    def cancel(self):
        self._cancelled = True
        for callback in self._callbacks:
            callback(self)

class Loop:
    def __init__(self, closed):
        self.closed = closed
        self.closed_checked = threading.Event()
        self.completion_scheduled = False

    def create_future(self):
        return Future()

    def is_closed(self):
        self.closed_checked.set()
        return self.closed

    def call_soon_threadsafe(self, callback):
        self.completion_scheduled = True
        callback()
"#,
    );
    module.getattr("Loop").unwrap().call1((closed,)).unwrap()
}

struct CancellationSignal(mpsc::Sender<()>);

impl Drop for CancellationSignal {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

#[test]
fn safe_future_into_py_settles_rust_panics() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        with_event_loop(py, |event_loop| {
            let locals = pyo3_async_runtimes::TaskLocals::new(event_loop.clone());
            let future = pyo3_async_runtimes::tokio::get_runtime()
                .block_on(pyo3_async_runtimes::tokio::scope(locals, async move {
                    Python::attach(|py| {
                        safe_future_into_py(py, async move { panic!("expected test panic") })
                            .map(Bound::unbind)
                    })
                }))
                .unwrap();

            let error = event_loop
                .call_method1("run_until_complete", (future,))
                .unwrap_err();
            assert!(error.is_instance_of::<pyo3_async_runtimes::err::RustPanic>(py));
            assert!(error.to_string().contains("expected test panic"));
        });
    });
}

#[test]
fn safe_future_into_py_cancels_rust_work() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let event_loop = test_loop(py, false);
        let locals = pyo3_async_runtimes::TaskLocals::new(event_loop.clone());
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (dropped_tx, dropped_rx) = mpsc::channel();
        let future = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(pyo3_async_runtimes::tokio::scope(locals, async move {
                Python::attach(|py| {
                    safe_future_into_py(py, async move {
                        started_tx.send(()).unwrap();
                        let _cancellation_signal = CancellationSignal(dropped_tx);
                        std::future::pending::<PyResult<Py<PyAny>>>().await
                    })
                    .map(Bound::unbind)
                })
            }))
            .unwrap();

        assert!(started_rx.recv_timeout(Duration::from_secs(1)).is_ok());
        future.bind(py).call_method0("cancel").unwrap();
        assert!(dropped_rx.recv_timeout(Duration::from_secs(1)).is_ok());
    });
}

#[test]
fn safe_future_into_py_skips_completion_on_closed_loop() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let event_loop = test_loop(py, true);
        let locals = pyo3_async_runtimes::TaskLocals::new(event_loop.clone());
        let _future = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(pyo3_async_runtimes::tokio::scope(locals, async move {
                Python::attach(|py| {
                    safe_future_into_py(py, async move { Python::attach(|py| Ok(py.None())) })
                        .map(Bound::unbind)
                })
            }))
            .unwrap();

        assert!(
            event_loop
                .getattr("closed_checked")
                .unwrap()
                .call_method1("wait", (1.0,))
                .unwrap()
                .is_truthy()
                .unwrap()
        );
        assert!(
            !event_loop
                .getattr("completion_scheduled")
                .unwrap()
                .is_truthy()
                .unwrap()
        );
    });
}

#[test]
fn py_api_helpers_and_scope_lifecycle_round_trip() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let module = PyModule::new(py, "_py_api_cov").unwrap();
        register(&module).unwrap();
        assert!(module.getattr("create_scope_stack").is_ok());
        assert!(module.getattr("llm_call_end").is_ok());

        let stack = create_scope_stack();
        set_thread_scope_stack(&stack);
        sync_thread_scope_stack(&stack);
        assert!(py_scope_stack_active());

        let thread_binding = capture_thread_scope_stack();
        assert_eq!(thread_binding.__repr__(), "<_ThreadScopeStackBinding>");
        let replacement = create_scope_stack();
        set_thread_scope_stack(&replacement);
        restore_thread_scope_stack(&thread_binding);
        assert!(py_scope_stack_active());

        let rooted_context = capture_propagation_context().unwrap();
        assert_eq!(rooted_context.inner.version, 1);
        assert!(rooted_context.inner.root_uuid.is_some());
        let rootless_context = capture_rootless_propagation_context().unwrap();
        assert_eq!(
            rootless_context.inner.parent_uuid,
            rooted_context.inner.parent_uuid
        );
        assert!(rootless_context.inner.root_uuid.is_none());
        let propagation_root = Uuid::now_v7();
        let rooted_context =
            capture_propagation_context_with_root(Some(&propagation_root.to_string())).unwrap();
        assert_eq!(rooted_context.inner.root_uuid, Some(propagation_root));
        let propagated_stack = create_scope_stack_from_propagation(&rooted_context).unwrap();
        set_thread_scope_stack(&propagated_stack);
        restore_thread_scope_stack(&thread_binding);
        assert!(capture_propagation_context_with_root(Some("not-a-uuid")).is_err());

        let handle = get_handle().unwrap();
        assert_eq!(handle.inner.name, "root");

        let data = py_dict(py, json!({"payload": true}));
        let metadata = py_dict(py, json!({"meta": true}));
        let child = push_scope(
            py,
            "child",
            PyScopeType::Tool,
            Some(handle.clone()),
            Some(PyScopeAttributes {
                inner: nemo_relay::api::scope::ScopeAttributes::PARALLEL,
            }),
            Some(&data),
            Some(&metadata),
            None,
            None,
        )
        .unwrap();
        assert_eq!(child.inner.name, "child");

        event(
            py,
            "mark",
            Some(child.clone()),
            Some(&py_dict(py, json!({"step": 1}))),
            None,
            Some(&py_dict(py, json!({"source": "cov"}))),
            None,
            None,
        )
        .unwrap();

        let tool = tool_call(
            py,
            "tool",
            &py_dict(py, json!({"arg": 1})),
            Some(child.clone()),
            Some(PyToolAttributes {
                inner: nemo_relay::api::tool::ToolAttributes::REMOTE,
            }),
            Some(&py_dict(py, json!({"tool_data": true}))),
            Some(&py_dict(py, json!({"tool_meta": true}))),
            Some("tool-call".to_string()),
            None,
        )
        .unwrap();
        tool_call_end(
            py,
            &tool,
            crate::py_types::PyToolExecutionResult::from_inner(
                py,
                nemo_relay::api::tool::ToolExecutionResult::new(json!({"result": 2})),
            )
            .unwrap(),
            Some(&py_dict(py, json!({"done": true}))),
            Some(&py_dict(py, json!({"status": "ok"}))),
            None,
        )
        .unwrap();

        let llm_request = PyLLMRequest {
            inner: nemo_relay::api::llm::LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": [], "model": "demo"}),
            },
        };
        let llm = llm_call(
            py,
            "llm",
            llm_request,
            Some(child.clone()),
            Some(PyLLMAttributes {
                inner: nemo_relay::api::llm::LlmAttributes::STATEFUL
                    | nemo_relay::api::llm::LlmAttributes::STREAMING,
            }),
            Some(&py_dict(py, json!({"llm_data": true}))),
            Some(&py_dict(py, json!({"llm_meta": true}))),
            Some("demo-model".to_string()),
            None,
        )
        .unwrap();
        llm_call_end(
            py,
            &llm,
            &py_dict(py, json!({"response": "ok"})),
            Some(&py_dict(py, json!({"tokens": 10}))),
            Some(&py_dict(py, json!({"finish_reason": "stop"}))),
            None,
            None,
            None,
        )
        .unwrap();

        pop_scope(py, &child, None, None, None).unwrap();
        assert_eq!(get_handle().unwrap().inner.name, "root");
    });
}

#[test]
fn py_api_execute_and_registry_paths_cover_global_and_scope_local_features() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let helpers = load_module(
            py,
            r#"
events = []
chunks = []

def subscriber(event):
    events.append((event.kind, getattr(event, "category", None), getattr(event, "scope_category", None), event.name))

def tool_sanitize_request(name, args):
    updated = dict(args)
    updated["value"] = updated["value"] + 1
    updated["tool_sanitized_request"] = True
    return updated

def tool_sanitize_response(name, result):
    updated = dict(result)
    updated["tool_sanitized_response"] = True
    return updated

def tool_conditional(name, args):
    return None if args["value"] >= 0 else "blocked"

async def async_tool_conditional(name, args):
    return None

def tool_request_intercept(name, args):
    updated = dict(args)
    updated["value"] = updated["value"] + 2
    return updated

async def tool_exec(args):
    return ToolExecutionResult(
        {"tool_result": args["value"]},
        {"source": "python-coverage"},
    )

async def tool_exec_intercept(name, args, next):
    downstream = await next({"value": args["value"] + 3})
    result = dict(downstream.result)
    result["tool_intercepted"] = True
    return ToolExecutionInterceptOutcome(result, annotation=downstream.annotation)

def llm_sanitize_request(request, context):
    return request

def llm_sanitize_response(response, context):
    updated = dict(response)
    updated["llm_sanitized_response"] = True
    return updated

def llm_conditional(request):
    return None if request.content.get("model") != "blocked" else "blocked"

def llm_request_intercept(name, request, annotated):
    headers = dict(request.headers)
    headers["x-intercepted"] = "1"
    if annotated is None:
        content = dict(request.content)
        content["messages"] = [{"role": "user", "content": "hello from intercept"}]
    else:
        content = request.content
        annotated.messages = [{"role": "user", "content": "hello from intercept"}]
    return LLMRequestInterceptOutcome(LLMRequest(headers, content), annotated)

async def llm_exec(request):
    return {
        "id": "chatcmpl-test",
        "model": "gpt-4o-mini",
        "choices": [{
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }]
    }

async def llm_exec_intercept(name, request, next):
    response = await next(request)
    response["from_intercept"] = True
    return response

def llm_stream_exec(request):
    async def gen():
        yield {"delta": 1}
        yield {"delta": 2}
    return gen()

async def llm_stream_intercept(request, next):
    stream = await next(request)

    async def gen():
        async for chunk in stream:
            yield {"delta": chunk["delta"] + 10}

    return gen()

def collector(chunk):
    chunks.append(chunk["delta"])

def finalizer():
    return {
        "id": "chatcmpl-stream",
        "model": "gpt-4o-mini",
        "choices": [{
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop"
        }],
        "chunks": list(chunks)
    }

async def await_value(awaitable):
    return await awaitable

async def collect_stream(awaitable):
    stream = await awaitable
    items = []
    async for chunk in stream:
        items.append(chunk)
    return items

class EchoCodec:
    def decode(self, request):
        return AnnotatedLLMRequest(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
            model="codec-model",
            extra={"codec": "decode"}
        )

    def encode(self, annotated, original):
        headers = dict(original.headers)
        headers["x-codec"] = "1"
        content = dict(original.content)
        content["messages"] = [{"role": "user", "content": annotated.last_user_message() or "missing"}]
        content["model"] = annotated.model
        return LLMRequest(headers, content)
"#,
        );
        let types_module = PyModule::new(py, "_py_api_types").unwrap();
        crate::py_types::register(&types_module).unwrap();
        let api_module = PyModule::new(py, "_py_api_registered").unwrap();
        register(&api_module).unwrap();
        let runner = load_module(
            py,
            r#"
async def run_tool(api, func, handle, attributes):
    return await api.tool_call_execute(
        "demo-tool",
        {"value": 1},
        func,
        handle=handle,
        attributes=attributes,
        data={"tool_data": True},
        metadata={"tool_meta": True},
    )

async def run_llm(api, request, func, handle, attributes, codec, response_codec):
    return await api.llm_call_execute(
        "demo-llm",
        request,
        func,
        handle=handle,
        attributes=attributes,
        data={"llm_data": True},
        metadata={"llm_meta": True},
        model_name="demo-model",
        codec=codec,
        response_codec=response_codec,
    )

async def run_standalone(api, request):
    tool_args = await api.tool_request_intercepts("demo-tool", {"value": 1})
    await api.tool_conditional_execution("demo-tool", tool_args)
    conditional_allowed = True
    llm_outcome = await api.llm_request_intercepts("demo-llm", request)
    await api.llm_conditional_execution(llm_outcome.request)
    return {
        "tool_value": tool_args["value"],
        "conditional_allowed": conditional_allowed,
        "llm_header": llm_outcome.request.headers["x-intercepted"],
    }

async def run_stream(api, request, func, collector, finalizer, handle, attributes, codec, response_codec):
    stream = await api.llm_stream_call_execute(
        "demo-stream",
        request,
        func,
        collector,
        finalizer,
        handle=handle,
        attributes=attributes,
        data={"stream_data": True},
        metadata={"stream_meta": True},
        model_name="demo-model",
        codec=codec,
        response_codec=response_codec,
    )
    items = []
    async for chunk in stream:
        items.append(chunk)
    return items
"#,
        );
        helpers
            .setattr("LLMRequest", types_module.getattr("LLMRequest").unwrap())
            .unwrap();
        helpers
            .setattr(
                "LLMRequestInterceptOutcome",
                types_module.getattr("LLMRequestInterceptOutcome").unwrap(),
            )
            .unwrap();
        helpers
            .setattr(
                "ToolExecutionInterceptOutcome",
                types_module
                    .getattr("ToolExecutionInterceptOutcome")
                    .unwrap(),
            )
            .unwrap();
        helpers
            .setattr(
                "ToolExecutionResult",
                types_module.getattr("ToolExecutionResult").unwrap(),
            )
            .unwrap();
        helpers
            .setattr(
                "AnnotatedLLMRequest",
                types_module.getattr("AnnotatedLLMRequest").unwrap(),
            )
            .unwrap();

        let stack = create_scope_stack();
        set_thread_scope_stack(&stack);
        let root = get_handle().unwrap();
        let child = push_scope(
            py,
            "child-exec",
            PyScopeType::Agent,
            Some(root.clone()),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let child_uuid = child.inner.uuid.to_string();

        let global_subscriber = format!("sub-{}", Uuid::now_v7());
        let tool_sanitize_request_name = format!("tsrq-{}", Uuid::now_v7());
        let tool_sanitize_response_name = format!("tsrs-{}", Uuid::now_v7());
        let tool_conditional_name = format!("tcond-{}", Uuid::now_v7());
        let tool_request_name = format!("treq-{}", Uuid::now_v7());
        let tool_exec_name = format!("texec-{}", Uuid::now_v7());
        let llm_sanitize_request_name = format!("lsrq-{}", Uuid::now_v7());
        let llm_sanitize_response_name = format!("lsrs-{}", Uuid::now_v7());
        let llm_conditional_name = format!("lcond-{}", Uuid::now_v7());
        let llm_request_name = format!("lreq-{}", Uuid::now_v7());
        let llm_exec_name = format!("lexec-{}", Uuid::now_v7());
        let llm_stream_name = format!("lstream-{}", Uuid::now_v7());

        register_subscriber(
            &global_subscriber,
            helpers.getattr("subscriber").unwrap().unbind(),
        )
        .unwrap();
        register_tool_sanitize_request_guardrail(
            &tool_sanitize_request_name,
            10,
            helpers.getattr("tool_sanitize_request").unwrap().unbind(),
        )
        .unwrap();
        register_tool_sanitize_response_guardrail(
            &tool_sanitize_response_name,
            10,
            helpers.getattr("tool_sanitize_response").unwrap().unbind(),
        )
        .unwrap();
        register_tool_conditional_execution_guardrail(
            &tool_conditional_name,
            10,
            helpers.getattr("tool_conditional").unwrap().unbind(),
        )
        .unwrap();
        register_tool_request_intercept(
            &tool_request_name,
            10,
            false,
            helpers.getattr("tool_request_intercept").unwrap().unbind(),
        )
        .unwrap();
        register_tool_execution_intercept(
            &tool_exec_name,
            10,
            helpers.getattr("tool_exec_intercept").unwrap().unbind(),
        )
        .unwrap();

        register_llm_sanitize_request_guardrail(
            &llm_sanitize_request_name,
            10,
            helpers.getattr("llm_sanitize_request").unwrap().unbind(),
        )
        .unwrap();
        register_llm_sanitize_response_guardrail(
            &llm_sanitize_response_name,
            10,
            helpers.getattr("llm_sanitize_response").unwrap().unbind(),
        )
        .unwrap();
        register_llm_conditional_execution_guardrail(
            &llm_conditional_name,
            10,
            helpers.getattr("llm_conditional").unwrap().unbind(),
        )
        .unwrap();
        register_llm_request_intercept(
            &llm_request_name,
            10,
            false,
            helpers.getattr("llm_request_intercept").unwrap().unbind(),
        )
        .unwrap();
        register_llm_execution_intercept(
            &llm_exec_name,
            10,
            helpers.getattr("llm_exec_intercept").unwrap().unbind(),
        )
        .unwrap();
        register_llm_stream_execution_intercept(
            &llm_stream_name,
            10,
            helpers.getattr("llm_stream_intercept").unwrap().unbind(),
        )
        .unwrap();

        fn assert_python_api_execution_paths(
            py: Python<'_>,
            helpers: Bound<'_, PyModule>,
            runner: Bound<'_, PyModule>,
            api_module: Bound<'_, PyModule>,
            types_module: Bound<'_, PyModule>,
            child: PyScopeHandle,
        ) {
            let tool_intercepted = tool_request_intercepts(
                py,
                "demo-tool".to_string(),
                &py_dict(py, json!({"value": 1})),
            )
            .unwrap();
            assert_eq!(
                crate::convert::py_to_json(&tool_intercepted).unwrap(),
                json!({"value": 3})
            );
            tool_conditional_execution(
                py,
                "demo-tool".to_string(),
                &py_dict(py, json!({"value": 1})),
            )
            .unwrap();
            assert!(
                tool_conditional_execution(
                    py,
                    "demo-tool".to_string(),
                    &py_dict(py, json!({"value": -1}))
                )
                .unwrap_err()
                .to_string()
                .contains("blocked")
            );
            let async_sync_rejection_name = format!("async-sync-{}", Uuid::now_v7());
            register_tool_conditional_execution_guardrail(
                &async_sync_rejection_name,
                20,
                helpers.getattr("async_tool_conditional").unwrap().unbind(),
            )
            .unwrap();
            assert!(
                tool_conditional_execution(
                    py,
                    "demo-tool".to_string(),
                    &py_dict(py, json!({"value": 1})),
                )
                .unwrap_err()
                .to_string()
                .contains("requires an async caller")
            );
            let llm_request = PyLLMRequest {
                inner: nemo_relay::api::llm::LlmRequest {
                    headers: serde_json::Map::new(),
                    content: json!({"messages": [{"role": "user", "content": "hello"}], "model": "demo-model"}),
                },
            };
            let intercepted_request =
                llm_request_intercepts(py, "demo-llm".to_string(), llm_request.clone()).unwrap();
            let intercepted_request: PyRef<'_, crate::py_types::PyLLMRequestInterceptOutcome> =
                intercepted_request.extract().unwrap();
            assert_eq!(
                intercepted_request
                    .inner
                    .request
                    .headers
                    .get("x-intercepted"),
                Some(&json!("1"))
            );
            llm_conditional_execution(py, llm_request.clone()).unwrap();
            assert!(
                llm_conditional_execution(
                    py,
                    PyLLMRequest {
                        inner: nemo_relay::api::llm::LlmRequest {
                            headers: serde_json::Map::new(),
                            content: json!({"messages": [], "model": "blocked"}),
                        },
                    }
                )
                .unwrap_err()
                .to_string()
                .contains("blocked")
            );

            with_event_loop(py, |event_loop| {
                let standalone = event_loop
                    .call_method1(
                        "run_until_complete",
                        (runner
                            .getattr("run_standalone")
                            .unwrap()
                            .call1((api_module.clone(), llm_request.clone()))
                            .unwrap(),),
                    )
                    .unwrap();
                assert_eq!(
                    crate::convert::py_to_json(&standalone).unwrap(),
                    json!({"tool_value": 3, "conditional_allowed": true, "llm_header": "1"})
                );

                let tool_result = event_loop
                    .call_method1(
                        "run_until_complete",
                        (runner
                            .getattr("run_tool")
                            .unwrap()
                            .call1((
                                api_module.clone(),
                                helpers.getattr("tool_exec").unwrap(),
                                child.clone(),
                                PyToolAttributes {
                                    inner: nemo_relay::api::tool::ToolAttributes::REMOTE,
                                },
                            ))
                            .unwrap(),),
                    )
                    .unwrap();
                let tool_json =
                    crate::convert::py_to_json(&tool_result.getattr("result").unwrap()).unwrap();
                assert_eq!(tool_json["tool_result"], json!(6));
                assert_eq!(tool_json["tool_intercepted"], json!(true));
                assert_eq!(
                    crate::convert::py_to_json(&tool_result.getattr("annotation").unwrap())
                        .unwrap(),
                    json!({"source": "python-coverage"})
                );

                let codec = helpers.getattr("EchoCodec").unwrap().call0().unwrap();
                let response_codec = types_module
                    .getattr("OpenAIChatCodec")
                    .unwrap()
                    .call0()
                    .unwrap();
                let llm_result = event_loop
                    .call_method1(
                        "run_until_complete",
                        (runner
                            .getattr("run_llm")
                            .unwrap()
                            .call1((
                                api_module.clone(),
                                llm_request.clone(),
                                helpers.getattr("llm_exec").unwrap(),
                                child.clone(),
                                PyLLMAttributes {
                                    inner: nemo_relay::api::llm::LlmAttributes::STATEFUL,
                                },
                                codec,
                                response_codec,
                            ))
                            .unwrap(),),
                    )
                    .unwrap();
                let llm_json = crate::convert::py_to_json(&llm_result).unwrap();
                assert_eq!(llm_json["id"], json!("chatcmpl-test"));
                assert_eq!(llm_json["from_intercept"], json!(true));

                let stream_codec = helpers.getattr("EchoCodec").unwrap().call0().unwrap();
                let stream_response_codec = types_module
                    .getattr("OpenAIChatCodec")
                    .unwrap()
                    .call0()
                    .unwrap();
                let stream_items = event_loop
                    .call_method1(
                        "run_until_complete",
                        (runner
                            .getattr("run_stream")
                            .unwrap()
                            .call1((
                                api_module.clone(),
                                llm_request.clone(),
                                helpers.getattr("llm_stream_exec").unwrap(),
                                helpers.getattr("collector").unwrap(),
                                helpers.getattr("finalizer").unwrap(),
                                child.clone(),
                                PyLLMAttributes {
                                    inner: nemo_relay::api::llm::LlmAttributes::STREAMING,
                                },
                                stream_codec,
                                stream_response_codec,
                            ))
                            .unwrap(),),
                    )
                    .unwrap();
                assert_eq!(
                    crate::convert::py_to_json(&stream_items).unwrap(),
                    json!([{"delta": 11}, {"delta": 12}])
                );
            });
            assert!(
                deregister_tool_conditional_execution_guardrail(&async_sync_rejection_name)
                    .unwrap()
            );
        }
        assert_python_api_execution_paths(
            py,
            helpers.clone(),
            runner,
            api_module,
            types_module.clone(),
            child.clone(),
        );

        fn assert_python_api_emitted_events(helpers: &Bound<'_, PyModule>) {
            let events = helpers.getattr("events").unwrap();
            let events_json = crate::convert::py_to_json(events.as_any()).unwrap();
            assert!(
                events_json
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|event| event[0] == "scope" && event[1] == "tool" && event[2] == "start")
            );
            assert!(
                events_json
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|event| event[0] == "scope" && event[1] == "llm" && event[2] == "end")
            );

            let chunks = helpers.getattr("chunks").unwrap();
            assert_eq!(
                crate::convert::py_to_json(chunks.as_any()).unwrap(),
                json!([11, 12])
            );
        }
        assert_python_api_emitted_events(&helpers);

        fn assert_scope_registry_paths(child_uuid: &str, helpers: &Bound<'_, PyModule>) {
            let scope_tool_sanitize_request_name = format!("scope-tsrq-{}", Uuid::now_v7());
            let scope_tool_sanitize_response_name = format!("scope-tsrs-{}", Uuid::now_v7());
            let scope_tool_conditional_name = format!("scope-tcond-{}", Uuid::now_v7());
            let scope_tool_request_name = format!("scope-treq-{}", Uuid::now_v7());
            let scope_tool_exec_name = format!("scope-texec-{}", Uuid::now_v7());
            let scope_llm_sanitize_request_name = format!("scope-lsrq-{}", Uuid::now_v7());
            let scope_llm_sanitize_response_name = format!("scope-lsrs-{}", Uuid::now_v7());
            let scope_llm_conditional_name = format!("scope-lcond-{}", Uuid::now_v7());
            let scope_llm_request_name = format!("scope-lreq-{}", Uuid::now_v7());
            let scope_llm_exec_name = format!("scope-lexec-{}", Uuid::now_v7());
            let scope_llm_stream_name = format!("scope-lstream-{}", Uuid::now_v7());
            let scope_subscriber = format!("scope-sub-{}", Uuid::now_v7());

            scope_register_tool_sanitize_request_guardrail(
                child_uuid,
                &scope_tool_sanitize_request_name,
                5,
                helpers.getattr("tool_sanitize_request").unwrap().unbind(),
            )
            .unwrap();
            scope_register_tool_sanitize_response_guardrail(
                child_uuid,
                &scope_tool_sanitize_response_name,
                5,
                helpers.getattr("tool_sanitize_response").unwrap().unbind(),
            )
            .unwrap();
            scope_register_tool_conditional_execution_guardrail(
                child_uuid,
                &scope_tool_conditional_name,
                5,
                helpers.getattr("tool_conditional").unwrap().unbind(),
            )
            .unwrap();
            scope_register_tool_request_intercept(
                child_uuid,
                &scope_tool_request_name,
                5,
                false,
                helpers.getattr("tool_request_intercept").unwrap().unbind(),
            )
            .unwrap();
            scope_register_tool_execution_intercept(
                child_uuid,
                &scope_tool_exec_name,
                5,
                helpers.getattr("tool_exec_intercept").unwrap().unbind(),
            )
            .unwrap();
            scope_register_llm_sanitize_request_guardrail(
                child_uuid,
                &scope_llm_sanitize_request_name,
                5,
                helpers.getattr("llm_sanitize_request").unwrap().unbind(),
            )
            .unwrap();
            scope_register_llm_sanitize_response_guardrail(
                child_uuid,
                &scope_llm_sanitize_response_name,
                5,
                helpers.getattr("llm_sanitize_response").unwrap().unbind(),
            )
            .unwrap();
            scope_register_llm_conditional_execution_guardrail(
                child_uuid,
                &scope_llm_conditional_name,
                5,
                helpers.getattr("llm_conditional").unwrap().unbind(),
            )
            .unwrap();
            scope_register_llm_request_intercept(
                child_uuid,
                &scope_llm_request_name,
                5,
                false,
                helpers.getattr("llm_request_intercept").unwrap().unbind(),
            )
            .unwrap();
            scope_register_llm_execution_intercept(
                child_uuid,
                &scope_llm_exec_name,
                5,
                helpers.getattr("llm_exec_intercept").unwrap().unbind(),
            )
            .unwrap();
            scope_register_llm_stream_execution_intercept(
                child_uuid,
                &scope_llm_stream_name,
                5,
                helpers.getattr("llm_stream_intercept").unwrap().unbind(),
            )
            .unwrap();
            scope_register_subscriber(
                child_uuid,
                &scope_subscriber,
                helpers.getattr("subscriber").unwrap().unbind(),
            )
            .unwrap();

            assert!(
                scope_register_subscriber(
                    "not-a-uuid",
                    "bad",
                    helpers.getattr("subscriber").unwrap().unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("invalid UUID")
            );

            assert!(
                scope_deregister_tool_sanitize_request_guardrail(
                    child_uuid,
                    &scope_tool_sanitize_request_name
                )
                .unwrap()
            );
            assert!(
                scope_deregister_tool_sanitize_response_guardrail(
                    child_uuid,
                    &scope_tool_sanitize_response_name
                )
                .unwrap()
            );
            assert!(
                scope_deregister_tool_conditional_execution_guardrail(
                    child_uuid,
                    &scope_tool_conditional_name
                )
                .unwrap()
            );
            assert!(
                scope_deregister_tool_request_intercept(child_uuid, &scope_tool_request_name)
                    .unwrap()
            );
            assert!(
                scope_deregister_tool_execution_intercept(child_uuid, &scope_tool_exec_name)
                    .unwrap()
            );
            assert!(
                scope_deregister_llm_sanitize_request_guardrail(
                    child_uuid,
                    &scope_llm_sanitize_request_name
                )
                .unwrap()
            );
            assert!(
                scope_deregister_llm_sanitize_response_guardrail(
                    child_uuid,
                    &scope_llm_sanitize_response_name
                )
                .unwrap()
            );
            assert!(
                scope_deregister_llm_conditional_execution_guardrail(
                    child_uuid,
                    &scope_llm_conditional_name
                )
                .unwrap()
            );
            assert!(
                scope_deregister_llm_request_intercept(child_uuid, &scope_llm_request_name)
                    .unwrap()
            );
            assert!(
                scope_deregister_llm_execution_intercept(child_uuid, &scope_llm_exec_name).unwrap()
            );
            assert!(
                scope_deregister_llm_stream_execution_intercept(child_uuid, &scope_llm_stream_name)
                    .unwrap()
            );
            assert!(scope_deregister_subscriber(child_uuid, &scope_subscriber).unwrap());
        }
        assert_scope_registry_paths(&child_uuid, &helpers);

        fn assert_global_tool_deregistration(
            tool_sanitize_request_name: &str,
            tool_sanitize_response_name: &str,
            tool_conditional_name: &str,
            tool_request_name: &str,
            tool_exec_name: &str,
        ) {
            assert!(
                deregister_tool_sanitize_request_guardrail(tool_sanitize_request_name).unwrap()
            );
            assert!(
                !deregister_tool_sanitize_request_guardrail(tool_sanitize_request_name).unwrap()
            );
            assert!(
                deregister_tool_sanitize_response_guardrail(tool_sanitize_response_name).unwrap()
            );
            assert!(
                !deregister_tool_sanitize_response_guardrail(tool_sanitize_response_name).unwrap()
            );
            assert!(
                deregister_tool_conditional_execution_guardrail(tool_conditional_name).unwrap()
            );
            assert!(
                !deregister_tool_conditional_execution_guardrail(tool_conditional_name).unwrap()
            );
            assert!(deregister_tool_request_intercept(tool_request_name).unwrap());
            assert!(!deregister_tool_request_intercept(tool_request_name).unwrap());
            assert!(deregister_tool_execution_intercept(tool_exec_name).unwrap());
            assert!(!deregister_tool_execution_intercept(tool_exec_name).unwrap());
        }
        assert_global_tool_deregistration(
            &tool_sanitize_request_name,
            &tool_sanitize_response_name,
            &tool_conditional_name,
            &tool_request_name,
            &tool_exec_name,
        );

        fn assert_global_llm_and_subscriber_deregistration(
            llm_sanitize_request_name: &str,
            llm_sanitize_response_name: &str,
            llm_conditional_name: &str,
            llm_request_name: &str,
            llm_exec_name: &str,
            llm_stream_name: &str,
            global_subscriber: &str,
        ) {
            assert!(deregister_llm_sanitize_request_guardrail(llm_sanitize_request_name).unwrap());
            assert!(!deregister_llm_sanitize_request_guardrail(llm_sanitize_request_name).unwrap());
            assert!(
                deregister_llm_sanitize_response_guardrail(llm_sanitize_response_name).unwrap()
            );
            assert!(
                !deregister_llm_sanitize_response_guardrail(llm_sanitize_response_name).unwrap()
            );
            assert!(deregister_llm_conditional_execution_guardrail(llm_conditional_name).unwrap());
            assert!(!deregister_llm_conditional_execution_guardrail(llm_conditional_name).unwrap());
            assert!(deregister_llm_request_intercept(llm_request_name).unwrap());
            assert!(!deregister_llm_request_intercept(llm_request_name).unwrap());
            assert!(deregister_llm_execution_intercept(llm_exec_name).unwrap());
            assert!(!deregister_llm_execution_intercept(llm_exec_name).unwrap());
            assert!(deregister_llm_stream_execution_intercept(llm_stream_name).unwrap());
            assert!(!deregister_llm_stream_execution_intercept(llm_stream_name).unwrap());
            assert!(deregister_subscriber(global_subscriber).unwrap());
            assert!(!deregister_subscriber(global_subscriber).unwrap());
        }
        assert_global_llm_and_subscriber_deregistration(
            &llm_sanitize_request_name,
            &llm_sanitize_response_name,
            &llm_conditional_name,
            &llm_request_name,
            &llm_exec_name,
            &llm_stream_name,
            &global_subscriber,
        );

        pop_scope(py, &child, None, None, None).unwrap();
    });
}

#[test]
fn to_py_err_and_forward_stream_to_channel_cover_private_helpers() {
    let _python = crate::test_support::init_python_test();
    let err = to_py_err(nemo_relay::error::FlowError::Internal("boom".into()));
    assert!(err.to_string().contains("boom"));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let stream = RustJsonStream::new(tokio_stream::iter(vec![
            Ok(json!({"chunk": 1})),
            Ok(json!({"chunk": 2})),
        ]));
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);
        let (_cancel, cancel_rx) = tokio::sync::watch::channel(false);
        let (closed, _closed_rx) = tokio::sync::watch::channel(None);

        forward_stream_to_channel(stream, tx, cancel_rx, closed).await;

        assert_eq!(rx.recv().await.unwrap().unwrap(), json!({"chunk": 1}));
        assert_eq!(rx.recv().await.unwrap().unwrap(), json!({"chunk": 2}));
        assert!(rx.recv().await.is_none());
    });
}

#[test]
fn synchronous_middleware_bridge_avoids_tokio_runtime_reentry() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let result = runtime.block_on(async {
        block_on_sync_middleware(async { Ok::<_, nemo_relay::error::FlowError>(7) })
    });
    assert_eq!(result.unwrap(), 7);
}

#[test]
fn llm_execution_uses_all_response_codec_selection_paths() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let helpers = load_module(
            py,
            r#"
async def llm_exec_responses(request):
    return {
        "id": "resp-1",
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done"}]
        }]
    }

async def llm_exec_anthropic(request):
    return {
        "id": "msg-1",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2}
    }

async def llm_exec_chat(request):
    return {
        "id": "chatcmpl-custom",
        "choices": [{
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }]
    }

def llm_stream_exec(request):
    async def gen():
        yield {"delta": "seen"}
    return gen()

def collector(_chunk):
    return None

def finalizer_responses():
    return {
        "id": "resp-1",
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done"}]
        }]
    }

def finalizer_anthropic():
    return {
        "id": "msg-1",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2}
    }

def finalizer_chat():
    return {
        "id": "chatcmpl-custom",
        "choices": [{
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }]
    }

class CustomResponseCodec:
    def decode_response(self, response):
        codec = OpenAIChatCodec()
        return codec.decode_response(response)
"#,
        );
        let types_module = PyModule::new(py, "_py_api_codec_types").unwrap();
        crate::py_types::register(&types_module).unwrap();
        let api_module = PyModule::new(py, "_py_api_codec_registered").unwrap();
        register(&api_module).unwrap();
        helpers
            .setattr(
                "OpenAIChatCodec",
                types_module.getattr("OpenAIChatCodec").unwrap(),
            )
            .unwrap();
        let runner = load_module(
            py,
            r#"
async def run_llm(api, request, func, response_codec):
    return await api.llm_call_execute("codec-llm", request, func, response_codec=response_codec)

async def run_stream(api, request, func, collector, finalizer, response_codec):
    stream = await api.llm_stream_call_execute(
        "codec-stream",
        request,
        func,
        collector,
        finalizer,
        response_codec=response_codec,
    )
    items = []
    async for chunk in stream:
        items.append(chunk)
    return items
"#,
        );
        let request = PyLLMRequest {
            inner: nemo_relay::api::llm::LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": [{"role": "user", "content": "hello"}], "model": "demo-model"}),
            },
        };

        with_event_loop(py, |event_loop| {
            let responses_result = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("run_llm")
                        .unwrap()
                        .call1((
                            api_module.clone(),
                            request.clone(),
                            helpers.getattr("llm_exec_responses").unwrap(),
                            types_module
                                .getattr("OpenAIResponsesCodec")
                                .unwrap()
                                .call0()
                                .unwrap(),
                        ))
                        .unwrap(),),
                )
                .unwrap();
            assert_eq!(
                crate::convert::py_to_json(&responses_result).unwrap()["id"],
                json!("resp-1")
            );

            let anthropic_result = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("run_llm")
                        .unwrap()
                        .call1((
                            api_module.clone(),
                            request.clone(),
                            helpers.getattr("llm_exec_anthropic").unwrap(),
                            types_module
                                .getattr("AnthropicMessagesCodec")
                                .unwrap()
                                .call0()
                                .unwrap(),
                        ))
                        .unwrap(),),
                )
                .unwrap();
            assert_eq!(
                crate::convert::py_to_json(&anthropic_result).unwrap()["id"],
                json!("msg-1")
            );

            let custom_result = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("run_llm")
                        .unwrap()
                        .call1((
                            api_module.clone(),
                            request.clone(),
                            helpers.getattr("llm_exec_chat").unwrap(),
                            helpers
                                .getattr("CustomResponseCodec")
                                .unwrap()
                                .call0()
                                .unwrap(),
                        ))
                        .unwrap(),),
                )
                .unwrap();
            assert_eq!(
                crate::convert::py_to_json(&custom_result).unwrap()["id"],
                json!("chatcmpl-custom")
            );

            let responses_stream = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("run_stream")
                        .unwrap()
                        .call1((
                            api_module.clone(),
                            request.clone(),
                            helpers.getattr("llm_stream_exec").unwrap(),
                            helpers.getattr("collector").unwrap(),
                            helpers.getattr("finalizer_responses").unwrap(),
                            types_module
                                .getattr("OpenAIResponsesCodec")
                                .unwrap()
                                .call0()
                                .unwrap(),
                        ))
                        .unwrap(),),
                )
                .unwrap();
            assert_eq!(
                crate::convert::py_to_json(&responses_stream).unwrap(),
                json!([{"delta": "seen"}])
            );

            let anthropic_stream = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("run_stream")
                        .unwrap()
                        .call1((
                            api_module.clone(),
                            request.clone(),
                            helpers.getattr("llm_stream_exec").unwrap(),
                            helpers.getattr("collector").unwrap(),
                            helpers.getattr("finalizer_anthropic").unwrap(),
                            types_module
                                .getattr("AnthropicMessagesCodec")
                                .unwrap()
                                .call0()
                                .unwrap(),
                        ))
                        .unwrap(),),
                )
                .unwrap();
            assert_eq!(
                crate::convert::py_to_json(&anthropic_stream).unwrap(),
                json!([{"delta": "seen"}])
            );

            let custom_stream = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("run_stream")
                        .unwrap()
                        .call1((
                            api_module.clone(),
                            request,
                            helpers.getattr("llm_stream_exec").unwrap(),
                            helpers.getattr("collector").unwrap(),
                            helpers.getattr("finalizer_chat").unwrap(),
                            helpers
                                .getattr("CustomResponseCodec")
                                .unwrap()
                                .call0()
                                .unwrap(),
                        ))
                        .unwrap(),),
                )
                .unwrap();
            assert_eq!(
                crate::convert::py_to_json(&custom_stream).unwrap(),
                json!([{"delta": "seen"}])
            );
        });

        assert!(
            parse_uuid("not-a-uuid")
                .unwrap_err()
                .to_string()
                .contains("invalid UUID")
        );
    });
}
