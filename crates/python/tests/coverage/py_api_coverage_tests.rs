// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use std::ffi::CString;

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

        let handle = get_handle().unwrap();
        assert_eq!(handle.inner.name, "root");

        let data = py_dict(py, json!({"payload": true}));
        let metadata = py_dict(py, json!({"meta": true}));
        let child = push_scope(
            "child",
            PyScopeType::Tool,
            Some(handle.clone()),
            Some(PyScopeAttributes {
                inner: nemo_flow::types::scope::ScopeAttributes::PARALLEL,
            }),
            Some(&data),
            Some(&metadata),
        )
        .unwrap();
        assert_eq!(child.inner.name, "child");

        event(
            "mark",
            Some(child.clone()),
            Some(&py_dict(py, json!({"step": 1}))),
            Some(&py_dict(py, json!({"source": "cov"}))),
        )
        .unwrap();

        let tool = tool_call(
            "tool",
            &py_dict(py, json!({"arg": 1})),
            Some(child.clone()),
            Some(PyToolAttributes {
                inner: nemo_flow::types::tool::ToolAttributes::LOCAL,
            }),
            Some(&py_dict(py, json!({"tool_data": true}))),
            Some(&py_dict(py, json!({"tool_meta": true}))),
            Some("tool-call".to_string()),
        )
        .unwrap();
        tool_call_end(
            &tool,
            &py_dict(py, json!({"result": 2})),
            Some(&py_dict(py, json!({"done": true}))),
            Some(&py_dict(py, json!({"status": "ok"}))),
        )
        .unwrap();

        let llm_request = PyLLMRequest {
            inner: nemo_flow::types::llm::LLMRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": [], "model": "demo"}),
            },
        };
        let llm = llm_call(
            "llm",
            llm_request,
            Some(child.clone()),
            Some(PyLLMAttributes {
                inner: nemo_flow::types::llm::LLMAttributes::STATELESS
                    | nemo_flow::types::llm::LLMAttributes::STREAMING,
            }),
            Some(&py_dict(py, json!({"llm_data": true}))),
            Some(&py_dict(py, json!({"llm_meta": true}))),
            Some("demo-model".to_string()),
        )
        .unwrap();
        llm_call_end(
            &llm,
            &py_dict(py, json!({"response": "ok"})),
            Some(&py_dict(py, json!({"tokens": 10}))),
            Some(&py_dict(py, json!({"finish_reason": "stop"}))),
        )
        .unwrap();

        pop_scope(&child).unwrap();
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
    events.append((event.kind, event.name))

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

def tool_request_intercept(name, args):
    updated = dict(args)
    updated["value"] = updated["value"] + 2
    return updated

async def tool_exec(args):
    return {"tool_result": args["value"]}

async def tool_exec_intercept(name, args, next):
    result = await next({"value": args["value"] + 3})
    result["tool_intercepted"] = True
    return result

def llm_sanitize_request(request):
    return request

def llm_sanitize_response(response):
    updated = dict(response)
    updated["llm_sanitized_response"] = True
    return updated

def llm_conditional(request):
    return None if request.content.get("model") != "blocked" else "blocked"

def llm_request_intercept(name, request, annotated):
    headers = dict(request.headers)
    headers["x-intercepted"] = "1"
    content = dict(request.content)
    content["messages"] = [{"role": "user", "content": "hello from intercept"}]
    return (LLMRequest(headers, content), annotated)

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
                "AnnotatedLLMRequest",
                types_module.getattr("AnnotatedLLMRequest").unwrap(),
            )
            .unwrap();

        let stack = create_scope_stack();
        set_thread_scope_stack(&stack);
        let root = get_handle().unwrap();
        let child = push_scope(
            "child-exec",
            PyScopeType::Agent,
            Some(root.clone()),
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

        let tool_intercepted =
            tool_request_intercepts(py, "demo-tool", &py_dict(py, json!({"value": 1}))).unwrap();
        assert_eq!(
            crate::convert::py_to_json(tool_intercepted.bind(py)).unwrap(),
            json!({"value": 3})
        );
        tool_conditional_execution("demo-tool", &py_dict(py, json!({"value": 1}))).unwrap();
        assert!(
            tool_conditional_execution("demo-tool", &py_dict(py, json!({"value": -1})))
                .unwrap_err()
                .to_string()
                .contains("blocked")
        );

        let llm_request = PyLLMRequest {
            inner: nemo_flow::types::llm::LLMRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": [{"role": "user", "content": "hello"}], "model": "demo-model"}),
            },
        };
        let intercepted_request = llm_request_intercepts("demo-llm", llm_request.clone()).unwrap();
        assert_eq!(
            intercepted_request.inner.headers.get("x-intercepted"),
            Some(&json!("1"))
        );
        llm_conditional_execution(llm_request.clone()).unwrap();
        assert!(
            llm_conditional_execution(PyLLMRequest {
                inner: nemo_flow::types::llm::LLMRequest {
                    headers: serde_json::Map::new(),
                    content: json!({"messages": [], "model": "blocked"}),
                },
            })
            .unwrap_err()
            .to_string()
            .contains("blocked")
        );

        with_event_loop(py, |event_loop| {
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
                                inner: nemo_flow::types::tool::ToolAttributes::LOCAL,
                            },
                        ))
                        .unwrap(),),
                )
                .unwrap();
            let tool_json = crate::convert::py_to_json(&tool_result).unwrap();
            assert_eq!(tool_json["tool_result"], json!(6));
            assert_eq!(tool_json["tool_intercepted"], json!(true));

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
                                inner: nemo_flow::types::llm::LLMAttributes::STATELESS,
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
                                inner: nemo_flow::types::llm::LLMAttributes::STREAMING,
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

        let events = helpers.getattr("events").unwrap();
        let events_json = crate::convert::py_to_json(events.as_any()).unwrap();
        assert!(
            events_json
                .as_array()
                .unwrap()
                .iter()
                .any(|event| event[0] == "ToolStart")
        );
        assert!(
            events_json
                .as_array()
                .unwrap()
                .iter()
                .any(|event| event[0] == "LLMEnd")
        );

        let chunks = helpers.getattr("chunks").unwrap();
        assert_eq!(
            crate::convert::py_to_json(chunks.as_any()).unwrap(),
            json!([11, 12])
        );

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
            &child_uuid,
            &scope_tool_sanitize_request_name,
            5,
            helpers.getattr("tool_sanitize_request").unwrap().unbind(),
        )
        .unwrap();
        scope_register_tool_sanitize_response_guardrail(
            &child_uuid,
            &scope_tool_sanitize_response_name,
            5,
            helpers.getattr("tool_sanitize_response").unwrap().unbind(),
        )
        .unwrap();
        scope_register_tool_conditional_execution_guardrail(
            &child_uuid,
            &scope_tool_conditional_name,
            5,
            helpers.getattr("tool_conditional").unwrap().unbind(),
        )
        .unwrap();
        scope_register_tool_request_intercept(
            &child_uuid,
            &scope_tool_request_name,
            5,
            false,
            helpers.getattr("tool_request_intercept").unwrap().unbind(),
        )
        .unwrap();
        scope_register_tool_execution_intercept(
            &child_uuid,
            &scope_tool_exec_name,
            5,
            helpers.getattr("tool_exec_intercept").unwrap().unbind(),
        )
        .unwrap();
        scope_register_llm_sanitize_request_guardrail(
            &child_uuid,
            &scope_llm_sanitize_request_name,
            5,
            helpers.getattr("llm_sanitize_request").unwrap().unbind(),
        )
        .unwrap();
        scope_register_llm_sanitize_response_guardrail(
            &child_uuid,
            &scope_llm_sanitize_response_name,
            5,
            helpers.getattr("llm_sanitize_response").unwrap().unbind(),
        )
        .unwrap();
        scope_register_llm_conditional_execution_guardrail(
            &child_uuid,
            &scope_llm_conditional_name,
            5,
            helpers.getattr("llm_conditional").unwrap().unbind(),
        )
        .unwrap();
        scope_register_llm_request_intercept(
            &child_uuid,
            &scope_llm_request_name,
            5,
            false,
            helpers.getattr("llm_request_intercept").unwrap().unbind(),
        )
        .unwrap();
        scope_register_llm_execution_intercept(
            &child_uuid,
            &scope_llm_exec_name,
            5,
            helpers.getattr("llm_exec_intercept").unwrap().unbind(),
        )
        .unwrap();
        scope_register_llm_stream_execution_intercept(
            &child_uuid,
            &scope_llm_stream_name,
            5,
            helpers.getattr("llm_stream_intercept").unwrap().unbind(),
        )
        .unwrap();
        scope_register_subscriber(
            &child_uuid,
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
                &child_uuid,
                &scope_tool_sanitize_request_name
            )
            .unwrap()
        );
        assert!(
            scope_deregister_tool_sanitize_response_guardrail(
                &child_uuid,
                &scope_tool_sanitize_response_name
            )
            .unwrap()
        );
        assert!(
            scope_deregister_tool_conditional_execution_guardrail(
                &child_uuid,
                &scope_tool_conditional_name
            )
            .unwrap()
        );
        assert!(
            scope_deregister_tool_request_intercept(&child_uuid, &scope_tool_request_name).unwrap()
        );
        assert!(
            scope_deregister_tool_execution_intercept(&child_uuid, &scope_tool_exec_name).unwrap()
        );
        assert!(
            scope_deregister_llm_sanitize_request_guardrail(
                &child_uuid,
                &scope_llm_sanitize_request_name
            )
            .unwrap()
        );
        assert!(
            scope_deregister_llm_sanitize_response_guardrail(
                &child_uuid,
                &scope_llm_sanitize_response_name
            )
            .unwrap()
        );
        assert!(
            scope_deregister_llm_conditional_execution_guardrail(
                &child_uuid,
                &scope_llm_conditional_name
            )
            .unwrap()
        );
        assert!(
            scope_deregister_llm_request_intercept(&child_uuid, &scope_llm_request_name).unwrap()
        );
        assert!(
            scope_deregister_llm_execution_intercept(&child_uuid, &scope_llm_exec_name).unwrap()
        );
        assert!(
            scope_deregister_llm_stream_execution_intercept(&child_uuid, &scope_llm_stream_name)
                .unwrap()
        );
        assert!(scope_deregister_subscriber(&child_uuid, &scope_subscriber).unwrap());

        assert!(deregister_tool_sanitize_request_guardrail(&tool_sanitize_request_name).unwrap());
        assert!(!deregister_tool_sanitize_request_guardrail(&tool_sanitize_request_name).unwrap());
        assert!(deregister_tool_sanitize_response_guardrail(&tool_sanitize_response_name).unwrap());
        assert!(
            !deregister_tool_sanitize_response_guardrail(&tool_sanitize_response_name).unwrap()
        );
        assert!(deregister_tool_conditional_execution_guardrail(&tool_conditional_name).unwrap());
        assert!(!deregister_tool_conditional_execution_guardrail(&tool_conditional_name).unwrap());
        assert!(deregister_tool_request_intercept(&tool_request_name).unwrap());
        assert!(!deregister_tool_request_intercept(&tool_request_name).unwrap());
        assert!(deregister_tool_execution_intercept(&tool_exec_name).unwrap());
        assert!(!deregister_tool_execution_intercept(&tool_exec_name).unwrap());

        assert!(deregister_llm_sanitize_request_guardrail(&llm_sanitize_request_name).unwrap());
        assert!(!deregister_llm_sanitize_request_guardrail(&llm_sanitize_request_name).unwrap());
        assert!(deregister_llm_sanitize_response_guardrail(&llm_sanitize_response_name).unwrap());
        assert!(!deregister_llm_sanitize_response_guardrail(&llm_sanitize_response_name).unwrap());
        assert!(deregister_llm_conditional_execution_guardrail(&llm_conditional_name).unwrap());
        assert!(!deregister_llm_conditional_execution_guardrail(&llm_conditional_name).unwrap());
        assert!(deregister_llm_request_intercept(&llm_request_name).unwrap());
        assert!(!deregister_llm_request_intercept(&llm_request_name).unwrap());
        assert!(deregister_llm_execution_intercept(&llm_exec_name).unwrap());
        assert!(!deregister_llm_execution_intercept(&llm_exec_name).unwrap());
        assert!(deregister_llm_stream_execution_intercept(&llm_stream_name).unwrap());
        assert!(!deregister_llm_stream_execution_intercept(&llm_stream_name).unwrap());
        assert!(deregister_subscriber(&global_subscriber).unwrap());
        assert!(!deregister_subscriber(&global_subscriber).unwrap());

        pop_scope(&child).unwrap();
    });
}

#[test]
fn to_py_err_and_forward_stream_to_channel_cover_private_helpers() {
    let _python = crate::test_support::init_python_test();
    let err = to_py_err(nemo_flow::error::FlowError::Internal("boom".into()));
    assert!(err.to_string().contains("boom"));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let stream: RustJsonStream = Box::pin(tokio_stream::iter(vec![
            Ok(json!({"chunk": 1})),
            Ok(json!({"chunk": 2})),
        ]));
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);

        forward_stream_to_channel(stream, tx).await;

        assert_eq!(rx.recv().await.unwrap().unwrap(), json!({"chunk": 1}));
        assert_eq!(rx.recv().await.unwrap().unwrap(), json!({"chunk": 2}));
        assert!(rx.recv().await.is_none());
    });
}
