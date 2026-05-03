// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for coverage in the NeMo Flow Python crate.

use std::ffi::CString;
use std::pin::Pin;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde_json::{Value as Json, json};
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
use nemo_flow::api::event::{BaseEvent, Event, EventCategory, ScopeCategory, ScopeEvent};
use nemo_flow::api::llm::LlmRequest;
use nemo_flow::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn};

fn load_module<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyModule> {
    let code = CString::new(code).unwrap();
    let file_name = CString::new("coverage_tests.py").unwrap();
    let module_name = CString::new("coverage_tests").unwrap();
    PyModule::from_code(py, &code, &file_name, &module_name).unwrap()
}

fn make_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::from_iter([("x-trace".into(), json!("1"))]),
        content: json!({"model": "test-model", "messages": []}),
    }
}

fn with_event_loop<T>(py: Python<'_>, f: impl FnOnce(Bound<'_, PyAny>) -> T) -> T {
    let asyncio = py.import("asyncio").unwrap();
    #[cfg(windows)]
    {
        let policy = asyncio
            .getattr("WindowsSelectorEventLoopPolicy")
            .unwrap()
            .call0()
            .unwrap();
        asyncio
            .call_method1("set_event_loop_policy", (policy,))
            .unwrap();
    }
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
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let module = PyModule::new(py, "_native_test").unwrap();
        crate::py_types::register(&module).unwrap();
        crate::py_api::register(&module).unwrap();
        crate::py_plugin::register(&module).unwrap();
        crate::py_adaptive::register(&module).unwrap();

        assert!(module.getattr("ScopeStack").is_ok());
        assert!(module.getattr("AtifExporter").is_ok());
        assert!(module.getattr("create_scope_stack").is_ok());
        assert!(module.getattr("llm_stream_call_execute").is_ok());
    });
}

#[test]
fn test_native_pymodule_entrypoint_registers_bindings() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let module = PyModule::new(py, "_native_entrypoint").unwrap();
        crate::_native(&module).unwrap();
        assert!(module.getattr("ScopeStack").is_ok());
        assert!(module.getattr("initialize_plugins").is_ok());
        assert!(module.getattr("set_latency_sensitivity").is_ok());
    });
}

#[test]
fn test_python_test_guard_restores_existing_runtime_env() {
    let lock = crate::test_support::lock_python_test();
    unsafe {
        std::env::set_var("NEMO_FLOW_BINDING_KIND", "python");
        std::env::set_var("NEMO_FLOW_RUNTIME_OWNER", "owner");
    }
    {
        let _guard = crate::test_support::init_python_test_locked(lock);
        assert!(std::env::var_os("NEMO_FLOW_BINDING_KIND").is_none());
        assert!(std::env::var_os("NEMO_FLOW_RUNTIME_OWNER").is_none());
    }
    unsafe {
        std::env::remove_var("NEMO_FLOW_BINDING_KIND");
        std::env::remove_var("NEMO_FLOW_RUNTIME_OWNER");
    }
}

#[test]
fn test_python_test_guard_keeps_absent_runtime_env_absent() {
    let lock = crate::test_support::lock_python_test();
    unsafe {
        std::env::remove_var("NEMO_FLOW_BINDING_KIND");
        std::env::remove_var("NEMO_FLOW_RUNTIME_OWNER");
    }
    {
        let _guard = crate::test_support::init_python_test_locked(lock);
        unsafe {
            std::env::set_var("NEMO_FLOW_BINDING_KIND", "mutated-binding");
            std::env::set_var("NEMO_FLOW_RUNTIME_OWNER", "mutated-owner");
        }
        assert_eq!(
            std::env::var_os("NEMO_FLOW_BINDING_KIND"),
            Some("mutated-binding".into())
        );
        assert_eq!(
            std::env::var_os("NEMO_FLOW_RUNTIME_OWNER"),
            Some("mutated-owner".into())
        );
    }
    unsafe {
        std::env::remove_var("NEMO_FLOW_BINDING_KIND");
        std::env::remove_var("NEMO_FLOW_RUNTIME_OWNER");
    }
    let _lock = crate::test_support::lock_python_test();
    assert!(std::env::var_os("NEMO_FLOW_BINDING_KIND").is_none());
    assert!(std::env::var_os("NEMO_FLOW_RUNTIME_OWNER").is_none());
}

#[test]
fn test_convert_helpers_error_on_non_json_python_objects() {
    let _python = crate::test_support::init_python_test();
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
    let _python = crate::test_support::init_python_test();
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
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let module = PyModule::new(py, "_api_test").unwrap();
        crate::py_api::register(&module).unwrap();

        let expected = [
            "create_scope_stack",
            "set_thread_scope_stack",
            "sync_thread_scope_stack",
            "scope_stack_active",
            "get_handle",
            "push_scope",
            "pop_scope",
            "event",
            "tool_call",
            "tool_call_end",
            "tool_call_execute",
            "llm_call",
            "llm_call_end",
            "llm_call_execute",
            "llm_stream_call_execute",
            "register_tool_sanitize_request_guardrail",
            "deregister_tool_sanitize_request_guardrail",
            "register_tool_sanitize_response_guardrail",
            "deregister_tool_sanitize_response_guardrail",
            "register_tool_conditional_execution_guardrail",
            "deregister_tool_conditional_execution_guardrail",
            "register_tool_request_intercept",
            "deregister_tool_request_intercept",
            "register_tool_execution_intercept",
            "deregister_tool_execution_intercept",
            "register_llm_sanitize_request_guardrail",
            "deregister_llm_sanitize_request_guardrail",
            "register_llm_sanitize_response_guardrail",
            "deregister_llm_sanitize_response_guardrail",
            "register_llm_conditional_execution_guardrail",
            "deregister_llm_conditional_execution_guardrail",
            "register_llm_request_intercept",
            "deregister_llm_request_intercept",
            "register_llm_execution_intercept",
            "deregister_llm_execution_intercept",
            "register_llm_stream_execution_intercept",
            "deregister_llm_stream_execution_intercept",
            "register_subscriber",
            "deregister_subscriber",
            "scope_register_tool_sanitize_request_guardrail",
            "scope_deregister_tool_sanitize_request_guardrail",
            "scope_register_tool_sanitize_response_guardrail",
            "scope_deregister_tool_sanitize_response_guardrail",
            "scope_register_tool_conditional_execution_guardrail",
            "scope_deregister_tool_conditional_execution_guardrail",
            "scope_register_tool_request_intercept",
            "scope_deregister_tool_request_intercept",
            "scope_register_tool_execution_intercept",
            "scope_deregister_tool_execution_intercept",
            "scope_register_llm_sanitize_request_guardrail",
            "scope_deregister_llm_sanitize_request_guardrail",
            "scope_register_llm_sanitize_response_guardrail",
            "scope_deregister_llm_sanitize_response_guardrail",
            "scope_register_llm_conditional_execution_guardrail",
            "scope_deregister_llm_conditional_execution_guardrail",
            "scope_register_llm_request_intercept",
            "scope_deregister_llm_request_intercept",
            "scope_register_llm_execution_intercept",
            "scope_deregister_llm_execution_intercept",
            "scope_register_llm_stream_execution_intercept",
            "scope_deregister_llm_stream_execution_intercept",
            "scope_register_subscriber",
            "scope_deregister_subscriber",
            "tool_request_intercepts",
            "tool_conditional_execution",
            "llm_request_intercepts",
            "llm_conditional_execution",
        ];

        for name in expected {
            assert!(module.getattr(name).is_ok(), "missing binding: {name}");
        }
    });
}

#[test]
fn test_py_adaptive_binding_rejects_zero_sensitivity() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let module = PyModule::new(py, "_adaptive_binding").unwrap();
        crate::py_adaptive::register(&module).unwrap();

        let err = module
            .getattr("set_latency_sensitivity")
            .unwrap()
            .call1((0_u32,))
            .unwrap_err();
        assert!(err.to_string().contains("sensitivity must be positive"));

        module
            .getattr("set_latency_sensitivity")
            .unwrap()
            .call1((1_u32,))
            .unwrap();
    });
}

#[test]
fn test_plugin_bindings_validate_configure_and_clear() {
    let _python = crate::test_support::init_python_test();
    let _plugin_test_state = crate::py_plugin::lock_plugin_test_state_for_tests();
    Python::attach(|py| {
        nemo_flow_adaptive::plugin_component::register_adaptive_component().unwrap();

        let plugin_module = PyModule::new(py, "_plugin_test").unwrap();
        crate::py_plugin::register(&plugin_module).unwrap();

        assert!(plugin_module.getattr("PluginContext").is_ok());
        assert!(plugin_module.getattr("validate_plugin_config").is_ok());
        assert!(plugin_module.getattr("initialize_plugins").is_ok());
        assert!(plugin_module.getattr("clear_plugin_configuration").is_ok());
        assert!(plugin_module.getattr("active_plugin_report").is_ok());
        assert!(plugin_module.getattr("list_plugin_kinds").is_ok());
        assert!(plugin_module.getattr("register_plugin").is_ok());
        assert!(plugin_module.getattr("deregister_plugin").is_ok());

        let adaptive_module = PyModule::new(py, "_adaptive_test").unwrap();
        crate::py_adaptive::register(&adaptive_module).unwrap();
        assert!(adaptive_module.getattr("AdaptiveRuntime").is_ok());
        assert!(adaptive_module.getattr("set_latency_sensitivity").is_ok());

        let report_config = crate::convert::json_to_py(
            py,
            &json!({
                "version": 1,
                "components": [{
                    "kind": "adaptive",
                    "enabled": true,
                    "config": {
                        "version": 1,
                        "telemetry": {},
                        "future_field": true
                    }
                }]
            }),
        )
        .unwrap();
        let report = plugin_module
            .getattr("validate_plugin_config")
            .unwrap()
            .call1((report_config.bind(py),))
            .unwrap();
        let report_json = crate::convert::py_to_json(&report).unwrap();
        assert!(
            report_json["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diag| diag["code"] == "adaptive.unknown_field")
        );

        let plugin_helpers = load_module(
            py,
            r#"
def tool_passthrough(name, value):
    return value

def tool_conditional(name, value):
    return None

def llm_sanitize_request(request):
    return request

def llm_sanitize_response(response):
    return response

def llm_conditional(request):
    return None

def llm_request_intercept(name, request, annotated):
    return (request, annotated)

async def llm_execution_intercept(name, request, next):
    return await next(request)

async def llm_stream_execution_intercept(request, next):
    return await next(request)

def tool_request_intercept(name, value):
    return value

async def tool_execution_intercept(name, value, next):
    return await next(value)

class CoveragePlugin:
    def validate(self, plugin_config):
        return [{
            "level": "warning",
            "code": "plugin.coverage_plugin_validate",
            "component": "coverage.python_plugin",
            "message": f"priority:{plugin_config.get('priority', 0)}",
        }]

    def register(self, plugin_config, context):
        context.register_subscriber("coverage_subscriber", lambda event: None)
        context.register_tool_sanitize_request_guardrail("tool_req", 1, tool_passthrough)
        context.register_tool_sanitize_response_guardrail("tool_resp", 1, tool_passthrough)
        context.register_tool_conditional_execution_guardrail("tool_cond", 1, tool_conditional)
        context.register_llm_sanitize_request_guardrail("llm_req", 1, llm_sanitize_request)
        context.register_llm_sanitize_response_guardrail("llm_resp", 1, llm_sanitize_response)
        context.register_llm_conditional_execution_guardrail("llm_cond", 1, llm_conditional)
        context.register_llm_request_intercept("llm_request", 1, False, llm_request_intercept)
        context.register_llm_execution_intercept("llm_exec", 1, llm_execution_intercept)
        context.register_llm_stream_execution_intercept("llm_stream", 1, llm_stream_execution_intercept)
        context.register_tool_request_intercept("tool_request", 1, False, tool_request_intercept)
        context.register_tool_execution_intercept("tool_exec", 1, tool_execution_intercept)
"#,
        );

        plugin_module
            .getattr("register_plugin")
            .unwrap()
            .call1((
                "coverage.python_plugin",
                plugin_helpers
                    .getattr("CoveragePlugin")
                    .unwrap()
                    .call0()
                    .unwrap(),
            ))
            .unwrap();

        let plugin_report_config = crate::convert::json_to_py(
            py,
            &json!({
                "version": 1,
                "components": [{
                    "kind": "coverage.python_plugin",
                    "enabled": true,
                    "config": {"priority": 9}
                }]
            }),
        )
        .unwrap();
        let plugin_report = plugin_module
            .getattr("validate_plugin_config")
            .unwrap()
            .call1((plugin_report_config.bind(py),))
            .unwrap();
        let plugin_report_json = crate::convert::py_to_json(&plugin_report).unwrap();
        assert!(
            plugin_report_json["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diag| diag["code"] == "plugin.coverage_plugin_validate")
        );

        let configured_plugin_config = crate::convert::json_to_py(
            py,
            &json!({
                "version": 1,
                "components": [
                    {
                        "kind": "adaptive",
                        "enabled": true,
                        "config": {
                            "version": 1,
                            "state": {
                                "backend": {
                                    "kind": "in_memory",
                                    "config": {}
                                }
                            },
                            "telemetry": {
                                "learners": ["latency_sensitivity"]
                            },
                            "adaptive_hints": {},
                            "tool_parallelism": {}
                        }
                    },
                    {
                        "kind": "coverage.python_plugin",
                        "enabled": true,
                        "config": {}
                    }
                ]
            }),
        )
        .unwrap();

        let helpers = load_module(
            py,
            r#"
import asyncio

async def initialize_plugins(module, config):
    return await module.initialize_plugins(config)
"#,
        );
        with_event_loop(py, |event_loop| {
            let configured = event_loop
                .call_method1(
                    "run_until_complete",
                    (helpers
                        .getattr("initialize_plugins")
                        .unwrap()
                        .call1((plugin_module.clone(), configured_plugin_config.bind(py)))
                        .unwrap(),),
                )
                .unwrap();
            let configured_json = crate::convert::py_to_json(&configured).unwrap();
            assert!(
                configured_json["diagnostics"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|diag| diag["code"] == "plugin.coverage_plugin_validate")
            );
        });

        let active_report = plugin_module
            .getattr("active_plugin_report")
            .unwrap()
            .call0()
            .unwrap();
        assert!(!active_report.is_none());
        let active_report_json = crate::convert::py_to_json(&active_report).unwrap();
        assert!(active_report_json.is_object());

        let kinds = plugin_module
            .getattr("list_plugin_kinds")
            .unwrap()
            .call0()
            .unwrap();
        let kinds_json = crate::convert::py_to_json(&kinds).unwrap();
        assert!(
            kinds_json
                .as_array()
                .unwrap()
                .iter()
                .any(|kind| kind == "adaptive")
        );

        plugin_module
            .getattr("clear_plugin_configuration")
            .unwrap()
            .call0()
            .unwrap();

        let removed = plugin_module
            .getattr("deregister_plugin")
            .unwrap()
            .call1(("coverage.python_plugin",))
            .unwrap();
        assert!(removed.extract::<bool>().unwrap());
    });
}

#[test]
fn test_sync_wrapper_fallbacks_and_helpers() {
    let _python = crate::test_support::init_python_test();
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
        assert!(
            tool_cond("demo", &json!({"x": 1}))
                .unwrap_err()
                .to_string()
                .contains("expected str or None")
        );

        let request = make_request();
        let llm_sanitize =
            wrap_py_llm_sanitize_request_fn(module.getattr("llm_sanitize_bad").unwrap().unbind());
        assert_eq!(llm_sanitize(request.clone()).content, request.content);

        let llm_cond = wrap_py_llm_conditional_fn(module.getattr("llm_cond_bad").unwrap().unbind());
        assert!(
            llm_cond(&request)
                .unwrap_err()
                .to_string()
                .contains("expected str or None")
        );
        let llm_cond_none =
            wrap_py_llm_conditional_fn(module.getattr("llm_cond_none").unwrap().unbind());
        assert_eq!(llm_cond_none(&request).unwrap(), None);

        let llm_req =
            wrap_py_llm_request_intercept_fn(module.getattr("llm_req_bad").unwrap().unbind());
        assert!(
            llm_req("demo", request.clone(), None)
                .unwrap_err()
                .to_string()
                .contains("intercept callable failed")
        );

        let tool_req =
            wrap_py_tool_request_intercept_fn(module.getattr("tool_fail").unwrap().unbind());
        assert!(
            tool_req("demo", json!({"x": 1}))
                .unwrap_err()
                .to_string()
                .contains("Python tool callable failed")
        );

        let llm_resp =
            wrap_py_llm_sanitize_response_fn(module.getattr("llm_resp_fail").unwrap().unbind());
        assert_eq!(llm_resp(json!({"ok": true})), json!({"ok": true}));

        let mut collector =
            wrap_py_collector_fn(module.getattr("collector_fail").unwrap().unbind());
        assert!(
            collector(json!({"chunk": 1}))
                .unwrap_err()
                .to_string()
                .contains("collector")
        );

        let finalizer = wrap_py_finalizer_fn(module.getattr("finalizer_fail").unwrap().unbind());
        assert_eq!(finalizer(), Json::Null);

        let subscriber = wrap_py_event_subscriber(module.getattr("event_fail").unwrap().unbind());
        let event = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .parent_uuid(Uuid::now_v7())
                .name("evt")
                .build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::tool(),
            None,
        ));
        subscriber(&event);
    });
}

#[test]
fn test_async_exec_and_intercept_wrappers() {
    let _python = crate::test_support::init_python_test();
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
    let _python = crate::test_support::init_python_test();
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
                                Box<dyn Stream<Item = nemo_flow::error::Result<Json>> + Send>,
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
    let _python = crate::test_support::init_python_test();
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
                assert!(
                    tool_exec(json!({"x": 1}))
                        .await
                        .unwrap_err()
                        .to_string()
                        .contains("tool exec boom")
                );

                let tool_intercept = wrap_py_tool_exec_intercept_fn(tool_intercept_fail_py);
                let tool_next: ToolExecutionNextFn =
                    Arc::new(|args| Box::pin(async move { Ok(args) }));
                assert!(
                    tool_intercept("tool", json!({"x": 1}), tool_next)
                        .await
                        .unwrap_err()
                        .to_string()
                        .contains("tool intercept boom")
                );

                let llm_exec = wrap_py_llm_exec_fn(llm_exec_fail_py);
                assert!(
                    llm_exec(make_request())
                        .await
                        .unwrap_err()
                        .to_string()
                        .contains("llm exec boom")
                );

                let llm_intercept = wrap_py_llm_exec_intercept_fn(llm_intercept_fail_py);
                let llm_next: LlmExecutionNextFn = Arc::new(|request| {
                    Box::pin(async move { Ok(json!({"model": request.content["model"]})) })
                });
                assert!(
                    llm_intercept("llm", make_request(), llm_next)
                        .await
                        .unwrap_err()
                        .to_string()
                        .contains("llm intercept boom")
                );

                let stream_exec = wrap_py_llm_stream_exec_fn(llm_stream_fail_py);
                let mut stream = stream_exec(make_request()).await.unwrap();
                assert!(
                    stream
                        .next()
                        .await
                        .unwrap()
                        .unwrap_err()
                        .to_string()
                        .contains("stream fail")
                );

                let stream_intercept =
                    wrap_py_llm_stream_exec_intercept_fn(llm_stream_intercept_sync_py);
                let stream_next: LlmStreamExecutionNextFn = Arc::new(|_request| {
                    Box::pin(async move {
                        let chunks = vec![Ok(json!({"chunk": "downstream"}))];
                        Ok(Box::pin(tokio_stream::iter(chunks))
                            as Pin<
                                Box<dyn Stream<Item = nemo_flow::error::Result<Json>> + Send>,
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
                                Box<dyn Stream<Item = nemo_flow::error::Result<Json>> + Send>,
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
