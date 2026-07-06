// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for py plugin coverage in the NeMo Relay Python crate.

use super::*;

use std::ffi::CString;
use std::sync::{Arc, Mutex};

use nemo_relay::plugin::rollback_registrations;
use pyo3::types::PyModule;
use serde_json::json;

fn load_module<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyModule> {
    let code = CString::new(code).unwrap();
    let file_name = CString::new("py_plugin_coverage_tests.py").unwrap();
    let module_name = CString::new("py_plugin_coverage_tests").unwrap();
    let module = PyModule::from_code(py, &code, &file_name, &module_name).unwrap();
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
    module
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
fn plugin_context_helpers_and_error_conversion_work() {
    let _python = crate::test_support::init_python_test();

    let context = PyPluginContext {
        registrations: Arc::new(Mutex::new(vec![])),
        namespace_prefix: "demo.".to_string(),
    };

    assert_eq!(context.qualify_name("subscriber"), "demo.subscriber");
    assert_eq!(context.__repr__(), "<PluginContext>");
    assert!(context.drain_registrations().unwrap().is_empty());

    let diag = plugin_callback_diag("demo.plugin", "demo.code", "message".to_string());
    assert_eq!(diag.code, "demo.code");
    assert_eq!(diag.component.as_deref(), Some("demo.plugin"));

    let err = to_py_err("boom");
    assert!(err.to_string().contains("boom"));
}

#[test]
fn register_adds_plugin_management_bindings() {
    let _python = crate::test_support::init_python_test();
    let _plugin_test_state = lock_plugin_test_state_for_tests();
    Python::attach(|py| {
        let module = PyModule::new(py, "_plugin_cov").unwrap();
        register(&module).unwrap();

        for name in [
            "PluginContext",
            "_PluginHostActivation",
            "validate_plugin_config",
            "initialize_plugins",
            "activate_dynamic_plugins",
            "clear_plugin_configuration",
            "active_plugin_report",
            "list_plugin_kinds",
            "register_plugin",
            "deregister_plugin",
        ] {
            assert!(module.getattr(name).is_ok(), "missing binding: {name}");
        }

        let listed = list_plugin_kinds_py(py).unwrap();
        let listed_json = crate::convert::py_to_json(listed.bind(py)).unwrap();
        assert!(listed_json.is_array());

        let active = active_plugin_report_py(py).unwrap();
        assert!(active.bind(py).is_none());

        let config = crate::convert::json_to_py(
            py,
            &json!({
                "version": 1,
                "components": []
            }),
        )
        .unwrap()
        .into_bound(py);
        let report = validate_plugin_config_py(py, &config).unwrap();
        let report_json = crate::convert::py_to_json(report.bind(py)).unwrap();
        assert!(report_json.get("diagnostics").unwrap().is_array());

        assert!(
            plugin_error_to_py_err(PluginError::InvalidConfig("bad".into()))
                .is_instance_of::<pyo3::exceptions::PyValueError>(py)
        );
        assert!(
            plugin_error_to_py_err(PluginError::NotFound("missing".into()))
                .is_instance_of::<pyo3::exceptions::PyFileNotFoundError>(py)
        );
        assert!(
            plugin_error_to_py_err(PluginError::Conflict("busy".into()))
                .is_instance_of::<pyo3::exceptions::PyRuntimeError>(py)
        );
        assert!(
            plugin_error_to_py_err(PluginError::RegistrationFailed("teardown".into()))
                .is_instance_of::<pyo3::exceptions::PyRuntimeError>(py)
        );
    });
}

#[test]
fn plugin_context_registers_all_runtime_hooks_and_drains_registrations() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let helpers = load_module(
            py,
            r#"
def subscriber(event):
    return None

def event_sanitize(event, fields):
    return fields

def tool_fn(name, value):
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
    return Outcome(request, annotated)

async def llm_execution_intercept(name, request, next):
    return await next(request)

async def llm_stream_execution_intercept(request, next):
    return await next(request)

def tool_request_intercept(name, value):
    return value

async def tool_execution_intercept(name, value, next):
    return ToolOutcome(await next(value))
"#,
        );

        let context = PyPluginContext {
            registrations: Arc::new(Mutex::new(vec![])),
            namespace_prefix: "demo.".to_string(),
        };

        context
            .register_subscriber(
                "subscriber",
                helpers.getattr("subscriber").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_mark_sanitize_guardrail(
                "mark_sanitize",
                1,
                helpers.getattr("event_sanitize").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_scope_sanitize_start_guardrail(
                "scope_start_sanitize",
                1,
                helpers.getattr("event_sanitize").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_scope_sanitize_end_guardrail(
                "scope_end_sanitize",
                1,
                helpers.getattr("event_sanitize").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_tool_sanitize_request_guardrail(
                "tool_sanitize_request",
                1,
                helpers.getattr("tool_fn").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_tool_sanitize_response_guardrail(
                "tool_sanitize_response",
                1,
                helpers.getattr("tool_fn").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_tool_conditional_execution_guardrail(
                "tool_conditional",
                1,
                helpers.getattr("tool_conditional").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_sanitize_request_guardrail(
                "llm_sanitize_request",
                1,
                helpers.getattr("llm_sanitize_request").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_sanitize_response_guardrail(
                "llm_sanitize_response",
                1,
                helpers.getattr("llm_sanitize_response").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_conditional_execution_guardrail(
                "llm_conditional",
                1,
                helpers.getattr("llm_conditional").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_request_intercept(
                "llm_request",
                1,
                false,
                helpers.getattr("llm_request_intercept").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_execution_intercept(
                "llm_execution",
                1,
                helpers.getattr("llm_execution_intercept").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_stream_execution_intercept(
                "llm_stream_execution",
                1,
                helpers
                    .getattr("llm_stream_execution_intercept")
                    .unwrap()
                    .unbind(),
            )
            .unwrap();
        context
            .register_tool_request_intercept(
                "tool_request",
                1,
                false,
                helpers.getattr("tool_request_intercept").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_tool_execution_intercept(
                "tool_execution",
                1,
                helpers
                    .getattr("tool_execution_intercept")
                    .unwrap()
                    .unbind(),
            )
            .unwrap();

        let registrations = context.drain_registrations().unwrap();
        assert_eq!(registrations.len(), 15);
        assert!(
            registrations
                .iter()
                .all(|registration| registration.name.starts_with("demo."))
        );

        assert!(deregister_subscriber("demo.subscriber").unwrap());
        assert!(deregister_mark_sanitize_guardrail("demo.mark_sanitize").unwrap());
        assert!(deregister_scope_sanitize_start_guardrail("demo.scope_start_sanitize").unwrap());
        assert!(deregister_scope_sanitize_end_guardrail("demo.scope_end_sanitize").unwrap());
        assert!(deregister_tool_sanitize_request_guardrail("demo.tool_sanitize_request").unwrap());
        assert!(
            deregister_tool_sanitize_response_guardrail("demo.tool_sanitize_response").unwrap()
        );
        assert!(deregister_tool_conditional_execution_guardrail("demo.tool_conditional").unwrap());
        assert!(deregister_llm_sanitize_request_guardrail("demo.llm_sanitize_request").unwrap());
        assert!(deregister_llm_sanitize_response_guardrail("demo.llm_sanitize_response").unwrap());
        assert!(deregister_llm_conditional_execution_guardrail("demo.llm_conditional").unwrap());
        assert!(deregister_llm_request_intercept("demo.llm_request").unwrap());
        assert!(deregister_llm_execution_intercept("demo.llm_execution").unwrap());
        assert!(deregister_llm_stream_execution_intercept("demo.llm_stream_execution").unwrap());
        assert!(deregister_tool_request_intercept("demo.tool_request").unwrap());
        assert!(deregister_tool_execution_intercept("demo.tool_execution").unwrap());
    });
}

#[test]
fn python_plugin_validation_and_initialization_cover_error_paths() {
    let _python = crate::test_support::init_python_test();
    let _plugin_test_state = lock_plugin_test_state_for_tests();
    Python::attach(|py| {
        let helpers = load_module(
            py,
            r#"
class MissingValidatePlugin:
    pass

class NonJsonValidatePlugin:
    def validate(self, plugin_config):
        return object()

class InvalidDiagnosticsPlugin:
    def validate(self, plugin_config):
        return [{"level": "warning", "code": 1, "message": "bad"}]

class RaisingValidatePlugin:
    def validate(self, plugin_config):
        raise RuntimeError("validate boom")

class NoneValidatePlugin:
    def validate(self, plugin_config):
        return None

class GoodPlugin:
    def validate(self, plugin_config):
        return [{
            "level": "warning",
            "code": "plugin.good_warning",
            "component": "demo.good",
            "message": "warn"
        }]

    def register(self, plugin_config, context):
        context.register_subscriber("sub", lambda event: None)

class FailingRegisterPlugin:
    def validate(self, plugin_config):
        return []

    def register(self, plugin_config, context):
        raise RuntimeError("register boom")

async def initialize_plugins(module, config):
    return await module.initialize_plugins(config)
"#,
        );

        let module = PyModule::new(py, "_plugin_cov_errors").unwrap();
        register(&module).unwrap();

        register_plugin_py(
            "demo.missing_validate",
            helpers
                .getattr("MissingValidatePlugin")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        )
        .unwrap();
        register_plugin_py(
            "demo.non_json_validate",
            helpers
                .getattr("NonJsonValidatePlugin")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        )
        .unwrap();
        register_plugin_py(
            "demo.invalid_diag_validate",
            helpers
                .getattr("InvalidDiagnosticsPlugin")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        )
        .unwrap();
        register_plugin_py(
            "demo.raising_validate",
            helpers
                .getattr("RaisingValidatePlugin")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        )
        .unwrap();
        register_plugin_py(
            "demo.none_validate",
            helpers
                .getattr("NoneValidatePlugin")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        )
        .unwrap();
        register_plugin_py(
            "demo.good",
            helpers
                .getattr("GoodPlugin")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        )
        .unwrap();
        register_plugin_py(
            "demo.failing_register",
            helpers
                .getattr("FailingRegisterPlugin")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        )
        .unwrap();

        let missing_validate = validate_plugin_config_py(
            py,
            crate::convert::json_to_py(
                py,
                &json!({
                    "version": 1,
                    "components": [{"kind": "demo.missing_validate", "enabled": true, "config": {}}]
                }),
            )
            .unwrap()
            .bind(py),
        )
        .unwrap();
        assert_eq!(
            crate::convert::py_to_json(missing_validate.bind(py)).unwrap()["diagnostics"],
            json!([])
        );

        let non_json_validate = validate_plugin_config_py(
            py,
            crate::convert::json_to_py(
                py,
                &json!({
                    "version": 1,
                    "components": [{"kind": "demo.non_json_validate", "enabled": true, "config": {}}]
                }),
            )
            .unwrap()
            .bind(py),
        )
        .unwrap();
        assert!(
            crate::convert::py_to_json(non_json_validate.bind(py)).unwrap()["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diag| diag["code"] == "plugin.validate_failed")
        );

        let invalid_diag_validate = validate_plugin_config_py(
            py,
            crate::convert::json_to_py(
                py,
                &json!({
                    "version": 1,
                    "components": [{"kind": "demo.invalid_diag_validate", "enabled": true, "config": {}}]
                }),
            )
            .unwrap()
            .bind(py),
        )
        .unwrap();
        assert!(
            crate::convert::py_to_json(invalid_diag_validate.bind(py)).unwrap()["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diag| diag["code"] == "plugin.validate_failed")
        );

        let raising_validate = validate_plugin_config_py(
            py,
            crate::convert::json_to_py(
                py,
                &json!({
                    "version": 1,
                    "components": [{"kind": "demo.raising_validate", "enabled": true, "config": {}}]
                }),
            )
            .unwrap()
            .bind(py),
        )
        .unwrap();
        assert!(
            crate::convert::py_to_json(raising_validate.bind(py)).unwrap()["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diag| diag["code"] == "plugin.validate_failed")
        );

        let none_validate = validate_plugin_config_py(
            py,
            crate::convert::json_to_py(
                py,
                &json!({
                    "version": 1,
                    "components": [{"kind": "demo.none_validate", "enabled": true, "config": {}}]
                }),
            )
            .unwrap()
            .bind(py),
        )
        .unwrap();
        assert_eq!(
            crate::convert::py_to_json(none_validate.bind(py)).unwrap()["diagnostics"],
            json!([])
        );

        let active_before = active_plugin_report_py(py).unwrap();
        assert!(active_before.bind(py).is_none());

        with_event_loop(py, |event_loop| {
            let good_config = crate::convert::json_to_py(
                py,
                &json!({
                    "version": 1,
                    "components": [{"kind": "demo.good", "enabled": true, "config": {}}]
                }),
            )
            .unwrap();
            let good_report = event_loop
                .call_method1(
                    "run_until_complete",
                    (helpers
                        .getattr("initialize_plugins")
                        .unwrap()
                        .call1((module.clone(), good_config.bind(py)))
                        .unwrap(),),
                )
                .unwrap();
            let good_json = crate::convert::py_to_json(&good_report).unwrap();
            assert!(
                good_json["diagnostics"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|diag| diag["code"] == "plugin.good_warning")
            );

            let active = active_plugin_report_py(py).unwrap();
            assert!(!active.bind(py).is_none());
            clear_plugin_configuration_py().unwrap();
            assert!(active_plugin_report_py(py).unwrap().bind(py).is_none());

            let failing_config = crate::convert::json_to_py(
                py,
                &json!({
                    "version": 1,
                    "components": [{"kind": "demo.failing_register", "enabled": true, "config": {}}]
                }),
            )
            .unwrap();
            let err = event_loop
                .call_method1(
                    "run_until_complete",
                    (helpers
                        .getattr("initialize_plugins")
                        .unwrap()
                        .call1((module.clone(), failing_config.bind(py)))
                        .unwrap(),),
                )
                .unwrap_err();
            assert!(err.to_string().contains("register boom"));
        });

        assert!(deregister_plugin_py("demo.missing_validate"));
        assert!(deregister_plugin_py("demo.non_json_validate"));
        assert!(deregister_plugin_py("demo.invalid_diag_validate"));
        assert!(deregister_plugin_py("demo.raising_validate"));
        assert!(deregister_plugin_py("demo.none_validate"));
        assert!(deregister_plugin_py("demo.good"));
        assert!(deregister_plugin_py("demo.failing_register"));
        assert!(!deregister_plugin_py("demo.good"));
    });
}

#[test]
fn plugin_context_rollback_from_non_runtime_owner_covers_deregistration_error_mappers() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let helpers = load_module(
            py,
            r#"
def subscriber(event):
    return None

def tool_fn(name, value):
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
    return Outcome(request, annotated)

async def llm_execution_intercept(name, request, next):
    return await next(request)

async def llm_stream_execution_intercept(request, next):
    return await next(request)

def tool_request_intercept(name, value):
    return value

async def tool_execution_intercept(name, value, next):
    return ToolOutcome(await next(value))
"#,
        );

        let context = PyPluginContext {
            registrations: Arc::new(Mutex::new(vec![])),
            namespace_prefix: "rollback.".to_string(),
        };

        context
            .register_subscriber(
                "subscriber",
                helpers.getattr("subscriber").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_tool_sanitize_request_guardrail(
                "tool_req",
                1,
                helpers.getattr("tool_fn").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_tool_sanitize_response_guardrail(
                "tool_resp",
                1,
                helpers.getattr("tool_fn").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_tool_conditional_execution_guardrail(
                "tool_cond",
                1,
                helpers.getattr("tool_conditional").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_sanitize_request_guardrail(
                "llm_req",
                1,
                helpers.getattr("llm_sanitize_request").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_sanitize_response_guardrail(
                "llm_resp",
                1,
                helpers.getattr("llm_sanitize_response").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_conditional_execution_guardrail(
                "llm_cond",
                1,
                helpers.getattr("llm_conditional").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_request_intercept(
                "llm_request",
                1,
                false,
                helpers.getattr("llm_request_intercept").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_execution_intercept(
                "llm_exec",
                1,
                helpers.getattr("llm_execution_intercept").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_llm_stream_execution_intercept(
                "llm_stream",
                1,
                helpers
                    .getattr("llm_stream_execution_intercept")
                    .unwrap()
                    .unbind(),
            )
            .unwrap();
        context
            .register_tool_request_intercept(
                "tool_request",
                1,
                false,
                helpers.getattr("tool_request_intercept").unwrap().unbind(),
            )
            .unwrap();
        context
            .register_tool_execution_intercept(
                "tool_exec",
                1,
                helpers
                    .getattr("tool_execution_intercept")
                    .unwrap()
                    .unbind(),
            )
            .unwrap();

        let mut registrations = context.drain_registrations().unwrap();
        assert_eq!(registrations.len(), 12);

        let previous_owner = std::env::var("NEMO_RELAY_RUNTIME_OWNER").ok();
        let conflicting_owner = format!(
            "pid={};binding=node;version={}",
            std::process::id(),
            env!("CARGO_PKG_VERSION").split('.').next().unwrap()
        );
        unsafe {
            std::env::set_var("NEMO_RELAY_RUNTIME_OWNER", &conflicting_owner);
        }
        rollback_registrations(&mut registrations);
        match previous_owner {
            Some(value) => unsafe { std::env::set_var("NEMO_RELAY_RUNTIME_OWNER", value) },
            None => unsafe { std::env::remove_var("NEMO_RELAY_RUNTIME_OWNER") },
        }
    });
}

#[test]
fn forced_plugin_conversion_and_context_allocation_failures_are_covered() {
    let _python = crate::test_support::init_python_test();
    let _plugin_test_state = lock_plugin_test_state_for_tests();
    Python::attach(|py| {
        let helpers = load_module(
            py,
            r#"
class GoodPlugin:
    def validate(self, plugin_config):
        return []

    def register(self, plugin_config, context):
        context.register_subscriber("sub", lambda event: None)

async def initialize_plugins(module, config):
    return await module.initialize_plugins(config)
"#,
        );

        let module = PyModule::new(py, "_plugin_cov_forced").unwrap();
        register(&module).unwrap();

        register_plugin_py(
            "demo.forced",
            helpers
                .getattr("GoodPlugin")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        )
        .unwrap();

        let config = crate::convert::json_to_py(
            py,
            &json!({
                "version": 1,
                "components": [{"kind": "demo.forced", "enabled": true, "config": {}}]
            }),
        )
        .unwrap();

        let _validate_error_guard = force_validate_config_to_py_error_for_tests("demo.forced");
        let report = validate_plugin_config_py(py, config.bind(py)).unwrap();
        assert!(
            crate::convert::py_to_json(report.bind(py)).unwrap()["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diag| diag["message"]
                    .as_str()
                    .unwrap()
                    .contains("forced plugin config conversion failure"))
        );
        drop(_validate_error_guard);

        with_event_loop(py, |event_loop| {
            let _context_new_error_guard = force_plugin_context_new_error_for_tests("demo.forced");
            let err = event_loop
                .call_method1(
                    "run_until_complete",
                    (helpers
                        .getattr("initialize_plugins")
                        .unwrap()
                        .call1((module.clone(), config.bind(py)))
                        .unwrap(),),
                )
                .unwrap_err();
            assert!(
                err.to_string()
                    .contains("forced plugin context allocation failure"),
                "{err}"
            );
        });

        assert!(deregister_plugin_py("demo.forced"));
    });
}

#[test]
fn invoke_python_plugin_register_rolls_back_partial_registrations_on_error() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let helpers = load_module(
            py,
            r#"
def subscriber(event):
    return None

class FailingPlugin:
    def register(self, plugin_config, context):
        context.register_subscriber("sub", subscriber)
        raise RuntimeError("boom")
"#,
        );

        let plugin = helpers.getattr("FailingPlugin").unwrap().call0().unwrap();
        let register_fn = plugin.getattr("register").unwrap();
        let namespace_prefix = "rollback.".to_string();

        for _ in 0..2 {
            let err = invoke_python_plugin_register(
                py,
                "demo.rollback",
                &register_fn,
                &serde_json::Map::new(),
                namespace_prefix.clone(),
            )
            .unwrap_err();
            assert!(err.to_string().contains("boom"), "{err}");

            let context = PyPluginContext {
                registrations: Arc::new(Mutex::new(vec![])),
                namespace_prefix: namespace_prefix.clone(),
            };
            context
                .register_subscriber("sub", helpers.getattr("subscriber").unwrap().unbind())
                .unwrap();
            let mut registrations = context.drain_registrations().unwrap();
            rollback_registrations(&mut registrations);
        }
    });
}

#[test]
fn plugin_context_lock_poisoning_covers_error_paths() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let helpers = load_module(
            py,
            r#"
def subscriber(event):
    return None

def tool_fn(name, value):
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
    return Outcome(request, annotated)

async def llm_execution_intercept(name, request, next):
    return await next(request)

async def llm_stream_execution_intercept(request, next):
    return await next(request)

def tool_request_intercept(name, value):
    return value

async def tool_execution_intercept(name, value, next):
    return ToolOutcome(await next(value))
"#,
        );

        let registrations = Arc::new(Mutex::new(vec![]));
        let poisoned = registrations.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poisoned.lock().unwrap();
            panic!("poison plugin registrations");
        })
        .join();

        let context = PyPluginContext {
            registrations,
            namespace_prefix: "poison.".to_string(),
        };

        assert!(
            context
                .drain_registrations()
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );

        assert!(
            context
                .register_subscriber(
                    "subscriber",
                    helpers.getattr("subscriber").unwrap().unbind()
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_subscriber("poison.subscriber").unwrap());

        assert!(
            context
                .register_tool_sanitize_request_guardrail(
                    "tool_req",
                    1,
                    helpers.getattr("tool_fn").unwrap().unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_tool_sanitize_request_guardrail("poison.tool_req").unwrap());

        assert!(
            context
                .register_tool_sanitize_response_guardrail(
                    "tool_resp",
                    1,
                    helpers.getattr("tool_fn").unwrap().unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_tool_sanitize_response_guardrail("poison.tool_resp").unwrap());

        assert!(
            context
                .register_tool_conditional_execution_guardrail(
                    "tool_cond",
                    1,
                    helpers.getattr("tool_conditional").unwrap().unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_tool_conditional_execution_guardrail("poison.tool_cond").unwrap());

        assert!(
            context
                .register_llm_sanitize_request_guardrail(
                    "llm_req",
                    1,
                    helpers.getattr("llm_sanitize_request").unwrap().unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_llm_sanitize_request_guardrail("poison.llm_req").unwrap());

        assert!(
            context
                .register_llm_sanitize_response_guardrail(
                    "llm_resp",
                    1,
                    helpers.getattr("llm_sanitize_response").unwrap().unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_llm_sanitize_response_guardrail("poison.llm_resp").unwrap());

        assert!(
            context
                .register_llm_conditional_execution_guardrail(
                    "llm_cond",
                    1,
                    helpers.getattr("llm_conditional").unwrap().unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_llm_conditional_execution_guardrail("poison.llm_cond").unwrap());

        assert!(
            context
                .register_llm_request_intercept(
                    "llm_request",
                    1,
                    false,
                    helpers.getattr("llm_request_intercept").unwrap().unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_llm_request_intercept("poison.llm_request").unwrap());

        assert!(
            context
                .register_llm_execution_intercept(
                    "llm_exec",
                    1,
                    helpers.getattr("llm_execution_intercept").unwrap().unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_llm_execution_intercept("poison.llm_exec").unwrap());

        assert!(
            context
                .register_llm_stream_execution_intercept(
                    "llm_stream",
                    1,
                    helpers
                        .getattr("llm_stream_execution_intercept")
                        .unwrap()
                        .unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_llm_stream_execution_intercept("poison.llm_stream").unwrap());

        assert!(
            context
                .register_tool_request_intercept(
                    "tool_request",
                    1,
                    false,
                    helpers.getattr("tool_request_intercept").unwrap().unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_tool_request_intercept("poison.tool_request").unwrap());

        assert!(
            context
                .register_tool_execution_intercept(
                    "tool_exec",
                    1,
                    helpers
                        .getattr("tool_execution_intercept")
                        .unwrap()
                        .unbind(),
                )
                .unwrap_err()
                .to_string()
                .contains("lock poisoned")
        );
        assert!(deregister_tool_execution_intercept("poison.tool_exec").unwrap());
    });
}
