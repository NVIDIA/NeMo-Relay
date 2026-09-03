// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for py plugin coverage in the NeMo Relay Python crate.

use super::*;

use std::collections::BTreeSet;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

use nemo_relay::api::registry::{RuntimeRegistrationKind, list_runtime_registrations};
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
    let event_loop = asyncio
        .getattr("SelectorEventLoop")
        .unwrap()
        .call0()
        .unwrap();
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

struct ErrorFixturePluginRegistrations;

impl Drop for ErrorFixturePluginRegistrations {
    fn drop(&mut self) {
        for kind in [
            "demo.raising_validate",
            "demo.invalid_diagnostics",
            "demo.failing_register",
            "demo.forced_failures",
        ] {
            let _ = deregister_plugin_py(kind);
        }
    }
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
fn plugin_context_rejects_legacy_and_uninspectable_llm_sanitizers() {
    let _python = crate::test_support::init_python_test();
    let context = PyPluginContext {
        registrations: Arc::new(Mutex::new(vec![])),
        namespace_prefix: "invalid.".to_string(),
    };

    Python::attach(|py| {
        let helpers = load_module(
            py,
            r#"
def one_argument(payload):
    return payload
"#,
        );
        for callback in [helpers.getattr("one_argument").unwrap().unbind(), py.None()] {
            let request_error = context
                .register_llm_sanitize_request_guardrail("request", 1, callback.clone_ref(py))
                .unwrap_err();
            assert!(request_error.to_string().contains("payload, context"));

            let response_error = context
                .register_llm_sanitize_response_guardrail("response", 1, callback)
                .unwrap_err();
            assert!(response_error.to_string().contains("payload, context"));
        }
    });
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
            "initialize",
            "validate",
            "list_plugin_kinds",
            "register_plugin",
            "deregister_plugin",
        ] {
            assert!(module.getattr(name).is_ok(), "missing binding: {name}");
        }

        let listed = list_plugin_kinds_py(py).unwrap();
        let listed_json = crate::convert::py_to_json(listed.bind(py)).unwrap();
        assert!(listed_json.is_array());

        let config = crate::convert::json_to_py(
            py,
            &json!({
                "version": 1,
                "components": []
            }),
        )
        .unwrap()
        .into_bound(py);
        let report = validate_py(py, &config, None).unwrap();
        let report_json = crate::convert::py_to_json(report.bind(py)).unwrap();
        assert!(report_json["config"]["diagnostics"].is_array());

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
fn python_plugin_validation_and_initialization_cover_error_paths() {
    let _python = crate::test_support::init_python_test();
    let _plugin_test_state = lock_plugin_test_state_for_tests();
    let _fixture_plugins = ErrorFixturePluginRegistrations;
    Python::attach(|py| {
        let helpers = load_module(
            py,
            r#"
class RaisingValidatePlugin:
    def validate(self, plugin_config):
        raise RuntimeError("validate boom")

class InvalidDiagnosticsPlugin:
    def validate(self, plugin_config):
        return [{"level": "warning", "code": 1, "message": "bad"}]

class FailingRegisterPlugin:
    def validate(self, plugin_config):
        return []

    def register(self, plugin_config, context):
        context.register_subscriber("sub", lambda event: None)
        raise RuntimeError("register boom")

class GoodPlugin:
    def validate(self, plugin_config):
        return []

    def register(self, plugin_config, context):
        context.register_subscriber("sub", lambda event: None)

async def initialize_plugin(module, config):
    return await module.initialize(config)

async def close_activation(activation):
    await activation.close()
"#,
        );
        let module = PyModule::new(py, "_plugin_cov_errors").unwrap();
        register(&module).unwrap();

        for (kind, class_name) in [
            ("demo.raising_validate", "RaisingValidatePlugin"),
            ("demo.invalid_diagnostics", "InvalidDiagnosticsPlugin"),
            ("demo.failing_register", "GoodPlugin"),
            ("demo.forced_failures", "GoodPlugin"),
        ] {
            register_plugin_py(
                kind,
                helpers
                    .getattr(class_name)
                    .unwrap()
                    .call0()
                    .unwrap()
                    .unbind(),
            )
            .unwrap();
        }

        let config_for = |kind: &str| {
            crate::convert::json_to_py(
                py,
                &json!({
                    "version": 1,
                    "components": [{"kind": kind, "enabled": true, "config": {}}]
                }),
            )
            .unwrap()
        };
        let has_validate_failure = |report: &Py<PyAny>| {
            crate::convert::py_to_json(report.bind(py)).unwrap()["config"]["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diagnostic| diagnostic["code"] == "plugin.validate_failed")
        };

        for kind in ["demo.raising_validate", "demo.invalid_diagnostics"] {
            let config = config_for(kind);
            let report = validate_py(py, config.bind(py), None).unwrap();
            assert!(has_validate_failure(&report));
        }

        let forced_config = config_for("demo.forced_failures");
        let conversion_guard = force_validate_config_to_py_error_for_tests("demo.forced_failures");
        let report = validate_py(py, forced_config.bind(py), None).unwrap();
        assert!(has_validate_failure(&report));
        drop(conversion_guard);

        with_event_loop(py, |event_loop| {
            let failing_config = config_for("demo.failing_register");
            let successful = helpers
                .getattr("initialize_plugin")
                .unwrap()
                .call1((module.clone(), failing_config.bind(py)))
                .unwrap();
            let activation = event_loop
                .call_method1("run_until_complete", (successful,))
                .unwrap();
            let subscriber_kinds = BTreeSet::from([RuntimeRegistrationKind::Subscriber]);
            let subscriber_name = list_runtime_registrations(Some(&subscriber_kinds))
                .unwrap()
                .into_iter()
                .find(|registration| {
                    registration.local_name == "sub"
                        && registration.owner.plugin_kind.as_deref()
                            == Some("demo.failing_register")
                })
                .expect("successful fixture registration should expose its effective name")
                .effective_name;
            let close = helpers
                .getattr("close_activation")
                .unwrap()
                .call1((activation,))
                .unwrap();
            event_loop
                .call_method1("run_until_complete", (close,))
                .unwrap();
            assert!(deregister_plugin_py("demo.failing_register"));
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

            let failing = helpers
                .getattr("initialize_plugin")
                .unwrap()
                .call1((module.clone(), failing_config.bind(py)))
                .unwrap();
            let error = event_loop
                .call_method1("run_until_complete", (failing,))
                .unwrap_err();
            assert!(error.to_string().contains("register boom"), "{error}");
            assert!(
                !deregister_subscriber(&subscriber_name).unwrap(),
                "failed initialization should roll back partial registrations"
            );

            let context_guard = force_plugin_context_new_error_for_tests("demo.forced_failures");
            let failing = helpers
                .getattr("initialize_plugin")
                .unwrap()
                .call1((module.clone(), forced_config.bind(py)))
                .unwrap();
            let error = event_loop
                .call_method1("run_until_complete", (failing,))
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("forced plugin context allocation failure"),
                "{error}"
            );
            drop(context_guard);
        });
    });
}

#[test]
#[allow(clippy::cognitive_complexity)]
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

def event_metadata(event):
    return {"python.plugin": event.name}

def tool_fn(name, value):
    return value

def tool_conditional(name, value):
    return None

def llm_sanitize_request(request, context):
    return request

def llm_sanitize_response(response, context):
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
    downstream = await next(value)
    return ToolOutcome(downstream.result, annotation=downstream.annotation)
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
            .register_event_metadata_injector(
                "event_metadata",
                1,
                helpers.getattr("event_metadata").unwrap().unbind(),
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
        assert_eq!(registrations.len(), 16);
        assert!(
            registrations
                .iter()
                .all(|registration| registration.name.starts_with("demo."))
        );

        assert!(deregister_subscriber("demo.subscriber").unwrap());
        assert!(deregister_event_metadata_injector("demo.event_metadata").unwrap());
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

def llm_sanitize_request(request, context):
    return request

def llm_sanitize_response(response, context):
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
    downstream = await next(value)
    return ToolOutcome(downstream.result, annotation=downstream.annotation)
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

def llm_sanitize_request(request, context):
    return request

def llm_sanitize_response(response, context):
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
    downstream = await next(value)
    return ToolOutcome(downstream.result, annotation=downstream.annotation)
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

        fn assert_poisoned_tool_registrations(
            context: &PyPluginContext,
            helpers: &Bound<'_, PyModule>,
        ) {
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
        }
        assert_poisoned_tool_registrations(&context, &helpers);

        fn assert_poisoned_llm_guardrail_registrations(
            context: &PyPluginContext,
            helpers: &Bound<'_, PyModule>,
        ) {
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
        }
        assert_poisoned_llm_guardrail_registrations(&context, &helpers);

        fn assert_poisoned_intercept_registrations(
            context: &PyPluginContext,
            helpers: &Bound<'_, PyModule>,
        ) {
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
        }
        assert_poisoned_intercept_registrations(&context, &helpers);
    });
}
