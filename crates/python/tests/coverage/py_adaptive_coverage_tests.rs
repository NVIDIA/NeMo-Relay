// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for py adaptive coverage in the NeMo Relay Python crate.

use super::*;

use std::fs;

use nemo_relay::api::scope::{ScopeHandle, ScopeType as CoreScopeType};
use pyo3::types::{PyDict, PyModule};
use serde_json::json;

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

fn adaptive_config<'py>(py: Python<'py>, provider: &str) -> Bound<'py, pyo3::types::PyAny> {
    crate::convert::json_to_py(
        py,
        &json!({
            "version": 1,
            "state": {
                "backend": {
                    "kind": "in_memory",
                    "config": {}
                }
            },
            "acg": {
                "provider": provider,
                "observation_window": 8,
                "priority": 12
            }
        }),
    )
    .unwrap()
    .into_bound(py)
}

#[test]
fn set_latency_sensitivity_rejects_zero_and_registers_binding() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let module = PyModule::new(py, "_adaptive_cov").unwrap();
        register(&module).unwrap();
        assert!(module.getattr("AdaptiveRuntime").is_ok());
        assert!(module.getattr("set_latency_sensitivity").is_ok());

        let err = set_latency_sensitivity(0).unwrap_err();
        assert!(err.to_string().contains("sensitivity must be positive"));

        set_latency_sensitivity(3).unwrap();
    });
}

#[test]
fn py_adaptive_uses_canonical_adaptive_acg_imports() {
    let source =
        fs::read_to_string(format!("{}/src/py_adaptive.rs", env!("CARGO_MANIFEST_DIR"))).unwrap();

    assert!(source.contains("nemo_relay_adaptive::acg"));
    assert!(!source.contains("nemo_relay_acg::"));
}

#[test]
fn python_crate_manifest_drops_direct_acg_dependency() {
    let manifest =
        fs::read_to_string(format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"))).unwrap();

    assert!(!manifest.contains("nemo-relay-acg ="));
}

#[test]
fn validate_adaptive_config_accepts_openai_provider_without_transport_fields() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_adaptive_cov").unwrap();
        register(&module).unwrap();

        let backend = PyDict::new(py);
        backend.set_item("kind", "in_memory").unwrap();
        backend.set_item("config", PyDict::new(py)).unwrap();

        let state = PyDict::new(py);
        state.set_item("backend", backend).unwrap();

        let acg = PyDict::new(py);
        acg.set_item("provider", "openai").unwrap();
        acg.set_item("observation_window", 8).unwrap();
        acg.set_item("priority", 12).unwrap();

        let config = PyDict::new(py);
        config.set_item("version", 1).unwrap();
        config.set_item("state", state).unwrap();
        config.set_item("acg", acg).unwrap();

        let report = module
            .getattr("validate_adaptive_config")
            .unwrap()
            .call1((config,))
            .unwrap();
        let diagnostics = report.get_item("diagnostics").unwrap();
        assert_eq!(diagnostics.len().unwrap(), 0);
    });
}

#[test]
fn adaptive_runtime_methods_cover_pending_ready_and_shutdown_paths() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let runner = PyModule::from_code(
            py,
            pyo3::ffi::c_str!(
                r#"
async def await_value(awaitable):
    return await awaitable

async def register_runtime(runtime):
    return await runtime.register()

async def shutdown_runtime(runtime):
    return await runtime.shutdown()

def bind_scope_and_translate(api_module, runtime, request):
    handle = api_module.push_scope("adaptive-runtime-cov", api_module.ScopeType.Agent)
    try:
        runtime.bind_scope(handle)
        return api_module.llm_request_intercepts("anthropic", request).request.content
    finally:
        api_module.pop_scope(handle)
"#
            ),
            pyo3::ffi::c_str!("py_adaptive_runtime_cov.py"),
            pyo3::ffi::c_str!("py_adaptive_runtime_cov"),
        )
        .unwrap();

        let module = PyModule::new(py, "_adaptive_runtime_cov").unwrap();
        crate::py_types::register(&module).unwrap();
        crate::py_api::register(&module).unwrap();
        register(&module).unwrap();
        let runtime = module
            .getattr("AdaptiveRuntime")
            .unwrap()
            .call1((adaptive_config(py, "anthropic"),))
            .unwrap();
        assert_eq!(
            runtime.repr().unwrap().to_str().unwrap(),
            "<AdaptiveRuntime>"
        );
        let request = module
            .getattr("LLMRequest")
            .unwrap()
            .call1((
                PyDict::new(py),
                crate::convert::json_to_py(
                    py,
                    &json!({
                        "messages": [{"role": "user", "content": "hello"}],
                        "system": "plan carefully",
                        "model": "claude-sonnet-4-20250514"
                    }),
                )
                .unwrap(),
            ))
            .unwrap();

        let pending_report =
            crate::convert::py_to_json(&runtime.call_method0("report").unwrap()).unwrap();
        assert_eq!(pending_report["diagnostics"], json!([]));
        runtime.call_method0("wait_for_idle").unwrap();
        runtime.call_method0("deregister").unwrap();

        let annotated_request = crate::convert::json_to_py(
            py,
            &json!({
                "messages": [
                    {"role": "system", "content": "plan carefully"},
                    {"role": "user", "content": "find cache facts"}
                ],
                "model": "claude-sonnet-4-20250514"
            }),
        )
        .unwrap();
        let pending_kwargs = PyDict::new(py);
        pending_kwargs.set_item("provider", "anthropic").unwrap();
        pending_kwargs
            .set_item("request_id", "00000000-0000-0000-0000-000000000201")
            .unwrap();
        pending_kwargs
            .set_item("annotated_request", annotated_request.bind(py))
            .unwrap();
        pending_kwargs
            .set_item("agent_id", "adaptive-runtime-cov")
            .unwrap();
        pending_kwargs
            .set_item("timestamp", "2026-01-01T00:00:00Z")
            .unwrap();
        let pending_err = runtime
            .call_method("build_cache_request_facts", (), Some(&pending_kwargs))
            .unwrap_err();
        assert!(
            pending_err
                .to_string()
                .contains("must be registered before building cache request facts")
        );
        let pending_scope = Py::new(
            py,
            crate::py_types::PyScopeHandle {
                inner: ScopeHandle::builder()
                    .name("adaptive-runtime-cov-pending")
                    .scope_type(CoreScopeType::Agent)
                    .build(),
            },
        )
        .unwrap();
        let pending_bind_err = runtime
            .call_method1("bind_scope", (pending_scope.clone_ref(py),))
            .unwrap_err();
        assert!(
            pending_bind_err
                .to_string()
                .contains("must be registered before binding ACG request intercepts")
        );

        with_event_loop(py, |event_loop| {
            event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("register_runtime")
                        .unwrap()
                        .call1((runtime.clone(),))
                        .unwrap(),),
                )
                .unwrap();
        });

        let ready_report =
            crate::convert::py_to_json(&runtime.call_method0("report").unwrap()).unwrap();
        assert_eq!(ready_report["diagnostics"], json!([]));
        runtime.call_method0("wait_for_idle").unwrap();

        let facts_kwargs = PyDict::new(py);
        facts_kwargs.set_item("provider", "anthropic").unwrap();
        facts_kwargs
            .set_item("request_id", "00000000-0000-0000-0000-000000000202")
            .unwrap();
        facts_kwargs
            .set_item("annotated_request", annotated_request.bind(py))
            .unwrap();
        facts_kwargs
            .set_item("agent_id", "adaptive-runtime-cov")
            .unwrap();
        let facts = runtime
            .call_method("build_cache_request_facts", (), Some(&facts_kwargs))
            .unwrap();
        let facts_json = crate::convert::py_to_json(&facts).unwrap();
        assert_eq!(facts_json["provider"], json!("anthropic"));
        let rewritten_content = with_event_loop(py, |_event_loop| {
            let result = runner
                .getattr("bind_scope_and_translate")
                .unwrap()
                .call1((&module, runtime.clone(), request.clone()))
                .unwrap();
            crate::convert::py_to_json(&result).unwrap()
        });
        assert_eq!(
            rewritten_content,
            json!({
                "messages": [{"role": "user", "content": "hello"}],
                "system": "plan carefully",
                "model": "claude-sonnet-4-20250514"
            })
        );

        with_event_loop(py, |event_loop| {
            event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("register_runtime")
                        .unwrap()
                        .call1((runtime.clone(),))
                        .unwrap(),),
                )
                .unwrap();
        });

        runtime.call_method0("deregister").unwrap();

        with_event_loop(py, |event_loop| {
            event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("shutdown_runtime")
                        .unwrap()
                        .call1((runtime.clone(),))
                        .unwrap(),),
                )
                .unwrap();
        });

        assert!(
            runtime
                .call_method0("report")
                .unwrap_err()
                .to_string()
                .contains("already shut down")
        );
        assert!(
            runtime
                .call_method("build_cache_request_facts", (), Some(&facts_kwargs))
                .unwrap_err()
                .to_string()
                .contains("already shut down")
        );
        let shutdown_scope = Py::new(
            py,
            crate::py_types::PyScopeHandle {
                inner: ScopeHandle::builder()
                    .name("adaptive-runtime-cov-shutdown")
                    .scope_type(CoreScopeType::Agent)
                    .build(),
            },
        )
        .unwrap();
        assert!(
            runtime
                .call_method1("bind_scope", (shutdown_scope.clone_ref(py),))
                .unwrap_err()
                .to_string()
                .contains("already shut down")
        );
    });
}

#[test]
fn adaptive_runtime_locking_and_helper_errors_are_covered() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let parsed_config: AdaptiveConfig = serde_json::from_value(json!({
            "version": 1,
            "state": {
                "backend": {
                    "kind": "in_memory",
                    "config": {}
                }
            },
            "acg": {
                "provider": "anthropic",
                "observation_window": 4,
                "priority": 7
            }
        }))
        .unwrap();
        let report = AdaptiveRuntime::validate_config(&parsed_config);
        let locked_runtime = PyAdaptiveRuntime {
            inner: std::sync::Arc::new(tokio::sync::Mutex::new(Some(
                PyAdaptiveRuntimeState::Pending {
                    config: parsed_config.clone(),
                    report: report.clone(),
                },
            ))),
        };

        let guard = locked_runtime.inner.try_lock().unwrap();
        let _llm_request = crate::py_types::PyLLMRequest {
            inner: nemo_relay::api::llm::LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "messages": [{"role": "user", "content": "hello"}],
                    "model": "claude-sonnet-4-20250514"
                }),
            },
        };
        let annotated_request = crate::convert::json_to_py(
            py,
            &json!({
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4.1-mini"
            }),
        )
        .unwrap();
        assert!(
            locked_runtime
                .deregister()
                .unwrap_err()
                .to_string()
                .contains("locked by an async operation")
        );
        assert!(
            locked_runtime
                .wait_for_idle()
                .unwrap_err()
                .to_string()
                .contains("locked by an async operation")
        );
        assert!(
            locked_runtime
                .report(py)
                .unwrap_err()
                .to_string()
                .contains("locked by an async operation")
        );
        let scope_handle = crate::py_types::PyScopeHandle {
            inner: ScopeHandle::builder()
                .name("adaptive-runtime-cov-locked")
                .scope_type(CoreScopeType::Agent)
                .build(),
        };
        assert!(match locked_runtime.bind_scope(&scope_handle) {
            Ok(_) => panic!("expected locked runtime rewrite to fail"),
            Err(err) => err.to_string().contains("locked by an async operation"),
        });
        assert!(
            locked_runtime
                .build_cache_request_facts(
                    py,
                    "openai",
                    "00000000-0000-0000-0000-000000000204",
                    annotated_request.bind(py),
                    "adaptive-runtime-cov",
                    None,
                )
                .unwrap_err()
                .to_string()
                .contains("locked by an async operation")
        );
        drop(guard);

        let empty_runtime = PyAdaptiveRuntime {
            inner: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
        };
        assert!(
            empty_runtime
                .deregister()
                .unwrap_err()
                .to_string()
                .contains("already shut down")
        );
        assert!(
            empty_runtime
                .wait_for_idle()
                .unwrap_err()
                .to_string()
                .contains("already shut down")
        );
        let scope_handle = crate::py_types::PyScopeHandle {
            inner: ScopeHandle::builder()
                .name("adaptive-runtime-cov-empty")
                .scope_type(CoreScopeType::Agent)
                .build(),
        };
        assert!(match empty_runtime.bind_scope(&scope_handle) {
            Ok(_) => panic!("expected shut down runtime rewrite to fail"),
            Err(err) => err.to_string().contains("already shut down"),
        });

        let valid_report = validate_adaptive_config_or_err(&parsed_config).unwrap();
        assert!(!valid_report.has_errors());

        let invalid_config: AdaptiveConfig = serde_json::from_value(json!({
            "version": 1,
            "policy": {
                "unknown_component": "warn",
                "unknown_field": "warn",
                "unsupported_value": "error"
            },
            "tool_parallelism": {
                "mode": "definitely_not_supported"
            }
        }))
        .unwrap();
        let invalid_err = validate_adaptive_config_or_err(&invalid_config).unwrap_err();
        assert!(invalid_err.to_string().contains("unsupported"));

        assert!(matches!(
            parse_cache_telemetry_provider("anthropic").unwrap(),
            CacheTelemetryProvider::Anthropic
        ));
        assert!(matches!(
            parse_cache_telemetry_provider("openai").unwrap(),
            CacheTelemetryProvider::OpenAI
        ));
        assert!(
            parse_cache_telemetry_provider("bogus")
                .unwrap_err()
                .to_string()
                .contains("unsupported provider")
        );
        assert!(
            parse_cache_telemetry_request_id("not-a-uuid")
                .unwrap_err()
                .to_string()
                .contains("invalid request_id UUID")
        );
        assert!(
            parse_cache_telemetry_timestamp(Some("not-a-timestamp"))
                .unwrap_err()
                .to_string()
                .contains("invalid timestamp")
        );
        assert!(parse_cache_telemetry_timestamp(None).is_ok());

        let types_module = PyModule::new(py, "_adaptive_types").unwrap();
        crate::py_types::register(&types_module).unwrap();
        let wrapped_messages =
            crate::convert::json_to_py(py, &json!([{"role": "user", "content": "wrapped"}]))
                .unwrap();
        let annotated_wrapper = types_module
            .getattr("AnnotatedLLMRequest")
            .unwrap()
            .call1((wrapped_messages,))
            .unwrap();
        assert_eq!(
            parse_annotated_request(annotated_wrapper.as_any())
                .unwrap()
                .last_user_message(),
            Some("wrapped")
        );
        assert_eq!(
            parse_annotated_request(
                crate::convert::json_to_py(
                    py,
                    &json!({
                        "messages": [{"role": "user", "content": "json"}],
                        "model": "gpt-4.1-mini"
                    }),
                )
                .unwrap()
                .bind(py),
            )
            .unwrap()
            .last_user_message(),
            Some("json")
        );
        assert!(
            parse_annotated_request(py.None().bind(py))
                .unwrap_err()
                .to_string()
                .contains("invalid annotated_request")
        );

        assert!(parse_cache_request_facts(None).unwrap().is_none());
        assert!(
            parse_cache_request_facts(Some(
                crate::convert::json_to_py(py, &json!({"not": "facts"}))
                    .unwrap()
                    .bind(py),
            ))
            .unwrap_err()
            .to_string()
            .contains("invalid request_facts")
        );

        let no_usage = build_cache_telemetry_event(
            py,
            "anthropic",
            "00000000-0000-0000-0000-000000000205",
            None,
            None,
            "adaptive-runtime-cov",
            "unknown",
            "unknown",
            "claude-sonnet-4-20250514",
            "default",
            None,
        )
        .unwrap();
        assert!(no_usage.bind(py).is_none());

        let invalid_usage = build_cache_telemetry_event(
            py,
            "anthropic",
            "00000000-0000-0000-0000-000000000206",
            Some(
                crate::convert::json_to_py(py, &json!("bad"))
                    .unwrap()
                    .bind(py),
            ),
            None,
            "adaptive-runtime-cov",
            "unknown",
            "unknown",
            "claude-sonnet-4-20250514",
            "default",
            None,
        )
        .unwrap_err();
        assert!(invalid_usage.to_string().contains("invalid usage"));

        let validate_err = validate_adaptive_config(py, py.None().bind(py)).unwrap_err();
        assert!(validate_err.to_string().contains("ValueError"));
        assert_eq!(to_py_err("boom").to_string(), "RuntimeError: boom");
    });
}

#[test]
fn adaptive_runtime_shutdown_and_register_error_paths_are_covered() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let runner = PyModule::from_code(
            py,
            pyo3::ffi::c_str!(
                r#"
async def register_runtime(runtime):
    return await runtime.register()

async def shutdown_runtime(runtime):
    return await runtime.shutdown()
"#
            ),
            pyo3::ffi::c_str!("py_adaptive_runtime_edges.py"),
            pyo3::ffi::c_str!("py_adaptive_runtime_edges"),
        )
        .unwrap();

        let module = PyModule::new(py, "_adaptive_runtime_edges").unwrap();
        register(&module).unwrap();

        let pending_runtime = module
            .getattr("AdaptiveRuntime")
            .unwrap()
            .call1((adaptive_config(py, "anthropic"),))
            .unwrap();
        with_event_loop(py, |event_loop| {
            event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("shutdown_runtime")
                        .unwrap()
                        .call1((pending_runtime.clone(),))
                        .unwrap(),),
                )
                .unwrap();
        });
        assert!(
            pending_runtime
                .call_method0("report")
                .unwrap_err()
                .to_string()
                .contains("already shut down")
        );
        with_event_loop(py, |event_loop| {
            let err = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("register_runtime")
                        .unwrap()
                        .call1((pending_runtime.clone(),))
                        .unwrap(),),
                )
                .unwrap_err();
            assert!(err.to_string().contains("already shut down"));
        });
        with_event_loop(py, |event_loop| {
            let err = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("shutdown_runtime")
                        .unwrap()
                        .call1((pending_runtime.clone(),))
                        .unwrap(),),
                )
                .unwrap_err();
            assert!(err.to_string().contains("already shut down"));
        });

        let valid_config: AdaptiveConfig = serde_json::from_value(json!({
            "version": 1,
            "state": {
                "backend": {
                    "kind": "in_memory",
                    "config": {}
                }
            },
            "acg": {
                "provider": "anthropic",
                "observation_window": 4,
                "priority": 7
            }
        }))
        .unwrap();
        let invalid_runtime = PyAdaptiveRuntime {
            inner: std::sync::Arc::new(tokio::sync::Mutex::new(Some(
                PyAdaptiveRuntimeState::Pending {
                    config: serde_json::from_value(json!({
                        "version": 1,
                        "state": {
                            "backend": {
                                "kind": "definitely_not_supported",
                                "config": {}
                            }
                        }
                    }))
                    .unwrap(),
                    report: AdaptiveRuntime::validate_config(&valid_config),
                },
            ))),
        };
        with_event_loop(py, |event_loop| {
            let err = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("register_runtime")
                        .unwrap()
                        .call1((pyo3::Py::new(py, invalid_runtime).unwrap(),))
                        .unwrap(),),
                )
                .unwrap_err();
            assert!(err.to_string().contains("unsupported backend"));
        });

        let no_prompt_tokens = build_cache_telemetry_event(
            py,
            "anthropic",
            "00000000-0000-0000-0000-000000000207",
            Some(
                crate::convert::json_to_py(
                    py,
                    &json!({
                        "completion_tokens": 2,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0
                    }),
                )
                .unwrap()
                .bind(py),
            ),
            None,
            "adaptive-runtime-cov",
            "unknown",
            "unknown",
            "claude-sonnet-4-20250514",
            "default",
            None,
        )
        .unwrap();
        assert!(no_prompt_tokens.bind(py).is_none());
    });
}
