// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::{Mutex, OnceLock};

use crate::convert::{clear_last_callback_error, record_callback_error};
#[cfg(target_arch = "wasm32")]
use serde_json::json;

static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn test_mutex() -> &'static Mutex<()> {
    TEST_MUTEX.get_or_init(|| Mutex::new(()))
}

#[test]
fn wasm_config_defaults_match_expected_values() {
    let otel_config = WasmOpenTelemetryConfig::default();
    assert_eq!(otel_config.transport.as_deref(), Some("http_binary"));
    assert_eq!(otel_config.service_name.as_deref(), Some("nemo-flow"));
    assert_eq!(
        otel_config.instrumentation_scope.as_deref(),
        Some("nemo-flow-otel")
    );
    assert_eq!(otel_config.timeout_millis, Some(3_000));

    let openinference_config = WasmOpenInferenceConfig::default();
    assert_eq!(
        openinference_config.transport.as_deref(),
        Some("http_binary")
    );
    assert_eq!(
        openinference_config.service_name.as_deref(),
        Some("nemo-flow")
    );
    assert_eq!(
        openinference_config.instrumentation_scope.as_deref(),
        Some("nemo-flow-openinference")
    );
    assert_eq!(openinference_config.timeout_millis, Some(3_000));
}

#[test]
fn config_builders_accept_explicit_overrides() {
    assert!(
        build_otel_config(Some(WasmOpenTelemetryConfig {
            transport: Some("grpc".to_string()),
            endpoint: Some("http://localhost:4317".to_string()),
            headers: Some(HashMap::from([(
                "authorization".to_string(),
                "Bearer token".to_string()
            )])),
            resource_attributes: Some(HashMap::from([(
                "deployment.environment".to_string(),
                "test".to_string(),
            )])),
            service_name: Some("demo-agent".to_string()),
            service_namespace: Some("agents".to_string()),
            service_version: Some("1.2.3".to_string()),
            instrumentation_scope: Some("demo-scope".to_string()),
            timeout_millis: Some(1_250),
        }))
        .is_ok()
    );

    assert!(
        build_openinference_config(Some(WasmOpenInferenceConfig {
            transport: Some("grpc".to_string()),
            endpoint: Some("http://localhost:4317".to_string()),
            headers: Some(HashMap::from([(
                "authorization".to_string(),
                "Bearer token".to_string()
            )])),
            resource_attributes: Some(HashMap::from([(
                "deployment.environment".to_string(),
                "test".to_string(),
            )])),
            service_name: Some("demo-agent".to_string()),
            service_namespace: Some("agents".to_string()),
            service_version: Some("1.2.3".to_string()),
            instrumentation_scope: Some("demo-scope".to_string()),
            timeout_millis: Some(1_250),
        }))
        .is_ok()
    );
}

fn wasm_atif_exporter_exports_full_trajectory_without_root_parameter() {
    let exporter = WasmAtifExporter::new(
        "session-wasm".to_string(),
        "test-agent".to_string(),
        "1.0.0".to_string(),
        Some("demo-model".to_string()),
    );
    let export: serde_json::Value = serde_json::from_str(&exporter.export_json().unwrap()).unwrap();
    assert_eq!(export["session_id"], "session-wasm");
    assert_eq!(export["agent"]["name"], "test-agent");
    assert!(export["steps"].as_array().unwrap().is_empty());

    exporter.clear();
    let cleared: serde_json::Value =
        serde_json::from_str(&exporter.export_json().unwrap()).unwrap();
    assert!(cleared["steps"].as_array().unwrap().is_empty());
}

#[cfg(target_arch = "wasm32")]
#[test]
fn adaptive_config_validation_and_runtime_report_round_trip() {
    let config = serde_wasm_bindgen::to_value(&serde_json::json!({
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
    }))
    .unwrap();

    let report = validate_plugin_config(config.clone()).unwrap();
    let report_json: serde_json::Value = serde_wasm_bindgen::from_value(report).unwrap();
    assert_eq!(report_json["diagnostics"], serde_json::json!([]));
}

#[test]
fn callback_error_wrapper_accessors_round_trip() {
    clear_last_callback_error();
    assert_eq!(get_last_callback_error(), None);

    record_callback_error("wasm wrapper callback failed");
    assert_eq!(
        get_last_callback_error(),
        Some("wasm wrapper callback failed".to_string())
    );

    clear_last_callback_error();
    assert_eq!(get_last_callback_error(), None);
}

#[test]
fn adaptive_plugin_context_helpers_work_natively() {
    let _guard = test_mutex().lock().unwrap_or_else(|e| e.into_inner());
    let context = WasmPluginContext {
        registrations: Arc::new(Mutex::new(Vec::new())),
        namespace_prefix: String::new(),
    };
    context
        .push_registration(ComponentRegistration::new(
            "plugin",
            "plugin.reg".to_string(),
            Box::new(|| Ok(())),
        ))
        .unwrap();
    let drained = context.drain_registrations().unwrap();
    assert_eq!(drained.len(), 1);
}

#[cfg(target_arch = "wasm32")]
#[test]
fn config_builders_reject_invalid_transport_values() {
    let otel_err = build_otel_config(Some(WasmOpenTelemetryConfig {
        transport: Some("invalid".to_string()),
        ..Default::default()
    }))
    .unwrap_err();
    assert!(
        otel_err
            .as_string()
            .unwrap()
            .contains("transport must be 'http_binary' or 'grpc'")
    );

    let openinference_err = build_openinference_config(Some(WasmOpenInferenceConfig {
        transport: Some("invalid".to_string()),
        ..Default::default()
    }))
    .unwrap_err();
    assert!(
        openinference_err
            .as_string()
            .unwrap()
            .contains("transport must be 'http_binary' or 'grpc'")
    );
}

#[cfg(target_arch = "wasm32")]
#[test]
fn scope_stack_and_lifecycle_wrappers_round_trip_natively() {
    let _guard = test_mutex().lock().unwrap_or_else(|e| e.into_inner());

    assert!(!scope_stack_active());
    let stack = create_scope_stack();
    set_thread_scope_stack(&stack);
    assert!(scope_stack_active());
    assert_eq!(
        current_scope_stack().inner.read().unwrap().top().name,
        "root"
    );

    let root = get_handle().unwrap();
    let child = push_scope(
        "child",
        SCOPE_TYPE_TOOL,
        Some(root),
        Some(SCOPE_PARALLEL),
        serde_wasm_bindgen::to_value(&json!({"payload": true})).unwrap(),
        serde_wasm_bindgen::to_value(&json!({"meta": true})).unwrap(),
    )
    .unwrap();
    assert_eq!(child.name(), "child");
    assert_eq!(child.scope_type(), SCOPE_TYPE_TOOL);
    assert_eq!(child.attributes(), SCOPE_PARALLEL);

    event(
        "mark",
        Some(get_handle().unwrap()),
        serde_wasm_bindgen::to_value(&json!({"step": 1})).unwrap(),
        serde_wasm_bindgen::to_value(&json!({"source": "api"})).unwrap(),
    )
    .unwrap();

    let tool = tool_call(
        "tool",
        serde_wasm_bindgen::to_value(&json!({"arg": 1})).unwrap(),
        Some(get_handle().unwrap()),
        Some(TOOL_LOCAL),
        serde_wasm_bindgen::to_value(&json!({"tool_data": true})).unwrap(),
        serde_wasm_bindgen::to_value(&json!({"tool_meta": true})).unwrap(),
        Some("tool-call".to_string()),
    )
    .unwrap();
    tool_call_end(
        &tool,
        serde_wasm_bindgen::to_value(&json!({"result": 2})).unwrap(),
        serde_wasm_bindgen::to_value(&json!({"done": true})).unwrap(),
        serde_wasm_bindgen::to_value(&json!({"status": "ok"})).unwrap(),
    )
    .unwrap();

    let llm = llm_call(
        "llm",
        serde_wasm_bindgen::to_value(&json!({
            "headers": {},
            "content": {"messages": [], "model": "demo"}
        }))
        .unwrap(),
        Some(get_handle().unwrap()),
        Some(LLM_STATELESS | LLM_STREAMING),
        serde_wasm_bindgen::to_value(&json!({"llm_data": true})).unwrap(),
        serde_wasm_bindgen::to_value(&json!({"llm_meta": true})).unwrap(),
        Some("demo-model".to_string()),
    )
    .unwrap();
    llm_call_end(
        &llm,
        serde_wasm_bindgen::to_value(&json!({"response": "ok"})).unwrap(),
        serde_wasm_bindgen::to_value(&json!({"tokens": 10})).unwrap(),
        serde_wasm_bindgen::to_value(&json!({"finish_reason": "stop"})).unwrap(),
    )
    .unwrap();

    pop_scope(&child).unwrap();
    assert_eq!(get_handle().unwrap().name(), "root");
}
