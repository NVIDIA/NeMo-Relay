// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use wasm_bindgen_test::*;

use nvidia_nat_nexus_wasm::api::{
    deregister_optimizer_plugin, register_optimizer_plugin, validate_optimizer_config,
    WasmOptimizerRuntime,
};

#[wasm_bindgen_test]
fn optimizer_config_validation_and_runtime_report_round_trip() {
    let config = serde_wasm_bindgen::to_value(&json!({
        "version": 1,
        "state": {
            "backend": {
                "kind": "in_memory",
                "config": {}
            }
        },
        "components": [
            {
                "kind": "telemetry",
                "enabled": true,
                "config": {
                    "learners": ["latency_sensitivity"]
                }
            },
            {
                "kind": "dynamo_hints",
                "enabled": true,
                "config": {}
            },
            {
                "kind": "tool_parallelism",
                "enabled": true,
                "config": {}
            }
        ]
    }))
    .unwrap();

    let report = validate_optimizer_config(config.clone()).unwrap();
    let report_json: serde_json::Value = serde_wasm_bindgen::from_value(report).unwrap();
    assert_eq!(report_json["diagnostics"], json!([]));

    let runtime = WasmOptimizerRuntime::new(config).unwrap();
    let report_json: serde_json::Value =
        serde_wasm_bindgen::from_value(runtime.report().unwrap()).unwrap();
    assert_eq!(report_json["diagnostics"], json!([]));
}

#[wasm_bindgen_test]
async fn optimizer_hosted_plugin_validation_and_registration_work() {
    let validate = js_sys::Function::new_with_args("instanceId, pluginConfig", "return [];");
    let register = js_sys::Function::new_with_args(
        "instanceId, pluginConfig, context",
        r#"
            context.registerToolRequestIntercept(
                `${instanceId}.toolRequest`,
                25,
                false,
                function(name, args) {
                    args.wasmToolPlugin = instanceId;
                    return args;
                },
            );
            context.registerLlmExecutionIntercept(
                `${instanceId}.llmExec`,
                25,
                function(request, next) {
                    return Promise.resolve(next(request)).then((result) => {
                        result.wasmLlmPlugin = instanceId;
                        return result;
                    });
                },
            );
            context.registerLlmStreamExecutionIntercept(
                `${instanceId}.llmStreamExec`,
                25,
                function(request, next) {
                    return next(request);
                },
            );
            return undefined;
        "#,
    );

    register_optimizer_plugin(
        "wasm.test.optimizer_plugin.register".to_string(),
        Some(validate),
        register,
    )
    .unwrap();

    let register_config = serde_wasm_bindgen::to_value(&json!({
        "version": 1,
        "components": [
            {
                "kind": "external_component",
                "enabled": true,
                "config": {
                    "plugin_kind": "wasm.test.optimizer_plugin.register",
                    "instance_id": "wasm-plugin-register",
                    "plugin_config": {
                        "threshold": 17
                    }
                }
            }
        ]
    }))
    .unwrap();

    let report = validate_optimizer_config(register_config.clone()).unwrap();
    let report_json: serde_json::Value = serde_wasm_bindgen::from_value(report).unwrap();
    assert_eq!(report_json["diagnostics"], json!([]));

    let runtime = WasmOptimizerRuntime::new(register_config).unwrap();
    let runtime_report: serde_json::Value =
        serde_wasm_bindgen::from_value(runtime.report().unwrap()).unwrap();
    assert_eq!(runtime_report["diagnostics"], json!([]));
    runtime.register().await.unwrap();

    runtime.deregister().unwrap();
    runtime.shutdown().await.unwrap();
    assert!(deregister_optimizer_plugin(
        "wasm.test.optimizer_plugin.register".to_string()
    ));
}

#[wasm_bindgen_test]
fn optimizer_hosted_plugin_registry_and_unknown_kind_diagnostics_work() {
    let validate = js_sys::Function::new_with_args("instanceId, pluginConfig", "return [];");
    let register =
        js_sys::Function::new_with_args("instanceId, pluginConfig, context", "return undefined;");

    assert!(!deregister_optimizer_plugin(
        "wasm.test.optimizer_plugin.missing".to_string()
    ));

    register_optimizer_plugin(
        "wasm.test.optimizer_plugin.duplicate".to_string(),
        Some(validate.clone()),
        register.clone(),
    )
    .unwrap();

    let duplicate = register_optimizer_plugin(
        "wasm.test.optimizer_plugin.duplicate".to_string(),
        Some(validate),
        register,
    );
    assert!(duplicate.is_err());

    assert!(deregister_optimizer_plugin(
        "wasm.test.optimizer_plugin.duplicate".to_string()
    ));
}
