// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use wasm_bindgen_test::*;

use nemo_flow_wasm::api::{
    clear_plugin_configuration, deregister_plugin, initialize_plugins, register_plugin,
    validate_plugin_config,
};

#[wasm_bindgen_test]
fn adaptive_config_validation_round_trip() {
    let config = serde_wasm_bindgen::to_value(&json!({
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
            }
        ]
    }))
    .unwrap();

    let report = validate_plugin_config(config.clone()).unwrap();
    let report_json: serde_json::Value = serde_wasm_bindgen::from_value(report).unwrap();
    assert_eq!(report_json["diagnostics"], json!([]));
}

#[wasm_bindgen_test]
async fn top_level_plugin_validation_and_registration_work() {
    let validate = js_sys::Function::new_with_args("pluginConfig", "return [];");
    let register = js_sys::Function::new_with_args(
        "pluginConfig, context",
        r#"
            context.registerToolRequestIntercept(
                "toolRequest",
                25,
                false,
                function(name, args) {
                    args.wasmToolPlugin = `threshold:${pluginConfig.threshold}`;
                    return args;
                },
            );
            context.registerLlmExecutionIntercept(
                "llmExec",
                25,
                function(request, next) {
                    return Promise.resolve(next(request)).then((result) => {
                        result.wasmLlmPlugin = `threshold:${pluginConfig.threshold}`;
                        return result;
                    });
                },
            );
            context.registerLlmStreamExecutionIntercept(
                "llmStreamExec",
                25,
                function(request, next) {
                    return next(request);
                },
            );
            return undefined;
        "#,
    );

    register_plugin(
        "wasm.test.plugin.register".to_string(),
        Some(validate),
        register,
    )
    .unwrap();

    let register_config = serde_wasm_bindgen::to_value(&json!({
        "version": 1,
        "components": [
            {
                "kind": "adaptive",
                "enabled": true,
                "config": {
                    "version": 1,
                    "adaptive_hints": {}
                }
            },
            {
                "kind": "wasm.test.plugin.register",
                "enabled": true,
                "config": {
                    "threshold": 17
                }
            }
        ]
    }))
    .unwrap();

    let report = validate_plugin_config(register_config.clone()).unwrap();
    let report_json: serde_json::Value = serde_wasm_bindgen::from_value(report).unwrap();
    assert_eq!(report_json["diagnostics"], json!([]));

    let runtime_report: serde_json::Value =
        serde_wasm_bindgen::from_value(initialize_plugins(register_config).await.unwrap()).unwrap();
    assert_eq!(runtime_report["diagnostics"], json!([]));

    clear_plugin_configuration().unwrap();
    assert!(deregister_plugin("wasm.test.plugin.register".to_string()));
}

#[wasm_bindgen_test]
fn hosted_plugin_registry_and_duplicate_kind_protection_work() {
    let validate = js_sys::Function::new_with_args("pluginConfig", "return [];");
    let register = js_sys::Function::new_with_args("pluginConfig, context", "return undefined;");

    assert!(!deregister_plugin("wasm.test.plugin.missing".to_string()));

    register_plugin(
        "wasm.test.plugin.duplicate".to_string(),
        Some(validate.clone()),
        register.clone(),
    )
    .unwrap();

    let duplicate = register_plugin(
        "wasm.test.plugin.duplicate".to_string(),
        Some(validate),
        register,
    );
    assert!(duplicate.is_err());

    assert!(deregister_plugin("wasm.test.plugin.duplicate".to_string()));
}
