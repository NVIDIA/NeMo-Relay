// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use std::sync::{Arc, Mutex};

use pyo3::types::PyModule;
use serde_json::json;

#[test]
fn plugin_context_helpers_and_error_conversion_work() {
    Python::initialize();

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
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_plugin_cov").unwrap();
        register(&module).unwrap();

        for name in [
            "PluginContext",
            "validate_plugin_config",
            "initialize_plugins",
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
    });
}
