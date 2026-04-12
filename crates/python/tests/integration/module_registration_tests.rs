// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use _native::{py_adaptive, py_api, py_plugin, py_types};
use nemo_flow::context::global::global_context;
use nemo_flow::context::state::NemoFlowContextState;
use pyo3::prelude::*;
use pyo3::types::PyModule;

fn reset_global() {
    let context = global_context();
    *context.write().unwrap() = NemoFlowContextState::new();
}

#[test]
fn registered_native_module_exposes_scope_stack_api() {
    Python::initialize();
    Python::attach(|py| {
        reset_global();

        let module = PyModule::new(py, "_native_test").unwrap();
        py_types::register(&module).unwrap();
        py_api::register(&module).unwrap();
        py_plugin::register(&module).unwrap();
        py_adaptive::register(&module).unwrap();

        let stack = module
            .getattr("create_scope_stack")
            .unwrap()
            .call0()
            .unwrap();
        module
            .getattr("set_thread_scope_stack")
            .unwrap()
            .call1((stack.clone(),))
            .unwrap();

        let active: bool = module
            .getattr("scope_stack_active")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert!(active);

        assert!(module.getattr("PluginContext").is_ok());
        assert!(module.getattr("set_latency_sensitivity").is_ok());
    });
}
