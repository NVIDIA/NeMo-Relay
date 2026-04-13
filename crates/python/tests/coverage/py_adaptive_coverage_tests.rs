// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use pyo3::types::PyModule;

#[test]
fn set_latency_sensitivity_rejects_zero_and_registers_binding() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let module = PyModule::new(py, "_adaptive_cov").unwrap();
        register(&module).unwrap();
        assert!(module.getattr("set_latency_sensitivity").is_ok());

        let err = set_latency_sensitivity(0).unwrap_err();
        assert!(err.to_string().contains("sensitivity must be positive"));

        set_latency_sensitivity(3).unwrap();
    });
}
