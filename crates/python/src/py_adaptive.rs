// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing adaptive helpers that remain outside the generic plugin host.

use nemo_flow_adaptive::context_helpers::set_latency_sensitivity as adaptive_set_latency_sensitivity;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (value: "int") -> "None", text_signature = "(value: int) -> None")]
fn set_latency_sensitivity(value: u32) -> PyResult<()> {
    if value == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sensitivity must be positive (> 0)",
        ));
    }
    adaptive_set_latency_sensitivity(value)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_latency_sensitivity, m)?)?;
    Ok(())
}

#[cfg(test)]
#[path = "../tests/coverage/py_adaptive_coverage_tests.rs"]
mod coverage_tests;
