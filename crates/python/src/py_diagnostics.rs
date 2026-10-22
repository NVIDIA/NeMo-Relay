// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only embedded diagnostics bridge. Copies never retain a core lease.
use nemo_relay::diagnostics::embedded::{self, EmbeddedDiagnosticError, EmbeddedDiagnosticsReader};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde_json::Value as Json;

fn error(py: Python<'_>, error: EmbeddedDiagnosticError) -> PyErr {
    let result = pyo3::exceptions::PyRuntimeError::new_err(error.reason());
    let _ = result.value(py).setattr("reason", error.reason());
    result
}

fn native_numbers(value: &mut Json, field: &str) {
    match value {
        Json::String(text) if embedded::is_diagnostic_number_field(field) => {
            if let Ok(number) = text.parse::<u64>() {
                *value = number.into();
            }
        }
        Json::Object(fields) => {
            for (key, child) in fields {
                native_numbers(child, key);
            }
        }
        Json::Array(values) => {
            for child in values {
                native_numbers(child, "");
            }
        }
        _ => {}
    }
}

/// Authority to inspect the runtime's restricted diagnostic evidence.
#[pyclass(name = "DiagnosticsReader")]
pub struct PyDiagnosticsReader {
    inner: EmbeddedDiagnosticsReader,
}
#[pymethods]
impl PyDiagnosticsReader {
    /// Return canonical JSON; query_json uses canonical snake_case query fields.
    #[pyo3(signature = (query_json = "{}"))]
    fn snapshot_json(&self, py: Python<'_>, query_json: &str) -> PyResult<String> {
        let page = self
            .inner
            .snapshot_json(query_json)
            .map_err(|e| error(py, e))?;
        Ok(page.as_str().to_owned())
    }
    /// Return a caller-owned dict with every diagnostic u64 represented as int.
    #[pyo3(signature = (query_json = "{}"))]
    fn snapshot(&self, py: Python<'_>, query_json: &str) -> PyResult<Py<PyAny>> {
        let page = self
            .inner
            .snapshot_json(query_json)
            .map_err(|e| error(py, e))?;
        let mut value: Json = serde_json::from_str(page.as_str())
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("invalid_snapshot"))?;
        native_numbers(&mut value, "");
        // Keep the adapter permit until Python conversion succeeds or fails.
        crate::convert::json_to_py(py, &value)
    }
    /// Return canonical memory JSON; query_json uses canonical snake_case query fields.
    #[pyo3(signature = (query_json = "{}"))]
    fn inspect_memory_json(&self, py: Python<'_>, query_json: &str) -> PyResult<String> {
        let page = self
            .inner
            .inspect_memory_json(query_json)
            .map_err(|e| error(py, e))?;
        Ok(page.as_str().to_owned())
    }
    /// Return a caller-owned dict with every diagnostic u64 represented as int.
    #[pyo3(signature = (query_json = "{}"))]
    fn inspect_memory(&self, py: Python<'_>, query_json: &str) -> PyResult<Py<PyAny>> {
        let page = self
            .inner
            .inspect_memory_json(query_json)
            .map_err(|e| error(py, e))?;
        let mut value: Json = serde_json::from_str(page.as_str())
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("invalid_snapshot"))?;
        native_numbers(&mut value, "");
        // Keep the adapter permit until Python conversion succeeds or fails.
        crate::convert::json_to_py(py, &value)
    }
    /// Return capacities, occupancy, and explicit logging integration coverage.
    fn info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let info = self.inner.info().map_err(|e| error(py, e))?;
        let mut value = serde_json::to_value(info)
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("invalid_metadata"))?;
        native_numbers(&mut value, "");
        crate::convert::json_to_py(py, &value)
    }
}

/// Obtain a reader without starting diagnostics or selecting default settings.
#[pyfunction]
fn diagnostics_reader(py: Python<'_>) -> PyResult<PyDiagnosticsReader> {
    embedded::reader()
        .map(|inner| PyDiagnosticsReader { inner })
        .map_err(|e| error(py, e))
}

/// Internal descriptor consumed by the pure-Python query adapter.
#[pyfunction(name = "_diagnostics_query_contract")]
fn diagnostics_query_contract() -> String {
    embedded::binding_query_contract_json()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDiagnosticsReader>()?;
    module.add_function(wrap_pyfunction!(diagnostics_reader, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_query_contract, module)?)?;
    Ok(())
}

#[cfg(test)]
#[path = "../tests/coverage/diagnostics_tests.rs"]
mod tests;
