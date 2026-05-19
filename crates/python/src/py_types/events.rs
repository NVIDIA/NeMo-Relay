// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

use super::{
    MarkEvent, PyAnnotatedLLMRequest, PyAnnotatedLLMResponse, PyAny, PyResult, Python, ScopeEvent,
    json_to_py, opt_json_to_py,
};

#[pyclass(name = "ScopeEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyScopeEvent {
    pub inner: ScopeEvent,
}

#[pymethods]
impl PyScopeEvent {
    #[getter]
    pub(crate) fn kind(&self) -> &'static str {
        "scope"
    }

    #[getter]
    pub(crate) fn scope_category(&self) -> &'static str {
        match self.inner.scope_category {
            nemo_flow::api::event::ScopeCategory::Start => "start",
            nemo_flow::api::event::ScopeCategory::End => "end",
        }
    }

    #[getter]
    pub(crate) fn atof_version(&self) -> String {
        self.inner.base.atof_version.clone()
    }

    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.base.parent_uuid.map(|u| u.to_string())
    }

    #[getter]
    pub(crate) fn uuid(&self) -> String {
        self.inner.base.uuid.to_string()
    }

    #[getter]
    pub(crate) fn timestamp(&self) -> String {
        self.inner.base.timestamp.to_rfc3339()
    }

    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.base.name.clone()
    }

    #[getter]
    pub(crate) fn attributes(&self) -> Vec<String> {
        self.inner.attributes.clone()
    }

    #[getter]
    pub(crate) fn category(&self) -> String {
        self.inner.category.as_str().to_string()
    }

    #[getter]
    pub(crate) fn category_profile(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::to_value(&self.inner.category_profile)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &value)
    }

    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.base.data)
    }

    #[getter]
    pub(crate) fn data_schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::to_value(&self.inner.base.data_schema)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &value)
    }

    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.base.metadata)
    }

    #[getter]
    pub(crate) fn annotated_request(&self) -> Option<PyAnnotatedLLMRequest> {
        self.inner
            .category_profile
            .as_ref()
            .and_then(|profile| profile.annotated_request.as_ref())
            .map(|request| PyAnnotatedLLMRequest {
                inner: (**request).clone(),
            })
    }

    #[getter]
    pub(crate) fn annotated_response(&self) -> Option<PyAnnotatedLLMResponse> {
        self.inner
            .category_profile
            .as_ref()
            .and_then(|profile| profile.annotated_response.as_ref())
            .map(|response| PyAnnotatedLLMResponse {
                inner: (**response).clone(),
            })
    }

    /// Return this event as the canonical subscriber JSON dictionary.
    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let event = nemo_flow::api::event::Event::Scope(self.inner.clone());
        let value = event
            .try_to_json_value()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &value)
    }

    /// Return this event as canonical subscriber JSON.
    pub(crate) fn to_json(&self) -> PyResult<String> {
        let event = nemo_flow::api::event::Event::Scope(self.inner.clone());
        event
            .to_json_string()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyclass(name = "MarkEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMarkEvent {
    pub inner: MarkEvent,
}

#[pymethods]
impl PyMarkEvent {
    #[getter]
    pub(crate) fn kind(&self) -> &'static str {
        "mark"
    }

    #[getter]
    pub(crate) fn atof_version(&self) -> String {
        self.inner.base.atof_version.clone()
    }

    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.base.parent_uuid.map(|u| u.to_string())
    }

    #[getter]
    pub(crate) fn uuid(&self) -> String {
        self.inner.base.uuid.to_string()
    }

    #[getter]
    pub(crate) fn timestamp(&self) -> String {
        self.inner.base.timestamp.to_rfc3339()
    }

    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.base.name.clone()
    }

    #[getter]
    pub(crate) fn category(&self) -> Option<String> {
        self.inner
            .category
            .as_ref()
            .map(|category| category.as_str().to_string())
    }

    #[getter]
    pub(crate) fn category_profile(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::to_value(&self.inner.category_profile)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &value)
    }

    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.base.data)
    }

    #[getter]
    pub(crate) fn data_schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::to_value(&self.inner.base.data_schema)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &value)
    }

    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.base.metadata)
    }

    /// Return this event as the canonical subscriber JSON dictionary.
    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let event = nemo_flow::api::event::Event::Mark(self.inner.clone());
        let value = event
            .try_to_json_value()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &value)
    }

    /// Return this event as canonical subscriber JSON.
    pub(crate) fn to_json(&self) -> PyResult<String> {
        let event = nemo_flow::api::event::Event::Mark(self.inner.clone());
        event
            .to_json_string()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
