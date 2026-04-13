// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::codecs::{PyAnnotatedLLMRequest, PyAnnotatedLLMResponse};
use super::*;

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

#[pyclass(name = "ScopeStartEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyScopeStartEvent {
    pub inner: ScopeStartEvent,
}

#[pymethods]
impl PyScopeStartEvent {
    #[getter]
    pub(crate) fn kind(&self) -> &'static str {
        "ScopeStart"
    }
    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    pub(crate) fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    pub(crate) fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    pub(crate) fn attributes(&self) -> PyScopeAttributes {
        PyScopeAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    pub(crate) fn scope_type(&self) -> PyScopeType {
        self.inner.scope_type.into()
    }
}

#[pyclass(name = "ScopeEndEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyScopeEndEvent {
    pub inner: ScopeEndEvent,
}

#[pymethods]
impl PyScopeEndEvent {
    #[getter]
    pub(crate) fn kind(&self) -> &'static str {
        "ScopeEnd"
    }
    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    pub(crate) fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    pub(crate) fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    pub(crate) fn attributes(&self) -> PyScopeAttributes {
        PyScopeAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    pub(crate) fn scope_type(&self) -> PyScopeType {
        self.inner.scope_type.into()
    }
}

#[pyclass(name = "ToolStartEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyToolStartEvent {
    pub inner: ToolStartEvent,
}

#[pymethods]
impl PyToolStartEvent {
    #[getter]
    pub(crate) fn kind(&self) -> &'static str {
        "ToolStart"
    }
    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    pub(crate) fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    pub(crate) fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    pub(crate) fn attributes(&self) -> PyToolAttributes {
        PyToolAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    pub(crate) fn input(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.input)
    }

    #[getter]
    pub(crate) fn tool_call_id(&self) -> Option<String> {
        self.inner.tool_call_id.clone()
    }
}

#[pyclass(name = "ToolEndEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyToolEndEvent {
    pub inner: ToolEndEvent,
}

#[pymethods]
impl PyToolEndEvent {
    #[getter]
    pub(crate) fn kind(&self) -> &'static str {
        "ToolEnd"
    }
    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    pub(crate) fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    pub(crate) fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    pub(crate) fn attributes(&self) -> PyToolAttributes {
        PyToolAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    pub(crate) fn output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.output)
    }

    #[getter]
    pub(crate) fn tool_call_id(&self) -> Option<String> {
        self.inner.tool_call_id.clone()
    }
}

#[pyclass(name = "LLMStartEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyLLMStartEvent {
    pub inner: LLMStartEvent,
}

#[pymethods]
impl PyLLMStartEvent {
    #[getter]
    pub(crate) fn kind(&self) -> &'static str {
        "LLMStart"
    }
    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    pub(crate) fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    pub(crate) fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    pub(crate) fn attributes(&self) -> PyLLMAttributes {
        PyLLMAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    pub(crate) fn input(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.input)
    }

    #[getter]
    pub(crate) fn model_name(&self) -> Option<String> {
        self.inner.model_name.clone()
    }

    #[getter]
    pub(crate) fn annotated_request(&self) -> Option<PyAnnotatedLLMRequest> {
        self.inner
            .annotated_request
            .as_ref()
            .map(|a| PyAnnotatedLLMRequest {
                inner: a.as_ref().clone(),
            })
    }
}

#[pyclass(name = "LLMEndEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyLLMEndEvent {
    pub inner: LLMEndEvent,
}

#[pymethods]
impl PyLLMEndEvent {
    #[getter]
    pub(crate) fn kind(&self) -> &'static str {
        "LLMEnd"
    }
    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    pub(crate) fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    pub(crate) fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
    #[getter]
    pub(crate) fn attributes(&self) -> PyLLMAttributes {
        PyLLMAttributes {
            inner: self.inner.attributes,
        }
    }

    #[getter]
    pub(crate) fn output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.output)
    }

    #[getter]
    pub(crate) fn model_name(&self) -> Option<String> {
        self.inner.model_name.clone()
    }

    #[getter]
    pub(crate) fn annotated_response(&self) -> Option<PyAnnotatedLLMResponse> {
        self.inner
            .annotated_response
            .as_ref()
            .map(|a| PyAnnotatedLLMResponse {
                inner: a.as_ref().clone(),
            })
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
        "Mark"
    }
    #[getter]
    pub(crate) fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
    #[getter]
    pub(crate) fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }
    #[getter]
    pub(crate) fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    pub(crate) fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.data)
    }
    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_json_to_py(py, &self.inner.metadata)
    }
}
