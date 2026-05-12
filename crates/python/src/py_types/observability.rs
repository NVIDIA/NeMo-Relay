// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use std::path::PathBuf;
use tokio::runtime::{Handle, Runtime};

use super::{
    Bound, Duration, HashMap, PyAny, PyRef, PyResult, Python, json_to_py, py_string_map,
    py_to_json, to_python_json_string, to_python_json_value,
};
#[cfg(test)]
use super::{
    FORCE_ATIF_EXPORT_JSON_SERIALIZATION_ERROR, FORCE_ATIF_EXPORT_VALUE_SERIALIZATION_ERROR,
};

// ---------------------------------------------------------------------------
// AtifExporter
// ---------------------------------------------------------------------------

/// ATIF trajectory exporter that collects events and exports ATIF trajectories.
///
/// Create an exporter, register it as an event subscriber, then call
/// ``export()`` or ``export_json()`` to produce an ATIF trajectory.
///
/// Example:
/// ```python
/// exporter = AtifExporter("session-1", "my-agent", "1.0.0", model_name="gpt-4")
/// exporter.register("atif")
/// # ... run agent ...
/// trajectory = exporter.export()
/// exporter.deregister("atif")
/// ```
#[pyclass(name = "AtifExporter")]
pub struct PyAtifExporter {
    inner: nemo_flow::observability::atif::AtifExporter,
}

#[pymethods]
impl PyAtifExporter {
    #[new]
    #[pyo3(signature = (session_id, agent_name, agent_version, *, model_name=None, tool_definitions=None, extra=None))]
    pub(crate) fn new(
        session_id: String,
        agent_name: String,
        agent_version: String,
        model_name: Option<String>,
        tool_definitions: Option<&Bound<'_, pyo3::types::PyList>>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let tool_defs = match tool_definitions {
            Some(list) => {
                let mut defs = Vec::new();
                for item in list.iter() {
                    defs.push(py_to_json(&item)?);
                }
                Some(defs)
            }
            None => None,
        };
        let extra_json = match extra {
            Some(obj) if !obj.is_none() => Some(py_to_json(obj)?),
            _ => None,
        };
        let agent_info = nemo_flow::observability::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: tool_defs,
            extra: extra_json,
        };
        Ok(Self {
            inner: nemo_flow::observability::atif::AtifExporter::new(session_id, agent_info),
        })
    }

    /// Register this exporter as an event subscriber with the given name.
    pub(crate) fn register(&self, name: String) -> PyResult<()> {
        let subscriber = self.inner.subscriber();
        nemo_flow::api::subscriber::register_subscriber(&name, subscriber)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Deregister the event subscriber with the given name.
    ///
    /// Returns ``True`` if a subscriber with that name was found and removed.
    pub(crate) fn deregister(&self, name: String) -> PyResult<bool> {
        nemo_flow::api::subscriber::deregister_subscriber(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Export the collected events as an ATIF trajectory dict.
    ///
    /// Returns:
    ///     A dict representing the ATIF trajectory.
    pub(crate) fn export(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let trajectory = self.inner.export();
        let value = to_python_json_value(
            &trajectory,
            "Serialization error",
            #[cfg(test)]
            FORCE_ATIF_EXPORT_VALUE_SERIALIZATION_ERROR,
        )?;
        json_to_py(py, &value)
    }

    /// Export the collected events as a JSON string.
    ///
    /// Returns:
    ///     A JSON string representing the ATIF trajectory.
    pub(crate) fn export_json(&self) -> PyResult<String> {
        let trajectory = self.inner.export();
        to_python_json_string(
            &trajectory,
            "Serialization error",
            #[cfg(test)]
            FORCE_ATIF_EXPORT_JSON_SERIALIZATION_ERROR,
        )
    }

    /// Clear all collected events.
    pub(crate) fn clear(&self) {
        self.inner.clear();
    }

    pub(crate) fn __repr__(&self) -> String {
        "<AtifExporter>".to_string()
    }
}

// ---------------------------------------------------------------------------
// ATOF JSONL exporter
// ---------------------------------------------------------------------------

/// File write behavior for ``AtofExporter``.
#[pyclass(name = "AtofExporterMode", eq, eq_int, from_py_object)]
#[derive(Clone, PartialEq)]
pub enum PyAtofExporterMode {
    Append = 0,
    Overwrite = 1,
}

impl From<PyAtofExporterMode> for nemo_flow::observability::atof::AtofExporterMode {
    fn from(value: PyAtofExporterMode) -> Self {
        match value {
            PyAtofExporterMode::Append => Self::Append,
            PyAtofExporterMode::Overwrite => Self::Overwrite,
        }
    }
}

impl From<nemo_flow::observability::atof::AtofExporterMode> for PyAtofExporterMode {
    fn from(value: nemo_flow::observability::atof::AtofExporterMode) -> Self {
        match value {
            nemo_flow::observability::atof::AtofExporterMode::Append => Self::Append,
            nemo_flow::observability::atof::AtofExporterMode::Overwrite => Self::Overwrite,
        }
    }
}

/// Mutable configuration object for the filesystem-backed ATOF JSONL exporter.
#[pyclass(name = "AtofExporterConfig")]
pub struct PyAtofExporterConfig {
    #[pyo3(get, set)]
    pub(crate) output_directory: String,
    #[pyo3(get, set)]
    pub(crate) mode: PyAtofExporterMode,
    #[pyo3(get, set)]
    pub(crate) filename: String,
}

impl PyAtofExporterConfig {
    fn to_rust_config(&self) -> nemo_flow::observability::atof::AtofExporterConfig {
        nemo_flow::observability::atof::AtofExporterConfig::new()
            .with_output_directory(PathBuf::from(self.output_directory.clone()))
            .with_mode(self.mode.clone().into())
            .with_filename(self.filename.clone())
    }
}

#[pymethods]
impl PyAtofExporterConfig {
    #[new]
    pub(crate) fn new() -> Self {
        let config = nemo_flow::observability::atof::AtofExporterConfig::new();
        Self {
            output_directory: config.output_directory.to_string_lossy().into_owned(),
            mode: config.mode.into(),
            filename: config.filename,
        }
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "<AtofExporterConfig output_directory={:?} filename={:?}>",
            self.output_directory, self.filename
        )
    }
}

/// Filesystem-backed ATOF JSONL exporter.
///
/// Register the exporter under a subscriber name, run instrumented application
/// code, then deregister and shut down the exporter to flush output.
#[pyclass(name = "AtofExporter")]
pub struct PyAtofExporter {
    inner: nemo_flow::observability::atof::AtofExporter,
}

#[pymethods]
impl PyAtofExporter {
    #[new]
    pub(crate) fn new(config: PyRef<'_, PyAtofExporterConfig>) -> PyResult<Self> {
        let inner = nemo_flow::observability::atof::AtofExporter::new(config.to_rust_config())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Return the JSONL output path.
    #[getter]
    pub(crate) fn path(&self) -> String {
        self.inner.path().to_string_lossy().into_owned()
    }

    /// Register this exporter globally under ``name``.
    pub(crate) fn register(&self, name: String) -> PyResult<()> {
        self.inner
            .register(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Deregister a global subscriber by name.
    pub(crate) fn deregister(&self, name: String) -> PyResult<bool> {
        self.inner
            .deregister(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Flush the output file.
    pub(crate) fn force_flush(&self) -> PyResult<()> {
        self.inner
            .force_flush()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Shut down the exporter by flushing output.
    pub(crate) fn shutdown(&self) -> PyResult<()> {
        self.inner
            .shutdown()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub(crate) fn __repr__(&self) -> String {
        "<AtofExporter>".to_string()
    }
}

// ---------------------------------------------------------------------------
// OpenTelemetry subscriber
// ---------------------------------------------------------------------------

/// Mutable configuration object for the OpenTelemetry subscriber.
///
/// Create the config, update fields as needed, then pass it to
/// ``OpenTelemetrySubscriber(config)``.
///
/// Example:
/// ```python
/// config = OpenTelemetryConfig()
/// config.endpoint = "http://localhost:4318/v1/traces"
/// config.service_name = "demo-agent"
/// config.headers = {"authorization": "Bearer token"}
/// ```
#[pyclass(name = "OpenTelemetryConfig")]
pub struct PyOpenTelemetryConfig {
    #[pyo3(get, set)]
    pub(crate) transport: String,
    #[pyo3(get, set)]
    pub(crate) endpoint: Option<String>,
    #[pyo3(get, set)]
    pub(crate) service_name: String,
    #[pyo3(get, set)]
    pub(crate) service_namespace: Option<String>,
    #[pyo3(get, set)]
    pub(crate) service_version: Option<String>,
    #[pyo3(get, set)]
    pub(crate) instrumentation_scope: String,
    #[pyo3(get, set)]
    pub(crate) timeout_millis: u64,
    pub(crate) headers: HashMap<String, String>,
    pub(crate) resource_attributes: HashMap<String, String>,
}

impl PyOpenTelemetryConfig {
    pub(crate) fn to_rust_config(
        &self,
    ) -> PyResult<nemo_flow::observability::otel::OpenTelemetryConfig> {
        let mut config = match self.transport.as_str() {
            "http_binary" => nemo_flow::observability::otel::OpenTelemetryConfig::http_binary(
                self.service_name.clone(),
            ),
            "grpc" => {
                nemo_flow::observability::otel::OpenTelemetryConfig::grpc(self.service_name.clone())
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "transport must be 'http_binary' or 'grpc', got {other:?}"
                )));
            }
        }
        .with_instrumentation_scope(self.instrumentation_scope.clone())
        .with_timeout(Duration::from_millis(self.timeout_millis));

        if let Some(endpoint) = &self.endpoint {
            config = config.with_endpoint(endpoint.clone());
        }
        if let Some(namespace) = &self.service_namespace {
            config = config.with_service_namespace(namespace.clone());
        }
        if let Some(version) = &self.service_version {
            config = config.with_service_version(version.clone());
        }
        for (key, value) in &self.headers {
            config = config.with_header(key.clone(), value.clone());
        }
        for (key, value) in &self.resource_attributes {
            config = config.with_resource_attribute(key.clone(), value.clone());
        }
        Ok(config)
    }
}

#[pymethods]
impl PyOpenTelemetryConfig {
    #[new]
    pub(crate) fn new() -> Self {
        Self {
            transport: "http_binary".to_string(),
            endpoint: None,
            service_name: "nemo-flow".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "nemo-flow-otel".to_string(),
            timeout_millis: 3_000,
            headers: HashMap::new(),
            resource_attributes: HashMap::new(),
        }
    }

    #[getter]
    pub(crate) fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &serde_json::to_value(&self.headers).unwrap_or_default())
    }

    #[setter]
    pub(crate) fn set_headers(&mut self, headers: &Bound<'_, PyAny>) -> PyResult<()> {
        self.headers = py_string_map(headers, "headers")?;
        Ok(())
    }

    #[getter]
    pub(crate) fn resource_attributes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &serde_json::to_value(&self.resource_attributes).unwrap_or_default(),
        )
    }

    #[setter]
    pub(crate) fn set_resource_attributes(
        &mut self,
        resource_attributes: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.resource_attributes = py_string_map(resource_attributes, "resource_attributes")?;
        Ok(())
    }

    pub(crate) fn set_header(&mut self, key: String, value: String) {
        self.headers.insert(key, value);
    }

    pub(crate) fn set_resource_attribute(&mut self, key: String, value: String) {
        self.resource_attributes.insert(key, value);
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "<OpenTelemetryConfig transport={:?} endpoint={:?}>",
            self.transport, self.endpoint
        )
    }
}

/// OpenTelemetry-backed event subscriber.
///
/// Construct it from an ``OpenTelemetryConfig``, register it with a subscriber
/// name, then call ``force_flush()`` or ``shutdown()`` when appropriate.
#[pyclass(name = "OpenTelemetrySubscriber")]
pub struct PyOpenTelemetrySubscriber {
    inner: nemo_flow::observability::otel::OpenTelemetrySubscriber,
}

#[pymethods]
impl PyOpenTelemetrySubscriber {
    #[new]
    pub(crate) fn new(config: PyRef<'_, PyOpenTelemetryConfig>) -> PyResult<Self> {
        let inner =
            nemo_flow::observability::otel::OpenTelemetrySubscriber::new(config.to_rust_config()?)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Register this subscriber globally with the given name.
    pub(crate) fn register(&self, name: String) -> PyResult<()> {
        self.inner
            .register(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Deregister a subscriber by name. Returns ``True`` if found.
    pub(crate) fn deregister(&self, name: String) -> PyResult<bool> {
        self.inner
            .deregister(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Force a flush of finished spans through the exporter.
    pub(crate) fn force_flush(&self) -> PyResult<()> {
        self.inner
            .force_flush()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Shut down the underlying tracer provider.
    pub(crate) fn shutdown(&self) -> PyResult<()> {
        self.inner
            .shutdown()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub(crate) fn __repr__(&self) -> String {
        "<OpenTelemetrySubscriber>".to_string()
    }
}

/// Mutable config object for ``OpenInferenceSubscriber``.
///
/// Example:
/// ```python
/// config = OpenInferenceConfig()
/// config.endpoint = "http://localhost:4318/v1/traces"
/// config.service_name = "demo-agent"
/// config.headers = {"authorization": "Bearer token"}
/// ```
#[pyclass(name = "OpenInferenceConfig")]
pub struct PyOpenInferenceConfig {
    #[pyo3(get, set)]
    pub(crate) transport: String,
    #[pyo3(get, set)]
    pub(crate) endpoint: Option<String>,
    #[pyo3(get, set)]
    pub(crate) service_name: String,
    #[pyo3(get, set)]
    pub(crate) service_namespace: Option<String>,
    #[pyo3(get, set)]
    pub(crate) service_version: Option<String>,
    #[pyo3(get, set)]
    pub(crate) instrumentation_scope: String,
    #[pyo3(get, set)]
    pub(crate) timeout_millis: u64,
    pub(crate) headers: HashMap<String, String>,
    pub(crate) resource_attributes: HashMap<String, String>,
}

impl PyOpenInferenceConfig {
    pub(crate) fn to_rust_config(
        &self,
    ) -> PyResult<nemo_flow::observability::openinference::OpenInferenceConfig> {
        let transport = match self.transport.as_str() {
            "http_binary" => nemo_flow::observability::openinference::OtlpTransport::HttpBinary,
            "grpc" => nemo_flow::observability::openinference::OtlpTransport::Grpc,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "transport must be 'http_binary' or 'grpc', got {other:?}"
                )));
            }
        };

        let mut config = nemo_flow::observability::openinference::OpenInferenceConfig::new()
            .with_transport(transport)
            .with_service_name(self.service_name.clone())
            .with_instrumentation_scope(self.instrumentation_scope.clone())
            .with_timeout(Duration::from_millis(self.timeout_millis));

        if let Some(endpoint) = &self.endpoint {
            config = config.with_endpoint(endpoint.clone());
        }
        if let Some(namespace) = &self.service_namespace {
            config = config.with_service_namespace(namespace.clone());
        }
        if let Some(version) = &self.service_version {
            config = config.with_service_version(version.clone());
        }
        for (key, value) in &self.headers {
            config = config.with_header(key.clone(), value.clone());
        }
        for (key, value) in &self.resource_attributes {
            config = config.with_resource_attribute(key.clone(), value.clone());
        }
        Ok(config)
    }
}

#[pymethods]
impl PyOpenInferenceConfig {
    #[new]
    pub(crate) fn new() -> Self {
        Self {
            transport: "http_binary".to_string(),
            endpoint: None,
            service_name: "nemo-flow".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "nemo-flow-openinference".to_string(),
            timeout_millis: 3_000,
            headers: HashMap::new(),
            resource_attributes: HashMap::new(),
        }
    }

    #[getter]
    pub(crate) fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &serde_json::to_value(&self.headers).unwrap_or_default())
    }

    #[setter]
    pub(crate) fn set_headers(&mut self, headers: &Bound<'_, PyAny>) -> PyResult<()> {
        self.headers = py_string_map(headers, "headers")?;
        Ok(())
    }

    #[getter]
    pub(crate) fn resource_attributes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &serde_json::to_value(&self.resource_attributes).unwrap_or_default(),
        )
    }

    #[setter]
    pub(crate) fn set_resource_attributes(
        &mut self,
        resource_attributes: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.resource_attributes = py_string_map(resource_attributes, "resource_attributes")?;
        Ok(())
    }

    pub(crate) fn set_header(&mut self, key: String, value: String) {
        self.headers.insert(key, value);
    }

    pub(crate) fn set_resource_attribute(&mut self, key: String, value: String) {
        self.resource_attributes.insert(key, value);
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "<OpenInferenceConfig transport={:?} endpoint={:?}>",
            self.transport, self.endpoint
        )
    }
}

/// OpenInference-backed event subscriber.
#[pyclass(name = "OpenInferenceSubscriber")]
pub struct PyOpenInferenceSubscriber {
    inner: nemo_flow::observability::openinference::OpenInferenceSubscriber,
    owned_runtime: Option<Runtime>,
}

#[pymethods]
impl PyOpenInferenceSubscriber {
    #[new]
    pub(crate) fn new(config: PyRef<'_, PyOpenInferenceConfig>) -> PyResult<Self> {
        let rust_config = config.to_rust_config()?;
        let needs_owned_runtime = config.transport == "grpc" && Handle::try_current().is_err();

        if needs_owned_runtime {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let _guard = runtime.enter();
            let inner =
                nemo_flow::observability::openinference::OpenInferenceSubscriber::new(rust_config)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                inner,
                owned_runtime: Some(runtime),
            })
        } else {
            let inner =
                nemo_flow::observability::openinference::OpenInferenceSubscriber::new(rust_config)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                inner,
                owned_runtime: None,
            })
        }
    }

    pub(crate) fn register(&self, name: String) -> PyResult<()> {
        self.with_runtime_context(|| {
            self.inner
                .register(&name)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    pub(crate) fn deregister(&self, name: String) -> PyResult<bool> {
        self.with_runtime_context(|| {
            self.inner
                .deregister(&name)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    pub(crate) fn force_flush(&self) -> PyResult<()> {
        self.with_runtime_context(|| {
            self.inner
                .force_flush()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    pub(crate) fn shutdown(&self) -> PyResult<()> {
        self.with_runtime_context(|| {
            self.inner
                .shutdown()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    pub(crate) fn __repr__(&self) -> String {
        "<OpenInferenceSubscriber>".to_string()
    }
}

impl PyOpenInferenceSubscriber {
    fn with_runtime_context<T>(&self, f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
        if let Some(runtime) = &self.owned_runtime {
            let _guard = runtime.enter();
            f()
        } else {
            f()
        }
    }
}
