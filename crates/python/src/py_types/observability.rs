// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use std::path::PathBuf;
use tokio::runtime::{Handle, Runtime};

use super::{
    Bound, Duration, HashMap, PyAny, PyLogSeverity, PyRef, PyResult, Python, json_to_py,
    py_string_map, py_to_json, to_python_json_string, to_python_json_value,
};
#[cfg(test)]
use super::{
    FORCE_ATIF_EXPORT_JSON_SERIALIZATION_ERROR, FORCE_ATIF_EXPORT_VALUE_SERIALIZATION_ERROR,
};

// ---------------------------------------------------------------------------
// AtifExporter
// ---------------------------------------------------------------------------

/// One bounded runtime diagnostic from an OpenTelemetry subscriber.
#[pyclass(name = "OpenTelemetryRuntimeDiagnostic", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyOpenTelemetryRuntimeDiagnostic {
    inner: nemo_relay::observability::OpenTelemetryRuntimeDiagnostic,
}

#[pymethods]
impl PyOpenTelemetryRuntimeDiagnostic {
    #[getter]
    fn code(&self) -> &str {
        &self.inner.code
    }

    #[getter]
    fn message(&self) -> &str {
        &self.inner.message
    }

    #[getter]
    fn count(&self) -> u64 {
        self.inner.count
    }
}

/// Bounded snapshot of runtime diagnostics from an OpenTelemetry subscriber.
#[pyclass(name = "OpenTelemetryRuntimeDiagnostics", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyOpenTelemetryRuntimeDiagnostics {
    inner: nemo_relay::observability::OpenTelemetryRuntimeDiagnostics,
}

#[pymethods]
impl PyOpenTelemetryRuntimeDiagnostics {
    /// Return diagnostics in stable code order.
    #[getter]
    fn entries(&self) -> Vec<PyOpenTelemetryRuntimeDiagnostic> {
        self.inner
            .entries()
            .iter()
            .cloned()
            .map(|inner| PyOpenTelemetryRuntimeDiagnostic { inner })
            .collect()
    }

    /// Return the diagnostic with ``code``, when present.
    fn get(&self, code: &str) -> Option<PyOpenTelemetryRuntimeDiagnostic> {
        self.inner
            .get(code)
            .cloned()
            .map(|inner| PyOpenTelemetryRuntimeDiagnostic { inner })
    }
}

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
    inner: nemo_relay::observability::atif::AtifExporter,
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
        let agent_info = nemo_relay::observability::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: tool_defs,
            extra: extra_json,
        };
        Ok(Self {
            inner: nemo_relay::observability::atif::AtifExporter::new(session_id, agent_info),
        })
    }

    /// Register this exporter as an event subscriber with the given name.
    pub(crate) fn register(&self, name: String) -> PyResult<()> {
        let subscriber = self.inner.subscriber();
        nemo_relay::api::subscriber::register_subscriber(&name, subscriber)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Deregister the event subscriber with the given name.
    ///
    /// Returns ``True`` if a subscriber with that name was found and removed.
    pub(crate) fn deregister(&self, name: String) -> PyResult<bool> {
        nemo_relay::api::subscriber::deregister_subscriber(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Export the collected events as an ATIF trajectory dict.
    ///
    /// Returns:
    ///     A dict representing the ATIF trajectory.
    pub(crate) fn export(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let trajectory = py
            .detach(|| self.inner.try_export())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
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
    pub(crate) fn export_json(&self, py: Python<'_>) -> PyResult<String> {
        let trajectory = py
            .detach(|| self.inner.try_export())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
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

impl From<PyAtofExporterMode> for nemo_relay::observability::atof::AtofExporterMode {
    fn from(value: PyAtofExporterMode) -> Self {
        match value {
            PyAtofExporterMode::Append => Self::Append,
            PyAtofExporterMode::Overwrite => Self::Overwrite,
        }
    }
}

impl From<nemo_relay::observability::atof::AtofExporterMode> for PyAtofExporterMode {
    fn from(value: nemo_relay::observability::atof::AtofExporterMode) -> Self {
        match value {
            nemo_relay::observability::atof::AtofExporterMode::Append => Self::Append,
            nemo_relay::observability::atof::AtofExporterMode::Overwrite => Self::Overwrite,
        }
    }
}

/// Mutable configuration object for an ATOF streaming endpoint.
///
/// Configures a remote endpoint URL, transport (`http_post`, `websocket`, or
/// `ndjson`), optional string headers, and a positive timeout in milliseconds.
#[pyclass(name = "AtofStreamSinkConfig", from_py_object)]
#[derive(Clone)]
pub struct PyAtofEndpointConfig {
    #[pyo3(get, set)]
    pub(crate) url: String,
    #[pyo3(get, set)]
    pub(crate) transport: String,
    #[pyo3(get, set)]
    pub(crate) headers: HashMap<String, String>,
    #[pyo3(get, set)]
    pub(crate) header_env: HashMap<String, String>,
    #[pyo3(get, set)]
    pub(crate) timeout_millis: u64,
    #[pyo3(get, set)]
    pub(crate) field_name_policy: String,
}

impl PyAtofEndpointConfig {
    fn to_rust_config(&self) -> PyResult<nemo_relay::observability::atof::AtofEndpointConfig> {
        let Some(transport) =
            nemo_relay::observability::atof::AtofEndpointTransport::parse(&self.transport)
        else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "endpoint transport must be 'http_post', 'websocket', or 'ndjson'",
            ));
        };
        let mut config =
            nemo_relay::observability::atof::AtofEndpointConfig::new(self.url.clone(), transport)
                .with_timeout_millis(self.timeout_millis);
        let Some(field_name_policy) =
            nemo_relay::observability::atof::AtofEndpointFieldNamePolicy::parse(
                &self.field_name_policy,
            )
        else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "endpoint field_name_policy must be 'preserve' or 'replace_dots'",
            ));
        };
        config = config.with_field_name_policy(field_name_policy);
        for (key, value) in &self.headers {
            config = config.with_header(key.clone(), value.clone());
        }
        for (key, variable) in &self.header_env {
            config = config.with_header_env(key.clone(), variable.clone());
        }
        Ok(config)
    }
}

#[pymethods]
impl PyAtofEndpointConfig {
    #[new]
    #[pyo3(signature = (url, *, transport="http_post".to_string(), headers=None, header_env=None, timeout_millis=3000, field_name_policy="preserve".to_string()))]
    pub(crate) fn new(
        url: String,
        transport: String,
        headers: Option<&Bound<'_, PyAny>>,
        header_env: Option<&Bound<'_, PyAny>>,
        timeout_millis: u64,
        field_name_policy: String,
    ) -> PyResult<Self> {
        let headers = match headers {
            Some(headers) if !headers.is_none() => py_string_map(headers, "headers")?,
            _ => HashMap::new(),
        };
        let header_env = match header_env {
            Some(header_env) if !header_env.is_none() => py_string_map(header_env, "header_env")?,
            _ => HashMap::new(),
        };
        Ok(Self {
            url,
            transport,
            headers,
            header_env,
            timeout_millis,
            field_name_policy,
        })
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "<AtofStreamSinkConfig transport={:?} url={:?}>",
            self.transport, self.url
        )
    }
}

/// One tagged ATOF sink configuration for the manual exporter API.
#[pyclass(name = "AtofExporterConfig")]
pub struct PyAtofExporterConfig {
    #[pyo3(get, set)]
    pub(crate) sink_type: String,
    #[pyo3(get, set)]
    pub(crate) output_directory: String,
    #[pyo3(get, set)]
    pub(crate) mode: PyAtofExporterMode,
    #[pyo3(get, set)]
    pub(crate) filename: String,
    #[pyo3(get, set)]
    pub(crate) url: String,
    #[pyo3(get, set)]
    pub(crate) transport: String,
    #[pyo3(get, set)]
    pub(crate) headers: HashMap<String, String>,
    #[pyo3(get, set)]
    pub(crate) header_env: HashMap<String, String>,
    #[pyo3(get, set)]
    pub(crate) timeout_millis: u64,
    #[pyo3(get, set)]
    pub(crate) field_name_policy: String,
}

impl PyAtofExporterConfig {
    fn to_rust_config(&self) -> PyResult<nemo_relay::observability::atof::AtofExporterConfig> {
        match self.sink_type.as_str() {
            "file" => Ok(nemo_relay::observability::atof::AtofExporterConfig::new()
                .with_output_directory(PathBuf::from(self.output_directory.clone()))
                .with_mode(self.mode.clone().into())
                .with_filename(self.filename.clone())),
            "stream" => {
                if self.url.trim().is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "stream sink requires url",
                    ));
                }
                PyAtofEndpointConfig {
                    url: self.url.clone(),
                    transport: self.transport.clone(),
                    headers: self.headers.clone(),
                    header_env: self.header_env.clone(),
                    timeout_millis: self.timeout_millis,
                    field_name_policy: self.field_name_policy.clone(),
                }
                .to_rust_config()
                .map(|sink| {
                    nemo_relay::observability::atof::AtofExporterConfig::new()
                        .with_stream_sink(sink)
                })
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "sink_type must be 'file' or 'stream'",
            )),
        }
    }
}

#[pymethods]
impl PyAtofExporterConfig {
    #[new]
    pub(crate) fn new() -> Self {
        let config = nemo_relay::observability::atof::AtofFileSinkConfig::new();
        Self {
            sink_type: "file".to_string(),
            output_directory: config.output_directory.to_string_lossy().into_owned(),
            mode: config.mode.into(),
            filename: config.filename,
            url: String::new(),
            transport: "http_post".to_string(),
            headers: HashMap::new(),
            header_env: HashMap::new(),
            timeout_millis: 3000,
            field_name_policy: "preserve".to_string(),
        }
    }

    pub(crate) fn __repr__(&self) -> String {
        format!("<AtofExporterConfig sink_type={:?}>", self.sink_type)
    }
}

/// Single-sink ATOF exporter.
///
/// Register the exporter under a subscriber name, run instrumented application
/// code, then deregister and shut down the exporter to flush output.
#[pyclass(name = "AtofExporter")]
pub struct PyAtofExporter {
    inner: nemo_relay::observability::atof::AtofExporter,
}

#[pymethods]
impl PyAtofExporter {
    #[new]
    pub(crate) fn new(config: PyRef<'_, PyAtofExporterConfig>) -> PyResult<Self> {
        let inner = nemo_relay::observability::atof::AtofExporter::new(config.to_rust_config()?)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Return the JSONL output path, or ``None`` for a stream sink.
    #[getter]
    pub(crate) fn path(&self) -> Option<String> {
        self.inner
            .path()
            .map(|path| path.to_string_lossy().into_owned())
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

    /// Outside subscriber and middleware callbacks, wait for queued subscriber delivery, then
    /// flush the file sink or ask the stream sink to drain for up to its timeout. A stream timeout
    /// is logged and does not by itself return an error.
    pub(crate) fn force_flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.force_flush())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Outside subscriber and middleware callbacks, wait for queued subscriber delivery, then
    /// flush the file sink or ask the stream sink to drain and close up to its timeout. A stream
    /// timeout is logged and does not by itself return an error.
    pub(crate) fn shutdown(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.shutdown())
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
/// config = OpenTelemetryConfig("full", "http://localhost:4318/v1/traces")
/// config.service_name = "demo-agent"
/// ```
#[pyclass(name = "OpenTelemetryConfig")]
pub struct PyOpenTelemetryConfig {
    #[pyo3(get, set, name = "type")]
    pub(crate) otel_type: String,
    #[pyo3(get, set)]
    pub(crate) transport: String,
    #[pyo3(get, set)]
    pub(crate) endpoint: String,
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
    #[pyo3(get, set)]
    pub(crate) completed_span_context_ttl_millis: u64,
    #[pyo3(get, set)]
    pub(crate) mark_projection: String,
    #[pyo3(get, set)]
    pub(crate) mark_exclude_names: Vec<String>,
    pub(crate) headers: HashMap<String, String>,
    pub(crate) header_env: HashMap<String, String>,
    pub(crate) resource_attributes: HashMap<String, String>,
    pub(crate) attribute_mappings: Vec<nemo_relay::observability::OtlpAttributeMapping>,
    #[pyo3(get, set)]
    pub(crate) promote_metadata_prefixes: Vec<String>,
    #[pyo3(get, set)]
    pub(crate) promote_resource_metadata_prefixes: Vec<String>,
}

impl PyOpenTelemetryConfig {
    pub(crate) fn to_rust_config(
        &self,
    ) -> PyResult<nemo_relay::observability::otel::OpenTelemetryConfig> {
        let otel_type = match self.otel_type.as_str() {
            "full" => nemo_relay::observability::OpenTelemetryType::Full,
            "gen_ai" => nemo_relay::observability::OpenTelemetryType::GenAi,
            "openinference" => nemo_relay::observability::OpenTelemetryType::OpenInference,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "type must be 'full', 'gen_ai', or 'openinference', got {other:?}"
                )));
            }
        };
        validate_otel_signal_endpoint(&self.endpoint)?;
        let transport = parse_otel_signal_transport(&self.transport)?;
        let mut config = nemo_relay::observability::otel::OpenTelemetryConfig::new(
            otel_type,
            self.endpoint.clone(),
        )
        .with_transport(transport)
        .with_service_name(self.service_name.clone())
        .with_instrumentation_scope(self.instrumentation_scope.clone())
        .with_timeout(Duration::from_millis(self.timeout_millis))
        .with_completed_span_context_ttl(Duration::from_millis(
            self.completed_span_context_ttl_millis,
        ));

        if let Some(namespace) = &self.service_namespace {
            config = config.with_service_namespace(namespace.clone());
        }
        if let Some(version) = &self.service_version {
            config = config.with_service_version(version.clone());
        }
        for (key, value) in &self.headers {
            config = config.with_header(key.clone(), value.clone());
        }
        for (key, variable) in &self.header_env {
            config = config.with_header_env(key.clone(), variable.clone());
        }
        for (key, value) in &self.resource_attributes {
            config = config.with_resource_attribute(key.clone(), value.clone());
        }
        let mark_projection =
            serde_json::from_value(serde_json::Value::String(self.mark_projection.clone()))
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        nemo_relay::observability::validate_attribute_mappings(&self.attribute_mappings)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        nemo_relay::observability::validate_metadata_promotion_prefixes(
            &self.promote_metadata_prefixes,
        )
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        nemo_relay::observability::validate_metadata_promotion_prefixes(
            &self.promote_resource_metadata_prefixes,
        )
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(config
            .with_mark_projection(mark_projection)
            .with_mark_exclude_names(self.mark_exclude_names.clone())
            .with_attribute_mappings(self.attribute_mappings.clone())
            .with_promote_metadata_prefixes(self.promote_metadata_prefixes.clone())
            .with_promote_resource_metadata_prefixes(
                self.promote_resource_metadata_prefixes.clone(),
            ))
    }
}

#[pymethods]
impl PyOpenTelemetryConfig {
    #[new]
    pub(crate) fn new(otel_type: String, endpoint: String) -> Self {
        Self {
            otel_type,
            transport: "http_binary".to_string(),
            endpoint,
            service_name: "unknown_service".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "opentelemetry".to_string(),
            timeout_millis: 3_000,
            completed_span_context_ttl_millis: 60_000,
            mark_projection: "inherit".to_string(),
            mark_exclude_names: nemo_relay::observability::default_mark_exclude_names(),
            headers: HashMap::new(),
            header_env: HashMap::new(),
            resource_attributes: HashMap::new(),
            attribute_mappings: Vec::new(),
            promote_metadata_prefixes: Vec::new(),
            promote_resource_metadata_prefixes: Vec::new(),
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
    pub(crate) fn header_env(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &serde_json::to_value(&self.header_env).unwrap_or_default(),
        )
    }

    #[setter]
    pub(crate) fn set_header_env(&mut self, header_env: &Bound<'_, PyAny>) -> PyResult<()> {
        self.header_env = py_string_map(header_env, "header_env")?;
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

    #[getter]
    pub(crate) fn attribute_mappings(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &serde_json::to_value(&self.attribute_mappings).unwrap_or_default(),
        )
    }

    #[setter]
    pub(crate) fn set_attribute_mappings(
        &mut self,
        attribute_mappings: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.attribute_mappings =
            serde_json::from_value(py_to_json(attribute_mappings)?).map_err(|error| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "attribute_mappings must be a list of mappings: {error}"
                ))
            })?;
        nemo_relay::observability::validate_attribute_mappings(&self.attribute_mappings)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(())
    }

    pub(crate) fn set_header(&mut self, key: String, value: String) {
        self.headers.insert(key, value);
    }

    pub(crate) fn set_header_from_env(&mut self, key: String, variable: String) {
        self.header_env.insert(key, variable);
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
    inner: nemo_relay::observability::otel::OpenTelemetrySubscriber,
    owned_runtime: Option<Runtime>,
}

#[pymethods]
impl PyOpenTelemetrySubscriber {
    #[new]
    pub(crate) fn new(config: PyRef<'_, PyOpenTelemetryConfig>) -> PyResult<Self> {
        let rust_config = config.to_rust_config()?;
        let needs_owned_runtime = config.transport == "grpc" && Handle::try_current().is_err();
        if needs_owned_runtime {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let _guard = runtime.enter();
            let inner = nemo_relay::observability::otel::OpenTelemetrySubscriber::new(rust_config)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                inner,
                owned_runtime: Some(runtime),
            })
        } else {
            let inner = nemo_relay::observability::otel::OpenTelemetrySubscriber::new(rust_config)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self {
                inner,
                owned_runtime: None,
            })
        }
    }

    /// Register this subscriber globally with the given name.
    pub(crate) fn register(&self, name: String) -> PyResult<()> {
        self.with_runtime_context(|| {
            self.inner
                .register(&name)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Deregister a subscriber by name. Returns ``True`` if found.
    pub(crate) fn deregister(&self, name: String) -> PyResult<bool> {
        self.with_runtime_context(|| {
            self.inner
                .deregister(&name)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Force a flush of finished spans through the exporter.
    ///
    /// A successful flush updates ``runtime_diagnostics()`` with queue drops observed so far.
    pub(crate) fn force_flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.with_runtime_context(|| {
                self.inner
                    .force_flush()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            })
        })
    }

    /// Shut down the underlying tracer provider.
    pub(crate) fn shutdown(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.with_runtime_context(|| {
                self.inner
                    .shutdown()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            })
        })
    }

    /// Return a bounded snapshot of exporter and event-processing diagnostics.
    pub(crate) fn runtime_diagnostics(&self) -> PyResult<PyOpenTelemetryRuntimeDiagnostics> {
        self.with_runtime_context(|| {
            Ok(PyOpenTelemetryRuntimeDiagnostics {
                inner: self.inner.runtime_diagnostics(),
            })
        })
    }

    pub(crate) fn __repr__(&self) -> String {
        "<OpenTelemetrySubscriber>".to_string()
    }
}

fn parse_otel_signal_transport(
    transport: &str,
) -> PyResult<nemo_relay::observability::otel::OtlpTransport> {
    match transport {
        "http_binary" => Ok(nemo_relay::observability::otel::OtlpTransport::HttpBinary),
        "grpc" => Ok(nemo_relay::observability::otel::OtlpTransport::Grpc),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "transport must be 'http_binary' or 'grpc', got {other:?}"
        ))),
    }
}

fn validate_otel_signal_endpoint(endpoint: &str) -> PyResult<()> {
    if endpoint.trim().is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "endpoint is required and must be nonblank",
        ));
    }
    Ok(())
}

/// Mutable configuration for an OTLP log subscriber.
#[pyclass(name = "OpenTelemetryLogConfig")]
pub struct PyOpenTelemetryLogConfig {
    #[pyo3(get, set)]
    pub(crate) transport: String,
    #[pyo3(get, set)]
    pub(crate) endpoint: String,
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
    #[pyo3(get, set)]
    pub(crate) minimum_severity: PyLogSeverity,
    #[pyo3(get, set)]
    pub(crate) max_queue_size: usize,
    #[pyo3(get, set)]
    pub(crate) max_export_batch_size: usize,
    #[pyo3(get, set)]
    pub(crate) scheduled_delay_millis: u64,
    #[pyo3(get, set)]
    pub(crate) completed_span_context_ttl_millis: u64,
    pub(crate) headers: HashMap<String, String>,
    pub(crate) header_env: HashMap<String, String>,
    pub(crate) resource_attributes: HashMap<String, String>,
}

impl PyOpenTelemetryLogConfig {
    fn to_rust_config(
        &self,
    ) -> PyResult<nemo_relay::observability::otel_logs::OpenTelemetryLogConfig> {
        validate_otel_signal_endpoint(&self.endpoint)?;
        let mut config = nemo_relay::observability::otel_logs::OpenTelemetryLogConfig::new(
            self.endpoint.clone(),
        )
        .with_transport(parse_otel_signal_transport(&self.transport)?)
        .with_service_name(self.service_name.clone())
        .with_instrumentation_scope(self.instrumentation_scope.clone())
        .with_timeout(Duration::from_millis(self.timeout_millis))
        .with_minimum_severity(self.minimum_severity.into())
        .with_max_queue_size(self.max_queue_size)
        .with_max_export_batch_size(self.max_export_batch_size)
        .with_scheduled_delay(Duration::from_millis(self.scheduled_delay_millis))
        .with_completed_span_context_ttl(Duration::from_millis(
            self.completed_span_context_ttl_millis,
        ));
        if let Some(namespace) = &self.service_namespace {
            config = config.with_service_namespace(namespace.clone());
        }
        if let Some(version) = &self.service_version {
            config = config.with_service_version(version.clone());
        }
        for (key, value) in &self.headers {
            config = config.with_header(key.clone(), value.clone());
        }
        for (key, variable) in &self.header_env {
            config = config.with_header_env(key.clone(), variable.clone());
        }
        for (key, value) in &self.resource_attributes {
            config = config.with_resource_attribute(key.clone(), value.clone());
        }
        Ok(config)
    }
}

#[pymethods]
impl PyOpenTelemetryLogConfig {
    #[new]
    fn new(endpoint: String) -> Self {
        Self {
            transport: "http_binary".into(),
            endpoint,
            service_name: "unknown_service".into(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "opentelemetry".into(),
            timeout_millis: 3_000,
            minimum_severity: PyLogSeverity::Info,
            max_queue_size: 2_048,
            max_export_batch_size: 512,
            scheduled_delay_millis: 1_000,
            completed_span_context_ttl_millis: 60_000,
            headers: HashMap::new(),
            header_env: HashMap::new(),
            resource_attributes: HashMap::new(),
        }
    }

    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &serde_json::to_value(&self.headers).unwrap_or_default())
    }

    #[setter]
    fn set_headers(&mut self, headers: &Bound<'_, PyAny>) -> PyResult<()> {
        self.headers = py_string_map(headers, "headers")?;
        Ok(())
    }

    #[getter]
    fn header_env(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &serde_json::to_value(&self.header_env).unwrap_or_default(),
        )
    }

    #[setter]
    fn set_header_env(&mut self, header_env: &Bound<'_, PyAny>) -> PyResult<()> {
        self.header_env = py_string_map(header_env, "header_env")?;
        Ok(())
    }

    #[getter]
    fn resource_attributes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &serde_json::to_value(&self.resource_attributes).unwrap_or_default(),
        )
    }

    #[setter]
    fn set_resource_attributes(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.resource_attributes = py_string_map(value, "resource_attributes")?;
        Ok(())
    }

    fn set_header(&mut self, key: String, value: String) {
        self.headers.insert(key, value);
    }

    fn set_header_from_env(&mut self, key: String, variable: String) {
        self.header_env.insert(key, variable);
    }

    fn set_resource_attribute(&mut self, key: String, value: String) {
        self.resource_attributes.insert(key, value);
    }

    fn __repr__(&self) -> String {
        format!(
            "<OpenTelemetryLogConfig transport={:?} endpoint={:?}>",
            self.transport, self.endpoint
        )
    }
}

/// OTLP log-backed Relay event subscriber.
#[pyclass(name = "OpenTelemetryLogSubscriber")]
pub struct PyOpenTelemetryLogSubscriber {
    inner: nemo_relay::observability::otel_logs::OpenTelemetryLogSubscriber,
}

#[pymethods]
impl PyOpenTelemetryLogSubscriber {
    #[new]
    fn new(config: PyRef<'_, PyOpenTelemetryLogConfig>) -> PyResult<Self> {
        let inner = nemo_relay::observability::otel_logs::OpenTelemetryLogSubscriber::new(
            config.to_rust_config()?,
        )
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    fn register(&self, name: String) -> PyResult<()> {
        self.inner
            .register(&name)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    fn deregister(&self, name: String) -> PyResult<bool> {
        self.inner
            .deregister(&name)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// Flush queued Relay events and the OTLP log processor.
    ///
    /// A successful flush updates ``runtime_diagnostics()`` with queue drops observed so far.
    fn force_flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.force_flush())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    fn shutdown(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.shutdown())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// Return a bounded snapshot of exporter and event-processing diagnostics.
    fn runtime_diagnostics(&self) -> PyOpenTelemetryRuntimeDiagnostics {
        PyOpenTelemetryRuntimeDiagnostics {
            inner: self.inner.runtime_diagnostics(),
        }
    }

    fn __repr__(&self) -> &'static str {
        "<OpenTelemetryLogSubscriber>"
    }
}

/// Preferred aggregation temporality for OTLP metrics.
#[pyclass(name = "MetricTemporality", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyMetricTemporality {
    Cumulative = 0,
    Delta = 1,
    LowMemory = 2,
}

impl From<PyMetricTemporality> for nemo_relay::observability::otel_metrics::MetricTemporality {
    fn from(value: PyMetricTemporality) -> Self {
        match value {
            PyMetricTemporality::Cumulative => Self::Cumulative,
            PyMetricTemporality::Delta => Self::Delta,
            PyMetricTemporality::LowMemory => Self::LowMemory,
        }
    }
}

/// Mutable configuration for an OTLP metric subscriber.
#[pyclass(name = "OpenTelemetryMetricConfig")]
pub struct PyOpenTelemetryMetricConfig {
    #[pyo3(get, set)]
    pub(crate) transport: String,
    #[pyo3(get, set)]
    pub(crate) endpoint: String,
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
    #[pyo3(get, set)]
    pub(crate) export_interval_millis: u64,
    #[pyo3(get, set)]
    pub(crate) temporality: PyMetricTemporality,
    #[pyo3(get, set)]
    pub(crate) max_instruments: usize,
    #[pyo3(get, set)]
    pub(crate) cardinality_limit: usize,
    pub(crate) headers: HashMap<String, String>,
    pub(crate) header_env: HashMap<String, String>,
    pub(crate) resource_attributes: HashMap<String, String>,
}

impl PyOpenTelemetryMetricConfig {
    fn to_rust_config(
        &self,
    ) -> PyResult<nemo_relay::observability::otel_metrics::OpenTelemetryMetricConfig> {
        validate_otel_signal_endpoint(&self.endpoint)?;
        let mut config = nemo_relay::observability::otel_metrics::OpenTelemetryMetricConfig::new(
            self.endpoint.clone(),
        )
        .with_transport(parse_otel_signal_transport(&self.transport)?)
        .with_service_name(self.service_name.clone())
        .with_instrumentation_scope(self.instrumentation_scope.clone())
        .with_timeout(Duration::from_millis(self.timeout_millis))
        .with_export_interval(Duration::from_millis(self.export_interval_millis))
        .with_temporality(self.temporality.into())
        .with_max_instruments(self.max_instruments)
        .with_cardinality_limit(self.cardinality_limit);
        if let Some(namespace) = &self.service_namespace {
            config = config.with_service_namespace(namespace.clone());
        }
        if let Some(version) = &self.service_version {
            config = config.with_service_version(version.clone());
        }
        for (key, value) in &self.headers {
            config = config.with_header(key.clone(), value.clone());
        }
        for (key, variable) in &self.header_env {
            config = config.with_header_env(key.clone(), variable.clone());
        }
        for (key, value) in &self.resource_attributes {
            config = config.with_resource_attribute(key.clone(), value.clone());
        }
        Ok(config)
    }
}

#[pymethods]
impl PyOpenTelemetryMetricConfig {
    #[new]
    fn new(endpoint: String) -> Self {
        Self {
            transport: "http_binary".into(),
            endpoint,
            service_name: "unknown_service".into(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "opentelemetry".into(),
            timeout_millis: 3_000,
            export_interval_millis: 60_000,
            temporality: PyMetricTemporality::Cumulative,
            max_instruments: 256,
            cardinality_limit: 2_000,
            headers: HashMap::new(),
            header_env: HashMap::new(),
            resource_attributes: HashMap::new(),
        }
    }

    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &serde_json::to_value(&self.headers).unwrap_or_default())
    }

    #[setter]
    fn set_headers(&mut self, headers: &Bound<'_, PyAny>) -> PyResult<()> {
        self.headers = py_string_map(headers, "headers")?;
        Ok(())
    }

    #[getter]
    fn header_env(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &serde_json::to_value(&self.header_env).unwrap_or_default(),
        )
    }

    #[setter]
    fn set_header_env(&mut self, header_env: &Bound<'_, PyAny>) -> PyResult<()> {
        self.header_env = py_string_map(header_env, "header_env")?;
        Ok(())
    }

    #[getter]
    fn resource_attributes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &serde_json::to_value(&self.resource_attributes).unwrap_or_default(),
        )
    }

    #[setter]
    fn set_resource_attributes(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.resource_attributes = py_string_map(value, "resource_attributes")?;
        Ok(())
    }

    fn set_header(&mut self, key: String, value: String) {
        self.headers.insert(key, value);
    }

    fn set_header_from_env(&mut self, key: String, variable: String) {
        self.header_env.insert(key, variable);
    }

    fn set_resource_attribute(&mut self, key: String, value: String) {
        self.resource_attributes.insert(key, value);
    }

    fn __repr__(&self) -> String {
        format!(
            "<OpenTelemetryMetricConfig transport={:?} endpoint={:?}>",
            self.transport, self.endpoint
        )
    }
}

/// OTLP metric-backed Relay event subscriber.
#[pyclass(name = "OpenTelemetryMetricSubscriber")]
pub struct PyOpenTelemetryMetricSubscriber {
    inner: nemo_relay::observability::otel_metrics::OpenTelemetryMetricSubscriber,
}

#[pymethods]
impl PyOpenTelemetryMetricSubscriber {
    #[new]
    fn new(config: PyRef<'_, PyOpenTelemetryMetricConfig>) -> PyResult<Self> {
        let inner = nemo_relay::observability::otel_metrics::OpenTelemetryMetricSubscriber::new(
            config.to_rust_config()?,
        )
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    fn register(&self, name: String) -> PyResult<()> {
        self.inner
            .register(&name)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    fn deregister(&self, name: String) -> PyResult<bool> {
        self.inner
            .deregister(&name)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    fn force_flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.force_flush())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    fn shutdown(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.shutdown())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// Return a bounded snapshot of exporter and event-processing diagnostics.
    fn runtime_diagnostics(&self) -> PyOpenTelemetryRuntimeDiagnostics {
        PyOpenTelemetryRuntimeDiagnostics {
            inner: self.inner.runtime_diagnostics(),
        }
    }

    fn __repr__(&self) -> &'static str {
        "<OpenTelemetryMetricSubscriber>"
    }
}

impl PyOpenTelemetrySubscriber {
    fn with_runtime_context<T>(&self, f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
        if let Some(runtime) = &self.owned_runtime {
            let _guard = runtime.enter();
            f()
        } else {
            f()
        }
    }
}
