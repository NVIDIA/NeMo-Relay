// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing adaptive helpers and runtime wrappers that remain outside the
//! generic plugin system.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use nemo_relay::codec::request::AnnotatedLlmRequest as AnnotatedLLMRequest;
use nemo_relay::codec::response::Usage;
use nemo_relay_adaptive::acg::{
    AgentIdentity, CacheRequestFacts, CacheTelemetryEvent, CacheTelemetryProvider,
};
use nemo_relay_adaptive::context_helpers::set_latency_sensitivity as adaptive_set_latency_sensitivity;
use nemo_relay_adaptive::{AdaptiveConfig, AdaptiveRuntime};
use pyo3::prelude::*;
use uuid::Uuid;

use crate::convert::{json_to_py, opt_py_to_json, py_to_json};
use crate::py_types::{PyAnnotatedLLMRequest, PyScopeHandle};

#[pyclass(name = "AdaptiveRuntime")]
pub struct PyAdaptiveRuntime {
    inner: Arc<tokio::sync::Mutex<Option<PyAdaptiveRuntimeState>>>,
}

enum PyAdaptiveRuntimeState {
    Pending {
        config: AdaptiveConfig,
        report: nemo_relay::plugin::ConfigReport,
    },
    Ready(AdaptiveRuntime),
}

#[pymethods]
impl PyAdaptiveRuntime {
    #[new]
    #[pyo3(signature = (config: "object"), text_signature = "(config: object)")]
    fn new(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let config_json = py_to_json(config)?;
        let config: AdaptiveConfig = serde_json::from_value(config_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let report = validate_adaptive_config_or_err(&config)?;
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(Some(
                PyAdaptiveRuntimeState::Pending { config, report },
            ))),
        })
    }

    #[pyo3(signature = () -> "object", text_signature = "() -> object")]
    fn register<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let state = {
                let mut guard = inner.lock().await;
                guard.take().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("adaptive runtime already shut down")
                })?
            };

            let (result, next_state) = match state {
                PyAdaptiveRuntimeState::Pending { config, report } => {
                    match AdaptiveRuntime::new(config.clone()).await {
                        Ok(mut runtime) => {
                            let result = runtime.register().await.map_err(to_py_err);
                            (result, Some(PyAdaptiveRuntimeState::Ready(runtime)))
                        }
                        Err(err) => (
                            Err(to_py_err(err)),
                            Some(PyAdaptiveRuntimeState::Pending { config, report }),
                        ),
                    }
                }
                PyAdaptiveRuntimeState::Ready(mut runtime) => {
                    let result = runtime.register().await.map_err(to_py_err);
                    (result, Some(PyAdaptiveRuntimeState::Ready(runtime)))
                }
            };

            let mut guard = inner.lock().await;
            *guard = next_state;
            result
        })
    }

    #[pyo3(signature = () -> "None", text_signature = "() -> None")]
    fn deregister(&self) -> PyResult<()> {
        let mut guard = self.inner.try_lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("adaptive runtime already shut down")
        })?;
        match state {
            PyAdaptiveRuntimeState::Pending { .. } => Ok(()),
            PyAdaptiveRuntimeState::Ready(runtime) => runtime.deregister().map_err(to_py_err),
        }
    }

    #[pyo3(signature = () -> "object", text_signature = "() -> object")]
    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let state = {
                let mut guard = inner.lock().await;
                guard.take().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("adaptive runtime already shut down")
                })?
            };
            match state {
                PyAdaptiveRuntimeState::Pending { .. } => Ok(()),
                PyAdaptiveRuntimeState::Ready(runtime) => {
                    runtime.shutdown().await.map_err(to_py_err)
                }
            }
        })
    }

    #[pyo3(signature = () -> "None", text_signature = "() -> None")]
    fn wait_for_idle(&self) -> PyResult<()> {
        let guard = self.inner.try_lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("adaptive runtime already shut down")
        })?;
        match state {
            PyAdaptiveRuntimeState::Pending { .. } => Ok(()),
            PyAdaptiveRuntimeState::Ready(runtime) => {
                runtime.wait_for_idle();
                Ok(())
            }
        }
    }

    #[pyo3(signature = () -> "object", text_signature = "() -> object")]
    fn report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let guard = self.inner.try_lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("adaptive runtime already shut down")
        })?;
        let report = match state {
            PyAdaptiveRuntimeState::Pending { report, .. } => report,
            PyAdaptiveRuntimeState::Ready(runtime) => runtime.report(),
        };
        let report = serde_json::to_value(report)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &report)
    }

    #[pyo3(
        signature = (
            *,
            provider,
            request_id,
            annotated_request,
            agent_id,
            timestamp = None
        ) -> "object",
        text_signature = "($self, *, provider: str, request_id: str, annotated_request: object, agent_id: str, timestamp: str | None = None) -> dict | None"
    )]
    fn build_cache_request_facts(
        &self,
        py: Python<'_>,
        provider: &str,
        request_id: &str,
        annotated_request: &Bound<'_, PyAny>,
        agent_id: &str,
        timestamp: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let _request_id = parse_cache_telemetry_request_id(request_id)?;
        let _timestamp = parse_cache_telemetry_timestamp(timestamp)?;
        let annotated_request = parse_annotated_request(annotated_request)?;
        let guard = self.inner.try_lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("adaptive runtime already shut down")
        })?;
        let Some(facts) = (match state {
            PyAdaptiveRuntimeState::Pending { .. } => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "adaptive runtime must be registered before building cache request facts",
                ));
            }
            PyAdaptiveRuntimeState::Ready(runtime) => {
                runtime.build_cache_request_facts(agent_id, provider, &annotated_request)
            }
        }) else {
            return Ok(py.None());
        };

        let facts = serde_json::to_value(&facts)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &facts)
    }

    #[pyo3(signature = (scope_handle), text_signature = "($self, scope_handle: ScopeHandle) -> None")]
    fn bind_scope(&self, scope_handle: &PyScopeHandle) -> PyResult<()> {
        let mut guard = self.inner.try_lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("adaptive runtime already shut down")
        })?;
        match state {
            PyAdaptiveRuntimeState::Pending { .. } => {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "adaptive runtime must be registered before binding ACG request intercepts",
                ))
            }
            PyAdaptiveRuntimeState::Ready(runtime) => runtime
                .bind_scope(scope_handle.inner.uuid)
                .map_err(to_py_err),
        }
    }

    fn __repr__(&self) -> String {
        "<AdaptiveRuntime>".to_string()
    }
}

fn validate_adaptive_config_or_err(
    config: &AdaptiveConfig,
) -> PyResult<nemo_relay::plugin::ConfigReport> {
    let report = AdaptiveRuntime::validate_config(config);
    if report.has_errors() {
        let joined = report
            .diagnostics
            .iter()
            .filter(|diag| diag.level == nemo_relay::plugin::DiagnosticLevel::Error)
            .map(|diag| diag.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(pyo3::exceptions::PyRuntimeError::new_err(joined));
    }
    Ok(report)
}

fn parse_cache_telemetry_provider(provider: &str) -> PyResult<CacheTelemetryProvider> {
    match provider {
        "anthropic" => Ok(CacheTelemetryProvider::Anthropic),
        "openai" => Ok(CacheTelemetryProvider::OpenAI),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported provider: {other}"
        ))),
    }
}

fn parse_cache_telemetry_request_id(request_id: &str) -> PyResult<Uuid> {
    Uuid::parse_str(request_id).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid request_id UUID: {e}"))
    })
}

fn parse_cache_telemetry_timestamp(timestamp: Option<&str>) -> PyResult<DateTime<Utc>> {
    match timestamp {
        Some(value) => DateTime::parse_from_rfc3339(value)
            .map(|value| value.with_timezone(&Utc))
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid timestamp: {e}"))
            }),
        None => Ok(Utc::now()),
    }
}

fn parse_annotated_request(value: &Bound<'_, PyAny>) -> PyResult<AnnotatedLLMRequest> {
    if let Ok(value) = value.extract::<PyAnnotatedLLMRequest>() {
        return Ok(value.inner);
    }

    let annotated_request = py_to_json(value)?;
    serde_json::from_value(annotated_request).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid annotated_request: {e}"))
    })
}

fn parse_cache_request_facts(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<CacheRequestFacts>> {
    let Some(request_facts) = opt_py_to_json(value)? else {
        return Ok(None);
    };

    serde_json::from_value(request_facts)
        .map(Some)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid request_facts: {e}")))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        *,
        provider,
        request_id,
        usage = None,
        request_facts = None,
        agent_id,
        template_version,
        toolset_hash,
        model_family,
        tenant_scope,
        timestamp = None
    ) -> "object",
    text_signature = "(*, provider: str, request_id: str, usage: dict | None = None, request_facts: object | None = None, agent_id: str, template_version: str, toolset_hash: str, model_family: str, tenant_scope: str, timestamp: str | None = None) -> dict | None"
)]
fn build_cache_telemetry_event(
    py: Python<'_>,
    provider: &str,
    request_id: &str,
    usage: Option<&Bound<'_, PyAny>>,
    request_facts: Option<&Bound<'_, PyAny>>,
    agent_id: &str,
    template_version: &str,
    toolset_hash: &str,
    model_family: &str,
    tenant_scope: &str,
    timestamp: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let provider = parse_cache_telemetry_provider(provider)?;
    let request_id = parse_cache_telemetry_request_id(request_id)?;
    let timestamp = parse_cache_telemetry_timestamp(timestamp)?;
    let Some(usage_json) = opt_py_to_json(usage)? else {
        return Ok(py.None());
    };
    let request_facts = parse_cache_request_facts(request_facts)?;
    let usage: Usage = serde_json::from_value(usage_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid usage: {e}")))?;
    let agent_identity = AgentIdentity {
        agent_id: agent_id.to_string(),
        template_version: template_version.to_string(),
        toolset_hash: toolset_hash.to_string(),
        model_family: model_family.to_string(),
        tenant_scope: tenant_scope.to_string(),
    };

    match CacheTelemetryEvent::from_usage(
        request_id,
        agent_identity,
        provider,
        &usage,
        timestamp,
        request_facts.as_ref(),
    ) {
        Some(event) => {
            let event = serde_json::to_value(&event)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            json_to_py(py, &event)
        }
        None => Ok(py.None()),
    }
}

#[pyfunction]
#[pyo3(signature = (config: "object") -> "object", text_signature = "(config: object) -> object")]
fn validate_adaptive_config(py: Python<'_>, config: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let config_json = py_to_json(config)?;
    let config: AdaptiveConfig = serde_json::from_value(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let report = AdaptiveRuntime::validate_config(&config);
    let report = serde_json::to_value(&report)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    json_to_py(py, &report)
}

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
    m.add_class::<PyAdaptiveRuntime>()?;
    m.add_function(wrap_pyfunction!(build_cache_telemetry_event, m)?)?;
    m.add_function(wrap_pyfunction!(validate_adaptive_config, m)?)?;
    m.add_function(wrap_pyfunction!(set_latency_sensitivity, m)?)?;
    Ok(())
}

fn to_py_err(err: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

#[cfg(test)]
#[path = "../tests/coverage/py_adaptive_coverage_tests.rs"]
mod coverage_tests;
