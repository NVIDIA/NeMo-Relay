// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing optimizer runtime wrappers and hosted plugin registration.

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use serde_json::{Map, Value as Json};

use nvidia_nat_nexus_core::{
    nat_nexus_deregister_llm_execution_intercept, nat_nexus_deregister_llm_request_intercept,
    nat_nexus_deregister_llm_stream_execution_intercept, nat_nexus_deregister_subscriber,
    nat_nexus_deregister_tool_execution_intercept, nat_nexus_deregister_tool_request_intercept,
    nat_nexus_register_llm_execution_intercept, nat_nexus_register_llm_request_intercept,
    nat_nexus_register_llm_stream_execution_intercept, nat_nexus_register_subscriber,
    nat_nexus_register_tool_execution_intercept, nat_nexus_register_tool_request_intercept,
};
use nvidia_nat_nexus_optimizer::{
    deregister_hosted_plugin_handler, register_hosted_plugin_handler, ComponentRegistration,
    ConfigDiagnostic, ConfigReport, DiagnosticLevel, HostedPluginHandler, OptimizerConfig,
    OptimizerRuntime,
};

use crate::convert::{json_to_py, py_to_json};
use crate::py_callable::{
    wrap_py_event_subscriber, wrap_py_llm_exec_intercept_fn, wrap_py_llm_request_intercept_fn,
    wrap_py_llm_stream_exec_intercept_fn, wrap_py_tool_exec_intercept_fn,
    wrap_py_tool_request_intercept_fn,
};

#[pyclass(name = "OptimizerRuntime")]
pub struct PyOptimizerRuntime {
    inner: Arc<tokio::sync::Mutex<Option<PyOptimizerRuntimeState>>>,
}

enum PyOptimizerRuntimeState {
    Pending {
        config: OptimizerConfig,
        report: ConfigReport,
    },
    Ready(OptimizerRuntime),
}

#[pyclass(name = "OptimizerPluginContext")]
pub struct PyOptimizerPluginContext {
    registrations: Arc<Mutex<Vec<ComponentRegistration>>>,
}

impl PyOptimizerPluginContext {
    fn drain_registrations(&self) -> PyResult<Vec<ComponentRegistration>> {
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "optimizer plugin context lock poisoned: {e}"
            ))
        })?;
        Ok(std::mem::take(&mut *guard))
    }
}

#[pymethods]
impl PyOptimizerPluginContext {
    #[pyo3(
        signature = (name: "str", callback: "object") -> "None",
        text_signature = "(name: str, callback: object) -> None"
    )]
    fn register_subscriber(&self, name: &str, callback: Py<PyAny>) -> PyResult<()> {
        nat_nexus_register_subscriber(name, wrap_py_event_subscriber(callback))
            .map_err(to_py_err)?;

        let name_owned = name.to_string();
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "optimizer plugin context lock poisoned: {e}"
            ))
        })?;
        guard.push(ComponentRegistration::new(
            "external_component",
            name_owned.clone(),
            Box::new(move || {
                nat_nexus_deregister_subscriber(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        nvidia_nat_nexus_optimizer::OptimizerError::RegistrationFailed(format!(
                            "subscriber deregistration failed: {e}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    #[pyo3(signature = (
        name: "str",
        priority: "int",
        break_chain: "bool",
        callback: "object"
    ) -> "None", text_signature = "(name: str, priority: int, break_chain: bool, callback: object) -> None")]
    fn register_llm_request_intercept(
        &self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        nat_nexus_register_llm_request_intercept(
            name,
            priority,
            break_chain,
            wrap_py_llm_request_intercept_fn(callback),
        )
        .map_err(to_py_err)?;

        let name_owned = name.to_string();
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "optimizer plugin context lock poisoned: {e}"
            ))
        })?;
        guard.push(ComponentRegistration::new(
            "external_component",
            name_owned.clone(),
            Box::new(move || {
                nat_nexus_deregister_llm_request_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        nvidia_nat_nexus_optimizer::OptimizerError::RegistrationFailed(format!(
                            "llm request intercept deregistration failed: {e}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_llm_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        nat_nexus_register_llm_execution_intercept(
            name,
            priority,
            wrap_py_llm_exec_intercept_fn(callback),
        )
        .map_err(to_py_err)?;

        let name_owned = name.to_string();
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "optimizer plugin context lock poisoned: {e}"
            ))
        })?;
        guard.push(ComponentRegistration::new(
            "external_component",
            name_owned.clone(),
            Box::new(move || {
                nat_nexus_deregister_llm_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        nvidia_nat_nexus_optimizer::OptimizerError::RegistrationFailed(format!(
                            "llm execution intercept deregistration failed: {e}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_llm_stream_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        nat_nexus_register_llm_stream_execution_intercept(
            name,
            priority,
            wrap_py_llm_stream_exec_intercept_fn(callback),
        )
        .map_err(to_py_err)?;

        let name_owned = name.to_string();
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "optimizer plugin context lock poisoned: {e}"
            ))
        })?;
        guard.push(ComponentRegistration::new(
            "external_component",
            name_owned.clone(),
            Box::new(move || {
                nat_nexus_deregister_llm_stream_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        nvidia_nat_nexus_optimizer::OptimizerError::RegistrationFailed(format!(
                            "llm stream execution intercept deregistration failed: {e}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    #[pyo3(signature = (
        name: "str",
        priority: "int",
        break_chain: "bool",
        callback: "object"
    ) -> "None", text_signature = "(name: str, priority: int, break_chain: bool, callback: object) -> None")]
    fn register_tool_request_intercept(
        &self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        nat_nexus_register_tool_request_intercept(
            name,
            priority,
            break_chain,
            wrap_py_tool_request_intercept_fn(callback),
        )
        .map_err(to_py_err)?;

        let name_owned = name.to_string();
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "optimizer plugin context lock poisoned: {e}"
            ))
        })?;
        guard.push(ComponentRegistration::new(
            "external_component",
            name_owned.clone(),
            Box::new(move || {
                nat_nexus_deregister_tool_request_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        nvidia_nat_nexus_optimizer::OptimizerError::RegistrationFailed(format!(
                            "tool request intercept deregistration failed: {e}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_tool_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        nat_nexus_register_tool_execution_intercept(
            name,
            priority,
            wrap_py_tool_exec_intercept_fn(callback),
        )
        .map_err(to_py_err)?;

        let name_owned = name.to_string();
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "optimizer plugin context lock poisoned: {e}"
            ))
        })?;
        guard.push(ComponentRegistration::new(
            "external_component",
            name_owned.clone(),
            Box::new(move || {
                nat_nexus_deregister_tool_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        nvidia_nat_nexus_optimizer::OptimizerError::RegistrationFailed(format!(
                            "tool execution intercept deregistration failed: {e}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    fn __repr__(&self) -> String {
        "<OptimizerPluginContext>".to_string()
    }
}

struct PyHostedPluginHandler {
    plugin_kind: String,
    handler: Py<PyAny>,
}

impl HostedPluginHandler for PyHostedPluginHandler {
    fn plugin_kind(&self) -> &str {
        &self.plugin_kind
    }

    fn validate(
        &self,
        instance_id: &str,
        plugin_config: &Map<String, Json>,
    ) -> Vec<ConfigDiagnostic> {
        Python::attach(|py| {
            let handler = self.handler.bind(py);
            let Ok(method) = handler.getattr("validate") else {
                return vec![];
            };

            let plugin_config_py = match json_to_py(py, &Json::Object(plugin_config.clone())) {
                Ok(value) => value,
                Err(err) => {
                    return vec![plugin_callback_diag(
                        "optimizer.plugin_validate_failed",
                        format!(
                            "hosted plugin '{}' failed to convert config for validate: {err}",
                            self.plugin_kind
                        ),
                    )];
                }
            };

            let result = match method.call1((instance_id, plugin_config_py)) {
                Ok(value) => value,
                Err(err) => {
                    return vec![plugin_callback_diag(
                        "optimizer.plugin_validate_failed",
                        format!(
                            "hosted plugin '{}' validate failed: {err}",
                            self.plugin_kind
                        ),
                    )];
                }
            };

            if result.is_none() {
                return vec![];
            }

            let diagnostics_json = match py_to_json(&result) {
                Ok(value) => value,
                Err(err) => {
                    return vec![plugin_callback_diag(
                        "optimizer.plugin_validate_failed",
                        format!(
                            "hosted plugin '{}' validate returned non-JSON diagnostics: {err}",
                            self.plugin_kind
                        ),
                    )];
                }
            };

            match serde_json::from_value::<Vec<ConfigDiagnostic>>(diagnostics_json) {
                Ok(diagnostics) => diagnostics,
                Err(err) => vec![plugin_callback_diag(
                    "optimizer.plugin_validate_failed",
                    format!(
                        "hosted plugin '{}' validate returned invalid diagnostics: {err}",
                        self.plugin_kind
                    ),
                )],
            }
        })
    }

    fn register(
        &self,
        instance_id: &str,
        plugin_config: &Map<String, Json>,
        ctx: &mut nvidia_nat_nexus_optimizer::HostedRegistrationContext,
    ) -> nvidia_nat_nexus_optimizer::Result<()> {
        let registrations = Python::attach(|py| -> PyResult<Vec<ComponentRegistration>> {
            let py_ctx = Py::new(
                py,
                PyOptimizerPluginContext {
                    registrations: Arc::new(Mutex::new(vec![])),
                },
            )?;
            let plugin_config_py = json_to_py(py, &Json::Object(plugin_config.clone()))?;
            self.handler.call_method1(
                py,
                "register",
                (instance_id, plugin_config_py, py_ctx.clone_ref(py)),
            )?;
            py_ctx.bind(py).borrow().drain_registrations()
        })
        .map_err(|err| {
            nvidia_nat_nexus_optimizer::OptimizerError::RegistrationFailed(err.to_string())
        })?;

        ctx.extend_registrations(registrations);

        Ok(())
    }
}

#[pymethods]
impl PyOptimizerRuntime {
    #[new]
    #[pyo3(signature = (config: "object"), text_signature = "(config: object)")]
    fn new(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let config_json = py_to_json(config)?;
        let config: OptimizerConfig = serde_json::from_value(config_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let report = validate_optimizer_config_or_err(&config)?;
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(Some(
                PyOptimizerRuntimeState::Pending { config, report },
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
                    pyo3::exceptions::PyRuntimeError::new_err("optimizer runtime already shut down")
                })?
            };

            let (result, next_state) = match state {
                PyOptimizerRuntimeState::Pending { config, report } => {
                    match OptimizerRuntime::new(config.clone()).await {
                        Ok(mut runtime) => {
                            let result = runtime.register().await.map_err(to_py_err);
                            (result, Some(PyOptimizerRuntimeState::Ready(runtime)))
                        }
                        Err(err) => (
                            Err(to_py_err(err)),
                            Some(PyOptimizerRuntimeState::Pending { config, report }),
                        ),
                    }
                }
                PyOptimizerRuntimeState::Ready(mut runtime) => {
                    let result = runtime.register().await.map_err(to_py_err);
                    (result, Some(PyOptimizerRuntimeState::Ready(runtime)))
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
                "optimizer runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("optimizer runtime already shut down")
        })?;
        match state {
            PyOptimizerRuntimeState::Pending { .. } => Ok(()),
            PyOptimizerRuntimeState::Ready(runtime) => runtime.deregister().map_err(to_py_err),
        }
    }

    #[pyo3(signature = () -> "object", text_signature = "() -> object")]
    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let state = {
                let mut guard = inner.lock().await;
                guard.take().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("optimizer runtime already shut down")
                })?
            };
            match state {
                PyOptimizerRuntimeState::Pending { .. } => Ok(()),
                PyOptimizerRuntimeState::Ready(runtime) => {
                    runtime.shutdown().await.map_err(to_py_err)
                }
            }
        })
    }

    #[pyo3(signature = () -> "object", text_signature = "() -> object")]
    fn report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let guard = self.inner.try_lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "optimizer runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("optimizer runtime already shut down")
        })?;
        let report = match state {
            PyOptimizerRuntimeState::Pending { report, .. } => report,
            PyOptimizerRuntimeState::Ready(runtime) => runtime.report(),
        };
        let report = serde_json::to_value(report)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &report)
    }

    fn __repr__(&self) -> String {
        "<OptimizerRuntime>".to_string()
    }
}

fn validate_optimizer_config_or_err(config: &OptimizerConfig) -> PyResult<ConfigReport> {
    let report = OptimizerRuntime::validate_config(config);
    if report.has_errors() {
        let joined = report
            .diagnostics
            .iter()
            .filter(|diag| diag.level == DiagnosticLevel::Error)
            .map(|diag| diag.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(pyo3::exceptions::PyRuntimeError::new_err(joined));
    }
    Ok(report)
}

#[pyfunction]
#[pyo3(signature = (config: "object") -> "object", text_signature = "(config: object) -> object")]
fn validate_optimizer_config(py: Python<'_>, config: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let config_json = py_to_json(config)?;
    let config: OptimizerConfig = serde_json::from_value(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let report = nvidia_nat_nexus_optimizer::OptimizerRuntime::validate_config(&config);
    let report = serde_json::to_value(&report)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    json_to_py(py, &report)
}

#[pyfunction]
#[pyo3(signature = (plugin_kind: "str", handler: "object") -> "None", text_signature = "(plugin_kind: str, handler: object) -> None")]
fn register_optimizer_plugin(plugin_kind: &str, handler: Py<PyAny>) -> PyResult<()> {
    register_hosted_plugin_handler(Arc::new(PyHostedPluginHandler {
        plugin_kind: plugin_kind.to_string(),
        handler,
    }))
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (plugin_kind: "str") -> "bool", text_signature = "(plugin_kind: str) -> bool")]
fn deregister_optimizer_plugin(plugin_kind: &str) -> bool {
    deregister_hosted_plugin_handler(plugin_kind)
}

#[pyfunction]
#[pyo3(signature = (value: "int") -> "None", text_signature = "(value: int) -> None")]
fn set_latency_sensitivity(value: u32) -> PyResult<()> {
    if value == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sensitivity must be positive (> 0)",
        ));
    }
    nvidia_nat_nexus_optimizer::set_latency_sensitivity(value).map_err(to_py_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOptimizerRuntime>()?;
    m.add_class::<PyOptimizerPluginContext>()?;
    m.add_function(wrap_pyfunction!(validate_optimizer_config, m)?)?;
    m.add_function(wrap_pyfunction!(register_optimizer_plugin, m)?)?;
    m.add_function(wrap_pyfunction!(deregister_optimizer_plugin, m)?)?;
    m.add_function(wrap_pyfunction!(set_latency_sensitivity, m)?)?;
    Ok(())
}

fn plugin_callback_diag(code: &str, message: String) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level: nvidia_nat_nexus_optimizer::DiagnosticLevel::Error,
        code: code.to_string(),
        component: Some("external_component".to_string()),
        field: None,
        message,
    }
}

fn to_py_err(err: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}
