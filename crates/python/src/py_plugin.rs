// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing generic plugin configuration and registration helpers.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use serde_json::{Map, Value as Json};

use nemo_flow_core::{
    ConfigDiagnostic, PluginConfig, PluginError, PluginHandler, PluginRegistration,
    PluginRegistrationContext, active_plugin_report, clear_plugin_configuration,
    deregister_plugin_handler, initialize_plugins, list_plugin_kinds, register_plugin_handler,
    validate_plugin_config,
};
use nemo_flow_core::{
    nemo_flow_deregister_llm_execution_intercept, nemo_flow_deregister_llm_request_intercept,
    nemo_flow_deregister_llm_stream_execution_intercept, nemo_flow_deregister_subscriber,
    nemo_flow_deregister_tool_execution_intercept, nemo_flow_deregister_tool_request_intercept,
    nemo_flow_register_llm_execution_intercept, nemo_flow_register_llm_request_intercept,
    nemo_flow_register_llm_stream_execution_intercept, nemo_flow_register_subscriber,
    nemo_flow_register_tool_execution_intercept, nemo_flow_register_tool_request_intercept,
};

use crate::convert::{json_to_py, py_to_json};
use crate::py_callable::{
    wrap_py_event_subscriber, wrap_py_llm_exec_intercept_fn, wrap_py_llm_request_intercept_fn,
    wrap_py_llm_stream_exec_intercept_fn, wrap_py_tool_exec_intercept_fn,
    wrap_py_tool_request_intercept_fn,
};

#[pyclass(name = "PluginContext")]
pub struct PyPluginContext {
    registrations: Arc<Mutex<Vec<PluginRegistration>>>,
    namespace_prefix: String,
}

impl PyPluginContext {
    fn drain_registrations(&self) -> PyResult<Vec<PluginRegistration>> {
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("plugin context lock poisoned: {e}"))
        })?;
        Ok(std::mem::take(&mut *guard))
    }

    fn qualify_name(&self, name: &str) -> String {
        format!("{}{}", self.namespace_prefix, name)
    }
}

#[pymethods]
impl PyPluginContext {
    #[pyo3(
        signature = (name: "str", callback: "object") -> "None",
        text_signature = "(name: str, callback: object) -> None"
    )]
    fn register_subscriber(&self, name: &str, callback: Py<PyAny>) -> PyResult<()> {
        let qualified_name = self.qualify_name(name);
        nemo_flow_register_subscriber(&qualified_name, wrap_py_event_subscriber(callback))
            .map_err(to_py_err)?;

        let name_owned = qualified_name;
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("plugin context lock poisoned: {e}"))
        })?;
        guard.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_subscriber(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
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
        let qualified_name = self.qualify_name(name);
        nemo_flow_register_llm_request_intercept(
            &qualified_name,
            priority,
            break_chain,
            wrap_py_llm_request_intercept_fn(callback),
        )
        .map_err(to_py_err)?;

        let name_owned = qualified_name;
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("plugin context lock poisoned: {e}"))
        })?;
        guard.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_llm_request_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
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
        let qualified_name = self.qualify_name(name);
        nemo_flow_register_llm_execution_intercept(
            &qualified_name,
            priority,
            wrap_py_llm_exec_intercept_fn(callback),
        )
        .map_err(to_py_err)?;

        let name_owned = qualified_name;
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("plugin context lock poisoned: {e}"))
        })?;
        guard.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_llm_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
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
        let qualified_name = self.qualify_name(name);
        nemo_flow_register_llm_stream_execution_intercept(
            &qualified_name,
            priority,
            wrap_py_llm_stream_exec_intercept_fn(callback),
        )
        .map_err(to_py_err)?;

        let name_owned = qualified_name;
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("plugin context lock poisoned: {e}"))
        })?;
        guard.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_llm_stream_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
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
        let qualified_name = self.qualify_name(name);
        nemo_flow_register_tool_request_intercept(
            &qualified_name,
            priority,
            break_chain,
            wrap_py_tool_request_intercept_fn(callback),
        )
        .map_err(to_py_err)?;

        let name_owned = qualified_name;
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("plugin context lock poisoned: {e}"))
        })?;
        guard.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_tool_request_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
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
        let qualified_name = self.qualify_name(name);
        nemo_flow_register_tool_execution_intercept(
            &qualified_name,
            priority,
            wrap_py_tool_exec_intercept_fn(callback),
        )
        .map_err(to_py_err)?;

        let name_owned = qualified_name;
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("plugin context lock poisoned: {e}"))
        })?;
        guard.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_tool_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "tool execution intercept deregistration failed: {e}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    fn __repr__(&self) -> String {
        "<PluginContext>".to_string()
    }
}

struct PyPluginHandler {
    plugin_kind: String,
    handler: Py<PyAny>,
}

impl PluginHandler for PyPluginHandler {
    fn plugin_kind(&self) -> &str {
        &self.plugin_kind
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        Python::attach(|py| {
            let handler = self.handler.bind(py);
            let Ok(method) = handler.getattr("validate") else {
                return vec![];
            };

            let plugin_config_py = match json_to_py(py, &Json::Object(plugin_config.clone())) {
                Ok(value) => value,
                Err(err) => {
                    return vec![plugin_callback_diag(
                        &self.plugin_kind,
                        "plugin.validate_failed",
                        format!(
                            "plugin '{}' failed to convert config for validate: {err}",
                            self.plugin_kind
                        ),
                    )];
                }
            };

            let result = match method.call1((plugin_config_py,)) {
                Ok(value) => value,
                Err(err) => {
                    return vec![plugin_callback_diag(
                        &self.plugin_kind,
                        "plugin.validate_failed",
                        format!("plugin '{}' validate failed: {err}", self.plugin_kind),
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
                        &self.plugin_kind,
                        "plugin.validate_failed",
                        format!(
                            "plugin '{}' validate returned non-JSON diagnostics: {err}",
                            self.plugin_kind
                        ),
                    )];
                }
            };

            match serde_json::from_value::<Vec<ConfigDiagnostic>>(diagnostics_json) {
                Ok(diagnostics) => diagnostics,
                Err(err) => vec![plugin_callback_diag(
                    &self.plugin_kind,
                    "plugin.validate_failed",
                    format!(
                        "plugin '{}' validate returned invalid diagnostics: {err}",
                        self.plugin_kind
                    ),
                )],
            }
        })
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), PluginError>> + Send + 'a>> {
        let namespace_prefix = ctx.qualify_name("");
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            let registrations = Python::attach(|py| -> PyResult<Vec<PluginRegistration>> {
                let py_ctx = Py::new(
                    py,
                    PyPluginContext {
                        registrations: Arc::new(Mutex::new(vec![])),
                        namespace_prefix,
                    },
                )?;
                let plugin_config_py = json_to_py(py, &Json::Object(plugin_config.clone()))?;
                self.handler.call_method1(
                    py,
                    "register",
                    (plugin_config_py, py_ctx.clone_ref(py)),
                )?;
                {
                    let py_ctx_ref = py_ctx.bind(py).borrow();
                    py_ctx_ref.drain_registrations()
                }
            })
            .map_err(|err| PluginError::RegistrationFailed(err.to_string()))?;

            ctx.extend_registrations(registrations);
            Ok(())
        })
    }
}

#[pyfunction(name = "validate_plugin_config")]
#[pyo3(signature = (config: "object") -> "object", text_signature = "(config: object) -> object")]
fn validate_plugin_config_py(py: Python<'_>, config: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let config_json = py_to_json(config)?;
    let config: PluginConfig = serde_json::from_value(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let report = validate_plugin_config(&config);
    let report = serde_json::to_value(&report)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    json_to_py(py, &report)
}

#[pyfunction(name = "initialize_plugins")]
#[pyo3(signature = (config: "object") -> "object", text_signature = "(config: object) -> object")]
fn initialize_plugins_py<'py>(
    py: Python<'py>,
    config: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let config_json = py_to_json(config)?;
    let config: PluginConfig = serde_json::from_value(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let report = initialize_plugins(config).await.map_err(to_py_err)?;
        Python::attach(|py| {
            let report = serde_json::to_value(&report)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            json_to_py(py, &report)
        })
    })
}

#[pyfunction(name = "clear_plugin_configuration")]
#[pyo3(signature = () -> "None", text_signature = "() -> None")]
fn clear_plugin_configuration_py() -> PyResult<()> {
    clear_plugin_configuration().map_err(to_py_err)
}

#[pyfunction(name = "active_plugin_report")]
#[pyo3(signature = () -> "object", text_signature = "() -> object")]
fn active_plugin_report_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    match active_plugin_report() {
        Some(report) => {
            let report = serde_json::to_value(&report)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            json_to_py(py, &report)
        }
        None => Ok(py.None()),
    }
}

#[pyfunction(name = "list_plugin_kinds")]
#[pyo3(signature = () -> "object", text_signature = "() -> object")]
fn list_plugin_kinds_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let kinds = serde_json::to_value(list_plugin_kinds())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    json_to_py(py, &kinds)
}

#[pyfunction(name = "register_plugin")]
#[pyo3(signature = (plugin_kind: "str", handler: "object") -> "None", text_signature = "(plugin_kind: str, handler: object) -> None")]
fn register_plugin_py(plugin_kind: &str, handler: Py<PyAny>) -> PyResult<()> {
    register_plugin_handler(Arc::new(PyPluginHandler {
        plugin_kind: plugin_kind.to_string(),
        handler,
    }))
    .map_err(to_py_err)
}

#[pyfunction(name = "deregister_plugin")]
#[pyo3(signature = (plugin_kind: "str") -> "bool", text_signature = "(plugin_kind: str) -> bool")]
fn deregister_plugin_py(plugin_kind: &str) -> bool {
    deregister_plugin_handler(plugin_kind)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPluginContext>()?;
    m.add_function(wrap_pyfunction!(validate_plugin_config_py, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_plugins_py, m)?)?;
    m.add_function(wrap_pyfunction!(clear_plugin_configuration_py, m)?)?;
    m.add_function(wrap_pyfunction!(active_plugin_report_py, m)?)?;
    m.add_function(wrap_pyfunction!(list_plugin_kinds_py, m)?)?;
    m.add_function(wrap_pyfunction!(register_plugin_py, m)?)?;
    m.add_function(wrap_pyfunction!(deregister_plugin_py, m)?)?;
    Ok(())
}

fn plugin_callback_diag(plugin_kind: &str, code: &str, message: String) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level: nemo_flow_core::DiagnosticLevel::Error,
        code: code.to_string(),
        component: Some(plugin_kind.to_string()),
        field: None,
        message,
    }
}

fn to_py_err(err: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}
