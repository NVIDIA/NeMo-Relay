// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing generic plugin configuration and registration helpers.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::{collections::HashSet, sync::LazyLock};

use pyo3::prelude::*;
use serde_json::{Map, Value as Json};

use nemo_relay::api::registry::{
    deregister_llm_conditional_execution_guardrail, deregister_llm_execution_intercept,
    deregister_llm_request_intercept, deregister_llm_sanitize_request_guardrail,
    deregister_llm_sanitize_response_guardrail, deregister_llm_stream_execution_intercept,
    deregister_mark_sanitize_guardrail, deregister_scope_sanitize_end_guardrail,
    deregister_scope_sanitize_start_guardrail, deregister_tool_conditional_execution_guardrail,
    deregister_tool_execution_intercept, deregister_tool_request_intercept,
    deregister_tool_sanitize_request_guardrail, deregister_tool_sanitize_response_guardrail,
    register_llm_conditional_execution_guardrail, register_llm_execution_intercept,
    register_llm_request_intercept, register_llm_sanitize_request_guardrail,
    register_llm_sanitize_response_guardrail, register_llm_stream_execution_intercept,
    register_mark_sanitize_guardrail, register_scope_sanitize_end_guardrail,
    register_scope_sanitize_start_guardrail, register_tool_conditional_execution_guardrail,
    register_tool_execution_intercept, register_tool_request_intercept,
    register_tool_sanitize_request_guardrail, register_tool_sanitize_response_guardrail,
};
use nemo_relay::api::subscriber::{deregister_subscriber, register_subscriber};
use nemo_relay::error::Result as FlowResult;
use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginConfig, PluginError, PluginRegistration,
    PluginRegistrationContext, active_plugin_report, clear_plugin_configuration, deregister_plugin,
    initialize_plugins, list_plugin_kinds, register_plugin, rollback_registrations,
    validate_plugin_config,
};

use crate::convert::{json_to_py, py_to_json};
use crate::py_callable::{
    wrap_py_event_sanitize_fn, wrap_py_event_subscriber, wrap_py_llm_conditional_fn,
    wrap_py_llm_exec_intercept_fn, wrap_py_llm_request_intercept_fn,
    wrap_py_llm_sanitize_request_fn, wrap_py_llm_sanitize_response_fn,
    wrap_py_llm_stream_exec_intercept_fn, wrap_py_tool_conditional_fn,
    wrap_py_tool_exec_intercept_fn, wrap_py_tool_fn, wrap_py_tool_request_intercept_fn,
};

#[cfg(test)]
static FORCE_VALIDATE_CONFIG_TO_PY_ERROR: LazyLock<Mutex<HashSet<String>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));
#[cfg(test)]
static FORCE_PLUGIN_CONTEXT_NEW_ERROR: LazyLock<Mutex<HashSet<String>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));
#[cfg(test)]
static PLUGIN_TEST_STATE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[cfg(test)]
enum ForcedPluginTestFlagKind {
    ValidateConfigToPyError,
    PluginContextNewError,
}

#[cfg(test)]
pub(crate) struct ForcedPluginTestFlagGuard {
    kind: ForcedPluginTestFlagKind,
    plugin_kind: String,
}

#[cfg(test)]
impl Drop for ForcedPluginTestFlagGuard {
    fn drop(&mut self) {
        match self.kind {
            ForcedPluginTestFlagKind::ValidateConfigToPyError => {
                if let Ok(mut forced_kinds) = FORCE_VALIDATE_CONFIG_TO_PY_ERROR.lock() {
                    forced_kinds.remove(&self.plugin_kind);
                }
            }
            ForcedPluginTestFlagKind::PluginContextNewError => {
                if let Ok(mut forced_kinds) = FORCE_PLUGIN_CONTEXT_NEW_ERROR.lock() {
                    forced_kinds.remove(&self.plugin_kind);
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn force_validate_config_to_py_error_for_tests(
    plugin_kind: &str,
) -> ForcedPluginTestFlagGuard {
    FORCE_VALIDATE_CONFIG_TO_PY_ERROR
        .lock()
        .expect("forced validate hook mutex poisoned")
        .insert(plugin_kind.to_string());
    ForcedPluginTestFlagGuard {
        kind: ForcedPluginTestFlagKind::ValidateConfigToPyError,
        plugin_kind: plugin_kind.to_string(),
    }
}

#[cfg(test)]
pub(crate) fn force_plugin_context_new_error_for_tests(
    plugin_kind: &str,
) -> ForcedPluginTestFlagGuard {
    FORCE_PLUGIN_CONTEXT_NEW_ERROR
        .lock()
        .expect("forced plugin context hook mutex poisoned")
        .insert(plugin_kind.to_string());
    ForcedPluginTestFlagGuard {
        kind: ForcedPluginTestFlagKind::PluginContextNewError,
        plugin_kind: plugin_kind.to_string(),
    }
}

#[cfg(test)]
pub(crate) fn lock_plugin_test_state_for_tests() -> std::sync::MutexGuard<'static, ()> {
    PLUGIN_TEST_STATE_LOCK
        .lock()
        .expect("plugin test state lock poisoned")
}

fn plugin_config_to_py(
    py: Python<'_>,
    _plugin_kind: &str,
    plugin_config: &Map<String, Json>,
) -> PyResult<Py<PyAny>> {
    #[cfg(test)]
    if FORCE_VALIDATE_CONFIG_TO_PY_ERROR
        .lock()
        .map(|forced_kinds| forced_kinds.contains(_plugin_kind))
        .unwrap_or(false)
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "forced plugin config conversion failure",
        ));
    }

    json_to_py(py, &Json::Object(plugin_config.clone()))
}

fn new_py_plugin_context(
    py: Python<'_>,
    _plugin_kind: &str,
    registrations: Arc<Mutex<Vec<PluginRegistration>>>,
    namespace_prefix: String,
) -> PyResult<Py<PyPluginContext>> {
    #[cfg(test)]
    if FORCE_PLUGIN_CONTEXT_NEW_ERROR
        .lock()
        .map(|forced_kinds| forced_kinds.contains(_plugin_kind))
        .unwrap_or(false)
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "forced plugin context allocation failure",
        ));
    }

    Py::new(
        py,
        PyPluginContext {
            registrations,
            namespace_prefix,
        },
    )
}

pub(crate) fn invoke_python_plugin_register(
    py: Python<'_>,
    plugin_kind: &str,
    register_fn: &Bound<'_, PyAny>,
    plugin_config: &Map<String, Json>,
    namespace_prefix: String,
) -> PyResult<Vec<PluginRegistration>> {
    let py_ctx = new_py_plugin_context(
        py,
        plugin_kind,
        Arc::new(Mutex::new(vec![])),
        namespace_prefix,
    )?;
    let plugin_config_py = plugin_config_to_py(py, plugin_kind, plugin_config)?;
    match register_fn.call1((plugin_config_py, py_ctx.clone_ref(py))) {
        Ok(_) => {
            let py_ctx_ref = py_ctx.bind(py).borrow();
            py_ctx_ref.drain_registrations()
        }
        Err(err) => {
            if let Ok(mut registrations) = py_ctx.bind(py).borrow().drain_registrations() {
                rollback_registrations(&mut registrations);
            }
            Err(err)
        }
    }
}

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

    fn register_callback(
        &self,
        name: &str,
        register: impl FnOnce(&str) -> FlowResult<()>,
        deregister: fn(&str) -> FlowResult<bool>,
        label: &'static str,
    ) -> PyResult<()> {
        let qualified_name = self.qualify_name(name);
        register(&qualified_name).map_err(to_py_err)?;

        let name_owned = qualified_name;
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("plugin context lock poisoned: {e}"))
        })?;
        guard.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister(&name_owned).map(|_| ()).map_err(|e| {
                    PluginError::RegistrationFailed(format!("{label} deregistration failed: {e}"))
                })
            }),
        ));
        Ok(())
    }
}

#[pymethods]
impl PyPluginContext {
    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_mark_sanitize_guardrail(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_mark_sanitize_guardrail(
                    qualified_name,
                    priority,
                    wrap_py_event_sanitize_fn(callback),
                )
            },
            deregister_mark_sanitize_guardrail,
            "mark sanitize guardrail",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_scope_sanitize_start_guardrail(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_scope_sanitize_start_guardrail(
                    qualified_name,
                    priority,
                    wrap_py_event_sanitize_fn(callback),
                )
            },
            deregister_scope_sanitize_start_guardrail,
            "scope start sanitize guardrail",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_scope_sanitize_end_guardrail(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_scope_sanitize_end_guardrail(
                    qualified_name,
                    priority,
                    wrap_py_event_sanitize_fn(callback),
                )
            },
            deregister_scope_sanitize_end_guardrail,
            "scope end sanitize guardrail",
        )
    }

    #[pyo3(
        signature = (name: "str", callback: "object") -> "None",
        text_signature = "(name: str, callback: object) -> None"
    )]
    fn register_subscriber(&self, name: &str, callback: Py<PyAny>) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_subscriber(qualified_name, wrap_py_event_subscriber(callback))
            },
            deregister_subscriber,
            "subscriber",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_tool_sanitize_request_guardrail(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_tool_sanitize_request_guardrail(
                    qualified_name,
                    priority,
                    wrap_py_tool_fn(callback),
                )
            },
            deregister_tool_sanitize_request_guardrail,
            "tool sanitize request guardrail",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_tool_sanitize_response_guardrail(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_tool_sanitize_response_guardrail(
                    qualified_name,
                    priority,
                    wrap_py_tool_fn(callback),
                )
            },
            deregister_tool_sanitize_response_guardrail,
            "tool sanitize response guardrail",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_tool_conditional_execution_guardrail(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_tool_conditional_execution_guardrail(
                    qualified_name,
                    priority,
                    wrap_py_tool_conditional_fn(callback),
                )
            },
            deregister_tool_conditional_execution_guardrail,
            "tool conditional execution guardrail",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_llm_sanitize_request_guardrail(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_llm_sanitize_request_guardrail(
                    qualified_name,
                    priority,
                    wrap_py_llm_sanitize_request_fn(callback),
                )
            },
            deregister_llm_sanitize_request_guardrail,
            "llm sanitize request guardrail",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_llm_sanitize_response_guardrail(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_llm_sanitize_response_guardrail(
                    qualified_name,
                    priority,
                    wrap_py_llm_sanitize_response_fn(callback),
                )
            },
            deregister_llm_sanitize_response_guardrail,
            "llm sanitize response guardrail",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_llm_conditional_execution_guardrail(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_llm_conditional_execution_guardrail(
                    qualified_name,
                    priority,
                    wrap_py_llm_conditional_fn(callback),
                )
            },
            deregister_llm_conditional_execution_guardrail,
            "llm conditional execution guardrail",
        )
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
        self.register_callback(
            name,
            |qualified_name| {
                register_llm_request_intercept(
                    qualified_name,
                    priority,
                    break_chain,
                    wrap_py_llm_request_intercept_fn(callback),
                )
            },
            deregister_llm_request_intercept,
            "llm request intercept",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_llm_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_llm_execution_intercept(
                    qualified_name,
                    priority,
                    wrap_py_llm_exec_intercept_fn(callback),
                )
            },
            deregister_llm_execution_intercept,
            "llm execution intercept",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_llm_stream_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_llm_stream_execution_intercept(
                    qualified_name,
                    priority,
                    wrap_py_llm_stream_exec_intercept_fn(callback),
                )
            },
            deregister_llm_stream_execution_intercept,
            "llm stream execution intercept",
        )
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
        self.register_callback(
            name,
            |qualified_name| {
                register_tool_request_intercept(
                    qualified_name,
                    priority,
                    break_chain,
                    wrap_py_tool_request_intercept_fn(callback),
                )
            },
            deregister_tool_request_intercept,
            "tool request intercept",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_tool_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_tool_execution_intercept(
                    qualified_name,
                    priority,
                    wrap_py_tool_exec_intercept_fn(callback),
                )
            },
            deregister_tool_execution_intercept,
            "tool execution intercept",
        )
    }

    fn __repr__(&self) -> String {
        "<PluginContext>".to_string()
    }
}

struct PyPlugin {
    plugin_kind: String,
    plugin: Py<PyAny>,
}

impl Plugin for PyPlugin {
    fn plugin_kind(&self) -> &str {
        &self.plugin_kind
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        Python::attach(|py| {
            let plugin = self.plugin.bind(py);
            let Ok(method) = plugin.getattr("validate") else {
                return vec![];
            };

            let plugin_config_py = match plugin_config_to_py(py, &self.plugin_kind, plugin_config) {
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
                let register_fn = self.plugin.getattr(py, "register")?.into_bound(py);
                invoke_python_plugin_register(
                    py,
                    &self.plugin_kind,
                    &register_fn,
                    &plugin_config,
                    namespace_prefix,
                )
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
#[pyo3(signature = (plugin_kind: "str", plugin: "object") -> "None", text_signature = "(plugin_kind: str, plugin: object) -> None")]
fn register_plugin_py(plugin_kind: &str, plugin: Py<PyAny>) -> PyResult<()> {
    register_plugin(Arc::new(PyPlugin {
        plugin_kind: plugin_kind.to_string(),
        plugin,
    }))
    .map_err(to_py_err)
}

#[pyfunction(name = "deregister_plugin")]
#[pyo3(signature = (plugin_kind: "str") -> "bool", text_signature = "(plugin_kind: str) -> bool")]
fn deregister_plugin_py(plugin_kind: &str) -> bool {
    deregister_plugin(plugin_kind)
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
        level: DiagnosticLevel::Error,
        code: code.to_string(),
        component: Some(plugin_kind.to_string()),
        field: None,
        message,
    }
}

fn to_py_err(err: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

#[cfg(test)]
#[path = "../tests/coverage/py_plugin_coverage_tests.rs"]
mod coverage_tests;
