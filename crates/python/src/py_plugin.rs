// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing generic plugin configuration and registration helpers.

use std::collections::BTreeSet;
#[cfg(test)]
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
#[cfg(test)]
use std::sync::LazyLock;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::PySet;
use serde_json::{Map, Value as Json};

use nemo_relay::api::registry::{
    RuntimeRegistrationKind, deregister_conditional_middleware_guardrail,
    deregister_event_metadata_injector, deregister_llm_conditional_execution_guardrail,
    deregister_llm_execution_intercept, deregister_llm_request_intercept,
    deregister_llm_sanitize_request_guardrail, deregister_llm_sanitize_response_guardrail,
    deregister_llm_stream_execution_intercept, deregister_mark_sanitize_guardrail,
    deregister_scope_sanitize_end_guardrail, deregister_scope_sanitize_start_guardrail,
    deregister_tool_conditional_execution_guardrail, deregister_tool_execution_intercept,
    deregister_tool_request_intercept, deregister_tool_sanitize_request_guardrail,
    deregister_tool_sanitize_response_guardrail, register_conditional_middleware_guardrail,
    register_event_metadata_injector, register_llm_conditional_execution_guardrail,
    register_llm_execution_intercept, register_llm_request_intercept,
    register_llm_sanitize_request_guardrail, register_llm_sanitize_response_guardrail,
    register_llm_stream_execution_intercept, register_mark_sanitize_guardrail,
    register_scope_sanitize_end_guardrail, register_scope_sanitize_start_guardrail,
    register_tool_conditional_execution_guardrail, register_tool_execution_intercept,
    register_tool_request_intercept, register_tool_sanitize_request_guardrail,
    register_tool_sanitize_response_guardrail,
};
use nemo_relay::api::subscriber::{deregister_subscriber, register_subscriber};
use nemo_relay::error::Result as FlowResult;
use nemo_relay::plugin::dynamic::{PluginHostReport, initialize, validate, validate_exact};
use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginConfig, PluginError, PluginHostActivation,
    PluginRegistration, PluginRegistrationContext, deregister_plugin, list_plugin_kinds,
    register_plugin, rollback_registrations,
};

use crate::convert::{json_to_py, py_to_json};
use crate::py_callable::{
    wrap_py_event_metadata_injector_fn, wrap_py_event_sanitize_fn, wrap_py_event_subscriber,
    wrap_py_llm_conditional_fn, wrap_py_llm_exec_intercept_fn, wrap_py_llm_request_intercept_fn,
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
        let forced_kinds = match self.kind {
            ForcedPluginTestFlagKind::ValidateConfigToPyError => &FORCE_VALIDATE_CONFIG_TO_PY_ERROR,
            ForcedPluginTestFlagKind::PluginContextNewError => &FORCE_PLUGIN_CONTEXT_NEW_ERROR,
        };
        if let Ok(mut forced_kinds) = forced_kinds.lock() {
            forced_kinds.remove(&self.plugin_kind);
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

fn runtime_registration_kind(kind: &str) -> PyResult<RuntimeRegistrationKind> {
    serde_json::from_value(Json::String(kind.to_string())).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown runtime registration kind: {kind}"
        ))
    })
}

#[pymethods]
impl PyPluginContext {
    #[pyo3(signature = (
        name: "str",
        kinds: "set[str]",
        registration_name: "str",
        guardrail: "object"
    ) -> "None", text_signature = "(name: str, kinds: set[str], registration_name: str, guardrail: object) -> None")]
    fn register_conditional_middleware_guardrail(
        &self,
        name: &str,
        kinds: BTreeSet<String>,
        registration_name: &str,
        guardrail: Py<PyAny>,
    ) -> PyResult<()> {
        let kinds = kinds
            .iter()
            .map(|kind| runtime_registration_kind(kind))
            .collect::<PyResult<BTreeSet<_>>>()?;
        let registration_name = registration_name.to_string();
        self.register_callback(
            name,
            move |qualified_name| {
                register_conditional_middleware_guardrail(
                    qualified_name,
                    kinds,
                    &registration_name,
                    Arc::new(move |kinds, effective_name| {
                        Python::attach(|py| {
                            let registration_kind = py
                                .import("nemo_relay.runtime_registrations")?
                                .getattr("RuntimeRegistrationKind")?;
                            let python_kinds = PySet::empty(py)?;
                            for kind in kinds {
                                python_kinds.add(registration_kind.call1((kind.as_str(),))?)?;
                            }
                            guardrail
                                .call1(py, (python_kinds, effective_name))?
                                .extract::<Option<String>>(py)
                        })
                        .unwrap_or_else(|error| {
                            Python::attach(|py| error.print(py));
                            None
                        })
                    }),
                )
            },
            deregister_conditional_middleware_guardrail,
            "conditional middleware guardrail",
        )
    }

    #[pyo3(signature = (name: "str", priority: "int", callback: "object") -> "None", text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_event_metadata_injector(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.register_callback(
            name,
            |qualified_name| {
                register_event_metadata_injector(
                    qualified_name,
                    priority,
                    wrap_py_event_metadata_injector_fn(callback),
                )
            },
            deregister_event_metadata_injector,
            "event metadata injector",
        )
    }
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
        let callback = wrap_py_llm_sanitize_request_fn(callback)?;
        self.register_callback(
            name,
            |qualified_name| {
                register_llm_sanitize_request_guardrail(qualified_name, priority, callback)
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
        let callback = wrap_py_llm_sanitize_response_fn(callback)?;
        self.register_callback(
            name,
            |qualified_name| {
                register_llm_sanitize_response_guardrail(qualified_name, priority, callback)
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

/// Owned dynamic plugin host activation.
///
/// The public Python wrapper retains this object until ``close()`` or context
/// manager exit. Dropping it without an explicit close still clears callbacks
/// before unloading plugin code.
#[pyclass(name = "_PluginHostActivation")]
struct PyPluginHostActivation {
    close_state: Arc<PluginHostCloseState>,
}

#[derive(Clone, Copy)]
enum PluginTeardownErrorKind {
    Value,
    NotFound,
    Runtime,
}

#[derive(Clone)]
struct PluginTeardownError {
    kind: PluginTeardownErrorKind,
    message: String,
}

impl PluginTeardownError {
    fn from_plugin_error(error: PluginError) -> Self {
        let message = error.to_string();
        let kind = match error {
            PluginError::InvalidConfig(_) | PluginError::Serialization(_) => {
                PluginTeardownErrorKind::Value
            }
            PluginError::NotFound(_) => PluginTeardownErrorKind::NotFound,
            PluginError::Conflict(_)
            | PluginError::Internal(_)
            | PluginError::RegistrationFailed(_) => PluginTeardownErrorKind::Runtime,
        };
        Self { kind, message }
    }

    fn runtime(message: impl Into<String>) -> Self {
        Self {
            kind: PluginTeardownErrorKind::Runtime,
            message: message.into(),
        }
    }

    fn to_py_err(&self) -> PyErr {
        match self.kind {
            PluginTeardownErrorKind::Value => {
                pyo3::exceptions::PyValueError::new_err(self.message.clone())
            }
            PluginTeardownErrorKind::NotFound => {
                pyo3::exceptions::PyFileNotFoundError::new_err(self.message.clone())
            }
            PluginTeardownErrorKind::Runtime => {
                pyo3::exceptions::PyRuntimeError::new_err(self.message.clone())
            }
        }
    }
}

type PluginTeardownResult = std::result::Result<(), PluginTeardownError>;

struct PluginTeardownCompletion {
    result: tokio::sync::watch::Sender<Option<PluginTeardownResult>>,
}

impl PluginTeardownCompletion {
    fn new() -> Self {
        let (result, _) = tokio::sync::watch::channel(None);
        Self { result }
    }

    fn finish(&self, result: PluginTeardownResult) {
        self.result.send_replace(Some(result));
    }

    fn reset(&self) {
        self.result.send_replace(None);
    }

    async fn wait(&self, operation: &'static str) -> PluginTeardownResult {
        let mut result = self.result.subscribe();
        loop {
            if let Some(result) = result.borrow().clone() {
                return result;
            }
            if result.changed().await.is_err() {
                return Err(PluginTeardownError::runtime(format!(
                    "{operation} result channel closed unexpectedly"
                )));
            }
        }
    }
}

enum PluginHostCloseStatus {
    Active(Option<PluginHostActivation>),
    Closing,
    Closed,
}

struct PluginHostCloseState {
    status: Mutex<PluginHostCloseStatus>,
    report: Mutex<PluginHostReport>,
    completion: PluginTeardownCompletion,
}

impl PluginHostCloseState {
    fn new(activation: PluginHostActivation) -> Self {
        let report = activation.report();
        Self {
            status: Mutex::new(PluginHostCloseStatus::Active(Some(activation))),
            report: Mutex::new(report),
            completion: PluginTeardownCompletion::new(),
        }
    }

    fn report(&self) -> PluginHostReport {
        let latest = {
            let status = self
                .status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match &*status {
                PluginHostCloseStatus::Active(Some(activation)) => Some(activation.report()),
                PluginHostCloseStatus::Active(None)
                | PluginHostCloseStatus::Closing
                | PluginHostCloseStatus::Closed => None,
            }
        };
        let mut report = self
            .report
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(latest) = latest {
            *report = latest;
        }
        report.clone()
    }

    fn is_active(&self) -> bool {
        let status = self
            .status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match &*status {
            PluginHostCloseStatus::Active(activation) => activation
                .as_ref()
                .is_some_and(PluginHostActivation::is_active),
            PluginHostCloseStatus::Closing | PluginHostCloseStatus::Closed => false,
        }
    }

    fn begin_close(self: &Arc<Self>, finalizer: bool) {
        let activation = {
            let mut status = self
                .status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match &mut *status {
                PluginHostCloseStatus::Active(activation) => {
                    if let Some(current) = activation.as_ref() {
                        *self
                            .report
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()) = current.report();
                    }
                    let activation = activation.take();
                    *status = PluginHostCloseStatus::Closing;
                    activation
                }
                PluginHostCloseStatus::Closing | PluginHostCloseStatus::Closed => None,
            }
        };
        let Some(activation) = activation else {
            return;
        };
        self.completion.reset();

        // Keep the activation outside the spawned closure so a thread-spawn
        // failure cannot drop it and synchronously run teardown on the caller.
        let activation = Arc::new(Mutex::new(Some(activation)));
        let worker_activation = Arc::clone(&activation);
        let close_state = Arc::clone(self);
        let spawn = std::thread::Builder::new()
            .name("nemo-relay-python-plugin-teardown".into())
            .spawn(move || {
                let activation = worker_activation
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take();
                let (activation, result) = match activation {
                    Some(mut activation) => {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            activation.close()
                        }))
                        .map_err(|_| {
                            PluginTeardownError::runtime("dynamic plugin teardown task panicked")
                        })
                        .and_then(|result| result.map_err(PluginTeardownError::from_plugin_error));
                        *close_state
                            .report
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()) = activation.report();
                        (Some(activation), result)
                    }
                    None => (
                        None,
                        Err(PluginTeardownError::runtime(
                            "dynamic plugin teardown task lost its activation",
                        )),
                    ),
                };
                close_state.finish(activation, result);
            });

        if let Err(error) = spawn {
            let activation = activation
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            let error = PluginTeardownError::runtime(format!(
                "failed to start dynamic plugin teardown task: {error}"
            ));
            if finalizer {
                // There is no caller left to retry, and teardown must not run
                // synchronously on the Python finalizer thread.
                if let Some(activation) = activation {
                    std::mem::forget(activation);
                }
                self.finish(None, Err(error));
            } else {
                self.finish(activation, Err(error));
            }
        }
    }

    fn finish(&self, activation: Option<PluginHostActivation>, result: PluginTeardownResult) {
        let retryable = result.is_err()
            && activation
                .as_ref()
                .is_some_and(PluginHostActivation::is_active);
        *self
            .status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = if retryable {
            PluginHostCloseStatus::Active(activation)
        } else {
            PluginHostCloseStatus::Closed
        };
        self.completion.finish(result);
    }

    async fn wait_for_close(&self) -> PluginTeardownResult {
        self.completion.wait("dynamic plugin teardown").await
    }
}

#[pymethods]
impl PyPluginHostActivation {
    /// Return the activation report captured during initialization.
    #[getter]
    fn report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let report = serde_json::to_value(self.close_state.report())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        json_to_py(py, &report)
    }

    /// Return whether this activation handle has not begun teardown.
    ///
    /// `False` does not guarantee another process-wide activation can start;
    /// failed teardown may intentionally retain the activation owner.
    #[getter]
    fn is_active(&self) -> PyResult<bool> {
        Ok(self.close_state.is_active())
    }

    /// Clear callbacks and unload the dynamic plugin host.
    #[pyo3(signature = () -> "None", text_signature = "($self) -> None")]
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let close_state = Arc::clone(&self.close_state);
        close_state.begin_close(false);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            close_state
                .wait_for_close()
                .await
                .map_err(|error| error.to_py_err())
        })
    }
}

impl Drop for PyPluginHostActivation {
    fn drop(&mut self) {
        self.close_state.begin_close(true);
    }
}

/// Validate the complete plugin host without activating it.
#[pyfunction(name = "validate")]
#[pyo3(signature = (config: "object", additional_plugins_toml: "str | None" = None) -> "object", text_signature = "(config: object, additional_plugins_toml: str | None = None) -> object")]
fn validate_py(
    py: Python<'_>,
    config: &Bound<'_, PyAny>,
    additional_plugins_toml: Option<String>,
) -> PyResult<Py<PyAny>> {
    let config: PluginConfig = serde_json::from_value(py_to_json(config)?)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let report = py
        .detach(|| validate(config, additional_plugins_toml.map(Into::into)))
        .map_err(plugin_error_to_py_err)?;
    let report = serde_json::to_value(report)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    json_to_py(py, &report)
}

/// Validate only the supplied static plugin configuration.
#[pyfunction(name = "validate_exact")]
#[pyo3(signature = (config: "object") -> "object", text_signature = "(config: object) -> object")]
fn validate_exact_py(py: Python<'_>, config: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let config: PluginConfig = serde_json::from_value(py_to_json(config)?)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let report = serde_json::to_value(validate_exact(config))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    json_to_py(py, &report)
}

/// Initialize the core-owned static and dynamic plugin host.
#[pyfunction(name = "initialize")]
#[pyo3(signature = (config: "object", additional_plugins_toml: "str | None" = None) -> "object", text_signature = "(config: object, additional_plugins_toml: str | None = None) -> object")]
fn initialize_py<'py>(
    py: Python<'py>,
    config: &Bound<'_, PyAny>,
    additional_plugins_toml: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let config: PluginConfig = serde_json::from_value(py_to_json(config)?)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let activation = initialize(config, additional_plugins_toml.map(Into::into))
            .await
            .map_err(plugin_error_to_py_err)?;
        Python::attach(|py| {
            Py::new(
                py,
                PyPluginHostActivation {
                    close_state: Arc::new(PluginHostCloseState::new(activation)),
                },
            )
        })
    })
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
    m.add_class::<PyPluginHostActivation>()?;
    m.add_function(wrap_pyfunction!(initialize_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_exact_py, m)?)?;
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

fn plugin_error_to_py_err(error: PluginError) -> PyErr {
    PluginTeardownError::from_plugin_error(error).to_py_err()
}

#[cfg(test)]
#[path = "../tests/coverage/py_plugin_coverage_tests.rs"]
mod coverage_tests;
