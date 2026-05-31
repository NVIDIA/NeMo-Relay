// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 native extension module for NeMo Relay.
//!
//! This crate compiles to a `_native` Python C extension that is imported by the
//! `nemo_relay` Python package. It exposes all core runtime types and API functions
//! to Python via PyO3.
//!
//! ## Modules
//!
//! - `py_types` — Python-facing type wrappers (`ScopeHandle`, `ToolHandle`, `Event`,
//!   `AtifExporter`, etc.). `Event` exposes typed lifecycle fields (`input`, `output`,
//!   `model_name`, `tool_call_id`). `AtifExporter` collects events and
//!   exports ATIF v1.7 trajectories.
//! - `py_api` — Python-facing API functions (`push_scope`, etc.). Tool calls
//!   accept `tool_call_id` and LLM calls accept `model_name` for ATIF correlation.
//! - `py_callable` — Bridges between Python callables and Rust callback types
//! - `py_context` — Notes on scope propagation between sync/async contexts
//! - `py_adaptive` — Python-facing adaptive helpers (`set_latency_sensitivity`)
//! - `py_plugin` — Python-facing generic plugin config/registration helpers
//! - `convert` — JSON ↔ Python conversion utilities
use nemo_relay::plugin::{PluginRegistrationContext, Result as PluginResult};
use nemo_relay::plugins::nemo_guardrails::component::{
    NeMoGuardrailsConfig, register_local_backend_provider,
};
use nemo_relay::shared_runtime::initialize_shared_runtime_binding;
use nemo_relay_adaptive::plugin_component::register_adaptive_component;
use pyo3::prelude::*;
use serde_json::Value as Json;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod convert;
#[doc(hidden)]
pub mod py_adaptive;
#[doc(hidden)]
pub mod py_api;
mod py_callable;
mod py_context;
#[doc(hidden)]
pub mod py_plugin;
mod py_storage;
#[doc(hidden)]
pub mod py_types;
#[cfg(test)]
mod test_support;

/// The `_native` PyO3 module entry point. Registers all types and functions.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    initialize_shared_runtime_binding("python").map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to initialize NeMo Relay runtime ownership: {e}"
        ))
    })?;
    register_adaptive_component().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to register adaptive plugin component: {e}"
        ))
    })?;
    register_local_backend_provider(Arc::new(register_python_local_guardrails_backend)).map_err(
        |e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to register NeMo Guardrails local backend provider: {e}"
            ))
        },
    )?;
    py_types::register(m)?;
    py_api::register(m)?;
    py_plugin::register(m)?;
    py_adaptive::register(m)?;
    Ok(())
}

fn register_python_local_guardrails_backend(
    config: NeMoGuardrailsConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    let plugin_config = match serde_json::to_value(config) {
        Ok(Json::Object(config)) => config,
        Ok(_) => {
            return Err(nemo_relay::plugin::PluginError::Internal(
                "NeMo Guardrails local config did not serialize to a JSON object".to_string(),
            ));
        }
        Err(err) => {
            return Err(nemo_relay::plugin::PluginError::Internal(format!(
                "failed to serialize NeMo Guardrails local config: {err}"
            )));
        }
    };

    let registrations = Python::attach(|py| {
        let register_fn = load_guardrails_local_register_fn(py)?;
        let namespace_prefix = ctx.qualify_name("");
        crate::py_plugin::invoke_python_plugin_register(
            py,
            "nemo_guardrails",
            &register_fn,
            &plugin_config,
            namespace_prefix,
        )
    })
    .map_err(|err| nemo_relay::plugin::PluginError::RegistrationFailed(err.to_string()))?;

    ctx.extend_registrations(registrations);
    Ok(())
}

fn load_guardrails_local_register_fn(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let module = match py.import("nemo_relay._guardrails_local") {
        Ok(module) => module,
        Err(err) => {
            let source_python_dir = guardrails_local_source_python_dir();
            if !source_python_dir.exists() {
                return Err(err);
            }

            prepend_python_path_if_missing(py, &source_python_dir)?;
            py.import("nemo_relay._guardrails_local")?
        }
    };
    module.getattr("register_local_backend")
}

fn guardrails_local_source_python_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python")
}

fn prepend_python_path_if_missing(py: Python<'_>, path: &Path) -> PyResult<()> {
    let sys = py.import("sys")?;
    let sys_path = sys.getattr("path")?;
    let path_str = path.to_string_lossy();

    if !sys_path.contains(path_str.as_ref())? {
        // Source-tree fallback for local development and in-repo tests where the
        // Python package has not been installed into the active environment yet.
        sys_path.call_method1("insert", (0, path_str.as_ref()))?;
    }

    Ok(())
}

#[cfg(test)]
#[path = "../tests/coverage/coverage_tests.rs"]
mod coverage_tests;
