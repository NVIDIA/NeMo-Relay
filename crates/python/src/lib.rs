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
use nemo_relay::shared_runtime::initialize_shared_runtime_binding;
use nemo_relay_adaptive::plugin_component::register_adaptive_component;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

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
    py_types::register(m)?;
    py_api::register(m)?;
    py_plugin::register(m)?;
    py_adaptive::register(m)?;
    install_native_module_alias(m)?;
    Ok(())
}

fn install_native_module_alias(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?.cast_into::<PyDict>()?;
    modules.set_item("nemo_relay._native", m)?;
    if let Ok(package) = py.import("nemo_relay") {
        let _ = package.setattr("_native", m);
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/coverage/coverage_tests.rs"]
mod coverage_tests;
