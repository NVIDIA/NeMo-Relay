//! PyO3 native extension module for NVAgentRT.
//!
//! This crate compiles to a `_native` Python C extension that is imported by the
//! `nvagentrt` Python package. It exposes all core runtime types and API functions
//! to Python via PyO3.
//!
//! ## Modules
//!
//! - `py_types` — Python-facing type wrappers (`ScopeHandle`, `ToolHandle`, etc.)
//! - `py_api` — Python-facing API functions (`nv_agentrt_push_scope`, etc.)
//! - `py_callable` — Bridges between Python callables and Rust callback types
//! - `py_context` — Notes on scope propagation between sync/async contexts
//! - `convert` — JSON ↔ Python conversion utilities

use pyo3::prelude::*;

mod convert;
mod py_api;
mod py_callable;
mod py_context;
mod py_types;

/// The `_native` PyO3 module entry point. Registers all types and functions.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_types::register(m)?;
    py_api::register(m)?;
    Ok(())
}
