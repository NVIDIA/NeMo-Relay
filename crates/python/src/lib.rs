// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 native extension module for Nexus.
//!
//! This crate compiles to a `_native` Python C extension that is imported by the
//! `nat_nexus` Python package. It exposes all core runtime types and API functions
//! to Python via PyO3.
//!
//! ## Modules
//!
//! - `py_types` — Python-facing type wrappers (`ScopeHandle`, `ToolHandle`, `Event`,
//!   `AtifExporter`, etc.). `Event` exposes typed lifecycle fields (`input`, `output`,
//!   `model_name`, `tool_call_id`, `root_uuid`). `AtifExporter` collects events and
//!   exports ATIF v1.6 trajectories.
//! - `py_api` — Python-facing API functions (`nat_nexus_push_scope`, etc.). Tool calls
//!   accept `tool_call_id` and LLM calls accept `model_name` for ATIF correlation.
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
