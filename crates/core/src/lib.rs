// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public Rust SDK composition for NeMo Relay.
//!
//! Use `nemo-relay-codecs` for provider implementations and
//! `nemo-relay-runtime` for execution internals, diagnostics and
//! plugin hosting. This crate assembles the default components and provides
//! high-level lifecycle APIs and exporter constructors.

pub mod api;
mod composition;
mod observability_runtime;
pub mod plugin;
pub use nemo_relay_types::{Json, editor_config, error};

/// Install default component services before using the runtime directly.
///
/// This does not activate configured plugins or start resource sampling.
pub fn initialize_runtime() {
    composition::ensure_runtime_services();
}
