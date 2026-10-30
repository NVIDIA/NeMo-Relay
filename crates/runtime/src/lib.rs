// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! Runtime execution, registries, scopes, diagnostics, and plugin hosting for NeMo Relay.
//! Component implementations are supplied by the embedding SDK or application.

#[cfg(test)]
extern crate self as nemo_relay_runtime;

pub mod api;
pub mod codec;
pub mod config_editor;
mod context;

pub mod error;
pub mod json;

#[cfg(test)]
#[path = "../tests/support/observability_runtime.rs"]
mod observability_runtime;
#[cfg(test)]
#[path = "../tests/support/observability/mod.rs"]
mod observability_test_support;
pub mod plugin;
mod registry;
#[cfg(test)]
#[path = "../tests/support/resource_metrics/mod.rs"]
mod resource_metrics;
#[doc(hidden)]
pub mod shared_runtime;
pub mod stream;

#[cfg(test)]
#[path = "../tests/unit/types_tests.rs"]
mod types_tests;

/// Declare typed configuration editor metadata.
pub use nemo_relay_types::editor_config;

/// Services supplied by the SDK or an embedding application.
pub mod composition;
/// Runtime pricing service adapters.
pub mod pricing;
#[cfg(test)]
#[path = "../tests/support/composition.rs"]
mod test_composition;

#[cfg(test)]
#[path = "../tests/support/mod.rs"]
pub(crate) mod test_support;
