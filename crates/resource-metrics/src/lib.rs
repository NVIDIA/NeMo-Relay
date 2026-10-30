// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resource collection and measurement configuration for NeMo Relay.

/// Collection settings and measurement units.
pub mod config;

#[doc(hidden)]
#[allow(missing_docs)]
pub mod collector;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod network;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod units;

/// Collection errors use the shared Relay runtime error contract.
pub use nemo_relay_types::error::FlowError as Error;
/// Result of configuring or collecting resource metrics.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
#[path = "../tests/support/snapshot.rs"]
mod snapshot_fixture;

/// Sampling ownership and polling lifecycle.
pub mod manager;
#[doc(hidden)]
#[allow(missing_docs)]
pub(crate) mod memory;
/// Collect a fresh resource snapshot on a Tokio blocking worker.
pub use manager::collect_on_demand as collect;

mod config_editor;

/// Built-in component implementation.
pub mod plugin;
