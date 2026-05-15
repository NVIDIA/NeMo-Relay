// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical Adaptive Cache Governor (ACG) module surface exposed from the
//! adaptive crate.

/// Minimum observations required before Adaptive Cache Governor (ACG) emits
/// optimization intents.
pub const MIN_ACG_OBSERVATIONS: u32 = 2;

pub mod anthropic_plugin;
pub mod canonicalize;
pub mod capability;
pub(crate) mod debug;
pub(crate) mod economics;
pub mod error;
pub mod ir_builder;
pub mod openai_plugin;
pub mod passthrough;
pub mod plugin;
pub mod plugin_registry;
pub mod policy;
pub mod profile;
pub mod prompt_ir;
pub(crate) mod request_surfaces;
pub mod retention;
pub mod stability;
pub mod telemetry;
pub(crate) mod translation;
pub mod types;
pub mod variable_extractor;

pub use anthropic_plugin::*;
pub use canonicalize::*;
pub use capability::*;
pub use error::{AcgError, Result};
pub use ir_builder::*;
pub use openai_plugin::*;
pub use passthrough::*;
pub use plugin::*;
pub use plugin_registry::*;
pub use policy::*;
pub use profile::*;
pub use prompt_ir::*;
pub use retention::*;
pub use stability::*;
pub use telemetry::*;
pub use types::*;
pub use variable_extractor::*;

#[cfg(test)]
#[path = "../../tests/unit/acg/mod.rs"]
mod tests;
