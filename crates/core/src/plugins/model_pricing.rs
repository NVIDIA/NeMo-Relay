// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime registration of the crate-owned pricing component.

use crate::plugin::{Result, register_builtin_plugin};
pub use nemo_relay_model_pricing::plugin::{PRICING_PLUGIN_KIND, PricingPlugin};
use std::sync::Arc;
pub(crate) fn register_pricing_component() -> Result<()> {
    register_builtin_plugin(Arc::new(PricingPlugin))
}
