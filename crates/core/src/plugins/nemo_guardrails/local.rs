// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(not(feature = "python"))]
use crate::plugin::PluginError;
use crate::plugin::{PluginRegistrationContext, Result as PluginResult};

use super::NeMoGuardrailsConfig;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
pub(super) fn register_local_backend(
    config: NeMoGuardrailsConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    python::register_local_backend(config, ctx)
}

#[cfg(not(feature = "python"))]
pub(super) fn register_local_backend(
    _config: NeMoGuardrailsConfig,
    _ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    Err(PluginError::RegistrationFailed(
        "built-in NeMo Guardrails local backend is unavailable in this build".to_string(),
    ))
}
