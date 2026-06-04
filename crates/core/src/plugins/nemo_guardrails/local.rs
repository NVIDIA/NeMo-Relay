// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::plugin::{PluginError, PluginRegistrationContext, Result as PluginResult};

use super::NeMoGuardrailsConfig;

#[cfg(feature = "python")]
mod python;

pub(super) fn register_local_backend(
    config: NeMoGuardrailsConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    #[cfg(feature = "python")]
    {
        return python::register_local_backend(config, ctx);
    }

    #[allow(unreachable_code)]
    Err(PluginError::RegistrationFailed(
        "built-in NeMo Guardrails local backend is unavailable in this build".to_string(),
    ))
}
