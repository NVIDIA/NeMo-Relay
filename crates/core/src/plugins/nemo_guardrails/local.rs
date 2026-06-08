// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::plugin::{PluginRegistrationContext, Result as PluginResult};

use super::NeMoGuardrailsConfig;

mod python;

pub(super) fn register_local_backend(
    config: NeMoGuardrailsConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    python::register_local_backend(config, ctx)
}
