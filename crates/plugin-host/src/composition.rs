// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional built-in plugin composition supplied by the SDK.
use crate::{ConfigDiagnostic, PluginConfig, Result};
use std::sync::{LazyLock, OnceLock};
/// Services supplied by the embedding SDK or application.
#[derive(Clone, Copy)]
pub struct HostServices {
    /// Register the embedding application's built-in plugin implementations.
    pub register_builtin_plugins: fn() -> Result<()>,
    /// Validate component-specific configuration before activation.
    pub preflight: fn(&PluginConfig) -> Vec<ConfigDiagnostic>,
}
impl Default for HostServices {
    fn default() -> Self {
        Self {
            register_builtin_plugins: || Ok(()),
            preflight: |_| Vec::new(),
        }
    }
}
static SERVICES: OnceLock<HostServices> = OnceLock::new();
static UNCOMPOSED: LazyLock<HostServices> = LazyLock::new(HostServices::default);
/// Install host composition once, preserving the chosen application.
pub fn install_host_services(factory: impl FnOnce() -> HostServices) -> &'static HostServices {
    SERVICES.get_or_init(factory)
}
pub(crate) fn services() -> &'static HostServices {
    #[cfg(test)]
    crate::test_composition::ensure_runtime_services();
    SERVICES.get().unwrap_or(&UNCOMPOSED)
}
