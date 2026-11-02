// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration activation entry points shared by Rust and language bindings.
use crate::{ConfigReport, PluginConfig, PluginHostActivation, Result, VerifiedDynamicPluginSpec};

/// Activate exactly the supplied configuration without implicit discovery.
pub async fn initialize_exact(config: PluginConfig) -> Result<PluginHostActivation> {
    crate::initialize_runtime();
    PluginHostActivation::initialize_exact(config).await
}

/// Activate verified artifacts under the host's configuration lease.
pub async fn initialize_with_verified_specs<I>(
    config: PluginConfig,
    specs: I,
) -> Result<(PluginHostActivation, ConfigReport)>
where
    I: IntoIterator<Item = VerifiedDynamicPluginSpec>,
{
    crate::initialize_runtime();
    PluginHostActivation::initialize_with_verified_specs(config, specs).await
}
