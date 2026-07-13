// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ConfigurationScope {
    pub(crate) user: bool,
    pub(crate) project: bool,
    pub(crate) global: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PluginsEditRequest {
    pub(crate) scope: ConfigurationScope,
}
#[derive(Debug, Clone, Default)]
pub(crate) struct PluginsAddRequest {
    pub(crate) scope: ConfigurationScope,
    pub(crate) path: PathBuf,
}
#[derive(Debug, Clone)]
pub(crate) struct PluginsValidateRequest {
    pub(crate) target: String,
    pub(crate) json: bool,
}
#[derive(Debug, Clone, Default)]
pub(crate) struct PluginsListRequest {
    pub(crate) all: bool,
    pub(crate) json: bool,
}
#[derive(Debug, Clone)]
pub(crate) struct PluginsInspectRequest {
    pub(crate) id: String,
    pub(crate) json: bool,
}
#[derive(Debug, Clone)]
pub(crate) struct PluginsEnableRequest {
    pub(crate) id: String,
}
#[derive(Debug, Clone)]
pub(crate) struct PluginsDisableRequest {
    pub(crate) id: String,
}
#[derive(Debug, Clone)]
pub(crate) struct PluginsRemoveRequest {
    pub(crate) id: String,
}

#[derive(Debug, Clone)]
pub(crate) struct PricingValidateRequest {
    pub(crate) path: PathBuf,
}
#[derive(Debug, Clone)]
pub(crate) struct PricingInitRequest {
    pub(crate) scope: ConfigurationScope,
}
#[derive(Debug, Clone)]
pub(crate) struct PricingAddSourceRequest {
    pub(crate) scope: ConfigurationScope,
    pub(crate) path: PathBuf,
    pub(crate) append: bool,
}
#[derive(Debug, Clone)]
pub(crate) struct PricingResolveRequest {
    pub(crate) model: String,
    pub(crate) provider: Option<String>,
    pub(crate) prompt_tokens: Option<u64>,
    pub(crate) completion_tokens: Option<u64>,
    pub(crate) cache_read_tokens: Option<u64>,
    pub(crate) cache_write_tokens: Option<u64>,
}
