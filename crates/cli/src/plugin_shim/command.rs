// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hidden `plugin-shim` CLI surface used by installed hooks and installer orchestration.

use clap::{Args, Subcommand, ValueEnum};

use crate::config::CodingAgent;

#[derive(Debug, Clone, Args)]
pub(crate) struct PluginShimCommand {
    #[command(subcommand)]
    pub(crate) command: PluginShimSubcommand,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum PluginShimSubcommand {
    Serve(PluginShimServeCommand),
    Hook(PluginShimHookCommand),
    Install(PluginShimInstallCommand),
    Uninstall(PluginShimUninstallCommand),
    Provider(PluginShimProviderCommand),
    Doctor(PluginShimDoctorCommand),
}

#[derive(Debug, Clone, Args)]
pub(crate) struct PluginShimServeCommand {
    #[arg(allow_hyphen_values = true, trailing_var_arg = true)]
    pub(crate) args: Vec<String>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct PluginShimHookCommand {
    #[arg(value_enum)]
    pub(crate) agent: CodingAgent,
    #[arg(long)]
    pub(crate) gateway_url: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct PluginShimInstallCommand {
    #[arg(value_enum)]
    pub(crate) agent: CodingAgent,
    #[arg(long, default_value = "http://127.0.0.1:47632")]
    pub(crate) gateway_url: String,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct PluginShimUninstallCommand {
    #[arg(value_enum)]
    pub(crate) agent: CodingAgent,
    #[arg(long, default_value = "http://127.0.0.1:47632")]
    pub(crate) gateway_url: String,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct PluginShimProviderCommand {
    #[arg(value_enum)]
    pub(crate) agent: CodingAgent,
    #[arg(value_enum)]
    pub(crate) action: PluginShimProviderAction,
    #[arg(long, default_value = "http://127.0.0.1:47632")]
    pub(crate) gateway_url: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum PluginShimProviderAction {
    Enable,
    Restore,
    Status,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct PluginShimDoctorCommand {
    #[arg(value_enum)]
    pub(crate) agent: CodingAgent,
    #[arg(long, default_value = "http://127.0.0.1:47632")]
    pub(crate) gateway_url: String,
}
