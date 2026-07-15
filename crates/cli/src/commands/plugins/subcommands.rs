// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::{ArgGroup, Args, Subcommand};
use std::path::PathBuf;

/// Args for `nemo-relay plugins`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PluginsCommand {
    #[command(subcommand)]
    pub(crate) command: PluginsSubcommand,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PluginJsonContext<'a> {
    pub(crate) command: &'static str,
    pub(crate) target: Option<&'a str>,
}

/// Plugin configuration subcommands.
#[derive(Debug, Clone, Subcommand)]
pub(crate) enum PluginsSubcommand {
    /// Interactively create or edit built-in and dynamic plugin configuration.
    Edit(PluginsEditCommand),
    /// Register a manifest-backed dynamic plugin in `plugins.toml`.
    Add(PluginsAddCommand),
    /// Validate a manifest-backed dynamic plugin by path or installed ID.
    Validate(PluginsValidateCommand),
    /// List discovered dynamic plugins from the resolved host config.
    List(PluginsListCommand),
    /// Inspect one discovered dynamic plugin by canonical ID.
    Inspect(PluginsInspectCommand),
    /// Mark a registered dynamic plugin enabled in desired state.
    Enable(PluginsEnableCommand),
    /// Mark a registered dynamic plugin disabled in desired state.
    Disable(PluginsDisableCommand),
    /// Tombstone a registered dynamic plugin and remove its host discovery reference.
    Remove(PluginsRemoveCommand),
}

impl PluginsSubcommand {
    pub(crate) fn json_context(&self) -> Option<PluginJsonContext<'_>> {
        match self {
            Self::Validate(command) if command.json => Some(PluginJsonContext {
                command: "plugins validate",
                target: Some(command.target.as_str()),
            }),
            Self::List(command) if command.json => Some(PluginJsonContext {
                command: "plugins list",
                target: None,
            }),
            Self::Inspect(command) if command.json => Some(PluginJsonContext {
                command: "plugins inspect",
                target: Some(command.id.as_str()),
            }),
            _ => None,
        }
    }
}

/// Args for `nemo-relay plugins edit`.
#[derive(Debug, Clone, Default, Args)]
#[command(group(
    ArgGroup::new("scope")
        .args(["user", "project", "global"])
        .multiple(false)
))]
pub(crate) struct PluginsScopeArgs {
    /// Edit the user config at `$XDG_CONFIG_HOME/nemo-relay/plugins.toml`.
    #[arg(long)]
    pub(crate) user: bool,
    /// Edit the nearest project config at `.nemo-relay/plugins.toml`.
    #[arg(long)]
    pub(crate) project: bool,
    /// Edit the system config at `/etc/nemo-relay/plugins.toml`.
    #[arg(long)]
    pub(crate) global: bool,
}

/// Args for `nemo-relay plugins edit`.
#[derive(Debug, Clone, Default, Args)]
pub(crate) struct PluginsEditCommand {
    #[command(flatten)]
    pub(crate) scope: PluginsScopeArgs,
}

/// Args for `nemo-relay plugins add`.
#[derive(Debug, Clone, Default, Args)]
pub(crate) struct PluginsAddCommand {
    #[command(flatten)]
    pub(crate) scope: PluginsScopeArgs,
    /// Path to a plugin directory or explicit `relay-plugin.toml`.
    pub(crate) path: PathBuf,
}

/// Args for `nemo-relay plugins validate`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PluginsValidateCommand {
    /// Canonical plugin ID or a local plugin directory / `relay-plugin.toml` path.
    pub(crate) target: String,
    /// Emit machine-readable JSON output.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Args for `nemo-relay plugins list`.
#[derive(Debug, Clone, Default, Args)]
pub(crate) struct PluginsListCommand {
    /// Include tombstoned dynamic plugin records in the output.
    #[arg(long)]
    pub(crate) all: bool,
    /// Emit machine-readable JSON output.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Args for `nemo-relay plugins inspect`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PluginsInspectCommand {
    /// Canonical plugin ID.
    pub(crate) id: String,
    /// Emit machine-readable JSON output.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Args for `nemo-relay plugins enable`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PluginsEnableCommand {
    /// Canonical plugin ID.
    pub(crate) id: String,
}

/// Args for `nemo-relay plugins disable`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PluginsDisableCommand {
    /// Canonical plugin ID.
    pub(crate) id: String,
}

/// Args for `nemo-relay plugins remove`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PluginsRemoveCommand {
    /// Canonical plugin ID.
    pub(crate) id: String,
}

impl From<PluginsScopeArgs> for crate::plugins::ConfigurationScope {
    fn from(value: PluginsScopeArgs) -> Self {
        match (value.user, value.project, value.global) {
            (false, false, false) => Self::Default,
            (true, false, false) => Self::User,
            (false, true, false) => Self::Project,
            (false, false, true) => Self::Global,
            _ => Self::Invalid,
        }
    }
}

impl PluginsEditCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PluginsEditRequest {
        crate::plugins::PluginsEditRequest {
            scope: self.scope.into(),
        }
    }
}
impl PluginsAddCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PluginsAddRequest {
        crate::plugins::PluginsAddRequest {
            scope: self.scope.into(),
            path: self.path,
        }
    }
}
impl PluginsValidateCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PluginsValidateRequest {
        crate::plugins::PluginsValidateRequest {
            target: self.target,
            json: self.json,
        }
    }
}
impl PluginsListCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PluginsListRequest {
        crate::plugins::PluginsListRequest {
            all: self.all,
            json: self.json,
        }
    }
}
impl PluginsInspectCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PluginsInspectRequest {
        crate::plugins::PluginsInspectRequest {
            id: self.id,
            json: self.json,
        }
    }
}
impl PluginsEnableCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PluginsEnableRequest {
        crate::plugins::PluginsEnableRequest { id: self.id }
    }
}
impl PluginsDisableCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PluginsDisableRequest {
        crate::plugins::PluginsDisableRequest { id: self.id }
    }
}
impl PluginsRemoveCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PluginsRemoveRequest {
        crate::plugins::PluginsRemoveRequest { id: self.id }
    }
}
