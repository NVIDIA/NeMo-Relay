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

impl PluginsCommand {
    pub(crate) fn is_edit(&self) -> bool {
        matches!(self.command, PluginsSubcommand::Edit(_))
    }
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
    /// Install a verified plugin bundle from a release source.
    Install(PluginsInstallCommand),
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
    /// Unregister and delete a CLI-managed plugin bundle.
    Uninstall(PluginsUninstallCommand),
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
        .args(["user", "global"])
        .multiple(false)
))]
pub(crate) struct PluginsScopeArgs {
    /// Select plugins in your user configuration (XDG unless an explicit target applies).
    #[arg(long)]
    pub(crate) user: bool,
    /// Select system plugins (`/etc/nemo-relay` on Unix; `%ProgramData%\nemo-relay` on Windows).
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

#[derive(Debug, Clone, Args)]
pub(crate) struct PluginsInstallCommand {
    #[command(flatten)]
    pub(crate) scope: PluginsScopeArgs,
    /// Release source, for example github:NVIDIA/NeMo-Relay-Plugins@switchyard-plugin-0.2.0.
    pub(crate) source: String,
    /// Register the plugin without enabling it.
    #[arg(long)]
    pub(crate) no_enable: bool,
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
    #[command(flatten)]
    pub(crate) scope: PluginsScopeArgs,
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
    #[command(flatten)]
    pub(crate) scope: PluginsScopeArgs,
    /// Canonical plugin ID.
    pub(crate) id: String,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct PluginsUninstallCommand {
    #[command(flatten)]
    pub(crate) scope: PluginsScopeArgs,
    /// Canonical plugin ID.
    pub(crate) id: String,
}

impl From<PluginsScopeArgs> for crate::plugins::ConfigurationScope {
    fn from(value: PluginsScopeArgs) -> Self {
        match (value.user, value.global) {
            (false, false) => Self::Default,
            (true, false) => Self::User,
            (false, true) => Self::Global,
            _ => Self::Invalid,
        }
    }
}

impl PluginsEditCommand {
    pub(crate) fn into_runtime(
        self,
        explicit_path: Option<PathBuf>,
    ) -> crate::plugins::PluginsEditRequest {
        let scope = self.scope.into();
        let explicit_path = matches!(
            scope,
            crate::plugins::ConfigurationScope::Default | crate::plugins::ConfigurationScope::User
        )
        .then_some(explicit_path)
        .flatten();
        crate::plugins::PluginsEditRequest {
            explicit_path,
            scope,
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
