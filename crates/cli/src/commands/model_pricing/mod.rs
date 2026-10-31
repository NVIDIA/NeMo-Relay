// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgGroup, Args, Subcommand};

use crate::error::CliError;

mod run;

/// Args for `nemo-relay model-pricing`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PricingCommand {
    #[command(subcommand)]
    pub(crate) command: PricingSubcommand,
}

/// Model pricing catalog and resolver subcommands.
#[derive(Debug, Clone, Subcommand)]
pub(crate) enum PricingSubcommand {
    /// Validate a model pricing catalog JSON file.
    Validate(PricingValidateCommand),
    /// Initialize model pricing in `plugins.toml`.
    Init(PricingInitCommand),
    /// Add a model pricing catalog file source to `plugins.toml`.
    AddSource(PricingAddSourceCommand),
    /// Resolve which model pricing entry matches a model and optional usage.
    Resolve(PricingResolveCommand),
}

/// Common target-scope flags for model pricing config mutations.
#[derive(Debug, Clone, Default, Args)]
#[command(group(
    ArgGroup::new("pricing_scope")
        .args(["user", "global"])
        .multiple(false)
))]
pub(crate) struct PricingScopeArgs {
    /// Edit the user config at `$XDG_CONFIG_HOME/nemo-relay/plugins.toml`.
    #[arg(long)]
    pub(crate) user: bool,
    /// Edit system config (`/etc/nemo-relay` on Unix; `%ProgramData%\nemo-relay` on Windows).
    #[arg(long)]
    pub(crate) global: bool,
}

/// Args for `nemo-relay model-pricing validate`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PricingValidateCommand {
    /// Path to a Relay model pricing catalog JSON file.
    pub(crate) path: PathBuf,
}

/// Args for `nemo-relay model-pricing init`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PricingInitCommand {
    #[command(flatten)]
    pub(crate) scope: PricingScopeArgs,
}

/// Args for `nemo-relay model-pricing add-source`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PricingAddSourceCommand {
    #[command(flatten)]
    pub(crate) scope: PricingScopeArgs,
    /// Path to a Relay model pricing catalog JSON file.
    pub(crate) path: PathBuf,
    /// Append as a lower-priority source instead of prepending as the highest-priority override.
    #[arg(long)]
    pub(crate) append: bool,
}

/// Args for `nemo-relay model-pricing resolve`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PricingResolveCommand {
    /// Model ID or routed model name to look up.
    pub(crate) model: String,
    /// Optional provider or route, such as `openai`, `anthropic`, or `azure/openai`.
    #[arg(long)]
    pub(crate) provider: Option<String>,
    /// Prompt/input token count to use for an estimate.
    #[arg(long)]
    pub(crate) prompt_tokens: Option<u64>,
    /// Completion/output token count to use for an estimate.
    #[arg(long)]
    pub(crate) completion_tokens: Option<u64>,
    /// Prompt-cache read token count to use for an estimate.
    #[arg(long)]
    pub(crate) cache_read_tokens: Option<u64>,
    /// Prompt-cache write token count to use for an estimate.
    #[arg(long)]
    pub(crate) cache_write_tokens: Option<u64>,
}
impl From<PricingScopeArgs> for crate::plugins::ConfigurationScope {
    fn from(value: PricingScopeArgs) -> Self {
        match (value.user, value.global) {
            (false, false) => Self::Default,
            (true, false) => Self::User,
            (false, true) => Self::Global,
            _ => Self::Invalid,
        }
    }
}

pub(super) fn execute(command: PricingCommand) -> Result<ExitCode, CliError> {
    match command.command {
        PricingSubcommand::Validate(command) => run::validate(command)?,
        PricingSubcommand::Init(command) => run::init(command)?,
        PricingSubcommand::AddSource(command) => run::add_source(command)?,
        PricingSubcommand::Resolve(command) => run::resolve(command)?,
    }
    Ok(ExitCode::SUCCESS)
}
