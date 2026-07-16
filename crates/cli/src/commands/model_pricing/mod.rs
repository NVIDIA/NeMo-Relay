// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgGroup, Args, Subcommand};

use crate::error::CliError;

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
        .args(["user", "project", "global"])
        .multiple(false)
))]
pub(crate) struct PricingScopeArgs {
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
        match (value.user, value.project, value.global) {
            (false, false, false) => Self::Default,
            (true, false, false) => Self::User,
            (false, true, false) => Self::Project,
            (false, false, true) => Self::Global,
            _ => Self::Invalid,
        }
    }
}
impl PricingValidateCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PricingValidateRequest {
        crate::plugins::PricingValidateRequest { path: self.path }
    }
}
impl PricingInitCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PricingInitRequest {
        crate::plugins::PricingInitRequest {
            scope: self.scope.into(),
        }
    }
}
impl PricingAddSourceCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PricingAddSourceRequest {
        crate::plugins::PricingAddSourceRequest {
            scope: self.scope.into(),
            path: self.path,
            append: self.append,
        }
    }
}
impl PricingResolveCommand {
    pub(crate) fn into_runtime(self) -> crate::plugins::PricingResolveRequest {
        crate::plugins::PricingResolveRequest {
            model: self.model,
            provider: self.provider,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            cache_read_tokens: self.cache_read_tokens,
            cache_write_tokens: self.cache_write_tokens,
        }
    }
}

pub(super) fn execute(command: PricingCommand) -> Result<ExitCode, CliError> {
    match command.command {
        PricingSubcommand::Validate(command) => {
            crate::plugins::pricing::validate(command.into_runtime())?
        }
        PricingSubcommand::Init(command) => crate::plugins::pricing::init(command.into_runtime())?,
        PricingSubcommand::AddSource(command) => {
            crate::plugins::pricing::add_source(command.into_runtime())?
        }
        PricingSubcommand::Resolve(command) => {
            crate::plugins::pricing::resolve(command.into_runtime())?
        }
    }
    Ok(ExitCode::SUCCESS)
}
