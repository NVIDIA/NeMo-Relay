// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `nemo-relay auth`: sign in to the provider named by `[upstream.<provider>_oidc]`.

use std::path::Path;
use std::process::ExitCode;

use clap::{Args, Subcommand, ValueEnum};

use super::serve::ServerArgs;
use crate::configuration::GatewayConfig;
use crate::error::CliError;
use crate::oidc::{BrowserApproval, LoginMethod, OidcSource};

#[derive(Debug, Clone, Args)]
pub(crate) struct AuthCommand {
    #[command(subcommand)]
    pub(crate) command: AuthSubcommand,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum AuthSubcommand {
    /// Sign in to the configured OpenID Connect provider in a browser.
    Login(LoginArgs),
    /// Print a current access token, refreshing it near expiry. For scripts; the gateway reads
    /// the sign-in itself.
    Token(ProviderArgs),
    /// Remove the stored sign-in. The next upstream request no longer uses it.
    Logout(ProviderArgs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ProviderArg {
    Openai,
    Anthropic,
}

#[derive(Debug, Clone, Default, Args)]
pub(crate) struct LoginArgs {
    #[command(flatten)]
    pub(crate) provider: ProviderArgs,
    /// Print the sign-in address instead of opening a browser.
    #[arg(long, conflicts_with = "device")]
    pub(crate) no_browser: bool,
    /// Loopback port for the browser redirect, for example one forwarded over SSH. A free port
    /// is picked by default.
    #[arg(long, default_value_t = 0, conflicts_with = "device")]
    pub(crate) callback_port: u16,
    /// Sign in with a code entered on the issuer's own page (OAuth device authorization) instead
    /// of the browser redirect.
    #[arg(long)]
    pub(crate) device: bool,
}

/// Without flags, the provider comes from `[upstream.openai_oidc]` or
/// `[upstream.anthropic_oidc]` in the effective configuration, including `--config`.
#[derive(Debug, Clone, Default, Args)]
pub(crate) struct ProviderArgs {
    /// Which configured provider to use when both have an `oidc` section.
    #[arg(long, value_enum)]
    pub(crate) provider: Option<ProviderArg>,
    /// OpenID Connect issuer URL, instead of the configured one.
    #[arg(long, requires = "client_id")]
    pub(crate) issuer: Option<String>,
    /// OAuth client identifier, instead of the configured one.
    #[arg(long, requires = "issuer")]
    pub(crate) client_id: Option<String>,
    /// Space-separated scopes for `login`, instead of the configured ones.
    #[arg(long)]
    pub(crate) scope: Option<String>,
}

pub(super) async fn execute(
    command: AuthCommand,
    server: &ServerArgs,
) -> Result<ExitCode, CliError> {
    let dir = crate::oidc::sessions_dir()?;
    let load = || configured(server);
    match command.command {
        AuthSubcommand::Login(args) => login(&dir, args, load).await?,
        AuthSubcommand::Token(args) => print_token(&dir, select_source(args, load)?).await?,
        AuthSubcommand::Logout(args) => logout(&dir, select_source(args, load)?).await?,
    }
    Ok(ExitCode::SUCCESS)
}

async fn login(
    dir: &Path,
    args: LoginArgs,
    load: impl FnOnce() -> Result<GatewayConfig, CliError>,
) -> Result<(), CliError> {
    let source = select_source(args.provider, load)?;
    let method = if args.device {
        LoginMethod::Device
    } else {
        LoginMethod::Browser(BrowserApproval::interactive(
            !args.no_browser,
            args.callback_port,
        ))
    };
    crate::oidc::login(dir, &source, method, |prompt| println!("{prompt}")).await?;
    println!("Signed in to {}.", source.issuer);
    Ok(())
}

async fn print_token(dir: &Path, source: OidcSource) -> Result<(), CliError> {
    let token = crate::oidc::current_token(dir, &source, &crate::oidc::now_secs).await?;
    println!("{token}");
    Ok(())
}

async fn logout(dir: &Path, source: OidcSource) -> Result<(), CliError> {
    match crate::oidc::logout(dir, &source).await? {
        crate::oidc::Logout::NotSignedIn => {
            println!("No sign-in is stored for {}.", source.issuer);
        }
        crate::oidc::Logout::Removed(crate::oidc::Revocation::Revoked) => {
            println!(
                "Removed the sign-in for {}. The issuer revoked its tokens.",
                source.issuer
            );
        }
        crate::oidc::Logout::Removed(crate::oidc::Revocation::NotOffered) => {
            println!(
                "Removed the sign-in for {0}. {0} offers no token revocation, so its tokens stay valid until they expire.",
                source.issuer
            );
        }
        crate::oidc::Logout::Removed(crate::oidc::Revocation::Failed(detail)) => {
            println!("Removed the sign-in for {}.", source.issuer);
            eprintln!(
                "warning: {} did not revoke the tokens ({detail}). They stay valid until they expire.",
                source.issuer
            );
        }
    }
    Ok(())
}

/// The effective configuration, including the user layer that `--config` replaces.
fn configured(server: &ServerArgs) -> Result<GatewayConfig, CliError> {
    crate::configuration::resolve_daemon_config(&server.to_runtime()).map(|config| config.gateway)
}

/// Explicit `--issuer` and `--client-id` win; otherwise the configuration names the provider.
fn select_source(
    args: ProviderArgs,
    load: impl FnOnce() -> Result<GatewayConfig, CliError>,
) -> Result<OidcSource, CliError> {
    if let (Some(issuer), Some(client_id)) = (&args.issuer, &args.client_id) {
        return Ok(OidcSource::new(issuer, client_id, args.scope.as_deref()));
    }
    let config = load()?;
    let mut source = match (args.provider, &config.openai_oidc, &config.anthropic_oidc) {
        (Some(ProviderArg::Openai), Some(source), _)
        | (Some(ProviderArg::Anthropic), _, Some(source))
        | (None, Some(source), None)
        | (None, None, Some(source)) => source.clone(),
        (None, Some(openai), Some(anthropic)) if openai == anthropic => openai.clone(),
        (None, Some(_), Some(_)) => {
            return Err(CliError::Config(
                "both [upstream.openai_oidc] and [upstream.anthropic_oidc] are configured; pass --provider openai or --provider anthropic".into(),
            ));
        }
        (provider, _, _) => {
            let section = match provider {
                Some(ProviderArg::Anthropic) => "[upstream.anthropic_oidc]",
                Some(ProviderArg::Openai) => "[upstream.openai_oidc]",
                None => "[upstream.openai_oidc] or [upstream.anthropic_oidc]",
            };
            return Err(CliError::Config(format!(
                "no {section} is configured; add one, or pass --issuer and --client-id"
            )));
        }
    };
    if let Some(scope) = args.scope {
        source.scope = scope;
    }
    Ok(source)
}

#[cfg(test)]
#[path = "../../tests/coverage/commands/auth_tests.rs"]
mod tests;
