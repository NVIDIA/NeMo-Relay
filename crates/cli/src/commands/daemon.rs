// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Subcommand, ValueEnum};

use super::root::AgentArg;
use crate::daemon;
use crate::daemon::common::address::{
    DEFAULT_DAEMON_BIND, DEFAULT_DAEMON_PORT, DEFAULT_WORKER_BIND,
};
use crate::error::CliError;

/// Run or connect to the multi-user NeMo Relay daemon.
#[derive(Debug, Clone, Args)]
#[command(args_conflicts_with_subcommands = true)]
pub(crate) struct DaemonCommand {
    /// Address on which the daemon accepts public requests.
    #[arg(long, default_value_t = DEFAULT_DAEMON_BIND, value_parser = parse_bind_address)]
    pub(crate) bind: Ipv4Addr,
    /// Port on which the daemon accepts public requests.
    #[arg(long, default_value_t = DEFAULT_DAEMON_PORT, value_parser = parse_nonzero_port)]
    pub(crate) port: u16,
    /// Concrete URL at which clients can reach a daemon bound to 0.0.0.0.
    #[arg(long)]
    pub(crate) advertise_address: Option<String>,
    /// PEM certificate chain for a native TLS daemon listener.
    #[arg(long, requires = "tls_key")]
    pub(crate) tls_cert: Option<PathBuf>,
    /// PKCS#8 PEM private key for a native TLS daemon listener.
    #[arg(long, requires = "tls_cert")]
    pub(crate) tls_key: Option<PathBuf>,
    /// Route directly to configured providers and never activate a worker.
    #[arg(long)]
    pub(crate) pass_through: bool,
    /// Reject requests until an authenticated worker is ready.
    #[arg(long, conflicts_with = "pass_through")]
    pub(crate) require_worker: bool,
    /// Distinct route tokens one machine-user identity may hold at once (1 through 64,
    /// default 4). At the limit, a new token releases the identity's least recently registered
    /// idle token. Falls back to NEMO_RELAY_DAEMON_MAX_TOKENS_PER_IDENTITY.
    #[arg(long, value_parser = parse_max_tokens_per_identity)]
    pub(crate) max_tokens_per_identity: Option<usize>,
    #[command(subcommand)]
    pub(crate) command: Option<DaemonSubcommand>,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum DaemonSubcommand {
    /// Register this MCP process with an explicitly selected daemon.
    Mcp(DaemonMcpCommand),
    /// Forward a managed coding-agent hook to an explicitly selected daemon.
    Hook(DaemonHookCommand),
    /// Run a worker activated and controlled by an explicitly selected daemon.
    Worker(DaemonWorkerCommand),
    /// Create an immutable administrator-managed integration bundle.
    ManagedBundle(DaemonManagedBundleCommand),
    /// Manage this user's daemon client route token file.
    Token(DaemonTokenCommand),
}

#[derive(Debug, Clone, Args)]
pub(crate) struct DaemonTokenCommand {
    #[command(subcommand)]
    pub(crate) command: DaemonTokenSubcommand,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum DaemonTokenSubcommand {
    /// Create this user's client token file if it does not already exist.
    #[command(
        long_about = "Create this user's owner-private daemon client token file if it does not already exist. Run as the user who runs the coding agent, not as root. The file is <user config dir>/.client-token (normally ~/.config/nemo-relay/.client-token; XDG_CONFIG_HOME is honored). Managed hooks, the managed MCP server, and doctor use it when NEMO_RELAY_CLIENT_TOKEN is not set. An existing valid file is left unchanged; an invalid or unsafe file is never overwritten. The token value is never printed."
    )]
    Ensure(DaemonTokenEnsureCommand),
}

#[derive(Debug, Clone, Args)]
pub(crate) struct DaemonTokenEnsureCommand {
    /// Print `{"status":"created"|"existing","path":"..."}` instead of a human-readable line.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct DaemonMcpCommand {
    /// Absolute daemon URL, including scheme, host, and port.
    #[arg(long, value_parser = parse_daemon_address)]
    pub(crate) daemon_address: String,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct DaemonHookCommand {
    /// Coding agent whose native hook payload is read from standard input.
    #[arg(value_enum)]
    pub(crate) agent: AgentArg,
    /// Absolute daemon URL, including scheme, host, and port.
    #[arg(long, value_parser = parse_daemon_address)]
    pub(crate) daemon_address: String,
    /// Allow the coding agent to continue when hook delivery fails.
    #[arg(long, conflicts_with = "fail_closed")]
    pub(crate) fail_open: bool,
    /// Return a failure when the hook cannot be delivered or is rejected.
    #[arg(long, conflicts_with = "fail_open")]
    pub(crate) fail_closed: bool,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct DaemonWorkerCommand {
    /// Absolute daemon URL, including scheme, host, and port.
    #[arg(long, value_parser = parse_daemon_address)]
    pub(crate) daemon_address: String,
    /// Address on which the worker accepts daemon requests.
    #[arg(long, default_value_t = DEFAULT_WORKER_BIND, value_parser = parse_bind_address)]
    pub(crate) bind: Ipv4Addr,
    /// Prescribed worker port. Omit to let the operating system select a port.
    #[arg(long, value_parser = parse_nonzero_port)]
    pub(crate) port: Option<u16>,
    /// Concrete daemon-reachable host or IP for a worker bound to 0.0.0.0.
    #[arg(long)]
    pub(crate) advertise_address: Option<String>,
}

#[derive(Debug, Clone, Args)]
#[command(
    long_about = "Create a new immutable administrator-managed integration bundle. This is separate from personal `nemo-relay install`: artifacts contain only fixed deployment values and an existing bundle is never rewritten with different bytes. The command prints the canonical bundle SHA-256 to stdout for separate administrator provisioning.",
    after_help = "On success, stdout contains only the canonical bundle SHA-256 for separate administrator provisioning. The dispatcher is checked lexically for the target platform. It must be an absolute stable system path outside known user and temporary directories. Filesystem ownership is not checked while building because the bundle may be created on a different operating system; deployment tooling must install the dispatcher with administrator-controlled ownership and permissions."
)]
pub(crate) struct DaemonManagedBundleCommand {
    /// New bundle directory. An existing byte-identical bundle is left untouched.
    #[arg(long)]
    pub(crate) output: PathBuf,
    /// Fixed absolute daemon URL embedded identically for every managed user.
    #[arg(long, value_parser = parse_daemon_address)]
    pub(crate) daemon_address: String,
    /// Absolute, stable administrator dispatcher path embedded in every artifact.
    ///
    /// The path is validated lexically for the selected target platform. It must be outside
    /// known user and temporary directories. Ownership is enforced at deployment time because
    /// cross-platform bundles may be built on a different operating system.
    #[arg(long, value_name = "ABSOLUTE-PATH")]
    pub(crate) dispatcher_command: String,
    /// Operating system on which the managed artifacts will be deployed.
    #[arg(long, value_enum)]
    pub(crate) platform: ManagedPlatformArg,
    /// Managed coding agent to include. Repeat this option to include multiple agents.
    #[arg(long = "agent", value_enum, required = true)]
    pub(crate) agents: Vec<AgentArg>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
pub(crate) enum ManagedPlatformArg {
    Linux,
    Macos,
    Windows,
}

impl From<ManagedPlatformArg> for daemon::managed::ManagedPlatform {
    fn from(platform: ManagedPlatformArg) -> Self {
        match platform {
            ManagedPlatformArg::Linux => Self::Linux,
            ManagedPlatformArg::Macos => Self::Macos,
            ManagedPlatformArg::Windows => Self::Windows,
        }
    }
}

pub(crate) async fn execute(
    command: DaemonCommand,
    server: &crate::commands::serve::ServerArgs,
) -> Result<ExitCode, CliError> {
    match command.command {
        None => {
            if command.bind == Ipv4Addr::UNSPECIFIED && command.advertise_address.is_none() {
                return Err(CliError::Config(
                    "a daemon bound to 0.0.0.0 requires --advertise-address".into(),
                ));
            }
            daemon::serve(daemon::ServerOptions {
                bind: command.bind,
                port: command.port,
                advertise_address: command.advertise_address,
                pass_through: command.pass_through,
                require_worker: command.require_worker,
                max_tokens_per_identity: resolve_max_tokens_per_identity(
                    command.max_tokens_per_identity,
                    std::env::var(MAX_TOKENS_PER_IDENTITY_ENV).ok().as_deref(),
                )?,
                gateway: server.to_runtime(),
                tls_cert: command.tls_cert,
                tls_key: command.tls_key,
            })
            .await?;
        }
        Some(DaemonSubcommand::Mcp(command)) => {
            daemon::mcp::run(daemon::mcp::Options {
                daemon_address: command.daemon_address,
            })
            .await?;
        }
        Some(DaemonSubcommand::Hook(command)) => {
            daemon::hook::run(daemon::hook::Options {
                agent: command.agent.into(),
                daemon_address: command.daemon_address,
                failure_policy: if command.fail_closed {
                    crate::hooks::HookFailurePolicy::FailClosed
                } else if command.fail_open {
                    crate::hooks::HookFailurePolicy::FailOpen
                } else {
                    crate::hooks::HookFailurePolicy::Default
                },
            })
            .await?;
        }
        Some(DaemonSubcommand::Worker(command)) => {
            if command.bind == Ipv4Addr::UNSPECIFIED && command.advertise_address.is_none() {
                return Err(CliError::Config(
                    "a worker bound to 0.0.0.0 requires --advertise-address".into(),
                ));
            }
            daemon::worker::run(daemon::worker::Options {
                daemon_address: command.daemon_address,
                bind: command.bind,
                port: command.port,
                advertise_address: command.advertise_address,
            })
            .await?;
        }
        Some(DaemonSubcommand::ManagedBundle(command)) => {
            let agents = command.agents.into_iter().map(|agent| match agent {
                AgentArg::Codex => daemon::managed::ManagedAgent::Codex,
                AgentArg::Claude => daemon::managed::ManagedAgent::ClaudeCode,
                AgentArg::Pi => daemon::managed::ManagedAgent::Pi,
            });
            let spec = daemon::managed::ManagedBundleSpec::new(
                command.daemon_address,
                command.dispatcher_command,
                command.platform.into(),
                agents,
            )?;
            let sha256 = daemon::managed::write_new_bundle(&command.output, &spec)?;
            println!("{sha256}");
        }
        Some(DaemonSubcommand::Token(DaemonTokenCommand {
            command: DaemonTokenSubcommand::Ensure(command),
        })) => {
            let path = daemon::common::client_token::client_token_path().ok_or_else(|| {
                CliError::Config(
                    "cannot determine the per-user NeMo Relay config directory; set HOME or XDG_CONFIG_HOME"
                        .into(),
                )
            })?;
            let status = daemon::common::client_token::ensure_client_token(&path)?;
            println!("{}", render_token_ensure(status, &path, command.json)?);
        }
    }

    Ok(ExitCode::SUCCESS)
}

fn render_token_ensure(
    status: daemon::common::client_token::EnsureStatus,
    path: &std::path::Path,
    json: bool,
) -> Result<String, CliError> {
    if json {
        return serde_json::to_string(&serde_json::json!({
            "status": status.as_str(),
            "path": path.display().to_string(),
        }))
        .map_err(|error| CliError::Config(format!("failed to encode token status: {error}")));
    }
    Ok(match status {
        daemon::common::client_token::EnsureStatus::Created => {
            format!("Created NeMo Relay client token file {}", path.display())
        }
        daemon::common::client_token::EnsureStatus::Existing => format!(
            "NeMo Relay client token file {} already exists",
            path.display()
        ),
    })
}

fn parse_bind_address(value: &str) -> Result<Ipv4Addr, String> {
    let address = value
        .parse::<Ipv4Addr>()
        .map_err(|_| "bind address must be 127.0.0.1 or 0.0.0.0".to_string())?;
    if matches!(address, Ipv4Addr::LOCALHOST | Ipv4Addr::UNSPECIFIED) {
        Ok(address)
    } else {
        Err("bind address must be 127.0.0.1 or 0.0.0.0".into())
    }
}

/// Environment fallback for `--max-tokens-per-identity`, read only when serving the daemon so a
/// malformed value cannot break the `mcp`, `hook`, or `worker` subcommands.
const MAX_TOKENS_PER_IDENTITY_ENV: &str = "NEMO_RELAY_DAEMON_MAX_TOKENS_PER_IDENTITY";

fn parse_max_tokens_per_identity(value: &str) -> Result<usize, String> {
    let limit = daemon::broker::registry::MAX_TOKENS_PER_IDENTITY_LIMIT;
    match value.trim().parse::<usize>() {
        Ok(tokens) if (1..=limit).contains(&tokens) => Ok(tokens),
        _ => Err(format!(
            "the per-identity token limit must be an integer from 1 through {limit}"
        )),
    }
}

fn resolve_max_tokens_per_identity(
    flag: Option<usize>,
    environment: Option<&str>,
) -> Result<usize, CliError> {
    match (flag, environment) {
        (Some(tokens), _) => Ok(tokens),
        (None, Some(value)) => parse_max_tokens_per_identity(value)
            .map_err(|error| CliError::Config(format!("{MAX_TOKENS_PER_IDENTITY_ENV}: {error}"))),
        (None, None) => Ok(daemon::broker::registry::DEFAULT_MAX_TOKENS_PER_IDENTITY),
    }
}

fn parse_nonzero_port(value: &str) -> Result<u16, String> {
    match value.parse::<u16>() {
        Ok(0) => Err("an explicitly supplied port must be between 1 and 65535".into()),
        Ok(port) => Ok(port),
        Err(_) => Err("port must be between 1 and 65535".into()),
    }
}

fn parse_daemon_address(value: &str) -> Result<String, String> {
    let uri = value
        .parse::<http::Uri>()
        .map_err(|_| "daemon address must be an absolute HTTP or HTTPS URL".to_string())?;
    let scheme = uri
        .scheme_str()
        .filter(|scheme| matches!(*scheme, "http" | "https"))
        .ok_or_else(|| "daemon address must use http or https".to_string())?;
    let authority = uri
        .authority()
        .ok_or_else(|| "daemon address must include a host and explicit port".to_string())?;
    if authority.port_u16().is_none() {
        return Err("daemon address must include an explicit port".into());
    }
    let url = reqwest::Url::parse(value)
        .map_err(|_| "daemon address must be an absolute HTTP or HTTPS URL".to_string())?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err(
            "daemon address cannot contain credentials, a non-root path, query, or fragment".into(),
        );
    }
    let host = url
        .host_str()
        .ok_or_else(|| "daemon address must include a host".to_string())?;
    let host = host.trim_matches(['[', ']']);
    if host
        .parse::<std::net::IpAddr>()
        .is_ok_and(|address| address.is_unspecified())
    {
        return Err("0.0.0.0 is a bind address and cannot be a daemon target".into());
    }
    let loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback());
    if scheme == "http" && !loopback {
        return Err("non-loopback daemon addresses must use https".into());
    }
    Ok(value.trim_end_matches('/').to_string())
}

#[cfg(test)]
#[path = "../../tests/coverage/commands/daemon_tests.rs"]
mod tests;
