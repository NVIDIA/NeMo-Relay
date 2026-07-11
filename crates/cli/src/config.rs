// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use axum::http::HeaderMap;
use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};
use nemo_relay::plugin::dynamic::{
    DYNAMIC_PLUGIN_MANIFEST_FILENAME, DynamicPluginManifest, DynamicPluginManifestLoad,
};
use nemo_relay::plugin::{PluginError, merge_plugin_config_documents};
use ring::rand::{SecureRandom, SystemRandom};
use ring::{digest, hmac};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use strum::{Display, IntoStaticStr};

pub(crate) use crate::coding_agent::CodingAgent;
use crate::error::CliError;
use crate::file_io::{LockAttempt, try_lock_exclusive};
#[cfg(test)]
use crate::plugins::lifecycle::active_dynamic_plugin_components;
use crate::plugins::lifecycle::{
    ActiveDynamicPluginComponent, active_dynamic_plugin_components_for_identity,
    dynamic_plugin_runtime_closure_digest, enforce_required_dynamic_plugin_startup,
};
use crate::plugins::policy::DynamicPluginHostPolicy;

pub(crate) const BOOTSTRAP_FINGERPRINT_ENV: &str = "NEMO_RELAY_BOOTSTRAP_FINGERPRINT";
pub(crate) const PLUGIN_IDLE_TIMEOUT_ENV: &str = "NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS";
/// Maximum regular-file size hashed into persistent gateway identity (512 MiB).
pub(crate) const MAX_BOOTSTRAP_IDENTITY_FILE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Parser)]
#[command(name = "nemo-relay")]
#[command(about = "Coding-agent gateway for NeMo Relay observability")]
#[command(version)]
pub(crate) struct Cli {
    #[command(flatten)]
    pub(crate) server: ServerArgs,
    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum Command {
    /// Run Claude Code with observability (setup on first use)
    #[command(
        long_about = "Run Anthropic's `claude` CLI under an ephemeral NeMo Relay gateway. \
                      Observability (ATIF + OpenInference) is wired in transparently via \
                      ANTHROPIC_BASE_URL. First-time use launches the setup wizard so the \
                      `[agents.claude]` block lands in `.nemo-relay/config.toml` and observation \
                      starts on the next invocation without prompts.",
        after_help = "Examples:\n  \
                      nemo-relay claude\n  \
                      nemo-relay claude -- chat \"refactor the launcher\"\n  \
                      nemo-relay claude -- --resume <session-id>"
    )]
    Claude(EasyPathCommand),
    /// Run Codex with observability (setup on first use)
    #[command(
        long_about = "Run OpenAI's `codex` CLI under an ephemeral NeMo Relay gateway. NeMo Relay \
                      injects a `nemo-relay-openai` provider override so codex points at the \
                      gateway; the gateway then forwards to `--openai-base-url` (defaults to \
                      api.openai.com) with `OPENAI_API_KEY` injected on the codex route (see \
                      NMF-86 — codex's own auth.json JWT is stripped). The supported host version \
                      is validated before launch.",
        after_help = "Examples:\n  \
                      nemo-relay codex\n  \
                      nemo-relay codex -- exec \"fix the bug in foo.rs\"\n  \
                      nemo-relay --openai-base-url https://inference-api.nvidia.com codex"
    )]
    Codex(EasyPathCommand),
    /// Run Hermes with observability (setup on first use)
    #[command(
        long_about = "Run Hermes Agent under an ephemeral NeMo Relay gateway. Persistent setup \
                      configures Hermes's user-level `mcp_servers` and shell hooks so bare Hermes \
                      processes can share the native Relay gateway on 127.0.0.1:47632. This \
                      wrapper temporarily suppresses that fixed MCP entry and uses a dynamic \
                      gateway for project-specific Relay configuration. Run \
                      `nemo-relay install hermes --force` to refresh the persistent integration.",
        after_help = "Examples:\n  \
                      nemo-relay hermes\n  \
                      nemo-relay hermes -- chat --provider custom"
    )]
    Hermes(EasyPathCommand),
    /// Keep a shared Relay gateway ready for an MCP client.
    #[command(
        long_about = "Start or reuse a shared native NeMo Relay gateway for an MCP stdio \
                      connection. The command acquires the gateway immediately, before reading \
                      MCP protocol frames. The gateway binds 127.0.0.1:47632 by default and MCP \
                      initialization completes only after Relay identity and readiness are \
                      verified. Multiple MCP clients share the gateway; it remains available \
                      until its idle timeout after the final client closes. This command \
                      advertises no MCP tools.",
        after_help = "Examples:\n  \
                      nemo-relay mcp\n  \
                      nemo-relay mcp --agent hermes\n  \
                      nemo-relay --bind 127.0.0.1:4041 mcp  # explicit standalone/test bind"
    )]
    Mcp(McpCommand),
    /// Run the interactive setup (writes `.nemo-relay/config.toml`)
    Config(ConfigCommand),
    /// Create or edit plugin configuration (writes `plugins.toml`)
    Plugins(PluginsCommand),
    /// Install coding-agent plugins from the local nemo-relay CLI.
    Install(InstallCommand),
    /// Uninstall coding-agent plugins installed by `nemo-relay install`.
    Uninstall(UninstallCommand),
    /// Validate and configure model pricing catalogs.
    ModelPricing(PricingCommand),
    /// Diagnose env, agents, config, observability (optionally scoped to one agent)
    Doctor(DoctorCommand),
    /// List supported and locally-detected agents (use `--json` for machine output)
    Agents(AgentsCommand),
    /// Print shell completion script (e.g. `nemo-relay completions zsh > ~/.zfunc/_nemo-relay`)
    Completions(CompletionsCommand),
    /// Run an agent deterministically (no wizard; errors if config is missing)
    Run(RunCommand),
    /// Internal: subprocess used by installed hooks to forward events. Not typed by humans.
    #[command(hide = true)]
    HookForward(HookForwardCommand),
}

/// Host identity for the lifecycle-bound MCP client.
#[derive(Debug, Clone, Args)]
pub(crate) struct McpCommand {
    /// Coding-agent host that launched this MCP client.
    #[arg(long, value_enum, default_value = "codex")]
    pub(crate) agent: CodingAgent,
}

/// Args for `nemo-relay doctor`. `--json` is on this command (rather than as a global flag)
/// so it doesn't pollute the help output of subcommands where it has no meaning.
#[derive(Debug, Clone, Args)]
pub(crate) struct DoctorCommand {
    /// Limit readiness checks to one supported agent.
    #[arg(value_enum)]
    pub(crate) agent: Option<CodingAgent>,
    /// Diagnose an installed coding-agent integration instead of the normal Relay config.
    #[arg(long, value_enum)]
    pub(crate) plugin: Option<IntegrationHost>,
    /// Plugin install state directory. Defaults to the platform data directory.
    #[arg(long)]
    pub(crate) install_dir: Option<PathBuf>,
    /// Emit machine-readable JSON instead of the formatted human report. Versioned via
    /// `schema_version`; stable shape for CI / evaluation harness consumption.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct InstallCommand {
    #[arg(value_enum)]
    pub(crate) host: IntegrationHost,
    #[arg(long)]
    pub(crate) install_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) force: bool,
    #[arg(long)]
    pub(crate) dry_run: bool,
    #[arg(long)]
    pub(crate) skip_doctor: bool,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct UninstallCommand {
    #[arg(value_enum)]
    pub(crate) host: IntegrationHost,
    #[arg(long)]
    pub(crate) install_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) dry_run: bool,
}

/// Args for `nemo-relay agents`. Shares the `--json` shape with `nemo-relay doctor`'s
/// `agents` field so the two outputs can be unified by downstream consumers.
#[derive(Debug, Clone, Args)]
pub(crate) struct AgentsCommand {
    /// Emit the supported + detected agent list as JSON instead of formatted text.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Args for `nemo-relay completions <shell>` (print to stdout) or `nemo-relay completions --install`
/// (auto-detect $SHELL and write to the standard fpath / completions directory).
///
/// The Homebrew / curl-install flows drop completion scripts automatically; this subcommand is
/// the escape hatch for CI, custom shells, regeneration, and `cargo install` users where no
/// post-install hook runs.
#[derive(Debug, Clone, Args)]
pub(crate) struct CompletionsCommand {
    /// Shell to generate the completion script for. Optional when used with `--install` (the
    /// installer auto-detects `$SHELL`).
    #[arg(value_enum)]
    pub(crate) shell: Option<clap_complete::Shell>,
    /// Write the completion script into the shell's standard completions directory instead of
    /// printing to stdout. Auto-detects `$SHELL` when no shell argument is given.
    #[arg(long)]
    pub(crate) install: bool,
}

/// Args for `nemo-relay config`. The setup wizard runs by default; `--reset` short-circuits to
/// a destructive clear. An optional positional agent name scopes both the wizard and `--reset`
/// to a single agent's settings, leaving other agents' blocks untouched.
#[derive(Debug, Clone, Args)]
pub(crate) struct ConfigCommand {
    /// Scope this run to one agent. Wizard skips the agent multi-select; `--reset` removes
    /// only that agent's block from the existing config file. Omit to operate on all agents.
    #[arg(value_enum)]
    pub(crate) agent: Option<CodingAgent>,
    /// Delete the project config file or the scoped agent block. A Hermes-scoped reset also
    /// removes Relay-owned MCP, hooks, and trust from the user Hermes config. The wizard does not
    /// run after reset; invoke `nemo-relay config` again to recreate configuration.
    #[arg(long)]
    pub(crate) reset: bool,
}

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

#[derive(Debug, Clone, Default, Args)]
pub(crate) struct ServerArgs {
    /// Path to an explicit config file (disables auto-discovery of workspace/global/system)
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,
    /// Address for the gateway to listen on in daemon mode (default 127.0.0.1:4040)
    #[arg(long, env = "NEMO_RELAY_GATEWAY_BIND")]
    pub(crate) bind: Option<SocketAddr>,
    /// Upstream OpenAI-compatible base URL (e.g. https://api.openai.com/v1, NVIDIA inference)
    #[arg(long, env = "NEMO_RELAY_OPENAI_BASE_URL")]
    pub(crate) openai_base_url: Option<String>,
    /// Upstream Anthropic base URL (e.g. https://api.anthropic.com)
    #[arg(long, env = "NEMO_RELAY_ANTHROPIC_BASE_URL")]
    pub(crate) anthropic_base_url: Option<String>,
    /// Internal override for the plugin configuration file.
    #[arg(long, env = "NEMO_RELAY_PLUGIN_CONFIG_PATH", hide = true)]
    pub(crate) plugin_config_path: Option<PathBuf>,
    /// Internal readiness file used by plugin sidecar bootstrap.
    #[arg(long, hide = true)]
    pub(crate) ready_file: Option<PathBuf>,
    /// Maximum accepted coding-agent hook payload size, in bytes.
    #[arg(long, env = "NEMO_RELAY_MAX_HOOK_PAYLOAD_BYTES")]
    pub(crate) max_hook_payload_bytes: Option<usize>,
    /// Maximum accepted provider passthrough request body size, in bytes.
    #[arg(long, env = "NEMO_RELAY_MAX_PASSTHROUGH_BODY_BYTES")]
    pub(crate) max_passthrough_body_bytes: Option<usize>,
}

impl ServerArgs {
    /// True when the user passed any flag that signals "I want the gateway, not the wizard." Used
    /// by the bare `nemo-relay` dispatch to choose between launching the long-running daemon and
    /// dropping into setup. `--config` is included: someone running `nemo-relay --config <path>`
    /// with no subcommand has explicitly pointed at a config file, which is only meaningful for
    /// daemon startup — the wizard creates configs, it doesn't consume them.
    pub(crate) fn requested_daemon_mode(&self) -> bool {
        self.bind.is_some()
            || self.openai_base_url.is_some()
            || self.anthropic_base_url.is_some()
            || self.plugin_config_path.is_some()
            || self.ready_file.is_some()
            || self.max_hook_payload_bytes.is_some()
            || self.max_passthrough_body_bytes.is_some()
            || self.config.is_some()
    }
}

pub(crate) const DEFAULT_MAX_HOOK_PAYLOAD_BYTES: usize = 20 * 1024 * 1024;
pub(crate) const DEFAULT_MAX_PASSTHROUGH_BODY_BYTES: usize = 100 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(crate) struct GatewayConfig {
    pub(crate) bind: SocketAddr,
    pub(crate) openai_base_url: String,
    pub(crate) anthropic_base_url: String,
    pub(crate) metadata: Option<Value>,
    pub(crate) plugin_config: Option<Value>,
    pub(crate) max_hook_payload_bytes: usize,
    pub(crate) max_passthrough_body_bytes: usize,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct HookForwardCommand {
    #[arg(value_enum)]
    pub(crate) agent: CodingAgent,
    #[arg(long)]
    pub(crate) gateway_url: Option<String>,
    #[arg(long)]
    pub(crate) profile: Option<String>,
    #[arg(long)]
    pub(crate) session_metadata: Option<String>,
    #[arg(long, value_enum)]
    pub(crate) gateway_mode: Option<GatewayMode>,
    #[arg(long)]
    pub(crate) fail_closed: bool,
}

/// Args for the easy-path agent shortcut (`nemo-relay claude`, `nemo-relay codex`, etc.).
/// Holds only pass-through agent args; the agent itself is selected by which subcommand variant
/// is invoked, and upstream settings come from the resolved config file. If no config file is
/// present, the dispatcher fires setup.
#[derive(Debug, Clone, Args)]
pub(crate) struct EasyPathCommand {
    /// Pass-through args forwarded to the underlying agent process. Use `--` to separate them
    /// from `nemo-relay`'s own flags. See the `Examples` section below for agent-specific shapes.
    #[arg(last = true)]
    pub(crate) command: Vec<String>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct RunCommand {
    #[arg(long, value_enum)]
    pub(crate) agent: Option<CodingAgent>,
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,
    #[arg(long)]
    pub(crate) openai_base_url: Option<String>,
    #[arg(long)]
    pub(crate) anthropic_base_url: Option<String>,
    #[arg(long)]
    pub(crate) session_metadata: Option<String>,
    /// Internal override for the plugin configuration file.
    #[arg(long, env = "NEMO_RELAY_PLUGIN_CONFIG_PATH", hide = true)]
    pub(crate) plugin_config_path: Option<PathBuf>,
    #[arg(long)]
    pub(crate) dry_run: bool,
    #[arg(long)]
    pub(crate) print: bool,
    #[arg(last = true)]
    pub(crate) command: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum IntegrationHost {
    Codex,
    #[value(name = "claude-code", alias = "claude")]
    ClaudeCode,
    Hermes,
    All,
}

impl IntegrationHost {
    pub(crate) const fn agent(self) -> Option<CodingAgent> {
        match self {
            Self::Codex => Some(CodingAgent::Codex),
            Self::ClaudeCode => Some(CodingAgent::ClaudeCode),
            Self::Hermes => Some(CodingAgent::Hermes),
            Self::All => None,
        }
    }

    pub(crate) const fn as_arg(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::ClaudeCode => "claude-code",
            Self::Hermes => "hermes",
            Self::All => "all",
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self.agent() {
            Some(agent) => agent.label(),
            None => "all",
        }
    }

    pub(crate) const fn executable(self) -> Option<&'static str> {
        match self.agent() {
            Some(agent) => Some(agent.executable()),
            None => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum GatewayMode {
    HookOnly,
    Passthrough,
    Required,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SessionConfig {
    pub(crate) metadata: Option<Value>,
    pub(crate) plugin_config: Option<Value>,
    pub(crate) profile: Option<String>,
    pub(crate) gateway_mode: Option<String>,
}

impl GatewayConfig {
    // Resolves per-session settings from hook/gateway headers with process config as fallback.
    // Header JSON fields are parsed opportunistically; invalid JSON is treated as absent here
    // because install and hook-forward validate generated header values before sending them.
    pub(crate) fn session_config_from_headers(&self, headers: &HeaderMap) -> SessionConfig {
        let metadata =
            header_json(headers, "x-nemo-relay-session-metadata").or_else(|| self.metadata.clone());
        let plugin_config = header_json(headers, "x-nemo-relay-plugin-config")
            .or_else(|| self.plugin_config.clone());
        let profile = header_string(headers, "x-nemo-relay-config-profile");
        let gateway_mode = header_string(headers, "x-nemo-relay-gateway-mode");
        SessionConfig {
            metadata,
            plugin_config,
            profile,
            gateway_mode,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ResolvedConfig {
    pub(crate) gateway: GatewayConfig,
    pub(crate) agents: AgentConfigs,
    pub(crate) dynamic_plugins: Vec<ResolvedDynamicPluginConfig>,
    pub(crate) dynamic_plugin_policy: DynamicPluginHostPolicy,
    pub(crate) bootstrap_fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedDynamicPluginConfig {
    pub(crate) plugin_id: String,
    pub(crate) manifest_ref: String,
    pub(crate) config: Map<String, Value>,
    pub(crate) has_explicit_config: bool,
    pub(crate) source: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Display, IntoStaticStr)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub(crate) enum DynamicPluginHostConfigStatus {
    Absent,
    Present,
}

impl ResolvedDynamicPluginConfig {
    pub(crate) fn host_config_status(&self) -> DynamicPluginHostConfigStatus {
        if self.has_explicit_config {
            DynamicPluginHostConfigStatus::Present
        } else {
            DynamicPluginHostConfigStatus::Absent
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AgentConfigs {
    pub(crate) claude: AgentCommandConfig,
    pub(crate) codex: AgentCommandConfig,
    pub(crate) hermes: AgentCommandConfig,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AgentCommandConfig {
    pub(crate) command: Option<String>,
    /// Recorded by `nemo-relay config` when it installs hermes shell hooks. Other agents leave
    /// this empty; the launcher reads it only to print a "hooks live here" pointer for hermes.
    pub(crate) hooks_path: Option<PathBuf>,
}

// TOML file shape grouped by user intent. Sections map 1:1 onto fields already present on
// `GatewayConfig` / `AgentConfigs`; plugin configuration lives in `plugins.toml`.
#[derive(Debug, Clone, Default, Deserialize)]
struct FileConfig {
    gateway: Option<FileGatewayConfig>,
    upstream: Option<FileUpstreamConfig>,
    agents: Option<FileAgentsConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileGatewayConfig {
    max_hook_payload_bytes: Option<usize>,
    max_passthrough_body_bytes: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileUpstreamConfig {
    openai_base_url: Option<String>,
    anthropic_base_url: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileAgentsConfig {
    // Keys match the agent's CLI invocation name (`claude`, `codex`, `hermes`) — the
    // word the user types at the shell — not the product name ("Claude Code") or the internal
    // `CodingAgent` enum kebab spelling. Same convention as the bare-agent shortcut in Phase 2.
    claude: Option<FileAgentCommandConfig>,
    codex: Option<FileAgentCommandConfig>,
    hermes: Option<FileAgentCommandConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileAgentCommandConfig {
    command: Option<String>,
    hooks_path: Option<PathBuf>,
}

impl Default for GatewayConfig {
    // Supplies conservative local gateway defaults: bind only to loopback, route OpenAI and
    // Anthropic requests to their public bases, and leave plugins disabled until config,
    // environment, or headers explicitly opt in.
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:4040"
                .parse()
                .expect("valid default bind address"),
            openai_base_url: "https://api.openai.com/v1".into(),
            anthropic_base_url: "https://api.anthropic.com".into(),
            metadata: None,
            plugin_config: None,
            max_hook_payload_bytes: DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
            max_passthrough_body_bytes: DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
        }
    }
}

/// Resolves server-mode configuration from shared config files plus server CLI/environment overrides.
///
/// File discovery and merge behavior live in `load_shared_config`; this function only applies the
/// server-facing command-line layer so launcher-only settings cannot leak into daemon mode.
pub(crate) fn resolve_server_config(args: &ServerArgs) -> Result<ResolvedConfig, CliError> {
    let mut resolved = load_shared_config(args.config.as_ref(), args.plugin_config_path.as_ref())?;
    apply_server_overrides(&mut resolved.gateway, args)?;
    enforce_required_dynamic_plugin_startup(args.config.as_ref(), &resolved)?;
    Ok(resolved)
}

/// Resolves the shared plugin MCP gateway from system and user layers only.
pub(crate) fn resolve_persistent_server_config(
    args: &ServerArgs,
) -> Result<ResolvedConfig, CliError> {
    if args.config.is_some() || args.plugin_config_path.is_some() || args.ready_file.is_some() {
        return Err(CliError::Config(
            "nemo-relay mcp uses system and user configuration only; use `nemo-relay run` for explicit or project configuration"
                .into(),
        ));
    }
    let mut resolved = load_shared_config_scoped(None, None, true)?;
    apply_server_overrides(&mut resolved.gateway, args)?;
    let active_dynamic_plugins = active_dynamic_plugin_components_for_identity(None, &resolved)?;
    resolved.bootstrap_fingerprint = Some(persistent_bootstrap_fingerprint(
        &resolved,
        &active_dynamic_plugins,
    )?);
    Ok(resolved)
}

/// Parent-computed identity and inputs needed to reverify a managed persistent gateway child.
#[derive(Debug, Clone)]
pub(crate) struct ManagedBootstrapIdentity {
    expected: String,
    persistent_args: ServerArgs,
    resolved: ResolvedConfig,
    active_dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
}

impl ManagedBootstrapIdentity {
    pub(crate) fn fingerprint(&self) -> &str {
        &self.expected
    }

    pub(crate) fn verify_current(&self) -> Result<(), CliError> {
        let snapshot_actual =
            persistent_bootstrap_fingerprint(&self.resolved, &self.active_dynamic_plugins)?;
        verify_managed_bootstrap_fingerprint(&self.expected, &snapshot_actual)?;
        let resolved = resolve_persistent_server_config(&self.persistent_args)?;
        let actual = resolved
            .bootstrap_fingerprint
            .expect("persistent gateway resolution sets a bootstrap fingerprint");
        verify_managed_bootstrap_fingerprint(&self.expected, &actual)
    }
}

/// Verifies and retains the parent-computed identity for a managed persistent gateway child.
///
/// Ordinary daemon launches remain stateless: the internal ready-file contract identifies a child
/// spawned by the plugin bootstrap path. The child recomputes identity from the configuration and
/// active lifecycle records it is about to activate before publishing ownership or readiness.
pub(crate) fn managed_bootstrap_identity(
    args: &ServerArgs,
    resolved: &ResolvedConfig,
    active_dynamic_plugins: &[ActiveDynamicPluginComponent],
) -> Result<Option<ManagedBootstrapIdentity>, CliError> {
    if args.ready_file.is_none() {
        return Ok(None);
    }
    let Some(expected) = env::var(BOOTSTRAP_FINGERPRINT_ENV)
        .ok()
        .filter(|fingerprint| !fingerprint.is_empty())
    else {
        return Ok(None);
    };
    let actual = persistent_bootstrap_fingerprint(resolved, active_dynamic_plugins)?;
    verify_managed_bootstrap_fingerprint(&expected, &actual)?;
    let mut persistent_args = args.clone();
    persistent_args.ready_file = None;
    Ok(Some(ManagedBootstrapIdentity {
        expected,
        persistent_args,
        resolved: resolved.clone(),
        active_dynamic_plugins: active_dynamic_plugins.to_vec(),
    }))
}

fn verify_managed_bootstrap_fingerprint(expected: &str, actual: &str) -> Result<(), CliError> {
    if actual == expected {
        return Ok(());
    }
    Err(CliError::Config(
        "persistent gateway identity changed during managed bootstrap; retry so the parent can resolve the current configuration"
            .into(),
    ))
}

fn persistent_bootstrap_fingerprint(
    resolved: &ResolvedConfig,
    active_dynamic_plugins: &[ActiveDynamicPluginComponent],
) -> Result<String, CliError> {
    let dynamic_plugins = active_dynamic_plugins
        .iter()
        .map(dynamic_plugin_bootstrap_identity)
        .collect::<Result<Vec<_>, _>>()?;
    let gateway = &resolved.gateway;
    let idle_timeout_secs = crate::sidecar::plugin_idle_timeout()
        .map_err(CliError::Config)?
        .as_secs();
    let document = serde_json::json!({
        "bootstrap_protocol": 1,
        "relay_version": env!("CARGO_PKG_VERSION"),
        "openai_base_url": gateway.openai_base_url,
        "anthropic_base_url": gateway.anthropic_base_url,
        "metadata": gateway.metadata,
        "plugin_config": gateway.plugin_config,
        "max_hook_payload_bytes": gateway.max_hook_payload_bytes,
        "max_passthrough_body_bytes": gateway.max_passthrough_body_bytes,
        "plugin_idle_timeout_secs": idle_timeout_secs,
        "dynamic_plugins": dynamic_plugins,
        "dynamic_plugin_policy": format!("{:?}", resolved.dynamic_plugin_policy),
    });
    let key = load_or_create_bootstrap_hmac_key()?;
    let key = hmac::Key::new(hmac::HMAC_SHA256, &key);
    let mut digest = hmac::Context::with_key(&key);
    digest.update(
        &serde_json::to_vec(&document).expect("persistent gateway fingerprint serializes to JSON"),
    );
    let environment = env::vars_os().filter_map(|(name, _)| name.into_string().ok());
    for name in crate::mcp_environment::forwarded_names(environment, gateway.plugin_config.as_ref())
    {
        if name == PLUGIN_IDLE_TIMEOUT_ENV {
            continue;
        }
        digest.update(&[0]);
        digest.update(name.as_bytes());
        digest.update(&[0]);
        if let Some(value) = env::var_os(&name) {
            digest.update(value.to_string_lossy().as_bytes());
        }
    }
    let tag = digest.sign();
    Ok(format!(
        "hmac-sha256:{}",
        tag.as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

fn dynamic_plugin_bootstrap_identity(
    plugin: &ActiveDynamicPluginComponent,
) -> Result<Value, CliError> {
    let manifest_identity = match (&plugin.activation_snapshot, plugin.manifest_ref.as_deref()) {
        (Some(snapshot), _) => Some(dynamic_plugin_snapshot_identity(snapshot)?),
        (None, Some(manifest_ref)) => Some(dynamic_plugin_manifest_identity(
            manifest_ref,
            plugin.environment_ref.as_deref(),
        )?),
        (None, None) => None,
    };
    Ok(serde_json::json!({
        "plugin_id": plugin.plugin_id,
        "kind": format!("{:?}", plugin.kind),
        "lifecycle_generation": plugin.lifecycle_generation,
        "manifest": manifest_identity,
        "environment_ref": plugin.environment_ref,
        "config": plugin.config,
    }))
}

fn dynamic_plugin_snapshot_identity(
    snapshot: &crate::plugins::lifecycle::DynamicPluginActivationSnapshot,
) -> Result<Value, CliError> {
    let (manifest, _) = load_bounded_dynamic_plugin_manifest(snapshot.identity_manifest())?;
    let manifest_path = PathBuf::from(snapshot.original_manifest_ref());
    let manifest_digest = bootstrap_file_digest(
        snapshot.identity_manifest(),
        "dynamic plugin manifest snapshot",
    )?;
    let artifact_ref = manifest
        .source
        .as_ref()
        .and_then(|source| source.artifact.as_deref())
        .or(match &manifest.load {
            DynamicPluginManifestLoad::RustDynamic(load) => load.library.as_deref(),
            DynamicPluginManifestLoad::Worker(_) => None,
        });
    let artifact = artifact_ref
        .map(|artifact_ref| {
            let logical_path = resolve_dynamic_plugin_relative_path(&manifest_path, artifact_ref);
            let snapshot_path = snapshot.identity_file(&logical_path).ok_or_else(|| {
                CliError::Config(format!(
                    "dynamic plugin activation snapshot is missing artifact {}",
                    logical_path.display()
                ))
            })?;
            bootstrap_file_digest(snapshot_path, "dynamic plugin artifact snapshot")
                .map(|digest| serde_json::json!({ "path": logical_path, "sha256": digest }))
        })
        .transpose()?;
    let signature = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.signature.as_deref())
        .map(|signature_ref| {
            let logical_path = resolve_dynamic_plugin_relative_path(&manifest_path, signature_ref);
            let snapshot_path = snapshot.identity_file(&logical_path).ok_or_else(|| {
                CliError::Config(format!(
                    "dynamic plugin activation snapshot is missing signature {}",
                    logical_path.display()
                ))
            })?;
            bootstrap_file_digest(snapshot_path, "dynamic plugin signature snapshot")
                .map(|digest| serde_json::json!({ "path": logical_path, "sha256": digest }))
        })
        .transpose()?;
    Ok(serde_json::json!({
        "path": snapshot.original_manifest_ref(),
        "sha256": manifest_digest,
        "artifact": artifact,
        "signature": signature,
        "runtime_closure_sha256": snapshot.closure_digest(),
    }))
}

fn dynamic_plugin_manifest_identity(
    manifest_ref: &str,
    environment_ref: Option<&str>,
) -> Result<Value, CliError> {
    let (manifest, normalized_ref) = load_bounded_dynamic_plugin_manifest(manifest_ref)?;
    let manifest_path = PathBuf::from(&normalized_ref);
    let manifest_digest = bootstrap_file_digest(&manifest_path, "dynamic plugin manifest")?;
    let artifact_ref = manifest
        .source
        .as_ref()
        .and_then(|source| source.artifact.as_deref())
        .or(match &manifest.load {
            DynamicPluginManifestLoad::RustDynamic(load) => load.library.as_deref(),
            DynamicPluginManifestLoad::Worker(_) => None,
        });
    let artifact = artifact_ref
        .map(|artifact_ref| {
            let path = resolve_dynamic_plugin_relative_path(&manifest_path, artifact_ref);
            bootstrap_file_digest(&path, "dynamic plugin artifact")
                .map(|digest| serde_json::json!({ "path": path, "sha256": digest }))
        })
        .transpose()?;
    let signature = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.signature.as_deref())
        .map(|signature_ref| {
            let path = resolve_dynamic_plugin_relative_path(&manifest_path, signature_ref);
            bootstrap_file_digest(&path, "dynamic plugin signature")
                .map(|digest| serde_json::json!({ "path": path, "sha256": digest }))
        })
        .transpose()?;
    let closure_digest = dynamic_plugin_runtime_closure_digest(&normalized_ref, environment_ref)?;
    Ok(serde_json::json!({
        "path": normalized_ref,
        "sha256": manifest_digest,
        "artifact": artifact,
        "signature": signature,
        "runtime_closure_sha256": closure_digest,
    }))
}

fn resolve_dynamic_plugin_relative_path(manifest_path: &Path, reference: &str) -> PathBuf {
    let path = PathBuf::from(reference);
    if path.is_absolute() {
        path
    } else {
        manifest_path
            .parent()
            .map(|parent| parent.join(&path))
            .unwrap_or(path)
    }
}

fn bootstrap_file_digest(path: &Path, description: &str) -> Result<String, CliError> {
    let mut context = digest::Context::new(&digest::SHA256);
    stream_bounded_regular_file(path, description, |bytes| context.update(bytes))
        .map_err(CliError::Config)?;
    Ok(context
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub(crate) fn load_bounded_dynamic_plugin_manifest(
    path: impl AsRef<Path>,
) -> Result<(DynamicPluginManifest, String), CliError> {
    let (manifest, normalized, _) = load_bounded_dynamic_plugin_manifest_bytes(path)?;
    Ok((manifest, normalized))
}

pub(crate) fn load_bounded_dynamic_plugin_manifest_bytes(
    path: impl AsRef<Path>,
) -> Result<(DynamicPluginManifest, String, Vec<u8>), CliError> {
    let path = path.as_ref();
    let manifest_path = if path.is_dir() {
        path.join(DYNAMIC_PLUGIN_MANIFEST_FILENAME)
    } else {
        path.to_path_buf()
    };
    let normalized = fs::canonicalize(&manifest_path).map_err(|error| {
        CliError::Config(format!(
            "failed to normalize dynamic plugin manifest {}: {error}",
            manifest_path.display()
        ))
    })?;
    let bytes = read_bounded_regular_file(&normalized, "dynamic plugin manifest")
        .map_err(CliError::Config)?;
    let contents = std::str::from_utf8(&bytes).map_err(|error| {
        CliError::Config(format!(
            "dynamic plugin manifest {} is not UTF-8: {error}",
            normalized.display()
        ))
    })?;
    let manifest = DynamicPluginManifest::parse_toml(contents)
        .map_err(|error| CliError::Config(error.to_string()))?;
    Ok((manifest, normalized.to_string_lossy().into_owned(), bytes))
}

pub(crate) fn read_bounded_regular_file(path: &Path, description: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    stream_bounded_regular_file(path, description, |chunk| bytes.extend_from_slice(chunk))?;
    Ok(bytes)
}

pub(crate) fn stream_bounded_regular_file(
    path: &Path,
    description: &str,
    mut consume: impl FnMut(&[u8]),
) -> Result<(), String> {
    const BUFFER_BYTES: usize = 64 * 1024;
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        format!(
            "failed to inspect {description} {} for persistent gateway identity: {error}",
            path.display()
        )
    })?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "{description} {} must be a regular file for persistent gateway identity",
            path.display()
        ));
    }
    if metadata.len() > MAX_BOOTSTRAP_IDENTITY_FILE_BYTES {
        return Err(format!(
            "{description} {} exceeds the {MAX_BOOTSTRAP_IDENTITY_FILE_BYTES}-byte persistent gateway identity budget",
            path.display()
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path).map_err(|error| {
        format!(
            "failed to read {description} {} for persistent gateway identity: {error}",
            path.display()
        )
    })?;
    let opened_metadata = file.metadata().map_err(|error| {
        format!(
            "failed to inspect {description} {} for persistent gateway identity: {error}",
            path.display()
        )
    })?;
    if !opened_metadata.file_type().is_file() {
        return Err(format!(
            "{description} {} must be a regular file for persistent gateway identity",
            path.display()
        ));
    }
    if opened_metadata.len() > MAX_BOOTSTRAP_IDENTITY_FILE_BYTES {
        return Err(format!(
            "{description} {} exceeds the {MAX_BOOTSTRAP_IDENTITY_FILE_BYTES}-byte persistent gateway identity budget",
            path.display()
        ));
    }
    let mut buffer = [0_u8; BUFFER_BYTES];
    let mut total = 0_u64;
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            format!(
                "failed to read {description} {} for persistent gateway identity: {error}",
                path.display()
            )
        })?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if total > MAX_BOOTSTRAP_IDENTITY_FILE_BYTES {
            return Err(format!(
                "{description} {} exceeds the {MAX_BOOTSTRAP_IDENTITY_FILE_BYTES}-byte persistent gateway identity budget",
                path.display()
            ));
        }
        consume(&buffer[..read]);
    }
    Ok(())
}

const BOOTSTRAP_HMAC_KEY_BYTES: usize = 32;
const BOOTSTRAP_HMAC_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const BOOTSTRAP_CHALLENGE_DOMAIN: &[u8] = b"nemo-relay/bootstrap-health/v1\0";
const PYTHON_ENVIRONMENT_ATTESTATION_DOMAIN: &[u8] =
    b"nemo-relay/python-environment-attestation/v1\0";

/// Per-user secret used to authenticate a managed bootstrap listener without exposing key bytes.
#[derive(Clone)]
pub(crate) struct BootstrapChallengeKey(hmac::Key);

impl BootstrapChallengeKey {
    pub(crate) fn load() -> Result<Self, CliError> {
        Ok(Self(hmac::Key::new(
            hmac::HMAC_SHA256,
            &load_or_create_bootstrap_hmac_key()?,
        )))
    }

    pub(crate) fn proof(&self, fingerprint: &str, nonce: &str) -> String {
        let mut context = hmac::Context::with_key(&self.0);
        context.update(BOOTSTRAP_CHALLENGE_DOMAIN);
        context.update(fingerprint.as_bytes());
        context.update(&[0]);
        context.update(nonce.as_bytes());
        let tag = context.sign();
        format!(
            "hmac-sha256:{}",
            tag.as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        )
    }

    pub(crate) fn verify(&self, fingerprint: &str, nonce: &str, proof: &str) -> bool {
        let Some(encoded) = proof.strip_prefix("hmac-sha256:") else {
            return false;
        };
        let Some(tag) = decode_fixed_hex::<32>(encoded) else {
            return false;
        };
        let mut message = Vec::with_capacity(
            BOOTSTRAP_CHALLENGE_DOMAIN.len() + fingerprint.len() + nonce.len() + 1,
        );
        message.extend_from_slice(BOOTSTRAP_CHALLENGE_DOMAIN);
        message.extend_from_slice(fingerprint.as_bytes());
        message.push(0);
        message.extend_from_slice(nonce.as_bytes());
        hmac::verify(&self.0, &message, &tag).is_ok()
    }

    #[cfg(test)]
    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
        Self(hmac::Key::new(hmac::HMAC_SHA256, bytes))
    }
}

fn decode_fixed_hex<const N: usize>(encoded: &str) -> Option<[u8; N]> {
    if encoded.len() != N * 2 || !encoded.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let mut decoded = [0_u8; N];
    for (index, byte) in decoded.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(decoded)
}

pub(crate) fn sign_python_environment_attestation(
    source_artifact_sha256: &str,
    environment_sha256: &str,
) -> Result<String, CliError> {
    let key = hmac::Key::new(hmac::HMAC_SHA256, &load_or_create_bootstrap_hmac_key()?);
    let message =
        python_environment_attestation_message(source_artifact_sha256, environment_sha256);
    let tag = hmac::sign(&key, &message);
    Ok(format!(
        "hmac-sha256:{}",
        tag.as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

pub(crate) fn verify_python_environment_attestation(
    source_artifact_sha256: &str,
    environment_sha256: &str,
    authentication: &str,
) -> Result<bool, CliError> {
    let Some(encoded) = authentication.strip_prefix("hmac-sha256:") else {
        return Ok(false);
    };
    let Some(tag) = decode_fixed_hex::<32>(encoded) else {
        return Ok(false);
    };
    let key = hmac::Key::new(hmac::HMAC_SHA256, &load_or_create_bootstrap_hmac_key()?);
    Ok(hmac::verify(
        &key,
        &python_environment_attestation_message(source_artifact_sha256, environment_sha256),
        &tag,
    )
    .is_ok())
}

fn python_environment_attestation_message(
    source_artifact_sha256: &str,
    environment_sha256: &str,
) -> Vec<u8> {
    let mut message = Vec::with_capacity(
        PYTHON_ENVIRONMENT_ATTESTATION_DOMAIN.len()
            + source_artifact_sha256.len()
            + environment_sha256.len()
            + 1,
    );
    message.extend_from_slice(PYTHON_ENVIRONMENT_ATTESTATION_DOMAIN);
    message.extend_from_slice(source_artifact_sha256.trim().as_bytes());
    message.push(0);
    message.extend_from_slice(environment_sha256.as_bytes());
    message
}

fn load_or_create_bootstrap_hmac_key() -> Result<[u8; BOOTSTRAP_HMAC_KEY_BYTES], CliError> {
    let path = user_config_dir()
        .map(|directory| directory.join("bootstrap").join("fingerprint-hmac.key"))
        .ok_or_else(|| {
            CliError::Config(
                "cannot determine the per-user NeMo Relay bootstrap state directory; set HOME or USERPROFILE"
                    .into(),
            )
        })?;
    load_or_create_bootstrap_hmac_key_at(&path)
}

fn load_or_create_bootstrap_hmac_key_at(
    path: &Path,
) -> Result<[u8; BOOTSTRAP_HMAC_KEY_BYTES], CliError> {
    load_or_create_bootstrap_hmac_key_at_with_timeout(path, BOOTSTRAP_HMAC_LOCK_TIMEOUT)
}

fn load_or_create_bootstrap_hmac_key_at_with_timeout(
    path: &Path,
    lock_timeout: Duration,
) -> Result<[u8; BOOTSTRAP_HMAC_KEY_BYTES], CliError> {
    let parent = path.parent().ok_or_else(|| {
        CliError::Config(format!(
            "bootstrap HMAC key path {} has no parent directory",
            path.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(|error| {
        CliError::Config(format!(
            "failed to create bootstrap state directory {}: {error}",
            parent.display()
        ))
    })?;
    #[cfg(unix)]
    fs::set_permissions(parent, {
        use std::os::unix::fs::PermissionsExt;
        fs::Permissions::from_mode(0o700)
    })
    .map_err(|error| {
        CliError::Config(format!(
            "failed to protect bootstrap state directory {}: {error}",
            parent.display()
        ))
    })?;

    let mut options = OpenOptions::new();
    options.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|error| {
        CliError::Config(format!(
            "failed to open bootstrap HMAC key {}: {error}",
            path.display()
        ))
    })?;
    let lock_deadline = Instant::now() + lock_timeout;
    loop {
        match try_lock_exclusive(&file) {
            Ok(LockAttempt::Acquired) => break,
            Ok(LockAttempt::Contended) => {
                if Instant::now() >= lock_deadline {
                    return Err(CliError::Config(format!(
                        "timed out waiting for bootstrap HMAC key lock {}",
                        path.display()
                    )));
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(error) => {
                return Err(CliError::Config(format!(
                    "failed to lock bootstrap HMAC key {}: {error}",
                    path.display()
                )));
            }
        }
    }
    #[cfg(unix)]
    file.set_permissions({
        use std::os::unix::fs::PermissionsExt;
        fs::Permissions::from_mode(0o600)
    })
    .map_err(|error| {
        CliError::Config(format!(
            "failed to protect bootstrap HMAC key {}: {error}",
            path.display()
        ))
    })?;

    let length = file
        .metadata()
        .map_err(|error| {
            CliError::Config(format!(
                "failed to inspect bootstrap HMAC key {}: {error}",
                path.display()
            ))
        })?
        .len();
    if length == 0 {
        let mut key = [0_u8; BOOTSTRAP_HMAC_KEY_BYTES];
        SystemRandom::new()
            .fill(&mut key)
            .map_err(|_| CliError::Config("failed to generate bootstrap HMAC key".into()))?;
        file.write_all(&key).map_err(|error| {
            CliError::Config(format!(
                "failed to write bootstrap HMAC key {}: {error}",
                path.display()
            ))
        })?;
        file.sync_all().map_err(|error| {
            CliError::Config(format!(
                "failed to persist bootstrap HMAC key {}: {error}",
                path.display()
            ))
        })?;
        return Ok(key);
    }
    if length != BOOTSTRAP_HMAC_KEY_BYTES as u64 {
        return Err(CliError::Config(format!(
            "bootstrap HMAC key {} has invalid length {length}; expected {BOOTSTRAP_HMAC_KEY_BYTES} bytes",
            path.display()
        )));
    }
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        CliError::Config(format!(
            "failed to read bootstrap HMAC key {}: {error}",
            path.display()
        ))
    })?;
    let mut key = [0_u8; BOOTSTRAP_HMAC_KEY_BYTES];
    file.read_exact(&mut key).map_err(|error| {
        CliError::Config(format!(
            "failed to read bootstrap HMAC key {}: {error}",
            path.display()
        ))
    })?;
    Ok(key)
}

/// Resolves shared config for plugin-facing CLI commands without mutating gateway runtime fields.
pub(crate) fn resolve_plugins_config(
    explicit: Option<&PathBuf>,
) -> Result<ResolvedConfig, CliError> {
    load_shared_config(explicit, None)
}

/// Resolves transparent `run` configuration and switches the gateway to an ephemeral bind address.
///
/// Explicit run arguments override inherited top-level server flags, which override shared config.
/// Session metadata and plugin config are parsed as JSON here so malformed CLI values fail before
/// the child agent is spawned.
pub(crate) fn resolve_run_config(
    command: &RunCommand,
    inherited: Option<&ServerArgs>,
) -> Result<ResolvedConfig, CliError> {
    let config = command
        .config
        .as_ref()
        .or_else(|| inherited.and_then(|args| args.config.as_ref()));
    let plugin_config_path = command
        .plugin_config_path
        .as_ref()
        .or_else(|| inherited.and_then(|args| args.plugin_config_path.as_ref()));
    let mut resolved = load_shared_config(config, plugin_config_path)?;
    if let Some(args) = inherited {
        apply_server_overrides(&mut resolved.gateway, args)?;
    }
    apply_run_overrides(&mut resolved.gateway, command)?;
    resolved.gateway.bind = "127.0.0.1:0"
        .parse()
        .expect("valid transparent bind address");
    if !command.dry_run {
        enforce_required_dynamic_plugin_startup(config, &resolved)?;
    }
    Ok(resolved)
}

// Applies subcommand-specific `run` overrides after inherited top-level flags. JSON-bearing fields
// are parsed here so invalid metadata or plugin config fails before the gateway binds a port.
fn apply_run_overrides(config: &mut GatewayConfig, command: &RunCommand) -> Result<(), CliError> {
    apply_run_url_overrides(config, command);
    apply_run_json_overrides(config, command)?;
    Ok(())
}

// Applies plain string/path run overrides. These fields do not need parsing, so they stay separate
// from JSON options whose errors should include field context.
fn apply_run_url_overrides(config: &mut GatewayConfig, command: &RunCommand) {
    if let Some(value) = &command.openai_base_url {
        config.openai_base_url = value.clone();
    }
    if let Some(value) = &command.anthropic_base_url {
        config.anthropic_base_url = value.clone();
    }
}

// Parses JSON-bearing run overrides after simple values. Invalid metadata or plugin config fails
// before transparent run mode binds its ephemeral gateway listener.
fn apply_run_json_overrides(
    config: &mut GatewayConfig,
    command: &RunCommand,
) -> Result<(), CliError> {
    if let Some(value) = &command.session_metadata {
        config.metadata = Some(parse_json_option("session metadata", value)?);
    }
    Ok(())
}

// Applies direct server flags on top of already-merged configuration. Only present options mutate
// the config so lower-priority file values survive when a flag was omitted.
fn apply_server_overrides(config: &mut GatewayConfig, args: &ServerArgs) -> Result<(), CliError> {
    if let Some(value) = args.bind {
        config.bind = value;
    }
    if let Some(value) = &args.openai_base_url {
        config.openai_base_url = value.clone();
    }
    if let Some(value) = &args.anthropic_base_url {
        config.anthropic_base_url = value.clone();
    }
    if let Some(value) = args.max_hook_payload_bytes {
        config.max_hook_payload_bytes = validate_body_limit("max hook payload bytes", value)?;
    }
    if let Some(value) = args.max_passthrough_body_bytes {
        config.max_passthrough_body_bytes =
            validate_body_limit("max passthrough body bytes", value)?;
    }
    Ok(())
}

pub(crate) const PLUGINS_TOML: &str = "plugins.toml";

// Loads config from the ordered shared locations, deep-merges TOML tables, maps the typed file
// shape onto runtime structs, applies a sibling/discovered plugins.toml when present, then lets
// environment variables override file values. Invalid TOML or typed shapes fail closed because
// they indicate an operator configuration error.
fn load_shared_config(
    explicit: Option<&PathBuf>,
    plugin_config_path: Option<&PathBuf>,
) -> Result<ResolvedConfig, CliError> {
    load_shared_config_scoped(explicit, plugin_config_path, user_config_scope())
}

fn load_shared_config_scoped(
    explicit: Option<&PathBuf>,
    plugin_config_path: Option<&PathBuf>,
    user_only: bool,
) -> Result<ResolvedConfig, CliError> {
    let mut merged = toml::Value::Table(toml::map::Map::new());
    for path in config_paths_scoped(explicit, user_only) {
        let Some(raw) = read_config_file(&path, explicit.is_some(), "configuration")? else {
            continue;
        };
        let parsed = raw
            .parse::<toml::Table>()
            .map(toml::Value::Table)
            .map_err(|error| {
                CliError::Config(format!("invalid TOML in {}: {error}", path.display()))
            })?;
        let legacy_observability = legacy_observability_sections(&parsed);
        if !legacy_observability.is_empty() {
            return Err(CliError::Config(format!(
                "legacy observability config in {} is no longer supported: {}; configure \
                 observability in plugins.toml with `nemo-relay plugins edit`",
                path.display(),
                legacy_observability.join(", ")
            )));
        }
        if parsed.get("plugins").is_some() {
            return Err(CliError::Config(format!(
                "plugin configuration in {} is no longer supported; move it to plugins.toml",
                path.display()
            )));
        }
        merge_toml(&mut merged, parsed);
    }
    let plugin_toml = load_plugin_toml_config_scoped(explicit, plugin_config_path, user_only)?;
    let mut resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        ..ResolvedConfig::default()
    };
    apply_file_config(&mut resolved, merged)?;
    apply_plugin_toml_config(&mut resolved, plugin_toml);
    apply_env_config(&mut resolved.gateway)?;
    Ok(resolved)
}

fn read_config_file(
    path: &Path,
    required: bool,
    description: &str,
) -> Result<Option<String>, CliError> {
    match path.try_exists() {
        Ok(false) if !required => Ok(None),
        Ok(false) => Err(CliError::Config(format!(
            "explicit {description} file {} does not exist",
            path.display()
        ))),
        Err(error) => Err(CliError::Config(format!(
            "failed to inspect {description} file {}: {error}",
            path.display()
        ))),
        Ok(true) => std::fs::read_to_string(path).map(Some).map_err(|error| {
            CliError::Config(format!(
                "failed to read {description} file {}: {error}",
                path.display()
            ))
        }),
    }
}

/// Returns true if any of the implicit config file locations exists on disk. Used by the
/// easy-path dispatcher to decide whether to launch setup (no config found) or proceed
/// with config-driven settings. Mirrors `config_paths(None)` but only checks existence.
pub(crate) fn any_config_file_exists() -> bool {
    config_paths(None).iter().any(|path| path.exists())
}

// Returns the config search path. An explicit path disables implicit discovery; otherwise system
// config is lowest priority, the nearest project config is next, and user config is merged last.
fn config_paths(explicit: Option<&PathBuf>) -> Vec<PathBuf> {
    config_paths_scoped(explicit, user_config_scope())
}

fn config_paths_scoped(explicit: Option<&PathBuf>, user_only: bool) -> Vec<PathBuf> {
    if let Some(path) = explicit {
        return vec![path.clone()];
    }
    let mut paths = vec![PathBuf::from("/etc/nemo-relay/config.toml")];
    if !user_only
        && let Ok(cwd) = std::env::current_dir()
        && let Some(project) = find_project_config(&cwd)
    {
        paths.push(project);
    }
    if let Some(user) = user_config_path() {
        paths.push(user);
    }
    paths
}

// Returns the plugin config search path. An explicit gateway config path scopes plugins.toml to the
// same directory so `--config path/to/config.toml` can be extended by `path/to/plugins.toml` without
// reading unrelated implicit project/user/global plugin files.
fn plugin_config_paths(
    explicit: Option<&PathBuf>,
    plugin_config_path: Option<&PathBuf>,
) -> Vec<PathBuf> {
    plugin_config_paths_scoped(explicit, plugin_config_path, user_config_scope())
}

fn plugin_config_paths_scoped(
    explicit: Option<&PathBuf>,
    plugin_config_path: Option<&PathBuf>,
    user_only: bool,
) -> Vec<PathBuf> {
    if let Some(path) = plugin_config_path {
        return vec![path.clone()];
    }
    if let Some(path) = explicit {
        return path
            .parent()
            .map(|parent| vec![parent.join(PLUGINS_TOML)])
            .unwrap_or_default();
    }
    if user_only {
        return implicit_plugin_config_paths(None, user_config_dir());
    }
    implicit_plugin_config_paths(std::env::current_dir().ok().as_deref(), user_config_dir())
}

fn user_config_scope() -> bool {
    std::env::var("NEMO_RELAY_CONFIG_SCOPE").ok().as_deref() == Some("user")
}

/// Returns the implicit `plugins.toml` discovery paths used by the gateway and doctor.
pub(crate) fn default_plugin_config_paths() -> Vec<PathBuf> {
    plugin_config_paths(None, None)
}

fn implicit_plugin_config_paths(
    cwd: Option<&std::path::Path>,
    user_config_dir: Option<PathBuf>,
) -> Vec<PathBuf> {
    // The search-path logic lives in core; the gateway shares it so discovery stays identical.
    nemo_relay::plugin::default_plugin_config_paths(cwd, user_config_dir)
}

// Walks upward from the current directory and returns the nearest project-local gateway config.
// The first hit wins so nested projects can override parent workspace defaults.
fn find_project_config(start: &std::path::Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        let path = ancestor.join(".nemo-relay/config.toml");
        if path.exists() {
            return Some(path);
        }
    }
    None
}

// The project-walk lives in core; the gateway shares it so discovery stays identical.
fn find_project_plugin_config(start: &std::path::Path) -> Option<PathBuf> {
    nemo_relay::plugin::nearest_project_plugin_config(start)
}

pub(crate) fn user_plugin_config_path() -> Option<PathBuf> {
    user_config_dir().map(|dir| dir.join(PLUGINS_TOML))
}

pub(crate) fn user_plugin_runtime_config() -> Result<Option<Value>, CliError> {
    Ok(
        load_plugin_toml_config_from_paths(implicit_plugin_config_paths(None, user_config_dir()))?
            .and_then(|config| config.value),
    )
}

pub(crate) fn project_plugin_config_path(start: &std::path::Path) -> PathBuf {
    find_project_plugin_config(start)
        .or_else(|| {
            find_project_config(start)
                .and_then(|path| path.parent().map(|parent| parent.join(PLUGINS_TOML)))
        })
        .unwrap_or_else(|| start.join(".nemo-relay").join(PLUGINS_TOML))
}

pub(crate) fn global_plugin_config_path() -> PathBuf {
    PathBuf::from("/etc/nemo-relay").join(PLUGINS_TOML)
}

// Resolves the user config using XDG first and HOME/USERPROFILE second. Returning `None` keeps
// config loading portable in minimal environments where no home directory is visible.
fn user_config_path() -> Option<PathBuf> {
    user_config_dir().map(|dir| dir.join("config.toml"))
}

/// Resolves the nemo-relay user config DIRECTORY (without trailing filename). Delegates to core's
/// resolver so the gateway, the editor, and the plugin runtime agree on the location.
pub(crate) fn user_config_dir() -> Option<PathBuf> {
    nemo_relay::plugin::user_config_dir()
}

// Applies the typed TOML config model to the resolved runtime config. Missing sections and fields
// are ignored, preserving defaults and prior merge layers.
fn apply_file_config(resolved: &mut ResolvedConfig, value: toml::Value) -> Result<(), CliError> {
    let config: FileConfig = value.try_into().map_err(|error| {
        CliError::Config(format!("invalid gateway configuration shape: {error}"))
    })?;
    apply_file_gateway_config(&mut resolved.gateway, config.gateway)?;
    apply_file_upstream_config(&mut resolved.gateway, config.upstream);
    apply_file_agents_config(&mut resolved.agents, config.agents);
    Ok(())
}

fn apply_file_gateway_config(
    gateway: &mut GatewayConfig,
    config: Option<FileGatewayConfig>,
) -> Result<(), CliError> {
    let Some(config) = config else {
        return Ok(());
    };
    if let Some(value) = config.max_hook_payload_bytes {
        gateway.max_hook_payload_bytes =
            validate_body_limit("gateway.max_hook_payload_bytes", value)?;
    }
    if let Some(value) = config.max_passthrough_body_bytes {
        gateway.max_passthrough_body_bytes =
            validate_body_limit("gateway.max_passthrough_body_bytes", value)?;
    }
    Ok(())
}

// Applies upstream LLM provider URLs. These are the bases for OpenAI- and Anthropic-shaped
// gateway routes; transparent `run` mode can still override them per invocation.
fn apply_file_upstream_config(gateway: &mut GatewayConfig, upstream: Option<FileUpstreamConfig>) {
    let Some(upstream) = upstream else {
        return;
    };
    if let Some(value) = upstream.openai_base_url {
        gateway.openai_base_url = value;
    }
    if let Some(value) = upstream.anthropic_base_url {
        gateway.anthropic_base_url = value;
    }
}

#[derive(Debug, Clone)]
struct PluginTomlConfig {
    value: Option<Value>,
    dynamic_plugins: Vec<ResolvedDynamicPluginConfig>,
    dynamic_plugin_policy: DynamicPluginHostPolicy,
    contributing_sources: Vec<PathBuf>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PluginTomlPluginsSection {
    #[serde(default)]
    dynamic: Vec<FileDynamicPluginConfig>,
    #[serde(default)]
    policy: Option<crate::plugins::policy::FileDynamicPluginHostPolicy>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileDynamicPluginConfig {
    manifest: String,
    #[serde(default)]
    config: Option<Map<String, Value>>,
}

fn load_plugin_toml_config(
    explicit: Option<&PathBuf>,
    plugin_config_path: Option<&PathBuf>,
) -> Result<Option<PluginTomlConfig>, CliError> {
    load_plugin_toml_config_scoped(explicit, plugin_config_path, user_config_scope())
}

fn load_plugin_toml_config_scoped(
    explicit: Option<&PathBuf>,
    plugin_config_path: Option<&PathBuf>,
    user_only: bool,
) -> Result<Option<PluginTomlConfig>, CliError> {
    load_plugin_toml_config_from_paths(plugin_config_paths_scoped(
        explicit,
        plugin_config_path,
        user_only,
    ))
}

/// Returns the physical `plugins.toml` files that contribute effective runtime or dynamic
/// plugin configuration under the default discovery rules.
pub(crate) fn effective_plugin_toml_sources() -> Result<Vec<PathBuf>, CliError> {
    let Some(config) = load_plugin_toml_config(None, None)? else {
        return Ok(Vec::new());
    };
    let mut sources = config.contributing_sources;
    sources.sort();
    sources.dedup();
    Ok(sources)
}

fn load_plugin_toml_config_from_paths<I>(paths: I) -> Result<Option<PluginTomlConfig>, CliError>
where
    I: IntoIterator<Item = PathBuf>,
{
    let paths = paths.into_iter().collect::<Vec<_>>();
    let mut dynamic_plugins = Vec::new();
    let mut dynamic_plugin_policy = DynamicPluginHostPolicy::default();
    let mut seen_plugin_ids = HashSet::new();
    let mut contributing_sources = Vec::new();
    let mut runtime_documents = Vec::new();

    for path in &paths {
        let Some(raw) = read_config_file(path, false, "plugin configuration")? else {
            continue;
        };
        let mut parsed = raw
            .parse::<toml::Table>()
            .map(toml::Value::Table)
            .map_err(|error| {
                CliError::Config(format!(
                    "invalid plugin TOML in {}: {error}",
                    path.display()
                ))
            })?;
        let resolved_plugins =
            resolve_dynamic_plugin_refs(path, &mut parsed, &mut seen_plugin_ids)?;
        if !resolved_plugins.dynamic_plugins.is_empty()
            || resolved_plugins.dynamic_plugin_policy != DynamicPluginHostPolicy::default()
        {
            contributing_sources.push(path.clone());
        }
        dynamic_plugins.extend(resolved_plugins.dynamic_plugins);
        dynamic_plugin_policy.merge_from(resolved_plugins.dynamic_plugin_policy);
        runtime_documents.push((
            path.clone(),
            serde_json::to_value(remove_dynamic_plugin_sections(parsed))
                .expect("toml value serializes to JSON"),
        ));
    }

    // Delegate merged runtime plugin config to the shared core primitive after dynamic refs have
    // been validated independently. File precedence stays unchanged for the generic runtime path.
    let resolved = merge_plugin_config_documents(runtime_documents).map_err(|err| match err {
        PluginError::InvalidConfig(message) => CliError::Config(message),
        other => CliError::Config(other.to_string()),
    })?;
    match resolved {
        Some((value, sources)) => {
            contributing_sources.extend(sources.iter().cloned());
            contributing_sources.sort();
            contributing_sources.dedup();
            Ok(Some(PluginTomlConfig {
                value: plugin_toml_runtime_value(value),
                dynamic_plugins,
                dynamic_plugin_policy,
                contributing_sources,
            }))
        }
        None => Ok((!dynamic_plugins.is_empty()
            || dynamic_plugin_policy != DynamicPluginHostPolicy::default())
        .then_some(PluginTomlConfig {
            value: None,
            dynamic_plugins,
            dynamic_plugin_policy,
            contributing_sources,
        })),
    }
}

fn apply_plugin_toml_config(resolved: &mut ResolvedConfig, plugin_toml: Option<PluginTomlConfig>) {
    let Some(plugin_toml) = plugin_toml else {
        return;
    };
    if let Some(value) = plugin_toml.value {
        resolved.gateway.plugin_config = Some(value);
    }
    resolved.dynamic_plugins = plugin_toml.dynamic_plugins;
    resolved.dynamic_plugin_policy = plugin_toml.dynamic_plugin_policy;
}

struct ResolvedDynamicPluginRefs {
    dynamic_plugins: Vec<ResolvedDynamicPluginConfig>,
    dynamic_plugin_policy: DynamicPluginHostPolicy,
}

fn resolve_dynamic_plugin_refs(
    source: &Path,
    value: &mut toml::Value,
    seen_plugin_ids: &mut HashSet<String>,
) -> Result<ResolvedDynamicPluginRefs, CliError> {
    let Some(root) = value.as_table_mut() else {
        return Ok(ResolvedDynamicPluginRefs {
            dynamic_plugins: Vec::new(),
            dynamic_plugin_policy: DynamicPluginHostPolicy::default(),
        });
    };

    let plugins_value = root.get("plugins").cloned();
    let Some(plugins_value) = plugins_value else {
        return Ok(ResolvedDynamicPluginRefs {
            dynamic_plugins: Vec::new(),
            dynamic_plugin_policy: DynamicPluginHostPolicy::default(),
        });
    };

    let plugins: PluginTomlPluginsSection = plugins_value.try_into().map_err(|error| {
        CliError::Config(format!(
            "invalid dynamic plugin config in {}: {error}",
            source.display()
        ))
    })?;

    if let Some(toml::Value::Table(plugins_table)) = root.get_mut("plugins") {
        plugins_table.remove("dynamic");
        plugins_table.remove("policy");
        if plugins_table.is_empty() {
            root.remove("plugins");
        }
    }

    let mut resolved = Vec::with_capacity(plugins.dynamic.len());
    for dynamic in plugins.dynamic {
        let manifest_path = resolve_dynamic_manifest_path(source, &dynamic.manifest);
        let (manifest, manifest_ref) = load_bounded_dynamic_plugin_manifest(&manifest_path)
            .map_err(|error| {
                CliError::Config(format!(
                    "invalid dynamic plugin manifest referenced by {}: {error}",
                    source.display()
                ))
            })?;
        let plugin_id = manifest.plugin.id.trim().to_owned();
        if !seen_plugin_ids.insert(plugin_id.clone()) {
            return Err(CliError::Config(format!(
                "duplicate dynamic plugin id '{}' in {} across plugins.toml sources",
                plugin_id,
                source.display()
            )));
        }
        resolved.push(ResolvedDynamicPluginConfig {
            plugin_id,
            manifest_ref,
            has_explicit_config: dynamic.config.is_some(),
            config: dynamic.config.unwrap_or_default(),
            source: source.to_path_buf(),
        });
    }
    Ok(ResolvedDynamicPluginRefs {
        dynamic_plugins: resolved,
        dynamic_plugin_policy: plugins.policy.map(Into::into).unwrap_or_default(),
    })
}

fn resolve_dynamic_manifest_path(source: &Path, manifest: &str) -> PathBuf {
    let manifest = PathBuf::from(manifest);
    if manifest.is_absolute() {
        manifest
    } else {
        source
            .parent()
            .map(|parent| parent.join(&manifest))
            .unwrap_or(manifest)
    }
}

fn plugin_toml_runtime_value(value: Value) -> Option<Value> {
    match value {
        Value::Object(ref object) if object.is_empty() => None,
        other => Some(other),
    }
}

fn remove_dynamic_plugin_sections(mut value: toml::Value) -> toml::Value {
    if let Some(root) = value.as_table_mut()
        && let Some(toml::Value::Table(plugins)) = root.get_mut("plugins")
    {
        plugins.remove("dynamic");
        plugins.remove("policy");
        if plugins.is_empty() {
            root.remove("plugins");
        }
    }
    value
}

// Applies configured agent commands from the merged file configuration.
fn apply_file_agents_config(agents: &mut AgentConfigs, file_agents: Option<FileAgentsConfig>) {
    let Some(file_agents) = file_agents else {
        return;
    };
    if let Some(value) = file_agents.claude {
        agents.claude.command = value.command;
    }
    if let Some(value) = file_agents.codex {
        agents.codex.command = value.command;
    }
    if let Some(value) = file_agents.hermes {
        agents.hermes.command = value.command;
        agents.hermes.hooks_path = value.hooks_path;
    }
}

// Applies environment variables after file configuration. Invalid bind values are ignored here to
// preserve existing startup behavior, while string values replace earlier layers when present.
fn apply_env_config(config: &mut GatewayConfig) -> Result<(), CliError> {
    if let Ok(value) = std::env::var("NEMO_RELAY_GATEWAY_BIND")
        && let Ok(value) = value.parse()
    {
        config.bind = value;
    }
    if let Ok(value) = std::env::var("NEMO_RELAY_OPENAI_BASE_URL") {
        config.openai_base_url = value;
    }
    if let Ok(value) = std::env::var("NEMO_RELAY_ANTHROPIC_BASE_URL") {
        config.anthropic_base_url = value;
    }
    if let Ok(value) = std::env::var("NEMO_RELAY_MAX_HOOK_PAYLOAD_BYTES") {
        config.max_hook_payload_bytes =
            parse_env_body_limit("NEMO_RELAY_MAX_HOOK_PAYLOAD_BYTES", &value)?;
    }
    if let Ok(value) = std::env::var("NEMO_RELAY_MAX_PASSTHROUGH_BODY_BYTES") {
        config.max_passthrough_body_bytes =
            parse_env_body_limit("NEMO_RELAY_MAX_PASSTHROUGH_BODY_BYTES", &value)?;
    }
    Ok(())
}

fn parse_env_body_limit(name: &str, raw: &str) -> Result<usize, CliError> {
    let value = raw.parse::<usize>().map_err(|error| {
        CliError::Config(format!("{name} must be a positive byte count: {error}"))
    })?;
    validate_body_limit(name, value)
}

fn validate_body_limit(name: &str, value: usize) -> Result<usize, CliError> {
    if value == 0 {
        return Err(CliError::Config(format!("{name} must be greater than 0")));
    }
    Ok(value)
}

// Recursively merges TOML tables and replaces scalar/array values from the higher-priority side.
// This lets user/project configs override individual nested keys without restating whole sections.
fn merge_toml(left: &mut toml::Value, right: toml::Value) {
    match (left, right) {
        (toml::Value::Table(left), toml::Value::Table(right)) => {
            for (key, value) in right {
                match left.get_mut(&key) {
                    Some(existing) => merge_toml(existing, value),
                    None => {
                        left.insert(key, value);
                    }
                }
            }
        }
        (left, right) => *left = right,
    }
}

fn legacy_observability_sections(value: &toml::Value) -> Vec<&'static str> {
    let mut sections = Vec::new();
    if value.get("exporters").is_some() {
        sections.push("[exporters]");
    }
    if value.get("observability").is_some() {
        sections.push("[observability]");
    }
    if value
        .get("export")
        .and_then(|export| export.get("openinference"))
        .is_some()
    {
        sections.push("[export.openinference]");
    }
    sections
}

// Parses JSON-valued CLI options into runtime metadata/config values and labels errors with the
// user-facing option name so callers can report which structured argument was malformed.
fn parse_json_option(name: &str, value: &str) -> Result<Value, CliError> {
    serde_json::from_str::<Value>(value)
        .map_err(|error| CliError::Config(format!("invalid {name}: {error}")))
}

/// Reads a non-empty UTF-8 header value as an owned string.
///
/// Invalid header bytes and empty strings are treated as absent so callers can preserve their
/// explicit fallback order without surfacing HTTP parsing details as gateway errors.
pub(crate) fn header_string(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn header_json(headers: &HeaderMap, name: &str) -> Option<Value> {
    header_string(headers, name).and_then(|raw| serde_json::from_str(&raw).ok())
}

impl GatewayMode {
    // Returns the installed hook-forward spelling for gateway mode headers. Keeping this separate
    // from debug output prevents enum formatting changes from affecting persisted hook commands.
    pub(crate) const fn as_arg(self) -> &'static str {
        match self {
            Self::HookOnly => "hook-only",
            Self::Passthrough => "passthrough",
            Self::Required => "required",
        }
    }
}

#[cfg(test)]
#[path = "../tests/coverage/config_tests.rs"]
mod tests;
