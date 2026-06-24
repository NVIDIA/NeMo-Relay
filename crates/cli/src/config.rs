// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use axum::http::HeaderMap;
use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};
use nemo_relay::plugin::dynamic::DynamicPluginManifest;
use nemo_relay::plugin::{PluginError, merge_plugin_config_documents};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use strum::Display;

use crate::error::CliError;
use crate::plugin_shim::PluginShimCommand;

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
                      NMF-86 — codex's own auth.json JWT is stripped). Requires codex-cli >= \
                      0.129.0.",
        after_help = "Examples:\n  \
                      nemo-relay codex\n  \
                      nemo-relay codex -- exec \"fix the bug in foo.rs\"\n  \
                      nemo-relay --openai-base-url https://inference-api.nvidia.com codex"
    )]
    Codex(EasyPathCommand),
    /// Run Cursor with observability (setup on first use)
    #[command(
        long_about = "Run Cursor's `cursor-agent` CLI under an ephemeral NeMo Relay gateway. The \
                      launcher temporarily patches `.cursor/hooks.json` in the project root \
                      during the run and restores it on exit. Disable that via \
                      `[agents.cursor] patch_restore_hooks = false` in config.toml if you \
                      maintain `.cursor/hooks.json` yourself.",
        after_help = "Examples:\n  \
                      nemo-relay cursor\n  \
                      nemo-relay cursor -- agent --resume <session-id>"
    )]
    Cursor(EasyPathCommand),
    /// Run Hermes with observability (setup on first use)
    #[command(
        long_about = "Run NVIDIA's Hermes agent under a NeMo Relay gateway. Hermes reads hooks \
                      from `.hermes/config.yaml`; first-run setup writes that file alongside \
                      `.nemo-relay/config.toml` so every subsequent invocation traces \
                      automatically. Re-run `nemo-relay config hermes` to refresh the hooks.",
        after_help = "Examples:\n  \
                      nemo-relay hermes\n  \
                      nemo-relay hermes -- chat --provider custom"
    )]
    Hermes(EasyPathCommand),
    /// Run the interactive setup (writes `.nemo-relay/config.toml`)
    Config(ConfigCommand),
    /// Create or edit plugin configuration (writes `plugins.toml`)
    Plugins(PluginsCommand),
    /// Install coding-agent plugins from the local nemo-relay CLI.
    Install(InstallCommand),
    /// Uninstall coding-agent plugins installed by `nemo-relay install`.
    Uninstall(UninstallCommand),
    /// Validate and configure model pricing catalogs.
    Pricing(PricingCommand),
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
    /// Internal: plugin-local hook and sidecar supervisor. Not typed by humans.
    #[command(hide = true)]
    PluginShim(PluginShimCommand),
}

/// Args for `nemo-relay doctor`. `--json` is on this command (rather than as a global flag)
/// so it doesn't pollute the help output of subcommands where it has no meaning.
#[derive(Debug, Clone, Args)]
pub(crate) struct DoctorCommand {
    /// Limit readiness checks to one supported agent.
    #[arg(value_enum)]
    pub(crate) agent: Option<CodingAgent>,
    /// Diagnose an installed coding-agent plugin instead of the normal relay config.
    #[arg(long, value_enum)]
    pub(crate) plugin: Option<PluginHost>,
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
    pub(crate) host: PluginHost,
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
    pub(crate) host: PluginHost,
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
    /// Delete the project config file (or remove just the scoped agent's block when an agent
    /// is named). The wizard does NOT run after a reset — invoke `nemo-relay config` again to
    /// re-create the file from scratch.
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
    /// Interactively create or edit built-in plugin configuration in `plugins.toml`.
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

/// Args for `nemo-relay pricing`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PricingCommand {
    #[command(subcommand)]
    pub(crate) command: PricingSubcommand,
}

/// Pricing catalog and resolver subcommands.
#[derive(Debug, Clone, Subcommand)]
pub(crate) enum PricingSubcommand {
    /// Validate a pricing catalog JSON file.
    Validate(PricingValidateCommand),
    /// Initialize the pricing plugin component in `plugins.toml`.
    Init(PricingInitCommand),
    /// Add a pricing catalog file source to `plugins.toml`.
    AddSource(PricingAddSourceCommand),
    /// Resolve which pricing entry matches a model and optional usage.
    Resolve(PricingResolveCommand),
}

/// Common target-scope flags for pricing config mutations.
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

/// Args for `nemo-relay pricing validate`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PricingValidateCommand {
    /// Path to a Relay pricing catalog JSON file.
    pub(crate) path: PathBuf,
}

/// Args for `nemo-relay pricing init`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PricingInitCommand {
    #[command(flatten)]
    pub(crate) scope: PricingScopeArgs,
}

/// Args for `nemo-relay pricing add-source`.
#[derive(Debug, Clone, Args)]
pub(crate) struct PricingAddSourceCommand {
    #[command(flatten)]
    pub(crate) scope: PricingScopeArgs,
    /// Path to a Relay pricing catalog JSON file.
    pub(crate) path: PathBuf,
    /// Append as a lower-priority source instead of prepending as the highest-priority override.
    #[arg(long)]
    pub(crate) append: bool,
}

/// Args for `nemo-relay pricing resolve`.
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
    /// Generic plugin configuration JSON for process-level gateway plugin activation.
    #[arg(long, env = "NEMO_RELAY_PLUGIN_CONFIG")]
    pub(crate) plugin_config: Option<String>,
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
            || self.plugin_config.is_some()
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
    #[arg(long)]
    pub(crate) plugin_config: Option<String>,
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
    #[arg(long)]
    pub(crate) plugin_config: Option<String>,
    #[arg(long)]
    pub(crate) dry_run: bool,
    #[arg(long)]
    pub(crate) print: bool,
    #[arg(last = true)]
    pub(crate) command: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum CodingAgent {
    /// Canonical CLI spelling is `claude` (matches Anthropic's own binary name and the TOML
    /// `[agents.claude]` key). `claude-code` is kept as an input alias for backward compat
    /// with hooks installed before this rename.
    #[value(name = "claude", alias = "claude-code")]
    ClaudeCode,
    Codex,
    Cursor,
    Hermes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum PluginHost {
    Codex,
    #[value(name = "claude-code", alias = "claude")]
    ClaudeCode,
    All,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedDynamicPluginConfig {
    pub(crate) plugin_id: String,
    pub(crate) manifest_ref: String,
    pub(crate) config: Map<String, Value>,
    pub(crate) source: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub(crate) enum DynamicPluginHostConfigStatus {
    Absent,
    Present,
}

impl ResolvedDynamicPluginConfig {
    pub(crate) fn host_config_status(&self) -> DynamicPluginHostConfigStatus {
        if self.config.is_empty() {
            DynamicPluginHostConfigStatus::Absent
        } else {
            DynamicPluginHostConfigStatus::Present
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AgentConfigs {
    pub(crate) claude: AgentCommandConfig,
    pub(crate) codex: AgentCommandConfig,
    pub(crate) cursor: CursorAgentConfig,
    pub(crate) hermes: AgentCommandConfig,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AgentCommandConfig {
    pub(crate) command: Option<String>,
    /// Recorded by `nemo-relay config` when it installs hermes shell hooks. Other agents leave
    /// this empty; the launcher reads it only to print a "hooks live here" pointer for hermes.
    pub(crate) hooks_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub(crate) struct CursorAgentConfig {
    pub(crate) command: Option<String>,
    pub(crate) patch_restore_hooks: bool,
}

impl Default for CursorAgentConfig {
    // Keeps Cursor run-mode patching enabled unless a config file opts out. Cursor's CLI discovers
    // hooks from project files, so the launcher needs permission to temporarily patch and restore
    // `.cursor/hooks.json` by default.
    fn default() -> Self {
        Self {
            command: None,
            patch_restore_hooks: true,
        }
    }
}

// TOML file shape grouped by user intent. Sections map 1:1 onto fields already present on
// `GatewayConfig` / `AgentConfigs`; plugin config is passed through to the runtime's generic
// `PluginConfig` activation path.
#[derive(Debug, Clone, Default, Deserialize)]
struct FileConfig {
    gateway: Option<FileGatewayConfig>,
    upstream: Option<FileUpstreamConfig>,
    plugins: Option<FilePluginsConfig>,
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
struct FilePluginsConfig {
    // Generic plugin initialization shape. The gateway activates this process-wide at startup.
    config: Option<Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileAgentsConfig {
    // Keys match the agent's CLI invocation name (`claude`, `codex`, `cursor`, `hermes`) — the
    // word the user types at the shell — not the product name ("Claude Code") or the internal
    // `CodingAgent` enum kebab spelling. Same convention as the bare-agent shortcut in Phase 2.
    claude: Option<FileAgentCommandConfig>,
    codex: Option<FileAgentCommandConfig>,
    cursor: Option<FileCursorAgentConfig>,
    hermes: Option<FileAgentCommandConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileAgentCommandConfig {
    command: Option<String>,
    hooks_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileCursorAgentConfig {
    command: Option<String>,
    patch_restore_hooks: Option<bool>,
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
    let mut resolved = load_shared_config(args.config.as_ref())?;
    apply_server_overrides(&mut resolved.gateway, args)?;
    Ok(resolved)
}

/// Resolves shared config for plugin-facing CLI commands without mutating gateway runtime fields.
pub(crate) fn resolve_plugins_config(
    explicit: Option<&PathBuf>,
) -> Result<ResolvedConfig, CliError> {
    load_shared_config(explicit)
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
    let mut resolved = load_shared_config(config)?;
    if let Some(args) = inherited {
        // Run-subcommand plugin config has higher precedence than inherited top-level plugin
        // config. Skip only that inherited field so file/plugins.toml conflicts are still caught
        // when the run-level override is applied below.
        if command.plugin_config.is_some() && args.plugin_config.is_some() {
            let mut inherited = args.clone();
            inherited.plugin_config = None;
            apply_server_overrides(&mut resolved.gateway, &inherited)?;
        } else {
            apply_server_overrides(&mut resolved.gateway, args)?;
        }
    }
    apply_run_overrides(&mut resolved.gateway, command)?;
    resolved.gateway.bind = "127.0.0.1:0"
        .parse()
        .expect("valid transparent bind address");
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
    if let Some(value) = &command.plugin_config {
        apply_cli_plugin_config(config, value)?;
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
    if let Some(value) = &args.plugin_config {
        apply_cli_plugin_config(config, value)?;
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
fn load_shared_config(explicit: Option<&PathBuf>) -> Result<ResolvedConfig, CliError> {
    let mut merged = toml::Value::Table(toml::map::Map::new());
    let mut config_toml_plugin_sources = Vec::new();
    for path in config_paths(explicit) {
        if path.exists() {
            let raw = std::fs::read_to_string(&path)?;
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
            if has_config_toml_plugin_config(&parsed) {
                config_toml_plugin_sources.push(path.clone());
            }
            merge_toml(&mut merged, parsed);
        }
    }
    if config_toml_plugin_sources.len() > 1 {
        return Err(CliError::Config(format!(
            "plugin config is defined in multiple config.toml files: {}; move it to one \
             [plugins].config block or use plugins.toml",
            format_paths(&config_toml_plugin_sources)
        )));
    }
    let plugin_toml = load_plugin_toml_config(explicit)?;
    let mut resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        ..ResolvedConfig::default()
    };
    apply_file_config(&mut resolved, merged)?;
    apply_plugin_toml_config(
        &mut resolved,
        config_toml_plugin_sources.first(),
        plugin_toml,
    )?;
    apply_env_config(&mut resolved.gateway)?;
    Ok(resolved)
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
    if let Some(path) = explicit {
        return vec![path.clone()];
    }
    let mut paths = vec![PathBuf::from("/etc/nemo-relay/config.toml")];
    if let Ok(cwd) = std::env::current_dir()
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
fn plugin_config_paths(explicit: Option<&PathBuf>) -> Vec<PathBuf> {
    if let Some(path) = explicit {
        return path
            .parent()
            .map(|parent| vec![parent.join(PLUGINS_TOML)])
            .unwrap_or_default();
    }
    implicit_plugin_config_paths(std::env::current_dir().ok().as_deref(), user_config_dir())
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
// are ignored, preserving defaults and prior merge layers; Cursor's patch-restore flag is only
// changed when explicitly present.
fn apply_file_config(resolved: &mut ResolvedConfig, value: toml::Value) -> Result<(), CliError> {
    let config: FileConfig = value.try_into().map_err(|error| {
        CliError::Config(format!("invalid gateway configuration shape: {error}"))
    })?;
    apply_file_gateway_config(&mut resolved.gateway, config.gateway)?;
    apply_file_upstream_config(&mut resolved.gateway, config.upstream);
    apply_file_plugins_config(&mut resolved.gateway, config.plugins);
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

// Applies plugin config. The gateway activates process-level plugin config at startup; hook headers
// still carry the value as session metadata until scoped plugin activation exists.
fn apply_file_plugins_config(gateway: &mut GatewayConfig, plugins: Option<FilePluginsConfig>) {
    let Some(plugins) = plugins else {
        return;
    };
    if let Some(value) = plugins.config {
        gateway.plugin_config = Some(value);
    }
}

#[derive(Debug, Clone)]
struct PluginTomlConfig {
    value: Option<Value>,
    dynamic_plugins: Vec<ResolvedDynamicPluginConfig>,
    sources: Vec<PathBuf>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PluginTomlPluginsSection {
    #[serde(default)]
    dynamic: Vec<FileDynamicPluginConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileDynamicPluginConfig {
    manifest: String,
    #[serde(default)]
    config: Map<String, Value>,
}

fn load_plugin_toml_config(
    explicit: Option<&PathBuf>,
) -> Result<Option<PluginTomlConfig>, CliError> {
    load_plugin_toml_config_from_paths(plugin_config_paths(explicit))
}

fn load_plugin_toml_config_from_paths<I>(paths: I) -> Result<Option<PluginTomlConfig>, CliError>
where
    I: IntoIterator<Item = PathBuf>,
{
    let paths = paths.into_iter().collect::<Vec<_>>();
    let mut dynamic_plugins = Vec::new();
    let mut seen_plugin_ids = HashSet::new();
    let mut runtime_documents = Vec::new();

    for path in &paths {
        if !path.exists() {
            continue;
        }
        let raw = std::fs::read_to_string(path)?;
        let mut parsed = raw
            .parse::<toml::Table>()
            .map(toml::Value::Table)
            .map_err(|error| {
                CliError::Config(format!(
                    "invalid plugin TOML in {}: {error}",
                    path.display()
                ))
            })?;
        dynamic_plugins.extend(resolve_dynamic_plugin_refs(
            path,
            &mut parsed,
            &mut seen_plugin_ids,
        )?);
        runtime_documents.push((
            path.clone(),
            serde_json::to_value(remove_dynamic_plugin_section(parsed))
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
        Some((value, sources)) => Ok(Some(PluginTomlConfig {
            value: plugin_toml_runtime_value(value),
            dynamic_plugins,
            sources,
        })),
        None => Ok((!dynamic_plugins.is_empty()).then_some(PluginTomlConfig {
            value: None,
            dynamic_plugins,
            sources: Vec::new(),
        })),
    }
}

fn apply_plugin_toml_config(
    resolved: &mut ResolvedConfig,
    config_toml_plugin_source: Option<&PathBuf>,
    plugin_toml: Option<PluginTomlConfig>,
) -> Result<(), CliError> {
    let Some(plugin_toml) = plugin_toml else {
        return Ok(());
    };
    if let Some(config_source) = config_toml_plugin_source
        && plugin_toml.value.is_some()
    {
        return Err(CliError::Config(format!(
            "plugin config is defined in both {} and {}; choose one source",
            config_source.display(),
            format_paths(&plugin_toml.sources)
        )));
    }
    if let Some(value) = plugin_toml.value {
        resolved.gateway.plugin_config = Some(value);
    }
    resolved.dynamic_plugins = plugin_toml.dynamic_plugins;
    Ok(())
}

fn resolve_dynamic_plugin_refs(
    source: &Path,
    value: &mut toml::Value,
    seen_plugin_ids: &mut HashSet<String>,
) -> Result<Vec<ResolvedDynamicPluginConfig>, CliError> {
    let Some(root) = value.as_table_mut() else {
        return Ok(Vec::new());
    };

    let plugins_value = root.get("plugins").cloned();
    let Some(plugins_value) = plugins_value else {
        return Ok(Vec::new());
    };

    let plugins: PluginTomlPluginsSection = plugins_value.try_into().map_err(|error| {
        CliError::Config(format!(
            "invalid dynamic plugin config in {}: {error}",
            source.display()
        ))
    })?;

    if let Some(toml::Value::Table(plugins_table)) = root.get_mut("plugins") {
        plugins_table.remove("dynamic");
        if plugins_table.is_empty() {
            root.remove("plugins");
        }
    }

    let mut resolved = Vec::with_capacity(plugins.dynamic.len());
    for dynamic in plugins.dynamic {
        let manifest_path = resolve_dynamic_manifest_path(source, &dynamic.manifest);
        let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
            .map_err(|error| CliError::Config(error.to_string()))?;
        let plugin_id = manifest.plugin.id.trim().to_owned();
        if !seen_plugin_ids.insert(plugin_id.clone()) {
            return Err(CliError::Config(format!(
                "duplicate dynamic plugin id '{}' across plugins.toml sources",
                plugin_id
            )));
        }
        resolved.push(ResolvedDynamicPluginConfig {
            plugin_id,
            manifest_ref,
            config: dynamic.config,
            source: source.to_path_buf(),
        });
    }
    Ok(resolved)
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

fn remove_dynamic_plugin_section(mut value: toml::Value) -> toml::Value {
    if let Some(root) = value.as_table_mut()
        && let Some(toml::Value::Table(plugins)) = root.get_mut("plugins")
    {
        plugins.remove("dynamic");
        if plugins.is_empty() {
            root.remove("plugins");
        }
    }
    value
}

fn apply_cli_plugin_config(config: &mut GatewayConfig, value: &str) -> Result<(), CliError> {
    if config.plugin_config.is_some() {
        return Err(CliError::Config(
            "plugin config is defined by both --plugin-config and file configuration; choose one source".into(),
        ));
    }
    config.plugin_config = Some(parse_json_option("plugin config", value)?);
    Ok(())
}

// Applies configured agent commands and Cursor's temporary-hook behavior. Cursor's
// `patch_restore_hooks` flag is intentionally tri-state in file config so omitted values preserve
// the safe default while explicit `false` disables temporary hook mutation.
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
    if let Some(value) = file_agents.cursor {
        agents.cursor.command = value.command;
        if let Some(patch_restore_hooks) = value.patch_restore_hooks {
            agents.cursor.patch_restore_hooks = patch_restore_hooks;
        }
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

fn has_config_toml_plugin_config(value: &toml::Value) -> bool {
    value
        .get("plugins")
        .and_then(|plugins| plugins.get("config"))
        .is_some()
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

fn format_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
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

impl CodingAgent {
    // Returns the gateway hook endpoint for the agent. These paths are stable integration surface
    // because installed hook commands persist them in user or project configuration.
    pub(crate) const fn hook_path(self) -> &'static str {
        match self {
            Self::ClaudeCode => "/hooks/claude-code",
            Self::Codex => "/hooks/codex",
            Self::Cursor => "/hooks/cursor",
            Self::Hermes => "/hooks/hermes",
        }
    }

    // Returns the canonical CLI spelling used in generated commands and diagnostics. Matches the
    // clap `#[value(name = ...)]` overrides on the enum so install/run output can be copied back
    // into commands. `claude` matches Anthropic's binary name and the TOML `[agents.claude]` key.
    pub(crate) const fn as_arg(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude",
            Self::Codex => "codex",
            Self::Cursor => "cursor",
            Self::Hermes => "hermes",
        }
    }

    // Infers an agent from the executable basename, accepting both canonical project names and
    // common command aliases. Path components are stripped so configured absolute commands work.
    pub(crate) fn infer(command: &str) -> Option<Self> {
        let name = std::path::Path::new(command)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(command);
        match name {
            "claude" | "claude-code" => Some(Self::ClaudeCode),
            "codex" => Some(Self::Codex),
            "cursor" | "cursor-agent" => Some(Self::Cursor),
            "hermes" | "hermes-agent" => Some(Self::Hermes),
            _ => None,
        }
    }
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
