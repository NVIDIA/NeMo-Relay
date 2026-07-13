// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};

use crate::agents::CodingAgent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum AgentArg {
    #[value(name = "claude", alias = "claude-code")]
    Claude,
    Codex,
    Hermes,
}

impl From<AgentArg> for CodingAgent {
    fn from(value: AgentArg) -> Self {
        match value {
            AgentArg::Claude => Self::ClaudeCode,
            AgentArg::Codex => Self::Codex,
            AgentArg::Hermes => Self::Hermes,
        }
    }
}

impl ServerArgs {
    pub(crate) fn to_runtime(&self) -> crate::configuration::GatewayOverrides {
        crate::configuration::GatewayOverrides {
            config: self.config.clone(),
            bind: self.bind,
            openai_base_url: self.openai_base_url.clone(),
            anthropic_base_url: self.anthropic_base_url.clone(),
            plugin_config_path: self.plugin_config_path.clone(),
            ready_file: self.ready_file.clone(),
            max_hook_payload_bytes: self.max_hook_payload_bytes,
            max_passthrough_body_bytes: self.max_passthrough_body_bytes,
        }
    }
}

impl RunCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::RunOverrides {
        crate::configuration::RunOverrides {
            agent: self.agent.map(Into::into),
            config: self.config,
            openai_base_url: self.openai_base_url,
            anthropic_base_url: self.anthropic_base_url,
            session_metadata: self.session_metadata,
            plugin_config_path: self.plugin_config_path,
            dry_run: self.dry_run,
            print: self.print,
            command: self.command,
        }
    }
}

impl HookForwardCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::HookForwardRequest {
        crate::configuration::HookForwardRequest {
            agent: self.agent.into(),
            gateway_url: self.gateway_url,
            generation_file: self.generation_file,
            generation_token: self.generation_token,
            forward_only: self.forward_only,
            transparent_run: self.transparent_run,
            profile: self.profile,
            session_metadata: self.session_metadata,
            gateway_mode: self.gateway_mode.map(Into::into),
            fail_closed: self.fail_closed,
        }
    }
}

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
        long_about = "Run Hermes Agent under an ephemeral NeMo Relay gateway. The wrapper uses a \
                      process-private HERMES_HOME overlay for dynamic hooks, without rewriting \
                      the user's Hermes configuration. Use `nemo-relay install hermes` when bare \
                      Hermes processes should load the shared native Relay gateway on \
                      127.0.0.1:47632 through MCP.",
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
                      nemo-relay --bind 127.0.0.1:4041 mcp  # explicit standalone/test bind"
    )]
    Mcp,
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

/// Args for `nemo-relay doctor`. `--json` is on this command (rather than as a global flag)
/// so it doesn't pollute the help output of subcommands where it has no meaning.
#[derive(Debug, Clone, Args)]
pub(crate) struct DoctorCommand {
    /// Limit readiness checks to one supported agent.
    #[arg(value_enum)]
    pub(crate) agent: Option<AgentArg>,
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
    pub(crate) agent: Option<AgentArg>,
    /// Delete the project config file or the scoped transparent-wrapper agent block. Persistent
    /// Hermes MCP, hooks, and trust are removed with `nemo-relay uninstall hermes`. The wizard
    /// does not run after reset; invoke `nemo-relay config` again to recreate configuration.
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

#[derive(Debug, Clone, Args)]
pub(crate) struct HookForwardCommand {
    #[arg(value_enum)]
    pub(crate) agent: AgentArg,
    #[arg(long)]
    pub(crate) gateway_url: Option<String>,
    /// Private install-generation fence used by generated persistent hooks.
    #[arg(long, hide = true)]
    pub(crate) generation_file: Option<PathBuf>,
    /// Immutable generation identity paired with `generation_file` by installed hooks.
    #[arg(long, hide = true)]
    pub(crate) generation_token: Option<String>,
    /// Forward to an existing gateway without starting or recovering Relay.
    #[arg(
        long,
        conflicts_with_all = ["generation_file", "generation_token"]
    )]
    pub(crate) forward_only: bool,
    /// Marks the process-private hook source injected by `nemo-relay run`.
    #[arg(
        long,
        hide = true,
        conflicts_with_all = ["generation_file", "generation_token", "forward_only"]
    )]
    pub(crate) transparent_run: bool,
    #[arg(long)]
    pub(crate) profile: Option<String>,
    #[arg(long)]
    pub(crate) session_metadata: Option<String>,
    #[arg(long, value_enum)]
    pub(crate) gateway_mode: Option<GatewayModeArg>,
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
    pub(crate) agent: Option<AgentArg>,
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

impl From<IntegrationHost> for crate::configuration::IntegrationHost {
    fn from(value: IntegrationHost) -> Self {
        match value {
            IntegrationHost::Codex => Self::Codex,
            IntegrationHost::ClaudeCode => Self::ClaudeCode,
            IntegrationHost::Hermes => Self::Hermes,
            IntegrationHost::All => Self::All,
        }
    }
}

impl InstallCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::InstallRequest {
        crate::configuration::InstallRequest {
            host: self.host.into(),
            install_dir: self.install_dir,
            force: self.force,
            dry_run: self.dry_run,
            skip_doctor: self.skip_doctor,
        }
    }
}

impl UninstallCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::UninstallRequest {
        crate::configuration::UninstallRequest {
            host: self.host.into(),
            install_dir: self.install_dir,
            dry_run: self.dry_run,
        }
    }
}

impl From<PluginsScopeArgs> for crate::configuration::PluginsScopeArgs {
    fn from(value: PluginsScopeArgs) -> Self {
        Self {
            user: value.user,
            project: value.project,
            global: value.global,
        }
    }
}

impl PluginsEditCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PluginsEditCommand {
        crate::configuration::PluginsEditCommand {
            scope: self.scope.into(),
        }
    }
}
impl PluginsAddCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PluginsAddCommand {
        crate::configuration::PluginsAddCommand {
            scope: self.scope.into(),
            path: self.path,
        }
    }
}
impl PluginsValidateCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PluginsValidateCommand {
        crate::configuration::PluginsValidateCommand {
            target: self.target,
            json: self.json,
        }
    }
}
impl PluginsListCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PluginsListCommand {
        crate::configuration::PluginsListCommand {
            all: self.all,
            json: self.json,
        }
    }
}
impl PluginsInspectCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PluginsInspectCommand {
        crate::configuration::PluginsInspectCommand {
            id: self.id,
            json: self.json,
        }
    }
}
impl PluginsEnableCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PluginsEnableCommand {
        crate::configuration::PluginsEnableCommand { id: self.id }
    }
}
impl PluginsDisableCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PluginsDisableCommand {
        crate::configuration::PluginsDisableCommand { id: self.id }
    }
}
impl PluginsRemoveCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PluginsRemoveCommand {
        crate::configuration::PluginsRemoveCommand { id: self.id }
    }
}

impl From<PricingScopeArgs> for crate::configuration::PricingScopeArgs {
    fn from(value: PricingScopeArgs) -> Self {
        Self {
            user: value.user,
            project: value.project,
            global: value.global,
        }
    }
}
impl PricingValidateCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PricingValidateCommand {
        crate::configuration::PricingValidateCommand { path: self.path }
    }
}
impl PricingInitCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PricingInitCommand {
        crate::configuration::PricingInitCommand {
            scope: self.scope.into(),
        }
    }
}
impl PricingAddSourceCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PricingAddSourceCommand {
        crate::configuration::PricingAddSourceCommand {
            scope: self.scope.into(),
            path: self.path,
            append: self.append,
        }
    }
}
impl PricingResolveCommand {
    pub(crate) fn into_runtime(self) -> crate::configuration::PricingResolveCommand {
        crate::configuration::PricingResolveCommand {
            model: self.model,
            provider: self.provider,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            cache_read_tokens: self.cache_read_tokens,
            cache_write_tokens: self.cache_write_tokens,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum GatewayModeArg {
    HookOnly,
    Passthrough,
    Required,
}

impl From<GatewayModeArg> for crate::configuration::GatewayMode {
    fn from(value: GatewayModeArg) -> Self {
        match value {
            GatewayModeArg::HookOnly => Self::HookOnly,
            GatewayModeArg::Passthrough => Self::Passthrough,
            GatewayModeArg::Required => Self::Required,
        }
    }
}
