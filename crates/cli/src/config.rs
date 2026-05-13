// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::http::HeaderMap;
use clap::{Args, Parser, Subcommand, ValueEnum};
use nemo_flow::observability::atof::AtofExporterMode;
use serde::Deserialize;
use serde_json::Value;

use crate::error::CliError;

#[derive(Debug, Clone, Parser)]
#[command(name = "nemo-flow")]
#[command(about = "Coding-agent gateway for NeMo Flow observability")]
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
        long_about = "Run Anthropic's `claude` CLI under an ephemeral NeMo Flow gateway. \
                      Observability (ATIF + OpenInference) is wired in transparently via \
                      ANTHROPIC_BASE_URL. First-time use launches the setup wizard so the \
                      `[agents.claude]` block lands in `.nemo-flow/config.toml` and observation \
                      starts on the next invocation without prompts.",
        after_help = "Examples:\n  \
                      nemo-flow claude\n  \
                      nemo-flow claude -- chat \"refactor the launcher\"\n  \
                      nemo-flow claude -- --resume <session-id>"
    )]
    Claude(EasyPathCommand),
    /// Run Codex with observability (setup on first use)
    #[command(
        long_about = "Run OpenAI's `codex` CLI under an ephemeral NeMo Flow gateway. NeMo Flow \
                      injects a `nemo-flow-openai` provider override so codex points at the \
                      gateway; the gateway then forwards to `--openai-base-url` (defaults to \
                      api.openai.com) with `OPENAI_API_KEY` injected on the codex route (see \
                      NMF-86 — codex's own auth.json JWT is stripped). Requires codex-cli >= \
                      0.129.0.",
        after_help = "Examples:\n  \
                      nemo-flow codex\n  \
                      nemo-flow codex -- exec \"fix the bug in foo.rs\"\n  \
                      nemo-flow --openai-base-url https://inference-api.nvidia.com codex"
    )]
    Codex(EasyPathCommand),
    /// Run Cursor with observability (setup on first use)
    #[command(
        long_about = "Run Cursor's `cursor-agent` CLI under an ephemeral NeMo Flow gateway. The \
                      launcher temporarily patches `.cursor/hooks.json` in the project root \
                      during the run and restores it on exit. Disable that via \
                      `[agents.cursor] patch_restore_hooks = false` in config.toml if you \
                      maintain `.cursor/hooks.json` yourself.",
        after_help = "Examples:\n  \
                      nemo-flow cursor\n  \
                      nemo-flow cursor -- agent --resume <session-id>"
    )]
    Cursor(EasyPathCommand),
    /// Run Hermes with observability (setup on first use)
    #[command(
        long_about = "Run NVIDIA's Hermes agent under a NeMo Flow gateway. Hermes reads hooks \
                      from `.hermes/config.yaml`; first-run setup writes that file alongside \
                      `.nemo-flow/config.toml` so every subsequent invocation traces \
                      automatically. Re-run `nemo-flow config hermes` to refresh the hooks.",
        after_help = "Examples:\n  \
                      nemo-flow hermes\n  \
                      nemo-flow hermes -- chat --provider custom"
    )]
    Hermes(EasyPathCommand),
    /// Run the interactive setup (writes `.nemo-flow/config.toml`)
    Config(ConfigCommand),
    /// Diagnose env, agents, config, observability (optionally scoped to one agent)
    Doctor(DoctorCommand),
    /// List supported and locally-detected agents (use `--json` for machine output)
    Agents(AgentsCommand),
    /// Print shell completion script (e.g. `nemo-flow completions zsh > ~/.zfunc/_nemo-flow`)
    Completions(CompletionsCommand),
    /// Run an agent deterministically (no wizard; errors if config is missing)
    Run(RunCommand),
    /// Internal: subprocess used by installed hooks to forward events. Not typed by humans.
    #[command(hide = true)]
    HookForward(HookForwardCommand),
}

/// Args for `nemo-flow doctor`. `--json` is on this command (rather than as a global flag)
/// so it doesn't pollute the help output of subcommands where it has no meaning.
#[derive(Debug, Clone, Args)]
pub(crate) struct DoctorCommand {
    /// Limit readiness checks to one supported agent.
    #[arg(value_enum)]
    pub(crate) agent: Option<CodingAgent>,
    /// Emit machine-readable JSON instead of the formatted human report. Versioned via
    /// `schema_version`; stable shape for CI / evaluation harness consumption.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Args for `nemo-flow agents`. Shares the `--json` shape with `nemo-flow doctor`'s
/// `agents` field so the two outputs can be unified by downstream consumers.
#[derive(Debug, Clone, Args)]
pub(crate) struct AgentsCommand {
    /// Emit the supported + detected agent list as JSON instead of formatted text.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Args for `nemo-flow completions <shell>` (print to stdout) or `nemo-flow completions --install`
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

/// Args for `nemo-flow config`. The setup wizard runs by default; `--reset` short-circuits to
/// a destructive clear. An optional positional agent name scopes both the wizard and `--reset`
/// to a single agent's settings, leaving other agents' blocks untouched.
#[derive(Debug, Clone, Args)]
pub(crate) struct ConfigCommand {
    /// Scope this run to one agent. Wizard skips the agent multi-select; `--reset` removes
    /// only that agent's block from the existing config file. Omit to operate on all agents.
    #[arg(value_enum)]
    pub(crate) agent: Option<CodingAgent>,
    /// Delete the project config file (or remove just the scoped agent's block when an agent
    /// is named). The wizard does NOT run after a reset — invoke `nemo-flow config` again to
    /// re-create the file from scratch.
    #[arg(long)]
    pub(crate) reset: bool,
}

#[derive(Debug, Clone, Default, Args)]
pub(crate) struct ServerArgs {
    /// Path to an explicit config file (disables auto-discovery of workspace/global/system)
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,
    /// Address for the gateway to listen on in daemon mode (default 127.0.0.1:4040)
    #[arg(long, env = "NEMO_FLOW_GATEWAY_BIND")]
    pub(crate) bind: Option<SocketAddr>,
    /// Upstream OpenAI-compatible base URL (e.g. https://api.openai.com, NVIDIA inference)
    #[arg(long, env = "NEMO_FLOW_OPENAI_BASE_URL")]
    pub(crate) openai_base_url: Option<String>,
    /// Upstream Anthropic base URL (e.g. https://api.anthropic.com)
    #[arg(long, env = "NEMO_FLOW_ANTHROPIC_BASE_URL")]
    pub(crate) anthropic_base_url: Option<String>,
    /// Directory to write ATIF trajectory JSON files into per session
    #[arg(long, env = "NEMO_FLOW_ATIF_DIR")]
    pub(crate) atif_dir: Option<PathBuf>,
    /// Directory to write per-event ATOF JSONL files into (one event per line, raw ATOF shape)
    #[arg(long, env = "NEMO_FLOW_ATOF_DIR")]
    pub(crate) atof_dir: Option<PathBuf>,
    /// OpenInference-compatible OTLP HTTP endpoint for streaming spans (Phoenix, Arize, etc.)
    #[arg(long, env = "NEMO_FLOW_OPENINFERENCE_ENDPOINT")]
    pub(crate) openinference_endpoint: Option<String>,
}

impl ServerArgs {
    /// True when the user passed any flag that signals "I want the gateway, not the wizard." Used
    /// by the bare `nemo-flow` dispatch to choose between launching the long-running daemon and
    /// dropping into setup. `--config` is included: someone running `nemo-flow --config <path>`
    /// with no subcommand has explicitly pointed at a config file, which is only meaningful for
    /// daemon startup — the wizard creates configs, it doesn't consume them.
    pub(crate) fn requested_daemon_mode(&self) -> bool {
        self.bind.is_some()
            || self.openai_base_url.is_some()
            || self.anthropic_base_url.is_some()
            || self.atif_dir.is_some()
            || self.atof_dir.is_some()
            || self.openinference_endpoint.is_some()
            || self.config.is_some()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GatewayConfig {
    pub(crate) bind: SocketAddr,
    pub(crate) openai_base_url: String,
    pub(crate) anthropic_base_url: String,
    pub(crate) exporters: ExportersConfig,
    pub(crate) metadata: Option<Value>,
    pub(crate) plugin_config: Option<Value>,
}

/// Sinks the gateway writes observability data to. Each exporter has its own nested config so
/// exporter-specific options (for example ATOF append/overwrite behavior) do not get flattened
/// into unrelated backends.
#[derive(Debug, Clone, Default)]
pub(crate) struct ExportersConfig {
    pub(crate) atif: AtifExporterSettings,
    pub(crate) atof: AtofExporterSettings,
    pub(crate) openinference: OpenInferenceExporterSettings,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AtifExporterSettings {
    pub(crate) dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub(crate) struct AtofExporterSettings {
    pub(crate) dir: Option<PathBuf>,
    pub(crate) mode: AtofExporterMode,
    pub(crate) filename_template: String,
}

impl Default for AtofExporterSettings {
    fn default() -> Self {
        Self {
            dir: None,
            mode: AtofExporterMode::Append,
            filename_template: "{session_id}.jsonl".into(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct OpenInferenceExporterSettings {
    pub(crate) endpoint: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct HookForwardCommand {
    #[arg(value_enum)]
    pub(crate) agent: CodingAgent,
    #[arg(long)]
    pub(crate) gateway_url: Option<String>,
    #[arg(long)]
    pub(crate) atif_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) atof_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) openinference_endpoint: Option<String>,
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

/// Args for the easy-path agent shortcut (`nemo-flow claude`, `nemo-flow codex`, etc.).
/// Holds only pass-through agent args; the agent itself is selected by which subcommand variant
/// is invoked, and all observability/upstream settings come from the resolved config file. If no
/// config file is present, the dispatcher fires setup (Phase 3). Phase 2 errors with a
/// pointer to `nemo-flow config` since setup isn't wired up yet.
#[derive(Debug, Clone, Args)]
pub(crate) struct EasyPathCommand {
    /// Pass-through args forwarded to the underlying agent process. Use `--` to separate them
    /// from `nemo-flow`'s own flags. See the `Examples` section below for agent-specific shapes.
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
    pub(crate) atif_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) atof_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) openinference_endpoint: Option<String>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum GatewayMode {
    HookOnly,
    Passthrough,
    Required,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SessionConfig {
    pub(crate) exporters: ExportersConfig,
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
        let exporters = ExportersConfig {
            atif: AtifExporterSettings {
                dir: header_string(headers, "x-nemo-flow-atif-dir")
                    .map(PathBuf::from)
                    .or_else(|| self.exporters.atif.dir.clone()),
            },
            atof: AtofExporterSettings {
                dir: header_string(headers, "x-nemo-flow-atof-dir")
                    .map(PathBuf::from)
                    .or_else(|| self.exporters.atof.dir.clone()),
                ..self.exporters.atof.clone()
            },
            openinference: OpenInferenceExporterSettings {
                endpoint: header_string(headers, "x-nemo-flow-openinference-endpoint")
                    .or_else(|| self.exporters.openinference.endpoint.clone()),
            },
        };
        let metadata =
            header_json(headers, "x-nemo-flow-session-metadata").or_else(|| self.metadata.clone());
        let plugin_config = header_json(headers, "x-nemo-flow-plugin-config")
            .or_else(|| self.plugin_config.clone());
        let profile = header_string(headers, "x-nemo-flow-config-profile");
        let gateway_mode = header_string(headers, "x-nemo-flow-gateway-mode");
        SessionConfig {
            exporters,
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
    /// Recorded by `nemo-flow config` when it installs hermes shell hooks. Other agents leave
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
    upstream: Option<FileUpstreamConfig>,
    exporters: Option<FileExportersConfig>,
    observability: Option<FileObservabilityConfig>,
    export: Option<FileExportConfig>,
    plugins: Option<FilePluginsConfig>,
    agents: Option<FileAgentsConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileUpstreamConfig {
    openai_base_url: Option<String>,
    anthropic_base_url: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileObservabilityConfig {
    atif_dir: Option<PathBuf>,
    atof_dir: Option<PathBuf>,
    metadata: Option<Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileExportersConfig {
    atif: Option<FileAtifExporterConfig>,
    atof: Option<FileAtofExporterConfig>,
    openinference: Option<FileOpenInferenceConfig>,
    // Legacy flat `[exporters]` keys from early CLI builds.
    atif_dir: Option<PathBuf>,
    atof_dir: Option<PathBuf>,
    openinference_endpoint: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileAtifExporterConfig {
    dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileAtofExporterConfig {
    dir: Option<PathBuf>,
    mode: Option<String>,
    filename_template: Option<String>,
}

// Legacy `[export.<backend>]` shape. New configs use `[exporters]`; this stays readable so
// existing user files do not break.
#[derive(Debug, Clone, Default, Deserialize)]
struct FileExportConfig {
    openinference: Option<FileOpenInferenceConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileOpenInferenceConfig {
    endpoint: Option<String>,
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
    // Anthropic requests to their public bases, and leave exporters/plugins disabled until config,
    // environment, or headers explicitly opt in.
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:4040"
                .parse()
                .expect("valid default bind address"),
            openai_base_url: "https://api.openai.com".into(),
            anthropic_base_url: "https://api.anthropic.com".into(),
            exporters: ExportersConfig::default(),
            metadata: None,
            plugin_config: None,
        }
    }
}

/// Resolves server-mode configuration from shared config files plus server CLI/environment overrides.
///
/// File discovery and merge behavior live in `load_shared_config`; this function only applies the
/// server-facing command-line layer so launcher-only settings cannot leak into daemon mode.
pub(crate) fn resolve_server_config(args: &ServerArgs) -> Result<ResolvedConfig, CliError> {
    let mut resolved = load_shared_config(args.config.as_ref())?;
    apply_server_overrides(&mut resolved.gateway, args);
    Ok(resolved)
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
        apply_server_overrides(&mut resolved.gateway, args);
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
    if let Some(value) = &command.atif_dir {
        config.exporters.atif.dir = Some(value.clone());
    }
    if let Some(value) = &command.atof_dir {
        config.exporters.atof.dir = Some(value.clone());
    }
    if let Some(value) = &command.openinference_endpoint {
        config.exporters.openinference.endpoint = Some(value.clone());
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
        config.plugin_config = Some(parse_json_option("plugin config", value)?);
    }
    Ok(())
}

// Applies direct server flags on top of already-merged configuration. Only present options mutate
// the config so lower-priority file values survive when a flag was omitted.
fn apply_server_overrides(config: &mut GatewayConfig, args: &ServerArgs) {
    if let Some(value) = args.bind {
        config.bind = value;
    }
    if let Some(value) = &args.openai_base_url {
        config.openai_base_url = value.clone();
    }
    if let Some(value) = &args.anthropic_base_url {
        config.anthropic_base_url = value.clone();
    }
    if let Some(value) = &args.atif_dir {
        config.exporters.atif.dir = Some(value.clone());
    }
    if let Some(value) = &args.atof_dir {
        config.exporters.atof.dir = Some(value.clone());
    }
    if let Some(value) = &args.openinference_endpoint {
        config.exporters.openinference.endpoint = Some(value.clone());
    }
}

// Loads config from the ordered shared locations, deep-merges TOML tables, maps the typed file
// shape onto runtime structs, then lets environment variables override file values. Invalid TOML
// or typed shapes fail closed because they indicate an operator configuration error.
fn load_shared_config(explicit: Option<&PathBuf>) -> Result<ResolvedConfig, CliError> {
    let mut merged = toml::Value::Table(toml::map::Map::new());
    for path in config_paths(explicit) {
        if path.exists() {
            let raw = std::fs::read_to_string(&path)?;
            let parsed = raw
                .parse::<toml::Table>()
                .map(toml::Value::Table)
                .map_err(|error| {
                    CliError::Config(format!("invalid TOML in {}: {error}", path.display()))
                })?;
            merge_toml(&mut merged, parsed);
        }
    }
    let mut resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        ..ResolvedConfig::default()
    };
    apply_file_config(&mut resolved, merged)?;
    apply_env_config(&mut resolved.gateway);
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
    let mut paths = vec![PathBuf::from("/etc/nemo-flow/config.toml")];
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

// Walks upward from the current directory and returns the nearest project-local gateway config.
// The first hit wins so nested projects can override parent workspace defaults.
fn find_project_config(start: &std::path::Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        let path = ancestor.join(".nemo-flow/config.toml");
        if path.exists() {
            return Some(path);
        }
    }
    None
}

// Resolves the user config using XDG first and HOME/USERPROFILE second. Returning `None` keeps
// config loading portable in minimal environments where no home directory is visible.
fn user_config_path() -> Option<PathBuf> {
    user_config_dir().map(|dir| dir.join("config.toml"))
}

/// Resolves the nemo-flow user config DIRECTORY (without trailing filename) using the same XDG
/// rules as `user_config_path`. Exposed so wizard/doctor code paths that write to or display
/// the global location stay in sync with the loader — without this, hard-coded
/// `$HOME/.config/nemo-flow` references silently ignore `$XDG_CONFIG_HOME`.
pub(crate) fn user_config_dir() -> Option<PathBuf> {
    if let Some(base) = std::env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(base).join("nemo-flow"));
    }
    home_dir().map(|home| home.join(".config/nemo-flow"))
}

// Applies the typed TOML config model to the resolved runtime config. Missing sections and fields
// are ignored, preserving defaults and prior merge layers; Cursor's patch-restore flag is only
// changed when explicitly present.
fn apply_file_config(resolved: &mut ResolvedConfig, value: toml::Value) -> Result<(), CliError> {
    let config: FileConfig = value.try_into().map_err(|error| {
        CliError::Config(format!("invalid gateway configuration shape: {error}"))
    })?;
    apply_file_upstream_config(&mut resolved.gateway, config.upstream);
    apply_file_observability_config(&mut resolved.gateway, config.observability);
    apply_file_export_config(&mut resolved.gateway, config.export);
    apply_file_exporters_config(&mut resolved.gateway, config.exporters)?;
    apply_file_plugins_config(&mut resolved.gateway, config.plugins);
    apply_file_agents_config(&mut resolved.agents, config.agents);
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

// Applies legacy observability sinks plus session metadata tags applied to every span/trajectory.
// New configs put exporter sinks under `[exporters]`; these fields remain readable for existing
// config files.
fn apply_file_observability_config(
    gateway: &mut GatewayConfig,
    observability: Option<FileObservabilityConfig>,
) {
    let Some(observability) = observability else {
        return;
    };
    if let Some(value) = observability.atif_dir {
        gateway.exporters.atif.dir = Some(value);
    }
    if let Some(value) = observability.atof_dir {
        gateway.exporters.atof.dir = Some(value);
    }
    if let Some(value) = observability.metadata {
        gateway.metadata = Some(value);
    }
}

// Applies legacy optional OpenInference export config. New configs use `[exporters]`.
fn apply_file_export_config(gateway: &mut GatewayConfig, export: Option<FileExportConfig>) {
    let Some(export) = export else {
        return;
    };
    if let Some(openinference) = export.openinference
        && let Some(value) = openinference.endpoint
    {
        gateway.exporters.openinference.endpoint = Some(value);
    }
}

// Applies the current exporter config shape. This runs after the legacy shapes so `[exporters]`
// wins when a file contains both old and new keys.
fn apply_file_exporters_config(
    gateway: &mut GatewayConfig,
    exporters: Option<FileExportersConfig>,
) -> Result<(), CliError> {
    let Some(exporters) = exporters else {
        return Ok(());
    };
    if let Some(value) = exporters.atif_dir {
        gateway.exporters.atif.dir = Some(value);
    }
    if let Some(value) = exporters.atof_dir {
        gateway.exporters.atof.dir = Some(value);
    }
    if let Some(value) = exporters.openinference_endpoint {
        gateway.exporters.openinference.endpoint = Some(value);
    }
    if let Some(atif) = exporters.atif
        && let Some(value) = atif.dir
    {
        gateway.exporters.atif.dir = Some(value);
    }
    if let Some(atof) = exporters.atof {
        if let Some(value) = atof.dir {
            gateway.exporters.atof.dir = Some(value);
        }
        if let Some(value) = atof.mode {
            gateway.exporters.atof.mode = AtofExporterMode::parse(&value).ok_or_else(|| {
                CliError::Config(format!(
                    "invalid [exporters.atof].mode `{value}`; expected append or overwrite"
                ))
            })?;
        }
        if let Some(value) = atof.filename_template {
            gateway.exporters.atof.filename_template = value;
        }
    }
    if let Some(openinference) = exporters.openinference
        && let Some(value) = openinference.endpoint
    {
        gateway.exporters.openinference.endpoint = Some(value);
    }
    Ok(())
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
// preserve existing startup behavior, while string/path values replace earlier layers when present.
fn apply_env_config(config: &mut GatewayConfig) {
    if let Ok(value) = std::env::var("NEMO_FLOW_GATEWAY_BIND")
        && let Ok(value) = value.parse()
    {
        config.bind = value;
    }
    if let Ok(value) = std::env::var("NEMO_FLOW_OPENAI_BASE_URL") {
        config.openai_base_url = value;
    }
    if let Ok(value) = std::env::var("NEMO_FLOW_ANTHROPIC_BASE_URL") {
        config.anthropic_base_url = value;
    }
    if let Some(value) = std::env::var_os("NEMO_FLOW_ATIF_DIR") {
        config.exporters.atif.dir = Some(PathBuf::from(value));
    }
    if let Some(value) = std::env::var_os("NEMO_FLOW_ATOF_DIR") {
        config.exporters.atof.dir = Some(PathBuf::from(value));
    }
    if let Ok(value) = std::env::var("NEMO_FLOW_OPENINFERENCE_ENDPOINT") {
        config.exporters.openinference.endpoint = Some(value);
    }
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

// Parses JSON-valued CLI options into runtime metadata/config values and labels errors with the
// user-facing option name so callers can report which structured argument was malformed.
fn parse_json_option(name: &str, value: &str) -> Result<Value, CliError> {
    serde_json::from_str::<Value>(value)
        .map_err(|error| CliError::Config(format!("invalid {name}: {error}")))
}

// Resolves a cross-platform home directory from environment only. The gateway avoids extra OS
// lookups here so tests can control install/config locations by setting env variables.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
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
