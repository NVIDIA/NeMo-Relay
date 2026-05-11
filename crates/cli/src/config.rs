// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::http::HeaderMap;
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Deserialize;
use serde_json::Value;

use crate::error::CliError;

#[derive(Debug, Clone, Parser)]
#[command(name = "nemo-flow")]
#[command(about = "Coding-agent gateway for NeMo Flow observability")]
pub(crate) struct Cli {
    #[command(flatten)]
    pub(crate) server: ServerArgs,
    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum Command {
    Install(InstallCommand),
    HookForward(HookForwardCommand),
    Run(RunCommand),
}

#[derive(Debug, Clone, Default, Args)]
pub(crate) struct ServerArgs {
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,
    #[arg(long, env = "NEMO_FLOW_GATEWAY_BIND")]
    pub(crate) bind: Option<SocketAddr>,
    #[arg(long, env = "NEMO_FLOW_OPENAI_BASE_URL")]
    pub(crate) openai_base_url: Option<String>,
    #[arg(long, env = "NEMO_FLOW_ANTHROPIC_BASE_URL")]
    pub(crate) anthropic_base_url: Option<String>,
    #[arg(long, env = "NEMO_FLOW_ATIF_DIR")]
    pub(crate) atif_dir: Option<PathBuf>,
    #[arg(long, env = "NEMO_FLOW_OPENINFERENCE_ENDPOINT")]
    pub(crate) openinference_endpoint: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct GatewayConfig {
    pub(crate) bind: SocketAddr,
    pub(crate) openai_base_url: String,
    pub(crate) anthropic_base_url: String,
    pub(crate) atif_dir: Option<PathBuf>,
    pub(crate) openinference_endpoint: Option<String>,
    pub(crate) metadata: Option<Value>,
    pub(crate) plugin_config: Option<Value>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct InstallCommand {
    #[arg(value_enum)]
    pub(crate) agent: CodingAgent,
    #[arg(long, value_enum, default_value = "user")]
    pub(crate) scope: InstallScope,
    #[arg(long, value_enum, default_value = "both")]
    pub(crate) target: InstallTarget,
    #[arg(long, default_value = "http://127.0.0.1:4040")]
    pub(crate) gateway_url: String,
    #[arg(long)]
    pub(crate) atif_dir: Option<PathBuf>,
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
    pub(crate) dry_run: bool,
    #[arg(long)]
    pub(crate) print: bool,
    #[arg(long, hide = true)]
    pub(crate) home_dir: Option<PathBuf>,
    #[arg(long, hide = true)]
    pub(crate) project_dir: Option<PathBuf>,
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
    ClaudeCode,
    Codex,
    Cursor,
    Hermes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum InstallScope {
    User,
    Project,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum InstallTarget {
    Cli,
    Gui,
    Both,
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
    pub(crate) atif_dir: Option<PathBuf>,
    pub(crate) openinference_endpoint: Option<String>,
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
        let atif_dir = header_string(headers, "x-nemo-flow-atif-dir")
            .map(PathBuf::from)
            .or_else(|| self.atif_dir.clone());
        let openinference_endpoint = header_string(headers, "x-nemo-flow-openinference-endpoint")
            .or_else(|| self.openinference_endpoint.clone());
        let metadata =
            header_json(headers, "x-nemo-flow-session-metadata").or_else(|| self.metadata.clone());
        let plugin_config = header_json(headers, "x-nemo-flow-plugin-config")
            .or_else(|| self.plugin_config.clone());
        let profile = header_string(headers, "x-nemo-flow-config-profile");
        let gateway_mode = header_string(headers, "x-nemo-flow-gateway-mode");
        SessionConfig {
            atif_dir,
            openinference_endpoint,
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
    pub(crate) claude_code: AgentCommandConfig,
    pub(crate) codex: AgentCommandConfig,
    pub(crate) cursor: CursorAgentConfig,
    pub(crate) hermes: AgentCommandConfig,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AgentCommandConfig {
    pub(crate) command: Option<String>,
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

#[derive(Debug, Clone, Default, Deserialize)]
struct FileConfig {
    server: Option<FileServerConfig>,
    session: Option<FileSessionConfig>,
    export: Option<FileExportConfig>,
    agents: Option<FileAgentsConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileServerConfig {
    openai_base_url: Option<String>,
    anthropic_base_url: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileSessionConfig {
    atif_dir: Option<PathBuf>,
    metadata: Option<Value>,
    plugin_config: Option<Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileExportConfig {
    openinference: Option<FileOpenInferenceConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileOpenInferenceConfig {
    endpoint: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileAgentsConfig {
    #[serde(rename = "claude-code")]
    claude_code: Option<FileAgentCommandConfig>,
    codex: Option<FileAgentCommandConfig>,
    cursor: Option<FileCursorAgentConfig>,
    hermes: Option<FileAgentCommandConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileAgentCommandConfig {
    command: Option<String>,
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
            atif_dir: None,
            openinference_endpoint: None,
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
        config.atif_dir = Some(value.clone());
    }
    if let Some(value) = &command.openinference_endpoint {
        config.openinference_endpoint = Some(value.clone());
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
        config.atif_dir = Some(value.clone());
    }
    if let Some(value) = &args.openinference_endpoint {
        config.openinference_endpoint = Some(value.clone());
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

// Returns the config search path. An explicit path disables implicit discovery; otherwise system
// config is lowest priority, the nearest project config is next, and user config is merged last.
fn config_paths(explicit: Option<&PathBuf>) -> Vec<PathBuf> {
    if let Some(path) = explicit {
        return vec![path.clone()];
    }
    let mut paths = vec![PathBuf::from("/etc/nemo-flow/gateway.toml")];
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
        let path = ancestor.join(".nemo-flow/gateway.toml");
        if path.exists() {
            return Some(path);
        }
    }
    None
}

// Resolves the user config using XDG first and HOME/USERPROFILE second. Returning `None` keeps
// config loading portable in minimal environments where no home directory is visible.
fn user_config_path() -> Option<PathBuf> {
    if let Some(base) = std::env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(base).join("nemo-flow/gateway.toml"));
    }
    home_dir().map(|home| home.join(".config/nemo-flow/gateway.toml"))
}

// Applies the typed TOML config model to the resolved runtime config. Missing sections and fields
// are ignored, preserving defaults and prior merge layers; Cursor's patch-restore flag is only
// changed when explicitly present.
fn apply_file_config(resolved: &mut ResolvedConfig, value: toml::Value) -> Result<(), CliError> {
    let config: FileConfig = value.try_into().map_err(|error| {
        CliError::Config(format!("invalid gateway configuration shape: {error}"))
    })?;
    apply_file_server_config(&mut resolved.gateway, config.server);
    apply_file_session_config(&mut resolved.gateway, config.session);
    apply_file_export_config(&mut resolved.gateway, config.export);
    apply_file_agents_config(&mut resolved.agents, config.agents);
    Ok(())
}

// Applies provider upstream defaults from file config. These values are the upstream targets used
// by direct gateway server mode; transparent `run` mode can still override them per invocation.
fn apply_file_server_config(gateway: &mut GatewayConfig, server: Option<FileServerConfig>) {
    let Some(server) = server else {
        return;
    };
    if let Some(value) = server.openai_base_url {
        gateway.openai_base_url = value;
    }
    if let Some(value) = server.anthropic_base_url {
        gateway.anthropic_base_url = value;
    }
}

// Applies session-level exporter and metadata defaults. Missing optional fields leave earlier
// merge layers intact, which preserves global or project defaults when user config is partial.
fn apply_file_session_config(gateway: &mut GatewayConfig, session: Option<FileSessionConfig>) {
    let Some(session) = session else {
        return;
    };
    if let Some(value) = session.atif_dir {
        gateway.atif_dir = Some(value);
    }
    if let Some(value) = session.metadata {
        gateway.metadata = Some(value);
    }
    if let Some(value) = session.plugin_config {
        gateway.plugin_config = Some(value);
    }
}

// Applies optional OpenInference export config. The nested shape mirrors the docs and leaves room
// for future exporter-specific fields without changing the top-level config parser.
fn apply_file_export_config(gateway: &mut GatewayConfig, export: Option<FileExportConfig>) {
    let Some(export) = export else {
        return;
    };
    if let Some(openinference) = export.openinference
        && let Some(value) = openinference.endpoint
    {
        gateway.openinference_endpoint = Some(value);
    }
}

// Applies configured agent commands and Cursor's temporary-hook behavior. Cursor's
// `patch_restore_hooks` flag is intentionally tri-state in file config so omitted values preserve
// the safe default while explicit `false` disables temporary hook mutation.
fn apply_file_agents_config(agents: &mut AgentConfigs, file_agents: Option<FileAgentsConfig>) {
    let Some(file_agents) = file_agents else {
        return;
    };
    if let Some(value) = file_agents.claude_code {
        agents.claude_code.command = value.command;
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
        config.atif_dir = Some(PathBuf::from(value));
    }
    if let Ok(value) = std::env::var("NEMO_FLOW_OPENINFERENCE_ENDPOINT") {
        config.openinference_endpoint = Some(value);
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

    // Returns the CLI spelling used in generated commands and diagnostics. The value intentionally
    // matches clap's kebab-case enum names so install/run output can be copied back into commands.
    pub(crate) const fn as_arg(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
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
