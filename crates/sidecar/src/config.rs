// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::http::HeaderMap;
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Deserialize;
use serde_json::Value;

use crate::error::SidecarError;

#[derive(Debug, Clone, Parser)]
#[command(name = "nemo-flow-sidecar")]
#[command(about = "Gateway sidecar for coding-agent NeMo Flow observability")]
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
    #[arg(long, env = "NEMO_FLOW_SIDECAR_BIND")]
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
pub(crate) struct SidecarConfig {
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
    pub(crate) sidecar_url: String,
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
    pub(crate) sidecar_url: Option<String>,
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

impl SidecarConfig {
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
    pub(crate) sidecar: SidecarConfig,
    pub(crate) agents: AgentConfigs,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AgentConfigs {
    pub(crate) claude_code: AgentCommandConfig,
    pub(crate) codex: AgentCommandConfig,
    pub(crate) cursor: CursorAgentConfig,
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

impl Default for SidecarConfig {
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

pub(crate) fn resolve_server_config(args: &ServerArgs) -> Result<ResolvedConfig, SidecarError> {
    let mut resolved = load_shared_config(args.config.as_ref())?;
    apply_server_overrides(&mut resolved.sidecar, args);
    Ok(resolved)
}

pub(crate) fn resolve_run_config(
    command: &RunCommand,
    inherited: Option<&ServerArgs>,
) -> Result<ResolvedConfig, SidecarError> {
    let config = command
        .config
        .as_ref()
        .or_else(|| inherited.and_then(|args| args.config.as_ref()));
    let mut resolved = load_shared_config(config)?;
    if let Some(args) = inherited {
        apply_server_overrides(&mut resolved.sidecar, args);
    }
    if let Some(value) = &command.openai_base_url {
        resolved.sidecar.openai_base_url = value.clone();
    }
    if let Some(value) = &command.anthropic_base_url {
        resolved.sidecar.anthropic_base_url = value.clone();
    }
    if let Some(value) = &command.atif_dir {
        resolved.sidecar.atif_dir = Some(value.clone());
    }
    if let Some(value) = &command.openinference_endpoint {
        resolved.sidecar.openinference_endpoint = Some(value.clone());
    }
    if let Some(value) = &command.session_metadata {
        resolved.sidecar.metadata = Some(parse_json_option("session metadata", value)?);
    }
    if let Some(value) = &command.plugin_config {
        resolved.sidecar.plugin_config = Some(parse_json_option("plugin config", value)?);
    }
    resolved.sidecar.bind = "127.0.0.1:0"
        .parse()
        .expect("valid transparent bind address");
    Ok(resolved)
}

fn apply_server_overrides(config: &mut SidecarConfig, args: &ServerArgs) {
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

fn load_shared_config(explicit: Option<&PathBuf>) -> Result<ResolvedConfig, SidecarError> {
    let mut merged = toml::Value::Table(toml::map::Map::new());
    for path in config_paths(explicit) {
        if path.exists() {
            let raw = std::fs::read_to_string(&path)?;
            let parsed = raw
                .parse::<toml::Table>()
                .map(toml::Value::Table)
                .map_err(|error| {
                    SidecarError::Config(format!("invalid TOML in {}: {error}", path.display()))
                })?;
            merge_toml(&mut merged, parsed);
        }
    }
    let mut resolved = ResolvedConfig {
        sidecar: SidecarConfig::default(),
        ..ResolvedConfig::default()
    };
    apply_file_config(&mut resolved, merged)?;
    apply_env_config(&mut resolved.sidecar);
    Ok(resolved)
}

fn config_paths(explicit: Option<&PathBuf>) -> Vec<PathBuf> {
    if let Some(path) = explicit {
        return vec![path.clone()];
    }
    let mut paths = vec![PathBuf::from("/etc/nemo-flow/sidecar.toml")];
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

fn find_project_config(start: &std::path::Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        let path = ancestor.join(".nemo-flow/sidecar.toml");
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn user_config_path() -> Option<PathBuf> {
    if let Some(base) = std::env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(base).join("nemo-flow/sidecar.toml"));
    }
    home_dir().map(|home| home.join(".config/nemo-flow/sidecar.toml"))
}

fn apply_file_config(
    resolved: &mut ResolvedConfig,
    value: toml::Value,
) -> Result<(), SidecarError> {
    let config: FileConfig = value.try_into().map_err(|error| {
        SidecarError::Config(format!("invalid sidecar configuration shape: {error}"))
    })?;
    if let Some(server) = config.server {
        if let Some(value) = server.openai_base_url {
            resolved.sidecar.openai_base_url = value;
        }
        if let Some(value) = server.anthropic_base_url {
            resolved.sidecar.anthropic_base_url = value;
        }
    }
    if let Some(session) = config.session {
        if let Some(value) = session.atif_dir {
            resolved.sidecar.atif_dir = Some(value);
        }
        if let Some(value) = session.metadata {
            resolved.sidecar.metadata = Some(value);
        }
        if let Some(value) = session.plugin_config {
            resolved.sidecar.plugin_config = Some(value);
        }
    }
    if let Some(export) = config.export
        && let Some(openinference) = export.openinference
        && let Some(value) = openinference.endpoint
    {
        resolved.sidecar.openinference_endpoint = Some(value);
    }
    if let Some(agents) = config.agents {
        if let Some(value) = agents.claude_code {
            resolved.agents.claude_code.command = value.command;
        }
        if let Some(value) = agents.codex {
            resolved.agents.codex.command = value.command;
        }
        if let Some(value) = agents.cursor {
            resolved.agents.cursor.command = value.command;
            if let Some(patch_restore_hooks) = value.patch_restore_hooks {
                resolved.agents.cursor.patch_restore_hooks = patch_restore_hooks;
            }
        }
    }
    Ok(())
}

fn apply_env_config(config: &mut SidecarConfig) {
    if let Ok(value) = std::env::var("NEMO_FLOW_SIDECAR_BIND")
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

fn parse_json_option(name: &str, value: &str) -> Result<Value, SidecarError> {
    serde_json::from_str::<Value>(value)
        .map_err(|error| SidecarError::Config(format!("invalid {name}: {error}")))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

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
    pub(crate) const fn hook_path(self) -> &'static str {
        match self {
            Self::ClaudeCode => "/hooks/claude-code",
            Self::Codex => "/hooks/codex",
            Self::Cursor => "/hooks/cursor",
        }
    }

    pub(crate) const fn as_arg(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Codex => "codex",
            Self::Cursor => "cursor",
        }
    }

    pub(crate) fn infer(command: &str) -> Option<Self> {
        let name = std::path::Path::new(command)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(command);
        match name {
            "claude" | "claude-code" => Some(Self::ClaudeCode),
            "codex" => Some(Self::Codex),
            "cursor" | "cursor-agent" => Some(Self::Cursor),
            _ => None,
        }
    }
}

impl GatewayMode {
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
