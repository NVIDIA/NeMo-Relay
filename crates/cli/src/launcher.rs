// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use nemo_relay::observability::plugin_component::{
    AtifStorageConfig, OBSERVABILITY_PLUGIN_KIND, ObservabilityConfig,
};
use nemo_relay::plugin::PluginConfig;
use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::process::Command;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::config::{
    AgentConfigs, CodingAgent, EasyPathCommand, GatewayConfig, ResolvedConfig, RunCommand,
    ServerArgs, any_config_file_exists, resolve_run_config,
};
use crate::error::CliError;
use crate::installer::{generated_hooks, hook_forward_command};
use crate::plugins::lifecycle::ActiveDynamicPluginComponent;
use crate::server;

/// Runs a child coding-agent command behind an ephemeral local gateway.
///
/// The gateway binds to an OS-assigned loopback port, prepares agent-specific hook/gateway wiring,
/// waits for health before spawning the child, and restores temporary files after the child and
/// server shut down. The child's exit status is preserved when it fits in `ExitCode`; otherwise the
/// launcher reports generic failure.
pub(crate) async fn run(
    command: RunCommand,
    inherited: Option<&ServerArgs>,
) -> Result<ExitCode, CliError> {
    let run = TransparentRun::new(command, inherited).await?;
    run.print_if_requested();
    run.execute().await
}

/// Runs the easy-path bare-agent shortcut (`nemo-relay claude`, `nemo-relay codex`, etc.).
///
/// If no config file is present at any discovery layer, this fires the interactive setup inline
/// (`crate::setup::run`) which writes a `config.toml`, then proceeds to launch the agent. When
/// config IS present, the easy path constructs a synthetic `RunCommand` and delegates to the
/// same transparent-run pipeline `nemo-relay run` uses — same observability wiring, same agent
/// argv resolution, same lifecycle management.
pub(crate) async fn easy_path(
    agent: CodingAgent,
    command: EasyPathCommand,
    inherited: Option<&ServerArgs>,
) -> Result<ExitCode, CliError> {
    // Explicit `--config <path>` short-circuits the discovery-based setup trigger: when the
    // user has pointed at a specific file, that file is the contract — fire setup only if it
    // doesn't exist yet, and never run setup just because no config lives at any default
    // discovery location.
    let explicit_config = inherited.and_then(|args| args.config.as_deref());
    let needs_setup = match explicit_config {
        Some(path) => !path.exists(),
        None => !any_config_file_exists(),
    };
    if needs_setup {
        // No config anywhere — fire setup inline, scoped to the agent the user typed. After
        // it returns, config discovery will pick up the freshly-written `config.toml` and
        // `run()` below will see a populated environment. If setup errors (non-TTY, user
        // cancelled), surface that directly.
        crate::setup::run(Some(agent)).await?;
    }
    let synthetic = RunCommand {
        agent: Some(agent),
        // Forward the explicit config path so `run` parses the same file the user asked for,
        // rather than re-discovering from defaults.
        config: explicit_config.map(std::path::Path::to_path_buf),
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: false,
        print: false,
        command: command.command,
    };
    run(synthetic, inherited).await
}

struct TransparentRun {
    agent: CodingAgent,
    prepared: PreparedRun,
    resolved: ResolvedConfig,
    dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
    listener: TcpListener,
    gateway_url: String,
    dry_run: bool,
    print: bool,
}

impl TransparentRun {
    // Resolves configuration, binds the ephemeral listener, and builds agent-specific launch wiring
    // without starting the gateway or spawning the child command.
    async fn new(command: RunCommand, inherited: Option<&ServerArgs>) -> Result<Self, CliError> {
        let dry_run = command.dry_run;
        let print = command.print;
        let explicit_config = command
            .config
            .as_ref()
            .or_else(|| inherited.and_then(|args| args.config.as_ref()));
        let mut resolved = resolve_run_config(&command, inherited)?;
        let dynamic_plugins = if dry_run {
            Vec::new()
        } else {
            crate::plugins::lifecycle::active_dynamic_plugin_components(explicit_config, &resolved)?
        };
        let (agent, argv) = resolve_agent_and_argv(&command, &resolved.agents)?;
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let gateway_url = format!("http://{address}");
        resolved.gateway.bind = address;

        let prepared = PreparedRun::new(agent, argv, &gateway_url, &resolved, dry_run)?;
        Ok(Self {
            agent,
            prepared,
            resolved,
            dynamic_plugins,
            listener,
            gateway_url,
            dry_run,
            print,
        })
    }

    // Emits the resolved run plan when requested. Dry runs always print because inspection is their
    // primary behavior; live runs print only when `--print` was passed.
    fn print_if_requested(&self) {
        if self.print || self.dry_run {
            self.prepared
                .print(self.agent, &self.gateway_url, &self.resolved);
        }
    }

    // Runs the prepared child command unless this is an inspection-only dry run.
    async fn execute(self) -> Result<ExitCode, CliError> {
        if self.dry_run {
            return Ok(ExitCode::SUCCESS);
        }
        self.prepared
            .print_live_status(self.agent, &self.gateway_url, &self.resolved);
        execute_live_run_with_dynamic(
            self.listener,
            self.resolved.gateway,
            self.dynamic_plugins,
            &self.gateway_url,
            self.prepared,
        )
        .await
    }
}

// Starts the gateway, waits for readiness, runs the child command, restores temporary state, and then
// maps the child process status to the launcher's exit code.
#[cfg(test)]
async fn execute_live_run(
    listener: TcpListener,
    gateway_config: GatewayConfig,
    gateway_url: &str,
    prepared: PreparedRun,
) -> Result<ExitCode, CliError> {
    execute_live_run_with_dynamic(listener, gateway_config, Vec::new(), gateway_url, prepared).await
}

async fn execute_live_run_with_dynamic(
    listener: TcpListener,
    gateway_config: GatewayConfig,
    dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
    gateway_url: &str,
    prepared: PreparedRun,
) -> Result<ExitCode, CliError> {
    let running_server = RunningGateway::start(listener, gateway_config, dynamic_plugins);
    if let Err(error) = wait_for_health(gateway_url).await {
        let restore = prepared.restore();
        let server_result = running_server.stop().await;
        restore?;
        server_result?;
        return Err(error);
    }
    let status = prepared.spawn_and_wait().await;
    let restore = prepared.restore();
    let server_result = running_server.stop().await;
    restore?;
    server_result?;

    Ok(exit_code(status?))
}

// Resolves the launched agent and argv from either an explicit command or a configured per-agent
// command. Agent inference only happens from argv[0] when `--agent` was omitted, so explicit agent
// selection can wrap commands whose executable name is not recognizable.
fn resolve_agent_and_argv(
    command: &RunCommand,
    agents: &AgentConfigs,
) -> Result<(CodingAgent, Vec<String>), CliError> {
    let argv = resolved_argv(command, agents)?;
    let agent = resolved_agent(command, &argv)?;
    Ok((agent, argv))
}

// Resolves the full argv to spawn. When `--agent` is set (the easy-path and explicit `--agent`
// flows both go through this case), the configured agent command is the base argv and anything
// after `--` is appended as pass-through args. When `--agent` is absent, `command.command` IS
// the full argv (e.g., `nemo-relay run -- codex --model X` runs that exact command and infers
// the agent from argv[0]).
fn resolved_argv(command: &RunCommand, agents: &AgentConfigs) -> Result<Vec<String>, CliError> {
    if let Some(agent) = command.agent {
        let mut argv = configured_command(agent, agents)
            .unwrap_or_else(|| vec![default_command_for(agent).to_string()]);
        argv.extend(command.command.iter().cloned());
        return Ok(argv);
    }
    if command.command.is_empty() {
        return Err(CliError::Launch(
            "missing command; pass -- <agent-command> or --agent with a configured command".into(),
        ));
    }
    Ok(command.command.clone())
}

// Default agent binary names used when no `[agents.<name>] command = "..."` override is in the
// resolved config. Matches the executable on $PATH that the wizard's detection probes for.
const fn default_command_for(agent: CodingAgent) -> &'static str {
    match agent {
        CodingAgent::ClaudeCode => "claude",
        CodingAgent::Codex => "codex",
        CodingAgent::Hermes => "hermes",
    }
}

// Uses an explicit `--agent` when present and otherwise infers the agent from argv[0]. Inference is
// intentionally late so configured commands and direct CLI commands share the same validation path.
fn resolved_agent(command: &RunCommand, argv: &[String]) -> Result<CodingAgent, CliError> {
    if let Some(agent) = command.agent {
        return Ok(agent);
    }
    CodingAgent::infer(&argv[0]).ok_or_else(|| {
        CliError::Launch(format!(
            "could not infer coding agent from command {:?}; pass --agent claude, --agent codex, or --agent hermes",
            argv[0]
        ))
    })
}

// Splits a configured command string into argv words for run mode. This intentionally uses simple
// whitespace splitting because config command values are a convenience fallback; complex shell
// commands should be passed after `--` by the caller.
fn configured_command(agent: CodingAgent, agents: &AgentConfigs) -> Option<Vec<String>> {
    let command = match agent {
        CodingAgent::ClaudeCode => agents.claude.command.as_ref(),
        CodingAgent::Codex => agents.codex.command.as_ref(),
        CodingAgent::Hermes => agents.hermes.command.as_ref(),
    }?;
    let argv: Vec<_> = command.split_whitespace().map(ToOwned::to_owned).collect();
    (!argv.is_empty()).then_some(argv)
}

struct PreparedRun {
    argv: Vec<String>,
    env: Vec<(String, String)>,
    temp_dirs: Vec<PathBuf>,
    hermes_restore: Option<HermesRestore>,
    notes: Vec<String>,
}

struct HermesRestore {
    path: PathBuf,
    backup_path: Option<PathBuf>,
    had_original: bool,
}

struct RunningGateway {
    shutdown_tx: oneshot::Sender<()>,
    task: JoinHandle<Result<(), CliError>>,
}

impl RunningGateway {
    // Starts the gateway listener on a background task and keeps the shutdown sender paired with the
    // task handle so health failures and normal exits use identical cleanup semantics.
    fn start(
        listener: TcpListener,
        config: crate::config::GatewayConfig,
        dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            server::serve_listener_with_dynamic(
                listener,
                config,
                dynamic_plugins,
                Some(shutdown_rx),
            )
            .await
        });
        Self { shutdown_tx, task }
    }

    // Requests shutdown and joins the server task. The send can fail only if the task already exited;
    // the join result still captures whether serving ended cleanly.
    async fn stop(self) -> Result<(), CliError> {
        let _ = self.shutdown_tx.send(());
        self.task
            .await
            .map_err(|error| CliError::Launch(format!("gateway task failed: {error}")))?
    }
}

impl PreparedRun {
    // Builds the launch plan and applies only the preparation needed by the selected agent.
    // Dry-run preparation records equivalent notes and argv/env changes without writing temporary
    // hook files or patching user/project configuration.
    fn new(
        agent: CodingAgent,
        argv: Vec<String>,
        gateway_url: &str,
        resolved: &ResolvedConfig,
        dry_run: bool,
    ) -> Result<Self, CliError> {
        let mut run = Self {
            argv,
            env: vec![("NEMO_RELAY_GATEWAY_URL".into(), gateway_url.into())],
            temp_dirs: Vec::new(),
            hermes_restore: None,
            notes: Vec::new(),
        };
        if let Some(path) = path_with_transparent_hook_dir() {
            run.env.push(("PATH".into(), path));
        }
        match agent {
            CodingAgent::ClaudeCode => {
                if dry_run {
                    run.prepare_claude_dry(gateway_url);
                } else {
                    run.prepare_claude(gateway_url)?;
                }
            }
            CodingAgent::Codex => run.prepare_codex(gateway_url),
            CodingAgent::Hermes => {
                if dry_run {
                    run.prepare_hermes_dry(resolved.agents.hermes.hooks_path.as_deref())?;
                } else {
                    run.prepare_hermes(resolved.agents.hermes.hooks_path.as_deref())?;
                }
            }
        }
        Ok(run)
    }

    // Records the Claude Code argv/env changes that would be made during a real run. The temporary
    // plugin path is symbolic so printed dry-run output is deterministic and non-mutating.
    fn prepare_claude_dry(&mut self, gateway_url: &str) {
        insert_after_agent(
            &mut self.argv,
            CodingAgent::ClaudeCode,
            [
                "--plugin-dir".into(),
                "<temporary-claude-plugin-dir>".into(),
            ],
        );
        self.env
            .push(("ANTHROPIC_BASE_URL".into(), gateway_url.to_string()));
        self.notes
            .push("would generate a temporary Claude Code plugin directory".into());
    }

    // Creates a temporary Claude Code plugin containing gateway hooks and points Claude at both
    // that plugin directory and the gateway Anthropic-compatible gateway URL.
    fn prepare_claude(&mut self, gateway_url: &str) -> Result<(), CliError> {
        let root = temp_dir("nemo-relay-claude-plugin")?;
        std::fs::create_dir_all(root.join(".claude-plugin"))?;
        std::fs::create_dir_all(root.join("hooks"))?;
        std::fs::write(
            root.join(".claude-plugin/plugin.json"),
            serde_json::to_vec_pretty(&json!({
                "name": "nemo-relay-cli",
                "version": env!("CARGO_PKG_VERSION"),
                "description": "Temporary NeMo Relay gateway hooks"
            }))
            .map_err(|error| CliError::Launch(error.to_string()))?,
        )?;
        write_hooks(
            &root.join("hooks/hooks.json"),
            generated_hooks(
                CodingAgent::ClaudeCode,
                &hook_forward_command(&transparent_hook_executable(), CodingAgent::ClaudeCode),
            ),
        )?;
        insert_after_agent(
            &mut self.argv,
            CodingAgent::ClaudeCode,
            ["--plugin-dir".into(), root.display().to_string()],
        );
        self.env
            .push(("ANTHROPIC_BASE_URL".into(), gateway_url.to_string()));
        self.temp_dirs.push(root);
        Ok(())
    }

    // Injects Codex hook and provider configuration through repeated `--config` flags. Codex
    // reserves built-in provider IDs, so run mode installs a temporary provider alias instead of
    // overriding `model_providers.openai`. Uses `features.hooks=true` introduced in codex-cli
    // current supported Codex releases. Requires codex-cli >= 0.143.0.
    fn prepare_codex(&mut self, gateway_url: &str) {
        // Codex resolves auth via `CodexAuth::from_auth_dot_json` (`codex-rs/login/src/auth/
        // manager.rs`): `auth_mode=ApiKey` uses `OPENAI_API_KEY`, `auth_mode=Chatgpt` uses the
        // OAuth token from `~/.codex/auth.json`. With `requires_openai_auth=true` the provider
        // config tells Codex to attach whichever credential it has. The gateway then either
        // substitutes `OPENAI_API_KEY` (routing to `api.openai.com`) or forwards the JWT as-is
        // (routing to `chatgpt.com/backend-api/codex`). Warn when neither source is present.
        let has_openai_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .is_some_and(|v| !v.is_empty());
        // Codex persists OAuth tokens to `~/.codex/auth.json` via `AuthDotJson` in
        // `codex-rs/login/src/auth/storage.rs`. Check for the file rather than parsing it —
        // Codex handles token refresh itself at runtime.
        let has_codex_auth = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(|h| {
                std::path::PathBuf::from(h)
                    .join(".codex/auth.json")
                    .exists()
            })
            .unwrap_or(false);
        if !has_openai_key && !has_codex_auth {
            eprintln!(
                "warning: No OpenAI credentials found. Either export OPENAI_API_KEY \
                 (e.g. `export OPENAI_API_KEY=sk-...`), log in to codex (`codex --login`), \
                 or pass `--openai-base-url` to an upstream that needs no key."
            );
        }
        let hook_command = hook_forward_command(&transparent_hook_executable(), CodingAgent::Codex);
        let mut args = vec![
            "--config".to_string(),
            "features.hooks=true".to_string(),
            "--config".to_string(),
            "model_provider=\"nemo-relay-openai\"".to_string(),
            "--config".to_string(),
            codex_gateway_provider_config(gateway_url),
        ];
        for (event, groups) in generated_hooks(CodingAgent::Codex, &hook_command)["hooks"]
            .as_object()
            .into_iter()
            .flatten()
        {
            args.push("--config".to_string());
            args.push(format!("hooks.{event}={}", hook_groups_toml(groups)));
        }
        insert_after_agent(&mut self.argv, CodingAgent::Codex, args);
    }

    // Hermes discovers hooks from `.hermes/config.yaml` instead of command-line flags. For
    // transparent runs, temporarily merge gateway hook-forward entries into the configured Hermes
    // hook file, then restore it after the child exits.
    fn prepare_hermes(&mut self, hooks_path: Option<&std::path::Path>) -> Result<(), CliError> {
        let path = hermes_hooks_path(hooks_path)?;
        let (had_original, backup_path) = backup_existing_hermes_hooks(&path)?;
        write_merged_hermes_hooks(&path)?;
        self.env.push(("HERMES_ACCEPT_HOOKS".into(), "1".into()));
        self.notes.push(format!(
            "temporarily merged NeMo Relay hooks into {}",
            path.display()
        ));
        self.hermes_restore = Some(HermesRestore {
            path,
            backup_path,
            had_original,
        });
        Ok(())
    }

    // Records the Hermes hook file that would be patched during a real run without touching the
    // filesystem, preserving dry-run as an inspection-only operation.
    fn prepare_hermes_dry(&mut self, hooks_path: Option<&std::path::Path>) -> Result<(), CliError> {
        let path = hermes_hooks_path(hooks_path)?;
        self.env.push(("HERMES_ACCEPT_HOOKS".into(), "1".into()));
        self.notes.push(format!(
            "would temporarily merge NeMo Relay hooks into {}",
            path.display()
        ));
        Ok(())
    }

    // Spawns the prepared child process with injected environment and waits for its exit status.
    // Stdio is inherited by default so agent interaction remains unchanged in transparent mode.
    async fn spawn_and_wait(&self) -> Result<std::process::ExitStatus, CliError> {
        let mut command = Command::new(&self.argv[0]);
        command.args(&self.argv[1..]);
        for (name, value) in &self.env {
            command.env(name, value);
        }
        let mut child = command.spawn()?;
        child.wait().await.map_err(CliError::from)
    }

    // Removes temporary directories and restores patched hook files after the child exits. Restore
    // errors are surfaced after the child status is collected so cleanup problems are not hidden.
    fn restore(&self) -> Result<(), CliError> {
        for dir in &self.temp_dirs {
            let _ = std::fs::remove_dir_all(dir);
        }

        if let Some(hermes) = &self.hermes_restore {
            restore_hook_file(
                &hermes.path,
                hermes.backup_path.as_deref(),
                hermes.had_original,
                "Hermes",
            )?;
        }
        Ok(())
    }

    // Prints a compact pre-launch status banner so users see at a glance which plugin
    // configuration is active, including plugin names and enabled/disabled state, before the
    // agent's own UI takes over the terminal. Always emitted on stderr so it never contaminates
    // piped/redirected agent output, and suppressed entirely when stdout is not a TTY — scripts
    // capturing the agent stream get a clean pipe, interactive users still get the bordered frame.
    // Distinct from `print()`, which is the verbose `--print` / `--dry-run` dump intended for
    // inspection.
    fn print_live_status(&self, agent: CodingAgent, gateway_url: &str, resolved: &ResolvedConfig) {
        // Suppress entirely on non-TTY stdout: when the user redirects the agent's stream to a
        // file or pipes it into another tool, no banner should appear ahead of that output.
        if !std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            return;
        }

        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("NeMo Relay → {}", agent.as_arg()));
        lines.push(format!("  Gateway        {gateway_url}"));
        let destinations = exporter_destinations(&resolved.gateway);
        if destinations.is_empty() {
            lines.push("  Exporters      not configured".into());
        } else {
            for (index, destination) in destinations.iter().enumerate() {
                lines.push(format!(
                    "  {}{}",
                    if index == 0 {
                        "Exporters      "
                    } else {
                        "               "
                    },
                    destination
                ));
            }
        }
        if !self.notes.is_empty() {
            lines.push(String::new());
            for note in &self.notes {
                lines.push(format!("⚠ {note}"));
            }
        }

        // Color decisions key off stderr (where we actually emit), not stdout.
        let use_color = std::io::IsTerminal::is_terminal(&std::io::stderr())
            && std::env::var_os("NO_COLOR").is_none();
        eprint!("{}", render_status_frame(&lines, use_color));
    }

    // Prints the resolved transparent-run plan, including dynamic gateway URL, upstream base URLs,
    // argv/env injection, and any agent-specific notes or temporary files.
    fn print(&self, agent: CodingAgent, gateway_url: &str, resolved: &ResolvedConfig) {
        println!("agent = {}", agent.as_arg());
        println!("gateway_url = {gateway_url}");
        println!("openai_base_url = {}", resolved.gateway.openai_base_url);
        println!(
            "anthropic_base_url = {}",
            resolved.gateway.anthropic_base_url
        );
        println!(
            "max_hook_payload_bytes = {}",
            resolved.gateway.max_hook_payload_bytes
        );
        println!(
            "max_passthrough_body_bytes = {}",
            resolved.gateway.max_passthrough_body_bytes
        );
        let destinations = exporter_destinations(&resolved.gateway);
        if destinations.is_empty() {
            println!("exporters = not_configured");
        } else {
            for destination in destinations {
                println!("exporter = {destination}");
            }
        }
        println!("argv = {}", self.argv.join(" "));
        for (name, value) in &self.env {
            println!("env.{name} = {value}");
        }
        for note in &self.notes {
            println!("note = {note}");
        }
    }
}

/// Renders a bordered status frame for daemon and transparent-run startup output.
pub(crate) fn render_status_frame(lines: &[String], color: bool) -> String {
    let max_w = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    // 1-char padding on each side of the longest line.
    let inner = max_w + 2;
    let mut output = String::new();

    output.push('\n');
    push_status_border(&mut output, '╭', '╮', inner, color);
    for line in lines {
        let pad = max_w - line.chars().count();
        let body = format!(" {line}{spaces} ", spaces = " ".repeat(pad));
        if color {
            output.push_str(&format!(
                "\x1b[38;5;112m│\x1b[0m{body}\x1b[38;5;112m│\x1b[0m\n"
            ));
        } else {
            output.push_str(&format!("│{body}│\n"));
        }
    }
    push_status_border(&mut output, '╰', '╯', inner, color);
    output.push('\n');
    output
}

pub(crate) fn exporter_destinations(config: &GatewayConfig) -> Vec<String> {
    let Some(plugin_config) = config.plugin_config.as_ref() else {
        return Vec::new();
    };
    let Ok(plugin_config) = serde_json::from_value::<PluginConfig>(plugin_config.clone()) else {
        return vec!["configured (invalid plugin config)".into()];
    };
    let Some(component) = plugin_config
        .components
        .iter()
        .find(|component| component.kind == OBSERVABILITY_PLUGIN_KIND)
    else {
        return Vec::new();
    };
    if !component.enabled {
        return Vec::new();
    }
    let Ok(observability) =
        serde_json::from_value::<ObservabilityConfig>(Value::Object(component.config.clone()))
    else {
        return vec!["Observability configured (invalid config)".into()];
    };
    observability_exporter_destinations(&observability)
}

fn observability_exporter_destinations(config: &ObservabilityConfig) -> Vec<String> {
    let mut destinations = Vec::new();
    if let Some(section) = config.atof.as_ref().filter(|section| section.enabled) {
        let directory = section
            .output_directory
            .clone()
            .unwrap_or_else(current_output_directory);
        let path = directory.join(
            section
                .filename
                .clone()
                .unwrap_or_else(|| "nemo-relay-events-<timestamp>.jsonl".into()),
        );
        destinations.push(format!("ATOF {}", path.display()));
    }
    if let Some(section) = config.atif.as_ref().filter(|section| section.enabled) {
        if section.storage.is_empty() {
            let directory = section
                .output_directory
                .clone()
                .unwrap_or_else(current_output_directory);
            destinations.push(format!(
                "ATIF {}",
                directory.join(&section.filename_template).display()
            ));
        } else {
            // Non-empty `storage` skips the local file write and uploads to each remote backend
            // instead, so report the actual upload destinations rather than a local path that is
            // never written.
            for backend in &section.storage {
                destinations.push(format!("ATIF {}", atif_storage_destination(backend)));
            }
        }
    }
    if let Some(section) = config
        .opentelemetry
        .as_ref()
        .filter(|section| section.enabled)
    {
        destinations.push(format!(
            "OpenTelemetry {}",
            section
                .endpoint
                .as_deref()
                .unwrap_or("OTLP endpoint from environment/default")
        ));
    }
    if let Some(section) = config
        .openinference
        .as_ref()
        .filter(|section| section.enabled)
    {
        destinations.push(format!(
            "OpenInference {}",
            section
                .endpoint
                .as_deref()
                .unwrap_or("OTLP endpoint from environment/default")
        ));
    }
    destinations
}

// Renders a single ATIF remote storage backend as a human-readable destination for the status
// banner. S3 keys are summarized as `s3://<bucket>/<key_prefix>`; the per-trajectory object suffix
// is omitted because it is only known once a session starts.
fn atif_storage_destination(storage: &AtifStorageConfig) -> String {
    match storage {
        AtifStorageConfig::Http(http) => http.endpoint.clone(),
        AtifStorageConfig::S3(s3) => {
            let prefix = s3.key_prefix.as_deref().unwrap_or("").trim_matches('/');
            if prefix.is_empty() {
                format!("s3://{}", s3.bucket)
            } else {
                format!("s3://{}/{}", s3.bucket, prefix)
            }
        }
    }
}

fn current_output_directory() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

// Converts a process status into the launcher status code while preserving normal 0-255 exits. Signal
// exits and platform-specific out-of-range codes become generic failure.
fn exit_code(status: std::process::ExitStatus) -> ExitCode {
    status
        .code()
        .and_then(|code| u8::try_from(code).ok())
        .map(ExitCode::from)
        .unwrap_or(ExitCode::FAILURE)
}

// Polls the ephemeral gateway health endpoint for roughly one second before launching the agent.
// Startup failures return a launcher error so the child command is never run against a dead proxy.
async fn wait_for_health(gateway_url: &str) -> Result<(), CliError> {
    let client = Client::new();
    let url = format!("{}/healthz", gateway_url.trim_end_matches('/'));
    for _ in 0..50 {
        if let Ok(response) = client.get(&url).send().await
            && response.status().is_success()
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    Err(CliError::Launch(format!(
        "gateway did not become ready at {url}"
    )))
}

fn codex_gateway_provider_config(gateway_url: &str) -> String {
    // `wire_api="responses"` is the only value codex 0.130+ accepts; the `chat` value was
    // removed (codex#7782). Codex transparent run therefore only works against upstreams that
    // implement `/v1/responses` (api.openai.com or a Responses-compatible proxy). For other
    // upstreams the user falls back to daemon mode and points codex directly at its configured
    // upstream — we observe hooks but not LLM calls.
    //
    // `requires_openai_auth=true` so Codex's `resolve_provider_auth` (`codex-rs/model-provider/
    // src/auth.rs`) attaches credentials via `BearerAuthProvider`. When the auth mode is
    // `Chatgpt` the token is an OAuth JWT or Codex access token; when `ApiKey` it is the
    // `OPENAI_API_KEY` value.
    // The gateway inspects the inbound `Authorization` header: if `OPENAI_API_KEY` is set in the
    // environment the ChatGPT token is replaced (see `alignment::gateway_forward_headers` and
    // `gateway.rs::inject_provider_auth`); otherwise it is forwarded to the ChatGPT backend.
    format!(
        "model_providers.nemo-relay-openai={{name=\"NeMo Relay OpenAI\",base_url={},wire_api=\"responses\",requires_openai_auth=true,supports_websockets=false}}",
        toml_string(gateway_url)
    )
}

// Appends one horizontal border line in NVIDIA green when color is enabled, otherwise plain
// ASCII-compatible box-drawing.
fn push_status_border(
    output: &mut String,
    left: char,
    right: char,
    inner_width: usize,
    color: bool,
) {
    let dashes = "─".repeat(inner_width);
    if color {
        output.push_str(&format!("\x1b[38;5;112m{left}{dashes}{right}\x1b[0m\n"));
    } else {
        output.push_str(&format!("{left}{dashes}{right}\n"));
    }
}

// Returns the absolute path of the running gateway binary so injected hooks can find it
// without relying on the user's `PATH`. Spawned hook subprocesses inherit the agent's
// environment; in transparent run, the dev/install location of the gateway is rarely on
// `PATH`, which would cause hooks to exit with status 127 (command not found). Falls back
// to the bare name when `current_exe` is unavailable so behavior degrades to the previous
// install-style assumption rather than failing to launch.
fn transparent_hook_executable() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.to_str().map(|s| {
                #[cfg(windows)]
                {
                    s.replace('\\', "/")
                }
                #[cfg(not(windows))]
                {
                    s.to_owned()
                }
            })
        })
        .unwrap_or_else(|| "nemo-relay".to_string())
}

// Appends the running gateway binary's directory to the child agent PATH. Transparent hooks use
// the absolute executable path when possible, but adding the directory also covers hook loaders or
// user-managed hook commands that resolve `nemo-relay` through PATH inside the launched agent. Keep
// user PATH precedence intact so normal agent tool resolution does not change.
fn path_with_transparent_hook_dir() -> Option<String> {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))?;
    let mut paths: Vec<PathBuf> = std::env::var_os("PATH")
        .as_deref()
        .map(std::env::split_paths)
        .into_iter()
        .flatten()
        .collect();
    if !paths.iter().any(|path| path == &dir) {
        paths.push(dir);
    }
    std::env::join_paths(paths)
        .ok()
        .map(|path| path.to_string_lossy().into_owned())
}

// Inserts generated agent flags immediately after the last argv element that looks like the agent
// executable. Falling back to index 0 keeps wrapper commands usable by inserting after the first
// word when the agent cannot be found later in argv.
fn insert_after_agent(
    argv: &mut Vec<String>,
    agent: CodingAgent,
    args: impl IntoIterator<Item = String>,
) {
    let index = argv
        .iter()
        .enumerate()
        .filter_map(|(index, arg)| (CodingAgent::infer(arg) == Some(agent)).then_some(index))
        .next_back()
        .unwrap_or(0);
    argv.splice(index + 1..index + 1, args);
}

// Writes pretty JSON hook config to a path whose parent has already been created by the caller.
// Serialization errors are converted to launch errors to keep temporary setup failures contextual.
fn write_hooks(path: &Path, hooks: Value) -> Result<(), CliError> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&hooks).map_err(|error| CliError::Launch(error.to_string()))?,
    )?;
    Ok(())
}

// Backs up an existing Hermes hook config before run-mode patching.
fn backup_existing_hermes_hooks(path: &Path) -> Result<(bool, Option<PathBuf>), CliError> {
    let had_original = path.exists();
    if !had_original {
        return Ok((false, None));
    }
    let backup = path.with_extension(format!("yaml.nemo-relay-run.bak.{}", timestamp()?));
    std::fs::copy(path, &backup)?;
    Ok((true, Some(backup)))
}

// Creates the Hermes config parent directory when needed, merges generated gateway hooks with any
// existing YAML config, and writes the patched YAML used for this transparent run.
fn write_merged_hermes_hooks(path: &Path) -> Result<(), CliError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let existing = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(CliError::Io(error)),
    };
    let relay = std::env::current_exe()
        .map(|path| path.canonicalize().unwrap_or(path))
        .map(crate::plugin_shim::portable_executable_path)
        .unwrap_or_else(|_| PathBuf::from("nemo-relay"));
    let contents = crate::hermes::transparent_config(&existing, &relay)?;
    std::fs::write(path, contents)?;
    Ok(())
}

// Chooses the Hermes hook file that transparent run should patch. If setup recorded a specific
// path, reuse it; otherwise fall back to the Hermes home config file that Hermes itself reads.
fn hermes_hooks_path(configured: Option<&Path>) -> Result<PathBuf, CliError> {
    if let Some(path) = configured {
        return Ok(path.to_path_buf());
    }
    if let Some(home) = std::env::var_os("HERMES_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(home).join("config.yaml"));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .ok_or_else(|| {
            CliError::Launch("could not resolve home directory for Hermes hooks".into())
        })?;
    Ok(PathBuf::from(home).join(".hermes").join("config.yaml"))
}

fn restore_hook_file(
    path: &Path,
    backup_path: Option<&Path>,
    had_original: bool,
    label: &str,
) -> Result<(), CliError> {
    match (backup_path, had_original) {
        (Some(backup), true) => {
            std::fs::copy(backup, path).map_err(|error| {
                CliError::Launch(format!(
                    "failed to restore {label} hooks from {}: {error}",
                    backup.display()
                ))
            })?;
            let _ = std::fs::remove_file(backup);
        }
        (_, false) if path.exists() => {
            std::fs::remove_file(path).map_err(|error| {
                CliError::Launch(format!(
                    "failed to remove temporary {label} hooks {}: {error}",
                    path.display()
                ))
            })?;
        }
        _ => {}
    }
    Ok(())
}

// Converts JSON hook groups into inline TOML arrays for Codex `--config` flags. The function
// preserves matchers when present and assumes generated hook groups contain one command hook.
fn hook_groups_toml(value: &Value) -> String {
    let mut groups = Vec::new();
    for group in value.as_array().into_iter().flatten() {
        let matcher = group
            .get("matcher")
            .and_then(Value::as_str)
            .map(|matcher| format!("matcher={},", toml_string(matcher)))
            .unwrap_or_default();
        let command = group["hooks"][0]["command"].as_str().unwrap_or_default();
        groups.push(format!(
            "{{{matcher}hooks=[{{type=\"command\",command={},timeout=30}}]}}",
            toml_string(command)
        ));
    }
    format!("[{}]", groups.join(","))
}

// Escapes a Rust string as a TOML basic string for inline Codex configuration values.
fn toml_string(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

// Creates a timestamped directory under the OS temp directory. The timestamp suffix avoids
// collisions between concurrent transparent runs without keeping persistent state.
fn temp_dir(prefix: &str) -> Result<PathBuf, CliError> {
    let path = std::env::temp_dir().join(format!("{prefix}-{}", timestamp()?));
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

// Returns a monotonic-enough wall-clock nanosecond stamp for temp and backup names. System time
// errors become launcher errors because paths cannot be safely generated without a timestamp.
fn timestamp() -> Result<u128, CliError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| CliError::Launch(error.to_string()))?
        .as_nanos())
}

#[cfg(test)]
#[path = "../tests/coverage/launcher_tests.rs"]
mod tests;
