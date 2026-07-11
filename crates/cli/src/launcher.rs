// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use nemo_relay::observability::plugin_component::{
    AtifStorageConfig, OBSERVABILITY_PLUGIN_KIND, ObservabilityConfig,
};
use nemo_relay::plugin::PluginConfig;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::config::{
    AgentConfigs, CodingAgent, EasyPathCommand, GatewayConfig, RELAY_PLUGIN_ID,
    RELAY_SOURCE_PLUGIN_ID, ResolvedConfig, RunCommand, ServerArgs, any_config_file_exists,
    resolve_run_config,
};
use crate::error::CliError;
use crate::installer::{generated_hooks, transparent_hook_forward_command};
use crate::plugins::lifecycle::ActiveDynamicPluginComponent;
use crate::server;

/// Runs a child coding-agent command behind an ephemeral local gateway.
///
/// The gateway binds to an OS-assigned loopback port, prepares agent-specific hook/gateway wiring,
/// waits for health before spawning the child, and removes temporary state after the child and
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
        let invocation = resolve_agent_invocation(&command, &resolved.agents)?;
        let agent = invocation.agent;
        if !dry_run {
            let probe = crate::agent_process::version_probe_argv(
                agent,
                &invocation.argv[..=invocation.host_index],
            );
            validate_agent_version(agent, &probe).await?;
        }
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let gateway_url = format!("http://{address}");
        resolved.gateway.bind = address;

        let prepared = PreparedRun::from_invocation(invocation, &gateway_url, &resolved, dry_run)?;
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
    let bootstrap_fingerprint = crate::config::transparent_gateway_fingerprint(gateway_url);
    let running_server = RunningGateway::start(
        listener,
        gateway_config,
        dynamic_plugins,
        bootstrap_fingerprint.clone(),
    );
    if let Err(error) = wait_for_health(gateway_url, &bootstrap_fingerprint).await {
        let restore = prepared.restore();
        let server_result = running_server.stop().await;
        restore?;
        server_result?;
        return Err(error);
    }
    supervise_prepared_run(&prepared, running_server).await
}

async fn supervise_prepared_run(
    prepared: &PreparedRun,
    mut running_server: RunningGateway,
) -> Result<ExitCode, CliError> {
    let mut child = match prepared.spawn().await {
        Ok(child) => child,
        Err(error) => {
            let restore = prepared.restore();
            let server_result = running_server.stop().await;
            restore?;
            server_result?;
            return Err(error);
        }
    };

    tokio::select! {
        status = child.wait() => {
            let restore = prepared.restore();
            let server_result = running_server.stop().await;
            restore?;
            server_result?;
            Ok(exit_code(status?))
        }
        gateway_result = running_server.wait() => {
            let child_result = child.terminate().await;
            let restore = prepared.restore();
            restore?;
            child_result?;
            match gateway_result {
                Err(error) => Err(error),
                Ok(()) => Err(CliError::Launch(
                    "transparent Relay gateway stopped before the coding agent exited".into(),
                )),
            }
        }
    }
}

// Resolves the launched agent and argv from either an explicit command or a configured per-agent
// command. Agent inference only happens from argv[0] when `--agent` was omitted, so explicit agent
// selection can wrap commands whose executable name is not recognizable.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AgentInvocation {
    agent: CodingAgent,
    argv: Vec<String>,
    host_index: usize,
}

fn resolve_agent_invocation(
    command: &RunCommand,
    agents: &AgentConfigs,
) -> Result<AgentInvocation, CliError> {
    if let Some(agent) = command.agent {
        let mut argv = configured_command(agent, agents)
            .unwrap_or_else(|| vec![default_command_for(agent).to_string()]);
        let host_index = argv
            .iter()
            .rposition(|argument| CodingAgent::infer(argument) == Some(agent))
            .unwrap_or(0);
        argv.extend(command.command.iter().cloned());
        return Ok(AgentInvocation {
            agent,
            argv,
            host_index,
        });
    }
    if command.command.is_empty() {
        return Err(CliError::Launch(
            "missing command; pass -- <agent-command> or --agent with a configured command".into(),
        ));
    }
    let argv = command.command.clone();
    let agent = CodingAgent::infer(&argv[0]).ok_or_else(|| {
        CliError::Launch(format!(
            "could not infer coding agent from command {:?}; pass --agent claude, --agent codex, or --agent hermes",
            argv[0]
        ))
    })?;
    Ok(AgentInvocation {
        agent,
        argv,
        host_index: 0,
    })
}

#[cfg(test)]
fn resolve_agent_and_argv(
    command: &RunCommand,
    agents: &AgentConfigs,
) -> Result<(CodingAgent, Vec<String>), CliError> {
    resolve_agent_invocation(command, agents).map(|invocation| (invocation.agent, invocation.argv))
}

// Default agent binary names used when no `[agents.<name>] command = "..."` override is in the
// resolved config. Matches the executable on $PATH that the wizard's detection probes for.
const fn default_command_for(agent: CodingAgent) -> &'static str {
    agent.executable()
}

/// Builds a version probe that preserves wrappers such as `npx codex` or `mise exec -- codex`.
/// Opaque wrappers remain supported when their `--version` output identifies the selected host.
async fn validate_agent_version(agent: CodingAgent, probe: &[String]) -> Result<(), CliError> {
    let mut command = crate::agent_process::tokio_command(probe);
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(5), command.output())
        .await
        .map_err(|_| {
            CliError::Launch(format!(
                "timed out while running version probe {:?} for {}; NeMo Relay requires {}",
                probe,
                agent.label(),
                agent.version_requirement()
            ))
        })??;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::Launch(format!(
            "version probe {:?} failed with {}{}",
            probe,
            output.status,
            if stderr.trim().is_empty() {
                String::new()
            } else {
                format!(": {}", stderr.trim())
            }
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    agent
        .validate_version_output(&stdout)
        .map(|_| ())
        .map_err(CliError::Launch)
}

// Splits a configured command string into argv words for run mode. This intentionally uses simple
// whitespace splitting because config command values are a convenience fallback; complex shell
// commands should be passed after `--` by the caller.
fn configured_command(agent: CodingAgent, agents: &AgentConfigs) -> Option<Vec<String>> {
    let command = agents.get(agent).command.as_ref()?;
    let argv = crate::agent_process::command_argv(command);
    (!argv.is_empty()).then_some(argv)
}

struct PreparedRun {
    argv: Vec<String>,
    host_index: usize,
    env: Vec<(String, String)>,
    temp_dirs: Vec<PathBuf>,
    notes: Vec<String>,
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
        bootstrap_fingerprint: String,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            server::serve_transparent_listener_with_dynamic(
                listener,
                config,
                dynamic_plugins,
                bootstrap_fingerprint,
                Some(shutdown_rx),
            )
            .await
        });
        Self { shutdown_tx, task }
    }

    async fn wait(&mut self) -> Result<(), CliError> {
        (&mut self.task)
            .await
            .map_err(|error| CliError::Launch(format!("gateway task failed: {error}")))?
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
    fn from_invocation(
        invocation: AgentInvocation,
        gateway_url: &str,
        resolved: &ResolvedConfig,
        dry_run: bool,
    ) -> Result<Self, CliError> {
        Self::build(
            invocation.agent,
            invocation.argv,
            invocation.host_index,
            gateway_url,
            resolved,
            dry_run,
        )
    }

    #[cfg(test)]
    fn new(
        agent: CodingAgent,
        argv: Vec<String>,
        gateway_url: &str,
        resolved: &ResolvedConfig,
        dry_run: bool,
    ) -> Result<Self, CliError> {
        let boundary = argv
            .iter()
            .position(|argument| argument == "--")
            .unwrap_or(argv.len());
        let host_index = argv[..boundary]
            .iter()
            .rposition(|argument| CodingAgent::infer(argument) == Some(agent))
            .unwrap_or(0);
        Self::build(agent, argv, host_index, gateway_url, resolved, dry_run)
    }

    // Builds the launch plan and applies only the preparation needed by the selected agent.
    // Dry-run preparation records equivalent notes and argv/env changes without writing temporary
    // hook files or patching user/project configuration.
    fn build(
        agent: CodingAgent,
        argv: Vec<String>,
        host_index: usize,
        gateway_url: &str,
        resolved: &ResolvedConfig,
        dry_run: bool,
    ) -> Result<Self, CliError> {
        let mut run = Self {
            argv,
            host_index,
            env: vec![
                (crate::config::GATEWAY_URL_ENV.into(), gateway_url.into()),
                (crate::config::TRANSPARENT_RUN_ENV.into(), "1".into()),
            ],
            temp_dirs: Vec::new(),
            notes: Vec::new(),
        };
        if let Some(path) = path_with_transparent_hook_dir() {
            run.env.push(("PATH".into(), path));
        }
        match agent {
            CodingAgent::ClaudeCode => {
                if dry_run {
                    run.prepare_claude_dry(gateway_url)?;
                } else {
                    run.prepare_claude(gateway_url)?;
                }
            }
            CodingAgent::Codex => run.prepare_codex(gateway_url)?,
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
    fn prepare_claude_dry(&mut self, gateway_url: &str) -> Result<(), CliError> {
        insert_after_host(
            &mut self.argv,
            self.host_index,
            [
                "--plugin-dir".into(),
                "<temporary-claude-plugin-dir>".into(),
                "--settings".into(),
                "<temporary-claude-settings>".into(),
            ],
        );
        self.env
            .push(("ANTHROPIC_BASE_URL".into(), gateway_url.to_string()));
        self.notes
            .push("would generate a temporary Claude Code plugin directory".into());
        Ok(())
    }

    // Creates a temporary Claude Code plugin containing gateway hooks and a process-private
    // settings overlay. Claude applies the first `--settings` argument, so the overlay preserves
    // the caller's first explicit settings source while overriding only the gateway URL.
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
        let hook_command = transparent_hook_forward_command(
            &transparent_hook_executable(),
            CodingAgent::ClaudeCode,
            gateway_url,
        )
        .map_err(CliError::Launch)?;
        write_hooks(
            &root.join("hooks/hooks.json"),
            generated_hooks(CodingAgent::ClaudeCode, &hook_command),
        )?;
        let settings_path = root.join("settings.json");
        let settings = claude_settings_overlay(&self.argv, self.host_index, gateway_url)?;
        let settings_bytes = serde_json::to_vec_pretty(&settings)
            .map_err(|error| CliError::Launch(error.to_string()))?;
        crate::file_io::atomic_write_private(&settings_path, &settings_bytes)
            .map_err(CliError::Launch)?;
        insert_after_host(
            &mut self.argv,
            self.host_index,
            [
                "--plugin-dir".into(),
                root.display().to_string(),
                "--settings".into(),
                settings_path.display().to_string(),
            ],
        );
        self.env
            .push(("ANTHROPIC_BASE_URL".into(), gateway_url.to_string()));
        self.temp_dirs.push(root);
        Ok(())
    }

    // Injects Codex hook and provider configuration through repeated `--config` flags. Codex
    // reserves built-in provider IDs, so run mode installs a temporary provider alias instead of
    // overriding `model_providers.openai`. Uses `features.hooks=true` introduced in codex-cli
    // current supported Codex releases. The centralized host policy validates the version first.
    fn prepare_codex(&mut self, gateway_url: &str) -> Result<(), CliError> {
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
        let hook_command = transparent_hook_forward_command(
            &transparent_hook_executable(),
            CodingAgent::Codex,
            gateway_url,
        )
        .map_err(CliError::Launch)?;
        let hook_groups = generated_hooks(CodingAgent::Codex, &hook_command);
        let mut args = vec![
            "--config".to_string(),
            "features.hooks=true".to_string(),
            "--config".to_string(),
            "model_provider=\"nemo-relay-openai\"".to_string(),
            "--config".to_string(),
            codex_gateway_provider_config(gateway_url),
        ];
        for (event, groups) in hook_groups["hooks"].as_object().into_iter().flatten() {
            args.push("--config".to_string());
            args.push(format!("hooks.{event}={}", hook_groups_toml(groups)));
        }
        args.push("--config".to_string());
        args.push(codex_session_hook_state_override(&hook_groups)?);
        insert_after_host(&mut self.argv, self.host_index, args);
        Ok(())
    }

    // Hermes discovers hooks from `.hermes/config.yaml` instead of command-line flags. A
    // process-private HERMES_HOME exposes dynamic hooks without rewriting user configuration.
    fn prepare_hermes(&mut self, hooks_path: Option<&std::path::Path>) -> Result<(), CliError> {
        let source_config = hermes_hooks_path(hooks_path)?;
        let source_home = source_config.parent().ok_or_else(|| {
            CliError::Launch(format!(
                "Hermes config path {} has no parent directory",
                source_config.display()
            ))
        })?;
        let gateway_url = self
            .env
            .iter()
            .find_map(|(name, value)| {
                (name == crate::config::GATEWAY_URL_ENV).then_some(value.as_str())
            })
            .expect("transparent runs always define their gateway URL");
        let overlay_home = create_hermes_overlay(source_home, &source_config, gateway_url)?;
        self.env.push(("HERMES_ACCEPT_HOOKS".into(), "1".into()));
        self.env
            .push(("HERMES_HOME".into(), overlay_home.display().to_string()));
        self.notes.push(format!(
            "using an isolated Hermes config overlay for {}",
            source_config.display()
        ));
        self.temp_dirs.push(overlay_home);
        Ok(())
    }

    // Records the Hermes hook file that would be patched during a real run without touching the
    // filesystem, preserving dry-run as an inspection-only operation.
    fn prepare_hermes_dry(&mut self, hooks_path: Option<&std::path::Path>) -> Result<(), CliError> {
        let path = hermes_hooks_path(hooks_path)?;
        self.env.push(("HERMES_ACCEPT_HOOKS".into(), "1".into()));
        self.notes.push(format!(
            "would create an isolated Hermes config overlay for {}",
            path.display()
        ));
        Ok(())
    }

    // Spawns the prepared child process with injected environment.
    // Stdio is inherited by default so agent interaction remains unchanged in transparent mode.
    async fn spawn(&self) -> Result<crate::agent_process::SupervisedChild, CliError> {
        let mut command = crate::agent_process::tokio_command(&self.argv);
        for (name, value) in &self.env {
            command.env(name, value);
        }
        crate::agent_process::SupervisedChild::spawn(&mut command)
            .await
            .map_err(CliError::from)
    }

    // Removes process-private plugin and configuration directories after the child exits.
    fn restore(&self) -> Result<(), CliError> {
        for dir in &self.temp_dirs {
            match std::fs::remove_dir_all(dir) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(CliError::Io(error)),
            }
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

// Claude Code honors only the first `--settings` source. Preserve that source in the generated
// overlay so inserting Relay's process-private gateway setting cannot discard user configuration.
fn claude_settings_overlay(
    argv: &[String],
    host_index: usize,
    gateway_url: &str,
) -> Result<Value, CliError> {
    let mut settings = match first_claude_settings(argv, host_index)? {
        Some(source) => read_claude_settings(source)?,
        None => json!({}),
    };
    let object = settings.as_object_mut().ok_or_else(|| {
        CliError::Launch("Claude Code --settings must contain a JSON object".into())
    })?;
    let environment = object.entry("env").or_insert_with(|| json!({}));
    let environment = environment.as_object_mut().ok_or_else(|| {
        CliError::Launch("Claude Code --settings field `env` must be a JSON object".into())
    })?;
    environment.insert(
        "ANTHROPIC_BASE_URL".into(),
        Value::String(gateway_url.into()),
    );
    Ok(settings)
}

fn first_claude_settings(argv: &[String], host_index: usize) -> Result<Option<&str>, CliError> {
    let boundary = argv
        .iter()
        .skip(host_index + 1)
        .position(|argument| argument == "--")
        .map_or(argv.len(), |offset| host_index + 1 + offset);
    let mut index = host_index + 1;
    while index < boundary {
        if argv[index] == "--settings" {
            if index + 1 >= boundary || argv[index + 1].is_empty() {
                return Err(CliError::Launch(
                    "Claude Code --settings is missing its value".into(),
                ));
            }
            return Ok(Some(argv[index + 1].as_str()));
        }
        if let Some(source) = argv[index].strip_prefix("--settings=") {
            if source.is_empty() {
                return Err(CliError::Launch(
                    "Claude Code --settings is missing its value".into(),
                ));
            }
            return Ok(Some(source));
        }
        index += 1;
    }
    Ok(None)
}

fn read_claude_settings(source: &str) -> Result<Value, CliError> {
    let raw = if source.trim_start().starts_with('{') {
        source.to_string()
    } else {
        std::fs::read_to_string(source).map_err(|error| {
            CliError::Launch(format!(
                "failed to read Claude Code settings {}: {error}",
                Path::new(source).display()
            ))
        })?
    };
    serde_json::from_str(&raw).map_err(|error| {
        CliError::Launch(format!(
            "failed to parse Claude Code --settings JSON: {error}"
        ))
    })
}

// Session hook definitions and their exact trust state share Codex's process-local CLI layer. This
// authorizes only the generated Relay command without rewriting the active user profile or using
// the process-wide hook-trust bypass.
fn codex_session_hook_state_override(generated: &Value) -> Result<String, CliError> {
    let events = generated
        .get("hooks")
        .and_then(Value::as_object)
        .ok_or_else(|| CliError::Launch("generated Codex hooks were malformed".into()))?;
    let mut states = Vec::new();
    for (event, groups) in events {
        let groups = groups.as_array().ok_or_else(|| {
            CliError::Launch(format!(
                "generated Codex {event} hook groups were malformed"
            ))
        })?;
        let event_key = codex_hook_event_key(event);
        for (group_index, group) in groups.iter().enumerate() {
            let group = group.as_object().ok_or_else(|| {
                CliError::Launch(format!("generated Codex {event} hook group was malformed"))
            })?;
            let handlers = group
                .get("hooks")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    CliError::Launch(format!(
                        "generated Codex {event} hook handlers were malformed"
                    ))
                })?;
            for (handler_index, handler) in handlers.iter().enumerate() {
                let hash = codex_command_hook_hash(&event_key, group, handler)?;
                let key = format!(
                    "/<session-flags>/config.toml:{event_key}:{group_index}:{handler_index}"
                );
                states.push(format!(
                    "{}={{trusted_hash={},enabled=true}}",
                    toml_string(&key),
                    toml_string(&hash)
                ));
                for plugin_id in [RELAY_PLUGIN_ID, RELAY_SOURCE_PLUGIN_ID] {
                    let key = format!(
                        "{plugin_id}:hooks/hooks.json:{event_key}:{group_index}:{handler_index}"
                    );
                    states.push(format!("{}={{enabled=false}}", toml_string(&key)));
                }
            }
        }
    }
    Ok(format!("hooks.state={{{}}}", states.join(",")))
}

fn codex_hook_event_key(event: &str) -> String {
    let mut normalized = String::with_capacity(event.len() + 2);
    for (index, character) in event.chars().enumerate() {
        if character.is_ascii_uppercase() {
            if index > 0 {
                normalized.push('_');
            }
            normalized.push(character.to_ascii_lowercase());
        } else {
            normalized.push(character);
        }
    }
    normalized
}

fn codex_command_hook_hash(
    event_key: &str,
    group: &serde_json::Map<String, Value>,
    handler: &Value,
) -> Result<String, CliError> {
    use sha2::{Digest, Sha256};

    let handler = handler.as_object().ok_or_else(|| {
        CliError::Launch(format!(
            "generated Codex {event_key} command hook was malformed"
        ))
    })?;
    if handler.get("type").and_then(Value::as_str) != Some("command") {
        return Err(CliError::Launch(format!(
            "generated Codex {event_key} hook was not a command"
        )));
    }
    let command = handler
        .get(if cfg!(windows) {
            "commandWindows"
        } else {
            "command"
        })
        .or_else(|| handler.get("command"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            CliError::Launch(format!(
                "generated Codex {event_key} hook command was missing"
            ))
        })?;
    let timeout = handler
        .get("timeout")
        .and_then(Value::as_u64)
        .unwrap_or(600)
        .max(1);
    let mut normalized_handler = serde_json::Map::new();
    normalized_handler.insert("type".into(), Value::String("command".into()));
    normalized_handler.insert("command".into(), Value::String(command.into()));
    normalized_handler.insert("timeout".into(), Value::Number(timeout.into()));
    normalized_handler.insert("async".into(), Value::Bool(false));
    if let Some(status) = handler.get("statusMessage").and_then(Value::as_str) {
        normalized_handler.insert("statusMessage".into(), Value::String(status.into()));
    }
    let mut identity = serde_json::Map::new();
    identity.insert("event_name".into(), Value::String(event_key.into()));
    if let Some(matcher) = group.get("matcher").and_then(Value::as_str) {
        identity.insert("matcher".into(), Value::String(matcher.into()));
    }
    identity.insert(
        "hooks".into(),
        Value::Array(vec![Value::Object(normalized_handler)]),
    );
    let canonical = canonical_json(Value::Object(identity));
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| CliError::Launch(format!("failed to hash Codex hook: {error}")))?;
    let digest = Sha256::digest(bytes);
    Ok(format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

fn canonical_json(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries = object.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, canonical_json(value)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonical_json).collect()),
        other => other,
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
async fn wait_for_health(gateway_url: &str, bootstrap_fingerprint: &str) -> Result<(), CliError> {
    for _ in 0..50 {
        let gateway_url = gateway_url.to_string();
        let bootstrap_fingerprint = bootstrap_fingerprint.to_string();
        if tokio::task::spawn_blocking(move || {
            crate::sidecar::healthz_compatible(&gateway_url, &bootstrap_fingerprint)
        })
        .await
        .map_err(|error| CliError::Launch(format!("gateway readiness task failed: {error}")))?
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    Err(CliError::Launch(format!(
        "gateway did not become ready at {}/healthz",
        gateway_url.trim_end_matches('/')
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
fn transparent_hook_executable() -> PathBuf {
    std::env::current_exe()
        .map(|path| path.canonicalize().unwrap_or(path))
        .map(crate::plugin_host::portable_executable_path)
        .unwrap_or_else(|_| PathBuf::from("nemo-relay"))
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

// The invocation resolver determines this index before pass-through arguments are appended. Using
// it here prevents a prompt token named `codex` or `claude` from becoming an accidental insertion
// target while preserving configured wrapper prefixes.
fn insert_after_host(
    argv: &mut Vec<String>,
    host_index: usize,
    args: impl IntoIterator<Item = String>,
) {
    debug_assert!(host_index < argv.len());
    argv.splice(host_index + 1..host_index + 1, args);
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

// Creates a per-process Hermes home whose user state points at the original profile while the
// config and hook approval files remain private to this transparent run. Hermes has no standalone
// config-file override, so `HERMES_HOME` is its supported process-scoped configuration boundary.
fn create_hermes_overlay(
    source_home: &Path,
    source_config: &Path,
    gateway_url: &str,
) -> Result<PathBuf, CliError> {
    // Prefer a sibling of HERMES_HOME so Windows file hard links remain on one volume. Fall back
    // to the OS temp directory when the profile parent is not writable; regular files then use a
    // copy fallback, while profile directories remain live through junctions.
    let overlay = source_home
        .parent()
        .filter(|parent| parent.is_dir())
        .and_then(|parent| private_temp_dir(parent, ".nemo-relay-hermes-home").ok())
        .map(Ok)
        .unwrap_or_else(|| temp_dir("nemo-relay-hermes-home"))?;
    if let Err(error) = populate_hermes_overlay(&overlay, source_home, source_config, gateway_url) {
        let _ = std::fs::remove_dir_all(&overlay);
        return Err(error);
    }
    Ok(overlay)
}

fn populate_hermes_overlay(
    overlay: &Path,
    source_home: &Path,
    source_config: &Path,
    gateway_url: &str,
) -> Result<(), CliError> {
    let absolute_overlay = overlay
        .canonicalize()
        .unwrap_or_else(|_| overlay.to_path_buf());
    match std::fs::read_dir(source_home) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry?;
                let name = entry.file_name();
                if name == "config.yaml" || name == "shell-hooks-allowlist.json" {
                    continue;
                }
                let source = entry.path();
                let absolute_source = source.canonicalize().unwrap_or_else(|_| source.clone());
                if absolute_overlay.starts_with(absolute_source) {
                    continue;
                }
                link_hermes_state(&source, &overlay.join(name), entry.file_type()?.is_dir())?;
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(CliError::Io(error)),
    }
    let existing = match std::fs::read_to_string(source_config) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(CliError::Io(error)),
    };
    let relay = std::env::current_exe()
        .map(|path| path.canonicalize().unwrap_or(path))
        .map(crate::plugin_host::portable_executable_path)
        .unwrap_or_else(|_| PathBuf::from("nemo-relay"));
    let contents = crate::hermes::transparent_config(&existing, &relay, gateway_url)?;
    std::fs::write(overlay.join("config.yaml"), contents)?;
    Ok(())
}

fn link_hermes_state(source: &Path, destination: &Path, directory: bool) -> Result<(), CliError> {
    #[cfg(unix)]
    {
        let _ = directory;
        std::os::unix::fs::symlink(source, destination)?;
        Ok(())
    }
    #[cfg(windows)]
    {
        if directory {
            create_windows_junction(source, destination)?;
        } else {
            match std::fs::hard_link(source, destination) {
                Ok(()) => {}
                Err(_) => {
                    std::fs::copy(source, destination)?;
                }
            }
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = directory;
        std::fs::copy(source, destination)?;
        Ok(())
    }
}

#[cfg(windows)]
fn create_windows_junction(source: &Path, destination: &Path) -> Result<(), CliError> {
    use std::os::windows::process::CommandExt;

    // Directory junctions do not require Developer Mode or SeCreateSymbolicLinkPrivilege. Paths
    // travel through environment variables so the fixed cmd program never interpolates user
    // content into shell syntax; delayed expansion is disabled for literal exclamation marks.
    let mut command = std::process::Command::new(
        std::env::var_os("COMSPEC").unwrap_or_else(|| std::ffi::OsString::from("cmd.exe")),
    );
    command.args(["/d", "/e:on", "/v:off", "/s", "/c"]);
    // `cmd.exe` parses the command after `/c` itself rather than with the Windows CRT rules used
    // by `Command::arg`. The outer quote pair is required so the inner path quotes survive `/s`.
    command
        .raw_arg(r#""mklink /J "%NEMO_RELAY_JUNCTION_DEST%" "%NEMO_RELAY_JUNCTION_SOURCE%" >nul""#);
    let status = command
        .env("NEMO_RELAY_JUNCTION_SOURCE", source)
        .env("NEMO_RELAY_JUNCTION_DEST", destination)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(CliError::Launch(format!(
            "failed to create Hermes state junction {} -> {}: {status}",
            destination.display(),
            source.display()
        )))
    }
}

// Chooses the Hermes config used as the source for a transparent-run overlay. If setup recorded a
// specific path, reuse it; otherwise fall back to the active Hermes home.
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

// Creates a uniquely named directory under the OS temp directory. UUIDv7 avoids collisions
// between concurrent transparent runs without keeping persistent coordination state.
fn temp_dir(prefix: &str) -> Result<PathBuf, CliError> {
    private_temp_dir(&std::env::temp_dir(), prefix)
}

fn private_temp_dir(parent: &Path, prefix: &str) -> Result<PathBuf, CliError> {
    let path = parent.join(format!("{prefix}-{}", uuid::Uuid::now_v7()));
    #[cfg(unix)]
    let builder = {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = std::fs::DirBuilder::new();
        builder.mode(0o700);
        builder
    };
    #[cfg(not(unix))]
    let builder = std::fs::DirBuilder::new();
    builder.create(&path)?;
    Ok(path)
}

#[cfg(test)]
#[path = "../tests/coverage/launcher_tests.rs"]
mod tests;
