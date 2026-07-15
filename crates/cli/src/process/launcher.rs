// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use nemo_relay::observability::plugin_component::{
    AtifStorageConfig, AtofSinkSectionConfig, OBSERVABILITY_PLUGIN_KIND, ObservabilityConfig,
};
use nemo_relay::plugin::PluginConfig;
use serde_json::Value;
#[cfg(test)]
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::agents::CodingAgent;
use crate::configuration::{AgentConfigs, GatewayConfig, ResolvedConfig, resolve_run_config};
use crate::error::CliError;
use crate::plugins::lifecycle::ActiveDynamicPluginComponent;
use crate::server;
use crate::server::GatewayOverrides;

use super::{PreparedAgentLaunch, RunOverrides};

/// Runs a child coding-agent command behind an ephemeral local gateway.
///
/// The gateway binds to an OS-assigned loopback port, prepares agent-specific hook/gateway wiring,
/// waits for health before spawning the child, and removes temporary state after the child and
/// server shut down. The child's exit status is preserved when it fits in `ExitCode`; otherwise the
/// launcher reports generic failure.
pub(crate) async fn run(
    command: RunOverrides,
    inherited: Option<&GatewayOverrides>,
) -> Result<ExitCode, CliError> {
    let run = TransparentRun::new(command, inherited).await?;
    run.print_if_requested();
    run.execute().await
}

struct TransparentRun {
    agent: CodingAgent,
    prepared: PreparedAgentLaunch,
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
    async fn new(
        command: RunOverrides,
        inherited: Option<&GatewayOverrides>,
    ) -> Result<Self, CliError> {
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
            let probe = crate::process::version_probe_argv(
                agent,
                &invocation.argv[..=invocation.host_index],
            );
            validate_agent_version(agent, &probe).await?;
        }
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let gateway_url = format!("http://{address}");
        resolved.gateway.bind = address;

        let prepared =
            PreparedAgentLaunch::from_invocation(invocation, &gateway_url, &resolved, dry_run)?;
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
    prepared: PreparedAgentLaunch,
) -> Result<ExitCode, CliError> {
    execute_live_run_with_dynamic(listener, gateway_config, Vec::new(), gateway_url, prepared).await
}

async fn execute_live_run_with_dynamic(
    listener: TcpListener,
    gateway_config: GatewayConfig,
    dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
    gateway_url: &str,
    prepared: PreparedAgentLaunch,
) -> Result<ExitCode, CliError> {
    let bootstrap_fingerprint = crate::configuration::transparent_gateway_fingerprint(gateway_url);
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
    prepared: &PreparedAgentLaunch,
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
    command: &RunOverrides,
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
    command: &RunOverrides,
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
    let mut command = crate::process::tokio_command(probe);
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
    let command = crate::agents::config(agent, agents).command.as_ref()?;
    let argv = crate::process::command_argv(command);
    (!argv.is_empty()).then_some(argv)
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
        config: crate::configuration::GatewayConfig,
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

impl PreparedAgentLaunch {
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
                (
                    crate::configuration::GATEWAY_URL_ENV.into(),
                    gateway_url.into(),
                ),
                (crate::configuration::TRANSPARENT_RUN_ENV.into(), "1".into()),
            ],
            temp_dirs: Vec::new(),
            notes: Vec::new(),
        };
        if let Some(path) = path_with_transparent_hook_dir() {
            run.env.push(("PATH".into(), path));
        }
        crate::agents::prepare_launch(agent, &mut run, gateway_url, resolved, dry_run)?;
        Ok(run)
    }

    // Injects Codex hook and provider configuration through repeated `--config` flags. Codex
    // reserves built-in provider IDs, so run mode installs a temporary provider alias instead of
    // overriding `model_providers.openai`. Uses `features.hooks=true` introduced in codex-cli
    // current supported Codex releases. The centralized host policy validates the version first.

    // Hermes discovers hooks from `.hermes/config.yaml` instead of command-line flags. A
    // process-private HERMES_HOME exposes dynamic hooks without rewriting user configuration.

    // Records the Hermes hook file that would be patched during a real run without touching the
    // filesystem, preserving dry-run as an inspection-only operation.

    // Spawns the prepared child process with injected environment.
    // Stdio is inherited by default so agent interaction remains unchanged in transparent mode.
    async fn spawn(&self) -> Result<crate::process::SupervisedChild, CliError> {
        let mut command = crate::process::tokio_command(&self.argv);
        for (name, value) in &self.env {
            command.env(name, value);
        }
        crate::process::SupervisedChild::spawn(&mut command)
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
// Session hook definitions and their exact trust state share Codex's process-local CLI layer. This
// authorizes only the generated Relay command without rewriting the active user profile or using
// the process-wide hook-trust bypass.

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
        for sink in &section.sinks {
            match sink {
                AtofSinkSectionConfig::File(file) => {
                    let directory = file
                        .output_directory
                        .clone()
                        .unwrap_or_else(current_output_directory);
                    let path = directory.join(
                        file.filename
                            .clone()
                            .unwrap_or_else(|| "nemo-relay-events-<timestamp>.jsonl".into()),
                    );
                    destinations.push(format!("ATOF {}", path.display()));
                }
                AtofSinkSectionConfig::Stream(stream) => {
                    destinations.push(format!("ATOF {}", sanitized_url(&stream.url)));
                }
            }
        }
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
                .map(sanitized_url)
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
                .map(sanitized_url)
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
        AtifStorageConfig::Http(http) => sanitized_url(&http.endpoint),
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

fn sanitized_url(value: &str) -> String {
    let Ok(mut url) = reqwest::Url::parse(value) else {
        return "configured endpoint".into();
    };
    let _ = url.set_username("");
    let _ = url.set_password(None);
    if url.query().is_some() {
        let keys = url
            .query_pairs()
            .map(|(key, _)| key.into_owned())
            .collect::<Vec<_>>();
        url.set_query(None);
        if !keys.is_empty() {
            let mut query = url.query_pairs_mut();
            for key in keys {
                query.append_pair(&key, "[REDACTED]");
            }
        }
    }
    url.to_string()
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
            crate::gateway::client::healthz_compatible(&gateway_url, &bootstrap_fingerprint)
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

// Chooses the Hermes config used as the source for a transparent-run overlay. If setup recorded a
// specific path, reuse it; otherwise fall back to the active Hermes home.

// Converts JSON hook groups into inline TOML arrays for Codex `--config` flags. The function
// preserves matchers when present and assumes generated hook groups contain one command hook.

// Escapes a Rust string as a TOML basic string for inline Codex configuration values.

// Creates a uniquely named directory under the OS temp directory. UUIDv7 avoids collisions
// between concurrent transparent runs without keeping persistent coordination state.

#[cfg(test)]
#[path = "../../tests/coverage/agents/launcher_tests.rs"]
mod tests;
