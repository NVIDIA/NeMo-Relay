// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::process::Command;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::config::{
    AgentConfigs, CodingAgent, ResolvedConfig, RunCommand, ServerArgs, SidecarConfig,
    resolve_run_config,
};
use crate::error::SidecarError;
use crate::installer::{generated_hooks, hook_forward_command, merge_hooks, read_json_file};
use crate::server;

/// Runs a child coding-agent command behind an ephemeral local sidecar.
///
/// The sidecar binds to an OS-assigned loopback port, prepares agent-specific hook/gateway wiring,
/// waits for health before spawning the child, and restores temporary files after the child and
/// server shut down. The child's exit status is preserved when it fits in `ExitCode`; otherwise the
/// launcher reports generic failure.
pub(crate) async fn run(
    command: RunCommand,
    inherited: Option<&ServerArgs>,
) -> Result<ExitCode, SidecarError> {
    let run = TransparentRun::new(command, inherited).await?;
    run.print_if_requested();
    run.execute().await
}

struct TransparentRun {
    agent: CodingAgent,
    prepared: PreparedRun,
    resolved: ResolvedConfig,
    listener: TcpListener,
    sidecar_url: String,
    dry_run: bool,
    print: bool,
}

impl TransparentRun {
    // Resolves configuration, binds the ephemeral listener, and builds agent-specific launch wiring
    // without starting the sidecar or spawning the child command.
    async fn new(
        command: RunCommand,
        inherited: Option<&ServerArgs>,
    ) -> Result<Self, SidecarError> {
        let dry_run = command.dry_run;
        let print = command.print;
        let mut resolved = resolve_run_config(&command, inherited)?;
        let (agent, argv) = resolve_agent_and_argv(&command, &resolved.agents)?;
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let sidecar_url = format!("http://{address}");
        resolved.sidecar.bind = address;

        let prepared = PreparedRun::new(agent, argv, &sidecar_url, &resolved, dry_run)?;
        Ok(Self {
            agent,
            prepared,
            resolved,
            listener,
            sidecar_url,
            dry_run,
            print,
        })
    }

    // Emits the resolved run plan when requested. Dry runs always print because inspection is their
    // primary behavior; live runs print only when `--print` was passed.
    fn print_if_requested(&self) {
        if self.print || self.dry_run {
            self.prepared
                .print(self.agent, &self.sidecar_url, &self.resolved);
        }
    }

    // Runs the prepared child command unless this is an inspection-only dry run.
    async fn execute(self) -> Result<ExitCode, SidecarError> {
        if self.dry_run {
            return Ok(ExitCode::SUCCESS);
        }
        execute_live_run(
            self.listener,
            self.resolved.sidecar,
            &self.sidecar_url,
            self.prepared,
        )
        .await
    }
}

// Starts the sidecar, waits for readiness, runs the child command, restores temporary state, and then
// maps the child process status to the launcher's exit code.
async fn execute_live_run(
    listener: TcpListener,
    sidecar_config: SidecarConfig,
    sidecar_url: &str,
    prepared: PreparedRun,
) -> Result<ExitCode, SidecarError> {
    let running_server = RunningSidecar::start(listener, sidecar_config);
    if let Err(error) = wait_for_health(sidecar_url).await {
        let _ = running_server.stop().await;
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
) -> Result<(CodingAgent, Vec<String>), SidecarError> {
    let argv = resolved_argv(command, agents)?;
    let agent = resolved_agent(command, &argv)?;
    Ok((agent, argv))
}

// Returns the command argv supplied on the CLI, or the configured command for an explicitly selected
// agent. Empty CLI argv without `--agent` is rejected before inference because there is no executable
// name to inspect.
fn resolved_argv(command: &RunCommand, agents: &AgentConfigs) -> Result<Vec<String>, SidecarError> {
    if !command.command.is_empty() {
        return Ok(command.command.clone());
    }
    let agent = command.agent.ok_or_else(|| {
        SidecarError::Launch(
            "missing command; pass -- <agent-command> or --agent with a configured command".into(),
        )
    })?;
    configured_command(agent, agents).ok_or_else(|| {
        SidecarError::Launch(format!(
            "no configured command for {}; pass -- <agent-command>",
            agent.as_arg()
        ))
    })
}

// Uses an explicit `--agent` when present and otherwise infers the agent from argv[0]. Inference is
// intentionally late so configured commands and direct CLI commands share the same validation path.
fn resolved_agent(command: &RunCommand, argv: &[String]) -> Result<CodingAgent, SidecarError> {
    if let Some(agent) = command.agent {
        return Ok(agent);
    }
    CodingAgent::infer(&argv[0]).ok_or_else(|| {
        SidecarError::Launch(format!(
            "could not infer coding agent from command {:?}; pass --agent claude-code, --agent codex, --agent cursor, or --agent hermes",
            argv[0]
        ))
    })
}

// Splits a configured command string into argv words for run mode. This intentionally uses simple
// whitespace splitting because config command values are a convenience fallback; complex shell
// commands should be passed after `--` by the caller.
fn configured_command(agent: CodingAgent, agents: &AgentConfigs) -> Option<Vec<String>> {
    let command = match agent {
        CodingAgent::ClaudeCode => agents.claude_code.command.as_ref(),
        CodingAgent::Codex => agents.codex.command.as_ref(),
        CodingAgent::Cursor => agents.cursor.command.as_ref(),
        CodingAgent::Hermes => agents.hermes.command.as_ref(),
    }?;
    let argv: Vec<_> = command.split_whitespace().map(ToOwned::to_owned).collect();
    (!argv.is_empty()).then_some(argv)
}

struct PreparedRun {
    argv: Vec<String>,
    env: Vec<(String, String)>,
    temp_dirs: Vec<PathBuf>,
    cursor_restore: Option<CursorRestore>,
    notes: Vec<String>,
}

struct CursorRestore {
    path: PathBuf,
    backup_path: Option<PathBuf>,
    had_original: bool,
}

struct RunningSidecar {
    shutdown_tx: oneshot::Sender<()>,
    task: JoinHandle<Result<(), SidecarError>>,
}

impl RunningSidecar {
    // Starts the sidecar listener on a background task and keeps the shutdown sender paired with the
    // task handle so health failures and normal exits use identical cleanup semantics.
    fn start(listener: TcpListener, config: crate::config::SidecarConfig) -> Self {
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            server::serve_listener(listener, config, Some(shutdown_rx)).await
        });
        Self { shutdown_tx, task }
    }

    // Requests shutdown and joins the server task. The send can fail only if the task already exited;
    // the join result still captures whether serving ended cleanly.
    async fn stop(self) -> Result<(), SidecarError> {
        let _ = self.shutdown_tx.send(());
        self.task
            .await
            .map_err(|error| SidecarError::Launch(format!("sidecar task failed: {error}")))?
    }
}

impl PreparedRun {
    // Builds the launch plan and applies only the preparation needed by the selected agent.
    // Dry-run preparation records equivalent notes and argv/env changes without writing temporary
    // hook files or patching user/project configuration.
    fn new(
        agent: CodingAgent,
        argv: Vec<String>,
        sidecar_url: &str,
        resolved: &ResolvedConfig,
        dry_run: bool,
    ) -> Result<Self, SidecarError> {
        let mut run = Self {
            argv,
            env: vec![("NEMO_FLOW_SIDECAR_URL".into(), sidecar_url.into())],
            temp_dirs: Vec::new(),
            cursor_restore: None,
            notes: Vec::new(),
        };
        match agent {
            CodingAgent::ClaudeCode => {
                if dry_run {
                    run.prepare_claude_dry(sidecar_url);
                } else {
                    run.prepare_claude(sidecar_url)?;
                }
            }
            CodingAgent::Codex => run.prepare_codex(sidecar_url),
            CodingAgent::Cursor => {
                if resolved.agents.cursor.patch_restore_hooks {
                    if dry_run {
                        run.prepare_cursor_dry()?;
                    } else {
                        run.prepare_cursor()?;
                    }
                }
            }
            CodingAgent::Hermes => run.prepare_hermes(),
        }
        Ok(run)
    }

    // Records the Claude Code argv/env changes that would be made during a real run. The temporary
    // plugin path is symbolic so printed dry-run output is deterministic and non-mutating.
    fn prepare_claude_dry(&mut self, sidecar_url: &str) {
        insert_after_agent(
            &mut self.argv,
            CodingAgent::ClaudeCode,
            [
                "--plugin-dir".into(),
                "<temporary-claude-plugin-dir>".into(),
            ],
        );
        self.env
            .push(("ANTHROPIC_BASE_URL".into(), sidecar_url.to_string()));
        self.notes
            .push("would generate a temporary Claude Code plugin directory".into());
    }

    // Creates a temporary Claude Code plugin containing sidecar hooks and points Claude at both
    // that plugin directory and the sidecar Anthropic-compatible gateway URL.
    fn prepare_claude(&mut self, sidecar_url: &str) -> Result<(), SidecarError> {
        let root = temp_dir("nemo-flow-claude-plugin")?;
        std::fs::create_dir_all(root.join(".claude-plugin"))?;
        std::fs::create_dir_all(root.join("hooks"))?;
        std::fs::write(
            root.join(".claude-plugin/plugin.json"),
            serde_json::to_vec_pretty(&json!({
                "name": "nemo-flow-sidecar",
                "version": env!("CARGO_PKG_VERSION"),
                "description": "Temporary NeMo Flow sidecar hooks"
            }))
            .map_err(|error| SidecarError::Launch(error.to_string()))?,
        )?;
        write_hooks(
            &root.join("hooks/hooks.json"),
            generated_hooks(
                CodingAgent::ClaudeCode,
                &hook_forward_command(CodingAgent::ClaudeCode),
            ),
        )?;
        insert_after_agent(
            &mut self.argv,
            CodingAgent::ClaudeCode,
            ["--plugin-dir".into(), root.display().to_string()],
        );
        self.env
            .push(("ANTHROPIC_BASE_URL".into(), sidecar_url.to_string()));
        self.temp_dirs.push(root);
        Ok(())
    }

    // Injects Codex hook and provider-base configuration through repeated `--config` flags. The
    // generated TOML hook groups are passed inline so transparent run mode does not edit the user's
    // persistent Codex config.
    fn prepare_codex(&mut self, sidecar_url: &str) {
        let hook_command = hook_forward_command(CodingAgent::Codex);
        let mut args = vec![
            "--config".to_string(),
            "features.codex_hooks=true".to_string(),
            "--config".to_string(),
            format!(
                "model_providers.openai.base_url={}",
                toml_string(sidecar_url)
            ),
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

    // Temporarily merges Cursor hooks into the nearest project `.cursor/hooks.json`, backing up the
    // original if it exists. Cursor discovers hooks from files, so run mode patches and later
    // restores project state rather than passing hook config on the command line.
    fn prepare_cursor(&mut self) -> Result<(), SidecarError> {
        let path = cursor_hooks_path()?;
        let (had_original, backup_path) = backup_existing_cursor_hooks(&path)?;
        write_merged_cursor_hooks(&path)?;
        self.cursor_restore = Some(CursorRestore {
            path,
            backup_path,
            had_original,
        });
        Ok(())
    }

    // Records the Cursor hook file that would be patched during a real run without touching the
    // filesystem, preserving dry-run as an inspection-only operation.
    fn prepare_cursor_dry(&mut self) -> Result<(), SidecarError> {
        let path = cursor_hooks_path()?;
        self.notes.push(format!(
            "would temporarily merge NeMo Flow hooks into {}",
            path.display()
        ));
        Ok(())
    }

    // Notes Hermes' persistent-hook requirement. Hermes hook approval is outside this launcher, so
    // run mode only exports the live sidecar URL for hooks that are already installed and approved.
    fn prepare_hermes(&mut self) {
        self.notes.push(
            "Hermes shell hooks must be configured with `nemo-flow-sidecar install hermes`; this run exports the dynamic sidecar URL for approved hooks".into(),
        );
    }

    // Spawns the prepared child process with injected environment and waits for its exit status.
    // Stdio is inherited by default so agent interaction remains unchanged in transparent mode.
    async fn spawn_and_wait(&self) -> Result<std::process::ExitStatus, SidecarError> {
        let mut command = Command::new(&self.argv[0]);
        command.args(&self.argv[1..]);
        for (name, value) in &self.env {
            command.env(name, value);
        }
        let mut child = command.spawn()?;
        child.wait().await.map_err(SidecarError::from)
    }

    // Removes temporary directories and restores Cursor hook files after the child exits. Restore
    // errors are surfaced after the child status is collected so cleanup problems are not hidden.
    fn restore(&self) -> Result<(), SidecarError> {
        for dir in &self.temp_dirs {
            let _ = std::fs::remove_dir_all(dir);
        }
        let Some(cursor) = &self.cursor_restore else {
            return Ok(());
        };
        match (&cursor.backup_path, cursor.had_original) {
            (Some(backup), true) => {
                std::fs::copy(backup, &cursor.path).map_err(|error| {
                    SidecarError::Launch(format!(
                        "failed to restore Cursor hooks from {}: {error}",
                        backup.display()
                    ))
                })?;
                let _ = std::fs::remove_file(backup);
            }
            (_, false) => {
                if cursor.path.exists() {
                    std::fs::remove_file(&cursor.path).map_err(|error| {
                        SidecarError::Launch(format!(
                            "failed to remove temporary Cursor hooks {}: {error}",
                            cursor.path.display()
                        ))
                    })?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    // Prints the resolved transparent-run plan, including dynamic sidecar URL, upstream base URLs,
    // argv/env injection, and any agent-specific notes or temporary files.
    fn print(&self, agent: CodingAgent, sidecar_url: &str, resolved: &ResolvedConfig) {
        println!("agent = {}", agent.as_arg());
        println!("sidecar_url = {sidecar_url}");
        println!("openai_base_url = {}", resolved.sidecar.openai_base_url);
        println!(
            "anthropic_base_url = {}",
            resolved.sidecar.anthropic_base_url
        );
        if let Some(path) = &resolved.sidecar.atif_dir {
            println!("atif_dir = {}", path.display());
        }
        if let Some(endpoint) = &resolved.sidecar.openinference_endpoint {
            println!("openinference_endpoint = {endpoint}");
        }
        println!("argv = {}", self.argv.join(" "));
        for (name, value) in &self.env {
            println!("env.{name} = {value}");
        }
        if let Some(cursor) = &self.cursor_restore {
            println!("cursor_hooks = {}", cursor.path.display());
        }
        for note in &self.notes {
            println!("note = {note}");
        }
    }
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

// Polls the ephemeral sidecar health endpoint for roughly one second before launching the agent.
// Startup failures return a launcher error so the child command is never run against a dead proxy.
async fn wait_for_health(sidecar_url: &str) -> Result<(), SidecarError> {
    let client = Client::new();
    let url = format!("{}/healthz", sidecar_url.trim_end_matches('/'));
    for _ in 0..50 {
        if let Ok(response) = client.get(&url).send().await
            && response.status().is_success()
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    Err(SidecarError::Launch(format!(
        "sidecar did not become ready at {url}"
    )))
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
fn write_hooks(path: &Path, hooks: Value) -> Result<(), SidecarError> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&hooks)
            .map_err(|error| SidecarError::Launch(error.to_string()))?,
    )?;
    Ok(())
}

// Backs up an existing Cursor hook file before run-mode patching. The return value records both the
// original-file state and backup path so restore can either copy back or remove the generated file.
fn backup_existing_cursor_hooks(path: &Path) -> Result<(bool, Option<PathBuf>), SidecarError> {
    let had_original = path.exists();
    if !had_original {
        return Ok((false, None));
    }
    let backup = path.with_extension(format!("json.nemo-flow-run.bak.{}", timestamp()?));
    std::fs::copy(path, &backup)?;
    Ok((true, Some(backup)))
}

// Creates the Cursor hooks parent directory when needed, merges generated sidecar hooks with any
// existing hook file, and writes the patched JSON used for this transparent run.
fn write_merged_cursor_hooks(path: &Path) -> Result<(), SidecarError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let contents = serde_json::to_string_pretty(&merge_hooks(
        read_json_file(path)?,
        generated_hooks(
            CodingAgent::Cursor,
            &hook_forward_command(CodingAgent::Cursor),
        ),
    )?)
    .map_err(|error| SidecarError::Launch(error.to_string()))?;
    std::fs::write(path, contents)?;
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
fn temp_dir(prefix: &str) -> Result<PathBuf, SidecarError> {
    let path = std::env::temp_dir().join(format!("{prefix}-{}", timestamp()?));
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

// Locates Cursor's project hook file by walking up to the nearest ancestor that already contains a
// `.cursor` directory, falling back to the current directory for first-time project setup.
fn cursor_hooks_path() -> Result<PathBuf, SidecarError> {
    let cwd = std::env::current_dir()?;
    let project = cwd
        .ancestors()
        .find(|ancestor| ancestor.join(".cursor").is_dir())
        .unwrap_or(cwd.as_path());
    Ok(project.join(".cursor/hooks.json"))
}

// Returns a monotonic-enough wall-clock nanosecond stamp for temp and backup names. System time
// errors become launcher errors because paths cannot be safely generated without a timestamp.
fn timestamp() -> Result<u128, SidecarError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| SidecarError::Launch(error.to_string()))?
        .as_nanos())
}

#[cfg(test)]
#[path = "../tests/coverage/launcher_tests.rs"]
mod tests;
