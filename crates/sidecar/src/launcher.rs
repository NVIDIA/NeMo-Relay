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

use crate::config::{
    AgentConfigs, CodingAgent, ResolvedConfig, RunCommand, ServerArgs, resolve_run_config,
};
use crate::error::SidecarError;
use crate::installer::{generated_hooks, hook_forward_command, merge_hooks, read_json_file};
use crate::server;

pub(crate) async fn run(
    command: RunCommand,
    inherited: Option<&ServerArgs>,
) -> Result<ExitCode, SidecarError> {
    let mut resolved = resolve_run_config(&command, inherited)?;
    let (agent, argv) = resolve_agent_and_argv(&command, &resolved.agents)?;
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let sidecar_url = format!("http://{address}");
    resolved.sidecar.bind = address;

    let prepared = PreparedRun::new(agent, argv, &sidecar_url, &resolved, command.dry_run)?;
    if command.print || command.dry_run {
        prepared.print(agent, &sidecar_url, &resolved);
    }
    if command.dry_run {
        return Ok(ExitCode::SUCCESS);
    }

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server_config = resolved.sidecar.clone();
    let server_task = tokio::spawn(async move {
        server::serve_listener(listener, server_config, Some(shutdown_rx)).await
    });
    if let Err(error) = wait_for_health(&sidecar_url).await {
        let _ = shutdown_tx.send(());
        let _ = server_task.await;
        return Err(error);
    }

    let status = prepared.spawn_and_wait().await;
    let restore = prepared.restore();
    let _ = shutdown_tx.send(());
    let server_result = server_task
        .await
        .map_err(|error| SidecarError::Launch(format!("sidecar task failed: {error}")))?;
    restore?;
    server_result?;

    let status = status?;
    Ok(status
        .code()
        .and_then(|code| u8::try_from(code).ok())
        .map(ExitCode::from)
        .unwrap_or(ExitCode::FAILURE))
}

fn resolve_agent_and_argv(
    command: &RunCommand,
    agents: &AgentConfigs,
) -> Result<(CodingAgent, Vec<String>), SidecarError> {
    let argv = if command.command.is_empty() {
        let agent = command.agent.ok_or_else(|| {
            SidecarError::Launch(
                "missing command; pass -- <agent-command> or --agent with a configured command"
                    .into(),
            )
        })?;
        configured_command(agent, agents).ok_or_else(|| {
            SidecarError::Launch(format!(
                "no configured command for {}; pass -- <agent-command>",
                agent.as_arg()
            ))
        })?
    } else {
        command.command.clone()
    };

    let agent = match command.agent {
        Some(agent) => agent,
        None => CodingAgent::infer(&argv[0]).ok_or_else(|| {
            SidecarError::Launch(format!(
                "could not infer coding agent from command {:?}; pass --agent claude-code, --agent codex, or --agent cursor",
                argv[0]
            ))
        })?,
    };
    Ok((agent, argv))
}

fn configured_command(agent: CodingAgent, agents: &AgentConfigs) -> Option<Vec<String>> {
    let command = match agent {
        CodingAgent::ClaudeCode => agents.claude_code.command.as_ref(),
        CodingAgent::Codex => agents.codex.command.as_ref(),
        CodingAgent::Cursor => agents.cursor.command.as_ref(),
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

impl PreparedRun {
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
        }
        Ok(run)
    }

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

    fn prepare_cursor(&mut self) -> Result<(), SidecarError> {
        let path = std::env::current_dir()?.join(".cursor/hooks.json");
        let had_original = path.exists();
        let backup_path = if had_original {
            let backup = path.with_extension(format!("json.nemo-flow-run.bak.{}", timestamp()?));
            std::fs::copy(&path, &backup)?;
            Some(backup)
        } else {
            None
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(&merge_hooks(
            read_json_file(&path)?,
            generated_hooks(
                CodingAgent::Cursor,
                &hook_forward_command(CodingAgent::Cursor),
            ),
        )?)
        .map_err(|error| SidecarError::Launch(error.to_string()))?;
        std::fs::write(&path, contents)?;
        self.cursor_restore = Some(CursorRestore {
            path,
            backup_path,
            had_original,
        });
        Ok(())
    }

    fn prepare_cursor_dry(&mut self) -> Result<(), SidecarError> {
        let path = std::env::current_dir()?.join(".cursor/hooks.json");
        self.notes.push(format!(
            "would temporarily merge NeMo Flow hooks into {}",
            path.display()
        ));
        Ok(())
    }

    async fn spawn_and_wait(&self) -> Result<std::process::ExitStatus, SidecarError> {
        let mut command = Command::new(&self.argv[0]);
        command.args(&self.argv[1..]);
        for (name, value) in &self.env {
            command.env(name, value);
        }
        let mut child = command.spawn()?;
        child.wait().await.map_err(SidecarError::from)
    }

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

fn write_hooks(path: &Path, hooks: Value) -> Result<(), SidecarError> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&hooks)
            .map_err(|error| SidecarError::Launch(error.to_string()))?,
    )?;
    Ok(())
}

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

fn toml_string(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn temp_dir(prefix: &str) -> Result<PathBuf, SidecarError> {
    let path = std::env::temp_dir().join(format!("{prefix}-{}", timestamp()?));
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

fn timestamp() -> Result<u128, SidecarError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| SidecarError::Launch(error.to_string()))?
        .as_nanos())
}

#[cfg(test)]
#[path = "../tests/coverage/launcher_tests.rs"]
mod tests;
