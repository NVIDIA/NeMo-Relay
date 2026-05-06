// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::config::{AgentCommandConfig, CursorAgentConfig, SidecarConfig};
use std::sync::{Mutex, OnceLock};

fn current_dir_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[test]
fn infers_agent_from_command_or_uses_override() {
    let command = RunCommand {
        agent: None,
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        atif_dir: None,
        openinference_endpoint: None,
        session_metadata: None,
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec!["/usr/bin/codex".into()],
    };
    let (agent, argv) = resolve_agent_and_argv(&command, &AgentConfigs::default()).unwrap();
    assert_eq!(agent, CodingAgent::Codex);
    assert_eq!(argv, vec!["/usr/bin/codex"]);

    let command = RunCommand {
        agent: Some(CodingAgent::ClaudeCode),
        command: vec!["wrapper".into()],
        ..command
    };
    let (agent, _) = resolve_agent_and_argv(&command, &AgentConfigs::default()).unwrap();
    assert_eq!(agent, CodingAgent::ClaudeCode);
}

#[test]
fn uses_configured_command_when_no_argv_is_supplied() {
    let agents = AgentConfigs {
        codex: AgentCommandConfig {
            command: Some("codex --full-auto".into()),
        },
        ..AgentConfigs::default()
    };
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        atif_dir: None,
        openinference_endpoint: None,
        session_metadata: None,
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec![],
    };

    let (agent, argv) = resolve_agent_and_argv(&command, &agents).unwrap();

    assert_eq!(agent, CodingAgent::Codex);
    assert_eq!(argv, vec!["codex", "--full-auto"]);
}

#[test]
fn uses_configured_hermes_command_when_no_argv_is_supplied() {
    let agents = AgentConfigs {
        hermes: AgentCommandConfig {
            command: Some("hermes --yolo chat".into()),
        },
        ..AgentConfigs::default()
    };
    let command = RunCommand {
        agent: Some(CodingAgent::Hermes),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        atif_dir: None,
        openinference_endpoint: None,
        session_metadata: None,
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec![],
    };

    let (agent, argv) = resolve_agent_and_argv(&command, &agents).unwrap();

    assert_eq!(agent, CodingAgent::Hermes);
    assert_eq!(argv, vec!["hermes", "--yolo", "chat"]);
}

#[test]
fn inference_failure_has_actionable_message() {
    let command = RunCommand {
        agent: None,
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        atif_dir: None,
        openinference_endpoint: None,
        session_metadata: None,
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec!["my-agent".into()],
    };

    let error = resolve_agent_and_argv(&command, &AgentConfigs::default())
        .unwrap_err()
        .to_string();

    assert!(error.contains("pass --agent claude-code"));
}

#[test]
fn prepares_codex_config_overrides() {
    let resolved = ResolvedConfig {
        sidecar: SidecarConfig::default(),
        agents: AgentConfigs::default(),
    };
    let prepared = PreparedRun::new(
        CodingAgent::Codex,
        vec!["codex".into()],
        "http://127.0.0.1:1234",
        &resolved,
        false,
    )
    .unwrap();

    assert!(prepared.argv.contains(&"features.codex_hooks=true".into()));
    assert!(
        prepared
            .argv
            .iter()
            .any(|arg| arg.contains("model_providers.openai.base_url"))
    );
    assert!(
        prepared
            .argv
            .iter()
            .any(|arg| arg.contains("hooks.SessionStart"))
    );
}

#[test]
fn prepares_hermes_hook_environment() {
    let resolved = ResolvedConfig {
        sidecar: SidecarConfig::default(),
        agents: AgentConfigs::default(),
    };
    let prepared = PreparedRun::new(
        CodingAgent::Hermes,
        vec!["hermes".into(), "chat".into()],
        "http://127.0.0.1:1234",
        &resolved,
        false,
    )
    .unwrap();

    assert_eq!(prepared.argv, vec!["hermes", "chat"]);
    assert!(prepared.env.contains(&(
        "NEMO_FLOW_SIDECAR_URL".into(),
        "http://127.0.0.1:1234".into()
    )));
    assert!(
        !prepared
            .env
            .iter()
            .any(|(name, _)| name == "HERMES_ACCEPT_HOOKS")
    );
    assert!(prepared.notes[0].contains("approved hooks"));
}

#[test]
fn prepares_claude_temp_plugin() {
    let resolved = ResolvedConfig {
        sidecar: SidecarConfig::default(),
        agents: AgentConfigs::default(),
    };
    let prepared = PreparedRun::new(
        CodingAgent::ClaudeCode,
        vec!["claude".into()],
        "http://127.0.0.1:1234",
        &resolved,
        false,
    )
    .unwrap();

    let plugin_index = prepared
        .argv
        .iter()
        .position(|arg| arg == "--plugin-dir")
        .unwrap();
    let plugin_dir = PathBuf::from(&prepared.argv[plugin_index + 1]);
    assert!(plugin_dir.join("hooks/hooks.json").exists());
    assert!(
        prepared
            .env
            .contains(&("ANTHROPIC_BASE_URL".into(), "http://127.0.0.1:1234".into()))
    );
    prepared.restore().unwrap();
}

#[test]
fn cursor_patch_restore_restores_original_file() {
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let previous = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp.path()).unwrap();
    std::fs::create_dir_all(".cursor").unwrap();
    std::fs::write(".cursor/hooks.json", r#"{"hooks":{"sessionStart":[]}}"#).unwrap();
    let resolved = ResolvedConfig {
        sidecar: SidecarConfig::default(),
        agents: AgentConfigs {
            cursor: CursorAgentConfig {
                command: None,
                patch_restore_hooks: true,
            },
            ..AgentConfigs::default()
        },
    };

    let prepared = PreparedRun::new(
        CodingAgent::Cursor,
        vec!["cursor-agent".into()],
        "http://s",
        &resolved,
        false,
    )
    .unwrap();
    assert!(
        std::fs::read_to_string(".cursor/hooks.json")
            .unwrap()
            .contains("hook-forward cursor")
    );
    prepared.restore().unwrap();
    assert_eq!(
        std::fs::read_to_string(".cursor/hooks.json").unwrap(),
        r#"{"hooks":{"sessionStart":[]}}"#
    );
    std::env::set_current_dir(previous).unwrap();
}

#[test]
fn cursor_patch_restore_uses_nearest_project_cursor_dir() {
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let previous = std::env::current_dir().unwrap();
    std::fs::create_dir_all(temp.path().join(".cursor")).unwrap();
    std::fs::create_dir_all(temp.path().join("nested")).unwrap();
    std::fs::write(
        temp.path().join(".cursor/hooks.json"),
        r#"{"hooks":{"sessionStart":[]}}"#,
    )
    .unwrap();
    std::env::set_current_dir(temp.path().join("nested")).unwrap();
    let resolved = ResolvedConfig {
        sidecar: SidecarConfig::default(),
        agents: AgentConfigs::default(),
    };

    let prepared = PreparedRun::new(
        CodingAgent::Cursor,
        vec!["cursor-agent".into()],
        "http://s",
        &resolved,
        false,
    )
    .unwrap();

    assert!(
        std::fs::read_to_string(temp.path().join(".cursor/hooks.json"))
            .unwrap()
            .contains("hook-forward cursor")
    );
    assert!(!Path::new(".cursor/hooks.json").exists());
    prepared.restore().unwrap();
    std::env::set_current_dir(previous).unwrap();
}

#[test]
fn cursor_patch_restore_removes_temporary_file() {
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let previous = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp.path()).unwrap();
    let resolved = ResolvedConfig {
        sidecar: SidecarConfig::default(),
        agents: AgentConfigs::default(),
    };

    let prepared = PreparedRun::new(
        CodingAgent::Cursor,
        vec!["cursor-agent".into()],
        "http://s",
        &resolved,
        false,
    )
    .unwrap();
    assert!(Path::new(".cursor/hooks.json").exists());
    prepared.restore().unwrap();
    assert!(!Path::new(".cursor/hooks.json").exists());
    std::env::set_current_dir(previous).unwrap();
}

#[test]
fn cursor_dry_run_does_not_write_hooks() {
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let previous = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp.path()).unwrap();
    let resolved = ResolvedConfig {
        sidecar: SidecarConfig::default(),
        agents: AgentConfigs::default(),
    };

    let prepared = PreparedRun::new(
        CodingAgent::Cursor,
        vec!["cursor-agent".into()],
        "http://s",
        &resolved,
        true,
    )
    .unwrap();

    assert!(!Path::new(".cursor/hooks.json").exists());
    assert!(prepared.notes[0].contains("would temporarily merge"));
    std::env::set_current_dir(previous).unwrap();
}

#[tokio::test]
async fn run_starts_sidecar_injects_env_and_returns_agent_exit_code() {
    let temp = tempfile::tempdir().unwrap();
    let script = temp.path().join("fake-agent.sh");
    let output = temp.path().join("env.txt");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nprintf '%s' \"$NEMO_FLOW_SIDECAR_URL\" > {}\nexit 7\n",
            output.display()
        ),
    )
    .unwrap();
    make_executable(&script);
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        atif_dir: None,
        openinference_endpoint: None,
        session_metadata: None,
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec![script.display().to_string()],
    };

    let code = run(command, None).await.unwrap();

    assert_eq!(code, ExitCode::from(7));
    let url = std::fs::read_to_string(output).unwrap();
    assert!(url.starts_with("http://127.0.0.1:"));
    assert!(!url.ends_with(":0"));
}

#[tokio::test]
async fn dry_run_does_not_spawn_agent() {
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        atif_dir: None,
        openinference_endpoint: None,
        session_metadata: None,
        plugin_config: None,
        dry_run: true,
        print: false,
        command: vec!["/path/that/does/not/exist".into()],
    };

    let code = run(command, None).await.unwrap();

    assert_eq!(code, ExitCode::SUCCESS);
}

#[cfg(unix)]
fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).unwrap();
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) {}
