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
fn missing_configured_command_has_actionable_messages() {
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
        command: vec![],
    };

    let error = resolve_agent_and_argv(&command, &AgentConfigs::default())
        .unwrap_err()
        .to_string();

    assert!(error.contains("missing command"));

    let command = RunCommand {
        agent: Some(CodingAgent::Cursor),
        ..command
    };
    let error = resolve_agent_and_argv(&command, &AgentConfigs::default())
        .unwrap_err()
        .to_string();

    assert!(error.contains("no configured command for cursor"));
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

    assert!(prepared.argv.contains(&"features.hooks=true".into()));
    assert!(
        prepared
            .argv
            .iter()
            .any(|arg| arg == "model_provider=\"nemo-flow-openai\"")
    );
    assert!(
        prepared
            .argv
            .iter()
            .any(|arg| arg.contains("model_providers.nemo-flow-openai")
                && arg.contains("base_url=\"http://127.0.0.1:1234\"")
                && arg.contains("requires_openai_auth=true")
                && arg.contains("supports_websockets=false"))
    );
    assert!(
        !prepared
            .argv
            .iter()
            .any(|arg| arg.contains("model_providers.openai"))
    );
    assert!(
        prepared
            .argv
            .iter()
            .any(|arg| arg.contains("hooks.SessionStart"))
    );
}

#[test]
fn prepares_claude_dry_run_without_writing_plugin() {
    let resolved = ResolvedConfig {
        sidecar: SidecarConfig::default(),
        agents: AgentConfigs::default(),
    };
    let prepared = PreparedRun::new(
        CodingAgent::ClaudeCode,
        vec!["claude".into()],
        "http://127.0.0.1:1234",
        &resolved,
        true,
    )
    .unwrap();

    assert_eq!(prepared.argv[1], "--plugin-dir");
    assert_eq!(prepared.argv[2], "<temporary-claude-plugin-dir>");
    assert!(
        prepared
            .env
            .contains(&("ANTHROPIC_BASE_URL".into(), "http://127.0.0.1:1234".into()))
    );
    assert!(prepared.notes[0].contains("would generate"));
}

#[test]
fn cursor_patching_can_be_disabled() {
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let previous = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp.path()).unwrap();
    let resolved = ResolvedConfig {
        sidecar: SidecarConfig::default(),
        agents: AgentConfigs {
            cursor: CursorAgentConfig {
                command: None,
                patch_restore_hooks: false,
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

    assert!(prepared.cursor_restore.is_none());
    assert!(!Path::new(".cursor/hooks.json").exists());
    std::env::set_current_dir(previous).unwrap();
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
fn cursor_restore_reports_failed_backup_restore() {
    let temp = tempfile::tempdir().unwrap();
    let prepared = PreparedRun {
        argv: vec![],
        env: vec![],
        temp_dirs: vec![],
        cursor_restore: Some(CursorRestore {
            path: temp.path().join("hooks.json"),
            backup_path: Some(temp.path().join("missing-backup.json")),
            had_original: true,
        }),
        notes: vec![],
    };

    let error = prepared.restore().unwrap_err().to_string();

    assert!(error.contains("failed to restore Cursor hooks"));
}

#[test]
fn cursor_restore_reports_failed_temporary_hook_removal() {
    let temp = tempfile::tempdir().unwrap();
    let hooks_path = temp.path().join("hooks.json");
    std::fs::create_dir(&hooks_path).unwrap();
    let prepared = PreparedRun {
        argv: vec![],
        env: vec![],
        temp_dirs: vec![],
        cursor_restore: Some(CursorRestore {
            path: hooks_path,
            backup_path: None,
            had_original: false,
        }),
        notes: vec![],
    };

    let error = prepared.restore().unwrap_err().to_string();

    assert!(error.contains("failed to remove temporary Cursor hooks"));
}

#[test]
fn cursor_restore_noops_when_original_was_declared_without_backup() {
    let prepared = PreparedRun {
        argv: vec![],
        env: vec![],
        temp_dirs: vec![],
        cursor_restore: Some(CursorRestore {
            path: PathBuf::from("unused"),
            backup_path: None,
            had_original: true,
        }),
        notes: vec![],
    };

    prepared.restore().unwrap();
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
    let output = temp.path().join("env.txt");
    let command_argv = fake_agent_command(temp.path(), &output);
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
        command: command_argv,
    };

    let code = run(command, None).await.unwrap();

    assert_eq!(code, ExitCode::from(7));
    let url = std::fs::read_to_string(output).unwrap();
    assert!(url.starts_with("http://127.0.0.1:"));
    assert!(!url.ends_with(":0"));
}

#[cfg(unix)]
fn fake_agent_command(temp: &Path, output: &Path) -> Vec<String> {
    let script = temp.join("fake-agent.sh");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nprintf '%s' \"$NEMO_FLOW_SIDECAR_URL\" > \"{}\"\nexit 7\n",
            output.display()
        ),
    )
    .unwrap();
    make_executable(&script);
    vec![script.display().to_string()]
}

#[cfg(windows)]
fn fake_agent_command(temp: &Path, output: &Path) -> Vec<String> {
    let script = temp.join("fake-agent.cmd");
    std::fs::write(
        &script,
        format!(
            "@echo off\r\n<nul set /p dummy=%NEMO_FLOW_SIDECAR_URL% > \"{}\"\r\nexit /b 7\r\n",
            output.display()
        ),
    )
    .unwrap();
    vec!["cmd.exe".into(), "/C".into(), script.display().to_string()]
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

#[tokio::test]
async fn wait_for_health_reports_unready_sidecar() {
    let error = wait_for_health("http://127.0.0.1:1")
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("sidecar did not become ready"));
}

#[cfg(unix)]
fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).unwrap();
}
