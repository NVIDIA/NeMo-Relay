// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::config::{AgentCommandConfig, CursorAgentConfig, GatewayConfig};
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
            hooks_path: None,
        },
        ..AgentConfigs::default()
    };
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
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
            hooks_path: None,
        },
        ..AgentConfigs::default()
    };
    let command = RunCommand {
        agent: Some(CodingAgent::Hermes),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
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
        session_metadata: None,
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec!["my-agent".into()],
    };

    let error = resolve_agent_and_argv(&command, &AgentConfigs::default())
        .unwrap_err()
        .to_string();

    assert!(error.contains("pass --agent claude"));
}

#[test]
fn missing_command_without_agent_errors() {
    // Bare `nemo-flow run` (no command, no --agent) errors — we have nothing to spawn and no
    // argv[0] to infer an agent from. With --agent set, we fall back to the agent's default
    // binary name (e.g., `cursor-agent`), so that branch is exercised in the resolution test
    // below rather than here.
    let command = RunCommand {
        agent: None,
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
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
}

#[test]
fn agent_without_configured_command_falls_back_to_default_binary() {
    // `--agent cursor` with no `[agents.cursor] command = "..."` override resolves to the
    // default executable name on $PATH (`cursor-agent` for the Cursor agent).
    let command = RunCommand {
        agent: Some(CodingAgent::Cursor),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec![],
    };

    let (agent, argv) = resolve_agent_and_argv(&command, &AgentConfigs::default()).unwrap();
    assert_eq!(agent, CodingAgent::Cursor);
    assert_eq!(argv, vec!["cursor-agent"]);
}

#[test]
fn agent_with_passthrough_args_appends_to_configured_command() {
    // The easy-path uses this code path: `nemo-flow codex -- --model X` resolves to the
    // configured (or default) codex command with `--model X` appended.
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec!["--model".into(), "openai/openai/gpt-5.1-codex".into()],
    };

    let (_, argv) = resolve_agent_and_argv(&command, &AgentConfigs::default()).unwrap();
    assert_eq!(
        argv,
        vec!["codex", "--model", "openai/openai/gpt-5.1-codex"]
    );
}

#[test]
fn prepares_codex_config_overrides() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
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
                // Codex sends its own credentials (ChatGPT-Plus OAuth or OPENAI_API_KEY).
                // When OPENAI_API_KEY is in the environment the gateway substitutes it;
                // otherwise codex's own auth is forwarded as-is.
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
    let path = prepared
        .env
        .iter()
        .find_map(|(name, value)| (name == "PATH").then_some(value))
        .expect("transparent run should set PATH for hook subprocesses");
    let current_exe_dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let entries = std::env::split_paths(path).collect::<Vec<_>>();
    assert!(entries.iter().any(|entry| entry == &current_exe_dir));
    if !std::env::var_os("PATH")
        .as_deref()
        .map(std::env::split_paths)
        .into_iter()
        .flatten()
        .any(|entry| entry == current_exe_dir)
    {
        assert_eq!(entries.last(), Some(&current_exe_dir));
    }
}

#[test]
fn exporter_destinations_describe_observability_outputs() {
    let gateway = GatewayConfig {
        plugin_config: Some(json!({
            "version": 1,
            "components": [{
                "kind": OBSERVABILITY_PLUGIN_KIND,
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "output_directory": "logs",
                        "filename": "events.jsonl"
                    },
                    "atif": {
                        "enabled": true,
                        "output_directory": "trajectories",
                        "filename_template": "agent-{session_id}.json"
                    },
                    "opentelemetry": {
                        "enabled": true,
                        "endpoint": "http://127.0.0.1:4318/v1/traces"
                    },
                    "openinference": {
                        "enabled": true
                    }
                }
            }]
        })),
        ..GatewayConfig::default()
    };

    let destinations = exporter_destinations(&gateway);

    assert!(destinations.iter().any(|line| line
        == &format!(
            "ATOF {}",
            PathBuf::from("logs").join("events.jsonl").display()
        )));
    assert!(destinations.iter().any(|line| line
        == &format!(
            "ATIF {}",
            PathBuf::from("trajectories")
                .join("agent-{session_id}.json")
                .display()
        )));
    assert!(
        destinations
            .iter()
            .any(|line| line == "OpenTelemetry http://127.0.0.1:4318/v1/traces")
    );
    assert!(
        destinations
            .iter()
            .any(|line| line == "OpenInference OTLP endpoint from environment/default")
    );
}

#[test]
fn prepares_claude_dry_run_without_writing_plugin() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
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
        gateway: GatewayConfig::default(),
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
        gateway: GatewayConfig::default(),
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
        "NEMO_FLOW_GATEWAY_URL".into(),
        "http://127.0.0.1:1234".into()
    )));
    assert!(
        !prepared
            .env
            .iter()
            .any(|(name, _)| name == "HERMES_ACCEPT_HOOKS")
    );
    assert!(prepared.notes[0].contains("nemo-flow config hermes"));
}

#[test]
fn prepares_claude_temp_plugin() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
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
        gateway: GatewayConfig::default(),
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
        gateway: GatewayConfig::default(),
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
        gateway: GatewayConfig::default(),
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
        gateway: GatewayConfig::default(),
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

// This e2e test relies on argv[0] being a script literally named after a known agent (so
// `CodingAgent::infer` recognises the basename without an explicit `--agent`). On Windows the
// only practical way to invoke a `.cmd` / `.bat` shim is via `cmd.exe /C script.cmd`, which
// makes argv[0] = `cmd.exe` and breaks inference. Gating Unix-only keeps cross-platform CI
// green; real Windows agent-spawn coverage can come back with a `.exe` fake binary once the
// launcher grows Windows support.
#[cfg(unix)]
#[tokio::test]
async fn run_starts_gateway_injects_env_and_returns_agent_exit_code() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("env.txt");
    let command_argv = fake_agent_command(temp.path(), &output);
    let command = RunCommand {
        // Leave `agent: None` so the launcher infers from argv[0] and uses `command_argv`
        // (our fake-agent.sh) as the full argv. With --agent set, the resolver appends
        // command as pass-through after the configured/default binary — not what this test
        // wants, since it specifically asserts that argv[0] is the fake script.
        agent: None,
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
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
    // Name the script `codex` (not `fake-agent.sh`) so `CodingAgent::infer` recognizes the
    // argv[0] basename without us needing to set `--agent` explicitly. With `--agent` set,
    // the resolver appends `command.command` as pass-through args after the configured/default
    // binary — wrong for this test, which wants the fake script itself to be argv[0].
    let script = temp.join("codex");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nprintf '%s' \"$NEMO_FLOW_GATEWAY_URL\" > \"{}\"\nexit 7\n",
            output.display()
        ),
    )
    .unwrap();
    make_executable(&script);
    vec![script.display().to_string()]
}

#[tokio::test]
async fn dry_run_does_not_spawn_agent() {
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
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
async fn wait_for_health_reports_unready_gateway() {
    let error = wait_for_health("http://127.0.0.1:1")
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("gateway did not become ready"));
}

#[cfg(unix)]
fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).unwrap();
}
