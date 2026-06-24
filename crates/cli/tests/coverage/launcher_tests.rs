// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::config::{AgentCommandConfig, CursorAgentConfig, GatewayConfig};
use std::ffi::OsString;
use std::sync::{Mutex, OnceLock};

fn current_dir_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvScope {
    _guard: std::sync::MutexGuard<'static, ()>,
    values: Vec<(&'static str, Option<OsString>)>,
}

impl EnvScope {
    fn set(values: &[(&'static str, Option<&std::ffi::OsStr>)]) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous = values
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect::<Vec<_>>();
        for (key, value) in values {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
        Self {
            _guard: guard,
            values: previous,
        }
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for (key, value) in self.values.drain(..) {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
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
    // Bare `nemo-relay run` (no command, no --agent) errors — we have nothing to spawn and no
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
    // The easy-path uses this code path: `nemo-relay codex -- --model X` resolves to the
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
fn default_and_configured_command_helpers_cover_empty_and_all_agents() {
    assert_eq!(default_command_for(CodingAgent::ClaudeCode), "claude");
    assert_eq!(default_command_for(CodingAgent::Codex), "codex");
    assert_eq!(default_command_for(CodingAgent::Cursor), "cursor-agent");
    assert_eq!(default_command_for(CodingAgent::Hermes), "hermes");

    let agents = AgentConfigs {
        codex: AgentCommandConfig {
            command: Some("   ".into()),
            hooks_path: None,
        },
        ..AgentConfigs::default()
    };
    assert!(configured_command(CodingAgent::Codex, &agents).is_none());
}

#[test]
fn prepares_codex_config_overrides() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
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
            .any(|arg| arg == "model_provider=\"nemo-relay-openai\"")
    );
    assert!(
        prepared
            .argv
            .iter()
            .any(|arg| arg.contains("model_providers.nemo-relay-openai")
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
fn prepares_codex_with_hooks_when_auth_missing() {
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::set(&[
        ("OPENAI_API_KEY", None),
        ("HOME", Some(temp.path().as_os_str())),
        ("USERPROFILE", None),
    ]);
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };

    let prepared = PreparedRun::new(
        CodingAgent::Codex,
        vec!["codex".into()],
        "http://127.0.0.1:1234",
        &resolved,
        false,
    )
    .unwrap();

    assert!(prepared.argv.iter().any(|arg| arg == "features.hooks=true"));
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
fn exporter_destinations_cover_invalid_disabled_and_missing_plugin_configs() {
    let invalid_plugin = GatewayConfig {
        plugin_config: Some(json!({"components": "not-a-list"})),
        ..GatewayConfig::default()
    };
    assert_eq!(
        exporter_destinations(&invalid_plugin),
        vec!["configured (invalid plugin config)".to_string()]
    );

    let disabled_observability = GatewayConfig {
        plugin_config: Some(json!({
            "version": 1,
            "components": [{
                "kind": OBSERVABILITY_PLUGIN_KIND,
                "enabled": false,
                "config": {"version": 1}
            }]
        })),
        ..GatewayConfig::default()
    };
    assert!(exporter_destinations(&disabled_observability).is_empty());

    let invalid_observability = GatewayConfig {
        plugin_config: Some(json!({
            "version": 1,
            "components": [{
                "kind": OBSERVABILITY_PLUGIN_KIND,
                "enabled": true,
                "config": {"version": "bad"}
            }]
        })),
        ..GatewayConfig::default()
    };
    assert_eq!(
        exporter_destinations(&invalid_observability),
        vec!["Observability configured (invalid config)".to_string()]
    );

    assert!(exporter_destinations(&GatewayConfig::default()).is_empty());
}

#[test]
fn insert_after_agent_uses_last_matching_agent_or_first_word_fallback() {
    let mut argv = vec![
        "wrapper".to_string(),
        "codex".to_string(),
        "subcommand".to_string(),
        "/usr/local/bin/codex".to_string(),
    ];
    insert_after_agent(&mut argv, CodingAgent::Codex, ["--config".to_string()]);
    assert_eq!(
        argv,
        vec![
            "wrapper",
            "codex",
            "subcommand",
            "/usr/local/bin/codex",
            "--config"
        ]
    );

    let mut wrapped = vec!["agent-wrapper".to_string(), "run".to_string()];
    insert_after_agent(&mut wrapped, CodingAgent::Hermes, ["--hook".to_string()]);
    assert_eq!(wrapped, vec!["agent-wrapper", "--hook", "run"]);
}

#[test]
fn prepares_claude_dry_run_without_writing_plugin() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
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
fn prepares_claude_dry_inserts_plugin_dir_after_last_agent_executable() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };
    let prepared = PreparedRun::new(
        CodingAgent::ClaudeCode,
        vec![
            "wrapper".into(),
            "claude".into(),
            "subcommand".into(),
            "/opt/bin/claude".into(),
            "--resume".into(),
        ],
        "http://127.0.0.1:1234",
        &resolved,
        true,
    )
    .unwrap();

    let plugin_index = prepared
        .argv
        .iter()
        .position(|arg| arg == "--plugin-dir")
        .expect("plugin dir arg");
    assert_eq!(prepared.argv[plugin_index - 1], "/opt/bin/claude");
    assert_eq!(
        prepared.argv[plugin_index + 1],
        "<temporary-claude-plugin-dir>"
    );
    assert_eq!(prepared.argv.last().map(String::as_str), Some("--resume"));
    assert!(prepared.temp_dirs.is_empty());
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
        ..ResolvedConfig::default()
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
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let previous = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp.path()).unwrap();
    let hooks_path = temp.path().join("hermes-home/config.yaml");
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs {
            hermes: AgentCommandConfig {
                command: None,
                hooks_path: Some(hooks_path.clone()),
            },
            ..AgentConfigs::default()
        },
        ..ResolvedConfig::default()
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
        "NEMO_RELAY_GATEWAY_URL".into(),
        "http://127.0.0.1:1234".into()
    )));
    assert!(
        prepared
            .env
            .contains(&("HERMES_ACCEPT_HOOKS".into(), "1".into()))
    );
    assert_eq!(
        prepared
            .hermes_restore
            .as_ref()
            .map(|restore| &restore.path),
        Some(&hooks_path)
    );
    let hooks = std::fs::read_to_string(&hooks_path).unwrap();
    assert!(hooks.contains("hook-forward hermes"));
    assert!(prepared.notes[0].contains("temporarily merged"));

    prepared.restore().unwrap();
    assert!(!hooks_path.exists());
    std::env::set_current_dir(previous).unwrap();
}

#[test]
fn prepares_hermes_dry_uses_home_path_without_writing_hooks() {
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::set(&[
        ("HERMES_HOME", None),
        ("HOME", Some(temp.path().as_os_str())),
        ("USERPROFILE", None),
    ]);
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };

    let prepared = PreparedRun::new(
        CodingAgent::Hermes,
        vec!["hermes".into()],
        "http://127.0.0.1:1234",
        &resolved,
        true,
    )
    .unwrap();

    let hook_path = temp.path().join(".hermes/config.yaml");
    assert!(prepared.notes[0].contains(".hermes"));
    assert!(prepared.notes[0].contains("config.yaml"));
    assert!(
        prepared
            .env
            .contains(&("HERMES_ACCEPT_HOOKS".into(), "1".into()))
    );
    assert!(!hook_path.exists());
}

#[test]
fn hermes_hooks_path_prefers_configured_then_env_then_home() {
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let configured = temp.path().join("configured.yaml");
    assert_eq!(hermes_hooks_path(Some(&configured)).unwrap(), configured);

    let _env = EnvScope::set(&[
        ("HERMES_HOME", Some(temp.path().as_os_str())),
        ("HOME", None),
        ("USERPROFILE", None),
    ]);
    assert_eq!(
        hermes_hooks_path(None).unwrap(),
        temp.path().join("config.yaml")
    );

    drop(_env);
    let _env = EnvScope::set(&[
        ("HERMES_HOME", None),
        ("HOME", Some(temp.path().as_os_str())),
        ("USERPROFILE", None),
    ]);
    assert_eq!(
        hermes_hooks_path(None).unwrap(),
        temp.path().join(".hermes/config.yaml")
    );

    drop(_env);
    let _env = EnvScope::set(&[("HERMES_HOME", None), ("HOME", None), ("USERPROFILE", None)]);
    let error = hermes_hooks_path(None).unwrap_err().to_string();
    assert!(error.contains("could not resolve home directory"));
}

#[test]
fn hermes_patch_restore_restores_original_file() {
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let previous = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp.path()).unwrap();
    let hooks_path = temp.path().join("hermes-home/config.yaml");
    std::fs::create_dir_all(hooks_path.parent().unwrap()).unwrap();
    let original = "hooks:\n  PreToolUse: []\n";
    std::fs::write(&hooks_path, original).unwrap();
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs {
            hermes: AgentCommandConfig {
                command: None,
                hooks_path: Some(hooks_path.clone()),
            },
            ..AgentConfigs::default()
        },
        ..ResolvedConfig::default()
    };

    let prepared = PreparedRun::new(
        CodingAgent::Hermes,
        vec!["hermes".into(), "chat".into()],
        "http://s",
        &resolved,
        false,
    )
    .unwrap();

    assert!(
        std::fs::read_to_string(&hooks_path)
            .unwrap()
            .contains("hook-forward hermes")
    );
    prepared.restore().unwrap();
    assert_eq!(std::fs::read_to_string(&hooks_path).unwrap(), original);
    std::env::set_current_dir(previous).unwrap();
}

#[test]
fn prepares_claude_temp_plugin() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
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
        ..ResolvedConfig::default()
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
    let patched: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(".cursor/hooks.json").unwrap()).unwrap();
    assert_eq!(patched["version"], json!(1));
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
        ..ResolvedConfig::default()
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
    let patched: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(temp.path().join(".cursor/hooks.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(patched["version"], json!(1));
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
        ..ResolvedConfig::default()
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
    let patched: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(".cursor/hooks.json").unwrap()).unwrap();
    assert_eq!(patched["version"], json!(1));
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
        hermes_restore: None,
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
        hermes_restore: None,
        notes: vec![],
    };

    let error = prepared.restore().unwrap_err().to_string();

    assert!(error.contains("failed to remove temporary Cursor hooks"));
}

#[test]
fn hermes_restore_reports_restore_and_temporary_removal_failures() {
    let temp = tempfile::tempdir().unwrap();
    let restore_missing_backup = PreparedRun {
        argv: vec![],
        env: vec![],
        temp_dirs: vec![],
        cursor_restore: None,
        hermes_restore: Some(HermesRestore {
            path: temp.path().join("config.yaml"),
            backup_path: Some(temp.path().join("missing-backup.yaml")),
            had_original: true,
        }),
        notes: vec![],
    };

    let error = restore_missing_backup.restore().unwrap_err().to_string();
    assert!(error.contains("failed to restore Hermes hooks"));

    let hooks_path = temp.path().join("hooks-dir");
    std::fs::create_dir(&hooks_path).unwrap();
    let remove_temporary_dir = PreparedRun {
        argv: vec![],
        env: vec![],
        temp_dirs: vec![],
        cursor_restore: None,
        hermes_restore: Some(HermesRestore {
            path: hooks_path,
            backup_path: None,
            had_original: false,
        }),
        notes: vec![],
    };

    let error = remove_temporary_dir.restore().unwrap_err().to_string();
    assert!(error.contains("failed to remove temporary Hermes hooks"));
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
        hermes_restore: None,
        notes: vec![],
    };

    prepared.restore().unwrap();
}

#[test]
fn hook_backup_and_write_helpers_cover_missing_existing_and_toml_escaping() {
    let temp = tempfile::tempdir().unwrap();
    let missing_cursor = temp.path().join("missing-hooks.json");
    assert_eq!(
        backup_existing_cursor_hooks(&missing_cursor).unwrap(),
        (false, None)
    );

    let cursor_hooks = temp.path().join("hooks.json");
    std::fs::write(&cursor_hooks, "{}").unwrap();
    let (had_original, cursor_backup) = backup_existing_cursor_hooks(&cursor_hooks).unwrap();
    assert!(had_original);
    assert!(cursor_backup.as_ref().unwrap().exists());

    let missing_hermes = temp.path().join("missing-config.yaml");
    assert_eq!(
        backup_existing_hermes_hooks(&missing_hermes).unwrap(),
        (false, None)
    );

    let hermes_hooks = temp.path().join("config.yaml");
    std::fs::write(&hermes_hooks, "hooks: {}\n").unwrap();
    let (had_original, hermes_backup) = backup_existing_hermes_hooks(&hermes_hooks).unwrap();
    assert!(had_original);
    assert!(hermes_backup.as_ref().unwrap().exists());

    let written_hooks = temp.path().join("written/hooks.json");
    std::fs::create_dir_all(written_hooks.parent().unwrap()).unwrap();
    write_hooks(&written_hooks, json!({"hooks": []})).unwrap();
    assert!(
        std::fs::read_to_string(&written_hooks)
            .unwrap()
            .contains("hooks")
    );

    let groups = hook_groups_toml(&json!([{
        "matcher": "Shell\"Run",
        "hooks": [{"command": "nemo-relay \"quoted\""}]
    }]));
    assert!(groups.contains("matcher=\"Shell\\\"Run\""));
    assert!(groups.contains("command=\"nemo-relay \\\"quoted\\\"\""));

    let escaped = toml_string(r#"C:\tmp\"quoted""#);
    assert!(escaped.starts_with('"'));
    assert!(escaped.ends_with('"'));
    assert!(escaped.contains(r#"C:\\tmp\\"#));
    assert!(escaped.contains(r#"\"quoted\""#));
}

#[cfg(unix)]
#[test]
fn exit_code_preserves_normal_and_shell_wrapped_codes() {
    let status = std::process::Command::new("/bin/sh")
        .args(["-c", "exit 7"])
        .status()
        .unwrap();
    assert_eq!(exit_code(status), ExitCode::from(7));

    let status = std::process::Command::new("/bin/sh")
        .args(["-c", "exit 300"])
        .status()
        .unwrap();
    assert_eq!(exit_code(status), ExitCode::from(44));
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
        ..ResolvedConfig::default()
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
    let config = temp.path().join("config.toml");
    std::fs::write(&config, "[upstream]\n").unwrap();
    let output = temp.path().join("env.txt");
    let command_argv = fake_agent_command(temp.path(), &output);
    let command = RunCommand {
        // Leave `agent: None` so the launcher infers from argv[0] and uses `command_argv`
        // (our fake-agent.sh) as the full argv. With --agent set, the resolver appends
        // command as pass-through after the configured/default binary — not what this test
        // wants, since it specifically asserts that argv[0] is the fake script.
        agent: None,
        config: Some(config),
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
            "#!/bin/sh\nprintf '%s' \"$NEMO_RELAY_GATEWAY_URL\" > \"{}\"\nexit 7\n",
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

#[tokio::test]
async fn execute_live_run_reports_gateway_startup_error_when_health_check_fails() {
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
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gateway_url = format!("http://{}", listener.local_addr().unwrap());
    let gateway_config = GatewayConfig {
        plugin_config: Some(json!({
            "version": 1,
            "components": [{
                "kind": OBSERVABILITY_PLUGIN_KIND,
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "mode": "invalid"
                    }
                }
            }]
        })),
        ..GatewayConfig::default()
    };

    let error = execute_live_run(listener, gateway_config, &gateway_url, prepared)
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("ATOF mode"));
    assert!(!error.contains("gateway did not become ready"));
}

#[tokio::test]
async fn execute_live_run_restores_hermes_hooks_when_health_check_fails() {
    let temp = tempfile::tempdir().unwrap();
    let hooks_path = temp.path().join("hermes-home/config.yaml");
    std::fs::create_dir_all(hooks_path.parent().unwrap()).unwrap();
    let original = "hooks:\n  PreToolUse: []\n";
    std::fs::write(&hooks_path, original).unwrap();
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs {
            hermes: AgentCommandConfig {
                command: None,
                hooks_path: Some(hooks_path.clone()),
            },
            ..AgentConfigs::default()
        },
        ..ResolvedConfig::default()
    };
    let prepared = PreparedRun::new(
        CodingAgent::Hermes,
        vec!["hermes".into(), "chat".into()],
        "http://127.0.0.1:1234",
        &resolved,
        false,
    )
    .unwrap();
    assert!(
        std::fs::read_to_string(&hooks_path)
            .unwrap()
            .contains("hook-forward hermes")
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let error = execute_live_run(
        listener,
        GatewayConfig::default(),
        "http://127.0.0.1:1",
        prepared,
    )
    .await
    .unwrap_err()
    .to_string();

    assert!(error.contains("gateway did not become ready"));
    assert_eq!(std::fs::read_to_string(&hooks_path).unwrap(), original);
}

#[cfg(unix)]
fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).unwrap();
}
