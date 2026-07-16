// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::configuration::{AgentCommandConfig, GatewayConfig};
use crate::hooks::generated_hooks;
use std::ffi::OsString;
use std::sync::Mutex;

fn current_dir_lock() -> &'static Mutex<()> {
    &crate::test_support::CWD_TEST_LOCK
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

    fn without_managed_bootstrap() -> Self {
        Self::set(&[
            (crate::bootstrap::state::BOOTSTRAP_STATE_DIR_ENV, None),
            ("NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN", None),
            (crate::configuration::BOOTSTRAP_FINGERPRINT_ENV, None),
        ])
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
    let command = RunOverrides {
        agent: None,
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: false,
        print: false,
        command: vec!["/usr/bin/codex".into()],
    };
    let (agent, argv) = resolve_agent_and_argv(&command, &AgentConfigs::default()).unwrap();
    assert_eq!(agent, CodingAgent::Codex);
    assert_eq!(argv, vec!["/usr/bin/codex"]);

    let command = RunOverrides {
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
    let command = RunOverrides {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
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
    let command = RunOverrides {
        agent: Some(CodingAgent::Hermes),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
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
    let command = RunOverrides {
        agent: None,
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
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
    // binary name (e.g., `hermes`), so that branch is exercised in the resolution test
    // below rather than here.
    let command = RunOverrides {
        agent: None,
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
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
    // `--agent hermes` with no `[agents.hermes] command = "..."` override resolves to the
    // default executable name on $PATH.
    let command = RunOverrides {
        agent: Some(CodingAgent::Hermes),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: false,
        print: false,
        command: vec![],
    };

    let (agent, argv) = resolve_agent_and_argv(&command, &AgentConfigs::default()).unwrap();
    assert_eq!(agent, CodingAgent::Hermes);
    assert_eq!(argv, vec!["hermes"]);
}

#[test]
fn agent_with_passthrough_args_appends_to_configured_command() {
    // The easy-path uses this code path: `nemo-relay codex -- --model X` resolves to the
    // configured (or default) codex command with `--model X` appended.
    let command = RunOverrides {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
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
    let _guard = current_dir_lock().lock().unwrap();
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };
    let prepared = PreparedAgentLaunch::new(
        CodingAgent::Codex,
        vec!["codex".into()],
        "http://127.0.0.1:1234",
        &resolved,
        false,
    )
    .unwrap();

    assert!(!prepared.argv.iter().any(|arg| arg == "--profile"));
    assert!(prepared.argv.contains(&"features.hooks=true".into()));
    assert!(
        prepared
            .argv
            .contains(&"features.multi_agent_v2.enabled=false".into())
    );
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
                && arg.contains("supports_websockets=false")
                && arg.contains("env_http_headers")
                && arg.contains(crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER)
                && arg.contains(crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_ENV))
    );
    assert!(
        !prepared
            .argv
            .iter()
            .any(|arg| arg.contains(prepared.proxy_credential.expose()))
    );
    assert!(prepared.env.contains(&(
        crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_ENV.into(),
        prepared.proxy_credential.expose().into()
    )));
    assert!(
        prepared
            .secret_env_names
            .contains(&crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_ENV.into())
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
    let trust = prepared
        .argv
        .iter()
        .find(|arg| arg.starts_with("hooks.state={"))
        .unwrap();
    let expected_hooks = generated_hooks(CodingAgent::Codex, "ignored")["hooks"]
        .as_object()
        .unwrap()
        .len();
    assert_eq!(
        trust.matches("trusted_hash=\"sha256:").count(),
        expected_hooks
    );
    assert_eq!(trust.matches("enabled=true").count(), expected_hooks);
    assert!(
        prepared
            .env
            .contains(&(crate::configuration::TRANSPARENT_RUN_ENV.into(), "1".into()))
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
    prepared.restore().unwrap();
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

    let prepared = PreparedAgentLaunch::new(
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
fn codex_session_hook_trust_matches_codex_discovery_identity() {
    let generated = generated_hooks(CodingAgent::Codex, "echo relay-probe");
    let group = generated["hooks"]["UserPromptSubmit"][0]
        .as_object()
        .unwrap();
    let handler = &group["hooks"].as_array().unwrap()[0];
    assert_eq!(
        crate::agents::codex::launch::command_hook_hash("user_prompt_submit", group, handler)
            .unwrap(),
        "sha256:83a9834ee494ffbd4acc85377c579d2c954f9797a9b8832924a326a6a44b0660"
    );

    let state = crate::agents::codex::launch::session_hook_state_override(&generated).unwrap();
    assert_eq!(
        state.matches("trusted_hash=\"sha256:").count(),
        generated["hooks"].as_object().unwrap().len()
    );
    assert!(state.contains("/<session-flags>/config.toml:user_prompt_submit:0:0"));
    assert!(
        state.contains("sha256:83a9834ee494ffbd4acc85377c579d2c954f9797a9b8832924a326a6a44b0660")
    );
    assert_eq!(
        state.matches("enabled=true").count(),
        generated["hooks"].as_object().unwrap().len()
    );
    assert_eq!(
        state.matches("enabled=false").count(),
        generated["hooks"].as_object().unwrap().len() * 2
    );
    assert!(
        state
            .contains("nemo-relay-plugin@nemo-relay-local:hooks/hooks.json:user_prompt_submit:0:0")
    );
    assert!(state.contains("nemo-relay-plugin@nemo-relay:hooks/hooks.json:user_prompt_submit:0:0"));
}

#[test]
fn codex_preserves_profiles_and_prompt_arguments_without_temporary_config() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };
    let prepared = PreparedAgentLaunch::new(
        CodingAgent::Codex,
        vec![
            "codex".into(),
            "--profile".into(),
            "root".into(),
            "exec".into(),
            "--profile=work".into(),
            "ping".into(),
            "--".into(),
            "codex".into(),
        ],
        "http://127.0.0.1:1234",
        &resolved,
        false,
    )
    .unwrap();

    let separator = prepared.argv.iter().position(|arg| arg == "--").unwrap();
    assert_eq!(&prepared.argv[separator..], &["--", "codex"]);
    assert!(
        prepared.argv[..separator]
            .windows(2)
            .any(|pair| pair == ["--profile", "root"])
    );
    assert!(
        prepared.argv[..separator]
            .iter()
            .any(|arg| arg == "--profile=work")
    );
    assert_eq!(
        prepared
            .argv
            .iter()
            .filter(|arg| arg.as_str() == "codex")
            .count(),
        2
    );
    assert!(
        prepared.argv[1..separator]
            .iter()
            .any(|arg| arg == "features.hooks=true")
    );
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
                    "version": 2,
                    "atof": {
                        "enabled": true,
                        "sinks": [
                            {
                                "type": "file",
                                "output_directory": "logs",
                                "filename": "events.jsonl"
                            },
                            {
                                "type": "stream",
                                "url": "https://user:secret@collector.example/atof?token=secret"
                            }
                        ]
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
    assert!(
        destinations
            .iter()
            .any(|line| line == "ATOF https://collector.example/atof?token=%5BREDACTED%5D")
    );
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
fn exporter_destinations_describe_atif_remote_storage_instead_of_local_path() {
    let gateway = GatewayConfig {
        plugin_config: Some(json!({
            "version": 1,
            "components": [{
                "kind": OBSERVABILITY_PLUGIN_KIND,
                "enabled": true,
                "config": {
                    "version": 1,
                    "atif": {
                        "enabled": true,
                        "output_directory": "trajectories",
                        "filename_template": "agent-{session_id}.json",
                        "storage": [
                            {"type": "s3", "bucket": "traj-bucket", "key_prefix": "runs/"},
                            {"type": "http", "endpoint": "https://collector.example/ingest"}
                        ]
                    }
                }
            }]
        })),
        ..GatewayConfig::default()
    };

    let destinations = exporter_destinations(&gateway);

    assert!(
        destinations
            .iter()
            .any(|line| line == "ATIF s3://traj-bucket/runs")
    );
    assert!(
        destinations
            .iter()
            .any(|line| line == "ATIF https://collector.example/ingest")
    );
    // The local path is skipped at runtime when storage is configured, so it must not be reported.
    assert!(
        !destinations
            .iter()
            .any(|line| line.contains("agent-{session_id}.json"))
    );
}

#[test]
fn exporter_destinations_redact_url_credentials_and_query_values() {
    assert_eq!(
        sanitized_url("https://user:secret@example.test/ingest?token=secret&tenant=acme"),
        "https://example.test/ingest?token=%5BREDACTED%5D&tenant=%5BREDACTED%5D"
    );
    assert_eq!(
        sanitized_url("not a url with secret"),
        "configured endpoint"
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
fn insert_after_host_uses_the_authoritative_executable_index() {
    let mut argv = vec![
        "wrapper".to_string(),
        "codex".to_string(),
        "exec".to_string(),
        "--".to_string(),
        "codex".to_string(),
    ];
    crate::process::insert_after_host(&mut argv, 1, ["--config".to_string()]);
    assert_eq!(
        argv,
        vec!["wrapper", "codex", "--config", "exec", "--", "codex"]
    );
}

#[test]
fn invocation_resolves_wrapper_host_before_appending_pass_through_arguments() {
    let agents = AgentConfigs {
        codex: AgentCommandConfig {
            command: Some("wrapper -- codex".into()),
            hooks_path: None,
        },
        ..AgentConfigs::default()
    };
    let command = RunOverrides {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: false,
        print: false,
        command: vec!["exec".into(), "--".into(), "codex".into()],
    };

    let invocation = resolve_agent_invocation(&command, &agents).unwrap();
    assert_eq!(invocation.host_index, 2);
    assert_eq!(
        invocation.argv,
        vec!["wrapper", "--", "codex", "exec", "--", "codex"]
    );
}

#[test]
fn version_probe_preserves_known_wrappers_and_validates_opaque_ones() {
    assert_eq!(
        crate::process::version_probe_argv(CodingAgent::Codex, &["codex".into(), "exec".into()]),
        vec!["codex", "--version"]
    );
    assert_eq!(
        crate::process::version_probe_argv(
            CodingAgent::Codex,
            &["npx".into(), "--yes".into(), "codex".into(), "exec".into(),],
        ),
        vec!["npx", "--yes", "codex", "--version"]
    );
    assert_eq!(
        crate::process::version_probe_argv(
            CodingAgent::Hermes,
            &["company-agent-wrapper".into(), "chat".into()],
        ),
        vec!["company-agent-wrapper", "chat", "--version"]
    );
}

#[cfg(unix)]
#[tokio::test]
async fn wrapped_agent_version_probe_runs_through_the_wrapper() {
    let temp = tempfile::tempdir().unwrap();
    let wrapper = temp.path().join("npx");
    std::fs::write(
        &wrapper,
        "#!/bin/sh\n[ \"$1\" = codex ] && [ \"$2\" = --version ] || exit 9\necho 'codex-cli 0.143.0'\n",
    )
    .unwrap();
    make_executable(&wrapper);
    let probe = crate::process::version_probe_argv(
        CodingAgent::Codex,
        &[wrapper.display().to_string(), "codex".into(), "exec".into()],
    );

    validate_agent_version(CodingAgent::Codex, &probe)
        .await
        .unwrap();
}

#[test]
fn prepares_claude_dry_run_without_writing_plugin() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };
    let prepared = PreparedAgentLaunch::new(
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
    let custom_headers = prepared
        .env
        .iter()
        .find_map(|(name, value)| (name == "ANTHROPIC_CUSTOM_HEADERS").then_some(value))
        .unwrap();
    assert!(custom_headers.starts_with(&format!(
        "{}: ",
        crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER
    )));
    assert!(custom_headers.ends_with(prepared.proxy_credential.expose()));
    assert!(
        prepared
            .secret_env_names
            .contains(&"ANTHROPIC_CUSTOM_HEADERS".into())
    );
}

#[test]
fn claude_transparent_proxy_header_preserves_existing_custom_headers() {
    let _env = EnvScope::set(&[(
        "ANTHROPIC_CUSTOM_HEADERS",
        Some(std::ffi::OsStr::new("x-existing: preserved")),
    )]);
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };

    let prepared = PreparedAgentLaunch::new(
        CodingAgent::ClaudeCode,
        vec!["claude".into()],
        "http://127.0.0.1:1234",
        &resolved,
        true,
    )
    .unwrap();
    let custom_headers = prepared
        .env
        .iter()
        .find_map(|(name, value)| (name == "ANTHROPIC_CUSTOM_HEADERS").then_some(value))
        .unwrap();

    assert_eq!(
        custom_headers,
        &format!(
            "x-existing: preserved\n{}: {}",
            crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER,
            prepared.proxy_credential.expose()
        )
    );
}

#[test]
fn claude_transparent_proxy_header_replaces_case_insensitive_existing_entries() {
    let _env = EnvScope::set(&[(
        "ANTHROPIC_CUSTOM_HEADERS",
        Some(std::ffi::OsStr::new(
            "X-NEMO-RELAY-PROXY-TOKEN: stale-first\nx-existing: preserved\nx-NeMo-ReLaY-PrOxY-ToKeN : stale-second",
        )),
    )]);
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };

    let prepared = PreparedAgentLaunch::new(
        CodingAgent::ClaudeCode,
        vec!["claude".into()],
        "http://127.0.0.1:1234",
        &resolved,
        true,
    )
    .unwrap();
    let custom_headers = prepared
        .env
        .iter()
        .find_map(|(name, value)| (name == "ANTHROPIC_CUSTOM_HEADERS").then_some(value))
        .unwrap();

    assert_eq!(
        custom_headers,
        &format!(
            "x-existing: preserved\n{}: {}",
            crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER,
            prepared.proxy_credential.expose()
        )
    );
    assert_eq!(
        custom_headers
            .lines()
            .filter(|line| line.split_once(':').is_some_and(|(name, _)| {
                name.trim()
                    .eq_ignore_ascii_case(crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER)
            }))
            .count(),
        1
    );
}

#[test]
fn prepares_claude_dry_inserts_plugin_dir_after_authoritative_agent_executable() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };
    let prepared = PreparedAgentLaunch::new(
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
fn prepares_hermes_hook_environment() {
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let hooks_path = temp.path().join("hermes-home/config.yaml");
    std::fs::create_dir_all(hooks_path.parent().unwrap()).unwrap();
    std::fs::write(&hooks_path, "model:\n  default: test\n").unwrap();
    let state = hooks_path.parent().unwrap().join("state.db");
    std::fs::write(&state, "state").unwrap();
    let cache = hooks_path.parent().unwrap().join("cache");
    std::fs::create_dir(&cache).unwrap();
    std::fs::write(cache.join("entry"), "cached").unwrap();
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs {
            hermes: AgentCommandConfig {
                command: None,
                hooks_path: Some(hooks_path.clone()),
            },
            ..AgentConfigs::default()
        },
        dynamic_plugins: Vec::new(),
        ..ResolvedConfig::default()
    };
    let prepared = PreparedAgentLaunch::new(
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
    let overlay = prepared
        .env
        .iter()
        .find_map(|(name, value)| (name == "HERMES_HOME").then(|| PathBuf::from(value)))
        .expect("Hermes overlay path");
    let hooks = std::fs::read_to_string(overlay.join("config.yaml")).unwrap();
    let hooks: serde_json::Value = serde_yaml::from_str(&hooks).unwrap();
    assert_eq!(hooks["model"]["provider"], json!("custom"));
    assert_eq!(
        hooks["model"]["api_key"],
        json!(format!(
            "${{{}}}",
            crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_ENV
        ))
    );
    assert!(crate::hook_assertions::value_has_command_arguments(
        &hooks,
        &[
            "hook-forward",
            "hermes",
            "--gateway-url",
            "http://127.0.0.1:1234",
            "--transparent-run",
        ],
    ));
    assert!(overlay.join("state.db").exists());
    assert_eq!(
        std::fs::read_to_string(overlay.join("cache/entry")).unwrap(),
        "cached"
    );
    std::fs::write(overlay.join("cache/through-overlay"), "live").unwrap();
    assert_eq!(
        std::fs::read_to_string(cache.join("through-overlay")).unwrap(),
        "live"
    );
    assert_eq!(
        std::fs::read_to_string(&hooks_path).unwrap(),
        "model:\n  default: test\n"
    );
    assert!(prepared.notes[0].contains("isolated Hermes config overlay"));

    prepared.restore().unwrap();
    assert!(hooks_path.exists());
    assert!(!overlay.exists());
}

#[cfg(unix)]
#[test]
fn process_private_directories_are_owner_only() {
    use std::os::unix::fs::PermissionsExt;

    let parent = tempfile::tempdir().unwrap();
    let path = crate::filesystem::temp::private_temp_dir(parent.path(), "relay-private").unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o700
    );
    std::fs::remove_dir(path).unwrap();
}

#[test]
fn concurrent_hermes_runs_use_independent_overlays_without_mutating_user_config() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("hermes/config.yaml");
    std::fs::create_dir_all(config.parent().unwrap()).unwrap();
    let original = "model:\n  default: test\n";
    std::fs::write(&config, original).unwrap();
    let resolved = ResolvedConfig {
        agents: AgentConfigs {
            hermes: AgentCommandConfig {
                hooks_path: Some(config.clone()),
                ..AgentCommandConfig::default()
            },
            ..AgentConfigs::default()
        },
        ..ResolvedConfig::default()
    };

    let resolved = std::sync::Arc::new(resolved);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let spawn = |url: &'static str| {
        let resolved = resolved.clone();
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            barrier.wait();
            PreparedAgentLaunch::new(
                CodingAgent::Hermes,
                vec!["hermes".into()],
                url,
                &resolved,
                false,
            )
            .unwrap()
        })
    };
    let first = spawn("http://127.0.0.1:4001");
    let second = spawn("http://127.0.0.1:4002");
    barrier.wait();
    let first = first.join().unwrap();
    let second = second.join().unwrap();
    let overlay = |run: &PreparedAgentLaunch| {
        run.env
            .iter()
            .find_map(|(name, value)| (name == "HERMES_HOME").then(|| PathBuf::from(value)))
            .unwrap()
    };
    let first_overlay = overlay(&first);
    let second_overlay = overlay(&second);

    assert_ne!(first_overlay, second_overlay);
    let first_config: serde_json::Value =
        serde_yaml::from_str(&std::fs::read_to_string(first_overlay.join("config.yaml")).unwrap())
            .unwrap();
    let second_config: serde_json::Value =
        serde_yaml::from_str(&std::fs::read_to_string(second_overlay.join("config.yaml")).unwrap())
            .unwrap();
    assert!(crate::hook_assertions::value_has_command_arguments(
        &first_config,
        &[
            "hook-forward",
            "hermes",
            "--gateway-url",
            "http://127.0.0.1:4001",
            "--transparent-run",
        ],
    ));
    assert!(crate::hook_assertions::value_has_command_arguments(
        &second_config,
        &[
            "hook-forward",
            "hermes",
            "--gateway-url",
            "http://127.0.0.1:4002",
            "--transparent-run",
        ],
    ));
    assert_eq!(std::fs::read_to_string(&config).unwrap(), original);

    first.restore().unwrap();
    assert!(!first_overlay.exists());
    assert!(second_overlay.exists());
    assert_eq!(std::fs::read_to_string(&config).unwrap(), original);
    second.restore().unwrap();
}

#[test]
fn hermes_overlay_does_not_link_an_ancestor_entry_that_contains_it() {
    let source_home = tempfile::tempdir().unwrap();
    let source_config = source_home.path().join("config.yaml");
    std::fs::write(&source_config, "model:\n  default: test\n").unwrap();
    let overlay = source_home.path().join("overlay");
    std::fs::create_dir(&overlay).unwrap();

    crate::agents::hermes::launch::populate_overlay(
        &overlay,
        source_home.path(),
        &source_config,
        "http://127.0.0.1:1234",
    )
    .unwrap();

    assert!(!overlay.join("overlay").exists());
    assert!(overlay.join("config.yaml").exists());
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

    let prepared = PreparedAgentLaunch::new(
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
    assert_eq!(
        crate::agents::hermes::launch::hooks_path_for_launch(Some(&configured)).unwrap(),
        configured
    );

    let _env = EnvScope::set(&[
        ("HERMES_HOME", Some(temp.path().as_os_str())),
        ("HOME", None),
        ("USERPROFILE", None),
    ]);
    assert_eq!(
        crate::agents::hermes::launch::hooks_path_for_launch(None).unwrap(),
        temp.path().join("config.yaml")
    );

    drop(_env);
    let _env = EnvScope::set(&[
        ("HERMES_HOME", None),
        ("HOME", Some(temp.path().as_os_str())),
        ("USERPROFILE", None),
    ]);
    assert_eq!(
        crate::agents::hermes::launch::hooks_path_for_launch(None).unwrap(),
        temp.path().join(".hermes/config.yaml")
    );

    drop(_env);
    let _env = EnvScope::set(&[("HERMES_HOME", None), ("HOME", None), ("USERPROFILE", None)]);
    let error = crate::agents::hermes::launch::hooks_path_for_launch(None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("could not resolve home directory"));
}

#[test]
fn hermes_overlay_preserves_ambiguous_manual_mcp_and_original_file() {
    let _guard = current_dir_lock().lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let hooks_path = temp.path().join("hermes-home/config.yaml");
    std::fs::create_dir_all(hooks_path.parent().unwrap()).unwrap();
    let original = r#"mcp_servers:
  nemo-relay:
    command: nemo-relay
    args: [mcp, --agent, hermes]
  filesystem:
    command: fs-mcp
hooks:
  PreToolUse: []
"#;
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

    let prepared = PreparedAgentLaunch::new(
        CodingAgent::Hermes,
        vec!["hermes".into(), "chat".into()],
        "http://s",
        &resolved,
        false,
    )
    .unwrap();

    let overlay = prepared
        .env
        .iter()
        .find_map(|(name, value)| (name == "HERMES_HOME").then(|| PathBuf::from(value)))
        .unwrap();
    let patched = std::fs::read_to_string(overlay.join("config.yaml")).unwrap();
    let patched_yaml: serde_json::Value = serde_yaml::from_str(&patched).unwrap();
    assert!(crate::hook_assertions::value_has_command_arguments(
        &patched_yaml,
        &[
            "hook-forward",
            "hermes",
            "--gateway-url",
            "http://s",
            "--transparent-run",
        ],
    ));
    assert_eq!(
        patched_yaml["mcp_servers"]["nemo-relay"]["args"],
        json!(["mcp", "--agent", "hermes"])
    );
    assert_eq!(
        patched_yaml["mcp_servers"]["filesystem"]["command"],
        json!("fs-mcp")
    );
    assert_eq!(std::fs::read_to_string(&hooks_path).unwrap(), original);
    prepared.restore().unwrap();
    assert!(!overlay.exists());
}

#[test]
fn prepares_claude_temp_plugin() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };
    let prepared = PreparedAgentLaunch::new(
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
    assert_eq!(prepared.argv[plugin_index + 2], "--settings");
    let settings_path = PathBuf::from(&prepared.argv[plugin_index + 3]);
    let settings: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&settings_path).unwrap()).unwrap();
    assert_eq!(
        settings["env"]["ANTHROPIC_BASE_URL"],
        "http://127.0.0.1:1234"
    );
    let hooks: serde_json::Value =
        serde_json::from_slice(&std::fs::read(plugin_dir.join("hooks/hooks.json")).unwrap())
            .unwrap();
    assert!(crate::hook_assertions::value_has_command_arguments(
        &hooks,
        &[
            "hook-forward",
            "claude",
            "--gateway-url",
            "http://127.0.0.1:1234",
            "--transparent-run",
        ],
    ));
    assert!(
        prepared
            .env
            .contains(&("ANTHROPIC_BASE_URL".into(), "http://127.0.0.1:1234".into()))
    );
    prepared.restore().unwrap();
    assert!(!plugin_dir.exists());
}

#[test]
fn claude_transparent_run_preserves_user_settings_and_prompt_boundary() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("claude-settings.json");
    let original = br#"{"model":"claude-user-setting-sentinel","enabledPlugins":{"other@market":true,"nemo-relay-plugin@nemo-relay-local":true},"env":{"PRIVATE":"kept"}}"#;
    std::fs::write(&source, original).unwrap();
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };
    let prepared = PreparedAgentLaunch::new(
        CodingAgent::ClaudeCode,
        vec![
            "claude".into(),
            "--settings".into(),
            source.display().to_string(),
            "--settings={\"model\":\"ignored-second-source\"}".into(),
            "--print".into(),
            "ping".into(),
            "--".into(),
            "--settings".into(),
            "literal-prompt-value".into(),
        ],
        "http://127.0.0.1:1234",
        &resolved,
        false,
    )
    .unwrap();

    assert_eq!(prepared.argv[1], "--plugin-dir");
    assert_eq!(prepared.argv[3], "--settings");
    let overlay: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&prepared.argv[4]).unwrap()).unwrap();
    assert_eq!(overlay["model"], "claude-user-setting-sentinel");
    assert_eq!(overlay["enabledPlugins"]["other@market"], true);
    assert_eq!(
        overlay["enabledPlugins"]["nemo-relay-plugin@nemo-relay-local"],
        true
    );
    assert_eq!(overlay["env"]["PRIVATE"], "kept");
    assert_eq!(
        overlay["env"]["ANTHROPIC_BASE_URL"],
        "http://127.0.0.1:1234"
    );
    assert_ne!(overlay["model"], "ignored-second-source");
    assert!(
        prepared
            .argv
            .windows(2)
            .any(|pair| { pair == ["--settings", source.to_string_lossy().as_ref()] })
    );
    assert!(
        prepared
            .argv
            .iter()
            .any(|arg| arg.contains("ignored-second-source"))
    );
    let separator = prepared.argv.iter().position(|arg| arg == "--").unwrap();
    assert_eq!(
        &prepared.argv[separator..],
        &["--", "--settings", "literal-prompt-value"]
    );
    assert_eq!(std::fs::read(&source).unwrap(), original);
    prepared.restore().unwrap();
}

#[test]
fn claude_settings_overlay_handles_inline_json_and_rejects_malformed_sources() {
    let inline = vec![
        "claude".into(),
        "--settings={\"model\":\"kept\",\"env\":{\"PRIVATE\":\"yes\"}}".into(),
    ];
    let overlay =
        crate::agents::claude::launch::settings_overlay(&inline, 0, "http://127.0.0.1:4321")
            .unwrap();
    assert_eq!(overlay["model"], "kept");
    assert_eq!(overlay["env"]["PRIVATE"], "yes");
    assert_eq!(
        overlay["env"]["ANTHROPIC_BASE_URL"],
        "http://127.0.0.1:4321"
    );

    let after_separator = vec![
        "claude".into(),
        "--".into(),
        "--settings".into(),
        "prompt-value".into(),
    ];
    let overlay = crate::agents::claude::launch::settings_overlay(
        &after_separator,
        0,
        "http://127.0.0.1:4321",
    )
    .unwrap();
    assert_eq!(overlay.as_object().unwrap().len(), 1);

    let missing = vec!["claude".into(), "--settings".into(), "--".into()];
    assert!(
        crate::agents::claude::launch::settings_overlay(&missing, 0, "http://127.0.0.1:4321")
            .unwrap_err()
            .to_string()
            .contains("missing its value")
    );

    let malformed_env = vec!["claude".into(), "--settings={\"env\":true}".into()];
    assert!(
        crate::agents::claude::launch::settings_overlay(&malformed_env, 0, "http://127.0.0.1:4321")
            .unwrap_err()
            .to_string()
            .contains("field `env` must be a JSON object")
    );

    let temp = tempfile::tempdir().unwrap();
    let non_object_path = temp.path().join("array-settings.json");
    std::fs::write(&non_object_path, "[]").unwrap();
    let non_object = vec![
        "claude".into(),
        format!("--settings={}", non_object_path.display()),
    ];
    assert!(
        crate::agents::claude::launch::settings_overlay(&non_object, 0, "http://127.0.0.1:4321")
            .unwrap_err()
            .to_string()
            .contains("must contain a JSON object")
    );

    let empty_inline = vec!["claude".into(), "--settings=".into()];
    assert!(
        crate::agents::claude::launch::settings_overlay(&empty_inline, 0, "http://127.0.0.1:4321")
            .unwrap_err()
            .to_string()
            .contains("missing its value")
    );

    let missing_file = vec![
        "claude".into(),
        "--verbose".into(),
        "--settings".into(),
        temp.path()
            .join("missing-settings.json")
            .display()
            .to_string(),
    ];
    assert!(
        crate::agents::claude::launch::settings_overlay(&missing_file, 0, "http://127.0.0.1:4321")
            .unwrap_err()
            .to_string()
            .contains("failed to read Claude Code settings")
    );

    let malformed_json = vec!["claude".into(), "--settings={not-json".into()];
    assert!(
        crate::agents::claude::launch::settings_overlay(
            &malformed_json,
            0,
            "http://127.0.0.1:4321"
        )
        .unwrap_err()
        .to_string()
        .contains("failed to parse Claude Code --settings JSON")
    );
}

#[test]
fn codex_session_hook_state_rejects_every_malformed_generated_shape() {
    let malformed = [
        (
            json!({"hooks": {"SessionStart": {}}}),
            "hook groups were malformed",
        ),
        (
            json!({"hooks": {"SessionStart": [true]}}),
            "hook group was malformed",
        ),
        (
            json!({"hooks": {"SessionStart": [{}]}}),
            "hook handlers were malformed",
        ),
        (
            json!({"hooks": {"SessionStart": [{"hooks": [true]}]}}),
            "command hook was malformed",
        ),
        (
            json!({"hooks": {"SessionStart": [{"hooks": [{"type": "prompt", "command": "relay"}]}]}}),
            "hook was not a command",
        ),
        (
            json!({"hooks": {"SessionStart": [{"hooks": [{"type": "command"}]}]}}),
            "hook command was missing",
        ),
    ];
    for (generated, expected) in malformed {
        let error = crate::agents::codex::launch::session_hook_state_override(&generated)
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{error}");
    }

    let generated = json!({
        "hooks": {
            "PreToolUse": [{
                "matcher": "*",
                "hooks": [{
                    "type": "command",
                    "command": "/opt/nemo relay/bin/nemo-relay hook-forward codex",
                    "timeout": 0,
                    "statusMessage": "Forwarding to Relay"
                }]
            }]
        }
    });
    let group = generated["hooks"]["PreToolUse"][0].as_object().unwrap();
    let handler = &group["hooks"].as_array().unwrap()[0];
    let hash =
        crate::agents::codex::launch::command_hook_hash("pre_tool_use", group, handler).unwrap();

    let normalized_handler = json!({
        "type": "command",
        "command": "/opt/nemo relay/bin/nemo-relay hook-forward codex",
        "timeout": 1,
        "statusMessage": "Forwarding to Relay"
    });
    assert_eq!(
        hash,
        crate::agents::codex::launch::command_hook_hash("pre_tool_use", group, &normalized_handler)
            .unwrap()
    );

    let mut without_matcher = group.clone();
    without_matcher.remove("matcher");
    assert_ne!(
        hash,
        crate::agents::codex::launch::command_hook_hash("pre_tool_use", &without_matcher, handler)
            .unwrap()
    );

    let mut without_status = normalized_handler;
    without_status
        .as_object_mut()
        .unwrap()
        .remove("statusMessage");
    assert_ne!(
        hash,
        crate::agents::codex::launch::command_hook_hash("pre_tool_use", group, &without_status)
            .unwrap()
    );

    let state = crate::agents::codex::launch::session_hook_state_override(&generated).unwrap();
    assert!(state.contains("pre_tool_use"));
    assert!(state.contains("trusted_hash"));
    assert!(state.contains("enabled=false"));
}

#[test]
fn claude_prompt_named_like_the_host_does_not_capture_relay_flags() {
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };
    let prepared = PreparedAgentLaunch::new(
        CodingAgent::ClaudeCode,
        vec!["claude".into(), "--".into(), "claude".into()],
        "http://127.0.0.1:1234",
        &resolved,
        true,
    )
    .unwrap();
    let separator = prepared.argv.iter().position(|arg| arg == "--").unwrap();
    assert_eq!(&prepared.argv[separator..], &["--", "claude"]);
    assert_eq!(prepared.argv[1], "--plugin-dir");
}

#[test]
fn hook_write_helpers_cover_toml_escaping() {
    let temp = tempfile::tempdir().unwrap();
    let written_hooks = temp.path().join("written/hooks.json");
    std::fs::create_dir_all(written_hooks.parent().unwrap()).unwrap();
    crate::agents::claude::launch::write_hooks(&written_hooks, json!({"hooks": []})).unwrap();
    assert!(
        std::fs::read_to_string(&written_hooks)
            .unwrap()
            .contains("hooks")
    );

    let groups = crate::agents::codex::launch::hook_groups_toml(&json!([{
        "matcher": "Shell\"Run",
        "hooks": [{"command": "nemo-relay \"quoted\""}]
    }]));
    assert!(groups.contains("matcher=\"Shell\\\"Run\""));
    assert!(groups.contains("command=\"nemo-relay \\\"quoted\\\"\""));

    let escaped = crate::agents::codex::launch::toml_string(r#"C:\tmp\"quoted""#);
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

// This e2e test uses Unix process exit semantics and a shell script named after the inferred host.
// Windows `.cmd` argv delivery is covered independently by `agent_process_tests`.
#[cfg(unix)]
#[tokio::test]
async fn run_starts_gateway_injects_env_and_returns_agent_exit_code() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config.toml");
    std::fs::write(&config, "[upstream]\n").unwrap();
    let output = temp.path().join("env.txt");
    let command_argv = fake_agent_command(temp.path(), &output);
    let command = RunOverrides {
        // Leave `agent: None` so the launcher infers from argv[0] and uses `command_argv`
        // (our fake-agent.sh) as the full argv. With --agent set, the resolver appends
        // command as pass-through after the configured/default binary — not what this test
        // wants, since it specifically asserts that argv[0] is the fake script.
        agent: None,
        config: Some(config),
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
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
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then\n  echo 'codex-cli 0.143.0'\n  exit 0\nfi\nprintf '%s' \"$NEMO_RELAY_GATEWAY_URL\" > \"{}\"\nexit 7\n",
            output.display()
        ),
    )
    .unwrap();
    make_executable(&script);
    vec![script.display().to_string()]
}

#[tokio::test]
async fn dry_run_does_not_spawn_agent() {
    let command = RunOverrides {
        agent: None,
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: true,
        print: false,
        command: vec!["/path/that/does/not/exist/codex".into()],
    };

    let code = run(command, None).await.unwrap();

    assert_eq!(code, ExitCode::SUCCESS);
}

#[tokio::test]
async fn dry_run_does_not_hydrate_dynamic_plugin_lifecycle_state() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = plugin_dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest_path,
        format!(
            r#"
manifest_version = 1

[plugin]
id = "acme.worker"
kind = "worker"

[compat]
relay = "={version}"
worker_protocol = "grpc-v1"

[capabilities]
items = ["plugin_worker"]

[defaults]

[load]
runtime = "python"
entrypoint = "acme.worker:create_plugin"
"#,
            version = env!("CARGO_PKG_VERSION"),
        ),
    )
    .unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(
        temp.path().join("plugins.toml"),
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let command = RunOverrides {
        agent: Some(CodingAgent::Codex),
        config: Some(config_path),
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: true,
        print: false,
        command: vec!["codex".into()],
    };

    let code = run(command, None).await.unwrap();

    assert_eq!(code, ExitCode::SUCCESS);
    assert!(!temp.path().join(".dynamic-plugins.json").exists());
}

#[tokio::test]
async fn wait_for_health_reports_unready_gateway() {
    let error = wait_for_health("http://127.0.0.1:1", "test-fingerprint")
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("gateway did not become ready"), "{error}");
}

#[cfg(unix)]
#[tokio::test]
async fn gateway_failure_terminates_the_agent_and_restores_private_state() {
    let temp = tempfile::tempdir().unwrap();
    let wrapper_pid_path = temp.path().join("wrapper.pid");
    let descendant_pid_path = temp.path().join("descendant.pid");
    let script = temp.path().join("test-agent");
    std::fs::write(
        &script,
        "#!/bin/sh\necho $$ > \"$1\"\nsh -c 'echo $$ > \"$1\"; while :; do :; done' descendant \"$2\" &\nwait \"$!\"\n",
    )
    .unwrap();
    make_executable(&script);
    let overlay = temp.path().join("private-overlay");
    std::fs::create_dir_all(&overlay).unwrap();
    let prepared = PreparedAgentLaunch {
        argv: vec![
            script.display().to_string(),
            wrapper_pid_path.display().to_string(),
            descendant_pid_path.display().to_string(),
        ],
        host_index: 0,
        env: Vec::new(),
        temp_dirs: vec![overlay.clone()],
        notes: Vec::new(),
        proxy_credential: crate::provider_auth::TransparentProxyCredential::from_static(
            "test-proxy-token",
        ),
        secret_env_names: Vec::new(),
    };
    let observed_wrapper_pid_path = wrapper_pid_path.clone();
    let observed_descendant_pid_path = descendant_pid_path.clone();
    let task = tokio::spawn(async move {
        for _ in 0..500 {
            if observed_wrapper_pid_path.exists() && observed_descendant_pid_path.exists() {
                return Err(CliError::Launch("injected gateway failure".into()));
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Err(CliError::Launch(
            "test agent did not publish its process ID".into(),
        ))
    });
    let (shutdown_tx, _shutdown_rx) = oneshot::channel();
    let running_server = RunningGateway { shutdown_tx, task };

    let error = tokio::time::timeout(
        Duration::from_secs(10),
        supervise_prepared_run(&prepared, running_server),
    )
    .await
    .expect("agent supervision did not finish")
    .unwrap_err()
    .to_string();

    assert!(error.contains("injected gateway failure"), "{error}");
    assert!(!overlay.exists());
    for pid_path in [wrapper_pid_path, descendant_pid_path] {
        let pid = std::fs::read_to_string(pid_path).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            // SAFETY: Signal 0 performs an existence check and does not alter the target process.
            let result = unsafe { libc::kill(pid.trim().parse().unwrap(), 0) };
            if result == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "agent process {pid} was not reaped"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

#[tokio::test]
async fn execute_live_run_reports_gateway_startup_error_when_health_check_fails() {
    let _guard = crate::test_support::PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _env = EnvScope::without_managed_bootstrap();
    let _ = nemo_relay::plugin::clear_plugin_configuration();
    let resolved = ResolvedConfig {
        gateway: GatewayConfig::default(),
        agents: AgentConfigs::default(),
        ..ResolvedConfig::default()
    };
    let prepared = PreparedAgentLaunch::new(
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
async fn execute_live_run_removes_hermes_overlay_when_health_check_fails() {
    let _guard = crate::test_support::PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let _env = EnvScope::without_managed_bootstrap();
    let _ = nemo_relay::plugin::clear_plugin_configuration();
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
    let prepared = PreparedAgentLaunch::new(
        CodingAgent::Hermes,
        vec!["hermes".into(), "chat".into()],
        "http://127.0.0.1:1234",
        &resolved,
        false,
    )
    .unwrap();
    let overlay = prepared
        .env
        .iter()
        .find_map(|(name, value)| (name == "HERMES_HOME").then(|| PathBuf::from(value)))
        .unwrap();
    let overlay_config: serde_json::Value =
        serde_yaml::from_str(&std::fs::read_to_string(overlay.join("config.yaml")).unwrap())
            .unwrap();
    assert!(crate::hook_assertions::value_has_command_arguments(
        &overlay_config,
        &[
            "hook-forward",
            "hermes",
            "--gateway-url",
            "http://127.0.0.1:1234",
            "--transparent-run",
        ],
    ));

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

    assert!(error.contains("gateway did not become ready"), "{error}");
    assert_eq!(std::fs::read_to_string(&hooks_path).unwrap(), original);
    assert!(!overlay.exists());
}

#[cfg(unix)]
fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).unwrap();
}
