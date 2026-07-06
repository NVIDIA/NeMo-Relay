// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::ffi::{OsStr, OsString};

struct EnvScope {
    _guard: std::sync::MutexGuard<'static, ()>,
    values: Vec<(OsString, Option<OsString>)>,
}

impl EnvScope {
    fn provider_test(values: &[(&str, Option<&OsStr>)]) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut names = std::env::vars_os()
            .map(|(name, _)| name)
            .filter(|name| {
                let name = name.to_string_lossy();
                name == "OPENAI_API_KEY"
                    || name == "OPENAI_API_KEYS"
                    || name == "OPENCLAW_LIVE_OPENAI_KEY"
                    || name.starts_with("OPENAI_API_KEY_")
                    || name == "ANTHROPIC_API_KEY"
                    || name == "ANTHROPIC_API_KEYS"
                    || name == "OPENCLAW_LIVE_ANTHROPIC_KEY"
                    || name.starts_with("ANTHROPIC_API_KEY_")
                    || name == OPENCLAW_CONFIG_PATH
                    || name == "OPENCLAW_INCLUDE_ROOTS"
                    || name == "OPENCLAW_STATE_DIR"
                    || name == "OPENCLAW_HOME"
            })
            .collect::<Vec<_>>();
        names.extend(values.iter().map(|(name, _)| OsString::from(name)));
        names.sort();
        names.dedup();
        let previous = names
            .iter()
            .map(|name| (name.clone(), std::env::var_os(name)))
            .collect::<Vec<_>>();
        for name in &names {
            unsafe { std::env::remove_var(name) };
        }
        for (name, value) in values {
            if let Some(value) = value {
                unsafe { std::env::set_var(name, value) };
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
        for (name, value) in self.values.drain(..) {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

#[test]
fn forces_model_serving_commands_local_and_rejects_remote_or_detached_paths() {
    let mut default = vec!["openclaw".into()];
    normalize_foreground_argv(&mut default).unwrap();
    assert_eq!(default, ["openclaw", "tui", "--local"]);

    let mut agent = vec![
        "openclaw".into(),
        "agent".into(),
        "--message".into(),
        "hello".into(),
    ];
    normalize_foreground_argv(&mut agent).unwrap();
    assert_eq!(agent[2], "--local");

    let mut tui = vec![
        "npx".into(),
        "openclaw".into(),
        "tui".into(),
        "--message".into(),
        "hi".into(),
    ];
    normalize_foreground_argv(&mut tui).unwrap();
    assert_eq!(tui[3], "--local");

    let mut foreground = vec!["openclaw".into(), "gateway".into(), "run".into()];
    normalize_foreground_argv(&mut foreground).unwrap();
    assert_eq!(foreground, ["openclaw", "gateway", "run"]);

    let mut message_named_openclaw = vec![
        "openclaw".into(),
        "agent".into(),
        "--message".into(),
        "openclaw".into(),
        "--json".into(),
    ];
    normalize_foreground_argv(&mut message_named_openclaw).unwrap();
    assert_eq!(message_named_openclaw[2], "--local");

    let mut root_option = vec![
        "openclaw".into(),
        "--log-level".into(),
        "debug".into(),
        "agent".into(),
    ];
    normalize_foreground_argv(&mut root_option).unwrap();
    assert_eq!(root_option[4], "--local");

    let remote = normalize_foreground_argv(&mut vec![
        "openclaw".into(),
        "tui".into(),
        "--url=ws://remote.example".into(),
    ])
    .unwrap_err()
    .to_string();
    assert!(remote.contains("remote TUI"));

    let detached = normalize_foreground_argv(&mut vec![
        "openclaw".into(),
        "gateway".into(),
        "start".into(),
    ])
    .unwrap_err()
    .to_string();
    assert!(detached.contains("foreground `gateway run`"));

    let container = normalize_foreground_argv(&mut vec![
        "openclaw".into(),
        "--container".into(),
        "sandbox".into(),
        "agent".into(),
    ])
    .unwrap_err()
    .to_string();
    assert!(container.contains("host-side temporary provider overlay"));

    let mut profiled = vec!["openclaw".into(), "--profile".into(), "work".into()];
    normalize_foreground_argv(&mut profiled).unwrap();
    assert_eq!(
        profiled,
        ["openclaw", "--profile", "work", "tui", "--local"]
    );

    let mut command_named_profile = vec![
        "openclaw".into(),
        "--profile".into(),
        "agent".into(),
        "tui".into(),
    ];
    normalize_foreground_argv(&mut command_named_profile).unwrap();
    assert_eq!(
        command_named_profile,
        ["openclaw", "--profile", "agent", "tui", "--local"]
    );
    assert_eq!(
        command_profile(&command_named_profile).unwrap().as_deref(),
        Some("agent")
    );

    let admin = normalize_foreground_argv(&mut vec![
        "openclaw".into(),
        "plugins".into(),
        "enable".into(),
        "nemo-relay".into(),
    ])
    .unwrap_err()
    .to_string();
    assert!(admin.contains("run configuration, plugin, model-management"));
}

#[test]
fn resolves_profile_config_after_environment_precedence() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::provider_test(&[("OPENCLAW_HOME", Some(temp.path().as_os_str()))]);

    assert_eq!(
        resolve_config_path(&["openclaw".into(), "--profile=work".into()]).unwrap(),
        temp.path().join(".openclaw-work/openclaw.json")
    );
    assert_eq!(
        resolve_config_path(&["openclaw".into(), "--dev".into()]).unwrap(),
        temp.path().join(".openclaw-dev/openclaw.json")
    );
}

#[test]
fn routes_anthropic_and_openai_completions_with_original_upstreams() {
    let _env = EnvScope::provider_test(&[]);
    let config: Value = json5::from_str(
        r#"{
          // Both providers are explicitly API-key-backed.
          models: { providers: {
            anthropic: {
              baseUrl: "https://anthropic.internal.example",
              api: "anthropic-messages",
              apiKey: { source: "env", provider: "default", id: "ANTHROPIC_API_KEY" },
            },
            openai: {
              baseUrl: "https://openai.internal.example/v1",
              api: "openai-completions",
              auth: "api-key",
            },
          } },
        }"#,
    )
    .unwrap();

    let plan = build_routing_plan(Some(&config), "http://127.0.0.1:43123");

    assert_eq!(
        plan.anthropic_upstream.as_deref(),
        Some("https://anthropic.internal.example")
    );
    assert_eq!(
        plan.openai_upstream.as_deref(),
        Some("https://openai.internal.example/v1")
    );
    assert_eq!(
        plan.provider_overrides["anthropic"]["baseUrl"],
        "http://127.0.0.1:43123"
    );
    assert_eq!(
        plan.provider_overrides["anthropic"]["agentRuntime"]["id"],
        "openclaw"
    );
    assert_eq!(
        plan.provider_overrides["openai"]["baseUrl"],
        "http://127.0.0.1:43123/v1"
    );
    assert_eq!(
        plan.provider_overrides["openai"]["agentRuntime"]["id"],
        "openclaw"
    );
    assert!(
        plan.notes
            .iter()
            .any(|note| note.contains("openai-completions"))
    );
    assert!(
        plan.notes
            .iter()
            .any(|note| note.contains("anthropic-messages"))
    );
}

#[test]
fn routes_openai_responses_and_uses_canonical_default_endpoint() {
    let _env = EnvScope::provider_test(&[("OPENAI_API_KEY", Some(OsStr::new("test-key")))]);
    let config = json!({
        "models": {"providers": {"openai": {"api": "openai-responses"}}}
    });

    let plan = build_routing_plan(Some(&config), "http://127.0.0.1:43124");

    assert_eq!(
        plan.openai_upstream.as_deref(),
        Some(DEFAULT_OPENAI_BASE_URL)
    );
    assert_eq!(
        plan.provider_overrides["openai"]["baseUrl"],
        "http://127.0.0.1:43124/v1"
    );
    assert!(
        plan.notes
            .iter()
            .any(|note| note.contains("openai-responses"))
    );
}

#[test]
fn leaves_custom_unsupported_oauth_and_model_override_providers_unchanged() {
    let _env = EnvScope::provider_test(&[]);
    let config = json!({
        "models": {"providers": {
            "custom-openai": {
                "baseUrl": "https://custom.example/v1",
                "api": "openai-completions",
                "apiKey": "secret"
            },
            "openai": {
                "api": "openai-responses",
                "auth": "oauth"
            },
            "anthropic": {
                "api": "anthropic-messages",
                "apiKey": "secret",
                "models": [{"id": "claude", "baseUrl": "https://model.example"}]
            }
        }}
    });

    let plan = build_routing_plan(Some(&config), "http://127.0.0.1:43125");

    assert!(plan.provider_overrides.is_empty());
    assert!(plan.openai_upstream.is_none());
    assert!(plan.anthropic_upstream.is_none());
    assert!(plan.notes.iter().any(|note| note.contains("custom-openai")));
    assert!(
        plan.notes
            .iter()
            .any(|note| note.contains("explicit authentication mode is not API-key based"))
    );
    assert!(plan.notes.iter().any(|note| note.contains("model-level")));
}

#[test]
fn provider_includes_fail_open_without_guessing_an_upstream() {
    let _env = EnvScope::provider_test(&[("OPENAI_API_KEY", Some(OsStr::new("test-key")))]);
    let config = json!({
        "models": {"providers": {"openai": {"$include": "./openai-provider.json5"}}}
    });

    let plan = build_routing_plan(Some(&config), "http://127.0.0.1:43126");

    assert!(plan.provider_overrides.is_empty());
    assert!(
        plan.notes
            .iter()
            .any(|note| note.contains("will not guess"))
    );

    let unrelated = json!({
        "plugins": {"$include": "./plugins.json5"},
        "models": {"providers": {"openai": {"api": "openai-responses"}}}
    });
    let plan = build_routing_plan(Some(&unrelated), "http://127.0.0.1:43126");
    assert!(plan.provider_overrides.contains_key("openai"));
}

#[test]
fn explicit_non_http_agent_runtime_is_not_claimed_as_routed() {
    let _env = EnvScope::provider_test(&[("OPENAI_API_KEY", Some(OsStr::new("test-key")))]);
    let config = json!({
        "models": {"providers": {"openai": {
            "api": "openai-responses",
            "agentRuntime": {"id": "codex"}
        }}}
    });

    let plan = build_routing_plan(Some(&config), "http://127.0.0.1:43126");

    assert!(plan.provider_overrides.get("openai").is_none());
    assert!(plan.openai_upstream.is_none());
    assert!(
        plan.notes
            .iter()
            .any(|note| note.contains("direct OpenClaw HTTP runtime"))
    );
}

#[test]
fn substituted_provider_endpoint_is_not_copied_unresolved_into_relay() {
    let _env = EnvScope::provider_test(&[("OPENAI_API_KEY", Some(OsStr::new("test-key")))]);
    let config = json!({
        "models": {"providers": {"openai": {
            "api": "openai-responses",
            "baseUrl": "${OPENAI_BASE}/v1"
        }}}
    });

    let plan = build_routing_plan(Some(&config), "http://127.0.0.1:43126");

    assert!(plan.provider_overrides.get("openai").is_none());
    assert!(plan.openai_upstream.is_none());
    assert!(
        plan.notes
            .iter()
            .any(|note| note.contains("environment-substituted baseUrl"))
    );
}

#[test]
fn agent_model_runtime_policy_is_not_overridden_or_claimed_as_routed() {
    let _env = EnvScope::provider_test(&[("OPENAI_API_KEY", Some(OsStr::new("test-key")))]);
    let config = json!({
        "agents": {
            "defaults": {"models": {
                "openai/gpt-5.5": {"agentRuntime": {"id": "codex"}}
            }},
            "list": [{
                "id": "main",
                "models": {
                    "anthropic/*": {"agentRuntime": {"id": "claude-cli"}}
                }
            }]
        },
        "models": {"providers": {
            "openai": {"api": "openai-responses"},
            "anthropic": {"api": "anthropic-messages", "apiKey": "secret"}
        }}
    });

    let plan = build_routing_plan(Some(&config), "http://127.0.0.1:43126");

    assert!(plan.provider_overrides.is_empty());
    assert!(plan.openai_upstream.is_none());
    assert!(plan.anthropic_upstream.is_none());
    assert_eq!(
        plan.notes
            .iter()
            .filter(|note| note.contains("agent model policy"))
            .count(),
        2
    );
}

#[cfg(unix)]
#[test]
fn temporary_overlay_preserves_source_config_metadata_and_explicit_relay_upstream() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("openclaw.json5");
    let original = r#"{
      // This comment and formatting must remain byte-for-byte unchanged.
      agents: { defaults: { model: { primary: "openai/gpt-test" } } },
      models: { providers: { openai: {
        baseUrl: "https://source-upstream.example/v1",
        api: "openai-responses",
        apiKey: "configured-secret",
      } } },
    }
"#;
    std::fs::write(&source, original).unwrap();
    let _env = EnvScope::provider_test(&[(OPENCLAW_CONFIG_PATH, Some(source.as_os_str()))]);
    let executable = temp.path().join("openclaw");
    std::fs::write(
        &executable,
        r#"#!/bin/sh
case "$*" in
  *"models auth list --provider openai --json"*|*"models auth list --provider anthropic --json"*)
    printf '%s' '{"profiles":[]}'
    ;;
  *) exit 1 ;;
esac
"#,
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&executable, permissions).unwrap();
    let mut argv = vec![
        executable.display().to_string(),
        "agent".into(),
        "--message".into(),
        "hello".into(),
    ];
    let mut gateway = GatewayConfig {
        openai_base_url: "https://explicit-relay-upstream.example/v1".into(),
        metadata: Some(json!({"team": "relay"})),
        ..GatewayConfig::default()
    };

    let prepared = prepare(
        &mut argv,
        "http://127.0.0.1:43127",
        &mut gateway,
        ExplicitUpstreams {
            openai: true,
            anthropic: false,
        },
        false,
    )
    .unwrap();

    assert_eq!(std::fs::read_to_string(&source).unwrap(), original);
    assert_eq!(
        gateway.openai_base_url,
        "https://explicit-relay-upstream.example/v1"
    );
    assert_eq!(
        gateway.metadata.as_ref().unwrap()["nemo_relay_invocation"]["agent"],
        "openclaw"
    );
    assert_eq!(gateway.metadata.as_ref().unwrap()["team"], "relay");
    assert_eq!(argv[2], "--local");

    let overlay_path = prepared
        .env
        .iter()
        .find_map(|(name, value)| (name == OPENCLAW_CONFIG_PATH).then(|| PathBuf::from(value)))
        .unwrap();
    let overlay: Value =
        serde_json::from_str(&std::fs::read_to_string(&overlay_path).unwrap()).unwrap();
    assert_eq!(overlay["$include"], source.display().to_string());
    assert_eq!(
        overlay["models"]["providers"]["openai"]["baseUrl"],
        "http://127.0.0.1:43127/v1"
    );
    assert_eq!(overlay_path.parent(), source.parent());
    std::fs::remove_file(&prepared.temp_files[0]).unwrap();
    assert!(!overlay_path.exists());
    assert!(source.exists());
}

#[cfg(unix)]
#[test]
fn read_only_probes_detect_plugin_state_and_deterministic_auth_profiles() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("openclaw.json5");
    std::fs::write(&source, "{}").unwrap();
    let executable = temp.path().join("openclaw");
    std::fs::write(
        &executable,
        r#"#!/bin/sh
case "$*" in
  "plugins inspect nemo-relay --json")
    printf '%s' '{"enabled":false}'
    ;;
  *"models auth --agent work list --provider openai --json"*)
    printf '%s' '{"profiles":[{"id":"openai:work","type":"api_key"}]}'
    ;;
  *"models auth list --provider openai --json"*)
    printf '%s' '{"profiles":[{"id":"openai:one","type":"api_key"},{"id":"openai:two","type":"api_key"}]}'
    ;;
  *"models auth list --provider anthropic --json"*)
    printf '%s' '{"profiles":[{"id":"anthropic:key","type":"api_key"},{"id":"anthropic:oauth","type":"oauth"}]}'
    ;;
  *)
    exit 2
    ;;
esac
"#,
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&executable, permissions).unwrap();
    let argv = vec![executable.display().to_string()];

    assert_eq!(
        probe_relay_plugin(&argv, &source),
        RelayPluginDetection::DetectedDisabled
    );
    assert_eq!(
        probe_auth_profiles(&argv, &source, None, ProviderFamily::OpenAi),
        AuthProfileEvidence::ApiKeyOnly
    );
    assert_eq!(
        probe_auth_profiles(&argv, &source, None, ProviderFamily::Anthropic),
        AuthProfileEvidence::NonApiKeyOrMixed
    );

    let selected = vec![
        executable.display().to_string(),
        "agent".into(),
        "--agent".into(),
        "work".into(),
    ];
    assert_eq!(selected_agent_id(&selected), Some("work"));
    assert_eq!(
        probe_auth_profiles(&selected, &source, None, ProviderFamily::OpenAi),
        AuthProfileEvidence::ApiKeyOnly
    );

    std::fs::write(
        &executable,
        "#!/bin/sh\nprintf '%s' '{\"plugin\":{\"enabled\":true,\"status\":\"error\"}}'\n",
    )
    .unwrap();
    assert_eq!(
        probe_relay_plugin(&argv, &source),
        RelayPluginDetection::DetectedError
    );
}

#[cfg(unix)]
#[test]
fn ambiguous_agent_selection_probes_every_configured_auth_store() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("openclaw.json5");
    std::fs::write(&source, "{}").unwrap();
    let executable = temp.path().join("openclaw");
    std::fs::write(
        &executable,
        r#"#!/bin/sh
case "$*" in
  "models auth list --provider openai --json"|*"models auth --agent work list --provider openai --json"*)
    printf '%s' '{"profiles":[{"type":"api_key"}]}'
    ;;
  *"models auth --agent personal list --provider openai --json"*)
    printf '%s' '{"profiles":[]}'
    ;;
  *)
    exit 2
    ;;
esac
"#,
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&executable, permissions).unwrap();
    let argv = vec![
        executable.display().to_string(),
        "agent".into(),
        "--session-key".into(),
        "agent:personal:main".into(),
    ];
    let config = json!({"agents": {"list": [{"id": "work"}, {"id": "personal"}]}});

    assert_eq!(
        probe_auth_profiles(&argv, &source, Some(&config), ProviderFamily::OpenAi),
        AuthProfileEvidence::ApiKeyOrNone
    );
}

#[test]
fn trusted_config_and_global_dotenv_api_keys_enable_profileless_routes() {
    assert!(!nonempty_dotenv_value(" # comment"));
    assert!(!nonempty_dotenv_value("\"\""));
    let temp = tempfile::tempdir().unwrap();
    let state = temp.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(
        state.join(".env"),
        "# values are never copied into diagnostics\nANTHROPIC_API_KEY=dotenv-secret\n",
    )
    .unwrap();
    let _env = EnvScope::provider_test(&[("OPENCLAW_STATE_DIR", Some(state.as_os_str()))]);
    let argv = vec!["openclaw".to_string(), "agent".into()];
    let config = json!({
        "env": {"vars": {"OPENAI_API_KEY": "config-secret"}},
        "models": {"providers": {
            "openai": {"api": "openai-responses"},
            "anthropic": {"api": "anthropic-messages"}
        }}
    });
    let trusted = trusted_api_key_sources(&argv, Some(&config));

    assert!(trusted.openai);
    assert!(trusted.anthropic);
    let plan = build_routing_plan_with_evidence(
        Some(&config),
        "http://127.0.0.1:43129",
        AuthProfileEvidenceSet::default(),
        trusted,
    );
    assert!(plan.provider_overrides.contains_key("openai"));
    assert!(plan.provider_overrides.contains_key("anthropic"));
    assert!(plan.notes.iter().all(|note| !note.contains("secret")));
}

#[test]
fn ambiguous_auth_profile_types_prevent_provider_redirection() {
    let _env = EnvScope::provider_test(&[("OPENAI_API_KEY", Some(OsStr::new("test-key")))]);
    let config = json!({
        "models": {"providers": {"openai": {"api": "openai-responses"}}}
    });
    let plan = build_routing_plan_with_auth(
        Some(&config),
        "http://127.0.0.1:43128",
        AuthProfileEvidenceSet {
            openai: AuthProfileEvidence::NonApiKeyOrMixed,
            anthropic: AuthProfileEvidence::None,
        },
    );

    assert!(plan.provider_overrides.is_empty());
    assert!(plan.notes.iter().any(|note| note.contains("ambiguous")));

    let explicit_but_unverified = json!({
        "models": {"providers": {"openai": {
            "api": "openai-responses",
            "apiKey": "configured-but-unverified"
        }}}
    });
    let plan = build_routing_plan_with_auth(
        Some(&explicit_but_unverified),
        "http://127.0.0.1:43128",
        AuthProfileEvidenceSet {
            openai: AuthProfileEvidence::Unknown,
            anthropic: AuthProfileEvidence::None,
        },
    );
    assert!(plan.provider_overrides.is_empty());
    assert!(
        plan.notes
            .iter()
            .any(|note| note.contains("could not be verified"))
    );
}

#[test]
fn invocation_metadata_preserves_non_object_user_metadata() {
    let mut gateway = GatewayConfig {
        metadata: Some(json!(["user", "metadata"])),
        ..GatewayConfig::default()
    };

    add_invocation_metadata(&mut gateway);

    let metadata = gateway.metadata.unwrap();
    assert_eq!(
        metadata["nemo_relay_original_metadata"],
        json!(["user", "metadata"])
    );
    assert_eq!(metadata["nemo_relay_invocation"]["agent"], "openclaw");
}
