// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::path::Path;
use std::time::Duration;

use reqwest::header::HeaderMap;
use serde_json::Value;

use crate::agents::CodingAgent;

#[test]
fn null_hook_configuration_normalizes_to_an_empty_object() {
    assert_eq!(
        super::merging::hook_config_root(serde_json::Value::Null).unwrap(),
        json!({})
    );
}

fn hook_request(agent: CodingAgent) -> HookForwardRequest {
    HookForwardRequest {
        agent,
        hook_config: None,
        gateway_url: None,
        generation_file: None,
        generation_token: None,
        forward_only: false,
        transparent_run: false,
        profile: None,
        session_metadata: None,
        gateway_mode: None,
        failure_policy: HookFailurePolicy::Default,
    }
}

fn write_private_hook_config(path: &std::path::Path, contents: &str) {
    std::fs::write(path, contents).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    #[cfg(windows)]
    crate::filesystem::protect_private_windows_path(path).unwrap();
}

#[test]
fn private_hook_config_round_trips_and_hydrates_a_hook_request() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("hook.json");
    HookCommandConfig::transparent(CodingAgent::Codex, "http://127.0.0.1:1234")
        .write(&path)
        .unwrap();

    let config = HookCommandConfig::load(&path).unwrap();
    let mut request = hook_request(CodingAgent::Codex);
    request.transparent_run = true;
    config.apply(&mut request).unwrap();
    assert_eq!(
        request.gateway_url.as_deref(),
        Some("http://127.0.0.1:1234")
    );
    assert!(request.transparent_run);
    assert!(request.generation_file.is_none());
    assert!(request.generation_token.is_none());
}

#[test]
fn private_hook_config_rejects_agent_mismatch_and_inline_configuration() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("hook.json");
    HookCommandConfig::transparent(CodingAgent::Codex, "http://127.0.0.1:1234")
        .write(&path)
        .unwrap();

    let mut agent_mismatch = hook_request(CodingAgent::ClaudeCode);
    assert!(
        HookCommandConfig::load(&path)
            .unwrap()
            .apply(&mut agent_mismatch)
            .unwrap_err()
            .contains("requested claude")
    );

    let mut inline_configuration = hook_request(CodingAgent::Codex);
    inline_configuration.gateway_url = Some("http://127.0.0.1:5678".into());
    assert!(
        HookCommandConfig::load(&path)
            .unwrap()
            .apply(&mut inline_configuration)
            .unwrap_err()
            .contains("cannot be combined")
    );
}

#[test]
fn private_hook_config_rejects_unknown_and_incomplete_values() {
    let directory = tempfile::tempdir().unwrap();
    let unknown = directory.path().join("unknown.json");
    write_private_hook_config(
        &unknown,
        r#"{"version":1,"agent":"codex","gateway_url":"http://127.0.0.1:1234","forward_only":false,"transparent_run":true,"unexpected":true}"#,
    );
    assert!(
        HookCommandConfig::load(&unknown)
            .unwrap_err()
            .contains("failed to parse")
    );

    let incomplete = directory.path().join("incomplete.json");
    write_private_hook_config(
        &incomplete,
        r#"{"version":1,"agent":"codex","gateway_url":"http://127.0.0.1:1234","generation_file":"generation","generation_token":null,"forward_only":false,"transparent_run":false}"#,
    );
    assert!(
        HookCommandConfig::load(&incomplete)
            .unwrap_err()
            .contains("both generation file and token")
    );
}

#[cfg(unix)]
#[test]
fn private_hook_config_rejects_symlinks_and_broad_permissions() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("hook.json");
    HookCommandConfig::transparent(CodingAgent::Codex, "http://127.0.0.1:1234")
        .write(&config)
        .unwrap();
    let link = directory.path().join("hook-link.json");
    symlink(&config, &link).unwrap();
    assert!(
        HookCommandConfig::load(&link)
            .unwrap_err()
            .contains("owner-only regular file")
    );

    std::fs::set_permissions(&config, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(
        HookCommandConfig::load(&config)
            .unwrap_err()
            .contains("owner-only regular file")
    );

    let unsafe_parent = directory.path().join("unsafe-parent");
    std::fs::create_dir(&unsafe_parent).unwrap();
    let private_config = unsafe_parent.join("hook.json");
    HookCommandConfig::transparent(CodingAgent::Codex, "http://127.0.0.1:1234")
        .write(&private_config)
        .unwrap();
    std::fs::set_permissions(&unsafe_parent, std::fs::Permissions::from_mode(0o777)).unwrap();
    assert!(
        HookCommandConfig::load(&private_config)
            .unwrap_err()
            .contains("parent must be current-user-owned and non-group/world-writable")
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)] // The process-wide environment lock must cover the hook call.
async fn transparent_run_skips_stale_persistent_hook_config_before_loading_it() {
    let _guard = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let previous = std::env::var_os(crate::configuration::TRANSPARENT_RUN_ENV);
    // SAFETY: The process-wide environment lock is held for this test.
    unsafe { std::env::set_var(crate::configuration::TRANSPARENT_RUN_ENV, "1") };
    let mut request = hook_request(CodingAgent::Codex);
    request.hook_config = Some(std::path::PathBuf::from(
        "missing-persistent-hook-config.json",
    ));
    request.failure_policy = HookFailurePolicy::FailClosed;
    let result = crate::hooks::hook_forward(request).await;
    // SAFETY: The process-wide environment lock is still held for this test.
    unsafe {
        match previous {
            Some(value) => std::env::set_var(crate::configuration::TRANSPARENT_RUN_ENV, value),
            None => std::env::remove_var(crate::configuration::TRANSPARENT_RUN_ENV),
        }
    }
    assert!(result.is_ok());
}

struct BootstrapConfigHome {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous: Option<std::ffi::OsString>,
}

impl BootstrapConfigHome {
    fn enter(path: &std::path::Path) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = std::env::var_os("XDG_CONFIG_HOME");
        // SAFETY: This scope holds the process-wide environment mutex.
        unsafe { std::env::set_var("XDG_CONFIG_HOME", path) };
        Self {
            _guard: guard,
            previous,
        }
    }
}

impl Drop for BootstrapConfigHome {
    fn drop(&mut self) {
        // SAFETY: This scope still holds the process-wide environment mutex.
        unsafe {
            match self.previous.take() {
                Some(previous) => std::env::set_var("XDG_CONFIG_HOME", previous),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
    }
}

struct ScopedEnvVar {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl ScopedEnvVar {
    fn set(key: &'static str, value: &std::ffi::OsStr) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: The caller holds the process-wide environment mutex through BootstrapConfigHome.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    fn drop(&mut self) {
        // SAFETY: BootstrapConfigHome outlives this guard and still holds the environment mutex.
        unsafe {
            match self.previous.take() {
                Some(previous) => std::env::set_var(self.key, previous),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[tokio::test]
async fn transparent_hook_delivery_authenticates_the_wrapper_gateway() {
    let _plugin_guard = crate::test_support::PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let _bootstrap_home = BootstrapConfigHome::enter(&temp.path().join("xdg"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bind = listener.local_addr().unwrap();
    let gateway_url = format!("http://{bind}");
    let fingerprint = crate::configuration::transparent_gateway_fingerprint(&gateway_url);
    let config = crate::configuration::GatewayConfig {
        bind,
        ..crate::configuration::GatewayConfig::default()
    };
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let proxy_credential = crate::provider_auth::TransparentProxyCredential::generate().unwrap();
    let _proxy_credential = ScopedEnvVar::set(
        crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_ENV,
        proxy_credential.expose().as_ref(),
    );
    let server = tokio::spawn(crate::server::serve_transparent_listener_with_dynamic(
        listener,
        config,
        Vec::new(),
        fingerprint.clone(),
        proxy_credential,
        Some(shutdown_rx),
    ));
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let url = gateway_url.clone();
            let fingerprint = fingerprint.clone();
            if tokio::task::spawn_blocking(move || {
                crate::gateway::client::healthz_compatible(&url, &fingerprint)
            })
            .await
            .unwrap()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("wrapper gateway did not become healthy");
    let command = HookForwardRequest {
        agent: CodingAgent::Codex,
        hook_config: None,
        gateway_url: Some(gateway_url.clone()),
        generation_file: None,
        generation_token: None,
        forward_only: false,
        transparent_run: true,
        profile: None,
        session_metadata: None,
        gateway_mode: None,
        failure_policy: HookFailurePolicy::FailClosed,
    };
    let gateway = transparent_gateway_spec(&gateway_url).unwrap();

    let response = send_verified_hook_forward_request(
        &command,
        &gateway,
        &gateway_url,
        json!({
            "session_id": "verified-transparent-hook",
            "hook_event_name": "SessionStart"
        })
        .to_string(),
    )
    .await
    .unwrap()
    .unwrap();

    assert_eq!(response.status, 200);
    let _ = shutdown_tx.send(());
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .expect("wrapper gateway did not stop")
        .unwrap()
        .unwrap();
}

#[test]
fn hook_payload_reader_normalizes_blank_input_and_accepts_the_exact_limit() {
    assert_eq!(read_hook_payload_from(" \n\t".as_bytes(), 3).unwrap(), "{}");
    assert_eq!(
        read_hook_payload_from("1234".as_bytes(), 4).unwrap(),
        "1234"
    );
}

#[test]
fn hook_payload_reader_rejects_oversized_invalid_and_unreadable_input() {
    let oversized = read_hook_payload_from("12345".as_bytes(), 4)
        .unwrap_err()
        .to_string();
    assert!(oversized.contains("exceeds the 4-byte limit"));

    let invalid = read_hook_payload_from([0xff].as_slice(), 1)
        .unwrap_err()
        .to_string();
    assert!(invalid.contains("not valid UTF-8"));

    struct FailingReader;
    impl std::io::Read for FailingReader {
        fn read(&mut self, _buffer: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("synthetic hook input failure"))
        }
    }
    assert!(
        read_hook_payload_from(FailingReader, 4)
            .unwrap_err()
            .to_string()
            .contains("synthetic hook input failure")
    );
}

#[test]
fn explicit_persistent_destinations_ignore_ambient_urls() {
    let destination = resolve_hook_destination(
        Some("http://installed".into()),
        Some("http://dynamic".into()),
        false,
        false,
    );
    assert_eq!(destination.gateway_url, "http://installed");
    assert_eq!(destination.lifecycle, HookGatewayLifecycle::Existing);

    let destination = resolve_hook_destination(None, Some("http://dynamic".into()), false, false);
    assert_eq!(destination.gateway_url, "http://dynamic");
    assert_eq!(destination.lifecycle, HookGatewayLifecycle::Transparent);

    let destination = resolve_hook_destination(
        Some("http://source-plugin".into()),
        Some("http://dynamic".into()),
        true,
        false,
    );
    assert_eq!(destination.gateway_url, "http://source-plugin");
    assert_eq!(destination.lifecycle, HookGatewayLifecycle::Existing);

    let destination = resolve_hook_destination(None, Some("http://dynamic".into()), true, false);
    assert_eq!(destination.gateway_url, crate::bootstrap::DEFAULT_URL);
    assert_eq!(destination.lifecycle, HookGatewayLifecycle::Existing);

    let destination = resolve_hook_destination(Some("http://embedded".into()), None, false, true);
    assert_eq!(destination.gateway_url, "http://embedded");
    assert_eq!(destination.lifecycle, HookGatewayLifecycle::Transparent);

    let destination = resolve_hook_destination(None, None, false, false);
    assert_eq!(destination.gateway_url, crate::bootstrap::DEFAULT_URL);
    assert_eq!(destination.lifecycle, HookGatewayLifecycle::Existing);
}

#[test]
fn verified_hook_response_rejects_invalid_status_and_fail_open_http_errors() {
    let error = handle_verified_hook_forward_response(
        Ok(crate::gateway::client::VerifiedHttpResponse {
            status: 0,
            body: Vec::new(),
        }),
        true,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("invalid status"), "{error}");

    handle_verified_hook_forward_response(
        Ok(crate::gateway::client::VerifiedHttpResponse {
            status: 0,
            body: Vec::new(),
        }),
        false,
    )
    .unwrap();

    handle_hook_forward_status(reqwest::StatusCode::BAD_GATEWAY, String::new(), false).unwrap();
}

#[test]
fn hook_response_statuses_preserve_guardrail_rejections_and_fail_closed_errors() {
    let rejection = handle_hook_forward_status(
        reqwest::StatusCode::FORBIDDEN,
        r#"{"error":{"type":"nemo_relay_guardrail_rejected","reason":"policy denied"}}"#.into(),
        false,
    )
    .unwrap_err()
    .to_string();
    assert!(rejection.contains("policy denied"), "{rejection}");

    let fallback = handle_hook_forward_status(
        reqwest::StatusCode::BAD_REQUEST,
        r#"{"error":{"type":"nemo_relay_guardrail_rejected","message":"fallback"}}"#.into(),
        false,
    )
    .unwrap_err()
    .to_string();
    assert!(fallback.contains("fallback"), "{fallback}");

    let error = handle_hook_forward_status(
        reqwest::StatusCode::BAD_GATEWAY,
        "not a guardrail response".into(),
        true,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("HTTP 502"), "{error}");
}

#[test]
fn merge_hooks_is_idempotent_and_preserves_existing_entries() {
    let existing = json!({
        "hooks": {
            "Stop": [{ "hooks": [{ "type": "command", "command": "existing" }] }]
        }
    });
    let generated = generated_hooks(CodingAgent::ClaudeCode, "nemo-relay hook-forward claude");
    let once = merge_hooks(existing, generated.clone()).unwrap();
    let twice = merge_hooks(once.clone(), generated).unwrap();
    assert_eq!(once, twice);
    assert_eq!(twice["hooks"]["Stop"].as_array().unwrap().len(), 2);
    assert_eq!(
        twice["hooks"]["UserPromptExpansion"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn merge_hooks_rejects_malformed_shapes() {
    let generated = generated_hooks(CodingAgent::Codex, "cmd");
    assert!(merge_hooks(json!([]), generated.clone()).is_err());
    assert!(merge_hooks(json!({ "hooks": [] }), generated.clone()).is_err());
    assert!(merge_hooks(json!({ "hooks": { "Stop": {} } }), generated).is_err());
    assert!(merge_hooks(json!({}), json!({ "hooks": [] })).is_err());
}

#[test]
fn helper_formatting_and_headers_cover_optional_paths() {
    assert!(event_matches_tools("PermissionRequest"));
    assert!(!event_matches_tools("SessionStart"));

    let headers = gateway_headers(
        Some("profile"),
        Some(r#"{"team":"obs"}"#),
        Some(GatewayMode::Passthrough),
    )
    .unwrap();
    assert_eq!(
        headers
            .get("x-nemo-relay-gateway-mode")
            .and_then(|value| value.to_str().ok()),
        Some("passthrough")
    );
    assert!(
        insert_header(
            &mut HeaderMap::new(),
            "x-nemo-relay-config-profile",
            Some("bad\nvalue")
        )
        .is_err()
    );

    let headers = gateway_headers(None, None, None).unwrap();
    assert!(headers.is_empty());
}

#[test]
fn generated_hook_dispatch_covers_all_agents() {
    assert_generated_hook_policies();
    let config = "/private/nemo-relay-hook.json";
    assert_eq!(
        transparent_hook_forward_commands_for_platform(
            Path::new("/abs/path/to/nemo-relay"),
            CodingAgent::Codex,
            config,
            false,
        )
        .for_event("PreToolUse"),
        "/abs/path/to/nemo-relay hook-forward codex --hook-config /private/nemo-relay-hook.json --transparent-run --fail-closed"
    );
    let relay = Path::new("/opt/NeMo Relay's & tools/nemo-relay");
    assert_eq!(
        transparent_hook_forward_commands_for_platform(relay, CodingAgent::Codex, config, false)
            .for_event("SessionStart"),
        r#"'/opt/NeMo Relay'\''s & tools/nemo-relay' hook-forward codex --hook-config /private/nemo-relay-hook.json --transparent-run --fail-open"#
    );
    let native =
        transparent_hook_forward_commands(Path::new("nemo-relay"), CodingAgent::Codex, config)
            .unwrap();
    if !cfg!(windows) {
        assert_eq!(
            native,
            transparent_hook_forward_commands_for_platform(
                Path::new("nemo-relay"),
                CodingAgent::Codex,
                config,
                false,
            )
        );
    }
    let windows = transparent_hook_forward_commands_for_platform(
        relay,
        CodingAgent::ClaudeCode,
        config,
        true,
    );
    let windows = windows.for_event("PreToolUse");
    assert!(windows.contains("--hook-config"));
    assert!(!windows.contains("PowerShell"));
    assert!(!windows.contains("EncodedCommand"));
}

fn assert_generated_hook_policies() {
    for agent in [CodingAgent::ClaudeCode, CodingAgent::Codex] {
        assert!(generated_hooks(agent, "cmd")["hooks"].is_object());
        let commands = GeneratedHookCommands::new("cmd --fail-open", "cmd --fail-closed");
        let generated = generated_policy_hooks(agent, &commands);
        for event in agent.hook_events() {
            let command = generated["hooks"][event][0]["hooks"][0]["command"]
                .as_str()
                .unwrap();
            assert_eq!(
                command,
                commands.for_event(event),
                "unexpected policy for {} {event}",
                agent.label()
            );
            assert_eq!(
                command.ends_with("--fail-closed"),
                event_requires_fail_closed(event),
                "unexpected enforcement classification for {} {event}",
                agent.label()
            );
        }
    }
}

#[test]
fn codex_generation_uses_exactly_the_supported_hook_schema() {
    let generated = generated_hooks(CodingAgent::Codex, "cmd");
    let events = generated["hooks"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(
        events,
        std::collections::BTreeSet::from([
            "PermissionRequest",
            "PostCompact",
            "PostToolUse",
            "PreCompact",
            "PreToolUse",
            "SessionStart",
            "Stop",
            "SubagentStart",
            "SubagentStop",
            "UserPromptSubmit",
        ])
    );
    for unsupported in ["PostToolUseFailure", "Notification", "SessionEnd"] {
        assert!(generated["hooks"].get(unsupported).is_none());
    }
}

#[test]
fn packaged_hook_configs_are_valid_json() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../integrations/coding-agents");
    for path in [
        root.join("../../.agents/plugins/marketplace.json"),
        root.join("../../.claude-plugin/marketplace.json"),
        root.join("claude-code/hooks/hooks.json"),
        root.join("codex/hooks/hooks.json"),
        root.join("claude-code/.mcp.json"),
        root.join("codex/.mcp.json"),
        root.join("claude-code/.claude-plugin/plugin.json"),
        root.join("codex/.codex-plugin/plugin.json"),
    ] {
        let raw = std::fs::read_to_string(&path).unwrap();
        serde_json::from_str::<Value>(&raw)
            .unwrap_or_else(|error| panic!("{} is invalid JSON: {error}", path.display()));
    }
}

#[test]
fn packaged_plugin_hooks_use_expected_forwarding_commands() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../integrations/coding-agents");
    let claude = serde_json::from_str::<Value>(
        &std::fs::read_to_string(root.join("claude-code/hooks/hooks.json")).unwrap(),
    )
    .unwrap();
    let codex = serde_json::from_str::<Value>(
        &std::fs::read_to_string(root.join("codex/hooks/hooks.json")).unwrap(),
    )
    .unwrap();

    assert_eq!(
        codex["description"],
        json!("SPDX-License-Identifier: Apache-2.0")
    );
    assert_eq!(
        codex.as_object().unwrap().keys().collect::<Vec<_>>(),
        vec!["description", "hooks"]
    );

    assert_eq!(
        claude["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        json!(format!(
            "nemo-relay hook-forward claude --gateway-url {} --forward-only --fail-open",
            crate::bootstrap::DEFAULT_URL
        ))
    );
    assert_eq!(
        codex["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        json!(format!(
            "nemo-relay hook-forward codex --gateway-url {} --forward-only --fail-open",
            crate::bootstrap::DEFAULT_URL
        ))
    );
    assert_eq!(
        claude["hooks"],
        generated_policy_hooks(
            CodingAgent::ClaudeCode,
            &GeneratedHookCommands::new(
                format!(
                    "nemo-relay hook-forward claude --gateway-url {} --forward-only --fail-open",
                    crate::bootstrap::DEFAULT_URL
                ),
                format!(
                    "nemo-relay hook-forward claude --gateway-url {} --forward-only --fail-closed",
                    crate::bootstrap::DEFAULT_URL
                ),
            ),
        )["hooks"]
    );
    assert_eq!(
        codex["hooks"],
        generated_policy_hooks(
            CodingAgent::Codex,
            &GeneratedHookCommands::new(
                format!(
                    "nemo-relay hook-forward codex --gateway-url {} --forward-only --fail-open",
                    crate::bootstrap::DEFAULT_URL
                ),
                format!(
                    "nemo-relay hook-forward codex --gateway-url {} --forward-only --fail-closed",
                    crate::bootstrap::DEFAULT_URL
                ),
            ),
        )["hooks"]
    );
    assert!(
        claude["hooks"]
            .as_object()
            .unwrap()
            .values()
            .flat_map(|groups| groups.as_array().unwrap())
            .flat_map(|group| group["hooks"].as_array().unwrap())
            .all(|hook| hook["command"]
                .as_str()
                .is_some_and(|command| command.starts_with("nemo-relay ")))
    );
    assert!(
        codex["hooks"]
            .as_object()
            .unwrap()
            .values()
            .flat_map(|groups| groups.as_array().unwrap())
            .flat_map(|group| group["hooks"].as_array().unwrap())
            .all(|hook| hook["command"]
                .as_str()
                .is_some_and(|command| command.starts_with("nemo-relay ")))
    );
}

#[test]
fn packaged_plugin_manifests_use_stable_plugin_name_and_version() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../integrations/coding-agents");

    assert_agent_plugin_manifests(&root);
    assert_agent_mcp_manifests(&root);
    assert_agent_marketplace_manifests(&root);
}

fn assert_agent_plugin_manifests(root: &std::path::Path) {
    let claude_path = root.join("claude-code/.claude-plugin/plugin.json");
    let claude =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&claude_path).unwrap()).unwrap();
    assert_eq!(claude["name"], json!("nemo-relay-plugin"));
    assert_eq!(claude["version"], json!(env!("CARGO_PKG_VERSION")));
    assert!(claude.get("hooks").is_none());
    assert_eq!(claude["mcpServers"], json!("./.mcp.json"));

    let codex_path = root.join("codex/.codex-plugin/plugin.json");
    let codex =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&codex_path).unwrap()).unwrap();
    assert_eq!(codex["name"], json!("nemo-relay-plugin"));
    assert_eq!(codex["version"], json!(env!("CARGO_PKG_VERSION")));
    assert!(codex.get("hooks").is_none());
    assert_eq!(codex["mcpServers"], json!("./.mcp.json"));
}

fn assert_agent_mcp_manifests(root: &std::path::Path) {
    let codex_mcp_path = root.join("codex/.mcp.json");
    let codex_mcp =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&codex_mcp_path).unwrap()).unwrap();
    let server = &codex_mcp["nemo-relay"];
    assert_eq!(server["command"], json!("nemo-relay"));
    assert_eq!(server["args"], json!(["mcp"]));
    assert_eq!(
        server["env"],
        json!({"NEMO_RELAY_GATEWAY_BIND": "127.0.0.1:47632"})
    );
    assert_eq!(server["required"], json!(true));
    assert_eq!(server["startup_timeout_sec"], json!(20));
    assert_eq!(
        server["env_vars"],
        json!(crate::mcp_environment::forwarded_names_for_platform(
            Vec::new(),
            None,
            false,
        ))
    );

    let claude_mcp_path = root.join("claude-code/.mcp.json");
    let claude_mcp =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&claude_mcp_path).unwrap()).unwrap();
    let claude_server = &claude_mcp["mcpServers"]["nemo-relay"];
    assert_eq!(claude_server["command"], json!("nemo-relay"));
    assert_eq!(claude_server["args"], json!(["mcp"]));
    assert_eq!(
        claude_server["env"],
        json!({"NEMO_RELAY_GATEWAY_BIND": "127.0.0.1:47632"})
    );
    assert_eq!(claude_server["alwaysLoad"], json!(true));
}

fn assert_agent_marketplace_manifests(root: &std::path::Path) {
    let codex_marketplace_path = root.join("../../.agents/plugins/marketplace.json");
    let codex_marketplace =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&codex_marketplace_path).unwrap())
            .unwrap();
    assert_eq!(codex_marketplace["name"], json!("nemo-relay"));
    assert_eq!(
        codex_marketplace["plugins"][0]["name"],
        json!("nemo-relay-plugin")
    );
    assert_eq!(
        codex_marketplace["plugins"][0]["source"]["path"],
        json!("./integrations/coding-agents/codex")
    );

    let claude_marketplace_path = root.join("../../.claude-plugin/marketplace.json");
    let claude_marketplace =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&claude_marketplace_path).unwrap())
            .unwrap();
    assert_eq!(claude_marketplace["name"], json!("nemo-relay"));
    assert_eq!(
        claude_marketplace["plugins"][0]["name"],
        json!("nemo-relay-plugin")
    );
    assert_eq!(
        claude_marketplace["plugins"][0]["source"],
        json!("./integrations/coding-agents/claude-code")
    );
}

#[test]
fn packaged_plugin_helpers_are_present() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../integrations/coding-agents");
    for path in [
        root.join("claude-code/hooks/hooks.json"),
        root.join("codex/hooks/hooks.json"),
        root.join("claude-code/.mcp.json"),
        root.join("codex/.mcp.json"),
    ] {
        let metadata = std::fs::metadata(&path)
            .unwrap_or_else(|error| panic!("{} missing: {error}", path.display()));
        assert!(metadata.is_file(), "{} is not a file", path.display());
    }
}

#[test]
fn claude_windows_hooks_preserve_drive_and_unc_arguments_in_bash() {
    for (raw, expected) in [
        (
            r"C:\Users\Relay Tools\nemo-relay.exe",
            "C:/Users/Relay Tools/nemo-relay.exe",
        ),
        (
            r"\\?\C:\Users\Relay Tools\nemo-relay.exe",
            "C:/Users/Relay Tools/nemo-relay.exe",
        ),
        (
            r"\\?\UNC\server\share\nemo-relay.exe",
            "//server/share/nemo-relay.exe",
        ),
        (
            r"\\server\share\nemo-relay.exe",
            "//server/share/nemo-relay.exe",
        ),
    ] {
        let config = r"\\?\C:\Users\Relay's $HOME `tools` & %USERPROFILE% !^\hook.json";
        let commands = transparent_hook_forward_commands_for_platform(
            Path::new(raw),
            CodingAgent::ClaudeCode,
            config,
            true,
        );
        let arguments = shell_words::split(commands.for_event("PreToolUse")).unwrap();
        assert_eq!(
            arguments,
            vec![
                expected,
                "hook-forward",
                "claude",
                "--hook-config",
                "C:/Users/Relay's $HOME `tools` & %USERPROFILE% !^/hook.json",
                "--transparent-run",
                "--fail-closed",
            ]
        );
    }
}

#[cfg(any(unix, windows))]
#[test]
fn claude_windows_hooks_execute_in_bash_with_exact_arguments_and_io() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let temp = tempfile::tempdir().unwrap();
    let bin = temp.path().join("Relay's $HOME `tools` & %USERPROFILE% !^");
    std::fs::create_dir(&bin).unwrap();
    let relay = bin.join("nemo-relay.exe");
    // Keep linker output paths simple; only the hook shell should parse the difficult path.
    let compiled_relay = temp.path().join("hook-fixture.exe");
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/windows_hook_relay.rs");
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let compiled = Command::new(rustc)
        .arg(source)
        .args(["--edition", "2024", "-o"])
        .arg(&compiled_relay)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    std::fs::rename(&compiled_relay, &relay).unwrap();

    let generation = bin.join(".nemo-relay-generation");
    let config = persistent_hook_config_path(&generation);
    let expected_config = config.display().to_string().replace('\\', "/");
    let raw_relay = relay.display().to_string().replace('/', "\\");
    let raw_config = format!(r"\\?\{}", config.display().to_string().replace('/', "\\"));
    let persistent = persistent_hook_forward_commands_for_platform(
        Path::new(&raw_relay),
        CodingAgent::ClaudeCode,
        &generation,
        "generation",
        true,
    );
    let transparent = transparent_hook_forward_commands_for_platform(
        Path::new(&raw_relay),
        CodingAgent::ClaudeCode,
        &raw_config,
        true,
    );
    #[cfg(windows)]
    let bash_executable = std::env::var_os("CLAUDE_CODE_GIT_BASH_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            // PATH may resolve bash.exe to the Windows WSL shim instead of Git Bash.
            let git =
                crate::process::resolve_executable("git").expect("Git for Windows is required");
            git.ancestors()
                .skip(1)
                .flat_map(|directory| {
                    [
                        directory.join("bin/bash.exe"),
                        directory.join("usr/bin/bash.exe"),
                    ]
                })
                .find(|candidate| candidate.is_file())
                .unwrap_or_else(|| panic!("Git Bash not found beside {}", git.display()))
        });
    #[cfg(unix)]
    let bash_executable = Path::new("bash");

    for (commands, transparent_run) in [(persistent, false), (transparent, true)] {
        for (event, policy) in [
            ("SessionEnd", "--fail-open"),
            ("PreToolUse", "--fail-closed"),
        ] {
            let marker = temp.path().join("stdin.txt");
            let command = commands.for_event(event);
            let mut bash = Command::new(bash_executable.as_os_str());
            bash.args(["-c", command])
                .env("NEMO_RELAY_HOOK_AGENT", "claude")
                .env("NEMO_RELAY_HOOK_CONFIG", &expected_config)
                .env("NEMO_RELAY_HOOK_POLICY", policy)
                .env("NEMO_RELAY_HOOK_INPUT_MARKER", &marker)
                .env("NEMO_RELAY_HOOK_EMIT_OUTPUT", "1")
                .env("NEMO_RELAY_HOOK_EXIT_CODE", "23")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            if transparent_run {
                bash.env("NEMO_RELAY_HOOK_TRANSPARENT", "1");
            }
            let mut child = bash.spawn().unwrap_or_else(|error| {
                panic!("failed to launch {}: {error}", bash_executable.display())
            });
            child
                .stdin
                .take()
                .unwrap()
                .write_all(b"hook-input\n")
                .unwrap();
            let output = child.wait_with_output().unwrap();
            assert_eq!(
                output.status.code(),
                Some(23),
                "command: {command}\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(std::fs::read(&marker).unwrap(), b"hook-input\n");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout).trim(),
                "hook-stdout"
            );
            assert_eq!(
                String::from_utf8_lossy(&output.stderr).trim(),
                "hook-stderr"
            );
        }
    }
}
