// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn hermes_config_merge_preserves_existing_yaml() {
    let existing = r#"
model:
  provider: auto
hooks:
  pre_tool_call:
    - command: ~/.hermes/agent-hooks/audit.sh
"#;
    let merged =
        merge_hermes_config(existing, hermes_hooks("nemo-relay hook-forward hermes")).unwrap();
    let yaml: Value = serde_yaml::from_str(&merged).unwrap();

    assert_eq!(yaml["model"]["provider"], json!("auto"));
    assert_eq!(yaml["hooks"]["pre_tool_call"].as_array().unwrap().len(), 2);
    assert_eq!(
        yaml["hooks"]["on_session_finalize"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn hermes_config_merge_rejects_invalid_yaml() {
    let error = merge_hermes_config(
        "hooks: [not valid",
        hermes_hooks("nemo-relay hook-forward hermes"),
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("invalid YAML in Hermes config"));
}

#[test]
fn hermes_hook_forward_prefers_dynamic_env_url() {
    assert_eq!(
        resolve_hook_gateway_url(
            CodingAgent::Hermes,
            Some("http://installed".into()),
            Some("http://dynamic".into()),
        )
        .as_deref(),
        Some("http://dynamic")
    );
    assert_eq!(
        resolve_hook_gateway_url(CodingAgent::Hermes, Some("http://installed".into()), None,)
            .as_deref(),
        Some("http://installed")
    );
    assert_eq!(
        resolve_hook_gateway_url(
            CodingAgent::Codex,
            Some("http://installed".into()),
            Some("http://dynamic".into()),
        )
        .as_deref(),
        Some("http://installed")
    );
}

#[test]
fn merge_hooks_is_idempotent_and_preserves_existing_entries() {
    let existing = json!({
        "hooks": {
            "Stop": [{ "hooks": [{ "type": "command", "command": "existing" }] }]
        }
    });
    let generated = codex_hooks("nemo-relay hook-forward codex");
    let once = merge_hooks(existing, generated.clone()).unwrap();
    let twice = merge_hooks(once.clone(), generated).unwrap();
    assert_eq!(once, twice);
    assert_eq!(twice["hooks"]["Stop"].as_array().unwrap().len(), 2);
}

#[test]
fn merge_hooks_rejects_malformed_shapes() {
    assert!(merge_hooks(json!([]), codex_hooks("cmd")).is_err());
    assert!(merge_hooks(json!({ "hooks": [] }), codex_hooks("cmd")).is_err());
    assert!(merge_hooks(json!({ "hooks": { "Stop": {} } }), codex_hooks("cmd")).is_err());
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
    for agent in [
        CodingAgent::ClaudeCode,
        CodingAgent::Codex,
        CodingAgent::Hermes,
    ] {
        assert!(generated_hooks(agent, "cmd")["hooks"].is_object());
    }
    assert_eq!(
        hook_forward_command("nemo-relay", CodingAgent::Hermes),
        "nemo-relay hook-forward hermes"
    );
    assert_eq!(
        hook_forward_command("/abs/path/to/nemo-relay", CodingAgent::Codex),
        "/abs/path/to/nemo-relay hook-forward codex"
    );
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
        root.join("claude-code/.claude-plugin/plugin.json"),
        root.join("codex/.codex-plugin/plugin.json"),
    ] {
        let raw = std::fs::read_to_string(&path).unwrap();
        serde_json::from_str::<Value>(&raw)
            .unwrap_or_else(|error| panic!("{} is invalid JSON: {error}", path.display()));
    }
}

#[test]
fn packaged_plugin_hooks_use_expected_shim_commands() {
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
        claude["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        json!("nemo-relay plugin-shim hook claude")
    );
    assert_eq!(
        codex["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        json!("nemo-relay plugin-shim hook codex")
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
    let claude_path = root.join("claude-code/.claude-plugin/plugin.json");
    let claude =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&claude_path).unwrap()).unwrap();
    assert_eq!(claude["name"], json!("nemo-relay-plugin"));
    assert_eq!(claude["version"], json!(env!("CARGO_PKG_VERSION")));
    assert!(claude.get("hooks").is_none());

    let codex_path = root.join("codex/.codex-plugin/plugin.json");
    let codex =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&codex_path).unwrap()).unwrap();
    assert_eq!(codex["name"], json!("nemo-relay-plugin"));
    assert_eq!(codex["version"], json!(env!("CARGO_PKG_VERSION")));

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
    ] {
        let metadata = std::fs::metadata(&path)
            .unwrap_or_else(|error| panic!("{} missing: {error}", path.display()));
        assert!(metadata.is_file(), "{} is not a file", path.display());
    }
}
