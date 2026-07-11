// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
    );
    assert_eq!(destination.gateway_url, "http://installed");
    assert!(destination.recover);

    let destination = resolve_hook_destination(None, Some("http://dynamic".into()));
    assert_eq!(destination.gateway_url, "http://dynamic");
    assert!(!destination.recover);
}

#[test]
fn merge_hooks_is_idempotent_and_preserves_existing_entries() {
    let existing = json!({
        "hooks": {
            "Stop": [{ "hooks": [{ "type": "command", "command": "existing" }] }]
        }
    });
    let generated = claude_hooks("nemo-relay hook-forward claude");
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
        transparent_hook_forward_command(Path::new("nemo-relay"), CodingAgent::Hermes),
        "nemo-relay hook-forward hermes"
    );
    assert_eq!(
        transparent_hook_forward_command(Path::new("/abs/path/to/nemo-relay"), CodingAgent::Codex),
        "/abs/path/to/nemo-relay hook-forward codex"
    );
    let relay = Path::new("/opt/NeMo Relay's & tools/nemo-relay");
    assert_eq!(
        transparent_hook_forward_command_for_platform(relay, CodingAgent::Codex, false),
        r#"'/opt/NeMo Relay'\''s & tools/nemo-relay' hook-forward codex"#
    );
    assert_eq!(
        transparent_hook_forward_command_for_platform(relay, CodingAgent::ClaudeCode, true),
        r#""/opt/NeMo Relay's & tools/nemo-relay" hook-forward claude"#
    );
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
            "nemo-relay hook-forward claude --gateway-url {}",
            crate::sidecar::DEFAULT_URL
        ))
    );
    assert_eq!(
        codex["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        json!(format!(
            "nemo-relay hook-forward codex --gateway-url {}",
            crate::sidecar::DEFAULT_URL
        ))
    );
    assert_eq!(
        claude["hooks"],
        generated_hooks(
            CodingAgent::ClaudeCode,
            &format!(
                "nemo-relay hook-forward claude --gateway-url {}",
                crate::sidecar::DEFAULT_URL
            ),
        )["hooks"]
    );
    assert_eq!(
        codex["hooks"],
        generated_hooks(
            CodingAgent::Codex,
            &format!(
                "nemo-relay hook-forward codex --gateway-url {}",
                crate::sidecar::DEFAULT_URL
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

    let codex_mcp_path = root.join("codex/.mcp.json");
    let codex_mcp =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&codex_mcp_path).unwrap()).unwrap();
    let server = &codex_mcp["nemo-relay"];
    assert_eq!(server["command"], json!("nemo-relay"));
    assert_eq!(server["args"], json!(["mcp", "--agent", "codex"]));
    assert_eq!(
        server["env"],
        json!({"NEMO_RELAY_GATEWAY_BIND": "127.0.0.1:47632"})
    );
    assert_eq!(server["required"], json!(true));
    assert_eq!(server["startup_timeout_sec"], json!(20));
    assert_eq!(
        server["env_vars"],
        json!(crate::mcp_environment::forwarded_names(Vec::new(), None))
    );

    let claude_mcp_path = root.join("claude-code/.mcp.json");
    let claude_mcp =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&claude_mcp_path).unwrap()).unwrap();
    let claude_server = &claude_mcp["mcpServers"]["nemo-relay"];
    assert_eq!(claude_server["command"], json!("nemo-relay"));
    assert_eq!(claude_server["args"], json!(["mcp", "--agent", "claude"]));
    assert_eq!(
        claude_server["env"],
        json!({"NEMO_RELAY_GATEWAY_BIND": "127.0.0.1:47632"})
    );
    assert_eq!(claude_server["alwaysLoad"], json!(true));

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
