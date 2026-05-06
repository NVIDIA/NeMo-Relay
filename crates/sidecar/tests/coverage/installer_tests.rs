// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn command(agent: CodingAgent, root: &Path) -> InstallCommand {
    InstallCommand {
        agent,
        scope: InstallScope::User,
        target: InstallTarget::Both,
        sidecar_url: "http://127.0.0.1:4040".into(),
        atif_dir: Some(root.join("atif")),
        openinference_endpoint: Some("http://otel:4318/v1/traces".into()),
        profile: Some("default".into()),
        session_metadata: Some(r#"{"team":"agent-observability"}"#.into()),
        plugin_config: Some(r#"{"components":[]}"#.into()),
        gateway_mode: Some(GatewayMode::Required),
        dry_run: false,
        print: false,
        home_dir: Some(root.to_path_buf()),
        project_dir: None,
    }
}

fn project_command(agent: CodingAgent, root: &Path) -> InstallCommand {
    InstallCommand {
        scope: InstallScope::Project,
        project_dir: Some(root.to_path_buf()),
        ..command(agent, root)
    }
}

#[test]
fn generates_claude_install_file() {
    let temp = tempfile::tempdir().unwrap();
    let files = planned_files(&command(CodingAgent::ClaudeCode, temp.path())).unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0].path.ends_with(".claude/settings.json"));
    let json: Value = serde_json::from_str(&files[0].contents).unwrap();
    assert!(json["hooks"]["SessionStart"].is_array());
    assert!(
        json["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap()
            .contains("hook-forward claude-code")
    );
}

#[test]
fn generates_codex_config_and_hooks() {
    let temp = tempfile::tempdir().unwrap();
    let files = planned_files(&command(CodingAgent::Codex, temp.path())).unwrap();
    assert_eq!(files.len(), 2);
    assert!(files[0].contents.contains("codex_hooks = true"));
    let json: Value = serde_json::from_str(&files[1].contents).unwrap();
    assert!(json["hooks"]["Stop"].is_array());
    assert!(
        json["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap()
            .contains("hook-forward codex")
    );
}

#[test]
fn generates_cursor_hooks() {
    let temp = tempfile::tempdir().unwrap();
    let files = planned_files(&command(CodingAgent::Cursor, temp.path())).unwrap();
    assert_eq!(files.len(), 1);
    let json: Value = serde_json::from_str(&files[0].contents).unwrap();
    assert!(json["hooks"]["beforeShellExecution"].is_array());
    assert!(
        json["hooks"]["beforeShellExecution"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap()
            .contains("hook-forward cursor")
    );
}

#[test]
fn merge_hooks_is_idempotent_and_preserves_existing_entries() {
    let existing = json!({
        "hooks": {
            "Stop": [{ "hooks": [{ "type": "command", "command": "existing" }] }]
        }
    });
    let generated = codex_hooks("nemo-flow-sidecar hook-forward codex");
    let once = merge_hooks(existing, generated.clone()).unwrap();
    let twice = merge_hooks(once.clone(), generated).unwrap();
    assert_eq!(once, twice);
    assert_eq!(twice["hooks"]["Stop"].as_array().unwrap().len(), 2);
}

#[test]
fn project_install_uses_project_dir_and_preserves_codex_toml() {
    let temp = tempfile::tempdir().unwrap();
    let codex_dir = temp.path().join(".codex");
    std::fs::create_dir_all(&codex_dir).unwrap();
    std::fs::write(
        codex_dir.join("config.toml"),
        "[features]\nother = true\n[model_providers.openai]\nbase_url = \"http://old\"\n",
    )
    .unwrap();

    let files = planned_files(&project_command(CodingAgent::Codex, temp.path())).unwrap();

    assert!(files[0].path.starts_with(temp.path()));
    assert!(files[0].contents.contains("other = true"));
    assert!(files[0].contents.contains("codex_hooks = true"));
    assert!(files[0].contents.contains("[model_providers.openai]"));
}

#[test]
fn install_writes_file_and_backs_up_existing_config() {
    let temp = tempfile::tempdir().unwrap();
    let claude_dir = temp.path().join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    let settings = claude_dir.join("settings.json");
    std::fs::write(&settings, r#"{"hooks":{"Stop":[]}}"#).unwrap();

    install(command(CodingAgent::ClaudeCode, temp.path())).unwrap();

    let installed = std::fs::read_to_string(&settings).unwrap();
    assert!(installed.contains("hook-forward claude-code"));
    let backups: Vec<_> = std::fs::read_dir(&claude_dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("settings.json.bak."))
        .collect();
    assert_eq!(backups.len(), 1);
}

#[test]
fn install_dry_run_does_not_write_files() {
    let temp = tempfile::tempdir().unwrap();
    let mut command = command(CodingAgent::Cursor, temp.path());
    command.dry_run = true;
    command.print = true;

    install(command).unwrap();

    assert!(!temp.path().join(".cursor/hooks.json").exists());
}

#[test]
fn invalid_json_config_is_rejected_before_planning() {
    let temp = tempfile::tempdir().unwrap();
    let mut command = command(CodingAgent::Codex, temp.path());
    command.session_metadata = Some("not-json".into());

    let error = install(command).unwrap_err().to_string();

    assert!(error.contains("invalid session metadata"));
}

#[test]
fn merge_hooks_rejects_malformed_shapes() {
    assert!(merge_hooks(json!([]), codex_hooks("cmd")).is_err());
    assert!(merge_hooks(json!({ "hooks": [] }), codex_hooks("cmd")).is_err());
    assert!(merge_hooks(json!({ "hooks": { "Stop": {} } }), codex_hooks("cmd")).is_err());
    assert!(merge_hooks(json!({}), json!({ "hooks": [] })).is_err());
}

#[test]
fn invalid_existing_files_are_reported() {
    let temp = tempfile::tempdir().unwrap();
    let cursor_dir = temp.path().join(".cursor");
    std::fs::create_dir_all(&cursor_dir).unwrap();
    std::fs::write(cursor_dir.join("hooks.json"), "not-json").unwrap();

    let error = planned_files(&command(CodingAgent::Cursor, temp.path()))
        .unwrap_err()
        .to_string();

    assert!(error.contains("invalid JSON"));

    let codex_dir = temp.path().join(".codex");
    std::fs::create_dir_all(&codex_dir).unwrap();
    std::fs::write(codex_dir.join("config.toml"), "not = [valid").unwrap();
    let error = planned_files(&command(CodingAgent::Codex, temp.path()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("invalid TOML"));
}

#[test]
fn helper_formatting_and_headers_cover_optional_paths() {
    assert_eq!(shell_quote("plain/arg-1"), "plain/arg-1");
    assert_eq!(shell_quote("needs space"), "'needs space'");
    assert_eq!(shell_quote("can't"), "'can'\\''t'");
    assert!(event_matches_tools("PermissionRequest"));
    assert!(!event_matches_tools("SessionStart"));

    let temp = tempfile::tempdir().unwrap();
    let headers = sidecar_headers(
        Some(temp.path()),
        Some("http://otel"),
        Some("profile"),
        Some(r#"{"team":"obs"}"#),
        Some(r#"{"plugins":[]}"#),
        Some(GatewayMode::Passthrough),
    )
    .unwrap();
    assert_eq!(
        headers
            .get("x-nemo-flow-gateway-mode")
            .and_then(|value| value.to_str().ok()),
        Some("passthrough")
    );
    assert!(
        insert_header(
            &mut HeaderMap::new(),
            "x-nemo-flow-config-profile",
            Some("bad\nvalue")
        )
        .is_err()
    );
}

#[test]
fn packaged_hook_configs_are_valid_json() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../integrations/coding-agents");
    for path in [
        root.join("claude-code/hooks/hooks.json"),
        root.join("codex/hooks/hooks.json"),
        root.join("cursor/.cursor/hooks.json"),
        root.join("claude-code/.claude-plugin/plugin.json"),
        root.join("codex/.codex-plugin/plugin.json"),
    ] {
        let raw = std::fs::read_to_string(&path).unwrap();
        serde_json::from_str::<Value>(&raw)
            .unwrap_or_else(|error| panic!("{} is invalid JSON: {error}", path.display()));
    }
}
