// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::http::HeaderValue;
use serde_json::json;

fn config() -> SidecarConfig {
    SidecarConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),
        anthropic_base_url: "http://anthropic".into(),
        atif_dir: Some(PathBuf::from("default-atif")),
        openinference_endpoint: Some("http://default-otel".into()),
        metadata: None,
        plugin_config: None,
    }
}

#[test]
fn session_config_prefers_headers_and_parses_json() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-flow-atif-dir",
        HeaderValue::from_static("header-atif"),
    );
    headers.insert(
        "x-nemo-flow-openinference-endpoint",
        HeaderValue::from_static("http://header-otel"),
    );
    headers.insert(
        "x-nemo-flow-config-profile",
        HeaderValue::from_static("profile-a"),
    );
    headers.insert(
        "x-nemo-flow-session-metadata",
        HeaderValue::from_static(r#"{"team":"obs"}"#),
    );
    headers.insert(
        "x-nemo-flow-plugin-config",
        HeaderValue::from_static(r#"{"components":[]}"#),
    );
    headers.insert(
        "x-nemo-flow-gateway-mode",
        HeaderValue::from_static("required"),
    );

    let session = config().session_config_from_headers(&headers);

    assert_eq!(session.atif_dir, Some(PathBuf::from("header-atif")));
    assert_eq!(
        session.openinference_endpoint.as_deref(),
        Some("http://header-otel")
    );
    assert_eq!(session.profile.as_deref(), Some("profile-a"));
    assert_eq!(session.metadata, Some(json!({ "team": "obs" })));
    assert_eq!(session.plugin_config, Some(json!({ "components": [] })));
    assert_eq!(session.gateway_mode.as_deref(), Some("required"));
}

#[test]
fn session_config_uses_defaults_and_ignores_bad_json() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-flow-session-metadata",
        HeaderValue::from_static("not-json"),
    );
    headers.insert("x-empty", HeaderValue::from_static(""));

    let session = config().session_config_from_headers(&headers);

    assert_eq!(session.atif_dir, Some(PathBuf::from("default-atif")));
    assert_eq!(
        session.openinference_endpoint.as_deref(),
        Some("http://default-otel")
    );
    assert_eq!(session.metadata, None);
    assert_eq!(header_string(&headers, "x-empty"), None);
}

#[test]
fn agent_and_gateway_mode_arguments_are_stable() {
    assert_eq!(CodingAgent::ClaudeCode.hook_path(), "/hooks/claude-code");
    assert_eq!(CodingAgent::Codex.hook_path(), "/hooks/codex");
    assert_eq!(CodingAgent::Cursor.hook_path(), "/hooks/cursor");
    assert_eq!(CodingAgent::Hermes.hook_path(), "/hooks/hermes");
    assert_eq!(GatewayMode::HookOnly.as_arg(), "hook-only");
    assert_eq!(GatewayMode::Passthrough.as_arg(), "passthrough");
    assert_eq!(GatewayMode::Required.as_arg(), "required");
}

#[test]
fn agent_inference_uses_executable_basename() {
    assert_eq!(
        CodingAgent::infer("/opt/bin/claude"),
        Some(CodingAgent::ClaudeCode)
    );
    assert_eq!(CodingAgent::infer("codex"), Some(CodingAgent::Codex));
    assert_eq!(
        CodingAgent::infer("cursor-agent"),
        Some(CodingAgent::Cursor)
    );
    assert_eq!(CodingAgent::infer("hermes"), Some(CodingAgent::Hermes));
    assert_eq!(CodingAgent::infer("wrapper"), None);
}

#[test]
fn explicit_toml_config_maps_supported_sections() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("sidecar.toml");
    std::fs::write(
        &path,
        r#"
[server]
openai_base_url = "http://openai"
anthropic_base_url = "http://anthropic"

[session]
atif_dir = "atif"
metadata = { team = "obs" }
plugin_config = { components = [] }

[export.openinference]
endpoint = "http://otel"

[agents.claude-code]
command = "claude"

[agents.codex]
command = "codex --approval-mode never"

[agents.cursor]
command = "cursor-agent"
patch_restore_hooks = false

[agents.hermes]
command = "hermes --yolo chat"
"#,
    )
    .unwrap();
    let command = RunCommand {
        agent: None,
        config: Some(path),
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

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(resolved.sidecar.bind.to_string(), "127.0.0.1:0");
    assert_eq!(resolved.sidecar.openai_base_url, "http://openai");
    assert_eq!(resolved.sidecar.anthropic_base_url, "http://anthropic");
    assert_eq!(resolved.sidecar.atif_dir, Some(PathBuf::from("atif")));
    assert_eq!(
        resolved.sidecar.openinference_endpoint.as_deref(),
        Some("http://otel")
    );
    assert_eq!(resolved.sidecar.metadata, Some(json!({ "team": "obs" })));
    assert_eq!(
        resolved.agents.codex.command.as_deref(),
        Some("codex --approval-mode never")
    );
    assert_eq!(
        resolved.agents.hermes.command.as_deref(),
        Some("hermes --yolo chat")
    );
    assert!(!resolved.agents.cursor.patch_restore_hooks);
}

#[test]
fn cli_run_overrides_config_values() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("sidecar.toml");
    std::fs::write(
        &path,
        r#"
[server]
openai_base_url = "http://file-openai"

[session]
atif_dir = "file-atif"
metadata = { team = "file" }
"#,
    )
    .unwrap();
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: Some(path),
        openai_base_url: Some("http://cli-openai".into()),
        anthropic_base_url: None,
        atif_dir: Some(PathBuf::from("cli-atif")),
        openinference_endpoint: None,
        session_metadata: Some(r#"{"team":"cli"}"#.into()),
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(resolved.sidecar.openai_base_url, "http://cli-openai");
    assert_eq!(resolved.sidecar.atif_dir, Some(PathBuf::from("cli-atif")));
    assert_eq!(resolved.sidecar.metadata, Some(json!({ "team": "cli" })));
}

#[test]
fn run_inherits_top_level_server_flags_when_subcommand_flags_are_absent() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("sidecar.toml");
    std::fs::write(
        &path,
        r#"
[server]
openai_base_url = "http://file-openai"
"#,
    )
    .unwrap();
    let server = ServerArgs {
        config: Some(path),
        openai_base_url: Some("http://top-level-openai".into()),
        ..ServerArgs::default()
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
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, Some(&server)).unwrap();

    assert_eq!(resolved.sidecar.openai_base_url, "http://top-level-openai");
}

#[test]
fn server_resolution_applies_all_server_overrides() {
    let args = ServerArgs {
        config: None,
        bind: Some("127.0.0.1:0".parse().unwrap()),
        openai_base_url: Some("http://cli-openai".into()),
        anthropic_base_url: Some("http://cli-anthropic".into()),
        atif_dir: Some(PathBuf::from("cli-atif")),
        openinference_endpoint: Some("http://cli-otel".into()),
    };

    let resolved = resolve_server_config(&args).unwrap();

    assert_eq!(resolved.sidecar.bind.to_string(), "127.0.0.1:0");
    assert_eq!(resolved.sidecar.openai_base_url, "http://cli-openai");
    assert_eq!(resolved.sidecar.anthropic_base_url, "http://cli-anthropic");
    assert_eq!(resolved.sidecar.atif_dir, Some(PathBuf::from("cli-atif")));
    assert_eq!(
        resolved.sidecar.openinference_endpoint.as_deref(),
        Some("http://cli-otel")
    );
}

#[test]
fn run_resolution_applies_all_run_overrides() {
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: Some("http://run-openai".into()),
        anthropic_base_url: Some("http://run-anthropic".into()),
        atif_dir: Some(PathBuf::from("run-atif")),
        openinference_endpoint: Some("http://run-otel".into()),
        session_metadata: Some(r#"{"team":"run"}"#.into()),
        plugin_config: Some(r#"{"components":["x"]}"#.into()),
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(resolved.sidecar.openai_base_url, "http://run-openai");
    assert_eq!(resolved.sidecar.anthropic_base_url, "http://run-anthropic");
    assert_eq!(resolved.sidecar.atif_dir, Some(PathBuf::from("run-atif")));
    assert_eq!(
        resolved.sidecar.openinference_endpoint.as_deref(),
        Some("http://run-otel")
    );
    assert_eq!(resolved.sidecar.metadata, Some(json!({ "team": "run" })));
    assert_eq!(
        resolved.sidecar.plugin_config,
        Some(json!({ "components": ["x"] }))
    );
}

#[test]
fn malformed_shared_config_reports_context() {
    let temp = tempfile::tempdir().unwrap();
    let invalid_toml = temp.path().join("invalid.toml");
    std::fs::write(&invalid_toml, "server = [").unwrap();
    let args = ServerArgs {
        config: Some(invalid_toml),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("invalid TOML"));

    let invalid_shape = temp.path().join("invalid-shape.toml");
    std::fs::write(&invalid_shape, "server = \"not-a-table\"").unwrap();
    let args = ServerArgs {
        config: Some(invalid_shape),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("invalid sidecar configuration shape"));
}

#[test]
fn recursive_toml_merge_replaces_scalars_and_preserves_tables() {
    let mut left: toml::Value = r#"
[server]
openai_base_url = "http://old"
anthropic_base_url = "http://anthropic"

[session.metadata]
team = "old"
env = "dev"
"#
    .parse::<toml::Table>()
    .map(toml::Value::Table)
    .unwrap();
    let right: toml::Value = r#"
[server]
openai_base_url = "http://new"

[session.metadata]
team = "new"
"#
    .parse::<toml::Table>()
    .map(toml::Value::Table)
    .unwrap();

    merge_toml(&mut left, right);

    assert_eq!(
        left["server"]["openai_base_url"].as_str(),
        Some("http://new")
    );
    assert_eq!(
        left["server"]["anthropic_base_url"].as_str(),
        Some("http://anthropic")
    );
    assert_eq!(left["session"]["metadata"]["team"].as_str(), Some("new"));
    assert_eq!(left["session"]["metadata"]["env"].as_str(), Some("dev"));
}
