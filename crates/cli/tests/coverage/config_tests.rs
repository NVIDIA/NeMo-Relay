// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::http::HeaderValue;
use serde_json::json;

fn config() -> GatewayConfig {
    GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),

        anthropic_base_url: "http://anthropic".into(),
        exporters: ExportersConfig {
            atif: AtifExporterSettings {
                dir: Some(PathBuf::from("default-atif")),
            },
            openinference: OpenInferenceExporterSettings {
                endpoint: Some("http://default-otel".into()),
            },
            ..Default::default()
        },
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

    assert_eq!(
        session.exporters.atif.dir,
        Some(PathBuf::from("header-atif"))
    );
    assert_eq!(
        session.exporters.openinference.endpoint.as_deref(),
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

    assert_eq!(
        session.exporters.atif.dir,
        Some(PathBuf::from("default-atif"))
    );
    assert_eq!(
        session.exporters.openinference.endpoint.as_deref(),
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
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
openai_base_url = "http://openai"
anthropic_base_url = "http://anthropic"

[exporters.atif]
dir = "atif"

[exporters.atof]
dir = "atof"
mode = "overwrite"
filename_template = "{session_id}-events.jsonl"

[exporters.openinference]
endpoint = "http://otel"

[observability]
metadata = { team = "obs" }

[plugins]
config = { components = [] }

[agents.claude]
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

        atof_dir: None,

        openinference_endpoint: None,
        session_metadata: None,
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec![],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(resolved.gateway.bind.to_string(), "127.0.0.1:0");
    assert_eq!(resolved.gateway.openai_base_url, "http://openai");
    assert_eq!(resolved.gateway.anthropic_base_url, "http://anthropic");
    assert_eq!(
        resolved.gateway.exporters.atif.dir,
        Some(PathBuf::from("atif"))
    );
    assert_eq!(
        resolved.gateway.exporters.atof.dir,
        Some(PathBuf::from("atof"))
    );
    assert_eq!(resolved.gateway.exporters.atof.mode.as_str(), "overwrite");
    assert_eq!(
        resolved.gateway.exporters.atof.filename_template,
        "{session_id}-events.jsonl"
    );
    assert_eq!(
        resolved.gateway.exporters.openinference.endpoint.as_deref(),
        Some("http://otel")
    );
    assert_eq!(resolved.gateway.metadata, Some(json!({ "team": "obs" })));
    assert_eq!(
        resolved.gateway.plugin_config,
        Some(json!({ "components": [] }))
    );
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
fn explicit_plugin_toml_maps_root_plugin_config() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(
        &config_path,
        r#"
[upstream]
openai_base_url = "http://openai"
"#,
    )
    .unwrap();
    std::fs::write(
        temp.path().join("plugin.toml"),
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
output_directory = "atof"
filename = "events.jsonl"
mode = "overwrite"
"#,
    )
    .unwrap();
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: Some(config_path),
        openai_base_url: None,
        anthropic_base_url: None,
        atif_dir: None,
        atof_dir: None,
        openinference_endpoint: None,
        session_metadata: None,
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(
        resolved.gateway.plugin_config,
        Some(json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": {
                        "version": 1,
                        "atof": {
                            "enabled": true,
                            "output_directory": "atof",
                            "filename": "events.jsonl",
                            "mode": "overwrite"
                        }
                    }
                }
            ]
        }))
    );
}

#[test]
fn plugin_toml_path_resolution_tracks_config_scope() {
    let temp = tempfile::tempdir().unwrap();
    let explicit = temp.path().join("custom-config.toml");
    assert_eq!(
        plugin_config_paths(Some(&explicit)),
        vec![temp.path().join("plugin.toml")]
    );

    let project = temp.path().join("workspace");
    let nested = project.join("a/b/c");
    std::fs::create_dir_all(project.join(".nemo-flow")).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    let plugin_path = project.join(".nemo-flow/plugin.toml");
    std::fs::write(&plugin_path, "version = 1").unwrap();

    assert_eq!(find_project_plugin_config(&nested), Some(plugin_path));
}

#[test]
fn plugin_toml_conflicts_with_config_toml_plugins_config() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(
        &config_path,
        r#"
[plugins]
config = { version = 1, components = [] }
"#,
    )
    .unwrap();
    std::fs::write(temp.path().join("plugin.toml"), "version = 1\n").unwrap();
    let args = ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("plugin config is defined in both"));
    assert!(error.contains("config.toml"));
    assert!(error.contains("plugin.toml"));
}

#[test]
fn cli_plugin_config_conflicts_with_file_plugin_config() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(temp.path().join("plugin.toml"), "version = 1\n").unwrap();
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: Some(config_path),
        openai_base_url: None,
        anthropic_base_url: None,
        atif_dir: None,
        atof_dir: None,
        openinference_endpoint: None,
        session_metadata: None,
        plugin_config: Some(r#"{"version":1,"components":[]}"#.into()),
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let error = resolve_run_config(&command, None).unwrap_err().to_string();

    assert!(error.contains("--plugin-config"));
    assert!(error.contains("file configuration"));
}

#[test]
fn cli_run_overrides_config_values() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
openai_base_url = "http://file-openai"

[observability]
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
        atof_dir: None,
        openinference_endpoint: None,
        session_metadata: Some(r#"{"team":"cli"}"#.into()),
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(resolved.gateway.openai_base_url, "http://cli-openai");
    assert_eq!(
        resolved.gateway.exporters.atif.dir,
        Some(PathBuf::from("cli-atif"))
    );
    assert_eq!(resolved.gateway.metadata, Some(json!({ "team": "cli" })));
}

#[test]
fn run_inherits_top_level_server_flags_when_subcommand_flags_are_absent() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
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

        atof_dir: None,

        openinference_endpoint: None,
        session_metadata: None,
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, Some(&server)).unwrap();

    assert_eq!(resolved.gateway.openai_base_url, "http://top-level-openai");
}

#[test]
fn server_resolution_applies_all_server_overrides() {
    let args = ServerArgs {
        config: None,
        bind: Some("127.0.0.1:0".parse().unwrap()),
        openai_base_url: Some("http://cli-openai".into()),
        anthropic_base_url: Some("http://cli-anthropic".into()),
        atif_dir: Some(PathBuf::from("cli-atif")),
        atof_dir: None,
        openinference_endpoint: Some("http://cli-otel".into()),
        plugin_config: Some(r#"{"version":1,"components":[]}"#.into()),
    };

    let resolved = resolve_server_config(&args).unwrap();

    assert_eq!(resolved.gateway.bind.to_string(), "127.0.0.1:0");
    assert_eq!(resolved.gateway.openai_base_url, "http://cli-openai");
    assert_eq!(resolved.gateway.anthropic_base_url, "http://cli-anthropic");
    assert_eq!(
        resolved.gateway.exporters.atif.dir,
        Some(PathBuf::from("cli-atif"))
    );
    assert_eq!(
        resolved.gateway.exporters.openinference.endpoint.as_deref(),
        Some("http://cli-otel")
    );
    assert_eq!(
        resolved.gateway.plugin_config,
        Some(json!({ "version": 1, "components": [] }))
    );
    assert!(args.requested_daemon_mode());
}

#[test]
fn run_resolution_applies_all_run_overrides() {
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: Some("http://run-openai".into()),
        anthropic_base_url: Some("http://run-anthropic".into()),
        atif_dir: Some(PathBuf::from("run-atif")),
        atof_dir: None,
        openinference_endpoint: Some("http://run-otel".into()),
        session_metadata: Some(r#"{"team":"run"}"#.into()),
        plugin_config: Some(r#"{"components":["x"]}"#.into()),
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(resolved.gateway.openai_base_url, "http://run-openai");
    assert_eq!(resolved.gateway.anthropic_base_url, "http://run-anthropic");
    assert_eq!(
        resolved.gateway.exporters.atif.dir,
        Some(PathBuf::from("run-atif"))
    );
    assert_eq!(
        resolved.gateway.exporters.openinference.endpoint.as_deref(),
        Some("http://run-otel")
    );
    assert_eq!(resolved.gateway.metadata, Some(json!({ "team": "run" })));
    assert_eq!(
        resolved.gateway.plugin_config,
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
    std::fs::write(&invalid_shape, "upstream = \"not-a-table\"").unwrap();
    let args = ServerArgs {
        config: Some(invalid_shape),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("invalid gateway configuration shape"));

    let plugin_config = temp.path().join("config-with-invalid-plugin.toml");
    std::fs::write(&plugin_config, "").unwrap();
    std::fs::write(temp.path().join("plugin.toml"), "version = [").unwrap();
    let args = ServerArgs {
        config: Some(plugin_config),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("invalid plugin TOML"));
}

#[test]
fn recursive_toml_merge_replaces_scalars_and_preserves_tables() {
    let mut left: toml::Value = r#"
[upstream]
openai_base_url = "http://old"
anthropic_base_url = "http://anthropic"

[observability.metadata]
team = "old"
env = "dev"
"#
    .parse::<toml::Table>()
    .map(toml::Value::Table)
    .unwrap();
    let right: toml::Value = r#"
[upstream]
openai_base_url = "http://new"

[observability.metadata]
team = "new"
"#
    .parse::<toml::Table>()
    .map(toml::Value::Table)
    .unwrap();

    merge_toml(&mut left, right);

    assert_eq!(
        left["upstream"]["openai_base_url"].as_str(),
        Some("http://new")
    );
    assert_eq!(
        left["upstream"]["anthropic_base_url"].as_str(),
        Some("http://anthropic")
    );
    assert_eq!(
        left["observability"]["metadata"]["team"].as_str(),
        Some("new")
    );
    assert_eq!(
        left["observability"]["metadata"]["env"].as_str(),
        Some("dev")
    );
}
