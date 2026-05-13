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
        metadata: None,
        plugin_config: None,
    }
}

#[test]
fn session_config_prefers_headers_and_parses_json() {
    let mut headers = HeaderMap::new();
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
    assert_eq!(resolved.gateway.metadata, None);
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
fn legacy_observability_config_sections_fail_clearly() {
    let temp = tempfile::tempdir().unwrap();
    for (name, contents, expected) in [
        (
            "exporters.toml",
            "[exporters]\natof_dir = \"atof\"\n",
            "[exporters]",
        ),
        (
            "observability.toml",
            "[observability]\natif_dir = \"atif\"\n",
            "[observability]",
        ),
        (
            "openinference.toml",
            "[export.openinference]\nendpoint = \"http://localhost:4318\"\n",
            "[export.openinference]",
        ),
    ] {
        let path = temp.path().join(name);
        std::fs::write(&path, contents).unwrap();
        let command = RunCommand {
            agent: None,
            config: Some(path),
            openai_base_url: None,
            anthropic_base_url: None,
            session_metadata: None,
            plugin_config: None,
            dry_run: false,
            print: false,
            command: vec![],
        };

        let error = resolve_run_config(&command, None).unwrap_err().to_string();

        assert!(error.contains("legacy observability config"));
        assert!(error.contains(expected));
        assert!(error.contains("plugins.toml"));
        assert!(error.contains("nemo-flow plugins edit"));
    }
}

#[test]
fn explicit_plugins_toml_maps_root_plugin_config() {
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
        temp.path().join("plugins.toml"),
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
fn plugins_toml_path_resolution_tracks_config_scope() {
    let temp = tempfile::tempdir().unwrap();
    let explicit = temp.path().join("custom-config.toml");
    assert_eq!(
        plugin_config_paths(Some(&explicit)),
        vec![temp.path().join("plugins.toml")]
    );

    let project = temp.path().join("workspace");
    let nested = project.join("a/b/c");
    std::fs::create_dir_all(project.join(".nemo-flow")).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    let plugin_path = project.join(".nemo-flow/plugins.toml");
    std::fs::write(&plugin_path, "version = 1").unwrap();
    let user_config = temp.path().join("xdg/nemo-flow");

    assert_eq!(find_project_plugin_config(&nested), Some(plugin_path));
    assert_eq!(
        project_plugin_config_path(&nested),
        project.join(".nemo-flow/plugins.toml")
    );
    assert_eq!(
        implicit_plugin_config_paths(Some(&nested), Some(user_config.clone())),
        vec![
            PathBuf::from("/etc/nemo-flow/plugins.toml"),
            project.join(".nemo-flow/plugins.toml"),
            user_config.join("plugins.toml"),
        ]
    );

    std::fs::remove_file(project.join(".nemo-flow/plugins.toml")).unwrap();
    std::fs::write(project.join(".nemo-flow/config.toml"), "").unwrap();
    assert_eq!(find_project_plugin_config(&nested), None);
    assert_eq!(
        project_plugin_config_path(&nested),
        project.join(".nemo-flow/plugins.toml")
    );
}

#[test]
fn discovered_plugins_toml_upserts_components_by_kind() {
    let temp = tempfile::tempdir().unwrap();
    let project_plugin = temp.path().join("project-plugins.toml");
    let user_plugin = temp.path().join("user-plugins.toml");
    std::fs::write(
        &project_plugin,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
filename = "project.jsonl"

[[components]]
kind = "adaptive"
enabled = true

[components.config]
mode = "project-only"
"#,
    )
    .unwrap();
    std::fs::write(
        &user_plugin,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true

[components.config.atif]
enabled = true
filename_template = "user-{session_id}.json"

[[components]]
kind = "custom"
enabled = true

[components.config]
source = "user"
"#,
    )
    .unwrap();

    let resolved = load_plugin_toml_config_from_paths(vec![project_plugin, user_plugin]).unwrap();

    assert_eq!(
        resolved.map(|config| config.value),
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
                            "filename": "project.jsonl"
                        },
                        "atif": {
                            "enabled": true,
                            "filename_template": "user-{session_id}.json"
                        }
                    }
                },
                {
                    "kind": "adaptive",
                    "enabled": true,
                    "config": {
                        "mode": "project-only"
                    }
                },
                {
                    "kind": "custom",
                    "enabled": true,
                    "config": {
                        "source": "user"
                    }
                }
            ]
        }))
    );
}

#[test]
fn discovered_plugins_toml_can_disable_lower_priority_observability_section() {
    let temp = tempfile::tempdir().unwrap();
    let project_plugin = temp.path().join("project-plugins.toml");
    let user_plugin = temp.path().join("user-plugins.toml");
    std::fs::write(
        &project_plugin,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
output_directory = "project-atof"
mode = "overwrite"
"#,
    )
    .unwrap();
    std::fs::write(
        &user_plugin,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = false
mode = "append"
"#,
    )
    .unwrap();

    let resolved = load_plugin_toml_config_from_paths(vec![project_plugin, user_plugin]).unwrap();

    assert_eq!(
        resolved.map(|config| config.value),
        Some(json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": {
                        "version": 1,
                        "atof": {
                            "enabled": false,
                            "output_directory": "project-atof",
                            "mode": "append"
                        }
                    }
                }
            ]
        }))
    );
}

#[test]
fn plugins_toml_rejects_duplicate_component_kinds_per_file() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_path = temp.path().join("plugins.toml");
    std::fs::write(
        &plugin_path,
        r#"
version = 1

[[components]]
kind = "observability"

[[components]]
kind = "observability"
"#,
    )
    .unwrap();

    let error = load_plugin_toml_config_from_paths(vec![plugin_path])
        .unwrap_err()
        .to_string();

    assert!(error.contains("duplicate plugin component kind"));
    assert!(error.contains("observability"));
}

#[test]
fn plugins_toml_conflicts_with_config_toml_plugins_config() {
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
    std::fs::write(temp.path().join("plugins.toml"), "version = 1\n").unwrap();
    let args = ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("plugin config is defined in both"));
    assert!(error.contains("config.toml"));
    assert!(error.contains("plugins.toml"));
}

#[test]
fn cli_plugin_config_conflicts_with_file_plugin_config() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(temp.path().join("plugins.toml"), "version = 1\n").unwrap();
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: Some(config_path),
        openai_base_url: None,
        anthropic_base_url: None,
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
"#,
    )
    .unwrap();
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: Some(path),
        openai_base_url: Some("http://cli-openai".into()),
        anthropic_base_url: None,
        session_metadata: Some(r#"{"team":"cli"}"#.into()),
        plugin_config: None,
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(resolved.gateway.openai_base_url, "http://cli-openai");
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
fn run_plugin_config_overrides_inherited_top_level_plugin_config() {
    let server = ServerArgs {
        plugin_config: Some(r#"{"components":["top-level"]}"#.into()),
        ..ServerArgs::default()
    };
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config: Some(r#"{"components":["run"]}"#.into()),
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, Some(&server)).unwrap();

    assert_eq!(
        resolved.gateway.plugin_config,
        Some(json!({ "components": ["run"] }))
    );
}

#[test]
fn server_resolution_applies_all_server_overrides() {
    let args = ServerArgs {
        config: None,
        bind: Some("127.0.0.1:0".parse().unwrap()),
        openai_base_url: Some("http://cli-openai".into()),
        anthropic_base_url: Some("http://cli-anthropic".into()),
        plugin_config: Some(r#"{"version":1,"components":[]}"#.into()),
    };

    let resolved = resolve_server_config(&args).unwrap();

    assert_eq!(resolved.gateway.bind.to_string(), "127.0.0.1:0");
    assert_eq!(resolved.gateway.openai_base_url, "http://cli-openai");
    assert_eq!(resolved.gateway.anthropic_base_url, "http://cli-anthropic");
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
        session_metadata: Some(r#"{"team":"run"}"#.into()),
        plugin_config: Some(r#"{"components":["x"]}"#.into()),
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(resolved.gateway.openai_base_url, "http://run-openai");
    assert_eq!(resolved.gateway.anthropic_base_url, "http://run-anthropic");
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

    let plugin_config = temp.path().join("config-with-invalid-plugins.toml");
    std::fs::write(&plugin_config, "").unwrap();
    std::fs::write(temp.path().join("plugins.toml"), "version = [").unwrap();
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

[plugins.config]
version = 1
policy = { unknown_component = "warn", unknown_field = "warn" }
"#
    .parse::<toml::Table>()
    .map(toml::Value::Table)
    .unwrap();
    let right: toml::Value = r#"
[upstream]
openai_base_url = "http://new"

[plugins.config.policy]
unknown_component = "error"
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
        left["plugins"]["config"]["policy"]["unknown_component"].as_str(),
        Some("error")
    );
    assert_eq!(
        left["plugins"]["config"]["policy"]["unknown_field"].as_str(),
        Some("warn")
    );
}
