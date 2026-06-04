// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvScope {
    values: Vec<(&'static str, Option<OsString>)>,
}

impl EnvScope {
    fn set(values: &[(&'static str, Option<&std::ffi::OsStr>)]) -> Self {
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
        Self { values: previous }
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

fn empty_report() -> DoctorReport {
    DoctorReport {
        schema_version: 1,
        binary_version: "0.0.0-test",
        target_agent: None,
        environment: EnvironmentInfo {
            os: "macos 25.3.0".into(),
            arch: "aarch64",
            shell: Some("zsh".into()),
        },
        configuration: ConfigurationInfo {
            workspace: ConfigLayer {
                path: PathBuf::from("/x/.nemo-relay/config.toml"),
                status: Status::Info,
                active: false,
                details: "not present".into(),
            },
            global: ConfigLayer {
                path: PathBuf::from("/x/.config/nemo-relay/config.toml"),
                status: Status::Info,
                active: false,
                details: "not present".into(),
            },
            system: ConfigLayer {
                path: PathBuf::from("/etc/nemo-relay/config.toml"),
                status: Status::Info,
                active: false,
                details: "not present".into(),
            },
            resolution: Check {
                name: "Resolution",
                status: Status::Pass,
                details: "valid".into(),
            },
            default_agent: None,
            configured_agents: vec![],
        },
        agents: vec![],
        observability: vec![],
        completions: vec![],
    }
}

#[test]
fn exit_code_passes_when_no_failures() {
    let report = empty_report();
    assert_eq!(exit_code(&report), 0);
}

#[test]
fn exit_code_fails_when_observability_check_fails() {
    let mut report = empty_report();
    report.observability.push(Check {
        name: "ATIF dir",
        status: Status::Fail,
        details: "not writable".into(),
    });
    assert_eq!(exit_code(&report), 1);
}

#[test]
fn exit_code_passes_with_warn_only() {
    let mut report = empty_report();
    report.observability.push(Check {
        name: "OpenInference endpoint",
        status: Status::Warn,
        details: "HTTP 500".into(),
    });
    assert_eq!(exit_code(&report), 0);
}

#[test]
fn exit_code_fails_when_workspace_config_is_invalid() {
    let mut report = empty_report();
    report.configuration.workspace.status = Status::Fail;
    report.configuration.workspace.details = "invalid TOML".into();
    assert_eq!(exit_code(&report), 1);
}

#[test]
fn exit_code_fails_when_config_resolution_fails() {
    let mut report = empty_report();
    report.configuration.resolution.status = Status::Fail;
    report.configuration.resolution.details = "invalid gateway configuration shape".into();
    assert_eq!(exit_code(&report), 1);
}

#[test]
fn exit_code_fails_when_agent_readiness_fails() {
    let mut report = empty_report();
    report.agents.push(AgentInfo {
        name: "codex",
        status: Status::Fail,
        configured: true,
        command: "codex".into(),
        path: None,
        version: None,
        annotation: "configured command not found on $PATH".into(),
    });
    assert_eq!(exit_code(&report), 1);
}

#[test]
fn format_human_emits_fixed_section_order() {
    let report = empty_report();
    let rendered = format_human(&report);

    // Locking in the section order so users can diff `doctor` output across machines.
    let env_idx = rendered.find("Environment").expect("Environment header");
    let cfg_idx = rendered
        .find("Configuration")
        .expect("Configuration header");
    let agents_idx = rendered.find("Agents detected").expect("Agents header");
    let obs_idx = rendered
        .find("Observability")
        .expect("Observability header");
    let comp_idx = rendered.find("Completions").expect("Completions header");

    assert!(env_idx < cfg_idx);
    assert!(cfg_idx < agents_idx);
    assert!(agents_idx < obs_idx);
    assert!(obs_idx < comp_idx);
}

#[test]
fn format_human_reports_all_checks_passed_on_clean_report() {
    let report = empty_report();
    let rendered = format_human(&report);
    assert!(rendered.contains("All checks passed."));
    assert!(!rendered.contains("warnings"));
}

#[test]
fn format_human_uses_symbols_for_agent_statuses() {
    let mut report = empty_report();
    report.agents = vec![
        AgentInfo {
            name: "claude",
            status: Status::Pass,
            configured: true,
            command: "claude".into(),
            path: Some(PathBuf::from("/bin/claude")),
            version: Some("1.0.0".into()),
            annotation: "hooks: injected during run".into(),
        },
        AgentInfo {
            name: "codex",
            status: Status::Info,
            configured: false,
            command: "codex".into(),
            path: None,
            version: None,
            annotation: "not configured".into(),
        },
    ];

    let rendered = format_human(&report);

    assert!(rendered.contains("    ✓  claude"));
    assert!(rendered.contains("    ·  codex"));
    assert!(!rendered.contains("    pass "));
    assert!(!rendered.contains("    info "));
}

#[test]
fn format_human_reports_failure_summary_when_anything_failed() {
    let mut report = empty_report();
    report.observability.push(Check {
        name: "ATIF dir",
        status: Status::Fail,
        details: "not writable".into(),
    });
    let rendered = format_human(&report);
    assert!(rendered.contains("Some checks FAILED"));
}

#[test]
fn format_human_reports_config_resolution_failure() {
    let mut report = empty_report();
    report.configuration.resolution.status = Status::Fail;
    report.configuration.resolution.details =
        "could not resolve merged config: invalid plugin TOML".into();

    let rendered = format_human(&report);

    assert!(rendered.contains("Resolution ✗ could not resolve merged config"));
    assert!(rendered.contains("Some checks FAILED"));
}

#[test]
fn format_human_distinguishes_pass_with_warnings_from_clean_pass() {
    let mut report = empty_report();
    report.observability.push(Check {
        name: "ATIF dir",
        status: Status::Warn,
        details: "directory missing — will be created on first write".into(),
    });
    let rendered = format_human(&report);
    // Exit code stays 0 (warns don't fail), but the footer must call out that warnings exist
    // so users aren't lulled by an "All checks passed." string.
    assert!(rendered.contains("All checks passed"));
    assert!(
        rendered.contains("warnings"),
        "warn-only report should surface the word `warnings` in the footer, got:\n{rendered}"
    );
}

#[test]
fn format_json_is_stable_and_versioned() {
    let report = empty_report();
    let json = format_json(&report).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    // schema_version pins the wire format. Bump only on breaking renames/removals.
    assert_eq!(parsed["schema_version"], 1);
    assert!(parsed["target_agent"].is_null());
    assert!(parsed["environment"]["os"].is_string());
    assert!(parsed["agents"].is_array());
}

#[test]
fn check_dir_writable_does_not_create_missing_dir() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing-atif");

    assert!(check_dir_writable(&missing).is_err());
    assert!(
        !missing.exists(),
        "doctor should not create missing ATIF directories while probing"
    );
}

#[test]
fn layer_status_reports_missing_valid_invalid_and_non_directory_paths() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing.toml");
    assert_eq!(layer_status(&missing).status, Status::Info);

    let valid = temp.path().join("config.toml");
    std::fs::write(&valid, "[upstream]\nopenai_base_url = \"http://local\"\n").unwrap();
    let valid_layer = layer_status(&valid);
    assert_eq!(valid_layer.status, Status::Pass);
    assert!(valid_layer.active);

    let invalid = temp.path().join("invalid.toml");
    std::fs::write(&invalid, "[upstream\n").unwrap();
    let invalid_layer = layer_status(&invalid);
    assert_eq!(invalid_layer.status, Status::Fail);
    assert!(invalid_layer.details.contains("invalid TOML"));

    let dir = temp.path().join("config-dir");
    std::fs::create_dir(&dir).unwrap();
    let dir_layer = layer_status(&dir);
    assert_eq!(dir_layer.status, Status::Fail);
    assert!(
        dir_layer.details.contains("unreadable") || dir_layer.details.contains("Is a directory")
    );
}

#[test]
fn agent_helper_statuses_cover_configured_target_and_hook_paths() {
    assert_eq!(command_executable("codex --full-auto"), "codex");
    assert_eq!(command_executable(""), "");
    assert_eq!(
        agent_command_status(Some(std::path::Path::new("/bin/codex")), false, true),
        Status::Warn
    );
    assert_eq!(agent_command_status(None, true, false), Status::Fail);
    assert_eq!(
        combine_status(Status::Pass, Status::Warn, true),
        Status::Warn
    );
    assert_eq!(
        combine_status(Status::Pass, Status::Warn, false),
        Status::Pass
    );

    let mut agents = AgentConfigs::default();
    agents.hermes.hooks_path = Some(PathBuf::from("/tmp/hermes.yaml"));
    assert!(agent_configured(CodingAgent::Hermes, &agents));
    assert_eq!(configured_agent_names(&agents), vec!["hermes".to_string()]);

    let temp = tempfile::tempdir().unwrap();
    let hook = temp.path().join("hooks.yaml");
    std::fs::write(&hook, "cmd: nemo-relay hook-forward hermes\n").unwrap();
    let (status, details) = hook_file_status(Ok(hook.clone()), CodingAgent::Hermes, true, "hooks");
    assert_eq!(status, Status::Pass);
    assert!(details.contains(hook.to_str().unwrap()));

    std::fs::write(&hook, "cmd: custom\n").unwrap();
    let (status, details) = hook_file_status(Ok(hook.clone()), CodingAgent::Hermes, true, "hooks");
    assert_eq!(status, Status::Fail);
    assert!(details.contains("missing NeMo Relay hook"));
    let (status, _) = hook_file_status(Ok(hook), CodingAgent::Hermes, false, "hooks");
    assert_eq!(status, Status::Info);
}

#[test]
fn collect_completions_reports_shell_specific_paths() {
    let _guard = ENV_LOCK.lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let zsh_completion = temp.path().join(".zfunc/_nemo-relay");
    std::fs::create_dir_all(zsh_completion.parent().unwrap()).unwrap();
    std::fs::write(&zsh_completion, "#compdef nemo-relay\n").unwrap();

    let _env = EnvScope::set(&[("SHELL", Some(std::ffi::OsStr::new("/bin/zsh")))]);
    let checks = collect_completions(Some(temp.path()));
    assert_eq!(checks[0].status, Status::Pass);
    assert!(checks[0].details.contains("_nemo-relay"));

    drop(_env);
    let _env = EnvScope::set(&[("SHELL", Some(std::ffi::OsStr::new("/bin/fish")))]);
    let checks = collect_completions(Some(temp.path()));
    assert_eq!(checks[0].status, Status::Info);
    assert!(checks[0].details.contains("nemo-relay.fish"));

    drop(_env);
    let _env = EnvScope::set(&[("SHELL", None)]);
    let checks = collect_completions(Some(temp.path()));
    assert_eq!(checks[0].status, Status::Info);
    assert!(checks[0].details.contains("no $SHELL"));
}

#[test]
fn observability_component_helpers_cover_disabled_and_default_paths() {
    let plugin = serde_json::json!({
        "version": 1,
        "components": [{
            "kind": OBSERVABILITY_PLUGIN_KIND,
            "enabled": true,
            "config": {
                "version": 1,
                "atof": { "enabled": true },
                "openinference": {
                    "enabled": true,
                    "endpoint": "http://127.0.0.1:1"
                }
            }
        }]
    });
    let config = observability_component_config(&plugin).unwrap();
    assert!(section_enabled(config, "atof"));
    assert_eq!(section_output_directory(config, "atof"), None);
    assert_eq!(
        section_endpoint(config, "openinference").as_deref(),
        Some("http://127.0.0.1:1")
    );
    assert!(
        observability_component_config(&serde_json::json!({
            "components": [{ "kind": "other", "config": {} }]
        }))
        .is_none()
    );
}

#[test]
fn check_directory_reports_pass_warn_and_fail() {
    let temp = tempfile::tempdir().unwrap();
    let pass = check_directory("ATOF dir", temp.path());
    assert_eq!(pass.status, Status::Pass);

    let missing = check_directory("ATOF dir", &temp.path().join("missing"));
    assert_eq!(missing.status, Status::Warn);

    let file = temp.path().join("file");
    std::fs::write(&file, "").unwrap();
    let fail = check_directory("ATOF dir", &file);
    assert_eq!(fail.status, Status::Fail);
}

#[test]
fn cursor_hook_status_rejects_grouped_entries() {
    let temp = tempfile::tempdir().unwrap();
    let hooks_path = temp.path().join("hooks.json");
    std::fs::write(
        &hooks_path,
        r#"{
          "version": 1,
          "hooks": {
            "beforeShellExecution": [
              {
                "matcher": "*",
                "hooks": [
                  {
                    "type": "command",
                    "command": "nemo-relay hook-forward cursor",
                    "timeout": 30
                  }
                ]
              }
            ]
          }
        }"#,
    )
    .unwrap();

    let (status, details) = hook_file_status(
        Ok(hooks_path),
        CodingAgent::Cursor,
        true,
        "hooks: user-managed",
    );

    assert_eq!(status, Status::Fail);
    assert!(details.contains("nested hook groups"));
    assert!(details.contains("direct command entries"));
}

#[test]
fn cursor_hook_status_rejects_any_grouped_entries_when_nemo_hook_is_direct() {
    let temp = tempfile::tempdir().unwrap();
    let hooks_path = temp.path().join("hooks.json");
    std::fs::write(
        &hooks_path,
        r#"{
          "version": 1,
          "hooks": {
            "sessionStart": [
              {
                "command": "nemo-relay hook-forward cursor",
                "timeout": 30
              }
            ],
            "beforeShellExecution": [
              {
                "matcher": "*",
                "hooks": [
                  {
                    "type": "command",
                    "command": "existing-audit-hook",
                    "timeout": 30
                  }
                ]
              }
            ]
          }
        }"#,
    )
    .unwrap();

    let (status, details) = hook_file_status(
        Ok(hooks_path),
        CodingAgent::Cursor,
        true,
        "hooks: user-managed",
    );

    assert_eq!(status, Status::Fail);
    assert!(details.contains("nested hook groups"));
    assert!(details.contains("direct command entries"));
}

#[test]
fn cursor_hook_status_requires_version_one() {
    let temp = tempfile::tempdir().unwrap();
    let hooks_path = temp.path().join("hooks.json");
    std::fs::write(
        &hooks_path,
        r#"{
          "hooks": {
            "beforeShellExecution": [
              {
                "command": "nemo-relay hook-forward cursor",
                "timeout": 30
              }
            ]
          }
        }"#,
    )
    .unwrap();

    let (status, details) = hook_file_status(
        Ok(hooks_path),
        CodingAgent::Cursor,
        true,
        "hooks: user-managed",
    );

    assert_eq!(status, Status::Fail);
    assert!(details.contains("version"));
    assert!(details.contains("1"));
}

#[test]
fn cursor_hook_status_rejects_non_one_version() {
    let temp = tempfile::tempdir().unwrap();
    let hooks_path = temp.path().join("hooks.json");
    std::fs::write(
        &hooks_path,
        r#"{
          "version": 2,
          "hooks": {
            "beforeShellExecution": [
              {
                "command": "nemo-relay hook-forward cursor",
                "timeout": 30
              }
            ]
          }
        }"#,
    )
    .unwrap();

    let (status, details) = hook_file_status(
        Ok(hooks_path),
        CodingAgent::Cursor,
        true,
        "hooks: user-managed",
    );

    assert_eq!(status, Status::Fail);
    assert!(details.contains("version"));
    assert!(details.contains("1"));
}

#[test]
fn cursor_hook_status_accepts_direct_versioned_entries() {
    let temp = tempfile::tempdir().unwrap();
    let hooks_path = temp.path().join("hooks.json");
    std::fs::write(
        &hooks_path,
        r#"{
          "version": 1,
          "hooks": {
            "beforeShellExecution": [
              {
                "command": "nemo-relay hook-forward cursor",
                "timeout": 30
              }
            ]
          }
        }"#,
    )
    .unwrap();

    let (status, details) = hook_file_status(
        Ok(hooks_path),
        CodingAgent::Cursor,
        true,
        "hooks: user-managed",
    );

    assert_eq!(status, Status::Pass);
    assert!(details.contains("installed"));
}

#[tokio::test]
async fn collect_observability_warns_for_missing_atif_dir_without_creating_it() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing-atif");
    let gateway = GatewayConfig {
        plugin_config: Some(serde_json::json!({
            "version": 1,
            "components": [{
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atif": {
                        "enabled": true,
                        "output_directory": missing
                    }
                }
            }]
        })),
        ..GatewayConfig::default()
    };

    let checks = collect_observability(&gateway).await;

    let atif_check = checks
        .iter()
        .find(|check| check.name == "ATIF dir")
        .expect("ATIF directory check");
    assert_eq!(atif_check.status, Status::Warn);
    assert!(!missing.exists());
}

#[tokio::test]
async fn collect_observability_registers_adaptive_before_validation() {
    let gateway = GatewayConfig {
        plugin_config: Some(serde_json::json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": { "version": 1 }
                },
                {
                    "kind": "adaptive",
                    "enabled": false,
                    "config": {
                        "policy": {
                            "unknown_component": "warn",
                            "unknown_field": "warn",
                            "unsupported_value": "error"
                        }
                    }
                }
            ]
        })),
        ..GatewayConfig::default()
    };

    let checks = collect_observability(&gateway).await;

    assert!(
        !checks.iter().any(|check| check
            .details
            .contains("plugin component kind 'adaptive' is unsupported")),
        "doctor should register adaptive before plugin validation: {checks:?}"
    );
}

#[test]
fn format_agents_human_lists_supported_and_separates_detected() {
    let agents = vec![
        AgentInfo {
            name: "claude",
            status: Status::Pass,
            configured: true,
            command: "claude".into(),
            path: Some(PathBuf::from("/opt/homebrew/bin/claude")),
            version: Some("2.1.4".into()),
            annotation: "hooks: injected during run".into(),
        },
        AgentInfo {
            name: "codex",
            status: Status::Info,
            configured: false,
            command: "codex".into(),
            path: None,
            version: None,
            annotation: "not configured".into(),
        },
    ];
    let rendered = format_agents_human(&agents);
    assert!(rendered.contains("Supported"));
    assert!(rendered.contains("Detected on this machine"));
    // Supported lists everything; detected only the one with a path.
    assert!(rendered.contains("claude\n"));
    assert!(rendered.contains("codex\n"));
    assert!(rendered.contains("/opt/homebrew/bin/claude"));
    // codex must NOT show up under the detected block because path is None.
    let detected_block = rendered.split("Detected on this machine").nth(1).unwrap();
    assert!(!detected_block.contains("codex"));
}

#[test]
fn format_agents_json_matches_doctor_agents_shape() {
    let agents = vec![AgentInfo {
        name: "claude",
        status: Status::Pass,
        configured: true,
        command: "claude".into(),
        path: Some(PathBuf::from("/opt/homebrew/bin/claude")),
        version: Some("2.1.4".into()),
        annotation: "hooks: injected during run".into(),
    }];
    let json = format_agents_json(&agents).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.is_array());
    assert_eq!(parsed[0]["name"], "claude");
    assert_eq!(parsed[0]["status"], "pass");
    assert_eq!(parsed[0]["configured"], true);
    assert_eq!(parsed[0]["command"], "claude");
    assert_eq!(parsed[0]["version"], "2.1.4");
    assert_eq!(parsed[0]["path"], "/opt/homebrew/bin/claude");
}
