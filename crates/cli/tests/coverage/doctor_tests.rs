// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::path::PathBuf;

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
                path: PathBuf::from("/x/.nemo-flow/config.toml"),
                status: Status::Info,
                active: false,
                details: "not present".into(),
            },
            global: ConfigLayer {
                path: PathBuf::from("/x/.config/nemo-flow/config.toml"),
                status: Status::Info,
                active: false,
                details: "not present".into(),
            },
            system: ConfigLayer {
                path: PathBuf::from("/etc/nemo-flow/config.toml"),
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

    assert!(checks.iter().any(|check| check.status == Status::Warn));
    assert!(!missing.exists());
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
