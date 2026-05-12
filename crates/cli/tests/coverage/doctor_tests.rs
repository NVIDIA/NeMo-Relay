// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::path::PathBuf;

fn empty_report() -> DoctorReport {
    DoctorReport {
        schema_version: 1,
        binary_version: "0.0.0-test",
        environment: EnvironmentInfo {
            os: "macos 25.3.0".into(),
            arch: "aarch64",
            shell: Some("zsh".into()),
        },
        configuration: ConfigurationInfo {
            workspace: ConfigLayer {
                path: PathBuf::from("/x/.nemo-flow/config.toml"),
                status: Status::Info,
                details: "not present".into(),
            },
            global: ConfigLayer {
                path: PathBuf::from("/x/.config/nemo-flow/config.toml"),
                status: Status::Info,
                details: "not present".into(),
            },
            system: ConfigLayer {
                path: PathBuf::from("/etc/nemo-flow/config.toml"),
                status: Status::Info,
                details: "not present".into(),
            },
            default_agent: None,
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
fn format_json_is_stable_and_versioned() {
    let report = empty_report();
    let json = format_json(&report).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    // schema_version pins the wire format. Bump only on breaking renames/removals.
    assert_eq!(parsed["schema_version"], 1);
    assert!(parsed["environment"]["os"].is_string());
    assert!(parsed["agents"].is_array());
}

#[test]
fn format_agents_human_lists_supported_and_separates_detected() {
    let agents = vec![
        AgentInfo {
            name: "claude",
            path: Some(PathBuf::from("/opt/homebrew/bin/claude")),
            version: Some("2.1.4".into()),
            annotation: String::new(),
        },
        AgentInfo {
            name: "codex",
            path: None,
            version: None,
            annotation: String::new(),
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
        path: Some(PathBuf::from("/opt/homebrew/bin/claude")),
        version: Some("2.1.4".into()),
        annotation: String::new(),
    }];
    let json = format_agents_json(&agents).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.is_array());
    assert_eq!(parsed[0]["name"], "claude");
    assert_eq!(parsed[0]["version"], "2.1.4");
    assert_eq!(parsed[0]["path"], "/opt/homebrew/bin/claude");
}
