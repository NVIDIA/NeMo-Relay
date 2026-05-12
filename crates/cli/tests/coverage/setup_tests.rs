// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

// Stub-binary detection relies on the Unix executable bit. Windows-side agent presence checks
// use a different mechanism (e.g. `.exe` extension matching), so this lookup test is gated to
// Unix to keep cross-platform CI green; covering the Windows code path is left to a separate
// test once the launcher grows real Windows support.
#[cfg(unix)]
#[test]
fn detect_installed_agents_finds_binaries_on_path() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    // Drop stub binaries for two of the four supported agents — confirming detection picks up
    // only the ones present and ignores the others.
    for exec in ["claude", "cursor-agent"] {
        let path = temp.path().join(exec);
        std::fs::write(&path, "#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    // SAFETY: we restore PATH on drop via the guard below. Tests are not run concurrently
    // within the same binary by default (cargo test --jobs 1 for parallel safety), and we
    // do not assert on agent ordering or unrelated PATH entries.
    let original_path = std::env::var_os("PATH");
    unsafe {
        std::env::set_var("PATH", temp.path());
    }

    let detected = detect_installed_agents();
    assert!(detected.contains(&CodingAgent::ClaudeCode));
    assert!(detected.contains(&CodingAgent::Cursor));
    assert!(!detected.contains(&CodingAgent::Codex));
    assert!(!detected.contains(&CodingAgent::Hermes));

    unsafe {
        if let Some(value) = original_path {
            std::env::set_var("PATH", value);
        } else {
            std::env::remove_var("PATH");
        }
    }
}

#[test]
fn build_config_emits_observability_section_when_atif_selected() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![],
        backends: vec![ObservabilityBackend::Atif],
        openinference_endpoint: None,
        openai_base_url: None,
        hermes_hooks_path: None,
    };

    let doc = build_config(&answers);
    let rendered = doc.to_string();

    assert!(rendered.contains("[observability]"));
    assert!(rendered.contains(r#"atif_dir = "./atif""#));
    assert!(!rendered.contains("[export"));
}

#[test]
fn build_config_emits_export_section_when_openinference_selected() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![],
        backends: vec![ObservabilityBackend::OpenInference],
        openinference_endpoint: Some("http://localhost:6006/v1/traces".into()),
        openai_base_url: None,
        hermes_hooks_path: None,
    };

    let doc = build_config(&answers);
    let rendered = doc.to_string();

    assert!(rendered.contains("[export.openinference]"));
    assert!(rendered.contains(r#"endpoint = "http://localhost:6006/v1/traces""#));
}

#[test]
fn build_config_skips_empty_sections_when_no_backends_selected() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![],
        backends: vec![],
        openinference_endpoint: None,
        openai_base_url: None,
        hermes_hooks_path: None,
    };

    let doc = build_config(&answers);
    let rendered = doc.to_string();

    assert!(!rendered.contains("[observability]"));
    assert!(!rendered.contains("[export"));
    assert!(!rendered.contains("[agents]"));
}

#[test]
fn build_config_emits_agents_block_with_user_facing_keys() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::ClaudeCode, CodingAgent::Codex],
        backends: vec![],
        openinference_endpoint: None,
        openai_base_url: None,
        hermes_hooks_path: None,
    };

    let doc = build_config(&answers);
    let rendered = doc.to_string();

    // Agent keys match the user-facing CLI shortcut names (`claude`, not `claude-code`).
    assert!(rendered.contains("[agents.claude]"));
    assert!(rendered.contains(r#"command = "claude""#));
    assert!(rendered.contains("[agents.codex]"));
    assert!(rendered.contains(r#"command = "codex""#));
}

#[test]
fn build_config_writes_upstream_block_for_custom_openai_base_url() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::Codex],
        backends: vec![ObservabilityBackend::Atif],
        openinference_endpoint: None,
        openai_base_url: Some("https://litellm.internal/v1".into()),
        hermes_hooks_path: None,
    };
    let rendered = build_config(&answers).to_string();
    assert!(rendered.contains("[upstream]"));
    assert!(rendered.contains(r#"openai_base_url = "https://litellm.internal/v1""#));
}

#[test]
fn build_config_omits_upstream_block_when_openai_base_url_is_none() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::Codex],
        backends: vec![ObservabilityBackend::Atif],
        openinference_endpoint: None,
        openai_base_url: None,
        hermes_hooks_path: None,
    };
    let rendered = build_config(&answers).to_string();
    assert!(!rendered.contains("[upstream]"));
}

#[test]
fn save_config_writes_project_scope_to_workspace_dir() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::ClaudeCode],
        backends: vec![ObservabilityBackend::Atif],
        openinference_endpoint: None,
        openai_base_url: None,
        hermes_hooks_path: None,
    };
    let doc = build_config(&answers);
    let temp = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();

    let written = save_config(&doc, ConfigScope::Project, temp.path(), home.path(), None).unwrap();

    assert_eq!(written.len(), 1);
    assert_eq!(written[0], temp.path().join(".nemo-flow/config.toml"));
    let contents = std::fs::read_to_string(&written[0]).unwrap();
    assert!(contents.contains("[observability]"));
    assert!(contents.contains("[agents.claude]"));
}

#[test]
fn save_config_scoped_merge_preserves_other_agents() {
    // Seed an existing config with claude AND codex blocks, plus a custom [upstream] that the
    // wizard does not touch. Then "re-run" the wizard scoped to claude and assert codex +
    // upstream survive while claude is updated and observability is written fresh.
    let temp = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let project_dir = temp.path().join(".nemo-flow");
    std::fs::create_dir_all(&project_dir).unwrap();
    let existing_path = project_dir.join("config.toml");
    std::fs::write(
        &existing_path,
        r#"[upstream]
openai_base_url = "http://old-openai"

[agents.claude]
command = "old-claude-binary"

[agents.codex]
command = "codex --full-auto"
"#,
    )
    .unwrap();

    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::ClaudeCode],
        backends: vec![ObservabilityBackend::Atif],
        openinference_endpoint: None,
        openai_base_url: None,
        hermes_hooks_path: None,
    };
    let doc = build_config(&answers);
    save_config(
        &doc,
        ConfigScope::Project,
        temp.path(),
        home.path(),
        Some(CodingAgent::ClaudeCode),
    )
    .unwrap();

    let merged = std::fs::read_to_string(&existing_path).unwrap();
    // Wizard-owned sections are replaced with the new doc's content.
    assert!(merged.contains("[observability]"));
    assert!(merged.contains("[agents.claude]"));
    assert!(merged.contains(r#"command = "claude""#));
    // Other agents and untouched sections survive.
    assert!(
        merged.contains("[agents.codex]"),
        "expected scoped merge to preserve [agents.codex], got:\n{merged}"
    );
    assert!(
        merged.contains("codex --full-auto"),
        "expected scoped merge to preserve codex command, got:\n{merged}"
    );
    assert!(
        merged.contains("http://old-openai"),
        "expected scoped merge to preserve untouched [upstream], got:\n{merged}"
    );
    // Old claude command should be gone.
    assert!(
        !merged.contains("old-claude-binary"),
        "expected scoped merge to overwrite [agents.claude].command, got:\n{merged}"
    );
}

#[test]
fn save_config_writes_both_scopes_when_both_selected() {
    let answers = SetupAnswers {
        scope: ConfigScope::Both,
        agents: vec![],
        backends: vec![ObservabilityBackend::Atif],
        openinference_endpoint: None,
        openai_base_url: None,
        hermes_hooks_path: None,
    };
    let doc = build_config(&answers);
    let cwd = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();

    let written = save_config(&doc, ConfigScope::Both, cwd.path(), home.path(), None).unwrap();

    assert_eq!(written.len(), 2);
    assert!(written.iter().any(|p| p.starts_with(cwd.path())));
    assert!(written.iter().any(|p| p.starts_with(home.path())));
}

#[test]
fn build_config_emits_hooks_path_for_hermes_when_set() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::Hermes],
        backends: vec![],
        openinference_endpoint: None,
        openai_base_url: None,
        hermes_hooks_path: Some(std::path::PathBuf::from("/tmp/proj/.hermes/config.yaml")),
    };
    let rendered = build_config(&answers).to_string();
    assert!(rendered.contains("[agents.hermes]"));
    assert!(rendered.contains(r#"hooks_path = "/tmp/proj/.hermes/config.yaml""#));
}

#[test]
fn install_hermes_hooks_writes_yaml_and_merges_existing() {
    let cwd = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    // Seed an existing hermes config so we can verify the merge preserves user state.
    let project_hermes = cwd.path().join(".hermes");
    std::fs::create_dir_all(&project_hermes).unwrap();
    std::fs::write(
        project_hermes.join("config.yaml"),
        "model:\n  provider: auto\n",
    )
    .unwrap();

    let written = install_hermes_hooks(ConfigScope::Both, cwd.path(), home.path()).unwrap();

    assert_eq!(written.len(), 2);
    let project_yaml = std::fs::read_to_string(cwd.path().join(".hermes/config.yaml")).unwrap();
    assert!(project_yaml.contains("nemo-flow hook-forward hermes"));
    assert!(
        project_yaml.contains("provider: auto"),
        "existing model block must survive merge"
    );
    let home_yaml = std::fs::read_to_string(home.path().join(".hermes/config.yaml")).unwrap();
    assert!(home_yaml.contains("nemo-flow hook-forward hermes"));
}
