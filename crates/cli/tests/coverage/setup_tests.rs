// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_support::CwdTestScope as CwdScope;
use std::ffi::OsString;

// Tests that exercise the global-config write path clear `$XDG_CONFIG_HOME`
// because CI runners commonly set it to a real `/home/runner/.config` path.
struct XdgScope {
    _guard: std::sync::MutexGuard<'static, ()>,
    prev: Option<std::ffi::OsString>,
}

impl XdgScope {
    fn cleared() -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let prev = std::env::var_os("XDG_CONFIG_HOME");
        unsafe {
            std::env::remove_var("XDG_CONFIG_HOME");
        }
        Self {
            _guard: guard,
            prev,
        }
    }
}

impl Drop for XdgScope {
    fn drop(&mut self) {
        unsafe {
            match self.prev.take() {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
    }
}

struct EnvScope {
    _guard: std::sync::MutexGuard<'static, ()>,
    values: Vec<(&'static str, Option<OsString>)>,
}

impl EnvScope {
    fn set(values: &[(&'static str, Option<&std::ffi::OsStr>)]) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
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
        Self {
            _guard: guard,
            values: previous,
        }
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

// Stub-binary detection relies on the Unix executable bit. Windows-side agent presence checks
// use a different mechanism (e.g. `.exe` extension matching), so this lookup test is gated to
// Unix to keep cross-platform CI green; covering the Windows code path is left to a separate
// test once the launcher grows real Windows support.
#[cfg(unix)]
#[test]
fn detect_installed_agents_finds_binaries_on_path() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    // Drop stub binaries for two of the three supported agents — confirming detection picks up
    // only the ones present and ignores the others.
    for exec in ["claude", "hermes"] {
        let path = temp.path().join(exec);
        std::fs::write(&path, "#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    // Use the pure-function variant that takes PATH as an arg instead of mutating the global
    // env var. Tests run in parallel by default; touching `std::env::set_var("PATH", ...)` would
    // race with every other test that reads the environment.
    let detected = detect_installed_agents_in(Some(temp.path().as_os_str()));
    assert!(detected.contains(&CodingAgent::ClaudeCode));
    assert!(detected.contains(&CodingAgent::Hermes));
    assert!(!detected.contains(&CodingAgent::Codex));
}

#[test]
fn detect_installed_agents_handles_missing_path() {
    assert!(detect_installed_agents_in(None).is_empty());
}

#[test]
fn build_config_does_not_emit_observability_exporters() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![],
    };

    let rendered = build_config(&answers).to_string();

    assert!(!rendered.contains("[exporters]"));
    assert!(!rendered.contains("[export."));
    assert!(!rendered.contains("[observability]"));
    assert!(!rendered.contains("[exporters.atif]"));
    assert!(!rendered.contains("[exporters.openinference]"));
}

#[test]
fn build_config_skips_empty_sections_when_no_backends_selected() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![],
    };

    let doc = build_config(&answers);
    let rendered = doc.to_string();

    assert!(!rendered.contains("[exporters]"));
    assert!(!rendered.contains("[observability]"));
    assert!(!rendered.contains("[export"));
    assert!(!rendered.contains("[agents]"));
}

#[test]
fn build_config_emits_agents_block_with_user_facing_keys() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::ClaudeCode, CodingAgent::Codex],
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
fn save_config_writes_project_scope_to_workspace_dir() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::ClaudeCode],
    };
    let doc = build_config(&answers);
    let temp = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();

    let written = save_config(&doc, ConfigScope::Project, temp.path(), home.path(), None).unwrap();

    assert_eq!(written.len(), 1);
    assert_eq!(written[0], temp.path().join(".nemo-relay/config.toml"));
    let contents = std::fs::read_to_string(&written[0]).unwrap();
    assert!(!contents.contains("[exporters]"));
    assert!(contents.contains("[agents.claude]"));
}

#[test]
fn save_config_scoped_merge_preserves_other_agents() {
    // Seed an existing config with claude AND codex blocks, plus a custom [upstream] that the
    // wizard does not touch. Then "re-run" the wizard scoped to claude and assert codex +
    // upstream survive while claude is updated and observability is written fresh.
    let temp = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let project_dir = temp.path().join(".nemo-relay");
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
    assert!(!merged.contains("[exporters]"));
    assert!(merged.contains("[agents.claude]"));
    assert!(merged.contains(r#"command = "claude""#));
    // Other agents (not touched by this scoped run) survive.
    assert!(
        merged.contains("[agents.codex]"),
        "expected scoped merge to preserve [agents.codex], got:\n{merged}"
    );
    assert!(
        merged.contains("codex --full-auto"),
        "expected scoped merge to preserve codex command, got:\n{merged}"
    );
    // Setup no longer owns upstream/provider settings.
    assert!(
        merged.contains("http://old-openai"),
        "expected scoped merge to preserve [upstream], got:\n{merged}"
    );
    // Old claude command should be gone.
    assert!(
        !merged.contains("old-claude-binary"),
        "expected scoped merge to overwrite [agents.claude].command, got:\n{merged}"
    );
}

#[test]
fn save_config_writes_both_scopes_when_both_selected() {
    let _xdg = XdgScope::cleared();
    let answers = SetupAnswers {
        scope: ConfigScope::Both,
        agents: vec![],
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
fn global_config_dir_and_preview_paths_prefer_xdg_when_set() {
    let xdg = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let _env = EnvScope::set(&[("XDG_CONFIG_HOME", Some(xdg.path().as_os_str()))]);

    assert_eq!(
        global_config_dir(home.path()),
        xdg.path().join("nemo-relay")
    );
    assert_eq!(
        preview_paths(ConfigScope::Both, cwd.path(), home.path()),
        vec![
            cwd.path().join(".nemo-relay/config.toml"),
            xdg.path().join("nemo-relay/config.toml"),
        ]
    );
}

#[test]
fn config_scope_labels_are_user_facing_and_stable() {
    assert!(
        ConfigScope::Project
            .label()
            .contains(".nemo-relay/config.toml")
    );
    assert!(
        ConfigScope::Global
            .label()
            .contains(".config/nemo-relay/config.toml")
    );
    assert!(
        ConfigScope::Both
            .label()
            .contains("project overrides global")
    );
}

#[test]
fn existing_defaults_detects_scope_and_agents_from_docs() {
    let empty = Defaults::default();
    assert!(!empty.has_any());
    assert!(
        Defaults {
            scope: Some(ConfigScope::Project),
            agents: vec![]
        }
        .has_any()
    );
    assert!(
        Defaults {
            scope: None,
            agents: vec![CodingAgent::Codex]
        }
        .has_any()
    );

    let doc: DocumentMut = r#"
[agents.claude]
command = "claude"

[agents.codex]
command = "codex"

[agents.unknown]
command = "custom"
"#
    .parse()
    .unwrap();
    let agents = read_agents_from_doc(&doc);
    assert_eq!(agents, vec![CodingAgent::ClaudeCode, CodingAgent::Codex]);
}

#[test]
fn read_existing_defaults_prefers_workspace_and_reports_scope_variants() {
    let cwd = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let _cwd = CwdScope::enter(cwd.path());
    let _env = EnvScope::set(&[
        ("XDG_CONFIG_HOME", None),
        ("HOME", Some(home.path().as_os_str())),
        ("USERPROFILE", None),
    ]);

    assert!(read_existing_defaults().is_none());

    let global_path = home.path().join(".config/nemo-relay/config.toml");
    std::fs::create_dir_all(global_path.parent().unwrap()).unwrap();
    std::fs::write(&global_path, "[agents.codex]\ncommand = \"codex\"\n").unwrap();
    let defaults = read_existing_defaults().unwrap();
    assert_eq!(defaults.scope, Some(ConfigScope::Global));
    assert_eq!(defaults.agents, vec![CodingAgent::Codex]);

    let workspace_path = cwd.path().join(".nemo-relay/config.toml");
    std::fs::create_dir_all(workspace_path.parent().unwrap()).unwrap();
    std::fs::write(&workspace_path, "[agents.claude]\ncommand = \"claude\"\n").unwrap();
    let defaults = read_existing_defaults().unwrap();
    assert_eq!(defaults.scope, Some(ConfigScope::Both));
    assert_eq!(defaults.agents, vec![CodingAgent::ClaudeCode]);

    std::fs::remove_file(&global_path).unwrap();
    let defaults = read_existing_defaults().unwrap();
    assert_eq!(defaults.scope, Some(ConfigScope::Project));
    assert_eq!(defaults.agents, vec![CodingAgent::ClaudeCode]);
}

#[test]
fn write_or_merge_recovers_from_non_table_agents_value() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
agents = "not-a-table"

[plugins]
config = { version = 1, components = [] }

"#,
    )
    .unwrap();
    let doc = build_config(&SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::Codex],
    });

    write_or_merge(&path, &doc, Some(CodingAgent::Codex)).unwrap();

    let merged = std::fs::read_to_string(path).unwrap();
    assert!(merged.contains("[agents.codex]"));
    assert!(merged.contains(r#"command = "codex""#));
    assert!(!merged.contains("[plugins]"));
}

#[test]
fn write_or_merge_overwrites_without_merge_scope_and_reports_malformed_existing_config() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(&path, "[agents.codex]\ncommand = \"old\"\n").unwrap();
    let doc = build_config(&SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::Hermes],
    });

    write_or_merge(&path, &doc, None).unwrap();
    let overwritten = std::fs::read_to_string(&path).unwrap();
    assert!(!overwritten.contains("[agents.codex]"));
    assert!(overwritten.contains("[agents.hermes]"));

    std::fs::write(&path, "[agents.codex\n").unwrap();
    let error = write_or_merge(&path, &doc, Some(CodingAgent::Hermes))
        .unwrap_err()
        .to_string();
    assert!(error.contains("could not parse existing config"));
}

#[test]
fn reset_removes_whole_project_config_or_one_agent() {
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CwdScope::enter(temp.path());
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&config_dir).unwrap();
    let path = config_dir.join("config.toml");
    std::fs::write(
        &path,
        r#"
[agents.claude]
command = "claude"

[agents.codex]
command = "codex"
"#,
    )
    .unwrap();

    reset(Some(CodingAgent::ClaudeCode)).unwrap();

    let scoped = std::fs::read_to_string(&path).unwrap();
    assert!(!scoped.contains("[agents.claude]"));
    assert!(scoped.contains("[agents.codex]"));

    reset(None).unwrap();

    assert!(!path.exists());
}

#[test]
fn reset_removes_empty_agents_table_when_last_agent_is_removed() {
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CwdScope::enter(temp.path());
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&config_dir).unwrap();
    let path = config_dir.join("config.toml");
    std::fs::write(&path, "[agents.codex]\ncommand = \"codex\"\n").unwrap();

    reset(Some(CodingAgent::Codex)).unwrap();

    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(!contents.contains("[agents]"));
    assert!(!contents.contains("[agents.codex]"));
}

#[test]
fn reset_noops_when_project_config_is_missing() {
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CwdScope::enter(temp.path());

    reset(None).unwrap();
    reset(Some(CodingAgent::Codex)).unwrap();
}

#[test]
fn reset_reports_missing_or_malformed_agent_blocks_without_rewriting() {
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CwdScope::enter(temp.path());
    let hermes_home = temp.path().join("hermes-home");
    let _env = EnvScope::set(&[
        ("HOME", Some(temp.path().as_os_str())),
        ("USERPROFILE", None),
        ("HERMES_HOME", Some(hermes_home.as_os_str())),
    ]);
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&config_dir).unwrap();
    let path = config_dir.join("config.toml");
    std::fs::write(&path, "agents = \"not-a-table\"\n").unwrap();

    reset(Some(CodingAgent::Hermes)).unwrap();

    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "agents = \"not-a-table\"\n"
    );

    std::fs::write(&path, "not valid toml = [\n").unwrap();
    let error = reset(Some(CodingAgent::Hermes)).unwrap_err().to_string();
    assert!(
        error.contains("could not parse existing config"),
        "error was: {error}"
    );
}

#[test]
fn plugins_edit_command_for_scope_targets_expected_plugin_scope() {
    use crate::plugins::config_io::{TargetScope, target_scope};

    let cases = [
        (ConfigScope::Project, TargetScope::Project),
        (ConfigScope::Global, TargetScope::User),
        (ConfigScope::Both, TargetScope::Project),
    ];

    for (scope, expected) in cases {
        let command = plugins_edit_command_for_scope(scope);
        assert_eq!(
            target_scope(&command.scope).unwrap(),
            expected,
            "unexpected plugin target scope for {scope:?}"
        );
    }
}

#[test]
fn plugins_resume_command_matches_scope() {
    let cases = [
        (ConfigScope::Project, "nemo-relay plugins edit --project"),
        (ConfigScope::Both, "nemo-relay plugins edit --project"),
        (ConfigScope::Global, "nemo-relay plugins edit"),
    ];

    for (scope, expected) in cases {
        assert_eq!(
            plugins_resume_command(scope),
            expected,
            "unexpected resume command for {scope:?}"
        );
    }
}

#[test]
fn plugin_prompt_interruption_recognizes_cancel_inputs() {
    for kind in [
        std::io::ErrorKind::Interrupted,
        std::io::ErrorKind::UnexpectedEof,
    ] {
        let error = dialoguer::Error::IO(std::io::Error::from(kind));
        assert!(plugin_prompt_was_interrupted(&error));
    }

    let error = dialoguer::Error::IO(std::io::Error::other("boom"));
    assert!(!plugin_prompt_was_interrupted(&error));
}
