// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

// Tests that exercise the global-config write path must run serially with respect to each other
// because `save_config` reads `$XDG_CONFIG_HOME`. CI runners commonly set this var to a real
// `/home/runner/.config` path, which would override the per-test `home` tempdir and make
// path-prefix assertions racy/false. This guard clears the var for the duration of the test
// and restores its prior value on drop.
fn xdg_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct XdgScope<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
    prev: Option<std::ffi::OsString>,
}

impl<'a> XdgScope<'a> {
    fn cleared() -> Self {
        let guard = xdg_env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("XDG_CONFIG_HOME");
        // SAFETY: We hold the mutex for the lifetime of this scope, and the only other tests
        // that touch XDG_CONFIG_HOME also go through this guard. Restored on drop.
        unsafe {
            std::env::remove_var("XDG_CONFIG_HOME");
        }
        Self {
            _guard: guard,
            prev,
        }
    }
}

impl<'a> Drop for XdgScope<'a> {
    fn drop(&mut self) {
        // SAFETY: see `cleared()` above — the env mutex is still held.
        unsafe {
            match self.prev.take() {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
    }
}

struct CwdScope<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
    prev: PathBuf,
}

impl<'a> CwdScope<'a> {
    fn enter(path: &std::path::Path) -> Self {
        let guard = xdg_env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(path).unwrap();
        Self {
            _guard: guard,
            prev,
        }
    }
}

impl<'a> Drop for CwdScope<'a> {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.prev).unwrap();
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
    // Drop stub binaries for two of the four supported agents — confirming detection picks up
    // only the ones present and ignores the others.
    for exec in ["claude", "cursor-agent"] {
        let path = temp.path().join(exec);
        std::fs::write(&path, "#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    // Use the pure-function variant that takes PATH as an arg instead of mutating the global
    // env var. Tests run in parallel by default; touching `std::env::set_var("PATH", ...)` would
    // race with every other test that reads the environment.
    let detected = detect_installed_agents_in(Some(temp.path().as_os_str()));
    assert!(detected.contains(&CodingAgent::ClaudeCode));
    assert!(detected.contains(&CodingAgent::Cursor));
    assert!(!detected.contains(&CodingAgent::Codex));
    assert!(!detected.contains(&CodingAgent::Hermes));
}

#[test]
fn build_config_does_not_emit_observability_exporters() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![],
        hermes_hooks_path: None,
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
        hermes_hooks_path: None,
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
fn save_config_writes_project_scope_to_workspace_dir() {
    let answers = SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::ClaudeCode],
        hermes_hooks_path: None,
    };
    let doc = build_config(&answers);
    let temp = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();

    let written = save_config(&doc, ConfigScope::Project, temp.path(), home.path(), None).unwrap();

    assert_eq!(written.len(), 1);
    assert_eq!(written[0], temp.path().join(".nemo-flow/config.toml"));
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
        hermes_hooks_path: Some(std::path::PathBuf::from("/tmp/proj/.hermes/config.yaml")),
    };
    let rendered = build_config(&answers).to_string();
    assert!(rendered.contains("[agents.hermes]"));
    assert!(rendered.contains(r#"hooks_path = "/tmp/proj/.hermes/config.yaml""#));
}

#[test]
fn config_scope_labels_are_user_facing_and_stable() {
    assert!(
        ConfigScope::Project
            .label()
            .contains(".nemo-flow/config.toml")
    );
    assert!(
        ConfigScope::Global
            .label()
            .contains(".config/nemo-flow/config.toml")
    );
    assert!(
        ConfigScope::Both
            .label()
            .contains("project overrides global")
    );
}

#[test]
fn hermes_hook_paths_follow_selected_scope() {
    let cwd = PathBuf::from("/workspace");
    let home = PathBuf::from("/home/user");
    let agents = [CodingAgent::Hermes];

    assert_eq!(
        hermes_hooks_path_for_scope(&agents, ConfigScope::Project, &cwd, &home),
        Some(PathBuf::from("/workspace/.hermes/config.yaml"))
    );
    assert_eq!(
        hermes_hooks_path_for_scope(&agents, ConfigScope::Both, &cwd, &home),
        Some(PathBuf::from("/workspace/.hermes/config.yaml"))
    );
    assert_eq!(
        hermes_hooks_path_for_scope(&agents, ConfigScope::Global, &cwd, &home),
        Some(PathBuf::from("/home/user/.hermes/config.yaml"))
    );
    assert_eq!(
        hermes_hooks_path_for_scope(&[], ConfigScope::Project, &cwd, &home),
        None
    );
    assert_eq!(
        hermes_hook_targets(ConfigScope::Both, &cwd, &home),
        vec![
            PathBuf::from("/workspace/.hermes/config.yaml"),
            PathBuf::from("/home/user/.hermes/config.yaml")
        ]
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
    assert!(project_yaml.contains("api_request_error"));
    assert!(
        project_yaml.contains("provider: auto"),
        "existing model block must survive merge"
    );
    let home_yaml = std::fs::read_to_string(home.path().join(".hermes/config.yaml")).unwrap();
    assert!(home_yaml.contains("nemo-flow hook-forward hermes"));
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
enabled = true
"#,
    )
    .unwrap();
    let doc = build_config(&SetupAnswers {
        scope: ConfigScope::Project,
        agents: vec![CodingAgent::Codex],
        hermes_hooks_path: None,
    });

    write_or_merge(&path, &doc, Some(CodingAgent::Codex)).unwrap();

    let merged = std::fs::read_to_string(path).unwrap();
    assert!(merged.contains("[agents.codex]"));
    assert!(merged.contains(r#"command = "codex""#));
    assert!(merged.contains("[plugins]"));
}

#[test]
fn reset_removes_whole_project_config_or_one_agent() {
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CwdScope::enter(temp.path());
    let config_dir = temp.path().join(".nemo-flow");
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
fn reset_reports_missing_or_malformed_agent_blocks_without_rewriting() {
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CwdScope::enter(temp.path());
    let config_dir = temp.path().join(".nemo-flow");
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
