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

// This stub-binary test specifically verifies Unix executable-bit handling. Platform-neutral
// PATH/PATHEXT resolution and Windows command-shim execution have separate focused coverage.
#[cfg(unix)]
#[test]
fn detect_installed_agents_finds_binaries_on_path() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    // Drop stub binaries for both supported agents.
    for exec in ["claude", "codex"] {
        let path = temp.path().join(exec);
        std::fs::write(&path, "#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    // Use the pure-function variant that takes PATH as an arg instead of mutating the global
    // env var. Tests run in parallel by default; touching `std::env::set_var("PATH", ...)` would
    // race with every other test that reads the environment.
    let detected = detect_installed_agents_in(Some(temp.path().as_os_str()));
    assert!(detected.contains(&CodingAgent::ClaudeCode));
    assert!(detected.contains(&CodingAgent::Codex));
}

#[test]
fn detect_installed_agents_handles_missing_path() {
    assert!(detect_installed_agents_in(None).is_empty());
}

#[test]
fn build_config_does_not_emit_observability_exporters() {
    let answers = SetupAnswers { agents: vec![] };

    let rendered = build_config(&answers).to_string();

    assert!(!rendered.contains("[exporters]"));
    assert!(!rendered.contains("[export."));
    assert!(!rendered.contains("[observability]"));
    assert!(!rendered.contains("[exporters.atif]"));
    assert!(!rendered.contains("[exporters.openinference]"));
}

#[test]
fn build_config_skips_empty_sections_when_no_backends_selected() {
    let answers = SetupAnswers { agents: vec![] };

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
fn save_config_writes_user_scope_to_user_config_dir() {
    let _xdg = XdgScope::cleared();
    let answers = SetupAnswers {
        agents: vec![CodingAgent::Codex],
    };
    let doc = build_config(&answers);
    let home = tempfile::tempdir().unwrap();
    let user_config_dir = home.path().join(".config/nemo-relay");
    assert!(!user_config_dir.exists());

    let written = save_config(&doc, home.path(), None).unwrap();

    assert_eq!(written.len(), 1);
    assert_eq!(
        written[0],
        home.path().join(".config/nemo-relay/config.toml")
    );
    let contents = std::fs::read_to_string(&written[0]).unwrap();
    assert!(user_config_dir.is_dir());
    assert!(!contents.contains("[exporters]"));
    assert!(contents.contains("[agents.codex]"));
}

#[test]
fn save_config_scoped_merge_preserves_other_agents() {
    // Seed an existing config with claude AND codex blocks, plus a custom [upstream] that the
    // wizard does not touch. Then "re-run" the wizard scoped to claude and assert codex +
    // upstream survive while claude is updated and observability is written fresh.
    let home = tempfile::tempdir().unwrap();
    let _xdg = XdgScope::cleared();
    let user_dir = home.path().join(".config/nemo-relay");
    std::fs::create_dir_all(&user_dir).unwrap();
    let existing_path = user_dir.join("config.toml");
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
        agents: vec![CodingAgent::ClaudeCode],
    };
    let doc = build_config(&answers);
    save_config(&doc, home.path(), Some(CodingAgent::ClaudeCode)).unwrap();

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
fn save_config_writes_only_user_scope() {
    let _xdg = XdgScope::cleared();
    let answers = SetupAnswers { agents: vec![] };
    let doc = build_config(&answers);
    let home = tempfile::tempdir().unwrap();

    let written = save_config(&doc, home.path(), None).unwrap();

    assert_eq!(
        written,
        vec![home.path().join(".config/nemo-relay/config.toml")]
    );
}

#[test]
fn user_config_dir_and_preview_paths_prefer_xdg_when_set() {
    let xdg = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let _env = EnvScope::set(&[("XDG_CONFIG_HOME", Some(xdg.path().as_os_str()))]);

    assert_eq!(user_config_dir(home.path()), xdg.path().join("nemo-relay"));
    assert_eq!(
        preview_paths(home.path()),
        vec![xdg.path().join("nemo-relay/config.toml")]
    );
}

#[test]
fn existing_defaults_detects_scope_and_agents_from_docs() {
    let empty = Defaults::default();
    assert!(!empty.has_any());
    assert!(
        Defaults {
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

// The parser used to hand-match "claude" and "codex", so `[agents.pi]` read as
// unknown. That is not merely unreported: an unscoped wizard run rewrites the whole
// `[agents]` table from what this returns, so an unparsed agent is *deleted*.
#[test]
fn read_agents_from_doc_recognizes_every_supported_agent() {
    let doc: DocumentMut = r#"
[agents.claude]
command = "claude"

[agents.codex]
command = "codex"

[agents.pi]
command = "pi"
"#
    .parse()
    .unwrap();
    assert_eq!(
        read_agents_from_doc(&doc),
        vec![CodingAgent::ClaudeCode, CodingAgent::Codex, CodingAgent::Pi]
    );
}

// The failure this guards: `nemo-relay config`, answered without deselecting anything,
// silently dropping an agent the wizard never offered.
#[test]
fn an_unscoped_save_preserves_every_configured_agent() {
    let home = tempfile::tempdir().unwrap();
    let path = home.path().join(".config/nemo-relay/config.toml");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        &path,
        "[agents.claude]\ncommand = \"claude\"\n\n[agents.pi]\ncommand = \"pi\"\n",
    )
    .unwrap();

    // What the wizard reads back as its pre-checked defaults...
    let existing: DocumentMut = std::fs::read_to_string(&path).unwrap().parse().unwrap();
    let configured = read_agents_from_doc(&existing);
    assert!(
        configured.contains(&CodingAgent::Pi),
        "pi must be seen as configured, or the wizard cannot re-offer it: {configured:?}"
    );

    // ...and what it writes back when the user changes nothing.
    let doc = build_config(&SetupAnswers { agents: configured });
    save_config(&doc, home.path(), None).unwrap();

    let saved: DocumentMut = std::fs::read_to_string(&path).unwrap().parse().unwrap();
    assert_eq!(
        read_agents_from_doc(&saved),
        vec![CodingAgent::ClaudeCode, CodingAgent::Pi],
        "an unscoped save replaces the whole [agents] table, so it must carry every agent forward"
    );
}

#[test]
fn read_existing_defaults_reads_user_config_and_ignores_project_config() {
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
    assert_eq!(defaults.agents, vec![CodingAgent::Codex]);

    let workspace_path = cwd.path().join(".nemo-relay/config.toml");
    std::fs::create_dir_all(workspace_path.parent().unwrap()).unwrap();
    std::fs::write(&workspace_path, "[agents.claude]\ncommand = \"claude\"\n").unwrap();
    let defaults = read_existing_defaults().unwrap();
    assert_eq!(defaults.agents, vec![CodingAgent::Codex]);

    std::fs::remove_file(&global_path).unwrap();
    assert!(read_existing_defaults().is_none());
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
        agents: vec![CodingAgent::Codex],
    });

    write_or_merge(&path, &doc, Some(CodingAgent::Codex)).unwrap();

    let merged = std::fs::read_to_string(path).unwrap();
    assert!(merged.contains("[agents.codex]"));
    assert!(merged.contains(r#"command = "codex""#));
    assert!(!merged.contains("[plugins]"));
}

#[test]
fn write_or_merge_replaces_agents_without_merge_scope_and_preserves_other_sections() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        "[agents.codex]\ncommand = \"old\"\n\n[upstream]\nopenai_base_url = \"https://example.test\"\n",
    )
    .unwrap();
    let doc = build_config(&SetupAnswers {
        agents: vec![CodingAgent::ClaudeCode],
    });

    write_or_merge(&path, &doc, None).unwrap();
    let overwritten = std::fs::read_to_string(&path).unwrap();
    assert!(!overwritten.contains("[agents.codex]"));
    assert!(overwritten.contains("[agents.claude]"));
    assert!(overwritten.contains("[upstream]"));
    assert!(overwritten.contains("https://example.test"));

    std::fs::write(&path, "[agents.codex\n").unwrap();
    let error = write_or_merge(&path, &doc, None).unwrap_err().to_string();
    assert!(error.contains("could not parse existing config"));
}

#[test]
fn reset_removes_whole_user_config_or_one_agent() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::set(&[
        ("HOME", Some(temp.path().as_os_str())),
        ("USERPROFILE", None),
        ("XDG_CONFIG_HOME", None),
    ]);
    let config_dir = temp.path().join(".config/nemo-relay");
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
    let _env = EnvScope::set(&[
        ("HOME", Some(temp.path().as_os_str())),
        ("USERPROFILE", None),
        ("XDG_CONFIG_HOME", None),
    ]);
    let config_dir = temp.path().join(".config/nemo-relay");
    std::fs::create_dir_all(&config_dir).unwrap();
    let path = config_dir.join("config.toml");
    std::fs::write(&path, "[agents.codex]\ncommand = \"codex\"\n").unwrap();

    reset(Some(CodingAgent::Codex)).unwrap();

    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(!contents.contains("[agents]"));
    assert!(!contents.contains("[agents.codex]"));
}

#[test]
fn reset_noops_when_user_config_is_missing() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::set(&[
        ("HOME", Some(temp.path().as_os_str())),
        ("USERPROFILE", None),
        ("XDG_CONFIG_HOME", None),
    ]);

    reset(None).unwrap();
    reset(Some(CodingAgent::Codex)).unwrap();
}

#[test]
fn reset_reports_missing_or_malformed_agent_blocks_without_rewriting() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::set(&[
        ("HOME", Some(temp.path().as_os_str())),
        ("USERPROFILE", None),
        ("XDG_CONFIG_HOME", None),
    ]);
    let config_dir = temp.path().join(".config/nemo-relay");
    std::fs::create_dir_all(&config_dir).unwrap();
    let path = config_dir.join("config.toml");
    std::fs::write(&path, "agents = \"not-a-table\"\n").unwrap();

    reset(Some(CodingAgent::Codex)).unwrap();

    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "agents = \"not-a-table\"\n"
    );

    std::fs::write(&path, "not valid toml = [\n").unwrap();
    let error = reset(Some(CodingAgent::Codex)).unwrap_err().to_string();
    assert!(
        error.contains("could not parse existing config"),
        "error was: {error}"
    );
}

#[test]
fn reset_removes_user_config_and_leaves_project_file_untouched() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    let home = temp.path().join("home");
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    let _cwd = CwdScope::enter(&project);
    let _env = EnvScope::set(&[
        ("HOME", Some(home.as_os_str())),
        ("USERPROFILE", Some(home.as_os_str())),
        ("XDG_CONFIG_HOME", Some(xdg.as_os_str())),
    ]);

    let project_path = project.join(".nemo-relay/config.toml");
    let user_path = user_config_dir(&home).join("config.toml");
    std::fs::create_dir_all(project_path.parent().unwrap()).unwrap();
    std::fs::create_dir_all(user_path.parent().unwrap()).unwrap();
    std::fs::write(&project_path, "[agents.codex]\ncommand = \"codex\"\n").unwrap();
    std::fs::write(&user_path, "[agents.codex]\ncommand = \"codex\"\n").unwrap();

    reset(None).unwrap();
    assert!(project_path.exists());
    assert!(!user_path.exists());
}

#[test]
fn plugins_edit_command_targets_user_scope() {
    use crate::plugins::config_io::{TargetScope, target_scope};

    let command = plugins_edit_command(None);
    assert_eq!(target_scope(&command.scope).unwrap(), TargetScope::User);
}

#[test]
fn plugins_edit_command_preserves_explicit_plugin_path() {
    let path = PathBuf::from("/managed/plugins.toml");

    let command = plugins_edit_command(Some(path.clone()));
    assert_eq!(command.explicit_path, Some(path));
    assert_eq!(command.scope, crate::plugins::ConfigurationScope::User);
}

#[test]
fn plugins_resume_command_targets_user_config() {
    assert_eq!(plugins_resume_command(None), "nemo-relay plugins edit");
}

#[test]
fn plugins_resume_command_preserves_explicit_plugin_path() {
    let path = PathBuf::from("/managed/plugin configs/plugins.toml");
    #[cfg(windows)]
    let expected = concat!(
        "nemo-relay --plugin-config-path ",
        "\"/managed/plugin configs/plugins.toml\" plugins edit"
    );
    #[cfg(not(windows))]
    let expected = concat!(
        "nemo-relay --plugin-config-path ",
        "'/managed/plugin configs/plugins.toml' plugins edit"
    );

    assert_eq!(plugins_resume_command(Some(&path)), expected);
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
