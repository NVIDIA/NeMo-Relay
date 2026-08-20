// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testable setup configuration model and file helpers owned by the configure command.

use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, Item, Table, value};

use crate::agents::CodingAgent;
use crate::error::CliError;
use crate::plugins::{ConfigurationScope, PluginsEditRequest};

/// Maps setup to the plugin editor target for the guided continuation. An explicit plugin path
/// follows the runtime contract and wins over the user file.
pub(crate) fn plugins_edit_command(explicit_path: Option<PathBuf>) -> PluginsEditRequest {
    PluginsEditRequest {
        scope: ConfigurationScope::User,
        explicit_path,
    }
}

/// Returns the exact command a user runs to resume plugin setup after skipping the continuation.
pub(crate) fn plugins_resume_command(explicit_path: Option<&Path>) -> String {
    if let Some(path) = explicit_path {
        let path = crate::process::shell_quote_arg_for_platform(
            &path.display().to_string(),
            cfg!(windows),
        );
        return format!("nemo-relay --plugin-config-path {path} plugins edit");
    }
    "nemo-relay plugins edit".into()
}

/// Resolved answers from setup. Built either by `prompt_user` (interactive) or by tests.
#[derive(Debug, Clone)]
pub(crate) struct SetupAnswers {
    pub agents: Vec<CodingAgent>,
}

/// Scans `$PATH` for the supported coding-agent binaries and returns the ones present.
///
/// The lookup uses the same set of executable names that `CodingAgent::infer` already recognizes;
/// detection is pure and deterministic given a fixed PATH so it can be exercised in tests by
/// constructing a tempdir with stub binaries and pointing `$PATH` at it.
pub(crate) fn detect_installed_agents() -> Vec<CodingAgent> {
    detect_installed_agents_in(std::env::var_os("PATH").as_deref())
}

pub(crate) fn detect_installed_agents_in(path_var: Option<&std::ffi::OsStr>) -> Vec<CodingAgent> {
    let Some(path_var) = path_var else {
        return Vec::new();
    };
    // Keep only agents whose canonical executable resolves on PATH.
    CodingAgent::ALL
        .into_iter()
        .filter(|agent| {
            crate::process::resolve_executable_in_path(agent.executable(), Some(path_var)).is_some()
        })
        .collect()
}

/// Builds the TOML document that represents the setup's answers. Pure and testable.
///
/// The shape mirrors the runtime model: agents live under `[agents.<name>]`.
/// Sections are only emitted when the user opted into the corresponding behavior so the resulting
/// file stays minimal.
pub(crate) fn build_config(answers: &SetupAnswers) -> DocumentMut {
    let mut doc = DocumentMut::new();

    if let Some(agents_table) = build_agents_table(answers) {
        doc["agents"] = Item::Table(agents_table);
    }

    doc
}

pub(crate) fn build_agents_table(answers: &SetupAnswers) -> Option<Table> {
    if answers.agents.is_empty() {
        return None;
    }

    let mut agents_table = Table::new();
    for agent in &answers.agents {
        let (key, command) = agent_key_and_command(*agent);
        let mut agent_table = Table::new();
        agent_table["command"] = value(command);
        agents_table.insert(key, Item::Table(agent_table));
    }
    Some(agents_table)
}

/// Writes the setup's TOML document to the user configuration path.
///
/// When `merge_scope` is `Some(agent)`, an existing `config.toml` at the target path is parsed
/// and only the single `[agents.<agent>]` block owned by THIS wizard run is replaced. Other
/// `[agents.*]` blocks are preserved when omitted from the wizard output. When `merge_scope` is
/// `None`, the full `[agents]` table is replaced while unrelated configuration sections are
/// preserved.
///
/// Returns the list of paths written. `home` and `cwd` are explicit so tests can drive this with
/// tempdirs.
pub(crate) fn save_config(
    doc: &DocumentMut,
    home: &Path,
    merge_scope: Option<CodingAgent>,
) -> Result<Vec<PathBuf>, CliError> {
    let user_dir = user_config_dir(home);
    std::fs::create_dir_all(&user_dir)?;
    let path = user_dir.join("config.toml");
    write_or_merge(&path, doc, merge_scope)?;
    Ok(vec![path])
}

// Resolves the user nemo-relay config directory. Prefers `$XDG_CONFIG_HOME/nemo-relay` (matches
// `config::user_config_dir`), falling back to `<home>/.config/nemo-relay`. Tests that pass a
// tempdir for `home` get hermetic paths unless they set XDG_CONFIG_HOME explicitly.
pub(crate) fn user_config_dir(home: &Path) -> PathBuf {
    if let Some(base) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(base).join("nemo-relay");
    }
    home.join(".config").join("nemo-relay")
}

// Writes the wizard-built `doc` to `path`. When `merge_scope` is `Some(agent)` and the file
// already exists, preserves any `[agents.<other>]` blocks while replacing the shared sections
// and the target agent's block. When `merge_scope` is `None`, replaces the complete agents table
// while preserving sections owned by other configuration commands.
pub(crate) fn write_or_merge(
    path: &Path,
    doc: &DocumentMut,
    merge_scope: Option<CodingAgent>,
) -> Result<(), CliError> {
    if !path.exists() {
        std::fs::write(path, doc.to_string())?;
        return Ok(());
    }
    let existing_raw = std::fs::read_to_string(path)?;
    let mut existing: DocumentMut = existing_raw
        .parse()
        .map_err(|err| CliError::Config(format!("could not parse existing config: {err}")))?;
    existing.remove("plugins");
    let Some(agent) = merge_scope else {
        match doc.get("agents") {
            Some(agents) => {
                existing["agents"] = agents.clone();
            }
            None => {
                existing.remove("agents");
            }
        }
        std::fs::write(path, existing.to_string())?;
        return Ok(());
    };
    let agent_key = agent_key_and_command(agent).0;
    // Remove the legacy plugin configuration block so the merged config remains loadable after
    // plugin configuration moved to plugins.toml.
    merge_agents_entry(&mut existing, doc, agent_key);
    std::fs::write(path, existing.to_string())?;
    Ok(())
}

// Replaces the single `[agents.<agent>]` block in `dst` with the one from `src`. If `src` does
// not contain that block, the existing entry in `dst` is left as-is.
pub(crate) fn merge_agents_entry(dst: &mut DocumentMut, src: &DocumentMut, agent_key: &str) {
    let Some(src_agent) = src
        .get("agents")
        .and_then(|item| item.as_table())
        .and_then(|table| table.get(agent_key))
    else {
        return;
    };
    // Defensive: if the existing config has `agents = "literal"` or `agents = [...]` (anything
    // not a table) the original `.as_table_mut().unwrap()` panicked. Replace any non-table
    // value with a fresh table so a malformed user file degrades to an overwrite, not a crash.
    let needs_init = dst
        .get("agents")
        .is_none_or(|item| item.as_table().is_none());
    if needs_init {
        dst["agents"] = Item::Table(Table::new());
    }
    let agents_table = dst["agents"]
        .as_table_mut()
        .expect("agents key is a table after the init guard above");
    agents_table.insert(agent_key, src_agent.clone());
}

/// Removes the user `config.toml` (or just one agent's block within it).
///
/// `agent_hint = None` deletes the whole user config file. `agent_hint = Some(agent)` parses
/// the existing file and removes only `[agents.<agent>]`, leaving every other section intact.
/// System configuration is left to direct editing because it is not owned by the wizard.
pub(crate) fn reset(agent_hint: Option<CodingAgent>) -> Result<(), CliError> {
    let home = home_dir().ok_or_else(|| {
        CliError::Config("cannot resolve the home directory for user reset".into())
    })?;
    reset_config_path(
        &user_config_dir(&home).join("config.toml"),
        "user",
        agent_hint,
    )
}

fn reset_config_path(
    path: &Path,
    scope: &str,
    agent_hint: Option<CodingAgent>,
) -> Result<(), CliError> {
    if !path.exists() {
        println!("  No {scope} config to reset at {}", path.display());
        return Ok(());
    }
    match agent_hint {
        None => {
            std::fs::remove_file(path)?;
            println!("  ✓ Removed {}", path.display());
            println!("  Run `nemo-relay config` to set up again.");
        }
        Some(agent) => {
            let agent_key = agent_key_and_command(agent).0;
            let raw = std::fs::read_to_string(path)?;
            let mut doc: DocumentMut = raw.parse().map_err(|err| {
                CliError::Config(format!("could not parse existing config: {err}"))
            })?;
            // Three reasons we have nothing to remove: no `[agents]` table at all, the `agents`
            // key holds a non-table value, or the table is missing this specific agent's block.
            // In every case we must report "nothing to reset" and skip the write — silently
            // printing "✓ Removed" when nothing changed misleads the user about file state.
            let Some(agents) = doc.get_mut("agents").and_then(Item::as_table_mut) else {
                println!(
                    "  No `[agents.{agent_key}]` block to reset in {}",
                    path.display()
                );
                return Ok(());
            };
            if agents.remove(agent_key).is_none() {
                println!(
                    "  No `[agents.{agent_key}]` block to reset in {}",
                    path.display()
                );
                return Ok(());
            }
            // Remove the empty `[agents]` table itself so the file stays tidy when no agent
            // entries remain.
            if agents.is_empty() {
                doc.remove("agents");
            }
            std::fs::write(path, doc.to_string())?;
            println!("  ✓ Removed `[agents.{agent_key}]` from {}", path.display());
        }
    }
    Ok(())
}

/// Pre-filled wizard defaults read from an existing `config.toml`. When the file is missing or
/// unparseable the defaults are all-empty and the wizard behaves like a first-run setup.
#[derive(Debug, Clone, Default)]
pub(crate) struct Defaults {
    pub(crate) agents: Vec<CodingAgent>,
}

impl Defaults {
    pub(crate) fn has_any(&self) -> bool {
        !self.agents.is_empty()
    }
}

/// Reads the user config file and derives wizard defaults from it. Missing or malformed files
/// yield `None` (the wizard then behaves as if no config existed).
pub(crate) fn read_existing_defaults() -> Option<Defaults> {
    let path = user_config_dir(&home_dir()?).join("config.toml");
    let doc: DocumentMut = std::fs::read_to_string(path).ok()?.parse().ok()?;

    Some(Defaults {
        agents: read_agents_from_doc(&doc),
    })
}

pub(crate) fn read_agents_from_doc(doc: &DocumentMut) -> Vec<CodingAgent> {
    let Some(table) = doc.get("agents").and_then(|i| i.as_table()) else {
        return Vec::new();
    };
    // Driven from `CodingAgent::ALL` rather than a hand-written match: an agent missing
    // from this list is not just unreported, it is *deleted*. An unscoped wizard run
    // rewrites the whole `[agents]` table from what it read here, so a key it cannot
    // parse silently disappears from the user's configuration.
    let mut found = Vec::new();
    for (key, _) in table.iter() {
        if let Some(agent) = CodingAgent::ALL
            .into_iter()
            .find(|agent| agent_key_and_command(*agent).0 == key)
        {
            found.push(agent);
        }
    }
    found
}

pub(crate) fn agent_key_and_command(agent: CodingAgent) -> (&'static str, &'static str) {
    (agent.as_arg(), agent.executable())
}

pub(crate) fn preview_paths(home: &Path) -> Vec<PathBuf> {
    vec![user_config_dir(home).join("config.toml")]
}

pub(crate) fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}
