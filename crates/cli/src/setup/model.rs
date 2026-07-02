// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testable setup configuration model and file helpers.

use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, Item, Table, value};

use crate::config::CodingAgent;
use crate::error::CliError;
use crate::installer::{hermes_hooks, hook_forward_command, merge_hermes_config};

/// Where the setup saves its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigScope {
    /// `./.nemo-relay/config.toml` (walked-up workspace dir).
    Project,
    /// `~/.config/nemo-relay/config.toml` (or `$XDG_CONFIG_HOME/nemo-relay/config.toml`).
    Global,
    /// Both project and global; project takes precedence per merge order.
    Both,
}

impl ConfigScope {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Project => "project   ./.nemo-relay/config.toml          (recommended)",
            Self::Global => "global    ~/.config/nemo-relay/config.toml",
            Self::Both => "both      project overrides global",
        }
    }
}

/// Resolved answers from setup. Built either by `prompt_user` (interactive) or by tests.
#[derive(Debug, Clone)]
pub(crate) struct SetupAnswers {
    pub scope: ConfigScope,
    pub agents: Vec<CodingAgent>,
    /// Path recorded under `[agents.hermes].hooks_path` when hermes is selected. Set by `run`
    /// from `hermes_hooks_path_for_scope` so the wizard preview shows the file the launcher
    /// will reference. `None` when hermes wasn't selected.
    pub hermes_hooks_path: Option<PathBuf>,
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
    // Pairs of (CodingAgent, exec name to look for on $PATH).
    let candidates = [
        (CodingAgent::ClaudeCode, "claude"),
        (CodingAgent::Codex, "codex"),
        (CodingAgent::Hermes, "hermes"),
    ];
    candidates
        .into_iter()
        .filter_map(|(agent, exec)| {
            let found = std::env::split_paths(path_var).any(|dir| {
                let candidate = dir.join(exec);
                candidate.is_file()
            });
            found.then_some(agent)
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

pub(super) fn build_agents_table(answers: &SetupAnswers) -> Option<Table> {
    if answers.agents.is_empty() {
        return None;
    }

    let mut agents_table = Table::new();
    for agent in &answers.agents {
        let (key, command) = agent_key_and_command(*agent);
        let mut agent_table = Table::new();
        agent_table["command"] = value(command);
        if matches!(agent, CodingAgent::Hermes)
            && let Some(path) = answers.hermes_hooks_path.as_deref()
        {
            agent_table["hooks_path"] = value(path.display().to_string());
        }
        agents_table.insert(key, Item::Table(agent_table));
    }
    Some(agents_table)
}

/// Writes the setup's TOML document to the scope-appropriate path(s).
///
/// When `merge_scope` is `Some(agent)`, an existing `config.toml` at the target path is parsed
/// and only the single `[agents.<agent>]` block owned by THIS wizard run is replaced. Other
/// `[agents.*]` blocks are preserved when omitted from the wizard output. When `merge_scope` is
/// `None`, the file is overwritten outright with the wizard's full output (the user explicitly
/// chose which agents to include).
///
/// Returns the list of paths written. `home` and `cwd` are explicit so tests can drive this with
/// tempdirs.
pub(crate) fn save_config(
    doc: &DocumentMut,
    scope: ConfigScope,
    cwd: &Path,
    home: &Path,
    merge_scope: Option<CodingAgent>,
) -> Result<Vec<PathBuf>, CliError> {
    let mut written = Vec::new();
    if matches!(scope, ConfigScope::Project | ConfigScope::Both) {
        let project_dir = cwd.join(".nemo-relay");
        std::fs::create_dir_all(&project_dir)?;
        let path = project_dir.join("config.toml");
        write_or_merge(&path, doc, merge_scope)?;
        written.push(path);
    }
    if matches!(scope, ConfigScope::Global | ConfigScope::Both) {
        let global_dir = global_config_dir(home);
        std::fs::create_dir_all(&global_dir)?;
        let path = global_dir.join("config.toml");
        write_or_merge(&path, doc, merge_scope)?;
        written.push(path);
    }
    Ok(written)
}

// Resolves the global nemo-relay config directory. Prefers `$XDG_CONFIG_HOME/nemo-relay` (matches
// `config::user_config_dir`), falling back to `<home>/.config/nemo-relay`. Tests that pass a
// tempdir for `home` get hermetic paths unless they set XDG_CONFIG_HOME explicitly.
pub(super) fn global_config_dir(home: &Path) -> PathBuf {
    if let Some(base) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(base).join("nemo-relay");
    }
    home.join(".config").join("nemo-relay")
}

// Writes the wizard-built `doc` to `path`. When `merge_scope` is `Some(agent)` and the file
// already exists, preserves any `[agents.<other>]` blocks while replacing the shared sections
// and the target agent's block. When `merge_scope` is `None`, just overwrites the file.
pub(super) fn write_or_merge(
    path: &Path,
    doc: &DocumentMut,
    merge_scope: Option<CodingAgent>,
) -> Result<(), CliError> {
    let Some(agent) = merge_scope else {
        std::fs::write(path, doc.to_string())?;
        return Ok(());
    };
    if !path.exists() {
        std::fs::write(path, doc.to_string())?;
        return Ok(());
    }
    let existing_raw = std::fs::read_to_string(path)?;
    let mut existing: DocumentMut = existing_raw
        .parse()
        .map_err(|err| CliError::Config(format!("could not parse existing config: {err}")))?;
    let agent_key = agent_key_and_command(agent).0;
    // Remove the legacy plugin configuration block so the merged config remains loadable after
    // plugin configuration moved to plugins.toml.
    existing.remove("plugins");
    merge_agents_entry(&mut existing, doc, agent_key);
    std::fs::write(path, existing.to_string())?;
    Ok(())
}

// Replaces the single `[agents.<agent>]` block in `dst` with the one from `src`. If `src` does
// not contain that block, the existing entry in `dst` is left as-is.
pub(super) fn merge_agents_entry(dst: &mut DocumentMut, src: &DocumentMut, agent_key: &str) {
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

/// Removes the project `config.toml` (or just one agent's block within it).
///
/// `agent_hint = None` deletes the whole project config file. `agent_hint = Some(agent)` parses
/// the existing file and removes only `[agents.<agent>]`, leaving every other section intact.
/// In both cases this targets the *project* layer; global and system layers are left to direct
/// editing because they typically aren't owned by the wizard.
pub(crate) fn reset(agent_hint: Option<CodingAgent>) -> Result<(), CliError> {
    let cwd = std::env::current_dir()?;
    let path = cwd.join(".nemo-relay").join("config.toml");
    if !path.exists() {
        println!("  No project config to reset at {}", path.display());
        return Ok(());
    }
    match agent_hint {
        None => {
            std::fs::remove_file(&path)?;
            println!("  ✓ Removed {}", path.display());
            println!("  Run `nemo-relay config` to set up again.");
        }
        Some(agent) => {
            let agent_key = agent_key_and_command(agent).0;
            let raw = std::fs::read_to_string(&path)?;
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
            std::fs::write(&path, doc.to_string())?;
            println!("  ✓ Removed `[agents.{agent_key}]` from {}", path.display());
        }
    }
    Ok(())
}

/// Returns the Hermes hooks file path that should be recorded for the selected setup scope.
pub(crate) fn hermes_hooks_path_for_scope(
    agents: &[CodingAgent],
    scope: ConfigScope,
    cwd: &Path,
    home: &Path,
) -> Option<PathBuf> {
    if !agents.contains(&CodingAgent::Hermes) {
        return None;
    }
    match scope {
        ConfigScope::Project | ConfigScope::Both => Some(cwd.join(".hermes").join("config.yaml")),
        ConfigScope::Global => Some(home.join(".hermes").join("config.yaml")),
    }
}

/// Writes/merges `.hermes/config.yaml` hook config for every scope-applicable location so hermes
/// fires `nemo-relay hook-forward hermes` on every hook event after setup. Idempotent: existing
/// hook entries are preserved and our generated groups are appended only when missing.
///
/// Returns the list of paths actually written so callers can surface them to the user.
pub(crate) fn install_hermes_hooks(
    scope: ConfigScope,
    cwd: &Path,
    home: &Path,
) -> Result<Vec<PathBuf>, CliError> {
    let generated = hermes_hooks(&hook_forward_command("nemo-relay", CodingAgent::Hermes));
    let mut written = Vec::new();
    for path in hermes_hook_targets(scope, cwd, home) {
        let existing = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(error) => return Err(CliError::Io(error)),
        };
        let merged = merge_hermes_config(&existing, generated.clone())?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, merged)?;
        written.push(path);
    }
    Ok(written)
}

pub(super) fn hermes_hook_targets(scope: ConfigScope, cwd: &Path, home: &Path) -> Vec<PathBuf> {
    let mut targets = Vec::new();
    if matches!(scope, ConfigScope::Project | ConfigScope::Both) {
        targets.push(cwd.join(".hermes").join("config.yaml"));
    }
    if matches!(scope, ConfigScope::Global | ConfigScope::Both) {
        targets.push(home.join(".hermes").join("config.yaml"));
    }
    targets
}

/// Pre-filled wizard defaults read from an existing `config.toml`. When the file is missing or
/// unparseable the defaults are all-empty and the wizard behaves like a first-run setup.
#[derive(Debug, Clone, Default)]
pub(super) struct Defaults {
    pub(super) scope: Option<ConfigScope>,
    pub(super) agents: Vec<CodingAgent>,
}

impl Defaults {
    pub(super) fn has_any(&self) -> bool {
        self.scope.is_some() || !self.agents.is_empty()
    }
}

/// Reads the highest-precedence existing config file and derives wizard defaults from it.
/// Workspace config wins over global; if both exist, scope defaults to `Both`. Missing or
/// malformed files yield `None` (the wizard then behaves as if no config existed).
pub(super) fn read_existing_defaults() -> Option<Defaults> {
    let cwd = std::env::current_dir().ok()?;
    let home = home_dir();

    let workspace_path = cwd.join(".nemo-relay").join("config.toml");
    let global_path = home
        .as_ref()
        .map(|h| global_config_dir(h).join("config.toml"));

    let workspace_exists = workspace_path.exists();
    let global_exists = global_path.as_ref().is_some_and(|p| p.exists());

    let read_doc =
        |path: &Path| -> Option<DocumentMut> { std::fs::read_to_string(path).ok()?.parse().ok() };

    let doc = match (workspace_exists, global_exists) {
        (true, _) => read_doc(&workspace_path)?,
        (false, true) => read_doc(global_path.as_ref()?)?,
        (false, false) => return None,
    };

    let scope = match (workspace_exists, global_exists) {
        (true, true) => Some(ConfigScope::Both),
        (true, false) => Some(ConfigScope::Project),
        (false, true) => Some(ConfigScope::Global),
        (false, false) => None,
    };

    Some(Defaults {
        scope,
        agents: read_agents_from_doc(&doc),
    })
}

pub(super) fn read_agents_from_doc(doc: &DocumentMut) -> Vec<CodingAgent> {
    let Some(table) = doc.get("agents").and_then(|i| i.as_table()) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for (key, _) in table.iter() {
        let agent = match key {
            "claude" => Some(CodingAgent::ClaudeCode),
            "codex" => Some(CodingAgent::Codex),
            "hermes" => Some(CodingAgent::Hermes),
            _ => None,
        };
        if let Some(agent) = agent {
            found.push(agent);
        }
    }
    found
}

pub(super) fn agent_key_and_command(agent: CodingAgent) -> (&'static str, &'static str) {
    match agent {
        CodingAgent::ClaudeCode => ("claude", "claude"),
        CodingAgent::Codex => ("codex", "codex"),
        CodingAgent::Hermes => ("hermes", "hermes"),
    }
}

pub(super) fn preview_paths(scope: ConfigScope, cwd: &Path, home: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if matches!(scope, ConfigScope::Project | ConfigScope::Both) {
        paths.push(cwd.join(".nemo-relay").join("config.toml"));
    }
    if matches!(scope, ConfigScope::Global | ConfigScope::Both) {
        paths.push(global_config_dir(home).join("config.toml"));
    }
    paths
}

pub(super) fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}
