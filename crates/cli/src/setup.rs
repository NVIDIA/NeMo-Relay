// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! First-run setup for `nemo-flow` configuration.
//!
//! Drives the three required prompts (scope, agents, observability backends) plus an optional
//! OpenInference endpoint follow-up, then writes a `config.toml` to the chosen scope. Pure
//! helpers (`detect_installed_agents`, `build_config`, `save_config`) are split out from the
//! `dialoguer`-driven orchestrator so the data path can be unit-tested without a TTY.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use dialoguer::theme::ColorfulTheme;
use dialoguer::{Confirm, Input, MultiSelect, Select};
use toml_edit::{DocumentMut, Item, Table, value};

use crate::config::CodingAgent;
use crate::error::CliError;
use crate::installer::{hermes_hooks, hook_forward_command, merge_hermes_config};

/// Where the setup saves its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigScope {
    /// `./.nemo-flow/config.toml` (walked-up workspace dir).
    Project,
    /// `~/.config/nemo-flow/config.toml` (or `$XDG_CONFIG_HOME/nemo-flow/config.toml`).
    Global,
    /// Both project and global; project takes precedence per merge order.
    Both,
}

impl ConfigScope {
    fn label(self) -> &'static str {
        match self {
            Self::Project => "project   ./.nemo-flow/config.toml          (recommended)",
            Self::Global => "global    ~/.config/nemo-flow/config.toml",
            Self::Both => "both      project overrides global",
        }
    }
}

/// One of the built-in observability backends offered in setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObservabilityBackend {
    /// Local ATIF trajectory files.
    Atif,
    /// OpenInference spans streamed to an HTTP endpoint (Phoenix, Arize, OTLP-compatible).
    OpenInference,
}

impl ObservabilityBackend {
    fn label(self) -> &'static str {
        match self {
            Self::Atif => "ATIF trajectory files    ./atif/                  (recommended)",
            Self::OpenInference => {
                "OpenInference spans      <endpoint URL>           (Phoenix / Arize / OTLP)"
            }
        }
    }
}

/// Resolved answers from setup. Built either by `prompt_user` (interactive) or by tests.
#[derive(Debug, Clone)]
pub(crate) struct SetupAnswers {
    pub scope: ConfigScope,
    pub agents: Vec<CodingAgent>,
    pub backends: Vec<ObservabilityBackend>,
    pub openinference_endpoint: Option<String>,
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
        (CodingAgent::Cursor, "cursor-agent"),
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
/// The shape mirrors the schema landed in Phase 1: `[observability]`, `[export.openinference]`,
/// `[agents.<name>]`. Sections are only emitted when the user opted into the corresponding
/// behavior so the resulting file stays minimal.
pub(crate) fn build_config(answers: &SetupAnswers) -> DocumentMut {
    let mut doc = DocumentMut::new();

    if answers.backends.contains(&ObservabilityBackend::Atif) {
        let mut observability = Table::new();
        observability["atif_dir"] = value("./atif");
        doc["observability"] = Item::Table(observability);
    }

    if answers
        .backends
        .contains(&ObservabilityBackend::OpenInference)
        && let Some(endpoint) = answers.openinference_endpoint.as_deref()
    {
        let mut export = Table::new();
        let mut openinference = Table::new();
        openinference["endpoint"] = value(endpoint);
        export.insert("openinference", Item::Table(openinference));
        doc["export"] = Item::Table(export);
    }

    if !answers.agents.is_empty() {
        let mut agents_table = Table::new();
        for agent in &answers.agents {
            let (key, command) = match agent {
                CodingAgent::ClaudeCode => ("claude", "claude"),
                CodingAgent::Codex => ("codex", "codex"),
                CodingAgent::Cursor => ("cursor", "cursor-agent"),
                CodingAgent::Hermes => ("hermes", "hermes"),
            };
            let mut agent_table = Table::new();
            agent_table["command"] = value(command);
            if matches!(agent, CodingAgent::Hermes)
                && let Some(path) = answers.hermes_hooks_path.as_deref()
            {
                agent_table["hooks_path"] = value(path.display().to_string());
            }
            agents_table.insert(key, Item::Table(agent_table));
        }
        doc["agents"] = Item::Table(agents_table);
    }

    doc
}

/// Writes the setup's TOML document to the scope-appropriate path(s).
///
/// When `merge_scope` is `Some(agent)`, an existing `config.toml` at the target path is parsed
/// and only the sections owned by THIS wizard run are replaced: `[observability]`,
/// `[export.openinference]`, `[plugins]`, and the single `[agents.<agent>]` block. Other
/// `[agents.*]` blocks are preserved. When `merge_scope` is `None`, the file is overwritten
/// outright with the wizard's full output (the user explicitly chose which agents to include).
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
        let project_dir = cwd.join(".nemo-flow");
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

// Resolves the global nemo-flow config directory. Prefers `$XDG_CONFIG_HOME/nemo-flow` (matches
// `config::user_config_dir`), falling back to `<home>/.config/nemo-flow`. Tests that pass a
// tempdir for `home` get hermetic paths unless they set XDG_CONFIG_HOME explicitly.
fn global_config_dir(home: &Path) -> PathBuf {
    if let Some(base) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(base).join("nemo-flow");
    }
    home.join(".config").join("nemo-flow")
}

// Writes the wizard-built `doc` to `path`. When `merge_scope` is `Some(agent)` and the file
// already exists, preserves any `[agents.<other>]` blocks while replacing the shared sections
// and the target agent's block. When `merge_scope` is `None`, just overwrites the file.
fn write_or_merge(
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
    // Wizard-owned sections use REPLACE semantics: if the user re-runs setup and the new doc
    // omits a section, the previous override is removed too. Otherwise accepting the default
    // (e.g. dropping a custom `openai_base_url`) could not actually revert the override —
    // the old value would silently survive.
    replace_section(&mut existing, doc, "observability");
    replace_section(&mut existing, doc, "export");
    replace_section(&mut existing, doc, "upstream");
    // `plugins` is not wizard-owned (users may hand-edit it). Preserve on omission.
    merge_section(&mut existing, doc, "plugins");
    merge_agents_entry(&mut existing, doc, agent_key);
    std::fs::write(path, existing.to_string())?;
    Ok(())
}

// Copies a top-level section from `src` into `dst`, replacing any existing entry under the same
// key. If `src` does not contain the section, the existing entry in `dst` is left as-is.
// Use for shared/hand-edited sections the wizard does not own.
fn merge_section(dst: &mut DocumentMut, src: &DocumentMut, key: &str) {
    if let Some(item) = src.get(key) {
        dst[key] = item.clone();
    }
}

// Like `merge_section`, but when `src` omits the key the existing entry in `dst` is removed.
// Use for wizard-owned sections (the wizard's output is authoritative for these keys).
fn replace_section(dst: &mut DocumentMut, src: &DocumentMut, key: &str) {
    match src.get(key) {
        Some(item) => dst[key] = item.clone(),
        None => {
            dst.remove(key);
        }
    }
}

// Replaces the single `[agents.<agent>]` block in `dst` with the one from `src`. If `src` does
// not contain that block, the existing entry in `dst` is left as-is.
fn merge_agents_entry(dst: &mut DocumentMut, src: &DocumentMut, agent_key: &str) {
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
    let path = cwd.join(".nemo-flow").join("config.toml");
    if !path.exists() {
        println!("  No project config to reset at {}", path.display());
        return Ok(());
    }
    match agent_hint {
        None => {
            std::fs::remove_file(&path)?;
            println!("  ✓ Removed {}", path.display());
            println!("  Run `nemo-flow config` to set up again.");
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

/// Drives the interactive setup. Returns the answers so callers can either save them or feed
/// the resulting config back into a launch path. Errors when stdin isn't a TTY so non-interactive
/// callers fall back to explicit flags instead of hanging on a prompt that nobody will see.
///
/// When `agent_hint` is `Some`, the agent multi-select is skipped — the user already declared
/// intent by typing `nemo-flow claude` (or another agent name), so respect that and only ask
/// scope + backends. To set up multiple agents, the user re-runs `nemo-flow config` later.
pub(crate) fn prompt_user(
    detected_agents: &[CodingAgent],
    agent_hint: Option<CodingAgent>,
) -> Result<SetupAnswers, CliError> {
    ensure_tty()?;
    let defaults = read_existing_defaults().unwrap_or_default();
    crate::banner::print_intro();
    match agent_hint {
        Some(agent) => {
            let (name, _) = agent_key_and_command(agent);
            println!("  Setting up observability for {name}.");
            println!("  Re-run `nemo-flow config` later to configure additional agents.");
        }
        None => {
            println!("  Let's set up observability for your coding agent.");
            println!("  This runs once. Re-run later with `nemo-flow config`.");
        }
    }
    // Only print the detected-agents listing for the unscoped wizard (`nemo-flow config`),
    // where the user is about to pick from the multi-select. When the agent was already chosen
    // via the easy-path shortcut (`nemo-flow codex`), listing the other three agents is noise.
    if agent_hint.is_none() {
        println!();
        print_detected_agents(detected_agents);
    }
    if defaults.has_any() {
        println!();
        println!("  Existing config detected — current values are pre-selected.");
    }
    println!();
    // Keybinding hint shown once: dialoguer's MultiSelect needs SPACE to toggle and ENTER to
    // confirm, but doesn't surface that itself. Without this line, users hit Enter expecting
    // to check a box and the prompt confirms with the wrong selection.
    println!(
        "  Tip: ↑/↓ to move, SPACE to toggle a checkbox, ENTER to confirm. Defaults are pre-selected."
    );
    println!();

    let theme = ColorfulTheme::default();
    let scope = ask_scope(&theme, defaults.scope)?;
    let agents = match agent_hint {
        Some(agent) => vec![agent],
        None => ask_agents(&theme, detected_agents, &defaults.agents)?,
    };
    let (backends, openinference_endpoint) = ask_backends(&theme, &defaults)?;

    if agents.contains(&CodingAgent::Codex) {
        print_codex_api_key_guide();
    }

    Ok(SetupAnswers {
        scope,
        agents,
        backends,
        openinference_endpoint,
        hermes_hooks_path: None,
    })
}

/// Returns the path the wizard will record under `[agents.hermes].hooks_path` and write hermes
/// hooks to. For `Both` scope, the project path wins (matches the config-merge precedence). For
/// `Global` scope, the home path wins. Returns `None` when hermes is not in the selection.
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
/// fires `nemo-flow hook-forward hermes` on every hook event after setup. Idempotent: existing
/// hook entries are preserved and our generated groups are appended only when missing.
///
/// Returns the list of paths actually written so callers can surface them to the user.
pub(crate) fn install_hermes_hooks(
    scope: ConfigScope,
    cwd: &Path,
    home: &Path,
) -> Result<Vec<PathBuf>, CliError> {
    let generated = hermes_hooks(&hook_forward_command("nemo-flow", CodingAgent::Hermes));
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

fn hermes_hook_targets(scope: ConfigScope, cwd: &Path, home: &Path) -> Vec<PathBuf> {
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
struct Defaults {
    scope: Option<ConfigScope>,
    agents: Vec<CodingAgent>,
    atif_enabled: bool,
    openinference_endpoint: Option<String>,
}

impl Defaults {
    fn has_any(&self) -> bool {
        self.scope.is_some()
            || !self.agents.is_empty()
            || self.atif_enabled
            || self.openinference_endpoint.is_some()
    }
}

/// Reads the highest-precedence existing config file and derives wizard defaults from it.
/// Workspace config wins over global; if both exist, scope defaults to `Both`. Missing or
/// malformed files yield `None` (the wizard then behaves as if no config existed).
fn read_existing_defaults() -> Option<Defaults> {
    let cwd = std::env::current_dir().ok()?;
    let home = home_dir();

    let workspace_path = cwd.join(".nemo-flow").join("config.toml");
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
        atif_enabled: doc
            .get("observability")
            .and_then(|i| i.as_table())
            .and_then(|t| t.get("atif_dir"))
            .is_some(),
        openinference_endpoint: doc
            .get("export")
            .and_then(|i| i.as_table())
            .and_then(|t| t.get("openinference"))
            .and_then(|i| i.as_table())
            .and_then(|t| t.get("endpoint"))
            .and_then(|i| i.as_str())
            .map(str::to_string),
    })
}

fn read_agents_from_doc(doc: &DocumentMut) -> Vec<CodingAgent> {
    let Some(table) = doc.get("agents").and_then(|i| i.as_table()) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for (key, _) in table.iter() {
        let agent = match key {
            "claude" => Some(CodingAgent::ClaudeCode),
            "codex" => Some(CodingAgent::Codex),
            "cursor" => Some(CodingAgent::Cursor),
            "hermes" => Some(CodingAgent::Hermes),
            _ => None,
        };
        if let Some(agent) = agent {
            found.push(agent);
        }
    }
    found
}

fn print_codex_api_key_guide() {
    // Codex supports two auth flows (see `codex-rs/login/src/auth/manager.rs`):
    //   1. ChatGPT-Plus PKCE OAuth via `codex --login` → tokens stored in `~/.codex/auth.json`
    //   2. OpenAI API key via `OPENAI_API_KEY` env var
    // The gateway routes to the correct upstream automatically: ChatGPT OAuth goes to
    // `chatgpt.com/backend-api/codex`, API key goes to `api.openai.com`.
    println!();
    println!("  ℹ Codex sends Responses-API requests through the gateway.");
    println!("    Authentication (pick one):");
    println!("      • ChatGPT-Plus login:  codex --login  (uses ~/.codex/auth.json)");
    println!("      • OpenAI API key:      export OPENAI_API_KEY=sk-...");
    println!("    When OPENAI_API_KEY is set the gateway uses it; otherwise the");
    println!("    ChatGPT-Plus OAuth token is forwarded to the ChatGPT backend.");
    println!();
}

fn ensure_tty() -> Result<(), CliError> {
    if !std::io::stdin().is_terminal() {
        return Err(CliError::Config(
            "interactive setup requires a TTY; pass `--config <path>` or set up \
             `.nemo-flow/config.toml` manually"
                .into(),
        ));
    }
    Ok(())
}

fn print_detected_agents(detected: &[CodingAgent]) {
    println!("  Detected agents on $PATH:");
    for agent in detected {
        let (name, _) = agent_key_and_command(*agent);
        println!("    ✓ {name}");
    }
    if detected.is_empty() {
        println!("    (none — you can still configure observability and add agents later)");
    }
}

fn ask_scope(
    theme: &ColorfulTheme,
    existing: Option<ConfigScope>,
) -> Result<ConfigScope, CliError> {
    let options = [ConfigScope::Project, ConfigScope::Global, ConfigScope::Both];
    let labels: Vec<&str> = options.iter().map(|s| s.label()).collect();
    // Cursor starts on the user's existing scope if there is one (so re-running the wizard
    // doesn't accidentally relocate their config), else `Project` per the design default.
    let default_idx = existing
        .and_then(|s| options.iter().position(|opt| *opt == s))
        .unwrap_or(0);
    let idx = Select::with_theme(theme)
        .with_prompt("Save config where?")
        .items(&labels)
        .default(default_idx)
        .interact()
        .map_err(setup_error)?;
    Ok(options[idx])
}

fn ask_agents(
    theme: &ColorfulTheme,
    detected: &[CodingAgent],
    configured: &[CodingAgent],
) -> Result<Vec<CodingAgent>, CliError> {
    let all_supported = [
        CodingAgent::ClaudeCode,
        CodingAgent::Codex,
        CodingAgent::Cursor,
        CodingAgent::Hermes,
    ];
    let labels: Vec<String> = all_supported
        .iter()
        .map(|a| {
            let (name, _) = agent_key_and_command(*a);
            name.to_string()
        })
        .collect();
    // Pre-check: union of "already in the existing config" and "detected on $PATH". The existing
    // entries take precedence — if the user previously deselected an agent that's on PATH, we
    // shouldn't re-check it for them. On first run (no existing config), this falls back to
    // pre-checking everything detected.
    let defaults: Vec<bool> = if configured.is_empty() {
        all_supported.iter().map(|a| detected.contains(a)).collect()
    } else {
        all_supported
            .iter()
            .map(|a| configured.contains(a))
            .collect()
    };
    let selected_idx = MultiSelect::with_theme(theme)
        .with_prompt("Which agents to observe?")
        .items(&labels)
        .defaults(&defaults)
        .interact()
        .map_err(setup_error)?;
    Ok(selected_idx.into_iter().map(|i| all_supported[i]).collect())
}

fn ask_backends(
    theme: &ColorfulTheme,
    existing: &Defaults,
) -> Result<(Vec<ObservabilityBackend>, Option<String>), CliError> {
    let options = [
        ObservabilityBackend::Atif,
        ObservabilityBackend::OpenInference,
    ];
    let labels: Vec<&str> = options.iter().map(|b| b.label()).collect();
    // Pre-check from existing config when present. On first run, falls back to ATIF on (zero
    // infra) and OpenInference off (needs an endpoint running).
    let defaults = if existing.has_any() {
        [
            existing.atif_enabled,
            existing.openinference_endpoint.is_some(),
        ]
    } else {
        [true, false]
    };
    let selected_idx = MultiSelect::with_theme(theme)
        .with_prompt("Observability backends?")
        .items(&labels)
        .defaults(&defaults)
        .interact()
        .map_err(setup_error)?;
    let backends: Vec<ObservabilityBackend> =
        selected_idx.into_iter().map(|i| options[i]).collect();

    let openinference_endpoint = if backends.contains(&ObservabilityBackend::OpenInference) {
        let initial = existing
            .openinference_endpoint
            .as_deref()
            .unwrap_or("http://localhost:6006/v1/traces");
        let endpoint: String = Input::with_theme(theme)
            .with_prompt("OpenInference endpoint URL")
            .with_initial_text(initial)
            .interact_text()
            .map_err(setup_error)?;
        Some(endpoint)
    } else {
        None
    };

    Ok((backends, openinference_endpoint))
}

/// Confirms the summary with the user before writing the file. Returns true if the user accepted.
/// Shows both the destination path(s) and the exact TOML body about to be written so the user
/// can verify what they're committing to instead of confirming a path blind.
pub(crate) fn confirm_summary(
    written_paths: &[PathBuf],
    doc: &DocumentMut,
) -> Result<bool, CliError> {
    println!();
    println!("  ─── Summary ─────────────────────────────────────────────");
    println!("  Will write to:");
    for path in written_paths {
        println!("    {}", path.display());
    }
    println!();
    println!("  Contents:");
    for line in doc.to_string().lines() {
        println!("    {line}");
    }
    println!();
    Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Looks good?")
        .default(true)
        .interact()
        .map_err(setup_error)
}

fn setup_error(err: dialoguer::Error) -> CliError {
    // dialoguer errors are mostly IO. Translate cancellation (Ctrl-C, EOF on stdin) into a
    // friendly "cancelled" message; surface anything else as the raw error.
    match err {
        dialoguer::Error::IO(io_err)
            if matches!(
                io_err.kind(),
                std::io::ErrorKind::Interrupted | std::io::ErrorKind::UnexpectedEof
            ) =>
        {
            CliError::Config("setup cancelled — no config saved".into())
        }
        other => CliError::Config(format!("setup error: {other}")),
    }
}

fn agent_key_and_command(agent: CodingAgent) -> (&'static str, &'static str) {
    match agent {
        CodingAgent::ClaudeCode => ("claude", "claude"),
        CodingAgent::Codex => ("codex", "codex"),
        CodingAgent::Cursor => ("cursor", "cursor-agent"),
        CodingAgent::Hermes => ("hermes", "hermes"),
    }
}

/// Top-level setup entry point used by `nemo-flow config` and the easy-path fallback.
/// Detects agents, prompts the user, writes the config, prints a final summary.
///
/// `agent_hint` carries the agent the user typed on the easy path (`nemo-flow claude`); when
/// `Some`, the agent multi-select is skipped because intent is already declared. `None` from
/// `nemo-flow config` asks the full set so users can configure multiple agents at once.
pub(crate) async fn run(agent_hint: Option<CodingAgent>) -> Result<(), CliError> {
    let detected = detect_installed_agents();
    let mut answers = prompt_user(&detected, agent_hint)?;

    let cwd = std::env::current_dir()?;
    let home = home_dir().ok_or_else(|| {
        CliError::Config("cannot determine home directory (set $HOME or $USERPROFILE)".into())
    })?;
    answers.hermes_hooks_path =
        hermes_hooks_path_for_scope(&answers.agents, answers.scope, &cwd, &home);

    let doc = build_config(&answers);
    let mut preview_paths = preview_paths(answers.scope, &cwd, &home);
    preview_paths.extend(
        hermes_hook_targets(answers.scope, &cwd, &home)
            .into_iter()
            .filter(|_| answers.agents.contains(&CodingAgent::Hermes)),
    );

    if !confirm_summary(&preview_paths, &doc)? {
        return Err(CliError::Config("setup cancelled — no config saved".into()));
    }

    let mut written = save_config(&doc, answers.scope, &cwd, &home, agent_hint)?;
    if answers.agents.contains(&CodingAgent::Hermes) {
        written.extend(install_hermes_hooks(answers.scope, &cwd, &home)?);
    }
    println!();
    println!("  ✓ Saved:");
    for path in &written {
        println!("    {}", path.display());
    }
    println!();
    Ok(())
}

fn preview_paths(scope: ConfigScope, cwd: &Path, home: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if matches!(scope, ConfigScope::Project | ConfigScope::Both) {
        paths.push(cwd.join(".nemo-flow").join("config.toml"));
    }
    if matches!(scope, ConfigScope::Global | ConfigScope::Both) {
        paths.push(global_config_dir(home).join("config.toml"));
    }
    paths
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[cfg(test)]
#[path = "../tests/coverage/setup_tests.rs"]
mod tests;
