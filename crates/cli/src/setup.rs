// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! First-run setup for `nemo-relay` configuration.
//!
//! Drives the required scope and agent prompts, then writes a `config.toml` to the chosen scope. Pure
//! helpers (`detect_installed_agents`, `build_config`, `save_config`) are split out from the
//! `dialoguer`-driven orchestrator so the data path can be unit-tested without a TTY.
//!
//! Keep this module focused on TTY and `dialoguer` orchestration. New testable setup behavior
//! should live in `setup/model.rs`, with focused unit tests, so Codecov does not depend on
//! exercising interactive prompt loops.

use std::io::IsTerminal;
use std::path::PathBuf;

use dialoguer::theme::ColorfulTheme;
use dialoguer::{Confirm, MultiSelect, Select};
use toml_edit::DocumentMut;

use crate::config::CodingAgent;
use crate::error::CliError;

mod model;

pub(crate) use self::model::reset;
use self::model::{
    ConfigScope, SetupAnswers, agent_key_and_command, build_config, detect_installed_agents,
    hermes_config_path_for_agents, home_dir, install_hermes_integration,
    plugins_edit_command_for_scope, plugins_resume_command, preview_paths, read_existing_defaults,
    save_config,
};

#[cfg(test)]
use self::model::{Defaults, global_config_dir, read_agents_from_doc, write_or_merge};

#[cfg(test)]
use self::model::detect_installed_agents_in;

///
/// When `agent_hint` is `Some`, the agent multi-select is skipped — the user already declared
/// intent by typing `nemo-relay claude` (or another agent name), so respect that and only ask
/// scope and agents. To set up multiple agents, the user re-runs `nemo-relay config` later.
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
            println!("  Setting up {name}.");
            println!("  Re-run `nemo-relay config` later to configure additional agents.");
        }
        None => {
            println!("  Let's set up your coding agent.");
            println!("  This runs once. Re-run later with `nemo-relay config`.");
        }
    }
    // Only print the detected-agents listing for the unscoped wizard (`nemo-relay config`),
    // where the user is about to pick from the multi-select. When the agent was already chosen
    // via the easy-path shortcut (`nemo-relay codex`), listing the other two agents is noise.
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
    if agents.contains(&CodingAgent::Codex) {
        print_codex_api_key_guide();
    }

    Ok(SetupAnswers {
        scope,
        agents,
        hermes_hooks_path: None,
    })
}

/// Top-level setup entry point used by `nemo-relay config` and the easy-path fallback.
/// Detects agents, prompts the user, writes the config, prints a final summary.
///
/// `agent_hint` carries the agent the user typed on the easy path (`nemo-relay claude`); when
/// `Some`, the agent multi-select is skipped because intent is already declared. `None` from
/// `nemo-relay config` asks the full set so users can configure multiple agents at once.
pub(crate) async fn run(agent_hint: Option<CodingAgent>) -> Result<(), CliError> {
    let detected = detect_installed_agents();
    let mut answers = prompt_user(&detected, agent_hint)?;

    let cwd = std::env::current_dir()?;
    let home = home_dir().ok_or_else(|| {
        CliError::Config("cannot determine home directory (set $HOME or $USERPROFILE)".into())
    })?;
    answers.hermes_hooks_path = hermes_config_path_for_agents(&answers.agents, &home);

    let doc = build_config(&answers);
    let mut preview_paths = preview_paths(answers.scope, &cwd, &home);
    preview_paths.extend(answers.hermes_hooks_path.iter().cloned());

    if !confirm_summary(&preview_paths, &doc)? {
        return Err(CliError::Config("setup cancelled — no config saved".into()));
    }

    let mut written = save_config(&doc, answers.scope, &cwd, &home, agent_hint)?;
    if answers.agents.contains(&CodingAgent::Hermes) {
        written.extend(install_hermes_integration(&home)?);
    }
    println!();
    println!("  ✓ Saved:");
    for path in &written {
        println!("    {}", path.display());
    }
    println!();
    continue_to_plugins(answers.scope)
}

/// After the base config is saved, offers to continue into plugin configuration in-process.
///
/// Prompts once. On acceptance it runs the existing plugin editor targeting the scope derived
/// from the base setup (project for `Project`/`Both`, user for `Global`). On decline it reports
/// that the base config was saved, that plugin setup was skipped, and prints the command to
/// resume later. Prompt interruption is treated as a skip; other prompt or editor failures
/// surface an error that makes clear the base config remains saved. The saved `config.toml`
/// is never rolled back here.
fn continue_to_plugins(scope: ConfigScope) -> Result<(), CliError> {
    let proceed = match Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Configure Relay plugins now?")
        .default(true)
        .interact()
    {
        Ok(proceed) => proceed,
        Err(error) if plugin_prompt_was_interrupted(&error) => {
            print_plugins_skipped(scope);
            return Ok(());
        }
        Err(error) => {
            return Err(CliError::Config(format!(
                "plugin setup did not complete; base configuration remains saved. \
                 Resume with `{}`. Cause: {error}",
                plugins_resume_command(scope)
            )));
        }
    };
    if !proceed {
        print_plugins_skipped(scope);
        return Ok(());
    }
    crate::plugins::edit(plugins_edit_command_for_scope(scope)).map_err(|error| {
        let cause = match error {
            CliError::Config(message) => message,
            other => other.to_string(),
        };
        CliError::Config(format!(
            "plugin setup did not complete; base configuration remains saved. \
             Resume with `{}`. Cause: {cause}",
            plugins_resume_command(scope)
        ))
    })
}

fn plugin_prompt_was_interrupted(error: &dialoguer::Error) -> bool {
    matches!(
        error,
        dialoguer::Error::IO(io_error)
            if matches!(
                io_error.kind(),
                std::io::ErrorKind::Interrupted | std::io::ErrorKind::UnexpectedEof
            )
    )
}

fn print_plugins_skipped(scope: ConfigScope) {
    println!();
    println!("  Base configuration saved. Plugin configuration skipped.");
    println!(
        "  Configure plugins later with `{}`.",
        plugins_resume_command(scope)
    );
    println!();
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
             `.nemo-relay/config.toml` manually"
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
        println!("    (none — you can still add agents later)");
    }
}

fn ask_scope(
    theme: &ColorfulTheme,
    existing: Option<ConfigScope>,
) -> Result<ConfigScope, CliError> {
    let options = [ConfigScope::Project, ConfigScope::Global, ConfigScope::Both];
    let labels: Vec<&str> = options.iter().map(|s| s.label()).collect();
    // Start on the user's existing scope if there is one (so re-running the wizard doesn't
    // accidentally relocate their config), else `Project` per the design default.
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

#[cfg(test)]
#[path = "../tests/coverage/setup_tests.rs"]
mod tests;
