// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Terminal-only prompt adapter for first-run configuration.

use std::io::IsTerminal;
use std::path::PathBuf;

use dialoguer::theme::ColorfulTheme;
use dialoguer::{Confirm, MultiSelect};
use toml_edit::DocumentMut;

use super::model::{
    SetupAnswers, agent_key_and_command, build_config, detect_installed_agents, home_dir,
    plugins_edit_command, plugins_resume_command, preview_paths, read_existing_defaults,
    save_config,
};
use crate::agents::CodingAgent;
use crate::error::CliError;

/// Prompts for the agents selected by the user.
///
/// When `agent_hint` is present, the agent picker is skipped because the command already
/// identified the requested agent.
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
    let agents = match agent_hint {
        Some(agent) => vec![agent],
        None => ask_agents(&theme, detected_agents, &defaults.agents)?,
    };
    if agents.contains(&CodingAgent::Codex) {
        print_codex_api_key_guide();
    }

    Ok(SetupAnswers { agents })
}

pub(super) async fn run(
    agent_hint: Option<CodingAgent>,
    explicit_plugin_path: Option<PathBuf>,
) -> Result<(), CliError> {
    let detected = detect_installed_agents();
    let answers = prompt_user(&detected, agent_hint)?;

    let home = home_dir().ok_or_else(|| {
        CliError::Config("cannot determine home directory (set $HOME or $USERPROFILE)".into())
    })?;
    let doc = build_config(&answers);
    let preview_paths = preview_paths(&home);

    if !confirm_summary(&preview_paths, &doc)? {
        return Err(CliError::Config("setup cancelled — no config saved".into()));
    }

    let written = save_config(&doc, &home, agent_hint)?;
    println!();
    println!("  ✓ Saved:");
    for path in &written {
        println!("    {}", path.display());
    }
    println!();
    if answers.agents.contains(&CodingAgent::Pi) {
        offer_pi_extension();
    }
    continue_to_plugins(explicit_plugin_path)
}

/// Offer to install the pi extension when setup selected pi and pi has no copy.
///
/// pi is the one agent whose shortcut is not install-free. Claude Code and Codex take
/// their hooks from the launcher, so setup is the last step they need; pi's hooks can only
/// originate inside an extension, so `nemo-relay pi` fails on a clean machine until one
/// exists. Without this the wizard would finish, congratulate the user, and then hand them
/// a command that does not work.
///
/// Nothing here can fail the wizard. The configuration is already saved by this point, the
/// install is optional, and a declined or broken install still leaves a working setup for
/// the manual routes -- so every path prints how to do it later and returns.
fn offer_pi_extension() {
    if !crate::agents::pi::install::setup_install_available() {
        return;
    }
    println!("  ℹ pi hooks originate inside the NeMo Relay pi extension, so `nemo-relay pi`");
    println!("    needs it present. Relay can write it into pi's own extensions directory.");
    println!();
    match confirm_pi_extension_install() {
        Ok(true) => run_pi_extension_install(),
        Ok(false) => print_pi_extension_skipped(),
        // An interrupted prompt is a skip, exactly as it is for plugin setup.
        Err(error) if super::plugin_prompt_was_interrupted(&error) => print_pi_extension_skipped(),
        Err(error) => {
            println!("  Could not read a response ({error}).");
            print_pi_extension_skipped();
        }
    }
}

fn confirm_pi_extension_install() -> Result<bool, dialoguer::Error> {
    Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Install the NeMo Relay pi extension now?")
        .default(true)
        .interact()
}

fn run_pi_extension_install() {
    let request = crate::installation::InstallRequest {
        install_dir: None,
        force: false,
        dry_run: false,
        skip_doctor: false,
    };
    if let Err(error) = crate::agents::pi::install::install(request) {
        println!();
        println!("  The pi extension was not installed: {error}");
        print_pi_extension_skipped();
    }
}

fn print_pi_extension_skipped() {
    println!();
    println!("  Install it later with `nemo-relay install pi`.");
    println!();
}

/// After the base config is saved, offers to continue into plugin configuration in-process.
///
/// Prompts once. On acceptance it runs the existing plugin editor targeting an explicit runtime
/// plugin path when present, otherwise the user plugin file. On decline it reports that the base config was saved,
/// that plugin setup was skipped, and prints the command to resume later. Prompt interruption is
/// treated as a skip; other prompt or editor failures surface an error that makes clear the base
/// config remains saved. The saved `config.toml` is never rolled back here.
fn continue_to_plugins(explicit_plugin_path: Option<PathBuf>) -> Result<(), CliError> {
    let resume_command = plugins_resume_command(explicit_plugin_path.as_deref());
    let proceed = match confirm_plugin_setup() {
        Ok(proceed) => proceed,
        Err(error) if super::plugin_prompt_was_interrupted(&error) => {
            print_plugins_skipped(&resume_command);
            return Ok(());
        }
        Err(error) => {
            return Err(CliError::Config(format!(
                "plugin setup did not complete; base configuration remains saved. \
                 Resume with `{}`. Cause: {error}",
                resume_command
            )));
        }
    };
    if !proceed {
        print_plugins_skipped(&resume_command);
        return Ok(());
    }
    let result = crate::plugins::edit(plugins_edit_command(explicit_plugin_path));
    result.map_err(|error| {
        let cause = match error {
            CliError::Config(message) => message,
            other => other.to_string(),
        };
        CliError::Config(format!(
            "plugin setup did not complete; base configuration remains saved. \
             Resume with `{}`. Cause: {cause}",
            resume_command
        ))
    })
}

pub(super) fn confirm_plugin_setup() -> Result<bool, dialoguer::Error> {
    Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Configure Relay plugins now?")
        .default(true)
        .interact()
}

pub(super) fn print_plugins_skipped(resume_command: &str) {
    println!();
    println!("  Base configuration saved. Plugin configuration skipped.");
    println!("  Configure plugins later with `{resume_command}`.");
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
             `$XDG_CONFIG_HOME/nemo-relay/config.toml` manually"
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

fn ask_agents(
    theme: &ColorfulTheme,
    detected: &[CodingAgent],
    configured: &[CodingAgent],
) -> Result<Vec<CodingAgent>, CliError> {
    // Every supported agent, not a hand-written pair. Saving an unscoped result replaces
    // the whole `[agents]` table, so an agent missing from this list is removed from the
    // user's config by a wizard that never offered it.
    let all_supported = CodingAgent::ALL;
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
