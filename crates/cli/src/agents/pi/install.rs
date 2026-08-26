// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Relay-managed installation of the pi extension.
//!
//! **Why the file drop rather than `pi install`.** pi records a `pi install <source>` in
//! the `packages` array of its own `settings.json`, under its own `withLock`, and shelling
//! out to it is often described as the safer route for exactly that reason. That reasoning
//! is about the settings write, and this route has none: pi auto-discovers every package
//! directory under `<agent dir>/extensions`, so writing one there registers the extension
//! without touching `settings.json` at all. No lock to contend for, no package-manager
//! semantics to inherit, no need for `pi` to be on `PATH`, and an uninstall that removes a
//! directory Relay created rather than un-editing an array under someone else's lock. It
//! is also the route the pi guide already tells users to run by hand.
//!
//! **Why a state file.** The guide's own `cp -r` lands in this same directory, so "a
//! directory exists here" cannot mean "Relay put it there". Every install records what it
//! wrote and the hash of each file, and uninstall removes only files that still match. A
//! directory with no state file is someone else's and is never touched, `--force`
//! included.
//!
//! **Why installing can refuse.** `pi -e` adds to pi's extension set rather than replacing
//! it, and pi de-duplicates by path, so a second copy elsewhere is a second package: every
//! hook fires twice, each turn closes as superseded by its own duplicate, and the
//! inline-shell gate decides one command twice. The launcher already refuses to start in
//! that state, so creating it here would break a working setup. This checks first.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::assets::{EXTENSION_FILES, EXTENSION_VERSION};
use crate::error::CliError;
use crate::installation::{InstallRequest, UninstallRequest};

/// Directory Relay writes under pi's auto-discovery directory.
///
/// The same name the guide's manual `cp -r` uses, deliberately: a user who followed the
/// guide and then runs `nemo-relay install pi` gets told their copy is already there,
/// rather than silently ending up with two.
const INSTALL_DIR_NAME: &str = "nemo-relay";

/// Marker recording what Relay wrote, so uninstall can be exact.
///
/// Dot-prefixed because pi resolves a package's entry points through its `package.json`
/// manifest and ignores everything else in the directory, but a leading dot keeps it out
/// of the way of anything that lists the directory.
const STATE_FILE: &str = ".nemo-relay-install.json";

/// Bumped when the state shape changes in a way an older CLI cannot read.
const STATE_SCHEMA: u64 = 1;

/// `~/.pi/agent/extensions/nemo-relay`, honoring pi's own directory override.
pub(crate) fn install_root() -> Option<PathBuf> {
    Some(super::doctor::user_extensions_dir()?.join(INSTALL_DIR_NAME))
}

/// Whether a Relay-managed install is present, for `uninstall all` and for diagnostics.
pub(crate) fn is_installed() -> bool {
    install_root().is_some_and(|root| matches!(occupant(&root), Occupant::Managed(_)))
}

/// The version of the extension a Relay-managed install left in place, if there is one.
pub(crate) fn installed_version() -> Option<String> {
    match occupant(&install_root()?) {
        Occupant::Managed(state) => Some(state.relay_version),
        _ => None,
    }
}

/// Whether first-run setup has an install worth offering.
///
/// Only when pi has no copy at all, in any scope. A copy that is already installed needs
/// no offer, and a project-scoped one -- the trap the guide warns about twice -- must not
/// quietly become the reason a *second* copy appears, because two copies double every hook
/// and stop the launcher. `doctor` reports a project-scoped copy on its own; setup stays
/// out of it.
pub(crate) fn setup_install_available() -> bool {
    install_root().is_some() && !super::doctor::extension_configured()
}

/// What is sitting at the install path.
enum Occupant {
    /// Nothing there.
    Vacant,
    /// A Relay-managed install this CLI can read.
    Managed(InstallState),
    /// A Relay-managed install written by a newer CLI than this one.
    ///
    /// Distinct from `Foreign` because it *is* ours and refusing to touch it forever
    /// would be wrong; the answer is to upgrade, not to delete by hand.
    FutureSchema(u64),
    /// Something Relay did not write -- a hand-placed copy, most likely.
    Foreign,
}

/// What a Relay-managed install recorded about itself.
struct InstallState {
    /// The `nemo-relay` version that wrote it, which is also the extension version it
    /// wrote: the two are tied by a test in the asset module.
    relay_version: String,
    files: Vec<RecordedFile>,
}

/// One file an install wrote, and what it looked like when it was written.
struct RecordedFile {
    path: String,
    sha256: String,
}

fn occupant(root: &Path) -> Occupant {
    if !root.exists() {
        return Occupant::Vacant;
    }
    let Ok(raw) = std::fs::read_to_string(root.join(STATE_FILE)) else {
        return Occupant::Foreign;
    };
    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return Occupant::Foreign;
    };
    let schema = value.get("schema").and_then(Value::as_u64).unwrap_or(0);
    if schema != STATE_SCHEMA {
        return Occupant::FutureSchema(schema);
    }
    let relay_version = value
        .get("relay_version")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let files = value
        .get("files")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| {
                    Some(RecordedFile {
                        path: entry.get("path").and_then(Value::as_str)?.to_string(),
                        sha256: entry.get("sha256").and_then(Value::as_str)?.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Occupant::Managed(InstallState {
        relay_version,
        files,
    })
}

fn digest(contents: &str) -> String {
    let digest = Sha256::digest(contents.as_bytes());
    format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

/// Files a managed install wrote that no longer match what it recorded.
///
/// These are the user's edits, not Relay's files, and both install and uninstall treat
/// them as the user's to keep.
fn modified_files(root: &Path, state: &InstallState) -> Vec<String> {
    state
        .files
        .iter()
        .filter(|recorded| {
            std::fs::read_to_string(root.join(&recorded.path))
                .map(|actual| digest(&actual) != recorded.sha256)
                .unwrap_or(false)
        })
        .map(|recorded| recorded.path.clone())
        .collect()
}

fn current_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn unsupported_install_dir(operation: &str) -> CliError {
    CliError::Install(format!(
        "`--install-dir` does not apply to pi: pi discovers extensions under its own agent \
         directory, so Relay has no install root to place elsewhere. Set \
         `PI_CODING_AGENT_DIR` to move pi's whole configuration, or drop the flag to \
         {operation} at pi's own location"
    ))
}

fn no_agent_dir() -> CliError {
    CliError::Install(
        "could not resolve pi's agent directory: no home directory was found and \
         `PI_CODING_AGENT_DIR` is unset"
            .into(),
    )
}

pub(crate) fn install(request: InstallRequest) -> Result<ExitCode, CliError> {
    if request.install_dir.is_some() {
        return Err(unsupported_install_dir("install"));
    }
    let root = install_root().ok_or_else(no_agent_dir)?;

    // Refuse before writing anything: a second loading copy doubles every hook, and the
    // launcher refuses to start once it exists. Checked before the occupant so a user who
    // already installed through `pi install` is told about *that* copy rather than about
    // this directory.
    if let Some(other) = super::doctor::conflicting_extension_site(&current_dir(), &root) {
        return Err(CliError::Install(format!(
            "another copy of the NeMo Relay pi extension is already installed at {}, and pi \
             would load it beside this one. pi de-duplicates its extension set by path \
             rather than by package, so both would register hooks and every turn, tool and \
             inline-shell event would be reported twice. Remove that copy first -- or keep \
             it and skip this install, because it already works",
            other.display()
        )));
    }

    let action = plan_install(&root, request.force)?;
    if request.dry_run {
        print_install_plan(&root, action);
        return Ok(ExitCode::SUCCESS);
    }

    write_extension(&root)?;
    println!("{action} the NeMo Relay pi extension at {}", root.display());
    println!(
        "  version {EXTENSION_VERSION}, {} files",
        EXTENSION_FILES.len()
    );

    note_missing_host();
    if !request.skip_doctor {
        verify_install(&root);
    }
    Ok(ExitCode::SUCCESS)
}

/// Note, rather than refuse, when pi itself is not on `PATH`.
///
/// This install writes a directory and never drives pi, so a missing pi does not stop it
/// and installing ahead of pi is a reasonable thing to do. It is still worth saying: the
/// extension does nothing until pi loads it, and silence here would read as a setup that
/// is ready. Version is deliberately not probed -- that means running pi, and `doctor`
/// already reports the floor and the verified ceiling.
fn note_missing_host() {
    if crate::process::resolve_executable(crate::agents::CodingAgent::Pi.executable()).is_none() {
        println!(
            "  note: `pi` was not found on PATH. The extension is installed and pi will \
             discover it once pi is installed; run `nemo-relay doctor pi` then to check the \
             version"
        );
    }
}

/// What an install would do here, or why it will not.
#[derive(Clone, Copy)]
enum InstallAction {
    Wrote,
    Reinstalled,
    Upgraded,
}

impl std::fmt::Display for InstallAction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Wrote => "Installed",
            Self::Reinstalled => "Reinstalled",
            Self::Upgraded => "Upgraded",
        })
    }
}

fn plan_install(root: &Path, force: bool) -> Result<InstallAction, CliError> {
    match occupant(root) {
        Occupant::Vacant => Ok(InstallAction::Wrote),
        Occupant::Foreign => Err(CliError::Install(format!(
            "{} already exists and was not written by NeMo Relay -- most likely the `cp -r` \
             the pi guide documents. It is left alone, `--force` included, because Relay \
             cannot tell an unmanaged copy from one you edited. It already works: \
             `nemo-relay run --agent pi` finds it. To replace it with a managed install, \
             remove the directory yourself and run this again",
            root.display()
        ))),
        Occupant::FutureSchema(schema) => Err(CliError::Install(format!(
            "{} was written by a newer nemo-relay whose install state is version {schema}; \
             this build understands version {STATE_SCHEMA}. Upgrade nemo-relay and run this \
             again rather than removing the directory by hand",
            root.display()
        ))),
        Occupant::Managed(state) => plan_over_managed(root, &state, force),
    }
}

fn plan_over_managed(
    root: &Path,
    state: &InstallState,
    force: bool,
) -> Result<InstallAction, CliError> {
    let modified = modified_files(root, state);
    if !modified.is_empty() && !force {
        return Err(CliError::Install(format!(
            "{} holds a NeMo Relay install whose files have been edited since it was \
             written: {}. Overwriting would discard those edits, so this stops instead. \
             Re-run with `--force` to overwrite them",
            root.display(),
            modified.join(", ")
        )));
    }
    // Replacing our own unmodified files needs no flag. `--force` guards the user's edits,
    // and there are none: refusing here would make every upgrade a two-step.
    if state.relay_version == EXTENSION_VERSION {
        Ok(InstallAction::Reinstalled)
    } else {
        Ok(InstallAction::Upgraded)
    }
}

fn print_install_plan(root: &Path, action: InstallAction) {
    println!("Plan: {action} at {}", root.display());
    println!("  version {EXTENSION_VERSION}");
    for file in EXTENSION_FILES {
        println!("  write {}", root.join(file.path).display());
    }
    println!("  write {}", root.join(STATE_FILE).display());
}

fn write_extension(root: &Path) -> Result<(), CliError> {
    let mut recorded = Vec::with_capacity(EXTENSION_FILES.len());
    for file in EXTENSION_FILES {
        let target = root.join(file.path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                CliError::Install(format!("could not create {}: {error}", parent.display()))
            })?;
        }
        std::fs::write(&target, file.contents).map_err(|error| {
            CliError::Install(format!("could not write {}: {error}", target.display()))
        })?;
        recorded.push(json!({ "path": file.path, "sha256": digest(file.contents) }));
    }
    let state = json!({
        "schema": STATE_SCHEMA,
        "relay_version": EXTENSION_VERSION,
        "files": recorded,
    });
    let rendered = serde_json::to_string_pretty(&state)
        .map_err(|error| CliError::Install(format!("could not render install state: {error}")))?;
    let state_path = root.join(STATE_FILE);
    std::fs::write(&state_path, format!("{rendered}\n")).map_err(|error| {
        CliError::Install(format!("could not write {}: {error}", state_path.display()))
    })
}

/// Confirm pi will actually load what was just written.
///
/// Writing the files is not the same as pi loading them: a `packages` filter in the user's
/// own `settings.json` can switch an auto-discovered package off, and the symptom of that
/// is missing spans rather than an error. Reported rather than failed, because the install
/// itself succeeded and the fix is in pi's configuration.
fn verify_install(root: &Path) {
    match super::doctor::launchable_extension_path(&current_dir()) {
        Some(resolved) if resolved.starts_with(root) => {
            println!("  pi extension load path: {}", resolved.display());
        }
        Some(resolved) => println!(
            "  warning: pi resolves the NeMo Relay extension to {} rather than the install \
             just written. Run `nemo-relay doctor pi`",
            resolved.display()
        ),
        None => println!(
            "  warning: the install was written but pi does not resolve it -- a `packages` \
             filter in pi's own settings.json can switch an auto-discovered package off. \
             Run `nemo-relay doctor pi`"
        ),
    }
}

pub(crate) fn uninstall(request: UninstallRequest) -> Result<ExitCode, CliError> {
    if request.install_dir.is_some() {
        return Err(unsupported_install_dir("uninstall"));
    }
    let root = install_root().ok_or_else(no_agent_dir)?;
    let state = match occupant(&root) {
        Occupant::Managed(state) => state,
        Occupant::Vacant => {
            return Err(CliError::Install(format!(
                "no NeMo Relay-managed pi extension install was found at {}",
                root.display()
            )));
        }
        Occupant::Foreign => {
            return Err(CliError::Install(format!(
                "{} was not written by `nemo-relay install pi`, so it is not Relay's to \
                 remove -- most likely the `cp -r` the pi guide documents. Remove it \
                 yourself if that is what you want",
                root.display()
            )));
        }
        Occupant::FutureSchema(schema) => {
            return Err(CliError::Install(format!(
                "{} records install state version {schema}, which this build does not \
                 understand; it understands version {STATE_SCHEMA}. Upgrade nemo-relay and \
                 run this again rather than removing the directory by hand",
                root.display()
            )));
        }
    };

    let kept = modified_files(&root, &state);
    if request.dry_run {
        print_uninstall_plan(&root, &state, &kept);
        return Ok(ExitCode::SUCCESS);
    }
    remove_recorded(&root, &state, &kept)?;
    report_uninstall(&root, &kept);
    Ok(ExitCode::SUCCESS)
}

fn print_uninstall_plan(root: &Path, state: &InstallState, kept: &[String]) {
    println!(
        "Plan: remove the NeMo Relay pi extension at {}",
        root.display()
    );
    for recorded in &state.files {
        let verb = if kept.contains(&recorded.path) {
            "keep (edited)"
        } else {
            "remove"
        };
        println!("  {verb} {}", root.join(&recorded.path).display());
    }
    println!("  remove {}", root.join(STATE_FILE).display());
}

fn remove_recorded(root: &Path, state: &InstallState, kept: &[String]) -> Result<(), CliError> {
    for recorded in &state.files {
        if kept.contains(&recorded.path) {
            continue;
        }
        let target = root.join(&recorded.path);
        match std::fs::remove_file(&target) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(CliError::Install(format!(
                    "could not remove {}: {error}",
                    target.display()
                )));
            }
        }
    }
    let state_path = root.join(STATE_FILE);
    if let Err(error) = std::fs::remove_file(&state_path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        return Err(CliError::Install(format!(
            "could not remove {}: {error}",
            state_path.display()
        )));
    }
    prune_empty_dirs(root, state);
    Ok(())
}

/// Remove directories the install created, but only while they are empty.
///
/// `remove_dir` rather than `remove_dir_all` throughout: a non-empty directory holds
/// something this install did not write, and that is the user's.
fn prune_empty_dirs(root: &Path, state: &InstallState) {
    let mut nested: Vec<PathBuf> = state
        .files
        .iter()
        .filter_map(|recorded| Path::new(&recorded.path).parent())
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(|parent| root.join(parent))
        .collect();
    nested.sort();
    nested.dedup();
    // Deepest first, so a nested directory does not keep its parent alive.
    nested.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for dir in nested {
        let _ = std::fs::remove_dir(&dir);
    }
    let _ = std::fs::remove_dir(root);
}

fn report_uninstall(root: &Path, kept: &[String]) {
    if kept.is_empty() {
        println!(
            "Removed the NeMo Relay pi extension from {}",
            root.display()
        );
        return;
    }
    println!(
        "Removed the NeMo Relay pi extension from {}, except for files edited since it was \
         installed: {}",
        root.display(),
        kept.join(", ")
    );
    println!("  those files and their directories are left in place");
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/pi_install_tests.rs"]
mod tests;
