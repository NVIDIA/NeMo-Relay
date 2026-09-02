// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use super::*;
use crate::agents::pi::doctor::{PI_AGENT_DIR_ENV, extension_configured};
use crate::agents::pi::launch::PI_EXTENSION_PATH_ENV;
use crate::test_support::EnvScope;

/// Point pi's whole configuration at a temp directory and unset the explicit override, so a
/// developer's own pi install can neither satisfy nor break these.
fn scoped(agent_dir: &Path) -> EnvScope {
    EnvScope::set(&[
        (PI_AGENT_DIR_ENV, Some(agent_dir.as_os_str())),
        (PI_EXTENSION_PATH_ENV, None),
    ])
}

fn request(force: bool, dry_run: bool) -> InstallRequest {
    InstallRequest {
        install_dir: None,
        force,
        dry_run,
        // The post-install check only prints; the tests assert on the filesystem instead.
        skip_doctor: true,
    }
}

fn removal(dry_run: bool) -> UninstallRequest {
    UninstallRequest {
        install_dir: None,
        dry_run,
        force: false,
    }
}

fn root() -> PathBuf {
    install_root().expect("PI_CODING_AGENT_DIR is set for these tests")
}

/// A copy of the extension pi would recognize, placed by hand rather than by Relay.
fn write_unmanaged_copy(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("package.json"),
        r#"{"name": "nemo-relay-pi", "pi": {"extensions": ["./index.ts"]}}"#,
    )
    .unwrap();
    std::fs::write(dir.join("index.ts"), "export default 1").unwrap();
}

#[test]
fn install_writes_every_file_and_records_what_it_wrote() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());

    assert_eq!(install(request(false, false)).unwrap(), ExitCode::SUCCESS);

    let root = root();
    for file in EXTENSION_FILES {
        let written = std::fs::read_to_string(root.join(file.path))
            .unwrap_or_else(|_| panic!("{} should have been written", file.path));
        assert_eq!(written, file.contents, "{} was written wrong", file.path);
    }
    // The point of installing: pi's own discovery finds it, with no variable set.
    assert!(is_installed());
    assert!(extension_configured());
    assert_eq!(installed_version().as_deref(), Some(EXTENSION_VERSION));
}

#[test]
fn a_dry_run_writes_nothing() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());

    assert_eq!(install(request(false, true)).unwrap(), ExitCode::SUCCESS);

    assert!(!root().exists());
    assert!(!is_installed());
}

/// Re-running over Relay's own untouched files needs no flag.
///
/// `--force` exists to guard the user's edits, and an unmodified managed install has none.
/// Demanding it here would make every upgrade a two-step for no safety gained.
#[test]
fn reinstalling_over_an_untouched_managed_install_needs_no_force() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());

    install(request(false, false)).unwrap();
    assert_eq!(install(request(false, false)).unwrap(), ExitCode::SUCCESS);
    assert!(is_installed());
}

/// The `cp -r` the guide documents lands in this same directory, and it is not Relay's.
#[test]
fn install_refuses_a_directory_relay_did_not_write_even_with_force() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    let root = root();
    write_unmanaged_copy(&root);
    let sentinel = std::fs::read_to_string(root.join("index.ts")).unwrap();

    for force in [false, true] {
        let error = install(request(force, false)).unwrap_err().to_string();
        assert!(
            error.contains("NeMo Relay did not write"),
            "force={force} gave: {error}"
        );
    }
    // Untouched, which is the whole point.
    assert_eq!(
        std::fs::read_to_string(root.join("index.ts")).unwrap(),
        sentinel
    );
}

#[test]
fn install_refuses_to_overwrite_an_edited_file_without_force() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    install(request(false, false)).unwrap();
    let edited = root().join("index.ts");
    std::fs::write(&edited, "// mine now").unwrap();

    let error = install(request(false, false)).unwrap_err().to_string();
    assert!(error.contains("index.ts"), "{error}");
    assert_eq!(std::fs::read_to_string(&edited).unwrap(), "// mine now");

    assert_eq!(install(request(true, false)).unwrap(), ExitCode::SUCCESS);
    assert_ne!(std::fs::read_to_string(&edited).unwrap(), "// mine now");
}

/// Installing beside an existing copy would break a setup that currently works.
///
/// pi de-duplicates its extension set by path rather than by package, so a second copy is a
/// second package: every hook fires twice and the launcher refuses to start at all.
#[test]
fn install_refuses_when_another_copy_would_load_beside_it() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    let other = temp.path().join("extensions").join("mine");
    write_unmanaged_copy(&other);

    let error = install(request(false, false)).unwrap_err().to_string();

    assert!(error.contains("already installed at"), "{error}");
    assert!(
        error.contains("mine"),
        "the other copy should be named: {error}"
    );
    assert!(!root().exists(), "nothing should have been written");
}

#[test]
fn uninstall_removes_what_it_wrote_and_prunes_the_directory() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    install(request(false, false)).unwrap();

    assert_eq!(uninstall(removal(false)).unwrap(), ExitCode::SUCCESS);

    assert!(!root().exists(), "the install directory should be gone");
    assert!(!is_installed());
}

/// A file the user edited is the user's, so uninstall leaves it and says so.
#[test]
fn uninstall_keeps_an_edited_file_and_the_directory_holding_it() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    install(request(false, false)).unwrap();
    let edited = root().join("src").join("user-bash.ts");
    std::fs::write(&edited, "// mine").unwrap();

    assert_eq!(uninstall(removal(false)).unwrap(), ExitCode::SUCCESS);

    assert_eq!(std::fs::read_to_string(&edited).unwrap(), "// mine");
    assert!(!root().join("index.ts").exists(), "unedited files still go");
    assert!(
        !is_installed(),
        "the state file is gone, so nothing is managed"
    );
}

#[test]
fn uninstall_refuses_a_directory_relay_did_not_write() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    let root = root();
    write_unmanaged_copy(&root);

    let error = uninstall(removal(false)).unwrap_err().to_string();

    assert!(error.contains("not Relay's to remove"), "{error}");
    assert!(root.join("index.ts").exists());
}

#[test]
fn uninstall_reports_when_there_is_nothing_installed() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());

    let error = uninstall(removal(false)).unwrap_err().to_string();

    assert!(
        error.contains("no NeMo Relay-managed pi extension install"),
        "{error}"
    );
}

#[test]
fn a_dry_run_uninstall_removes_nothing() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    install(request(false, false)).unwrap();

    assert_eq!(uninstall(removal(true)).unwrap(), ExitCode::SUCCESS);

    assert!(is_installed());
}

/// `--install-dir` addresses Relay's own marketplace root, and pi has none.
#[test]
fn install_dir_is_rejected_rather_than_ignored() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    let elsewhere = Some(temp.path().join("elsewhere"));

    let install_error = install(InstallRequest {
        install_dir: elsewhere.clone(),
        ..request(false, false)
    })
    .unwrap_err()
    .to_string();
    let uninstall_error = uninstall(UninstallRequest {
        install_dir: elsewhere,
        dry_run: false,
        force: false,
    })
    .unwrap_err()
    .to_string();

    for error in [install_error, uninstall_error] {
        assert!(error.contains("does not apply to pi"), "{error}");
    }
}

/// State a newer CLI wrote is still Relay's, and deleting it by hand is the wrong advice.
#[test]
fn a_newer_install_state_is_reported_as_upgradable_not_as_foreign() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    let root = root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join(".nemo-relay-install.json"),
        r#"{"schema": 99, "relay_version": "99.0.0", "files": []}"#,
    )
    .unwrap();

    for error in [
        install(request(true, false)).unwrap_err().to_string(),
        uninstall(removal(false)).unwrap_err().to_string(),
    ] {
        assert!(error.contains("version 99"), "{error}");
        assert!(error.contains("Upgrade nemo-relay"), "{error}");
    }
}

/// The wizard offers only when pi has nothing.
#[test]
fn setup_offers_an_install_only_when_nothing_is_there() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());

    assert!(
        setup_install_available(),
        "a clean machine should be offered one"
    );

    install(request(false, false)).unwrap();
    assert!(
        !setup_install_available(),
        "an install already written needs no offer"
    );
}

/// A copy the user placed themselves already works, so setup stays quiet about it.
#[test]
fn setup_makes_no_offer_over_an_unmanaged_copy() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    write_unmanaged_copy(&temp.path().join("extensions").join("mine"));

    assert!(!setup_install_available());
}

/// The two collectors answer different questions, and conflating them panicked `doctor`.
///
/// `installed_integrations` feeds marketplace-only readiness code that is `unreachable!()`
/// for pi, so a managed install must not appear there. `uninstallable_integrations` is what
/// `uninstall all` asks, and pi belongs in that one.
#[test]
fn a_managed_install_is_uninstallable_but_never_a_marketplace_integration() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    let marketplace = temp.path().join("plugins");
    let all = crate::agents::CodingAgent::ALL;

    install(request(false, false)).unwrap();

    // Both hold whether or not force-cleanup targets are included: `include_local_install` widens
    // the marketplace question, it does not give pi a marketplace answer.
    for include_local_install in [false, true] {
        assert!(
            !crate::agents::installed_integrations(&all, Some(&marketplace), include_local_install)
                .contains(&crate::agents::CodingAgent::Pi),
            "pi in this list reaches marketplace code that aborts on it              (include_local_install={include_local_install})"
        );
        assert!(
            crate::agents::uninstallable_integrations(
                &all,
                Some(&marketplace),
                include_local_install
            )
            .contains(&crate::agents::CodingAgent::Pi),
            "`uninstall all` has to see a managed pi install              (include_local_install={include_local_install})"
        );
    }
}

/// Rewrite the recorded file list, the way a tampered or corrupted state file would read.
fn record_paths(root: &Path, entries: &[(&str, &str)]) {
    let state = root.join(".nemo-relay-install.json");
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&state).unwrap()).unwrap();
    value["files"] = entries
        .iter()
        .map(|(path, sha)| serde_json::json!({ "path": path, "sha256": sha }))
        .collect();
    std::fs::write(&state, value.to_string()).unwrap();
}

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(std::fs::read(path).unwrap());
    format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

/// ⚠️ Uninstall must never become an arbitrary-file delete.
///
/// Every recorded path is joined to the install root and removed, and `Path::join` with an
/// absolute path replaces the root outright. The recorded hash is no guard: it is compared
/// against whatever sits at the resolved path, so pairing the traversal with the victim's
/// own digest satisfies it. Both shapes are covered because they escape by different means.
#[test]
fn uninstall_refuses_state_that_names_a_path_outside_the_install() {
    for escape in ["../../victim.txt", "/tmp/nemo-relay-pi-victim-absolute.txt"] {
        let temp = tempfile::tempdir().unwrap();
        let _scope = scoped(temp.path());
        install(request(false, false)).unwrap();

        let victim = if escape.starts_with('/') {
            let victim = std::env::temp_dir().join("nemo-relay-pi-victim-absolute.txt");
            std::fs::write(&victim, "not yours to delete").unwrap();
            victim
        } else {
            let victim = temp.path().join("victim.txt");
            std::fs::write(&victim, "not yours to delete").unwrap();
            victim
        };
        let recorded = if escape.starts_with('/') {
            victim.display().to_string()
        } else {
            escape.to_string()
        };
        record_paths(&root(), &[(recorded.as_str(), &sha256_of(&victim))]);

        let error = uninstall(removal(false)).unwrap_err().to_string();

        assert!(
            error.contains("not a path inside the install directory"),
            "{escape} was accepted: {error}"
        );
        assert!(
            victim.exists(),
            "{escape} deleted a file outside the install"
        );
        let _ = std::fs::remove_file(&victim);
    }
}

/// The same state is refused on the way in, so `--force` cannot launder it either.
#[test]
fn install_refuses_state_that_names_a_path_outside_the_install() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    install(request(false, false)).unwrap();
    record_paths(&root(), &[("../escape.ts", "sha256:0")]);

    let error = install(request(true, false)).unwrap_err().to_string();

    assert!(
        error.contains("not a path inside the install directory"),
        "{error}"
    );
}

/// An edit Relay cannot read as text is still an edit, and uninstall must keep it.
///
/// Hashing through `read_to_string` made any non-UTF-8 edit unreadable, and collapsing that
/// to "unmodified" deleted the very file the keep-edited rule exists to protect.
#[test]
fn uninstall_keeps_a_file_edited_into_something_unreadable() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    install(request(false, false)).unwrap();
    let edited = root().join("index.ts");
    std::fs::write(&edited, [0xff, 0xfe, 0x00, 0x01]).unwrap();

    uninstall(removal(false)).unwrap();

    assert!(
        edited.exists(),
        "a non-UTF-8 edit must be kept like any other"
    );
    assert_eq!(std::fs::read(&edited).unwrap(), [0xff, 0xfe, 0x00, 0x01]);
}

/// A symlinked parent escapes component validation, so removal resolves the real path.
///
/// Both halves matter and each was reachable on its own: `remove_recorded` follows the
/// symlink to delete a file, and `prune_empty_dirs` follows it to remove the directory. The
/// second was missed the first time round because every recorded path was one level deep;
/// a forged deeper path reaches it.
#[test]
fn uninstall_does_not_follow_a_symlinked_parent_out_of_the_install() {
    #[cfg(unix)]
    {
        let temp = tempfile::tempdir().unwrap();
        let _scope = scoped(temp.path());
        install(request(false, false)).unwrap();

        let outside = temp.path().join("outside");
        // Left *empty*, which is what makes the pruning half reachable: `remove_dir` fails
        // on a non-empty directory, so a victim with a file in it hides the escape.
        let victim_dir = outside.join("deep");
        std::fs::create_dir_all(&victim_dir).unwrap();

        // Replace the install's own `src` with a symlink pointing out of the tree.
        let src = root().join("src");
        std::fs::remove_dir_all(&src).unwrap();
        std::os::unix::fs::symlink(&outside, &src).unwrap();
        record_paths(&root(), &[("src/deep/x.ts", "sha256:0")]);

        let code = uninstall(removal(false)).unwrap();

        assert!(
            victim_dir.exists(),
            "pruning followed the symlink out of the install and removed {}",
            victim_dir.display()
        );
        assert_ne!(
            code,
            ExitCode::SUCCESS,
            "an uninstall that could not act on its own state must not report success"
        );
    }
}

/// A third-party extension at the install path is not Relay's to overwrite.
///
/// pi loads a package through its `package.json` when there is one and falls back to a bare
/// `index.ts` when there is not, so "the manifest does not name Relay" is not "nothing of
/// value here". Reading it that way let `--force` overwrite a manifest-less extension and
/// somebody else's package alike, against the guarantee the guide makes.
#[test]
fn force_never_overwrites_an_extension_relay_did_not_write() {
    for (label, write) in [
        (
            "someone else's package",
            &(|dir: &Path| {
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(dir.join("package.json"), r#"{"name": "someone-elses-ext"}"#)
                    .unwrap();
                std::fs::write(dir.join("index.ts"), "export default 1").unwrap();
            }) as &dyn Fn(&Path),
        ),
        (
            "a manifest-less index.ts pi still loads",
            &(|dir: &Path| {
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(dir.join("index.ts"), "export default 1").unwrap();
            }) as &dyn Fn(&Path),
        ),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let _scope = scoped(temp.path());
        write(&root());
        let before = std::fs::read_to_string(root().join("index.ts")).unwrap();

        let error = install(request(true, false)).unwrap_err().to_string();

        assert!(
            error.contains("did not write"),
            "{label} was overwritten rather than refused: {error}"
        );
        assert_eq!(
            std::fs::read_to_string(root().join("index.ts")).unwrap(),
            before,
            "{label} was modified"
        );
    }
}

/// Leftovers pi cannot load are still not Relay's to overwrite.
#[test]
fn force_never_overwrites_leftovers_either() {
    let temp = tempfile::tempdir().unwrap();
    let _scope = scoped(temp.path());
    install(request(false, false)).unwrap();
    std::fs::write(root().join("src").join("user-bash.ts"), "// mine").unwrap();
    uninstall(removal(false)).unwrap();

    let error = install(request(true, false)).unwrap_err().to_string();

    assert!(error.contains("pi would not load it"), "{error}");
    assert!(
        !error.contains("already"),
        "the refusal must not claim the leftovers work: {error}"
    );
    assert_eq!(
        std::fs::read_to_string(root().join("src").join("user-bash.ts")).unwrap(),
        "// mine"
    );
}
