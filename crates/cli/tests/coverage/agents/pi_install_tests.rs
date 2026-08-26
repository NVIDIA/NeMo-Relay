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
            error.contains("was not written by NeMo Relay"),
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
