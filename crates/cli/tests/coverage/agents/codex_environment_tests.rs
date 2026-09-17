// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn dotenv_install_preserves_user_bytes_and_reinstall_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.toml");
    let original = "# user settings\r\nOPENAI_PROJECT='project-user'\r\nOTHER=untouched";
    fs::write(path(&config), original).unwrap();
    let key = BootstrapChallengeKey::from_bytes(b"dotenv-test-key");
    install(&config, &key.client_token()).unwrap();
    let installed = fs::read_to_string(path(&config)).unwrap();
    assert!(installed.starts_with(original));
    assert!(has_proof(&config, &key));
    install(&config, &key.client_token()).unwrap();
    assert_eq!(fs::read_to_string(path(&config)).unwrap(), installed);
    // Edits made after installation survive uninstall, including other environment settings.
    fs::write(path(&config), format!("{installed}LATER=value\n")).unwrap();
    uninstall(&config).unwrap();
    assert_eq!(
        fs::read_to_string(path(&config)).unwrap(),
        format!("{original}\nLATER=value\n")
    );
}

#[test]
fn dotenv_uninstall_distinguishes_absent_and_empty_original_files() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.toml");
    let key = BootstrapChallengeKey::from_bytes(b"dotenv-test-key");
    install(&config, &key.client_token()).unwrap();
    uninstall(&config).unwrap();
    assert!(!path(&config).exists());
    fs::write(path(&config), "").unwrap();
    install(&config, &key.client_token()).unwrap();
    uninstall(&config).unwrap();
    assert_eq!(fs::read_to_string(path(&config)).unwrap(), "");
}

#[test]
fn dotenv_rejects_invalid_syntax_without_exposing_or_changing_it() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.toml");
    let original = "SECRET='unclosed-sensitive-value";
    fs::write(path(&config), original).unwrap();
    let key = BootstrapChallengeKey::from_bytes(b"dotenv-test-key");
    let error = install(&config, &key.client_token()).unwrap_err();
    assert!(!error.contains("sensitive-value"));
    assert_eq!(fs::read_to_string(path(&config)).unwrap(), original);
}

#[test]
fn dotenv_modified_managed_block_is_not_overwritten_or_removed() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.toml");
    let key = BootstrapChallengeKey::from_bytes(b"dotenv-test-key");
    install(&config, &key.client_token()).unwrap();
    let modified = fs::read_to_string(path(&config))
        .unwrap()
        .replace("${OPENAI_PROJECT}", "edited");
    fs::write(path(&config), &modified).unwrap();
    assert!(install(&config, &key.client_token()).is_err());
    assert!(uninstall(&config).is_err());
    assert_eq!(fs::read_to_string(path(&config)).unwrap(), modified);
}

#[cfg(unix)]
#[test]
fn dotenv_install_preserves_symlink_and_protects_credential_permissions() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.toml");
    let target = dir.path().join("actual-env");
    fs::write(&target, "USER_SETTING=yes\n").unwrap();
    symlink(&target, path(&config)).unwrap();
    let key = BootstrapChallengeKey::from_bytes(b"dotenv-test-key");
    install(&config, &key.client_token()).unwrap();
    assert!(
        fs::symlink_metadata(path(&config))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        fs::metadata(&target).unwrap().permissions().mode() & 0o777,
        0o600
    );
    uninstall(&config).unwrap();
    assert_eq!(fs::read_to_string(&target).unwrap(), "USER_SETTING=yes\n");
    assert!(
        fs::symlink_metadata(path(&config))
            .unwrap()
            .file_type()
            .is_symlink()
    );
}
