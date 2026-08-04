// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Mutex, OnceLock};

use tempfile::tempdir;

use super::*;

fn cwd_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

struct CurrentDirectoryGuard(PathBuf);

impl CurrentDirectoryGuard {
    fn enter(path: &Path) -> Self {
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(path).unwrap();
        Self(original)
    }
}

impl Drop for CurrentDirectoryGuard {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.0).unwrap();
    }
}

#[test]
fn relative_existing_source_is_pinned_across_cwd_changes() {
    let _lock = cwd_lock();
    let temp = tempdir().unwrap();
    let first = temp.path().join("first");
    let second = temp.path().join("second");
    std::fs::create_dir_all(&first).unwrap();
    std::fs::create_dir_all(&second).unwrap();
    std::fs::write(first.join("plugins.toml"), "version = 1\n").unwrap();

    let pinned = {
        let _cwd = CurrentDirectoryGuard::enter(&first);
        pin_plugin_config_path(Path::new("plugins.toml")).unwrap()
    };
    let _cwd = CurrentDirectoryGuard::enter(&second);
    assert_eq!(
        sibling_lifecycle_state_path(&pinned),
        dunce::canonicalize(&first)
            .unwrap()
            .join(DYNAMIC_PLUGIN_STATE_FILENAME)
    );
}

#[cfg(windows)]
#[test]
fn missing_and_existing_windows_sources_pin_to_portable_physical_paths() {
    let temp = tempdir().unwrap();
    let config = temp.path().join("plugins.toml");

    let missing = pin_plugin_config_path(&config).unwrap();
    assert!(missing.is_absolute());
    assert!(!missing.to_string_lossy().starts_with(r"\\?\"));

    std::fs::write(&config, "version = 1\n").unwrap();
    let existing = pin_plugin_config_path(&config).unwrap();
    assert_eq!(existing, missing);
    assert!(!existing.to_string_lossy().starts_with(r"\\?\"));
    assert_eq!(
        std::fs::canonicalize(existing).unwrap(),
        std::fs::canonicalize(config).unwrap()
    );

    let verbatim_root = std::fs::canonicalize(temp.path()).unwrap();
    let reserved = pin_plugin_config_path(&verbatim_root.join("CON").join("plugins.toml")).unwrap();
    assert!(reserved.to_string_lossy().starts_with(r"\\?\"));

    let overlong =
        pin_plugin_config_path(&verbatim_root.join("a".repeat(240)).join("plugins.toml")).unwrap();
    assert!(overlong.to_string_lossy().starts_with(r"\\?\"));
}

#[test]
fn legacy_state_without_declaration_sources_is_readable() {
    let temp = tempdir().unwrap();
    let state_path = temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME);
    std::fs::write(&state_path, r#"{"schema_version":1,"records":[]}"#).unwrap();

    let state = read_lifecycle_state(&state_path).unwrap();

    assert!(state.list(true).is_empty());
    assert_eq!(state.declaration_source("missing"), None);
}

#[test]
fn save_drops_declaration_sources_for_absent_records() {
    let temp = tempdir().unwrap();
    let state_path = temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME);
    std::fs::write(
        &state_path,
        r#"{"schema_version":1,"records":[],"declaration_sources":{"stale":"/tmp/plugins.toml"}}"#,
    )
    .unwrap();
    let lock = lock_lifecycle_state(&state_path).unwrap();
    let state = read_locked_lifecycle_state(&lock).unwrap();

    save_locked_lifecycle_state(&lock, &state).unwrap();

    let persisted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&state_path).unwrap()).unwrap();
    assert!(persisted.get("declaration_sources").is_none());
}

#[cfg(unix)]
#[test]
fn symlinked_source_and_target_share_lifecycle_state() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let target_dir = temp.path().join("target");
    let alias_dir = temp.path().join("alias");
    std::fs::create_dir_all(&target_dir).unwrap();
    std::fs::create_dir_all(&alias_dir).unwrap();
    let target = target_dir.join("plugins.toml");
    std::fs::write(&target, "version = 1\n").unwrap();
    let alias = alias_dir.join("plugins.toml");
    symlink(&target, &alias).unwrap();

    let pinned_alias = pin_plugin_config_path(&alias).unwrap();
    assert_eq!(pinned_alias, target.canonicalize().unwrap());
    assert_eq!(
        sibling_lifecycle_state_path(&pinned_alias),
        target_dir
            .canonicalize()
            .unwrap()
            .join(DYNAMIC_PLUGIN_STATE_FILENAME)
    );
}

#[cfg(unix)]
#[test]
fn missing_source_and_parents_pin_through_nearest_existing_symlink() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let physical = temp.path().join("physical");
    std::fs::create_dir_all(&physical).unwrap();
    let alias = temp.path().join("alias");
    symlink(&physical, &alias).unwrap();
    let logical = alias.join("missing").join("nested").join("plugins.toml");
    let expected = physical
        .canonicalize()
        .unwrap()
        .join("missing")
        .join("nested")
        .join("plugins.toml");

    let before_creation = pin_plugin_config_path(&logical).unwrap();
    std::fs::create_dir_all(logical.parent().unwrap()).unwrap();
    std::fs::write(&logical, "version = 1\n").unwrap();
    let after_creation = pin_plugin_config_path(&logical).unwrap();

    assert_eq!(before_creation, expected);
    assert_eq!(after_creation, expected);
}

#[cfg(unix)]
#[test]
fn lifecycle_replace_syncs_the_containing_directory() {
    use std::cell::RefCell;

    let temp = tempdir().unwrap();
    let target = temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME);
    let staged = temp.path().join(".staged-state.tmp");
    std::fs::write(&target, b"old").unwrap();
    std::fs::write(&staged, b"new").unwrap();
    let synced_directory = RefCell::new(None);

    replace_lifecycle_state_with_directory_sync(&staged, &target, |directory| {
        synced_directory.replace(Some(directory.to_path_buf()));
        Ok(())
    })
    .unwrap();

    assert_eq!(std::fs::read(&target).unwrap(), b"new");
    assert!(!staged.exists());
    assert_eq!(
        synced_directory.into_inner(),
        Some(temp.path().to_path_buf())
    );
}

#[cfg(unix)]
#[test]
fn special_file_lifecycle_state_is_rejected_without_reading_it() {
    let error = read_lifecycle_registry(Path::new("/dev/zero"))
        .expect_err("a character device must not be read as lifecycle state");
    assert!(error.to_string().contains("must be a regular file"));
}
