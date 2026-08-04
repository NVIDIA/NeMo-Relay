// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Mutex, OnceLock};

use tempfile::tempdir;

use nemo_relay::plugin::dynamic::{DynamicPluginManifest, DynamicPluginRegistry};

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

fn fixture_record(id: &str) -> DynamicPluginRecord {
    DynamicPluginManifest::parse_toml(&format!(
        r#"manifest_version = 1
[plugin]
id = "{id}"
kind = "rust_dynamic"
[compat]
relay = ">=0.5,<1.0"
native_api = "v1"
[capabilities]
items = ["plugin_native"]
[defaults]
enabled = false
[load]
library = "plugin.so"
symbol = "nemo_relay_plugin_entrypoint_v1"
[source]
artifact = "plugin.so"
[integrity]
sha256 = "sha256:placeholder"
"#,
    ))
    .unwrap()
    .into_record(Some(format!("/{id}/relay-plugin.toml")))
    .unwrap()
}

#[test]
fn lifecycle_state_ownership_methods_reject_unknown_ids_and_follow_registry_replacement() {
    let mut state = DynamicPluginLifecycleState::new(DynamicPluginRegistry::default());
    let error = state
        .set_declaration_source("missing", "/tmp/plugins.toml".into())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unknown dynamic plugin 'missing'")
    );

    state.add(fixture_record("owned")).unwrap();
    state
        .set_declaration_source("owned", "/tmp/plugins.toml".into())
        .unwrap();
    assert_eq!(state.declaration_source("owned"), Some("/tmp/plugins.toml"));
    state.clear_declaration_source("owned");
    assert_eq!(state.declaration_source("owned"), None);

    state
        .set_declaration_source("owned", "/tmp/plugins.toml".into())
        .unwrap();
    state.replace_registry(DynamicPluginRegistry::default());
    assert!(state.list(true).is_empty());
    assert_eq!(state.declaration_source("owned"), None);
}

#[test]
fn lifecycle_state_defaults_schema_and_reports_invalid_documents() {
    let temp = tempdir().unwrap();
    let state_path = temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME);

    assert!(
        read_lifecycle_state(&state_path)
            .unwrap()
            .list(true)
            .is_empty()
    );

    std::fs::write(&state_path, r#"{"records":[]}"#).unwrap();
    assert!(
        read_lifecycle_state(&state_path)
            .unwrap()
            .list(true)
            .is_empty()
    );

    std::fs::write(&state_path, r#"{"schema_version":99,"records":[]}"#).unwrap();
    let error = read_lifecycle_state(&state_path).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unsupported dynamic plugin registry schema_version 99")
    );

    std::fs::write(&state_path, r#"{"schema_version":1,"records":["secret"}"#).unwrap();
    let error = read_lifecycle_state(&state_path).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("invalid dynamic plugin registry state")
    );
    assert!(!error.to_string().contains("secret"));
}

#[test]
fn locked_registry_wrappers_persist_records_and_declaration_ownership() {
    let temp = tempdir().unwrap();
    let state_path = temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME);
    let lock = lock_lifecycle_state(&state_path).unwrap();
    assert!(format!("{lock:?}").contains("LifecycleStateLock"));

    let mut state = read_locked_lifecycle_state(&lock).unwrap();
    state.add(fixture_record("persisted")).unwrap();
    state
        .set_declaration_source(
            "persisted",
            temp.path().join("plugins.toml").display().to_string(),
        )
        .unwrap();
    save_locked_lifecycle_state(&lock, &state).unwrap();

    let registry = read_locked_lifecycle_registry(&lock).unwrap();
    assert!(registry.get("persisted").is_some());
    save_locked_lifecycle_registry(&lock, &registry).unwrap();
    drop(lock);

    let persisted = read_lifecycle_state(&state_path).unwrap();
    assert!(persisted.get("persisted").is_some());
    assert!(persisted.declaration_source("persisted").is_some());
    assert!(
        read_lifecycle_registry(&state_path)
            .unwrap()
            .get("persisted")
            .is_some()
    );
}

#[test]
fn sibling_state_path_handles_a_parentless_plugin_filename() {
    assert_eq!(
        sibling_lifecycle_state_path(Path::new("")),
        PathBuf::from(DYNAMIC_PLUGIN_STATE_FILENAME)
    );
}

#[test]
fn lifecycle_lock_directory_creation_errors_are_contextualized() {
    let temp = tempdir().unwrap();
    let blocking_file = temp.path().join("blocking-file");
    std::fs::write(&blocking_file, b"file").unwrap();

    let error =
        lock_lifecycle_state(&blocking_file.join(DYNAMIC_PLUGIN_STATE_FILENAME)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("create lifecycle state directory")
    );
}

#[cfg(unix)]
#[test]
fn lifecycle_temp_creation_and_directory_sync_errors_are_contextualized() {
    let temp = tempdir().unwrap();
    let state_directory = temp.path().join("state");
    let state_path = state_directory.join(DYNAMIC_PLUGIN_STATE_FILENAME);
    let lock = lock_lifecycle_state(&state_path).unwrap();
    let state = DynamicPluginLifecycleState::default();

    // Keep the lock descriptor open while replacing its unlinked parent directory with a file.
    // This makes temporary-file creation fail deterministically, including under a privileged
    // test runner that could bypass directory permission bits.
    std::fs::remove_file(state_directory.join(DYNAMIC_PLUGIN_STATE_LOCK_FILENAME)).unwrap();
    std::fs::remove_dir(&state_directory).unwrap();
    std::fs::write(&state_directory, b"blocking file").unwrap();

    let error = save_locked_lifecycle_state(&lock, &state).unwrap_err();
    assert!(error.to_string().contains("create lifecycle state"));

    let error = sync_lifecycle_state_directory(&temp.path().join("missing-directory")).unwrap_err();
    assert!(error.to_string().contains("open lifecycle state directory"));
}

#[cfg(windows)]
#[test]
fn windows_lifecycle_replace_reports_a_missing_staged_file() {
    let temp = tempdir().unwrap();
    let error = replace_lifecycle_state(
        &temp.path().join("missing.tmp"),
        &temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME),
    )
    .unwrap_err();
    assert!(error.to_string().contains("replace lifecycle state"));
}
