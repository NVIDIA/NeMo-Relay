// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(unix)]
use std::fs::File;

use tempfile::tempdir;

use super::*;

#[test]
fn plugin_retirement_rejects_an_external_lock_outside_its_layout() {
    let dir = tempdir().unwrap();
    let marker = dir.path().join("plugin").join(GENERATION_FILE_NAME);
    let expected_lock = dir.path().join("expected.lock");
    let unrelated_lock = dir.path().join("unrelated.lock");
    write_new_generation_with_token_at(&marker, &unrelated_lock).unwrap();

    let error = match GenerationRetirement::acquire_for_plugin(&marker, &expected_lock) {
        Err(error) => error,
        Ok(_) => panic!("out-of-layout lock was accepted"),
    };

    assert!(error.contains("outside its plugin layout"), "{error}");
    assert!(unrelated_lock.exists());
}

#[test]
fn generation_markers_have_one_canonical_encoding() {
    let lock_path = PathBuf::from("generation.lock");
    let active = GenerationMarker::active("generation-a", &lock_path);
    let retired = active.retired();

    assert_eq!(active.token(), "generation-a");
    assert_eq!(retired.token(), "generation-a");
    let active_encoded = active.encoded();
    let retired_encoded = retired.encoded();
    assert_eq!(active_encoded.lines().next(), Some("generation-a"));
    assert_eq!(retired_encoded.lines().next(), Some("retired:generation-a"));
    assert_eq!(
        decode_lock_path(
            active_encoded
                .lines()
                .nth(1)
                .unwrap()
                .strip_prefix(GENERATION_LOCK_PATH_PREFIX)
                .unwrap()
        )
        .unwrap(),
        lock_path
    );
    assert!(!active.is_retired());
    assert!(retired.is_retired());
}

#[test]
fn an_explicit_missing_external_lock_is_rejected_without_creating_it() {
    let dir = tempdir().unwrap();
    let marker_path = dir.path().join(GENERATION_FILE_NAME);
    let lock_path = dir.path().join("missing-external.lock");
    let marker = GenerationMarker::active("generation-a", &lock_path);
    std::fs::write(&marker_path, marker.encoded()).unwrap();

    let error = InstallGeneration::capture(marker_path).unwrap_err();

    assert!(
        error.contains("failed to open MCP install generation lock"),
        "{error}"
    );
    assert!(!lock_path.exists());
}

#[test]
fn an_explicit_relative_external_lock_path_is_rejected() {
    let dir = tempdir().unwrap();
    let marker_path = dir.path().join(GENERATION_FILE_NAME);
    let marker = GenerationMarker::active("generation-a", "relative-generation.lock");
    std::fs::write(&marker_path, marker.encoded()).unwrap();

    let error = InstallGeneration::capture(marker_path).unwrap_err();

    assert!(error.contains("non-absolute external lock path"), "{error}");
}

#[test]
fn an_explicit_empty_external_lock_is_rejected_without_modifying_it() {
    let dir = tempdir().unwrap();
    let marker_path = dir.path().join(GENERATION_FILE_NAME);
    let lock_path = dir.path().join("empty-external.lock");
    std::fs::write(&lock_path, []).unwrap();
    let marker = GenerationMarker::active("generation-a", &lock_path);
    std::fs::write(&marker_path, marker.encoded()).unwrap();

    let error = InstallGeneration::capture(marker_path.clone()).unwrap_err();
    assert!(error.contains("generation lock"), "{error}");
    assert!(error.contains("is empty"), "{error}");
    assert_eq!(std::fs::read(&lock_path).unwrap(), b"");

    let error = GenerationRetirement::acquire(&marker_path)
        .err()
        .expect("empty external lock was accepted");
    assert!(error.contains("generation lock"), "{error}");
    assert!(error.contains("is empty"), "{error}");
    assert_eq!(std::fs::read(lock_path).unwrap(), b"");
}

#[test]
fn retirement_without_invalidation_only_releases_the_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let mut retirement = GenerationRetirement::acquire(&path).unwrap().unwrap();

    retirement.restore_after_rollback().unwrap();

    assert!(retirement.lock.is_none());
    assert!(!retirement.changed);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn active_generation_guard_fences_retirement_until_startup_finishes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    let generation = InstallGeneration::capture(path.clone()).unwrap();
    let guard = generation.guard_current().unwrap();

    let error = match GenerationRetirement::acquire_with_timeout(&path, Duration::from_millis(20)) {
        Err(error) => error,
        Ok(_) => panic!("retirement must wait for the active startup guard"),
    };
    assert!(error.contains("timed out waiting"), "{error}");

    drop(guard);
    assert!(
        GenerationRetirement::acquire_with_timeout(&path, Duration::from_secs(1))
            .unwrap()
            .is_some()
    );
}

#[test]
fn guarded_capture_fences_retirement_without_a_reacquisition_gap() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    let (_generation, guard) = InstallGeneration::capture_guarded(path.clone()).unwrap();

    let error = GenerationRetirement::acquire_with_timeout(&path, Duration::from_millis(20))
        .err()
        .expect("retirement entered between generation capture and its guard");
    assert!(error.contains("timed out waiting"), "{error}");

    drop(guard);
    assert!(
        GenerationRetirement::acquire_with_timeout(&path, Duration::from_secs(1))
            .unwrap()
            .is_some()
    );
}

#[test]
fn expected_token_rejects_a_stale_launcher_after_same_path_rotation() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    let stale_token = write_new_generation_with_token(&path).unwrap();
    let current_token = write_new_generation_with_token(&path).unwrap();

    let error = InstallGeneration::capture_guarded_expected(path.clone(), &stale_token)
        .err()
        .expect("stale launcher adopted the replacement generation");

    assert!(error.contains("has been retired"), "{error}");
    let (generation, guard) =
        InstallGeneration::capture_guarded_expected(path.clone(), &current_token).unwrap();
    assert_eq!(generation.token(), current_token);
    assert_eq!(
        InstallGeneration::capture(path).unwrap().token(),
        current_token
    );
    drop(guard);
}

#[test]
fn generation_environment_requires_and_verifies_the_complete_identity_pair() {
    struct EnvironmentRestore {
        file: Option<std::ffi::OsString>,
        token: Option<std::ffi::OsString>,
    }
    impl Drop for EnvironmentRestore {
        fn drop(&mut self) {
            // SAFETY: This test holds the repository-wide environment mutex.
            unsafe {
                match self.file.take() {
                    Some(value) => std::env::set_var(GENERATION_FILE_ENV, value),
                    None => std::env::remove_var(GENERATION_FILE_ENV),
                }
                match self.token.take() {
                    Some(value) => std::env::set_var(GENERATION_TOKEN_ENV, value),
                    None => std::env::remove_var(GENERATION_TOKEN_ENV),
                }
            }
        }
    }

    let _environment = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _restore = EnvironmentRestore {
        file: std::env::var_os(GENERATION_FILE_ENV),
        token: std::env::var_os(GENERATION_TOKEN_ENV),
    };
    // SAFETY: This test holds the repository-wide environment mutex.
    unsafe {
        std::env::remove_var(GENERATION_FILE_ENV);
        std::env::remove_var(GENERATION_TOKEN_ENV);
    }
    assert!(
        InstallGeneration::capture_guarded_from_env()
            .unwrap()
            .is_none()
    );

    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    let token = write_new_generation_with_token(&path).unwrap();
    // SAFETY: This test holds the repository-wide environment mutex.
    unsafe { std::env::set_var(GENERATION_FILE_ENV, &path) };
    let error = InstallGeneration::capture_guarded_from_env()
        .err()
        .expect("generation path without identity was accepted");
    assert!(error.contains(GENERATION_TOKEN_ENV), "{error}");

    // SAFETY: This test holds the repository-wide environment mutex.
    unsafe {
        std::env::remove_var(GENERATION_FILE_ENV);
        std::env::set_var(GENERATION_TOKEN_ENV, &token);
    }
    let error = InstallGeneration::capture_guarded_from_env()
        .err()
        .expect("generation identity without path was accepted");
    assert!(error.contains(GENERATION_FILE_ENV), "{error}");

    // SAFETY: This test holds the repository-wide environment mutex.
    unsafe { std::env::set_var(GENERATION_FILE_ENV, &path) };
    let (generation, guard) = InstallGeneration::capture_guarded_from_env()
        .unwrap()
        .unwrap();
    assert_eq!(generation.token(), token);
    drop(guard);

    // SAFETY: This test holds the repository-wide environment mutex.
    unsafe { std::env::set_var(GENERATION_TOKEN_ENV, "stale-generation") };
    let error = InstallGeneration::capture_guarded_from_env()
        .err()
        .expect("stale generation identity was accepted");
    assert!(error.contains("has been retired"), "{error}");
}

#[test]
fn guarded_capture_rejects_a_marker_and_lock_replaced_after_open() {
    let dir = tempdir().unwrap();
    let old_path = dir.path().join("old").join(GENERATION_FILE_NAME);
    let path = dir.path().join("visible").join(GENERATION_FILE_NAME);
    write_new_generation(&old_path).unwrap();
    write_new_generation(&path).unwrap();
    std::fs::write(&old_path, "same-marker\n").unwrap();
    std::fs::write(&path, "same-marker\n").unwrap();
    let old_marker = open_generation(&old_path).unwrap();
    let old_lock = open_generation_lock(&old_path).unwrap();

    // Model a force install promoting generation B after capture opened generation A's files.
    // Both markers intentionally match, so the visible lock-file identity is what must reject the
    // stale handle pair. Distinct paths model the post-promotion view without relying on replacing
    // an open file, which Windows does not permit through MoveFileEx.

    let observed = read_generation_marker(&old_marker, &old_path).unwrap();
    let error = InstallGeneration::capture_guarded_open_files_with_lock(
        path.clone(),
        old_marker,
        old_lock,
        observed,
    )
    .err()
    .expect("capture adopted a new marker through the old generation lock");

    assert!(error.contains("has been retired"), "{error}");
    assert!(error.contains(&path.display().to_string()), "{error}");
}

#[test]
fn rollback_does_not_publish_the_old_token_through_a_replacement_lock() {
    let dir = tempdir().unwrap();
    let plugin = dir.path().join("plugin");
    let backup = dir.path().join("plugin-backup");
    let path = plugin.join(GENERATION_FILE_NAME);
    let lock_path = dir.path().join("generation-transaction.lock");
    write_new_generation_with_token_at(&path, &lock_path).unwrap();
    let mut retirement = GenerationRetirement::acquire(&path).unwrap().unwrap();
    retirement.invalidate_for_replacement().unwrap();

    std::fs::rename(&plugin, &backup).unwrap();
    let replacement_token = write_staged_generation_with_token(&path, &lock_path).unwrap();
    let replacement_marker = std::fs::read(&path).unwrap();

    let error = retirement.restore_after_rollback().unwrap_err();

    assert!(error.contains("lock identity changed"), "{error}");
    assert_eq!(std::fs::read(&path).unwrap(), replacement_marker);
    drop(retirement);
    assert_eq!(
        InstallGeneration::capture(path).unwrap().token(),
        replacement_token
    );
}

#[test]
fn staged_generation_lock_remains_held_across_marketplace_promotion() {
    let dir = tempdir().unwrap();
    let staged_plugin = dir.path().join("staged").join("plugin");
    let live_plugin = dir.path().join("live").join("plugin");
    let staged_marker = staged_plugin.join(GENERATION_FILE_NAME);
    let live_marker = live_plugin.join(GENERATION_FILE_NAME);
    let lock_path = dir.path().join("replacement-generation.lock");
    write_new_generation_with_token_at(&staged_marker, &lock_path).unwrap();
    let mut retirement = GenerationRetirement::acquire(&staged_marker)
        .unwrap()
        .unwrap();

    std::fs::create_dir_all(live_plugin.parent().unwrap()).unwrap();
    std::fs::rename(&staged_plugin, &live_plugin).unwrap();
    retirement.retarget_promoted_marker(&live_marker).unwrap();

    let error = GenerationRetirement::acquire_with_timeout(&live_marker, Duration::from_millis(20))
        .err()
        .expect("promoted generation escaped its staged transaction lock");
    assert!(error.contains("timed out waiting"), "{error}");

    drop(retirement);
    assert!(
        GenerationRetirement::acquire_with_timeout(&live_marker, Duration::from_secs(1))
            .unwrap()
            .is_some()
    );
}

#[test]
fn legacy_sibling_lock_can_be_released_for_tree_move_and_reacquired_for_rollback() {
    let dir = tempdir().unwrap();
    let plugin = dir.path().join("plugin");
    let backup = dir.path().join("plugin-backup");
    let marker_path = plugin.join(GENERATION_FILE_NAME);
    write_legacy_generation(&marker_path, "generation-a").unwrap();
    let mut retirement = GenerationRetirement::acquire(&marker_path)
        .unwrap()
        .unwrap();
    retirement.invalidate_for_replacement().unwrap();
    retirement.release_legacy_lock_for_tree_mutation().unwrap();

    std::fs::rename(&plugin, &backup).unwrap();
    std::fs::rename(&backup, &plugin).unwrap();
    retirement.restore_after_rollback().unwrap();

    assert_eq!(
        InstallGeneration::capture(marker_path).unwrap().token(),
        "generation-a"
    );
}

#[test]
fn relative_legacy_marker_reencodes_an_absolute_lock_for_rollback() {
    let current_dir = std::env::current_dir().unwrap();
    let dir = tempfile::Builder::new()
        .prefix(".relay-relative-generation-")
        .tempdir_in(&current_dir)
        .unwrap();
    let relative_root = dir.path().strip_prefix(&current_dir).unwrap();
    let plugin = relative_root.join("plugin");
    let backup = relative_root.join("plugin-backup");
    let marker_path = plugin.join(GENERATION_FILE_NAME);
    assert!(!marker_path.is_absolute());
    write_legacy_generation(&marker_path, "generation-a").unwrap();
    let mut retirement = GenerationRetirement::acquire(&marker_path)
        .unwrap()
        .unwrap();

    retirement.invalidate_for_replacement().unwrap();
    let retired = read_generation_marker_path(&marker_path).unwrap();
    assert!(retired.is_retired());
    assert!(retired.lock_path().is_absolute());
    retirement.release_legacy_lock_for_tree_mutation().unwrap();
    std::fs::rename(&plugin, &backup).unwrap();
    std::fs::rename(&backup, &plugin).unwrap();
    retirement.restore_after_rollback().unwrap();

    assert_eq!(
        InstallGeneration::capture(marker_path).unwrap().token(),
        "generation-a"
    );
}

#[test]
fn generation_lock_identity_is_independent_of_path_spelling() {
    let dir = tempdir().unwrap();
    let alias_dir = dir.path().join("alias");
    std::fs::create_dir(&alias_dir).unwrap();
    let marker_path = dir.path().join(GENERATION_FILE_NAME);
    let lock_path = dir.path().join("generation-transaction.lock");
    write_new_generation_with_token_at(&marker_path, &lock_path).unwrap();
    let retirement = GenerationRetirement::acquire(&marker_path)
        .unwrap()
        .unwrap();
    let aliased_lock_path = alias_dir.join("..").join("generation-transaction.lock");

    assert_ne!(retirement.lock_path(), aliased_lock_path);
    assert!(retirement.uses_lock_path(&aliased_lock_path).unwrap());
}

#[test]
fn installer_reads_the_promoted_token_through_its_existing_transaction() {
    let dir = tempdir().unwrap();
    let plugin = dir.path().join("plugin");
    let backup = dir.path().join("plugin-backup");
    let marker_path = plugin.join(GENERATION_FILE_NAME);
    let lock_path = dir.path().join("generation-transaction.lock");
    write_new_generation_with_token_at(&marker_path, &lock_path).unwrap();
    let mut retirement = GenerationRetirement::acquire(&marker_path)
        .unwrap()
        .unwrap();
    retirement.invalidate_for_replacement().unwrap();
    std::fs::rename(&plugin, &backup).unwrap();
    let replacement = write_staged_generation_with_token(&marker_path, &lock_path).unwrap();

    assert_eq!(retirement.active_visible_token().unwrap(), replacement);
    retirement.commit_replacement();
}

#[cfg(windows)]
#[test]
fn windows_visible_lock_validation_uses_the_owning_locked_handle() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    let lock_path = generation_lock_path(&path);
    let file = open_generation_lock(&path).unwrap();
    lock_exclusive_with_timeout(&file, &path, Duration::from_secs(1)).unwrap();
    let identity = ensure_generation_lock_identity_locked(&file, &lock_path).unwrap();

    assert!(visible_generation_lock_matches(&file, &lock_path, &identity).unwrap());

    unlock_file(&file).unwrap();
}

#[cfg(windows)]
#[test]
fn windows_retirement_lock_survives_tree_rename_restore_and_removal() {
    let dir = tempdir().unwrap();
    let plugin = dir.path().join("plugin");
    let backup = dir.path().join("plugin-backup");
    let path = plugin.join(GENERATION_FILE_NAME);
    let lock_path = dir.path().join("generation-transaction.lock");
    write_new_generation_with_token_at(&path, &lock_path).unwrap();
    let original = std::fs::read(&path).unwrap();
    let mut retirement = GenerationRetirement::acquire(&path).unwrap().unwrap();
    retirement.invalidate_for_replacement().unwrap();

    std::fs::rename(&plugin, &backup).unwrap();
    std::fs::rename(&backup, &plugin).unwrap();
    retirement.restore_after_rollback().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), original);

    let mut retirement = GenerationRetirement::acquire(&path).unwrap().unwrap();
    retirement.invalidate_for_replacement().unwrap();
    retirement.commit_replacement();
    std::fs::remove_dir_all(&plugin).unwrap();
    assert!(!plugin.exists());
}

#[test]
fn guarded_capture_rejects_a_rolled_back_marker_with_the_replacement_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    let old_marker = open_generation(&path).unwrap();

    // Model capture opening marker A, then opening generation B's lock while B is promoted. The
    // installer rolls the visible tree back to A before capture obtains the shared lock. Marker
    // equality alone would adopt A while retaining B's unlinked lock.
    let replacement_path = dir.path().join("replacement-generation");
    write_new_generation(&replacement_path).unwrap();
    let replacement_lock = open_generation_lock(&replacement_path).unwrap();

    let observed = read_generation_marker(&old_marker, &path).unwrap();
    let error = InstallGeneration::capture_guarded_open_files_with_lock(
        path.clone(),
        old_marker,
        replacement_lock,
        observed,
    )
    .err()
    .expect("capture adopted the rolled-back marker through the replacement lock");

    assert!(error.contains("has been retired"), "{error}");
    assert!(error.contains(&path.display().to_string()), "{error}");
}

#[test]
fn rollback_can_restore_with_the_original_lock_still_held() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    std::fs::write(&path, "retired:generation-a\n").unwrap();
    let lock = open_generation_lock(&path).unwrap();
    lock_exclusive_with_timeout(&lock, &path, Duration::from_secs(1)).unwrap();
    let lock_id =
        ensure_generation_lock_identity_locked(&lock, &generation_lock_path(&path)).unwrap();
    let mut retirement = GenerationRetirement {
        lock: Some(lock),
        lock_id,
        path: path.clone(),
        original: GenerationMarker::active("generation-a", generation_lock_path(&path)),
        changed: true,
        committed: false,
        lock_released_for_tree_mutation: false,
    };

    retirement.restore_after_rollback().unwrap();

    assert_eq!(
        read_generation_marker_path(&path).unwrap(),
        GenerationMarker::active("generation-a", generation_lock_path(&path))
    );
    assert!(!retirement.changed);
}

#[test]
fn rollback_restores_the_visible_path_after_atomic_marker_replacement() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    let original_bytes = std::fs::read(&path).unwrap();
    let original = InstallGeneration::capture(path.clone()).unwrap();
    let mut retirement = GenerationRetirement::acquire(&path).unwrap().unwrap();
    retirement.invalidate_for_replacement().unwrap();

    atomic_write(&path, retirement.original.retired().encoded().as_bytes()).unwrap();

    retirement.restore_after_rollback().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), original_bytes);

    // A rollback must not replace the inode that old clients use for fencing. Otherwise a later
    // retirement can lock the visible generation while an old client remains guarded through an
    // unlinked inode (an ABA race).
    let guard = original.guard_current().unwrap();
    let error = GenerationRetirement::acquire_with_timeout(&path, Duration::from_millis(20))
        .err()
        .expect("retirement bypassed the pre-rollback generation guard");
    assert!(error.contains("timed out waiting"), "{error}");

    drop(guard);
    assert!(
        GenerationRetirement::acquire_with_timeout(&path, Duration::from_secs(1))
            .unwrap()
            .is_some()
    );
}

#[test]
fn invalidation_requires_a_live_exclusive_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    let mut retirement = GenerationRetirement {
        lock: None,
        lock_id: uuid::Uuid::nil().to_string(),
        path: path.clone(),
        original: GenerationMarker::active("generation-a", generation_lock_path(&path)),
        changed: false,
        committed: false,
        lock_released_for_tree_mutation: false,
    };

    let error = retirement.invalidate_for_replacement().unwrap_err();

    assert!(error.contains("is not locked"), "{error}");
    assert!(error.contains(&path.display().to_string()), "{error}");
}

#[test]
fn failed_invalidation_restores_a_marker_changed_by_partial_io() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    let mut retirement = GenerationRetirement::acquire(&path).unwrap().unwrap();

    let error = retirement
        .invalidate_with(|path, _retired| {
            std::fs::write(path, "").unwrap();
            Err("injected generation write failure after truncation".into())
        })
        .unwrap_err();

    assert!(
        error.contains("injected generation write failure"),
        "{error}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert!(!retirement.changed);
}

#[test]
fn an_already_retired_generation_keeps_its_replacement_transaction_lock() {
    let dir = tempdir().unwrap();
    let marker_path = dir.path().join(GENERATION_FILE_NAME);
    let lock_path = dir.path().join("generation-transaction.lock");
    write_new_generation_with_token_at(&marker_path, &lock_path).unwrap();
    let mut first = GenerationRetirement::acquire(&marker_path)
        .unwrap()
        .unwrap();
    first.invalidate_for_replacement().unwrap();
    first.commit_replacement();
    drop(first);

    let mut retry = GenerationRetirement::acquire(&marker_path)
        .unwrap()
        .unwrap();
    retry.invalidate_for_replacement().unwrap();
    let error = GenerationRetirement::acquire_with_timeout(&marker_path, Duration::from_millis(20))
        .err()
        .expect("already-retired retry released its transaction lock");
    assert!(error.contains("timed out waiting"), "{error}");

    let replacement = write_staged_generation_with_token(&marker_path, &lock_path).unwrap();
    assert_eq!(retry.active_visible_token().unwrap(), replacement);
}

#[test]
fn dropping_an_uncommitted_retirement_restores_the_original_marker() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    let original = std::fs::read(&path).unwrap();

    {
        let mut retirement = GenerationRetirement::acquire(&path).unwrap().unwrap();
        retirement.invalidate_for_replacement().unwrap();
    }

    assert_eq!(std::fs::read(path).unwrap(), original);
}

#[test]
fn rollback_requires_the_original_transaction_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("missing-generation");
    let mut retirement = GenerationRetirement {
        lock: None,
        lock_id: uuid::Uuid::nil().to_string(),
        path: path.clone(),
        original: GenerationMarker::active("generation-a", generation_lock_path(&path)),
        changed: true,
        committed: false,
        lock_released_for_tree_mutation: false,
    };

    let error = retirement.restore_after_rollback().unwrap_err();

    assert!(error.contains("has no transaction lock"), "{error}");
    assert!(error.contains(&path.display().to_string()), "{error}");
}

#[test]
fn marker_replacement_preserves_the_operation_in_io_errors() {
    let dir = tempdir().unwrap();
    let parent = dir.path().join("not-a-directory");
    std::fs::write(&parent, "file").unwrap();
    let path = parent.join(GENERATION_FILE_NAME);

    let error = replace_generation_marker(
        &path,
        &GenerationMarker::active("generation-a", generation_lock_path(&path)).retired(),
        "invalidate",
    )
    .unwrap_err();

    assert!(error.contains("failed to invalidate"), "{error}");
    assert!(error.contains(&path.display().to_string()), "{error}");
}

#[test]
fn malformed_retirement_requires_a_token() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    std::fs::write(&path, "retired:\n").unwrap();

    let error = match GenerationRetirement::acquire(&path) {
        Err(error) => error,
        Ok(_) => panic!("a retired marker without a token must be rejected"),
    };

    assert!(error.contains("retired marker without a token"), "{error}");
}

#[test]
fn generation_creation_reports_an_invalid_parent() {
    let dir = tempdir().unwrap();
    let parent = dir.path().join("not-a-directory");
    std::fs::write(&parent, "file").unwrap();

    let error = write_new_generation(&parent.join(GENERATION_FILE_NAME)).unwrap_err();

    assert!(error.contains("failed to create"), "{error}");
}

#[test]
fn generation_creation_reports_an_unwritable_target_shape() {
    let dir = tempdir().unwrap();

    let error = write_new_generation(dir.path()).unwrap_err();

    assert!(error.contains("failed to replace"), "{error}");
}

#[test]
fn generation_creation_provisions_a_stable_sibling_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);

    write_new_generation(&path).unwrap();

    assert!(generation_lock_path(&path).is_file());
    uuid::Uuid::parse_str(&read_generation_lock_identity_path(&path).unwrap()).unwrap();
}

#[test]
fn an_empty_generation_lock_is_initialized_before_capture() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    std::fs::write(generation_lock_path(&path), "").unwrap();

    InstallGeneration::capture(path.clone()).unwrap();

    uuid::Uuid::parse_str(&read_generation_lock_identity_path(&path).unwrap()).unwrap();
}

#[test]
fn a_malformed_generation_lock_identity_is_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    std::fs::write(generation_lock_path(&path), "not-a-uuid\n").unwrap();

    let capture_error = InstallGeneration::capture(path.clone()).unwrap_err();
    assert!(
        capture_error.contains("invalid identity"),
        "{capture_error}"
    );

    let retirement_error = GenerationRetirement::acquire(&path)
        .err()
        .expect("retirement accepted a malformed lock identity");
    assert!(
        retirement_error.contains("invalid identity"),
        "{retirement_error}"
    );
}

#[test]
fn generation_marker_parser_rejects_every_malformed_record_shape() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    let absolute_lock = dir.path().join("generation.lock");
    let encoded_lock = encode_lock_path(&absolute_lock);
    let cases = [
        (String::new(), "is empty"),
        (
            format!("{}\n", "x".repeat(MAX_GENERATION_TOKEN_BYTES + 1)),
            "token in",
        ),
        (
            format!("generation-a\nwrong-prefix:{encoded_lock}\n"),
            "invalid lock-path record",
        ),
        (
            format!("generation-a\n{GENERATION_LOCK_PATH_PREFIX}{encoded_lock}\nunexpected\n"),
            "unexpected trailing records",
        ),
        (
            format!("generation-a\n{GENERATION_LOCK_PATH_PREFIX}not-base64!\n"),
            "invalid lock path",
        ),
        (
            format!("generation-a\n{GENERATION_LOCK_PATH_PREFIX}\n"),
            "lock path is empty",
        ),
        (
            format!("{}\n", "x".repeat(MAX_GENERATION_MARKER_BYTES + 1)),
            "byte limit",
        ),
    ];

    for (contents, expected) in cases {
        std::fs::write(&path, contents).unwrap();
        let error = read_generation_marker_path(&path).unwrap_err();
        assert!(
            error.contains(expected),
            "expected {expected:?} in {error:?}"
        );
    }
    let oversized = std::fs::read_to_string(&path).unwrap();
    assert_eq!(oversized.len(), MAX_GENERATION_MARKER_BYTES + 2);
    assert!(
        read_generation_marker_path(&path)
            .unwrap_err()
            .contains("byte limit")
    );
}

#[test]
fn generation_lock_identity_reader_rejects_oversized_records() {
    let dir = tempdir().unwrap();
    let marker = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&marker).unwrap();
    let lock = generation_lock_path(&marker);
    std::fs::write(&lock, "x".repeat(MAX_GENERATION_LOCK_ID_BYTES + 1)).unwrap();

    let capture_error = InstallGeneration::capture(marker.clone()).unwrap_err();
    assert!(capture_error.contains("byte limit"), "{capture_error}");
    let retirement_error = GenerationRetirement::acquire(&marker)
        .err()
        .expect("retirement accepted an oversized lock identity");
    assert!(
        retirement_error.contains("byte limit"),
        "{retirement_error}"
    );
}

#[cfg(unix)]
#[test]
fn generation_environment_rejects_non_unicode_expected_identity() {
    use std::os::unix::ffi::OsStringExt;

    struct Restore {
        path: Option<std::ffi::OsString>,
        token: Option<std::ffi::OsString>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            // SAFETY: The test holds the process-wide environment lock.
            unsafe {
                match self.path.take() {
                    Some(value) => std::env::set_var(GENERATION_FILE_ENV, value),
                    None => std::env::remove_var(GENERATION_FILE_ENV),
                }
                match self.token.take() {
                    Some(value) => std::env::set_var(GENERATION_TOKEN_ENV, value),
                    None => std::env::remove_var(GENERATION_TOKEN_ENV),
                }
            }
        }
    }

    let _environment = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _restore = Restore {
        path: std::env::var_os(GENERATION_FILE_ENV),
        token: std::env::var_os(GENERATION_TOKEN_ENV),
    };
    unsafe {
        std::env::set_var(GENERATION_FILE_ENV, "/unused/generation");
        std::env::set_var(
            GENERATION_TOKEN_ENV,
            std::ffi::OsString::from_vec(vec![0xff]),
        );
    }

    let error = InstallGeneration::capture_guarded_from_env()
        .err()
        .expect("non-Unicode expected identity was accepted");
    assert!(error.contains("not valid Unicode"), "{error}");
}

#[test]
fn retirement_state_machine_rejects_invalid_transitions_without_losing_its_lock() {
    let dir = tempdir().unwrap();
    let legacy_marker = dir.path().join("legacy").join(GENERATION_FILE_NAME);
    write_legacy_generation(&legacy_marker, "generation-a").unwrap();
    let mut legacy = GenerationRetirement::acquire(&legacy_marker)
        .unwrap()
        .unwrap();

    let active_release_error = legacy.release_legacy_lock_for_tree_mutation().unwrap_err();
    assert!(
        active_release_error.contains("cannot release active"),
        "{active_release_error}"
    );
    assert!(
        !legacy
            .uses_lock_path(&dir.path().join("missing.lock"))
            .unwrap()
    );
    let invalid_shape = dir.path().join("not-a-directory");
    std::fs::write(&invalid_shape, b"file").unwrap();
    let inspection_error = legacy
        .uses_lock_path(&invalid_shape.join("nested").join("lock"))
        .unwrap_err();
    assert!(
        inspection_error.contains("failed to inspect"),
        "{inspection_error}"
    );

    legacy.invalidate_for_replacement().unwrap();
    assert!(
        legacy
            .active_visible_token()
            .unwrap_err()
            .contains("retired")
    );
    legacy.release_legacy_lock_for_tree_mutation().unwrap();
    legacy.release_legacy_lock_for_tree_mutation().unwrap();
    legacy.restore_after_rollback().unwrap();

    let external_marker = dir.path().join("external").join(GENERATION_FILE_NAME);
    let external_lock = dir.path().join("external-generation.lock");
    write_new_generation_with_token_at(&external_marker, &external_lock).unwrap();
    let mut external = GenerationRetirement::acquire(&external_marker)
        .unwrap()
        .unwrap();
    external.release_legacy_lock_for_tree_mutation().unwrap();
    let replacement_lock = dir.path().join("replacement.lock");
    write_new_generation_with_token_at(&external_marker, &replacement_lock).unwrap();
    assert!(
        external
            .active_visible_token()
            .unwrap_err()
            .contains("retired")
    );
    let marker_error = external
        .retarget_promoted_marker(&external_marker)
        .unwrap_err();
    assert!(marker_error.contains("marker changed"), "{marker_error}");
    external.commit_replacement();
    let committed_error = external
        .retarget_promoted_marker(&external_marker)
        .unwrap_err();
    assert!(
        committed_error.contains("cannot retarget mutated"),
        "{committed_error}"
    );
}

#[test]
fn captured_generation_rejects_visible_marker_rotation_on_every_verification_path() {
    let dir = tempdir().unwrap();
    let marker = dir.path().join(GENERATION_FILE_NAME);
    let lock = dir.path().join("generation.lock");
    write_new_generation_with_token_at(&marker, &lock).unwrap();
    let generation = InstallGeneration::capture(marker.clone()).unwrap();

    write_staged_generation_with_token(&marker, &lock).unwrap();

    assert!(
        generation
            .try_verify_current()
            .unwrap_err()
            .contains("retired")
    );
    assert!(
        generation
            .guard_current()
            .err()
            .expect("rotated marker was guarded")
            .contains("retired")
    );
}

#[cfg(unix)]
#[test]
fn promoted_generation_rejects_replaced_external_lock_inode() {
    let dir = tempdir().unwrap();
    let marker = dir.path().join(GENERATION_FILE_NAME);
    let lock = dir.path().join("generation.lock");
    write_new_generation_with_token_at(&marker, &lock).unwrap();
    let mut retirement = GenerationRetirement::acquire(&marker).unwrap().unwrap();

    std::fs::remove_file(&lock).unwrap();
    std::fs::write(&lock, format!("{}\n", uuid::Uuid::now_v7())).unwrap();

    let error = retirement.retarget_promoted_marker(&marker).unwrap_err();
    assert!(error.contains("lock identity changed"), "{error}");
}

#[test]
fn retirement_reports_marker_inspection_errors_and_repeated_invalidation_is_idempotent() {
    let dir = tempdir().unwrap();
    let invalid_parent = dir.path().join("not-a-directory");
    std::fs::write(&invalid_parent, b"file").unwrap();
    let inspection_error =
        GenerationRetirement::acquire(&invalid_parent.join("nested").join(GENERATION_FILE_NAME))
            .err()
            .expect("invalid marker parent was accepted");
    assert!(
        inspection_error.contains("failed to inspect"),
        "{inspection_error}"
    );

    let marker = dir.path().join("valid").join(GENERATION_FILE_NAME);
    write_new_generation(&marker).unwrap();
    let mut retirement = GenerationRetirement::acquire(&marker).unwrap().unwrap();
    retirement.invalidate_for_replacement().unwrap();
    let retired = std::fs::read(&marker).unwrap();
    retirement.invalidate_for_replacement().unwrap();
    assert_eq!(std::fs::read(&marker).unwrap(), retired);
    retirement.restore_after_rollback().unwrap();
}

#[test]
fn failed_invalidation_aggregates_a_failed_marker_restore() {
    let dir = tempdir().unwrap();
    let plugin = dir.path().join("plugin");
    let marker = plugin.join(GENERATION_FILE_NAME);
    let lock = dir.path().join("external.lock");
    write_new_generation_with_token_at(&marker, &lock).unwrap();
    let mut retirement = GenerationRetirement::acquire(&marker).unwrap().unwrap();

    let error = retirement
        .invalidate_with(|path, _retired| {
            crate::file_io::fail_next_atomic_write(path);
            Err("injected invalidation failure".into())
        })
        .unwrap_err();

    assert!(error.contains("injected invalidation failure"), "{error}");
    assert!(error.contains("additionally"), "{error}");
    retirement.commit_replacement();
}

#[test]
fn generation_writers_report_each_invalid_parent_lock_and_identity_shape() {
    let dir = tempdir().unwrap();
    let invalid_parent = dir.path().join("not-a-directory");
    std::fs::write(&invalid_parent, b"file").unwrap();
    let marker_under_file = invalid_parent.join(GENERATION_FILE_NAME);
    let external_lock = dir.path().join("external.lock");

    let legacy_error = write_legacy_generation(&marker_under_file, "generation-a").unwrap_err();
    assert!(legacy_error.contains("failed to create"), "{legacy_error}");
    let marker_parent_error =
        write_new_generation_with_token_at(&marker_under_file, &external_lock).unwrap_err();
    assert!(
        marker_parent_error.contains("failed to create"),
        "{marker_parent_error}"
    );
    let lock_parent_error = write_new_generation_with_token_at(
        &dir.path().join("marker"),
        &invalid_parent.join("lock"),
    )
    .unwrap_err();
    assert!(
        lock_parent_error.contains("failed to create"),
        "{lock_parent_error}"
    );
    let staged_error =
        write_staged_generation_with_token(&marker_under_file, &external_lock).unwrap_err();
    assert!(staged_error.contains("failed to create"), "{staged_error}");

    let malformed_lock = dir.path().join("malformed.lock");
    std::fs::write(&malformed_lock, b"not-a-uuid\n").unwrap();
    let malformed_error =
        write_new_generation_with_token_at(&dir.path().join("malformed-marker"), &malformed_lock)
            .unwrap_err();
    assert!(
        malformed_error.contains("invalid identity"),
        "{malformed_error}"
    );

    let directory_lock_error =
        write_new_generation_with_token_at(&dir.path().join("directory-lock-marker"), dir.path())
            .unwrap_err();
    assert!(
        directory_lock_error.contains("failed to open"),
        "{directory_lock_error}"
    );
}

#[test]
fn direct_generation_lock_identity_read_rejects_an_empty_legacy_lock() {
    let dir = tempdir().unwrap();
    let marker = dir.path().join(GENERATION_FILE_NAME);
    std::fs::write(&marker, b"generation-a\n").unwrap();
    std::fs::write(generation_lock_path(&marker), b"").unwrap();

    let error = read_generation_lock_identity_path(&marker).unwrap_err();

    assert!(error.contains("is empty"), "{error}");
}

#[cfg(windows)]
#[test]
fn windows_lock_path_decoder_rejects_odd_utf16_byte_length() {
    let encoded = base64::engine::general_purpose::STANDARD.encode([0_u8]);

    let error = decode_lock_path(&encoded).unwrap_err();

    assert!(error.contains("odd byte length"), "{error}");
}

#[test]
fn marker_rotation_preserves_the_lock_inode_identity() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(GENERATION_FILE_NAME);
    write_new_generation(&path).unwrap();
    let lock_id = read_generation_lock_identity_path(&path).unwrap();
    let mut retirement = GenerationRetirement::acquire(&path).unwrap().unwrap();
    retirement.invalidate_for_replacement().unwrap();
    atomic_write(&path, b"generation-b\n").unwrap();
    retirement.commit_replacement();
    drop(retirement);

    InstallGeneration::capture(path.clone()).unwrap();

    assert_eq!(read_generation_lock_identity_path(&path).unwrap(), lock_id);
}

#[cfg(unix)]
#[test]
fn generation_reader_reports_directory_read_errors() {
    let dir = tempdir().unwrap();
    let directory = File::open(dir.path()).unwrap();

    let error = read_generation_marker(&directory, dir.path()).unwrap_err();

    assert!(
        error.contains("failed to read MCP install generation"),
        "{error}"
    );
}
