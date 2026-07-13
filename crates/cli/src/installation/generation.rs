// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private install-generation fencing for lifecycle-bound plugin MCP clients.

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine;

use crate::filesystem::{
    LockAttempt, atomic_write, try_lock_exclusive, try_lock_shared, unlock_file,
};

pub(crate) const GENERATION_FILE_ENV: &str = "NEMO_RELAY_MCP_GENERATION_FILE";
pub(crate) const GENERATION_TOKEN_ENV: &str = "NEMO_RELAY_MCP_GENERATION";
pub(crate) const GENERATION_FILE_NAME: &str = ".nemo-relay-generation";
const MAX_GENERATION_TOKEN_BYTES: usize = 128;
const MAX_GENERATION_MARKER_BYTES: usize = 16 * 1024;
const MAX_GENERATION_LOCK_ID_BYTES: usize = 64;
const RETIRED_GENERATION_PREFIX: &str = "retired:";
const GENERATION_LOCK_PATH_PREFIX: &str = "lock-path:";
const GENERATION_LOCK_SUFFIX: &str = ".lock";
const DEFAULT_GENERATION_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const GENERATION_LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Clone, Debug)]
pub(crate) struct InstallGeneration {
    path: PathBuf,
    marker: GenerationMarker,
    lock_id: String,
    // The lock lives outside movable plugin trees while the marker remains plugin-owned.
    // Retaining this handle preserves fencing across replacement and rollback without preventing
    // Windows from moving the marketplace that contains the marker.
    file: Arc<File>,
}

/// Shared generation lock held across one gateway adoption or startup.
///
/// Retirement takes the exclusive side of the same file lock, so an installer cannot invalidate
/// and stop an endpoint until every startup that observed the old marker has either published a
/// ready gateway or failed.
pub(crate) struct ActiveGenerationGuard {
    lock: Arc<File>,
}

impl Drop for ActiveGenerationGuard {
    fn drop(&mut self) {
        let _ = unlock_file(&self.lock);
    }
}

impl InstallGeneration {
    pub(crate) fn capture_guarded_from_env() -> Result<Option<(Self, ActiveGenerationGuard)>, String>
    {
        match (
            env::var_os(GENERATION_FILE_ENV),
            env::var_os(GENERATION_TOKEN_ENV),
        ) {
            (None, None) => Ok(None),
            (Some(path), Some(expected)) => {
                let expected = expected.into_string().map_err(|_| {
                    format!("{GENERATION_TOKEN_ENV} is not valid Unicode; reinstall the plugin")
                })?;
                Self::capture_guarded_expected(PathBuf::from(path), &expected).map(Some)
            }
            (Some(_), None) => Err(format!(
                "{GENERATION_TOKEN_ENV} is required with {GENERATION_FILE_ENV}; reinstall the plugin"
            )),
            (None, Some(_)) => Err(format!(
                "{GENERATION_FILE_ENV} is required with {GENERATION_TOKEN_ENV}; reinstall the plugin"
            )),
        }
    }

    pub(crate) fn capture(path: PathBuf) -> Result<Self, String> {
        let (generation, guard) = Self::capture_guarded(path)?;
        drop(guard);
        Ok(generation)
    }

    pub(crate) fn capture_guarded(path: PathBuf) -> Result<(Self, ActiveGenerationGuard), String> {
        // Open the marker first. A force install can replace the entire plugin tree while this
        // process waits on the external lock named by that marker. The retained marker handle
        // lets us reject an old-marker/new-tree pairing after the wait.
        let marker = open_generation(&path)?;
        Self::capture_guarded_open_files(path, marker)
    }

    pub(crate) fn capture_guarded_expected(
        path: PathBuf,
        expected: &str,
    ) -> Result<(Self, ActiveGenerationGuard), String> {
        let (generation, guard) = Self::capture_guarded(path)?;
        if generation.token() != expected {
            drop(guard);
            return Err(retired_generation_error(&generation.path));
        }
        Ok((generation, guard))
    }

    pub(crate) fn token(&self) -> &str {
        self.marker.token()
    }

    fn capture_guarded_open_files(
        path: PathBuf,
        marker: File,
    ) -> Result<(Self, ActiveGenerationGuard), String> {
        let observed = read_generation_marker(&marker, &path)?;
        let lock_path = observed.lock_path().to_owned();
        let file = open_marker_generation_lock(&path, &lock_path)?;
        Self::capture_guarded_open_files_with_lock(path, marker, file, observed)
    }

    fn capture_guarded_open_files_with_lock(
        path: PathBuf,
        marker: File,
        file: File,
        observed: GenerationMarker,
    ) -> Result<(Self, ActiveGenerationGuard), String> {
        let lock_path = observed.lock_path().to_owned();
        let file = Arc::new(file);
        let lock_id =
            lock_shared_with_identity(&file, &path, &lock_path, DEFAULT_GENERATION_LOCK_TIMEOUT)?;
        let locked_marker = read_generation_marker(&marker, &path)?;
        let visible_marker = read_generation_marker_path(&path)?;
        let visible_lock_matches = visible_generation_lock_matches(&file, &lock_path, &lock_id)?;
        if observed != locked_marker
            || locked_marker != visible_marker
            || !visible_lock_matches
            || locked_marker.is_retired()
        {
            let _ = unlock_file(&file);
            return Err(retired_generation_error(&path));
        }
        Ok((
            Self {
                path,
                marker: visible_marker,
                lock_id,
                file: file.clone(),
            },
            ActiveGenerationGuard { lock: file },
        ))
    }

    #[cfg(test)]
    pub(crate) fn verify_current(&self) -> Result<(), String> {
        loop {
            if self.try_verify_current()? {
                return Ok(());
            }
            thread::sleep(GENERATION_LOCK_RETRY_INTERVAL);
        }
    }

    /// Check one lifecycle snapshot without waiting when an installer owns the transaction lock.
    pub(crate) fn try_verify_current(&self) -> Result<bool, String> {
        match try_lock_shared(&self.file) {
            Ok(LockAttempt::Contended) => return Ok(false),
            Ok(LockAttempt::Acquired) => {}
            Err(_) => return Err(retired_generation_error(&self.path)),
        }
        let result = self.try_validate_locked();
        let _ = unlock_file(&self.file);
        result
            .map(|()| true)
            .map_err(|_| retired_generation_error(&self.path))
    }

    pub(crate) fn guard_current(&self) -> Result<ActiveGenerationGuard, String> {
        lock_shared_with_timeout(&self.file, &self.path, DEFAULT_GENERATION_LOCK_TIMEOUT)
            .map_err(|_| retired_generation_error(&self.path))?;
        if self.try_validate_locked().is_err() {
            let _ = unlock_file(&self.file);
            return Err(retired_generation_error(&self.path));
        }
        Ok(ActiveGenerationGuard {
            lock: self.file.clone(),
        })
    }

    fn try_validate_locked(&self) -> Result<(), String> {
        let current_marker = read_generation_marker_path(&self.path)?;
        let current_lock_matches =
            visible_generation_lock_matches(&self.file, self.marker.lock_path(), &self.lock_id)?;
        if current_marker == self.marker && current_lock_matches && !current_marker.is_retired() {
            Ok(())
        } else {
            Err(retired_generation_error(&self.path))
        }
    }
}

pub(crate) struct GenerationRetirement {
    lock: Option<File>,
    lock_id: String,
    path: PathBuf,
    original: GenerationMarker,
    changed: bool,
    committed: bool,
    lock_released_for_tree_mutation: bool,
}

pub(crate) struct VisibleGenerationMarker(GenerationMarker);

impl Drop for GenerationRetirement {
    fn drop(&mut self) {
        if self.changed && !self.committed {
            let _ = self.restore_after_rollback();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum GenerationMarker {
    Active { token: String, lock_path: PathBuf },
    Retired { token: String, lock_path: PathBuf },
}

impl GenerationMarker {
    fn active(token: impl Into<String>, lock_path: impl Into<PathBuf>) -> Self {
        Self::Active {
            token: token.into(),
            lock_path: lock_path.into(),
        }
    }

    fn token(&self) -> &str {
        match self {
            Self::Active { token, .. } | Self::Retired { token, .. } => token,
        }
    }

    fn lock_path(&self) -> &Path {
        match self {
            Self::Active { lock_path, .. } | Self::Retired { lock_path, .. } => lock_path,
        }
    }

    fn retired(&self) -> Self {
        Self::Retired {
            token: self.token().to_owned(),
            lock_path: self.lock_path().to_owned(),
        }
    }

    fn encoded(&self) -> String {
        let token = match self {
            Self::Active { token, .. } => token.to_owned(),
            Self::Retired { token, .. } => format!("{RETIRED_GENERATION_PREFIX}{token}"),
        };
        format!(
            "{token}\n{GENERATION_LOCK_PATH_PREFIX}{}\n",
            encode_lock_path(self.lock_path())
        )
    }

    fn is_retired(&self) -> bool {
        matches!(self, Self::Retired { .. })
    }
}

impl GenerationRetirement {
    pub(crate) fn lock_path(&self) -> &Path {
        self.original.lock_path()
    }

    pub(crate) fn marker_path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn uses_lock_path(&self, path: &Path) -> Result<bool, String> {
        let path = absolute_lock_path(path)?;
        if !inspected_path_exists(&path, "MCP install generation lock")? {
            return Ok(false);
        }
        let lock = self.lock.as_ref().ok_or_else(|| {
            format!(
                "MCP install generation {} has no transaction lock",
                self.path.display()
            )
        })?;
        visible_generation_lock_matches(lock, &path, &self.lock_id)
    }

    /// Follow an unchanged staged marker after its marketplace is promoted to the live path.
    ///
    /// The external lock remains held throughout the rename, so no MCP can adopt the replacement
    /// between staging and host registration.
    pub(crate) fn retarget_promoted_marker(&mut self, path: &Path) -> Result<(), String> {
        if self.changed || self.committed {
            return Err(format!(
                "cannot retarget mutated MCP install generation {}",
                self.path.display()
            ));
        }
        if !self.visible_lock_identity_matches()? {
            return Err(format!(
                "failed to adopt promoted MCP install generation {} because its lock identity changed",
                path.display()
            ));
        }
        let visible = read_generation_marker_path(path)?;
        if visible != self.original {
            return Err(format!(
                "failed to adopt promoted MCP install generation {} because its marker changed",
                path.display()
            ));
        }
        self.path = path.to_owned();
        Ok(())
    }

    /// Read the active marker protected by this already-held transaction lock.
    pub(crate) fn active_visible_token(&self) -> Result<String, String> {
        let visible = read_generation_marker_path(&self.path)?;
        if visible.is_retired() || !self.uses_lock_path(visible.lock_path())? {
            return Err(retired_generation_error(&self.path));
        }
        Ok(visible.token().to_owned())
    }

    pub(crate) fn visible_marker_uses_transaction_lock(&self) -> Result<bool, String> {
        let visible = read_generation_marker_path(&self.path)?;
        self.uses_lock_path(visible.lock_path())
    }

    /// Release a legacy sibling lock before Windows moves or removes its containing plugin tree.
    ///
    /// The marker must already be retired. A rollback reacquires this exact lock identity after
    /// the tree is restored and before the old active marker is republished.
    pub(crate) fn release_legacy_lock_for_tree_mutation(&mut self) -> Result<(), String> {
        if self.lock_released_for_tree_mutation {
            return Ok(());
        }
        if !self.uses_lock_path(&generation_lock_path(&self.path))? {
            return Ok(());
        }
        if !self.original.is_retired() && !self.changed {
            return Err(format!(
                "cannot release active MCP install generation lock {}",
                self.original.lock_path().display()
            ));
        }
        let Some(file) = self.lock.take() else {
            return Err(format!(
                "MCP install generation {} is not locked",
                self.path.display()
            ));
        };
        if let Err(error) = unlock_file(&file) {
            self.lock = Some(file);
            return Err(format!(
                "failed to release MCP install generation lock {} before moving its plugin tree: {error}",
                self.original.lock_path().display()
            ));
        }
        self.lock_released_for_tree_mutation = true;
        Ok(())
    }

    pub(crate) fn acquire(path: &Path) -> Result<Option<Self>, String> {
        Self::acquire_with_timeout(path, DEFAULT_GENERATION_LOCK_TIMEOUT)
    }

    pub(crate) fn acquire_for_plugin(
        path: &Path,
        external_lock: &Path,
    ) -> Result<Option<Self>, String> {
        Self::acquire_impl(path, DEFAULT_GENERATION_LOCK_TIMEOUT, Some(external_lock))
    }

    pub(crate) fn acquire_with_timeout(
        path: &Path,
        timeout: Duration,
    ) -> Result<Option<Self>, String> {
        Self::acquire_impl(path, timeout, None)
    }

    fn acquire_impl(
        path: &Path,
        timeout: Duration,
        allowed_external_lock: Option<&Path>,
    ) -> Result<Option<Self>, String> {
        if !inspected_path_exists(path, "MCP install generation")? {
            return Ok(None);
        }
        // Open the marker before its external lock so validation can detect a plugin-tree swap
        // between the two opens. The immutable lock UUID additionally detects lock replacement.
        let marker = open_generation(path)?;
        let observed = read_generation_marker(&marker, path)?;
        let lock_path = observed.lock_path().to_owned();
        if let Some(allowed) = allowed_external_lock
            && !is_legacy_sibling_lock(path, &lock_path)?
            && absolute_lock_path(allowed)? != absolute_lock_path(&lock_path)?
        {
            return Err(format!(
                "MCP install generation {} references an external lock outside its plugin layout",
                path.display()
            ));
        }
        let file = open_marker_generation_lock(path, &lock_path)?;
        lock_exclusive_with_timeout(&file, path, timeout)?;
        let lock_id = if is_legacy_sibling_lock(path, &lock_path)? {
            ensure_generation_lock_identity_locked(&file, &lock_path)?
        } else {
            read_generation_lock_identity(&file, &lock_path)?
                .ok_or_else(|| empty_generation_lock_error(&lock_path))?
        };
        let original = read_generation_marker(&marker, path)?;
        let visible = read_generation_marker_path(path)?;
        let visible_lock_matches = visible_generation_lock_matches(&file, &lock_path, &lock_id)?;
        if observed != original || original != visible || !visible_lock_matches {
            let _ = unlock_file(&file);
            return Err(retired_generation_error(path));
        }
        Ok(Some(Self {
            lock: Some(file),
            lock_id,
            path: path.to_owned(),
            original,
            changed: false,
            committed: false,
            lock_released_for_tree_mutation: false,
        }))
    }

    /// Persistently invalidate this generation while retaining its exclusive transaction lock.
    ///
    /// Existing MCPs retain the stable generation lock while the marker can be atomically
    /// replaced or moved. Call [`Self::commit_replacement`] to make retirement permanent;
    /// otherwise dropping the transaction restores the token before releasing the lock.
    pub(crate) fn invalidate_for_replacement(&mut self) -> Result<(), String> {
        self.invalidate_with(|path, retired| replace_generation_marker(path, retired, "invalidate"))
    }

    fn invalidate_with(
        &mut self,
        write_retired: impl FnOnce(&Path, &GenerationMarker) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.original.is_retired() {
            return Ok(());
        }
        if self.changed {
            return Ok(());
        }
        let retired = self.original.retired();
        self.lock.as_ref().ok_or_else(|| {
            format!(
                "MCP install generation {} is not locked",
                self.path.display()
            )
        })?;
        self.changed = true;
        if let Err(error) = write_retired(&self.path, &retired) {
            let restore_error = replace_generation_marker(
                &self.path,
                &self.original,
                "restore after failed invalidation",
            )
            .err();
            if restore_error.is_none() {
                self.changed = false;
            }
            return match restore_error {
                Some(restore_error) => Err(format!("{error}; additionally {restore_error}")),
                None => Err(error),
            };
        }
        Ok(())
    }

    /// Commit the retired marker while retaining the transaction lock until this value is dropped.
    pub(crate) fn commit_replacement(&mut self) {
        self.committed = true;
    }

    /// Retire a promoted marker while this transaction still owns its external lock.
    ///
    /// Force-install rollback uses this after a staged tree is promoted. The old and replacement
    /// markers intentionally share one external transaction lock, so reacquiring a second
    /// retirement would deadlock. Returning the active marker lets a failed gateway refresh
    /// restore the promoted generation without exposing it during destructive rollback.
    pub(crate) fn retire_visible_replacement(&mut self) -> Result<VisibleGenerationMarker, String> {
        let visible = read_generation_marker_path(&self.path)?;
        if visible.is_retired() {
            return Err(format!(
                "replacement MCP install generation {} is already retired",
                self.path.display()
            ));
        }
        if !self.uses_lock_path(visible.lock_path())? {
            return Err(format!(
                "failed to retire replacement MCP install generation {} because its lock identity changed",
                self.path.display()
            ));
        }
        let retired = visible.retired();
        replace_generation_marker(&self.path, &retired, "retire replacement")?;
        self.verify_visible_state_for_rollback(&retired)?;
        Ok(VisibleGenerationMarker(visible))
    }

    pub(crate) fn restore_visible_replacement(
        &mut self,
        visible: VisibleGenerationMarker,
    ) -> Result<(), String> {
        self.verify_visible_state_for_rollback(&visible.0.retired())?;
        replace_generation_marker(&self.path, &visible.0, "restore replacement")?;
        self.verify_visible_state_for_rollback(&visible.0)
    }

    /// Restore an invalidated marker before a rolled-back plugin is registered again.
    pub(crate) fn restore_after_rollback(&mut self) -> Result<(), String> {
        if !self.changed {
            self.lock = None;
            self.lock_released_for_tree_mutation = false;
            return Ok(());
        }
        self.reacquire_transaction_lock()?;
        // Never publish the retired generation's token through a replacement tree. A failed
        // filesystem rollback can leave the promoted tree visible at the same path while this
        // transaction still owns the shared external lock.
        self.verify_visible_state_for_rollback(&self.original.retired())?;
        replace_generation_marker(&self.path, &self.original, "restore")?;
        // Retain the post-write check so an unexpected path swap during restoration is still
        // reported before the old generation is considered active again.
        self.verify_visible_state_for_rollback(&self.original)?;
        self.changed = false;
        self.committed = false;
        self.lock = None;
        self.lock_released_for_tree_mutation = false;
        Ok(())
    }

    fn verify_visible_state_for_rollback(
        &self,
        expected_marker: &GenerationMarker,
    ) -> Result<(), String> {
        if !self.visible_lock_identity_matches()? {
            return Err(format!(
                "failed to restore MCP install generation {} because its marker or lock identity changed",
                self.path.display()
            ));
        }
        let visible_marker = read_generation_marker_path(&self.path)?;
        if &visible_marker != expected_marker {
            return Err(format!(
                "failed to restore MCP install generation {} because its marker or lock identity changed",
                self.path.display()
            ));
        }
        Ok(())
    }

    fn visible_lock_identity_matches(&self) -> Result<bool, String> {
        let lock = self.lock.as_ref().ok_or_else(|| {
            format!(
                "MCP install generation {} has no transaction lock for rollback",
                self.path.display()
            )
        })?;
        visible_generation_lock_matches(lock, self.original.lock_path(), &self.lock_id)
    }

    fn reacquire_transaction_lock(&mut self) -> Result<(), String> {
        if self.lock.is_some() {
            return Ok(());
        }
        if !self.lock_released_for_tree_mutation {
            return Err(format!(
                "MCP install generation {} has no transaction lock for rollback",
                self.path.display()
            ));
        }
        let lock_path = self.original.lock_path();
        let file = open_existing_generation_lock_path(lock_path)?;
        lock_exclusive_with_timeout(&file, &self.path, DEFAULT_GENERATION_LOCK_TIMEOUT)?;
        let visible_identity = read_generation_lock_identity(&file, lock_path)?
            .ok_or_else(|| empty_generation_lock_error(lock_path))?;
        if visible_identity != self.lock_id {
            let _ = unlock_file(&file);
            return Err(format!(
                "failed to reacquire MCP install generation lock {} because its identity changed",
                lock_path.display()
            ));
        }
        self.lock = Some(file);
        self.lock_released_for_tree_mutation = false;
        Ok(())
    }
}

/// Distinguishes an absent path from Windows reporting `NotFound` when an intermediate component
/// is a file. Treating the latter as absent would silently skip retirement or lock validation.
fn inspected_path_exists(path: &Path, description: &str) -> Result<bool, String> {
    match fs::metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut ancestor = path.parent();
            while let Some(parent) = ancestor {
                match fs::metadata(parent) {
                    Ok(metadata) if metadata.is_dir() => return Ok(false),
                    Ok(_) => {
                        return Err(format!(
                            "failed to inspect {description} {}: parent {} is not a directory",
                            path.display(),
                            parent.display()
                        ));
                    }
                    Err(parent_error) if parent_error.kind() == std::io::ErrorKind::NotFound => {
                        ancestor = parent.parent();
                    }
                    Err(parent_error) => {
                        return Err(format!(
                            "failed to inspect {description} {}: failed to inspect parent {}: {parent_error}",
                            path.display(),
                            parent.display()
                        ));
                    }
                }
            }
            Ok(false)
        }
        Err(error) => Err(format!(
            "failed to inspect {description} {}: {error}",
            path.display()
        )),
    }
}

fn replace_generation_marker(
    path: &Path,
    marker: &GenerationMarker,
    operation: &str,
) -> Result<(), String> {
    atomic_write(path, marker.encoded().as_bytes()).map_err(|error| {
        format!(
            "failed to {operation} MCP install generation {}: {error}",
            path.display()
        )
    })
}

fn lock_shared_with_timeout(file: &File, path: &Path, timeout: Duration) -> Result<(), String> {
    lock_with_timeout(file, path, timeout, false)
}

/// Acquire a shared generation lock whose immutable inode identity is initialized and readable.
///
/// A zero-length legacy sibling lock can remain after an interrupted first install. Initialization
/// briefly upgrades through an exclusive lock, then reacquires the shared lock and lets the
/// caller's marker/visible-path validation close the upgrade gap. Explicit external locks are
/// never initialized through a marker because their paths are marker-controlled.
fn lock_shared_with_identity(
    file: &File,
    path: &Path,
    lock_path: &Path,
    timeout: Duration,
) -> Result<String, String> {
    lock_shared_with_timeout(file, path, timeout)?;
    match read_generation_lock_identity(file, lock_path) {
        Ok(Some(identity)) => return Ok(identity),
        Ok(None) if !is_legacy_sibling_lock(path, lock_path)? => {
            let _ = unlock_file(file);
            return Err(empty_generation_lock_error(lock_path));
        }
        Ok(None) => {
            let _ = unlock_file(file);
        }
        Err(error) => {
            let _ = unlock_file(file);
            return Err(error);
        }
    }

    lock_exclusive_with_timeout(file, path, timeout)?;
    let initialized = ensure_generation_lock_identity_locked(file, lock_path);
    let _ = unlock_file(file);
    initialized?;

    lock_shared_with_timeout(file, path, timeout)?;
    read_generation_lock_identity(file, lock_path)?.ok_or_else(|| {
        format!(
            "MCP install generation lock {} remained empty after initialization",
            lock_path.display()
        )
    })
}

fn empty_generation_lock_error(lock_path: &Path) -> String {
    format!(
        "MCP install generation lock {} is empty",
        lock_path.display()
    )
}

fn lock_exclusive_with_timeout(file: &File, path: &Path, timeout: Duration) -> Result<(), String> {
    lock_with_timeout(file, path, timeout, true)
}

fn lock_with_timeout(
    file: &File,
    path: &Path,
    timeout: Duration,
    exclusive: bool,
) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    loop {
        let result = if exclusive {
            try_lock_exclusive(file)
        } else {
            try_lock_shared(file)
        };
        match result {
            Ok(LockAttempt::Acquired) => return Ok(()),
            Ok(LockAttempt::Contended) => {
                if Instant::now() >= deadline {
                    return Err(format!(
                        "timed out waiting for MCP install generation lock {}",
                        path.display()
                    ));
                }
                thread::sleep(GENERATION_LOCK_RETRY_INTERVAL.min(timeout));
            }
            Err(error) => {
                return Err(format!(
                    "failed to lock MCP install generation {}: {error}",
                    path.display()
                ));
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn write_new_generation(path: &Path) -> Result<(), String> {
    write_new_generation_with_token(path).map(|_| ())
}

#[cfg(test)]
pub(crate) fn write_new_generation_with_token(path: &Path) -> Result<String, String> {
    let lock_path = generation_lock_path(path);
    write_new_generation_with_token_at(path, &lock_path)
}

#[cfg(test)]
pub(crate) fn write_legacy_generation(path: &Path, token: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let lock_path = generation_lock_path(path);
    let file = open_generation_lock_path(&lock_path)?;
    lock_exclusive_with_timeout(&file, path, DEFAULT_GENERATION_LOCK_TIMEOUT)?;
    if let Err(error) = ensure_generation_lock_identity_locked(&file, &lock_path) {
        let _ = unlock_file(&file);
        return Err(error);
    }
    let result = atomic_write(path, format!("{token}\n").as_bytes());
    let _ = unlock_file(&file);
    result
}

pub(crate) fn write_new_generation_with_token_at(
    path: &Path,
    lock_path: &Path,
) -> Result<String, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let file = open_generation_lock_path(lock_path)?;
    lock_exclusive_with_timeout(&file, path, DEFAULT_GENERATION_LOCK_TIMEOUT)?;
    if let Err(error) = ensure_generation_lock_identity_locked(&file, lock_path) {
        let _ = unlock_file(&file);
        return Err(error);
    }
    let token = uuid::Uuid::now_v7().to_string();
    let marker = GenerationMarker::active(&token, absolute_lock_path(lock_path)?);
    let result = atomic_write(path, marker.encoded().as_bytes());
    let _ = unlock_file(&file);
    result.map(|()| token)
}

pub(crate) fn write_staged_generation_with_token(
    path: &Path,
    active_lock_path: &Path,
) -> Result<String, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let token = uuid::Uuid::now_v7().to_string();
    let marker = GenerationMarker::active(&token, absolute_lock_path(active_lock_path)?);
    atomic_write(path, marker.encoded().as_bytes()).map(|()| token)
}

fn absolute_lock_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        return Ok(path.to_owned());
    }
    env::current_dir()
        .map(|current| current.join(path))
        .map_err(|error| {
            format!(
                "failed to resolve generation lock {}: {error}",
                path.display()
            )
        })
}

fn generation_lock_path(path: &Path) -> PathBuf {
    let mut lock = path.as_os_str().to_os_string();
    lock.push(GENERATION_LOCK_SUFFIX);
    PathBuf::from(lock)
}

fn is_legacy_sibling_lock(marker_path: &Path, lock_path: &Path) -> Result<bool, String> {
    Ok(absolute_lock_path(&generation_lock_path(marker_path))? == absolute_lock_path(lock_path)?)
}

#[cfg(test)]
fn open_generation_lock(path: &Path) -> Result<File, String> {
    let lock_path = generation_lock_path(path);
    open_generation_lock_path(&lock_path)
}

fn open_generation_lock_path(lock_path: &Path) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(lock_path).map_err(|error| {
        format!(
            "failed to open MCP install generation lock {}: {error}",
            lock_path.display()
        )
    })
}

fn open_marker_generation_lock(marker_path: &Path, lock_path: &Path) -> Result<File, String> {
    // Legacy one-line markers derive a sibling lock. Creating only that deterministic path keeps
    // old installs and Hermes upgrades compatible without allowing a marker to create an
    // arbitrary external file. New plugin markers always point at a pre-initialized state lock.
    if is_legacy_sibling_lock(marker_path, lock_path)? {
        open_generation_lock_path(lock_path)
    } else {
        open_existing_generation_lock_path(lock_path)
    }
}

fn open_existing_generation_lock_path(lock_path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .read(true)
        .open(lock_path)
        .map_err(|error| {
            format!(
                "failed to open MCP install generation lock {}: {error}",
                lock_path.display()
            )
        })
}

fn ensure_generation_lock_identity_locked(file: &File, lock_path: &Path) -> Result<String, String> {
    if let Some(identity) = read_generation_lock_identity(file, lock_path)? {
        return Ok(identity);
    }
    let identity = uuid::Uuid::now_v7().to_string();
    write_generation_lock_identity(file, lock_path, &identity)?;
    Ok(identity)
}

fn write_generation_lock_identity(
    file: &File,
    lock_path: &Path,
    identity: &str,
) -> Result<(), String> {
    let mut writer = file;
    writer.seek(SeekFrom::Start(0)).map_err(|error| {
        format!(
            "failed to seek MCP install generation lock {}: {error}",
            lock_path.display()
        )
    })?;
    file.set_len(0).map_err(|error| {
        format!(
            "failed to truncate MCP install generation lock {}: {error}",
            lock_path.display()
        )
    })?;
    writer
        .write_all(format!("{identity}\n").as_bytes())
        .map_err(|error| {
            format!(
                "failed to write MCP install generation lock {}: {error}",
                lock_path.display()
            )
        })?;
    writer.sync_all().map_err(|error| {
        format!(
            "failed to sync MCP install generation lock {}: {error}",
            lock_path.display()
        )
    })
}

#[cfg(test)]
fn read_generation_lock_identity_path(path: &Path) -> Result<String, String> {
    let marker = read_generation_marker_path(path)?;
    let lock_path = marker.lock_path();
    let file = open_existing_generation_lock_path(lock_path)?;
    read_generation_lock_identity(&file, lock_path)?.ok_or_else(|| {
        format!(
            "MCP install generation lock {} is empty",
            lock_path.display()
        )
    })
}

/// Verify that the visible lock path still names the locked generation and retains its UUID.
///
/// Windows byte-range locks reject reads through every other handle, including handles opened by
/// the locking process. Compare file identities there, then read the UUID through the owning
/// handle. Unix locks are advisory, so reading the visible path preserves the same check directly.
fn visible_generation_lock_matches(
    locked: &File,
    lock_path: &Path,
    expected_identity: &str,
) -> Result<bool, String> {
    #[cfg(windows)]
    {
        let visible = open_existing_generation_lock_path(lock_path)?;
        if windows_file_identity(locked, lock_path)? != windows_file_identity(&visible, lock_path)?
        {
            return Ok(false);
        }
        return read_generation_lock_identity(locked, lock_path)
            .map(|identity| identity.as_deref() == Some(expected_identity));
    }
    #[cfg(not(windows))]
    {
        let visible = open_existing_generation_lock_path(lock_path)?;
        #[cfg(unix)]
        if unix_file_identity(locked, lock_path)? != unix_file_identity(&visible, lock_path)? {
            return Ok(false);
        }
        read_generation_lock_identity(&visible, lock_path)
            .map(|identity| identity.as_deref() == Some(expected_identity))
    }
}

#[cfg(unix)]
fn unix_file_identity(file: &File, lock_path: &Path) -> Result<(u64, u64), String> {
    use std::os::unix::fs::MetadataExt;

    file.metadata()
        .map(|metadata| (metadata.dev(), metadata.ino()))
        .map_err(|error| {
            format!(
                "failed to identify MCP install generation lock {}: {error}",
                lock_path.display()
            )
        })
}

#[cfg(windows)]
fn windows_file_identity(file: &File, lock_path: &Path) -> Result<(u64, [u8; 16]), String> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_ID_INFO, FileIdInfo, GetFileInformationByHandleEx,
    };

    let mut information = FILE_ID_INFO::default();
    // SAFETY: `file` owns a live handle and `information` is writable for the duration of the
    // synchronous call.
    if unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle(),
            FileIdInfo,
            (&raw mut information).cast(),
            std::mem::size_of::<FILE_ID_INFO>() as u32,
        )
    } == 0
    {
        return Err(format!(
            "failed to identify MCP install generation lock {}: {}",
            lock_path.display(),
            std::io::Error::last_os_error()
        ));
    }
    Ok((
        information.VolumeSerialNumber,
        information.FileId.Identifier,
    ))
}

fn read_generation_lock_identity(file: &File, lock_path: &Path) -> Result<Option<String>, String> {
    let mut raw = String::new();
    let mut reader = file;
    reader.seek(SeekFrom::Start(0)).map_err(|error| {
        format!(
            "failed to seek MCP install generation lock {}: {error}",
            lock_path.display()
        )
    })?;
    reader
        .take(MAX_GENERATION_LOCK_ID_BYTES.saturating_add(1) as u64)
        .read_to_string(&mut raw)
        .map_err(|error| {
            format!(
                "failed to read MCP install generation lock {}: {error}",
                lock_path.display()
            )
        })?;
    if raw.len() > MAX_GENERATION_LOCK_ID_BYTES {
        return Err(format!(
            "MCP install generation lock {} exceeds the {MAX_GENERATION_LOCK_ID_BYTES}-byte limit",
            lock_path.display()
        ));
    }
    let identity = raw.trim();
    if identity.is_empty() {
        return Ok(None);
    }
    uuid::Uuid::parse_str(identity)
        .map(|identity| Some(identity.to_string()))
        .map_err(|error| {
            format!(
                "MCP install generation lock {} has an invalid identity: {error}",
                lock_path.display()
            )
        })
}

fn open_generation(path: &Path) -> Result<File, String> {
    OpenOptions::new().read(true).open(path).map_err(|error| {
        format!(
            "failed to open MCP install generation {}: {error}",
            path.display()
        )
    })
}

fn read_generation_marker_path(path: &Path) -> Result<GenerationMarker, String> {
    let file = open_generation(path)?;
    read_generation_marker(&file, path)
}

fn read_generation_marker(file: &File, path: &Path) -> Result<GenerationMarker, String> {
    let mut raw = String::new();
    let mut reader = file;
    reader.seek(SeekFrom::Start(0)).map_err(|error| {
        format!(
            "failed to seek MCP install generation {}: {error}",
            path.display()
        )
    })?;
    reader
        .take(MAX_GENERATION_MARKER_BYTES.saturating_add(1) as u64)
        .read_to_string(&mut raw)
        .map_err(|error| {
            format!(
                "failed to read MCP install generation {}: {error}",
                path.display()
            )
        })?;
    if raw.len() > MAX_GENERATION_MARKER_BYTES {
        return Err(format!(
            "MCP install generation {} exceeds the {MAX_GENERATION_MARKER_BYTES}-byte limit",
            path.display()
        ));
    }
    let mut lines = raw.lines();
    let token = lines.next().unwrap_or_default().trim();
    if token.is_empty() {
        return Err(format!(
            "MCP install generation {} is empty",
            path.display()
        ));
    }
    let (retired, token) = match token.strip_prefix(RETIRED_GENERATION_PREFIX) {
        Some("") => {
            return Err(format!(
                "MCP install generation {} has a retired marker without a token",
                path.display()
            ));
        }
        Some(token) => (true, token),
        None => (false, token),
    };
    if token.len() > MAX_GENERATION_TOKEN_BYTES {
        return Err(format!(
            "MCP install generation token in {} exceeds the {MAX_GENERATION_TOKEN_BYTES}-byte limit",
            path.display()
        ));
    }
    let lock_path = match lines.next() {
        Some(encoded) => {
            let encoded = encoded
                .strip_prefix(GENERATION_LOCK_PATH_PREFIX)
                .ok_or_else(|| {
                    format!(
                        "MCP install generation {} has an invalid lock-path record",
                        path.display()
                    )
                })?;
            if lines.next().is_some() {
                return Err(format!(
                    "MCP install generation {} has unexpected trailing records",
                    path.display()
                ));
            }
            let lock_path = decode_lock_path(encoded).map_err(|error| {
                format!(
                    "MCP install generation {} has an invalid lock path: {error}",
                    path.display()
                )
            })?;
            if !lock_path.is_absolute() {
                return Err(format!(
                    "MCP install generation {} has a non-absolute external lock path",
                    path.display()
                ));
            }
            lock_path
        }
        None => absolute_lock_path(&generation_lock_path(path))?,
    };
    Ok(if retired {
        GenerationMarker::Retired {
            token: token.to_owned(),
            lock_path,
        }
    } else {
        GenerationMarker::active(token, lock_path)
    })
}

fn encode_lock_path(path: &Path) -> String {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    };
    #[cfg(windows)]
    let bytes = {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str()
            .encode_wide()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    };
    #[cfg(not(any(unix, windows)))]
    let bytes = path.to_string_lossy().as_bytes().to_vec();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn decode_lock_path(encoded: &str) -> Result<PathBuf, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| error.to_string())?;
    #[cfg(unix)]
    let path = {
        use std::os::unix::ffi::OsStringExt;
        PathBuf::from(std::ffi::OsString::from_vec(bytes))
    };
    #[cfg(windows)]
    let path = {
        use std::os::windows::ffi::OsStringExt;
        let pairs = bytes.chunks_exact(2);
        if !pairs.remainder().is_empty() {
            return Err("UTF-16 lock path has an odd byte length".into());
        }
        let wide = pairs
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        PathBuf::from(std::ffi::OsString::from_wide(&wide))
    };
    #[cfg(not(any(unix, windows)))]
    let path = PathBuf::from(String::from_utf8(bytes).map_err(|error| error.to_string())?);
    if path.as_os_str().is_empty() {
        return Err("lock path is empty".into());
    }
    Ok(path)
}

fn retired_generation_error(path: &Path) -> String {
    format!(
        "plugin MCP install generation at {} has been retired",
        path.display()
    )
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/install_generation_tests.rs"]
mod tests;
