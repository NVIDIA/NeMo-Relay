// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owner-private daemon identity, trust, and environment state.

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::identity::{Fingerprint, MachineIdentity, PublicIdentity, TokenDigest};
use crate::error::CliError;
use crate::filesystem::{LockAttempt, atomic_write_private, try_lock_exclusive, unlock_file};

pub(crate) const ROUTE_TOKEN_ENV: &str = "NEMO_RELAY_CLIENT_TOKEN";
const IDENTITY_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_IDENTITY_BYTES: u64 = 4 * 1024;
const ACTIVE_WORKER_GENERATIONS_FILENAME: &str = "active-worker-generations.json";
const ACTIVE_WORKER_GENERATIONS_SCHEMA_VERSION: u32 = 1;
const MAX_ACTIVE_WORKER_GENERATIONS: usize = 4_096;
const MAX_ACTIVE_WORKER_GENERATIONS_BYTES: u64 = 2 * 1024 * 1024;
const MAX_GENERATION_ID_BYTES: usize = 128;

#[derive(Clone)]
pub(crate) struct RouteCredential {
    value: String,
    digest: TokenDigest,
}

impl fmt::Debug for RouteCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RouteCredential")
            .field("digest", &self.digest)
            .finish_non_exhaustive()
    }
}

impl RouteCredential {
    pub(crate) fn from_environment() -> Result<Self, CliError> {
        let value = std::env::var(ROUTE_TOKEN_ENV).map_err(|_| {
            CliError::Config(format!(
                "managed daemon integration requires {ROUTE_TOKEN_ENV}; contact the managed environment administrator"
            ))
        })?;
        Self::parse(value)
    }

    pub(crate) fn parse(value: String) -> Result<Self, CliError> {
        if value.trim() != value || value.bytes().any(|byte| byte.is_ascii_whitespace()) {
            return Err(CliError::Config(format!(
                "{ROUTE_TOKEN_ENV} must be an unpadded base64url credential without whitespace"
            )));
        }
        let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(&value)
            .map_err(|_| {
                CliError::Config(format!(
                    "{ROUTE_TOKEN_ENV} must be an unpadded base64url credential"
                ))
            })?;
        if decoded.len() != 32 {
            return Err(CliError::Config(format!(
                "{ROUTE_TOKEN_ENV} must decode to exactly 32 bytes"
            )));
        }
        let digest = TokenDigest::from_token(value.as_bytes());
        Ok(Self { value, digest })
    }

    pub(crate) fn expose(&self) -> &str {
        &self.value
    }

    pub(crate) const fn digest(&self) -> TokenDigest {
        self.digest
    }
}

pub(crate) fn load_or_create_machine_identity() -> Result<MachineIdentity, CliError> {
    load_or_create_identity(&daemon_state_dir()?.join("machine-identity.pk8"))
}

pub(crate) fn load_or_create_daemon_identity() -> Result<MachineIdentity, CliError> {
    load_or_create_identity(&daemon_state_dir()?.join("daemon-identity.pk8"))
}

/// Owner-private durable record of the only worker generation allowed to recover per route.
#[derive(Debug)]
pub(crate) struct ActiveWorkerGenerations {
    path: PathBuf,
}

impl ActiveWorkerGenerations {
    /// Loads and validates the durable generation record before the daemon accepts traffic.
    pub(crate) fn load() -> Result<Self, CliError> {
        Self::load_from_path(daemon_state_dir()?.join(ACTIVE_WORKER_GENERATIONS_FILENAME))
    }

    fn load_from_path(path: PathBuf) -> Result<Self, CliError> {
        let state = Self { path };
        state.with_locked_generations(|_| Ok(((), false)))?;
        Ok(state)
    }

    /// Returns whether `generation_id` is the exact active generation for `fingerprint`.
    pub(crate) fn matches(
        &self,
        fingerprint: Fingerprint,
        generation_id: &str,
    ) -> Result<bool, CliError> {
        validate_generation_id(generation_id)?;
        self.with_locked_generations(|generations| {
            Ok((
                generations
                    .get(&fingerprint)
                    .is_some_and(|active| active == generation_id),
                false,
            ))
        })
    }

    /// Publishes a ready generation, atomically replacing any prior generation for the route.
    pub(crate) fn publish(
        &self,
        fingerprint: Fingerprint,
        generation_id: &str,
    ) -> Result<Option<String>, CliError> {
        validate_generation_id(generation_id)?;
        self.with_locked_generations(|generations| {
            if !generations.contains_key(&fingerprint)
                && generations.len() >= MAX_ACTIVE_WORKER_GENERATIONS
            {
                return Err(CliError::Config(format!(
                    "active worker generation state exceeds {MAX_ACTIVE_WORKER_GENERATIONS} routes"
                )));
            }
            let previous = generations.insert(fingerprint, generation_id.to_owned());
            let changed = previous.as_deref() != Some(generation_id);
            Ok((previous, changed))
        })
    }

    /// Revokes a generation only if it is still active, protecting a newer replacement.
    pub(crate) fn revoke_if_matches(
        &self,
        fingerprint: Fingerprint,
        generation_id: &str,
    ) -> Result<bool, CliError> {
        validate_generation_id(generation_id)?;
        self.with_locked_generations(|generations| {
            let matches = generations
                .get(&fingerprint)
                .is_some_and(|active| active == generation_id);
            if matches {
                generations.remove(&fingerprint);
            }
            Ok((matches, matches))
        })
    }

    /// Restores the prior value if a broker publication loses a race after durable publication.
    pub(crate) fn restore_if_matches(
        &self,
        fingerprint: Fingerprint,
        expected_generation_id: &str,
        previous_generation_id: Option<&str>,
    ) -> Result<bool, CliError> {
        validate_generation_id(expected_generation_id)?;
        if let Some(previous) = previous_generation_id {
            validate_generation_id(previous)?;
        }
        self.with_locked_generations(|generations| {
            let matches = generations
                .get(&fingerprint)
                .is_some_and(|active| active == expected_generation_id);
            if !matches {
                return Ok((false, false));
            }
            match previous_generation_id {
                Some(previous) => {
                    generations.insert(fingerprint, previous.to_owned());
                }
                None => {
                    generations.remove(&fingerprint);
                }
            }
            Ok((true, true))
        })
    }

    fn with_locked_generations<T>(
        &self,
        operation: impl FnOnce(&mut HashMap<Fingerprint, String>) -> Result<(T, bool), CliError>,
    ) -> Result<T, CliError> {
        let parent = self
            .path
            .parent()
            .ok_or_else(|| CliError::Config("worker generation state path has no parent".into()))?;
        create_private_directory(parent)?;
        let lock_path = self.path.with_extension("lock");
        let lock = open_private_lock(&lock_path)?;
        acquire_private_lock(&lock, &lock_path)?;
        let result = (|| {
            let mut generations = read_active_worker_generations(&self.path)?;
            let (output, changed) = operation(&mut generations)?;
            if changed {
                write_active_worker_generations(&self.path, &generations)?;
            }
            Ok(output)
        })();
        let _ = unlock_file(&lock);
        result
    }

    #[cfg(test)]
    pub(crate) fn load_for_test(path: PathBuf) -> Result<Self, CliError> {
        Self::load_from_path(path)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedActiveWorkerGenerations {
    schema_version: u32,
    generations: Vec<PersistedActiveWorkerGeneration>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedActiveWorkerGeneration {
    fingerprint: Fingerprint,
    generation_id: String,
}

fn read_active_worker_generations(path: &Path) -> Result<HashMap<Fingerprint, String>, CliError> {
    let Some(bytes) = read_bounded(
        path,
        MAX_ACTIVE_WORKER_GENERATIONS_BYTES,
        "active worker generation state",
    )?
    else {
        return Ok(HashMap::new());
    };
    let persisted: PersistedActiveWorkerGenerations =
        serde_json::from_slice(&bytes).map_err(|_| {
            CliError::Config(format!(
                "active worker generation state {} is corrupt",
                path.display()
            ))
        })?;
    if persisted.schema_version != ACTIVE_WORKER_GENERATIONS_SCHEMA_VERSION {
        return Err(CliError::Config(format!(
            "active worker generation state {} has unsupported schema version {}",
            path.display(),
            persisted.schema_version
        )));
    }
    if persisted.generations.len() > MAX_ACTIVE_WORKER_GENERATIONS {
        return Err(CliError::Config(format!(
            "active worker generation state {} exceeds {MAX_ACTIVE_WORKER_GENERATIONS} routes",
            path.display()
        )));
    }
    let mut generations = HashMap::with_capacity(persisted.generations.len());
    for entry in persisted.generations {
        validate_generation_id(&entry.generation_id)?;
        if generations
            .insert(entry.fingerprint, entry.generation_id)
            .is_some()
        {
            return Err(CliError::Config(format!(
                "active worker generation state {} contains a duplicate fingerprint",
                path.display()
            )));
        }
    }
    Ok(generations)
}

fn write_active_worker_generations(
    path: &Path,
    generations: &HashMap<Fingerprint, String>,
) -> Result<(), CliError> {
    let mut entries = generations
        .iter()
        .map(
            |(fingerprint, generation_id)| PersistedActiveWorkerGeneration {
                fingerprint: *fingerprint,
                generation_id: generation_id.clone(),
            },
        )
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.fingerprint.to_string());
    let document = PersistedActiveWorkerGenerations {
        schema_version: ACTIVE_WORKER_GENERATIONS_SCHEMA_VERSION,
        generations: entries,
    };
    let bytes = serde_json::to_vec(&document).map_err(|error| {
        CliError::Config(format!(
            "failed to serialize active worker generation state: {error}"
        ))
    })?;
    if bytes.len() as u64 > MAX_ACTIVE_WORKER_GENERATIONS_BYTES {
        return Err(CliError::Config(format!(
            "active worker generation state exceeds {MAX_ACTIVE_WORKER_GENERATIONS_BYTES} bytes"
        )));
    }
    atomic_write_private(path, &bytes).map_err(CliError::Config)?;
    sync_parent_directory(path)
}

fn validate_generation_id(generation_id: &str) -> Result<(), CliError> {
    if generation_id.is_empty() || generation_id.len() > MAX_GENERATION_ID_BYTES {
        return Err(CliError::Config(
            "active worker generation ID is invalid".into(),
        ));
    }
    Ok(())
}

pub(crate) fn verify_or_store_daemon_pin(
    daemon_origin: &str,
    identity: PublicIdentity,
) -> Result<(), CliError> {
    let name = hex_digest(daemon_origin.as_bytes());
    let path = daemon_state_dir()?
        .join("pins")
        .join(format!("{name}.ed25519"));
    let parent = path
        .parent()
        .ok_or_else(|| CliError::Config("daemon pin path has no parent".into()))?;
    create_private_directory(parent)?;
    let lock_path = path.with_extension("lock");
    let lock = open_private_lock(&lock_path)?;
    acquire_private_lock(&lock, &lock_path)?;
    let result = match read_bounded(&path, MAX_IDENTITY_BYTES, "daemon trust pin")? {
        Some(existing) if existing == identity.as_bytes() => Ok(()),
        Some(_) => Err(CliError::Unauthorized(format!(
            "daemon identity changed for {daemon_origin}; remove the owner-private trust pin only after verifying the daemon replacement"
        ))),
        None => atomic_write_private(&path, identity.as_bytes()).map_err(CliError::Config),
    };
    let _ = unlock_file(&lock);
    result
}

fn load_or_create_identity(path: &Path) -> Result<MachineIdentity, CliError> {
    let parent = path
        .parent()
        .ok_or_else(|| CliError::Config("daemon identity path has no parent".into()))?;
    create_private_directory(parent)?;
    let lock_path = path.with_extension("lock");
    let lock = open_private_lock(&lock_path)?;
    acquire_private_lock(&lock, &lock_path)?;
    let result = match read_bounded(path, MAX_IDENTITY_BYTES, "daemon identity state")? {
        Some(bytes) => MachineIdentity::from_pkcs8(&bytes)
            .map_err(|error| CliError::Config(format!("invalid daemon identity: {error}"))),
        None => {
            let generated = MachineIdentity::generate().map_err(|error| {
                CliError::Config(format!("failed to generate identity: {error}"))
            })?;
            atomic_write_private(path, &generated.pkcs8).map_err(CliError::Config)?;
            Ok(generated.identity)
        }
    };
    let _ = unlock_file(&lock);
    result
}

fn open_private_lock(path: &Path) -> Result<std::fs::File, CliError> {
    #[cfg(windows)]
    let file = crate::filesystem::open_private_windows_file(path).map_err(|error| {
        CliError::Config(format!(
            "failed to open owner-private daemon lock {}: {error}",
            path.display()
        ))
    })?;
    #[cfg(not(windows))]
    let file = {
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        options.open(path).map_err(|error| {
            CliError::Config(format!(
                "failed to open owner-private daemon lock {}: {error}",
                path.display()
            ))
        })?
    };
    validate_private_file(&file, path)?;
    Ok(file)
}

fn acquire_private_lock(file: &std::fs::File, path: &Path) -> Result<(), CliError> {
    let deadline = Instant::now() + IDENTITY_LOCK_TIMEOUT;
    loop {
        match try_lock_exclusive(file) {
            Ok(LockAttempt::Acquired) => return Ok(()),
            Ok(LockAttempt::Contended) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(25));
            }
            Ok(LockAttempt::Contended) => {
                return Err(CliError::Config(format!(
                    "timed out waiting for daemon identity lock {}",
                    path.display()
                )));
            }
            Err(error) => return Err(CliError::Io(error)),
        }
    }
}

fn read_bounded(
    path: &Path,
    max_bytes: u64,
    description: &str,
) -> Result<Option<Vec<u8>>, CliError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(CliError::Io(error)),
    };
    validate_private_file(&file, path)?;
    let length = file.metadata()?.len();
    if length > max_bytes {
        return Err(CliError::Config(format!(
            "{description} {} exceeds {max_bytes} bytes",
            path.display(),
        )));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.read_to_end(&mut bytes)?;
    Ok(Some(bytes))
}

fn daemon_state_dir() -> Result<PathBuf, CliError> {
    crate::configuration::user_config_dir()
        .map(|directory| directory.join("daemon"))
        .ok_or_else(|| {
            CliError::Config(
                "cannot determine the per-user daemon state directory; set HOME or USERPROFILE"
                    .into(),
            )
        })
}

fn create_private_directory(path: &Path) -> Result<(), CliError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_dir() => {
            return Err(CliError::Config(format!(
                "daemon state directory {} must be a real directory",
                path.display()
            )));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => fs::create_dir_all(path)?,
        Err(error) => return Err(CliError::Io(error)),
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        use std::os::unix::fs::MetadataExt;
        let metadata = fs::symlink_metadata(path)?;
        if metadata.uid() != unsafe { libc::geteuid() } {
            return Err(CliError::Config(format!(
                "daemon state directory {} is not owned by the current user",
                path.display()
            )));
        }
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(windows)]
    crate::filesystem::protect_private_windows_path(path)?;
    Ok(())
}

fn validate_private_file(file: &std::fs::File, path: &Path) -> Result<(), CliError> {
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(CliError::Config(format!(
            "daemon state {} must be a regular file",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if metadata.uid() != unsafe { libc::geteuid() } {
            return Err(CliError::Config(format!(
                "daemon state {} is not owned by the current user",
                path.display()
            )));
        }
        if metadata.mode() & 0o077 != 0 {
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
    }
    #[cfg(windows)]
    crate::filesystem::protect_private_windows_path(path)?;
    Ok(())
}

fn sync_parent_directory(path: &Path) -> Result<(), CliError> {
    #[cfg(unix)]
    {
        let parent = path
            .parent()
            .ok_or_else(|| CliError::Config("worker generation state path has no parent".into()))?;
        std::fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/state_tests.rs"]
mod tests;
