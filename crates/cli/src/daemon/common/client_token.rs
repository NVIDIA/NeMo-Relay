// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-user daemon route credential resolution and the owner-private client token file.
//!
//! Managed clients resolve their route credential from `NEMO_RELAY_CLIENT_TOKEN` first and then
//! from `<user config dir>/.client-token`. The file exists because environment pushed after
//! login (for example with `launchctl setenv`) never reaches processes started earlier, while a
//! file in the user's own config directory is visible to every process of that user. The token
//! value is never logged, printed, or included in an error.

use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use base64::Engine;
use ring::rand::{SecureRandom, SystemRandom};
use serde::Serialize;

use super::state::{ROUTE_TOKEN_ENV, RouteCredential};
use crate::error::CliError;

/// Hidden token file name under the per-user NeMo Relay config directory.
pub(crate) const CLIENT_TOKEN_FILENAME: &str = ".client-token";
/// A 32-byte unpadded base64url token is 43 bytes; the bound leaves room for a line ending only.
const MAX_CLIENT_TOKEN_FILE_BYTES: u64 = 128;

/// Where a resolved route credential came from. Diagnostics only; never the value itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CredentialSource {
    Environment,
    File,
}

impl CredentialSource {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Environment => "env",
            Self::File => "file",
        }
    }
}

impl fmt::Display for CredentialSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A usable route credential plus its source.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedCredential {
    pub(crate) credential: RouteCredential,
    pub(crate) source: CredentialSource,
}

/// Why an existing token file was not used. Display output never contains file contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TokenFileProblem {
    Symlink,
    NotRegularFile,
    /// Only detectable from Unix metadata; Windows checks ownership through the DACL.
    #[cfg(unix)]
    NotOwnedByCurrentUser,
    InsecurePermissions,
    TooLarge,
    InvalidContents,
    Unreadable(std::io::ErrorKind),
}

impl TokenFileProblem {
    const fn as_str(&self) -> &'static str {
        match self {
            Self::Symlink => "symlink",
            Self::NotRegularFile => "not_regular_file",
            #[cfg(unix)]
            Self::NotOwnedByCurrentUser => "not_owned_by_current_user",
            Self::InsecurePermissions => "insecure_permissions",
            Self::TooLarge => "too_large",
            Self::InvalidContents => "invalid_contents",
            Self::Unreadable(_) => "unreadable",
        }
    }
}

impl fmt::Display for TokenFileProblem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Symlink => formatter.write_str("is a symbolic link"),
            Self::NotRegularFile => formatter.write_str("is not a regular file"),
            #[cfg(unix)]
            Self::NotOwnedByCurrentUser => formatter.write_str("is not owned by the current user"),
            Self::InsecurePermissions => formatter.write_str(
                "is accessible by other users; it must be readable and writable only by its owner (mode 0600)",
            ),
            Self::TooLarge => write!(
                formatter,
                "exceeds {MAX_CLIENT_TOKEN_FILE_BYTES} bytes"
            ),
            Self::InvalidContents => formatter.write_str(
                "does not contain one unpadded base64url credential encoding exactly 32 bytes",
            ),
            Self::Unreadable(kind) => write!(formatter, "cannot be read ({kind})"),
        }
    }
}

/// An unusable token file, naming the path and the problem but never the contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenFileError {
    pub(crate) path: PathBuf,
    pub(crate) problem: TokenFileProblem,
}

impl fmt::Display for TokenFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "client token file {} {}",
            self.path.display(),
            self.problem
        )
    }
}

/// Returns `<user config dir>/.client-token`, when the user config directory is resolvable.
pub(crate) fn client_token_path() -> Option<PathBuf> {
    crate::configuration::user_config_dir().map(|directory| directory.join(CLIENT_TOKEN_FILENAME))
}

/// Resolves the route credential: a valid environment value, else a safe valid token file.
///
/// Invalid or unsafe sources are skipped with one redacted warning per process rather than
/// failing, so a caller with no usable credential can fall back to pass-through behavior.
pub(crate) fn resolve_route_credential() -> Option<ResolvedCredential> {
    resolve_route_credential_from(
        std::env::var_os(ROUTE_TOKEN_ENV),
        client_token_path().as_deref(),
    )
}

pub(crate) fn resolve_route_credential_from(
    environment: Option<OsString>,
    token_file: Option<&Path>,
) -> Option<ResolvedCredential> {
    if let Some(value) = environment.filter(|value| !value.is_empty()) {
        match value
            .into_string()
            .ok()
            .and_then(|value| RouteCredential::parse(value).ok())
        {
            Some(credential) => {
                return Some(ResolvedCredential {
                    credential,
                    source: CredentialSource::Environment,
                });
            }
            None => warn_once(&INVALID_ENVIRONMENT_WARNED, || {
                log::warn!(
                    target: "nemo_relay.daemon",
                    event = "route_credential_skipped",
                    source = CredentialSource::Environment.as_str(),
                    reason = "invalid_value";
                    "{ROUTE_TOKEN_ENV} is not a valid route credential; ignoring it"
                );
            }),
        }
    }
    let path = token_file?;
    match read_client_token_file(path) {
        Ok(Some(credential)) => Some(ResolvedCredential {
            credential,
            source: CredentialSource::File,
        }),
        Ok(None) => None,
        Err(error) => {
            warn_once(&INVALID_FILE_WARNED, || {
                log::warn!(
                    target: "nemo_relay.daemon",
                    event = "route_credential_skipped",
                    source = CredentialSource::File.as_str(),
                    reason = error.problem.as_str();
                    "{error}; ignoring it"
                );
            });
            None
        }
    }
}

static INVALID_ENVIRONMENT_WARNED: AtomicBool = AtomicBool::new(false);
static INVALID_FILE_WARNED: AtomicBool = AtomicBool::new(false);

fn warn_once(flag: &AtomicBool, emit: impl FnOnce()) {
    if !flag.swap(true, Ordering::Relaxed) {
        emit();
    }
}

/// Reads and validates the token file. `Ok(None)` means the file does not exist.
pub(crate) fn read_client_token_file(
    path: &Path,
) -> Result<Option<RouteCredential>, TokenFileError> {
    let fail = |problem| TokenFileError {
        path: path.to_path_buf(),
        problem,
    };
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(fail(TokenFileProblem::Unreadable(error.kind()))),
    };
    if metadata.file_type().is_symlink() {
        return Err(fail(TokenFileProblem::Symlink));
    }
    if !metadata.is_file() {
        return Err(fail(TokenFileProblem::NotRegularFile));
    }
    let file = open_token_file_for_read(path).map_err(fail)?;
    // Validate the opened handle, not the earlier path lookup, so a swapped path cannot pass.
    validate_token_file_handle(&file).map_err(fail)?;
    let mut bytes = Vec::new();
    file.take(MAX_CLIENT_TOKEN_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| fail(TokenFileProblem::Unreadable(error.kind())))?;
    if bytes.len() as u64 > MAX_CLIENT_TOKEN_FILE_BYTES {
        return Err(fail(TokenFileProblem::TooLarge));
    }
    parse_token_file_contents(&bytes)
        .map(Some)
        .ok_or_else(|| fail(TokenFileProblem::InvalidContents))
}

fn parse_token_file_contents(bytes: &[u8]) -> Option<RouteCredential> {
    let text = std::str::from_utf8(bytes).ok()?;
    let text = text
        .strip_suffix("\r\n")
        .or_else(|| text.strip_suffix('\n'))
        .unwrap_or(text);
    RouteCredential::parse(text.to_owned()).ok()
}

#[cfg(unix)]
fn open_token_file_for_read(path: &Path) -> Result<fs::File, TokenFileProblem> {
    use std::os::unix::fs::OpenOptionsExt;

    fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| {
            if error.raw_os_error() == Some(libc::ELOOP) {
                TokenFileProblem::Symlink
            } else {
                TokenFileProblem::Unreadable(error.kind())
            }
        })
}

#[cfg(windows)]
fn open_token_file_for_read(path: &Path) -> Result<fs::File, TokenFileProblem> {
    // Rejects reparse points and requires the same protected owner/System DACL as other
    // owner-private Relay state.
    crate::filesystem::open_private_windows_file_for_read(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::PermissionDenied {
            TokenFileProblem::InsecurePermissions
        } else {
            TokenFileProblem::Unreadable(error.kind())
        }
    })
}

#[cfg(not(any(unix, windows)))]
fn open_token_file_for_read(path: &Path) -> Result<fs::File, TokenFileProblem> {
    fs::File::open(path).map_err(|error| TokenFileProblem::Unreadable(error.kind()))
}

fn validate_token_file_handle(file: &fs::File) -> Result<(), TokenFileProblem> {
    let metadata = file
        .metadata()
        .map_err(|error| TokenFileProblem::Unreadable(error.kind()))?;
    if !metadata.is_file() {
        return Err(TokenFileProblem::NotRegularFile);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        if metadata.uid() != current_euid() {
            return Err(TokenFileProblem::NotOwnedByCurrentUser);
        }
        if metadata.mode() & 0o077 != 0 {
            return Err(TokenFileProblem::InsecurePermissions);
        }
    }
    if metadata.len() > MAX_CLIENT_TOKEN_FILE_BYTES {
        return Err(TokenFileProblem::TooLarge);
    }
    Ok(())
}

#[cfg(unix)]
fn current_euid() -> u32 {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() }
}

/// Result of `nemo-relay daemon token ensure`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EnsureStatus {
    Created,
    Existing,
}

impl EnsureStatus {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Existing => "existing",
        }
    }
}

/// Ensures the current user's token file exists, creating it from the OS CSPRNG when absent.
///
/// An existing valid, safe file is left untouched. An existing invalid or unsafe file is never
/// overwritten. Concurrent callers converge on one file because publication is exclusive.
pub(crate) fn ensure_client_token(path: &Path) -> Result<EnsureStatus, CliError> {
    refuse_privileged_user()?;
    ensure_client_token_unprivileged(path)
}

fn ensure_client_token_unprivileged(path: &Path) -> Result<EnsureStatus, CliError> {
    match read_client_token_file(path) {
        Ok(Some(_)) => return Ok(EnsureStatus::Existing),
        Ok(None) => {}
        Err(error) => return Err(unusable_existing_file(error)),
    }
    let parent = path
        .parent()
        .ok_or_else(|| CliError::Config("client token path has no parent directory".into()))?;
    ensure_parent_directory(parent)?;

    let mut token = [0_u8; 32];
    SystemRandom::new().fill(&mut token).map_err(|_| {
        CliError::Config("failed to generate a client token from the OS random source".into())
    })?;
    let mut contents = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(token);
    contents.push('\n');

    let temporary = parent.join(format!(
        "{CLIENT_TOKEN_FILENAME}.{}.tmp",
        uuid::Uuid::now_v7()
    ));
    let published = (|| {
        write_private_temporary(&temporary, contents.as_bytes())?;
        // A hard link never replaces an existing name, so exactly one concurrent writer wins.
        match fs::hard_link(&temporary, path) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
            Err(error) => Err(CliError::Config(format!(
                "failed to publish client token file {}: {error}",
                path.display()
            ))),
        }
    })();
    let _ = fs::remove_file(&temporary);
    if published? {
        sync_directory(parent);
        return Ok(EnsureStatus::Created);
    }
    // Another process published first; accept its file only if it is valid and safe.
    match read_client_token_file(path) {
        Ok(Some(_)) => Ok(EnsureStatus::Existing),
        Ok(None) => Err(CliError::Config(format!(
            "client token file {} disappeared during creation; retry the command",
            path.display()
        ))),
        Err(error) => Err(unusable_existing_file(error)),
    }
}

fn unusable_existing_file(error: TokenFileError) -> CliError {
    CliError::Config(format!(
        "{error}; it was left unchanged. Review and remove it, then run `nemo-relay daemon token ensure` again"
    ))
}

fn refuse_privileged_user() -> Result<(), CliError> {
    #[cfg(unix)]
    if current_euid() == 0 {
        return Err(CliError::Config(
            "refusing to create a client token as root; run `nemo-relay daemon token ensure` as the user who runs the coding agent".into(),
        ));
    }
    Ok(())
}

fn ensure_parent_directory(parent: &Path) -> Result<(), CliError> {
    // Follows a symlinked config directory (common with dotfile managers) and checks the target,
    // which is what decides who can replace the token. An existing directory is the user's
    // general Relay config directory, so it is rejected rather than repaired.
    match fs::metadata(parent) {
        Ok(metadata) if metadata.is_dir() => return validate_existing_parent(parent, &metadata),
        Ok(_) => {
            return Err(CliError::Config(format!(
                "client token directory {} is not a directory",
                parent.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(CliError::Io(error)),
    }
    fs::create_dir_all(parent)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(windows)]
    crate::filesystem::protect_private_windows_path(parent)?;
    Ok(())
}

#[cfg(unix)]
fn validate_existing_parent(parent: &Path, metadata: &fs::Metadata) -> Result<(), CliError> {
    use std::os::unix::fs::MetadataExt;

    if metadata.uid() != current_euid() {
        return Err(CliError::Config(format!(
            "client token directory {} is not owned by the current user; it was left unchanged",
            parent.display()
        )));
    }
    if metadata.mode() & 0o022 != 0 {
        return Err(CliError::Config(format!(
            "client token directory {} is writable by other users; remove group and other write permission (for example `chmod go-w`) and run `nemo-relay daemon token ensure` again",
            parent.display()
        )));
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_existing_parent(_parent: &Path, _metadata: &fs::Metadata) -> Result<(), CliError> {
    // The token file itself is still required to carry the protected owner-only DACL.
    Ok(())
}

#[cfg(unix)]
fn write_private_temporary(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    // The creation mode is filtered by umask; make the owner-only mode explicit.
    file.set_permissions(fs::Permissions::from_mode(0o600))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(windows)]
fn write_private_temporary(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    // The temporary name is unique, so this creates a new file with the protected owner/System
    // DACL. Hard links share that security descriptor with the published name.
    let mut file = crate::filesystem::open_private_windows_file(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn write_private_temporary(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(_directory: &Path) {
    #[cfg(unix)]
    if let Ok(directory) = fs::File::open(_directory) {
        let _ = directory.sync_all();
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/client_token_tests.rs"]
mod tests;
