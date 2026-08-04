// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(windows)]
use std::fs::File;
#[cfg(not(windows))]
use std::fs::OpenOptions;
#[cfg(windows)]
use std::io;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use fs2::FileExt;
use nemo_relay::plugin::dynamic::{
    DynamicPluginCheckState, DynamicPluginManifest, DynamicPluginManifestLoad, WorkerRuntime,
};
use ring::hmac;
use ring::rand::{SecureRandom, SystemRandom};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::{PluginHostConfigError, Result};
use crate::io::read_bounded_utf8_regular_file;
use crate::io::{MAX_BOUNDED_FILE_BYTES, read_bounded_regular_file};

#[doc(hidden)]
pub const MANAGED_ENVIRONMENTS_DIR: &str = ".dynamic-plugin-environments";
#[doc(hidden)]
pub const ENVIRONMENT_ATTESTATION_FILE: &str = ".nemo-relay-environment.sha256";
const MAX_ENVIRONMENT_FILES: usize = 100_000;
const MAX_ENVIRONMENT_DEPTH: usize = 128;
const HMAC_KEY_BYTES: usize = 32;
const HMAC_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const PYTHON_ENVIRONMENT_ATTESTATION_DOMAIN: &[u8] =
    b"nemo-relay/python-environment-attestation/v1\0";

#[derive(Deserialize)]
struct EnvironmentAttestation {
    version: u8,
    source_artifact_sha256: String,
    environment_sha256: String,
    authentication: String,
}

#[doc(hidden)]
pub fn validate_python_entrypoint_artifact(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
) -> std::result::Result<(), String> {
    let DynamicPluginManifestLoad::Worker(load) = &manifest.load else {
        return Ok(());
    };
    if load.runtime != Some(WorkerRuntime::Python) {
        return Ok(());
    }
    let source = manifest.source.as_ref().ok_or_else(|| {
        "Python worker plugins must declare source.manifest_root and source.artifact".to_owned()
    })?;
    let manifest_root = source
        .manifest_root
        .as_deref()
        .map(str::trim)
        .filter(|root| !root.is_empty())
        .ok_or_else(|| "Python worker plugins must declare source.manifest_root".to_owned())?;
    let artifact = source
        .artifact
        .as_deref()
        .map(str::trim)
        .filter(|artifact| !artifact.is_empty())
        .ok_or_else(|| "Python worker plugins must declare source.artifact".to_owned())?;
    let entrypoint = load
        .entrypoint
        .as_deref()
        .map(str::trim)
        .filter(|entrypoint| !entrypoint.is_empty())
        .ok_or_else(|| "Python worker plugins must declare load.entrypoint".to_owned())?;
    let (module, callable) = entrypoint.split_once(':').ok_or_else(|| {
        format!(
            "Python worker load.entrypoint '{entrypoint}' must use the unambiguous module:function form"
        )
    })?;
    if callable.is_empty()
        || callable.contains(':')
        || module.is_empty()
        || module
            .split('.')
            .any(|segment| segment.is_empty() || segment.contains(['/', '\\', ':']))
    {
        return Err(format!(
            "Python worker load.entrypoint '{entrypoint}' must use the unambiguous module:function form"
        ));
    }
    let manifest_path = Path::new(manifest_ref);
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let unresolved_manifest_root = resolve_relative_path(manifest_dir, manifest_root);
    let manifest_root = unresolved_manifest_root.canonicalize().map_err(|error| {
        format!(
            "could not resolve Python plugin source.manifest_root {}: {error}",
            unresolved_manifest_root.display()
        )
    })?;
    let artifact = resolve_relative_path(manifest_dir, artifact)
        .canonicalize()
        .map_err(|error| format!("could not resolve Python source.artifact: {error}"))?;
    let module_path = module
        .split('.')
        .fold(manifest_root, |path, segment| path.join(segment));
    let module_file = module_path.with_extension("py");
    let package_file = module_path.join("__init__.py");
    let mut candidates = [module_file, package_file]
        .into_iter()
        .filter(|path| path.is_file())
        .map(|path| {
            path.canonicalize().map_err(|error| {
                format!(
                    "could not resolve Python entrypoint module file {}: {error}",
                    path.display()
                )
            })
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    candidates.sort();
    candidates.dedup();
    let [entrypoint_artifact] = candidates.as_slice() else {
        return Err(format!(
            "Python worker load.entrypoint '{entrypoint}' must resolve to exactly one source module under source.manifest_root; expected {} or {}",
            module_path.with_extension("py").display(),
            module_path.join("__init__.py").display()
        ));
    };
    if entrypoint_artifact != &artifact {
        return Err(format!(
            "Python worker load.entrypoint '{entrypoint}' resolves to {}, but integrity-checked source.artifact resolves to {}; the executed entrypoint module must be the integrity-checked artifact",
            entrypoint_artifact.display(),
            artifact.display()
        ));
    }
    Ok(())
}

#[doc(hidden)]
pub fn environment_state(
    manifest: &DynamicPluginManifest,
    state_path: &Path,
    environment_ref: Option<&str>,
) -> DynamicPluginCheckState {
    validate_environment_state(manifest, state_path, environment_ref)
        .unwrap_or(DynamicPluginCheckState::Invalid)
}

#[doc(hidden)]
pub fn validate_environment_state(
    manifest: &DynamicPluginManifest,
    state_path: &Path,
    environment_ref: Option<&str>,
) -> Result<DynamicPluginCheckState> {
    if !is_python_worker(manifest) {
        return Ok(DynamicPluginCheckState::Unknown);
    }
    let environment_ref = environment_ref.ok_or_else(|| {
        PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin '{}' has no lifecycle-managed Python environment",
            manifest.plugin.id
        ))
    })?;
    let expected = managed_environment_path(state_path, &manifest.plugin.id)?;
    let configured = absolute_path(Path::new(environment_ref))?;
    let expected_metadata = std::fs::symlink_metadata(&expected).map_err(|error| {
        PluginHostConfigError::io(
            "inspect lifecycle-managed Python environment",
            &expected,
            error,
        )
    })?;
    if !expected_metadata.file_type().is_dir() {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "lifecycle-managed Python environment {} must be a directory and not a symbolic link",
            expected.display()
        )));
    }
    let configured_metadata = std::fs::symlink_metadata(&configured).map_err(|error| {
        PluginHostConfigError::io("inspect configured Python environment", &configured, error)
    })?;
    if !configured_metadata.file_type().is_dir() {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "configured Python environment {} must be a directory and not a symbolic link",
            configured.display()
        )));
    }
    let same_physical_environment = configured == expected
        || std::fs::canonicalize(&configured)
            .ok()
            .zip(std::fs::canonicalize(&expected).ok())
            .is_some_and(|(configured, expected)| configured == expected);
    if !same_physical_environment {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin '{}' configured Python environment {} is not its lifecycle-managed environment {}",
            manifest.plugin.id,
            configured.display(),
            expected.display()
        )));
    }
    let python = environment_python_path(&configured);
    if !python.is_file() {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "managed Python environment {} has no Python launcher at {}",
            configured.display(),
            python.display()
        )));
    }
    let digest = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.sha256.as_deref())
        .ok_or_else(|| {
            PluginHostConfigError::InvalidConfig(format!(
                "Python worker dynamic plugin '{}' requires integrity.sha256 for its lifecycle-managed environment",
                manifest.plugin.id
            ))
        })?;
    verify_environment_attestation(&configured, digest)?;
    Ok(DynamicPluginCheckState::Valid)
}

/// Reads and authenticates a lifecycle-managed environment attestation.
#[doc(hidden)]
pub fn read_environment_attestation(
    environment: &Path,
    expected_source_artifact_sha256: &str,
) -> Result<String> {
    let attestation_path = environment.join(ENVIRONMENT_ATTESTATION_FILE);
    let raw = read_bounded_utf8_regular_file(
        &attestation_path,
        "managed Python environment attestation",
    )?;
    let attestation = serde_json::from_str::<EnvironmentAttestation>(&raw).map_err(|error| {
        PluginHostConfigError::InvalidConfig(format!(
            "managed Python environment attestation {} is invalid at line {}, column {}: {}",
            attestation_path.display(),
            error.line(),
            error.column(),
            crate::error::sanitize_parser_reason(&error.to_string())
        ))
    })?;
    if attestation.version != 1
        || attestation.source_artifact_sha256 != expected_source_artifact_sha256.trim()
        || attestation.environment_sha256.len() != 64
        || !attestation
            .environment_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "managed Python environment attestation {} does not match the trusted source artifact",
            attestation_path.display()
        )));
    }
    if !verify_environment_authentication(
        &attestation.source_artifact_sha256,
        &attestation.environment_sha256,
        &attestation.authentication,
    )? {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "managed Python environment attestation {} failed authentication",
            attestation_path.display()
        )));
    }
    Ok(attestation.environment_sha256)
}

/// Authenticates an environment attestation and verifies the complete environment tree.
#[doc(hidden)]
pub fn verify_environment_attestation(
    environment: &Path,
    expected_source_artifact_sha256: &str,
) -> Result<String> {
    let expected = read_environment_attestation(environment, expected_source_artifact_sha256)?;
    let actual = environment_tree_digest(environment)?;
    if actual != expected {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "managed Python environment {} changed after provisioning",
            environment.display()
        )));
    }
    Ok(actual)
}

fn is_python_worker(manifest: &DynamicPluginManifest) -> bool {
    matches!(
        &manifest.load,
        DynamicPluginManifestLoad::Worker(load)
            if load.runtime == Some(WorkerRuntime::Python)
    )
}

fn environment_python_path(environment: &Path) -> PathBuf {
    if cfg!(windows) {
        environment.join("Scripts").join("python.exe")
    } else {
        environment.join("bin").join("python")
    }
}

fn managed_environment_path(state_path: &Path, plugin_id: &str) -> Result<PathBuf> {
    let state_path = absolute_path(state_path)?;
    let parent = state_path.parent().ok_or_else(|| {
        PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin lifecycle state {} has no parent directory",
            state_path.display()
        ))
    })?;
    let digest = Sha256::digest(plugin_id.trim().as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(parent.join(MANAGED_ENVIRONMENTS_DIR).join(digest))
}

fn resolve_relative_path(base: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|current| current.join(path))
            .map_err(|error| PluginHostConfigError::io("resolve path", path, error))
    }
}

fn environment_tree_digest(environment: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    let mut total = 0_u64;
    let mut entries = 0_usize;
    digest_environment_directory(
        environment,
        Path::new(""),
        &mut Vec::new(),
        &mut digest,
        &mut total,
        &mut entries,
    )?;
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn digest_environment_directory(
    directory: &Path,
    relative_directory: &Path,
    ancestors: &mut Vec<PathBuf>,
    digest: &mut Sha256,
    total: &mut u64,
    entries: &mut usize,
) -> Result<()> {
    if ancestors.len() >= MAX_ENVIRONMENT_DEPTH {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "managed Python environment exceeds the {MAX_ENVIRONMENT_DEPTH}-directory traversal depth at {}",
            directory.display()
        )));
    }
    let canonical = std::fs::canonicalize(directory).map_err(|error| {
        PluginHostConfigError::io("normalize environment directory", directory, error)
    })?;
    if ancestors.contains(&canonical) {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "managed Python environment contains a directory symlink cycle at {}",
            directory.display()
        )));
    }
    ancestors.push(canonical.clone());
    let mut children = std::fs::read_dir(&canonical)
        .map_err(|error| {
            PluginHostConfigError::io("read environment directory", &canonical, error)
        })?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| {
            PluginHostConfigError::io("read environment directory entry", &canonical, error)
        })?;
    *entries = entries.saturating_add(children.len());
    if *entries > MAX_ENVIRONMENT_FILES {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "managed Python environment exceeds the {MAX_ENVIRONMENT_FILES}-entry attestation budget at {}",
            directory.display()
        )));
    }
    children.sort_by_key(std::fs::DirEntry::file_name);
    for child in children {
        let path = child.path();
        let relative = relative_directory.join(child.file_name());
        if relative == Path::new(ENVIRONMENT_ATTESTATION_FILE)
            || path.file_name().and_then(|name| name.to_str()) == Some("__pycache__")
            || path.extension().and_then(|extension| extension.to_str()) == Some("pyc")
        {
            continue;
        }
        let source = resolve_environment_entry(&path)?;
        let metadata = std::fs::metadata(&source).map_err(|error| {
            PluginHostConfigError::io("inspect environment entry", &source, error)
        })?;
        if metadata.is_dir() {
            update_tree_digest(digest, b'd', &relative, &[]);
            digest_environment_directory(&source, &relative, ancestors, digest, total, entries)?;
        } else if metadata.is_file() {
            let bytes = read_bounded_regular_file(&source, "managed Python environment file")?;
            *total = total.saturating_add(bytes.len() as u64);
            if *total > MAX_BOUNDED_FILE_BYTES {
                return Err(PluginHostConfigError::InvalidConfig(format!(
                    "managed Python environment exceeds the {MAX_BOUNDED_FILE_BYTES}-byte attestation budget"
                )));
            }
            update_tree_digest(digest, b'f', &relative, &bytes);
        } else {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "managed Python environment entry {} must resolve to a regular file or directory",
                path.display()
            )));
        }
    }
    ancestors.pop();
    Ok(())
}

fn resolve_environment_entry(path: &Path) -> Result<PathBuf> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| PluginHostConfigError::io("inspect environment entry", path, error))?;
    if metadata.file_type().is_symlink() {
        std::fs::canonicalize(path)
            .map_err(|error| PluginHostConfigError::io("resolve environment symlink", path, error))
    } else {
        Ok(path.to_path_buf())
    }
}

fn update_tree_digest(digest: &mut Sha256, entry_type: u8, path: &Path, payload: &[u8]) {
    let path = raw_path_bytes(path);
    digest.update([entry_type]);
    digest.update((path.len() as u64).to_le_bytes());
    digest.update(&path);
    digest.update((payload.len() as u64).to_le_bytes());
    digest.update(payload);
}

#[cfg(unix)]
fn raw_path_bytes(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes().to_vec()
}

#[cfg(windows)]
fn raw_path_bytes(path: &Path) -> Vec<u8> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str()
        .encode_wide()
        .flat_map(u16::to_le_bytes)
        .collect()
}

fn verify_environment_authentication(
    source_artifact_sha256: &str,
    environment_sha256: &str,
    authentication: &str,
) -> Result<bool> {
    let Some(encoded) = authentication.strip_prefix("hmac-sha256:") else {
        return Ok(false);
    };
    let Some(tag) = decode_fixed_hex::<32>(encoded) else {
        return Ok(false);
    };
    let key = hmac::Key::new(hmac::HMAC_SHA256, &load_or_create_hmac_key()?);
    Ok(hmac::verify(
        &key,
        &environment_attestation_message(source_artifact_sha256, environment_sha256),
        &tag,
    )
    .is_ok())
}

fn environment_attestation_message(
    source_artifact_sha256: &str,
    environment_sha256: &str,
) -> Vec<u8> {
    let mut message = Vec::with_capacity(
        PYTHON_ENVIRONMENT_ATTESTATION_DOMAIN.len()
            + source_artifact_sha256.len()
            + environment_sha256.len()
            + 1,
    );
    message.extend_from_slice(PYTHON_ENVIRONMENT_ATTESTATION_DOMAIN);
    message.extend_from_slice(source_artifact_sha256.trim().as_bytes());
    message.push(0);
    message.extend_from_slice(environment_sha256.as_bytes());
    message
}

fn decode_fixed_hex<const N: usize>(encoded: &str) -> Option<[u8; N]> {
    if encoded.len() != N * 2 || !encoded.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let mut decoded = [0_u8; N];
    for (index, byte) in decoded.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(decoded)
}

fn load_or_create_hmac_key() -> Result<[u8; HMAC_KEY_BYTES]> {
    let path = nemo_relay::plugin::user_config_dir()
        .map(|directory| directory.join("bootstrap").join("fingerprint-hmac.key"))
        .ok_or_else(|| {
            PluginHostConfigError::InvalidConfig(
                "cannot determine the per-user NeMo Relay bootstrap state directory; set HOME or USERPROFILE"
                    .into(),
            )
        })?;
    let parent = path.parent().expect("bootstrap HMAC key has a parent");
    std::fs::create_dir_all(parent).map_err(|error| {
        PluginHostConfigError::io("create bootstrap state directory", parent, error)
    })?;
    #[cfg(unix)]
    std::fs::set_permissions(parent, {
        use std::os::unix::fs::PermissionsExt;
        std::fs::Permissions::from_mode(0o700)
    })
    .map_err(|error| {
        PluginHostConfigError::io("protect bootstrap state directory", parent, error)
    })?;

    #[cfg(windows)]
    protect_private_windows_path(parent).map_err(|error| {
        PluginHostConfigError::io("protect bootstrap state directory", parent, error)
    })?;

    #[cfg(windows)]
    let mut file = open_private_windows_file(&path)
        .map_err(|error| PluginHostConfigError::io("open bootstrap HMAC key", &path, error))?;
    #[cfg(not(windows))]
    let mut file = {
        let mut options = OpenOptions::new();
        options.create(true).truncate(false).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options
            .open(&path)
            .map_err(|error| PluginHostConfigError::io("open bootstrap HMAC key", &path, error))?
    };
    let deadline = Instant::now() + HMAC_LOCK_TIMEOUT;
    loop {
        match file.try_lock_exclusive() {
            Ok(()) => break,
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                thread::sleep(Duration::from_millis(25));
            }
            Err(error) => {
                return Err(PluginHostConfigError::io(
                    "lock bootstrap HMAC key",
                    &path,
                    error,
                ));
            }
        }
    }
    #[cfg(unix)]
    file.set_permissions({
        use std::os::unix::fs::PermissionsExt;
        std::fs::Permissions::from_mode(0o600)
    })
    .map_err(|error| PluginHostConfigError::io("protect bootstrap HMAC key", &path, error))?;
    let length = file
        .metadata()
        .map_err(|error| PluginHostConfigError::io("inspect bootstrap HMAC key", &path, error))?
        .len();
    if length == 0 {
        let mut key = [0_u8; HMAC_KEY_BYTES];
        SystemRandom::new().fill(&mut key).map_err(|_| {
            PluginHostConfigError::InvalidConfig("failed to generate bootstrap HMAC key".into())
        })?;
        file.write_all(&key)
            .map_err(|error| PluginHostConfigError::io("write bootstrap HMAC key", &path, error))?;
        file.sync_all()
            .map_err(|error| PluginHostConfigError::io("sync bootstrap HMAC key", &path, error))?;
        return Ok(key);
    }
    if length != HMAC_KEY_BYTES as u64 {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "bootstrap HMAC key {} has invalid length {length}; expected {HMAC_KEY_BYTES} bytes",
            path.display()
        )));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| PluginHostConfigError::io("seek bootstrap HMAC key", &path, error))?;
    let mut key = [0_u8; HMAC_KEY_BYTES];
    file.read_exact(&mut key)
        .map_err(|error| PluginHostConfigError::io("read bootstrap HMAC key", &path, error))?;
    Ok(key)
}

#[cfg(windows)]
fn open_private_windows_file(path: &Path) -> io::Result<File> {
    use windows_sys::Win32::Foundation::{GENERIC_READ, GENERIC_WRITE};
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_ALWAYS,
    };

    let file = with_private_windows_descriptor(|descriptor| {
        open_windows_file(
            path,
            descriptor,
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            OPEN_ALWAYS,
        )
    })?;
    protect_private_windows_path(path)?;
    Ok(file)
}

#[cfg(windows)]
pub(crate) fn create_private_windows_file(path: &Path) -> io::Result<File> {
    use windows_sys::Win32::Foundation::GENERIC_WRITE;
    use windows_sys::Win32::Storage::FileSystem::CREATE_NEW;

    with_private_windows_descriptor(|descriptor| {
        open_windows_file(path, descriptor, GENERIC_WRITE, 0, CREATE_NEW)
    })
}

#[cfg(windows)]
fn protect_private_windows_path(path: &Path) -> io::Result<()> {
    use windows_sys::Win32::Security::{
        DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION, SetFileSecurityW,
    };

    if !windows_path_owned_by_current_user(path)? {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("{} is not owned by the current user", path.display()),
        ));
    }
    let path_wide = windows_wide(path.as_os_str());
    with_private_windows_descriptor(|descriptor| {
        // SAFETY: The path and descriptor remain valid for the duration of the call.
        if unsafe {
            SetFileSecurityW(
                path_wide.as_ptr(),
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                descriptor,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    })?;
    if !windows_path_is_private(path)? {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "failed to verify protected owner/System access on {}",
                path.display()
            ),
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn with_private_windows_descriptor<T>(
    operation: impl FnOnce(windows_sys::Win32::Security::PSECURITY_DESCRIPTOR) -> io::Result<T>,
) -> io::Result<T> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::PSECURITY_DESCRIPTOR;

    let descriptor_sddl = windows_wide("D:P(A;;FA;;;OW)(A;;FA;;;SY)");
    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
    // SAFETY: The SDDL string is NUL-terminated and `descriptor` points to writable storage.
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            descriptor_sddl.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    let result = operation(descriptor);
    // SAFETY: The descriptor was allocated by the conversion API and is still owned here.
    unsafe { LocalFree(descriptor.cast()) };
    result
}

#[cfg(windows)]
fn windows_path_owned_by_current_user(path: &Path) -> io::Result<bool> {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::Security::{
        EqualSid, GetSecurityDescriptorOwner, GetTokenInformation, OWNER_SECURITY_INFORMATION,
        PSID, TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    let mut descriptor = read_windows_security_descriptor(path, OWNER_SECURITY_INFORMATION)?;
    let mut owner: PSID = std::ptr::null_mut();
    let mut defaulted = 0;
    // SAFETY: The descriptor and output storage remain valid for the duration of the call.
    if unsafe {
        GetSecurityDescriptorOwner(descriptor.as_mut_ptr().cast(), &mut owner, &mut defaulted)
    } == 0
        || owner.is_null()
    {
        return Err(io::Error::last_os_error());
    }

    let mut token: HANDLE = std::ptr::null_mut();
    // SAFETY: GetCurrentProcess returns a valid pseudo-handle and `token` is writable.
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let result = (|| {
        let mut required = 0;
        // SAFETY: This sizing call intentionally supplies a null output buffer.
        unsafe { GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut required) };
        if required == 0 {
            return Err(io::Error::last_os_error());
        }
        let word = std::mem::size_of::<usize>();
        let mut buffer = vec![0_usize; (required as usize).div_ceil(word)];
        // SAFETY: The aligned buffer has at least `required` writable bytes.
        if unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                buffer.as_mut_ptr().cast(),
                required,
                &mut required,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: GetTokenInformation initialized a TOKEN_USER at this aligned address.
        let user = unsafe { &*buffer.as_ptr().cast::<TOKEN_USER>() };
        // SAFETY: Both SID pointers remain valid while their backing buffers are alive.
        Ok(unsafe { EqualSid(owner, user.User.Sid) != 0 })
    })();
    // SAFETY: `token` is an owned handle returned by OpenProcessToken.
    unsafe { CloseHandle(token) };
    result
}

#[cfg(windows)]
fn windows_path_is_private(path: &Path) -> io::Result<bool> {
    use windows_sys::Win32::Security::{DACL_SECURITY_INFORMATION, OWNER_SECURITY_INFORMATION};

    if !windows_path_owned_by_current_user(path)? {
        return Ok(false);
    }
    let mut actual = read_windows_security_descriptor(
        path,
        OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
    )?;
    let actual = windows_dacl_sddl(actual.as_mut_ptr().cast())?;
    with_private_windows_descriptor(|expected| Ok(actual == windows_dacl_sddl(expected)?))
}

#[cfg(windows)]
fn windows_dacl_sddl(
    descriptor: windows_sys::Win32::Security::PSECURITY_DESCRIPTOR,
) -> io::Result<String> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Authorization::{
        ConvertSecurityDescriptorToStringSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::DACL_SECURITY_INFORMATION;

    let mut rendered = std::ptr::null_mut();
    let mut rendered_len = 0;
    // SAFETY: The descriptor is valid and both output pointers reference writable storage.
    if unsafe {
        ConvertSecurityDescriptorToStringSecurityDescriptorW(
            descriptor,
            SDDL_REVISION_1,
            DACL_SECURITY_INFORMATION,
            &mut rendered,
            &mut rendered_len,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: The API returned `rendered_len` initialized UTF-16 code units.
    let value = String::from_utf16_lossy(unsafe {
        std::slice::from_raw_parts(rendered, rendered_len as usize)
    })
    .trim_end_matches('\0')
    .to_string();
    // SAFETY: `rendered` was allocated by the conversion API above.
    unsafe { LocalFree(rendered.cast()) };
    Ok(value)
}

#[cfg(windows)]
fn read_windows_security_descriptor(
    path: &Path,
    information: windows_sys::Win32::Security::OBJECT_SECURITY_INFORMATION,
) -> io::Result<Vec<u8>> {
    use windows_sys::Win32::Security::GetFileSecurityW;

    let path = windows_wide(path.as_os_str());
    let mut required = 0;
    // SAFETY: This sizing call intentionally supplies a null output buffer.
    unsafe {
        GetFileSecurityW(
            path.as_ptr(),
            information,
            std::ptr::null_mut(),
            0,
            &mut required,
        )
    };
    if required == 0 {
        return Err(io::Error::last_os_error());
    }
    let mut descriptor = vec![0_u8; required as usize];
    // SAFETY: The NUL-terminated path and allocated output buffer remain valid for the call.
    if unsafe {
        GetFileSecurityW(
            path.as_ptr(),
            information,
            descriptor.as_mut_ptr().cast(),
            required,
            &mut required,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(descriptor)
}

#[cfg(windows)]
fn open_windows_file(
    path: &Path,
    descriptor: windows_sys::Win32::Security::PSECURITY_DESCRIPTOR,
    desired_access: u32,
    share_mode: u32,
    creation_disposition: u32,
) -> io::Result<File> {
    use std::os::windows::io::FromRawHandle;
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
    use windows_sys::Win32::Storage::FileSystem::{CreateFileW, FILE_ATTRIBUTE_NORMAL};

    let path = windows_wide(path.as_os_str());
    let attributes = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: descriptor,
        bInheritHandle: 0,
    };
    // SAFETY: The path and security descriptor remain valid; the returned handle is owned.
    let handle = unsafe {
        CreateFileW(
            path.as_ptr(),
            desired_access,
            share_mode,
            &attributes,
            creation_disposition,
            FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `handle` is a newly returned valid owned file handle.
    Ok(unsafe { File::from_raw_handle(handle) })
}

#[cfg(windows)]
fn windows_wide(value: impl AsRef<std::ffi::OsStr>) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;

    value.as_ref().encode_wide().chain(Some(0)).collect()
}
