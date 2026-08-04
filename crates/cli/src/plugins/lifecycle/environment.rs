// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use nemo_relay::plugin::dynamic::{
    DynamicPluginCheckState, DynamicPluginManifest, DynamicPluginManifestLoad, WorkerRuntime,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub(super) use nemo_relay_plugin_host_config::{
    ENVIRONMENT_ATTESTATION_FILE, MANAGED_ENVIRONMENTS_DIR,
};
const MAX_ENVIRONMENT_FILES: usize = 100_000;
pub(super) const MAX_ENVIRONMENT_DEPTH: usize = 128;
#[cfg(test)]
static ENVIRONMENT_TREE_DIGEST_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Deserialize, Serialize)]
struct EnvironmentAttestation {
    version: u8,
    source_artifact_sha256: String,
    environment_sha256: String,
    authentication: String,
}

pub(super) trait PythonEnvironmentCommandRunner {
    fn run(&self, program: &OsStr, args: &[OsString]) -> Result<(), String>;
}

pub(super) struct ProcessPythonEnvironmentCommandRunner;

impl PythonEnvironmentCommandRunner for ProcessPythonEnvironmentCommandRunner {
    fn run(&self, program: &OsStr, args: &[OsString]) -> Result<(), String> {
        let status = Command::new(program).args(args).status().map_err(|error| {
            format!("failed to start {}: {error}", Path::new(program).display())
        })?;
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "{} exited with status {status}",
                Path::new(program).display()
            ))
        }
    }
}

pub(super) fn is_python_worker(manifest: &DynamicPluginManifest) -> bool {
    matches!(
        &manifest.load,
        DynamicPluginManifestLoad::Worker(load)
            if load.runtime == Some(WorkerRuntime::Python)
    )
}

pub(super) fn validate_python_entrypoint_artifact(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
) -> Result<(), String> {
    nemo_relay_plugin_host_config::validate_python_entrypoint_artifact(manifest, manifest_ref)
}

pub(super) fn provision_python_environment(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    state_path: &Path,
    runner: &impl PythonEnvironmentCommandRunner,
) -> Result<Option<PathBuf>, String> {
    if !is_python_worker(manifest) {
        return Ok(None);
    }
    validate_python_entrypoint_artifact(manifest, manifest_ref)?;

    let manifest_root = manifest
        .source
        .as_ref()
        .and_then(|source| source.manifest_root.as_deref())
        .map(str::trim)
        .filter(|root| !root.is_empty())
        .ok_or_else(|| {
            "Python worker plugins added through the CLI must declare source.manifest_root"
                .to_string()
        })?;
    let manifest_path = Path::new(manifest_ref);
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let manifest_root = resolve_relative_path(manifest_dir, manifest_root);
    let manifest_root = manifest_root.canonicalize().map_err(|error| {
        format!(
            "could not resolve Python plugin source.manifest_root {}: {error}",
            manifest_root.display()
        )
    })?;
    if !manifest_root.is_dir() {
        return Err(format!(
            "Python plugin source.manifest_root {} is not a directory",
            manifest_root.display()
        ));
    }

    let environment = managed_environment_path(state_path, &manifest.plugin.id)?;
    remove_directory_if_present(&environment, "reset")?;
    let environment_parent = environment.parent().ok_or_else(|| {
        format!(
            "managed Python environment {} has no parent directory",
            environment.display()
        )
    })?;
    std::fs::create_dir_all(environment_parent).map_err(|error| {
        format!(
            "could not create managed Python environment directory {}: {error}",
            environment_parent.display()
        )
    })?;

    let base_python = configured_python_executable();
    let create_args = vec![
        OsString::from("-m"),
        OsString::from("venv"),
        environment.as_os_str().to_owned(),
    ];
    if let Err(error) = runner.run(&base_python, &create_args) {
        let _ = remove_directory_if_present(&environment, "clean up");
        return Err(format!(
            "failed to create managed Python environment {}: {error}",
            environment.display()
        ));
    }

    let environment_python = environment_python_path(&environment);
    if !environment_python.is_file() {
        let _ = remove_directory_if_present(&environment, "clean up");
        return Err(format!(
            "managed Python environment {} did not create interpreter {}",
            environment.display(),
            environment_python.display()
        ));
    }

    let install_args = vec![
        OsString::from("-m"),
        OsString::from("pip"),
        OsString::from("install"),
        manifest_root.as_os_str().to_owned(),
    ];
    if let Err(error) = runner.run(environment_python.as_os_str(), &install_args) {
        let _ = remove_directory_if_present(&environment, "clean up");
        return Err(format!(
            "failed to install Python plugin from {} into {}: {error}",
            manifest_root.display(),
            environment.display()
        ));
    }

    let source_artifact_sha256 = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.sha256.as_deref())
        .map(str::trim)
        .filter(|digest| !digest.is_empty())
        .ok_or_else(|| {
            "Python worker plugins require integrity.sha256 to bind the installed environment to the trusted source artifact"
                .to_string()
        })?;
    write_environment_attestation(&environment, source_artifact_sha256)?;

    Ok(Some(environment))
}

pub(super) fn read_environment_attestation(
    environment: &Path,
    expected_source_artifact_sha256: &str,
) -> Result<String, String> {
    nemo_relay_plugin_host_config::read_environment_attestation(
        environment,
        expected_source_artifact_sha256,
    )
    .map_err(|error| error.to_string())
}

#[cfg(test)]
pub(super) fn verify_environment_attestation(
    environment: &Path,
    expected_source_artifact_sha256: &str,
) -> Result<String, String> {
    nemo_relay_plugin_host_config::verify_environment_attestation(
        environment,
        expected_source_artifact_sha256,
    )
    .map_err(|error| error.to_string())
}

pub(super) fn write_environment_attestation(
    environment: &Path,
    source_artifact_sha256: &str,
) -> Result<(), String> {
    let digest = environment_tree_digest(environment)?;
    let path = environment.join(ENVIRONMENT_ATTESTATION_FILE);
    let authentication =
        crate::configuration::sign_python_environment_attestation(source_artifact_sha256, &digest)
            .map_err(|error| error.to_string())?;
    let mut bytes = serde_json::to_vec_pretty(&EnvironmentAttestation {
        version: 1,
        source_artifact_sha256: source_artifact_sha256.trim().to_owned(),
        environment_sha256: digest,
        authentication,
    })
    .map_err(|error| format!("failed to encode {}: {error}", path.display()))?;
    bytes.push(b'\n');
    std::fs::write(&path, bytes)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

pub(super) fn environment_tree_digest(environment: &Path) -> Result<String, String> {
    #[cfg(test)]
    ENVIRONMENT_TREE_DIGEST_CALLS.fetch_add(1, Ordering::Relaxed);
    environment_tree_digest_with_limit(environment, MAX_ENVIRONMENT_FILES)
}

fn environment_tree_digest_with_limit(
    environment: &Path,
    max_entries: usize,
) -> Result<String, String> {
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
        max_entries,
    )?;
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(test)]
pub(super) fn test_environment_tree_digest_with_entry_limit(
    environment: &Path,
    max_entries: usize,
) -> Result<String, String> {
    environment_tree_digest_with_limit(environment, max_entries)
}

#[cfg(test)]
pub(super) fn reset_environment_tree_digest_calls() {
    ENVIRONMENT_TREE_DIGEST_CALLS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(super) fn environment_tree_digest_calls() -> usize {
    ENVIRONMENT_TREE_DIGEST_CALLS.load(Ordering::Relaxed)
}

fn digest_environment_directory(
    directory: &Path,
    relative_directory: &Path,
    ancestors: &mut Vec<PathBuf>,
    digest: &mut Sha256,
    total: &mut u64,
    entries: &mut usize,
    max_entries: usize,
) -> Result<(), String> {
    if ancestors.len() >= MAX_ENVIRONMENT_DEPTH {
        return Err(format!(
            "managed Python environment exceeds the {MAX_ENVIRONMENT_DEPTH}-directory traversal depth at {}",
            directory.display()
        ));
    }
    let canonical_directory = std::fs::canonicalize(directory)
        .map_err(|error| format!("failed to normalize {}: {error}", directory.display()))?;
    if ancestors.contains(&canonical_directory) {
        return Err(format!(
            "managed Python environment contains a directory symlink cycle at {}",
            directory.display()
        ));
    }
    ancestors.push(canonical_directory.clone());
    let mut children = Vec::new();
    for child in std::fs::read_dir(directory)
        .map_err(|error| format!("failed to read {}: {error}", directory.display()))?
    {
        *entries = entries.saturating_add(1);
        if *entries > max_entries {
            return Err(format!(
                "managed Python environment exceeds the {max_entries}-entry attestation budget at {}",
                directory.display()
            ));
        }
        children.push(
            child.map_err(|error| format!("failed to read {}: {error}", directory.display()))?,
        );
    }
    children.sort_by_key(std::fs::DirEntry::file_name);
    for child in children {
        digest_environment_entry(
            child,
            relative_directory,
            ancestors,
            digest,
            total,
            entries,
            max_entries,
        )?;
    }
    ancestors.pop();
    Ok(())
}

fn digest_environment_entry(
    child: std::fs::DirEntry,
    relative_directory: &Path,
    ancestors: &mut Vec<PathBuf>,
    digest: &mut Sha256,
    total: &mut u64,
    entries: &mut usize,
    max_entries: usize,
) -> Result<(), String> {
    let path = child.path();
    let relative = relative_directory.join(child.file_name());
    if environment_entry_is_ignored(&path, &relative) {
        return Ok(());
    }
    let source = resolve_environment_entry(&path)?;
    let metadata = std::fs::metadata(&source)
        .map_err(|error| format!("failed to inspect {}: {error}", source.display()))?;
    if metadata.is_dir() {
        update_tree_digest(digest, b'd', &relative, &[]);
        return digest_environment_directory(
            &source,
            &relative,
            ancestors,
            digest,
            total,
            entries,
            max_entries,
        );
    }
    if !metadata.is_file() {
        return Err(format!(
            "managed Python environment entry {} must resolve to a regular file or directory",
            path.display()
        ));
    }
    let bytes = crate::filesystem::bounded::read_bounded_regular_file(
        &source,
        "managed Python environment file",
    )?;
    *total = total.saturating_add(bytes.len() as u64);
    if *total > crate::filesystem::bounded::MAX_BOUNDED_FILE_BYTES {
        return Err(format!(
            "managed Python environment exceeds the {}-byte attestation budget",
            crate::filesystem::bounded::MAX_BOUNDED_FILE_BYTES
        ));
    }
    update_tree_digest(digest, b'f', &relative, &bytes);
    Ok(())
}

fn environment_entry_is_ignored(path: &Path, relative: &Path) -> bool {
    relative == Path::new(ENVIRONMENT_ATTESTATION_FILE)
        || path.file_name().and_then(|name| name.to_str()) == Some("__pycache__")
        || path.extension().and_then(|extension| extension.to_str()) == Some("pyc")
}

fn resolve_environment_entry(path: &Path) -> Result<PathBuf, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        std::fs::canonicalize(path)
            .map_err(|error| format!("failed to resolve {}: {error}", path.display()))
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

pub(super) fn remove_managed_environment(
    state_path: &Path,
    plugin_id: &str,
    environment_ref: &str,
) -> Result<(), String> {
    let expected = managed_environment_path(state_path, plugin_id)?;
    let configured = absolute_path(Path::new(environment_ref))?;
    if configured != expected {
        return Err(format!(
            "refusing to delete Python environment {} because the lifecycle-managed path is {}",
            configured.display(),
            expected.display()
        ));
    }
    remove_directory_if_present(&configured, "delete")
}

pub(super) fn validate_environment_state(
    manifest: &DynamicPluginManifest,
    state_path: &Path,
    environment_ref: Option<&str>,
) -> Result<DynamicPluginCheckState, String> {
    nemo_relay_plugin_host_config::validate_environment_state(manifest, state_path, environment_ref)
        .map_err(|error| error.to_string())
}

pub(super) fn environment_python_path(environment: &Path) -> PathBuf {
    if cfg!(windows) {
        environment.join("Scripts").join("python.exe")
    } else {
        environment.join("bin").join("python")
    }
}

fn managed_environment_path(state_path: &Path, plugin_id: &str) -> Result<PathBuf, String> {
    let state_path = absolute_path(state_path)?;
    let parent = state_path.parent().ok_or_else(|| {
        format!(
            "dynamic plugin lifecycle state {} has no parent directory",
            state_path.display()
        )
    })?;
    let digest = Sha256::digest(plugin_id.trim().as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(parent.join(MANAGED_ENVIRONMENTS_DIR).join(digest))
}

fn configured_python_executable() -> OsString {
    std::env::var_os("NEMO_RELAY_PYTHON").unwrap_or_else(|| {
        if cfg!(windows) {
            OsString::from("python")
        } else {
            OsString::from("python3")
        }
    })
}

fn resolve_relative_path(base: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|current| current.join(path))
            .map_err(|error| format!("could not resolve {}: {error}", path.display()))
    }
}

fn remove_directory_if_present(path: &Path, operation: &str) -> Result<(), String> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "could not inspect managed Python environment {} before {operation}: {error}",
                path.display()
            ));
        }
    };
    if !metadata.file_type().is_dir() {
        return Err(format!(
            "refusing to {operation} managed Python environment {} because it is not a directory",
            path.display()
        ));
    }
    std::fs::remove_dir_all(path).map_err(|error| {
        format!(
            "could not {operation} managed Python environment {}: {error}",
            path.display()
        )
    })
}
