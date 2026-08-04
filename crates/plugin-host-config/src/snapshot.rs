// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nemo_relay::plugin::Result as PluginResult;
use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationResource, DynamicPluginKind, DynamicPluginManifest,
    DynamicPluginManifestLoad, DynamicPluginStartupClass, WorkerRuntime,
};
use sha2::{Digest, Sha256};

use crate::environment::{
    ENVIRONMENT_ATTESTATION_FILE, MANAGED_ENVIRONMENTS_DIR, validate_python_entrypoint_artifact,
    verify_environment_attestation,
};
use crate::error::{PluginHostConfigError, Result};
#[cfg(target_os = "macos")]
use crate::io::read_bounded_utf8_regular_file;
use crate::io::{
    MAX_BOUNDED_FILE_BYTES, load_bounded_dynamic_plugin_manifest_bytes, read_bounded_regular_file,
};
use crate::policy::{DynamicPluginHostPolicy, evaluate_dynamic_plugin_host_policy};
use crate::trust::evaluate_dynamic_plugin_trust;

const MAX_SNAPSHOT_FILES: usize = 100_000;
const MAX_SNAPSHOT_DEPTH: usize = 128;

/// Immutable filesystem snapshot retained for one dynamic plugin's runtime lifetime.
#[derive(Debug, PartialEq, Eq)]
pub struct DynamicPluginActivationSnapshot {
    root: PathBuf,
    original_manifest_ref: String,
    identity_manifest: PathBuf,
    activation_manifest: PathBuf,
    activation_environment_ref: Option<String>,
    identity_files: HashMap<PathBuf, PathBuf>,
    closure_digest: String,
    verification_digest: String,
}

impl DynamicPluginActivationSnapshot {
    /// Creates and verifies an immutable activation snapshot.
    pub fn create(
        manifest_ref: &str,
        expected_plugin_id: &str,
        expected_kind: DynamicPluginKind,
        environment_ref: Option<&str>,
        host_policy: &DynamicPluginHostPolicy,
    ) -> Result<Arc<Self>> {
        let (mut manifest, original_manifest_ref, manifest_bytes) =
            load_bounded_dynamic_plugin_manifest_bytes(manifest_ref)?;
        if manifest.plugin.id.trim() != expected_plugin_id || manifest.plugin.kind != expected_kind
        {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin manifest identity changed before activation for '{expected_plugin_id}'"
            )));
        }
        validate_python_entrypoint_artifact(&manifest, &original_manifest_ref)
            .map_err(PluginHostConfigError::InvalidConfig)?;
        let policy = evaluate_dynamic_plugin_host_policy(host_policy, &manifest);

        let root = std::env::temp_dir().join(format!(
            "nemo-relay-plugin-snapshot-{}",
            uuid::Uuid::now_v7().simple()
        ));
        fs::create_dir(&root).map_err(|error| {
            PluginHostConfigError::io("create activation snapshot", &root, error)
        })?;
        let mut root_guard = SnapshotRootGuard(Some(root.clone()));
        #[cfg(unix)]
        fs::set_permissions(&root, {
            use std::os::unix::fs::PermissionsExt;
            fs::Permissions::from_mode(0o700)
        })
        .map_err(|error| PluginHostConfigError::io("protect activation snapshot", &root, error))?;

        let identity_manifest = root.join("identity-manifest.toml");
        fs::write(&identity_manifest, &manifest_bytes).map_err(|error| {
            PluginHostConfigError::io(
                "write activation identity manifest",
                &identity_manifest,
                error,
            )
        })?;
        let original_manifest_path = PathBuf::from(&original_manifest_ref);
        let manifest_directory = original_manifest_path.parent().ok_or_else(|| {
            PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin manifest {} has no parent directory",
                original_manifest_path.display()
            ))
        })?;
        let runtime_root = root.join("runtime");
        let mut budget = SnapshotBudget::default();
        let mut copied_files = HashMap::new();
        copy_snapshot_directory(
            manifest_directory,
            &runtime_root,
            &mut copied_files,
            &mut budget,
            false,
            &mut Vec::new(),
        )?;
        let declared_artifact = manifest
            .source
            .as_ref()
            .and_then(|source| source.artifact.as_deref())
            .map(|artifact| {
                fs::canonicalize(resolve_manifest_relative_path(&original_manifest_path, artifact))
            })
            .transpose()
            .map_err(|error| {
                PluginHostConfigError::InvalidConfig(format!(
                    "failed to normalize dynamic plugin artifact for '{expected_plugin_id}': {error}"
                ))
            })?;
        let mut identity_files = HashMap::new();

        match &mut manifest.load {
            DynamicPluginManifestLoad::RustDynamic(load) => {
                if let Some(library) = load.library.as_deref() {
                    let (logical, canonical, copied) = copy_snapshot_file(
                        &root,
                        &original_manifest_path,
                        library,
                        "library",
                        &mut copied_files,
                        &mut budget,
                    )?;
                    if declared_artifact.as_ref() != Some(&canonical) {
                        return Err(PluginHostConfigError::InvalidConfig(format!(
                            "native dynamic plugin '{expected_plugin_id}' must declare its load.library as the integrity-checked source.artifact"
                        )));
                    }
                    identity_files
                        .entry(logical)
                        .or_insert_with(|| copied.clone());
                    load.library = Some(copied.to_string_lossy().into_owned());
                }
            }
            DynamicPluginManifestLoad::Worker(load)
                if matches!(
                    load.runtime,
                    Some(WorkerRuntime::Rust | WorkerRuntime::Command)
                ) =>
            {
                if let Some(entrypoint) = load.entrypoint.as_deref() {
                    let (logical, canonical, copied) = copy_snapshot_file(
                        &root,
                        &original_manifest_path,
                        entrypoint,
                        "entrypoint",
                        &mut copied_files,
                        &mut budget,
                    )?;
                    if declared_artifact.as_ref() != Some(&canonical) {
                        return Err(PluginHostConfigError::InvalidConfig(format!(
                            "command worker dynamic plugin '{expected_plugin_id}' must declare its load.entrypoint as the integrity-checked source.artifact"
                        )));
                    }
                    identity_files
                        .entry(logical)
                        .or_insert_with(|| copied.clone());
                    load.entrypoint = Some(copied.to_string_lossy().into_owned());
                }
            }
            DynamicPluginManifestLoad::Worker(_) => {}
        }

        if let Some(source) = manifest.source.as_mut()
            && let Some(artifact) = source.artifact.as_deref()
        {
            let (logical, _, copied) = copy_snapshot_file(
                &root,
                &original_manifest_path,
                artifact,
                "artifact",
                &mut copied_files,
                &mut budget,
            )?;
            identity_files.insert(logical, copied.clone());
            source.artifact = Some(copied.to_string_lossy().into_owned());
        }
        if let Some(integrity) = manifest.integrity.as_mut()
            && let Some(signature) = integrity.signature.as_deref()
        {
            let (logical, _, copied) = copy_snapshot_file(
                &root,
                &original_manifest_path,
                signature,
                "signature",
                &mut copied_files,
                &mut budget,
            )?;
            identity_files.insert(logical, copied.clone());
            integrity.signature = Some(copied.to_string_lossy().into_owned());
        }

        let activation_environment_ref = snapshot_python_environment(
            &manifest,
            environment_ref,
            expected_plugin_id,
            &root,
            &mut copied_files,
            &mut budget,
        )?;
        let activation_manifest = runtime_root.join("relay-plugin.toml");
        let rendered = toml::to_string(&manifest).map_err(|error| {
            PluginHostConfigError::InvalidConfig(format!(
                "failed to encode dynamic plugin activation snapshot for '{expected_plugin_id}': {error}"
            ))
        })?;
        if rendered.len() as u64 > MAX_BOUNDED_FILE_BYTES {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin activation manifest for '{expected_plugin_id}' exceeds the {MAX_BOUNDED_FILE_BYTES}-byte activation snapshot budget"
            )));
        }
        fs::write(&activation_manifest, rendered).map_err(|error| {
            PluginHostConfigError::io("write activation manifest", &activation_manifest, error)
        })?;

        let trust = evaluate_dynamic_plugin_trust(
            &manifest,
            activation_manifest.to_string_lossy().as_ref(),
            &policy,
        );
        if policy.startup_class == DynamicPluginStartupClass::Required {
            if !policy.policy_satisfied {
                return Err(PluginHostConfigError::InvalidConfig(format!(
                    "dynamic plugin '{expected_plugin_id}' activation snapshot violates host policy"
                )));
            }
            if let Some(failure) = trust.failure() {
                return Err(PluginHostConfigError::InvalidConfig(
                    failure.display(expected_plugin_id).to_string(),
                ));
            }
        }

        let closure_digest = snapshot_tree_digest(&root, true)?;
        let verification_digest = snapshot_tree_digest(&root, false)?;
        protect_snapshot_tree(&root)?;
        root_guard.0 = None;
        Ok(Arc::new(Self {
            root,
            original_manifest_ref,
            identity_manifest,
            activation_manifest,
            activation_environment_ref,
            identity_files,
            closure_digest,
            verification_digest,
        }))
    }

    /// Returns the rewritten manifest consumed by the native or worker loader.
    pub fn activation_manifest_ref(&self) -> String {
        self.activation_manifest.to_string_lossy().into_owned()
    }

    /// Returns the copied lifecycle-managed Python environment, when applicable.
    pub fn activation_environment_ref(&self) -> Option<&str> {
        self.activation_environment_ref.as_deref()
    }

    /// Returns a stable digest of the snapshotted runtime closure.
    pub fn closure_digest(&self) -> &str {
        &self.closure_digest
    }

    /// Verifies that the protected snapshot has not changed since construction.
    pub fn verify_current(&self) -> Result<()> {
        let actual = snapshot_tree_digest(&self.root, false)?;
        if actual == self.verification_digest {
            Ok(())
        } else {
            Err(PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin activation snapshot {} changed before code load",
                self.root.display()
            )))
        }
    }

    /// Returns the original canonical manifest path.
    pub fn original_manifest_ref(&self) -> &str {
        &self.original_manifest_ref
    }

    /// Returns the immutable authored manifest copy used for identity reporting.
    pub fn identity_manifest(&self) -> &Path {
        &self.identity_manifest
    }

    /// Returns the snapshotted copy corresponding to one authored logical path.
    pub fn identity_file(&self, logical_path: &Path) -> Option<&Path> {
        self.identity_files.get(logical_path).map(PathBuf::as_path)
    }
}

impl DynamicPluginActivationResource for DynamicPluginActivationSnapshot {
    fn verify(&self) -> PluginResult<()> {
        self.verify_current()
            .map_err(|error| error.into_plugin_error())
    }
}

impl Drop for DynamicPluginActivationSnapshot {
    fn drop(&mut self) {
        make_snapshot_removable(&self.root);
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn snapshot_python_environment(
    manifest: &DynamicPluginManifest,
    environment_ref: Option<&str>,
    expected_plugin_id: &str,
    root: &Path,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
    budget: &mut SnapshotBudget,
) -> Result<Option<String>> {
    if !matches!(
        &manifest.load,
        DynamicPluginManifestLoad::Worker(load) if load.runtime == Some(WorkerRuntime::Python)
    ) {
        return Ok(None);
    }
    let environment = environment_ref.ok_or_else(|| {
        PluginHostConfigError::InvalidConfig(format!(
            "Python worker dynamic plugin '{expected_plugin_id}' has no managed environment"
        ))
    })?;
    let digest = trusted_source_artifact_sha256(manifest)?;
    let environment = PathBuf::from(environment);
    verify_environment_attestation(&environment, digest)?;
    let environment_name = environment.file_name().ok_or_else(|| {
        PluginHostConfigError::InvalidConfig(format!(
            "managed Python environment {} has no lifecycle environment name",
            environment.display()
        ))
    })?;
    let copied_environment = root.join(MANAGED_ENVIRONMENTS_DIR).join(environment_name);
    copy_snapshot_directory(
        &environment,
        &copied_environment,
        copied_files,
        budget,
        true,
        &mut Vec::new(),
    )?;
    verify_environment_attestation(&copied_environment, digest)?;
    #[cfg(target_os = "macos")]
    // Some relocatable CPython builds link their launcher through
    // `@rpath/libpython*.dylib`. The launcher is materialized to keep it pinned,
    // so retain that runtime library in the copied environment as well.
    snapshot_macos_python_runtime_library(&copied_environment, copied_files, budget)?;
    Ok(Some(copied_environment.to_string_lossy().into_owned()))
}

#[cfg(target_os = "macos")]
fn snapshot_macos_python_runtime_library(
    copied_environment: &Path,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
    budget: &mut SnapshotBudget,
) -> Result<()> {
    let pyvenv_config = copied_environment.join("pyvenv.cfg");
    let contents = read_bounded_utf8_regular_file(&pyvenv_config, "Python environment config")?;
    let value = |expected: &str| {
        contents.lines().find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.trim() == expected)
                .then_some(value.trim())
                .filter(|value| !value.is_empty())
        })
    };
    let Some(home) = value("home") else {
        return Ok(());
    };
    let Some(version) = value("version_info").or_else(|| value("version")) else {
        return Ok(());
    };
    let mut version = version.split('.');
    let Some(major) = version
        .next()
        .filter(|part| !part.is_empty() && part.chars().all(|value| value.is_ascii_digit()))
    else {
        return Ok(());
    };
    let Some(minor) = version
        .next()
        .filter(|part| !part.is_empty() && part.chars().all(|value| value.is_ascii_digit()))
    else {
        return Ok(());
    };
    let home = PathBuf::from(home);
    let Some(prefix) = home.parent() else {
        return Ok(());
    };
    let library_name = format!("libpython{major}.{minor}.dylib");
    let source = prefix.join("lib").join(&library_name);
    let destination = copied_environment.join("lib").join(&library_name);
    match fs::symlink_metadata(&destination) {
        Ok(metadata) if metadata.file_type().is_file() => return Ok(()),
        Ok(_) => {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "snapshotted Python runtime library {} must be a regular file",
                destination.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(PluginHostConfigError::io(
                "inspect snapshotted Python runtime library",
                &destination,
                error,
            ));
        }
    }
    let source_metadata = match fs::symlink_metadata(&source) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(PluginHostConfigError::io(
                "inspect Python runtime library",
                &source,
                error,
            ));
        }
    };
    let canonical = if source_metadata.file_type().is_symlink() {
        fs::canonicalize(&source).map_err(|error| {
            PluginHostConfigError::io("normalize Python runtime library", &source, error)
        })?
    } else {
        source.clone()
    };
    if !fs::metadata(&canonical)
        .map_err(|error| {
            PluginHostConfigError::io("inspect Python runtime library", &canonical, error)
        })?
        .is_file()
    {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "Python runtime library {} must resolve to a regular file",
            source.display()
        )));
    }
    let copied_library_directory = destination.parent().ok_or_else(|| {
        PluginHostConfigError::InvalidConfig(format!(
            "snapshotted Python runtime library {} has no parent directory",
            destination.display()
        ))
    })?;
    fs::create_dir_all(copied_library_directory).map_err(|error| {
        PluginHostConfigError::io(
            "create Python runtime library snapshot directory",
            copied_library_directory,
            error,
        )
    })?;
    budget.record_entry(&source)?;
    copy_snapshot_regular_file(&canonical, &destination, copied_files, budget)?;
    Ok(())
}

fn trusted_source_artifact_sha256(manifest: &DynamicPluginManifest) -> Result<&str> {
    manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.sha256.as_deref())
        .map(str::trim)
        .filter(|digest| !digest.is_empty())
        .ok_or_else(|| {
            PluginHostConfigError::InvalidConfig(format!(
                "Python worker dynamic plugin '{}' requires integrity.sha256 to bind its complete installed environment to the trusted source artifact",
                manifest.plugin.id
            ))
        })
}

struct SnapshotRootGuard(Option<PathBuf>);

impl Drop for SnapshotRootGuard {
    fn drop(&mut self) {
        if let Some(root) = self.0.take() {
            make_snapshot_removable(&root);
            let _ = fs::remove_dir_all(root);
        }
    }
}

#[derive(Default)]
struct SnapshotBudget {
    entries: usize,
    bytes: u64,
}

impl SnapshotBudget {
    fn record_entry(&mut self, path: &Path) -> Result<()> {
        self.entries = self.entries.saturating_add(1);
        if self.entries > MAX_SNAPSHOT_FILES {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin runtime closure exceeds the {MAX_SNAPSHOT_FILES}-entry activation snapshot budget at {}",
                path.display()
            )));
        }
        Ok(())
    }

    fn record_bytes(&mut self, path: &Path, bytes: usize) -> Result<()> {
        self.bytes = self.bytes.saturating_add(bytes as u64);
        if self.bytes > MAX_BOUNDED_FILE_BYTES {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin runtime closure exceeds the {MAX_BOUNDED_FILE_BYTES}-byte activation snapshot budget at {}",
                path.display()
            )));
        }
        Ok(())
    }
}

fn copy_snapshot_directory(
    source: &Path,
    destination: &Path,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
    budget: &mut SnapshotBudget,
    skip_python_cache: bool,
    ancestors: &mut Vec<PathBuf>,
) -> Result<()> {
    budget.record_entry(source)?;
    copy_snapshot_directory_contents(
        source,
        destination,
        copied_files,
        budget,
        skip_python_cache,
        ancestors,
    )
}

fn copy_snapshot_directory_contents(
    source: &Path,
    destination: &Path,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
    budget: &mut SnapshotBudget,
    skip_python_cache: bool,
    ancestors: &mut Vec<PathBuf>,
) -> Result<()> {
    if ancestors.len() >= MAX_SNAPSHOT_DEPTH {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin runtime closure exceeds the {MAX_SNAPSHOT_DEPTH}-directory traversal depth at {}",
            source.display()
        )));
    }
    let canonical = fs::canonicalize(source)
        .map_err(|error| PluginHostConfigError::io("normalize runtime directory", source, error))?;
    if ancestors.contains(&canonical) {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin runtime closure contains a directory symlink cycle at {}",
            source.display()
        )));
    }
    ancestors.push(canonical.clone());
    fs::create_dir_all(destination).map_err(|error| {
        PluginHostConfigError::io("create snapshot directory", destination, error)
    })?;
    let mut entries = fs::read_dir(&canonical)
        .map_err(|error| PluginHostConfigError::io("read runtime directory", &canonical, error))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| {
            PluginHostConfigError::io("read runtime directory entry", &canonical, error)
        })?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        budget.record_entry(&entry.path())?;
        copy_snapshot_entry(
            entry,
            destination,
            copied_files,
            budget,
            skip_python_cache,
            ancestors,
        )?;
    }
    ancestors.pop();
    Ok(())
}

fn copy_snapshot_entry(
    entry: fs::DirEntry,
    destination: &Path,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
    budget: &mut SnapshotBudget,
    skip_python_cache: bool,
    ancestors: &mut Vec<PathBuf>,
) -> Result<()> {
    let source_path = entry.path();
    if skip_python_cache
        && (entry.file_name() == "__pycache__"
            || source_path.extension().and_then(|value| value.to_str()) == Some("pyc"))
    {
        return Ok(());
    }
    let destination_path = destination.join(entry.file_name());
    let metadata = fs::symlink_metadata(&source_path)
        .map_err(|error| PluginHostConfigError::io("inspect runtime entry", &source_path, error))?;
    let resolved = if metadata.file_type().is_symlink() {
        fs::canonicalize(&source_path).map_err(|error| {
            PluginHostConfigError::io("resolve runtime symlink", &source_path, error)
        })?
    } else {
        source_path.clone()
    };
    let resolved_metadata = fs::metadata(&resolved).map_err(|error| {
        PluginHostConfigError::io("inspect resolved runtime entry", &resolved, error)
    })?;
    if resolved_metadata.is_dir() {
        return copy_snapshot_directory_contents(
            &resolved,
            &destination_path,
            copied_files,
            budget,
            skip_python_cache,
            ancestors,
        );
    }
    if !resolved_metadata.is_file() {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin runtime entry {} must resolve to a regular file or directory",
            source_path.display()
        )));
    }
    if preserve_python_launcher(
        &source_path,
        &destination_path,
        &resolved,
        &metadata,
        copied_files,
    )? {
        return Ok(());
    }
    copy_snapshot_regular_file(&resolved, &destination_path, copied_files, budget)
}

#[cfg(unix)]
fn preserve_python_launcher(
    source: &Path,
    destination: &Path,
    resolved: &Path,
    metadata: &fs::Metadata,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
) -> Result<bool> {
    let is_versioned_launcher_alias = source.parent().and_then(Path::file_name)
        == Some(std::ffi::OsStr::new("bin"))
        && source
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("python3"));
    if !metadata.file_type().is_symlink() || !is_versioned_launcher_alias {
        return Ok(false);
    }
    let target = fs::read_link(source).map_err(|error| {
        PluginHostConfigError::io("read Python launcher symlink", source, error)
    })?;
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            PluginHostConfigError::io("create Python launcher directory", parent, error)
        })?;
    }
    std::os::unix::fs::symlink(target, destination).map_err(|error| {
        PluginHostConfigError::io("preserve Python launcher symlink", destination, error)
    })?;
    copied_files.insert(resolved.to_path_buf(), destination.to_path_buf());
    Ok(true)
}

#[cfg(not(unix))]
fn preserve_python_launcher(
    _source: &Path,
    _destination: &Path,
    _resolved: &Path,
    _metadata: &fs::Metadata,
    _copied_files: &mut HashMap<PathBuf, PathBuf>,
) -> Result<bool> {
    Ok(false)
}

fn copy_snapshot_file(
    root: &Path,
    manifest_path: &Path,
    reference: &str,
    label: &str,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
    budget: &mut SnapshotBudget,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let logical = resolve_manifest_relative_path(manifest_path, reference);
    let canonical = fs::canonicalize(&logical).map_err(|error| {
        PluginHostConfigError::io("normalize dynamic plugin artifact", &logical, error)
    })?;
    if let Some(copied) = copied_files.get(&canonical) {
        return Ok((logical, canonical, copied.clone()));
    }
    let external = root.join(format!("external-{label}"));
    if matches!(label, "library" | "entrypoint") {
        let parent = canonical.parent().ok_or_else(|| {
            PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin {label} {} has no parent directory",
                canonical.display()
            ))
        })?;
        copy_snapshot_directory(
            parent,
            &external,
            copied_files,
            budget,
            false,
            &mut Vec::new(),
        )?;
    } else {
        fs::create_dir_all(&external).map_err(|error| {
            PluginHostConfigError::io("create external snapshot directory", &external, error)
        })?;
        let destination = external.join(canonical.file_name().unwrap_or_default());
        copy_snapshot_regular_file(&canonical, &destination, copied_files, budget)?;
    }
    let copied = copied_files.get(&canonical).cloned().ok_or_else(|| {
        PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin {label} {} was not included in its activation snapshot",
            canonical.display()
        ))
    })?;
    Ok((logical, canonical, copied))
}

fn copy_snapshot_regular_file(
    source: &Path,
    destination: &Path,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
    budget: &mut SnapshotBudget,
) -> Result<()> {
    let bytes = read_bounded_regular_file(source, "dynamic plugin runtime file")?;
    budget.record_bytes(source, bytes.len())?;
    fs::write(destination, bytes)
        .map_err(|error| PluginHostConfigError::io("write snapshot file", destination, error))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(source)
            .map_err(|error| PluginHostConfigError::io("inspect snapshot source", source, error))?
            .permissions()
            .mode();
        fs::set_permissions(destination, fs::Permissions::from_mode(mode)).map_err(|error| {
            PluginHostConfigError::io("preserve snapshot permissions", destination, error)
        })?;
    }
    copied_files.insert(source.to_path_buf(), destination.to_path_buf());
    Ok(())
}

fn resolve_manifest_relative_path(manifest_path: &Path, reference: &str) -> PathBuf {
    let path = PathBuf::from(reference);
    if path.is_absolute() {
        path
    } else {
        manifest_path
            .parent()
            .map(|parent| parent.join(&path))
            .unwrap_or(path)
    }
}

fn snapshot_tree_digest(root: &Path, stable_identity: bool) -> Result<String> {
    let mut files = Vec::new();
    collect_snapshot_files(root, root, &mut files, 0, &mut 0)?;
    files.sort();
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    for relative in files {
        if stable_identity {
            let activation_manifest = Path::new("runtime").join("relay-plugin.toml");
            let python_environment_content = relative.starts_with(MANAGED_ENVIRONMENTS_DIR)
                && relative.file_name() != Some(std::ffi::OsStr::new(ENVIRONMENT_ATTESTATION_FILE));
            if relative == activation_manifest || python_environment_content {
                continue;
            }
        }
        let path = root.join(&relative);
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| PluginHostConfigError::io("inspect snapshot entry", &path, error))?;
        let (kind, payload) = if metadata.file_type().is_symlink() {
            let target = fs::read_link(&path).map_err(|error| {
                PluginHostConfigError::io("read snapshot symlink", &path, error)
            })?;
            (1_u8, target.as_os_str().as_encoded_bytes().to_vec())
        } else {
            (
                0_u8,
                read_bounded_regular_file(&path, "dynamic plugin activation snapshot file")?,
            )
        };
        bytes = bytes.saturating_add(payload.len() as u64);
        if bytes > MAX_BOUNDED_FILE_BYTES {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin activation snapshot exceeds the {MAX_BOUNDED_FILE_BYTES}-byte verification budget"
            )));
        }
        let relative_bytes = relative.as_os_str().as_encoded_bytes();
        digest.update([kind]);
        digest.update((relative_bytes.len() as u64).to_le_bytes());
        digest.update(relative_bytes);
        digest.update((payload.len() as u64).to_le_bytes());
        digest.update(payload);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn collect_snapshot_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<PathBuf>,
    depth: usize,
    entries: &mut usize,
) -> Result<()> {
    if depth >= MAX_SNAPSHOT_DEPTH {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin activation snapshot exceeds the {MAX_SNAPSHOT_DEPTH}-directory traversal depth at {}",
            directory.display()
        )));
    }
    for entry in fs::read_dir(directory)
        .map_err(|error| PluginHostConfigError::io("read activation snapshot", directory, error))?
    {
        *entries = entries.saturating_add(1);
        if *entries > MAX_SNAPSHOT_FILES {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin activation snapshot exceeds the {MAX_SNAPSHOT_FILES}-entry verification budget at {}",
                directory.display()
            )));
        }
        let path = entry
            .map_err(|error| {
                PluginHostConfigError::io("read activation snapshot entry", directory, error)
            })?
            .path();
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            PluginHostConfigError::io("inspect activation snapshot entry", &path, error)
        })?;
        if metadata.is_dir() {
            collect_snapshot_files(root, &path, files, depth + 1, entries)?;
        } else {
            files.push(
                path.strip_prefix(root)
                    .map_err(|error| PluginHostConfigError::InvalidConfig(error.to_string()))?
                    .to_path_buf(),
            );
        }
    }
    Ok(())
}

#[cfg(unix)]
fn protect_snapshot_tree(root: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    for entry in fs::read_dir(root)
        .map_err(|error| PluginHostConfigError::io("read activation snapshot", root, error))?
    {
        let path = entry
            .map_err(|error| {
                PluginHostConfigError::io("read activation snapshot entry", root, error)
            })?
            .path();
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            PluginHostConfigError::io("inspect activation snapshot entry", &path, error)
        })?;
        if metadata.is_dir() {
            protect_snapshot_tree(&path)?;
        } else if !metadata.file_type().is_symlink() {
            let mode = metadata.permissions().mode() & !0o222;
            fs::set_permissions(&path, fs::Permissions::from_mode(mode)).map_err(|error| {
                PluginHostConfigError::io("protect activation snapshot entry", &path, error)
            })?;
        }
    }
    fs::set_permissions(root, fs::Permissions::from_mode(0o500)).map_err(|error| {
        PluginHostConfigError::io("protect activation snapshot directory", root, error)
    })
}

#[cfg(windows)]
fn protect_snapshot_tree(root: &Path) -> Result<()> {
    for entry in fs::read_dir(root)
        .map_err(|error| PluginHostConfigError::io("read activation snapshot", root, error))?
    {
        let path = entry
            .map_err(|error| {
                PluginHostConfigError::io("read activation snapshot entry", root, error)
            })?
            .path();
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            PluginHostConfigError::io("inspect activation snapshot entry", &path, error)
        })?;
        if metadata.is_dir() {
            protect_snapshot_tree(&path)?;
        } else if !metadata.file_type().is_symlink() {
            let mut permissions = metadata.permissions();
            permissions.set_readonly(true);
            fs::set_permissions(&path, permissions).map_err(|error| {
                PluginHostConfigError::io("protect activation snapshot entry", &path, error)
            })?;
        }
    }
    Ok(())
}

fn make_snapshot_removable(root: &Path) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(root, fs::Permissions::from_mode(0o700));
    }
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            make_snapshot_removable(&path);
        } else {
            #[cfg(windows)]
            if !metadata.file_type().is_symlink() {
                let mut permissions = metadata.permissions();
                permissions.set_readonly(false);
                let _ = fs::set_permissions(&path, permissions);
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/snapshot.rs"]
mod tests;
