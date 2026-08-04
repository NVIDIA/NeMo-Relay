// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared lifecycle-registry persistence and transaction locking.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};

use fs2::FileExt;
use nemo_relay::plugin::dynamic::{DynamicPluginRecord, DynamicPluginRegistry};
use serde::{Deserialize, Serialize};

use crate::error::{PluginHostConfigError, Result};
use crate::io::read_bounded_utf8_regular_file;

pub(crate) const DYNAMIC_PLUGIN_STATE_FILENAME: &str = ".dynamic-plugins.json";
const DYNAMIC_PLUGIN_STATE_LOCK_FILENAME: &str = ".dynamic-plugins.lock";
const DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Default, Serialize, Deserialize)]
struct PersistedDynamicPluginRegistry {
    #[serde(default = "default_state_schema_version")]
    schema_version: u32,
    #[serde(default)]
    records: Vec<DynamicPluginRecord>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    declaration_sources: BTreeMap<String, String>,
}

const fn default_state_schema_version() -> u32 {
    DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION
}

/// Exclusive transaction lock for one sibling dynamic-plugin lifecycle registry.
///
/// Callers must hold this guard continuously from the registry read through its atomic save.
#[doc(hidden)]
pub struct LifecycleStateLock {
    state_path: PathBuf,
    lock_path: PathBuf,
    file: File,
}

/// In-memory lifecycle state, including internal physical declaration ownership.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct DynamicPluginLifecycleState {
    registry: DynamicPluginRegistry,
    declaration_sources: BTreeMap<String, String>,
}

impl DynamicPluginLifecycleState {
    /// Creates lifecycle state around a registry with no declaration ownership metadata.
    pub fn new(registry: DynamicPluginRegistry) -> Self {
        Self {
            registry,
            declaration_sources: BTreeMap::new(),
        }
    }

    /// Returns the source that owns `plugin_id`, if ownership has been recorded.
    pub fn declaration_source(&self, plugin_id: &str) -> Option<&str> {
        self.declaration_sources.get(plugin_id).map(String::as_str)
    }

    /// Records the canonical physical source that owns `plugin_id`.
    pub fn set_declaration_source(&mut self, plugin_id: &str, source: String) -> Result<()> {
        if self.registry.get(plugin_id).is_none() {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "cannot assign declaration ownership for unknown dynamic plugin '{plugin_id}'"
            )));
        }
        self.declaration_sources
            .insert(plugin_id.to_owned(), source);
        Ok(())
    }

    /// Clears physical declaration ownership for `plugin_id`.
    pub fn clear_declaration_source(&mut self, plugin_id: &str) {
        self.declaration_sources.remove(plugin_id);
    }

    /// Replaces the record registry and drops ownership for records no longer present.
    pub fn replace_registry(&mut self, registry: DynamicPluginRegistry) {
        self.registry = registry;
        self.retain_live_declaration_sources();
    }

    fn retain_live_declaration_sources(&mut self) {
        self.declaration_sources
            .retain(|plugin_id, _| self.registry.get(plugin_id).is_some());
    }

    fn into_registry(self) -> DynamicPluginRegistry {
        self.registry
    }
}

impl Deref for DynamicPluginLifecycleState {
    type Target = DynamicPluginRegistry;

    fn deref(&self) -> &Self::Target {
        &self.registry
    }
}

impl DerefMut for DynamicPluginLifecycleState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.registry
    }
}

impl std::fmt::Debug for LifecycleStateLock {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LifecycleStateLock")
            .field("state_path", &self.state_path)
            .field("lock_path", &self.lock_path)
            .finish_non_exhaustive()
    }
}

impl Drop for LifecycleStateLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

/// Returns the lifecycle-state file adjacent to a physical `plugins.toml` source.
#[doc(hidden)]
pub fn sibling_lifecycle_state_path(plugins_toml_path: &Path) -> PathBuf {
    plugins_toml_path
        .parent()
        .map(|parent| parent.join(DYNAMIC_PLUGIN_STATE_FILENAME))
        .unwrap_or_else(|| PathBuf::from(DYNAMIC_PLUGIN_STATE_FILENAME))
}

/// Pins a plugin configuration path to its portable physical location.
///
/// Existing files resolve through symlinks. For a not-yet-created file, the nearest existing
/// parent is canonicalized so the CLI control plane and embedded runtime choose the same sibling
/// lifecycle registry after creation. On Windows, canonical paths use the legacy representation
/// whenever it is unambiguous so child runtimes do not receive a verbatim path they cannot use.
#[doc(hidden)]
pub fn pin_plugin_config_path(path: &Path) -> Result<PathBuf> {
    match dunce::canonicalize(path) {
        Ok(path) => return Ok(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "failed to normalize plugin configuration file {}: {error}",
                path.display()
            )));
        }
    }

    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| {
                PluginHostConfigError::io("resolve plugin configuration path", path, error)
            })?
            .join(path)
    };
    let mut unresolved_components = Vec::new();
    let mut candidate = absolute.as_path();
    loop {
        match std::fs::canonicalize(candidate) {
            Ok(mut pinned) => {
                for component in unresolved_components.iter().rev() {
                    pinned.push(component);
                }
                return Ok(dunce::simplified(&pinned).to_path_buf());
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let Some(file_name) = candidate.file_name() else {
                    return Err(PluginHostConfigError::InvalidConfig(format!(
                        "plugin configuration path {} does not name a file",
                        path.display()
                    )));
                };
                unresolved_components.push(file_name.to_os_string());
                candidate = candidate.parent().ok_or_else(|| {
                    PluginHostConfigError::InvalidConfig(format!(
                        "plugin configuration path {} has no existing ancestor",
                        path.display()
                    ))
                })?;
            }
            Err(error) => {
                return Err(PluginHostConfigError::InvalidConfig(format!(
                    "failed to normalize plugin configuration file {}: {error}",
                    path.display()
                )));
            }
        }
    }
}

/// Acquires the stable sibling lock used for lifecycle read-modify-write transactions.
#[doc(hidden)]
pub fn lock_lifecycle_state(state_path: &Path) -> Result<LifecycleStateLock> {
    let parent = state_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&parent).map_err(|error| {
        PluginHostConfigError::io("create lifecycle state directory", &parent, error)
    })?;
    let lock_path = parent.join(DYNAMIC_PLUGIN_STATE_LOCK_FILENAME);
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(&lock_path).map_err(|error| {
        PluginHostConfigError::io("open lifecycle state lock", &lock_path, error)
    })?;
    FileExt::lock_exclusive(&file)
        .map_err(|error| PluginHostConfigError::io("lock lifecycle state", &lock_path, error))?;
    Ok(LifecycleStateLock {
        state_path: state_path.to_path_buf(),
        lock_path,
        file,
    })
}

/// Reads a lifecycle registry while its sibling transaction lock is held.
#[doc(hidden)]
pub fn read_locked_lifecycle_registry(lock: &LifecycleStateLock) -> Result<DynamicPluginRegistry> {
    Ok(read_locked_lifecycle_state(lock)?.into_registry())
}

/// Reads lifecycle records and internal declaration ownership while its lock is held.
#[doc(hidden)]
pub fn read_locked_lifecycle_state(
    lock: &LifecycleStateLock,
) -> Result<DynamicPluginLifecycleState> {
    read_lifecycle_state(&lock.state_path)
}

/// Reads a lifecycle registry without acquiring its sibling mutation lock.
///
/// Lifecycle writers replace the complete JSON document atomically, so read-only callers can
/// safely inspect the current durable snapshot without requiring write access to its directory.
#[doc(hidden)]
pub fn read_lifecycle_registry(state_path: &Path) -> Result<DynamicPluginRegistry> {
    Ok(read_lifecycle_state(state_path)?.into_registry())
}

/// Reads lifecycle records and internal declaration ownership without acquiring a mutation lock.
#[doc(hidden)]
pub fn read_lifecycle_state(state_path: &Path) -> Result<DynamicPluginLifecycleState> {
    let raw = match read_bounded_utf8_regular_file(state_path, "dynamic plugin lifecycle state") {
        Ok(raw) => raw,
        Err(PluginHostConfigError::Io { source, .. })
            if source.kind() == std::io::ErrorKind::NotFound =>
        {
            return Ok(DynamicPluginLifecycleState::default());
        }
        Err(error) => return Err(error),
    };
    let state: PersistedDynamicPluginRegistry = serde_json::from_str(&raw).map_err(|error| {
        PluginHostConfigError::json_parse("dynamic plugin registry state", state_path, &error)
    })?;
    if state.schema_version != DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "unsupported dynamic plugin registry schema_version {} in {}; expected {}",
            state.schema_version,
            state_path.display(),
            DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION
        )));
    }
    let registry = DynamicPluginRegistry::from_records(state.records)?;
    let mut lifecycle_state = DynamicPluginLifecycleState {
        registry,
        declaration_sources: state.declaration_sources,
    };
    lifecycle_state.retain_live_declaration_sources();
    Ok(lifecycle_state)
}

/// Atomically saves a lifecycle registry while its sibling transaction lock is held.
#[doc(hidden)]
pub fn save_locked_lifecycle_registry(
    lock: &LifecycleStateLock,
    registry: &DynamicPluginRegistry,
) -> Result<()> {
    let mut state = read_locked_lifecycle_state(lock)?;
    state.replace_registry(DynamicPluginRegistry::from_records(
        registry.cloned_records(true),
    )?);
    save_locked_lifecycle_state(lock, &state)
}

/// Atomically saves lifecycle records and internal declaration ownership while locked.
#[doc(hidden)]
pub fn save_locked_lifecycle_state(
    lock: &LifecycleStateLock,
    state: &DynamicPluginLifecycleState,
) -> Result<()> {
    let declaration_sources = state
        .declaration_sources
        .iter()
        .filter(|(plugin_id, _)| state.registry.get(plugin_id).is_some())
        .map(|(plugin_id, source)| (plugin_id.clone(), source.clone()))
        .collect();
    let mut rendered = serde_json::to_vec_pretty(&PersistedDynamicPluginRegistry {
        schema_version: DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION,
        records: state.registry.cloned_records(true),
        declaration_sources,
    })?;
    rendered.push(b'\n');
    let parent = lock
        .state_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let (temp_path, mut file) = create_lifecycle_state_temp(
        &parent,
        lock.state_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("dynamic-plugins"),
    )?;
    let write_result = (|| -> Result<()> {
        file.write_all(&rendered).map_err(|error| {
            PluginHostConfigError::io("write lifecycle state", &temp_path, error)
        })?;
        file.sync_all().map_err(|error| {
            PluginHostConfigError::io("sync lifecycle state", &temp_path, error)
        })?;
        drop(file);
        replace_lifecycle_state(&temp_path, &lock.state_path)?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    write_result
}

fn create_lifecycle_state_temp(parent: &Path, name: &str) -> Result<(PathBuf, File)> {
    for _ in 0..16 {
        let path = parent.join(format!(".{name}.{}.tmp", uuid::Uuid::now_v7().simple()));
        #[cfg(windows)]
        let opened = crate::environment::create_private_windows_file(&path);
        #[cfg(not(windows))]
        let opened = {
            let mut options = OpenOptions::new();
            options.create_new(true).write(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            options.open(&path)
        };
        match opened {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(PluginHostConfigError::io(
                    "create lifecycle state",
                    &path,
                    error,
                ));
            }
        }
    }
    Err(PluginHostConfigError::InvalidConfig(format!(
        "could not allocate a collision-free lifecycle state temporary file in {}",
        parent.display()
    )))
}

#[cfg(unix)]
fn replace_lifecycle_state(temp: &Path, target: &Path) -> Result<()> {
    replace_lifecycle_state_with_directory_sync(temp, target, sync_lifecycle_state_directory)
}

#[cfg(unix)]
fn replace_lifecycle_state_with_directory_sync(
    temp: &Path,
    target: &Path,
    sync_directory: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    std::fs::rename(temp, target)
        .map_err(|error| PluginHostConfigError::io("replace lifecycle state", target, error))?;
    let parent = target
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    sync_directory(parent)
}

#[cfg(unix)]
fn sync_lifecycle_state_directory(directory: &Path) -> Result<()> {
    let file = File::open(directory).map_err(|error| {
        PluginHostConfigError::io("open lifecycle state directory", directory, error)
    })?;
    file.sync_all().map_err(|error| {
        PluginHostConfigError::io("sync lifecycle state directory", directory, error)
    })
}

#[cfg(all(not(unix), not(windows)))]
fn replace_lifecycle_state(temp: &Path, target: &Path) -> Result<()> {
    std::fs::rename(temp, target)
        .map_err(|error| PluginHostConfigError::io("replace lifecycle state", target, error))
}

#[cfg(windows)]
fn replace_lifecycle_state(temp: &Path, target: &Path) -> Result<()> {
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    let temp_wide = windows_wide(temp.as_os_str());
    let target_wide = windows_wide(target.as_os_str());
    // SAFETY: Both paths are NUL-terminated and remain valid for this same-directory replace.
    if unsafe {
        MoveFileExW(
            temp_wide.as_ptr(),
            target_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    } == 0
    {
        return Err(PluginHostConfigError::io(
            "replace lifecycle state",
            target,
            std::io::Error::last_os_error(),
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn windows_wide(value: impl AsRef<std::ffi::OsStr>) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;

    value.as_ref().encode_wide().chain(Some(0)).collect()
}

#[cfg(test)]
#[path = "../tests/unit/state.rs"]
mod tests;
