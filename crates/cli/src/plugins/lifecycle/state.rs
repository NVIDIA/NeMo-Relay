// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use nemo_relay::plugin::dynamic::{DynamicPluginRecord, DynamicPluginRegistry};
use serde::{Deserialize, Serialize};

use crate::config::{
    global_plugin_config_path, project_plugin_config_path, user_config_dir, user_plugin_config_path,
};
use crate::error::CliError;

use super::super::config_io::TargetScope;

const DYNAMIC_PLUGIN_STATE_FILENAME: &str = "dynamic-plugins.json";
const DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RegistryScope {
    User,
    Project,
    Global,
    Explicit,
}

impl RegistryScope {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Project => "project",
            Self::Global => "global",
            Self::Explicit => "explicit",
        }
    }
}

#[derive(Debug)]
pub(super) struct ScopedRegistry {
    pub(super) scope: RegistryScope,
    pub(super) plugins_toml_path: PathBuf,
    pub(super) state_path: PathBuf,
    pub(super) registry: DynamicPluginRegistry,
}

#[derive(Debug, Clone)]
pub(super) struct ScopedDynamicPluginRecord {
    pub(super) scope_index: usize,
    pub(super) scope: RegistryScope,
    pub(super) plugins_toml_path: PathBuf,
    pub(super) state_path: PathBuf,
    pub(super) record: DynamicPluginRecord,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PersistedDynamicPluginRegistry {
    #[serde(default = "default_state_schema_version")]
    schema_version: u32,
    #[serde(default)]
    records: Vec<DynamicPluginRecord>,
}

const fn default_state_schema_version() -> u32 {
    DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION
}

impl ScopedRegistry {
    pub(super) fn save(&self) -> Result<(), CliError> {
        let mut rendered = serde_json::to_vec_pretty(&PersistedDynamicPluginRegistry {
            schema_version: DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION,
            records: self.registry.cloned_records(true),
        })
        .map_err(|error| {
            CliError::Config(format!(
                "could not serialize dynamic plugin registry state {}: {error}",
                self.state_path.display()
            ))
        })?;
        rendered.push(b'\n');
        let parent = self
            .state_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        std::fs::create_dir_all(&parent)?;

        let temp_path = parent.join(format!(
            ".{}.{}.tmp",
            self.state_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("dynamic-plugins"),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock is after unix epoch")
                .as_nanos()
        ));

        let write_result = (|| -> Result<(), CliError> {
            let mut file = std::fs::File::create(&temp_path)?;
            file.write_all(&rendered)?;
            file.sync_all()?;
            std::fs::rename(&temp_path, &self.state_path)?;
            Ok(())
        })();

        if write_result.is_err() {
            let _ = std::fs::remove_file(&temp_path);
        }
        write_result?;
        Ok(())
    }
}

pub(super) fn load_scoped_registries(
    explicit: Option<&PathBuf>,
) -> Result<Vec<ScopedRegistry>, CliError> {
    scoped_registry_layouts(explicit)?
        .into_iter()
        .map(|(scope, plugins_toml_path, state_path)| {
            Ok(ScopedRegistry {
                scope,
                plugins_toml_path,
                registry: read_registry(&state_path)?,
                state_path,
            })
        })
        .collect()
}

pub(super) fn scoped_paths_for_add(
    scope: TargetScope,
    explicit: Option<&PathBuf>,
) -> Result<(PathBuf, PathBuf, RegistryScope), CliError> {
    if let Some(explicit) = explicit {
        let parent = explicit.parent().ok_or_else(|| {
            CliError::Config(format!(
                "explicit config path {} has no parent directory",
                explicit.display()
            ))
        })?;
        return Ok((
            parent.join("plugins.toml"),
            parent.join(DYNAMIC_PLUGIN_STATE_FILENAME),
            RegistryScope::Explicit,
        ));
    }

    let plugins_toml_path = match scope {
        TargetScope::User => user_plugin_config_path().ok_or_else(|| {
            CliError::Config(
                "cannot determine user config directory; set HOME or XDG_CONFIG_HOME".into(),
            )
        })?,
        TargetScope::Project => {
            let cwd = std::env::current_dir()?;
            project_plugin_config_path(&cwd)
        }
        TargetScope::Global => global_plugin_config_path(),
    };
    let state_path = sibling_state_path(&plugins_toml_path);
    let scope = match scope {
        TargetScope::User => RegistryScope::User,
        TargetScope::Project => RegistryScope::Project,
        TargetScope::Global => RegistryScope::Global,
    };
    Ok((plugins_toml_path, state_path, scope))
}

pub(super) fn collect_records(
    scopes: &[ScopedRegistry],
    include_tombstoned: bool,
) -> Vec<ScopedDynamicPluginRecord> {
    let mut records = Vec::new();
    for (scope_index, scope) in scopes.iter().enumerate() {
        for record in scope.registry.cloned_records(include_tombstoned) {
            records.push(ScopedDynamicPluginRecord {
                scope_index,
                scope: scope.scope,
                plugins_toml_path: scope.plugins_toml_path.clone(),
                state_path: scope.state_path.clone(),
                record,
            });
        }
    }
    records.sort_by(|left, right| left.record.metadata.id.cmp(&right.record.metadata.id));
    records
}

pub(super) fn find_record_by_id(
    scopes: &[ScopedRegistry],
    plugin_id: &str,
) -> Result<Option<ScopedDynamicPluginRecord>, CliError> {
    let mut live = Vec::new();
    let mut tombstoned = Vec::new();
    for record in collect_records(scopes, true)
        .into_iter()
        .filter(|record| record.record.metadata.id == plugin_id)
    {
        if record.record.is_tombstoned() {
            tombstoned.push(record);
        } else {
            live.push(record);
        }
    }

    if live.len() > 1 {
        return Err(CliError::Config(format!(
            "dynamic plugin '{}' is configured in multiple lifecycle scopes; inspect {}",
            plugin_id,
            live.iter()
                .map(|record| record.scope.label())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if let Some(record) = live.into_iter().next() {
        return Ok(Some(record));
    }
    if tombstoned.len() > 1 {
        return Err(CliError::Config(format!(
            "dynamic plugin '{}' has multiple tombstoned lifecycle records; inspect {}",
            plugin_id,
            tombstoned
                .iter()
                .map(|record| record.scope.label())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(tombstoned.into_iter().next())
}

fn scoped_registry_layouts(
    explicit: Option<&PathBuf>,
) -> Result<Vec<(RegistryScope, PathBuf, PathBuf)>, CliError> {
    if let Some(explicit) = explicit {
        let parent = explicit.parent().ok_or_else(|| {
            CliError::Config(format!(
                "explicit config path {} has no parent directory",
                explicit.display()
            ))
        })?;
        let plugins_toml_path = parent.join("plugins.toml");
        return Ok(vec![(
            RegistryScope::Explicit,
            plugins_toml_path.clone(),
            sibling_state_path(&plugins_toml_path),
        )]);
    }

    let mut layouts = vec![(
        RegistryScope::Global,
        global_plugin_config_path(),
        sibling_state_path(&global_plugin_config_path()),
    )];
    if let Ok(cwd) = std::env::current_dir() {
        let plugins_toml_path = project_plugin_config_path(&cwd);
        layouts.push((
            RegistryScope::Project,
            plugins_toml_path.clone(),
            sibling_state_path(&plugins_toml_path),
        ));
    }
    if let Some(user_dir) = user_config_dir() {
        let plugins_toml_path = user_dir.join("plugins.toml");
        layouts.push((
            RegistryScope::User,
            plugins_toml_path.clone(),
            sibling_state_path(&plugins_toml_path),
        ));
    }
    Ok(layouts)
}

fn read_registry(path: &Path) -> Result<DynamicPluginRegistry, CliError> {
    if !path.exists() {
        return Ok(DynamicPluginRegistry::new());
    }
    let raw = std::fs::read_to_string(path)?;
    let state: PersistedDynamicPluginRegistry = serde_json::from_str(&raw).map_err(|error| {
        CliError::Config(format!(
            "invalid dynamic plugin registry state in {}: {error}",
            path.display()
        ))
    })?;
    if state.schema_version != DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION {
        return Err(CliError::Config(format!(
            "unsupported dynamic plugin registry schema_version {} in {}; expected {}",
            state.schema_version,
            path.display(),
            DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION
        )));
    }
    DynamicPluginRegistry::from_records(state.records)
        .map_err(|error| CliError::Config(error.to_string()))
}

fn sibling_state_path(plugins_toml_path: &Path) -> PathBuf {
    plugins_toml_path
        .parent()
        .map(|parent| parent.join(DYNAMIC_PLUGIN_STATE_FILENAME))
        .unwrap_or_else(|| PathBuf::from(DYNAMIC_PLUGIN_STATE_FILENAME))
}
