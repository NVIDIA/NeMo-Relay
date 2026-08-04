// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::PathBuf;

use nemo_relay::plugin::dynamic::{DynamicPluginRecord, DynamicPluginRegistry};
use nemo_relay_plugin_host_config::{
    DynamicPluginLifecycleState, LifecycleStateLock, lock_lifecycle_state, pin_plugin_config_path,
    read_lifecycle_state, read_locked_lifecycle_state, save_locked_lifecycle_state,
    sibling_lifecycle_state_path,
};
use serde::Serialize;
use strum::{Display, IntoStaticStr};

use crate::configuration::{
    global_plugin_config_path, project_plugin_config_path, user_plugin_config_path,
};
use crate::error::CliError;

use super::super::config_io::TargetScope;

// Internal CLI-managed lifecycle state. This file is not intended to be user-edited.
#[derive(Display, IntoStaticStr, Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub(super) enum RegistryScope {
    User,
    Project,
    Global,
    Explicit,
}

#[derive(Debug)]
pub(super) struct ScopedRegistry {
    pub(super) sources: Vec<ScopedRegistrySource>,
    pub(super) state_path: PathBuf,
    pub(super) registry: DynamicPluginLifecycleState,
    pub(super) state_lock: Option<LifecycleStateLock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ScopedRegistrySource {
    pub(super) scope: RegistryScope,
    pub(super) plugins_toml_path: PathBuf,
}

#[derive(Debug, Clone)]
pub(super) struct ScopedDynamicPluginRecord {
    pub(super) scope_index: usize,
    pub(super) scope: RegistryScope,
    pub(super) plugins_toml_path: PathBuf,
    pub(super) state_path: PathBuf,
    pub(super) record: DynamicPluginRecord,
}

impl ScopedRegistry {
    pub(super) fn save(&self) -> Result<(), CliError> {
        let lock = self.state_lock.as_ref().ok_or_else(|| {
            CliError::Config(format!(
                "dynamic plugin lifecycle state {} is not locked for mutation",
                self.state_path.display()
            ))
        })?;
        save_locked_lifecycle_state(lock, &self.registry)
            .map_err(|error| CliError::Config(error.to_string()))
    }

    pub(super) fn ensure_locked(&mut self) -> Result<(), CliError> {
        if self.state_lock.is_some() {
            return Ok(());
        }
        if !self.registry.cloned_records(true).is_empty() {
            return Err(CliError::Config(format!(
                "refusing to lock lifecycle state {} after in-memory mutation",
                self.state_path.display()
            )));
        }
        let lock = lock_lifecycle_state(&self.state_path)
            .map_err(|error| CliError::Config(error.to_string()))?;
        self.registry = read_locked_lifecycle_state(&lock)
            .map_err(|error| CliError::Config(error.to_string()))?;
        self.state_lock = Some(lock);
        Ok(())
    }

    pub(super) fn add_source(&mut self, scope: RegistryScope, plugins_toml_path: PathBuf) {
        if !self
            .sources
            .iter()
            .any(|source| source.plugins_toml_path == plugins_toml_path)
        {
            self.sources.push(ScopedRegistrySource {
                scope,
                plugins_toml_path,
            });
        }
    }

    pub(super) fn source_for_path(&self, path: &PathBuf) -> Option<&ScopedRegistrySource> {
        self.sources
            .iter()
            .find(|source| &source.plugins_toml_path == path)
    }

    pub(super) fn source_for_record(&self, plugin_id: &str) -> ScopedRegistrySource {
        if let Some(owner) = self.registry.declaration_source(plugin_id) {
            let owner = PathBuf::from(owner);
            if let Some(source) = self.source_for_path(&owner) {
                return source.clone();
            }
            return ScopedRegistrySource {
                scope: RegistryScope::Explicit,
                plugins_toml_path: owner,
            };
        }
        self.sources
            .first()
            .cloned()
            .expect("every lifecycle registry must retain at least one physical source")
    }
}

pub(super) fn load_scoped_registries(
    explicit_plugin_config: Option<&PathBuf>,
) -> Result<Vec<ScopedRegistry>, CliError> {
    let layouts = scoped_registry_layouts(explicit_plugin_config, None)?;
    layouts
        .into_iter()
        .map(|layout| {
            let registry = read_lifecycle_state(&layout.state_path)
                .map_err(|error| CliError::Config(error.to_string()))?;
            Ok(ScopedRegistry {
                sources: layout.sources,
                state_path: layout.state_path,
                registry,
                state_lock: None,
            })
        })
        .collect()
}

pub(super) fn load_scoped_registries_for_update(
    explicit_plugin_config: Option<&PathBuf>,
    mutation_target: Option<(RegistryScope, PathBuf, PathBuf)>,
) -> Result<Vec<ScopedRegistry>, CliError> {
    let force_locked_state = mutation_target
        .as_ref()
        .map(|(_, _, state_path)| state_path.clone());
    let layouts = scoped_registry_layouts(explicit_plugin_config, mutation_target)?;
    let mut lock_order = Vec::new();
    for (index, layout) in layouts.iter().enumerate() {
        let mut plugin_exists = false;
        for source in &layout.sources {
            plugin_exists |= source.plugins_toml_path.try_exists()?;
        }
        let state_exists = layout.state_path.try_exists()?;
        if plugin_exists
            || state_exists
            || force_locked_state
                .as_ref()
                .is_some_and(|forced| forced == &layout.state_path)
        {
            lock_order.push((layout.state_path.clone(), index));
        }
    }
    lock_order.sort_by(|left, right| left.0.cmp(&right.0));
    let mut locks = HashMap::new();
    for (_, index) in lock_order {
        let lock = lock_lifecycle_state(&layouts[index].state_path)
            .map_err(|error| CliError::Config(error.to_string()))?;
        locks.insert(index, lock);
    }
    let mut locked = HashMap::new();
    for (index, lock) in locks {
        let registry = read_locked_lifecycle_state(&lock)
            .map_err(|error| CliError::Config(error.to_string()))?;
        locked.insert(index, (lock, registry));
    }
    Ok(layouts
        .into_iter()
        .enumerate()
        .map(|(index, layout)| {
            let (state_lock, registry) = locked.remove(&index).map_or_else(
                || {
                    (
                        None,
                        DynamicPluginLifecycleState::new(DynamicPluginRegistry::new()),
                    )
                },
                |(lock, registry)| (Some(lock), registry),
            );
            ScopedRegistry {
                sources: layout.sources,
                state_path: layout.state_path,
                registry,
                state_lock,
            }
        })
        .collect())
}

pub(super) fn scoped_paths_for_add(
    scope: TargetScope,
    explicit_plugin_config: Option<&PathBuf>,
) -> Result<(PathBuf, PathBuf, RegistryScope), CliError> {
    if let Some(explicit_plugin_config) = explicit_plugin_config {
        let plugins_toml_path = pin_plugin_config_path(explicit_plugin_config)
            .map_err(|error| CliError::Config(error.to_string()))?;
        return Ok((
            plugins_toml_path.clone(),
            sibling_lifecycle_state_path(&plugins_toml_path),
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
    let plugins_toml_path = pin_plugin_config_path(&plugins_toml_path)
        .map_err(|error| CliError::Config(error.to_string()))?;
    let state_path = sibling_lifecycle_state_path(&plugins_toml_path);
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
            let source = scope.source_for_record(&record.metadata.id);
            records.push(ScopedDynamicPluginRecord {
                scope_index,
                scope: source.scope,
                plugins_toml_path: source.plugins_toml_path,
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
                .map(|record| record.scope.to_string())
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
                .map(|record| record.scope.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(tombstoned.into_iter().next())
}

fn scoped_registry_layouts(
    explicit_plugin_config: Option<&PathBuf>,
    mutation_target: Option<(RegistryScope, PathBuf, PathBuf)>,
) -> Result<Vec<ScopedRegistryLayout>, CliError> {
    let mut layouts = Vec::new();
    if let Some(explicit_plugin_config) = explicit_plugin_config {
        layouts.push((RegistryScope::Explicit, explicit_plugin_config.clone()));
    } else if let Some(plugins_toml_path) = user_plugin_config_path() {
        layouts.push((RegistryScope::User, plugins_toml_path));
    }

    let user_only = std::env::var("NEMO_RELAY_CONFIG_SCOPE").ok().as_deref() == Some("user");
    if !user_only && let Ok(cwd) = std::env::current_dir() {
        let plugins_toml_path = project_plugin_config_path(&cwd);
        layouts.push((RegistryScope::Project, plugins_toml_path));
    }
    let plugins_toml_path = global_plugin_config_path();
    layouts.push((RegistryScope::Global, plugins_toml_path));
    if let Some((scope, plugins_toml_path, _)) = mutation_target {
        layouts.push((scope, plugins_toml_path));
    }

    let mut grouped = Vec::<ScopedRegistryLayout>::new();
    for (scope, plugins_toml_path) in layouts {
        let plugins_toml_path = pin_plugin_config_path(&plugins_toml_path)
            .map_err(|error| CliError::Config(error.to_string()))?;
        let state_path = sibling_lifecycle_state_path(&plugins_toml_path);
        if let Some(existing) = grouped
            .iter_mut()
            .find(|existing| existing.state_path == state_path)
        {
            if !existing
                .sources
                .iter()
                .any(|source| source.plugins_toml_path == plugins_toml_path)
            {
                existing.sources.push(ScopedRegistrySource {
                    scope,
                    plugins_toml_path,
                });
            }
        } else {
            grouped.push(ScopedRegistryLayout {
                sources: vec![ScopedRegistrySource {
                    scope,
                    plugins_toml_path,
                }],
                state_path,
            });
        }
    }
    Ok(grouped)
}

#[derive(Debug)]
struct ScopedRegistryLayout {
    sources: Vec<ScopedRegistrySource>,
    state_path: PathBuf,
}
