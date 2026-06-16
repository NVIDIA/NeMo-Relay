// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use super::{
    DynamicPluginFailure, DynamicPluginId, DynamicPluginManifest, DynamicPluginMetadata,
    DynamicPluginRecord, DynamicPluginRuntimeStatus, DynamicPluginValidationStatus,
    bump_generation, stamp_creation_metadata, touch_metadata,
};
use crate::plugin::{PluginError, Result};

/// In-memory dynamic plugin registry used by the control plane.
#[derive(Debug, Default)]
pub struct DynamicPluginRegistry {
    records: BTreeMap<DynamicPluginId, DynamicPluginRecord>,
}

impl DynamicPluginRegistry {
    /// Creates an empty dynamic plugin registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the registered record for `plugin_id`, if present.
    pub fn get(&self, plugin_id: &str) -> Option<&DynamicPluginRecord> {
        self.records.get(plugin_id)
    }

    /// Lists records, hiding tombstones unless requested.
    pub fn list(&self, include_tombstoned: bool) -> Vec<&DynamicPluginRecord> {
        self.records
            .values()
            .filter(|record| include_tombstoned || !record.is_tombstoned())
            .collect()
    }

    /// Adds a new dynamic plugin record.
    ///
    /// This is a trusted internal control-plane API. Callers that start from an
    /// authored `relay-plugin.toml` manifest should prefer [`Self::add_manifest`]
    /// so the manifest contract is enforced before record creation.
    pub fn add(&mut self, mut record: DynamicPluginRecord) -> Result<&DynamicPluginRecord> {
        validate_record_shape(&record)?;

        let plugin_id = record.metadata.id.clone();
        record.spec.present = true;

        if let Some(existing) = self.records.get(&plugin_id) {
            if !existing.is_tombstoned() {
                return Err(PluginError::Conflict(format!(
                    "dynamic plugin '{plugin_id}' is already registered"
                )));
            }

            inherit_tombstoned_lineage(&mut record.metadata, &existing.metadata);
        }

        stamp_creation_metadata(&mut record.metadata);
        touch_metadata(&mut record.metadata);

        self.records.insert(plugin_id.clone(), record);
        Ok(self
            .records
            .get(&plugin_id)
            .expect("dynamic plugin record must exist immediately after insert"))
    }

    /// Validates an authored manifest and registers the resulting dynamic plugin record.
    pub fn add_manifest(
        &mut self,
        manifest: DynamicPluginManifest,
        manifest_ref: Option<String>,
    ) -> Result<&DynamicPluginRecord> {
        let record = manifest.into_record(manifest_ref)?;
        self.add(record)
    }

    /// Marks the plugin enabled in desired state.
    pub fn enable(&mut self, plugin_id: &str) -> Result<bool> {
        let record = self.lookup_mut(plugin_id)?;
        ensure_live_record(record, plugin_id)?;
        if record.spec.enabled {
            return Ok(false);
        }
        record.spec.enabled = true;
        bump_generation(record);
        Ok(true)
    }

    /// Marks the plugin disabled in desired state.
    pub fn disable(&mut self, plugin_id: &str) -> Result<bool> {
        let record = self.lookup_mut(plugin_id)?;
        ensure_live_record(record, plugin_id)?;
        if !record.spec.enabled {
            return Ok(false);
        }
        record.spec.enabled = false;
        bump_generation(record);
        Ok(true)
    }

    /// Tombstones the plugin record and disables desired runtime realization.
    pub fn remove(&mut self, plugin_id: &str) -> Result<bool> {
        let record = self.lookup_mut(plugin_id)?;
        if record.is_tombstoned() {
            return Ok(false);
        }
        record.spec.present = false;
        record.spec.enabled = false;
        bump_generation(record);
        Ok(true)
    }

    /// Replaces the current validation status without mutating desired state.
    pub fn update_validation_status(
        &mut self,
        plugin_id: &str,
        mut validation: DynamicPluginValidationStatus,
    ) -> Result<()> {
        validation.checked_at = Some(super::current_timestamp());
        let record = self.lookup_mut(plugin_id)?;
        record.status.validation = validation;
        Ok(())
    }

    /// Replaces the current runtime status without mutating desired state.
    pub fn update_runtime_status(
        &mut self,
        plugin_id: &str,
        mut runtime: DynamicPluginRuntimeStatus,
    ) -> Result<()> {
        runtime.updated_at = Some(super::current_timestamp());
        let record = self.lookup_mut(plugin_id)?;
        record.status.runtime = runtime;
        Ok(())
    }

    /// Records the most recent dynamic-plugin failure summary.
    pub fn update_last_error(
        &mut self,
        plugin_id: &str,
        last_error: Option<DynamicPluginFailure>,
    ) -> Result<()> {
        let record = self.lookup_mut(plugin_id)?;
        record.status.last_error = last_error;
        Ok(())
    }

    fn lookup_mut(&mut self, plugin_id: &str) -> Result<&mut DynamicPluginRecord> {
        self.records.get_mut(plugin_id).ok_or_else(|| {
            PluginError::NotFound(format!("dynamic plugin '{plugin_id}' is not registered"))
        })
    }
}

fn ensure_live_record(record: &DynamicPluginRecord, plugin_id: &str) -> Result<()> {
    if record.is_tombstoned() {
        return Err(PluginError::Conflict(format!(
            "dynamic plugin '{plugin_id}' has been removed"
        )));
    }
    Ok(())
}

fn inherit_tombstoned_lineage(
    metadata: &mut DynamicPluginMetadata,
    existing: &DynamicPluginMetadata,
) {
    let next_generation = existing.generation.saturating_add(1);
    if metadata.created_at.is_none() {
        metadata.created_at = existing.created_at.clone();
    }
    metadata.generation = next_generation;
}

fn validate_record_shape(record: &DynamicPluginRecord) -> Result<()> {
    if record.metadata.id.trim().is_empty() {
        return Err(PluginError::InvalidConfig(
            "dynamic plugin id must not be empty".into(),
        ));
    }

    let has_native = record
        .capabilities
        .contains(&super::DynamicPluginCapability::PluginNative);
    let has_worker = record
        .capabilities
        .contains(&super::DynamicPluginCapability::PluginWorker);

    match record.metadata.kind {
        super::DynamicPluginKind::RustDynamic => {
            if !has_native || has_worker {
                return Err(PluginError::InvalidConfig(
                    "dynamic rust_dynamic record has invalid capability shape".into(),
                ));
            }
            if record.compatibility.native_api.is_none()
                || record.compatibility.worker_protocol.is_some()
            {
                return Err(PluginError::InvalidConfig(
                    "dynamic rust_dynamic record has invalid compatibility shape".into(),
                ));
            }
            if record
                .load
                .library
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
                || record
                    .load
                    .symbol
                    .as_deref()
                    .is_none_or(|value| value.trim().is_empty())
                || record.load.worker_runtime.is_some()
                || record.load.entrypoint.is_some()
            {
                return Err(PluginError::InvalidConfig(
                    "dynamic rust_dynamic record has invalid load shape".into(),
                ));
            }
        }
        super::DynamicPluginKind::Worker => {
            if !has_worker || has_native {
                return Err(PluginError::InvalidConfig(
                    "dynamic worker record has invalid capability shape".into(),
                ));
            }
            if record.compatibility.worker_protocol.is_none()
                || record.compatibility.native_api.is_some()
            {
                return Err(PluginError::InvalidConfig(
                    "dynamic worker record has invalid compatibility shape".into(),
                ));
            }
            if record
                .load
                .entrypoint
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
                || record.load.worker_runtime.is_none()
                || record.load.library.is_some()
                || record.load.symbol.is_some()
            {
                return Err(PluginError::InvalidConfig(
                    "dynamic worker record has invalid load shape".into(),
                ));
            }
        }
    }

    Ok(())
}
