// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::de::{self, Deserializer};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

use super::{
    DYNAMIC_PLUGIN_MANIFEST_FILENAME, DynamicPluginCapability, DynamicPluginCheckState,
    DynamicPluginCompatibility, DynamicPluginId, DynamicPluginKind, DynamicPluginLoadContract,
    DynamicPluginMetadata, DynamicPluginRecord, DynamicPluginRustLoadContract, DynamicPluginSource,
    DynamicPluginSpec, DynamicPluginStatus, DynamicPluginValidationStatus,
    DynamicPluginWorkerLoadContract, WorkerRuntime, current_timestamp,
};
use crate::plugin::{PluginError, Result};

/// Authored `relay-plugin.toml` manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginManifest {
    /// Relay plugin manifest schema version.
    pub manifest_version: u32,
    /// Plugin identity and lane declaration.
    pub plugin: DynamicPluginManifestPlugin,
    /// Relay compatibility declarations.
    pub compat: DynamicPluginManifestCompat,
    /// Default desired-state settings.
    pub defaults: DynamicPluginManifestDefaults,
    /// Required capability declarations.
    pub capabilities: DynamicPluginManifestCapabilities,
    /// Runtime load contract.
    #[cfg_attr(feature = "schema", schemars(with = "DynamicPluginManifestLoadSchema"))]
    pub load: DynamicPluginManifestLoad,
    /// Optional source-oriented author metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<DynamicPluginManifestSource>,
    /// Optional integrity/authenticity evidence references.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub integrity: Option<DynamicPluginManifestIntegrity>,
    /// Optional human-oriented description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Plugin identity block for `relay-plugin.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginManifestPlugin {
    /// Stable plugin identifier.
    pub id: DynamicPluginId,
    /// Optional display name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional plugin version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Execution lane.
    pub kind: DynamicPluginKind,
}

/// Compatibility block for `relay-plugin.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginManifestCompat {
    /// Supported Relay version or version range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay: Option<String>,
    /// Native plugin contract version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_api: Option<String>,
    /// Worker protocol version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_protocol: Option<String>,
}

/// Defaults block for authored manifests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginManifestDefaults {
    /// Explicit default desired enabled state.
    #[serde(default)]
    pub enabled: bool,
}

/// Capability declarations for authored manifests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginManifestCapabilities {
    /// Required capability set.
    pub items: Vec<DynamicPluginCapability>,
}

/// Load block for authored manifests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicPluginManifestLoad {
    /// Worker registration target.
    Worker(DynamicPluginManifestWorkerLoad),
    /// Native shared-library registration target.
    RustDynamic(DynamicPluginManifestRustDynamicLoad),
}

/// Worker lane authored load block.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginManifestWorkerLoad {
    /// Worker runtime when `kind = "worker"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<WorkerRuntime>,
    /// Worker entrypoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entrypoint: Option<String>,
}

/// Native shared-library authored load block.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginManifestRustDynamicLoad {
    /// Native library path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library: Option<String>,
    /// Native registration symbol.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
}

#[cfg(feature = "schema")]
#[derive(schemars::JsonSchema)]
#[serde(untagged)]
enum DynamicPluginManifestLoadSchema {
    Worker(DynamicPluginManifestWorkerLoad),
    RustDynamic(DynamicPluginManifestRustDynamicLoad),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct RawDynamicPluginManifestLoad {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime: Option<WorkerRuntime>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    entrypoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    library: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    symbol: Option<String>,
}

impl<'de> Deserialize<'de> for DynamicPluginManifestLoad {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawDynamicPluginManifestLoad::deserialize(deserializer)?;
        let has_worker_fields = raw.runtime.is_some() || raw.entrypoint.is_some();
        let has_native_fields = raw.library.is_some() || raw.symbol.is_some();

        match (has_worker_fields, has_native_fields) {
            (true, false) => Ok(Self::Worker(DynamicPluginManifestWorkerLoad {
                runtime: raw.runtime,
                entrypoint: raw.entrypoint,
            })),
            (false, true) => Ok(Self::RustDynamic(DynamicPluginManifestRustDynamicLoad {
                library: raw.library,
                symbol: raw.symbol,
            })),
            (true, true) => Err(de::Error::custom(
                "load must declare either worker fields (runtime, entrypoint) or rust_dynamic fields (library, symbol), not both",
            )),
            (false, false) => Err(de::Error::custom(
                "load must declare either worker fields (runtime, entrypoint) or rust_dynamic fields (library, symbol)",
            )),
        }
    }
}

impl Serialize for DynamicPluginManifestLoad {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let raw = match self {
            Self::Worker(load) => RawDynamicPluginManifestLoad {
                runtime: load.runtime,
                entrypoint: load.entrypoint.clone(),
                library: None,
                symbol: None,
            },
            Self::RustDynamic(load) => RawDynamicPluginManifestLoad {
                runtime: None,
                entrypoint: None,
                library: load.library.clone(),
                symbol: load.symbol.clone(),
            },
        };
        raw.serialize(serializer)
    }
}

/// Source block for authored manifests.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginManifestSource {
    /// Author-facing manifest root or package root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_root: Option<String>,
    /// Author-facing artifact hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
}

/// Integrity/authenticity evidence block for authored manifests.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginManifestIntegrity {
    /// Expected artifact digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Optional signature reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl DynamicPluginManifest {
    /// Parses a `relay-plugin.toml` manifest from TOML text.
    pub fn parse_toml(toml_source: &str) -> Result<Self> {
        let manifest = toml::from_str::<Self>(toml_source).map_err(|err| {
            PluginError::InvalidConfig(format!("invalid relay-plugin.toml: {err}"))
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Loads and validates a `relay-plugin.toml` manifest from a file path or directory.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<(Self, String)> {
        let path = path.as_ref();
        let manifest_path = if path.is_dir() {
            path.join(DYNAMIC_PLUGIN_MANIFEST_FILENAME)
        } else {
            path.to_path_buf()
        };

        let display_ref = manifest_path.to_string_lossy().into_owned();
        let exists = manifest_path.try_exists().map_err(|err| {
            PluginError::Internal(format!("failed to inspect '{}': {err}", display_ref))
        })?;
        if !exists {
            return Err(PluginError::NotFound(format!(
                "dynamic plugin manifest '{}' does not exist",
                display_ref
            )));
        }

        let normalized_manifest_path = fs::canonicalize(&manifest_path).map_err(|err| {
            PluginError::Internal(format!(
                "failed to normalize dynamic plugin manifest '{}': {err}",
                display_ref
            ))
        })?;
        let manifest_ref = normalized_manifest_path.to_string_lossy().into_owned();

        let contents = fs::read_to_string(&normalized_manifest_path).map_err(|err| {
            PluginError::Internal(format!(
                "failed to read dynamic plugin manifest '{}': {err}",
                manifest_ref
            ))
        })?;
        let manifest = Self::parse_toml(&contents)?;
        Ok((manifest, manifest_ref))
    }

    /// Validates the authored manifest against the v1 contract.
    pub fn validate(&self) -> Result<()> {
        if self.manifest_version != 1 {
            return Err(PluginError::InvalidConfig(format!(
                "unsupported relay-plugin.toml manifest_version {}; expected 1",
                self.manifest_version
            )));
        }
        if self.plugin.id.trim().is_empty() {
            return Err(PluginError::InvalidConfig(
                "plugin.id must not be empty".into(),
            ));
        }
        ensure_optional_string_non_empty(self.plugin.name.as_deref(), "plugin.name")?;
        ensure_optional_string_non_empty(self.plugin.version.as_deref(), "plugin.version")?;
        ensure_optional_string_non_empty(self.description.as_deref(), "description")?;
        ensure_optional_string_non_empty(
            self.source
                .as_ref()
                .and_then(|source| source.manifest_root.as_deref()),
            "source.manifest_root",
        )?;
        ensure_optional_string_non_empty(
            self.source
                .as_ref()
                .and_then(|source| source.artifact.as_deref()),
            "source.artifact",
        )?;
        ensure_optional_string_non_empty(
            self.integrity
                .as_ref()
                .and_then(|integrity| integrity.sha256.as_deref()),
            "integrity.sha256",
        )?;
        ensure_optional_string_non_empty(
            self.integrity
                .as_ref()
                .and_then(|integrity| integrity.signature.as_deref()),
            "integrity.signature",
        )?;

        required_trimmed_string(self.compat.relay.as_deref(), "compat.relay")?;
        if self.capabilities.items.is_empty() {
            return Err(PluginError::InvalidConfig(
                "capabilities.items must declare at least one capability".into(),
            ));
        }
        reject_duplicate_capabilities(&self.capabilities.items)?;

        if self.defaults.enabled {
            return Err(PluginError::InvalidConfig(
                "defaults.enabled=true is not supported for dynamic plugins; plugins are added disabled and require explicit enablement".into(),
            ));
        }

        validate_capability_shape(self.plugin.kind, &self.capabilities.items)?;
        validate_load_shape(self.plugin.kind, &self.load)?;
        validate_compat_shape(self.plugin.kind, &self.compat)?;
        Ok(())
    }

    /// Converts the authored manifest into a durable control-plane record.
    pub fn into_record(self, manifest_ref: Option<String>) -> Result<DynamicPluginRecord> {
        self.validate()?;
        let validation = self.validation_status();
        let plugin = self.plugin;
        let compat = self.compat;
        let load = self.load;

        Ok(DynamicPluginRecord {
            metadata: DynamicPluginMetadata {
                id: plugin.id.trim().to_owned(),
                name: plugin.name,
                version: plugin.version,
                kind: plugin.kind,
                generation: 0,
                created_at: None,
                updated_at: None,
            },
            capabilities: self.capabilities.items,
            source: DynamicPluginSource {
                manifest_ref,
                artifact_ref: self
                    .source
                    .as_ref()
                    .and_then(|source| source.artifact.clone()),
                environment_ref: None,
                artifact_digest: self
                    .integrity
                    .as_ref()
                    .and_then(|integrity| integrity.sha256.clone()),
            },
            spec: DynamicPluginSpec {
                present: true,
                enabled: false,
                config_ref: None,
            },
            compatibility: DynamicPluginCompatibility {
                relay: compat.relay.map(|value| value.trim().to_owned()),
                native_api: compat.native_api.map(|value| value.trim().to_owned()),
                worker_protocol: compat.worker_protocol.map(|value| value.trim().to_owned()),
            },
            load: load.into_record_load_contract(),
            status: DynamicPluginStatus {
                validation,
                ..DynamicPluginStatus::default()
            },
        })
    }

    /// Produces the initial validation status for a successfully validated manifest.
    pub fn validation_status(&self) -> DynamicPluginValidationStatus {
        DynamicPluginValidationStatus {
            manifest: DynamicPluginCheckState::Valid,
            compatibility: DynamicPluginCheckState::Unknown,
            integrity: DynamicPluginCheckState::Unknown,
            environment: DynamicPluginCheckState::Unknown,
            authenticity: DynamicPluginCheckState::Unknown,
            policy_satisfied: DynamicPluginCheckState::Unknown,
            checked_at: Some(current_timestamp()),
            message: Some("manifest validated".into()),
        }
    }
}

fn required_string<'a>(value: Option<&'a str>, field: &str) -> Result<&'a str> {
    value.ok_or_else(|| PluginError::InvalidConfig(format!("{field} is required")))
}

fn required_trimmed_string<'a>(value: Option<&'a str>, field: &str) -> Result<&'a str> {
    let value = required_string(value, field)?;
    if value.trim().is_empty() {
        return Err(PluginError::InvalidConfig(format!(
            "{field} must not be empty"
        )));
    }
    Ok(value)
}

fn ensure_optional_string_non_empty(value: Option<&str>, field: &str) -> Result<()> {
    if value.is_some_and(|value| value.trim().is_empty()) {
        return Err(PluginError::InvalidConfig(format!(
            "{field} must not be empty when provided"
        )));
    }
    Ok(())
}

fn reject_duplicate_capabilities(capabilities: &[DynamicPluginCapability]) -> Result<()> {
    let mut seen = HashSet::with_capacity(capabilities.len());
    for capability in capabilities {
        if !seen.insert(*capability) {
            return Err(PluginError::InvalidConfig(format!(
                "capabilities.items contains duplicate capability '{capability:?}'"
            )));
        }
    }
    Ok(())
}

fn validate_capability_shape(
    kind: DynamicPluginKind,
    capabilities: &[DynamicPluginCapability],
) -> Result<()> {
    let has_native = capabilities.contains(&DynamicPluginCapability::PluginNative);
    let has_worker = capabilities.contains(&DynamicPluginCapability::PluginWorker);
    match kind {
        DynamicPluginKind::RustDynamic => {
            if !has_native {
                return Err(PluginError::InvalidConfig(
                    "rust_dynamic plugins must declare capabilities.items containing plugin_native"
                        .into(),
                ));
            }
            if has_worker {
                return Err(PluginError::InvalidConfig(
                    "rust_dynamic plugins must not declare plugin_worker".into(),
                ));
            }
        }
        DynamicPluginKind::Worker => {
            if !has_worker {
                return Err(PluginError::InvalidConfig(
                    "worker plugins must declare capabilities.items containing plugin_worker"
                        .into(),
                ));
            }
            if has_native {
                return Err(PluginError::InvalidConfig(
                    "worker plugins must not declare plugin_native".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_load_shape(kind: DynamicPluginKind, load: &DynamicPluginManifestLoad) -> Result<()> {
    match (kind, load) {
        (DynamicPluginKind::RustDynamic, DynamicPluginManifestLoad::Worker(_)) => {
            return Err(PluginError::InvalidConfig(
                "rust_dynamic plugins must not declare load.runtime or load.entrypoint".into(),
            ));
        }
        (DynamicPluginKind::Worker, DynamicPluginManifestLoad::RustDynamic(_)) => {
            return Err(PluginError::InvalidConfig(
                "worker plugins must not declare load.library or load.symbol".into(),
            ));
        }
        (DynamicPluginKind::RustDynamic, DynamicPluginManifestLoad::RustDynamic(load)) => {
            required_trimmed_string(load.library.as_deref(), "load.library")?;
            required_trimmed_string(load.symbol.as_deref(), "load.symbol")?;
        }
        (DynamicPluginKind::Worker, DynamicPluginManifestLoad::Worker(load)) => {
            required_trimmed_string(load.entrypoint.as_deref(), "load.entrypoint")?;
            match load.runtime {
                Some(WorkerRuntime::Python) => {}
                None => {
                    return Err(PluginError::InvalidConfig(
                        "worker plugins must declare load.runtime".into(),
                    ));
                }
            }
        }
    }
    Ok(())
}

impl DynamicPluginManifestLoad {
    fn into_record_load_contract(self) -> DynamicPluginLoadContract {
        match self {
            Self::Worker(load) => {
                DynamicPluginLoadContract::Worker(DynamicPluginWorkerLoadContract {
                    runtime: load
                        .runtime
                        .expect("validated worker manifest must carry load.runtime"),
                    entrypoint: load
                        .entrypoint
                        .expect("validated worker manifest must carry load.entrypoint")
                        .trim()
                        .to_owned(),
                })
            }
            Self::RustDynamic(load) => {
                DynamicPluginLoadContract::RustDynamic(DynamicPluginRustLoadContract {
                    library: load
                        .library
                        .expect("validated rust_dynamic manifest must carry load.library")
                        .trim()
                        .to_owned(),
                    symbol: load
                        .symbol
                        .expect("validated rust_dynamic manifest must carry load.symbol")
                        .trim()
                        .to_owned(),
                })
            }
        }
    }
}

fn validate_compat_shape(
    kind: DynamicPluginKind,
    compat: &DynamicPluginManifestCompat,
) -> Result<()> {
    match kind {
        DynamicPluginKind::RustDynamic => {
            required_trimmed_string(compat.native_api.as_deref(), "compat.native_api")?;
            if compat.worker_protocol.is_some() {
                return Err(PluginError::InvalidConfig(
                    "rust_dynamic plugins must not declare compat.worker_protocol".into(),
                ));
            }
        }
        DynamicPluginKind::Worker => {
            required_trimmed_string(compat.worker_protocol.as_deref(), "compat.worker_protocol")?;
            if compat.native_api.is_some() {
                return Err(PluginError::InvalidConfig(
                    "worker plugins must not declare compat.native_api".into(),
                ));
            }
        }
    }
    Ok(())
}
