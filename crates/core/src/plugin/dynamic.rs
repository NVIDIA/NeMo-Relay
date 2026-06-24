// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamic plugin control-plane and registry model.
//!
//! This module owns the durable control-plane record shape for dynamic plugins.
//! Authored manifest parsing/validation and in-memory registry mutation logic
//! live in dedicated submodules so those responsibilities do not accumulate in
//! one file as the feature grows.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use strum::IntoStaticStr;

/// Canonical identifier for one dynamic plugin record.
pub type DynamicPluginId = String;

/// Canonical filename for authored Relay plugin manifests.
pub const DYNAMIC_PLUGIN_MANIFEST_FILENAME: &str = "relay-plugin.toml";

mod manifest;
mod registry;

pub use manifest::*;
pub use registry::*;

/// Plugin execution lane.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DynamicPluginKind {
    /// Trusted in-process native plugin.
    RustDynamic,
    /// Isolated worker-based plugin runtime.
    Worker,
}

/// Managed runtime identity for worker-based plugins.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum WorkerRuntime {
    /// Python worker runtime.
    Python,
}

/// Relay-enforced capability declared by a dynamic plugin.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DynamicPluginCapability {
    /// Trusted in-process native extension capability.
    PluginNative,
    /// Isolated worker-based extension capability.
    PluginWorker,
    /// Typed configuration schema contribution capability.
    ConfigSchema,
}

/// Host policy startup classification for a plugin.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DynamicPluginStartupClass {
    /// Failure is tolerated and the host may start in degraded mode.
    Optional,
    /// Failure is startup-fatal under current host policy.
    Required,
}

/// Host attestation policy mode.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DynamicPluginAttestationMode {
    /// Integrity verification only.
    IntegrityOnly,
    /// Verify signatures when present but do not require them.
    SignatureIfPresent,
    /// Require trusted signature verification.
    SignatureRequired,
}

/// High-level verification state for one validation axis.
#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash, IntoStaticStr,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum DynamicPluginCheckState {
    /// No verification result is currently known.
    #[default]
    Unknown,
    /// Verification passed.
    Valid,
    /// Verification failed.
    Invalid,
}

/// Observed runtime state for a dynamic plugin.
#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash, IntoStaticStr,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum DynamicPluginRuntimeState {
    /// Not currently active.
    #[default]
    Stopped,
    /// Activation is in progress.
    Starting,
    /// Currently active.
    Running,
    /// Activation failed or the active runtime crashed.
    Failed,
}

/// Recent failure phase for diagnostics and operator UX.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DynamicPluginFailurePhase {
    /// Failure occurred during validation.
    Validation,
    /// Failure occurred during activation or reconciliation.
    Activation,
    /// Failure occurred after activation while running.
    Runtime,
    /// Failure occurred because policy no longer permits realization.
    Policy,
}

/// Stable metadata for one durable plugin record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginMetadata {
    /// Canonical plugin identifier.
    pub id: DynamicPluginId,
    /// Optional human-friendly display label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional plugin version mirrored from packaging metadata when desired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Execution lane used by the plugin.
    pub kind: DynamicPluginKind,
    /// Monotonic desired-state generation.
    #[serde(default)]
    pub generation: u64,
    /// Creation timestamp in RFC 3339 form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Last durable record update time in RFC 3339 form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

/// Source and resolved artifact facts for a plugin.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginSource {
    /// Canonical manifest location or reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_ref: Option<String>,
    /// Resolved runtime artifact location.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_ref: Option<String>,
    /// Resolved environment location for worker-based plugins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_ref: Option<String>,
    /// Pinned artifact digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_digest: Option<String>,
}

/// Desired-state fields owned by user-facing operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginSpec {
    /// Whether the plugin should be present in desired state.
    #[serde(default = "default_present")]
    pub present: bool,
    /// Whether the plugin should be enabled in desired state.
    #[serde(default)]
    pub enabled: bool,
    /// Optional config reference controlled by higher-level config surfaces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_ref: Option<String>,
}

pub(crate) fn default_present() -> bool {
    true
}

impl Default for DynamicPluginSpec {
    fn default() -> Self {
        Self {
            present: true,
            enabled: false,
            config_ref: None,
        }
    }
}

/// Lane-specific compatibility declarations and resolved compatibility facts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DynamicPluginCompatibility {
    /// Native shared-library compatibility contract.
    RustDynamic(DynamicPluginRustCompatibility),
    /// Worker runtime compatibility contract.
    Worker(DynamicPluginWorkerCompatibility),
}

/// Compatibility contract for worker plugins.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginWorkerCompatibility {
    /// Compatible NeMo Relay version or version range.
    pub relay: String,
    /// Worker protocol version for `worker`.
    pub worker_protocol: String,
}

/// Compatibility contract for native shared libraries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginRustCompatibility {
    /// Compatible NeMo Relay version or version range.
    pub relay: String,
    /// Native host API/ABI contract version for `rust_dynamic`.
    pub native_api: String,
}

/// Runtime entry contract for the resolved plugin.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DynamicPluginLoadContract {
    /// Worker-based plugin registration target.
    Worker(DynamicPluginWorkerLoadContract),
    /// Native shared-library registration target.
    RustDynamic(DynamicPluginRustLoadContract),
}

/// Lane-specific load contract for worker plugins.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginWorkerLoadContract {
    /// Managed worker runtime identity.
    pub runtime: WorkerRuntime,
    /// Worker entrypoint or registration target.
    pub entrypoint: String,
}

/// Lane-specific load contract for native shared libraries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginRustLoadContract {
    /// Native dynamic library path.
    pub library: String,
    /// Native exported registration symbol.
    pub symbol: String,
}

/// One structured recent failure summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginFailure {
    /// Failure phase.
    pub phase: DynamicPluginFailurePhase,
    /// Machine-readable failure code.
    pub code: String,
    /// Human-readable summary.
    pub message: String,
}

/// Decomposed validation results for one plugin record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginValidationStatus {
    /// Manifest schema/result state.
    #[serde(default)]
    pub manifest: DynamicPluginCheckState,
    /// Relay/native/worker compatibility state.
    #[serde(default)]
    pub compatibility: DynamicPluginCheckState,
    /// Artifact integrity state.
    #[serde(default)]
    pub integrity: DynamicPluginCheckState,
    /// Environment/runtime readiness state.
    #[serde(default)]
    pub environment: DynamicPluginCheckState,
    /// Signature/authenticity state.
    #[serde(default)]
    pub authenticity: DynamicPluginCheckState,
    /// Whether the current host policy is satisfied.
    #[serde(default)]
    pub policy_satisfied: DynamicPluginCheckState,
    /// Most recent validation time in RFC 3339 form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_at: Option<String>,
    /// Concise operator-facing validation summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl Default for DynamicPluginValidationStatus {
    fn default() -> Self {
        Self {
            manifest: DynamicPluginCheckState::Unknown,
            compatibility: DynamicPluginCheckState::Unknown,
            integrity: DynamicPluginCheckState::Unknown,
            environment: DynamicPluginCheckState::Unknown,
            authenticity: DynamicPluginCheckState::Unknown,
            policy_satisfied: DynamicPluginCheckState::Unknown,
            checked_at: None,
            message: None,
        }
    }
}

/// Observed runtime state for one plugin record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginRuntimeStatus {
    /// Current observed runtime state.
    #[serde(default)]
    pub state: DynamicPluginRuntimeState,
    /// Desired-state generation this runtime status reflects.
    #[serde(default)]
    pub observed_generation: u64,
    /// Most recent successful start/activation time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    /// Most recent runtime-status refresh time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// Concise operator-facing runtime summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl Default for DynamicPluginRuntimeStatus {
    fn default() -> Self {
        Self {
            state: DynamicPluginRuntimeState::Stopped,
            observed_generation: 0,
            started_at: None,
            updated_at: None,
            message: None,
        }
    }
}

/// Durable observed state for a plugin record.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginStatus {
    /// Validation and policy status.
    #[serde(default)]
    pub validation: DynamicPluginValidationStatus,
    /// Runtime state observed by the control plane.
    #[serde(default)]
    pub runtime: DynamicPluginRuntimeStatus,
    /// Host policy startup classification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_class: Option<DynamicPluginStartupClass>,
    /// Effective attestation mode for this plugin under host policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attestation_mode: Option<DynamicPluginAttestationMode>,
    /// Most recent meaningful failure summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<DynamicPluginFailure>,
}

/// Durable control-plane record for a dynamic plugin.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginRecord {
    /// Stable plugin metadata.
    pub metadata: DynamicPluginMetadata,
    /// Source and artifact facts.
    #[serde(default)]
    pub source: DynamicPluginSource,
    /// Desired state.
    #[serde(default)]
    pub spec: DynamicPluginSpec,
    /// Compatibility declarations and resolved compatibility facts.
    pub compatibility: DynamicPluginCompatibility,
    /// Resolved runtime entry contract.
    pub load: DynamicPluginLoadContract,
    /// Observed state.
    #[serde(default)]
    pub status: DynamicPluginStatus,
}

impl DynamicPluginRecord {
    /// Returns `true` when the runtime has observed the current desired-state generation.
    pub fn is_reconciled(&self) -> bool {
        self.status.runtime.observed_generation == self.metadata.generation
    }

    /// Returns `true` when the record is tombstoned.
    pub fn is_tombstoned(&self) -> bool {
        !self.spec.present
    }
}

pub(crate) fn current_timestamp() -> String {
    Utc::now().to_rfc3339()
}

pub(crate) fn stamp_creation_metadata(metadata: &mut DynamicPluginMetadata) {
    if metadata.created_at.is_none() {
        metadata.created_at = Some(current_timestamp());
    }
    if metadata.updated_at.is_none() {
        metadata.updated_at = metadata.created_at.clone();
    }
}

pub(crate) fn touch_metadata(metadata: &mut DynamicPluginMetadata) {
    metadata.updated_at = Some(current_timestamp());
}

pub(crate) fn bump_generation(record: &mut DynamicPluginRecord) {
    record.metadata.generation = record.metadata.generation.saturating_add(1);
    touch_metadata(&mut record.metadata);
}

#[cfg(test)]
#[path = "../../tests/unit/plugin_dynamic_tests.rs"]
mod tests;
