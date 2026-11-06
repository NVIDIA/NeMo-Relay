// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native dynamic plugin loader and host-side ABI adapter.

#[path = "native_diagnostics.rs"]
mod native_diagnostics;
pub(super) use native_diagnostics::CheckedDiagnostics;

use super::trust::DynamicPluginLoadTrust;
use super::{
    DynamicPluginKind, DynamicPluginManifest, DynamicPluginManifestLoad,
    DynamicPluginTeardownOutcome, deregister_tracked_registrations_checked,
};
use crate::{Plugin, PluginError, deregister_plugin_registration_checked, register_plugin_tracked};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
/// Native plugin load request derived from host dynamic-plugin state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePluginLoadSpec {
    /// Expected plugin kind.
    pub plugin_id: String,
    /// Path to the authored `relay-plugin.toml`.
    pub manifest_ref: String,
}

pub(super) struct PreparedNativePlugin {
    pub(super) manifest: DynamicPluginManifest,
    pub(super) manifest_ref: String,
    pub(super) artifact_bytes: Option<Vec<u8>>,
}

impl PreparedNativePlugin {
    pub(super) fn new(
        manifest: DynamicPluginManifest,
        manifest_ref: String,
        artifact_bytes: Option<Vec<u8>>,
    ) -> crate::Result<Self> {
        if let (Some(bytes), DynamicPluginManifestLoad::RustDynamic(load)) =
            (&artifact_bytes, &manifest.load)
        {
            let library = load
                .library
                .as_deref()
                .expect("validated native load.library");
            let source = manifest
                .source
                .as_ref()
                .and_then(|source| source.artifact.as_deref())
                .expect("integrity-checked source.artifact");
            let library_path = super::resolve_dynamic_plugin_artifact_path(&manifest_ref, library);
            let source_path = super::resolve_dynamic_plugin_artifact_path(&manifest_ref, source);
            // Preserve the existing requirement that load.library also matches
            // the artifact digest when the manifest names two distinct files.
            if library_path != source_path {
                let library_bytes =
                    super::read_bounded_regular_file(&library_path, "native plugin library")
                        .map_err(PluginError::InvalidConfig)?;
                if library_bytes != *bytes {
                    return Err(PluginError::InvalidConfig(format!(
                        "native plugin library '{}' differs from the verified artifact",
                        library_path.display()
                    )));
                }
            }
        }
        Ok(Self {
            manifest,
            manifest_ref,
            artifact_bytes,
        })
    }
}

/// Owns native dynamic libraries registered into the plugin registry.
///
/// Dropping this value deregisters plugin definitions. Libraries and their staged
/// dependency closures stay resident until process exit. Component cleanup is
/// owned by the runtime registrations and must finish before clearing the host.
pub struct NativePluginActivation {
    plugins: Vec<Arc<super::native_checked::CheckedPlugin>>,
    plugin_registrations: Vec<(String, u64)>,
}

impl NativePluginActivation {
    /// Verified resident paths associated with this activation's definitions.
    pub fn resident_libraries(&self) -> impl Iterator<Item = &Path> {
        self.plugins.iter().map(|plugin| plugin.resident_path())
    }
    /// Returns `true` when no native plugins were loaded.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Consumes the activation and deregisters loaded plugin kinds.
    pub fn clear(self) {}

    pub(crate) fn deregister_plugin_kinds_checked(&mut self) -> DynamicPluginTeardownOutcome {
        deregister_tracked_registrations_checked(&mut self.plugin_registrations, "native")
    }

    #[cfg(test)]
    pub(super) fn with_plugin_kind_for_test(plugin_kind: impl Into<String>) -> Self {
        Self {
            plugins: Vec::new(),
            plugin_registrations: vec![(plugin_kind.into(), 0)],
        }
    }
}

impl Drop for NativePluginActivation {
    fn drop(&mut self) {
        for (plugin_kind, registration_id) in self.plugin_registrations.iter().rev() {
            let _ = deregister_plugin_registration_checked(plugin_kind, *registration_id);
        }
    }
}

/// Loads native dynamic plugins and registers their plugin kinds.
///
/// The returned activation must be kept alive until after active plugin
/// configuration has been cleared.
pub fn load_native_plugins<I>(specs: I) -> crate::Result<NativePluginActivation>
where
    I: IntoIterator<Item = NativePluginLoadSpec>,
{
    load_native_plugins_with_diagnostics(
        specs,
        nemo_relay_diagnostics::Diagnostics::process_default(),
    )
}

/// Load native plugins with the runtime owner's diagnostics service.
pub fn load_native_plugins_with_diagnostics<I>(
    specs: I,
    diagnostics: &nemo_relay_diagnostics::Diagnostics,
) -> crate::Result<NativePluginActivation>
where
    I: IntoIterator<Item = NativePluginLoadSpec>,
{
    load_native_plugins_with_trust(
        specs,
        &HashMap::new(),
        &mut Vec::new(),
        Vec::new(),
        diagnostics,
    )
}

/// Load trusted native plugins and retain their registrations in an activation owner.
pub(super) fn load_native_plugins_with_trust<I>(
    specs: I,
    trust: &HashMap<String, DynamicPluginLoadTrust>,
    skipped: &mut Vec<super::DynamicPluginValidationReport>,
    artifacts: Vec<PreparedNativePlugin>,
    diagnostics: &nemo_relay_diagnostics::Diagnostics,
) -> crate::Result<NativePluginActivation>
where
    I: IntoIterator<Item = NativePluginLoadSpec>,
{
    let mut artifacts = artifacts
        .into_iter()
        .map(|artifact| (artifact.manifest.plugin.id.trim().to_owned(), artifact))
        .collect::<HashMap<_, _>>();
    let mut activation = NativePluginActivation {
        plugins: Vec::new(),
        plugin_registrations: Vec::new(),
    };
    for spec in specs {
        // Core resolution supplies the already verified artifact. Do not
        // re-check an authored path after successful verification.
        let artifact = match artifacts.remove(&spec.plugin_id) {
            Some(artifact) => Some(artifact),
            None => match trust.get(&spec.plugin_id) {
                Some(trust) => match trust.manifest_for_load(&spec.manifest_ref, skipped)? {
                    Some(manifest) => Some(PreparedNativePlugin::new(
                        manifest,
                        spec.manifest_ref.clone(),
                        None,
                    )?),
                    None => continue,
                },
                None => None,
            },
        };
        let observer = super::diagnostics::PluginDiagnostics::new(
            diagnostics,
            &spec.plugin_id,
            nemo_relay_diagnostics::PluginExecution::NativeAbi,
            uuid::Uuid::now_v7(),
        );
        let instance = load_one_native_plugin(
            &spec,
            artifact,
            observer.clone(),
            diagnostics.clone(),
            trust
                .get(&spec.plugin_id)
                .is_some_and(|t| t.policy.native_diagnostics_read_context),
        )
        .inspect_err(|_| {
            observer.failure(
                nemo_relay_diagnostics::PluginOperationFailureCode::NativeLoad,
                nemo_relay_diagnostics::FailureClass::Unknown,
                nemo_relay_diagnostics::PluginPhase::Activation,
            );
        })?;
        let plugin_kind = instance.plugin_kind().to_owned();
        observer.success(
            nemo_relay_diagnostics::SuccessCode::PluginNativeLoaded,
            nemo_relay_diagnostics::PluginPhase::Activation,
        );
        let registration_id = register_plugin_tracked(instance.clone())?;
        activation.plugins.push(instance);
        activation
            .plugin_registrations
            .push((plugin_kind, registration_id));
    }
    Ok(activation)
}

/// Use verified artifact bytes to check and construct one native plugin definition.
fn load_one_native_plugin(
    spec: &NativePluginLoadSpec,
    artifact: Option<PreparedNativePlugin>,
    observer: super::diagnostics::PluginDiagnostics,
    diagnostics: nemo_relay_diagnostics::Diagnostics,
    read: bool,
) -> crate::Result<Arc<super::native_checked::CheckedPlugin>> {
    let (manifest, manifest_ref, verified) = match artifact {
        Some(a) => (a.manifest, a.manifest_ref, a.artifact_bytes),
        None => {
            let (manifest, reference) = DynamicPluginManifest::load_from_path(&spec.manifest_ref)?;
            (manifest, reference, None)
        }
    };
    if manifest.plugin.id.trim() != spec.plugin_id
        || manifest.plugin.kind != DynamicPluginKind::RustDynamic
    {
        return Err(PluginError::InvalidConfig(
            "native manifest identity or kind disagrees with activation request".into(),
        ));
    }
    let api = manifest.compat.native_api.as_deref().unwrap_or("");
    super::native_api_support(api)?;
    super::validate_dynamic_plugin_relay_compatibility(manifest.compat.relay.as_deref(), "native")?;
    let DynamicPluginManifestLoad::RustDynamic(load) = &manifest.load else {
        return Err(PluginError::InvalidConfig(
            "invalid native load contract".into(),
        ));
    };
    if load.symbol.is_some() {
        return Err(PluginError::InvalidConfig("load.symbol is unsupported for checked native plugins; rebuild and use the standard bootstrap header".into()));
    }
    let path = resolve_manifest_relative_path(
        Path::new(&manifest_ref),
        load.library.as_deref().expect("validated library"),
    );
    let bytes = match verified {
        Some(bytes) => bytes,
        None => {
            if !path.exists() {
                return Err(PluginError::NotFound(format!(
                    "native plugin library '{}' does not exist",
                    path.display()
                )));
            }
            super::read_bounded_regular_file(&path, "native plugin library")
                .map_err(PluginError::InvalidConfig)?
        }
    };
    if let Some(digest) = manifest
        .integrity
        .as_ref()
        .and_then(|i| i.sha256.as_deref())
    {
        verify_sha256_bytes(&path, &bytes, digest)?;
    }
    let manifest_root = Path::new(&manifest_ref)
        .parent()
        .expect("manifest has parent");
    let artifact = super::native_dependencies::capture(&path, manifest_root, bytes)
        .map_err(PluginError::InvalidConfig)?;
    let library = super::native_library::CheckedNativeLibrary::load(artifact, api, &spec.plugin_id)
        .map_err(PluginError::InvalidConfig)?;
    Ok(Arc::new(
        super::native_checked::CheckedPlugin::with_observer(&library, diagnostics, read, observer)?,
    ))
}
fn resolve_manifest_relative_path(manifest_path: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        manifest_path
            .parent()
            .map(|parent| parent.join(&path))
            .unwrap_or(path)
    }
}

fn verify_sha256_bytes(path: &Path, bytes: &[u8], expected: &str) -> crate::Result<()> {
    let expected = expected
        .trim()
        .strip_prefix("sha256:")
        .unwrap_or(expected.trim());
    let actual = hex_digest(Sha256::digest(bytes));
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(PluginError::InvalidConfig(format!(
            "native plugin library '{}' sha256 mismatch",
            path.display()
        )))
    }
}

/// Format artifact digest bytes as lowercase hexadecimal for integrity checks.
fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
