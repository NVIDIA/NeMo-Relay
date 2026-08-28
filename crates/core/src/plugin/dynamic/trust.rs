// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Artifact integrity and signature attestation for dynamic plugins.

use std::path::{Path, PathBuf};

use base64::Engine;
use ring::signature::{ED25519, UnparsedPublicKey};
use sha2::{Digest, Sha256};

use super::{
    DynamicPluginAttestationMode, DynamicPluginCheckState, DynamicPluginFailure,
    DynamicPluginFailurePhase, DynamicPluginManifest, EvaluatedDynamicPluginHostPolicy,
    read_bounded_regular_file, stream_bounded_regular_file,
};

/// A failed artifact trust check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicPluginTrustFailure {
    /// The manifest omitted `source.artifact`.
    MissingArtifact,
    /// The manifest omitted `integrity.sha256`.
    MissingIntegrityDigest,
    /// The artifact could not be safely read.
    ArtifactRead {
        /// Artifact path.
        path: PathBuf,
        /// Read failure detail.
        error: String,
    },
    /// The artifact digest differs from the manifest value.
    IntegrityMismatch {
        /// Artifact path.
        path: PathBuf,
        /// Manifest digest.
        expected: String,
        /// Observed digest.
        actual: String,
    },
    /// The effective policy requires a signature but none was declared.
    MissingSignature,
    /// Signature verification was requested without trusted keys.
    MissingTrustedKeys,
    /// The signature cannot be safely read or decoded.
    SignatureRead {
        /// Signature path.
        path: PathBuf,
        /// Read or decode failure detail.
        error: String,
    },
    /// A configured trusted key is malformed.
    InvalidTrustedKey {
        /// Configured key value.
        key: String,
        /// Parse failure detail.
        error: String,
    },
    /// No trusted key verified the signature.
    SignatureVerification {
        /// Signature path.
        path: PathBuf,
        /// Malformed configured key diagnostics.
        parse_errors: Vec<String>,
    },
}

impl DynamicPluginTrustFailure {
    /// Stable refusal code used in structured reports.
    pub fn refusal_code(&self) -> &'static str {
        match self {
            Self::MissingArtifact
            | Self::MissingIntegrityDigest
            | Self::ArtifactRead { .. }
            | Self::IntegrityMismatch { .. } => "integrity_failed",
            Self::MissingSignature
            | Self::MissingTrustedKeys
            | Self::SignatureRead { .. }
            | Self::InvalidTrustedKey { .. }
            | Self::SignatureVerification { .. } => "attestation_failed",
        }
    }

    /// Human-readable failure description.
    pub fn display(&self, plugin_id: &str) -> String {
        match self {
            Self::MissingArtifact => format!(
                "dynamic plugin '{plugin_id}' is missing source.artifact required for integrity verification"
            ),
            Self::MissingIntegrityDigest => format!(
                "dynamic plugin '{plugin_id}' is missing integrity.sha256 required for host trust verification"
            ),
            Self::ArtifactRead { path, error } => format!(
                "dynamic plugin '{plugin_id}' artifact {} could not be read for trust verification: {error}",
                path.display()
            ),
            Self::IntegrityMismatch {
                path,
                expected,
                actual,
            } => format!(
                "dynamic plugin '{plugin_id}' failed integrity verification for {}: expected {expected}, got {actual}",
                path.display()
            ),
            Self::MissingSignature => format!(
                "dynamic plugin '{plugin_id}' requires integrity.signature under host policy"
            ),
            Self::MissingTrustedKeys => format!(
                "dynamic plugin '{plugin_id}' requires signature verification, but no trusted_public_keys are configured in host policy"
            ),
            Self::SignatureRead { path, error } => format!(
                "dynamic plugin '{plugin_id}' signature {} could not be read: {error}",
                path.display()
            ),
            Self::InvalidTrustedKey { key, error } => format!(
                "dynamic plugin '{plugin_id}' has invalid trusted public key '{key}': {error}"
            ),
            Self::SignatureVerification { path, parse_errors } => {
                let suffix = if parse_errors.is_empty() {
                    String::new()
                } else {
                    format!("; key parse errors: {}", parse_errors.join("; "))
                };
                format!(
                    "dynamic plugin '{plugin_id}' failed signature verification for {} against configured host policy keys{suffix}",
                    path.display()
                )
            }
        }
    }
}

/// Result of integrity and authenticity verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatedDynamicPluginTrust {
    /// Digest verification state.
    pub integrity: DynamicPluginCheckState,
    /// Signature verification state.
    pub authenticity: DynamicPluginCheckState,
    /// Failure when required trust cannot be established.
    pub failure: Option<DynamicPluginTrustFailure>,
}

impl EvaluatedDynamicPluginTrust {
    /// Whether every required trust check succeeded.
    pub fn is_satisfied(&self) -> bool {
        self.failure.is_none()
    }

    /// Returns the trust failure, if required verification failed.
    pub fn failure(&self) -> Option<&DynamicPluginTrustFailure> {
        self.failure.as_ref()
    }

    /// Returns the stable refusal code, if any.
    pub fn refusal_code(&self) -> Option<&'static str> {
        self.failure
            .as_ref()
            .map(DynamicPluginTrustFailure::refusal_code)
    }

    /// Converts a failed trust check into the canonical failure form.
    pub fn last_error(&self, plugin_id: &str) -> Option<DynamicPluginFailure> {
        self.failure.as_ref().map(|failure| DynamicPluginFailure {
            phase: DynamicPluginFailurePhase::Validation,
            code: failure.refusal_code().into(),
            message: failure.display(plugin_id),
        })
    }
}

/// Validates the manifest artifact under the already-evaluated host policy.
pub fn evaluate_dynamic_plugin_trust(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    policy: &EvaluatedDynamicPluginHostPolicy,
) -> EvaluatedDynamicPluginTrust {
    if !policy.policy_satisfied {
        return EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Unknown,
            authenticity: DynamicPluginCheckState::Unknown,
            failure: None,
        };
    }
    let artifact = match verify_integrity(manifest, manifest_ref) {
        Ok(path) => path,
        Err(failure) => {
            return EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Invalid,
                authenticity: DynamicPluginCheckState::Unknown,
                failure: Some(failure),
            };
        }
    };
    match verify_authenticity(manifest, manifest_ref, &artifact, policy) {
        Ok(authenticity) => EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Valid,
            authenticity,
            failure: None,
        },
        Err(failure) => EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Valid,
            authenticity: DynamicPluginCheckState::Invalid,
            failure: Some(failure),
        },
    }
}

/// Resolves an artifact or signature reference relative to its manifest.
pub fn resolve_dynamic_plugin_artifact_path(manifest_ref: &str, reference: &str) -> PathBuf {
    let path = PathBuf::from(reference);
    if path.is_absolute() {
        path
    } else {
        Path::new(manifest_ref)
            .parent()
            .map(|parent| parent.join(&path))
            .unwrap_or(path)
    }
}

fn verify_integrity(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
) -> Result<PathBuf, DynamicPluginTrustFailure> {
    let artifact = manifest
        .source
        .as_ref()
        .and_then(|source| source.artifact.as_deref())
        .ok_or(DynamicPluginTrustFailure::MissingArtifact)?;
    let expected = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.sha256.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or(DynamicPluginTrustFailure::MissingIntegrityDigest)?;
    let artifact = resolve_dynamic_plugin_artifact_path(manifest_ref, artifact);
    let actual =
        sha256_file(&artifact).map_err(|error| DynamicPluginTrustFailure::ArtifactRead {
            path: artifact.clone(),
            error,
        })?;
    if actual != expected {
        return Err(DynamicPluginTrustFailure::IntegrityMismatch {
            path: artifact,
            expected: expected.into(),
            actual,
        });
    }
    Ok(artifact)
}

fn verify_authenticity(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    artifact: &Path,
    policy: &EvaluatedDynamicPluginHostPolicy,
) -> Result<DynamicPluginCheckState, DynamicPluginTrustFailure> {
    let signature = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.signature.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    match (policy.attestation_mode, signature) {
        (DynamicPluginAttestationMode::IntegrityOnly, _) => Ok(DynamicPluginCheckState::Unknown),
        (DynamicPluginAttestationMode::SignatureIfPresent, None) => {
            Ok(DynamicPluginCheckState::Unknown)
        }
        (DynamicPluginAttestationMode::SignatureRequired, None) => {
            Err(DynamicPluginTrustFailure::MissingSignature)
        }
        (_, Some(signature)) => {
            verify_signature(
                manifest_ref,
                artifact,
                signature,
                &policy.trusted_public_keys,
            )?;
            Ok(DynamicPluginCheckState::Valid)
        }
    }
}

fn verify_signature(
    manifest_ref: &str,
    artifact: &Path,
    signature_ref: &str,
    trusted_keys: &[String],
) -> Result<(), DynamicPluginTrustFailure> {
    if trusted_keys.is_empty() {
        return Err(DynamicPluginTrustFailure::MissingTrustedKeys);
    }
    let signature_path = resolve_dynamic_plugin_artifact_path(manifest_ref, signature_ref);
    let raw_signature = read_bounded_regular_file(&signature_path, "dynamic plugin signature")
        .map_err(|error| DynamicPluginTrustFailure::SignatureRead {
            path: signature_path.clone(),
            error,
        })?;
    let signature = base64::engine::general_purpose::STANDARD
        .decode(
            String::from_utf8_lossy(&raw_signature)
                .trim()
                .strip_prefix("ed25519:")
                .unwrap_or(String::from_utf8_lossy(&raw_signature).trim())
                .trim(),
        )
        .map_err(|error| DynamicPluginTrustFailure::SignatureRead {
            path: signature_path.clone(),
            error: format!("invalid base64 signature: {error}"),
        })?;
    let artifact_bytes =
        read_bounded_regular_file(artifact, "dynamic plugin artifact").map_err(|error| {
            DynamicPluginTrustFailure::ArtifactRead {
                path: artifact.to_path_buf(),
                error,
            }
        })?;
    let mut parse_errors = Vec::new();
    for key in trusted_keys {
        match parse_ed25519_key(key) {
            Ok(key) => {
                if UnparsedPublicKey::new(&ED25519, key)
                    .verify(&artifact_bytes, &signature)
                    .is_ok()
                {
                    return Ok(());
                }
            }
            Err(error) => parse_errors.push(error),
        }
    }
    Err(DynamicPluginTrustFailure::SignatureVerification {
        path: signature_path,
        parse_errors,
    })
}

fn parse_ed25519_key(value: &str) -> Result<Vec<u8>, String> {
    let encoded = value
        .trim()
        .strip_prefix("ed25519:")
        .ok_or_else(|| format!("unsupported trusted public key format '{value}'"))?;
    base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .map_err(|error| format!("invalid ed25519 trusted public key '{value}': {error}"))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut digest = Sha256::new();
    stream_bounded_regular_file(path, "dynamic plugin artifact", |chunk| {
        digest.update(chunk)
    })?;
    Ok(format!(
        "sha256:{}",
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}
