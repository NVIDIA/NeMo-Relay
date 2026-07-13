// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::path::{Path, PathBuf};

use base64::Engine;
use nemo_relay::plugin::dynamic::{
    DynamicPluginAttestationMode, DynamicPluginCheckState, DynamicPluginFailure,
    DynamicPluginFailurePhase, DynamicPluginManifest,
};
use ring::signature::{ED25519, UnparsedPublicKey};
use sha2::{Digest, Sha256};

use crate::plugins::policy::EvaluatedDynamicPluginHostPolicy;

type TrustResult<T> = Result<T, DynamicPluginTrustFailure>;

#[derive(Debug, Clone)]
pub(super) enum DynamicPluginTrustFailure {
    MissingArtifact,
    MissingIntegrityDigest,
    ArtifactRead {
        path: PathBuf,
        error: String,
    },
    IntegrityMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    MissingSignature,
    MissingTrustedKeys,
    SignatureRead {
        path: PathBuf,
        error: String,
    },
    InvalidTrustedKey {
        key: String,
        error: String,
    },
    SignatureVerification {
        path: PathBuf,
        parse_errors: Vec<String>,
    },
}

impl DynamicPluginTrustFailure {
    pub(super) fn display<'a>(
        &'a self,
        plugin_id: &'a str,
    ) -> DynamicPluginTrustFailureDisplay<'a> {
        DynamicPluginTrustFailureDisplay {
            failure: self,
            plugin_id,
        }
    }

    pub(super) fn refusal_code(&self) -> &'static str {
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
}

pub(super) struct DynamicPluginTrustFailureDisplay<'a> {
    failure: &'a DynamicPluginTrustFailure,
    plugin_id: &'a str,
}

impl fmt::Display for DynamicPluginTrustFailureDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.failure {
            DynamicPluginTrustFailure::MissingArtifact => write!(
                f,
                "dynamic plugin '{}' is missing source.artifact required for integrity verification",
                self.plugin_id
            ),
            DynamicPluginTrustFailure::MissingIntegrityDigest => write!(
                f,
                "dynamic plugin '{}' is missing integrity.sha256 required for host trust verification",
                self.plugin_id
            ),
            DynamicPluginTrustFailure::ArtifactRead { path, error } => write!(
                f,
                "dynamic plugin '{}' artifact {} could not be read for trust verification: {}",
                self.plugin_id,
                path.display(),
                error
            ),
            DynamicPluginTrustFailure::IntegrityMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "dynamic plugin '{}' failed integrity verification for {}: expected {}, got {}",
                self.plugin_id,
                path.display(),
                expected,
                actual
            ),
            DynamicPluginTrustFailure::MissingSignature => write!(
                f,
                "dynamic plugin '{}' requires integrity.signature under host policy",
                self.plugin_id
            ),
            DynamicPluginTrustFailure::MissingTrustedKeys => write!(
                f,
                "dynamic plugin '{}' requires signature verification, but no trusted_public_keys are configured in host policy",
                self.plugin_id
            ),
            DynamicPluginTrustFailure::SignatureRead { path, error } => write!(
                f,
                "dynamic plugin '{}' signature {} could not be read: {}",
                self.plugin_id,
                path.display(),
                error
            ),
            DynamicPluginTrustFailure::InvalidTrustedKey { key, error } => write!(
                f,
                "dynamic plugin '{}' has invalid trusted public key '{}': {}",
                self.plugin_id, key, error
            ),
            DynamicPluginTrustFailure::SignatureVerification { path, parse_errors } => {
                write!(
                    f,
                    "dynamic plugin '{}' failed signature verification for {} against configured host policy keys",
                    self.plugin_id,
                    path.display()
                )?;
                if !parse_errors.is_empty() {
                    write!(f, "; key parse errors: {}", parse_errors.join("; "))?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct EvaluatedDynamicPluginTrust {
    pub(super) integrity: DynamicPluginCheckState,
    pub(super) authenticity: DynamicPluginCheckState,
    pub(super) failure: Option<DynamicPluginTrustFailure>,
}

impl EvaluatedDynamicPluginTrust {
    fn valid(authenticity: DynamicPluginCheckState) -> Self {
        Self {
            integrity: DynamicPluginCheckState::Valid,
            authenticity,
            failure: None,
        }
    }

    fn failed(
        integrity: DynamicPluginCheckState,
        authenticity: DynamicPluginCheckState,
        failure: DynamicPluginTrustFailure,
    ) -> Self {
        Self {
            integrity,
            authenticity,
            failure: Some(failure),
        }
    }

    pub(super) fn failure(&self) -> Option<&DynamicPluginTrustFailure> {
        self.failure.as_ref()
    }

    pub(super) fn refusal_code(&self) -> Option<&'static str> {
        self.failure
            .as_ref()
            .map(DynamicPluginTrustFailure::refusal_code)
    }

    pub(super) fn is_satisfied(&self) -> bool {
        self.failure.is_none()
    }

    pub(super) fn last_error(&self, plugin_id: &str) -> Option<DynamicPluginFailure> {
        self.failure.as_ref().map(|failure| DynamicPluginFailure {
            phase: DynamicPluginFailurePhase::Validation,
            code: failure.refusal_code().into(),
            message: failure.display(plugin_id).to_string(),
        })
    }
}

pub(super) fn evaluate_dynamic_plugin_trust(
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

    let artifact_path = match verify_integrity(manifest, manifest_ref) {
        Ok(artifact_path) => artifact_path,
        Err(failure) => {
            return EvaluatedDynamicPluginTrust::failed(
                DynamicPluginCheckState::Invalid,
                DynamicPluginCheckState::Unknown,
                failure,
            );
        }
    };

    match evaluate_authenticity(manifest, manifest_ref, artifact_path.as_path(), policy) {
        Ok(authenticity) => EvaluatedDynamicPluginTrust::valid(authenticity),
        Err(failure) => EvaluatedDynamicPluginTrust::failed(
            DynamicPluginCheckState::Valid,
            DynamicPluginCheckState::Invalid,
            failure,
        ),
    }
}

fn verify_integrity(manifest: &DynamicPluginManifest, manifest_ref: &str) -> TrustResult<PathBuf> {
    let artifact = manifest
        .source
        .as_ref()
        .and_then(|source| source.artifact.as_deref())
        .ok_or(DynamicPluginTrustFailure::MissingArtifact)?;
    let expected_digest = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.sha256.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or(DynamicPluginTrustFailure::MissingIntegrityDigest)?;

    let artifact_path = resolve_artifact_path(manifest_ref, artifact);
    let actual_digest =
        file_sha256(&artifact_path).map_err(|error| DynamicPluginTrustFailure::ArtifactRead {
            path: artifact_path.clone(),
            error: error.to_string(),
        })?;

    if actual_digest != expected_digest {
        return Err(DynamicPluginTrustFailure::IntegrityMismatch {
            path: artifact_path,
            expected: expected_digest.to_owned(),
            actual: actual_digest,
        });
    }

    Ok(artifact_path)
}

fn evaluate_authenticity(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    artifact_path: &Path,
    policy: &EvaluatedDynamicPluginHostPolicy,
) -> TrustResult<DynamicPluginCheckState> {
    let signature_ref = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.signature.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty());

    match policy.attestation_mode {
        DynamicPluginAttestationMode::IntegrityOnly => Ok(DynamicPluginCheckState::Unknown),
        DynamicPluginAttestationMode::SignatureIfPresent => match signature_ref {
            Some(signature_ref) => {
                verify_signature(
                    manifest_ref,
                    artifact_path,
                    signature_ref,
                    &policy.trusted_public_keys,
                )?;
                Ok(DynamicPluginCheckState::Valid)
            }
            None => Ok(DynamicPluginCheckState::Unknown),
        },
        DynamicPluginAttestationMode::SignatureRequired => match signature_ref {
            Some(signature_ref) => {
                verify_signature(
                    manifest_ref,
                    artifact_path,
                    signature_ref,
                    &policy.trusted_public_keys,
                )?;
                Ok(DynamicPluginCheckState::Valid)
            }
            None => Err(DynamicPluginTrustFailure::MissingSignature),
        },
    }
}

fn verify_signature(
    manifest_ref: &str,
    artifact_path: &Path,
    signature_ref: &str,
    trusted_public_keys: &[String],
) -> TrustResult<()> {
    if trusted_public_keys.is_empty() {
        return Err(DynamicPluginTrustFailure::MissingTrustedKeys);
    }

    let signature_path = resolve_artifact_path(manifest_ref, signature_ref);
    let signature_bytes = read_signature_bytes(&signature_path)?;
    let artifact_bytes =
        crate::configuration::read_bounded_regular_file(artifact_path, "dynamic plugin artifact")
            .map_err(|error| DynamicPluginTrustFailure::ArtifactRead {
            path: artifact_path.to_path_buf(),
            error,
        })?;

    let mut parse_errors = Vec::new();
    for trusted_public_key in trusted_public_keys {
        let public_key_bytes = match parse_ed25519_public_key(trusted_public_key) {
            Ok(public_key_bytes) => public_key_bytes,
            Err(DynamicPluginTrustFailure::InvalidTrustedKey { key: _, error }) => {
                parse_errors.push(error);
                continue;
            }
            Err(other) => return Err(other),
        };

        let verifier = UnparsedPublicKey::new(&ED25519, public_key_bytes);
        if verifier.verify(&artifact_bytes, &signature_bytes).is_ok() {
            return Ok(());
        }
    }

    Err(DynamicPluginTrustFailure::SignatureVerification {
        path: signature_path,
        parse_errors,
    })
}

fn read_signature_bytes(path: &Path) -> TrustResult<Vec<u8>> {
    let raw = crate::configuration::read_bounded_regular_file(path, "dynamic plugin signature")
        .map_err(|error| DynamicPluginTrustFailure::SignatureRead {
            path: path.to_path_buf(),
            error,
        })?;
    let trimmed = String::from_utf8_lossy(&raw).trim().to_owned();
    if trimmed.is_empty() {
        return Err(DynamicPluginTrustFailure::SignatureRead {
            path: path.to_path_buf(),
            error: "signature file is empty".into(),
        });
    }

    let encoded = trimmed
        .strip_prefix("ed25519:")
        .unwrap_or(trimmed.as_str())
        .trim();
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| DynamicPluginTrustFailure::SignatureRead {
            path: path.to_path_buf(),
            error: format!("invalid base64 signature: {error}"),
        })
}

fn parse_ed25519_public_key(value: &str) -> TrustResult<Vec<u8>> {
    let encoded = value.trim().strip_prefix("ed25519:").ok_or_else(|| {
        DynamicPluginTrustFailure::InvalidTrustedKey {
            key: value.to_owned(),
            error: format!("unsupported trusted public key format '{value}'"),
        }
    })?;
    base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .map_err(|error| DynamicPluginTrustFailure::InvalidTrustedKey {
            key: value.to_owned(),
            error: format!("invalid ed25519 trusted public key '{value}': {error}"),
        })
}

fn resolve_artifact_path(manifest_ref: &str, artifact_ref: &str) -> PathBuf {
    let artifact_path = PathBuf::from(artifact_ref);
    if artifact_path.is_absolute() {
        artifact_path
    } else {
        Path::new(manifest_ref)
            .parent()
            .map(|parent| parent.join(&artifact_path))
            .unwrap_or(artifact_path)
    }
}

fn file_sha256(path: &Path) -> Result<String, std::io::Error> {
    let mut digest = Sha256::new();
    crate::configuration::stream_bounded_regular_file(path, "dynamic plugin artifact", |bytes| {
        digest.update(bytes);
    })
    .map_err(std::io::Error::other)?;
    Ok(format!(
        "sha256:{}",
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}
