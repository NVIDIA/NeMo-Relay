// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use nemo_relay::plugin::dynamic::{
    DynamicPluginAttestationMode, DynamicPluginCheckState, DynamicPluginFailure,
    DynamicPluginFailurePhase, DynamicPluginManifest,
};
use ring::signature::{ED25519, UnparsedPublicKey};
use sha2::{Digest, Sha256};

use crate::plugins::policy::EvaluatedDynamicPluginHostPolicy;

#[derive(Debug)]
pub(super) struct EvaluatedDynamicPluginTrust {
    pub(super) integrity: DynamicPluginCheckState,
    pub(super) authenticity: DynamicPluginCheckState,
    pub(super) message: Option<String>,
}

impl EvaluatedDynamicPluginTrust {
    pub(super) fn last_error(
        &self,
        attestation_mode: DynamicPluginAttestationMode,
    ) -> Option<DynamicPluginFailure> {
        self.message.as_ref().map(|message| DynamicPluginFailure {
            phase: DynamicPluginFailurePhase::Validation,
            code: match attestation_mode {
                DynamicPluginAttestationMode::IntegrityOnly => "integrity_verification_failed",
                DynamicPluginAttestationMode::SignatureIfPresent
                | DynamicPluginAttestationMode::SignatureRequired => {
                    "attestation_verification_failed"
                }
            }
            .into(),
            message: message.clone(),
        })
    }
}

pub(super) fn evaluate_dynamic_plugin_trust(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    policy: &EvaluatedDynamicPluginHostPolicy,
) -> EvaluatedDynamicPluginTrust {
    let Some(artifact) = manifest
        .source
        .as_ref()
        .and_then(|source| source.artifact.as_deref())
    else {
        return EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Invalid,
            authenticity: DynamicPluginCheckState::Unknown,
            message: Some(format!(
                "dynamic plugin '{}' is missing source.artifact required for integrity verification",
                manifest.plugin.id
            )),
        };
    };

    let Some(expected_digest) = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.sha256.as_deref())
    else {
        return EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Invalid,
            authenticity: DynamicPluginCheckState::Unknown,
            message: Some(format!(
                "dynamic plugin '{}' is missing integrity.sha256 required for host trust verification",
                manifest.plugin.id
            )),
        };
    };

    let artifact_path = resolve_artifact_path(manifest_ref, artifact);
    let actual_digest = match file_sha256(&artifact_path) {
        Ok(digest) => digest,
        Err(error) => {
            return EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Invalid,
                authenticity: DynamicPluginCheckState::Unknown,
                message: Some(format!(
                    "dynamic plugin '{}' artifact {} could not be read for integrity verification: {}",
                    manifest.plugin.id,
                    artifact_path.display(),
                    error
                )),
            };
        }
    };

    if actual_digest != expected_digest.trim() {
        return EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Invalid,
            authenticity: DynamicPluginCheckState::Unknown,
            message: Some(format!(
                "dynamic plugin '{}' failed integrity verification for {}: expected {}, got {}",
                manifest.plugin.id,
                artifact_path.display(),
                expected_digest.trim(),
                actual_digest
            )),
        };
    }

    evaluate_authenticity(manifest, manifest_ref, artifact_path.as_path(), policy)
}

fn evaluate_authenticity(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    artifact_path: &Path,
    policy: &EvaluatedDynamicPluginHostPolicy,
) -> EvaluatedDynamicPluginTrust {
    let signature_ref = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.signature.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty());

    match policy.attestation_mode {
        DynamicPluginAttestationMode::IntegrityOnly => EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Valid,
            authenticity: DynamicPluginCheckState::Unknown,
            message: None,
        },
        DynamicPluginAttestationMode::SignatureIfPresent => match signature_ref {
            Some(signature_ref) => verify_signature(
                manifest,
                manifest_ref,
                artifact_path,
                signature_ref,
                &policy.trusted_public_keys,
            ),
            None => EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Unknown,
                message: None,
            },
        },
        DynamicPluginAttestationMode::SignatureRequired => match signature_ref {
            Some(signature_ref) => verify_signature(
                manifest,
                manifest_ref,
                artifact_path,
                signature_ref,
                &policy.trusted_public_keys,
            ),
            None => EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Invalid,
                message: Some(format!(
                    "dynamic plugin '{}' requires integrity.signature under host policy",
                    manifest.plugin.id
                )),
            },
        },
    }
}

fn verify_signature(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    artifact_path: &Path,
    signature_ref: &str,
    trusted_public_keys: &[String],
) -> EvaluatedDynamicPluginTrust {
    if trusted_public_keys.is_empty() {
        return EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Valid,
            authenticity: DynamicPluginCheckState::Invalid,
            message: Some(format!(
                "dynamic plugin '{}' requires signature verification, but no trusted_public_keys are configured in host policy",
                manifest.plugin.id
            )),
        };
    }

    let signature_path = resolve_artifact_path(manifest_ref, signature_ref);
    let signature_bytes = match read_signature_bytes(&signature_path) {
        Ok(signature_bytes) => signature_bytes,
        Err(error) => {
            return EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Invalid,
                message: Some(format!(
                    "dynamic plugin '{}' signature {} could not be read: {}",
                    manifest.plugin.id,
                    signature_path.display(),
                    error
                )),
            };
        }
    };

    let artifact_bytes = match fs::read(artifact_path) {
        Ok(artifact_bytes) => artifact_bytes,
        Err(error) => {
            return EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Invalid,
                message: Some(format!(
                    "dynamic plugin '{}' artifact {} could not be read for signature verification: {}",
                    manifest.plugin.id,
                    artifact_path.display(),
                    error
                )),
            };
        }
    };

    let mut parse_errors = Vec::new();
    for trusted_public_key in trusted_public_keys {
        let public_key_bytes = match parse_ed25519_public_key(trusted_public_key) {
            Ok(public_key_bytes) => public_key_bytes,
            Err(error) => {
                parse_errors.push(error);
                continue;
            }
        };

        let verifier = UnparsedPublicKey::new(&ED25519, public_key_bytes);
        if verifier.verify(&artifact_bytes, &signature_bytes).is_ok() {
            return EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Valid,
                message: None,
            };
        }
    }

    let parse_error_suffix = if parse_errors.is_empty() {
        String::new()
    } else {
        format!("; key parse errors: {}", parse_errors.join("; "))
    };

    EvaluatedDynamicPluginTrust {
        integrity: DynamicPluginCheckState::Valid,
        authenticity: DynamicPluginCheckState::Invalid,
        message: Some(format!(
            "dynamic plugin '{}' failed signature verification for {} against configured host policy keys{}",
            manifest.plugin.id,
            signature_path.display(),
            parse_error_suffix
        )),
    }
}

fn read_signature_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let raw = fs::read(path).map_err(|error| error.to_string())?;
    let trimmed = String::from_utf8_lossy(&raw).trim().to_owned();
    if trimmed.is_empty() {
        return Err("signature file is empty".into());
    }

    let encoded = trimmed
        .strip_prefix("ed25519:")
        .unwrap_or(trimmed.as_str())
        .trim();
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| format!("invalid base64 signature: {error}"))
}

fn parse_ed25519_public_key(value: &str) -> Result<Vec<u8>, String> {
    let encoded = value
        .trim()
        .strip_prefix("ed25519:")
        .ok_or_else(|| format!("unsupported trusted public key format '{value}'"))?;
    base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .map_err(|error| format!("invalid ed25519 trusted public key '{value}': {error}"))
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
    let bytes = fs::read(path)?;
    let mut digest = Sha256::new();
    digest.update(&bytes);
    Ok(format!(
        "sha256:{}",
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}
