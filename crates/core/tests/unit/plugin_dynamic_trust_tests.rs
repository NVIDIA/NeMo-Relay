// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use base64::Engine;
use ring::rand::SystemRandom;
use ring::signature::{Ed25519KeyPair, KeyPair};
use sha2::{Digest, Sha256};

use super::*;

fn manifest(
    artifact: Option<&str>,
    digest: Option<&str>,
    signature: Option<&str>,
) -> DynamicPluginManifest {
    let source = artifact
        .map(|artifact| format!("[source]\nartifact = {artifact:?}\n"))
        .unwrap_or_default();
    let integrity = if digest.is_some() || signature.is_some() {
        format!(
            "[integrity]\n{}{}",
            digest
                .map(|digest| format!("sha256 = {digest:?}\n"))
                .unwrap_or_default(),
            signature
                .map(|signature| format!("signature = {signature:?}\n"))
                .unwrap_or_default(),
        )
    } else {
        String::new()
    };
    DynamicPluginManifest::parse_toml(&format!(
        r#"
manifest_version = 1

[plugin]
id = "fixture.trust"
kind = "worker"

[compat]
relay = ">=0.8.0,<1.0"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "command"
entrypoint = "fixture-worker"

{source}
{integrity}
"#,
    ))
    .unwrap()
}

fn policy(
    satisfied: bool,
    attestation_mode: DynamicPluginAttestationMode,
    trusted_public_keys: Vec<String>,
) -> EvaluatedDynamicPluginHostPolicy {
    EvaluatedDynamicPluginHostPolicy {
        policy_satisfied: satisfied,
        startup_class: super::super::DynamicPluginStartupClass::Required,
        attestation_mode,
        trusted_public_keys,
        failure: (!satisfied).then_some(super::super::DynamicPluginHostPolicyFailure::Blocked),
    }
}

fn digest(bytes: &[u8]) -> String {
    format!(
        "sha256:{}",
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn signing_key() -> Ed25519KeyPair {
    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).unwrap();
    Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap()
}

#[test]
fn trust_failures_have_stable_codes_messages_and_structured_errors() {
    let artifact = PathBuf::from("artifact.bin");
    let signature = PathBuf::from("artifact.sig");
    let failures = vec![
        DynamicPluginTrustFailure::MissingArtifact,
        DynamicPluginTrustFailure::MissingIntegrityDigest,
        DynamicPluginTrustFailure::ArtifactRead {
            path: artifact.clone(),
            error: "unreadable".into(),
        },
        DynamicPluginTrustFailure::IntegrityMismatch {
            path: artifact,
            expected: "sha256:expected".into(),
            actual: "sha256:actual".into(),
        },
        DynamicPluginTrustFailure::MissingSignature,
        DynamicPluginTrustFailure::MissingTrustedKeys,
        DynamicPluginTrustFailure::SignatureRead {
            path: signature.clone(),
            error: "unreadable".into(),
        },
        DynamicPluginTrustFailure::InvalidTrustedKey {
            key: "bad-key".into(),
            error: "invalid".into(),
        },
        DynamicPluginTrustFailure::SignatureVerification {
            path: signature,
            parse_errors: vec!["bad key".into()],
        },
    ];

    for failure in failures {
        let evaluated = EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Invalid,
            authenticity: DynamicPluginCheckState::Invalid,
            failure: Some(failure),
        };
        assert!(!evaluated.is_satisfied());
        assert!(matches!(
            evaluated.refusal_code(),
            Some("integrity_failed" | "attestation_failed")
        ));
        assert!(evaluated.failure().is_some());
        let error = evaluated.last_error("fixture.trust").unwrap();
        assert_eq!(error.phase, DynamicPluginFailurePhase::Validation);
        assert!(error.message.contains("fixture.trust"));
    }

    let without_parse_errors = DynamicPluginTrustFailure::SignatureVerification {
        path: PathBuf::from("artifact.sig"),
        parse_errors: Vec::new(),
    };
    assert!(
        !without_parse_errors
            .display("fixture.trust")
            .contains("key parse errors")
    );
}

#[test]
fn integrity_verification_fails_closed_and_accepts_a_matching_digest() {
    let temp = tempfile::tempdir().unwrap();
    let manifest_ref = temp.path().join("relay-plugin.toml");
    let manifest_ref = manifest_ref.to_str().unwrap();
    let integrity_only = policy(
        true,
        DynamicPluginAttestationMode::IntegrityOnly,
        Vec::new(),
    );

    for (manifest, expected) in [
        (
            manifest(None, None, None),
            DynamicPluginTrustFailure::MissingArtifact,
        ),
        (
            manifest(Some("artifact.bin"), None, None),
            DynamicPluginTrustFailure::MissingIntegrityDigest,
        ),
    ] {
        let evaluated = evaluate_dynamic_plugin_trust(&manifest, manifest_ref, &integrity_only);
        assert_eq!(evaluated.failure, Some(expected));
        assert_eq!(evaluated.integrity, DynamicPluginCheckState::Invalid);
        assert_eq!(evaluated.authenticity, DynamicPluginCheckState::Unknown);
    }

    let missing = manifest(Some("missing.bin"), Some("sha256:missing"), None);
    assert!(matches!(
        evaluate_dynamic_plugin_trust(&missing, manifest_ref, &integrity_only).failure,
        Some(DynamicPluginTrustFailure::ArtifactRead { .. })
    ));

    let bytes = b"trusted plugin artifact";
    std::fs::write(temp.path().join("artifact.bin"), bytes).unwrap();
    let mismatch = manifest(Some("artifact.bin"), Some("sha256:wrong"), None);
    assert!(matches!(
        evaluate_dynamic_plugin_trust(&mismatch, manifest_ref, &integrity_only).failure,
        Some(DynamicPluginTrustFailure::IntegrityMismatch { .. })
    ));

    let valid = manifest(Some("artifact.bin"), Some(&digest(bytes)), None);
    let evaluated = evaluate_dynamic_plugin_trust(&valid, manifest_ref, &integrity_only);
    assert!(evaluated.is_satisfied());
    assert_eq!(evaluated.integrity, DynamicPluginCheckState::Valid);
    assert_eq!(evaluated.authenticity, DynamicPluginCheckState::Unknown);
}

#[test]
fn authenticity_modes_cover_optional_required_and_malformed_evidence() {
    let temp = tempfile::tempdir().unwrap();
    let bytes = b"signed plugin artifact";
    std::fs::write(temp.path().join("artifact.bin"), bytes).unwrap();
    let manifest_ref = temp.path().join("relay-plugin.toml");
    let manifest_ref = manifest_ref.to_str().unwrap();
    let unsigned = manifest(Some("artifact.bin"), Some(&digest(bytes)), None);

    let optional = policy(
        true,
        DynamicPluginAttestationMode::SignatureIfPresent,
        Vec::new(),
    );
    assert!(evaluate_dynamic_plugin_trust(&unsigned, manifest_ref, &optional).is_satisfied());

    let required = policy(
        true,
        DynamicPluginAttestationMode::SignatureRequired,
        Vec::new(),
    );
    assert_eq!(
        evaluate_dynamic_plugin_trust(&unsigned, manifest_ref, &required).failure,
        Some(DynamicPluginTrustFailure::MissingSignature)
    );

    let signed = manifest(
        Some("artifact.bin"),
        Some(&digest(bytes)),
        Some("artifact.sig"),
    );
    assert_eq!(
        evaluate_dynamic_plugin_trust(&signed, manifest_ref, &required).failure,
        Some(DynamicPluginTrustFailure::MissingTrustedKeys)
    );

    let key = signing_key();
    let trusted_key = format!(
        "ed25519:{}",
        base64::engine::general_purpose::STANDARD.encode(key.public_key().as_ref())
    );
    let with_key = policy(
        true,
        DynamicPluginAttestationMode::SignatureRequired,
        vec![trusted_key],
    );
    assert!(matches!(
        evaluate_dynamic_plugin_trust(&signed, manifest_ref, &with_key).failure,
        Some(DynamicPluginTrustFailure::SignatureRead { .. })
    ));

    std::fs::write(temp.path().join("artifact.sig"), "not-base64!").unwrap();
    match evaluate_dynamic_plugin_trust(&signed, manifest_ref, &with_key).failure {
        Some(DynamicPluginTrustFailure::SignatureRead { error, .. }) => {
            assert!(error.contains("invalid base64 signature"), "{error}");
        }
        other => panic!("expected malformed base64 signature failure, got {other:?}"),
    }
}

#[test]
fn valid_signature_is_accepted_and_wrong_or_malformed_keys_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let bytes = b"signed plugin artifact";
    std::fs::write(temp.path().join("artifact.bin"), bytes).unwrap();
    let key = signing_key();
    let signature = key.sign(bytes);
    std::fs::write(
        temp.path().join("artifact.sig"),
        format!(
            "ed25519:{}\n",
            base64::engine::general_purpose::STANDARD.encode(signature.as_ref())
        ),
    )
    .unwrap();
    let manifest = manifest(
        Some("artifact.bin"),
        Some(&digest(bytes)),
        Some("artifact.sig"),
    );
    let manifest_ref = temp.path().join("relay-plugin.toml");
    let manifest_ref = manifest_ref.to_str().unwrap();
    let trusted_key = format!(
        "ed25519:{}",
        base64::engine::general_purpose::STANDARD.encode(key.public_key().as_ref())
    );
    let accepted = policy(
        true,
        DynamicPluginAttestationMode::SignatureRequired,
        vec!["unsupported:key".into(), trusted_key],
    );
    let evaluated = evaluate_dynamic_plugin_trust(&manifest, manifest_ref, &accepted);
    assert!(evaluated.is_satisfied());
    assert_eq!(evaluated.authenticity, DynamicPluginCheckState::Valid);

    let wrong_key = signing_key();
    let rejected = policy(
        true,
        DynamicPluginAttestationMode::SignatureRequired,
        vec![
            "missing-prefix".into(),
            "ed25519:not-base64".into(),
            format!(
                "ed25519:{}",
                base64::engine::general_purpose::STANDARD.encode(wrong_key.public_key().as_ref())
            ),
        ],
    );
    assert!(matches!(
        evaluate_dynamic_plugin_trust(&manifest, manifest_ref, &rejected).failure,
        Some(DynamicPluginTrustFailure::SignatureVerification { parse_errors, .. })
            if parse_errors.len() == 2
    ));
}

#[test]
fn blocked_policy_short_circuits_io_and_path_resolution_preserves_anchors() {
    let manifest = manifest(Some("missing.bin"), Some("sha256:missing"), None);
    let blocked = policy(
        false,
        DynamicPluginAttestationMode::SignatureRequired,
        Vec::new(),
    );
    let evaluated = evaluate_dynamic_plugin_trust(&manifest, "relay-plugin.toml", &blocked);
    assert!(evaluated.is_satisfied());
    assert_eq!(evaluated.integrity, DynamicPluginCheckState::Unknown);
    assert_eq!(evaluated.authenticity, DynamicPluginCheckState::Unknown);

    assert_eq!(
        resolve_dynamic_plugin_artifact_path("plugins/relay-plugin.toml", "artifact.bin"),
        Path::new("plugins").join("artifact.bin")
    );
    let absolute = std::env::temp_dir().join("absolute-artifact.bin");
    assert_eq!(
        resolve_dynamic_plugin_artifact_path(
            "plugins/relay-plugin.toml",
            absolute.to_str().unwrap()
        ),
        absolute
    );
}
