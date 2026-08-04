// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;

use nemo_relay::plugin::dynamic::DynamicPluginStartupClass;
use tempfile::tempdir;

use super::*;

#[test]
fn configured_public_key_values_are_redacted_from_trust_errors() {
    let temp = tempdir().unwrap();
    let artifact = temp.path().join("artifact.bin");
    let signature = temp.path().join("signature.txt");
    let manifest = temp.path().join("relay-plugin.toml");
    fs::write(&artifact, b"artifact").unwrap();
    fs::write(&signature, "AA==").unwrap();
    let secret = "do-not-leak-trusted-key";

    let failure = verify_signature(
        manifest.to_string_lossy().as_ref(),
        &artifact,
        "signature.txt",
        &[secret.to_owned()],
    )
    .unwrap_err();
    let rendered = failure.display("fixture").to_string();
    assert!(!rendered.contains(secret), "{rendered}");
}

#[test]
fn trust_helpers_classify_missing_files_empty_signatures_and_invalid_keys() {
    let temp = tempdir().unwrap();
    let manifest_ref = temp.path().join("relay-plugin.toml");
    let missing_artifact = temp.path().join("missing-artifact");
    let signature = temp.path().join("signature.txt");
    fs::write(&signature, "AA==").unwrap();

    let failure = verify_signature(
        manifest_ref.to_string_lossy().as_ref(),
        &missing_artifact,
        "signature.txt",
        &["ed25519:AA==".into()],
    )
    .unwrap_err();
    assert!(matches!(
        failure,
        DynamicPluginTrustFailure::ArtifactRead { .. }
    ));

    let missing = read_signature_bytes(&temp.path().join("missing-signature")).unwrap_err();
    assert!(matches!(
        missing,
        DynamicPluginTrustFailure::SignatureRead { .. }
    ));

    fs::write(&signature, "  \n").unwrap();
    let empty = read_signature_bytes(&signature).unwrap_err();
    assert!(empty.display("fixture").to_string().contains("empty"));

    fs::write(&signature, "not-base64").unwrap();
    let invalid = read_signature_bytes(&signature).unwrap_err();
    assert!(
        invalid
            .display("fixture")
            .to_string()
            .contains("invalid base64")
    );

    let unsupported = parse_ed25519_public_key("plain-key").unwrap_err();
    assert!(matches!(
        unsupported,
        DynamicPluginTrustFailure::InvalidTrustedKey { .. }
    ));
    let malformed = parse_ed25519_public_key("ed25519:not-base64").unwrap_err();
    assert!(
        malformed
            .display("fixture")
            .to_string()
            .contains("invalid trusted public key")
    );

    let absolute = if cfg!(windows) {
        PathBuf::from(r"C:\absolute\artifact")
    } else {
        PathBuf::from("/absolute/artifact")
    };
    assert_eq!(
        resolve_artifact_path("relay-plugin.toml", absolute.to_string_lossy().as_ref()),
        absolute
    );
}

#[test]
fn authenticity_without_an_optional_signature_remains_unknown() {
    let temp = tempdir().unwrap();
    let artifact = temp.path().join("artifact.bin");
    fs::write(&artifact, b"artifact").unwrap();
    let manifest_ref = temp.path().join("relay-plugin.toml");
    let manifest = DynamicPluginManifest::parse_toml(
        r#"manifest_version = 1
[plugin]
id = "optional-signature"
kind = "rust_dynamic"
[compat]
relay = ">=0.5,<1.0"
native_api = "v1"
[capabilities]
items = ["plugin_native"]
[defaults]
enabled = false
[load]
library = "artifact.bin"
symbol = "nemo_relay_plugin_entrypoint_v1"
[source]
artifact = "artifact.bin"
[integrity]
sha256 = "sha256:placeholder"
"#,
    )
    .unwrap();
    let policy = EvaluatedDynamicPluginHostPolicy {
        policy_satisfied: true,
        startup_class: DynamicPluginStartupClass::Optional,
        attestation_mode: DynamicPluginAttestationMode::SignatureIfPresent,
        trusted_public_keys: Vec::new(),
        failure: None,
    };

    assert_eq!(
        evaluate_authenticity(
            &manifest,
            manifest_ref.to_string_lossy().as_ref(),
            &artifact,
            &policy
        )
        .unwrap(),
        DynamicPluginCheckState::Unknown
    );

    let mut missing = manifest;
    missing.source.as_mut().unwrap().artifact = Some("missing.bin".into());
    let trust =
        evaluate_dynamic_plugin_trust(&missing, manifest_ref.to_string_lossy().as_ref(), &policy);
    assert!(matches!(
        trust.failure(),
        Some(DynamicPluginTrustFailure::ArtifactRead { .. })
    ));
}
