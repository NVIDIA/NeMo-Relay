// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dynamic::*;
use base64::Engine as _;
use ring::signature::{Ed25519KeyPair, KeyPair as _};

#[test]
fn captured_dependencies_require_digest_and_matching_trusted_signature() {
    let directory = tempfile::tempdir().unwrap();
    let reference = directory.path().join("relay-plugin.toml");
    let reference = reference.to_str().unwrap();
    let bytes = b"trusted dependency";
    let artifact = super::super::native_library::NativeLibraryArtifact::new(
        "main.so".into(),
        [
            ("main.so".into(), b"main".to_vec()),
            ("lib/dep.so".into(), bytes.to_vec()),
        ],
    )
    .unwrap();
    let mut manifest = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1
[plugin]
id = "dependency.trust"
kind = "rust_dynamic"
[compat]
relay = ">=1.0.0,<2.0"
native_api = "v1.0"
[defaults]
enabled = false
[capabilities]
items = ["plugin_native"]
[load]
library = "main.so"
"#,
    )
    .unwrap();
    let mut policy =
        evaluate_dynamic_plugin_host_policy(&DynamicPluginHostPolicy::default(), &manifest);
    policy.attestation_mode = DynamicPluginAttestationMode::SignatureRequired;
    assert!(
        verify_dependency_closure(&artifact, &manifest, reference, Some(&policy))
            .unwrap_err()
            .to_string()
            .contains("requires integrity.dependencies")
    );
    let evidence = DynamicPluginDependencyIntegrity {
        sha256: format!("sha256:{}", hex_digest(Sha256::digest(bytes))),
        signature: None,
    };
    manifest.integrity = Some(DynamicPluginManifestIntegrity {
        dependencies: [("lib/dep.so".into(), evidence)].into(),
        ..Default::default()
    });
    assert!(
        verify_dependency_closure(&artifact, &manifest, reference, Some(&policy))
            .unwrap_err()
            .to_string()
            .contains("signature")
    );
    let key = Ed25519KeyPair::from_seed_unchecked(&[7; 32]).unwrap();
    policy.trusted_public_keys = vec![format!(
        "ed25519:{}",
        base64::engine::general_purpose::STANDARD.encode(key.public_key().as_ref())
    )];
    std::fs::write(
        directory.path().join("dep.sig"),
        base64::engine::general_purpose::STANDARD.encode(key.sign(bytes).as_ref()),
    )
    .unwrap();
    manifest
        .integrity
        .as_mut()
        .unwrap()
        .dependencies
        .get_mut("lib/dep.so")
        .unwrap()
        .signature = Some("dep.sig".into());
    verify_dependency_closure(&artifact, &manifest, reference, Some(&policy)).unwrap();

    // A replacement before capture must fail even when the main library is unchanged.
    let replaced = super::super::native_library::NativeLibraryArtifact::new(
        "main.so".into(),
        [
            ("main.so".into(), b"main".to_vec()),
            ("lib/dep.so".into(), b"replacement".to_vec()),
        ],
    )
    .unwrap();
    assert!(
        verify_dependency_closure(&replaced, &manifest, reference, Some(&policy))
            .unwrap_err()
            .to_string()
            .contains("sha256 mismatch")
    );
    manifest
        .integrity
        .as_mut()
        .unwrap()
        .dependencies
        .get_mut("lib/dep.so")
        .unwrap()
        .sha256 = format!("sha256:{}", hex_digest(Sha256::digest(b"replacement")));
    assert!(
        verify_dependency_closure(&replaced, &manifest, reference, Some(&policy))
            .unwrap_err()
            .to_string()
            .contains("signature verification")
    );
    // A later filesystem replacement cannot change the already verified capture.
    std::fs::write(directory.path().join("dep.so"), b"replacement").unwrap();
    manifest
        .integrity
        .as_mut()
        .unwrap()
        .dependencies
        .get_mut("lib/dep.so")
        .unwrap()
        .sha256 = format!("sha256:{}", hex_digest(Sha256::digest(bytes)));
    verify_dependency_closure(&artifact, &manifest, reference, Some(&policy)).unwrap();
}
