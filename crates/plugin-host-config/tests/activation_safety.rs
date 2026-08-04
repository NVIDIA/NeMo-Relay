// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for file-backed native activation and lifecycle validation state.

use std::fs;
use std::path::{Path, PathBuf};

use nemo_relay::plugin::dynamic::DynamicPluginKind;
use nemo_relay_plugin_host_config::{
    DynamicPluginActivationSnapshot, DynamicPluginHostPolicy, reconcile_plugin_lifecycle,
    resolve_plugin_files_from_paths,
};
use sha2::{Digest, Sha256};
use tempfile::tempdir;

const LIFECYCLE_STATE_FILE: &str = ".dynamic-plugins.json";

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn write_native_manifest(root: &Path, plugin_id: &str, artifact: &str, library: &str) -> PathBuf {
    fs::create_dir_all(root).unwrap();
    let trusted_bytes = b"integrity-checked native artifact";
    fs::write(root.join(artifact), trusted_bytes).unwrap();
    if library != artifact {
        fs::write(root.join(library), b"different native library").unwrap();
    }
    let manifest = root.join("relay-plugin.toml");
    fs::write(
        &manifest,
        format!(
            r#"manifest_version = 1
[plugin]
id = "{plugin_id}"
kind = "rust_dynamic"
[compat]
relay = ">=0.5,<1.0"
native_api = "v1"
[capabilities]
items = ["plugin_native"]
[defaults]
enabled = false
[load]
library = "{library}"
symbol = "nemo_relay_plugin_entrypoint_v1"
[source]
artifact = "{artifact}"
[integrity]
sha256 = "sha256:{}"
"#,
            sha256_hex(trusted_bytes)
        ),
    )
    .unwrap();
    manifest
}

#[test]
fn native_snapshot_rejects_a_library_other_than_the_integrity_checked_artifact() {
    let temp = tempdir().unwrap();
    let manifest = write_native_manifest(
        temp.path(),
        "native-artifact-mismatch",
        "trusted.so",
        "loaded.so",
    );

    let error = DynamicPluginActivationSnapshot::create(
        manifest.to_string_lossy().as_ref(),
        "native-artifact-mismatch",
        DynamicPluginKind::RustDynamic,
        None,
        &DynamicPluginHostPolicy::default(),
    )
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("must declare its load.library as the integrity-checked source.artifact")
    );
}

#[test]
fn native_snapshot_accepts_the_integrity_checked_library() {
    let temp = tempdir().unwrap();
    let manifest = write_native_manifest(
        temp.path(),
        "native-artifact-match",
        "plugin.so",
        "plugin.so",
    );

    let snapshot = DynamicPluginActivationSnapshot::create(
        manifest.to_string_lossy().as_ref(),
        "native-artifact-match",
        DynamicPluginKind::RustDynamic,
        None,
        &DynamicPluginHostPolicy::default(),
    )
    .unwrap();

    snapshot.verify_current().unwrap();
}

#[test]
fn newly_hydrated_validation_is_timestamped_without_bumping_generation() {
    let temp = tempdir().unwrap();
    let manifest = write_native_manifest(
        &temp.path().join("plugin"),
        "hydrated-validation",
        "plugin.so",
        "plugin.so",
    );
    let config = temp.path().join("plugins.toml");
    fs::write(
        &config,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest.to_string_lossy()
        ),
    )
    .unwrap();
    let resolved = resolve_plugin_files_from_paths([config], None).unwrap();

    reconcile_plugin_lifecycle(&resolved).unwrap();
    let state_path = temp.path().join(LIFECYCLE_STATE_FILE);
    let first: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&state_path).unwrap()).unwrap();
    let first_record = &first["records"][0];
    let first_generation = first_record["metadata"]["generation"].as_u64().unwrap();
    assert!(
        first_record["status"]["validation"]["checked_at"]
            .as_str()
            .is_some_and(|timestamp| !timestamp.is_empty())
    );

    reconcile_plugin_lifecycle(&resolved).unwrap();
    let second: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(state_path).unwrap()).unwrap();
    let second_record = &second["records"][0];
    assert_eq!(
        second_record["metadata"]["generation"].as_u64(),
        Some(first_generation)
    );
    assert!(
        second_record["status"]["validation"]["checked_at"]
            .as_str()
            .is_some_and(|timestamp| !timestamp.is_empty())
    );
}
