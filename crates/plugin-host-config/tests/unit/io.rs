// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::OpenOptions;
use std::path::Path;

use tempfile::tempdir;

use super::*;

#[test]
fn oversized_regular_file_is_rejected_from_metadata_without_allocation() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("oversized.toml");
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .unwrap();
    file.set_len(MAX_BOUNDED_FILE_BYTES + 1).unwrap();

    let error = read_bounded_regular_file(&path, "plugin configuration file").unwrap_err();
    assert!(error.to_string().contains("exceeds"));
    assert!(error.to_string().contains("byte limit"));
}

#[test]
fn utf8_reader_rejects_invalid_bytes_without_disclosing_contents() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("invalid.toml");
    std::fs::write(&path, [0xff, 0xfe]).unwrap();

    let error = read_bounded_utf8_regular_file(&path, "fixture configuration").unwrap_err();

    assert!(error.to_string().contains("is not valid UTF-8"));
    assert!(!error.to_string().contains("255"));
}

#[test]
fn manifest_loader_accepts_a_directory_and_preserves_authored_bytes() {
    let temp = tempdir().unwrap();
    let contents = r#"manifest_version = 1
[plugin]
id = "fixture"
kind = "rust_dynamic"
[compat]
relay = ">=0.5,<1.0"
native_api = "v1"
[capabilities]
items = ["plugin_native"]
[defaults]
enabled = false
[load]
library = "plugin.so"
symbol = "nemo_relay_plugin_entrypoint_v1"
[source]
artifact = "plugin.so"
[integrity]
sha256 = "sha256:placeholder"
"#;
    std::fs::write(temp.path().join(DYNAMIC_PLUGIN_MANIFEST_FILENAME), contents).unwrap();

    let (manifest, normalized, bytes) =
        load_bounded_dynamic_plugin_manifest_bytes(temp.path()).unwrap();

    assert_eq!(manifest.plugin.id, "fixture");
    assert_eq!(
        Path::new(&normalized),
        temp.path()
            .join(DYNAMIC_PLUGIN_MANIFEST_FILENAME)
            .canonicalize()
            .unwrap()
    );
    assert_eq!(bytes, contents.as_bytes());
}

#[test]
fn manifest_loader_maps_missing_invalid_utf8_and_invalid_toml_errors() {
    let temp = tempdir().unwrap();
    let missing =
        load_bounded_dynamic_plugin_manifest_bytes(temp.path().join("missing.toml")).unwrap_err();
    assert!(matches!(missing, PluginHostConfigError::NotFound { .. }));

    let manifest = temp.path().join("invalid-utf8.toml");
    std::fs::write(&manifest, [0xff]).unwrap();
    let error = load_bounded_dynamic_plugin_manifest_bytes(&manifest).unwrap_err();
    assert!(error.to_string().contains("manifest") && error.to_string().contains("not UTF-8"));

    std::fs::write(&manifest, "manifest_version = [").unwrap();
    let error = load_bounded_dynamic_plugin_manifest_bytes(&manifest).unwrap_err();
    assert!(error.to_string().contains("manifest") && error.to_string().contains("invalid"));
}
