// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;

use serde_json::json;

use super::*;

fn manifest_with_schema(path: &str) -> DynamicPluginManifest {
    DynamicPluginManifest::parse_toml(&format!(
        r#"
manifest_version = 1

[plugin]
id = "fixture.schema"
kind = "worker"

[compat]
relay = ">=0.8.0,<1.0"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker", "config_schema"]

[config_schema]
path = {path:?}

[load]
runtime = "command"
entrypoint = "fixture-worker"
"#
    ))
    .unwrap()
}

fn manifest_without_schema() -> DynamicPluginManifest {
    DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "fixture.schema-less"
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
"#,
    )
    .unwrap()
}

#[test]
fn schema_validation_accepts_supported_drafts_and_local_references() {
    let temp = tempfile::tempdir().unwrap();
    let manifest_path = temp.path().join("relay-plugin.toml");
    let schema_path = temp.path().join("config.schema.json");
    let manifest = manifest_with_schema("config.schema.json");

    for draft in [
        "http://json-schema.org/draft-07/schema",
        "https://json-schema.org/draft/2020-12/schema",
        "https://json-schema.org/draft/2020-12/schema#",
        "http://json-schema.org/draft/2020-12/schema",
        "http://json-schema.org/draft/2020-12/schema#",
    ] {
        let definitions = if draft.contains("draft-07") {
            json!({"definitions": {"name": {"type": "string"}}})
        } else {
            json!({"$defs": {"name": {"type": "string"}}})
        };
        let reference = if draft.contains("draft-07") {
            "#/definitions/name"
        } else {
            "#/$defs/name"
        };
        let mut schema = json!({
            "$schema": draft,
            "type": "object",
            "properties": {
                "name": {"$ref": reference},
                "items": {"type": "array", "items": {"$ref": reference}}
            },
            "required": ["name"]
        });
        schema
            .as_object_mut()
            .unwrap()
            .extend(definitions.as_object().unwrap().clone());
        fs::write(&schema_path, serde_json::to_vec(&schema).unwrap()).unwrap();

        validate_dynamic_plugin_config_schema(
            &manifest,
            manifest_path.to_string_lossy().as_ref(),
            &json!({"name": "relay", "items": ["one"]}),
        )
        .unwrap();

        let error = validate_dynamic_plugin_config_schema(
            &manifest,
            manifest_path.to_string_lossy().as_ref(),
            &json!({"name": 7}),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("does not satisfy"), "{error}");
    }

    validate_dynamic_plugin_config_schema(
        &manifest_without_schema(),
        manifest_path.to_string_lossy().as_ref(),
        &json!({}),
    )
    .unwrap();
}

#[test]
fn schema_loader_rejects_invalid_documents_and_unsupported_dialects() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("schema.json");

    let cases = [
        (b"{".as_slice(), "not valid JSON"),
        (
            br#"{"$schema":"https://example.com/schema","type":"object"}"#,
            "unsupported $schema",
        ),
        (br#"{"type":"string"}"#, "must have object root type"),
        (
            br#"{"type":"object","properties":{"value":{"type":"not-a-type"}}}"#,
            "cannot be compiled",
        ),
    ];

    for (contents, expected) in cases {
        fs::write(&path, contents).unwrap();
        let error = load_schema("fixture.schema", &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{error}");
    }

    let directory_error = load_schema("fixture.schema", temp.path())
        .unwrap_err()
        .to_string();
    assert!(directory_error.contains("must be a regular file"));
}

#[test]
fn schema_loader_rejects_nonlocal_references_and_oversized_documents() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("schema.json");
    fs::write(
        &path,
        br#"{"type":"object","allOf":[{"$ref":"https://example.com/remote.json"}]}"#,
    )
    .unwrap();
    let error = load_schema("fixture.schema", &path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("forbidden non-local $ref"), "{error}");

    let oversized = temp.path().join("oversized.schema.json");
    let file = fs::File::create(&oversized).unwrap();
    file.set_len(MAX_CONFIG_SCHEMA_BYTES as u64 + 1).unwrap();
    let error = load_schema("fixture.schema", &oversized)
        .unwrap_err()
        .to_string();
    assert!(error.contains("exceeds the"), "{error}");
}
