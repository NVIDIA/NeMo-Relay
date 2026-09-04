// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Non-interactive dynamic-plugin JSON Schema validation.

use std::path::Path;

use jsonschema::Draft;
use serde_json::Value as Json;

use super::{DynamicPluginManifest, read_regular_file_with_limit};
use crate::plugin::{PluginError, Result};

const MAX_CONFIG_SCHEMA_BYTES: usize = 1024 * 1024;
const DRAFT_7_URIS: [&str; 4] = [
    "http://json-schema.org/draft-07/schema",
    "http://json-schema.org/draft-07/schema#",
    "https://json-schema.org/draft-07/schema",
    "https://json-schema.org/draft-07/schema#",
];
const DRAFT_2020_12_URIS: [&str; 4] = [
    "https://json-schema.org/draft/2020-12/schema",
    "https://json-schema.org/draft/2020-12/schema#",
    "http://json-schema.org/draft/2020-12/schema",
    "http://json-schema.org/draft/2020-12/schema#",
];

/// Validates dynamic component configuration against its manifest-declared local schema.
pub fn validate_dynamic_plugin_config_schema(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    config: &Json,
) -> Result<()> {
    let Some(path) = manifest.resolve_config_schema_path(manifest_ref)? else {
        return Ok(());
    };
    let schema = load_schema(manifest.plugin.id.trim(), &path)?;
    schema.validate(config).map_err(|error| {
        PluginError::InvalidConfig(format!(
            "dynamic plugin '{}' configuration does not satisfy {} at {}: {}",
            manifest.plugin.id,
            path.display(),
            error.instance_path(),
            error
        ))
    })
}

fn load_schema(plugin_id: &str, path: &Path) -> Result<jsonschema::Validator> {
    let bytes = read_regular_file_with_limit(
        path,
        "dynamic plugin config schema",
        MAX_CONFIG_SCHEMA_BYTES as u64,
    )
    .map_err(PluginError::InvalidConfig)?;
    let schema: Json = serde_json::from_slice(&bytes).map_err(|error| {
        PluginError::InvalidConfig(format!(
            "dynamic plugin '{plugin_id}' config schema {} is not valid JSON: {error}",
            path.display()
        ))
    })?;
    reject_nonlocal_references(plugin_id, path, &schema)?;
    let draft = schema
        .get("$schema")
        .and_then(Json::as_str)
        .map(|uri| {
            if DRAFT_7_URIS.contains(&uri) {
                Ok(Draft::Draft7)
            } else if DRAFT_2020_12_URIS.contains(&uri) {
                Ok(Draft::Draft202012)
            } else {
                Err(PluginError::InvalidConfig(format!(
                    "dynamic plugin '{plugin_id}' config schema {} uses unsupported $schema '{uri}'",
                    path.display()
                )))
            }
        })
        .transpose()?
        .unwrap_or(Draft::Draft202012);
    if schema.get("type").and_then(Json::as_str) != Some("object") {
        return Err(PluginError::InvalidConfig(format!(
            "dynamic plugin '{plugin_id}' config schema {} must have object root type",
            path.display()
        )));
    }
    jsonschema::options()
        .with_draft(draft)
        .build(&schema)
        .map_err(|error| {
            PluginError::InvalidConfig(format!(
                "dynamic plugin '{plugin_id}' config schema {} cannot be compiled: {error}",
                path.display()
            ))
        })
}

fn reject_nonlocal_references(plugin_id: &str, path: &Path, value: &Json) -> Result<()> {
    match value {
        Json::Object(object) => {
            if let Some(reference) = object.get("$ref").and_then(Json::as_str)
                && !reference.starts_with('#')
            {
                return Err(PluginError::InvalidConfig(format!(
                    "dynamic plugin '{plugin_id}' config schema {} has forbidden non-local $ref '{reference}'",
                    path.display()
                )));
            }
            for child in object.values() {
                reject_nonlocal_references(plugin_id, path, child)?;
            }
        }
        Json::Array(values) => {
            for child in values {
                reject_nonlocal_references(plugin_id, path, child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/plugin_dynamic_schema_tests.rs"]
mod tests;
