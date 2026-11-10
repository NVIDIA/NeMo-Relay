// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration-defined defaults, resolved before explicit file layers are applied.

use super::{CliError, merge_gateway_config_toml};
use std::collections::BTreeMap;

#[derive(Default)]
struct Scope {
    selected: Option<String>,
    definitions: toml::Table,
}

/// Collect metadata across file layers, expand selected defaults, then replay explicit layers.
/// Named definitions replace whole definitions; explicit settings retain the normal merge rules.
pub(super) fn resolve(layers: Vec<toml::Value>) -> Result<toml::Value, CliError> {
    resolve_layers(layers, true)
}

/// A doctor/editor layer may select definitions supplied by another file. Validate available
/// definitions here; the shared loader checks missing names against all files at startup.
pub(super) fn validate_layer(value: toml::Value) -> Result<toml::Value, CliError> {
    resolve_layers(vec![value], false)
}

fn resolve_layers(
    mut layers: Vec<toml::Value>,
    require_selected: bool,
) -> Result<toml::Value, CliError> {
    let mut scopes = BTreeMap::new();
    for layer in layers.iter_mut() {
        collect(layer, &mut Vec::new(), &mut scopes)?;
    }
    let mut merged = toml::Value::Table(toml::Table::new());
    for (path, scope) in &scopes {
        let Some(name) = &scope.selected else {
            continue;
        };
        // A selector can use a containing scope's definition without moving that definition.
        // The nearest definition wins; take only the subtree matching the selector's path.
        let definition = (0..=path.len()).rev().find_map(|depth| {
            scopes
                .get(&path[..depth])
                .and_then(|scope| scope.definitions.get(name))
                .map(|definition| (depth, definition))
        });
        let Some((depth, definition)) = definition else {
            if !require_selected {
                continue;
            }
            return Err(CliError::Config(format!(
                "unknown preset {name:?} at {}; define it in {}presets",
                location(path),
                prefix(path),
            )));
        };
        let mut defaults = path[depth..]
            .iter()
            .try_fold(definition, |value, key| value.get(key))
            .filter(|value| value.is_table())
            .cloned()
            .ok_or_else(|| {
                CliError::Config(format!(
                    "preset {name:?} has no matching configuration table at {}",
                    location(path),
                ))
            })?;
        // Inheritance is deliberately unsupported, including indirect self references.
        if contains_metadata(&defaults) {
            return Err(CliError::Config(format!(
                "preset {name:?} at {} cannot contain preset selectors or definitions",
                location(path),
            )));
        }
        for key in path.iter().rev() {
            defaults = toml::Value::Table([(key.clone(), defaults)].into_iter().collect());
        }
        merge_gateway_config_toml(&mut merged, defaults);
    }
    for layer in layers {
        merge_gateway_config_toml(&mut merged, layer);
    }
    Ok(merged)
}

fn contains_metadata(value: &toml::Value) -> bool {
    value.as_table().is_some_and(|table| {
        table.contains_key("preset")
            || table.contains_key("presets")
            || table.values().any(contains_metadata)
    })
}

fn prefix(path: &[String]) -> String {
    if path.is_empty() {
        String::new()
    } else {
        format!("{}.", path.join("."))
    }
}

fn location(path: &[String]) -> String {
    if path.is_empty() {
        "the root".into()
    } else {
        format!("[{}]", path.join("."))
    }
}

/// Remove reserved metadata while preserving the original explicit layer boundaries.
/// Return true for containers emptied by metadata removal, so they cannot replace credentials.
fn collect(
    value: &mut toml::Value,
    path: &mut Vec<String>,
    scopes: &mut BTreeMap<Vec<String>, Scope>,
) -> Result<bool, CliError> {
    let Some(table) = value.as_table_mut() else {
        return Ok(false);
    };
    let had_contents = !table.is_empty();
    if let Some(definitions) = table.remove("presets") {
        let definitions = definitions
            .as_table()
            .ok_or_else(|| CliError::Config(format!("{}presets must be a table", prefix(path))))?;
        for (name, definition) in definitions {
            if !definition.is_table() {
                return Err(CliError::Config(format!(
                    "{}presets.{name} must be a table",
                    prefix(path),
                )));
            }
            scopes
                .entry(path.clone())
                .or_default()
                .definitions
                .insert(name.clone(), definition.clone());
        }
    }
    if let Some(selected) = table.remove("preset") {
        let name = selected
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                CliError::Config(format!("{}preset must be a non-empty string", prefix(path)))
            })?;
        scopes.entry(path.clone()).or_default().selected = Some(name.into());
    }
    let mut empty = Vec::new();
    for (key, child) in table.iter_mut() {
        path.push(key.clone());
        if collect(child, path, scopes)? {
            empty.push(key.clone());
        }
        path.pop();
    }
    for key in empty {
        table.remove(&key);
    }
    Ok(had_contents && table.is_empty())
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/preset_tests.rs"]
mod tests;
