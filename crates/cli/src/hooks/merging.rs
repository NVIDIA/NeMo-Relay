// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Non-destructive lifecycle-hook configuration merging.

use super::*;

pub(crate) fn merge_hooks(existing: Value, generated: Value) -> Result<Value, CliError> {
    let mut root = hook_config_root(existing)?;
    let hooks = hooks_object_mut(&mut root)?;
    let generated_hooks = generated_hooks_object(&generated)?;
    for (event, groups) in generated_hooks {
        merge_event_hook_groups(hooks, event, groups)?;
    }
    Ok(root)
}

// Normalizes an existing hook config root. Missing files arrive as `Null`, valid JSON/YAML config
// roots remain objects, and other shapes are rejected before any write can occur.
pub(super) fn hook_config_root(existing: Value) -> Result<Value, CliError> {
    match existing {
        Value::Null => Ok(json!({})),
        Value::Object(object) => Ok(Value::Object(object)),
        _ => Err(CliError::Install(
            "hook config must be a JSON object".into(),
        )),
    }
}

// Returns the mutable `hooks` object from a config root, creating it when absent. A non-object
// `hooks` field is considered user config corruption and is not overwritten.
pub(super) fn hooks_object_mut(
    root: &mut Value,
) -> Result<&mut serde_json::Map<String, Value>, CliError> {
    root.as_object_mut()
        .expect("root checked as object")
        .entry("hooks")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| CliError::Install("hooks must be a JSON object".into()))
}

// Validates generated hook shape before merging. Generated hooks are internal data, but checking
// here keeps test failures localized if an agent bundle generator regresses.
pub(super) fn generated_hooks_object(
    generated: &Value,
) -> Result<&serde_json::Map<String, Value>, CliError> {
    generated
        .get("hooks")
        .and_then(Value::as_object)
        .ok_or_else(|| CliError::Install("generated hooks were malformed".into()))
}

// Appends missing generated groups for one hook event. Equality comparison is exact so repeated
// writes are idempotent without trying to interpret vendor-specific hook group schemas.
pub(super) fn merge_event_hook_groups(
    hooks: &mut serde_json::Map<String, Value>,
    event: &str,
    groups: &Value,
) -> Result<(), CliError> {
    let groups = groups
        .as_array()
        .ok_or_else(|| CliError::Install("generated hook groups were malformed".into()))?;
    let event_groups = hooks.entry(event.to_string()).or_insert_with(|| json!([]));
    let event_groups = event_groups
        .as_array_mut()
        .ok_or_else(|| CliError::Install(format!("{event} hooks must be an array")))?;
    for group in groups {
        if !event_groups.iter().any(|existing| existing == group) {
            event_groups.push(group.clone());
        }
    }
    Ok(())
}

// Validates optional JSON strings before they are embedded into hook-forward headers. Catches
// quoting/config mistakes at hook-fire time rather than after the request reaches the gateway.
