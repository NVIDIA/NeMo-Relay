// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact Hermes shell-hook trust generation and verification.

use std::fs;
use std::path::Path;
use std::time::SystemTime;

use chrono::{DateTime, SecondsFormat, Utc};
use serde_json::{Value, json};

use super::config::is_managed_hook_command;
use crate::error::CliError;
use crate::installer::HERMES_HOOK_EVENTS;

pub(super) fn trusted_hooks(
    existing: Option<&str>,
    command: &str,
    relay: &Path,
    now: SystemTime,
) -> Result<Value, CliError> {
    let mut root = parse_json_object(existing, "Hermes shell-hook allowlist")?;
    let approvals = root
        .as_object_mut()
        .expect("JSON root checked as object")
        .entry("approvals")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| {
            CliError::Install("Hermes shell-hook allowlist approvals must be an array".into())
        })?;
    approvals.retain(|entry| {
        entry
            .get("command")
            .and_then(Value::as_str)
            .is_none_or(|candidate| !is_managed_hook_command(candidate))
    });
    let approved_at = timestamp(now);
    let script_mtime_at_approval = fs::metadata(relay)
        .and_then(|metadata| metadata.modified())
        .ok()
        .map(timestamp);
    approvals.extend(HERMES_HOOK_EVENTS.iter().map(|event| {
        json!({
            "event": event,
            "command": command,
            "approved_at": approved_at,
            "script_mtime_at_approval": script_mtime_at_approval,
        })
    }));
    Ok(root)
}

fn timestamp(time: SystemTime) -> String {
    DateTime::<Utc>::from(time).to_rfc3339_opts(SecondsFormat::Micros, true)
}

pub(super) fn verify_trust(allowlist_path: &Path, command: &str) -> Result<(), String> {
    let raw = fs::read_to_string(allowlist_path)
        .map_err(|error| format!("failed to read {}: {error}", allowlist_path.display()))?;
    let allowlist =
        parse_json_object(Some(&raw), "Hermes shell-hook allowlist").map_err(|e| e.to_string())?;
    let approvals = allowlist
        .get("approvals")
        .and_then(Value::as_array)
        .ok_or_else(|| "Hermes shell-hook approvals are missing".to_string())?;
    for event in HERMES_HOOK_EVENTS {
        let matching = approvals
            .iter()
            .filter(|entry| {
                entry.get("event").and_then(Value::as_str) == Some(event)
                    && entry.get("command").and_then(Value::as_str) == Some(command)
            })
            .count();
        if matching != 1 {
            return Err(format!(
                "Hermes hook {event} expected exactly one trust approval"
            ));
        }
    }
    Ok(())
}

pub(super) fn parse_json_object(raw: Option<&str>, description: &str) -> Result<Value, CliError> {
    let value = match raw.filter(|raw| !raw.trim().is_empty()) {
        Some(raw) => serde_json::from_str(raw)
            .map_err(|error| CliError::Install(format!("invalid {description}: {error}")))?,
        None => json!({}),
    };
    if value.is_object() {
        Ok(value)
    } else {
        Err(CliError::Install(format!(
            "{description} must contain a JSON object"
        )))
    }
}

pub(super) fn json_bytes(value: &Value) -> Result<Vec<u8>, CliError> {
    let mut bytes =
        serde_json::to_vec_pretty(value).map_err(|error| CliError::Install(error.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}
