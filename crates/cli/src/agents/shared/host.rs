// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Filesystem and platform helpers shared by host configuration.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use toml_edit::{DocumentMut, Item, Table};

pub(crate) use crate::bootstrap::current_exe;
pub(crate) use crate::filesystem::{atomic_write, atomic_write_private};
pub(crate) use crate::gateway::client::healthz;

pub(crate) fn shell_quote(path: &Path) -> String {
    shell_quote_for_platform(path, cfg!(windows))
}

pub(crate) fn shell_quote_for_platform(path: &Path, windows: bool) -> String {
    crate::process::shell_quote_arg_for_platform(&path.display().to_string(), windows)
}

pub(crate) fn ensure_table<'a>(doc: &'a mut DocumentMut, name: &str) -> &'a mut Table {
    if !doc.as_table().contains_key(name) || !doc[name].is_table() {
        doc[name] = Item::Table(Table::new());
    }
    doc[name].as_table_mut().expect("table was just inserted")
}

pub(crate) fn read_json_object(path: &Path) -> Result<Value, String> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let value = serde_json::from_str::<Value>(&raw)
        .map_err(|error| format!("invalid JSON in {}: {error}", path.display()))?;
    if value.is_object() {
        Ok(value)
    } else {
        Err(format!("{} must contain a JSON object", path.display()))
    }
}

pub(crate) fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

pub(crate) fn home_dir() -> Result<PathBuf, String> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| "cannot determine home directory (set HOME or USERPROFILE)".into())
}

pub(crate) fn print_check(label: &str, ok: bool) -> bool {
    println!("{} {label}", if ok { "ok" } else { "missing" });
    ok
}

pub(crate) fn print_info(label: &str, message: &str) {
    println!("info {label}: {message}");
}
