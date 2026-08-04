// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::OpenOptions;
use std::io::Read;
use std::path::Path;

use nemo_relay::plugin::dynamic::{DYNAMIC_PLUGIN_MANIFEST_FILENAME, DynamicPluginManifest};

use crate::error::{PluginHostConfigError, Result};

pub(crate) const MAX_BOUNDED_FILE_BYTES: u64 = 512 * 1024 * 1024;

pub(crate) fn read_bounded_regular_file(path: &Path, description: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    stream_bounded_regular_file(path, description, |chunk| bytes.extend_from_slice(chunk))?;
    Ok(bytes)
}

pub(crate) fn read_bounded_utf8_regular_file(path: &Path, description: &str) -> Result<String> {
    let bytes = read_bounded_regular_file(path, description)?;
    String::from_utf8(bytes).map_err(|error| {
        PluginHostConfigError::InvalidConfig(format!(
            "{description} {} is not valid UTF-8: {error}",
            path.display()
        ))
    })
}

pub(crate) fn stream_bounded_regular_file(
    path: &Path,
    description: &str,
    mut consume: impl FnMut(&[u8]),
) -> Result<()> {
    const BUFFER_BYTES: usize = 64 * 1024;
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        PluginHostConfigError::io(format!("inspect {description}"), path, error)
    })?;
    if !metadata.file_type().is_file() {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "{description} {} must be a regular file",
            path.display()
        )));
    }
    if metadata.len() > MAX_BOUNDED_FILE_BYTES {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "{description} {} exceeds the {MAX_BOUNDED_FILE_BYTES}-byte limit",
            path.display()
        )));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options
        .open(path)
        .map_err(|error| PluginHostConfigError::io(format!("read {description}"), path, error))?;
    let opened_metadata = file.metadata().map_err(|error| {
        PluginHostConfigError::io(format!("inspect {description}"), path, error)
    })?;
    if !opened_metadata.file_type().is_file() {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "{description} {} must be a regular file",
            path.display()
        )));
    }
    let mut buffer = [0_u8; BUFFER_BYTES];
    let mut total = 0_u64;
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            PluginHostConfigError::io(format!("read {description}"), path, error)
        })?;
        if read == 0 {
            return Ok(());
        }
        total = total.saturating_add(read as u64);
        if total > MAX_BOUNDED_FILE_BYTES {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "{description} {} exceeds the {MAX_BOUNDED_FILE_BYTES}-byte limit",
                path.display()
            )));
        }
        consume(&buffer[..read]);
    }
}

pub(crate) fn load_bounded_dynamic_plugin_manifest(
    path: impl AsRef<Path>,
) -> Result<(DynamicPluginManifest, String)> {
    let (manifest, normalized, _) = load_bounded_dynamic_plugin_manifest_bytes(path)?;
    Ok((manifest, normalized))
}

pub(crate) fn load_bounded_dynamic_plugin_manifest_bytes(
    path: impl AsRef<Path>,
) -> Result<(DynamicPluginManifest, String, Vec<u8>)> {
    let path = path.as_ref();
    let manifest_path = if path.is_dir() {
        path.join(DYNAMIC_PLUGIN_MANIFEST_FILENAME)
    } else {
        path.to_path_buf()
    };
    let normalized = std::fs::canonicalize(&manifest_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PluginHostConfigError::NotFound {
                path: manifest_path.clone(),
                message: error.to_string(),
            }
        } else {
            PluginHostConfigError::io("normalize dynamic plugin manifest", &manifest_path, error)
        }
    })?;
    let bytes = read_bounded_regular_file(&normalized, "dynamic plugin manifest")?;
    let contents = std::str::from_utf8(&bytes).map_err(|error| {
        PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin manifest {} is not UTF-8: {error}",
            normalized.display()
        ))
    })?;
    let manifest = DynamicPluginManifest::parse_toml(contents).map_err(|_| {
        PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin manifest {} is invalid",
            normalized.display()
        ))
    })?;
    Ok((manifest, normalized.to_string_lossy().into_owned(), bytes))
}

#[cfg(test)]
#[path = "../tests/unit/io.rs"]
mod tests;
