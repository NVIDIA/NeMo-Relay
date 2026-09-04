// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded regular-file reads used by dynamic-plugin trust verification.

use std::io::Read;
use std::path::Path;

/// Maximum accepted size for one plugin manifest, artifact, schema, or signature.
pub const MAX_DYNAMIC_PLUGIN_FILE_BYTES: u64 = 512 * 1024 * 1024;

/// Reads a regular file without following a Unix symlink and with a fixed byte budget.
pub fn read_bounded_regular_file(path: &Path, description: &str) -> Result<Vec<u8>, String> {
    read_regular_file_with_limit(path, description, MAX_DYNAMIC_PLUGIN_FILE_BYTES)
}

/// Reads a regular file without following a Unix symlink and with a caller-provided byte budget.
pub fn read_regular_file_with_limit(
    path: &Path,
    description: &str,
    max_bytes: u64,
) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    stream_regular_file_with_limit(path, description, max_bytes, |chunk| {
        bytes.extend_from_slice(chunk)
    })?;
    Ok(bytes)
}

/// Streams a regular file through `consume` with a fixed byte budget.
pub fn stream_bounded_regular_file(
    path: &Path,
    description: &str,
    consume: impl FnMut(&[u8]),
) -> Result<(), String> {
    stream_regular_file_with_limit(path, description, MAX_DYNAMIC_PLUGIN_FILE_BYTES, consume)
}

/// Streams a regular file through `consume` with a caller-provided byte budget.
pub fn stream_regular_file_with_limit(
    path: &Path,
    description: &str,
    max_bytes: u64,
    mut consume: impl FnMut(&[u8]),
) -> Result<(), String> {
    const BUFFER_BYTES: usize = 64 * 1024;
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        format!(
            "failed to inspect {description} {}: {error}",
            path.display()
        )
    })?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "{description} {} must be a regular file",
            path.display()
        ));
    }
    if metadata.len() > max_bytes {
        return Err(format!(
            "{description} {} exceeds the {max_bytes}-byte limit",
            path.display()
        ));
    }

    #[cfg(unix)]
    let mut file = {
        use rustix::fs::{Mode, OFlags, openat};
        let fd = openat(
            rustix::fs::CWD,
            path,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(|error| format!("failed to read {description} {}: {error}", path.display()))?;
        std::fs::File::from(fd)
    };
    #[cfg(not(unix))]
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("failed to read {description} {}: {error}", path.display()))?;

    let opened = file.metadata().map_err(|error| {
        format!(
            "failed to inspect {description} {}: {error}",
            path.display()
        )
    })?;
    if !opened.file_type().is_file() {
        return Err(format!(
            "{description} {} must be a regular file",
            path.display()
        ));
    }
    if opened.len() > max_bytes {
        return Err(format!(
            "{description} {} exceeds the {max_bytes}-byte limit",
            path.display()
        ));
    }
    let mut buffer = [0_u8; BUFFER_BYTES];
    let mut total = 0_u64;
    loop {
        let count = match file.read(&mut buffer) {
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => {
                return Err(format!(
                    "failed to read {description} {}: {error}",
                    path.display()
                ));
            }
        };
        if count == 0 {
            return Ok(());
        }
        total = total.saturating_add(count as u64);
        if total > max_bytes {
            return Err(format!(
                "{description} {} exceeds the {max_bytes}-byte limit",
                path.display()
            ));
        }
        consume(&buffer[..count]);
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/plugin_dynamic_bounded_tests.rs"]
mod tests;
