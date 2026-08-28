// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded regular-file reads used by dynamic-plugin trust verification.

use std::io::Read;
use std::path::Path;

/// Maximum accepted size for one plugin manifest, artifact, schema, or signature.
pub const MAX_DYNAMIC_PLUGIN_FILE_BYTES: u64 = 512 * 1024 * 1024;

/// Reads a regular file without following a Unix symlink and with a fixed byte budget.
pub fn read_bounded_regular_file(path: &Path, description: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    stream_bounded_regular_file(path, description, |chunk| bytes.extend_from_slice(chunk))?;
    Ok(bytes)
}

/// Streams a regular file through `consume` with a fixed byte budget.
pub fn stream_bounded_regular_file(
    path: &Path,
    description: &str,
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
    if metadata.len() > MAX_DYNAMIC_PLUGIN_FILE_BYTES {
        return Err(format!(
            "{description} {} exceeds the {MAX_DYNAMIC_PLUGIN_FILE_BYTES}-byte limit",
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
    if opened.len() > MAX_DYNAMIC_PLUGIN_FILE_BYTES {
        return Err(format!(
            "{description} {} exceeds the {MAX_DYNAMIC_PLUGIN_FILE_BYTES}-byte limit",
            path.display()
        ));
    }
    let mut buffer = [0_u8; BUFFER_BYTES];
    let mut total = 0_u64;
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {description} {}: {error}", path.display()))?;
        if count == 0 {
            return Ok(());
        }
        total = total.saturating_add(count as u64);
        if total > MAX_DYNAMIC_PLUGIN_FILE_BYTES {
            return Err(format!(
                "{description} {} exceeds the {MAX_DYNAMIC_PLUGIN_FILE_BYTES}-byte limit",
                path.display()
            ));
        }
        consume(&buffer[..count]);
    }
}
