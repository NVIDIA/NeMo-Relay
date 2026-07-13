// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::OpenOptions;
use std::io::Read;
use std::path::Path;

pub(crate) const MAX_BOUNDED_FILE_BYTES: u64 = 512 * 1024 * 1024;

pub(crate) fn read_bounded_regular_file(path: &Path, description: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    stream_bounded_regular_file(path, description, |chunk| bytes.extend_from_slice(chunk))?;
    Ok(bytes)
}

pub(crate) fn stream_bounded_regular_file(
    path: &Path,
    description: &str,
    mut consume: impl FnMut(&[u8]),
) -> Result<(), String> {
    const BUFFER_BYTES: usize = 64 * 1024;
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        format!(
            "failed to inspect {description} {} for persistent gateway identity: {error}",
            path.display()
        )
    })?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "{description} {} must be a regular file for persistent gateway identity",
            path.display()
        ));
    }
    if metadata.len() > MAX_BOUNDED_FILE_BYTES {
        return Err(format!(
            "{description} {} exceeds the {MAX_BOUNDED_FILE_BYTES}-byte persistent gateway identity budget",
            path.display()
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path).map_err(|error| {
        format!(
            "failed to read {description} {} for persistent gateway identity: {error}",
            path.display()
        )
    })?;
    let opened_metadata = file.metadata().map_err(|error| {
        format!(
            "failed to inspect {description} {} for persistent gateway identity: {error}",
            path.display()
        )
    })?;
    if !opened_metadata.file_type().is_file() {
        return Err(format!(
            "{description} {} must be a regular file for persistent gateway identity",
            path.display()
        ));
    }
    if opened_metadata.len() > MAX_BOUNDED_FILE_BYTES {
        return Err(format!(
            "{description} {} exceeds the {MAX_BOUNDED_FILE_BYTES}-byte persistent gateway identity budget",
            path.display()
        ));
    }
    let mut buffer = [0_u8; BUFFER_BYTES];
    let mut total = 0_u64;
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            format!(
                "failed to read {description} {} for persistent gateway identity: {error}",
                path.display()
            )
        })?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if total > MAX_BOUNDED_FILE_BYTES {
            return Err(format!(
                "{description} {} exceeds the {MAX_BOUNDED_FILE_BYTES}-byte persistent gateway identity budget",
                path.display()
            ));
        }
        consume(&buffer[..read]);
    }
    Ok(())
}
