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
    open_bounded_regular_file(path, description)?.stream(|chunk| {
        consume(chunk);
        Ok(())
    })
}

/// An open regular file with a checked size.
/// Hashes use the length and bytes from the same file handle. Reads fail if
/// the file grows or shrinks.
pub(crate) struct BoundedFile<'a> {
    file: std::fs::File,
    length: usize,
    path: &'a Path,
    description: &'a str,
}

pub(crate) fn open_bounded_regular_file<'a>(
    path: &'a Path,
    description: &'a str,
) -> Result<BoundedFile<'a>, String> {
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
    if metadata.len() > MAX_BOUNDED_FILE_BYTES {
        return Err(format!(
            "{description} {} exceeds the {MAX_BOUNDED_FILE_BYTES}-byte limit",
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
    let file = options
        .open(path)
        .map_err(|error| format!("failed to read {description} {}: {error}", path.display()))?;
    let opened_metadata = file.metadata().map_err(|error| {
        format!(
            "failed to inspect {description} {}: {error}",
            path.display()
        )
    })?;
    if !opened_metadata.file_type().is_file() {
        return Err(format!(
            "{description} {} must be a regular file",
            path.display()
        ));
    }
    if opened_metadata.len() > MAX_BOUNDED_FILE_BYTES {
        return Err(format!(
            "{description} {} exceeds the {MAX_BOUNDED_FILE_BYTES}-byte limit",
            path.display()
        ));
    }
    let length = usize::try_from(opened_metadata.len())
        .map_err(|_| format!("{description} {} is too large", path.display()))?;
    Ok(BoundedFile {
        file,
        length,
        path,
        description,
    })
}

impl BoundedFile<'_> {
    pub(crate) fn len(&self) -> usize {
        self.length
    }

    pub(crate) fn stream(
        self,
        mut consume: impl FnMut(&[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        const BUFFER_BYTES: usize = 64 * 1024;
        let Self {
            mut file,
            length,
            path,
            description,
        } = self;
        let mut buffer = [0_u8; BUFFER_BYTES];
        let mut total = 0_usize;
        loop {
            let read = file.read(&mut buffer).map_err(|error| {
                format!("failed to read {description} {}: {error}", path.display())
            })?;
            if read == 0 {
                break;
            }
            total = total.saturating_add(read);
            if total > length {
                return Err(format!(
                    "{description} {} changed length while reading",
                    path.display()
                ));
            }
            consume(&buffer[..read])?;
        }
        if total != length {
            return Err(format!(
                "{description} {} changed length while reading",
                path.display()
            ));
        }
        Ok(())
    }
}
