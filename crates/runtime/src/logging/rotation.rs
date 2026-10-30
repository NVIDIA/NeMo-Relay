// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Size-based file rotation for operational log sinks.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

pub(crate) struct SizeRotatingFileWriter {
    base_path: PathBuf,
    file: Option<BufWriter<File>>,
    current_size: u64,
    max_file_size_bytes: u64,
    retained_files: usize,
}

impl SizeRotatingFileWriter {
    pub(crate) fn new(
        base_path: PathBuf,
        max_file_size_bytes: u64,
        retained_files: usize,
    ) -> io::Result<Self> {
        create_parent_directory(&base_path)?;
        let file = open_active_file(&base_path, false)?;
        let current_size = file.get_ref().metadata()?.len();

        Ok(Self {
            base_path,
            file: Some(file),
            current_size,
            max_file_size_bytes,
            retained_files,
        })
    }

    fn rotate_if_needed(&mut self, incoming_bytes: usize) -> io::Result<()> {
        if self.current_size == 0
            || self.current_size.saturating_add(incoming_bytes as u64) <= self.max_file_size_bytes
        {
            return Ok(());
        }

        self.rotate()
    }

    fn rotate(&mut self) -> io::Result<()> {
        let mut file = self
            .file
            .take()
            .ok_or_else(|| io::Error::other("rotating log file is not open"))?;
        if let Err(error) = file.flush() {
            self.file = Some(file);
            return Err(error);
        }
        drop(file);

        if let Err(error) = rotate_files(&self.base_path, self.retained_files) {
            return match self.reopen_after_failed_rotation() {
                Ok(()) => Err(error),
                Err(reopen_error) => Err(io::Error::new(
                    reopen_error.kind(),
                    format!(
                        "log rotation failed: {error}; failed to reopen active log file: \
                         {reopen_error}"
                    ),
                )),
            };
        }

        self.file = Some(open_active_file(&self.base_path, true)?);
        self.current_size = 0;
        Ok(())
    }

    fn reopen_after_failed_rotation(&mut self) -> io::Result<()> {
        let file = open_active_file(&self.base_path, false)?;
        self.current_size = file.get_ref().metadata()?.len();
        self.file = Some(file);
        Ok(())
    }
}

impl Write for SizeRotatingFileWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.rotate_if_needed(buffer.len())?;
        self.file
            .as_mut()
            .ok_or_else(|| io::Error::other("rotating log file is not open"))?
            .write_all(buffer)?;
        self.current_size = self.current_size.saturating_add(buffer.len() as u64);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file
            .as_mut()
            .ok_or_else(|| io::Error::other("rotating log file is not open"))?
            .flush()
    }
}

fn create_parent_directory(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn open_active_file(path: &Path, truncate: bool) -> io::Result<BufWriter<File>> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(!truncate)
        .truncate(truncate)
        .open(path)?;
    Ok(BufWriter::new(file))
}

fn rotate_files(base_path: &Path, retained_files: usize) -> io::Result<()> {
    for index in (1..=retained_files).rev() {
        let source = if index == 1 {
            base_path.to_path_buf()
        } else {
            rotated_log_path(base_path, index - 1)
        };
        if !source.exists() {
            continue;
        }

        let destination = rotated_log_path(base_path, index);
        fs::rename(source, destination)?;
    }
    Ok(())
}

pub(crate) fn rotated_log_path(base_path: &Path, index: usize) -> PathBuf {
    let stem = base_path.file_stem().unwrap_or(base_path.as_os_str());
    let mut file_name = stem.to_os_string();
    file_name.push(format!(".{index}"));
    if let Some(extension) = base_path.extension() {
        file_name.push(".");
        file_name.push(extension);
    }
    base_path.with_file_name(file_name)
}

#[cfg(test)]
#[path = "../../tests/coverage/logging_rotation_tests.rs"]
mod tests;
