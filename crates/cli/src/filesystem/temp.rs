// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use crate::error::CliError;

pub(crate) fn private_temp_dir(parent: &Path, prefix: &str) -> Result<PathBuf, CliError> {
    let path = parent.join(format!("{prefix}-{}", uuid::Uuid::now_v7()));
    #[cfg(unix)]
    let builder = {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = std::fs::DirBuilder::new();
        builder.mode(0o700);
        builder
    };
    #[cfg(not(unix))]
    let builder = std::fs::DirBuilder::new();
    builder.create(&path)?;
    #[cfg(windows)]
    if let Err(error) = crate::filesystem::protect_private_windows_path(&path) {
        let cleanup = std::fs::remove_dir(&path);
        return Err(CliError::Io(match cleanup {
            Ok(()) => error,
            Err(cleanup_error) => std::io::Error::new(
                cleanup_error.kind(),
                format!(
                    "{error}; additionally failed to remove {}: {cleanup_error}",
                    path.display()
                ),
            ),
        }));
    }
    Ok(path)
}

pub(crate) fn private_system_temp_dir(prefix: &str) -> Result<PathBuf, CliError> {
    private_temp_dir(&std::env::temp_dir(), prefix)
}
