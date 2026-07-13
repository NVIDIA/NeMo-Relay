// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded filesystem and network diagnostic probes.

use super::{Check, NETWORK_TIMEOUT, Status};
use std::path::Path;

pub(super) fn check_directory(name: &'static str, path: &Path) -> Check {
    match check_dir_writable(path) {
        Ok(()) => Check {
            name,
            status: Status::Pass,
            details: format!("{} (appears writable)", path.display()),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Check {
            name,
            status: Status::Warn,
            details: format!("{}: not present; runtime will create it", path.display()),
        },
        Err(error) => Check {
            name,
            status: Status::Fail,
            details: format!("{}: {error}", path.display()),
        },
    }
}

pub(super) fn check_dir_writable(directory: &Path) -> Result<(), std::io::Error> {
    let metadata = std::fs::metadata(directory)?;
    if !metadata.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path is not a directory",
        ));
    }
    if metadata.permissions().readonly() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "directory is read-only",
        ));
    }
    Ok(())
}

pub(super) async fn probe_http_named(name: &'static str, url: &str) -> Check {
    let client = match reqwest::Client::builder().timeout(NETWORK_TIMEOUT).build() {
        Ok(client) => client,
        Err(error) => {
            return Check {
                name,
                status: Status::Fail,
                details: format!("could not build HTTP client: {error}"),
            };
        }
    };
    match client.get(url).send().await {
        Ok(response) => Check {
            name,
            status: if response.status().is_success() || response.status().is_redirection() {
                Status::Pass
            } else {
                Status::Warn
            },
            details: format!("{} (HTTP {})", url, response.status().as_u16()),
        },
        Err(error) => Check {
            name,
            status: Status::Fail,
            details: format!("{url}: {error}"),
        },
    }
}
