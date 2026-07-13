// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process and operating-system diagnostic collection.

use super::EnvironmentInfo;

pub(super) fn collect_environment() -> EnvironmentInfo {
    EnvironmentInfo {
        os: format!("{} {}", std::env::consts::OS, os_version()),
        arch: std::env::consts::ARCH,
        shell: std::env::var("SHELL").ok().and_then(|path| {
            std::path::Path::new(&path)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        }),
    }
}

fn os_version() -> String {
    if cfg!(windows) {
        return String::new();
    }
    match std::process::Command::new("uname").arg("-r").output() {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => String::new(),
    }
}
