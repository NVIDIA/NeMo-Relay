// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process and operating-system diagnostic collection.

use super::EnvironmentInfo;

pub(super) fn collect_environment() -> EnvironmentInfo {
    let version = os_version();
    let os = if version.is_empty() {
        std::env::consts::OS.to_string()
    } else {
        format!("{} {version}", std::env::consts::OS)
    };
    let shell_variable = if cfg!(windows) { "COMSPEC" } else { "SHELL" };
    EnvironmentInfo {
        os,
        arch: std::env::consts::ARCH,
        shell: std::env::var(shell_variable).ok().and_then(|path| {
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
