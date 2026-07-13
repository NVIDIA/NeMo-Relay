// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-process serialization for per-user host state and one installation root.

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use crate::agents::CodingAgent;
use crate::filesystem::{LockAttempt, try_lock_exclusive};

pub(super) const DEFAULT_OPERATION_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(25);

pub(super) struct PluginOperationLock {
    _global_file: File,
    _root_file: File,
}

impl PluginOperationLock {
    pub(super) fn acquire(
        host: CodingAgent,
        global_lock_dir: &Path,
        install_dir: &Path,
        timeout: Duration,
    ) -> Result<Self, String> {
        let deadline = Instant::now() + timeout;
        let global_file = acquire_lock_file(host, global_lock_dir, deadline, "global")?;
        let root_file = acquire_lock_file(host, install_dir, deadline, "install-root")?;
        Ok(Self {
            _global_file: global_file,
            _root_file: root_file,
        })
    }
}

fn acquire_lock_file(
    host: CodingAgent,
    directory: &Path,
    deadline: Instant,
    scope: &str,
) -> Result<File, String> {
    fs::create_dir_all(directory).map_err(|error| {
        format!(
            "failed to create plugin operation lock directory {}: {error}",
            directory.display()
        )
    })?;
    let path = operation_lock_path(host, directory);
    let mut options = OpenOptions::new();
    options.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(&path).map_err(|error| {
        format!(
            "failed to open plugin operation lock {}: {error}",
            path.display()
        )
    })?;
    loop {
        match try_lock_exclusive(&file) {
            Ok(LockAttempt::Acquired) => return Ok(file),
            Ok(LockAttempt::Contended) => {
                if Instant::now() >= deadline {
                    return Err(format!(
                        "timed out waiting for another {} plugin install or uninstall operation on the {scope} lock at {}; wait for it to finish and retry",
                        host_name(host),
                        directory.display()
                    ));
                }
                thread::sleep(
                    LOCK_RETRY_INTERVAL.min(deadline.saturating_duration_since(Instant::now())),
                );
            }
            Err(error) => {
                return Err(format!(
                    "failed to lock plugin operation {}: {error}",
                    path.display()
                ));
            }
        }
    }
}

pub(super) fn operation_lock_path(host: CodingAgent, install_dir: &Path) -> PathBuf {
    install_dir.join(format!(".nemo-relay-{}-operation.lock", host_name(host)))
}

fn host_name(host: CodingAgent) -> &'static str {
    match host {
        CodingAgent::Codex => "codex",
        CodingAgent::ClaudeCode => "claude-code",
        CodingAgent::Hermes => {
            unreachable!("all is expanded before operation locking")
        }
    }
}
