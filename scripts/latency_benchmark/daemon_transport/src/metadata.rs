// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct BinarySpec {
    pub profile: String,
    pub path: PathBuf,
}

impl FromStr for BinarySpec {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (profile, path) = value
            .split_once('=')
            .context("binary metadata must use PROFILE=PATH")?;
        ensure!(!profile.is_empty(), "binary profile is empty");
        let path = PathBuf::from(path);
        ensure!(path.is_file(), "binary does not exist: {}", path.display());
        Ok(Self {
            profile: profile.to_owned(),
            path,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct EnvironmentRecord {
    generated_unix_seconds: u64,
    git_commit: String,
    git_dirty: bool,
    operating_system: String,
    architecture: String,
    logical_cpus: usize,
    rustc: String,
    benchmark_binary: String,
    relay_binaries: Vec<BinaryRecord>,
}

#[derive(Debug, Serialize)]
struct BinaryRecord {
    profile: String,
    path: String,
    size_bytes: u64,
    sha256: String,
    version: Option<String>,
}

pub fn collect(binaries: &[BinarySpec]) -> Result<EnvironmentRecord> {
    let benchmark_binary =
        std::env::current_exe().context("failed to resolve benchmark executable")?;
    Ok(EnvironmentRecord {
        generated_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock precedes Unix epoch")?
            .as_secs(),
        git_commit: command_output("git", &["rev-parse", "HEAD"])
            .unwrap_or_else(|| "unknown".to_owned()),
        git_dirty: command_output("git", &["status", "--porcelain"])
            .is_some_and(|value| !value.is_empty()),
        operating_system: std::env::consts::OS.to_owned(),
        architecture: std::env::consts::ARCH.to_owned(),
        logical_cpus: std::thread::available_parallelism().map_or(1, usize::from),
        rustc: command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_owned()),
        benchmark_binary: benchmark_binary.display().to_string(),
        relay_binaries: binaries.iter().map(binary_record).collect::<Result<_>>()?,
    })
}

fn binary_record(spec: &BinarySpec) -> Result<BinaryRecord> {
    let contents = std::fs::read(&spec.path)
        .with_context(|| format!("failed to read binary {}", spec.path.display()))?;
    let version = Command::new(&spec.path)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    Ok(BinaryRecord {
        profile: spec.profile.clone(),
        path: spec.path.display().to_string(),
        size_bytes: contents.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&contents)),
        version,
    })
}

fn command_output(command: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(command).args(arguments).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let serialized =
        serde_json::to_vec_pretty(value).context("failed to serialize benchmark report")?;
    std::fs::write(path, [serialized.as_slice(), b"\n"].concat())
        .with_context(|| format!("failed to write {}", path.display()))
}
