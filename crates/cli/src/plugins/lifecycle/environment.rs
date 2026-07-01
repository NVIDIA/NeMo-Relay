// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

use nemo_relay::plugin::dynamic::{
    DynamicPluginCheckState, DynamicPluginManifest, DynamicPluginManifestLoad, WorkerRuntime,
};
use sha2::{Digest, Sha256};

const MANAGED_ENVIRONMENTS_DIR: &str = ".dynamic-plugin-environments";

pub(super) trait PythonEnvironmentCommandRunner {
    fn run(&self, program: &OsStr, args: &[OsString]) -> Result<(), String>;
}

pub(super) struct ProcessPythonEnvironmentCommandRunner;

impl PythonEnvironmentCommandRunner for ProcessPythonEnvironmentCommandRunner {
    fn run(&self, program: &OsStr, args: &[OsString]) -> Result<(), String> {
        let status = Command::new(program).args(args).status().map_err(|error| {
            format!("failed to start {}: {error}", Path::new(program).display())
        })?;
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "{} exited with status {status}",
                Path::new(program).display()
            ))
        }
    }
}

pub(super) fn is_python_worker(manifest: &DynamicPluginManifest) -> bool {
    matches!(
        &manifest.load,
        DynamicPluginManifestLoad::Worker(load)
            if load.runtime == Some(WorkerRuntime::Python)
    )
}

pub(super) fn provision_python_environment(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    state_path: &Path,
    runner: &impl PythonEnvironmentCommandRunner,
) -> Result<Option<PathBuf>, String> {
    if !is_python_worker(manifest) {
        return Ok(None);
    }

    let manifest_root = manifest
        .source
        .as_ref()
        .and_then(|source| source.manifest_root.as_deref())
        .map(str::trim)
        .filter(|root| !root.is_empty())
        .ok_or_else(|| {
            "Python worker plugins added through the CLI must declare source.manifest_root"
                .to_string()
        })?;
    let manifest_path = Path::new(manifest_ref);
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let manifest_root = resolve_relative_path(manifest_dir, manifest_root);
    let manifest_root = manifest_root.canonicalize().map_err(|error| {
        format!(
            "could not resolve Python plugin source.manifest_root {}: {error}",
            manifest_root.display()
        )
    })?;
    if !manifest_root.is_dir() {
        return Err(format!(
            "Python plugin source.manifest_root {} is not a directory",
            manifest_root.display()
        ));
    }

    let environment = managed_environment_path(state_path, &manifest.plugin.id)?;
    remove_directory_if_present(&environment, "reset")?;
    let environment_parent = environment.parent().ok_or_else(|| {
        format!(
            "managed Python environment {} has no parent directory",
            environment.display()
        )
    })?;
    std::fs::create_dir_all(environment_parent).map_err(|error| {
        format!(
            "could not create managed Python environment directory {}: {error}",
            environment_parent.display()
        )
    })?;

    let base_python = configured_python_executable();
    let create_args = vec![
        OsString::from("-m"),
        OsString::from("venv"),
        environment.as_os_str().to_owned(),
    ];
    if let Err(error) = runner.run(&base_python, &create_args) {
        let _ = remove_directory_if_present(&environment, "clean up");
        return Err(format!(
            "failed to create managed Python environment {}: {error}",
            environment.display()
        ));
    }

    let environment_python = environment_python_path(&environment);
    if !environment_python.is_file() {
        let _ = remove_directory_if_present(&environment, "clean up");
        return Err(format!(
            "managed Python environment {} did not create interpreter {}",
            environment.display(),
            environment_python.display()
        ));
    }

    let install_args = vec![
        OsString::from("-m"),
        OsString::from("pip"),
        OsString::from("install"),
        manifest_root.as_os_str().to_owned(),
    ];
    if let Err(error) = runner.run(environment_python.as_os_str(), &install_args) {
        let _ = remove_directory_if_present(&environment, "clean up");
        return Err(format!(
            "failed to install Python plugin from {} into {}: {error}",
            manifest_root.display(),
            environment.display()
        ));
    }

    Ok(Some(environment))
}

pub(super) fn remove_managed_environment(
    state_path: &Path,
    plugin_id: &str,
    environment_ref: &str,
) -> Result<(), String> {
    let expected = managed_environment_path(state_path, plugin_id)?;
    let configured = absolute_path(Path::new(environment_ref))?;
    if configured != expected {
        return Err(format!(
            "refusing to delete Python environment {} because the lifecycle-managed path is {}",
            configured.display(),
            expected.display()
        ));
    }
    remove_directory_if_present(&configured, "delete")
}

pub(super) fn environment_state(
    manifest: &DynamicPluginManifest,
    state_path: &Path,
    environment_ref: Option<&str>,
) -> DynamicPluginCheckState {
    if !is_python_worker(manifest) {
        return DynamicPluginCheckState::Unknown;
    }
    let Some(environment_ref) = environment_ref else {
        return DynamicPluginCheckState::Invalid;
    };
    let Ok(expected) = managed_environment_path(state_path, &manifest.plugin.id) else {
        return DynamicPluginCheckState::Invalid;
    };
    let Ok(configured) = absolute_path(Path::new(environment_ref)) else {
        return DynamicPluginCheckState::Invalid;
    };
    if configured != expected
        || std::fs::symlink_metadata(&configured)
            .map(|metadata| !metadata.file_type().is_dir())
            .unwrap_or(true)
        || !environment_python_path(&configured).is_file()
    {
        return DynamicPluginCheckState::Invalid;
    }
    DynamicPluginCheckState::Valid
}

pub(super) fn environment_python_path(environment: &Path) -> PathBuf {
    if cfg!(windows) {
        environment.join("Scripts").join("python.exe")
    } else {
        environment.join("bin").join("python")
    }
}

fn managed_environment_path(state_path: &Path, plugin_id: &str) -> Result<PathBuf, String> {
    let state_path = absolute_path(state_path)?;
    let parent = state_path.parent().ok_or_else(|| {
        format!(
            "dynamic plugin lifecycle state {} has no parent directory",
            state_path.display()
        )
    })?;
    let digest = Sha256::digest(plugin_id.trim().as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(parent.join(MANAGED_ENVIRONMENTS_DIR).join(digest))
}

fn configured_python_executable() -> OsString {
    std::env::var_os("NEMO_RELAY_PYTHON").unwrap_or_else(|| {
        if cfg!(windows) {
            OsString::from("python")
        } else {
            OsString::from("python3")
        }
    })
}

fn resolve_relative_path(base: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|current| current.join(path))
            .map_err(|error| format!("could not resolve {}: {error}", path.display()))
    }
}

fn remove_directory_if_present(path: &Path, operation: &str) -> Result<(), String> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "could not inspect managed Python environment {} before {operation}: {error}",
                path.display()
            ));
        }
    };
    if !metadata.file_type().is_dir() {
        return Err(format!(
            "refusing to {operation} managed Python environment {} because it is not a directory",
            path.display()
        ));
    }
    std::fs::remove_dir_all(path).map_err(|error| {
        format!(
            "could not {operation} managed Python environment {}: {error}",
            path.display()
        )
    })
}
