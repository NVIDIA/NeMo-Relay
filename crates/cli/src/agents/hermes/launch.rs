// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use crate::error::CliError;
use crate::process::PreparedAgentLaunch;

pub(crate) fn prepare(
    launch: &mut PreparedAgentLaunch,
    hooks_path: Option<&Path>,
    dry_run: bool,
) -> Result<(), CliError> {
    let source_config = hooks_path_for_launch(hooks_path)?;
    let gateway_url = launch
        .env
        .iter()
        .find_map(|(name, value)| {
            (name == crate::configuration::GATEWAY_URL_ENV).then_some(value.as_str())
        })
        .expect("transparent runs always define their gateway URL")
        .to_owned();
    launch.env.push(("HERMES_ACCEPT_HOOKS".into(), "1".into()));
    launch.env.push((
        "OPENAI_BASE_URL".into(),
        format!("{}/v1", gateway_url.trim_end_matches('/')),
    ));
    if dry_run {
        launch.notes.push(format!(
            "would create an isolated Hermes config overlay for {}",
            source_config.display()
        ));
        return Ok(());
    }
    let source_home = source_config.parent().ok_or_else(|| {
        CliError::Launch(format!(
            "Hermes config path {} has no parent directory",
            source_config.display()
        ))
    })?;
    let overlay_home = create_overlay(source_home, &source_config, &gateway_url)?;
    launch
        .env
        .push(("HERMES_HOME".into(), overlay_home.display().to_string()));
    launch.notes.push(format!(
        "using an isolated Hermes config overlay for {}",
        source_config.display()
    ));
    launch.temp_dirs.push(overlay_home);
    Ok(())
}

fn create_overlay(
    source_home: &Path,
    source_config: &Path,
    gateway_url: &str,
) -> Result<PathBuf, CliError> {
    ensure_durable_state_paths(source_home)?;
    let overlay = source_home
        .parent()
        .filter(|parent| parent.is_dir())
        .and_then(|parent| {
            crate::filesystem::temp::private_temp_dir(parent, ".nemo-relay-hermes-home").ok()
        })
        .map(Ok)
        .unwrap_or_else(|| {
            crate::filesystem::temp::private_system_temp_dir("nemo-relay-hermes-home")
        })?;
    if let Err(error) = populate_overlay(&overlay, source_home, source_config, gateway_url) {
        let _ = std::fs::remove_dir_all(&overlay);
        return Err(error);
    }
    Ok(overlay)
}

fn ensure_durable_state_paths(source_home: &Path) -> Result<(), CliError> {
    std::fs::create_dir_all(source_home.join("sessions"))?;
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(source_home.join("state.db"))
    {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(CliError::Io(error)),
    }
}

pub(crate) fn populate_overlay(
    overlay: &Path,
    source_home: &Path,
    source_config: &Path,
    gateway_url: &str,
) -> Result<(), CliError> {
    let absolute_overlay = overlay
        .canonicalize()
        .unwrap_or_else(|_| overlay.to_path_buf());
    match std::fs::read_dir(source_home) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry?;
                let name = entry.file_name();
                if name == "config.yaml" || name == "shell-hooks-allowlist.json" {
                    continue;
                }
                let source = entry.path();
                let absolute_source = source.canonicalize().unwrap_or_else(|_| source.clone());
                if absolute_overlay.starts_with(absolute_source) {
                    continue;
                }
                link_state(&source, &overlay.join(name), entry.file_type()?.is_dir())?;
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(CliError::Io(error)),
    }
    let existing = match std::fs::read_to_string(source_config) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(CliError::Io(error)),
    };
    let relay = std::env::current_exe()
        .map(|path| path.canonicalize().unwrap_or(path))
        .map(crate::agents::portable_executable_path)
        .unwrap_or_else(|_| PathBuf::from("nemo-relay"));
    let contents = crate::agents::hermes::transparent_config(&existing, &relay, gateway_url)?;
    std::fs::write(overlay.join("config.yaml"), contents)?;
    Ok(())
}

fn link_state(source: &Path, destination: &Path, directory: bool) -> Result<(), CliError> {
    #[cfg(unix)]
    {
        let _ = directory;
        std::os::unix::fs::symlink(source, destination)?;
        Ok(())
    }
    #[cfg(windows)]
    {
        if directory {
            create_windows_junction(source, destination)?;
        } else if std::fs::hard_link(source, destination).is_err() {
            std::fs::copy(source, destination)?;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = directory;
        std::fs::copy(source, destination)?;
        Ok(())
    }
}

#[cfg(windows)]
fn create_windows_junction(source: &Path, destination: &Path) -> Result<(), CliError> {
    use std::os::windows::process::CommandExt;

    let mut command = std::process::Command::new(
        std::env::var_os("COMSPEC").unwrap_or_else(|| std::ffi::OsString::from("cmd.exe")),
    );
    command.args(["/d", "/e:on", "/v:off", "/s", "/c"]);
    command
        .raw_arg(r#""mklink /J "%NEMO_RELAY_JUNCTION_DEST%" "%NEMO_RELAY_JUNCTION_SOURCE%" >nul""#);
    let status = command
        .env("NEMO_RELAY_JUNCTION_SOURCE", source)
        .env("NEMO_RELAY_JUNCTION_DEST", destination)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(CliError::Launch(format!(
            "failed to create Hermes state junction {} -> {}: {status}",
            destination.display(),
            source.display()
        )))
    }
}

pub(crate) fn hooks_path_for_launch(configured: Option<&Path>) -> Result<PathBuf, CliError> {
    if let Some(path) = configured {
        return Ok(path.to_path_buf());
    }
    if let Some(home) = std::env::var_os("HERMES_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(home).join("config.yaml"));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .ok_or_else(|| {
            CliError::Launch("could not resolve home directory for Hermes hooks".into())
        })?;
    Ok(PathBuf::from(home).join(".hermes").join("config.yaml"))
}
