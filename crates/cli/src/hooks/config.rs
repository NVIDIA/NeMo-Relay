// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private, installer-owned configuration for generated coding-agent hooks.

use std::path::{Path, PathBuf};

#[cfg(any(unix, windows))]
use std::io::Read;

use serde::{Deserialize, Serialize};

use crate::agents::CodingAgent;

use super::{GatewayMode, HookForwardRequest};

const HOOK_CONFIG_VERSION: u32 = 1;
pub(crate) const NATIVE_INVOCATION_CONFIG: &str = ".nemo-relay-invocation.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HookCommandConfig {
    version: u32,
    agent: String,
    gateway_url: String,
    generation_file: Option<PathBuf>,
    generation_token: Option<String>,
    forward_only: bool,
    transparent_run: bool,
    profile: Option<String>,
    session_metadata: Option<String>,
    gateway_mode: Option<GatewayMode>,
    #[serde(default)]
    proxy_credential: Option<String>,
    #[serde(default)]
    invocation_state_dir: Option<PathBuf>,
}

impl HookCommandConfig {
    pub(crate) fn persistent(
        agent: CodingAgent,
        gateway_url: impl Into<String>,
        generation_file: PathBuf,
        generation_token: impl Into<String>,
    ) -> Self {
        Self {
            version: HOOK_CONFIG_VERSION,
            agent: agent.as_arg().into(),
            gateway_url: gateway_url.into(),
            generation_file: Some(generation_file),
            generation_token: Some(generation_token.into()),
            forward_only: false,
            transparent_run: false,
            profile: None,
            session_metadata: None,
            gateway_mode: None,
            proxy_credential: None,
            invocation_state_dir: None,
        }
    }

    pub(crate) fn transparent(agent: CodingAgent, gateway_url: impl Into<String>) -> Self {
        Self {
            version: HOOK_CONFIG_VERSION,
            agent: agent.as_arg().into(),
            gateway_url: gateway_url.into(),
            generation_file: None,
            generation_token: None,
            forward_only: false,
            transparent_run: true,
            profile: None,
            session_metadata: None,
            gateway_mode: None,
            proxy_credential: None,
            invocation_state_dir: None,
        }
    }

    pub(crate) fn write(&self, path: &Path) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| format!("failed to serialize hook configuration: {error}"))?;
        crate::filesystem::atomic_write_private(path, &bytes)
    }

    pub(crate) fn with_proxy_credential(mut self, credential: &str) -> Self {
        self.proxy_credential = Some(credential.into());
        self
    }

    pub(crate) fn with_invocation_state_dir(mut self, path: &Path) -> Self {
        self.invocation_state_dir = Some(path.to_path_buf());
        self
    }

    pub(crate) fn prepared_gateway(&self, agent: CodingAgent) -> Result<&str, String> {
        if self.agent != agent.as_arg()
            || !self.transparent_run
            || self.proxy_credential.as_deref().is_none_or(str::is_empty)
        {
            return Err("invalid prepared native invocation".into());
        }
        Ok(&self.gateway_url)
    }

    pub(crate) fn prepared_gateway_from_native_home() -> Result<Option<String>, String> {
        Self::prepared_config_from_native_home()?
            .map(|(config, agent)| config.prepared_gateway(agent).map(str::to_owned))
            .transpose()
    }

    pub(crate) fn prepared_state_dir_from_native_home() -> Result<Option<PathBuf>, String> {
        Self::prepared_config_from_native_home()?
            .map(|(config, agent)| {
                config.prepared_gateway(agent)?;
                config
                    .invocation_state_dir
                    .ok_or_else(|| "prepared native invocation has no state directory".into())
            })
            .transpose()
    }

    fn prepared_config_from_native_home() -> Result<Option<(Self, CodingAgent)>, String> {
        let mut prepared = None;
        for (variable, agent) in [
            ("CODEX_HOME", CodingAgent::Codex),
            ("CLAUDE_CONFIG_DIR", CodingAgent::ClaudeCode),
        ] {
            let Some(home) = std::env::var_os(variable).map(PathBuf::from) else {
                continue;
            };
            let path = home.join(NATIVE_INVOCATION_CONFIG);
            match std::fs::symlink_metadata(&path) {
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(format!(
                        "cannot inspect prepared native invocation: {error}"
                    ));
                }
            }
            if !home.is_absolute() || home.canonicalize().ok().as_ref() != Some(&home) {
                return Err("prepared native home is not an absolute resolved directory".into());
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                let metadata =
                    std::fs::symlink_metadata(&home).map_err(|error| error.to_string())?;
                if !metadata.is_dir()
                    || metadata.uid() != unsafe { libc::geteuid() }
                    || metadata.mode() & 0o077 != 0
                {
                    return Err(
                        "prepared native home must be current-user-owned and owner-only".into(),
                    );
                }
            }
            let config = Self::load(&path)?;
            config.prepared_gateway(agent)?;
            if prepared.replace((config, agent)).is_some() {
                return Err("multiple prepared native invocations are active".into());
            }
        }
        Ok(prepared)
    }

    pub(crate) fn prepared_native_home_present() -> bool {
        ["CODEX_HOME", "CLAUDE_CONFIG_DIR"].iter().any(|variable| {
            std::env::var_os(variable)
                .map(PathBuf::from)
                .is_some_and(|home| {
                    std::fs::symlink_metadata(home.join(NATIVE_INVOCATION_CONFIG)).is_ok()
                })
        })
    }

    pub(crate) fn load(path: &Path) -> Result<Self, String> {
        let bytes = read_private_hook_config(path).map_err(|error| {
            format!(
                "failed to read hook configuration {}: {error}",
                path.display()
            )
        })?;
        let config = serde_json::from_slice::<Self>(&bytes).map_err(|error| {
            format!(
                "failed to parse hook configuration {}: {error}",
                path.display()
            )
        })?;
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn apply(self, request: &mut HookForwardRequest) -> Result<(), String> {
        if self.agent != request.agent.as_arg() {
            return Err(format!(
                "hook configuration is for {} but the command requested {}",
                self.agent,
                request.agent.as_arg()
            ));
        }
        if request.has_inline_configuration() || request.transparent_run != self.transparent_run {
            return Err(
                "--hook-config cannot be combined with inline hook configuration options".into(),
            );
        }
        request.gateway_url = Some(self.gateway_url);
        request.generation_file = self.generation_file;
        request.generation_token = self.generation_token;
        request.forward_only = self.forward_only;
        request.transparent_run = self.transparent_run;
        request.profile = self.profile;
        request.session_metadata = self.session_metadata;
        request.gateway_mode = self.gateway_mode;
        request.proxy_credential = self.proxy_credential;
        Ok(())
    }

    fn validate(&self) -> Result<(), String> {
        if self.version != HOOK_CONFIG_VERSION {
            return Err(format!(
                "unsupported hook configuration version {}; expected {HOOK_CONFIG_VERSION}",
                self.version
            ));
        }
        if self.agent.trim().is_empty() || self.gateway_url.trim().is_empty() {
            return Err("hook configuration requires an agent and gateway URL".into());
        }
        if self.generation_file.is_some() != self.generation_token.is_some() {
            return Err("hook configuration must include both generation file and token".into());
        }
        if self.forward_only && (self.generation_file.is_some() || self.transparent_run) {
            return Err("forward-only hook configuration cannot include a generation fence or transparent mode".into());
        }
        if self.transparent_run && self.generation_file.is_some() {
            return Err("transparent hook configuration cannot include a generation fence".into());
        }
        if self
            .invocation_state_dir
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
        {
            return Err("prepared invocation state directory must be absolute".into());
        }
        Ok(())
    }
}

/// Reads a Relay-owned hook config without following a replacement symlink.
///
/// The configuration carries generation credentials or a process-private gateway URL. On Unix,
/// require an owner-only file and a current-user-owned, non-group/world-writable parent before
/// opening it with `O_NOFOLLOW`. The post-open metadata check closes the replacement race between
/// inspecting the path and reading its bytes.
#[cfg(unix)]
fn read_private_hook_config(path: &Path) -> Result<Vec<u8>, std::io::Error> {
    use std::fs::{self, OpenOptions};
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    let expected_uid = unsafe { libc::geteuid() };
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file()
        || metadata.uid() != expected_uid
        || metadata.mode() & 0o077 != 0
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "hook configuration must be a current-user-owned owner-only regular file",
        ));
    }
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "hook configuration has no parent",
        )
    })?;
    let parent_metadata = fs::symlink_metadata(parent)?;
    if !parent_metadata.file_type().is_dir()
        || parent_metadata.uid() != expected_uid
        || parent_metadata.mode() & 0o022 != 0
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "hook configuration parent must be current-user-owned and non-group/world-writable",
        ));
    }

    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file()
        || opened.uid() != expected_uid
        || opened.mode() & 0o077 != 0
        || opened.dev() != metadata.dev()
        || opened.ino() != metadata.ino()
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "hook configuration changed while it was opened",
        ));
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[cfg(windows)]
fn read_private_hook_config(path: &Path) -> Result<Vec<u8>, std::io::Error> {
    let mut file = crate::filesystem::open_private_windows_file_for_read(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[cfg(not(any(unix, windows)))]
fn read_private_hook_config(path: &Path) -> Result<Vec<u8>, std::io::Error> {
    std::fs::read(path)
}
