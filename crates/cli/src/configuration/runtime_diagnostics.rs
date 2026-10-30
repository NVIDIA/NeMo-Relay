// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup staging within the existing shared configuration resolver.
use super::*;
use nemo_relay_runtime::diagnostics::{DiagnosticLimits, DiagnosticsSettings};

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct RuntimeDiagnosticsConfig {
    pub enabled: bool,
    pub limits: DiagnosticLimits,
    /// Permit whole-context reads using a separate private-bootstrap capability.
    pub http_read: bool,
}
impl RuntimeDiagnosticsConfig {
    pub(crate) fn settings(&self) -> DiagnosticsSettings {
        DiagnosticsSettings {
            enabled: self.enabled,
            limits: self.limits.clone(),
        }
    }
    pub(crate) fn validate(&self) -> Result<(), CliError> {
        self.limits
            .validate()
            .map_err(|_| CliError::Config("invalid diagnostics capacity profile".into()))
    }
}

/// Owns the parsed startup inputs until runtime resolution completes. No plugin
/// loading, logging, credential creation, or process-default lookup happens here.
pub(crate) struct PreparedConfig {
    document: std::sync::Mutex<Option<PreparedSharedDocument>>,
    pub diagnostics: RuntimeDiagnosticsConfig,
    pub collector: crate::daemon::diagnostics::CollectorConfig,
    daemon_logging: bool,
}
impl std::fmt::Debug for PreparedConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedConfig").finish_non_exhaustive()
    }
}
impl PreparedConfig {
    pub(crate) fn load(explicit: Option<&Path>) -> Result<Self, CliError> {
        Self::load_role(explicit, false)
    }
    fn load_role(explicit: Option<&Path>, daemon_logging: bool) -> Result<Self, CliError> {
        let explicit = explicit.map(Path::to_path_buf);
        let paths = config_paths(explicit.as_ref())
            .into_iter()
            .map(|path| {
                let required = explicit.as_ref() == Some(&path);
                (path, required)
            })
            .collect();
        Self::from_paths(paths, daemon_logging)
    }
    pub(crate) fn load_daemon(explicit: Option<&Path>) -> Result<Self, CliError> {
        Self::load_role(explicit, true)
    }
    pub(crate) fn load_managed_worker() -> Result<Self, CliError> {
        let mut paths = Vec::new();
        if let Some(user) = user_config_path() {
            paths.push((user, false));
        }
        paths.push((system_config_dir().join("config.toml"), false));
        Self::from_paths(paths, false)
    }
    fn from_paths(paths: Vec<(PathBuf, bool)>, daemon_logging: bool) -> Result<Self, CliError> {
        let document = read_shared_documents_for_startup(paths, false)?;
        let diagnostics = match document.merged.get("diagnostics") {
            Some(value) => value.clone().try_into().map_err(|error| {
                CliError::Config(format!("invalid diagnostics configuration shape: {error}"))
            })?,
            None => RuntimeDiagnosticsConfig::default(),
        };
        let collector = if daemon_logging {
            document
                .merged
                .get("daemon")
                .and_then(|v| v.get("diagnostics"))
                .map(|v| v.clone().try_into())
                .transpose()
                .map_err(|_| {
                    CliError::Config("invalid daemon diagnostics collection settings".into())
                })?
                .unwrap_or_default()
        } else {
            Default::default()
        };
        Ok(Self {
            collector,
            daemon_logging,
            document: std::sync::Mutex::new(Some(document)),
            diagnostics,
        })
    }
    pub(super) fn take_document(&self) -> Option<PreparedSharedDocument> {
        self.document
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }
    pub(crate) fn logging(&self) -> Result<LoggingConfig, CliError> {
        let document = self.document.lock().unwrap_or_else(|e| e.into_inner());
        let document = document
            .as_ref()
            .ok_or_else(|| CliError::Config("startup configuration already consumed".into()))?;
        resolve_logging_from_document(
            &document.merged,
            if self.daemon_logging {
                &["daemon", "logging"]
            } else {
                &["logging"]
            },
        )
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/diagnostic_configuration_tests.rs"]
mod tests;
