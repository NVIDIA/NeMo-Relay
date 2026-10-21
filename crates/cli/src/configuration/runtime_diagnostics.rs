// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup staging within the existing shared configuration resolver.
use super::*;
use nemo_relay::diagnostics::{DiagnosticLimits, DiagnosticsSettings};

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
    document: std::sync::Mutex<Option<toml::Value>>,
    pub diagnostics: RuntimeDiagnosticsConfig,
}
impl std::fmt::Debug for PreparedConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedConfig").finish_non_exhaustive()
    }
}
impl PreparedConfig {
    pub(crate) fn load(explicit: Option<&Path>) -> Result<Self, CliError> {
        let explicit = explicit.map(Path::to_path_buf);
        let paths = config_paths(explicit.as_ref())
            .into_iter()
            .map(|path| {
                let required = explicit.as_ref() == Some(&path);
                (path, required)
            })
            .collect();
        let document = read_shared_documents_for_startup(paths, false)?;
        let diagnostics = match document.get("diagnostics") {
            Some(value) => value
                .clone()
                .try_into()
                .map_err(|_| CliError::Config("invalid diagnostics configuration shape".into()))?,
            None => RuntimeDiagnosticsConfig::default(),
        };
        Ok(Self {
            document: std::sync::Mutex::new(Some(document)),
            diagnostics,
        })
    }
    pub(super) fn take_document(&self) -> Option<toml::Value> {
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
        resolve_logging_from_document(document, &["logging"])
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/diagnostic_configuration_tests.rs"]
mod tests;
