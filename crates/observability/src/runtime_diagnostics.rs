// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(missing_docs)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

#[doc(hidden)]
pub const MAX_RUNTIME_DIAGNOSTICS: usize = 32;
#[doc(hidden)]
pub const MAX_RUNTIME_DIAGNOSTIC_MESSAGE_CHARS: usize = 1_024;

/// One runtime diagnostic recorded by an exporter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExporterRuntimeDiagnostic {
    /// Stable diagnostic code.
    pub code: String,
    /// Distinct messages for this code, newline separated, capped in length.
    pub message: String,
    /// Sum of the counts recorded for this code.
    pub count: u64,
}

/// Snapshot of runtime diagnostics recorded by an exporter.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExporterRuntimeDiagnostics {
    pub diagnostics: Vec<ExporterRuntimeDiagnostic>,
}

impl ExporterRuntimeDiagnostics {
    /// Return diagnostics sorted by code.
    pub fn entries(&self) -> &[ExporterRuntimeDiagnostic] {
        &self.diagnostics
    }

    /// Return diagnostic for `code`, if recorded.
    pub fn get(&self, code: &str) -> Option<&ExporterRuntimeDiagnostic> {
        self.diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code == code)
    }
}

#[derive(Debug, Default)]
#[doc(hidden)]
pub struct RuntimeDiagnosticState {
    pub diagnostics: BTreeMap<String, RuntimeDiagnosticEntry>,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct RuntimeDiagnosticEntry {
    pub diagnostic: ExporterRuntimeDiagnostic,
    pub messages: BTreeSet<String>,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct ExporterRuntimeDiagnosticsRecorder {
    pub host: crate::host::HostCell,
    pub state: Arc<Mutex<RuntimeDiagnosticState>>,
    pub plugin_field: Option<String>,
}

impl ExporterRuntimeDiagnosticsRecorder {
    #[doc(hidden)]
    pub fn new(plugin_field: Option<String>) -> Self {
        Self {
            host: crate::host::HostCell::default(),
            state: Arc::new(Mutex::new(RuntimeDiagnosticState::default())),
            plugin_field,
        }
    }

    #[doc(hidden)]
    pub fn record(&self, code: impl Into<String>, message: String, count: u64) -> u64 {
        let code = code.into();
        let count = count.max(1);
        let message = truncate_runtime_diagnostic_message(message);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let total = if let Some(diagnostic) = state.diagnostics.get_mut(&code) {
            diagnostic.diagnostic.count = diagnostic.diagnostic.count.saturating_add(count);
            if diagnostic.messages.insert(message.clone()) {
                let combined = combine_runtime_diagnostic_messages(&diagnostic.messages);
                if combined.chars().count() <= MAX_RUNTIME_DIAGNOSTIC_MESSAGE_CHARS {
                    diagnostic.diagnostic.message = combined;
                } else {
                    diagnostic.messages.remove(&message);
                }
            }
            diagnostic.diagnostic.count
        } else if state.diagnostics.len() < MAX_RUNTIME_DIAGNOSTICS {
            let mut messages = BTreeSet::new();
            messages.insert(message.clone());
            state.diagnostics.insert(
                code.clone(),
                RuntimeDiagnosticEntry {
                    diagnostic: ExporterRuntimeDiagnostic {
                        code: code.clone(),
                        message: message.clone(),
                        count,
                    },
                    messages,
                },
            );
            count
        } else {
            return 0;
        };
        drop(state);

        if let Some(field) = &self.plugin_field {
            self.host
                .get()
                .record_runtime_diagnostic(&code, Some(field.clone()), message, count);
        }
        total
    }

    #[doc(hidden)]
    pub fn snapshot(&self) -> ExporterRuntimeDiagnostics {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ExporterRuntimeDiagnostics {
            diagnostics: state
                .diagnostics
                .values()
                .map(|entry| entry.diagnostic.clone())
                .collect(),
        }
    }

    #[doc(hidden)]
    pub fn has_plugin_mirror(&self) -> bool {
        self.plugin_field.is_some()
    }
}

pub(crate) fn truncate_runtime_diagnostic_message(message: String) -> String {
    if message.chars().count() <= MAX_RUNTIME_DIAGNOSTIC_MESSAGE_CHARS {
        return message;
    }
    let mut truncated = message
        .chars()
        .take(MAX_RUNTIME_DIAGNOSTIC_MESSAGE_CHARS - 1)
        .collect::<String>();
    truncated.push('…');
    truncated
}

pub(crate) fn combine_runtime_diagnostic_messages(messages: &BTreeSet<String>) -> String {
    messages
        .iter()
        .fold(String::new(), |mut combined, message| {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(message);
            combined
        })
}

impl ExporterRuntimeDiagnosticsRecorder {
    pub fn bind_host(&self, host: Arc<dyn crate::host::ObservabilityHost>) {
        self.host.bind(host);
    }
}

impl ExporterRuntimeDiagnosticsRecorder {
    pub fn with_host(self, host: Arc<dyn crate::host::ObservabilityHost>) -> Self {
        self.bind_host(host);
        self
    }
}
