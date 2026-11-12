// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust snapshots of the host runtime diagnostic report.

use serde::Serialize;

/// One bounded runtime diagnostic reported by the Relay host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct RuntimeDiagnostic {
    /// Stable identifier for the diagnostic condition.
    pub code: String,
    /// Most recently recorded message for this condition.
    pub message: String,
    /// Total number of occurrences recorded for this condition.
    pub count: u64,
}

/// Bounded snapshot of active host runtime diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct RuntimeDiagnostics {
    entries: Vec<RuntimeDiagnostic>,
}

impl RuntimeDiagnostics {
    /// Return diagnostics in stable code order.
    pub fn entries(&self) -> &[RuntimeDiagnostic] {
        &self.entries
    }

    /// Return a diagnostic by its stable code.
    pub fn get(&self, code: &str) -> Option<&RuntimeDiagnostic> {
        self.entries
            .iter()
            .find(|diagnostic| diagnostic.code == code)
    }
}
