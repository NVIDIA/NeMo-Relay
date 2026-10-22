// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_plugin::diagnostics::{
    DiagnosticDiscovery, DiagnosticError, DiagnosticMemoryPage, DiagnosticQuery,
    DiagnosticSnapshot, Diagnostics,
};

/// Read separate discovery, snapshot, and memory views after an explicit grant.
pub fn inspect_authorized_context(
    diagnostics: &Diagnostics,
) -> Result<
    (
        DiagnosticDiscovery,
        DiagnosticSnapshot,
        DiagnosticMemoryPage,
    ),
    DiagnosticError,
> {
    let reader = diagnostics.reader()?;
    let discovery = reader.discovery()?;
    let query = DiagnosticQuery {
        max_components: Some(1),
        max_events: Some(1),
        ..Default::default()
    };
    let snapshot = reader.snapshot(&query)?;
    let memory = reader.inspect_memory(&query)?;
    // Keep these caller-owned values inside the authorized inspection boundary.
    Ok((discovery, snapshot, memory))
}
