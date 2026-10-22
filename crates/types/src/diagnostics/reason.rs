// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Portable read failure reasons. Transport and owner-lifecycle errors are separate.

/// Stable reasons shared by canonical snapshot adapters. These are operation
/// results, never a source health classification or a producer failure code.
/// Include a fallback arm when matching: future releases can add reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DiagnosticReadReason {
    /// Collection was disabled at startup.
    Disabled,
    /// Collection is enabled but snapshot workspaces were disabled.
    QueryDisabled,
    /// Requested snapshot version is unsupported.
    UnsupportedSchema,
    /// Malformed, oversized, unknown, or inadmissible query input.
    InvalidQuery,
    /// Continuation source, selection, or revision no longer matches.
    InvalidCursor,
    /// Core or adapter concurrency permits are temporarily exhausted.
    Busy,
    /// The admitted page cannot fit a required envelope or atomic row.
    SnapshotCapacity,
    /// Invalid startup capacity profile, including arithmetic overflow.
    InvalidLimits,
    /// An admitted allocation could not be reserved.
    AllocationFailed,
    /// An unexpected internal read failure; no original error text is exposed.
    Internal,
}
impl DiagnosticReadReason {
    /// Stable wire spelling, also used by language error helpers.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::QueryDisabled => "query_disabled",
            Self::UnsupportedSchema => "unsupported_schema",
            Self::InvalidQuery => "invalid_query",
            Self::InvalidCursor => "invalid_cursor",
            Self::Busy => "busy",
            Self::SnapshotCapacity => "snapshot_capacity",
            Self::InvalidLimits => "invalid_limits",
            Self::AllocationFailed => "allocation_failed",
            Self::Internal => "internal",
        }
    }
}
impl std::fmt::Display for DiagnosticReadReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
impl std::error::Error for DiagnosticReadReason {}
