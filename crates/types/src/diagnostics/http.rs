// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic inspection HTTP contracts, independent of any HTTP framework.

use super::{DIAGNOSTIC_SCHEMA_VERSION, DiagnosticReadReason};

/// Maximum error envelope accepted by diagnostic HTTP consumers, in bytes.
/// Error responses never contain retained diagnostic evidence or raw source errors.
pub const MAX_DIAGNOSTIC_HTTP_ERROR_BYTES: usize = 1024;

/// Stable reasons for HTTP inspection, including snapshot, access, and lifecycle failures.
/// This is separate from the transport-neutral [`DiagnosticReadReason`].
/// Unknown future reasons must not be inferred from HTTP status alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DiagnosticHttpReason {
    /// Diagnostic recording was disabled at startup.
    Disabled,
    /// Snapshot query workspaces were disabled at startup.
    QueryDisabled,
    /// The requested snapshot schema is unsupported.
    UnsupportedSchema,
    /// The query is invalid.
    InvalidQuery,
    /// The continuation is invalid or stale.
    InvalidCursor,
    /// Read concurrency is temporarily exhausted.
    Busy,
    /// The snapshot cannot fit the admitted envelope.
    SnapshotCapacity,
    /// The startup capacity profile is invalid.
    InvalidLimits,
    /// A backing allocation failed.
    AllocationFailed,
    /// An unexpected internal read failure occurred.
    Internal,
    /// Inspection requires Relay's authenticated transport.
    AuthenticatedTransportRequired,
    /// The read credential is absent.
    Unauthorized,
    /// The credential or request origin is not permitted.
    Forbidden,
    /// The diagnostic reader is shutting down.
    ShuttingDown,
    /// HTTP diagnostic reads are disabled.
    HttpReadDisabled,
    /// Managed-worker collection is disabled.
    CollectionDisabled,
    /// The selected worker assignment does not exist, after authorization.
    /// This does not describe a current worker whose cache is empty.
    Missing,
    /// The selected peer does not implement diagnostics.
    Unsupported,
    /// The current worker session has no usable diagnostic read authority.
    ReadAuthorityUnavailable,
}

impl DiagnosticHttpReason {
    /// Canonical HTTP status for this reason. Adapters construct framework responses.
    pub const fn http_status(self) -> u16 {
        match self {
            Self::Disabled
            | Self::QueryDisabled
            | Self::ShuttingDown
            | Self::HttpReadDisabled
            | Self::CollectionDisabled
            | Self::ReadAuthorityUnavailable => 503,
            Self::UnsupportedSchema => 406,
            Self::InvalidQuery => 400,
            Self::InvalidCursor => 409,
            Self::Busy => 429,
            Self::SnapshotCapacity => 422,
            Self::InvalidLimits | Self::AllocationFailed | Self::Internal => 500,
            Self::AuthenticatedTransportRequired => 426,
            Self::Unauthorized => 401,
            Self::Forbidden => 403,
            Self::Missing => 404,
            Self::Unsupported => 501,
        }
    }
}

impl From<DiagnosticReadReason> for DiagnosticHttpReason {
    fn from(reason: DiagnosticReadReason) -> Self {
        match reason {
            DiagnosticReadReason::Disabled => Self::Disabled,
            DiagnosticReadReason::QueryDisabled => Self::QueryDisabled,
            DiagnosticReadReason::UnsupportedSchema => Self::UnsupportedSchema,
            DiagnosticReadReason::InvalidQuery => Self::InvalidQuery,
            DiagnosticReadReason::InvalidCursor => Self::InvalidCursor,
            DiagnosticReadReason::Busy => Self::Busy,
            DiagnosticReadReason::SnapshotCapacity => Self::SnapshotCapacity,
            DiagnosticReadReason::InvalidLimits => Self::InvalidLimits,
            DiagnosticReadReason::AllocationFailed => Self::AllocationFailed,
            DiagnosticReadReason::Internal => Self::Internal,
        }
    }
}

/// Error envelope returned by diagnostic HTTP routes.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticHttpError {
    /// Safe machine-readable error details.
    pub error: DiagnosticHttpErrorDetails,
}

/// Error details without query values, credentials, or retained evidence.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticHttpErrorDetails {
    /// Stable operation failure reason.
    pub reason: DiagnosticHttpReason,
    /// Supported version, supplied for an unsupported snapshot schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supported_schema_version: Option<u32>,
}

impl DiagnosticHttpError {
    /// Construct a canonical error body, including schema negotiation metadata.
    pub fn new(reason: DiagnosticHttpReason) -> Self {
        Self {
            error: DiagnosticHttpErrorDetails {
                reason,
                supported_schema_version: (reason == DiagnosticHttpReason::UnsupportedSchema)
                    .then_some(DIAGNOSTIC_SCHEMA_VERSION),
            },
        }
    }

    /// Decode a bounded known error whose reason agrees with the HTTP status.
    /// Unknown reasons, malformed bodies, and status mismatches require a consumer fallback.
    /// Unknown fields are accepted for additive protocol evolution.
    pub fn from_response(status: u16, body: &[u8]) -> Option<Self> {
        if body.len() > MAX_DIAGNOSTIC_HTTP_ERROR_BYTES {
            return None;
        }
        let value: Self = serde_json::from_slice(body).ok()?;
        (value.error.reason.http_status() == status).then_some(value)
    }
}
