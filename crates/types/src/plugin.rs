// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared plugin diagnostic data types.

use serde::{Deserialize, Serialize};

use crate::Json;

/// Remote exporter target presented to an activation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ExportActivationTargetKind {
    /// OpenTelemetry trace destination.
    OtlpTrace,
    /// OpenTelemetry log destination.
    OtlpLog,
    /// OpenTelemetry metric destination.
    OtlpMetric,
    /// Remote ATOF stream sink.
    AtofStream,
    /// Remote ATIF HTTP storage.
    AtifHttp,
    /// Remote ATIF S3-compatible storage.
    AtifS3,
}

/// Input supplied to an export-activation policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportActivationRequest {
    /// Kind of remote exporter target being considered.
    pub target_kind: ExportActivationTargetKind,
    /// Opaque target-local policy configuration.
    #[serde(default)]
    pub config: Json,
}

/// Decision returned by an export-activation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ExportActivationDecision {
    /// Construct and activate the remote exporter target.
    Allow,
    /// Suppress the remote exporter target for this activation.
    Deny,
}

/// Diagnostic severity returned by plugin validation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLevel {
    /// Non-fatal compatibility or validation issue.
    Warning,
    /// Fatal validation issue that blocks initialization.
    Error,
}

/// Structured validation diagnostic for plugin validation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ConfigDiagnostic {
    /// Severity level for the diagnostic.
    pub level: DiagnosticLevel,
    /// Stable diagnostic code suitable for machine checks.
    pub code: String,
    /// Optional component identifier associated with the diagnostic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component: Option<String>,
    /// Optional field path associated with the diagnostic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    /// Human-readable diagnostic message.
    pub message: String,
}
