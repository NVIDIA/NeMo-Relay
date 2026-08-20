// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared plugin diagnostic data types.

use std::borrow::Cow;

use serde::{Deserialize, Deserializer, Serialize};

use crate::Json;

/// Export target kind presented to an activation policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct ExportActivationTargetKind(Cow<'static, str>);

impl ExportActivationTargetKind {
    /// OpenTelemetry trace destination.
    pub const OTLP_TRACE: Self = Self(Cow::Borrowed("nemo_relay.otlp.trace"));
    /// OpenTelemetry log destination.
    pub const OTLP_LOG: Self = Self(Cow::Borrowed("nemo_relay.otlp.log"));
    /// OpenTelemetry metric destination.
    pub const OTLP_METRIC: Self = Self(Cow::Borrowed("nemo_relay.otlp.metric"));
    /// Local ATOF file sink.
    pub const ATOF_FILE: Self = Self(Cow::Borrowed("nemo_relay.atof.file"));
    /// Remote ATOF stream sink.
    pub const ATOF_STREAM: Self = Self(Cow::Borrowed("nemo_relay.atof.stream"));
    /// Local ATIF file destination.
    pub const ATIF_FILE: Self = Self(Cow::Borrowed("nemo_relay.atif.file"));
    /// Remote ATIF HTTP storage.
    pub const ATIF_HTTP: Self = Self(Cow::Borrowed("nemo_relay.atif.http"));
    /// Remote ATIF S3-compatible storage.
    pub const ATIF_S3: Self = Self(Cow::Borrowed("nemo_relay.atif.s3"));

    // Compatibility spellings retained for the 0.8 pre-release SDK surface.
    /// Compatibility alias for [`Self::OTLP_TRACE`].
    #[allow(non_upper_case_globals)]
    pub const OtlpTrace: Self = Self::OTLP_TRACE;
    /// Compatibility alias for [`Self::OTLP_LOG`].
    #[allow(non_upper_case_globals)]
    pub const OtlpLog: Self = Self::OTLP_LOG;
    /// Compatibility alias for [`Self::OTLP_METRIC`].
    #[allow(non_upper_case_globals)]
    pub const OtlpMetric: Self = Self::OTLP_METRIC;
    /// Compatibility alias for [`Self::ATOF_FILE`].
    #[allow(non_upper_case_globals)]
    pub const AtofFile: Self = Self::ATOF_FILE;
    /// Compatibility alias for [`Self::ATOF_STREAM`].
    #[allow(non_upper_case_globals)]
    pub const AtofStream: Self = Self::ATOF_STREAM;
    /// Compatibility alias for [`Self::ATIF_FILE`].
    #[allow(non_upper_case_globals)]
    pub const AtifFile: Self = Self::ATIF_FILE;
    /// Compatibility alias for [`Self::ATIF_HTTP`].
    #[allow(non_upper_case_globals)]
    pub const AtifHttp: Self = Self::ATIF_HTTP;
    /// Compatibility alias for [`Self::ATIF_S3`].
    #[allow(non_upper_case_globals)]
    pub const AtifS3: Self = Self::ATIF_S3;

    /// Creates a custom namespaced export-target kind.
    pub fn new(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into();
        let valid = value.len() <= 255
            && value.contains('.')
            && value.split('.').all(|segment| {
                !segment.is_empty()
                    && segment
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
            });
        if !valid {
            return Err(
                "export activation target kind must be a dot-separated namespaced identifier"
                    .into(),
            );
        }
        Ok(Self(Cow::Owned(value)))
    }

    /// Returns the serialized target-kind name.
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

impl<'de> Deserialize<'de> for ExportActivationTargetKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Input supplied to an export-activation policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportActivationRequest {
    /// Kind of local or remote export target being considered.
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
    /// Construct and activate the exporter target.
    Allow,
    /// Suppress the exporter target for this activation.
    Deny,
}

/// Activation policy attached to one export target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportActivationPolicyConfig {
    /// Plugin kind or dynamic manifest identifier that owns the callback.
    pub provider: String,
    /// Maximum policy evaluation time in milliseconds.
    #[serde(default = "default_export_activation_timeout_millis")]
    pub timeout_millis: u64,
    /// Opaque target-local configuration passed to the policy.
    #[serde(default)]
    pub config: Json,
}

/// Deferred export target registered by one plugin component.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportTargetRegistration {
    /// Component-local identifier, unique within one component instance.
    pub id: String,
    /// Namespaced kind presented to the policy provider.
    pub target_kind: ExportActivationTargetKind,
    /// Optional policy controlling target construction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_policy: Option<ExportActivationPolicyConfig>,
}

/// Minimum accepted policy timeout.
pub const MIN_EXPORT_ACTIVATION_TIMEOUT_MILLIS: u64 = 1_000;
/// Default policy timeout.
pub const DEFAULT_EXPORT_ACTIVATION_TIMEOUT_MILLIS: u64 = 30_000;
/// Maximum accepted policy timeout.
pub const MAX_EXPORT_ACTIVATION_TIMEOUT_MILLIS: u64 = 300_000;

const fn default_export_activation_timeout_millis() -> u64 {
    DEFAULT_EXPORT_ACTIVATION_TIMEOUT_MILLIS
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
