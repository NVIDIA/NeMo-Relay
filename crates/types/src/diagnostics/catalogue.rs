// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed-size authoring catalogue; stable wire names never depend on Rust layout.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use super::Classification;

/// Supported general cause, independent of health, retries, and operation outcome.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    strum::EnumString,
    strum::IntoStaticStr,
    enumset::EnumSetType,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
#[enumset(no_super_impls)]
pub enum FailureClass {
    /// The supplied input is invalid.
    InvalidInput,
    /// The operation is invalid in the observed state.
    InvalidState,
    /// The requested resource was not found.
    NotFound,
    /// The operation conflicts with an existing resource or state.
    Conflict,
    /// Authentication or the existing authentication classification failed.
    Authentication,
    /// Structured evidence establishes denied permission.
    PermissionDenied,
    /// The requested capability is unsupported.
    Unsupported,
    /// Observed versions or protocols are incompatible.
    Incompatible,
    /// The operation exceeded its deadline.
    Timeout,
    /// The dependency or service was unavailable.
    Unavailable,
    /// An observed capacity limit was reached.
    ResourceExhausted,
    /// Structured evidence establishes an I/O failure.
    Io,
    /// A callback or operation failed during execution.
    ExecutionFailed,
    /// An existing policy refused the operation.
    PolicyRejected,
    /// The source reported an internal failure.
    Internal,
    /// A failure was observed without a supported more specific cause.
    Unknown,
}

impl FailureClass {
    /// Interpret an exact supported canonical name without allocating.
    #[must_use]
    pub fn from_canonical_name(name: &str) -> Option<Self> {
        name.parse().ok()
    }
    /// Stable portable name, also available in const contexts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidInput => "invalid_input",
            Self::InvalidState => "invalid_state",
            Self::NotFound => "not_found",
            Self::Conflict => "conflict",
            Self::Authentication => "authentication",
            Self::PermissionDenied => "permission_denied",
            Self::Unsupported => "unsupported",
            Self::Incompatible => "incompatible",
            Self::Timeout => "timeout",
            Self::Unavailable => "unavailable",
            Self::ResourceExhausted => "resource_exhausted",
            Self::Io => "io",
            Self::ExecutionFailed => "execution_failed",
            Self::PolicyRejected => "policy_rejected",
            Self::Internal => "internal",
            Self::Unknown => "unknown",
        }
    }
}

/// Top-level owning subsystem, derived from its catalogue code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticDomain {
    /// CLI-owned execution and transport operations.
    Cli,
    /// Plugin configuration and activation.
    Plugin,
    /// Operational logging and its sinks.
    Logging,
}

/// Semantic family of related plugin diagnostic codes.
///
/// Family is derived from a code's catalogue ownership. Execution mechanism is
/// separately supplied by producers through [`PluginExecution`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum PluginDiagnosticFamily {
    /// Shared plugin-operation codes applicable across execution mechanisms.
    Common,
}

/// Producer-authored plugin execution mechanism, excluded from counter identity.
///
/// Phase-one core activation recording leaves execution unset. These values
/// support later native/worker producer integrations; they do not describe
/// currently implemented automatic collection.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    strum::EnumString,
    strum::IntoStaticStr,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum PluginExecution {
    /// The plugin executes through the native ABI.
    NativeAbi,
    /// The plugin executes through the dynamic-worker protocol.
    WorkerProtocol,
}

impl PluginExecution {
    /// Interpret an exact supported canonical name without allocating.
    #[must_use]
    pub fn from_canonical_name(name: &str) -> Option<Self> {
        name.parse().ok()
    }
    /// Stable portable name, also available in const contexts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeAbi => "native_abi",
            Self::WorkerProtocol => "worker_protocol",
        }
    }
}

impl From<PluginExecution> for Cow<'static, str> {
    fn from(value: PluginExecution) -> Self {
        Self::Borrowed(value.as_str())
    }
}

/// Observed plugin lifecycle phase or policy-gate context, independent of
/// classification.
///
/// The phase-one core adapter automatically supplies only [`Self::Activation`].
/// Other values are authoring/wire contracts for producers integrated later;
/// declaring a value does not mean its stage is currently observed by Relay.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    strum::EnumString,
    strum::IntoStaticStr,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum PluginPhase {
    /// Validation lifecycle phase.
    Validation,
    /// Activation lifecycle phase.
    Activation,
    /// Runtime lifecycle phase.
    Runtime,
    /// Host policy evaluation that permits or blocks plugin realization.
    Policy,
}

impl PluginPhase {
    /// Interpret an exact supported canonical name without allocating.
    #[must_use]
    pub fn from_canonical_name(name: &str) -> Option<Self> {
        name.parse().ok()
    }
    /// Stable portable name, also available in const contexts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::Activation => "activation",
            Self::Runtime => "runtime",
            Self::Policy => "policy",
        }
    }
}

impl From<PluginPhase> for Cow<'static, str> {
    fn from(value: PluginPhase) -> Self {
        Self::Borrowed(value.as_str())
    }
}

/// Plugin failures supported by phase-one activation recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(strum::EnumIter))]
pub enum CommonPluginFailureCode {
    /// Observation identified by `plugin.configuration_invalid`.
    /// Generic configuration errors may include file-read failures, so their
    /// cause remains unknown without structured producer evidence.
    ConfigurationInvalid,
    /// Observation identified by `plugin.registration_failed`.
    RegistrationFailed,
    /// Observation identified by `plugin.state_conflict`.
    StateConflict,
    /// Observation identified by `plugin.resource_not_found`.
    ResourceNotFound,
    /// Observation identified by `plugin.serialization_failed`.
    SerializationFailed,
    /// Observation identified by `plugin.internal`.
    Internal,
}

impl CommonPluginFailureCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::ConfigurationInvalid => "plugin.configuration_invalid",
            Self::RegistrationFailed => "plugin.registration_failed",
            Self::StateConflict => "plugin.state_conflict",
            Self::ResourceNotFound => "plugin.resource_not_found",
            Self::SerializationFailed => "plugin.serialization_failed",
            Self::Internal => "plugin.internal",
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::ConfigurationInvalid => &[FailureClass::Unknown, FailureClass::InvalidInput],
            Self::RegistrationFailed => &[
                FailureClass::Unknown,
                FailureClass::InvalidInput,
                FailureClass::Conflict,
                FailureClass::Internal,
                FailureClass::ResourceExhausted,
                FailureClass::Unavailable,
            ],
            Self::StateConflict => &[FailureClass::Conflict],
            Self::ResourceNotFound => &[FailureClass::NotFound],
            Self::SerializationFailed => &[
                FailureClass::Unknown,
                FailureClass::InvalidInput,
                FailureClass::Internal,
            ],
            Self::Internal => &[FailureClass::Internal],
        }
    }
}

/// Initial Logging failure vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(strum::EnumIter))]
pub enum LoggingFailureCode {
    /// Observation identified by `logging.initialize_failed`.
    InitializeFailed,
    /// Observation identified by `logging.shutdown_failed`.
    ShutdownFailed,
    /// Observation identified by `logging.projection_unavailable`.
    ProjectionUnavailable,
    /// Observation identified by `logging.sink.initialize_failed`.
    SinkInitializeFailed,
    /// Observation identified by `logging.sink.record_dropped`.
    SinkRecordDropped,
    /// Observation identified by `logging.sink.flush_dropped`.
    SinkFlushDropped,
    /// Observation identified by `logging.sink.flush_failed`.
    SinkFlushFailed,
    /// Observation identified by `logging.sink.write_failed`.
    SinkWriteFailed,
    /// Observation identified by `logging.sink.shutdown_failed`.
    SinkShutdownFailed,
}

impl LoggingFailureCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::InitializeFailed => "logging.initialize_failed",
            Self::ShutdownFailed => "logging.shutdown_failed",
            Self::ProjectionUnavailable => "logging.projection_unavailable",
            Self::SinkInitializeFailed => "logging.sink.initialize_failed",
            Self::SinkRecordDropped => "logging.sink.record_dropped",
            Self::SinkFlushDropped => "logging.sink.flush_dropped",
            Self::SinkFlushFailed => "logging.sink.flush_failed",
            Self::SinkWriteFailed => "logging.sink.write_failed",
            Self::SinkShutdownFailed => "logging.sink.shutdown_failed",
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::InitializeFailed => &[
                FailureClass::Unknown,
                FailureClass::Io,
                FailureClass::InvalidInput,
                FailureClass::ResourceExhausted,
            ],
            Self::ShutdownFailed => &[
                FailureClass::Unknown,
                FailureClass::Io,
                FailureClass::Timeout,
            ],
            Self::ProjectionUnavailable => {
                &[FailureClass::Unknown, FailureClass::ResourceExhausted]
            }
            Self::SinkInitializeFailed => &[
                FailureClass::Unknown,
                FailureClass::Io,
                FailureClass::InvalidInput,
            ],
            Self::SinkRecordDropped => &[FailureClass::ResourceExhausted],
            Self::SinkFlushDropped => &[FailureClass::ResourceExhausted],
            Self::SinkFlushFailed => &[
                FailureClass::Unknown,
                FailureClass::Io,
                FailureClass::Unavailable,
                FailureClass::Timeout,
            ],
            Self::SinkWriteFailed => &[
                FailureClass::Unknown,
                FailureClass::Io,
                FailureClass::Unavailable,
                FailureClass::Timeout,
            ],
            Self::SinkShutdownFailed => &[
                FailureClass::Unknown,
                FailureClass::Io,
                FailureClass::Timeout,
            ],
        }
    }
}

/// Plugin failure hierarchy for integrated producers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum PluginFailureCode {
    /// Common plugin failures.
    Common(CommonPluginFailureCode),
}

impl PluginFailureCode {
    /// Plugin catalogue family selected by this code.
    #[must_use]
    pub const fn family(self) -> PluginDiagnosticFamily {
        match self {
            Self::Common(_) => PluginDiagnosticFamily::Common,
        }
    }
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Common(code) => code.canonical_name(),
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::Common(code) => code.allowed_classes(),
        }
    }
}

/// Failures at integrated CLI operation boundaries; causes come from producers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(strum::EnumIter))]
pub enum CliFailureCode {
    /// Failure to clean up a supervised child tree.
    ChildCleanup,
    /// Failure while reading or decoding an upstream response body.
    UpstreamResponse,
    /// Observation identified by `cli.command_failed`.
    Command,
    /// Observation identified by `cli.configuration_failed`.
    Configuration,
    /// Observation identified by `cli.gateway.bind_failed`.
    GatewayBind,
    /// Observation identified by `cli.gateway.serve_failed`.
    GatewayServe,
    /// Observation identified by `cli.acquire_failed`.
    Acquire,
    /// Observation identified by `cli.child.spawn_failed`.
    ChildSpawn,
    /// Observation identified by `cli.child.wait_failed`.
    ChildWait,
    /// Observation identified by `cli.hook.forward_failed`.
    HookForward,
    /// Observation identified by `cli.upstream.transport_failed`.
    UpstreamTransport,
}
impl CliFailureCode {
    /// Stable catalogue name.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::ChildCleanup => "cli.child.cleanup_failed",
            Self::UpstreamResponse => "cli.upstream.response_failed",
            Self::Command => "cli.command_failed",
            Self::Configuration => "cli.configuration_failed",
            Self::GatewayBind => "cli.gateway.bind_failed",
            Self::GatewayServe => "cli.gateway.serve_failed",
            Self::Acquire => "cli.acquire_failed",
            Self::ChildSpawn => "cli.child.spawn_failed",
            Self::ChildWait => "cli.child.wait_failed",
            Self::HookForward => "cli.hook.forward_failed",
            Self::UpstreamTransport => "cli.upstream.transport_failed",
        }
    }
    /// Causes admissible at CLI boundaries; producers must not infer from text.
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        &[
            FailureClass::Unknown,
            FailureClass::Io,
            FailureClass::InvalidInput,
            FailureClass::Timeout,
            FailureClass::Authentication,
            FailureClass::PolicyRejected,
            FailureClass::Unavailable,
            FailureClass::PermissionDenied,
            FailureClass::NotFound,
            FailureClass::Conflict,
            FailureClass::ExecutionFailed,
        ]
    }
}

/// Failure event hierarchy for integrated producers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum FailureCode {
    /// CLI execution and transport failures.
    Cli(CliFailureCode),
    /// Plugin failures.
    Plugin(PluginFailureCode),
    /// Logging failures.
    Logging(LoggingFailureCode),
}

impl FailureCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Cli(code) => code.canonical_name(),
            Self::Plugin(code) => code.canonical_name(),
            Self::Logging(code) => code.canonical_name(),
        }
    }
    /// Owning subsystem derived from the code.
    #[must_use]
    pub const fn domain(self) -> DiagnosticDomain {
        match self {
            Self::Cli(_) => DiagnosticDomain::Cli,
            Self::Plugin(_) => DiagnosticDomain::Plugin,
            Self::Logging(_) => DiagnosticDomain::Logging,
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::Cli(code) => code.allowed_classes(),
            Self::Plugin(code) => code.allowed_classes(),
            Self::Logging(code) => code.allowed_classes(),
        }
    }
}

/// Explicit successful operations; successes do not recover or clear failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(strum::EnumIter))]
pub enum SuccessCode {
    /// An upstream transport returned headers, independently of provider status.
    CliUpstreamHeadersReceived,
    /// An upstream response body was fully read, independently of provider status.
    CliUpstreamBodyCompleted,
    /// A supervised child tree was cleaned up.
    CliChildCleaned,
    /// Observation identified by `cli.command_completed`.
    CliCommandCompleted,
    /// Observation identified by `cli.gateway.bound`.
    CliGatewayBound,
    /// Observation identified by `cli.gateway.stopped`.
    CliGatewayStopped,
    /// Observation identified by `cli.acquired`.
    CliAcquired,
    /// Observation identified by `cli.child.spawned`.
    CliChildSpawned,
    /// Observation identified by `cli.child.waited`.
    CliChildWaited,
    /// Observation identified by `cli.hook.forwarded`.
    CliHookForwarded,

    /// Observation identified by `logging.initialized`.
    LoggingInitialized,
    /// Observation identified by `logging.shutdown_completed`.
    LoggingShutdownCompleted,
    /// Observation identified by `logging.sink.initialized`.
    SinkInitialized,
}

impl SuccessCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::CliUpstreamHeadersReceived => "cli.upstream.headers_received",
            Self::CliUpstreamBodyCompleted => "cli.upstream.body_completed",
            Self::CliChildCleaned => "cli.child.cleaned",
            Self::CliCommandCompleted => "cli.command_completed",
            Self::CliGatewayBound => "cli.gateway.bound",
            Self::CliGatewayStopped => "cli.gateway.stopped",
            Self::CliAcquired => "cli.acquired",
            Self::CliChildSpawned => "cli.child.spawned",
            Self::CliChildWaited => "cli.child.waited",
            Self::CliHookForwarded => "cli.hook.forwarded",

            Self::LoggingInitialized => "logging.initialized",
            Self::LoggingShutdownCompleted => "logging.shutdown_completed",
            Self::SinkInitialized => "logging.sink.initialized",
        }
    }
}

/// Bounded non-failure observations; they never acquire a failure class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(strum::EnumIter))]
pub enum NoticeCode {
    /// Observation identified by `cli.nonzero_exit`.
    CliNonzeroExit,
    /// Observation identified by `cli.policy_rejected`.
    CliPolicyRejected,
    /// Observation identified by `cli.child.exited`.
    CliChildExited,

    /// Observation identified by `plugin.configuration_inherited`.
    ConfigurationInherited,
}

impl NoticeCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::CliNonzeroExit => "cli.nonzero_exit",
            Self::CliPolicyRejected => "cli.policy_rejected",
            Self::CliChildExited => "cli.child.exited",

            Self::ConfigurationInherited => "plugin.configuration_inherited",
        }
    }
}

/// Fixed-size canonical authoring value with one authoritative hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum BuiltinDiagnosticCode {
    /// Failure observation.
    Failure(FailureCode),
    /// Success observation.
    Success(SuccessCode),
    /// Notice observation.
    Notice(NoticeCode),
}

impl BuiltinDiagnosticCode {
    /// Stable name used on the wire and in operational projections.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Failure(code) => code.canonical_name(),
            Self::Success(code) => code.canonical_name(),
            Self::Notice(code) => code.canonical_name(),
        }
    }
    /// Parent domain derived from the individual catalogue entry.
    ///
    /// Outcome does not determine ownership. Match each success/notice explicitly
    /// so adding another entry requires choosing its owning domain.
    #[must_use]
    pub const fn domain(self) -> DiagnosticDomain {
        match self {
            Self::Failure(code) => code.domain(),
            Self::Success(
                SuccessCode::CliUpstreamHeadersReceived
                | SuccessCode::CliUpstreamBodyCompleted
                | SuccessCode::CliChildCleaned
                | SuccessCode::CliCommandCompleted
                | SuccessCode::CliGatewayBound
                | SuccessCode::CliGatewayStopped
                | SuccessCode::CliAcquired
                | SuccessCode::CliChildSpawned
                | SuccessCode::CliChildWaited
                | SuccessCode::CliHookForwarded,
            )
            | Self::Notice(
                NoticeCode::CliNonzeroExit
                | NoticeCode::CliPolicyRejected
                | NoticeCode::CliChildExited,
            ) => DiagnosticDomain::Cli,
            Self::Success(
                SuccessCode::LoggingInitialized
                | SuccessCode::LoggingShutdownCompleted
                | SuccessCode::SinkInitialized,
            ) => DiagnosticDomain::Logging,
            Self::Notice(NoticeCode::ConfigurationInherited) => DiagnosticDomain::Plugin,
        }
    }
    /// Plugin catalogue family, independent of the execution mechanism.
    ///
    /// Each success/notice explicitly selects a family or None; future plugin
    /// successes must choose their family when their producer is integrated.
    #[must_use]
    pub const fn plugin_family(self) -> Option<PluginDiagnosticFamily> {
        match self {
            Self::Failure(FailureCode::Plugin(code)) => Some(code.family()),
            Self::Notice(NoticeCode::ConfigurationInherited) => {
                Some(PluginDiagnosticFamily::Common)
            }
            Self::Failure(FailureCode::Cli(_))
            | Self::Success(
                SuccessCode::CliUpstreamHeadersReceived
                | SuccessCode::CliUpstreamBodyCompleted
                | SuccessCode::CliChildCleaned
                | SuccessCode::CliCommandCompleted
                | SuccessCode::CliGatewayBound
                | SuccessCode::CliGatewayStopped
                | SuccessCode::CliAcquired
                | SuccessCode::CliChildSpawned
                | SuccessCode::CliChildWaited
                | SuccessCode::CliHookForwarded,
            )
            | Self::Notice(
                NoticeCode::CliNonzeroExit
                | NoticeCode::CliPolicyRejected
                | NoticeCode::CliChildExited,
            )
            | Self::Failure(FailureCode::Logging(_))
            | Self::Success(
                SuccessCode::LoggingInitialized
                | SuccessCode::LoggingShutdownCompleted
                | SuccessCode::SinkInitialized,
            ) => None,
        }
    }
    /// Permitted causes; success and notice codes accept no failure class.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::Failure(code) => code.allowed_classes(),
            _ => &[],
        }
    }
    /// Check a class against the entry-specific catalogue contract.
    #[must_use]
    pub fn allows_class(self, class: Option<FailureClass>) -> bool {
        match (self, class) {
            (Self::Failure(_), Some(class)) => self.allowed_classes().contains(&class),
            (Self::Success(_) | Self::Notice(_), None) => true,
            _ => false,
        }
    }
    /// All implemented catalogue entries.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Failure(FailureCode::Cli(CliFailureCode::ChildCleanup)),
            Self::Failure(FailureCode::Cli(CliFailureCode::UpstreamResponse)),
            Self::Success(SuccessCode::CliUpstreamHeadersReceived),
            Self::Success(SuccessCode::CliUpstreamBodyCompleted),
            Self::Success(SuccessCode::CliChildCleaned),
            Self::Failure(FailureCode::Cli(CliFailureCode::Command)),
            Self::Failure(FailureCode::Cli(CliFailureCode::Configuration)),
            Self::Failure(FailureCode::Cli(CliFailureCode::GatewayBind)),
            Self::Failure(FailureCode::Cli(CliFailureCode::GatewayServe)),
            Self::Failure(FailureCode::Cli(CliFailureCode::Acquire)),
            Self::Failure(FailureCode::Cli(CliFailureCode::ChildSpawn)),
            Self::Failure(FailureCode::Cli(CliFailureCode::ChildWait)),
            Self::Failure(FailureCode::Cli(CliFailureCode::HookForward)),
            Self::Failure(FailureCode::Cli(CliFailureCode::UpstreamTransport)),
            Self::Success(SuccessCode::CliCommandCompleted),
            Self::Success(SuccessCode::CliGatewayBound),
            Self::Success(SuccessCode::CliGatewayStopped),
            Self::Success(SuccessCode::CliAcquired),
            Self::Success(SuccessCode::CliChildSpawned),
            Self::Success(SuccessCode::CliChildWaited),
            Self::Success(SuccessCode::CliHookForwarded),
            Self::Notice(NoticeCode::CliNonzeroExit),
            Self::Notice(NoticeCode::CliPolicyRejected),
            Self::Notice(NoticeCode::CliChildExited),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::ConfigurationInvalid,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::RegistrationFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::StateConflict,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::ResourceNotFound,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::SerializationFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::Internal,
            ))),
            Self::Failure(FailureCode::Logging(LoggingFailureCode::InitializeFailed)),
            Self::Failure(FailureCode::Logging(LoggingFailureCode::ShutdownFailed)),
            Self::Failure(FailureCode::Logging(
                LoggingFailureCode::ProjectionUnavailable,
            )),
            Self::Failure(FailureCode::Logging(
                LoggingFailureCode::SinkInitializeFailed,
            )),
            Self::Failure(FailureCode::Logging(LoggingFailureCode::SinkRecordDropped)),
            Self::Failure(FailureCode::Logging(LoggingFailureCode::SinkFlushDropped)),
            Self::Failure(FailureCode::Logging(LoggingFailureCode::SinkFlushFailed)),
            Self::Failure(FailureCode::Logging(LoggingFailureCode::SinkWriteFailed)),
            Self::Failure(FailureCode::Logging(LoggingFailureCode::SinkShutdownFailed)),
            Self::Success(SuccessCode::LoggingInitialized),
            Self::Success(SuccessCode::LoggingShutdownCompleted),
            Self::Success(SuccessCode::SinkInitialized),
            Self::Notice(NoticeCode::ConfigurationInherited),
        ]
    }
    /// Look up a built-in code without accepting prefixes or inspecting messages.
    #[must_use]
    pub fn from_canonical_name(name: &str) -> Option<Self> {
        Self::all()
            .iter()
            .copied()
            .find(|code| code.canonical_name() == name)
    }
}

/// A cause is missing, forbidden, or unsupported by the selected catalogue code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassificationError;
impl std::fmt::Display for ClassificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("failure class is invalid for diagnostic code")
    }
}
impl std::error::Error for ClassificationError {}

/// Pure producer behavior: borrow original evidence and return shared classification metadata.
///
/// Owning producer crates implement local adapters. Classification must not record,
/// log, retry, change health, or modify the source operation's outcome.
pub trait DiagnosticClassifier<E: ?Sized> {
    /// Interpret supported structured evidence without retaining or altering it.
    fn classify(&self, error: &E) -> Classification;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use strum::IntoEnumIterator;

    #[test]
    fn catalogue_contains_every_leaf_enum_variant_exactly_once() {
        // EnumIter derives directly from each leaf definition, independently
        // of all(). Adding a leaf variant without registering it must fail.
        let expected: BTreeSet<_> = CommonPluginFailureCode::iter()
            .map(|code| {
                BuiltinDiagnosticCode::Failure(FailureCode::Plugin(PluginFailureCode::Common(code)))
            })
            .chain(
                LoggingFailureCode::iter()
                    .map(|code| BuiltinDiagnosticCode::Failure(FailureCode::Logging(code))),
            )
            .chain(
                CliFailureCode::iter()
                    .map(|code| BuiltinDiagnosticCode::Failure(FailureCode::Cli(code))),
            )
            .chain(SuccessCode::iter().map(BuiltinDiagnosticCode::Success))
            .chain(NoticeCode::iter().map(BuiltinDiagnosticCode::Notice))
            .collect();
        let actual: BTreeSet<_> = BuiltinDiagnosticCode::all().iter().copied().collect();
        assert_eq!(
            actual.len(),
            BuiltinDiagnosticCode::all().len(),
            "duplicate catalogue entry"
        );
        assert_eq!(
            actual, expected,
            "catalogue must contain every leaf variant"
        );
    }
}
