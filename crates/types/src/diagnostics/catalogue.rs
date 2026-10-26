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
    /// Daemon control, runtime, forwarding, and collection operations.
    Daemon,
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
    /// Host-observed loading, validation, execution, and cleanup operations.
    Operations,
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
    /// Plugin cleanup and confirmed process termination.
    Shutdown,
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
            Self::Shutdown => "shutdown",
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
    /// Nonfatal exporter construction failed during plugin activation.
    /// Observation identified by `plugin.exporter_initialization_failed`.
    ExporterInitializationFailed,
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
            Self::ExporterInitializationFailed => "plugin.exporter_initialization_failed",
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
            Self::ExporterInitializationFailed => &[
                FailureClass::InvalidInput,
                FailureClass::PolicyRejected,
                FailureClass::Io,
                FailureClass::Unknown,
            ],
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

/// Failures at observed dynamic-plugin operation boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(strum::EnumIter))]
pub enum PluginOperationFailureCode {
    /// Failure of the plugin artifact_verification operation.
    ArtifactVerification,
    /// Failure of the plugin policy operation.
    Policy,
    /// Failure of the plugin schema_validation operation.
    SchemaValidation,
    /// Failure of the plugin native load operation.
    NativeLoad,
    /// Failure of the plugin native validation operation.
    NativeValidation,
    /// Failure of the plugin native registration operation.
    NativeRegistration,
    /// Failure of the plugin native callback operation.
    NativeCallback,
    /// Failure of the plugin worker launch operation.
    WorkerLaunch,
    /// Failure of the plugin worker handshake operation.
    WorkerHandshake,
    /// Failure of the plugin worker validation operation.
    WorkerValidation,
    /// Failure of the plugin worker registration operation.
    WorkerRegistration,
    /// Failure of the plugin worker transport operation.
    WorkerTransport,
    /// Failure of the plugin worker callback operation.
    WorkerCallback,
    /// Failure of the plugin worker cancellation operation.
    WorkerCancellation,
    /// Failure of the plugin worker shutdown operation.
    WorkerShutdown,
    /// Failure of the plugin worker exit operation.
    WorkerExit,
}
impl PluginOperationFailureCode {
    /// Stable operation-specific name.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::ArtifactVerification => "plugin.artifact_verification_failed",
            Self::Policy => "plugin.policy_failed",
            Self::SchemaValidation => "plugin.schema_validation_failed",
            Self::NativeLoad => "plugin.native.load_failed",
            Self::NativeValidation => "plugin.native.validation_failed",
            Self::NativeRegistration => "plugin.native.registration_failed",
            Self::NativeCallback => "plugin.native.callback_failed",
            Self::WorkerLaunch => "plugin.worker.launch_failed",
            Self::WorkerHandshake => "plugin.worker.handshake_failed",
            Self::WorkerValidation => "plugin.worker.validation_failed",
            Self::WorkerRegistration => "plugin.worker.registration_failed",
            Self::WorkerTransport => "plugin.worker.transport_failed",
            Self::WorkerCallback => "plugin.worker.callback_failed",
            Self::WorkerCancellation => "plugin.worker.cancellation_failed",
            Self::WorkerShutdown => "plugin.worker.shutdown_failed",
            Self::WorkerExit => "plugin.worker.exit_failed",
        }
    }
    /// A producer must establish the cause from structured operation evidence.
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        &[
            FailureClass::Unknown,
            FailureClass::InvalidInput,
            FailureClass::InvalidState,
            FailureClass::NotFound,
            FailureClass::Conflict,
            FailureClass::Authentication,
            FailureClass::PermissionDenied,
            FailureClass::Unsupported,
            FailureClass::Incompatible,
            FailureClass::Timeout,
            FailureClass::Unavailable,
            FailureClass::ResourceExhausted,
            FailureClass::Io,
            FailureClass::ExecutionFailed,
            FailureClass::PolicyRejected,
            FailureClass::Internal,
        ]
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
    /// Host-observed operations across native and worker execution.
    Operation(PluginOperationFailureCode),
    /// Common plugin failures.
    Common(CommonPluginFailureCode),
}

impl PluginFailureCode {
    /// Plugin catalogue family selected by this code.
    #[must_use]
    pub const fn family(self) -> PluginDiagnosticFamily {
        match self {
            Self::Operation(_) => PluginDiagnosticFamily::Operations,
            Self::Common(_) => PluginDiagnosticFamily::Common,
        }
    }
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Operation(code) => code.canonical_name(),
            Self::Common(code) => code.canonical_name(),
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::Operation(code) => code.allowed_classes(),
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

/// Failures observed by daemon operation owners, independent of health policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(strum::EnumIter))]
pub enum DaemonFailureCode {
    /// Observation identified by `daemon.registration_failed`.
    Registration,
    /// Observation identified by `daemon.activation_reported_failed`.
    ActivationReported,
    /// Observation identified by `daemon.activation_deadline_exceeded`.
    ActivationDeadline,
    /// Observation identified by `daemon.readiness_failed`.
    Readiness,
    /// Observation identified by `daemon.control_lost`.
    ControlLost,
    /// Observation identified by `daemon.forward_failed`.
    Forward,
    /// Observation identified by `daemon.collection_failed`.
    Collection,
    /// Observation identified by `daemon.worker.initialize_failed`.
    WorkerInitialize,
    /// Observation identified by `daemon.worker.listener_failed`.
    WorkerListener,
    /// Observation identified by `daemon.worker.teardown_failed`.
    WorkerTeardown,
}
impl DaemonFailureCode {
    /// Stable wire name.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Registration => "daemon.registration_failed",
            Self::ActivationReported => "daemon.activation_reported_failed",
            Self::ActivationDeadline => "daemon.activation_deadline_exceeded",
            Self::Readiness => "daemon.readiness_failed",
            Self::ControlLost => "daemon.control_lost",
            Self::Forward => "daemon.forward_failed",
            Self::Collection => "daemon.collection_failed",
            Self::WorkerInitialize => "daemon.worker.initialize_failed",
            Self::WorkerListener => "daemon.worker.listener_failed",
            Self::WorkerTeardown => "daemon.worker.teardown_failed",
        }
    }
    /// Structured causes accepted from the owning operation.
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        &[
            FailureClass::Unknown,
            FailureClass::Io,
            FailureClass::Timeout,
            FailureClass::Unavailable,
            FailureClass::Authentication,
            FailureClass::InvalidInput,
            FailureClass::InvalidState,
            FailureClass::ResourceExhausted,
            FailureClass::Unsupported,
            FailureClass::PermissionDenied,
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
    /// Daemon-owned failures.
    Daemon(DaemonFailureCode),
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
            Self::Daemon(code) => code.canonical_name(),
            Self::Plugin(code) => code.canonical_name(),
            Self::Logging(code) => code.canonical_name(),
        }
    }
    /// Owning subsystem derived from the code.
    #[must_use]
    pub const fn domain(self) -> DiagnosticDomain {
        match self {
            Self::Cli(_) => DiagnosticDomain::Cli,
            Self::Daemon(_) => DiagnosticDomain::Daemon,
            Self::Plugin(_) => DiagnosticDomain::Plugin,
            Self::Logging(_) => DiagnosticDomain::Logging,
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::Cli(code) => code.allowed_classes(),
            Self::Daemon(code) => code.allowed_classes(),
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
    /// Observation identified by `plugin.native.loaded`.
    PluginNativeLoaded,
    /// Observation identified by `plugin.native.validated`.
    PluginNativeValidated,
    /// Observation identified by `plugin.native.registered`.
    PluginNativeRegistered,
    /// Observation identified by `plugin.worker.started`.
    PluginWorkerStarted,
    /// Observation identified by `plugin.worker.handshake_completed`.
    PluginWorkerHandshakeCompleted,
    /// Observation identified by `plugin.worker.validated`.
    PluginWorkerValidated,
    /// Observation identified by `plugin.worker.registered`.
    PluginWorkerRegistered,
    /// Observation identified by `plugin.worker.process_cleaned`.
    PluginWorkerProcessCleaned,

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
            Self::PluginNativeLoaded => "plugin.native.loaded",
            Self::PluginNativeValidated => "plugin.native.validated",
            Self::PluginNativeRegistered => "plugin.native.registered",
            Self::PluginWorkerStarted => "plugin.worker.started",
            Self::PluginWorkerHandshakeCompleted => "plugin.worker.handshake_completed",
            Self::PluginWorkerValidated => "plugin.worker.validated",
            Self::PluginWorkerRegistered => "plugin.worker.registered",
            Self::PluginWorkerProcessCleaned => "plugin.worker.process_cleaned",

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
                SuccessCode::PluginNativeLoaded
                | SuccessCode::PluginNativeValidated
                | SuccessCode::PluginNativeRegistered
                | SuccessCode::PluginWorkerStarted
                | SuccessCode::PluginWorkerHandshakeCompleted
                | SuccessCode::PluginWorkerValidated
                | SuccessCode::PluginWorkerRegistered
                | SuccessCode::PluginWorkerProcessCleaned,
            ) => DiagnosticDomain::Plugin,
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
            Self::Success(
                SuccessCode::PluginNativeLoaded
                | SuccessCode::PluginNativeValidated
                | SuccessCode::PluginNativeRegistered
                | SuccessCode::PluginWorkerStarted
                | SuccessCode::PluginWorkerHandshakeCompleted
                | SuccessCode::PluginWorkerValidated
                | SuccessCode::PluginWorkerRegistered
                | SuccessCode::PluginWorkerProcessCleaned,
            ) => Some(PluginDiagnosticFamily::Operations),
            Self::Notice(NoticeCode::ConfigurationInherited) => {
                Some(PluginDiagnosticFamily::Common)
            }
            Self::Failure(FailureCode::Daemon(_))
            | Self::Failure(FailureCode::Cli(_))
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
            Self::Success(SuccessCode::PluginNativeLoaded),
            Self::Success(SuccessCode::PluginNativeValidated),
            Self::Success(SuccessCode::PluginNativeRegistered),
            Self::Success(SuccessCode::PluginWorkerStarted),
            Self::Success(SuccessCode::PluginWorkerHandshakeCompleted),
            Self::Success(SuccessCode::PluginWorkerValidated),
            Self::Success(SuccessCode::PluginWorkerRegistered),
            Self::Success(SuccessCode::PluginWorkerProcessCleaned),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::ArtifactVerification,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::Policy,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::SchemaValidation,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::NativeLoad,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::NativeValidation,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::NativeRegistration,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::NativeCallback,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::WorkerLaunch,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::WorkerHandshake,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::WorkerValidation,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::WorkerRegistration,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::WorkerTransport,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::WorkerCallback,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::WorkerCancellation,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::WorkerShutdown,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                PluginOperationFailureCode::WorkerExit,
            ))),
            Self::Failure(FailureCode::Daemon(DaemonFailureCode::Registration)),
            Self::Failure(FailureCode::Daemon(DaemonFailureCode::ActivationReported)),
            Self::Failure(FailureCode::Daemon(DaemonFailureCode::ActivationDeadline)),
            Self::Failure(FailureCode::Daemon(DaemonFailureCode::Readiness)),
            Self::Failure(FailureCode::Daemon(DaemonFailureCode::ControlLost)),
            Self::Failure(FailureCode::Daemon(DaemonFailureCode::Forward)),
            Self::Failure(FailureCode::Daemon(DaemonFailureCode::Collection)),
            Self::Failure(FailureCode::Daemon(DaemonFailureCode::WorkerInitialize)),
            Self::Failure(FailureCode::Daemon(DaemonFailureCode::WorkerListener)),
            Self::Failure(FailureCode::Daemon(DaemonFailureCode::WorkerTeardown)),
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
                CommonPluginFailureCode::ExporterInitializationFailed,
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
            .chain(PluginOperationFailureCode::iter().map(|code| {
                BuiltinDiagnosticCode::Failure(FailureCode::Plugin(PluginFailureCode::Operation(
                    code,
                )))
            }))
            .chain(
                LoggingFailureCode::iter()
                    .map(|code| BuiltinDiagnosticCode::Failure(FailureCode::Logging(code))),
            )
            .chain(
                CliFailureCode::iter()
                    .map(|code| BuiltinDiagnosticCode::Failure(FailureCode::Cli(code))),
            )
            .chain(
                DaemonFailureCode::iter()
                    .map(|code| BuiltinDiagnosticCode::Failure(FailureCode::Daemon(code))),
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
