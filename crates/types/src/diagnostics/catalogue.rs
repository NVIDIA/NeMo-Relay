// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed-size authoring catalogue; stable wire names never depend on Rust layout.

use serde::{Deserialize, Serialize};

/// Supported general cause, independent of health, retries, and operation outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
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
    /// Stable portable name of this cause.
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
    /// Core runtime operations.
    Runtime,
    /// Plugin configuration, native ABI, and dynamic workers.
    Plugin,
    /// Provider and upstream operations.
    Upstream,
    /// Telemetry instrumentation and delivery.
    Observability,
    /// Operational logging and its sinks.
    Logging,
    /// Language bridge callbacks.
    Binding,
}

/// Mechanism within the plugin catalogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum PluginDiagnosticFamily {
    /// Mechanism-independent plugin evidence.
    Common,
    /// Native ABI evidence.
    Native,
    /// Dynamic-plugin worker protocol evidence.
    Worker,
}

/// Telemetry signal; catalogue availability does not enable a pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum Signal {
    /// Telemetry logs.
    Logs,
    /// Trace spans.
    Traces,
    /// Metric measurements.
    Metrics,
}

/// Plugin execution mechanism, retained as context rather than counter identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum PluginExecution {
    /// The plugin executes through the native ABI.
    NativeAbi,
    /// The plugin executes through the dynamic-worker protocol.
    WorkerProtocol,
}

/// Observed plugin lifecycle phase, independent of classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum PluginPhase {
    /// Validation lifecycle phase.
    Validation,
    /// Activation lifecycle phase.
    Activation,
    /// Runtime lifecycle phase.
    Runtime,
    /// Policy lifecycle phase.
    Policy,
}

/// Initial Runtime failure vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum RuntimeFailureCode {
    /// Observation identified by `runtime.already_exists`.
    AlreadyExists,
    /// Observation identified by `runtime.not_found`.
    NotFound,
    /// Observation identified by `runtime.invalid_argument`.
    InvalidArgument,
    /// Observation identified by `runtime.scope_stack_empty`.
    ScopeStackEmpty,
    /// Observation identified by `runtime.internal`.
    Internal,
    /// Observation identified by `runtime.remote_error`.
    RemoteError,
}

impl RuntimeFailureCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::AlreadyExists => "runtime.already_exists",
            Self::NotFound => "runtime.not_found",
            Self::InvalidArgument => "runtime.invalid_argument",
            Self::ScopeStackEmpty => "runtime.scope_stack_empty",
            Self::Internal => "runtime.internal",
            Self::RemoteError => "runtime.remote_error",
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::AlreadyExists => &[FailureClass::Conflict],
            Self::NotFound => &[FailureClass::NotFound],
            Self::InvalidArgument => &[FailureClass::InvalidInput],
            Self::ScopeStackEmpty => &[FailureClass::InvalidState],
            Self::Internal => &[FailureClass::Internal],
            Self::RemoteError => &[FailureClass::Unknown],
        }
    }
}

/// Initial Upstream failure vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum UpstreamFailureCode {
    /// Observation identified by `upstream.connection_failed`.
    ConnectionFailed,
    /// Observation identified by `upstream.timed_out`.
    TimedOut,
    /// Observation identified by `upstream.status_rejected`.
    StatusRejected,
    /// Observation identified by `upstream.context_limit`.
    ContextLimit,
    /// Observation identified by `upstream.model_unavailable`.
    ModelUnavailable,
    /// Observation identified by `upstream.authentication_failed`.
    AuthenticationFailed,
    /// Observation identified by `upstream.request_invalid`.
    RequestInvalid,
    /// Observation identified by `upstream.failed`.
    Failed,
}

impl UpstreamFailureCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::ConnectionFailed => "upstream.connection_failed",
            Self::TimedOut => "upstream.timed_out",
            Self::StatusRejected => "upstream.status_rejected",
            Self::ContextLimit => "upstream.context_limit",
            Self::ModelUnavailable => "upstream.model_unavailable",
            Self::AuthenticationFailed => "upstream.authentication_failed",
            Self::RequestInvalid => "upstream.request_invalid",
            Self::Failed => "upstream.failed",
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::ConnectionFailed => &[FailureClass::Unavailable],
            Self::TimedOut => &[FailureClass::Timeout],
            Self::StatusRejected => &[FailureClass::Unknown],
            Self::ContextLimit => &[FailureClass::ResourceExhausted],
            Self::ModelUnavailable => &[FailureClass::Unavailable],
            Self::AuthenticationFailed => &[FailureClass::Authentication],
            Self::RequestInvalid => &[FailureClass::InvalidInput],
            Self::Failed => &[FailureClass::Unknown],
        }
    }
}

/// Initial Binding failure vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum BindingFailureCode {
    /// Observation identified by `binding.callback_exception`.
    CallbackException,
}

impl BindingFailureCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::CallbackException => "binding.callback_exception",
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::CallbackException => &[FailureClass::ExecutionFailed],
        }
    }
}

/// Initial Common plugin  failure vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum CommonPluginFailureCode {
    /// Observation identified by `plugin.configuration_invalid`.
    ConfigurationInvalid,
    /// Observation identified by `plugin.registration_failed`.
    RegistrationFailed,
    /// Observation identified by `plugin.registration_conflict`.
    RegistrationConflict,
    /// Observation identified by `plugin.state_conflict`.
    StateConflict,
    /// Observation identified by `plugin.resource_not_found`.
    ResourceNotFound,
    /// Observation identified by `plugin.invocation_failed`.
    InvocationFailed,
    /// Observation identified by `plugin.serialization_failed`.
    SerializationFailed,
    /// Observation identified by `plugin.internal`.
    Internal,
    /// Observation identified by `plugin.configuration_version_unsupported`.
    ConfigurationVersionUnsupported,
    /// Observation identified by `plugin.component_unknown`.
    ComponentUnknown,
    /// Observation identified by `plugin.configuration_file_missing`.
    ConfigurationFileMissing,
    /// Observation identified by `plugin.configuration_schema_failed`.
    ConfigurationSchemaFailed,
    /// Observation identified by `plugin.integrity_check_failed`.
    IntegrityCheckFailed,
    /// Observation identified by `plugin.attestation_failed`.
    AttestationFailed,
    /// Observation identified by `plugin.policy_rejected`.
    PolicyRejected,
}

impl CommonPluginFailureCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::ConfigurationInvalid => "plugin.configuration_invalid",
            Self::RegistrationFailed => "plugin.registration_failed",
            Self::RegistrationConflict => "plugin.registration_conflict",
            Self::StateConflict => "plugin.state_conflict",
            Self::ResourceNotFound => "plugin.resource_not_found",
            Self::InvocationFailed => "plugin.invocation_failed",
            Self::SerializationFailed => "plugin.serialization_failed",
            Self::Internal => "plugin.internal",
            Self::ConfigurationVersionUnsupported => "plugin.configuration_version_unsupported",
            Self::ComponentUnknown => "plugin.component_unknown",
            Self::ConfigurationFileMissing => "plugin.configuration_file_missing",
            Self::ConfigurationSchemaFailed => "plugin.configuration_schema_failed",
            Self::IntegrityCheckFailed => "plugin.integrity_check_failed",
            Self::AttestationFailed => "plugin.attestation_failed",
            Self::PolicyRejected => "plugin.policy_rejected",
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::ConfigurationInvalid => &[FailureClass::InvalidInput],
            Self::RegistrationFailed => &[
                FailureClass::Unknown,
                FailureClass::InvalidInput,
                FailureClass::Conflict,
                FailureClass::Internal,
                FailureClass::ResourceExhausted,
                FailureClass::Unavailable,
            ],
            Self::RegistrationConflict => &[FailureClass::Conflict],
            Self::StateConflict => &[FailureClass::Conflict],
            Self::ResourceNotFound => &[FailureClass::NotFound],
            Self::InvocationFailed => &[
                FailureClass::Timeout,
                FailureClass::ExecutionFailed,
                FailureClass::Unknown,
            ],
            Self::SerializationFailed => &[
                FailureClass::Unknown,
                FailureClass::InvalidInput,
                FailureClass::Internal,
            ],
            Self::Internal => &[FailureClass::Internal],
            Self::ConfigurationVersionUnsupported => &[FailureClass::Unsupported],
            Self::ComponentUnknown => &[FailureClass::NotFound],
            Self::ConfigurationFileMissing => &[FailureClass::NotFound],
            Self::ConfigurationSchemaFailed => &[FailureClass::Unknown, FailureClass::InvalidInput],
            Self::IntegrityCheckFailed => &[
                FailureClass::Unknown,
                FailureClass::InvalidInput,
                FailureClass::Io,
            ],
            Self::AttestationFailed => &[FailureClass::Unknown, FailureClass::InvalidInput],
            Self::PolicyRejected => &[FailureClass::PolicyRejected],
        }
    }
}

/// Initial Native plugin  failure vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum NativePluginFailureCode {
    /// Observation identified by `plugin.native.library_load_failed`.
    LibraryLoadFailed,
    /// Observation identified by `plugin.native.entry_point_missing`.
    EntryPointMissing,
    /// Observation identified by `plugin.native.abi_mismatch`.
    AbiMismatch,
    /// Observation identified by `plugin.native.validation_failed`.
    ValidationFailed,
    /// Observation identified by `plugin.native.callback_panicked`.
    CallbackPanicked,
    /// Observation identified by `plugin.native.abi_call_failed`.
    AbiCallFailed,
}

impl NativePluginFailureCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::LibraryLoadFailed => "plugin.native.library_load_failed",
            Self::EntryPointMissing => "plugin.native.entry_point_missing",
            Self::AbiMismatch => "plugin.native.abi_mismatch",
            Self::ValidationFailed => "plugin.native.validation_failed",
            Self::CallbackPanicked => "plugin.native.callback_panicked",
            Self::AbiCallFailed => "plugin.native.abi_call_failed",
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::LibraryLoadFailed => &[
                FailureClass::Unknown,
                FailureClass::Io,
                FailureClass::NotFound,
                FailureClass::Unavailable,
            ],
            Self::EntryPointMissing => &[FailureClass::Incompatible, FailureClass::NotFound],
            Self::AbiMismatch => &[FailureClass::Incompatible],
            Self::ValidationFailed => &[
                FailureClass::Unknown,
                FailureClass::InvalidInput,
                FailureClass::Internal,
                FailureClass::ExecutionFailed,
            ],
            Self::CallbackPanicked => &[FailureClass::ExecutionFailed],
            Self::AbiCallFailed => &[
                FailureClass::Conflict,
                FailureClass::NotFound,
                FailureClass::InvalidState,
                FailureClass::Internal,
                FailureClass::InvalidInput,
                FailureClass::Unknown,
            ],
        }
    }
}

/// Initial Worker plugin  failure vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum WorkerPluginFailureCode {
    /// Observation identified by `plugin.worker.launch_failed`.
    LaunchFailed,
    /// Observation identified by `plugin.worker.handshake_failed`.
    HandshakeFailed,
    /// Observation identified by `plugin.worker.protocol_mismatch`.
    ProtocolMismatch,
    /// Observation identified by `plugin.worker.connection_lost`.
    ConnectionLost,
    /// Observation identified by `plugin.worker.transport_failed`.
    TransportFailed,
    /// Observation identified by `plugin.worker.response_invalid`.
    ResponseInvalid,
    /// Observation identified by `plugin.worker.process_exited`.
    ProcessExited,
    /// Observation identified by `plugin.worker.configuration_mismatch`.
    ConfigurationMismatch,
    /// Observation identified by `plugin.worker.invalid_input`.
    InvalidInput,
    /// Observation identified by `plugin.worker.serialization_failed`.
    SerializationFailed,
    /// Observation identified by `plugin.worker.remote_error`.
    RemoteError,
    /// Observation identified by `plugin.worker.remote_sdk_error`.
    RemoteSdkError,
    /// Observation identified by `plugin.worker.invocation_timed_out`.
    InvocationTimedOut,
    /// Observation identified by `plugin.worker.rpc_failed`.
    RpcFailed,
}

impl WorkerPluginFailureCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::LaunchFailed => "plugin.worker.launch_failed",
            Self::HandshakeFailed => "plugin.worker.handshake_failed",
            Self::ProtocolMismatch => "plugin.worker.protocol_mismatch",
            Self::ConnectionLost => "plugin.worker.connection_lost",
            Self::TransportFailed => "plugin.worker.transport_failed",
            Self::ResponseInvalid => "plugin.worker.response_invalid",
            Self::ProcessExited => "plugin.worker.process_exited",
            Self::ConfigurationMismatch => "plugin.worker.configuration_mismatch",
            Self::InvalidInput => "plugin.worker.invalid_input",
            Self::SerializationFailed => "plugin.worker.serialization_failed",
            Self::RemoteError => "plugin.worker.remote_error",
            Self::RemoteSdkError => "plugin.worker.remote_sdk_error",
            Self::InvocationTimedOut => "plugin.worker.invocation_timed_out",
            Self::RpcFailed => "plugin.worker.rpc_failed",
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::LaunchFailed => &[
                FailureClass::Unknown,
                FailureClass::Io,
                FailureClass::NotFound,
                FailureClass::Unavailable,
                FailureClass::ResourceExhausted,
            ],
            Self::HandshakeFailed => &[
                FailureClass::Unknown,
                FailureClass::Incompatible,
                FailureClass::Timeout,
                FailureClass::Unavailable,
            ],
            Self::ProtocolMismatch => &[FailureClass::Incompatible],
            Self::ConnectionLost => &[
                FailureClass::Unavailable,
                FailureClass::Io,
                FailureClass::Unknown,
            ],
            Self::TransportFailed => &[
                FailureClass::Unknown,
                FailureClass::Io,
                FailureClass::Unavailable,
                FailureClass::Timeout,
            ],
            Self::ResponseInvalid => &[
                FailureClass::InvalidInput,
                FailureClass::Incompatible,
                FailureClass::Unknown,
            ],
            Self::ProcessExited => &[
                FailureClass::Unavailable,
                FailureClass::ExecutionFailed,
                FailureClass::Unknown,
            ],
            Self::ConfigurationMismatch => &[FailureClass::Conflict],
            Self::InvalidInput => &[FailureClass::InvalidInput],
            Self::SerializationFailed => &[
                FailureClass::Unknown,
                FailureClass::InvalidInput,
                FailureClass::Internal,
            ],
            Self::RemoteError => &[FailureClass::Unknown],
            Self::RemoteSdkError => &[FailureClass::Unknown],
            Self::InvocationTimedOut => &[FailureClass::Timeout],
            Self::RpcFailed => &[FailureClass::Unknown, FailureClass::Unavailable],
        }
    }
}

/// Initial Logging failure vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
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

/// Initial telemetry vocabulary; pipeline integration is a later phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ObservabilityFailureCode {
    /// A telemetry export attempt failed.
    ExportFailed(Signal),
    /// Telemetry records were dropped.
    RecordsDropped(Signal),
    /// Observation identified by `otel.metric_mark_invalid`.
    MetricMarkInvalid,
    /// Observation identified by `otel.log_mark_invalid_severity`.
    LogSeverityInvalid,
    /// Observation identified by `otel.timestamp_out_of_range`.
    TimestampOutOfRange,
    /// Observation identified by `otel.metric_cardinality_limit`.
    MetricCardinalityLimit,
    /// Observation identified by `otel.metric_instrument_limit`.
    MetricInstrumentLimit,
    /// Observation identified by `otel.metric_descriptor_conflict`.
    MetricDescriptorConflict,
    /// Observation identified by `otel.resource_metadata_pipeline_limit`.
    ResourcePipelineLimit,
    /// Observation identified by `otel.resource_metadata_active_scope_limit`.
    ResourceScopeLimit,
    /// Observation identified by `otel.resource_metadata_pipeline_build_failed`.
    ResourcePipelineBuildFailed,
    /// Observation identified by `otel.metadata_value_unsupported`.
    MetadataValueUnsupported,
    /// Observation identified by `observability.invalid_otel_file_sink`.
    FileSinkConfigurationInvalid,
    /// Observation identified by `observability.invalid_plugin_config`.
    ConfigurationInvalid,
    /// Observation identified by `observability.unsafe_otel_destination_collision`.
    DestinationConflict,
    /// Observation identified by `observability.invalid_otel_endpoint`.
    EndpointInvalid,
}

impl ObservabilityFailureCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::ExportFailed(Signal::Logs) => "otel.logs_export_failed",
            Self::ExportFailed(Signal::Traces) => "otel.traces_export_failed",
            Self::ExportFailed(Signal::Metrics) => "otel.metrics_export_failed",
            Self::RecordsDropped(Signal::Logs) => "otel.logs_dropped",
            Self::RecordsDropped(Signal::Traces) => "otel.spans_dropped",
            Self::RecordsDropped(Signal::Metrics) => "otel.metrics_dropped",
            Self::MetricMarkInvalid => "otel.metric_mark_invalid",
            Self::LogSeverityInvalid => "otel.log_mark_invalid_severity",
            Self::TimestampOutOfRange => "otel.timestamp_out_of_range",
            Self::MetricCardinalityLimit => "otel.metric_cardinality_limit",
            Self::MetricInstrumentLimit => "otel.metric_instrument_limit",
            Self::MetricDescriptorConflict => "otel.metric_descriptor_conflict",
            Self::ResourcePipelineLimit => "otel.resource_metadata_pipeline_limit",
            Self::ResourceScopeLimit => "otel.resource_metadata_active_scope_limit",
            Self::ResourcePipelineBuildFailed => "otel.resource_metadata_pipeline_build_failed",
            Self::MetadataValueUnsupported => "otel.metadata_value_unsupported",
            Self::FileSinkConfigurationInvalid => "observability.invalid_otel_file_sink",
            Self::ConfigurationInvalid => "observability.invalid_plugin_config",
            Self::DestinationConflict => "observability.unsafe_otel_destination_collision",
            Self::EndpointInvalid => "observability.invalid_otel_endpoint",
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::ExportFailed(_) => &[
                FailureClass::Unknown,
                FailureClass::Timeout,
                FailureClass::Unavailable,
                FailureClass::Io,
            ],
            Self::RecordsDropped(_) => &[FailureClass::Unknown, FailureClass::ResourceExhausted],
            Self::MetricMarkInvalid => &[FailureClass::InvalidInput],
            Self::LogSeverityInvalid => &[FailureClass::InvalidInput],
            Self::TimestampOutOfRange => &[FailureClass::InvalidInput],
            Self::MetricCardinalityLimit => &[FailureClass::ResourceExhausted],
            Self::MetricInstrumentLimit => &[FailureClass::ResourceExhausted],
            Self::MetricDescriptorConflict => &[FailureClass::Conflict],
            Self::ResourcePipelineLimit => &[FailureClass::ResourceExhausted],
            Self::ResourceScopeLimit => &[FailureClass::ResourceExhausted],
            Self::ResourcePipelineBuildFailed => &[FailureClass::Unknown],
            Self::MetadataValueUnsupported => &[FailureClass::Unsupported],
            Self::FileSinkConfigurationInvalid => &[FailureClass::InvalidInput],
            Self::ConfigurationInvalid => &[FailureClass::InvalidInput],
            Self::DestinationConflict => &[FailureClass::Conflict],
            Self::EndpointInvalid => &[FailureClass::InvalidInput],
        }
    }
}

/// Nested common, native, and worker plugin classifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum PluginFailureCode {
    /// Common plugin failures.
    Common(CommonPluginFailureCode),
    /// Native plugin failures.
    Native(NativePluginFailureCode),
    /// Worker plugin failures.
    Worker(WorkerPluginFailureCode),
}

impl PluginFailureCode {
    /// Plugin mechanism selected by this code.
    #[must_use]
    pub const fn family(self) -> PluginDiagnosticFamily {
        match self {
            Self::Common(_) => PluginDiagnosticFamily::Common,
            Self::Native(_) => PluginDiagnosticFamily::Native,
            Self::Worker(_) => PluginDiagnosticFamily::Worker,
        }
    }
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Common(code) => code.canonical_name(),
            Self::Native(code) => code.canonical_name(),
            Self::Worker(code) => code.canonical_name(),
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::Common(code) => code.allowed_classes(),
            Self::Native(code) => code.allowed_classes(),
            Self::Worker(code) => code.allowed_classes(),
        }
    }
}

/// Failure event hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum FailureCode {
    /// Runtime failures.
    Runtime(RuntimeFailureCode),
    /// Plugin failures.
    Plugin(PluginFailureCode),
    /// Upstream failures.
    Upstream(UpstreamFailureCode),
    /// Observability failures.
    Observability(ObservabilityFailureCode),
    /// Logging failures.
    Logging(LoggingFailureCode),
    /// Binding failures.
    Binding(BindingFailureCode),
}

impl FailureCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Runtime(code) => code.canonical_name(),
            Self::Plugin(code) => code.canonical_name(),
            Self::Upstream(code) => code.canonical_name(),
            Self::Observability(code) => code.canonical_name(),
            Self::Logging(code) => code.canonical_name(),
            Self::Binding(code) => code.canonical_name(),
        }
    }
    /// Owning subsystem derived from the code.
    #[must_use]
    pub const fn domain(self) -> DiagnosticDomain {
        match self {
            Self::Runtime(_) => DiagnosticDomain::Runtime,
            Self::Plugin(_) => DiagnosticDomain::Plugin,
            Self::Upstream(_) => DiagnosticDomain::Upstream,
            Self::Observability(_) => DiagnosticDomain::Observability,
            Self::Logging(_) => DiagnosticDomain::Logging,
            Self::Binding(_) => DiagnosticDomain::Binding,
        }
    }
    /// Permitted cause classes for this observation.
    #[must_use]
    pub const fn allowed_classes(self) -> &'static [FailureClass] {
        match self {
            Self::Runtime(code) => code.allowed_classes(),
            Self::Plugin(code) => code.allowed_classes(),
            Self::Upstream(code) => code.allowed_classes(),
            Self::Observability(code) => code.allowed_classes(),
            Self::Logging(code) => code.allowed_classes(),
            Self::Binding(code) => code.allowed_classes(),
        }
    }
}

/// Explicit successful operations; successes do not recover or clear failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum SuccessCode {
    /// Observation identified by `logging.initialized`.
    LoggingInitialized,
    /// Observation identified by `logging.shutdown_completed`.
    LoggingShutdownCompleted,
    /// Observation identified by `logging.sink.initialized`.
    SinkInitialized,
    /// A telemetry export attempt succeeded; producer integration is deferred.
    ExportSucceeded(Signal),
}

impl SuccessCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::LoggingInitialized => "logging.initialized",
            Self::LoggingShutdownCompleted => "logging.shutdown_completed",
            Self::SinkInitialized => "logging.sink.initialized",
            Self::ExportSucceeded(Signal::Logs) => "otel.logs_export_succeeded",
            Self::ExportSucceeded(Signal::Traces) => "otel.traces_export_succeeded",
            Self::ExportSucceeded(Signal::Metrics) => "otel.metrics_export_succeeded",
        }
    }
}

/// Observed decisions and normal completion, independent of failure severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum NoticeCode {
    /// Observation identified by `plugin.configuration_inherited`.
    ConfigurationInherited,
    /// Observation identified by `plugin.component_reenabled`.
    ComponentReenabled,
    /// Observation identified by `runtime.guardrail_rejected`.
    GuardrailRejected,
    /// Observation identified by `plugin.worker.cancelled`.
    WorkerCancelled,
    /// Observation identified by `plugin.native.stream_ended`.
    NativeStreamEnded,
    /// Observation identified by `plugin.native.stream_backpressured`.
    NativeStreamBackpressured,
    /// A completed telemetry context expired; producer integration is deferred.
    CompletedContextExpired(Signal),
}

impl NoticeCode {
    /// Stable catalogue name.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::ConfigurationInherited => "plugin.configuration_inherited",
            Self::ComponentReenabled => "plugin.component_reenabled",
            Self::GuardrailRejected => "runtime.guardrail_rejected",
            Self::WorkerCancelled => "plugin.worker.cancelled",
            Self::NativeStreamEnded => "plugin.native.stream_ended",
            Self::NativeStreamBackpressured => "plugin.native.stream_backpressured",
            Self::CompletedContextExpired(Signal::Logs) => "otel.completed_log_context_expired",
            Self::CompletedContextExpired(Signal::Traces) => "otel.completed_span_context_expired",
            Self::CompletedContextExpired(Signal::Metrics) => {
                "otel.completed_metric_context_expired"
            }
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
    /// Parent domain; no independent parent value can conflict with the code.
    #[must_use]
    pub const fn domain(self) -> DiagnosticDomain {
        match self {
            Self::Failure(code) => code.domain(),
            Self::Success(SuccessCode::ExportSucceeded(_))
            | Self::Notice(NoticeCode::CompletedContextExpired(_)) => {
                DiagnosticDomain::Observability
            }
            Self::Success(_) => DiagnosticDomain::Logging,
            Self::Notice(NoticeCode::ConfigurationInherited) => DiagnosticDomain::Plugin,
            Self::Notice(NoticeCode::ComponentReenabled) => DiagnosticDomain::Plugin,
            Self::Notice(NoticeCode::GuardrailRejected) => DiagnosticDomain::Runtime,
            Self::Notice(NoticeCode::WorkerCancelled) => DiagnosticDomain::Plugin,
            Self::Notice(NoticeCode::NativeStreamEnded) => DiagnosticDomain::Plugin,
            Self::Notice(NoticeCode::NativeStreamBackpressured) => DiagnosticDomain::Plugin,
        }
    }
    /// Plugin mechanism when the code itself describes one.
    #[must_use]
    pub const fn plugin_family(self) -> Option<PluginDiagnosticFamily> {
        match self {
            Self::Failure(FailureCode::Plugin(code)) => Some(code.family()),
            Self::Notice(NoticeCode::ConfigurationInherited | NoticeCode::ComponentReenabled) => {
                Some(PluginDiagnosticFamily::Common)
            }
            Self::Notice(NoticeCode::WorkerCancelled) => Some(PluginDiagnosticFamily::Worker),
            Self::Notice(NoticeCode::NativeStreamEnded | NoticeCode::NativeStreamBackpressured) => {
                Some(PluginDiagnosticFamily::Native)
            }
            _ => None,
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
    /// All initial catalogue entries, including deferred producer vocabulary.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Failure(FailureCode::Runtime(RuntimeFailureCode::AlreadyExists)),
            Self::Failure(FailureCode::Runtime(RuntimeFailureCode::NotFound)),
            Self::Failure(FailureCode::Runtime(RuntimeFailureCode::InvalidArgument)),
            Self::Failure(FailureCode::Runtime(RuntimeFailureCode::ScopeStackEmpty)),
            Self::Failure(FailureCode::Runtime(RuntimeFailureCode::Internal)),
            Self::Failure(FailureCode::Runtime(RuntimeFailureCode::RemoteError)),
            Self::Failure(FailureCode::Upstream(UpstreamFailureCode::ConnectionFailed)),
            Self::Failure(FailureCode::Upstream(UpstreamFailureCode::TimedOut)),
            Self::Failure(FailureCode::Upstream(UpstreamFailureCode::StatusRejected)),
            Self::Failure(FailureCode::Upstream(UpstreamFailureCode::ContextLimit)),
            Self::Failure(FailureCode::Upstream(UpstreamFailureCode::ModelUnavailable)),
            Self::Failure(FailureCode::Upstream(
                UpstreamFailureCode::AuthenticationFailed,
            )),
            Self::Failure(FailureCode::Upstream(UpstreamFailureCode::RequestInvalid)),
            Self::Failure(FailureCode::Upstream(UpstreamFailureCode::Failed)),
            Self::Failure(FailureCode::Binding(BindingFailureCode::CallbackException)),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::ConfigurationInvalid,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::RegistrationFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::RegistrationConflict,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::StateConflict,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::ResourceNotFound,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::InvocationFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::SerializationFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::Internal,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::ConfigurationVersionUnsupported,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::ComponentUnknown,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::ConfigurationFileMissing,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::ConfigurationSchemaFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::IntegrityCheckFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::AttestationFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                CommonPluginFailureCode::PolicyRejected,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Native(
                NativePluginFailureCode::LibraryLoadFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Native(
                NativePluginFailureCode::EntryPointMissing,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Native(
                NativePluginFailureCode::AbiMismatch,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Native(
                NativePluginFailureCode::ValidationFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Native(
                NativePluginFailureCode::CallbackPanicked,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Native(
                NativePluginFailureCode::AbiCallFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::LaunchFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::HandshakeFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::ProtocolMismatch,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::ConnectionLost,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::TransportFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::ResponseInvalid,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::ProcessExited,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::ConfigurationMismatch,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::InvalidInput,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::SerializationFailed,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::RemoteError,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::RemoteSdkError,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::InvocationTimedOut,
            ))),
            Self::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
                WorkerPluginFailureCode::RpcFailed,
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
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::MetricMarkInvalid,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::LogSeverityInvalid,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::TimestampOutOfRange,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::MetricCardinalityLimit,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::MetricInstrumentLimit,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::MetricDescriptorConflict,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::ResourcePipelineLimit,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::ResourceScopeLimit,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::ResourcePipelineBuildFailed,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::MetadataValueUnsupported,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::FileSinkConfigurationInvalid,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::ConfigurationInvalid,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::DestinationConflict,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::EndpointInvalid,
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::ExportFailed(Signal::Logs),
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::ExportFailed(Signal::Traces),
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::ExportFailed(Signal::Metrics),
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::RecordsDropped(Signal::Logs),
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::RecordsDropped(Signal::Traces),
            )),
            Self::Failure(FailureCode::Observability(
                ObservabilityFailureCode::RecordsDropped(Signal::Metrics),
            )),
            Self::Success(SuccessCode::LoggingInitialized),
            Self::Success(SuccessCode::LoggingShutdownCompleted),
            Self::Success(SuccessCode::SinkInitialized),
            Self::Success(SuccessCode::ExportSucceeded(Signal::Logs)),
            Self::Success(SuccessCode::ExportSucceeded(Signal::Traces)),
            Self::Success(SuccessCode::ExportSucceeded(Signal::Metrics)),
            Self::Notice(NoticeCode::ConfigurationInherited),
            Self::Notice(NoticeCode::ComponentReenabled),
            Self::Notice(NoticeCode::GuardrailRejected),
            Self::Notice(NoticeCode::WorkerCancelled),
            Self::Notice(NoticeCode::NativeStreamEnded),
            Self::Notice(NoticeCode::NativeStreamBackpressured),
            Self::Notice(NoticeCode::CompletedContextExpired(Signal::Logs)),
            Self::Notice(NoticeCode::CompletedContextExpired(Signal::Traces)),
            Self::Notice(NoticeCode::CompletedContextExpired(Signal::Metrics)),
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

/// Validated authoring metadata; no source error or classifier is retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Classification {
    code: BuiltinDiagnosticCode,
    failure_class: Option<FailureClass>,
}

impl Classification {
    /// Validate the required cause and the catalogue entry's permitted classes.
    pub fn new(
        code: BuiltinDiagnosticCode,
        failure_class: Option<FailureClass>,
    ) -> Result<Self, ClassificationError> {
        if !code.allows_class(failure_class) {
            return Err(ClassificationError);
        }
        Ok(Self {
            code,
            failure_class,
        })
    }
    /// Classify a failure using an explicitly supported cause.
    pub fn failure(
        code: FailureCode,
        failure_class: FailureClass,
    ) -> Result<Self, ClassificationError> {
        Self::new(BuiltinDiagnosticCode::Failure(code), Some(failure_class))
    }
    /// Classify an observed success, without clearing previous failures.
    #[must_use]
    pub const fn success(code: SuccessCode) -> Self {
        Self {
            code: BuiltinDiagnosticCode::Success(code),
            failure_class: None,
        }
    }
    /// Classify an observed notice, without inventing a failure cause.
    #[must_use]
    pub const fn notice(code: NoticeCode) -> Self {
        Self {
            code: BuiltinDiagnosticCode::Notice(code),
            failure_class: None,
        }
    }
    /// Canonical built-in classification.
    #[must_use]
    pub const fn code(self) -> BuiltinDiagnosticCode {
        self.code
    }
    /// Supported cause; absent for success and notice.
    #[must_use]
    pub const fn failure_class(self) -> Option<FailureClass> {
        self.failure_class
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

/// Pure producer behavior: borrow original evidence and return fixed-size metadata.
///
/// Owning producer crates implement local adapters. Classification must not record,
/// log, retry, change health, or modify the source operation's outcome.
pub trait DiagnosticClassifier<E: ?Sized> {
    /// Interpret supported structured evidence without retaining or altering it.
    fn classify(&self, error: &E) -> Classification;
}
