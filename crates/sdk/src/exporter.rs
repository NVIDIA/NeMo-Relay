// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-bound exporters and their configuration contracts.

pub use nemo_relay_observability::header_file::{
    HeaderFiles, HostRef, PlaintextPolicy, TrustedHost,
};
pub use nemo_relay_observability::{
    ExporterRuntimeDiagnostic, ExporterRuntimeDiagnostics, MarkProjection, OpenTelemetryTlsConfig,
    OpenTelemetryType, OtlpAttributeMapping,
};

macro_rules! subscriber_methods {
    () => {
        /// Return the callback for global or scope-local subscriber registration.
        pub fn subscriber(&self) -> crate::middleware::EventSubscriberFn {
            self.inner.subscriber()
        }
        /// Read a bounded snapshot of exporter diagnostics.
        pub fn runtime_diagnostics(&self) -> super::ExporterRuntimeDiagnostics {
            self.inner.runtime_diagnostics()
        }
        /// Register with the runtime used by the Rust SDK.
        pub fn register(&self, name: &str) -> Result<()> {
            self.inner.register(name)
        }
        /// Remove a global registration from the runtime.
        pub fn deregister(&self, name: &str) -> Result<bool> {
            self.inner.deregister(name)
        }
        /// Wait for accepted events and flush the exporter's pending output.
        pub fn force_flush(&self) -> Result<()> {
            self.inner.force_flush()
        }
        /// Flush pending output and shut down the exporter.
        pub fn shutdown(&self) -> Result<()> {
            self.inner.shutdown()
        }
    };
}

/// ATIF trajectories and a runtime-bound trajectory collector.
pub mod atif {
    use crate::Result;
    pub use nemo_relay_observability::atif::{
        ATIF_SCHEMA_VERSION, AtifAgentInfo, AtifAncestry, AtifFinalMetrics, AtifInvocationInfo,
        AtifMetrics, AtifObservation, AtifObservationResult, AtifStep, AtifStepExtra,
        AtifSubagentTrajectoryRef, AtifToolCall, AtifTrajectory,
    };

    /// Collect lifecycle events using the SDK's scope and normalization services.
    #[derive(Clone)]
    pub struct AtifExporter {
        inner: nemo_relay_observability::atif::AtifExporter,
    }
    impl AtifExporter {
        /// Create a collector bound to the SDK runtime and built-in services.
        pub fn new(session_id: String, agent_info: AtifAgentInfo) -> Self {
            Self {
                inner: nemo_relay_builtins::exporter::atif(session_id, agent_info),
            }
        }
        /// Create a bound collector with an explicit trajectory identifier.
        pub fn new_for_trajectory(
            session_id: String,
            trajectory_id: String,
            agent_info: AtifAgentInfo,
        ) -> Self {
            Self {
                inner: nemo_relay_builtins::exporter::atif_for_trajectory(
                    session_id,
                    trajectory_id,
                    agent_info,
                ),
            }
        }
        /// Return a callback for global or scope-local registration.
        pub fn subscriber(&self) -> crate::middleware::EventSubscriberFn {
            self.inner.subscriber()
        }
        /// Wait for delivery and export the collected trajectory.
        pub fn export(&self) -> Result<AtifTrajectory> {
            self.inner.export()
        }
        /// Wait for delivery and export the collected trajectory.
        pub fn try_export(&self) -> Result<AtifTrajectory> {
            self.inner.try_export()
        }
        /// Clear collected events while keeping session metadata.
        pub fn clear(&self) {
            self.inner.clear()
        }
    }
}

/// ATOF JSONL destinations and a runtime-bound exporter.
pub mod atof {
    pub use nemo_relay_observability::atof::{
        AtofEndpointConfig, AtofEndpointFieldNamePolicy, AtofEndpointTransport, AtofExporterConfig,
        AtofExporterError, AtofExporterMode, AtofFileSinkConfig, AtofSinkConfig,
        AtofStreamSinkConfig, Result,
    };
    /// ATOF exporter whose constructors bind the SDK runtime automatically.
    pub struct AtofExporter {
        inner: nemo_relay_observability::atof::AtofExporter,
    }
    impl AtofExporter {
        /// Open configured destinations and bind the SDK runtime.
        pub fn new(config: AtofExporterConfig) -> Result<Self> {
            nemo_relay_builtins::exporter::atof(config).map(|inner| Self { inner })
        }
        /// Return the configured file destination, when present.
        pub fn path(&self) -> Option<&std::path::Path> {
            self.inner.path()
        }
        subscriber_methods!();
    }
}

/// OpenTelemetry and OpenInference trace configuration and subscribers.
pub mod otel {
    use super::{OpenTelemetryType, OtlpAttributeMapping};
    pub use nemo_relay_observability::otel::{
        DEFAULT_COMPLETED_SPAN_CONTEXT_TTL, ExporterInitializationError,
        ExporterInitializationResult, OpenTelemetryConfig, OpenTelemetryError,
        OpenTelemetryFileSinkConfig, OpenTelemetrySubscriberOptions, OtlpFileSinkMode,
        OtlpFileSinkSettings, OtlpTransport, Result, SharedTraceOptions,
    };
    use opentelemetry_sdk::trace::SdkTracerProvider;

    /// Trace subscriber bound to the SDK's runtime services.
    ///
    /// ```compile_fail,E0599
    /// fn host(subscriber: &nemo_relay::exporter::otel::OpenTelemetrySubscriber) {
    ///     let _ = subscriber.host_ref();
    /// }
    /// ```
    #[derive(Clone)]
    pub struct OpenTelemetrySubscriber {
        inner: nemo_relay_observability::otel::OpenTelemetrySubscriber,
    }
    impl OpenTelemetrySubscriber {
        /// Build a collector-backed subscriber bound to the SDK runtime.
        pub fn new(config: OpenTelemetryConfig) -> Result<Self> {
            nemo_relay_builtins::exporter::opentelemetry(config).map(|inner| Self { inner })
        }
        /// Build a file subscriber bound to the SDK runtime.
        pub fn new_file_sink(config: OpenTelemetryFileSinkConfig) -> Result<Self> {
            nemo_relay_builtins::exporter::opentelemetry_file(config).map(|inner| Self { inner })
        }
        /// Bind an application-owned provider to the SDK runtime.
        pub fn from_tracer_provider(
            provider: SdkTracerProvider,
            instrumentation_scope: impl Into<String>,
        ) -> Self {
            Self {
                inner: nemo_relay_builtins::exporter::opentelemetry_from_provider(
                    provider,
                    instrumentation_scope,
                ),
            }
        }
        /// Bind a provider with an explicit trace projection.
        pub fn from_tracer_provider_with_type(
            provider: SdkTracerProvider,
            instrumentation_scope: impl Into<String>,
            projection: OpenTelemetryType,
        ) -> Self {
            Self {
                inner: nemo_relay_builtins::exporter::opentelemetry_from_provider_with_type(
                    provider,
                    instrumentation_scope,
                    projection,
                ),
            }
        }
        /// Bind a provider with typed attribute copies.
        pub fn from_tracer_provider_with_attribute_mappings<I>(
            provider: SdkTracerProvider,
            instrumentation_scope: impl Into<String>,
            attribute_mappings: I,
        ) -> Result<Self>
        where
            I: IntoIterator<Item = OtlpAttributeMapping>,
        {
            Self::from_tracer_provider_with_options(
                provider,
                instrumentation_scope,
                OpenTelemetrySubscriberOptions {
                    attribute_mappings: attribute_mappings.into_iter().collect(),
                    ..Default::default()
                },
            )
        }
        /// Bind a provider with composable projection options.
        pub fn from_tracer_provider_with_options(
            provider: SdkTracerProvider,
            instrumentation_scope: impl Into<String>,
            options: OpenTelemetrySubscriberOptions,
        ) -> Result<Self> {
            nemo_relay_builtins::exporter::opentelemetry_from_provider_with_options(
                provider,
                instrumentation_scope,
                options,
            )
            .map(|inner| Self { inner })
        }
        /// Bind a typed provider with composable projection options.
        pub fn from_tracer_provider_with_type_and_options(
            provider: SdkTracerProvider,
            instrumentation_scope: impl Into<String>,
            projection: OpenTelemetryType,
            options: OpenTelemetrySubscriberOptions,
        ) -> Result<Self> {
            nemo_relay_builtins::exporter::opentelemetry_from_provider_with_type_and_options(
                provider,
                instrumentation_scope,
                projection,
                options,
            )
            .map(|inner| Self { inner })
        }
        /// Flush and shut down the provider after subscriber delivery settles.
        pub fn shutdown_provider(&self) -> Result<()> {
            self.inner.shutdown_provider()
        }
        subscriber_methods!();
    }
}

/// OTLP file encodings and file-export errors.
pub mod otel_file {
    pub use nemo_relay_observability::otel_file::{OtlpFileExporterError, OtlpFileFormat, Result};
}

/// Runtime-bound logs export.
pub mod otel_logs {
    pub use super::otel::{OpenTelemetryError, Result};
    pub use nemo_relay_observability::otel_logs::OpenTelemetryLogConfig;
    /// Subscriber bound to the SDK's scope and diagnostic services.
    #[derive(Clone)]
    pub struct OpenTelemetryLogSubscriber {
        inner: nemo_relay_observability::otel_logs::OpenTelemetryLogSubscriber,
    }
    impl OpenTelemetryLogSubscriber {
        /// Build the subscriber and bind the SDK runtime automatically.
        pub fn new(config: OpenTelemetryLogConfig) -> Result<Self> {
            nemo_relay_builtins::exporter::opentelemetry_logs(config).map(|inner| Self { inner })
        }
        /// Flush and shut down the provider after subscriber delivery settles.
        pub fn shutdown_provider(&self) -> Result<()> {
            self.inner.shutdown_provider()
        }
        /// Return a summary of unresolved delivery failures, if any.
        pub fn delivery_failure_summary(&self) -> Option<String> {
            self.inner.delivery_failure_summary()
        }
        subscriber_methods!();
    }
}

/// Runtime-bound metrics export.
pub mod otel_metrics {
    pub use super::otel::{OpenTelemetryError, Result};
    pub use nemo_relay_observability::otel_metrics::{
        MetricTemporality, OpenTelemetryMetricConfig,
    };
    /// Subscriber bound to the SDK's scope and diagnostic services.
    #[derive(Clone)]
    pub struct OpenTelemetryMetricSubscriber {
        inner: nemo_relay_observability::otel_metrics::OpenTelemetryMetricSubscriber,
    }
    impl OpenTelemetryMetricSubscriber {
        /// Build the subscriber and bind the SDK runtime automatically.
        pub fn new(config: OpenTelemetryMetricConfig) -> Result<Self> {
            nemo_relay_builtins::exporter::opentelemetry_metrics(config).map(|inner| Self { inner })
        }
        /// Flush and shut down the provider after subscriber delivery settles.
        pub fn shutdown_provider(&self) -> Result<()> {
            self.inner.shutdown_provider()
        }
        /// Return a summary of unresolved delivery failures, if any.
        pub fn delivery_failure_summary(&self) -> Option<String> {
            self.inner.delivery_failure_summary()
        }
        subscriber_methods!();
    }
}

/// Create a runtime-bound exporter through the SDK.
pub fn atif(session_id: String, agent_info: atif::AtifAgentInfo) -> atif::AtifExporter {
    atif::AtifExporter::new(session_id, agent_info)
}

/// Create a runtime-bound exporter through the SDK.
pub fn atof(config: atof::AtofExporterConfig) -> atof::Result<atof::AtofExporter> {
    atof::AtofExporter::new(config)
}

/// Create a runtime-bound exporter through the SDK.
pub fn opentelemetry(
    config: otel::OpenTelemetryConfig,
) -> otel::Result<otel::OpenTelemetrySubscriber> {
    otel::OpenTelemetrySubscriber::new(config)
}

/// Create a runtime-bound exporter through the SDK.
pub fn opentelemetry_file(
    config: otel::OpenTelemetryFileSinkConfig,
) -> otel::Result<otel::OpenTelemetrySubscriber> {
    otel::OpenTelemetrySubscriber::new_file_sink(config)
}

/// Create a runtime-bound exporter through the SDK.
pub fn opentelemetry_logs(
    config: otel_logs::OpenTelemetryLogConfig,
) -> otel::Result<otel_logs::OpenTelemetryLogSubscriber> {
    otel_logs::OpenTelemetryLogSubscriber::new(config)
}

/// Create a runtime-bound exporter through the SDK.
pub fn opentelemetry_metrics(
    config: otel_metrics::OpenTelemetryMetricConfig,
) -> otel::Result<otel_metrics::OpenTelemetryMetricSubscriber> {
    otel_metrics::OpenTelemetryMetricSubscriber::new(config)
}

/// Create a runtime-bound exporter through the SDK.
pub fn opentelemetry_from_provider(
    provider: opentelemetry_sdk::trace::SdkTracerProvider,
    instrumentation_scope: impl Into<String>,
) -> otel::OpenTelemetrySubscriber {
    otel::OpenTelemetrySubscriber::from_tracer_provider(provider, instrumentation_scope)
}

/// Create a runtime-bound exporter through the SDK.
pub fn opentelemetry_from_provider_with_type(
    provider: opentelemetry_sdk::trace::SdkTracerProvider,
    instrumentation_scope: impl Into<String>,
    projection: OpenTelemetryType,
) -> otel::OpenTelemetrySubscriber {
    otel::OpenTelemetrySubscriber::from_tracer_provider_with_type(
        provider,
        instrumentation_scope,
        projection,
    )
}
