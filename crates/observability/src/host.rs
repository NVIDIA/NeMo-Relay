// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime services supplied by the embedding Relay runtime.

use nemo_relay_types::api::event::Event;
use nemo_relay_types::codec::normalization::LlmNormalizer;
use nemo_relay_types::error::{FlowError, Result};
use std::sync::{Arc, OnceLock};

/// Callback receiving one canonical lifecycle event.
pub type EventSubscriberFn = Arc<dyn Fn(&Event) + Send + Sync>;

/// Runtime operations used by the built-in exporters.
pub trait ObservabilityHost: LlmNormalizer {
    /// Capture the selected estimator for one projection operation.
    /// Detached exporters use the standalone catalog; runtime adapters override
    /// this with the embedding application's selected pricing service.
    fn pricing_snapshot(&self) -> Arc<dyn nemo_relay_types::codec::pricing::CostEstimator>;

    /// Register an exporter callback.
    fn register_subscriber(&self, name: &str, callback: EventSubscriberFn) -> Result<()>;
    /// Remove an exporter callback.
    fn deregister_subscriber(&self, name: &str) -> Result<bool>;
    /// Establish the runtime's subscriber-delivery barrier.
    fn flush_subscribers(&self) -> Result<()>;
    /// Parse remote W3C propagation context using the runtime's policy.
    fn w3c_span_context(
        &self,
        traceparent: Option<&str>,
        tracestate: Option<&str>,
    ) -> Option<opentelemetry::trace::SpanContext>;
    /// Mirror an exporter diagnostic into the active plugin session.
    fn record_runtime_diagnostic(
        &self,
        code: &str,
        field: Option<String>,
        message: String,
        count: u64,
    );
    /// Configure whether active scopes retain full observability payloads.
    fn set_full_payloads_enabled(&self, enabled: bool) -> Result<()>;
    /// Register a subscriber on an open scope without emitting lookup diagnostics.
    fn try_scope_register_subscriber(
        &self,
        scope: &uuid::Uuid,
        name: &str,
        callback: EventSubscriberFn,
    ) -> Result<()>;
    /// Remove a scoped subscriber without emitting lookup diagnostics.
    fn try_scope_deregister_subscriber(&self, scope: &uuid::Uuid, name: &str) -> Result<bool>;
    /// Remove a scoped subscriber with ordinary runtime lookup behavior.
    fn scope_deregister_subscriber(&self, scope: &uuid::Uuid, name: &str) -> Result<bool>;
    /// Whether a scope's parent is the active root or a propagated parent.
    fn is_trajectory_parent(&self, parent: uuid::Uuid) -> bool;
    /// Record an ATIF diagnostic, retaining its session correlation.
    fn record_plugin_diagnostic(&self, diagnostic: nemo_relay_plugin::component::RuntimeDiagnostic);
}

#[derive(Clone, Default)]
pub struct HostCell(Arc<OnceLock<Arc<dyn ObservabilityHost>>>);

impl std::fmt::Debug for HostCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostCell")
            .field("bound", &self.0.get().is_some())
            .finish()
    }
}

impl HostCell {
    pub fn bind(&self, host: Arc<dyn ObservabilityHost>) {
        let _ = self.0.set(host);
    }
    pub fn get(&self) -> Arc<dyn ObservabilityHost> {
        self.0
            .get()
            .cloned()
            .unwrap_or_else(|| Arc::new(DetachedHost))
    }
}

pub struct DetachedHost;
impl LlmNormalizer for DetachedHost {
    fn normalize_request(
        &self,
        _: &nemo_relay_types::api::llm::LlmRequest,
        _: Option<&str>,
    ) -> Option<nemo_relay_types::codec::request::AnnotatedLlmRequest> {
        None
    }
    fn normalize_response(
        &self,
        _: &nemo_relay_types::Json,
    ) -> Option<nemo_relay_types::codec::response::AnnotatedLlmResponse> {
        None
    }
}
impl ObservabilityHost for DetachedHost {
    fn pricing_snapshot(&self) -> Arc<dyn nemo_relay_types::codec::pricing::CostEstimator> {
        nemo_relay_model_pricing::active_pricing_resolver()
    }

    fn register_subscriber(&self, _: &str, _: EventSubscriberFn) -> Result<()> {
        Err(FlowError::InvalidArgument(
            "observability exporter has no runtime host".into(),
        ))
    }
    fn deregister_subscriber(&self, _: &str) -> Result<bool> {
        Err(FlowError::InvalidArgument(
            "observability exporter has no runtime host".into(),
        ))
    }
    fn flush_subscribers(&self) -> Result<()> {
        Ok(())
    }
    fn w3c_span_context(
        &self,
        _: Option<&str>,
        _: Option<&str>,
    ) -> Option<opentelemetry::trace::SpanContext> {
        None
    }
    fn record_runtime_diagnostic(&self, _: &str, _: Option<String>, _: String, _: u64) {}
    fn set_full_payloads_enabled(&self, _: bool) -> Result<()> {
        Err(FlowError::InvalidArgument(
            "observability exporter has no runtime host".into(),
        ))
    }
    fn try_scope_register_subscriber(
        &self,
        _: &uuid::Uuid,
        _: &str,
        _: EventSubscriberFn,
    ) -> Result<()> {
        Err(FlowError::InvalidArgument(
            "observability exporter has no runtime host".into(),
        ))
    }
    fn try_scope_deregister_subscriber(&self, _: &uuid::Uuid, _: &str) -> Result<bool> {
        Ok(false)
    }
    fn scope_deregister_subscriber(&self, _: &uuid::Uuid, _: &str) -> Result<bool> {
        Ok(false)
    }
    fn is_trajectory_parent(&self, _: uuid::Uuid) -> bool {
        false
    }
    fn record_plugin_diagnostic(&self, _: nemo_relay_plugin::component::RuntimeDiagnostic) {}
}

/// Identifies recoverable OpenTelemetry delivery failures in plugin teardown.
#[doc(hidden)]
pub const OTEL_RUNTIME_DELIVERY_FAILURE_MARKER: &str = "OpenTelemetry runtime delivery failures";

/// Identifies recoverable ATIF delivery failures in plugin teardown.
#[doc(hidden)]
pub const ATIF_RUNTIME_DELIVERY_FAILURE_MARKER: &str = "ATIF runtime delivery failures";
