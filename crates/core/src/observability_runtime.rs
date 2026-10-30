// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core runtime services supplied to observability exporters.

struct CoreObservabilityHost;
impl nemo_relay_types::codec::normalization::LlmNormalizer for CoreObservabilityHost {
    fn normalize_request(
        &self,
        request: &crate::api::llm::LlmRequest,
        provider_hint: Option<&str>,
    ) -> Option<crate::codec::request::AnnotatedLlmRequest> {
        crate::codec::resolve::normalize_request_with_hint(request, provider_hint)
    }
    fn normalize_response(
        &self,
        response: &crate::json::Json,
    ) -> Option<crate::codec::response::AnnotatedLlmResponse> {
        crate::codec::resolve::normalize_response(response)
    }
}
impl nemo_relay_observability::host::ObservabilityHost for CoreObservabilityHost {
    fn register_subscriber(
        &self,
        name: &str,
        callback: crate::api::runtime::EventSubscriberFn,
    ) -> crate::error::Result<()> {
        crate::api::subscriber::register_subscriber(name, callback)
    }
    fn deregister_subscriber(&self, name: &str) -> crate::error::Result<bool> {
        crate::api::subscriber::deregister_subscriber(name)
    }
    fn flush_subscribers(&self) -> crate::error::Result<()> {
        crate::api::subscriber::flush_subscribers()
    }
    fn w3c_span_context(
        &self,
        traceparent: Option<&str>,
        tracestate: Option<&str>,
    ) -> Option<opentelemetry::trace::SpanContext> {
        crate::api::runtime::scope_stack::w3c_span_context(traceparent, tracestate)
    }
    fn record_runtime_diagnostic(
        &self,
        code: &str,
        field: Option<String>,
        message: String,
        count: u64,
    ) {
        record_plugin_runtime_diagnostic(code, field, message, count);
    }
    fn set_full_payloads_enabled(&self, enabled: bool) -> crate::error::Result<()> {
        crate::api::runtime::global_context()
            .write()
            .map_err(|error| crate::error::FlowError::Internal(error.to_string()))?
            .observability_full_payloads_enabled = enabled;
        Ok(())
    }
    fn try_scope_register_subscriber(
        &self,
        scope: &uuid::Uuid,
        name: &str,
        callback: crate::api::runtime::EventSubscriberFn,
    ) -> crate::error::Result<()> {
        crate::api::subscriber::try_scope_register_subscriber(scope, name, callback)
    }
    fn try_scope_deregister_subscriber(
        &self,
        scope: &uuid::Uuid,
        name: &str,
    ) -> crate::error::Result<bool> {
        crate::api::subscriber::try_scope_deregister_subscriber(scope, name)
    }
    fn scope_deregister_subscriber(
        &self,
        scope: &uuid::Uuid,
        name: &str,
    ) -> crate::error::Result<bool> {
        crate::api::subscriber::scope_deregister_subscriber(scope, name)
    }
    fn is_trajectory_parent(&self, parent: uuid::Uuid) -> bool {
        crate::api::runtime::current_scope_stack()
            .read()
            .map(|stack| stack.root_uuid() == parent || stack.is_propagated_parent(parent))
            .unwrap_or(false)
    }
    fn record_plugin_diagnostic(
        &self,
        diagnostic: nemo_relay_plugin::component::RuntimeDiagnostic,
    ) {
        crate::plugin::record_active_plugin_runtime_diagnostic(diagnostic);
    }
}
pub(crate) fn core_host() -> std::sync::Arc<dyn nemo_relay_observability::host::ObservabilityHost> {
    std::sync::Arc::new(CoreObservabilityHost)
}
pub(crate) fn event_view(
    event: &crate::api::event::Event,
) -> nemo_relay_observability::normalization::EventView<'_> {
    nemo_relay_observability::normalization::EventView::new(event, &CoreObservabilityHost)
}

pub(crate) fn record_plugin_runtime_diagnostic(
    code: &str,
    field: Option<String>,
    message: String,
    count: u64,
) {
    crate::plugin::record_active_plugin_runtime_diagnostic(crate::plugin::RuntimeDiagnostic {
        code: code.to_string(),
        component: "observability".to_string(),
        field,
        message,
        session_id: None,
        count,
    });
}
