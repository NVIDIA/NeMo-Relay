// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core runtime services supplied to observability exporters.

struct RuntimeObservabilityHost;
impl nemo_relay_types::codec::normalization::LlmNormalizer for RuntimeObservabilityHost {
    fn normalize_request(
        &self,
        request: &nemo_relay_runtime::api::llm::LlmRequest,
        provider_hint: Option<&str>,
    ) -> Option<nemo_relay_types::codec::request::AnnotatedLlmRequest> {
        nemo_relay_codecs::resolve::normalize_request_with_hint(request, provider_hint)
    }
    fn normalize_response(
        &self,
        response: &nemo_relay_types::Json,
    ) -> Option<nemo_relay_types::codec::response::AnnotatedLlmResponse> {
        nemo_relay_runtime::codec::RuntimeLlmNormalizer.normalize_response(response)
    }
}
impl nemo_relay_observability::host::ObservabilityHost for RuntimeObservabilityHost {
    fn register_subscriber(
        &self,
        name: &str,
        callback: nemo_relay_runtime::api::runtime::EventSubscriberFn,
    ) -> nemo_relay_types::error::Result<()> {
        nemo_relay_runtime::api::subscriber::register_subscriber(name, callback)
    }
    fn deregister_subscriber(&self, name: &str) -> nemo_relay_types::error::Result<bool> {
        nemo_relay_runtime::api::subscriber::deregister_subscriber(name)
    }
    fn flush_subscribers(&self) -> nemo_relay_types::error::Result<()> {
        nemo_relay_runtime::api::subscriber::flush_subscribers()
    }
    fn w3c_span_context(
        &self,
        traceparent: Option<&str>,
        tracestate: Option<&str>,
    ) -> Option<opentelemetry::trace::SpanContext> {
        nemo_relay_runtime::api::runtime::scope_stack::w3c_span_context(traceparent, tracestate)
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
    fn set_full_payloads_enabled(&self, enabled: bool) -> nemo_relay_types::error::Result<()> {
        nemo_relay_runtime::api::runtime::global_context()
            .write()
            .map_err(|error| nemo_relay_types::error::FlowError::Internal(error.to_string()))?
            .set_observability_full_payloads_enabled(enabled);
        Ok(())
    }
    fn try_scope_register_subscriber(
        &self,
        scope: &uuid::Uuid,
        name: &str,
        callback: nemo_relay_runtime::api::runtime::EventSubscriberFn,
    ) -> nemo_relay_types::error::Result<()> {
        nemo_relay_runtime::api::subscriber::try_scope_register_subscriber(scope, name, callback)
    }
    fn try_scope_deregister_subscriber(
        &self,
        scope: &uuid::Uuid,
        name: &str,
    ) -> nemo_relay_types::error::Result<bool> {
        nemo_relay_runtime::api::subscriber::try_scope_deregister_subscriber(scope, name)
    }
    fn scope_deregister_subscriber(
        &self,
        scope: &uuid::Uuid,
        name: &str,
    ) -> nemo_relay_types::error::Result<bool> {
        nemo_relay_runtime::api::subscriber::scope_deregister_subscriber(scope, name)
    }
    fn is_trajectory_parent(&self, parent: uuid::Uuid) -> bool {
        nemo_relay_runtime::api::runtime::current_scope_stack()
            .read()
            .map(|stack| stack.root_uuid() == parent || stack.is_propagated_parent(parent))
            .unwrap_or(false)
    }
    fn record_plugin_diagnostic(
        &self,
        diagnostic: nemo_relay_plugin::component::RuntimeDiagnostic,
    ) {
        crate::record_active_plugin_runtime_diagnostic(diagnostic);
    }
}
pub(crate) fn runtime_host() -> std::sync::Arc<dyn nemo_relay_observability::host::ObservabilityHost>
{
    std::sync::Arc::new(RuntimeObservabilityHost)
}

pub(crate) fn record_plugin_runtime_diagnostic(
    code: &str,
    field: Option<String>,
    message: String,
    count: u64,
) {
    crate::record_active_plugin_runtime_diagnostic(crate::RuntimeDiagnostic {
        code: code.to_string(),
        component: "observability".to_string(),
        field,
        message,
        session_id: None,
        count,
    });
}

#[cfg(test)]
pub(crate) fn event_view(
    event: &nemo_relay_types::api::event::Event,
) -> nemo_relay_observability::normalization::EventView<'_> {
    crate::initialize_runtime();
    nemo_relay_observability::normalization::EventView::new(event, &RuntimeObservabilityHost)
}
