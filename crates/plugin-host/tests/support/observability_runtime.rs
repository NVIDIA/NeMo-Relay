// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core runtime services supplied to observability exporters.

struct CoreObservabilityHost;
impl nemo_relay_types::codec::normalization::LlmNormalizer for CoreObservabilityHost {
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
impl nemo_relay_observability::host::ObservabilityHost for CoreObservabilityHost {
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
            .observability_full_payloads_enabled = enabled;
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
        nemo_relay_plugin_host::record_active_plugin_runtime_diagnostic(diagnostic);
    }
}
pub(crate) fn core_host() -> std::sync::Arc<dyn nemo_relay_observability::host::ObservabilityHost> {
    crate::test_composition::ensure_runtime_services();
    std::sync::Arc::new(CoreObservabilityHost)
}
pub(crate) fn event_view(
    event: &nemo_relay_runtime::api::event::Event,
) -> nemo_relay_observability::normalization::EventView<'_> {
    crate::test_composition::ensure_runtime_services();
    nemo_relay_observability::normalization::EventView::new(event, &CoreObservabilityHost)
}

pub(crate) fn record_plugin_runtime_diagnostic(
    code: &str,
    field: Option<String>,
    message: String,
    count: u64,
) {
    nemo_relay_plugin_host::record_active_plugin_runtime_diagnostic(
        nemo_relay_plugin_host::RuntimeDiagnostic {
            code: code.to_string(),
            component: "observability".to_string(),
            field,
            message,
            session_id: None,
            count,
        },
    );
}
