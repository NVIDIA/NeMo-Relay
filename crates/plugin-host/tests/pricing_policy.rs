// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selected runtime pricing applies to all runtime-bound exporters. Run with nextest.

use nemo_relay_model_pricing::{PricingCatalog, PricingResolver, set_active_pricing_resolver};
use nemo_relay_observability::{OpenTelemetryType, atif::AtifAgentInfo};
use nemo_relay_runtime::composition::{RuntimeServices, install_runtime_services};
use nemo_relay_types::{
    api::event::{BaseEvent, Event, EventCategory, ScopeCategory, ScopeEvent},
    codec::{
        pricing::CostEstimator,
        response::{CostEstimate, Usage},
    },
};
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};
use serde_json::json;
use std::sync::Arc;

struct SelectedPrice;
impl CostEstimator for SelectedPrice {
    fn estimate_cost_for_provider(
        &self,
        _: Option<&str>,
        _: &str,
        _: &Usage,
    ) -> Option<CostEstimate> {
        Some(
            serde_json::from_value(
                json!({"total": 0.25, "currency": "USD", "source": "model_pricing"}),
            )
            .unwrap(),
        )
    }
}

fn assert_exported_cost(expected: Option<f64>) {
    // A conflicting standalone catalog must not override the selected runtime policy.
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1, "entries": [{"provider": "openai", "model_id": "policy-model",
            "pricing_as_of": "2026-10-30", "pricing_source": "test",
            "rates": {"input_per_million": 1.0, "output_per_million": 2.0},
            "prompt_cache": {"read_accounting": "included_in_prompt_tokens"}}]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();

    let uuid = uuid::Uuid::now_v7();
    let start = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .name("openai")
            .data(json!({"headers": {}, "content": {
            "model": "policy-model", "messages": [{"role": "user", "content": "hello"}]}}))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::llm(),
        None,
    ));
    let end = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().uuid(uuid).name("openai").data(json!({"model": "policy-model",
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000}})).build(),
        ScopeCategory::End, Vec::new(), EventCategory::llm(), None,
    ));

    let atif = nemo_relay_plugin_host::exporter::atif(
        "policy".into(),
        AtifAgentInfo {
            name: "policy".into(),
            version: "1".into(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        },
    );
    for event in [&start, &end] {
        atif.subscriber()(event);
    }
    let trajectory = atif.export().unwrap();
    let actual = trajectory
        .steps
        .iter()
        .filter_map(|step| step.metrics.as_ref())
        .find_map(|metrics| metrics.cost_usd);
    assert_eq!(actual, expected, "ATIF must use selected runtime pricing");

    for (projection, key) in [
        (OpenTelemetryType::Full, "nemo_relay.llm.cost.total"),
        (OpenTelemetryType::OpenInference, "llm.cost.total"),
    ] {
        let sink = InMemorySpanExporterBuilder::new().build();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(sink.clone())
            .build();
        let subscriber = nemo_relay_plugin_host::exporter::opentelemetry_from_provider_with_type(
            provider, "policy", projection,
        );
        for event in [&start, &end] {
            subscriber.subscriber()(event);
        }
        subscriber.force_flush().unwrap();
        let spans = sink.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        let actual = spans[0]
            .attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == key)
            .map(|attribute| attribute.value.to_string().parse::<f64>().unwrap());
        assert_eq!(
            actual, expected,
            "{projection:?} must use selected runtime pricing"
        );
    }
}

#[test]
fn disabled_runtime_pricing_stays_disabled_in_exporters() {
    install_runtime_services(RuntimeServices::default);
    assert_exported_cost(None);
}

#[test]
fn custom_runtime_estimator_controls_exporter_fallbacks() {
    install_runtime_services(|| RuntimeServices {
        pricing_snapshot: || Arc::new(SelectedPrice),
        ..Default::default()
    });
    assert_exported_cost(Some(0.25));
}
