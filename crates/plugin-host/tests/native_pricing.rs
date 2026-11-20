// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native plugins price usage with the host's selected pricing policy.

use std::sync::Arc;

use nemo_relay_plugin::{CostEstimate, PluginRuntime, Usage};
use nemo_relay_plugin_host::dynamic::native_checked::HostRuntime;
use nemo_relay_runtime::composition::{RuntimeServices, install_runtime_services};
use nemo_relay_types::codec::pricing::CostEstimator;
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use serde_json::json;

struct HostPrice;
impl CostEstimator for HostPrice {
    fn estimate_cost_for_provider(
        &self,
        provider: Option<&str>,
        model: &str,
        usage: &Usage,
    ) -> Option<CostEstimate> {
        (provider == Some("openai") && model == "priced-model").then(|| {
            serde_json::from_value(json!({
                "total": usage.prompt_tokens.unwrap_or_default() as f64 / 1000.0,
                "currency": "USD",
                "source": "model_pricing"
            }))
            .unwrap()
        })
    }
}

#[tokio::test]
async fn sdk_pricing_operations_use_the_host_pricing_policy() {
    install_runtime_services(|| RuntimeServices {
        pricing_snapshot: || Arc::new(HostPrice),
        attach_estimated_cost: |response, provider| {
            if let (Some(model), Some(usage)) = (response.model.clone(), response.usage.as_mut())
                && usage.cost.is_none()
            {
                usage.cost = HostPrice.estimate_cost_for_provider(provider, &model, usage);
            }
        },
        ..RuntimeServices::default()
    });
    let owner = HostRuntime::new("pricing".into()).unwrap();
    let runtime = PluginRuntime::from_abi(owner.runtime_box());
    let usage = Usage {
        prompt_tokens: Some(250),
        ..Usage::default()
    };

    let priced = runtime.estimate_cost_for_provider(Some("openai"), "priced-model", &usage);
    let unpriced = runtime.estimate_cost_for_provider(None, "priced-model", &usage);
    let response: AnnotatedLlmResponse = serde_json::from_value(json!({
        "model": "priced-model",
        "usage": {"prompt_tokens": 500}
    }))
    .unwrap();
    let response_cost = runtime.response_cost_for_provider(&response, Some("openai"));
    let reported: AnnotatedLlmResponse = serde_json::from_value(json!({
        "model": "priced-model",
        "usage": {"prompt_tokens": 500, "cost": {"total": 9.0, "currency": "USD", "source": "provider_reported"}}
    }))
    .unwrap();
    let reported_cost = runtime.response_cost_for_provider(&reported, Some("openai"));
    let raw = owner.runtime_box();
    let invalid = raw.call_sync(
        "pricing.estimate_cost_for_provider".into(),
        nemo_relay_plugin::abi::Json::Null,
    );
    owner.shutdown().await.unwrap();

    assert_eq!(priced.unwrap().and_then(|cost| cost.total), Some(0.25));
    assert_eq!(unpriced.unwrap(), None);
    assert_eq!(
        response_cost.unwrap().and_then(|cost| cost.total),
        Some(0.5)
    );
    assert_eq!(
        reported_cost.unwrap().and_then(|cost| cost.total),
        Some(9.0)
    );
    assert!(matches!(
        invalid,
        nemo_relay_plugin::abi::v1_0::Result::Err(_)
    ));
}
