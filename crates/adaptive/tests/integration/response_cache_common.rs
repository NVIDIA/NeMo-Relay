// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by the response-cache integration test binaries. Included
//! via `#[path]` from each `[[test]]` target, so this file is not a test
//! target of its own.

use nemo_relay::api::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay::api::runtime::LlmExecutionNextFn;
use nemo_relay::plugin::{PluginConfig, initialize_plugins};
use nemo_relay_adaptive::plugin_component::{ComponentSpec, register_adaptive_component};
use nemo_relay_adaptive::{AdaptiveConfig, ResponseCacheConfig};
use serde_json::{Value as Json, json};

pub fn chat_request(prompt: &str) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }),
    }
}

/// Activates the response cache as a section of the adaptive plugin component
/// (its only supported configuration surface — there is no standalone kind).
pub async fn activate_cache(config: ResponseCacheConfig) {
    register_adaptive_component().unwrap();
    let adaptive = AdaptiveConfig {
        response_cache: Some(config),
        ..AdaptiveConfig::default()
    };
    let report = initialize_plugins(PluginConfig {
        components: vec![ComponentSpec::new(adaptive).into()],
        ..PluginConfig::default()
    })
    .await
    .unwrap();
    assert!(
        report.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        report.diagnostics
    );
}

pub async fn call(provider: &LlmExecutionNextFn, request: LlmRequest) -> Json {
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(request)
            .func(provider.clone())
            .model_name("gpt-4o")
            .build(),
    )
    .await
    .unwrap()
}
