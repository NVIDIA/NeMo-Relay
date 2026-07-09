// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility tests for the public LLM optimization wire contract.

use nemo_relay_types::codec::optimization::{
    LlmOptimizationContribution, LlmOptimizationEvidenceQuality, LlmOptimizationKind,
    LlmOptimizationPayload, LlmOptimizationTokenImpact, LlmOptimizationTokens,
};
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Serialize)]
struct CustomPayload {
    evidence: String,
}

impl LlmOptimizationPayload for CustomPayload {
    const SCHEMA_NAME: &'static str = "example.custom_optimization";
    const SCHEMA_VERSION: &'static str = "3";
}

#[test]
fn standard_kind_constants_keep_their_exact_wire_values() {
    assert_eq!(LlmOptimizationKind::INPUT_COMPRESSION, "input_compression");
    assert_eq!(LlmOptimizationKind::MODEL_ROUTING, "model_routing");
    assert_eq!(
        serde_json::to_value(LlmOptimizationKind::input_compression()).unwrap(),
        json!("input_compression")
    );
    assert_eq!(
        serde_json::to_value(LlmOptimizationKind::model_routing()).unwrap(),
        json!("model_routing")
    );
}

#[test]
fn custom_kinds_payloads_and_future_fields_round_trip() {
    let mut contribution = LlmOptimizationContribution::new("example", "energy_reduction")
        .with_payload(&CustomPayload {
            evidence: "measured".to_string(),
        })
        .unwrap();
    contribution
        .extra
        .insert("future_field".to_string(), json!({"v": 2}));
    let decoded: LlmOptimizationContribution =
        serde_json::from_value(serde_json::to_value(&contribution).unwrap()).unwrap();
    assert_eq!(decoded.kind.as_str(), "energy_reduction");
    assert_eq!(
        decoded.payload_schema.as_ref().unwrap().name,
        "example.custom_optimization"
    );
    assert_eq!(decoded.extra["future_field"], json!({"v": 2}));
}

#[test]
fn saved_prompt_tokens_remain_explicit_on_the_wire() {
    let impact = LlmOptimizationTokenImpact {
        saved: Some(LlmOptimizationTokens::saved_prompt(42)),
        quality: Some(LlmOptimizationEvidenceQuality::Estimated),
        estimation_method: Some("tokenizer-v1".to_string()),
        ..LlmOptimizationTokenImpact::default()
    };
    let wire = serde_json::to_value(impact).unwrap();
    assert_eq!(wire["saved"]["prompt_tokens"], 42);
    assert_eq!(wire["saved"]["total_tokens"], 42);
}

#[test]
fn canonical_all_fields_fixture_is_lossless_and_open() {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/llm_optimization_contribution_v1.json"
    ))
    .unwrap();
    let contribution: LlmOptimizationContribution =
        serde_json::from_value(fixture.clone()).unwrap();

    assert_eq!(contribution.kind.as_str(), "custom_energy_optimization");
    assert_eq!(
        contribution
            .token_impact
            .as_ref()
            .and_then(|impact| impact.saved.as_ref())
            .and_then(|saved| saved.cache_write_tokens),
        Some(4)
    );
    assert_eq!(
        contribution.extra["future_top_level_field"],
        json!({"preserved": true})
    );
    assert_eq!(serde_json::to_value(contribution).unwrap(), fixture);
}
