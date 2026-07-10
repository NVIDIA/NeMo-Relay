// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conformance tests for the native plugin optimization contribution surface.

use nemo_relay_plugin::{
    DataSchema, LlmOptimizationContribution, LlmOptimizationEvidenceQuality, LlmOptimizationKind,
    LlmOptimizationModel, LlmOptimizationModelTransition, LlmOptimizationSummary,
    LlmOptimizationSummaryStatus, LlmOptimizationTokenImpact, LlmOptimizationTokens, LlmRequest,
    LlmRequestInterceptOutcome,
};
use serde_json::{Value as Json, json};

#[test]
fn native_plugin_sdk_round_trips_the_canonical_contribution_contract() {
    let fixture: Json = serde_json::from_str(include_str!(
        "../../types/tests/fixtures/llm_optimization_contribution_v1.json"
    ))
    .unwrap();
    let contribution: LlmOptimizationContribution =
        serde_json::from_value(fixture.clone()).unwrap();

    assert_ne!(contribution.kind, LlmOptimizationKind::input_compression());
    assert_ne!(contribution.kind, LlmOptimizationKind::model_routing());
    assert_eq!(
        contribution.extra["future_top_level_field"],
        fixture["future_top_level_field"]
    );

    let outcome = LlmRequestInterceptOutcome::new(
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({"model": "test"}),
        },
        None,
    )
    .with_optimization_contribution(contribution);
    let wire = serde_json::to_value(outcome).unwrap();
    assert_eq!(wire["optimization_contributions"][0], fixture);

    // Keep every nested public contract type covered by the SDK's compile-time surface.
    fn exported<T>() {}
    exported::<DataSchema>();
    exported::<LlmOptimizationEvidenceQuality>();
    exported::<LlmOptimizationModel>();
    exported::<LlmOptimizationModelTransition>();
    exported::<LlmOptimizationSummary>();
    exported::<LlmOptimizationSummaryStatus>();
    exported::<LlmOptimizationTokenImpact>();
    exported::<LlmOptimizationTokens>();
}
