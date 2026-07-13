// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the public Switchyard Decision API contract.

use nemo_relay_switchyard::contract::{DecisionAttempt, RoutingDecision};
use serde_json::json;

#[test]
fn current_switchyard_decision_contract_accepts_additive_fields() {
    let decision: RoutingDecision = serde_json::from_value(json!({
        "schema_version": "switchyard.routing_decision.v1",
        "decision_id": "decision-1",
        "router": {"name": "stage_router", "version": "1"},
        "route": {
            "tier": "capable",
            "target_model": "model-a",
            "backend_id": "backend-a",
            "target_protocol_profile": "openai_chat",
            "target_endpoint": "/v1/chat/completions"
        },
        "confidence": 0.8,
        "future_field": {"safe": true}
    }))
    .unwrap();
    assert_eq!(
        decision.extra.get("future_field"),
        Some(&json!({"safe": true}))
    );
    assert!(decision.baseline_route.is_none());
}

#[test]
fn current_switchyard_decision_contract_accepts_an_explicit_baseline() {
    let decision: RoutingDecision = serde_json::from_value(json!({
        "schema_version": "switchyard.routing_decision.v1",
        "decision_id": "decision-1",
        "router": {"name": "stage", "version": "1"},
        "route": {
            "tier": "efficient",
            "target_model": "model-small",
            "backend_id": "backend-small",
            "target_protocol_profile": "openai_chat",
            "target_endpoint": "/v1/chat/completions"
        },
        "baseline_route": {
            "tier": "capable",
            "target_model": "model-large",
            "backend_id": "backend-large",
            "target_protocol_profile": "anthropic_messages",
            "target_endpoint": "/v1/messages"
        }
    }))
    .unwrap();
    let baseline = decision.baseline_route.unwrap();
    assert_eq!(baseline.backend_id, "backend-large");
    assert_eq!(baseline.target_protocol_profile, "anthropic_messages");
}

#[test]
fn malformed_decisions_are_rejected_by_serde() {
    let missing_route = json!({
        "schema_version": "switchyard.routing_decision.v1",
        "decision_id": "decision-1",
        "router": {"name": "stage_router", "version": "1"}
    });
    assert!(serde_json::from_value::<RoutingDecision>(missing_route).is_err());
}

#[test]
fn additive_retry_metadata_stays_optional_on_the_wire() {
    let attempt: DecisionAttempt = serde_json::from_value(json!({
        "routing_attempt": 1,
        "max_routing_attempts": 4
    }))
    .unwrap();
    assert!(attempt.previous_route.is_none());
    assert!(attempt.retry_reason.is_none());
}
