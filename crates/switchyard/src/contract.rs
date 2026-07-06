// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON compatibility types for the Switchyard Decision API.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Routing request schema supported by this plugin.
pub const ROUTING_REQUEST_SCHEMA_VERSION: &str = "switchyard.routing_request.v1";
/// Routing decision schema supported by this plugin.
pub const ROUTING_DECISION_SCHEMA_VERSION: &str = "switchyard.routing_decision.v1";

/// Request-time materialization supplied to Switchyard.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum RequestMaterialization {
    /// Identity, protocol, summary, and attempt only.
    None,
    /// Baseline summary without current request material.
    SummaryOnly,
    /// Latest user prompt.
    LatestUserPrompt,
    /// Bounded recent message window.
    RecentMessageWindow,
    /// Relay-normalized request plus its provider body.
    AnnotatedRequest,
    /// Complete provider request body.
    FullBody,
}

/// Switchyard profile selection.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DecisionProfile {
    /// Profile ID loaded by Switchyard.
    pub profile_id: String,
    /// Request materialization mode.
    pub request_materialization: RequestMaterialization,
}

/// Normalized Relay identity.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RequestIdentity {
    /// Stable session identifier.
    pub session_id: String,
    /// Per-request identifier.
    pub request_id: String,
    /// Optional turn identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    /// Optional parent scope identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_scope_id: Option<String>,
    /// Optional root scope identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_scope_id: Option<String>,
    /// Harness name.
    pub harness: String,
    /// Request source.
    pub source: String,
    /// Optional resolved work owner.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner_id: Option<String>,
    /// Native, explicit, or synthetic identity quality.
    pub quality: String,
}

/// Inbound protocol context.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RequestProtocol {
    /// Inbound protocol profile.
    pub inbound_profile: String,
    /// Inbound endpoint.
    pub inbound_endpoint: String,
    /// Response profile expected by the harness.
    pub desired_response_profile: String,
}

/// Cheap provider-request summary.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RequestSummary {
    /// Client-requested model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_requested_model: Option<String>,
    /// Optional prompt-token estimate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_estimate: Option<u64>,
    /// Number of tools in the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_count_in_payload: Option<u64>,
    /// Whether a system prompt is present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_system_prompt: Option<bool>,
}

/// Routing-attempt context. Additive fields are ignored by older Switchyard servers.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DecisionAttempt {
    /// One-indexed routing attempt.
    pub routing_attempt: u32,
    /// Maximum Decision API attempts.
    pub max_routing_attempts: u32,
    /// Previously selected backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_route: Option<String>,
    /// Reason another decision is requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_reason: Option<String>,
}

/// Canonical routing request sent by Relay.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RoutingRequest {
    /// Schema identifier.
    pub schema_version: String,
    /// Switchyard profile selection.
    pub decision_profile: DecisionProfile,
    /// Normalized identity.
    pub identity: RequestIdentity,
    /// Inbound protocol.
    pub protocol: RequestProtocol,
    /// Cheap request summary.
    pub request_summary: RequestSummary,
    /// Optional request materialization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_request: Option<Value>,
    /// Attempt metadata.
    pub attempt: DecisionAttempt,
}

/// Router metadata returned by Switchyard.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DecisionProvider {
    /// Router name.
    pub name: String,
    /// Router version.
    pub version: String,
}

/// Selected target returned by Switchyard.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RoutingTarget {
    /// Tier label.
    pub tier: String,
    /// Selected model.
    pub target_model: String,
    /// Backend binding ID.
    pub backend_id: String,
    /// Target protocol profile.
    pub target_protocol_profile: String,
    /// Target endpoint.
    pub target_endpoint: String,
}

/// Canonical Switchyard routing decision.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RoutingDecision {
    /// Schema identifier.
    pub schema_version: String,
    /// Decision identifier.
    pub decision_id: String,
    /// Router metadata.
    pub router: DecisionProvider,
    /// Selected target.
    pub route: RoutingTarget,
    /// Optional confidence.
    #[serde(default)]
    pub confidence: Option<f64>,
    /// Optional reason code.
    #[serde(default)]
    pub reason_code: Option<String>,
    /// Optional reason summary.
    #[serde(default)]
    pub reason_summary: Option<String>,
    /// Additive router metadata.
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    /// Unknown additive response fields.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn current_switchyard_decision_contract_accepts_additive_fields() {
        let decision: RoutingDecision = serde_json::from_value(json!({
            "schema_version": "switchyard.routing_decision.v1",
            "decision_id": "decision-1",
            "router": {"name": "cascade", "version": "1"},
            "route": {
                "tier": "strong",
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
    }

    #[test]
    fn malformed_decisions_are_rejected_by_serde() {
        let missing_route = json!({
            "schema_version": "switchyard.routing_decision.v1",
            "decision_id": "decision-1",
            "router": {"name": "cascade", "version": "1"}
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
}
