// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for profile in the NeMo Relay adaptive crate.

use chrono::Utc;

use crate::acg::profile::*;
use crate::acg::prompt_ir::SpanId;
use crate::acg::types::AgentIdentity;

fn sample_agent_identity() -> AgentIdentity {
    AgentIdentity {
        agent_id: "agent-profile".to_string(),
        template_version: "v1".to_string(),
        toolset_hash: "hash".to_string(),
        model_family: "claude".to_string(),
        tenant_scope: "tenant-a".to_string(),
    }
}

#[test]
fn behavioral_profile_has_sufficient_data_respects_threshold() {
    let profile = BehavioralProfile {
        agent_identity: sample_agent_identity(),
        profile_version: "1".to_string(),
        block_stability: vec![BlockStabilityScore {
            span_id: SpanId("system-0".to_string()),
            classification: StabilityClass::Stable,
            score: 0.99,
            confidence: 0.95,
            observation_count: 6,
        }],
        session_duration: Some(DistributionSummary {
            p50: 12.0,
            p90: 20.0,
            p99: 25.0,
            sample_count: 6,
        }),
        inter_call_gap: Some(DistributionSummary {
            p50: 1.5,
            p90: 2.5,
            p99: 3.0,
            sample_count: 6,
        }),
        parallelism: Some(ParallelismPattern {
            has_fanouts: true,
            typical_fanout_width: Some(2),
            predominantly_serial: false,
        }),
        tool_usage_phases: vec![ToolUsagePhase {
            phase_label: "research".to_string(),
            tools: vec!["search".to_string(), "browse".to_string()],
            phase_reach_rate: 0.8,
        }],
        dominant_archetype: Some(SessionArchetype::ToolHeavyLoop),
        observation_count: 6,
        minimum_observations: 5,
        updated_at: Utc::now(),
    };

    assert!(profile.has_sufficient_data());
    assert!(
        !BehavioralProfile {
            minimum_observations: 7,
            ..profile
        }
        .has_sufficient_data()
    );
}

#[test]
fn profile_types_round_trip_through_serde() {
    for stability in [
        StabilityClass::Stable,
        StabilityClass::SemiStable,
        StabilityClass::Variable,
    ] {
        let json = serde_json::to_string(&stability).unwrap();
        let restored: StabilityClass = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, stability);
    }

    for archetype in [
        SessionArchetype::FastAnswer,
        SessionArchetype::ToolHeavyLoop,
        SessionArchetype::LongRunningWorkflow,
        SessionArchetype::MultiTurnTroubleshooting,
    ] {
        let json = serde_json::to_string(&archetype).unwrap();
        let restored: SessionArchetype = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, archetype);
    }

    let summary = DistributionSummary {
        p50: 1.0,
        p90: 2.0,
        p99: 3.0,
        sample_count: 4,
    };
    let restored: DistributionSummary =
        serde_json::from_str(&serde_json::to_string(&summary).unwrap()).unwrap();
    assert_eq!(restored, summary);
}
