// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for topological convergence detection in the ACG learner.

use std::sync::{Arc, RwLock};

use chrono::Utc;
use nemo_relay::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use nemo_relay_adaptive::acg::stability::StabilityThresholds;
use nemo_relay_adaptive::acg_learner::AcgLearner;
use nemo_relay_adaptive::learner::traits::Learner;
use nemo_relay_adaptive::types::cache::HotCache;
use nemo_relay_adaptive::types::records::{CallKind, CallRecord, RunRecord};
use nemo_relay_adaptive::{ConvergenceConfig, InMemoryBackend, StorageBackendDyn};
use uuid::Uuid;

fn identical_request() -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are a helpful assistant.".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Summarize this.".to_string()),
                name: None,
            },
        ],
        model: Some("gpt-4o".to_string()),
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    }
}

fn run_with_requests(requests: Vec<AnnotatedLlmRequest>) -> RunRecord {
    let now = Utc::now();
    RunRecord {
        id: Uuid::now_v7(),
        agent_id: "convergence-agent".to_string(),
        calls: requests
            .into_iter()
            .map(|request| CallRecord {
                kind: CallKind::Llm,
                name: "planner".to_string(),
                started_at: now,
                ended_at: Some(now),
                metadata_snapshot: None,
                output_tokens: None,
                prompt_tokens: None,
                total_tokens: None,
                model_name: None,
                tool_call_count: None,
                annotated_request: Some(request.into()),
                annotated_response: None,
            })
            .collect(),
        started_at: now,
        ended_at: Some(now),
    }
}

fn empty_cache() -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }))
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_declares_convergence_before_window_exhausted() {
    let observation_window = 20;
    let stability_window = 3;
    let learner = AcgLearner::new_with_convergence(
        "convergence-agent",
        observation_window,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window,
        }),
    );
    let backend = InMemoryBackend::new();
    let hot_cache = empty_cache();
    let request = identical_request();

    let mut converged_at = None;
    let mut agent_observations_at_convergence = 0;
    for iteration in 0..observation_window {
        let run = run_with_requests(vec![request.clone()]);
        learner
            .process_run(&run, &backend, &hot_cache)
            .await
            .unwrap();

        let stability = backend
            .load_stability("convergence-agent")
            .await
            .unwrap()
            .expect("stability should be stored");
        if stability.converged {
            converged_at = Some(iteration + 1);
            agent_observations_at_convergence = backend
                .load_observations("convergence-agent")
                .await
                .unwrap()
                .map(|observations| observations.len())
                .unwrap_or(0);
            break;
        }
    }

    assert!(
        converged_at.is_some(),
        "expected convergence to be declared before exhausting the observation window"
    );
    assert!(
        converged_at.unwrap() < observation_window,
        "convergence should be declared before the observation window is exhausted"
    );
    assert!(
        converged_at.unwrap() >= stability_window,
        "convergence should require at least stability_window epochs"
    );

    // Continue running after convergence to verify the cached result is reused
    // and observations are no longer updated.
    for _ in 0..3 {
        let run = run_with_requests(vec![request.clone()]);
        learner
            .process_run(&run, &backend, &hot_cache)
            .await
            .unwrap();
    }

    let final_stability = backend
        .load_stability("convergence-agent")
        .await
        .unwrap()
        .expect("stability should still be stored");
    assert!(
        final_stability.converged,
        "cached stability result should remain converged"
    );

    let final_agent_observations = backend
        .load_observations("convergence-agent")
        .await
        .unwrap()
        .expect("agent aggregate observations should remain stored after convergence");
    assert!(
        !final_agent_observations.is_empty(),
        "agent aggregate observations should be non-empty after convergence"
    );
    assert_eq!(
        final_agent_observations.len(),
        agent_observations_at_convergence,
        "agent aggregate observation storage should be skipped after convergence"
    );
}
