// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Benchmark comparing observations-to-decision for the ACG learner with and
//! without topological convergence detection.
//!
//! The synthetic prompt profile consists of 50 identical observations. Without
//! convergence detection the learner processes the full observation sequence
//! before deciding, while topological convergence detection declares
//! convergence after the configured stability window.
#![allow(missing_docs)]

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use chrono::Utc;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nemo_relay::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use nemo_relay_adaptive::acg::build_prompt_ir;
use nemo_relay_adaptive::acg::prompt_ir::PromptIR;
use nemo_relay_adaptive::acg::stability::StabilityThresholds;
use nemo_relay_adaptive::acg_learner::AcgLearner;
use nemo_relay_adaptive::config::ConvergenceConfig;
use nemo_relay_adaptive::learner::traits::Learner;
use nemo_relay_adaptive::types::cache::HotCache;
use nemo_relay_adaptive::types::records::{CallKind, CallRecord, RunRecord};
use nemo_relay_adaptive::{InMemoryBackend, StorageBackendDyn};
use uuid::Uuid;

static RUNTIME: LazyLock<tokio::runtime::Runtime> =
    LazyLock::new(|| tokio::runtime::Runtime::new().expect("tokio runtime"));

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

fn build_stable_observations(count: usize) -> Vec<PromptIR> {
    let request = identical_request();
    (0..count)
        .map(|_| build_prompt_ir(&request).expect("valid prompt IR"))
        .collect()
}

fn build_run(request: AnnotatedLlmRequest) -> RunRecord {
    let now = Utc::now();
    RunRecord {
        id: Uuid::now_v7(),
        agent_id: "convergence-agent".to_string(),
        calls: vec![CallRecord {
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
        }],
        started_at: now,
        ended_at: Some(now),
    }
}

fn empty_cache() -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: HashMap::new(),
        acg_profile_observation_counts: HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }))
}

fn observations_without_convergence(observations: &[PromptIR]) -> usize {
    RUNTIME.block_on(async {
        let learner = AcgLearner::new("convergence-agent", 100, StabilityThresholds::default());
        let backend = InMemoryBackend::new();
        let cache = empty_cache();

        for _ in 0..observations.len() {
            let run = build_run(identical_request());
            learner
                .process_run(&run, &backend, &cache)
                .await
                .expect("process run");
        }
        observations.len()
    })
}

fn observations_with_convergence(observations: &[PromptIR]) -> usize {
    RUNTIME.block_on(async {
        let config = ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        };
        let learner = AcgLearner::new_with_convergence(
            "convergence-agent",
            100,
            StabilityThresholds::default(),
            Some(config),
        );
        let backend = InMemoryBackend::new();
        let cache = empty_cache();

        for index in 0..observations.len() {
            let run = build_run(identical_request());
            learner
                .process_run(&run, &backend, &cache)
                .await
                .expect("process run");

            let stability = backend
                .load_stability("convergence-agent")
                .await
                .expect("load stability")
                .expect("stability exists");
            if stability.converged {
                return index + 1;
            }
        }
        observations.len()
    })
}

fn bench_convergence(c: &mut Criterion) {
    let observations = build_stable_observations(50);

    c.bench_function("without_convergence", |b| {
        b.iter(|| observations_without_convergence(black_box(&observations)))
    });

    c.bench_function("with_convergence", |b| {
        b.iter(|| observations_with_convergence(black_box(&observations)))
    });

    let without = observations_without_convergence(&observations);
    let with = observations_with_convergence(&observations);
    println!("observations-to-decision: without={without}, with={with}");
    assert!(
        with <= without,
        "convergence path should use fewer or equal observations: {with} <= {without}"
    );
}

criterion_group!(benches, bench_convergence);
criterion_main!(benches);
