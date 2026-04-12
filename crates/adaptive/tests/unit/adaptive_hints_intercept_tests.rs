// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::{Mutex, OnceLock};

use crate::trie::data_models::{LlmCallPrediction, PredictionMetrics};
use nemo_flow::api::scope::{pop_scope, push_scope};
use nemo_flow::codec::request::{AnnotatedLLMRequest, Message, MessageContent};
use nemo_flow::context::scope_stack::current_scope_stack;
use nemo_flow::types::scope::{ScopeAttributes, ScopeType};

static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn test_mutex() -> &'static Mutex<()> {
    TEST_MUTEX.get_or_init(|| Mutex::new(()))
}

fn reset_root_metadata() {
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = None;
}

fn make_prediction(latency_sensitivity: Option<u32>) -> LlmCallPrediction {
    LlmCallPrediction {
        remaining_calls: PredictionMetrics {
            sample_count: 10,
            mean: 5.0,
            p50: 5.0,
            p90: 8.0,
            p95: 9.0,
        },
        interarrival_ms: PredictionMetrics {
            sample_count: 10,
            mean: 200.0,
            p50: 180.0,
            p90: 300.0,
            p95: 350.0,
        },
        output_tokens: PredictionMetrics {
            sample_count: 10,
            mean: 100.0,
            p50: 90.0,
            p90: 150.0,
            p95: 180.0,
        },
        latency_sensitivity,
    }
}

#[test]
fn test_build_agent_hints_from_prediction() {
    let pred = make_prediction(Some(4));

    let hints = build_agent_hints(Some(&pred), &None, "test-agent", 2, 3).unwrap();
    assert_eq!(hints.osl, 150, "osl = output_tokens.p90");
    assert_eq!(hints.iat, 200, "iat = interarrival_ms.mean");
    assert_eq!(hints.priority, 1, "priority = 5 - 4 = 1");
    assert!((hints.latency_sensitivity - 4.0).abs() < f64::EPSILON);
    assert_eq!(hints.prefix_id, "test-agent-d3");
    assert_eq!(hints.total_requests, 7, "total_requests = 5 + 2 = 7");
}

#[test]
fn test_build_agent_hints_falls_back_to_defaults() {
    let defaults = AgentHints {
        osl: 42,
        iat: 99,
        priority: 1,
        latency_sensitivity: 4.0,
        prefix_id: "fallback".into(),
        total_requests: 10,
    };
    let hints = build_agent_hints(None, &Some(defaults.clone()), "agent", 1, 0).unwrap();
    assert_eq!(hints.osl, 42);
    assert_eq!(hints.prefix_id, "fallback");
}

#[test]
fn test_build_agent_hints_none_when_no_prediction_and_no_defaults() {
    let hints = build_agent_hints(None, &None, "agent", 1, 0);
    assert!(hints.is_none());
}

#[test]
fn test_build_agent_hints_defaults_missing_latency_sensitivity_to_one() {
    let pred = make_prediction(None);
    let hints = build_agent_hints(Some(&pred), &None, "fallback-agent", 4, 1).unwrap();
    assert_eq!(hints.priority, 4);
    assert_eq!(hints.latency_sensitivity, 1.0);
    assert_eq!(hints.prefix_id, "fallback-agent-d1");
    assert_eq!(hints.total_requests, 9);
}

#[test]
fn test_adaptive_hints_intercept_new() {
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
    }));
    let intercept = AdaptiveHintsIntercept::new(hot_cache, "test".to_string());
    assert_eq!(intercept.call_counter.load(Ordering::Relaxed), 1);
    assert_eq!(intercept.agent_id, "test");
}

#[test]
fn test_adaptive_hints_intercept_into_request_fn_compiles() {
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
    }));
    let intercept = AdaptiveHintsIntercept::new(hot_cache, "test".to_string());
    let _req_fn: LlmRequestInterceptFn = intercept.into_request_fn();
    // If this compiles and runs, the type is correct.
}

#[test]
fn test_adaptive_hints_intercept_injects_prediction_hints_and_manual_override() {
    let _guard = test_mutex().lock().unwrap();
    reset_root_metadata();

    let mut root = crate::trie::data_models::PredictionTrieNode::new("root");
    let mut agent_node = crate::trie::data_models::PredictionTrieNode::new("scope-agent");
    let mut function_node = crate::trie::data_models::PredictionTrieNode::new("step");
    function_node
        .predictions_by_call_index
        .insert(1, make_prediction(Some(2)));
    agent_node.children.insert("step".into(), function_node);
    root.children.insert("scope-agent".into(), agent_node);

    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: Some(root),
        agent_hints_default: None,
    }));
    let intercept = AdaptiveHintsIntercept::new(hot_cache, "fallback-agent".to_string());
    let req_fn = intercept.into_request_fn();

    let agent_scope = push_scope(
        "scope-agent",
        ScopeType::Agent,
        None,
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap();
    let function_scope = push_scope(
        "step",
        ScopeType::Function,
        Some(&agent_scope),
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap();
    crate::context_helpers::set_latency_sensitivity(5).unwrap();

    let annotated = AnnotatedLLMRequest {
        messages: vec![Message::User {
            content: MessageContent::Text("hello".into()),
            name: None,
        }],
        model: Some("model".into()),
        params: None,
        tools: None,
        tool_choice: None,
        extra: serde_json::Map::new(),
    };
    let (request, returned_annotated) = req_fn(
        "model",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        Some(annotated.clone()),
    )
    .unwrap();

    let body_hints = &request.content["nvext"]["agent_hints"];
    assert_eq!(body_hints["osl"], serde_json::json!(150));
    assert_eq!(body_hints["iat"], serde_json::json!(200));
    assert_eq!(body_hints["latency_sensitivity"], serde_json::json!(5.0));
    assert_eq!(body_hints["priority"], serde_json::json!(0));
    assert_eq!(body_hints["prefix_id"], serde_json::json!("scope-agent-d2"));
    assert_eq!(body_hints["total_requests"], serde_json::json!(6));
    assert_eq!(
        request.headers.get(AGENT_HINTS_HEADER_KEY).unwrap(),
        body_hints
    );
    assert_eq!(returned_annotated, Some(annotated));

    pop_scope(&function_scope.uuid).unwrap();
    pop_scope(&agent_scope.uuid).unwrap();
    reset_root_metadata();
}

#[test]
fn test_adaptive_hints_intercept_uses_defaults_and_ignores_poisoned_cache() {
    let _guard = test_mutex().lock().unwrap();
    reset_root_metadata();

    let defaults = AgentHints {
        osl: 9,
        iat: 12,
        priority: 3,
        latency_sensitivity: 2.0,
        prefix_id: "defaults".into(),
        total_requests: 11,
    };
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: Some(defaults.clone()),
    }));
    let req_fn =
        AdaptiveHintsIntercept::new(hot_cache, "fallback-agent".to_string()).into_request_fn();
    let (request, annotated) = req_fn(
        "model",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        None,
    )
    .unwrap();
    assert_eq!(
        request.headers.get(AGENT_HINTS_HEADER_KEY),
        Some(&serde_json::to_value(&defaults).unwrap())
    );
    assert_eq!(
        request.content["nvext"]["agent_hints"]["prefix_id"],
        "defaults"
    );
    assert!(annotated.is_none());

    let poisoned_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: Some(defaults),
    }));
    let _ = std::panic::catch_unwind({
        let poisoned_cache = poisoned_cache.clone();
        move || {
            let _guard = poisoned_cache.write().unwrap();
            panic!("poison adaptive cache lock");
        }
    });
    let poisoned_req_fn =
        AdaptiveHintsIntercept::new(poisoned_cache, "fallback-agent".to_string()).into_request_fn();
    let (poisoned_request, _) = poisoned_req_fn(
        "model",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({"existing": true}),
        },
        None,
    )
    .unwrap();
    assert!(
        poisoned_request
            .headers
            .get(AGENT_HINTS_HEADER_KEY)
            .is_none()
    );
    assert_eq!(
        poisoned_request.content,
        serde_json::json!({"existing": true})
    );

    reset_root_metadata();
}
