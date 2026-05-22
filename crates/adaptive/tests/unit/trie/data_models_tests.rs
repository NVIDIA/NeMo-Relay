// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for data models in the NeMo Relay adaptive crate.

use super::*;

// -----------------------------------------------------------------------
// PredictionMetrics tests
// -----------------------------------------------------------------------

#[test]
fn test_prediction_metrics_default_all_zeros() {
    let m = PredictionMetrics::default();
    assert_eq!(m.sample_count, 0);
    assert_eq!(m.mean, 0.0);
    assert_eq!(m.p50, 0.0);
    assert_eq!(m.p90, 0.0);
    assert_eq!(m.p95, 0.0);
}

#[test]
fn test_prediction_metrics_serde_roundtrip() {
    let m = PredictionMetrics {
        sample_count: 10,
        mean: 2.5,
        p50: 2.0,
        p90: 4.0,
        p95: 4.5,
    };
    let value = serde_json::to_value(&m).unwrap();
    let restored: PredictionMetrics = serde_json::from_value(value).unwrap();
    assert_eq!(restored, m);
}

#[test]
fn test_prediction_metrics_partial_eq() {
    let a = PredictionMetrics {
        sample_count: 5,
        mean: 1.0,
        p50: 1.0,
        p90: 2.0,
        p95: 2.5,
    };
    let b = a.clone();
    assert_eq!(a, b);
}

// -----------------------------------------------------------------------
// LlmCallPrediction tests
// -----------------------------------------------------------------------

#[test]
fn test_llm_call_prediction_default_has_default_metrics() {
    let p = LlmCallPrediction::default();
    assert_eq!(p.remaining_calls, PredictionMetrics::default());
    assert_eq!(p.interarrival_ms, PredictionMetrics::default());
    assert_eq!(p.output_tokens, PredictionMetrics::default());
    assert!(p.latency_sensitivity.is_none());
}

#[test]
fn test_llm_call_prediction_latency_sensitivity_some() {
    let p = LlmCallPrediction {
        latency_sensitivity: Some(3),
        ..LlmCallPrediction::default()
    };
    let value = serde_json::to_value(&p).unwrap();
    assert_eq!(value["latency_sensitivity"], serde_json::json!(3));
}

#[test]
fn test_llm_call_prediction_latency_sensitivity_none() {
    let p = LlmCallPrediction::default();
    let value = serde_json::to_value(&p).unwrap();
    assert_eq!(
        value["latency_sensitivity"],
        serde_json::Value::Null,
        "latency_sensitivity=None should serialize as null"
    );
}

// -----------------------------------------------------------------------
// PredictionTrieNode tests
// -----------------------------------------------------------------------

#[test]
fn test_prediction_trie_node_new_defaults() {
    let node = PredictionTrieNode::new("root");
    assert_eq!(node.name, "root");
    assert!(node.children.is_empty());
    assert!(node.predictions_by_call_index.is_empty());
    assert!(node.predictions_any_index.is_none());
}

#[test]
fn test_prediction_trie_node_nested_serde_roundtrip() {
    let mut root = PredictionTrieNode::new("root");
    let mut child = PredictionTrieNode::new("agent_fn");
    child.predictions_any_index = Some(LlmCallPrediction::default());
    root.children.insert("agent_fn".to_string(), child);

    let json = serde_json::to_value(&root).unwrap();
    let restored: PredictionTrieNode = serde_json::from_value(json).unwrap();

    assert_eq!(restored.name, "root");
    assert!(restored.children.contains_key("agent_fn"));
    let child = &restored.children["agent_fn"];
    assert_eq!(child.name, "agent_fn");
    assert!(child.predictions_any_index.is_some());
}

#[test]
fn test_predictions_by_call_index_string_keys() {
    let mut node = PredictionTrieNode::new("test");
    node.predictions_by_call_index
        .insert(1, LlmCallPrediction::default());
    node.predictions_by_call_index
        .insert(2, LlmCallPrediction::default());

    let value = serde_json::to_value(&node).unwrap();
    let index_map = value["predictions_by_call_index"]
        .as_object()
        .expect("predictions_by_call_index should be a JSON object");

    // HashMap<u32, T> serializes with string keys in serde_json
    assert!(
        index_map.contains_key("1"),
        "key '1' should be a string in JSON"
    );
    assert!(
        index_map.contains_key("2"),
        "key '2' should be a string in JSON"
    );
}

// -----------------------------------------------------------------------
// Send + Sync compile-time assertions
// -----------------------------------------------------------------------

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn test_all_types_send_sync() {
    assert_send_sync::<PredictionMetrics>();
    assert_send_sync::<LlmCallPrediction>();
    assert_send_sync::<PredictionTrieNode>();
}
