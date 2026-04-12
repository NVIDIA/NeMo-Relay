// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::trie::data_models::PredictionMetrics;

/// Helper to create a test prediction with distinct values for identification.
fn make_prediction(remaining_mean: f64, sensitivity: Option<u32>) -> LlmCallPrediction {
    LlmCallPrediction {
        remaining_calls: PredictionMetrics {
            sample_count: 10,
            mean: remaining_mean,
            ..Default::default()
        },
        interarrival_ms: PredictionMetrics::default(),
        output_tokens: PredictionMetrics::default(),
        latency_sensitivity: sensitivity,
    }
}

#[test]
fn test_find_exact_path_and_index_match() {
    // root -> "workflow" -> "agent" with prediction at index 2
    let mut root = PredictionTrieNode::new("root");
    let mut workflow = PredictionTrieNode::new("workflow");
    let mut agent = PredictionTrieNode::new("agent");
    agent
        .predictions_by_call_index
        .insert(2, make_prediction(99.0, Some(5)));
    workflow.children.insert("agent".to_string(), agent);
    root.children.insert("workflow".to_string(), workflow);

    let lookup = PredictionTrieLookup::new(&root);
    let path = vec!["workflow".to_string(), "agent".to_string()];
    let result = lookup.find(&path, 2);
    assert!(result.is_some());
    assert_eq!(result.unwrap().remaining_calls.mean, 99.0);
}

#[test]
fn test_find_falls_back_to_any_index() {
    // Node has predictions_any_index but not the exact call_index
    let mut root = PredictionTrieNode::new("root");
    let mut child = PredictionTrieNode::new("child");
    child.predictions_any_index = Some(make_prediction(42.0, Some(2)));
    root.children.insert("child".to_string(), child);

    let lookup = PredictionTrieLookup::new(&root);
    let path = vec!["child".to_string()];
    // Request index 7, which doesn't exist -- should fall back to any_index
    let result = lookup.find(&path, 7);
    assert!(result.is_some());
    assert_eq!(result.unwrap().remaining_calls.mean, 42.0);
}

#[test]
fn test_find_deepest_ancestor_when_path_extends_beyond_trie() {
    // Trie has root -> "a" -> "b" with prediction at "b"
    // Path is ["a", "b", "c"] -- "c" doesn't exist, should return "b"'s prediction
    let mut root = PredictionTrieNode::new("root");
    let mut a = PredictionTrieNode::new("a");
    let mut b = PredictionTrieNode::new("b");
    b.predictions_any_index = Some(make_prediction(77.0, None));
    a.children.insert("b".to_string(), b);
    root.children.insert("a".to_string(), a);

    let lookup = PredictionTrieLookup::new(&root);
    let path = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let result = lookup.find(&path, 1);
    assert!(result.is_some());
    assert_eq!(result.unwrap().remaining_calls.mean, 77.0);
}

#[test]
fn test_find_returns_root_prediction_when_no_child_match() {
    // Root has a prediction, path has ["unknown"] which doesn't exist
    let mut root = PredictionTrieNode::new("root");
    root.predictions_any_index = Some(make_prediction(10.0, Some(1)));

    let lookup = PredictionTrieLookup::new(&root);
    let path = vec!["unknown".to_string()];
    let result = lookup.find(&path, 1);
    assert!(result.is_some());
    assert_eq!(result.unwrap().remaining_calls.mean, 10.0);
}

#[test]
fn test_find_returns_none_when_no_predictions_at_all() {
    // Empty trie: root with no predictions and no children
    let root = PredictionTrieNode::new("root");

    let lookup = PredictionTrieLookup::new(&root);
    let path = vec!["something".to_string()];
    let result = lookup.find(&path, 1);
    assert!(result.is_none());
}

#[test]
fn test_find_prefers_deeper_match_over_shallow() {
    // Both root and child "a" -> "b" have predictions; find should return "b"'s
    let mut root = PredictionTrieNode::new("root");
    root.predictions_any_index = Some(make_prediction(1.0, Some(1)));

    let mut a = PredictionTrieNode::new("a");
    a.predictions_any_index = Some(make_prediction(2.0, Some(2)));

    let mut b = PredictionTrieNode::new("b");
    b.predictions_any_index = Some(make_prediction(3.0, Some(3)));

    a.children.insert("b".to_string(), b);
    root.children.insert("a".to_string(), a);

    let lookup = PredictionTrieLookup::new(&root);
    let path = vec!["a".to_string(), "b".to_string()];
    let result = lookup.find(&path, 1);
    assert!(result.is_some());
    assert_eq!(result.unwrap().remaining_calls.mean, 3.0);
}

#[test]
fn test_find_empty_path_returns_root_prediction() {
    let mut root = PredictionTrieNode::new("root");
    root.predictions_any_index = Some(make_prediction(5.0, None));

    let lookup = PredictionTrieLookup::new(&root);
    let path: Vec<String> = vec![];
    let result = lookup.find(&path, 1);
    assert!(result.is_some());
    assert_eq!(result.unwrap().remaining_calls.mean, 5.0);
}

#[test]
fn test_find_exact_index_beats_any_index_at_deeper_node() {
    // Deeper node has both exact index and any_index -- exact should win
    let mut root = PredictionTrieNode::new("root");
    let mut child = PredictionTrieNode::new("child");
    child
        .predictions_by_call_index
        .insert(3, make_prediction(100.0, Some(9)));
    child.predictions_any_index = Some(make_prediction(50.0, Some(5)));
    root.children.insert("child".to_string(), child);

    let lookup = PredictionTrieLookup::new(&root);
    let path = vec!["child".to_string()];
    let result = lookup.find(&path, 3);
    assert!(result.is_some());
    assert_eq!(result.unwrap().remaining_calls.mean, 100.0);
}
