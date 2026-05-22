// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for serialization in the NeMo Relay adaptive crate.

use super::*;
use crate::trie::data_models::{LlmCallPrediction, PredictionMetrics};

const NAT_GOLDEN_JSON: &str = r#"{
      "version": "1.0",
      "generated_at": "2026-03-31T12:00:00+00:00",
      "workflow_name": "my_agent",
      "root": {
        "name": "root",
        "predictions_by_call_index": {
          "1": {
            "remaining_calls": { "sample_count": 10, "mean": 2.5, "p50": 2.0, "p90": 4.0, "p95": 4.5 },
            "interarrival_ms": { "sample_count": 10, "mean": 150.0, "p50": 140.0, "p90": 200.0, "p95": 220.0 },
            "output_tokens": { "sample_count": 10, "mean": 256.0, "p50": 240.0, "p90": 400.0, "p95": 450.0 },
            "latency_sensitivity": 3
          }
        },
        "predictions_any_index": {
          "remaining_calls": { "sample_count": 30, "mean": 1.8, "p50": 2.0, "p90": 3.0, "p95": 3.5 },
          "interarrival_ms": { "sample_count": 30, "mean": 160.0, "p50": 150.0, "p90": 210.0, "p95": 230.0 },
          "output_tokens": { "sample_count": 30, "mean": 280.0, "p50": 250.0, "p90": 420.0, "p95": 470.0 },
          "latency_sensitivity": 2
        },
        "children": {
          "react_agent": {
            "name": "react_agent",
            "predictions_by_call_index": {},
            "predictions_any_index": null,
            "children": {}
          }
        }
      }
    }"#;

#[test]
fn test_envelope_serializes_with_required_top_level_keys() {
    let root = PredictionTrieNode::new("root");
    let envelope = TrieEnvelope::new(root, "test_workflow");
    let json = envelope.to_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(value.get("version").is_some());
    assert!(value.get("generated_at").is_some());
    assert!(value.get("workflow_name").is_some());
    assert!(value.get("root").is_some());
}

#[test]
fn test_envelope_version_is_always_1_0() {
    let root = PredictionTrieNode::new("root");
    let envelope = TrieEnvelope::new(root, "test");
    assert_eq!(envelope.version, "1.0");
}

#[test]
fn test_envelope_generated_at_is_iso8601() {
    let root = PredictionTrieNode::new("root");
    let envelope = TrieEnvelope::new(root, "test");
    // Verify it parses as a valid RFC 3339 / ISO 8601 timestamp
    let parsed = chrono::DateTime::parse_from_rfc3339(&envelope.generated_at);
    assert!(
        parsed.is_ok(),
        "generated_at should be valid ISO 8601: {}",
        envelope.generated_at
    );
}

#[test]
fn test_predictions_by_call_index_serializes_with_string_keys() {
    let mut root = PredictionTrieNode::new("root");
    root.predictions_by_call_index
        .insert(1, LlmCallPrediction::default());
    root.predictions_by_call_index
        .insert(2, LlmCallPrediction::default());

    let envelope = TrieEnvelope::new(root, "test");
    let json = envelope.to_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let index_map = value["root"]["predictions_by_call_index"]
        .as_object()
        .unwrap();
    assert!(index_map.contains_key("1"), "key should be string '1'");
    assert!(index_map.contains_key("2"), "key should be string '2'");
}

#[test]
fn test_predictions_any_index_null_when_none() {
    let root = PredictionTrieNode::new("root");
    let envelope = TrieEnvelope::new(root, "test");
    let json = envelope.to_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(
        value["root"]["predictions_any_index"],
        serde_json::Value::Null
    );
}

#[test]
fn test_predictions_any_index_object_when_some() {
    let mut root = PredictionTrieNode::new("root");
    root.predictions_any_index = Some(LlmCallPrediction {
        remaining_calls: PredictionMetrics {
            sample_count: 5,
            mean: 1.0,
            ..Default::default()
        },
        ..Default::default()
    });

    let envelope = TrieEnvelope::new(root, "test");
    let json = envelope.to_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(value["root"]["predictions_any_index"].is_object());
}

#[test]
fn test_children_serializes_as_nested_objects() {
    let mut root = PredictionTrieNode::new("root");
    let mut child = PredictionTrieNode::new("child_a");
    let grandchild = PredictionTrieNode::new("grandchild_b");
    child
        .children
        .insert("grandchild_b".to_string(), grandchild);
    root.children.insert("child_a".to_string(), child);

    let envelope = TrieEnvelope::new(root, "test");
    let json = envelope.to_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    let child_val = &value["root"]["children"]["child_a"];
    assert_eq!(child_val["name"], "child_a");
    let grandchild_val = &child_val["children"]["grandchild_b"];
    assert_eq!(grandchild_val["name"], "grandchild_b");
}

#[test]
fn test_golden_nat_json_deserializes_and_reserializes() {
    // Deserialize NAT golden JSON
    let envelope = TrieEnvelope::from_json(NAT_GOLDEN_JSON).unwrap();
    assert_eq!(envelope.version, "1.0");
    assert_eq!(envelope.workflow_name, "my_agent");
    assert_eq!(envelope.root.name, "root");

    // Verify root predictions_by_call_index has key 1 with sample_count=10
    let pred1 = envelope
        .root
        .predictions_by_call_index
        .get(&1)
        .expect("should have key 1");
    assert_eq!(pred1.remaining_calls.sample_count, 10);

    // Verify root predictions_any_index is Some with sample_count=30
    let any = envelope
        .root
        .predictions_any_index
        .as_ref()
        .expect("should have any_index");
    assert_eq!(any.remaining_calls.sample_count, 30);

    // Verify children has "react_agent" with predictions_any_index=None
    let react = envelope
        .root
        .children
        .get("react_agent")
        .expect("should have react_agent");
    assert!(react.predictions_any_index.is_none());

    // Re-serialize and compare as serde_json::Value (not string equality)
    let reserialized = envelope.to_json().unwrap();
    let original_val: serde_json::Value = serde_json::from_str(NAT_GOLDEN_JSON).unwrap();
    let reserialized_val: serde_json::Value = serde_json::from_str(&reserialized).unwrap();
    assert_eq!(
        original_val, reserialized_val,
        "re-serialized JSON should be equivalent to original"
    );
}

#[test]
fn test_roundtrip_construct_serialize_deserialize() {
    let mut root = PredictionTrieNode::new("test_root");
    root.predictions_any_index = Some(LlmCallPrediction::default());
    let envelope = TrieEnvelope::new(root, "roundtrip_workflow");

    let json = envelope.to_json().unwrap();
    let restored = TrieEnvelope::from_json(&json).unwrap();

    assert_eq!(restored.version, envelope.version);
    assert_eq!(restored.generated_at, envelope.generated_at);
    assert_eq!(restored.workflow_name, envelope.workflow_name);
    assert_eq!(restored.root.name, "test_root");
    assert!(restored.root.predictions_any_index.is_some());
}

#[test]
fn test_new_helper_creates_with_current_timestamp_and_version() {
    let root = PredictionTrieNode::new("root");
    let before = chrono::Utc::now();
    let envelope = TrieEnvelope::new(root, "my_workflow");
    let after = chrono::Utc::now();

    assert_eq!(envelope.version, CURRENT_VERSION);
    assert_eq!(envelope.workflow_name, "my_workflow");

    let ts = chrono::DateTime::parse_from_rfc3339(&envelope.generated_at)
        .expect("should be valid RFC 3339");
    let before_fixed: chrono::DateTime<chrono::FixedOffset> = before.into();
    let after_fixed: chrono::DateTime<chrono::FixedOffset> = after.into();
    assert!(
        ts >= before_fixed && ts <= after_fixed,
        "generated_at should be between before and after creation time"
    );
}
